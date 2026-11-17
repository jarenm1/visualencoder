"""FP16 TensorRT engine for a frozen encoder.

The encoder is exported to ONNX in float16 (strongly typed, so TensorRT keeps the graph's own
precisions, including the backbone's fp32 RoPE), built once per weights fingerprint, input size,
GPU and TensorRT version, and cached under ``$XDG_CACHE_HOME/visenc/tensorrt``. Inference reads
and writes torch tensors in place on the caller's current CUDA stream.
"""
from __future__ import annotations

import copy
import io
import os
import re
from pathlib import Path

import tensorrt as trt
import torch
from torch import nn

OPSET = 17
OPT_BATCH = 128
MAX_BATCH = 256
# Bump when the exported graph or builder settings change; old cache entries are then ignored.
ENGINE_VERSION = 1
CACHE = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "visenc" / "tensorrt"


class _Float16(nn.Module):
    """float32 [B,3,H,W] in [0,1] -> the encoder's (cls, patches), computed in float16."""

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, images):
        return self.encoder(images.half())


class TensorRTEncoder:
    """Drop-in for the frozen encoder: float32 [B,3,H,W] -> float16 (cls [B,D], patches [B,P,D])."""

    def __init__(self, encoder, fingerprint, device):
        self.encoder = encoder  # float32 module; exported only when an engine is not cached
        self.fingerprint = fingerprint
        self.device = device
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engines = {}

    def __call__(self, images):
        if images.dtype != torch.float32 or images.device != self.device or images.dim() != 4:
            raise ValueError("TensorRT encoder expects float32 [B,3,H,W] images on its device")
        images = images.contiguous()
        batch, _, height, width = images.shape
        _, context = self._engine(height, width)
        context.set_input_shape("image", (min(batch, MAX_BATCH), 3, height, width))
        cls = torch.empty((batch, *context.get_tensor_shape("cls")[1:]), dtype=torch.float16, device=self.device)
        patches = torch.empty((batch, *context.get_tensor_shape("patches")[1:]), dtype=torch.float16,
                              device=self.device)
        stream = torch.cuda.current_stream(self.device).cuda_stream
        for start in range(0, batch, MAX_BATCH):
            end = min(start + MAX_BATCH, batch)
            context.set_input_shape("image", (end - start, 3, height, width))
            context.set_tensor_address("image", images[start:end].data_ptr())
            context.set_tensor_address("cls", cls[start:end].data_ptr())
            context.set_tensor_address("patches", patches[start:end].data_ptr())
            if not context.execute_async_v3(stream):
                raise RuntimeError("TensorRT encoder execution failed")
        return cls, patches

    def _engine(self, height, width):
        key = (height, width)
        if key not in self.engines:
            path = CACHE / self._engine_name(height, width)
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                partial = path.with_suffix(f".{os.getpid()}.partial")
                partial.write_bytes(self._build(height, width))
                partial.replace(path)
            engine = self.runtime.deserialize_cuda_engine(path.read_bytes())
            if engine is None:
                raise RuntimeError(f"cannot load TensorRT engine {path}; delete it to rebuild")
            self.engines[key] = (engine, engine.create_execution_context())
        return self.engines[key]

    def _engine_name(self, height, width):
        properties = torch.cuda.get_device_properties(self.device)
        gpu = re.sub(r"[^A-Za-z0-9]+", "-", properties.name).strip("-")
        return (f"{self.fingerprint[:24]}-{width}x{height}-b{OPT_BATCH}-{MAX_BATCH}-fp16-v{ENGINE_VERSION}"
                f"-trt{trt.__version__}-{gpu}-sm{properties.major}{properties.minor}.engine")

    def _build(self, height, width):
        model = _Float16(copy.deepcopy(self.encoder)).to(self.device).half().eval()
        onnx = io.BytesIO()
        with torch.no_grad():
            torch.onnx.export(
                model, (torch.rand(2, 3, height, width, device=self.device),), onnx,
                input_names=["image"], output_names=["cls", "patches"],
                dynamic_axes={"image": {0: "batch"}, "cls": {0: "batch"}, "patches": {0: "batch"}},
                opset_version=OPSET, dynamo=False, do_constant_folding=True,
            )  # fmt: skip
        del model
        torch.cuda.empty_cache()
        builder = trt.Builder(self.logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
        parser = trt.OnnxParser(network, self.logger)
        if not parser.parse(onnx.getvalue()):
            errors = "\n".join(str(parser.get_error(index)) for index in range(parser.num_errors))
            raise RuntimeError(f"TensorRT cannot parse the exported encoder:\n{errors}")
        config = builder.create_builder_config()
        profile = builder.create_optimization_profile()
        profile.set_shape("image", (1, 3, height, width), (OPT_BATCH, 3, height, width),
                          (MAX_BATCH, 3, height, width))
        config.add_optimization_profile(profile)
        plan = builder.build_serialized_network(network, config)
        if plan is None:
            raise RuntimeError("TensorRT engine build failed")
        return bytes(plan)
