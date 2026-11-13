"""Deployment artefacts for Jetson Orin Nano (TensorRT, JetPack 6).

`visenc deploy --encoder run:runs/x --calib data/train --out deploy/x` writes:
    encoder.onnx       fp32 weights, static [1,3,H,W] float input in [0,1] (normalisation inside);
                       build an FP16 engine on the device with `trtexec --fp16`
    encoder_int8.onnx  explicit INT8 Q/DQ graph (ModelOpt, calibrated on rendered views);
                       build with `trtexec --int8 --fp16` (Q/DQ scales are used as given)
    int8_modelopt.pth  ModelOpt state; `int8:deploy/x` evaluates the simulated-INT8 model
    deploy.json        source spec, input size, calibration set, trtexec commands

Engines are built on the Jetson itself: TensorRT engines are tied to the GPU and TRT version.
"""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .data import Split
from .models import Encoder, load_encoder

OPSET = 17


class FP16(Encoder):
    """Encoder run entirely in float16 (what an FP16 TensorRT engine computes), no autocast."""

    autocast = False

    def __init__(self, inner: Encoder):
        super().__init__()
        self.inner = inner.half()
        self.dim = inner.dim

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cls, patches = self.inner(images.half())
        return cls.float(), patches.float()


class Simulated(Encoder):
    """ModelOpt fake-quantised encoder: INT8 Q/DQ on Linear/Conv inputs and weights, fp32 elsewhere."""

    autocast = False

    def __init__(self, inner: nn.Module, dim: int):
        super().__init__()
        self.inner, self.dim = inner, dim

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.inner(images.float())


def calibration_images(split: Split, count: int, seed: int = 0) -> torch.Tensor:
    """uint8 [count,3,H,W] drawn evenly from every appearance view of random poses."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(split), count, replace=False)
    views = [split.views[k % len(split.views)] for k in range(count)]
    return torch.from_numpy(np.stack([split.rgb[v][i] for v, i in zip(views, idx)])).permute(0, 3, 1, 2)


def int8_config() -> dict:
    """SmoothQuant INT8 on every Linear/Conv; LayerNorm inputs stay FP16.

    The ViT residual stream has outlier channels (final-norm input range ~±400), so per-tensor
    INT8 on norm inputs destroys features (patch cosine to FP32 0.58 with ModelOpt's default).
    Measured on the LingBot-S student: SmoothQuant without norm quantizers keeps cosine 0.998.
    """
    import modelopt.torch.quantization as mtq

    cfg = copy.deepcopy(mtq.INT8_SMOOTHQUANT_CFG)
    cfg["quant_cfg"].append({"quantizer_name": "*norm*", "enable": False})  # later rules override earlier ones
    return cfg


def quantize_int8(encoder: Encoder, calib: torch.Tensor, device: torch.device, batch_size: int = 32) -> nn.Module:
    """Quantises `encoder` in place (ModelOpt) and returns it."""
    import modelopt.torch.quantization as mtq

    def forward_loop(model: nn.Module) -> None:
        for start in range(0, len(calib), batch_size):
            model(calib[start : start + batch_size].to(device).float().div_(255))

    return mtq.quantize(encoder, int8_config(), forward_loop)


def export_onnx(model: nn.Module, path: Path, height: int, width: int) -> None:
    """Exports from CPU: tracing ModelOpt's fake-quant ops on CUDA segfaults without its CUDA extension
    (which needs ninja + nvcc). The TorchScript exporter is used because the dynamo exporter emits an
    opset-18 Split attribute that ONNX Runtime rejects at opset 17."""
    model = model.cpu()
    torch.onnx.export(
        model, (torch.rand(1, 3, height, width),), str(path), input_names=["image"], output_names=["cls", "patches"],
        opset_version=OPSET, dynamo=False, do_constant_folding=True,
    )  # fmt: skip


def mean_cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.cosine_similarity(a.float(), b.float(), dim=-1).mean().item()


def onnx_outputs(path: Path, images: torch.Tensor) -> list[torch.Tensor]:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    outs = [sess.run(None, {"image": images[i : i + 1].numpy()}) for i in range(len(images))]
    return [torch.from_numpy(np.concatenate([o[k] for o in outs])) for k in range(2)]


def run(args: argparse.Namespace) -> None:
    import modelopt.torch.opt as mto

    device = torch.device("cuda")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    split = Split(args.calib)
    height, width = split.size
    encoder = load_encoder(args.encoder).eval()
    dim = encoder.dim
    check = calibration_images(split, 8, seed=1).float().div(255)  # disjoint seed from calibration
    with torch.no_grad():
        ref = encoder(check)

    export_onnx(encoder, out / "encoder.onnx", height, width)
    fp32_onnx = onnx_outputs(out / "encoder.onnx", check)

    calib = calibration_images(split, args.calib_count)
    with torch.no_grad():
        quantized = quantize_int8(encoder.to(device), calib, device)  # in place: `encoder` is now quantized
    mto.save(quantized, out / "int8_modelopt.pth")
    export_onnx(quantized, out / "encoder_int8.onnx", height, width)
    with torch.no_grad():
        sim = quantized(check)
    int8_onnx = onnx_outputs(out / "encoder_int8.onnx", check)

    names = ("cls", "patches")
    meta = {
        "encoder": args.encoder,
        "dim": dim,
        "input": {"shape": [1, 3, height, width], "dtype": "float32", "range": "[0,1] RGB, ImageNet normalisation inside"},
        "outputs": {"cls": [1, dim], "patches": [1, (height // 16) * (width // 16), dim]},
        "calibration": {"split": str(args.calib), "images": args.calib_count, "config": "SmoothQuant INT8, LayerNorm inputs FP16"},
        "cosine_to_pytorch_fp32": {
            "onnx_fp32": {n: mean_cosine(fp32_onnx[k], ref[k]) for k, n in enumerate(names)},
            "simulated_int8": {n: mean_cosine(sim[k], ref[k]) for k, n in enumerate(names)},
            "onnx_int8": {n: mean_cosine(int8_onnx[k], ref[k]) for k, n in enumerate(names)},
        },
        "trtexec": {
            "fp16": "trtexec --onnx=encoder.onnx --fp16 --saveEngine=encoder_fp16.engine",
            "int8": "trtexec --onnx=encoder_int8.onnx --int8 --fp16 --saveEngine=encoder_int8.engine",
        },
    }
    (out / "deploy.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


def load_int8(deploy_dir: Path) -> Simulated:
    import modelopt.torch.opt as mto

    meta = json.loads((deploy_dir / "deploy.json").read_text())
    base = load_encoder(meta["encoder"]).eval()
    return Simulated(mto.restore(base, deploy_dir / "int8_modelopt.pth"), meta["dim"])


def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--encoder", required=True, help="encoder spec to deploy, usually run:runs/<name>")
    p.add_argument("--calib", required=True, help="split whose rendered views calibrate INT8 ranges")
    p.add_argument("--calib-count", type=int, default=512)
    p.add_argument("--out", required=True, help="new deploy directory")
