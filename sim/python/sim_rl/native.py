"""Borrowed CUDA tensors over the sim_env C ABI v4; no observation transport."""
from __future__ import annotations

import ctypes as ct
import json
import os
from contextlib import contextmanager
from pathlib import Path

import torch

DEFAULT_CITY = Path(__file__).resolve().parents[3] / "assets" / "city"
TASK = "city-drone-goal-v1"
ABI_VERSION = 4
ACTION_DIM = 4
HORIZON = 450
CONTROL_HZ = 30
CONTROL_MODES = {"velocity": 0, "rates": 1}
# Zero velocity commands hover; rates mode keeps the v3 hover throttle bias.
HOVER_ACTIONS = {"velocity": (0.0, 0.0, 0.0, 0.0), "rates": (-0.2, 0.0, 0.0, 0.0)}
# Fields: 0 obs, 1 proprio, 2 reward, 3 done, 4 success, 5 privileged, 6 collision,
# 7 episodes, 8 diagnostics, 9 design, 10 goal images, 11 curriculum (writable),
# 12 ended bucket, 13 out_of_bounds, 14 ended gps_available.
FIELD_COUNT = 15
DIAGNOSTICS_WIDTH = 24
CURRICULUM_WIDTH = 4  # w_visible, w_partial, w_hidden, p_no_gps
PROPRIO_WIDTH = 19
PRIVILEGED_WIDTH = 21
REQUIRED_METADATA = ("abi_version", "task", "action_dim", "control_hz", "physics_hz", "horizon",
                     "design_count", "city_fingerprint", "city_path", "control_mode", "action_names",
                     "action_scales")
SPLIT_SPECIFIC_METADATA = ("city_path", "city", "style", "split", "design_count", "designs",
                           "target_designs", "district", "catalog", "fields")


class _CudaAllocation:
    def __init__(self, owner, pointer, shape, typestr):
        self.owner = owner
        self.__cuda_array_interface__ = {
            "shape": shape, "typestr": typestr, "data": (pointer, False),
            "version": 3, "strides": None, "stream": owner.stream.cuda_stream,
        }


class CudaEnv:
    """All methods that enqueue work must run inside ``with env.execution()``.

    Returned tensors/DLPack objects borrow native allocations. They must not be
    used after close; the environment retains their owners until its final wait.
    Vulkan shared buffers are internal: every exposed pointer is CUDA-only.
    """
    def __init__(self, library, n, seed, width, height, style, control_mode, city_path=DEFAULT_CITY):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required; no CPU environment fallback")
        if n < 1 or width < 48 or height < 32 or width % 16 or height % 16:
            raise ValueError("positive env count and image dimensions divisible by 16 required")
        if control_mode not in CONTROL_MODES:
            raise ValueError(f"control_mode must be one of {tuple(CONTROL_MODES)}")
        torch.cuda.set_device(0)
        torch.cuda.init()  # Establish PyTorch's primary context before native imports.
        self.device = torch.device("cuda:0")
        self.n, self.width, self.height, self.style = n, width, height, style
        self.control_mode = control_mode
        self.library = Path(library).resolve(strict=True)
        self.city_path = Path(city_path).resolve(strict=True)
        self.lib = ct.CDLL(str(self.library))
        declarations = {
            "sim_env_create": ([ct.c_uint32, ct.c_uint64, ct.c_uint32, ct.c_uint32, ct.c_uint32, ct.c_uint32,
                                ct.c_char_p], ct.c_void_p),
            "sim_env_metadata": ([ct.c_void_p], ct.c_char_p),
            "sim_env_reset": ([ct.c_void_p, ct.c_uint64], ct.c_int32),
            "sim_env_step": ([ct.c_void_p, ct.c_void_p], ct.c_int32),
            "sim_env_buffer": ([ct.c_void_p, ct.c_int32], ct.c_uint64),
            "sim_env_stream": ([ct.c_void_p], ct.c_uint64),
            "sim_env_destroy": ([ct.c_void_p], None),
            "sim_env_error": ([], ct.c_char_p),
        }
        for name, (args, result) in declarations.items():
            try:
                function = getattr(self.lib, name)
            except AttributeError as error:
                raise RuntimeError(f"city navigation requires sim_env ABI v4 symbol {name}") from error
            function.argtypes, function.restype = args, result
        self.handle = self.lib.sim_env_create(n, seed, width, height, style, CONTROL_MODES[control_mode],
                                              os.fsencode(self.city_path))
        if not self.handle:
            raise RuntimeError(self.error())
        self._owners, self.tensors = [], []
        try:
            raw_metadata = self.lib.sim_env_metadata(self.handle)
            if not raw_metadata:
                raise RuntimeError(f"native city metadata unavailable: {self.error()}")
            self.metadata = json.loads(raw_metadata.decode("utf-8"))
            if self.metadata.get("abi_version") != ABI_VERSION or self.metadata.get("task") != TASK:
                raise ValueError("city navigation requires sim_env ABI v4 task city-drone-goal-v1; "
                                 "older libraries are incompatible")
            missing = [key for key in REQUIRED_METADATA if key not in self.metadata]
            if missing:
                raise ValueError(f"incomplete native city metadata: {missing}")
            if self.metadata["control_mode"] != control_mode:
                raise ValueError(f"native control_mode {self.metadata['control_mode']!r} != {control_mode!r}")
            self.action_dim = int(self.metadata["action_dim"])
            self.horizon = int(self.metadata["horizon"])
            self.control_hz = int(self.metadata["control_hz"])
            self.design_count = int(self.metadata["design_count"])
            self.action_names = tuple(self.metadata["action_names"])
            self.action_scales = tuple(float(value) for value in self.metadata["action_scales"])
            if (self.action_dim, self.horizon, self.control_hz) != (ACTION_DIM, HORIZON, CONTROL_HZ):
                raise ValueError("unsupported city-drone-goal-v1 action/horizon/control contract")
            if len(self.action_names) != ACTION_DIM or len(self.action_scales) != ACTION_DIM:
                raise ValueError("native metadata must name and scale exactly four control channels")
            if self.design_count < 1:
                raise ValueError("native split has no goal designs")
            # Location and split-specific content (district, catalog, held-out designs,
            # goal-image field shape) are excluded; dynamics, sensors, reward, camera,
            # city content, control mode and every other rule must match checkpoints exactly.
            self.task_contract = {key: value for key, value in self.metadata.items()
                                  if key not in SPLIT_SPECIFIC_METADATA}
            pointer = self.lib.sim_env_stream(self.handle)
            if not pointer:
                raise RuntimeError(self.error())
            self.stream = torch.cuda.ExternalStream(pointer, device=self.device)
            # Native creation/reset and torch module transfers are initialization,
            # never part of the collection range or its zero-staging claim.
            self.stream.synchronize()
            goal_shape = (self.design_count, 3, height, width)
            specifications = [
                ((n, 3, height, width), "<f4"),        # 0 obs
                ((n, PROPRIO_WIDTH), "<f4"),           # 1 proprio
                ((n,), "<f4"),                         # 2 reward
                ((n,), "<f4"),                         # 3 done
                ((n,), "<f4"),                         # 4 success
                ((n, PRIVILEGED_WIDTH), "<f4"),        # 5 privileged
                ((n,), "<f4"),                         # 6 collision
                ((n,), "<u8"),                         # 7 episode counters
                ((n, DIAGNOSTICS_WIDTH), "<f4"),       # 8 diagnostics
                ((n,), "<i4"),                         # 9 design ids
                (goal_shape, "<f4"),                   # 10 goal images
                ((CURRICULUM_WIDTH,), "<f4"),          # 11 curriculum lanes (writable)
                ((n,), "<f4"),                         # 12 ended bucket, -1 if not done
                ((n,), "<f4"),                         # 13 out_of_bounds
                ((n,), "<f4"),                         # 14 ended gps_available
            ]
            with self.execution():
                for field, (shape, typestr) in enumerate(specifications):
                    address = self.lib.sim_env_buffer(self.handle, field)
                    if not address:
                        raise RuntimeError(f"field {field}: {self.error()}")
                    owner = _CudaAllocation(self, address, shape, typestr)
                    view = torch.as_tensor(owner, device=self.device)
                    if view.data_ptr() != address:
                        raise RuntimeError("CUDA array import unexpectedly copied native memory")
                    view._sim_env_owner = owner
                    self._owners.append(owner)
                    self.tensors.append(view)
            (self.obs, self.proprio, self.reward, self.done, self.success, self.privileged,
             self.collision, self.episode_counts, self.diagnostics, self.design, self.goal_images,
             self.curriculum, self.ended_bucket, self.out_of_bounds, self.ended_gps) = self.tensors
        except BaseException:
            self.close()
            raise

    def error(self):
        message = self.lib.sim_env_error()
        return message.decode("utf-8", errors="replace") if message else "sim_env failed without an error message"

    @contextmanager
    def execution(self):
        if not self.handle:
            raise RuntimeError("environment is closed")
        with torch.cuda.device(self.device), torch.cuda.stream(self.stream):
            yield self

    def _check_stream(self):
        if torch.cuda.current_stream(self.device).cuda_stream != self.stream.cuda_stream:
            raise RuntimeError("native and Torch work must share env.execution() ExternalStream")

    def reset(self, seed):
        self._check_stream()
        if self.lib.sim_env_reset(self.handle, seed) != 0:
            raise RuntimeError(self.error())

    def step(self, actions):
        self._check_stream()
        if (actions.device != self.device or actions.dtype != torch.float32
                or actions.shape != (self.n, self.action_dim) or not actions.is_contiguous()):
            raise ValueError("actions must be contiguous CUDA:0 float32[n, 4]")
        # The caller retains actions in its preallocated buffer for the entire
        # asynchronous native read. No .to(), host scalar, or hidden staging.
        if self.lib.sim_env_step(self.handle, ct.c_void_p(actions.data_ptr())) != 0:
            raise RuntimeError(self.error())

    def write_curriculum(self, lanes):
        """Enqueue an in-place device copy of [w_visible, w_partial, w_hidden, p_no_gps]."""
        self._check_stream()
        if lanes.device != self.device or lanes.shape != (CURRICULUM_WIDTH,) or lanes.dtype != torch.float32:
            raise ValueError("curriculum lanes must be CUDA:0 float32[4]")
        self.curriculum.copy_(lanes)

    def dlpack(self, field, stream=None):
        """Export a borrowed tensor through the standard stream-aware protocol."""
        return self.tensors[field].__dlpack__(stream=stream)

    def close(self):
        if getattr(self, "handle", None):
            if hasattr(self, "stream"):
                self.stream.synchronize()  # Explicit end-batch/lifetime boundary.
            self.lib.sim_env_destroy(self.handle)
            self.handle = None
            self._owners.clear()
            self.tensors.clear()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
