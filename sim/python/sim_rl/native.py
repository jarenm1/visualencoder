"""Borrowed CUDA tensors over the sim_env C ABI v7; no observation transport."""
from __future__ import annotations

import ctypes as ct
import json
import os
from contextlib import contextmanager
from pathlib import Path

import torch

DEFAULT_CITY = Path(__file__).resolve().parents[3] / "assets" / "city"
TASK = "city-drone-goal-v5"
ABI_VERSION = 8
ACTION_DIM = 4
HORIZON = 1200
CONTROL_HZ = 30
CATALOG_SIZE = 65536                   # native default episode catalogue per environment
DEFAULT_START_DISTANCE = (8.0, 30.0)   # native default start distance (m from the target box)
CONTROL_MODES = {"velocity": 0, "rates": 1}
# Zero velocity commands hover; rates mode keeps the v3 hover throttle bias.
HOVER_ACTIONS = {"velocity": (0.0, 0.0, 0.0, 0.0), "rates": (-0.2, 0.0, 0.0, 0.0)}
# Fields: 0 obs, 1 proprio, 2 reward, 3 done, 4 success, 5 privileged, 6 collision,
# 7 episodes, 8 diagnostics, 9 catalog index, 10 goal views, 11 curriculum (writable),
# 12 ended bucket, 13 out_of_bounds, 14 ended hint mode, 15 goal crops, 16 episode diagnostics,
# 17 patch depth, 18 patch object (both training labels, never actor input).
FIELD_COUNT = 19
PATCH = 16
DIAGNOSTICS_WIDTH = 28
# w_visible, w_partial, w_hidden, p_no_gps (no hint), appearance strength, p_coarse, p_wrong (of hinted)
CURRICULUM_WIDTH = 7
PROPRIO_WIDTH = 20
HINT_MODES = ("none", "precise", "coarse", "wrong")
PRIVILEGED_WIDTH = 29
REQUIRED_METADATA = ("abi_version", "task", "action_dim", "control_hz", "physics_hz", "horizon",
                     "catalog_size", "city_fingerprint", "city_path", "control_mode", "action_names",
                     "action_scales", "target_classes")
# Which city (and where in it) an environment simulates is not part of the policy contract:
# multi-city training and held-out-city evaluation depend on that.
SPLIT_SPECIFIC_METADATA = ("city_path", "city", "city_fingerprint", "landable", "style", "split", "district",
                           "catalog", "catalog_size", "fields")


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
    def __init__(self, library, n, seed, width, height, style, control_mode, city_path=DEFAULT_CITY,
                 start_distance=None, catalog_size=None):
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
            "sim_env_render_goals": ([ct.c_void_p, ct.c_uint32], ct.c_int32),
            "sim_env_retarget": ([ct.c_void_p, ct.c_void_p], ct.c_int32),
            "sim_env_set_free_flight": ([ct.c_void_p, ct.c_uint32], ct.c_int32),
            "sim_env_create_with_options": ([ct.c_uint32, ct.c_uint64, ct.c_uint32, ct.c_uint32,
                ct.c_uint32, ct.c_uint32, ct.c_char_p, ct.c_float, ct.c_float, ct.c_uint32], ct.c_void_p),
            "sim_env_destroy": ([ct.c_void_p], None),
            "sim_env_error": ([], ct.c_char_p),
        }
        for name, (args, result) in declarations.items():
            try:
                function = getattr(self.lib, name)
            except AttributeError as error:
                raise RuntimeError(f"city navigation requires sim_env ABI v{ABI_VERSION} symbol {name}") from error
            function.argtypes, function.restype = args, result
        create_args = (n, seed, width, height, style, CONTROL_MODES[control_mode], os.fsencode(self.city_path))
        if start_distance is None and catalog_size is None:
            self.handle = self.lib.sim_env_create(*create_args)
        else:
            self.handle = self.lib.sim_env_create_with_options(
                *create_args, *(start_distance or DEFAULT_START_DISTANCE), catalog_size or CATALOG_SIZE)
        if not self.handle:
            raise RuntimeError(self.error())
        self._owners, self.tensors = [], []
        try:
            raw_metadata = self.lib.sim_env_metadata(self.handle)
            if not raw_metadata:
                raise RuntimeError(f"native city metadata unavailable: {self.error()}")
            self.metadata = json.loads(raw_metadata.decode("utf-8"))
            if self.metadata.get("abi_version") != ABI_VERSION or self.metadata.get("task") != TASK:
                raise ValueError(f"city navigation requires sim_env ABI v{ABI_VERSION} task {TASK}; "
                                 "older libraries are incompatible")
            missing = [key for key in REQUIRED_METADATA if key not in self.metadata]
            if missing:
                raise ValueError(f"incomplete native city metadata: {missing}")
            if self.metadata["control_mode"] != control_mode:
                raise ValueError(f"native control_mode {self.metadata['control_mode']!r} != {control_mode!r}")
            self.action_dim = int(self.metadata["action_dim"])
            self.horizon = int(self.metadata["horizon"])
            self.control_hz = int(self.metadata["control_hz"])
            self.catalog_size = int(self.metadata["catalog_size"])
            self.city_fingerprints = (self.metadata["city_fingerprint"],)
            self.target_classes = tuple(self.metadata["target_classes"])
            self.layout_classes = tuple(self.metadata["city"]["classes"])
            self.action_names = tuple(self.metadata["action_names"])
            self.action_scales = tuple(float(value) for value in self.metadata["action_scales"])
            if (self.action_dim, self.horizon, self.control_hz) != (ACTION_DIM, HORIZON, CONTROL_HZ):
                raise ValueError(f"unsupported {TASK} action/horizon/control contract")
            if len(self.action_names) != ACTION_DIM or len(self.action_scales) != ACTION_DIM:
                raise ValueError("native metadata must name and scale exactly four control channels")
            # Location and split-specific content (city, district, catalog, field shapes) are
            # excluded; dynamics, sensors, reward, camera, appearance, control mode and every
            # other rule must match checkpoints exactly.
            self.task_contract = {key: value for key, value in self.metadata.items()
                                  if key not in SPLIT_SPECIFIC_METADATA}
            pointer = self.lib.sim_env_stream(self.handle)
            if not pointer:
                raise RuntimeError(self.error())
            self.stream = torch.cuda.ExternalStream(pointer, device=self.device)
            # Native creation/reset and torch module transfers are initialization,
            # never part of the collection range or its zero-staging claim.
            self.stream.synchronize()
            image_shape = (n, 3, height, width)
            patches = (height // PATCH) * (width // PATCH)
            specifications = [
                (image_shape, "<f4"),                  # 0 obs
                ((n, PROPRIO_WIDTH), "<f4"),           # 1 proprio
                ((n,), "<f4"),                         # 2 reward
                ((n,), "<f4"),                         # 3 done
                ((n,), "<f4"),                         # 4 success
                ((n, PRIVILEGED_WIDTH), "<f4"),        # 5 privileged
                ((n,), "<f4"),                         # 6 collision
                ((n,), "<u8"),                         # 7 episode counters
                ((n, DIAGNOSTICS_WIDTH), "<f4"),       # 8 diagnostics
                ((n,), "<i4"),                         # 9 catalog index
                (image_shape, "<f4"),                  # 10 goal views
                ((CURRICULUM_WIDTH,), "<f4"),          # 11 curriculum lanes (writable)
                ((n,), "<f4"),                         # 12 ended bucket, -1 if not done
                ((n,), "<f4"),                         # 13 out_of_bounds
                ((n,), "<f4"),                         # 14 ended hint mode (HINT_MODES index), -1 if not done
                ((n, 4), "<f4"),                       # 15 goal crops
                ((n, 3), "<f4"),                       # 16 episode diagnostics, never actor input
                ((n, patches), "<f4"),                 # 17 patch depth (label)
                ((n, patches), "<i4"),                 # 18 patch object (label)
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
             self.collision, self.episode_counts, self.diagnostics, self.catalog_index, self.goal_views,
             self.curriculum, self.ended_bucket, self.out_of_bounds, self.ended_hint, self.goal_crops,
             self.episode_diagnostics, self.patch_depth, self.patch_object) = self.tensors
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

    def render_goals(self, first):
        """Enqueue goal reference views of catalog entries first..first+n-1 into goal_views/goal_crops."""
        self._check_stream()
        if self.lib.sim_env_render_goals(self.handle, first) != 0:
            raise RuntimeError(self.error())

    def retarget(self, targets):
        """Replace current-episode targets: CUDA float32 [n, 8] rows [centre xyz, layout object
        (< 0 keeps the catalog target), half xyz, layout class]. Kept alive by the caller until done."""
        self._check_stream()
        if (targets.device != self.device or targets.dtype != torch.float32 or targets.shape != (self.n, 8)
                or not targets.is_contiguous()):
            raise ValueError("targets must be contiguous CUDA:0 float32[n, 8]")
        if self.lib.sim_env_retarget(self.handle, ct.c_void_p(targets.data_ptr())) != 0:
            raise RuntimeError(self.error())

    def set_free_flight(self, enabled):
        """Interactive free flight (never training): only the city edge and a 100 m ceiling end a
        flight, success is reported once without ending the episode, and there is no timeout."""
        if self.lib.sim_env_set_free_flight(self.handle, int(bool(enabled))) != 0:
            raise RuntimeError(self.error())

    def write_curriculum(self, lanes):
        """Enqueue an in-place device copy of the CURRICULUM_WIDTH lanes (curriculum.py)."""
        self._check_stream()
        if lanes.device != self.device or lanes.shape != (CURRICULUM_WIDTH,) or lanes.dtype != torch.float32:
            raise ValueError(f"curriculum lanes must be CUDA:0 float32[{CURRICULUM_WIDTH}]")
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
