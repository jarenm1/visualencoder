"""Borrowed CUDA tensors over the sim_env C ABI; no observation transport."""
from __future__ import annotations

import ctypes as ct
from contextlib import contextmanager
from pathlib import Path

import torch


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
    def __init__(self, library, n=64, seed=0, width=192, height=128, style=0):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required; no CPU environment fallback")
        if n < 1 or width < 48 or height < 32 or width % 16 or height % 16:
            raise ValueError("positive env count and image dimensions divisible by 16 required")
        torch.cuda.set_device(0)
        torch.cuda.init()  # Establish PyTorch's primary context before native imports.
        self.device = torch.device("cuda:0")
        self.n, self.width, self.height, self.style = n, width, height, style
        self.library = Path(library).resolve(strict=True)
        self.lib = ct.CDLL(str(self.library))
        declarations = {
            "sim_env_create": ([ct.c_uint32, ct.c_uint64, ct.c_uint32, ct.c_uint32, ct.c_uint32], ct.c_void_p),
            "sim_env_reset": ([ct.c_void_p, ct.c_uint64], ct.c_int32),
            "sim_env_step": ([ct.c_void_p, ct.c_void_p], ct.c_int32),
            "sim_env_buffer": ([ct.c_void_p, ct.c_int32], ct.c_uint64),
            "sim_env_stream": ([ct.c_void_p], ct.c_uint64),
            "sim_env_destroy": ([ct.c_void_p], None),
            "sim_env_error": ([], ct.c_char_p),
        }
        for name, (args, result) in declarations.items():
            function = getattr(self.lib, name)
            function.argtypes, function.restype = args, result
        self.handle = self.lib.sim_env_create(n, seed, width, height, style)
        if not self.handle:
            raise RuntimeError(self.error())
        self._owners, self.tensors = [], []
        try:
            pointer = self.lib.sim_env_stream(self.handle)
            if not pointer:
                raise RuntimeError(self.error())
            self.stream = torch.cuda.ExternalStream(pointer, device=self.device)
            # Native creation/reset and torch module transfers are initialization,
            # never part of the collection range or its zero-staging claim.
            self.stream.synchronize()
            specifications = [((n, 3, height, width), "<f4"), ((n, 8), "<f4"),
                              ((n,), "<f4"), ((n,), "<f4"), ((n,), "<f4"),
                              ((n, 4), "<f4"), ((n,), "<f4"), ((n,), "<u8")]
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
            (self.obs, self.proprio, self.reward, self.done, self.success,
             self.privileged, self.collision, self.episode_counts) = self.tensors
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
        if (actions.device != self.device or actions.dtype != torch.int64
                or actions.shape != (self.n,) or not actions.is_contiguous()):
            raise ValueError("actions must be contiguous CUDA:0 int64[n]")
        # The caller retains actions in its preallocated rollout for the entire
        # asynchronous native read. No .to(), host scalar, or hidden staging.
        if self.lib.sim_env_step(self.handle, ct.c_void_p(actions.data_ptr())) != 0:
            raise RuntimeError(self.error())

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
