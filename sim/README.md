# GPU-owned simulation

A separate Rust workspace derived from Triage's rendering and simulation concepts. Linux, NVIDIA Vulkan and CUDA only. No software-renderer or CPU-physics fallback.

## Crates

| Crate | Responsibility |
|---|---|
| `sim-state` | `no_std`, runtime-independent body/action/integration layouts; Y-up metres, xyzw quaternions |
| `sim-interop` | UUID-matched Vulkan/CUDA device, exportable allocations, FD ownership, stream/timeline synchronization |
| `sim-render` | Meshes, materials, cameras, GPU-state-driven rasterization and sensor packing; no physics/CUDA dependency |
| `sim-physics` | NVIDIA Rust CUDA rigid-body integration, RGBA→NCHW conversion and image reductions |
| `sim-env` | GPU-owned flight-through-gap task and borrowed-device-pointer C ABI for learning |
| `sim-smoke` | Executable consumer of the reusable primitives; explicit final image/correctness inspection |

The renderer adapts mesh, camera and lighting ideas from `jarenm1/triage`, particularly `crates/sim-graphics`. It does not copy Triage's Python staging adapter, web application or readback graph. Original attribution is retained in adapted source.

## Build

```sh
cd sim
nix develop
cargo host                          # ordinary incremental host-only build
cargo test -p sim-state -p sim-interop --lib
cargo test -p sim-interop --test exchange -- --ignored --test-threads=1
cargo oxide build --arch sm_120 --device-codegen-crate sim_physics -- -p sim-smoke
../.sim-target/debug/sim-smoke --frames 64 --output smoke.ppm
```

NVIDIA CUDA Rust and the compiler backend resolve to the same pinned commit. The shell supplies CUDA 13, Clang, Vulkan tools, mold and Rust 1.99.1 with compiler development components. `sim-physics` must be built with `cargo oxide`; ordinary Cargo does not silently produce a device-code-free substitute. Device kernels are Rust, not CUDA C++, handwritten PTX or WGSL physics.

Build outputs live in `../.sim-target`. Development uses opt-level 0, incremental compilation, 256 codegen units and no LTO; it does not inherit Triage's expensive optimized-dependency development profile. Compilation is limited to eight jobs to avoid exhausting a workstation while other workloads run.

### Fast host iteration

```sh
nix develop .#fast
sim-fast build
sim-fast test -p sim-state
```

`sim-fast` uses a separately pinned nightly Cranelift toolchain and separate target directory. It never changes the compiler used by `cargo oxide`. Cranelift's debug speed is for host iteration, not kernel optimization. Indexable's Blaze also replaces the host toolchain; it is not used to compile CUDA kernels or override NVIDIA's required compiler.

For optimized GPU execution:

```sh
cargo oxide build --arch sm_120 --device-codegen-crate sim_physics -- -p sim-smoke --profile training
../.sim-target/training/sim-smoke --frames 1024 --output smoke.ppm
```

The `training` profile uses opt-level 3, one codegen unit and line tables. Host LTO is disabled: NVIDIA's embedded CUDA artifact objects do not contain the `.llvmbc` section required by Rust's cross-crate LTO. Kernel build/JIT warmup is distinct from timed training.

## Ownership and synchronization

`Gpu` selects the requested CUDA ordinal and requires a Vulkan adapter with the same physical-device UUID. Its CUDA context retains the primary context, allowing PyTorch interoperability. `SharedBuffer` owns Vulkan device-local memory exported as an opaque FD and imported into CUDA. The FD is transferred only after successful CUDA import; borrowed CUDA pointers must never be freed independently.

`Exchange` exclusively enrolls its shared buffers. A cycle schedules:

1. wgpu rendering/compute on the owned Vulkan queue.
2. Vulkan EXTERNAL ownership release and timeline signal.
3. CUDA stream wait, caller-enqueued CUDA work, and timeline signal.
4. Vulkan ownership acquire before the next use.

The handoff reuses recorded barrier command buffers. It does not map pixels, copy state through the CPU, or wait on the host per frame. Startup and teardown may synchronize. `cycle` is unsafe because raw CUDA launches must obey the same-stream, exclusive-access and allocation-lifetime contract. Its returned wgpu submission index is **not** completion of the complete CUDA cycle; use an explicit batch-boundary `wait_idle` for inspection or shutdown.

## Sensor storage

For `N = cameras × height × width`, `sim-render` writes one storage buffer:

| Region | Byte offset | Bytes |
|---|---|---|
| sRGB RGBA8 | 0 | 4N |
| optical-Z depth, f32 metres | 4N | 4N |
| object IDs, u32 | 8N | 4N |

Images are camera-major, top-left origin, tightly packed rows. Background depth/ID are zero. CUDA conversion produces camera-major NCHW f32 RGB in `[0,1]`; model normalization remains the encoder's responsibility. This GPU packing/conversion is device work, not host staging.

For a learner, preprocess into **ordinary CUDA-owned storage**, not an exchange-owned buffer after it has returned to Vulkan. Wrap borrowed device addresses without copying, retain the native owner, and execute model/policy/native calls on the native CUDA stream. Actions are passed as device pointers. Host metrics and saved images belong outside rollout collection.

## Scope

The physics primitive is semi-implicit rigid-body integration with gravity, force, torque, damping and quaternion normalization. It does not claim a contact solver, rotor aerodynamics or a complete drone controller. Contact/task logic belongs in the environment consumer. The initial learning task isolates visual steering through a randomized gap; it is not evidence of real-world drone flight.

`sim-smoke` verifies actual rendered geometry, metric depth, object IDs, analytic translation, quaternion normalization and every converted pixel. Its final PPM and correctness readbacks are explicit inspection steps outside the hot loop. Profiler captures must likewise distinguish initialization/inspection from collection; GPU pointers alone are not proof of zero host staging.

Run `./scripts/check-validation.sh` inside the shell after the optimized smoke build to exercise the same draw and handoff path with `VK_LAYER_KHRONOS_validation`. The check fails on reported Vulkan errors even if the driver continues rendering. Stage outputs use direct struct constructors: function-local output structs triggered `VUID-StandaloneSpirv-None-10684` in Naga-generated SPIR-V.
