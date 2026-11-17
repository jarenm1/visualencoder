# GPU-owned simulation

A separate Rust workspace derived from Triage's rendering and simulation concepts. Linux, NVIDIA Vulkan and CUDA only. No software-renderer or CPU-physics fallback.

## Crates

| Crate | Responsibility |
|---|---|
| `sim-state` | `no_std` body/action/integration layouts, class-masked swept-sphere/BVH collision, quadrotor constants, `EpisodeSpec` and the shared body-mounted camera basis |
| `sim-interop` | UUID-matched Vulkan/CUDA device, exportable allocations, FD ownership, stream/timeline synchronization |
| `sim-render` | Textured meshes, procedural target shapes, GPU frustum culling with per-camera instance ownership, indirect draws, cameras and sensor packing; no physics/CUDA dependency |
| `sim-scene` | City GLB/material loading with semantic class tags, landable-plane check, BVH, fingerprint; 48 target designs, split districts and the visibility-bucketed episode catalog |
| `sim-physics` | NVIDIA Rust CUDA rigid-body integration, RGBA→NCHW conversion and image reductions |
| `sim-env` | GPU-owned quadrotor visual-goal task (CTBR dynamics, landing/crash, targets, curriculum sampling) and borrowed-device-pointer C ABI v3 |
| `sim-smoke` | Executable consumer of the reusable primitives; explicit final image/correctness inspection |

The renderer adapts mesh, camera and lighting ideas from `jarenm1/triage`, particularly `crates/sim-graphics`. It does not copy Triage's Python staging adapter, web application or readback graph. Original attribution is retained in adapted source.

## Build

```sh
cd sim
nix develop
cargo host                          # ordinary incremental host-only build
cargo test -p sim-state -p sim-interop -p sim-scene --lib
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

The city learning task (`city-drone-goal-v1`, ABI v3, see `crates/sim-env/sim_env.h`) is quadrotor flight with the controls a flight controller accepts: collective thrust plus roll/pitch/yaw body rates. CUDA integrates a 1 kg quadrotor at 240 Hz (8 substeps per 30 Hz action) with thrust lag, a saturated rate inner loop with gyroscopic torque, body-frame drag and gravity; the camera is rigidly mounted with 15° uptilt. Episodes start landed (take-off required) or already flying, and end on success (within 1.5 m of the target at ≤3 m/s), crash (any non-ground geometry, or touchdown faster than 2 m/s or tilted over 30°), leaving the district or the 40 m ceiling, or the 15 s horizon. Rotor aerodynamics, ground effect and battery are not modelled.

Each batch shares the actual 1,799,347-triangle city across its independent cameras. Rendering uses 16m spatial chunks, GPU frustum culling and persistent indirect draws. Collision uses the same transformed GLB triangles, not layout AABBs. The renderer imports palette textures and material factors; it does not reproduce the legacy renderer's Poly Haven normal maps or sun shadows. Texture sampling is nearest/bilinear without mipmaps.

Ground and road triangles are landable: every landable vertex lies within 0.25 m of y=0 (roads have 0.2 m curbs), so contact uses one analytic plane at the highest landable vertex and swept collision ignores those classes. Training, validation and test use disjoint districts and disjoint target designs (32/8/8). Each split has a fixed 65,536-episode catalog; visibility of the target from the start camera (64 surface samples, exact occlusion) sorts episodes into visible (≥50%), partial (5–50%) and hidden (<5%, including outside the view) buckets, and reset samples buckets from GPU-resident curriculum weights. Hidden episodes are common because start yaw is random. The policy gets a GPS-like hint with per-episode error up to 25% of the start distance; precise approach needs vision. See [`python/README.md`](python/README.md) for training and provenance.

`sim-smoke` verifies actual rendered geometry, metric depth, object IDs, analytic translation, quaternion normalization and every converted pixel. Its final PPM and correctness readbacks are explicit inspection steps outside the hot loop. Profiler captures must likewise distinguish initialization/inspection from collection; GPU pointers alone are not proof of zero host staging.

Run `./scripts/check-validation.sh` inside the shell after the optimized smoke build to exercise the same draw and handoff path with `VK_LAYER_KHRONOS_validation`. The check fails on reported Vulkan errors even if the driver continues rendering. Stage outputs use direct struct constructors: function-local output structs triggered `VUID-StandaloneSpirv-None-10684` in Naga-generated SPIR-V.
