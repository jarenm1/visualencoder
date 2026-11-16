# Provenance

- Triage: `jarenm1/triage`, local source revision `e85785a` (reported by the existing checkout). Workspace packages declare `MIT OR Apache-2.0`. `sim-render/src/mesh.rs` adapts cube/plane generation from `crates/sim-graphics/src/primitive.rs`; `camera.rs` adapts camera matrices from `crates/sim-graphics/src/camera/mod.rs`. Lighting and instancing reuse the design, not the staged transport. Local Triage files are not modified by this workspace.
- NVIDIA CUDA Rust: dependency/compiler revision `348895e421a92d13a2c3a843ccb20ed1fb8b7f8b`, Apache-2.0. Cargo and Nix lockfiles pin the sources. Kernels in this workspace are Rust compiled by NVIDIA `cuda-oxide`; NVIDIA source is consumed as a dependency, not copied into the workspace.
- wgpu/wgpu-hal: 30.x, pinned by Cargo.lock, MIT OR Apache-2.0. Vulkan operations use `ash` 0.38 and the wgpu-hal Vulkan escape hatch.
- The external-memory/timeline exchange, task implementation, native ABI and executable are new workspace code. No C++ CUDA source, legacy Triage CUDA build tree, Python host-staging environment or complete Triage repository is vendored.
