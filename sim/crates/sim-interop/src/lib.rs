//! Linux NVIDIA Vulkan/CUDA zero-copy buffers.
//!
//! [`Exchange`] schedules Vulkan → CUDA → Vulkan ownership transfers. Its CUDA
//! callback only enqueues work; neither frame handoff nor buffer access maps or
//! copies data through the host. Allocation and destruction may wait for the GPU.
//!
//! Raw CUDA pointers are borrowed, never owned. Keep their [`SharedBuffer`] and
//! its exchange alive, launch on [`Gpu::cuda_stream`], and do not access shared
//! buffers outside the exchange's Vulkan encoder or CUDA callback.

#![deny(unsafe_op_in_unsafe_fn)]

#[cfg(not(target_os = "linux"))]
compile_error!("sim-interop requires Linux OPAQUE_FD Vulkan/CUDA interop");

mod context;
mod memory;
mod sync;

pub use context::Gpu;
pub use memory::SharedBuffer;
pub use sync::Exchange;

pub(crate) fn cuda_result(code: cuda_bindings::CUresult, operation: &str) -> anyhow::Result<()> {
    if code == cuda_bindings::cudaError_enum_CUDA_SUCCESS {
        Ok(())
    } else {
        Err(anyhow::anyhow!("{operation} failed with CUDA driver status {code}"))
    }
}
