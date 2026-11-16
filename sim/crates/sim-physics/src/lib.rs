//! Rust CUDA primitives for externally owned GPU storage.
//!
//! Build the consuming executable with NVIDIA cuda-oxide, for example:
//! `cargo oxide build --arch sm_120 --device-codegen-crate sim_physics -- -p sim-smoke --release`.
//! Kernels live in this library, are compiled from Rust, and travel in its
//! embedded artifact section. There is no CPU, WGSL, or prewritten PTX fallback.
//!
//! Load [`Physics`] once using the interop CUDA context. Launch methods borrow
//! raw device addresses, enqueue on the supplied stream, and return immediately:
//! no allocation, host transfer, synchronization, or implicit ownership change.
//! The owner must retain allocations and this module until all queued uses finish.

mod image;
mod kernels;

use std::{ffi::c_void, sync::Arc};

use cuda_core::{CudaContext, CudaStream, Function, launch_kernel_on_stream};
use sim_state::{BodyAction, BodyState, StepParams};

pub use image::{ImageShape, ImageStatistics};

#[derive(Debug, thiserror::Error)]
pub enum PhysicsError {
    #[error(transparent)]
    Module(#[from] cuda_host::EmbeddedModuleError),
    #[error(transparent)]
    Driver(#[from] cuda_core::DriverError),
    #[error("image dimensions must be positive with total pixels fitting u32 and cameras fitting a CUDA x-grid")]
    InvalidImageShape,
}

/// Loaded CUDA kernels and their owning module, reusable across streams in one
/// context. Loading may compile embedded IR; it belongs outside the frame loop.
pub struct Physics {
    integrate: Function,
    preprocess: Function,
    statistics: Function,
}

impl Physics {
    /// Load this library's embedded CUDA bundle in the interop-owned context.
    pub fn load(context: &Arc<CudaContext>) -> Result<Self, PhysicsError> {
        // Generated load() references the library artifact anchor, ensuring the
        // linker retains device code even though the kernels live in an rlib.
        let loaded = kernels::device::load(context)?;
        let module = loaded.as_cuda_module();
        Ok(Self {
            integrate: module.load_function(kernels::device::integrate_CUDA_SIGNATURE.entry)?,
            preprocess: module.load_function(kernels::device::rgba8_to_nchw_CUDA_SIGNATURE.entry)?,
            statistics: module.load_function(kernels::device::reduce_images_CUDA_SIGNATURE.entry)?,
        })
    }

    /// Advance each independent body once using semi-implicit Euler and a
    /// normalized quaternion update. No contact solver or aerodynamics is applied.
    /// A zero count enqueues nothing and permits null pointers.
    ///
    /// # Safety
    /// `stream` must belong to the context used by [`Self::load`]. `states` must
    /// address `count` aligned, initialized, exclusively writable [`BodyState`]s;
    /// `actions` must address that many aligned, initialized [`BodyAction`]s in a
    /// disjoint allocation. State/actions must be finite, quaternions nonzero,
    /// and parameters chosen so intermediate values remain finite and the
    /// quaternion squared norm stays positive and representable. Construct
    /// parameters with [`StepParams::new`] (or use its valid default).
    ///
    /// Caller must acquire external/Vulkan ownership before launch, prevent
    /// conflicting access, and retain memory and `self` until stream completion.
    /// A later external consumer needs a stream-ordered release/signal; this call
    /// does not wait for or signal graphics work and does not retain raw buffers.
    pub unsafe fn step(
        &self,
        stream: &CudaStream,
        mut states: *mut BodyState,
        mut actions: *const BodyAction,
        mut count: u32,
        mut params: StepParams,
    ) -> Result<(), PhysicsError> {
        if count == 0 {
            return Ok(());
        }
        let mut arguments = [
            (&raw mut states).cast::<c_void>(),
            (&raw mut actions).cast::<c_void>(),
            (&raw mut count).cast::<c_void>(),
            (&raw mut params).cast::<c_void>(),
        ];
        // SAFETY: argument storage matches the concrete kernel's by-value ABI;
        // geometry owns one disjoint body per thread. Caller ensures lifetimes.
        unsafe {
            launch_kernel_on_stream(
                &self.integrate,
                (count.div_ceil(256), 1, 1),
                (256, 1, 1),
                0,
                stream,
                &mut arguments,
            )?;
        }
        Ok(())
    }

    /// Convert the leading RGBA8 region of a shared render output into a separate
    /// tightly packed NCHW f32 RGB buffer on the GPU. RGB values are scaled by
    /// 1/255, without gamma conversion; alpha, depth and object IDs are ignored.
    ///
    /// # Safety
    /// `stream` must use this module's context. `rgba` must cover at least
    /// `shape.rgba_bytes()` initialized device bytes; `rgb` must cover at least
    /// `shape.rgb_bytes()` exclusively writable device bytes aligned for f32.
    /// The input and output extents must not overlap. The ownership, exclusion,
    /// external synchronization and completion-lifetime contract of [`Self::step`]
    /// also applies, including retaining `self` until execution completes.
    pub unsafe fn preprocess_rgba8(
        &self,
        stream: &CudaStream,
        mut rgba: *const u8,
        mut rgb: *mut f32,
        shape: ImageShape,
    ) -> Result<(), PhysicsError> {
        let mut count = shape.pixels();
        let mut plane = shape.pixels_per_image();
        let mut arguments = [
            (&raw mut rgba).cast::<c_void>(),
            (&raw mut rgb).cast::<c_void>(),
            (&raw mut count).cast::<c_void>(),
            (&raw mut plane).cast::<c_void>(),
        ];
        // SAFETY: shape validates dimensions and the caller ensures storage.
        unsafe {
            launch_kernel_on_stream(
                &self.preprocess,
                (count.div_ceil(256), 1, 1),
                (256, 1, 1),
                0,
                stream,
                &mut arguments,
            )?;
        }
        Ok(())
    }

    /// Reduce preprocessed images to one GPU-resident result per camera. Each
    /// 256-thread block scans one image, then reduces in block-local storage.
    /// No scratch allocation, atomics, or initialization of the output is needed.
    ///
    /// # Safety
    /// `rgb` must hold `shape.rgb_bytes()` initialized, finite NCHW f32 RGB in
    /// [0,1]. `statistics` must point to `shape.cameras()` aligned, exclusively
    /// writable [`ImageStatistics`] elements, disjoint from RGB input. The same
    /// context, external ownership, exclusion and completion-lifetime obligations
    /// as [`Self::step`] apply. Ordering on the same CUDA stream is sufficient
    /// after [`Self::preprocess_rgba8`]; no host synchronization is required.
    pub unsafe fn image_statistics(
        &self,
        stream: &CudaStream,
        mut rgb: *const f32,
        mut statistics: *mut ImageStatistics,
        shape: ImageShape,
    ) -> Result<(), PhysicsError> {
        let mut plane = shape.pixels_per_image();
        let mut arguments = [
            (&raw mut rgb).cast::<c_void>(),
            (&raw mut statistics).cast::<c_void>(),
            (&raw mut plane).cast::<c_void>(),
        ];
        // SAFETY: exact reduction block geometry; the caller ensures storage.
        unsafe {
            launch_kernel_on_stream(
                &self.statistics,
                (shape.cameras(), 1, 1),
                (256, 1, 1),
                0,
                stream,
                &mut arguments,
            )?;
        }
        Ok(())
    }
}
