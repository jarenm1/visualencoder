// SPDX-License-Identifier: MIT OR Apache-2.0
//! Pure GPU mesh rendering; no adapter creation, CUDA context, or host readback.
//!
//! Construct the scene and [`RenderBindings`] once. Each frame, order CUDA body /
//! camera writes before Vulkan access using the interop owner's synchronization,
//! call [`Renderer::render`], then release the output to its GPU consumer. No queue
//! submission or ownership transfer is performed here. A renderer's attachments
//! are reused, so overlapping frames must be ordered on the same GPU queue.
//!
//! Body storage is an array of 64-byte, 16-byte-aligned structs containing four
//! `vec4<f32>` fields: position, velocity, xyzw rotation, angular_velocity. Position
//! is in metres, world +Y is up, and rotations must be unit quaternions. This is
//! the sim-state `BodyState` ABI, without a physics dependency.
mod camera;
mod mesh;
mod renderer;

pub use camera::{Camera, CameraGpu};
pub use mesh::{BaseColorImage, BaseColorTexture, MAX_LODS, MeshData, MeshLod, MeshMaterial, MeshVertex, WrapMode};
pub use renderer::{RENDER_TIMESTAMPS, RenderBindings, Renderer};

/// Surface parameters in linear RGB; the packed output is sRGB-encoded RGBA8.
#[derive(Clone, Copy, Debug)]
pub struct Material {
    pub color: [f32; 4],
    pub shininess: f32,
    pub specular: f32,
    /// World-space checker frequency. Zero disables the pattern.
    pub checker_scale: f32,
    pub checker_contrast: f32,
}

impl Default for Material {
    fn default() -> Self {
        Self {
            color: [0.7, 0.7, 0.7, 1.0],
            shininess: 32.0,
            specular: 0.25,
            checker_scale: 0.0,
            checker_contrast: 0.3,
        }
    }
}

/// Local mesh transform, optionally attached to a GPU body. Composition is
/// body_translation * body_rotation * local_translation * local_rotation * scale.
/// Mesh instances are GPU frustum-culled per camera and drawn with indexed multi-draw;
/// the camera filter is applied in the same cull pass.
#[derive(Clone, Copy, Debug)]
pub struct Instance {
    pub mesh: usize,
    pub body_index: Option<u32>,
    pub position: [f32; 3],
    pub rotation: [f32; 4],
    /// Nonzero positive dimensions; normals use the inverse transpose scale.
    pub scale: [f32; 3],
    pub material: Material,
    /// Zero is reserved for background.
    pub object_id: u32,
    /// `Some(c)` draws only into camera index `c` of the batch; `None` draws into every camera.
    pub camera: Option<u32>,
}

impl Instance {
    pub fn static_mesh(mesh: usize) -> Self {
        Self {
            mesh,
            body_index: None,
            position: [0.0; 3],
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [1.0; 3],
            material: Material::default(),
            object_id: 1,
            camera: None,
        }
    }

    pub fn body(mesh: usize, body_index: u32) -> Self {
        Self { body_index: Some(body_index), ..Self::static_mesh(mesh) }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Lighting {
    /// Direction from a surface toward the light.
    pub direction: [f32; 3],
    pub color: [f32; 3],
    pub ambient: [f32; 3],
}

impl Default for Lighting {
    fn default() -> Self {
        Self {
            direction: [0.5, 1.0, 0.4],
            color: [0.95, 0.91, 0.84],
            ambient: [0.16, 0.18, 0.22],
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct RenderConfig {
    pub width: u32,
    pub height: u32,
    pub lighting: Lighting,
    /// sRGB-encoded background, matching the packed output color space.
    pub background: [f64; 4],
    /// Level-of-detail budget: each camera draws a mesh's coarsest level whose geometric error
    /// projects to at most this many pixels at the mesh's nearest bounding-sphere point.
    /// Zero always draws the full mesh.
    pub lod_pixels: f32,
}

impl Default for RenderConfig {
    fn default() -> Self {
        Self {
            width: 320,
            height: 240,
            lighting: Lighting::default(),
            background: [0.12, 0.18, 0.25, 1.0],
            lod_pixels: 0.0,
        }
    }
}

/// Byte offsets in one caller-owned STORAGE buffer. Each region is camera-major,
/// top-left-origin, row-major, and tightly packed with no row padding. RGBA has
/// interleaved u8 channels; depth is f32 optical-Z in metres (not Euclidean range
/// or NDC depth); IDs are u32. Background depth and ID are zero. All scalars use
/// the native little-endian GPU layout. Each pixel occupies four bytes per region.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OutputLayout {
    pub width: u32,
    pub height: u32,
    pub cameras: u32,
    pub row_stride_bytes: u64,
    pub camera_stride_bytes: u64,
    pub rgba_offset_bytes: u64,
    pub depth_offset_bytes: u64,
    pub object_id_offset_bytes: u64,
    pub total_bytes: u64,
}
