// SPDX-License-Identifier: MIT OR Apache-2.0
// Camera matrices adapted from Triage crates/sim-graphics/src/camera/mod.rs.
// Triage is distributed under MIT OR Apache-2.0; see workspace provenance.
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};

/// Right-handed, Y-up camera. Visible view-space Z is negative.
#[derive(Clone, Copy, Debug)]
pub struct Camera {
    pub eye: Vec3,
    pub target: Vec3,
    pub up: Vec3,
    pub vertical_fov_radians: f32,
    pub near: f32,
    pub far: f32,
}

impl Camera {
    pub fn view(self) -> Mat4 {
        Mat4::look_at_rh(self.eye, self.target, self.up)
    }

    /// WebGPU/Vulkan depth range 0..1, not OpenGL's -1..1.
    pub fn view_projection(self, aspect: f32) -> Mat4 {
        Mat4::perspective_rh(self.vertical_fov_radians, aspect, self.near, self.far) * self.view()
    }

    pub fn gpu(self, aspect: f32) -> CameraGpu {
        CameraGpu {
            view: self.view().to_cols_array_2d(),
            view_projection: self.view_projection(aspect).to_cols_array_2d(),
            eye: self.eye.extend(1.0).to_array(),
        }
    }
}

/// Device-owned camera array element: 144 bytes, aligned to 16 bytes.
/// Matrices are column-major. CUDA may write this layout directly before Vulkan
/// acquires the buffer. `view` must be rigid for metric optical-Z output.
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct CameraGpu {
    pub view: [[f32; 4]; 4],
    pub view_projection: [[f32; 4]; 4],
    pub eye: [f32; 4],
}

const _: () = assert!(size_of::<CameraGpu>() == 144);
