// SPDX-License-Identifier: MIT OR Apache-2.0
// Cube/plane generation adapted from Triage crates/sim-graphics/src/primitive.rs.
// Triage is distributed under MIT OR Apache-2.0; see workspace provenance.
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;

/// RGBA8 sRGB image, shared by all meshes using the same imported image.
#[derive(Debug)]
pub struct BaseColorImage {
    pub width: u32,
    pub height: u32,
    pub rgba8: Vec<u8>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum WrapMode { Repeat, MirroredRepeat, ClampToEdge }

#[derive(Clone, Debug)]
pub struct BaseColorTexture {
    pub image: Arc<BaseColorImage>,
    pub wrap_u: WrapMode,
    pub wrap_v: WrapMode,
    pub linear: bool,
}

/// glTF-style opaque surface. Factors are linear; texture texels are sRGB.
#[derive(Clone, Debug)]
pub struct MeshMaterial {
    pub base_color_factor: [f32; 4],
    pub base_color_texture: Option<BaseColorTexture>,
    pub metallic_factor: f32,
    pub roughness_factor: f32,
    pub double_sided: bool,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct MeshVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
}

pub struct MeshData {
    pub vertices: Vec<MeshVertex>,
    pub indices: Vec<u32>,
    /// None retains the instance-only material used by procedural smoke scenes.
    pub material: Option<MeshMaterial>,
}

impl MeshData {
    /// Unit cube centered on the origin, with separate vertices for flat normals.
    pub fn cube() -> Self {
        let faces = [
            ([1.0, 0.0, 0.0], [[0.5, -0.5, -0.5], [0.5, -0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, -0.5]]),
            ([-1.0, 0.0, 0.0], [[-0.5, -0.5, 0.5], [-0.5, -0.5, -0.5], [-0.5, 0.5, -0.5], [-0.5, 0.5, 0.5]]),
            ([0.0, 1.0, 0.0], [[-0.5, 0.5, -0.5], [0.5, 0.5, -0.5], [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5]]),
            ([0.0, -1.0, 0.0], [[-0.5, -0.5, 0.5], [0.5, -0.5, 0.5], [0.5, -0.5, -0.5], [-0.5, -0.5, -0.5]]),
            ([0.0, 0.0, 1.0], [[0.5, -0.5, 0.5], [-0.5, -0.5, 0.5], [-0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]),
            ([0.0, 0.0, -1.0], [[-0.5, -0.5, -0.5], [0.5, -0.5, -0.5], [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5]]),
        ];
        let mut vertices = Vec::with_capacity(24);
        for (normal, positions) in faces {
            vertices.extend(positions.into_iter().map(|position| MeshVertex { position, normal, uv: [0.0; 2] }));
        }
        let mut indices = Vec::with_capacity(36);
        for face in 0..6_u32 {
            let base = face * 4;
            indices.extend_from_slice(&[base, base + 2, base + 1, base, base + 3, base + 2]);
        }
        Self { vertices, indices, material: None }
    }

    /// Unit XZ plane centered on the origin, with its front face pointing +Y.
    pub fn plane() -> Self {
        Self {
            vertices: vec![
                MeshVertex { position: [-0.5, 0.0, -0.5], normal: [0.0, 1.0, 0.0], uv: [0.0, 0.0] },
                MeshVertex { position: [0.5, 0.0, -0.5], normal: [0.0, 1.0, 0.0], uv: [1.0, 0.0] },
                MeshVertex { position: [0.5, 0.0, 0.5], normal: [0.0, 1.0, 0.0], uv: [1.0, 1.0] },
                MeshVertex { position: [-0.5, 0.0, 0.5], normal: [0.0, 1.0, 0.0], uv: [0.0, 1.0] },
            ],
            indices: vec![0, 2, 1, 0, 3, 2],
            material: None,
        }
    }
}
