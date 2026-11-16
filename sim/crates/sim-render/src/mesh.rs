// SPDX-License-Identifier: MIT OR Apache-2.0
// Cube/plane generation adapted from Triage crates/sim-graphics/src/primitive.rs.
// Triage is distributed under MIT OR Apache-2.0; see workspace provenance.
use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct MeshVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
}

pub struct MeshData {
    pub vertices: Vec<MeshVertex>,
    pub indices: Vec<u32>,
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
            vertices.extend(positions.into_iter().map(|position| MeshVertex { position, normal }));
        }
        let mut indices = Vec::with_capacity(36);
        for face in 0..6_u32 {
            let base = face * 4;
            indices.extend_from_slice(&[base, base + 2, base + 1, base, base + 3, base + 2]);
        }
        Self { vertices, indices }
    }

    /// Unit XZ plane centered on the origin, with its front face pointing +Y.
    pub fn plane() -> Self {
        Self {
            vertices: vec![
                MeshVertex { position: [-0.5, 0.0, -0.5], normal: [0.0, 1.0, 0.0] },
                MeshVertex { position: [0.5, 0.0, -0.5], normal: [0.0, 1.0, 0.0] },
                MeshVertex { position: [0.5, 0.0, 0.5], normal: [0.0, 1.0, 0.0] },
                MeshVertex { position: [-0.5, 0.0, 0.5], normal: [0.0, 1.0, 0.0] },
            ],
            indices: vec![0, 2, 1, 0, 3, 2],
        }
    }
}
