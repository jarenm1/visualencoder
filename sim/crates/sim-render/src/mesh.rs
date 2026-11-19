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
    /// Coarser index lists over the same `vertices`, in increasing `error` (at most MAX_LODS).
    pub lods: Vec<MeshLod>,
}

/// One simplified level of detail: `indices` reference the parent mesh's vertices and `error`
/// bounds its geometric deviation from the full mesh in mesh units (metres for the city).
pub struct MeshLod {
    pub indices: Vec<u32>,
    pub error: f32,
}

/// Levels of detail per mesh beyond the full mesh (renderer DrawInfo and cull.wgsl ABI).
pub const MAX_LODS: usize = 4;

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
        Self { vertices, indices, material: None, lods: Vec::new() }
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
            lods: Vec::new(),
        }
    }

    /// Unit sphere, diameter 1, centered on the origin. `segments` wraps around Y;
    /// `rings` runs pole to pole. Normals are radial; winding matches `cube()`.
    pub fn sphere(segments: u32, rings: u32) -> Self {
        assert!(segments >= 3 && rings >= 2, "sphere needs at least 3 segments and 2 rings");
        let mut vertices = Vec::with_capacity(((rings + 1) * (segments + 1)) as usize);
        for ring in 0..=rings {
            let phi = std::f32::consts::PI * ring as f32 / rings as f32;
            for segment in 0..=segments {
                let theta = std::f32::consts::TAU * segment as f32 / segments as f32;
                let normal = [phi.sin() * theta.cos(), phi.cos(), phi.sin() * theta.sin()];
                vertices.push(MeshVertex {
                    position: normal.map(|c| 0.5 * c),
                    normal,
                    uv: [segment as f32 / segments as f32, ring as f32 / rings as f32],
                });
            }
        }
        let at = |ring: u32, segment: u32| ring * (segments + 1) + segment;
        let mut indices = Vec::with_capacity((rings * segments * 6) as usize);
        for ring in 0..rings {
            for segment in 0..segments {
                let a = at(ring, segment);
                let b = at(ring + 1, segment);
                let c = at(ring + 1, segment + 1);
                let d = at(ring, segment + 1);
                // Pole rings collapse one edge of their quads; skip the zero-area triangle.
                if ring != 0 { indices.extend_from_slice(&[a, d, c]); }
                if ring + 1 != rings { indices.extend_from_slice(&[a, c, b]); }
            }
        }
        Self { vertices, indices, material: None, lods: Vec::new() }
    }

    /// Unit cylinder: radius 0.5, height 1 along Y, centered on the origin, with flat caps.
    pub fn cylinder(segments: u32) -> Self {
        assert!(segments >= 3, "cylinder needs at least 3 segments");
        let mut vertices = Vec::with_capacity(((segments + 1) * 2 + 2 * (segments + 1)) as usize);
        let mut indices = Vec::with_capacity((segments * 12) as usize);
        for segment in 0..=segments {
            let theta = std::f32::consts::TAU * segment as f32 / segments as f32;
            let (sin, cos) = theta.sin_cos();
            let u = segment as f32 / segments as f32;
            let normal = [cos, 0.0, sin];
            vertices.push(MeshVertex { position: [0.5 * cos, -0.5, 0.5 * sin], normal, uv: [u, 0.0] });
            vertices.push(MeshVertex { position: [0.5 * cos, 0.5, 0.5 * sin], normal, uv: [u, 1.0] });
        }
        for segment in 0..segments {
            let bottom = 2 * segment;
            let top = bottom + 1;
            let next_bottom = bottom + 2;
            let next_top = bottom + 3;
            indices.extend_from_slice(&[bottom, top, next_top, bottom, next_top, next_bottom]);
        }
        push_disc(&mut vertices, &mut indices, segments, 0.5, 1.0);
        push_disc(&mut vertices, &mut indices, segments, -0.5, -1.0);
        Self { vertices, indices, material: None, lods: Vec::new() }
    }

    /// Unit cone: base radius 0.5 at y = -0.5, apex at y = +0.5, with a flat base.
    pub fn cone(segments: u32) -> Self {
        assert!(segments >= 3, "cone needs at least 3 segments");
        let mut vertices = Vec::with_capacity(((segments + 1) * 2 + segments + 1) as usize);
        let mut indices = Vec::with_capacity((segments * 6) as usize);
        // Surface r = 0.5 - y/2 has gradient (cos, 0.5, sin); normalise by sqrt(1.25).
        let k = 1.25f32.sqrt().recip();
        for segment in 0..=segments {
            let theta = std::f32::consts::TAU * segment as f32 / segments as f32;
            let (sin, cos) = theta.sin_cos();
            let u = segment as f32 / segments as f32;
            let normal = [cos * k, 0.5 * k, sin * k];
            vertices.push(MeshVertex { position: [0.5 * cos, -0.5, 0.5 * sin], normal, uv: [u, 0.0] });
            vertices.push(MeshVertex { position: [0.0, 0.5, 0.0], normal, uv: [u, 1.0] });
        }
        for segment in 0..segments {
            let base = 2 * segment;
            let apex = base + 1;
            let next_base = base + 2;
            indices.extend_from_slice(&[base, apex, next_base]);
        }
        push_disc(&mut vertices, &mut indices, segments, -0.5, -1.0);
        Self { vertices, indices, material: None, lods: Vec::new() }
    }

    /// Unit octahedron with vertices at ±0.5 on the coordinate axes; flat-shaded faces.
    pub fn octahedron() -> Self {
        let mut vertices = Vec::with_capacity(24);
        let mut indices = Vec::with_capacity(24);
        for sx in [-1.0f32, 1.0] {
            for sy in [-1.0f32, 1.0] {
                for sz in [-1.0f32, 1.0] {
                    let normal = [sx, sy, sz].map(|c| c / 3f32.sqrt());
                    let x = [0.5 * sx, 0.0, 0.0];
                    let y = [0.0, 0.5 * sy, 0.0];
                    let z = [0.0, 0.0, 0.5 * sz];
                    // (X, Y, Z) winds along sx*sy*sz; odd sign parity needs the opposite order.
                    let corners = if sx * sy * sz > 0.0 { [x, y, z] } else { [x, z, y] };
                    let base = vertices.len() as u32;
                    for (position, uv) in corners.into_iter().zip([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]) {
                        vertices.push(MeshVertex { position, normal, uv });
                    }
                    indices.extend_from_slice(&[base, base + 1, base + 2]);
                }
            }
        }
        Self { vertices, indices, material: None, lods: Vec::new() }
    }

    /// Unit torus in the XZ plane, centered on the origin; outer extent is exactly 0.5 in x and z.
    /// `segments` wraps the major circle, `tube_segments` wraps the tube. `ratio` is the tube
    /// radius divided by the major radius and must be positive.
    pub fn torus(segments: u32, tube_segments: u32, ratio: f32) -> Self {
        assert!(segments >= 3 && tube_segments >= 3, "torus needs at least 3 segments in each direction");
        assert!(ratio.is_finite() && ratio > 0.0, "torus ratio must be finite and positive");
        let major = 0.5 / (1.0 + ratio);
        let tube = ratio * major;
        let mut vertices = Vec::with_capacity(((segments + 1) * (tube_segments + 1)) as usize);
        for i in 0..=segments {
            let (sin_u, cos_u) = (std::f32::consts::TAU * i as f32 / segments as f32).sin_cos();
            for j in 0..=tube_segments {
                let (sin_v, cos_v) = (std::f32::consts::TAU * j as f32 / tube_segments as f32).sin_cos();
                let normal = [cos_v * cos_u, sin_v, cos_v * sin_u];
                let ring = major + tube * cos_v;
                vertices.push(MeshVertex {
                    position: [ring * cos_u, tube * sin_v, ring * sin_u],
                    normal,
                    uv: [i as f32 / segments as f32, j as f32 / tube_segments as f32],
                });
            }
        }
        let at = |i: u32, j: u32| i * (tube_segments + 1) + j;
        let mut indices = Vec::with_capacity((segments * tube_segments * 6) as usize);
        for i in 0..segments {
            for j in 0..tube_segments {
                let a = at(i, j);
                let b = at(i + 1, j);
                let c = at(i + 1, j + 1);
                let d = at(i, j + 1);
                indices.extend_from_slice(&[a, c, b, a, d, c]);
            }
        }
        Self { vertices, indices, material: None, lods: Vec::new() }
    }
 }

/// Flat cap of radius 0.5 at height `y`. `normal_y` is +1 or -1 and fixes the winding.
fn push_disc(vertices: &mut Vec<MeshVertex>, indices: &mut Vec<u32>, segments: u32, y: f32, normal_y: f32) {
    let normal = [0.0, normal_y, 0.0];
    let center = vertices.len() as u32;
    vertices.push(MeshVertex { position: [0.0, y, 0.0], normal, uv: [0.5, 0.5] });
    let ring = center + 1;
    for segment in 0..segments {
        let (sin, cos) = (std::f32::consts::TAU * segment as f32 / segments as f32).sin_cos();
        vertices.push(MeshVertex { position: [0.5 * cos, y, 0.5 * sin], normal, uv: [0.5 + 0.5 * cos, 0.5 + 0.5 * sin] });
    }
    for segment in 0..segments {
        let current = ring + segment;
        let next = ring + (segment + 1) % segments;
        // Seen from +Y, (center, next, current) is counterclockwise; the -Y cap reverses it.
        if normal_y > 0.0 { indices.extend_from_slice(&[center, next, current]); }
        else { indices.extend_from_slice(&[center, current, next]); }
    }
}
