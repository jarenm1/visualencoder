//! Import the city's actual transformed triangles once for both graphics and collision.
//!
//! `layout.json` supplies task extent only, never collision proxies. Six embedded
//! GLBs supply indexed geometry, inverse-transpose normals and palette UVs.
//! Material-compatible triangle-centroid chunks are 16m across in X/Z; their
//! unclipped vertices are exactly the triangles reordered into the collision BVH.
//! All imported file bytes and names enter the SHA-256 provenance fingerprint.
use std::{collections::{BTreeMap, HashMap}, path::Path, sync::Arc};

use anyhow::{Context, Result, ensure};
use glam::{Mat3, Mat4, Vec3};
use sha2::{Digest, Sha256};
use sim_render::{BaseColorImage, BaseColorTexture, Instance, MeshData, MeshMaterial, MeshVertex, WrapMode};
use sim_state::collision::{BvhNode, LANDABLE, Triangle};

pub mod drone;

pub struct CityScene {
    pub meshes: Vec<MeshData>,
    pub instances: Vec<Instance>,
    pub nodes: Vec<BvhNode>,
    pub triangles: Vec<Triangle>,
    pub half_extent: f32,
    pub landable: Landable,
    pub fingerprint: String,
}

/// Landable geometry. Every landable vertex lies within the level band, so the plane at
/// `height` (the highest landable vertex) is at or above every landable feature.
#[derive(Clone, Copy, Debug)]
pub struct Landable {
    pub height: f32,
    pub triangles: usize,
    pub non_level: usize,
}

// Triangle-centroid bins bound the work submitted to each camera without replacing
// or clipping any geometry. Large triangles retain their full conservative bounds.
const CHUNK_METRES: f32 = 16.0;
const LEAF_TRIANGLES: usize = 8;
const ASSETS: [(&str, &str); 6] = [("building.glb", "building"), ("vegetation.glb", "vegetation"),
    ("vehicle.glb", "vehicle"), ("road.glb", "road"), ("prop.glb", "prop"), ("ground.glb", "ground")];
// Every landable vertex must lie within this band of y = 0.
const LEVEL_BAND: f32 = 0.25;
// Landable triangles with |n_y| / |n| at or below this cosine are counted as non-level (curbs, steps).
const LEVEL_COS: f32 = 0.98;

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
struct MaterialKey {
    factor: [u32; 4],
    metallic: u32,
    roughness: u32,
    image: usize,
    sampler: [u32; 3],
    double_sided: bool,
}

struct Chunk {
    mesh: MeshData,
    vertices: HashMap<u64, u32>,
}

#[derive(Default)]
struct Importer {
    chunks: BTreeMap<(i32, i32, MaterialKey), Chunk>,
    images: Vec<Arc<BaseColorImage>>,
    triangles: Vec<Triangle>,
    vertex_serial: u64,
}

impl CityScene {
    pub fn load(path: &Path) -> Result<Self> {
        let layout = std::fs::read(path.join("layout.json")).context("reading city layout")?;
        let json: serde_json::Value = serde_json::from_slice(&layout)?;
        let half_extent = json["city_half_extent_m"].as_f64().context("layout lacks city_half_extent_m")? as f32;
        ensure!(half_extent.is_finite() && half_extent > 0.0, "invalid city half extent");
        let mut hash = Sha256::new();
        hash_file(&mut hash, "layout.json", &layout);
        let mut importer = Importer::default();
        let classes = json["classes"].as_array().context("layout lacks classes")?;
        for (name, class_name) in ASSETS {
            let class = class_index(classes, class_name)?;
            let bytes = std::fs::read(path.join(name)).with_context(|| format!("reading {name}"))?;
            hash_file(&mut hash, name, &bytes);
            // These city GLBs are self-contained. import_slice resolves embedded
            // image/buffer contents, so fingerprinting the GLB covers every byte.
            let (document, buffers, images) = gltf::import_slice(&bytes).with_context(|| format!("importing {name}"))?;
            let images = images.into_iter().map(|data| importer.image(data)).collect::<Result<Vec<_>>>()?;
            let scene = document.default_scene().or_else(|| document.scenes().next()).context("GLB lacks a scene")?;
            for node in scene.nodes() {
                importer.node(node, Mat4::IDENTITY, &buffers, &images, class)?;
            }
        }
        ensure!(!importer.triangles.is_empty(), "city contains no triangles");
        ensure!(importer.triangles.len() <= u32::MAX as usize, "city triangle count exceeds collision ABI");
        let landable = landable_plane(&importer.triangles)?;
        let meshes: Vec<_> = importer.chunks.into_values().map(|chunk| chunk.mesh).collect();
        let instances = (0..meshes.len()).map(|mesh| {
            let mut instance = Instance::static_mesh(mesh);
            instance.material.color = [1.0; 4];
            instance.object_id = mesh as u32 + 1;
            instance
        }).collect();
        let leaves = importer.triangles.len().div_ceil(LEAF_TRIANGLES).next_power_of_two();
        let mut nodes = Vec::with_capacity(leaves * 2 - 1);
        build_bvh(&mut importer.triangles, 0, &mut nodes);
        Ok(Self { meshes, instances, nodes, triangles: importer.triangles, half_extent, landable,
            fingerprint: format!("sha256:{:x}", hash.finalize()) })
    }

    /// Bit k of `ignore` skips triangles of class k; the BVH is still traversed in full.
    pub fn sweep_hits(&self, from: [f32; 3], to: [f32; 3], radius: f32, ignore: u32) -> bool {
        // The immutable vectors own every node/range visited by the shared
        // stackless CPU/CUDA implementation for the duration of this call.
        unsafe { sim_state::collision::sweep_hits(self.nodes.as_ptr(), self.nodes.len() as u32,
            self.triangles.as_ptr(), from, to, radius, ignore) }
    }
}

fn hash_file(hash: &mut Sha256, name: &str, bytes: &[u8]) {
    hash.update((name.len() as u64).to_le_bytes());
    hash.update(name.as_bytes());
    hash.update((bytes.len() as u64).to_le_bytes());
    hash.update(bytes);
}

impl Importer {
    fn image(&mut self, data: gltf::image::Data) -> Result<Arc<BaseColorImage>> {
        use gltf::image::Format;
        let (channels, channel_bytes, float) = match data.format {
            Format::R8 => (1, 1, false), Format::R8G8 => (2, 1, false),
            Format::R8G8B8 => (3, 1, false), Format::R8G8B8A8 => (4, 1, false),
            Format::R16 => (1, 2, false), Format::R16G16 => (2, 2, false),
            Format::R16G16B16 => (3, 2, false), Format::R16G16B16A16 => (4, 2, false),
            Format::R32G32B32FLOAT => (3, 4, true), Format::R32G32B32A32FLOAT => (4, 4, true),
        };
        let mut rgba8 = Vec::with_capacity(data.width as usize * data.height as usize * 4);
        for pixel in data.pixels.chunks_exact(channels * channel_bytes) {
            let channel = |i: usize| -> u8 {
                let start = i * channel_bytes;
                if float {
                    (f32::from_ne_bytes(pixel[start..start + 4].try_into().unwrap()).clamp(0.0, 1.0) * 255.0).round() as u8
                } else if channel_bytes == 2 {
                    (u16::from_ne_bytes(pixel[start..start + 2].try_into().unwrap()) / 257) as u8
                } else { pixel[start] }
            };
            let r = channel(0);
            rgba8.extend_from_slice(&[r, if channels >= 3 { channel(1) } else { r },
                if channels >= 3 { channel(2) } else { r },
                if channels == 2 { channel(1) } else if channels == 4 { channel(3) } else { 255 }]);
        }
        if let Some(image) = self.images.iter().find(|image| image.width == data.width && image.height == data.height && image.rgba8 == rgba8) {
            return Ok(image.clone());
        }
        let image = Arc::new(BaseColorImage { width: data.width, height: data.height, rgba8 });
        self.images.push(image.clone());
        Ok(image)
    }

    fn node(&mut self, node: gltf::Node<'_>, parent: Mat4, buffers: &[gltf::buffer::Data], images: &[Arc<BaseColorImage>], class: u32) -> Result<()> {
        let world = parent * Mat4::from_cols_array_2d(&node.transform().matrix());
        let linear = Mat3::from_mat4(world);
        ensure!(world.is_finite() && linear.determinant().abs() > 1e-12, "invalid GLB node transform");
        let mirrored = linear.determinant() < 0.0;
        let normal_transform = linear.inverse().transpose();
        if let Some(mesh) = node.mesh() {
            for primitive in mesh.primitives() {
                ensure!(primitive.mode() == gltf::mesh::Mode::Triangles, "city primitive is not triangles");
                let source = primitive.material();
                ensure!(source.alpha_mode() == gltf::material::AlphaMode::Opaque, "city material is not opaque");
                let pbr = source.pbr_metallic_roughness();
                let texture = pbr.base_color_texture().map(|info| {
                    ensure!(info.tex_coord() == 0, "base-color texture requires unsupported UV set");
                    let texture = info.texture();
                    let sampler = texture.sampler();
                    Ok(BaseColorTexture { image: images[texture.source().index()].clone(),
                        wrap_u: wrap(sampler.wrap_s()), wrap_v: wrap(sampler.wrap_t()),
                        linear: sampler.mag_filter() != Some(gltf::texture::MagFilter::Nearest) })
                }).transpose()?;
                let image = texture.as_ref().map_or(0, |texture|
                    self.images.iter().position(|image| Arc::ptr_eq(image, &texture.image)).unwrap() + 1);
                let sampler = texture.as_ref().map_or([0; 3], |texture| [texture.wrap_u as u32, texture.wrap_v as u32, texture.linear as u32]);
                let material = MeshMaterial { base_color_factor: pbr.base_color_factor(), base_color_texture: texture,
                    metallic_factor: pbr.metallic_factor(), roughness_factor: pbr.roughness_factor(), double_sided: source.double_sided() };
                let key = MaterialKey { factor: material.base_color_factor.map(f32::to_bits),
                    metallic: material.metallic_factor.to_bits(), roughness: material.roughness_factor.to_bits(), image, sampler,
                    double_sided: material.double_sided };
                let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()].0));
                let positions: Vec<_> = reader.read_positions().context("primitive lacks positions")?.collect();
                let normals: Vec<_> = reader.read_normals().context("primitive lacks normals")?.collect();
                let uvs = reader.read_tex_coords(0).map(|uvs| uvs.into_f32().collect::<Vec<_>>());
                ensure!(normals.len() == positions.len() && uvs.as_ref().is_none_or(|uvs| uvs.len() == positions.len()), "inconsistent vertex attributes");
                ensure!(material.base_color_texture.is_none() || uvs.is_some(), "textured primitive lacks UVs");
                // COLOR_0 is deliberately not read: these assets encode class and
                // variant metadata in R/G rather than a surface reflectance.
                let vertices: Vec<_> = positions.iter().zip(&normals).enumerate().map(|(i, (p, n))| MeshVertex {
                    position: world.transform_point3(Vec3::from_array(*p)).to_array(),
                    normal: (normal_transform * Vec3::from_array(*n)).normalize_or_zero().to_array(),
                    uv: uvs.as_ref().map_or([0.0; 2], |uvs| uvs[i]),
                }).collect();
                ensure!(vertices.iter().all(|v| v.position.iter().chain(v.normal.iter()).chain(v.uv.iter()).all(|v| v.is_finite())), "non-finite imported vertex");
                let indices: Vec<u32> = reader.read_indices().context("primitive lacks triangle indices")?.into_u32().collect();
                ensure!(indices.len() % 3 == 0 && indices.iter().all(|&i| (i as usize) < vertices.len()), "invalid triangle indices");
                let serial = self.vertex_serial;
                self.vertex_serial += vertices.len() as u64;
                for face in indices.chunks_exact(3) {
                    let mut face = [face[0], face[1], face[2]];
                    if mirrored { face.swap(1, 2); }
                    let p = face.map(|i| vertices[i as usize].position);
                    self.triangles.push(Triangle { a: tagged(p[0], class), b: lane(p[1]), c: lane(p[2]) });
                    let center = (Vec3::from_array(p[0]) + Vec3::from_array(p[1]) + Vec3::from_array(p[2])) / 3.0;
                    let chunk = self.chunks.entry(((center.x / CHUNK_METRES).floor() as i32,
                        (center.z / CHUNK_METRES).floor() as i32, key.clone())).or_insert_with(|| Chunk {
                            mesh: MeshData { vertices: Vec::new(), indices: Vec::new(), material: Some(material.clone()) }, vertices: HashMap::new() });
                    for index in face {
                        let next = chunk.mesh.vertices.len() as u32;
                        let remapped = *chunk.vertices.entry(serial + u64::from(index)).or_insert_with(|| {
                            chunk.mesh.vertices.push(vertices[index as usize]); next
                        });
                        chunk.mesh.indices.push(remapped);
                    }
                }
            }
        }
        for child in node.children() { self.node(child, world, buffers, images, class)?; }
        Ok(())
    }
}

fn wrap(mode: gltf::texture::WrappingMode) -> WrapMode {
    match mode { gltf::texture::WrappingMode::ClampToEdge => WrapMode::ClampToEdge,
        gltf::texture::WrappingMode::MirroredRepeat => WrapMode::MirroredRepeat,
        gltf::texture::WrappingMode::Repeat => WrapMode::Repeat }
}
fn lane(p: [f32; 3]) -> [f32; 4] { [p[0], p[1], p[2], 0.0] }
fn tagged(p: [f32; 3], class: u32) -> [f32; 4] { [p[0], p[1], p[2], class as f32] }
fn class_index(classes: &[serde_json::Value], name: &str) -> Result<u32> {
    classes.iter().position(|c| c.as_str() == Some(name)).map(|i| i as u32)
        .with_context(|| format!("layout lacks class {name}"))
}
/// Verifies that every landable vertex lies in the level band and summarises the landable
/// surface. Zero-area triangles still constrain the band but do not count as non-level.
fn landable_plane(triangles: &[Triangle]) -> Result<Landable> {
    let mut height = f32::NEG_INFINITY;
    let mut landable = Landable { height: 0.0, triangles: 0, non_level: 0 };
    for t in triangles {
        let class = t.a[3] as u32;
        if class >= 32 || (LANDABLE >> class) & 1 == 0 {
            continue;
        }
        let p = [t.a, t.b, t.c].map(|v| Vec3::new(v[0], v[1], v[2]));
        for v in p {
            ensure!(v.y.abs() <= LEVEL_BAND, "landable vertex at y = {} lies outside the level band", v.y);
            height = height.max(v.y);
        }
        landable.triangles += 1;
        let n = (p[1] - p[0]).cross(p[2] - p[0]);
        let len = n.length();
        if len >= 1e-6 && (n.y / len).abs() <= LEVEL_COS {
            landable.non_level += 1;
        }
    }
    ensure!(landable.triangles > 0, "city has no landable triangles");
    landable.height = height;
    Ok(landable)
}

fn build_bvh(triangles: &mut [Triangle], first: usize, nodes: &mut Vec<BvhNode>) {
    let mut min = Vec3::splat(f32::INFINITY);
    let mut max = Vec3::splat(f32::NEG_INFINITY);
    let mut centroid_min = min;
    let mut centroid_max = max;
    for triangle in triangles.iter() {
        for p in [triangle.a, triangle.b, triangle.c] {
            let p = Vec3::new(p[0], p[1], p[2]);
            min = min.min(p); max = max.max(p);
        }
        let center = centroid(triangle);
        centroid_min = centroid_min.min(center); centroid_max = centroid_max.max(center);
    }
    let index = nodes.len();
    nodes.push(BvhNode { bounds_min: lane(min.to_array()), bounds_max: lane(max.to_array()), meta: [first as u32, 0, 0, 0] });
    if triangles.len() <= LEAF_TRIANGLES {
        nodes[index].meta[1] = triangles.len() as u32;
    } else {
        let extent = centroid_max - centroid_min;
        let axis = if extent.x >= extent.y && extent.x >= extent.z { 0 } else if extent.y >= extent.z { 1 } else { 2 };
        let mid = triangles.len() / 2;
        triangles.select_nth_unstable_by(mid, |a, b| centroid(a)[axis].total_cmp(&centroid(b)[axis]));
        let (left, right) = triangles.split_at_mut(mid);
        build_bvh(left, first, nodes);
        build_bvh(right, first + mid, nodes);
    }
    nodes[index].meta[2] = nodes.len() as u32;
}
fn centroid(t: &Triangle) -> Vec3 {
    Vec3::new((t.a[0] + t.b[0] + t.c[0]) / 3.0, (t.a[1] + t.b[1] + t.c[1]) / 3.0, (t.a[2] + t.b[2] + t.c[2]) / 3.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn verify_bvh(nodes: &[BvhNode], triangles: &[Triangle]) {
        assert_eq!(nodes[0].meta[2] as usize, nodes.len());
        let mut next_triangle = 0usize;
        for (index, node) in nodes.iter().enumerate() {
            let escape = node.meta[2] as usize;
            assert!(escape > index && escape <= nodes.len());
            if node.meta[1] == 0 {
                let left = index + 1;
                let right = nodes[left].meta[2] as usize;
                assert!(right < escape);
                assert_eq!(nodes[right].meta[2] as usize, escape);
                for child in [&nodes[left], &nodes[right]] {
                    for axis in 0..3 {
                        assert!(child.bounds_min[axis] >= node.bounds_min[axis]);
                        assert!(child.bounds_max[axis] <= node.bounds_max[axis]);
                    }
                }
            } else {
                assert_eq!(escape, index + 1);
                assert_eq!(node.meta[0] as usize, next_triangle);
                assert!(node.meta[1] as usize <= LEAF_TRIANGLES);
                for triangle in &triangles[next_triangle..next_triangle + node.meta[1] as usize] {
                    for p in [triangle.a, triangle.b, triangle.c] {
                        for axis in 0..3 {
                            assert!(p[axis] >= node.bounds_min[axis] && p[axis] <= node.bounds_max[axis]);
                        }
                    }
                }
                next_triangle += node.meta[1] as usize;
            }
        }
        assert_eq!(next_triangle, triangles.len());
    }

    #[test]
    fn balanced_preorder_has_contiguous_leaf_ranges() {
        let mut triangles = (0..35).rev().map(|x| Triangle {
            a: [x as f32, 0.0, 0.0, 0.0], b: [x as f32, 1.0, 0.0, 0.0],
            c: [x as f32, 0.0, 1.0, 0.0],
        }).collect::<Vec<_>>();
        let mut nodes = Vec::new();
        build_bvh(&mut triangles, 0, &mut nodes);
        verify_bvh(&nodes, &triangles);
    }

    #[test]
    #[ignore = "requires generated assets/city; geometry agreement is an explicit asset smoke check"]
    fn real_city_render_and_collision_geometry_agree() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../assets/city");
        let scene = CityScene::load(&path).unwrap();
        verify_bvh(&scene.nodes, &scene.triangles);
        let mut render_faces: Vec<_> = scene.meshes.iter().flat_map(|mesh| mesh.indices.chunks_exact(3)
            .map(move |face| face_positions(mesh, face))).collect();
        let mut collision_faces: Vec<_> = scene.triangles.iter().map(|t|
            [t.a[0], t.a[1], t.a[2], t.b[0], t.b[1], t.b[2], t.c[0], t.c[1], t.c[2]].map(f32::to_bits)).collect();
        render_faces.sort_unstable();
        collision_faces.sort_unstable();
        assert_eq!(render_faces, collision_faces);
    }

    fn face_positions(mesh: &MeshData, face: &[u32]) -> [u32; 9] {
        let a = mesh.vertices[face[0] as usize].position;
        let b = mesh.vertices[face[1] as usize].position;
        let c = mesh.vertices[face[2] as usize].position;
        [a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2]].map(f32::to_bits)
    }

    #[test]
    fn landable_plane_accepts_any_orientation_inside_band_and_rejects_outside() {
        let flat = |y: f32, class: f32| Triangle {
            a: [0.0, y, 0.0, class], b: [1.0, y, 0.0, 0.0], c: [0.0, y, 1.0, 0.0],
        };
        let level = landable_plane(&[flat(0.1, 1.0), flat(0.2, 2.0), flat(0.5, 3.0)]).unwrap();
        assert_eq!((level.height, level.triangles, level.non_level), (0.2, 2, 0));
        let curb = Triangle { a: [0.0, 0.0, 0.0, 2.0], b: [1.0, 0.0, 0.0, 0.0], c: [0.0, 0.25, 0.5, 0.0] };
        let step = landable_plane(&[curb]).unwrap();
        assert_eq!((step.height, step.triangles, step.non_level), (0.25, 1, 1));
        let slope = Triangle { a: [0.0, 0.0, 0.0, 1.0], b: [1.0, 0.0, 0.0, 0.0], c: [0.0, 1.0, 1.0, 0.0] };
        assert!(landable_plane(&[slope]).is_err());
        assert!(landable_plane(&[flat(0.5, 1.0)]).is_err());
        assert!(landable_plane(&[flat(0.0, 3.0)]).is_err());
    }
}
