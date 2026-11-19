// SPDX-License-Identifier: MIT OR Apache-2.0
use std::{num::NonZeroU64, sync::Arc};

use anyhow::{Result, ensure};
use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::{BaseColorImage, Camera, CameraGpu, Instance, MeshData, MeshVertex, OutputLayout, RenderConfig, WrapMode};

/// Timestamps written by `Renderer::render_timed`: cull begin/end, camera pass begin/end,
/// pack begin/end.
pub const RENDER_TIMESTAMPS: u32 = 6;

/// GPU instance ABI, shared with cull.wgsl and render.wgsl. `metadata`: body index
/// (u32::MAX = static), object ID, camera filter (0 = all cameras, else camera + 1),
/// base-color texture index. `pbr`: metallic, roughness, imported flag, double-sided flag.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuInstance {
    position: [f32; 4],
    rotation: [f32; 4],
    scale: [f32; 4],
    color: [f32; 4],
    surface: [f32; 4],
    metadata: [u32; 4],
    pbr: [f32; 4],
}

impl From<&Instance> for GpuInstance {
    fn from(instance: &Instance) -> Self {
        Self {
            position: [instance.position[0], instance.position[1], instance.position[2], 0.0],
            rotation: instance.rotation,
            scale: [instance.scale[0], instance.scale[1], instance.scale[2], 0.0],
            color: instance.material.color,
            surface: [instance.material.shininess, instance.material.specular,
                instance.material.checker_scale, instance.material.checker_contrast],
            metadata: [instance.body_index.unwrap_or(u32::MAX), instance.object_id,
                instance.camera.map_or(0, |camera| camera + 1), 0],
            pbr: [0.0, 0.5, 0.0, 0.0],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Globals {
    light_direction: [f32; 4],
    light_color: [f32; 4],
    ambient: [f32; 4],
    camera_index: [u32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct DrawInfo {
    bounds_min: [f32; 4],
    bounds_max: [f32; 4],
    // index_count, first_index, base_vertex (bitcast i32), instance_index
    command: [u32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct TextureInfo {
    // atlas Y offset, width, height, packed wrapping/filter flags
    data: [u32; 4],
}

/// One 2D attachment holding every camera's image: camera `c` renders into the tile at column
/// `c % columns`, row `c / columns`, so all cameras share a single render pass.
struct Target {
    view: wgpu::TextureView,
}

impl Target {
    fn new(device: &wgpu::Device, atlas: Atlas, format: wgpu::TextureFormat) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("sim-render camera atlas"),
            size: wgpu::Extent3d { width: atlas.width(), height: atlas.height(), depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        Self { view: texture.create_view(&wgpu::TextureViewDescriptor::default()) }
    }
}

/// Camera tiling of the attachments; mirrored by pack.wgsl's `Grid`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Atlas {
    tile_width: u32,
    tile_height: u32,
    columns: u32,
    cameras: u32,
}

impl Atlas {
    fn new(tile_width: u32, tile_height: u32, cameras: u32, max_dimension: u32) -> Result<Self> {
        let columns = (max_dimension / tile_width).min(cameras);
        ensure!(columns > 0 && cameras.div_ceil(columns) * tile_height <= max_dimension,
            "{cameras} cameras of {tile_width}x{tile_height} do not fit a {max_dimension}px atlas");
        Ok(Self { tile_width, tile_height, columns, cameras })
    }
    fn width(&self) -> u32 { self.columns * self.tile_width }
    fn height(&self) -> u32 { self.cameras.div_ceil(self.columns) * self.tile_height }
    fn origin(&self, camera: u32) -> (u32, u32) {
        ((camera % self.columns) * self.tile_width, (camera / self.columns) * self.tile_height)
    }
}

/// Persistent bindings to one body/output/camera buffer set. Create one per
/// externally synchronized ring-buffer slot, not once per frame. Bind groups
/// retain the buffers. Do not destroy the underlying external allocations while
/// bindings or submitted GPU work still reference them.
pub struct RenderBindings {
    cameras: Vec<wgpu::BindGroup>,
    pack: wgpu::BindGroup,
    cull: wgpu::BindGroup,
}

pub struct Renderer {
    config: RenderConfig,
    output: OutputLayout,
    render_pipeline: wgpu::RenderPipeline,
    render_layout: wgpu::BindGroupLayout,
    pack_pipeline: wgpu::ComputePipeline,
    pack_layout: wgpu::BindGroupLayout,
    vertices: wgpu::Buffer,
    indices: wgpu::Buffer,
    draws: wgpu::Buffer,
    draw_count: u32,
    indirect: wgpu::Buffer,
    cull_pipeline: wgpu::ComputePipeline,
    cull_layout: wgpu::BindGroupLayout,
    texture_pixels: wgpu::TextureView,
    texture_info: wgpu::Buffer,
    instances: wgpu::Buffer,
    cameras: wgpu::Buffer,
    globals: Vec<wgpu::Buffer>,
    body_binding_bytes: u64,
    tiles: Atlas,
    atlas_grid: wgpu::Buffer,
    color: Target,
    optical_depth: Target,
    object_id: Target,
    raster_depth: Target,
}

impl Renderer {
    /// Upload immutable scene resources and preallocate all batch attachments.
    /// Requires INDIRECT_FIRST_INSTANCE and indirect execution support.
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        config: RenderConfig,
        meshes: &[MeshData],
        instances: &[Instance],
        cameras: &[Camera],
    ) -> Result<Self> {
        let limits = device.limits();
        ensure!(device.features().contains(wgpu::Features::INDIRECT_FIRST_INSTANCE),
            "renderer requires INDIRECT_FIRST_INSTANCE");
        ensure!(config.width > 0 && config.height > 0, "render dimensions must be nonzero");
        ensure!(config.width <= limits.max_texture_dimension_2d && config.height <= limits.max_texture_dimension_2d,
            "render dimensions exceed device texture limits");
        let camera_count = u32::try_from(cameras.len())?;
        ensure!(camera_count > 0, "renderer needs at least one camera");
        let tiles = Atlas::new(config.width, config.height, camera_count, limits.max_texture_dimension_2d)?;
        ensure!(config.width.div_ceil(8) <= limits.max_compute_workgroups_per_dimension
            && config.height.div_ceil(8) <= limits.max_compute_workgroups_per_dimension
            && camera_count <= limits.max_compute_workgroups_per_dimension,
            "render dimensions exceed packing dispatch limits");
        let camera_stride = u64::from(config.width) * u64::from(config.height) * 4;
        let region_bytes = camera_stride * u64::from(camera_count);
        let total_bytes = region_bytes.checked_mul(3).ok_or_else(|| anyhow::anyhow!("output size overflow"))?;
        ensure!(total_bytes <= u64::from(limits.max_storage_buffer_binding_size)
            && total_bytes <= limits.max_buffer_size && total_bytes / 4 <= u64::from(u32::MAX),
            "packed output exceeds device storage limits");
        let output = OutputLayout {
            width: config.width,
            height: config.height,
            cameras: camera_count,
            row_stride_bytes: u64::from(config.width) * 4,
            camera_stride_bytes: camera_stride,
            rgba_offset_bytes: 0,
            depth_offset_bytes: region_bytes,
            object_id_offset_bytes: region_bytes * 2,
            total_bytes,
        };
        ensure!(!meshes.is_empty() && !instances.is_empty(), "scene must contain meshes and instances");
        let instance_count = u32::try_from(instances.len())?;
        ensure!(u64::from(instance_count) * size_of::<GpuInstance>() as u64 <= u64::from(limits.max_storage_buffer_binding_size),
            "instances exceed device storage limits");
        ensure!(u64::from(camera_count) * size_of::<CameraGpu>() as u64 <= u64::from(limits.max_storage_buffer_binding_size),
            "cameras exceed device storage limits");
        for instance in instances {
            ensure!(instance.mesh < meshes.len(), "instance references missing mesh");
            ensure!(instance.body_index != Some(u32::MAX), "body index u32::MAX is reserved for static instances");
            ensure!(instance.scale.iter().all(|v| v.is_finite() && *v > 0.0), "instance scale must be finite and positive");
            ensure!(instance.object_id != 0, "object ID zero is reserved for background");
            ensure!(instance.camera.is_none_or(|camera| camera < camera_count),
                "instance camera is outside the camera batch");
        }
        let body_count = instances.iter().filter_map(|instance| instance.body_index)
            .max().map_or(1, |index| u64::from(index) + 1);
        let body_binding_bytes = body_count * 64;
        ensure!(body_binding_bytes <= u64::from(limits.max_storage_buffer_binding_size), "body indices exceed device storage limits");
        let mut grouped_instances = Vec::<GpuInstance>::with_capacity(instances.len());
        let mut vertices = Vec::new();
        let mut indices = Vec::new();
        let mut draws = Vec::new();
        // Deduplicated sRGB images share a vertical atlas. Each image keeps its
        // original resolution; shader-side texel wrapping prevents atlas bleed.
        let mut atlas_width = 1u32;
        let mut atlas_height = 1u32;
        let mut texture_info = vec![TextureInfo { data: [0, 1, 1, 0] }];
        let mut images: Vec<(Arc<BaseColorImage>, u32)> = Vec::new();
        for (mesh_index, mesh) in meshes.iter().enumerate() {
            ensure!(!mesh.vertices.is_empty() && !mesh.indices.is_empty() && mesh.indices.len() % 3 == 0,
                "mesh {mesh_index} must contain indexed triangles");
            ensure!(mesh.indices.iter().all(|&index| (index as usize) < mesh.vertices.len()), "mesh {mesh_index} has an invalid index");
            let mut min = glam::Vec3::splat(f32::INFINITY);
            let mut max = glam::Vec3::splat(f32::NEG_INFINITY);
            for vertex in &mesh.vertices {
                let p = glam::Vec3::from_array(vertex.position);
                ensure!(p.is_finite(), "mesh {mesh_index} has non-finite vertices");
                min = min.min(p); max = max.max(p);
            }
            let first_index = u32::try_from(indices.len())?;
            let base_vertex = i32::try_from(vertices.len())?;
            let index_count = u32::try_from(mesh.indices.len())?;
            vertices.extend_from_slice(&mesh.vertices);
            indices.extend_from_slice(&mesh.indices);
            let texture_index = if let Some(texture) = mesh.material.as_ref().and_then(|m| m.base_color_texture.as_ref()) {
                let image = &texture.image;
                ensure!(image.width > 0 && image.height > 0
                    && image.rgba8.len() as u64 == u64::from(image.width) * u64::from(image.height) * 4,
                    "invalid base-color image");
                let offset = if let Some((_, offset)) = images.iter().find(|(other, _)| Arc::ptr_eq(other, image)
                    || (other.width == image.width && other.height == image.height && other.rgba8 == image.rgba8)) {
                    *offset
                } else {
                    let offset = atlas_height;
                    atlas_height = atlas_height.checked_add(image.height).ok_or_else(|| anyhow::anyhow!("texture atlas height overflow"))?;
                    atlas_width = atlas_width.max(image.width);
                    images.push((image.clone(), offset));
                    offset
                };
                let mode = |wrap| match wrap { WrapMode::Repeat => 0, WrapMode::MirroredRepeat => 1, WrapMode::ClampToEdge => 2 };
                let descriptor = [offset, image.width, image.height,
                    mode(texture.wrap_u) | (mode(texture.wrap_v) << 2) | (u32::from(texture.linear) << 4)];
                if let Some(index) = texture_info.iter().position(|info| info.data == descriptor) { index as u32 }
                else { texture_info.push(TextureInfo { data: descriptor }); (texture_info.len() - 1) as u32 }
            } else { 0 };
            for instance in instances.iter().filter(|instance| instance.mesh == mesh_index) {
                let instance_index = u32::try_from(grouped_instances.len())?;
                let mut gpu = GpuInstance::from(instance);
                gpu.metadata[3] = texture_index;
                if let Some(material) = &mesh.material {
                    for channel in 0..4 { gpu.color[channel] *= material.base_color_factor[channel]; }
                    let double_sided = if material.double_sided { 1.0 } else { 0.0 };
                    gpu.pbr = [material.metallic_factor, material.roughness_factor, 1.0, double_sided];
                }
                grouped_instances.push(gpu);
                draws.push(DrawInfo { bounds_min: [min.x, min.y, min.z, 0.0],
                    bounds_max: [max.x, max.y, max.z, 0.0],
                    command: [index_count, first_index, base_vertex as u32, instance_index] });
            }
        }
        let draw_count = u32::try_from(draws.len())?;
        ensure!(draw_count.div_ceil(64) <= limits.max_compute_workgroups_per_dimension, "too many culling workgroups");
        let indirect_bytes = u64::from(draw_count) * u64::from(camera_count) * 20;
        for (label, bytes) in [("draws", draws.len() as u64 * size_of::<DrawInfo>() as u64),
            ("indirect", indirect_bytes),
            ("texture descriptors", texture_info.len() as u64 * size_of::<TextureInfo>() as u64)] {
            ensure!(bytes <= u64::from(limits.max_storage_buffer_binding_size) && bytes <= limits.max_buffer_size,
                "{label} exceeds device storage limits");
        }
        let vertices = upload(device, "scene vertices", &vertices, wgpu::BufferUsages::VERTEX);
        let indices = upload(device, "scene indices", &indices, wgpu::BufferUsages::INDEX);
        let draws = upload(device, "scene draw bounds", &draws, wgpu::BufferUsages::STORAGE);
        let indirect = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("persistent culled draw commands"), size: indirect_bytes,
            // COPY_SRC: the profiler reads the culled commands back on request.
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        ensure!(atlas_width <= limits.max_texture_dimension_2d && atlas_height <= limits.max_texture_dimension_2d,
            "base-color image atlas exceeds texture dimensions");
        let atlas = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("base-color sRGB image atlas"),
            size: wgpu::Extent3d { width: atlas_width, height: atlas_height, depth_or_array_layers: 1 },
            mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST, view_formats: &[],
        });
        let write_image = |y, width, height, rgba8: &[u8]| {
            queue.write_texture(wgpu::TexelCopyTextureInfo {
                texture: &atlas, mip_level: 0, origin: wgpu::Origin3d { x: 0, y, z: 0 },
                aspect: wgpu::TextureAspect::All,
            }, rgba8, wgpu::TexelCopyBufferLayout {
                offset: 0, bytes_per_row: Some(width * 4), rows_per_image: Some(height),
            }, wgpu::Extent3d { width, height, depth_or_array_layers: 1 });
        };
        write_image(0, 1, 1, &[255; 4]);
        for (image, y) in &images { write_image(*y, image.width, image.height, &image.rgba8); }
        let texture_pixels = atlas.create_view(&wgpu::TextureViewDescriptor::default());
        let texture_info = upload(device, "base-color image descriptors", &texture_info, wgpu::BufferUsages::STORAGE);
        let instances = upload(device, "scene instances", &grouped_instances, wgpu::BufferUsages::STORAGE);
        let camera_data: Vec<_> = cameras.iter().map(|camera| camera.gpu(config.width as f32 / config.height as f32)).collect();
        let cameras = upload(device, "scene cameras", &camera_data, wgpu::BufferUsages::STORAGE);
        let lighting = config.lighting;
        let globals = (0..camera_count).map(|camera_index| {
            let data = Globals {
                light_direction: [lighting.direction[0], lighting.direction[1], lighting.direction[2], 0.0],
                light_color: [lighting.color[0], lighting.color[1], lighting.color[2], 0.0],
                ambient: [lighting.ambient[0], lighting.ambient[1], lighting.ambient[2], 0.0],
                camera_index: [camera_index, 0, 0, 0],
            };
            upload(device, "camera lighting", &[data], wgpu::BufferUsages::UNIFORM)
        }).collect();
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("sim-render mesh shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("render.wgsl").into()),
        });
        let attributes = wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Float32x2];
        let targets = [wgpu::TextureFormat::Rgba8Unorm, wgpu::TextureFormat::R32Float, wgpu::TextureFormat::R32Uint]
            .map(|format| Some(wgpu::ColorTargetState { format, blend: None, write_mask: wgpu::ColorWrites::ALL }));
        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("sim-render mesh pipeline"),
            layout: None,
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: size_of::<MeshVertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &attributes,
                })],
            },
            primitive: wgpu::PrimitiveState { cull_mode: None, ..Default::default() },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &targets,
            }),
            multiview_mask: None,
            cache: None,
        });
        let render_layout = render_pipeline.get_bind_group_layout(0);
        let cull_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("sim-render frustum culling"),
            source: wgpu::ShaderSource::Wgsl(include_str!("cull.wgsl").into()),
        });
        let cull_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("sim-render frustum culling"), layout: None,
            module: &cull_shader, entry_point: Some("cull"),
            compilation_options: Default::default(), cache: None,
        });
        let cull_layout = cull_pipeline.get_bind_group_layout(0);
        let pack_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sim-render pack bindings"),
            entries: &[
                texture_entry(0, wgpu::TextureSampleType::Float { filterable: false }),
                texture_entry(1, wgpu::TextureSampleType::Float { filterable: false }),
                texture_entry(2, wgpu::TextureSampleType::Uint),
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(size_of::<Atlas>() as u64),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(total_bytes),
                    },
                    count: None,
                },
            ],
        });
        let pack_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("sim-render pack layout"),
            bind_group_layouts: &[Some(&pack_layout)],
            immediate_size: 0,
        });
        let pack_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("sim-render pack shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("pack.wgsl").into()),
        });
        let pack_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("sim-render pack pipeline"),
            layout: Some(&pack_pipeline_layout),
            module: &pack_shader,
            entry_point: Some("pack"),
            compilation_options: Default::default(),
            cache: None,
        });
        Ok(Self {
            config, output, render_pipeline, render_layout, pack_pipeline, pack_layout,
            vertices, indices, draws, draw_count, indirect, cull_pipeline, cull_layout,
            texture_pixels, texture_info, instances, cameras, globals, body_binding_bytes,
            tiles,
            atlas_grid: upload(device, "camera atlas grid", &[tiles], wgpu::BufferUsages::UNIFORM),
            color: Target::new(device, tiles, wgpu::TextureFormat::Rgba8Unorm),
            optical_depth: Target::new(device, tiles, wgpu::TextureFormat::R32Float),
            object_id: Target::new(device, tiles, wgpu::TextureFormat::R32Uint),
            raster_depth: Target::new(device, tiles, wgpu::TextureFormat::Depth32Float),
        })
    }

    pub fn output_layout(&self) -> OutputLayout { self.output }

    /// Number of chunk draws per camera (each camera has its own indirect command per draw).
    pub fn draw_count(&self) -> u32 { self.draw_count }

    /// Culled indexed-indirect commands of the last render: `cameras x draw_count` records of five
    /// u32 (index count, instance count 0/1, first index, base vertex, first instance).
    pub fn indirect(&self) -> &wgpu::Buffer { &self.indirect }

    /// Bind persistent resources once. Buffers must belong to this device and
    /// have STORAGE usage. `states` uses the documented BodyState64 ABI (at least
    /// one element even for an all-static scene). Output may additionally have
    /// COPY_SRC/COPY_DST for an explicit diagnostic, but never needs MAP_READ.
    /// Optional external cameras are an array of CameraGpu with this batch size.
    /// The producer must synchronize writes before rendering; do not concurrently
    /// write any bound resource from CUDA or another queue.
    pub fn bind(
        &self,
        device: &wgpu::Device,
        states: &wgpu::Buffer,
        output: &wgpu::Buffer,
        cameras: Option<&wgpu::Buffer>,
    ) -> Result<RenderBindings> {
        let cameras = cameras.unwrap_or(&self.cameras);
        let camera_bytes = u64::from(self.output.cameras) * size_of::<CameraGpu>() as u64;
        for (name, buffer, bytes) in [
            ("body state", states, self.body_binding_bytes),
            ("output", output, self.output.total_bytes),
            ("camera", cameras, camera_bytes),
        ] {
            ensure!(buffer.usage().contains(wgpu::BufferUsages::STORAGE), "{name} buffer requires STORAGE usage");
            ensure!(buffer.size() >= bytes, "{name} buffer is smaller than the binding layout ({bytes} bytes)");
        }
        let camera_bindings = self.globals.iter().map(|globals| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("sim-render camera bindings"),
                layout: &self.render_layout,
                entries: &[
                    buffer_entry(0, states, self.body_binding_bytes),
                    wgpu::BindGroupEntry { binding: 1, resource: self.instances.as_entire_binding() },
                    buffer_entry(2, cameras, camera_bytes),
                    wgpu::BindGroupEntry { binding: 3, resource: globals.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(&self.texture_pixels) },
                    wgpu::BindGroupEntry { binding: 5, resource: self.texture_info.as_entire_binding() },
                ],
            })
        }).collect();
        let pack = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sim-render output bindings"),
            layout: &self.pack_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&self.color.view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&self.optical_depth.view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&self.object_id.view) },
                buffer_entry(3, output, self.output.total_bytes),
                wgpu::BindGroupEntry { binding: 4, resource: self.atlas_grid.as_entire_binding() },
            ],
        });
        let cull = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sim-render culling bindings"), layout: &self.cull_layout,
            entries: &[
                buffer_entry(0, states, self.body_binding_bytes),
                wgpu::BindGroupEntry { binding: 1, resource: self.instances.as_entire_binding() },
                buffer_entry(2, cameras, camera_bytes),
                wgpu::BindGroupEntry { binding: 3, resource: self.draws.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: self.indirect.as_entire_binding() },
            ],
        });
        Ok(RenderBindings { cameras: camera_bindings, pack, cull })
    }

    /// Record all camera draws and one texture-to-contiguous-buffer compute pass.
    /// `bindings` must have been created by this renderer. There are no per-frame
    /// host vectors, resource creations, queue writes, submissions, or readbacks.
    pub fn render(&self, encoder: &mut wgpu::CommandEncoder, bindings: &RenderBindings) {
        self.render_timed(encoder, bindings, None);
    }

    /// `render` with pass timestamps written into `timestamps` (TIMESTAMP_QUERY, at least
    /// RENDER_TIMESTAMPS entries): cull begin/end, camera pass begin/end, pack begin/end.
    pub fn render_timed(&self, encoder: &mut wgpu::CommandEncoder, bindings: &RenderBindings,
        timestamps: Option<&wgpu::QuerySet>) {
        let compute_writes = |begin, end| timestamps.map(|query_set| wgpu::ComputePassTimestampWrites {
            query_set, beginning_of_pass_write_index: Some(begin), end_of_pass_write_index: Some(end),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("sim-render spatial culling"), timestamp_writes: compute_writes(0, 1),
            });
            pass.set_pipeline(&self.cull_pipeline);
            pass.set_bind_group(0, &bindings.cull, &[]);
            pass.dispatch_workgroups(self.draw_count.div_ceil(64), self.output.cameras, 1);
        }
        let background = self.config.background;
        let attachments = [
            Some(color_attachment(&self.color.view, wgpu::Color {
                r: background[0], g: background[1], b: background[2], a: background[3],
            })),
            Some(color_attachment(&self.optical_depth.view, wgpu::Color::TRANSPARENT)),
            Some(color_attachment(&self.object_id.view, wgpu::Color::TRANSPARENT)),
        ];
        {
            // Every camera draws into its own atlas tile (viewport + scissor) of one render pass.
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("sim-render cameras"),
                color_attachments: &attachments,
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.raster_depth.view,
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Discard }),
                    stencil_ops: None,
                }),
                timestamp_writes: timestamps.map(|query_set| wgpu::RenderPassTimestampWrites {
                    query_set, beginning_of_pass_write_index: Some(2), end_of_pass_write_index: Some(3),
                }),
                ..Default::default()
            });
            pass.set_pipeline(&self.render_pipeline);
            pass.set_vertex_buffer(0, self.vertices.slice(..));
            pass.set_index_buffer(self.indices.slice(..), wgpu::IndexFormat::Uint32);
            let (width, height) = (self.tiles.tile_width, self.tiles.tile_height);
            for (camera_index, camera) in bindings.cameras.iter().enumerate() {
                let (x, y) = self.tiles.origin(camera_index as u32);
                pass.set_bind_group(0, camera, &[]);
                pass.set_viewport(x as f32, y as f32, width as f32, height as f32, 0.0, 1.0);
                pass.set_scissor_rect(x, y, width, height);
                pass.multi_draw_indexed_indirect(&self.indirect,
                    camera_index as u64 * u64::from(self.draw_count) * 20, self.draw_count);
            }
        }
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("sim-render contiguous output"),
            timestamp_writes: compute_writes(4, 5),
        });
        pass.set_pipeline(&self.pack_pipeline);
        pass.set_bind_group(0, &bindings.pack, &[]);
        pass.dispatch_workgroups(self.output.width.div_ceil(8), self.output.height.div_ceil(8), self.output.cameras);
    }
}

fn upload<T: Pod>(device: &wgpu::Device, label: &str, data: &[T], usage: wgpu::BufferUsages) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label), contents: bytemuck::cast_slice(data), usage,
    })
}

fn buffer_entry(binding: u32, buffer: &wgpu::Buffer, bytes: u64) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer, offset: 0, size: NonZeroU64::new(bytes),
        }),
    }
}

fn texture_entry(binding: u32, sample_type: wgpu::TextureSampleType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding, visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Texture { sample_type, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false },
        count: None,
    }
}

fn color_attachment(view: &wgpu::TextureView, clear: wgpu::Color) -> wgpu::RenderPassColorAttachment<'_> {
    wgpu::RenderPassColorAttachment {
        view, depth_slice: None, resolve_target: None,
        ops: wgpu::Operations { load: wgpu::LoadOp::Clear(clear), store: wgpu::StoreOp::Store },
    }
}
