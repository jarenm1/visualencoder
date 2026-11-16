// SPDX-License-Identifier: MIT OR Apache-2.0
use std::{num::NonZeroU64, ops::Range};

use anyhow::{Result, ensure};
use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::{Camera, CameraGpu, Instance, MeshData, MeshVertex, OutputLayout, RenderConfig};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuInstance {
    position: [f32; 4],
    rotation: [f32; 4],
    scale: [f32; 4],
    color: [f32; 4],
    surface: [f32; 4],
    metadata: [u32; 4],
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
            metadata: [instance.body_index.unwrap_or(u32::MAX), instance.object_id, 0, 0],
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

struct MeshBatch {
    vertices: wgpu::Buffer,
    indices: wgpu::Buffer,
    index_count: u32,
    instances: Range<u32>,
}

struct Target {
    array: wgpu::TextureView,
    layers: Vec<wgpu::TextureView>,
}

impl Target {
    fn new(device: &wgpu::Device, layout: OutputLayout, format: wgpu::TextureFormat) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("sim-render batch target"),
            size: wgpu::Extent3d {
                width: layout.width,
                height: layout.height,
                depth_or_array_layers: layout.cameras,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let array = texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
        let layers = (0..layout.cameras).map(|layer| {
            texture.create_view(&wgpu::TextureViewDescriptor {
                dimension: Some(wgpu::TextureViewDimension::D2),
                base_array_layer: layer,
                array_layer_count: Some(1),
                ..Default::default()
            })
        }).collect();
        Self { array, layers }
    }
}

/// Persistent bindings to one body/output/camera buffer set. Create one per
/// externally synchronized ring-buffer slot, not once per frame. Bind groups
/// retain the buffers. Do not destroy the underlying external allocations while
/// bindings or submitted GPU work still reference them.
pub struct RenderBindings {
    cameras: Vec<wgpu::BindGroup>,
    pack: wgpu::BindGroup,
}

pub struct Renderer {
    config: RenderConfig,
    output: OutputLayout,
    render_pipeline: wgpu::RenderPipeline,
    render_layout: wgpu::BindGroupLayout,
    pack_pipeline: wgpu::ComputePipeline,
    pack_layout: wgpu::BindGroupLayout,
    meshes: Vec<MeshBatch>,
    instances: wgpu::Buffer,
    cameras: wgpu::Buffer,
    globals: Vec<wgpu::Buffer>,
    body_binding_bytes: u64,
    color: Target,
    optical_depth: Target,
    object_id: Target,
    raster_depth: Target,
}

impl Renderer {
    /// Upload immutable scene resources and preallocate all batch attachments.
    /// No device features beyond baseline Vulkan wgpu limits are required.
    pub fn new(
        device: &wgpu::Device,
        config: RenderConfig,
        meshes: &[MeshData],
        instances: &[Instance],
        cameras: &[Camera],
    ) -> Result<Self> {
        let limits = device.limits();
        ensure!(config.width > 0 && config.height > 0, "render dimensions must be nonzero");
        ensure!(config.width <= limits.max_texture_dimension_2d && config.height <= limits.max_texture_dimension_2d,
            "render dimensions exceed device texture limits");
        let camera_count = u32::try_from(cameras.len())?;
        ensure!(camera_count > 0 && camera_count <= limits.max_texture_array_layers,
            "camera count must fit device texture array limits");
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
        }
        let body_count = instances.iter().filter_map(|instance| instance.body_index)
            .max().map_or(1, |index| u64::from(index) + 1);
        let body_binding_bytes = body_count * 64;
        ensure!(body_binding_bytes <= u64::from(limits.max_storage_buffer_binding_size), "body indices exceed device storage limits");
        let mut grouped_instances = Vec::<GpuInstance>::with_capacity(instances.len());
        let mut batches = Vec::with_capacity(meshes.len());
        for (mesh_index, mesh) in meshes.iter().enumerate() {
            ensure!(!mesh.vertices.is_empty() && !mesh.indices.is_empty() && mesh.indices.len() % 3 == 0,
                "mesh {mesh_index} must contain indexed triangles");
            ensure!(mesh.indices.iter().all(|&index| (index as usize) < mesh.vertices.len()), "mesh {mesh_index} has an invalid index");
            let index_count = u32::try_from(mesh.indices.len())?;
            let start = grouped_instances.len() as u32;
            grouped_instances.extend(instances.iter().filter(|instance| instance.mesh == mesh_index).map(GpuInstance::from));
            let end = grouped_instances.len() as u32;
            if start == end { continue; }
            batches.push(MeshBatch {
                vertices: upload(device, "mesh vertices", &mesh.vertices, wgpu::BufferUsages::VERTEX),
                indices: upload(device, "mesh indices", &mesh.indices, wgpu::BufferUsages::INDEX),
                index_count,
                instances: start..end,
            });
        }
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
        let attributes = wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3];
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
            primitive: wgpu::PrimitiveState { cull_mode: Some(wgpu::Face::Back), ..Default::default() },
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
        let pack_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sim-render pack bindings"),
            entries: &[
                texture_entry(0, wgpu::TextureSampleType::Float { filterable: false }),
                texture_entry(1, wgpu::TextureSampleType::Float { filterable: false }),
                texture_entry(2, wgpu::TextureSampleType::Uint),
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
            meshes: batches, instances, cameras, globals, body_binding_bytes,
            color: Target::new(device, output, wgpu::TextureFormat::Rgba8Unorm),
            optical_depth: Target::new(device, output, wgpu::TextureFormat::R32Float),
            object_id: Target::new(device, output, wgpu::TextureFormat::R32Uint),
            raster_depth: Target::new(device, output, wgpu::TextureFormat::Depth32Float),
        })
    }

    pub fn output_layout(&self) -> OutputLayout { self.output }

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
                ],
            })
        }).collect();
        let pack = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sim-render output bindings"),
            layout: &self.pack_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&self.color.array) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&self.optical_depth.array) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&self.object_id.array) },
                buffer_entry(3, output, self.output.total_bytes),
            ],
        });
        Ok(RenderBindings { cameras: camera_bindings, pack })
    }

    /// Record all camera draws and one texture-to-contiguous-buffer compute pass.
    /// `bindings` must have been created by this renderer. There are no per-frame
    /// host vectors, resource creations, queue writes, submissions, or readbacks.
    pub fn render(&self, encoder: &mut wgpu::CommandEncoder, bindings: &RenderBindings) {
        let background = self.config.background;
        for (camera_index, camera) in bindings.cameras.iter().enumerate() {
            let attachments = [
                Some(color_attachment(&self.color.layers[camera_index], wgpu::Color {
                    r: background[0], g: background[1], b: background[2], a: background[3],
                })),
                Some(color_attachment(&self.optical_depth.layers[camera_index], wgpu::Color::TRANSPARENT)),
                Some(color_attachment(&self.object_id.layers[camera_index], wgpu::Color::TRANSPARENT)),
            ];
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("sim-render camera"),
                color_attachments: &attachments,
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.raster_depth.layers[camera_index],
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Discard }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });
            pass.set_pipeline(&self.render_pipeline);
            pass.set_bind_group(0, camera, &[]);
            for mesh in &self.meshes {
                pass.set_vertex_buffer(0, mesh.vertices.slice(..));
                pass.set_index_buffer(mesh.indices.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..mesh.index_count, 0, mesh.instances.clone());
            }
        }
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("sim-render contiguous output"),
            timestamp_writes: None,
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
        ty: wgpu::BindingType::Texture { sample_type, view_dimension: wgpu::TextureViewDimension::D2Array, multisampled: false },
        count: None,
    }
}

fn color_attachment(view: &wgpu::TextureView, clear: wgpu::Color) -> wgpu::RenderPassColorAttachment<'_> {
    wgpu::RenderPassColorAttachment {
        view, depth_slice: None, resolve_target: None,
        ops: wgpu::Operations { load: wgpu::LoadOp::Clear(clear), store: wgpu::StoreOp::Store },
    }
}
