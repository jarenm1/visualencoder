use std::{io::Write, sync::{Arc, mpsc}};

use anyhow::{Context, Result, ensure};
use glam::Vec3;
use sim_interop::Gpu;
use sim_render::{Camera, CameraGpu, RenderBindings, RenderConfig, Renderer};
use sim_scene::CityScene;
use sim_state::BodyState;

use crate::{proxy, recording::Recording};

pub struct Capture {
    gpu: Arc<Gpu>,
    renderer: Renderer,
    bindings: RenderBindings,
    bodies: wgpu::Buffer,
    camera: wgpu::Buffer,
    output: wgpu::Buffer,
    readback: wgpu::Buffer,
    frame_bytes: u64,
    aspect: f32,
    send: mpsc::SyncSender<Result<(), wgpu::BufferAsyncError>>,
    receive: mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>,
}

impl Capture {
    pub fn new(scene: &CityScene, recording: &Recording) -> Result<Self> {
        let gpu = Gpu::new(0).context("creating offline Vulkan device")?;
        let camera = chase_camera(scene, &recording.environments[0].states[0])?;
        let config = RenderConfig { width: recording.frame_width, height: recording.frame_height, ..Default::default() };
        // SAFETY: startup texture uploads only; the queue is not retained or
        // reentered, and no exchange or shared CUDA allocations are involved.
        let renderer = unsafe { gpu.with_queue(|queue| Renderer::new(
            gpu.device(), queue, config, &scene.meshes, &scene.instances, &[camera],
        )) }?;
        let layout = renderer.output_layout();
        let buffer = |label, size, usage| gpu.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some(label), size, usage, mapped_at_creation: false,
        });
        let bodies = buffer("replay proxy body states", (size_of::<BodyState>() * 2) as u64,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST);
        let camera = buffer("replay chase camera", size_of::<CameraGpu>() as u64,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST);
        let output = buffer("replay packed RGBA depth IDs", layout.total_bytes,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC);
        let frame_bytes = layout.camera_stride_bytes;
        let readback = buffer("offline RGBA readback", frame_bytes,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST);
        let bindings = renderer.bind(gpu.device(), &bodies, &output, Some(&camera))?;
        let (send, receive) = mpsc::sync_channel(1);
        Ok(Self { gpu, renderer, bindings, bodies, camera, output, readback, frame_bytes,
            aspect: recording.frame_width as f32 / recording.frame_height as f32, send, receive })
    }

    pub fn write_frame(&self, scene: &CityScene, state: &[f32; 8], destination: &mut impl Write) -> Result<()> {
        let bodies = proxy::bodies(state);
        let camera = chase_camera(scene, state)?.gpu(self.aspect);
        self.gpu.write_buffer(&self.bodies, 0, bytemuck::cast_slice(&bodies));
        self.gpu.write_buffer(&self.camera, 0, bytemuck::bytes_of(&camera));
        let mut encoder = self.gpu.device().create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("offline replay frame and explicit CPU transfer"),
        });
        self.renderer.render(&mut encoder, &self.bindings);
        encoder.copy_buffer_to_buffer(&self.output, 0, &self.readback, 0, self.frame_bytes);
        self.gpu.submit([encoder.finish()]);
        let send = self.send.clone();
        self.readback.slice(..).map_async(wgpu::MapMode::Read, move |result| {
            // A dropped receiver means export has already failed; do not panic
            // from a device callback during cleanup.
            let _ = send.send(result);
        });
        // This blocking transfer is intentionally OFFLINE export, never the
        // training/render interop path. Reuse the same readback each frame and
        // write its mapped view directly to FFmpeg without another CPU copy.
        self.gpu.device().poll(wgpu::PollType::wait_indefinitely()).context("waiting for offline RGBA transfer")?;
        self.receive.recv().context("receiving offline map completion")?.context("mapping offline RGBA frame")?;
        let result = (|| {
            let view = self.readback.slice(..).get_mapped_range().context("reading mapped offline RGBA frame")?;
            destination.write_all(&view).context("writing third-person RGBA frame to ffmpeg")
        })();
        self.readback.unmap();
        result
    }
}

fn chase_camera(scene: &CityScene, state: &[f32; 8]) -> Result<Camera> {
    let target = Vec3::new(state[0], state[1], state[2]);
    let goal = Vec3::new(state[3], state[4], state[5]);
    let forward = proxy::forward(target, goal);
    let right = forward.cross(Vec3::Y);
    let radius = 0.12;
    ensure!(!scene.sweep_hits(target.to_array(), target.to_array(), radius),
        "recorded drone pose is inside city geometry; no collision-safe chase camera origin");
    let desired = -forward * 4.6 + Vec3::Y * 2.5;
    let mut offset = clip(scene, target, desired, radius);
    // Prefer the normal behind-and-above camera, even when mildly clipped.
    // In a narrow alley, choose a longer unobstructed elevated/side view instead
    // of putting the camera inside a wall or collapsing it into the proxy.
    if offset.length() < 2.0 {
        for candidate in [
            -forward * 2.0 + Vec3::Y * 4.5,
            -forward * 0.3 + Vec3::Y * 5.0,
            -forward * 2.5 + right * 3.0 + Vec3::Y * 2.5,
            -forward * 2.5 - right * 3.0 + Vec3::Y * 2.5,
            forward * 3.5 + Vec3::Y * 2.5,
        ] {
            let clear = clip(scene, target, candidate, radius);
            if clear.length_squared() > offset.length_squared() {
                offset = clear;
            }
        }
    }
    ensure!(offset.length_squared() > 0.01, "no collision-safe chase camera position around recorded pose");
    Ok(Camera { eye: target + offset, target, up: Vec3::Y,
        vertical_fov_radians: 60.0_f32.to_radians(), near: 0.025,
        far: (scene.half_extent * 4.0).max(100.0) })
}

fn clip(scene: &CityScene, origin: Vec3, offset: Vec3, radius: f32) -> Vec3 {
    let hits = |fraction: f32| scene.sweep_hits(origin.to_array(), (origin + offset * fraction).to_array(), radius);
    if !hits(1.0) {
        return offset;
    }
    // Exact triangle swept spheres, not layout/AABB proxies. Binary search is
    // bounded and the final margin keeps the near plane away from the surface.
    let (mut low, mut high) = (0.0, 1.0);
    for _ in 0..12 {
        let middle = (low + high) * 0.5;
        if hits(middle) { high = middle; } else { low = middle; }
    }
    offset * (low - 0.04 / offset.length()).max(0.0)
}
