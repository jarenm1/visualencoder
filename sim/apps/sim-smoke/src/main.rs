//! Real Vulkan rasterization → shared allocation → Rust CUDA → Vulkan reuse.
//! Host transfers are restricted to initialization and explicit final inspection.

use std::{fs, io::Write, path::PathBuf, time::Instant};

use anyhow::{Context, Result, bail, ensure};
use glam::Vec3;
use sim_interop::{Exchange, Gpu, SharedBuffer};
use sim_physics::{ImageShape, ImageStatistics, Physics};
use sim_render::{Camera, Instance, MeshData, RenderConfig, Renderer};
use sim_state::{BodyAction, BodyState, StepParams};

const WIDTH: u32 = 320;
const HEIGHT: u32 = 240;
const DT: f32 = 1.0 / 120.0;

struct Options {
    frames: u32,
    cameras: u32,
    output: PathBuf,
}

impl Options {
    fn parse() -> Result<Self> {
        let mut options = Self {
            frames: 64,
            cameras: 2,
            output: PathBuf::from("smoke.ppm"),
        };
        let mut arguments = std::env::args().skip(1);
        while let Some(argument) = arguments.next() {
            match argument.as_str() {
                "--frames" => options.frames = arguments.next().context("--frames requires a count")?.parse()?,
                "--cameras" => options.cameras = arguments.next().context("--cameras requires a count")?.parse()?,
                "--output" => options.output = arguments.next().context("--output requires a path")?.into(),
                "--help" => {
                    println!("sim-smoke [--frames 64] [--cameras 2] [--output smoke.ppm]");
                    std::process::exit(0);
                }
                _ => bail!("unknown argument {argument:?}"),
            }
        }
        ensure!(options.frames > 0 && options.cameras > 0, "frame and camera counts must be positive");
        Ok(options)
    }
}

fn main() -> Result<()> {
    let options = Options::parse()?;
    let gpu = Gpu::new(0)?;
    let physics = Physics::load(gpu.cuda_context())?;
    let shape = ImageShape::new(WIDTH, HEIGHT, options.cameras)?;
    let params = StepParams::new(DT, 1.0, 1.0, 0.0, 0.0, [0.0, -9.81, 0.0])?;
    let bodies = [
        BodyState {
            position: [-0.6, 0.8, 0.0, 0.0],
            velocity: [0.4, 0.0, 0.0, 0.0],
            rotation: [0.0, 0.0, 0.0, 1.0],
            angular_velocity: [0.0, 0.0, 0.0, 0.0],
        },
        BodyState {
            position: [1.0, 0.6, -1.0, 0.0],
            velocity: [-0.2, 0.0, 0.0, 0.0],
            rotation: [0.0, 0.0, 0.0, 1.0],
            angular_velocity: [0.0, 0.0, 0.0, 0.0],
        },
    ];
    let actions = [BodyAction { force: [0.0, 9.81, 0.0, 0.0], torque: [0.0, 0.2, 0.0, 0.0] }; 2];
    let state = SharedBuffer::new(gpu.clone(), size_of_val(&bodies) as u64, wgpu::BufferUsages::STORAGE, "body state")?;
    let action = SharedBuffer::new(gpu.clone(), size_of_val(&actions) as u64, wgpu::BufferUsages::STORAGE, "body forces")?;

    let mut cube = Instance::body(0, 0);
    cube.object_id = 2;
    cube.material.color = [0.8, 0.12, 0.04, 1.0];
    let mut second_cube = Instance::body(0, 1);
    second_cube.object_id = 3;
    second_cube.scale = [0.7; 3];
    second_cube.material.color = [0.05, 0.35, 0.8, 1.0];
    let mut ground = Instance::static_mesh(1);
    ground.scale = [20.0, 1.0, 20.0];
    ground.object_id = 1;
    ground.material.color = [0.2, 0.35, 0.15, 1.0];
    ground.material.checker_scale = 1.0;
    ground.material.checker_contrast = 0.35;
    let cameras: Vec<_> = (0..options.cameras).map(|index| Camera {
        eye: if index % 2 == 0 { Vec3::new(3.0, 2.5, 5.0) } else { Vec3::new(-3.0, 2.0, 4.0) },
        target: Vec3::new(0.0, 0.7, 0.0),
        up: Vec3::Y,
        vertical_fov_radians: 60.0_f32.to_radians(),
        near: 0.1,
        far: 100.0,
    }).collect();
    let renderer = Renderer::new(gpu.device(), RenderConfig { width: WIDTH, height: HEIGHT, ..Default::default() },
                                 &[MeshData::cube(), MeshData::plane()], &[cube, second_cube, ground], &cameras)?;
    let layout = renderer.output_layout();
    let pixels = SharedBuffer::new(gpu.clone(), layout.total_bytes, wgpu::BufferUsages::STORAGE, "RGBA depth IDs")?;
    let rgb = SharedBuffer::new(gpu.clone(), shape.rgb_bytes() as u64, wgpu::BufferUsages::STORAGE, "CUDA planar RGB")?;
    let statistics = SharedBuffer::new(gpu.clone(), (size_of::<ImageStatistics>() * options.cameras as usize) as u64,
                                       wgpu::BufferUsages::STORAGE, "CUDA image statistics")?;
    let bindings = renderer.bind(gpu.device(), state.buffer(), pixels.buffer(), None)?;
    let mut exchange = Exchange::new(gpu.clone(), vec![state.clone(), action.clone(), pixels.clone(), rgb.clone(), statistics.clone()])?;
    let state_pointer = state.device_ptr() as *mut BodyState;
    let action_pointer = action.device_ptr() as *const BodyAction;
    let pixel_pointer = pixels.device_ptr() as *const u8;
    let rgb_pointer = rgb.device_ptr() as *mut f32;
    let stats_pointer = statistics.device_ptr() as *mut ImageStatistics;

    // Startup upload only. The host arrays stay alive through the following wait.
    let encoder = gpu.device().create_command_encoder(&Default::default());
    // SAFETY: one Exchange owns all shared allocations, with no other submissions;
    // every CUDA access uses its stream and buffers outlive all queued operations.
    unsafe {
        exchange.cycle(encoder, || {
            cuda_core::memcpy_htod_async(state.device_ptr(), bodies.as_ptr(), bodies.len(), gpu.cuda_stream())?;
            cuda_core::memcpy_htod_async(action.device_ptr(), actions.as_ptr(), actions.len(), gpu.cuda_stream())?;
            Ok(())
        })?;
    }
    gpu.wait_idle()?;

    let started = Instant::now();
    for _ in 0..options.frames {
        let mut encoder = gpu.device().create_command_encoder(&Default::default());
        renderer.render(&mut encoder, &bindings);
        // SAFETY: render → CUDA → render ownership is serialized by Exchange.
        // Physics updates feed the NEXT render; no host state is read or staged.
        unsafe {
            exchange.cycle(encoder, || {
                physics.preprocess_rgba8(gpu.cuda_stream(), pixel_pointer, rgb_pointer, shape)?;
                physics.step(gpu.cuda_stream(), state_pointer, action_pointer, bodies.len() as u32, params)?;
                Ok(())
            })?;
        }
    }
    // One batch-boundary wait, never a per-frame host wait.
    gpu.wait_idle()?;
    let elapsed = started.elapsed();

    let mut final_bodies = bodies;
    let mut final_pixels = vec![0_u8; layout.total_bytes as usize];
    let mut final_rgb = vec![0.0_f32; shape.rgb_bytes() / size_of::<f32>()];
    let mut final_stats = vec![ImageStatistics::default(); options.cameras as usize];
    let encoder = gpu.device().create_command_encoder(&Default::default());
    // Explicit inspection phase, OUTSIDE the profiled/hot loop. Keep host
    // destinations alive until wait_idle completes both APIs' queued work.
    unsafe {
        exchange.cycle(encoder, || {
            physics.image_statistics(gpu.cuda_stream(), rgb_pointer, stats_pointer, shape)?;
            cuda_core::memcpy_dtoh_async(final_bodies.as_mut_ptr(), state.device_ptr(), final_bodies.len(), gpu.cuda_stream())?;
            cuda_core::memcpy_dtoh_async(final_pixels.as_mut_ptr(), pixels.device_ptr(), final_pixels.len(), gpu.cuda_stream())?;
            cuda_core::memcpy_dtoh_async(final_rgb.as_mut_ptr(), rgb.device_ptr(), final_rgb.len(), gpu.cuda_stream())?;
            cuda_core::memcpy_dtoh_async(final_stats.as_mut_ptr(), statistics.device_ptr(), final_stats.len(), gpu.cuda_stream())?;
            Ok(())
        })?;
    }
    gpu.wait_idle()?;
    verify(&bodies, &final_bodies, &final_pixels, &final_rgb, &final_stats, options.frames, options.cameras)?;
    save_ppm(&options.output, &final_pixels[..(WIDTH * HEIGHT * 4) as usize])?;
    println!("PASS: {} Vulkan/CUDA handoffs; {} cameras at {}x{}; {:.2} ms total ({:.1} camera frames/s)",
             options.frames, options.cameras, WIDTH, HEIGHT, elapsed.as_secs_f64() * 1000.0,
             f64::from(options.frames) * f64::from(options.cameras) / elapsed.as_secs_f64());
    println!("PASS: analytic body translation, normalized rotation, every RGBA→NCHW pixel, metric depth and object IDs");
    println!("Final camera image: {}", options.output.display());
    Ok(())
}

fn verify(initial: &[BodyState], final_bodies: &[BodyState], pixels: &[u8], rgb: &[f32],
          statistics: &[ImageStatistics], frames: u32, cameras: u32) -> Result<()> {
    for (before, after) in initial.iter().zip(final_bodies) {
        let expected_x = before.position[0] + before.velocity[0] * DT * frames as f32;
        ensure!((after.position[0] - expected_x).abs() < 1e-4, "CUDA translation differs from analytic integration");
        ensure!((after.position[1] - before.position[1]).abs() < 1e-4, "gravity cancellation failed");
        let norm = after.rotation.iter().map(|v| v * v).sum::<f32>();
        ensure!(norm.is_finite() && (norm - 1.0).abs() < 1e-4, "invalid CUDA quaternion");
        ensure!(after.rotation[1].abs() > 1e-6, "nonzero torque did not rotate the body");
    }
    let per_camera = (WIDTH * HEIGHT) as usize;
    let total_pixels = per_camera * cameras as usize;
    for camera in 0..cameras as usize {
        for pixel in 0..per_camera {
            for channel in 0..3 {
                let source = pixels[(camera * per_camera + pixel) * 4 + channel] as f32 / 255.0;
                let actual = rgb[camera * per_camera * 3 + channel * per_camera + pixel];
                ensure!(actual.is_finite() && (actual - source).abs() < 1e-6, "CUDA channel packing mismatch");
            }
        }
        let stat = &statistics[camera];
        ensure!(stat.pixel_count == per_camera as u32, "CUDA statistics lost pixels");
        ensure!(stat.mean_luminance.is_finite() && stat.max_luminance > stat.min_luminance,
                "rendered camera has no visible scene variation");
    }
    let ids_offset = total_pixels * 8;
    let depth_offset = total_pixels * 4;
    let mut ground_pixels = 0;
    let mut cube_pixels = 0;
    for pixel in 0..total_pixels {
        let id = u32::from_le_bytes(pixels[ids_offset + pixel * 4..ids_offset + pixel * 4 + 4].try_into()?);
        if id == 0 { continue; }
        let depth = f32::from_le_bytes(pixels[depth_offset + pixel * 4..depth_offset + pixel * 4 + 4].try_into()?);
        ensure!(depth.is_finite() && depth > 0.0, "invalid optical-Z depth for visible geometry");
        if id == 1 { ground_pixels += 1; }
        if id == 2 || id == 3 { cube_pixels += 1; }
    }
    ensure!(ground_pixels > 0 && cube_pixels > 0, "expected ground and moving meshes are not visible");
    Ok(())
}

fn save_ppm(path: &std::path::Path, rgba: &[u8]) -> Result<()> {
    let mut data = Vec::with_capacity((WIDTH * HEIGHT * 3) as usize);
    for pixel in rgba.chunks_exact(4) { data.extend_from_slice(&pixel[..3]); }
    let mut file = fs::File::create(path)?;
    write!(file, "P6\n{WIDTH} {HEIGHT}\n255\n")?;
    file.write_all(&data)?;
    Ok(())
}
