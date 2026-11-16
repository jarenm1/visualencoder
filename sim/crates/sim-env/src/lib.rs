//! GPU-resident, first-person gap-flight task. See `sim_env.h` for the C ABI.
//! Build: cargo oxide build --arch sm_120 --device-codegen-crate sim_physics,sim_env -- -p sim-env --release
//! Both CUDA modules use NVIDIA's image-anchor loader, so ctypes loads kernels
//! from this cdylib's embedded artifact rather than the Python executable.
mod ffi;
mod kernels;
mod task;

use std::{ffi::c_void, sync::Arc};
use anyhow::{Result, ensure};
use cuda_core::{CudaStream, DeviceBuffer, Function, launch_kernel_on_stream};
use glam::{Mat4, Vec3};
use sim_interop::{Exchange, Gpu, SharedBuffer};
use sim_physics::{ImageShape, Physics};
use sim_render::{Camera, CameraGpu, Instance, Lighting, MeshData, RenderBindings, RenderConfig, Renderer};
use sim_state::{BodyAction, BodyState, StepParams};
use kernels::{Buffers, Config, TaskState};

struct TaskKernels {
    reset: Function,
    control: Function,
    finish: Function,
}

impl TaskKernels {
    fn load(gpu: &Gpu) -> Result<Self> {
        let loaded = kernels::device::load(gpu.cuda_context())?;
        let module = loaded.as_cuda_module();
        Ok(Self {
            reset: module.load_function(kernels::device::reset_CUDA_SIGNATURE.entry)?,
            control: module.load_function(kernels::device::control_CUDA_SIGNATURE.entry)?,
            finish: module.load_function(kernels::device::finish_CUDA_SIGNATURE.entry)?,
        })
    }

    unsafe fn launch(&self, function: &Function, stream: &CudaStream,
        mut buffers: Buffers, mut config: Config, actions: Option<*const i64>) -> Result<()> {
        let mut actions = actions.unwrap_or(std::ptr::null());
        let mut args = [
            (&raw mut buffers).cast::<c_void>(),
            (&raw mut config).cast::<c_void>(),
            (&raw mut actions).cast::<c_void>(),
        ];
        let count = if actions.is_null() { 2 } else { 3 };
        unsafe {
            launch_kernel_on_stream(function, (config.n.div_ceil(256), 1, 1),
                (256, 1, 1), 0, stream, &mut args[..count])?;
        }
        Ok(())
    }
}

/// Owns all device allocations for one batch. Calls must be serialized by the
/// caller. All exported tensor storage is ordinary CUDA-only memory, not Vulkan
/// memory, and remains valid until this object is destroyed. The caller must
/// use `stream()` or establish explicit CUDA stream dependencies for every view.
pub struct Env {
    exchange: Exchange,
    renderer: Renderer,
    bindings: RenderBindings,
    physics: Physics,
    kernels: TaskKernels,
    state: Arc<SharedBuffer>,
    cameras: Arc<SharedBuffer>,
    rendered: Arc<SharedBuffer>,
    forces: DeviceBuffer<u8>,
    task: DeviceBuffer<u8>,
    outputs: [DeviceBuffer<u8>; 8],
    config: Config,
    shape: ImageShape,
    step_params: StepParams,
    gpu: Arc<Gpu>,
}

impl Env {
    pub fn new(n: u32, seed: u64, width: u32, height: u32, style: u32) -> Result<Self> {
        ensure!(n > 0 && n <= u32::MAX / 3, "environment count must be positive and body indices must fit u32");
        ensure!(style <= 2, "style must be 0 (training), 1 (validation), or 2 (final test)");
        let shape = ImageShape::new(width, height, n)?;
        let gpu = Gpu::new(0)?;
        let physics = Physics::load(gpu.cuda_context())?;
        let kernels = TaskKernels::load(&gpu)?;
        let (render_config, instances, camera_list) = scene(n, seed, width, height, style);
        let renderer = Renderer::new(gpu.device(), render_config,
            &[MeshData::cube(), MeshData::plane()], &instances, &camera_list)?;
        let state = SharedBuffer::new(gpu.clone(), u64::from(n) * 3 * size_of::<BodyState>() as u64,
            wgpu::BufferUsages::STORAGE, "gap-flight bodies")?;
        let cameras = SharedBuffer::new(gpu.clone(), u64::from(n) * size_of::<CameraGpu>() as u64,
            wgpu::BufferUsages::STORAGE, "gap-flight cameras")?;
        let rendered = SharedBuffer::new(gpu.clone(), renderer.output_layout().total_bytes,
            wgpu::BufferUsages::STORAGE, "gap-flight render output")?;
        let bindings = renderer.bind(gpu.device(), state.buffer(), rendered.buffer(), Some(cameras.buffer()))?;
        let exchange = Exchange::new(gpu.clone(), vec![state.clone(), cameras.clone(), rendered.clone()])?;
        let allocate = |bytes: usize| -> Result<DeviceBuffer<u8>> {
            // All bytes are written by reset/render before a tensor is exposed.
            Ok(unsafe { DeviceBuffer::<u8>::uninitialized_async(gpu.cuda_stream(), bytes)? })
        };
        let count = n as usize;
        let forces = allocate(count * size_of::<BodyAction>())?;
        let task = allocate(count * size_of::<TaskState>())?;
        let outputs = [
            allocate(shape.rgb_bytes())?, allocate(count * 8 * 4)?,
            allocate(count * 4)?, allocate(count * 4)?, allocate(count * 4)?,
            allocate(count * 4 * 4)?, allocate(count * 4)?, allocate(count * 8)?,
        ];
        let projection = Mat4::perspective_rh(std::f32::consts::FRAC_PI_2,
            width as f32 / height as f32, 0.05, 20.0).to_cols_array();
        let config = Config { n, style, seed, projection_x: projection[0], projection_y: projection[5],
            projection_z: projection[10], projection_w: projection[14] };
        let mut env = Self {
            exchange, renderer, bindings, physics, kernels, state, cameras, rendered,
            forces, task, outputs, config, shape,
            step_params: StepParams::new(1.0 / 120.0, 1.0, 1.0, 0.0, 0.0, [0.0, -9.81, 0.0])?,
            gpu,
        };
        env.reset(seed)?;
        // Startup only: CUDA async allocations and the initial observation are
        // ready before Python creates its borrowed tensor views.
        env.gpu.wait_idle()?;
        Ok(env)
    }

    fn buffers(&self) -> Buffers {
        Buffers {
            bodies: self.state.device_ptr() as *mut BodyState,
            forces: self.forces.cu_deviceptr() as *mut BodyAction,
            task: self.task.cu_deviceptr() as *mut TaskState,
            cameras: self.cameras.device_ptr() as *mut f32,
            proprio: self.outputs[1].cu_deviceptr() as *mut f32,
            rewards: self.outputs[2].cu_deviceptr() as *mut f32,
            done: self.outputs[3].cu_deviceptr() as *mut f32,
            success: self.outputs[4].cu_deviceptr() as *mut f32,
            privileged: self.outputs[5].cu_deviceptr() as *mut f32,
            collision: self.outputs[6].cu_deviceptr() as *mut f32,
            episodes: self.outputs[7].cu_deviceptr() as *mut u64,
        }
    }

    pub fn reset(&mut self, seed: u64) -> Result<()> {
        self.config.seed = seed;
        let buffers = self.buffers();
        let config = self.config;
        let encoder = self.gpu.device().create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("gap-flight reset ownership"),
        });
        unsafe {
            self.exchange.cycle(encoder, || self.kernels.launch(&self.kernels.reset,
                self.gpu.cuda_stream(), buffers, config, None))?;
        }
        self.render_observation()
    }

    /// Actions point to n CUDA int64 values (0=left, 1=neutral, 2=right), on
    /// stream(). No transfer, tensor allocation, or host wait is performed.
    ///
    /// # Safety
    /// Actions must be initialized and remain live until this stream completes;
    /// callers must order any producer before the environment's CUDA stream.
    pub unsafe fn step(&mut self, actions: *const i64) -> Result<()> {
        ensure!(!actions.is_null(), "actions pointer must not be null");
        let buffers = self.buffers();
        let config = self.config;
        let encoder = self.gpu.device().create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("gap-flight physics ownership"),
        });
        unsafe {
            self.exchange.cycle(encoder, || {
                let stream = self.gpu.cuda_stream();
                self.kernels.launch(&self.kernels.control, stream, buffers, config, Some(actions))?;
                for _ in 0..6 {
                    self.physics.step(stream, buffers.bodies, buffers.forces, config.n, self.step_params)?;
                }
                self.kernels.launch(&self.kernels.finish, stream, buffers, config, None)
            })?;
        }
        self.render_observation()
    }

    fn render_observation(&mut self) -> Result<()> {
        let mut encoder = self.gpu.device().create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("gap-flight rendered observation"),
        });
        self.renderer.render(&mut encoder, &self.bindings);
        unsafe {
            self.exchange.cycle(encoder, || {
                self.physics.preprocess_rgba8(self.gpu.cuda_stream(),
                    self.rendered.device_ptr() as *const u8,
                    self.outputs[0].cu_deviceptr() as *mut f32, self.shape)?;
                Ok(())
            })?;
        }
        Ok(())
    }

    pub fn buffer(&self, field: i32) -> Result<u64> {
        ensure!((0..8).contains(&field), "buffer field must be in 0..8");
        Ok(self.outputs[field as usize].cu_deviceptr())
    }

    pub fn stream(&self) -> u64 {
        self.gpu.cuda_stream().cu_stream() as usize as u64
    }
}

impl Drop for Env {
    fn drop(&mut self) {
        // End boundary only. Covers foreign Torch uses on the borrowed stream,
        // raw kernel accesses, module lifetimes, and async allocation retirement.
        let _ = self.gpu.wait_idle();
    }
}

fn scene(n: u32, seed: u64, width: u32, height: u32, style: u32)
    -> (RenderConfig, Vec<Instance>, Vec<Camera>) {
    let mut config = RenderConfig { width, height, ..Default::default() };
    if style == 1 {
        config.lighting = Lighting { direction: [-0.6, 0.8, -0.4], color: [0.65, 0.8, 1.0], ambient: [0.24, 0.12, 0.2] };
        config.background = [0.22, 0.12, 0.28, 1.0];
    } else if style == 2 {
        config.lighting = Lighting { direction: [0.7, 0.5, -0.6], color: [1.0, 0.78, 0.5], ambient: [0.12, 0.24, 0.18] };
        config.background = [0.08, 0.24, 0.2, 1.0];
    }
    let mut instances = Vec::with_capacity(n as usize * 6);
    let mut cameras = Vec::with_capacity(n as usize);
    for i in 0..n {
        let base = i as f32 * 32.0;
        let random = |lane: u64| (task::episode_gap(seed ^ lane, i, 0, style) + 2.5) / 5.0;
        let color = if style == 0 {
            [0.15 + 0.7 * random(11), 0.15 + 0.7 * random(23), 0.15 + 0.7 * random(37), 1.0]
        } else if style == 1 {
            [0.7 + 0.2 * random(41), 0.1 + 0.2 * random(53), 0.6 + 0.3 * random(67), 1.0]
        } else {
            [0.75 + 0.2 * random(41), 0.55 + 0.3 * random(53), 0.05 + 0.15 * random(67), 1.0]
        };
        for side in 0..2 {
            let mut wall = Instance::body(0, n + i * 2 + side);
            wall.scale = [8.0, 3.0, 0.3];
            wall.material.color = color;
            wall.material.checker_scale = 1.5 + random(79);
            wall.material.checker_contrast = 0.15;
            wall.object_id = 1 + i * 6 + side;
            instances.push(wall);
        }
        let mut floor = Instance::static_mesh(1);
        floor.position = [base, 0.0, -6.0];
        floor.scale = [16.0, 1.0, 16.0];
        floor.material.color = [0.12 + random(83) * 0.25, 0.2, 0.22, 1.0];
        floor.material.checker_scale = 1.0;
        floor.material.checker_contrast = 0.4;
        floor.object_id = 3 + i * 6;
        instances.push(floor);
        for side in 0..2 {
            let mut boundary = Instance::static_mesh(0);
            boundary.position = [base + if side == 0 { -4.15 } else { 4.15 }, 1.5, -6.0];
            boundary.scale = [0.3, 3.0, 14.0];
            boundary.material.color = [0.18, 0.28 + 0.2 * random(89), 0.32, 1.0];
            boundary.material.checker_scale = 1.0;
            boundary.object_id = 4 + i * 6 + side;
            instances.push(boundary);
        }
        let mut goal = Instance::static_mesh(0);
        goal.position = [base, 1.2, -12.5];
        goal.scale = [1.4, 1.4, 0.1];
        goal.material.color = [0.15, 0.95, 0.25, 1.0];
        goal.object_id = 6 + i * 6;
        instances.push(goal);
        cameras.push(Camera { eye: Vec3::new(base, 1.2, 0.0), target: Vec3::new(base, 1.2, -1.0),
            up: Vec3::Y, vertical_fov_radians: std::f32::consts::FRAC_PI_2, near: 0.05, far: 20.0 });
    }
    (config, instances, cameras)
}
