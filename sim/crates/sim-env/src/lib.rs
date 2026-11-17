//! GPU-resident city block navigation. See `sim_env.h` for ABI v2.
//! Build with cargo oxide and device-codegen-crate sim_physics,sim_env.
mod ffi;
mod kernels;
mod routes;
mod task;

use std::{ffi::{CString, c_char, c_void}, path::Path, sync::Arc};
use anyhow::{Result, ensure};
use cuda_core::{CudaStream, DeviceBuffer, Function, launch_kernel_on_stream};
use glam::{Mat4, Vec3};
use sim_interop::{Exchange, Gpu, SharedBuffer};
use sim_physics::{ImageShape, Physics};
use sim_render::{Camera, CameraGpu, RenderBindings, RenderConfig, Renderer};
use sim_scene::CityScene;
use sim_state::{BodyState, collision::{BvhNode, Triangle}};
use kernels::{Buffers, Config, TaskState};
use routes::Catalogs;
use task::*;

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
    task: DeviceBuffer<u8>,
    nodes: DeviceBuffer<u8>,
    triangles: DeviceBuffer<u8>,
    routes: DeviceBuffer<u8>,
    waypoints: DeviceBuffer<u8>,
    outputs: [DeviceBuffer<u8>; 9],
    config: Config,
    shape: ImageShape,
    metadata: CString,
    gpu: Arc<Gpu>,
}

impl Env {
    pub fn new(n: u32, seed: u64, width: u32, height: u32, style: u32, city_path: &Path) -> Result<Self> {
        ensure!(n > 0, "environment count must be positive");
        ensure!(style <= 2, "style must be 0 (training), 1 (validation), or 2 (final test)");
        let shape = ImageShape::new(width, height, n)?;
        let city_path = city_path.canonicalize()?;
        let city = CityScene::load(&city_path)?;
        ensure!(!city.triangles.is_empty() && !city.nodes.is_empty(), "city collision geometry is empty");
        ensure!(city.nodes.len() <= u32::MAX as usize && city.triangles.len() <= u32::MAX as usize,
            "city collision index counts exceed u32");
        let catalogs = Catalogs::build(&city)?;
        let metadata = task_metadata(&city, &catalogs, &city_path, style)?;
        let gpu = Gpu::new(0)?;
        let physics = Physics::load(gpu.cuda_context())?;
        let kernels = TaskKernels::load(&gpu)?;
        let camera_list = cameras(n, &catalogs, style);
        let render_config = RenderConfig { width, height, ..Default::default() };
        let device = gpu.device();
        // SAFETY: the queue is used only for startup-owned texture uploads; it
        // is not retained, no exchange buffers exist yet, and Gpu is not reentered.
        let renderer = unsafe { gpu.with_queue(|queue| Renderer::new(
            device, queue, render_config, &city.meshes, &city.instances, &camera_list))? };
        // Independent agents share one static city. Agents have no render mesh,
        // so one camera can never see another environment's body.
        let state = SharedBuffer::new(gpu.clone(), u64::from(n) * size_of::<BodyState>() as u64,
            wgpu::BufferUsages::STORAGE, "city agent bodies")?;
        let cameras = SharedBuffer::new(gpu.clone(), u64::from(n) * size_of::<CameraGpu>() as u64,
            wgpu::BufferUsages::STORAGE, "city agent cameras")?;
        let rendered = SharedBuffer::new(gpu.clone(), renderer.output_layout().total_bytes,
            wgpu::BufferUsages::STORAGE, "city render output")?;
        let bindings = renderer.bind(gpu.device(), state.buffer(), rendered.buffer(), Some(cameras.buffer()))?;
        let exchange = Exchange::new(gpu.clone(), vec![state.clone(), cameras.clone(), rendered.clone()])?;
        let allocate = |bytes: usize| -> Result<DeviceBuffer<u8>> {
            Ok(unsafe { DeviceBuffer::<u8>::uninitialized_async(gpu.cuda_stream(), bytes)? })
        };
        // Startup-only uploads. No host geometry or route state participates in
        // subsequent control, collision, reward, reset or observation updates.
        let nodes = DeviceBuffer::from_host(gpu.cuda_stream(), bytemuck::cast_slice(&city.nodes))?;
        let triangles = DeviceBuffer::from_host(gpu.cuda_stream(), bytemuck::cast_slice(&city.triangles))?;
        let routes = DeviceBuffer::from_host(gpu.cuda_stream(), bytemuck::cast_slice(&catalogs.routes))?;
        let waypoints = DeviceBuffer::from_host(gpu.cuda_stream(), bytemuck::cast_slice(&catalogs.waypoints))?;
        let count = n as usize;
        let task = allocate(count * size_of::<TaskState>())?;
        let outputs = [
            allocate(shape.rgb_bytes())?, allocate(count * 8 * 4)?,
            allocate(count * 4)?, allocate(count * 4)?, allocate(count * 4)?,
            allocate(count * 4 * 4)?, allocate(count * 4)?, allocate(count * 8)?,
            allocate(count * 8 * 4)?,
        ];
        let projection = Mat4::perspective_rh(std::f32::consts::FRAC_PI_2,
            width as f32 / height as f32, 0.05, 80.0).to_cols_array();
        let config = Config { n, route_first: style * ROUTES_PER_SPLIT as u32,
            route_count: ROUTES_PER_SPLIT as u32, node_count: city.nodes.len() as u32, seed,
            projection_x: projection[0], projection_y: projection[5],
            projection_z: projection[10], projection_w: projection[14] };
        let mut env = Self { exchange, renderer, bindings, physics, kernels, state, cameras, rendered,
            task, nodes, triangles, routes, waypoints, outputs, config, shape, metadata, gpu };
        env.reset(seed)?;
        // Startup only: make CUDA-owned tensor storage ready before exposure.
        env.gpu.wait_idle()?;
        Ok(env)
    }

    fn buffers(&self) -> Buffers {
        Buffers {
            bodies: self.state.device_ptr() as *mut BodyState,
            nodes: self.nodes.cu_deviceptr() as *const BvhNode,
            triangles: self.triangles.cu_deviceptr() as *const Triangle,
            routes: self.routes.cu_deviceptr() as *const Route,
            waypoints: self.waypoints.cu_deviceptr() as *const Waypoint,
            task: self.task.cu_deviceptr() as *mut TaskState,
            cameras: self.cameras.device_ptr() as *mut f32,
            proprio: self.outputs[1].cu_deviceptr() as *mut f32,
            rewards: self.outputs[2].cu_deviceptr() as *mut f32,
            done: self.outputs[3].cu_deviceptr() as *mut f32,
            success: self.outputs[4].cu_deviceptr() as *mut f32,
            privileged: self.outputs[5].cu_deviceptr() as *mut f32,
            collision: self.outputs[6].cu_deviceptr() as *mut f32,
            episodes: self.outputs[7].cu_deviceptr() as *mut u64,
            diagnostics: self.outputs[8].cu_deviceptr() as *mut f32,
        }
    }

    pub fn reset(&mut self, seed: u64) -> Result<()> {
        self.config.seed = seed;
        let buffers = self.buffers();
        let config = self.config;
        let encoder = self.gpu.device().create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("city reset ownership"),
        });
        unsafe {
            self.exchange.cycle(encoder, || self.kernels.launch(&self.kernels.reset,
                self.gpu.cuda_stream(), buffers, config, None))?;
        }
        self.render_observation()
    }

    /// Actions point to n CUDA int64 values (0=hover, 1..8 clockwise from forward),
    /// on stream(). No transfer, tensor allocation, or host wait is performed.
    ///
    /// # Safety
    /// Actions must be initialized and remain live until this stream completes;
    /// callers must order any producer before the environment's CUDA stream.
    pub unsafe fn step(&mut self, actions: *const i64) -> Result<()> {
        ensure!(!actions.is_null(), "actions pointer must not be null");
        let buffers = self.buffers();
        let config = self.config;
        let encoder = self.gpu.device().create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("city control ownership"),
        });
        unsafe {
            self.exchange.cycle(encoder, || {
                let stream = self.gpu.cuda_stream();
                self.kernels.launch(&self.kernels.control, stream, buffers, config, Some(actions))?;
                self.kernels.launch(&self.kernels.finish, stream, buffers, config, None)
            })?;
        }
        self.render_observation()
    }

    fn render_observation(&mut self) -> Result<()> {
        let mut encoder = self.gpu.device().create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("city rendered observation"),
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
        ensure!((0..9).contains(&field), "buffer field must be in 0..9");
        Ok(self.outputs[field as usize].cu_deviceptr())
    }

    pub fn stream(&self) -> u64 {
        self.gpu.cuda_stream().cu_stream() as usize as u64
    }

    pub fn metadata(&self) -> *const c_char { self.metadata.as_ptr() }
}

impl Drop for Env {
    fn drop(&mut self) {
        // End boundary only. Covers foreign Torch uses on the borrowed stream,
        // raw kernel accesses, module lifetimes, and async allocation retirement.
        let _ = self.gpu.wait_idle();
    }
}

fn cameras(n: u32, catalogs: &Catalogs, style: u32) -> Vec<Camera> {
    let route = catalogs.routes[style as usize * ROUTES_PER_SPLIT];
    let start = catalogs.waypoints[route.first as usize].position;
    let goal = catalogs.waypoints[(route.first + route.count - 1) as usize].position;
    vec![Camera { eye: Vec3::from_array(start), target: Vec3::from_array(goal), up: Vec3::Y,
        vertical_fov_radians: std::f32::consts::FRAC_PI_2, near: 0.05, far: 80.0 }; n as usize]
}

fn task_metadata(city: &CityScene, catalogs: &Catalogs, path: &Path, style: u32) -> Result<CString> {
    let value = serde_json::json!({
        "abi_version": 2, "task": "city-block-navigation-v1", "action_count": 9,
        "neutral_action": 0, "horizon": HORIZON, "control_hz": 20, "physics_hz": PHYSICS_HZ,
        "city_fingerprint": city.fingerprint, "city_path": path.to_string_lossy(),
        "style": style, "split": (["train", "validation", "test"][style as usize]),
        "altitude_m": ALTITUDE, "speed_m_s": SPEED, "collision_radius_m": DRONE_RADIUS,
        "goal_radius_m": GOAL_RADIUS, "waypoint_radius_m": WAYPOINT_RADIUS,
        "triangle_count": city.triangles.len(), "bvh_node_count": city.nodes.len(),
        "route_counts": {"train": ROUTES_PER_SPLIT, "validation": ROUTES_PER_SPLIT, "test": ROUTES_PER_SPLIT},
        "route_split_definition": {
            "catalog_version": 1, "catalog_seeds": CATALOG_SEEDS,
            "district_centers_xz_m": catalogs.centers, "district_half_size_m": DISTRICT_HALF_SIZE,
            "grid_spacing_m": GRID_SPACING, "planning_radius_m": PLANNING_RADIUS,
            "length_range_m": [MIN_ROUTE_LENGTH, MAX_ROUTE_LENGTH],
            "actual_length_ranges_m": catalogs.lengths,
            "minimum_direct_distance_m": 8.0, "minimum_detour_m": 2.0, "minimum_detour_ratio": 1.08,
            "route_rule": "All splits require direct swept-sphere line blocked at radius .35m. Starts/goals and every path chord are clear at .85m. Samples must be sky-clear to y=256m. Dijkstra shortest paths on the exact-sweep eight-connected 2m grid, then greedy farthest-visible shortcut. 128 distinct unordered start/goal pairs per split, at most four routes per start. Fixed independently seeded catalogs; reset seed selects episodes only; no easy curriculum or fallback."
        },
        "controls": "Altitude exactly stabilized at6m. Immediate3m/s horizontal velocity; six exact swept120Hz substeps per20Hz action. Level gimballed camera yaws toward final goal each observation; action frame held within the control step. One shared city; agents neither render nor collide with each other.",
        "actions": ["hover", "forward", "forward_right", "right", "backward_right", "backward", "backward_left", "left", "forward_left"],
        "reward": {
            "progress_per_m": 0.2, "time_per_step": -0.01, "success": 5.0, "collision": -5.0,
            "potential": "remaining=distance to next waypoint + its suffix polyline distance; reward=.2*(old_remaining-new_remaining)-.01+terminal_bonus; advance nonfinal waypoint within.25m; success actual final-goal distance<=1m; collision takes precedence; horizon320 is terminal with zero extra bonus"
        },
        "proprio": ["goal_right_m=0", "goal_up_m", "goal_view_z_m=-distance", "velocity_right_m_s", "velocity_up_m_s", "velocity_forward_m_s", "remaining_clock_fraction", "previous_action_div8"],
        "privileged": ["next_waypoint_right_m/32", "next_waypoint_forward_m/32", "remaining_path_m/48", "elapsed_clock_fraction"],
        "privileged_convention": "Right positive camera-right; forward positive toward goal-facing camera forward; identical32m scale for waypoint components. Privileged state only for explicit state ceiling, never default visual/proprio actor.",
        "diagnostics": ["world_x", "world_y", "world_z", "goal_x", "goal_y", "goal_z", "route_id", "clock_steps"],
        "autoreset": "Terminal reward/done/success/collision retained; observation and diagnostics belong to reset episode. Episode counter increments on any terminal. Explicit reset clears counters/outputs."
    });
    Ok(CString::new(serde_json::to_string(&value)?)?)
}
