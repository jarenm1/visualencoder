//! GPU-resident city drone visual-goal environment (city-drone-goal-v1, ABI v3).
//! See `sim_env.h` for the ABI. Build with cargo oxide and device-codegen-crate sim_physics,sim_env.
mod ffi;
mod kernels;
mod task;

use std::{ffi::{CString, c_char, c_void}, path::Path, sync::Arc};
use anyhow::{Result, ensure};
use cuda_core::{CudaStream, DeviceBuffer, Function, launch_kernel_on_stream};
use glam::{Mat4, Vec3};
use sim_interop::{Exchange, Gpu, SharedBuffer};
use sim_physics::{ImageShape, Physics};
use sim_render::{Camera, Instance, Material, RenderBindings, RenderConfig, Renderer};
use sim_scene::{CityScene, drone};
use sim_state::BodyState;
use sim_state::collision::{BvhNode, Triangle};
use sim_state::drone as dynamics;
use sim_state::drone::EpisodeSpec;
use kernels::{Buffers, Config, EnvState};
use task::*;

/// Goal camera distance from the design origin (metres): a 0.7-1.0 m design
/// fills roughly half the 75-degree frame, so encoder goal tokens describe the
/// object rather than background.
const GOAL_CAMERA_DISTANCE: f32 = 1.2;
/// Fixed visibility and projection aspect (width / height).
const CAMERA_ASPECT: f32 = 4.0 / 3.0;
/// Goal camera view direction (before normalisation).
const GOAL_CAMERA_DIRECTION: [f32; 3] = [0.7, 0.25, 0.7];

struct TaskKernels {
    reset: Function,
    control: Function,
    finish: Function,
    observe: Function,
}

impl TaskKernels {
    fn load(gpu: &Gpu) -> Result<Self> {
        let loaded = kernels::device::load(gpu.cuda_context())?;
        let module = loaded.as_cuda_module();
        Ok(Self {
            reset: module.load_function(kernels::device::reset_CUDA_SIGNATURE.entry)?,
            control: module.load_function(kernels::device::control_CUDA_SIGNATURE.entry)?,
            finish: module.load_function(kernels::device::finish_CUDA_SIGNATURE.entry)?,
            observe: module.load_function(kernels::device::observe_CUDA_SIGNATURE.entry)?,
        })
    }

    unsafe fn launch(&self, function: &Function, stream: &CudaStream,
        mut buffers: Buffers, mut config: Config, actions: Option<*const f32>) -> Result<()> {
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

/// Target design metadata kept on the host after the meshes move to the renderer.
struct DesignMeta {
    name: String,
    scale: f32,
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
    states: DeviceBuffer<u8>,
    nodes: DeviceBuffer<u8>,
    triangles: DeviceBuffer<u8>,
    catalog: DeviceBuffer<u8>,
    exclusions: DeviceBuffer<u8>,
    outputs: [DeviceBuffer<u8>; FIELD_COUNT],
    config: Config,
    shape: ImageShape,
    metadata: CString,
    gpu: Arc<Gpu>,
}

impl Env {
    pub fn new(n: u32, seed: u64, width: u32, height: u32, style: u32, city_path: &Path) -> Result<Self> {
        ensure!(n > 0, "environment count must be positive");
        ensure!(style <= 2, "style must be 0 (training), 1 (validation), or 2 (final test)");
        ensure!(u64::from(width) * 3 == u64::from(height) * 4,
            "image must be 4:3 (width * 3 == height * 4), got {width}x{height}");
        let shape = ImageShape::new(width, height, n)?;
        let city_path = city_path.canonicalize()?;
        let mut city = CityScene::load(&city_path)?;
        ensure!(!city.triangles.is_empty() && !city.nodes.is_empty(), "city collision geometry is empty");
        ensure!(city.nodes.len() <= u32::MAX as usize && city.triangles.len() <= u32::MAX as usize,
            "city collision index counts exceed u32");

        let catalog = drone::build_catalog(&city, style, CATALOG_SIZE, seed)?;
        ensure!(catalog.episodes.len() == CATALOG_SIZE, "catalog size mismatch");
        let district = drone::district(style, city.half_extent);

        let designs: Vec<_> = drone::target_designs().into_iter().filter(|d| d.split == style).collect();
        ensure!(!designs.is_empty(), "no target designs for style {style}");
        let mut design_meta = Vec::with_capacity(designs.len());
        let mut design_meshes = Vec::with_capacity(designs.len());
        let mut design_materials: Vec<Material> = Vec::with_capacity(designs.len());
        for (k, d) in designs.into_iter().enumerate() {
            ensure!(d.index as usize == k, "design index {} does not match split-local position {k}", d.index);
            design_meta.push(DesignMeta { name: d.name, scale: d.scale });
            design_materials.push(d.material);
            design_meshes.push(d.mesh);
        }
        let k_designs = design_meta.len() as u32;
        let shape_goal = ImageShape::new(width, height, k_designs)?;

        let gpu = Gpu::new(0)?;
        let physics = Physics::load(gpu.cuda_context())?;
        let kernels = TaskKernels::load(&gpu)?;
        let device = gpu.device();
        let render_config = RenderConfig { width, height, ..Default::default() };

        // Bodies: n drones, then n*K target slots (slot = n + i*K + k).
        let body_count = u64::from(n) + u64::from(n) * u64::from(k_designs);
        let state = SharedBuffer::new(gpu.clone(), body_count * size_of::<BodyState>() as u64,
            wgpu::BufferUsages::STORAGE, "city drone and target bodies")?;
        let cameras = SharedBuffer::new(gpu.clone(), u64::from(n) * CAMERA_FLOATS as u64 * 4,
            wgpu::BufferUsages::STORAGE, "city drone cameras")?;

        // Goal renderer: design k alone in camera k, static pose at the origin.
        let goal_eye = Vec3::from(GOAL_CAMERA_DIRECTION).normalize() * GOAL_CAMERA_DISTANCE;
        let goal_cameras: Vec<Camera> = (0..k_designs).map(|_| Camera {
            eye: goal_eye,
            target: Vec3::ZERO,
            up: Vec3::Y,
            vertical_fov_radians: dynamics::CAMERA_VFOV_RAD,
            near: dynamics::CAMERA_NEAR,
            far: dynamics::CAMERA_FAR,
        }).collect();
        let goal_instances: Vec<Instance> = design_meta.iter().enumerate().map(|(k, m)| {
            let mut inst = Instance::static_mesh(k);
            inst.scale = [m.scale; 3];
            inst.material = design_materials[k];
            inst.object_id = 1 + k as u32;
            inst.camera = Some(k as u32);
            inst
        }).collect();
        // SAFETY: the queue is used only for startup-owned texture uploads; it
        // is not retained, no exchange buffers exist yet, and Gpu is not reentered.
        let goal_renderer = unsafe { gpu.with_queue(|queue| Renderer::new(
            device, queue, render_config, &design_meshes, &goal_instances, &goal_cameras))? };
        let goal_rendered = SharedBuffer::new(gpu.clone(), goal_renderer.output_layout().total_bytes,
            wgpu::BufferUsages::STORAGE, "city goal render output")?;
        let goal_bindings = goal_renderer.bind(device, state.buffer(), goal_rendered.buffer(), None)?;

        // Main renderer: city geometry, then design meshes; per-env target instances keyed by body.
        let mut meshes = std::mem::take(&mut city.meshes);
        let design_mesh_base = meshes.len();
        meshes.extend(design_meshes);
        let mut instances = std::mem::take(&mut city.instances);
        let design_object_base = instances.iter().map(|i| i.object_id).max().unwrap_or(0) + 1;
        instances.reserve(n as usize * k_designs as usize);
        for i in 0..n {
            for k in 0..k_designs {
                let body = n + i * k_designs + k;
                let mut inst = Instance::body(design_mesh_base + k as usize, body);
                inst.scale = [design_meta[k as usize].scale; 3];
                inst.material = design_materials[k as usize];
                inst.object_id = design_object_base + k;
                inst.camera = Some(i);
                instances.push(inst);
            }
        }
        // Internal cameras are never bound: the CUDA-written camera buffer supplies every view.
        let internal_cameras: Vec<Camera> = (0..n).map(|_| Camera {
            eye: Vec3::ZERO,
            target: Vec3::NEG_Z,
            up: Vec3::Y,
            vertical_fov_radians: dynamics::CAMERA_VFOV_RAD,
            near: dynamics::CAMERA_NEAR,
            far: dynamics::CAMERA_FAR,
        }).collect();
        // SAFETY: as above.
        let renderer = unsafe { gpu.with_queue(|queue| Renderer::new(
            device, queue, render_config, &meshes, &instances, &internal_cameras))? };
        let rendered = SharedBuffer::new(gpu.clone(), renderer.output_layout().total_bytes,
            wgpu::BufferUsages::STORAGE, "city render output")?;
        let bindings = renderer.bind(device, state.buffer(), rendered.buffer(), Some(cameras.buffer()))?;

        // Startup-only uploads of static geometry and the episode catalog.
        let nodes = DeviceBuffer::from_host(gpu.cuda_stream(), bytemuck::cast_slice(&city.nodes))?;
        let triangles = DeviceBuffer::from_host(gpu.cuda_stream(), bytemuck::cast_slice(&city.triangles))?;
        let catalog_device = DeviceBuffer::from_host(gpu.cuda_stream(), bytemuck::cast_slice(&catalog.episodes))?;
        // Exclusions are read only up to exclusion_count; the single zero rectangle is never read when empty.
        let mut exclusion_rects: Vec<[f32; 4]> = district.exclusions.iter()
            .map(|&(lo, hi)| [lo[0], lo[1], hi[0], hi[1]]).collect();
        let exclusion_count = exclusion_rects.len() as u32;
        if exclusion_rects.is_empty() {
            exclusion_rects.push([0.0; 4]);
        }
        let exclusions = DeviceBuffer::from_host(gpu.cuda_stream(), bytemuck::cast_slice(&exclusion_rects))?;
        let weights_init = [1.0f32 / 3.0; 3];
        let weights = DeviceBuffer::from_host(gpu.cuda_stream(), bytemuck::cast_slice(&weights_init))?;

        let count = n as usize;
        let states = unsafe {
            DeviceBuffer::<u8>::uninitialized_async(gpu.cuda_stream(), count * size_of::<EnvState>())?
        };
        let allocate = |bytes: usize| -> Result<DeviceBuffer<u8>> {
            Ok(unsafe { DeviceBuffer::<u8>::uninitialized_async(gpu.cuda_stream(), bytes)? })
        };
        let outputs: [DeviceBuffer<u8>; FIELD_COUNT] = [
            allocate(shape.rgb_bytes())?,
            allocate(count * SENSOR_DIM * 4)?,
            allocate(count * 4)?,
            allocate(count * 4)?,
            allocate(count * 4)?,
            allocate(count * PRIVILEGED_DIM * 4)?,
            allocate(count * 4)?,
            allocate(count * 8)?,
            allocate(count * DIAGNOSTIC_DIM * 4)?,
            allocate(count * 4)?,
            allocate(shape_goal.rgb_bytes())?,
            weights,
            allocate(count * 4)?,
            allocate(count * 4)?,
        ];

        let mut exchange = Exchange::new(gpu.clone(), vec![state.clone(), cameras.clone(),
            rendered.clone(), goal_rendered.clone()])?;
        let goal_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("city goal observation"),
        });
        let mut goal_encoder = goal_encoder;
        goal_renderer.render(&mut goal_encoder, &goal_bindings);
        unsafe {
            exchange.cycle(goal_encoder, || {
                physics.preprocess_rgba8(gpu.cuda_stream(), goal_rendered.device_ptr() as *const u8,
                    outputs[10].cu_deviceptr() as *mut f32, shape_goal)?;
                Ok(())
            })?;
        }

        let projection = Mat4::perspective_rh(dynamics::CAMERA_VFOV_RAD, CAMERA_ASPECT,
            dynamics::CAMERA_NEAR, dynamics::CAMERA_FAR).to_cols_array();
        let config = Config {
            n,
            designs: k_designs,
            catalog_size: CATALOG_SIZE as u32,
            node_count: city.nodes.len() as u32,
            seed,
            bucket_first: [catalog.bucket_ranges[0][0], catalog.bucket_ranges[1][0], catalog.bucket_ranges[2][0]],
            bucket_count: [catalog.bucket_ranges[0][1], catalog.bucket_ranges[1][1], catalog.bucket_ranges[2][1]],
            exclusion_count,
            district_min: district.min,
            district_max: district.max,
            landable_height: city.landable.height,
            tan_half_vfov: (dynamics::CAMERA_VFOV_RAD * 0.5).tan(),
            aspect: CAMERA_ASPECT,
            projection_x: projection[0],
            projection_y: projection[5],
            projection_z: projection[10],
            projection_w: projection[14],
        };
        let metadata = task_metadata(&city, &catalog, &design_meta, &district, &city_path, style, k_designs)?;
        let mut env = Self {
            exchange,
            renderer,
            bindings,
            physics,
            kernels,
            state,
            cameras,
            rendered,
            states,
            nodes,
            triangles,
            catalog: catalog_device,
            exclusions,
            outputs,
            config,
            shape,
            metadata,
            gpu,
        };
        env.reset(seed)?;
        // Startup only: make CUDA-owned tensor storage ready before exposure.
        env.gpu.wait_idle()?;
        Ok(env)
    }

    fn buffers(&self) -> Buffers {
        Buffers {
            bodies: self.state.device_ptr() as *mut BodyState,
            states: self.states.cu_deviceptr() as *mut EnvState,
            nodes: self.nodes.cu_deviceptr() as *const BvhNode,
            triangles: self.triangles.cu_deviceptr() as *const Triangle,
            catalog: self.catalog.cu_deviceptr() as *const EpisodeSpec,
            exclusions: self.exclusions.cu_deviceptr() as *const [f32; 4],
            weights: self.outputs[11].cu_deviceptr() as *const f32,
            cameras: self.cameras.device_ptr() as *mut f32,
            sensors: self.outputs[1].cu_deviceptr() as *mut f32,
            privileged: self.outputs[5].cu_deviceptr() as *mut f32,
            diagnostics: self.outputs[8].cu_deviceptr() as *mut f32,
            rewards: self.outputs[2].cu_deviceptr() as *mut f32,
            done: self.outputs[3].cu_deviceptr() as *mut f32,
            success: self.outputs[4].cu_deviceptr() as *mut f32,
            collision: self.outputs[6].cu_deviceptr() as *mut f32,
            oob: self.outputs[13].cu_deviceptr() as *mut f32,
            bucket: self.outputs[12].cu_deviceptr() as *mut f32,
            episodes: self.outputs[7].cu_deviceptr() as *mut u64,
            design: self.outputs[9].cu_deviceptr() as *mut i32,
        }
    }

    /// Restarts every environment with `seed`; episode counters restart at 0.
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

    /// Steps all environments with `actions` (n * 4 float32 on stream()).
    /// No transfer, tensor allocation, or host wait is performed.
    ///
    /// # Safety
    /// Actions must be initialized and remain live until this stream completes;
    /// callers must order any producer before the environment's CUDA stream.
    pub unsafe fn step(&mut self, actions: *const f32) -> Result<()> {
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
                self.kernels.launch(&self.kernels.finish, stream, buffers, config, None)?;
                self.kernels.launch(&self.kernels.observe, stream, buffers, config, None)
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
        ensure!((0..FIELD_COUNT as i32).contains(&field), "buffer field must be in 0..{FIELD_COUNT}");
        Ok(self.outputs[field as usize].cu_deviceptr())
    }

    pub fn stream(&self) -> u64 {
        self.gpu.cuda_stream().cu_stream() as usize as u64
    }

    pub fn metadata(&self) -> *const c_char { self.metadata.as_ptr() }
}

impl Drop for Env {
    fn drop(&mut self) {
        let _ = self.gpu.wait_idle();
    }
}

fn task_metadata(city: &CityScene, catalog: &drone::Catalog, designs: &[DesignMeta],
    district: &drone::District, city_path: &Path, style: u32, k_designs: u32) -> Result<CString> {
    let designs_json: Vec<serde_json::Value> = designs.iter().enumerate()
        .map(|(k, m)| serde_json::json!({"index": k, "name": m.name, "scale_m": m.scale}))
        .collect();
    let exclusions: Vec<serde_json::Value> = district.exclusions.iter()
        .map(|(lo, hi)| serde_json::json!({"min": lo, "max": hi})).collect();
    const SPLIT_NAMES: [&str; 3] = ["train", "validation", "test"];
    let mut json = serde_json::json!({
        "abi_version": ABI_VERSION,
        "task": TASK_NAME,
        "city": {
            "path": city_path.display().to_string(),
            "fingerprint": city.fingerprint,
            "half_extent_m": city.half_extent,
        },
        "action": {
            "dim": ACTION_DIM,
            "order": ["thrust", "roll_rate", "pitch_rate", "yaw_rate"],
            "range": [-1.0, 1.0],
            "thrust_map": "thrust = (a0 + 1) / 2 * max_thrust_N; a0 = -1 is zero thrust",
            "rate_map": "omega_cmd_body = [3.5*a2 (pitch, +: nose up), 2.0*a3 (yaw, +: nose left), -3.5*a1 (roll, +: right side down)]",
            "nan": "NaN a0 -> -1, other NaN -> 0",
        },
        "control_hz": dynamics::CONTROL_HZ,
        "physics_hz": dynamics::PHYSICS_HZ,
        "substeps": dynamics::SUBSTEPS,
        "horizon": dynamics::HORIZON,
        "dynamics": {
            "gravity": dynamics::GRAVITY,
            "mass": dynamics::MASS,
            "inertia": dynamics::INERTIA,
            "max_thrust": dynamics::MAX_THRUST,
            "thrust_tau": dynamics::THRUST_TAU,
            "rate_tau": dynamics::RATE_TAU,
            "max_rates": dynamics::MAX_RATES,
            "max_torque": dynamics::MAX_TORQUE,
            "drag": dynamics::DRAG,
            "radius": dynamics::RADIUS,
            "hover_throttle": 2.0 * dynamics::GRAVITY * dynamics::MASS / dynamics::MAX_THRUST - 1.0,
            "quaternion_layout": "xyzw, body-to-world",
        },
        "camera": {
            "vertical_fov_rad": dynamics::CAMERA_VFOV_RAD,
            "aspect": CAMERA_ASPECT,
            "uptilt_rad": dynamics::CAMERA_UPTILT_RAD,
            "offset_body_m": dynamics::CAMERA_OFFSET,
            "near_m": dynamics::CAMERA_NEAR,
            "far_m": dynamics::CAMERA_FAR,
        },
        "success": {"radius_m": dynamics::SUCCESS_RADIUS, "speed_mps": dynamics::SUCCESS_SPEED},
        "landing": {"max_speed_mps": dynamics::LANDING_SPEED, "max_tilt_rad": dynamics::LANDING_TILT_RAD},
        "ceiling_m": dynamics::CEILING,
        "reward": {
            "terms": "(d_prev - d_now) - 0.01 - 0.02*|a - a_prev|^2/4 - 0.005*|w_b|^2/3.5^2 + 10*success - 10*crash - 5*oob",
            "done": "crash || oob || success || step >= horizon; no bootstrap",
        },
        "sensors": {
            "dim": SENSOR_DIM,
            "layout": [
                "0-2 gravity direction in body frame",
                "3-5 body rates (+noise sd 0.02) / 3.5",
                "6-8 body velocity (+noise sd 0.1) / 5",
                "9 height above landable surface (+noise sd 0.03) / 10",
                "10-13 last applied action",
                "14-16 noisy target offset in heading frame (x right, y up, z = -forward) / 32",
                "17 remaining horizon fraction",
            ],
        },
        "privileged": {
            "dim": PRIVILEGED_DIM,
            "layout": [
                "0-2 target in body frame / 32", "3 distance / 32", "4-6 world velocity / 5",
                "7-9 body rates / 3.5", "10-13 quaternion xyzw", "14 target visible (0/1)",
                "15-17 one-hot bucket",
            ],
        },
        "diagnostics": {
            "dim": DIAGNOSTIC_DIM,
            "layout": [
                "0-2 position", "3-5 target", "6-9 quaternion xyzw", "10-12 world velocity",
                "13 thrust N", "14 design", "15 bucket", "16 catalog index", "17 step",
                "18 landed", "19 start landed", "20 initial distance", "21 visible fraction at start",
                "22 episode (low 24 bits)", "23 reserved 0",
            ],
        },
        "designs": designs_json,
        "district": {
            "min": district.min,
            "max": district.max,
            "margin_m": DISTRICT_MARGIN,
            "exclusions": exclusions,
        },
        "landable": {
            "height_m": city.landable.height,
            "triangles": city.landable.triangles,
            "non_level": city.landable.non_level,
        },
        "catalog": {
            "size": CATALOG_SIZE,
            "bucket_ranges": catalog.bucket_ranges,
            "rules": catalog.rules,
        },
        "target_designs": k_designs,
        "fields": [
            {"index": 0, "name": "rgb", "dtype": "float32", "shape": ["n", 3, "H", "W"]},
            {"index": 1, "name": "sensors", "dtype": "float32", "shape": ["n", SENSOR_DIM]},
            {"index": 2, "name": "reward", "dtype": "float32", "shape": ["n"]},
            {"index": 3, "name": "done", "dtype": "float32", "shape": ["n"]},
            {"index": 4, "name": "success", "dtype": "float32", "shape": ["n"]},
            {"index": 5, "name": "privileged", "dtype": "float32", "shape": ["n", PRIVILEGED_DIM]},
            {"index": 6, "name": "collision", "dtype": "float32", "shape": ["n"]},
            {"index": 7, "name": "episodes", "dtype": "uint64", "shape": ["n"]},
            {"index": 8, "name": "diagnostics", "dtype": "float32", "shape": ["n", DIAGNOSTIC_DIM]},
            {"index": 9, "name": "design", "dtype": "int32", "shape": ["n"]},
            {"index": 10, "name": "goal", "dtype": "float32", "shape": [k_designs, 3, "H", "W"]},
            {"index": 11, "name": "bucket_weights", "dtype": "float32", "shape": [3]},
            {"index": 12, "name": "bucket", "dtype": "float32", "shape": ["n"]},
            {"index": 13, "name": "oob", "dtype": "float32", "shape": ["n"]},
        ],
    });
    // Flat keys consumed by learner/replay checks (inserted after the macro to
    // stay within json!'s recursion limit).
    json["style"] = style.into();
    json["split"] = SPLIT_NAMES[style as usize].into();
    json["action_dim"] = ACTION_DIM.into();
    json["design_count"] = k_designs.into();
    json["city_fingerprint"] = city.fingerprint.clone().into();
    json["city_path"] = city_path.display().to_string().into();
    json["goal_camera"] = serde_json::json!({
        "distance_m": GOAL_CAMERA_DISTANCE, "direction": GOAL_CAMERA_DIRECTION,
        "vertical_fov_rad": dynamics::CAMERA_VFOV_RAD,
        "scene": "each split-local design alone at the origin, split lighting/background, rendered once at create",
    });
    Ok(CString::new(json.to_string())?)
}
