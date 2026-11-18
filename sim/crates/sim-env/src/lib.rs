//! GPU-resident city drone visual-goal environment (city-drone-goal-v4, ABI v7).
//! See `sim_env.h` for the ABI. Build with cargo oxide and device-codegen-crate sim_physics,sim_env.
//!
//! Targets are placed objects of the imported city (buildings, vehicles, vegetation, props);
//! nothing extra is rendered. The goal is a crop of a reference camera view of the object:
//! `render_goals` renders catalog entries' reference views into the goal field so the
//! learner can encode the whole catalog once.
mod ffi;
mod kernels;
mod task;

use std::{ffi::{CString, c_char, c_void}, path::Path, sync::Arc};
use anyhow::{Result, ensure};
use cuda_core::{CudaStream, DeviceBuffer, Function, launch_kernel_on_stream};
use glam::{Mat4, Vec3};
use sim_interop::{Exchange, Gpu, SharedBuffer};
use sim_render::{Camera, RenderBindings, RenderConfig, Renderer};
use sim_scene::{CityScene, drone};
use sim_state::BodyState;
use sim_state::collision::{BvhNode, Triangle};
use sim_state::drone as dynamics;
use sim_state::drone::EpisodeSpec;
use kernels::{Buffers, Config, EnvState, ImageArgs, LabelArgs};
use task::*;

/// Fixed visibility and projection aspect (width / height).
const CAMERA_ASPECT: f32 = 4.0 / 3.0;
/// Default curriculum lanes: uniform buckets, GPS always available, full appearance randomization.
const DEFAULT_LANES: [f32; CURRICULUM_LANES] = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 0.0, 1.0];

struct TaskKernels {
    reset: Function,
    control: Function,
    finish: Function,
    observe: Function,
    goal_view: Function,
    appearance: Function,
    retarget: Function,
    coverage: Function,
    patch_labels: Function,
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
            goal_view: module.load_function(kernels::device::goal_view_CUDA_SIGNATURE.entry)?,
            appearance: module.load_function(kernels::device::appearance_CUDA_SIGNATURE.entry)?,
            retarget: module.load_function(kernels::device::retarget_CUDA_SIGNATURE.entry)?,
            coverage: module.load_function(kernels::device::coverage_CUDA_SIGNATURE.entry)?,
            patch_labels: module.load_function(kernels::device::patch_labels_CUDA_SIGNATURE.entry)?,
        })
    }

    /// Per-env kernel; `extra` is the actions pointer (control) or the first catalog index (goal_view).
    unsafe fn launch(&self, function: &Function, stream: &CudaStream,
        mut buffers: Buffers, mut config: Config, extra: Option<*mut c_void>) -> Result<()> {
        let mut args = [
            (&raw mut buffers).cast::<c_void>(),
            (&raw mut config).cast::<c_void>(),
            extra.unwrap_or(std::ptr::null_mut()),
        ];
        let count = if extra.is_some() { 3 } else { 2 };
        unsafe {
            launch_kernel_on_stream(function, (config.n.div_ceil(256), 1, 1),
                (256, 1, 1), 0, stream, &mut args[..count])?;
        }
        Ok(())
    }

    unsafe fn launch_appearance(&self, stream: &CudaStream, mut args: ImageArgs) -> Result<()> {
        let blocks = args.pixels.div_ceil(256);
        let mut pointers = [(&raw mut args).cast::<c_void>()];
        unsafe { launch_kernel_on_stream(&self.appearance, (blocks, 1, 1), (256, 1, 1), 0, stream, &mut pointers)? };
        Ok(())
    }

    unsafe fn launch_coverage(&self, stream: &CudaStream, mut buffers: Buffers,
        mut config: Config, mut image: ImageArgs) -> Result<()> {
        let mut args = [
            (&raw mut buffers).cast::<c_void>(),
            (&raw mut config).cast::<c_void>(),
            (&raw mut image).cast::<c_void>(),
        ];
        unsafe {
            launch_kernel_on_stream(&self.coverage, (config.n.div_ceil(256), 1, 1),
                (256, 1, 1), 0, stream, &mut args)?;
        }
        Ok(())
    }

    unsafe fn launch_labels(&self, stream: &CudaStream, mut buffers: Buffers, mut config: Config,
        mut image: ImageArgs, mut labels: LabelArgs, patches: u32) -> Result<()> {
        let mut args = [
            (&raw mut buffers).cast::<c_void>(),
            (&raw mut config).cast::<c_void>(),
            (&raw mut image).cast::<c_void>(),
            (&raw mut labels).cast::<c_void>(),
        ];
        unsafe {
            launch_kernel_on_stream(&self.patch_labels, ((config.n * patches).div_ceil(128), 1, 1),
                (128, 1, 1), 0, stream, &mut args)?;
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
    kernels: TaskKernels,
    state: Arc<SharedBuffer>,
    cameras: Arc<SharedBuffer>,
    rendered: Arc<SharedBuffer>,
    states: DeviceBuffer<u8>,
    nodes: DeviceBuffer<u8>,
    triangles: DeviceBuffer<u8>,
    catalog: DeviceBuffer<u8>,
    exclusions: DeviceBuffer<u8>,
    materials: DeviceBuffer<u8>,
    keys: DeviceBuffer<u8>,
    coverage_seen: DeviceBuffer<u8>,
    label_boxes: DeviceBuffer<u8>,
    label_cell_start: DeviceBuffer<u8>,
    label_cell_objects: DeviceBuffer<u8>,
    label_cells: u32,
    outputs: [DeviceBuffer<u8>; FIELD_COUNT],
    config: Config,
    width: u32,
    height: u32,
    mesh_count: u32,
    frame: u32,
    metadata: CString,
    gpu: Arc<Gpu>,
}

impl Env {
    pub fn new(n: u32, seed: u64, width: u32, height: u32, style: u32, control_mode: u32,
        city_path: &Path) -> Result<Self> {
        Self::new_with_start_distance(n, seed, width, height, style, control_mode, city_path, [8.0, 30.0])
    }

    pub fn new_with_start_distance(n: u32, seed: u64, width: u32, height: u32, style: u32,
        control_mode: u32, city_path: &Path, start_distance: [f32; 2]) -> Result<Self> {
        ensure!(n > 0, "environment count must be positive");
        ensure!(style <= 2, "style must be 0 (training), 1 (validation), or 2 (final test)");
        ensure!(control_mode == CONTROL_VELOCITY || control_mode == CONTROL_RATES,
            "control_mode must be 0 (velocity) or 1 (rates), got {control_mode}");
        ensure!(u64::from(width) * 3 == u64::from(height) * 4,
            "image must be 4:3 (width * 3 == height * 4), got {width}x{height}");
        ensure!(width > 0 && u64::from(width) * u64::from(height) * u64::from(n) <= u64::from(u32::MAX),
            "image batch exceeds the u32 pixel index");
        ensure!(width % PATCH == 0 && height % PATCH == 0,
            "image dimensions must be multiples of the {PATCH}-pixel label patch, got {width}x{height}");
        let city_path = city_path.canonicalize()?;
        let mut city = CityScene::load(&city_path)?;
        ensure!(!city.triangles.is_empty() && !city.nodes.is_empty(), "city collision geometry is empty");
        ensure!(city.nodes.len() <= u32::MAX as usize && city.triangles.len() <= u32::MAX as usize,
            "city collision index counts exceed u32");

        let catalog = drone::build_catalog_with_start_distance(&city, style, CATALOG_SIZE, seed, start_distance)?;
        ensure!(catalog.episodes.len() == CATALOG_SIZE, "catalog size mismatch");
        let district = drone::district(style, city.half_extent);

        let gpu = Gpu::new(0)?;
        let kernels = TaskKernels::load(&gpu)?;
        let device = gpu.device();
        let render_config = RenderConfig { width, height, ..Default::default() };
        let state = SharedBuffer::new(gpu.clone(), u64::from(n) * size_of::<BodyState>() as u64,
            wgpu::BufferUsages::STORAGE, "city drone bodies")?;
        let cameras = SharedBuffer::new(gpu.clone(), u64::from(n) * CAMERA_FLOATS as u64 * 4,
            wgpu::BufferUsages::STORAGE, "city drone cameras")?;

        // Internal cameras are never bound: the CUDA-written camera buffer supplies every view.
        let internal_cameras: Vec<Camera> = (0..n).map(|_| Camera {
            eye: Vec3::ZERO,
            target: Vec3::NEG_Z,
            up: Vec3::Y,
            vertical_fov_radians: dynamics::CAMERA_VFOV_RAD,
            near: dynamics::CAMERA_NEAR,
            far: dynamics::CAMERA_FAR,
        }).collect();
        let meshes = std::mem::take(&mut city.meshes);
        let instances = std::mem::take(&mut city.instances);
        let mesh_count = meshes.len() as u32;
        // SAFETY: the queue is used only for startup-owned texture uploads; it is not
        // retained, no exchange buffers exist yet, and Gpu is not reentered.
        let renderer = unsafe { gpu.with_queue(|queue| Renderer::new(
            device, queue, render_config, &meshes, &instances, &internal_cameras))? };
        let rendered = SharedBuffer::new(gpu.clone(), renderer.output_layout().total_bytes,
            wgpu::BufferUsages::STORAGE, "city render output")?;
        let bindings = renderer.bind(device, state.buffer(), rendered.buffer(), Some(cameras.buffer()))?;

        // Startup-only uploads of static geometry, materials and the episode catalog.
        let stream = gpu.cuda_stream();
        let nodes = DeviceBuffer::from_host(stream, bytemuck::cast_slice(&city.nodes))?;
        let triangles = DeviceBuffer::from_host(stream, bytemuck::cast_slice(&city.triangles))?;
        let catalog_device = DeviceBuffer::from_host(stream, bytemuck::cast_slice(&catalog.episodes))?;
        let materials = DeviceBuffer::from_host(stream, bytemuck::cast_slice(&city.mesh_materials))?;
        // Exclusions are read only up to exclusion_count; the single zero rectangle is never read when empty.
        let mut exclusion_rects: Vec<[f32; 4]> = district.exclusions.iter()
            .map(|&(lo, hi)| [lo[0], lo[1], hi[0], hi[1]]).collect();
        let exclusion_count = exclusion_rects.len() as u32;
        if exclusion_rects.is_empty() {
            exclusion_rects.push([0.0; 4]);
        }
        let exclusions = DeviceBuffer::from_host(stream, bytemuck::cast_slice(&exclusion_rects))?;
        let lanes = DeviceBuffer::from_host(stream, bytemuck::cast_slice(&DEFAULT_LANES))?;
        let (boxes, cell_start, cell_objects, label_cells) = label_grid(&city.objects, city.half_extent);
        let label_boxes = DeviceBuffer::from_host(stream, bytemuck::cast_slice(&boxes))?;
        let label_cell_start = DeviceBuffer::from_host(stream, bytemuck::cast_slice(&cell_start))?;
        let label_cell_objects = DeviceBuffer::from_host(stream, bytemuck::cast_slice(&cell_objects))?;

        let count = n as usize;
        let allocate = |bytes: usize| -> Result<DeviceBuffer<u8>> {
            Ok(unsafe { DeviceBuffer::<u8>::uninitialized_async(stream, bytes)? })
        };
        let rgb_bytes = count * 3 * height as usize * width as usize * 4;
        let states = allocate(count * size_of::<EnvState>())?;
        let keys = allocate(count * 4)?;
        let coverage_seen = allocate(count * 512 * 4)?;
        let patches = (width / PATCH * (height / PATCH)) as usize;
        let outputs: [DeviceBuffer<u8>; FIELD_COUNT] = [
            allocate(rgb_bytes)?,
            allocate(count * SENSOR_DIM * 4)?,
            allocate(count * 4)?,
            allocate(count * 4)?,
            allocate(count * 4)?,
            allocate(count * PRIVILEGED_DIM * 4)?,
            allocate(count * 4)?,
            allocate(count * 8)?,
            allocate(count * DIAGNOSTIC_DIM * 4)?,
            allocate(count * 4)?,
            allocate(rgb_bytes)?,
            lanes,
            allocate(count * 4)?,
            allocate(count * 4)?,
            allocate(count * 4)?,
            allocate(count * 4 * 4)?,
            allocate(count * 3 * 4)?,
            allocate(count * patches * 4)?,
            allocate(count * patches * 4)?,
        ];

        let exchange = Exchange::new(gpu.clone(), vec![state.clone(), cameras.clone(), rendered.clone()])?;
        let projection = Mat4::perspective_rh(dynamics::CAMERA_VFOV_RAD, CAMERA_ASPECT,
            dynamics::CAMERA_NEAR, dynamics::CAMERA_FAR).to_cols_array();
        let config = Config {
            n,
            catalog_size: CATALOG_SIZE as u32,
            node_count: city.nodes.len() as u32,
            seed,
            control_mode,
            bucket_first: [catalog.bucket_ranges[0][0], catalog.bucket_ranges[1][0], catalog.bucket_ranges[2][0]],
            bucket_count: [catalog.bucket_ranges[0][1], catalog.bucket_ranges[1][1], catalog.bucket_ranges[2][1]],
            exclusion_count,
            district_min: district.min,
            district_max: district.max,
            landable_height: city.landable.height,
            free_flight: 0,
            city_half_extent: city.half_extent,
            tan_half_vfov: (dynamics::CAMERA_VFOV_RAD * 0.5).tan(),
            aspect: CAMERA_ASPECT,
            projection_x: projection[0],
            projection_y: projection[5],
            projection_z: projection[10],
            projection_w: projection[14],
        };
        let metadata = task_metadata(&city, &catalog, &district, &city_path, style, control_mode)?;
        let mut env = Self {
            exchange,
            renderer,
            bindings,
            kernels,
            state,
            cameras,
            rendered,
            states,
            nodes,
            triangles,
            catalog: catalog_device,
            exclusions,
            materials,
            keys,
            coverage_seen,
            label_boxes,
            label_cell_start,
            label_cell_objects,
            label_cells,
            outputs,
            config,
            width,
            height,
            mesh_count,
            frame: 0,
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
            catalog_index: self.outputs[9].cu_deviceptr() as *mut i32,
            gps_available: self.outputs[14].cu_deviceptr() as *mut f32,
            crops: self.outputs[15].cu_deviceptr() as *mut f32,
            keys: self.keys.cu_deviceptr() as *mut u32,
            coverage_seen: self.coverage_seen.cu_deviceptr() as *mut u32,
            episode_diagnostics: self.outputs[16].cu_deviceptr() as *mut f32,
        }
    }

    /// Runs per-env kernels between Vulkan accesses: `run` receives the kernels, the CUDA
    /// stream, buffer pointers and config while the exchange owns the shared buffers.
    fn cycle(&mut self, label: &str, run: impl FnOnce(&TaskKernels, &CudaStream, Buffers, Config) -> Result<()>) -> Result<()> {
        let buffers = self.buffers();
        let config = self.config;
        let encoder = self.gpu.device().create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) });
        let (kernels, gpu) = (&self.kernels, &self.gpu);
        unsafe { self.exchange.cycle(encoder, || run(kernels, gpu.cuda_stream(), buffers, config))? };
        Ok(())
    }

    /// Restarts every environment with `seed`; episode counters restart at 0.
    pub fn reset(&mut self, seed: u64) -> Result<()> {
        self.config.seed = seed;
        self.cycle("city reset ownership", |kernels, stream, buffers, config| unsafe {
            kernels.launch(&kernels.reset, stream, buffers, config, None)
        })?;
        self.render_into(0, true)
    }

    /// Steps all environments with `actions` (n * 4 float32 on stream()).
    /// No transfer, tensor allocation, or host wait is performed.
    ///
    /// # Safety
    /// Actions must be initialized and remain live until this stream completes;
    /// callers must order any producer before the environment's CUDA stream.
    pub unsafe fn step(&mut self, actions: *const f32) -> Result<()> {
        ensure!(!actions.is_null(), "actions pointer must not be null");
        let mut actions = actions;
        self.cycle("city control ownership", |kernels, stream, buffers, config| unsafe {
            kernels.launch(&kernels.control, stream, buffers, config, Some((&raw mut actions).cast()))?;
            kernels.launch(&kernels.finish, stream, buffers, config, None)?;
            kernels.launch(&kernels.observe, stream, buffers, config, None)
        })?;
        self.render_into(0, true)
    }

    /// Renders the goal reference views of catalog entries `first .. first + n` (clamped to the
    /// last entry) into the goal field and their crop rectangles into the crop field, then
    /// restores the current observation. Startup-time API: the learner encodes the catalog once.
    pub fn render_goals(&mut self, first: u32) -> Result<()> {
        ensure!(first < self.config.catalog_size, "first catalog index {first} out of range");
        let mut first = first;
        self.cycle("city goal views", |kernels, stream, buffers, config| unsafe {
            kernels.launch(&kernels.goal_view, stream, buffers, config, Some((&raw mut first).cast()))
        })?;
        self.render_into(10, false)?;
        self.cycle("city observation restore", |kernels, stream, buffers, config| unsafe {
            kernels.launch(&kernels.observe, stream, buffers, config, None)
        })?;
        self.render_into(0, false)
    }

    /// Replaces current-episode targets: `targets` is a device pointer to n x 8 float32 rows
    /// `[centre xyz, layout object (< 0 keeps the catalog target), half xyz, layout class]` on
    /// stream(). Observations are refreshed; the next autoreset returns to catalog targets.
    ///
    /// # Safety
    /// `targets` must hold n x 8 initialized floats that stay live until the stream completes.
    pub unsafe fn retarget(&mut self, targets: *const f32) -> Result<()> {
        ensure!(!targets.is_null(), "targets pointer must not be null");
        let mut targets = targets;
        self.cycle("city retarget", |kernels, stream, buffers, config| unsafe {
            kernels.launch(&kernels.retarget, stream, buffers, config, Some((&raw mut targets).cast()))
        })?;
        self.render_into(0, true)
    }

    /// Free flight for interactive use (never training): only the city edge and a 100 m ceiling are
    /// out of bounds, success is reported once without ending the episode, and there is no timeout.
    pub fn set_free_flight(&mut self, enabled: bool) {
        self.config.free_flight = enabled as u32;
    }


    /// Renders every camera slot, then applies the randomized appearance into `field`
    /// (0: observation, 10: goal views). Observation renders also write the per-patch labels.
    fn render_into(&mut self, field: usize, track_coverage: bool) -> Result<()> {
        let mut encoder = self.gpu.device().create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("city rendered observation"),
        });
        self.renderer.render(&mut encoder, &self.bindings);
        self.frame = self.frame.wrapping_add(1);
        let observation = field == 0;
        let args = ImageArgs {
            rendered: self.rendered.device_ptr() as *const u32,
            output: self.outputs[field].cu_deviceptr() as *mut f32,
            keys: self.keys.cu_deviceptr() as *const u32,
            materials: self.materials.cu_deviceptr() as *const u32,
            lanes: self.outputs[11].cu_deviceptr() as *const f32,
            bodies: self.state.device_ptr() as *const BodyState,
            width: self.width,
            height: self.height,
            pixels: self.width * self.height * self.config.n,
            mesh_count: self.mesh_count,
            frame: self.frame,
            moving: observation as u32,
            tan_half_vfov: self.config.tan_half_vfov,
        };
        let labels = LabelArgs {
            boxes: self.label_boxes.cu_deviceptr() as *const [f32; 8],
            cell_start: self.label_cell_start.cu_deviceptr() as *const u32,
            cell_objects: self.label_cell_objects.cu_deviceptr() as *const u32,
            cells: self.label_cells,
            patch_depth: self.outputs[17].cu_deviceptr() as *mut f32,
            patch_object: self.outputs[18].cu_deviceptr() as *mut i32,
        };
        let patches = (self.width / PATCH) * (self.height / PATCH);
        let buffers = self.buffers();
        let config = self.config;
        unsafe {
            self.exchange.cycle(encoder, || {
                self.kernels.launch_appearance(self.gpu.cuda_stream(), args)?;
                if observation {
                    self.kernels.launch_labels(self.gpu.cuda_stream(), buffers, config, args, labels, patches)?;
                }
                if track_coverage {
                    self.kernels.launch_coverage(self.gpu.cuda_stream(), buffers, config, args)?;
                }
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

fn task_metadata(city: &CityScene, catalog: &drone::Catalog, district: &drone::District, city_path: &Path,
    style: u32, control_mode: u32) -> Result<CString> {
    let exclusions: Vec<serde_json::Value> = district.exclusions.iter()
        .map(|(lo, hi)| serde_json::json!({"min": lo, "max": hi})).collect();
    const SPLIT_NAMES: [&str; 3] = ["train", "validation", "test"];
    let (control_mode_name, action_names, action_scales, action_map): (&str, Vec<&str>, Vec<f32>, &str) =
        if control_mode == CONTROL_RATES {
            ("rates", vec!["thrust", "roll_rate", "pitch_rate", "yaw_rate"],
                vec![dynamics::MAX_THRUST * 0.5, dynamics::MAX_RATES[2], dynamics::MAX_RATES[0], dynamics::MAX_RATES[1]],
                "thrust = (a0 + 1) / 2 * max_thrust_N; omega_cmd_body = [3.5*a2 (x, pitch, +: nose up), 2.0*a3 (y, yaw, +: nose left), -3.5*a1 (z, roll, +: right side down)]")
        } else {
            ("velocity", vec!["forward", "right", "up", "yaw_rate"],
                vec![dynamics::VELOCITY_MAX_HORIZONTAL, dynamics::VELOCITY_MAX_HORIZONTAL, dynamics::VELOCITY_MAX_VERTICAL, dynamics::YAW_RATE_MAX],
                "heading-frame command: horizontal = 4*(a0 forward + a1 right), |(a0,a1)| clamped to 1; up = 2*a2 m/s; yaw_rate = 1.5*a3 rad/s (+: nose left)")
        };
    let mut json = serde_json::json!({
        "abi_version": ABI_VERSION,
        "task": TASK_NAME,
        "city": {
            "path": city_path.display().to_string(),
            "fingerprint": city.fingerprint,
            "half_extent_m": city.half_extent,
            "classes": city.classes,
            "objects": city.objects.len(),
            "materials": city.material_count,
        },
        "action": {
            "dim": ACTION_DIM,
            "control_mode": control_mode_name,
            "order": action_names,
            "scale": action_scales,
            "range": [-1.0, 1.0],
            "map": action_map,
            "nan": "velocity: NaN -> 0; rates: NaN a0 -> -1, other NaN -> 0",
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
        "success": {
            "standoff_max_m": dynamics::STANDOFF_MAX, "hold_speed_mps": dynamics::HOLD_SPEED,
            "hold_steps": dynamics::HOLD_STEPS, "arrival_steps": dynamics::ARRIVAL_STEPS,
            "timing": "success is the first three-second uninterrupted hold and ends the episode; the 40 s horizon otherwise fails it",
            "facing": "anchor (target-box point nearest the camera ray) inside the frustum, sight line clear to 2 cm before the box",
        },
        "landing": {"max_speed_mps": dynamics::LANDING_SPEED, "max_tilt_rad": dynamics::LANDING_TILT_RAD},
        "ceiling_m": dynamics::CEILING,
        "reward": {
            "terms": "(max(d_prev, 3) - max(d_box, 3)) - 0.01 + 0.1*near*clamp(1 - speed/3, 0, 1) + 10*terminal_success - 10*terminal_failure; near = within 6 m of the box and facing it",
            "done": "success || crash || oob || step >= horizon; no bootstrap",
        },
        "sensors": {
            "dim": SENSOR_DIM,
            "layout": [
                "0-2 gravity direction in body frame",
                "3-5 body rates (+noise sd 0.02) / 3.5",
                "6-8 body velocity (+noise sd 0.1) / 5",
                "9 height above landable surface (+noise sd 0.03) / 10",
                "10-13 last applied action",
                "14-16 noisy target-centre offset in heading frame (x right, y up, z = -forward) / 32; 0 when GPS unavailable",
                "17 remaining horizon fraction", "18 gps_available (0/1)",
            ],
        },
        "privileged": {
            "dim": PRIVILEGED_DIM,
            "layout": [
                "0-2 target centre in body frame / 32", "3 box distance / 32", "4-6 world velocity / 5",
                "7-9 body rates / 3.5", "10-13 quaternion xyzw", "14 anchor visible (0/1)",
                "15-17 one-hot bucket", "18-19 anchor NDC x,y (0 outside frustum)", "20 anchor in frustum (0/1)",
                "21-23 target half extents / 32", "24 hold progress (steps / hold_steps)",
                "25-28 target box projection [x_min, y_min, x_max, y_max] in NDC (y up) when the anchor is visible, else 0",
            ],
        },
        "diagnostics": {
            "dim": DIAGNOSTIC_DIM,
            "layout": [
                "0-2 position", "3-5 target centre", "6-9 quaternion xyzw", "10-12 world velocity",
                "13 thrust N", "14 target class (layout index)", "15 bucket", "16 catalog index", "17 step",
                "18 landed", "19 start landed", "20 initial box distance", "21 visible fraction at start",
                "22 episode (low 24 bits)", "23 gps_available (0/1)", "24-26 target half extents",
                "27 target object (layout index)",
            ],
        },
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
            {"index": 9, "name": "catalog_index", "dtype": "int32", "shape": ["n"]},
            {"index": 10, "name": "goal_views", "dtype": "float32", "shape": ["n", 3, "H", "W"]},
            {"index": 11, "name": "curriculum", "dtype": "float32", "shape": [CURRICULUM_LANES]},
            {"index": 12, "name": "bucket", "dtype": "float32", "shape": ["n"]},
            {"index": 13, "name": "oob", "dtype": "float32", "shape": ["n"]},
            {"index": 14, "name": "gps_available", "dtype": "float32", "shape": ["n"]},
            {"index": 15, "name": "goal_crops", "dtype": "float32", "shape": ["n", 4]},
            {"index": 16, "name": "episode_diagnostics", "dtype": "float32", "shape": ["n", 3],
             "layout": ["current unique observed tiles", "ended episode unique observed tiles",
                        "ended episode consecutive hold steps"]},
        ],
    });
    // Flat keys consumed by learner/replay checks (inserted after the macro to
    // stay within json!'s recursion limit).
    json["style"] = style.into();
    json["split"] = SPLIT_NAMES[style as usize].into();
    json["action_dim"] = ACTION_DIM.into();
    json["control_mode"] = control_mode_name.into();
    json["action_names"] = action_names.into();
    json["action_scales"] = action_scales.into();
    json["catalog_size"] = CATALOG_SIZE.into();
    json["target_classes"] = drone::TARGET_CLASSES.into();
    if control_mode != CONTROL_RATES {
        json["velocity_loop"] = serde_json::json!({
            "p_gain_per_s": dynamics::VELOCITY_GAIN,
            "drag_feedforward": "a_des = p_gain * (v_cmd - v) + rotate(q, DRAG * R(q)^T v_cmd) / mass (m/s^2); cancels the body-frame linear drag (force = -DRAG * v_body) at the commanded velocity",
            "drag_n_s_per_m": dynamics::DRAG,
            "mass_kg": dynamics::MASS,
        });
    }
    json["gps"] = serde_json::json!({
        "radius_m": [GPS_MIN_RADIUS, GPS_MAX_RADIUS], "vertical_m": GPS_VERTICAL,
        "per_episode_available": "unit >= lanes[3] (p_no_gps): 1 never available, 0 always",
    });
    json["appearance"] = serde_json::json!({
        "strength": "curriculum lane 4 in [0, 1]; 0 = plain render",
        "key": "catalog index: an episode and its goal view share one appearance",
        "per_material": "YIQ hue +-0.35 rad, saturation x[0.6, 1.4], value x[0.75, 1.25]",
        "global": "exposure 2^+-0.6, white balance +-12% per channel, contrast +-20%, gamma 2^+-0.3, noise sd <= 0.03",
        "sky": "vertical gradient, blue/overcast top, warm horizon", "fog": "1 - exp(-k depth), k <= 0.02 per m, horizon colour",
        "unchanged": "sun direction and shadows",
    });
    if let Some(fields) = json["fields"].as_array_mut() {
        fields.push(serde_json::json!({"index": 17, "name": "patch_depth", "dtype": "float32", "shape": ["n", "P"],
            "layout": "training label, never actor input: mean planar depth (m) of the patch's hit samples, 0 if none"}));
        fields.push(serde_json::json!({"index": 18, "name": "patch_object", "dtype": "int32", "shape": ["n", "P"],
            "layout": "training label, never actor input: layout object index hit by most samples, -1 if none"}));
    }
    json["camera_effects"] = serde_json::json!({
        "scope": "monocular lens and sensor timing, each a per-episode uniform times appearance strength",
        "distortion": format!("radial, source r = r (1 + k r^2), r over the half-diagonal, k <= {LENS_DISTORTION}"),
        "vignette": format!("brightness x (1 - v r^2), v <= {LENS_VIGNETTE}"),
        "motion_blur": format!("{LENS_BLUR_TAPS} taps over exposure <= {LENS_EXPOSURE:.4} s of first-order image motion from body rates"),
        "rolling_shutter": format!("row time offset over readout <= {LENS_READOUT:.4} s, same image motion"),
        "goal_views": "same distortion and vignette, no motion",
        "labels": "patch labels follow the same distortion and rolling shutter; privileged target NDC and boxes stay undistorted",
    });
    json["patch_labels"] = serde_json::json!({
        "patch_px": PATCH, "samples": LABEL_SIDE * LABEL_SIDE, "min_hits": LABEL_MIN_HITS,
        "object_margin_m": LABEL_MARGIN,
        "object": "smallest layout box (grown by the margin) containing the depth hit point",
    });
    json["city_fingerprint"] = city.fingerprint.clone().into();
    json["city_path"] = city_path.display().to_string().into();
    Ok(CString::new(json.to_string())?)
}

/// Layout object boxes [centre xyz, class, half xyz, 0] and a CSR x/z grid of LABEL_CELL metres
/// over the city listing every object whose grown footprint overlaps each cell.
fn label_grid(objects: &[sim_scene::CityObject], half_extent: f32) -> (Vec<[f32; 8]>, Vec<u32>, Vec<u32>, u32) {
    let cells = (2.0 * half_extent / LABEL_CELL).ceil() as u32;
    let cell_of = |x: f32| (((x + half_extent) / LABEL_CELL).floor().max(0.0) as u32).min(cells - 1);
    let mut lists = vec![Vec::new(); (cells * cells) as usize];
    let boxes = objects.iter().enumerate().map(|(index, object)| {
        let (lo, hi) = (object.center - object.half - LABEL_MARGIN, object.center + object.half + LABEL_MARGIN);
        for z in cell_of(lo.z)..=cell_of(hi.z) {
            for x in cell_of(lo.x)..=cell_of(hi.x) {
                lists[(z * cells + x) as usize].push(index as u32);
            }
        }
        [object.center.x, object.center.y, object.center.z, object.class as f32,
         object.half.x, object.half.y, object.half.z, 0.0]
    }).collect();
    let mut start = Vec::with_capacity(lists.len() + 1);
    let mut flat = Vec::new();
    for list in &lists {
        start.push(flat.len() as u32);
        flat.extend_from_slice(list);
    }
    start.push(flat.len() as u32);
    if flat.is_empty() {
        flat.push(0);
    }
    (boxes, start, flat, cells)
}
