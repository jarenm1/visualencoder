//! CUDA kernels for city-drone-goal-v4 (ABI v7).
//!
//! Each env thread owns one environment `i`. It writes only drone body `i`, camera
//! matrices `i`, and row `i` of every per-env field. The appearance kernel owns one
//! output pixel per thread; the label kernel owns one (env, patch) per thread.

use cuda_device::cuda_module;
use sim_state::BodyState;
use sim_state::collision::{BvhNode, Triangle};
use sim_state::drone::EpisodeSpec;

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct Config {
    pub n: u32,
    pub catalog_size: u32,
    pub node_count: u32,
    pub seed: u64,
    /// 0 velocity (PX4-style offboard), 1 rates. See task::CONTROL_*.
    pub control_mode: u32,
    pub bucket_first: [u32; 3],
    pub bucket_count: [u32; 3],
    pub exclusion_count: u32,
    pub district_min: [f32; 2],
    pub district_max: [f32; 2],
    pub landable_height: f32,
    /// Free flight (interactive use, never training): only the city edge and FREE_CEILING are out
    /// of bounds, success is reported once without ending the episode, and there is no timeout.
    pub free_flight: u32,
    pub city_half_extent: f32,
    pub tan_half_vfov: f32,
    /// Fixed 4:3 visibility/projection aspect.
    pub aspect: f32,
    /// Perspective terms P[0], P[5], P[10], P[14] for near/far/vfov.
    pub projection_x: f32,
    pub projection_y: f32,
    pub projection_z: f32,
    pub projection_w: f32,
}

/// Per-environment episode state. Lives in device memory only.
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct EnvState {
    pub episode: u64,
    /// Target box centre (xyz, w = 0).
    pub target: [f32; 4],
    /// Target box half extents (xyz, w = 0).
    pub target_half: [f32; 4],
    /// Clamped action applied on the last step.
    pub action: [f32; 4],
    /// Per-episode hint error: hinted target = target + hint_offset.
    pub hint_offset: [f32; 4],
    /// Yaw-only setpoint quaternion (xyzw, rotation about world +y), velocity mode only.
    pub yaw_setpoint: [f32; 4],
    /// Box distance at the start.
    pub initial_distance: f32,
    /// max(box distance, STANDOFF_NEAR) after the previous step (progress reward state).
    pub previous_distance: f32,
    pub visible_start: f32,
    pub thrust: f32,
    pub step: u32,
    pub catalog: u32,
    /// City layout object index and class of the target.
    pub object: u32,
    pub class: u32,
    pub bucket: u32,
    pub start_landed: u32,
    pub landed: u32,
    /// Consecutive control steps inside the standoff hold condition.
    pub hold: u32,
    /// EVENT_* bits set by the control kernel.
    pub event: u32,
    /// Per-episode hint mode (task::HINT_*) and claimed radius (metres), sampled at reset.
    pub hint: u32,
    pub hint_radius: f32,
    pub observed_tiles: u32,
    /// observed_tiles already rewarded (search reward pays only new tiles).
    pub rewarded_tiles: u32,
    /// The target's anchor has been in view this episode (search -> approach).
    pub seen: u32,
    /// The reach bonus has been paid this episode.
    pub reached: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct Buffers {
    pub bodies: *mut BodyState,
    pub states: *mut EnvState,
    pub nodes: *const BvhNode,
    pub triangles: *const Triangle,
    pub catalog: *const EpisodeSpec,
    /// Train-split exclusion rectangles [lo.x, lo.z, hi.x, hi.z].
    pub exclusions: *const [f32; 4],
    pub weights: *const f32,
    pub cameras: *mut f32,
    pub sensors: *mut f32,
    pub privileged: *mut f32,
    pub diagnostics: *mut f32,
    pub rewards: *mut f32,
    pub done: *mut f32,
    pub success: *mut f32,
    pub collision: *mut f32,
    pub oob: *mut f32,
    pub bucket: *mut f32,
    pub episodes: *mut u64,
    /// Current catalog index per env (the goal-table row).
    pub catalog_index: *mut i32,
    /// Hint mode of each episode that ended this step (-1 otherwise).
    pub ended_hint: *mut f32,
    /// Goal crop rectangles written by `goal_view` (n x [x0, y0, x1, y1]).
    pub crops: *mut f32,
    /// Writable per-env reset override: catalog index of the next episode (-1: sample), consumed.
    pub next_catalog: *mut i32,
    /// Appearance key per camera slot: the catalog index whose appearance it renders.
    pub keys: *mut u32,
    /// Per-env 128x128 observed surface-tile ledger, 512 words; never policy input.
    pub coverage_seen: *mut u32,
    /// [current unique tiles, terminal unique tiles, terminal consecutive hold steps].
    pub episode_diagnostics: *mut f32,
}

/// Arguments of the occupancy kernel: a voxel grid (x fastest, then y, then z) tested against the
/// collision geometry with a sphere of `radius` at each voxel centre.
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct OccupancyArgs {
    pub origin: [f32; 4],
    pub dims: [u32; 4],
    pub cell: f32,
    pub radius: f32,
    pub out: *mut u8,
}

/// Arguments of the per-pixel appearance kernel.
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct ImageArgs {
    /// Renderer output words: packed sRGB RGBA8, then f32 depth bits, then object IDs,
    /// each region `pixels` words, camera-major.
    pub rendered: *const u32,
    /// NCHW f32 RGB in [0, 1].
    pub output: *mut f32,
    pub keys: *const u32,
    /// Unique-material index per render object id - 1 (object id 0 is sky).
    pub materials: *const u32,
    /// Curriculum lanes; lane 4 is the appearance strength in [0, 1].
    pub lanes: *const f32,
    /// Drone bodies; camera slot `i` is env `i`'s body camera when `moving` is 1.
    pub bodies: *const BodyState,
    pub width: u32,
    pub height: u32,
    pub pixels: u32,
    pub mesh_count: u32,
    /// Render counter: decorrelates sensor noise between frames.
    pub frame: u32,
    /// 1 for body-camera observations (motion blur and rolling shutter from body rates), 0 for
    /// static goal reference views.
    pub moving: u32,
    pub tan_half_vfov: f32,
    /// Per-camera appearance constants (n entries), written by `camera_look` before the image
    /// kernels of the same frame read them.
    pub looks: *mut Look,
}

/// Per-camera monocular lens and sensor timing for one frame (all zero at appearance 0).
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct Lens {
    /// Radial distortion: source radius = r (1 + k r^2), r normalized by the half-diagonal.
    pub k: f32,
    pub vignette: f32,
    pub exposure: f32,
    pub readout: f32,
    /// Image motion from body rates: pan and tilt (pixels/s) and roll (rad/s).
    pub pan: f32,
    pub tilt: f32,
    pub roll: f32,
    pub cx: f32,
    pub cy: f32,
    pub radius: f32,
}

/// Everything about one camera's frame that is constant across its pixels: the strength, lens,
/// sky gradient and the global photometric transform. Computed once per camera per frame.
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct Look {
    pub strength: f32,
    pub lens: Lens,
    pub sky_top: [f32; 3],
    pub sky_horizon: [f32; 3],
    pub fog_density: f32,
    /// Exposure gain before vignetting.
    pub exposure_gain: f32,
    pub contrast: f32,
    pub gamma: f32,
    pub sigma: f32,
    pub balance: [f32; 3],
}

/// Per-patch training labels (never actor input): metric depth and the layout object hit.
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct LabelArgs {
    /// Layout objects: [centre x, y, z, layout class, half x, y, z, 0].
    pub boxes: *const [f32; 8],
    /// CSR over a `cells` x `cells` x/z grid of LABEL_CELL metres: cell c lists
    /// `cell_objects[cell_start[c] .. cell_start[c + 1]]`.
    pub cell_start: *const u32,
    pub cell_objects: *const u32,
    pub cells: u32,
    /// n x P mean planar depth of the patch object's samples (all hits if none; 0: no geometry).
    pub patch_depth: *mut f32,
    /// n x P layout object index hit by most samples (-1: none or too few).
    pub patch_object: *mut i32,
}

#[cuda_module]
pub(crate) mod device {
    use super::{Buffers, Config, EnvState, ImageArgs, LabelArgs, Lens, Look, OccupancyArgs};
    use crate::task::{
        APPEARANCE_SALT, CAMERA_FLOATS, COS_LANDING_TILT, CONTROL_RATES, DIAGNOSTIC_DIM, DISTRICT_MARGIN,
        EVENT_CRASH, EVENT_OOB, HEIGHT_SCALE, HINT_COARSE, HINT_COARSE_RADIUS, HINT_NONE, HINT_PRECISE,
        HINT_PRECISE_RADIUS, HINT_VERTICAL, HINT_WRONG, HINT_WRONG_ERROR, LABEL_CELL, LABEL_MARGIN,
        LABEL_MIN_HITS, LABEL_SIDE, LANE_BUCKET, LANE_CATALOG, LANE_GYRO, LANE_HEIGHT, LANE_HINT_ANGLE,
        LANE_HINT_AVAILABLE, LANE_HINT_ERROR, LANE_HINT_MODE, LANE_HINT_RADIUS, LANE_HINT_VERTICAL,
        LANE_VELOCITY, LENS_BLUR_TAPS, LENS_DISTORTION, LENS_EXPOSURE, LENS_READOUT, LENS_VIGNETTE,
        NOISE_GYRO, NOISE_HEIGHT, NOISE_VELOCITY, PATCH, POSITION_SCALE, PRIVILEGED_DIM, RATE_SCALE,
        SENSOR_DIM, SETTLE_RATE, VELOCITY_SCALE, mix, stream, unit,
    };
    use cuda_device::{float, kernel, thread};
    use sim_state::BodyState;
    use sim_state::collision::{LANDABLE, sweep_hits};
    use sim_state::drone::{
        ATTITUDE_GAIN, CAMERA_FAR, CAMERA_NEAR, CEILING, DRAG, GRAVITY, HOLD_SPEED, HOLD_STEPS, HORIZON,
        INERTIA, LANDING_SPEED, MASS, MAX_RATES, MAX_THRUST, MAX_TILT_TAN, MAX_TORQUE,
        MIN_VERTICAL_ACCEL_FRACTION, PHYSICS_HZ, RADIUS, RATE_TAU, STANDOFF_MAX, STANDOFF_NEAR, SUBSTEPS,
        THRUST_TAU, VELOCITY_GAIN, VELOCITY_MAX_HORIZONTAL, VELOCITY_MAX_VERTICAL, YAW_RATE_MAX,
        box_anchor, box_distance, camera_basis, segment_box_entry,
    };
    use sim_state::drone::{goal_reward, hold_outcome};

    /// Free-flight ceiling above the landable plane (metres): above the tallest towers.
    const FREE_CEILING: f32 = 100.0;
    /// The occlusion test stops this far before the target box (metres).
    const BOX_STANDOFF_EPS: f32 = 0.02;

    #[inline(always)]
    fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
        a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    }

    #[inline(always)]
    fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
        [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
    }

    #[inline(always)]
    fn add(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
        [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
    }

    #[inline(always)]
    fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
        [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
    }

    #[inline(always)]
    fn scale(a: [f32; 3], s: f32) -> [f32; 3] {
        [a[0] * s, a[1] * s, a[2] * s]
    }

    #[inline(always)]
    fn length(a: [f32; 3]) -> f32 {
        float::sqrt_rn_f32(dot(a, a))
    }

    #[inline(always)]
    fn normalize(a: [f32; 3]) -> [f32; 3] {
        let l = length(a);
        if l > 1.0e-9 { scale(a, 1.0 / l) } else { [0.0; 3] }
    }

    #[inline(always)]
    fn xyz(v: [f32; 4]) -> [f32; 3] {
        [v[0], v[1], v[2]]
    }

    #[inline(always)]
    fn pack(v: [f32; 3]) -> [f32; 4] {
        [v[0], v[1], v[2], 0.0]
    }

    #[inline(always)]
    fn clampf(x: f32, lo: f32, hi: f32) -> f32 {
        if x < lo { lo } else if x > hi { hi } else { x }
    }

    /// Curriculum lane `k` as a probability in [0, 1] (NaN reads as 0).
    #[inline(always)]
    unsafe fn probability(buffers: Buffers, k: usize) -> f32 {
        let raw = unsafe { *buffers.weights.add(k) };
        if raw != raw { 0.0 } else { clampf(raw, 0.0, 1.0) }
    }

    #[inline(always)]
    fn maxf(a: f32, b: f32) -> f32 {
        if a > b { a } else { b }
    }

    // v' = v + 2w(u x v) + 2u x (u x v), u = (x, y, z); unit quaternion xyzw.
    #[inline(always)]
    fn rotate(q: [f32; 4], v: [f32; 3]) -> [f32; 3] {
        let u = [q[0], q[1], q[2]];
        let t = scale(cross(u, v), 2.0);
        let c = cross(u, t);
        [v[0] + q[3] * t[0] + c[0], v[1] + q[3] * t[1] + c[1], v[2] + q[3] * t[2] + c[2]]
    }

    #[inline(always)]
    fn rotate_inverse(q: [f32; 4], v: [f32; 3]) -> [f32; 3] {
        rotate([-q[0], -q[1], -q[2], q[3]], v)
    }

    // Hamilton product, xyzw layout.
    #[inline(always)]
    fn quat_mul(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
        [
            a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
            a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
            a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
            a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
        ]
    }

    #[inline(always)]
    fn quat_normalize(q: [f32; 4]) -> [f32; 4] {
        let l = float::sqrt_rn_f32(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
        [q[0] / l, q[1] / l, q[2] / l, q[3] / l]
    }

    // Standard normal via Box-Muller on two counter-hash uniforms. Uses unit lanes 2*lane, 2*lane+1.
    #[inline(always)]
    fn gauss(base: u64, lane: u32) -> f32 {
        let u1 = unit(base, 2 * lane);
        let u2 = unit(base, 2 * lane + 1);
        let r = float::sqrt_rn_f32(-2.0 * float::lg2_approx_f32(u1) * core::f32::consts::LN_2);
        r * float::cos_approx_f32(core::f32::consts::TAU * u2)
    }

    #[inline(always)]
    unsafe fn put(ptr: *mut f32, offset: usize, value: f32) {
        unsafe { *ptr.add(offset) = value };
    }

    #[inline(always)]
    unsafe fn put3(ptr: *mut f32, offset: usize, v: [f32; 3]) {
        unsafe {
            put(ptr, offset, v[0]);
            put(ptr, offset + 1, v[1]);
            put(ptr, offset + 2, v[2]);
        }
    }

    #[inline(always)]
    unsafe fn put4(ptr: *mut f32, offset: usize, v: [f32; 4]) {
        unsafe {
            put3(ptr, offset, [v[0], v[1], v[2]]);
            put(ptr, offset + 3, v[3]);
        }
    }

    // Matches the district test used on the host catalog. Exclusions are not expanded by the margin.
    #[inline(always)]
    unsafe fn inside_district(buffers: Buffers, config: Config, x: f32, z: f32) -> bool {
        let lo = [config.district_min[0] + DISTRICT_MARGIN, config.district_min[1] + DISTRICT_MARGIN];
        let hi = [config.district_max[0] - DISTRICT_MARGIN, config.district_max[1] - DISTRICT_MARGIN];
        if !(x >= lo[0] && x <= hi[0] && z >= lo[1] && z <= hi[1]) {
            return false;
        }
        let mut k = 0u32;
        while k < config.exclusion_count {
            let r = unsafe { *buffers.exclusions.add(k as usize) };
            if x >= r[0] && x <= r[2] && z >= r[1] && z <= r[3] {
                return false;
            }
            k += 1;
        }
        true
    }

    // Writes view (column-major, rigid), view_projection and eye for one camera slot.
    // View rows are (right, up, -forward) with right = forward x up.
    #[inline(always)]
    unsafe fn write_camera(cam: *mut f32, config: Config, eye: [f32; 3], right: [f32; 3],
        up: [f32; 3], forward: [f32; 3]) {
        let row2 = scale(forward, -1.0);
        let t = [-dot(right, eye), -dot(up, eye), -dot(row2, eye)];
        let view = [
            [right[0], up[0], row2[0], 0.0],
            [right[1], up[1], row2[1], 0.0],
            [right[2], up[2], row2[2], 0.0],
            [t[0], t[1], t[2], 1.0],
        ];
        let a = config.projection_x;
        let b = config.projection_y;
        let c = config.projection_z;
        let d = config.projection_w;
        let mut j = 0usize;
        while j < 4 {
            let col = view[j];
            let proj = [a * col[0], b * col[1], c * col[2] + d * col[3], -col[2]];
            let mut r = 0usize;
            while r < 4 {
                unsafe {
                    put(cam, j * 4 + r, col[r]);
                    put(cam, 16 + j * 4 + r, proj[r]);
                }
                r += 1;
            }
            j += 1;
        }
        unsafe {
            put(cam, 32, eye[0]);
            put(cam, 33, eye[1]);
            put(cam, 34, eye[2]);
            put(cam, 35, 1.0);
        }
    }

    // Target projected into the 4:3 frustum: NDC (x right, y up) and whether it lies inside.
    // NDC is [0, 0] when outside. Occlusion is not considered here.
    #[inline(always)]
    fn project(config: Config, eye: [f32; 3], forward: [f32; 3], up: [f32; 3], target: [f32; 3])
        -> ([f32; 2], bool) {
        let d = sub(target, eye);
        let right = cross(forward, up);
        let z = dot(d, forward);
        let x = dot(d, right);
        let y = dot(d, up);
        if !(z > CAMERA_NEAR && z < CAMERA_FAR) {
            return ([0.0; 2], false);
        }
        let half_y = z * config.tan_half_vfov;
        let half_x = half_y * config.aspect;
        if x < -half_x || x > half_x || y < -half_y || y > half_y {
            return ([0.0; 2], false);
        }
        ([x / half_x, y / half_y], true)
    }

    // Starts an episode for env `i` at `episode`: samples a bucket and catalog entry and writes
    // the drone body, the K target slots, and the episode state.
    #[inline(always)]
    unsafe fn reset_env(buffers: Buffers, config: Config, i: usize, episode: u64) {
        let base = stream(config.seed, i as u32, episode, 0);
        let mut w = [0.0f32; 3];
        let mut sum = 0.0f32;
        let mut k = 0usize;
        while k < 3 {
            let v = unsafe { *buffers.weights.add(k) };
            w[k] = if v >= 0.0 { v } else { 0.0 };
            sum += w[k];
            k += 1;
        }
        if !(sum > 0.0 && sum < 3.0e38) {
            w = [1.0; 3];
            sum = 3.0;
        }
        let p_no_hint = unsafe { probability(buffers, 3) };
        let pick = unit(base, LANE_BUCKET) * sum;
        let sampled_bucket = if pick < w[0] { 0usize } else if pick < w[0] + w[1] { 1usize } else { 2usize };
        let count = config.bucket_count[sampled_bucket];
        let within = unit(base, LANE_CATALOG) * count as f32;
        let mut within = within as u32;
        if within >= count {
            within = count - 1;
        }
        // A caller-chosen catalog entry (verifier labelling, prioritized replay) replaces the
        // sampled one once; its bucket is the catalog's.
        let forced = unsafe { *buffers.next_catalog.add(i) };
        let index = if forced >= 0 && (forced as u32) < config.catalog_size { forced as u32 }
            else { config.bucket_first[sampled_bucket] + within };
        unsafe { *buffers.next_catalog.add(i) = -1 };
        let spec = unsafe { *buffers.catalog.add(index as usize) };
        let bucket = spec.meta[1] as usize;

        let rot = spec.start_rotation;
        let position = xyz(spec.start_position);
        let velocity = xyz(spec.start_velocity);
        let rates = rotate(rot, xyz(spec.start_rates));
        let target = xyz(spec.target);
        let half = xyz(spec.target_half);
        let start_landed = spec.meta[2];

        // Hint: none with probability p_no_hint; otherwise coarse / wrong with lanes 5 / 6 of the
        // hinted episodes and precise for the rest. Precise and coarse errors are uniform in the
        // claimed disc; a wrong hint is 30-80 m off while claiming a precise radius.
        let hinted = unit(base, LANE_HINT_AVAILABLE) >= p_no_hint;
        let mode_pick = unit(base, LANE_HINT_MODE);
        let p_coarse = unsafe { probability(buffers, 5) };
        let p_wrong = unsafe { probability(buffers, 6) };
        let hint = if !hinted { HINT_NONE } else if mode_pick < p_coarse { HINT_COARSE }
            else if mode_pick < p_coarse + p_wrong { HINT_WRONG } else { HINT_PRECISE };
        let range = if hint == HINT_COARSE { HINT_COARSE_RADIUS } else { HINT_PRECISE_RADIUS };
        let claimed = range[0] + (range[1] - range[0]) * unit(base, LANE_HINT_RADIUS);
        let error = if hint == HINT_WRONG {
            HINT_WRONG_ERROR[0] + (HINT_WRONG_ERROR[1] - HINT_WRONG_ERROR[0]) * unit(base, LANE_HINT_ERROR)
        } else {
            claimed * float::sqrt_rn_f32(unit(base, LANE_HINT_ERROR))
        };
        let angle = core::f32::consts::TAU * unit(base, LANE_HINT_ANGLE);
        let offset = [
            error * float::cos_approx_f32(angle),
            (2.0 * unit(base, LANE_HINT_VERTICAL) - 1.0) * HINT_VERTICAL,
            error * float::sin_approx_f32(angle),
        ];

        let thrust = if start_landed != 0 { 0.0 } else { MASS * GRAVITY };
        let state = EnvState {
            episode,
            target: pack(target),
            target_half: pack(half),
            action: [0.0; 4],
            hint_offset: pack(offset),
            yaw_setpoint: heading_quat(rot),
            initial_distance: spec.stats[1],
            previous_distance: maxf(box_distance(position, target, half), STANDOFF_NEAR),
            visible_start: spec.stats[0],
            thrust,
            step: 0,
            catalog: index,
            object: spec.meta[0],
            class: spec.stats[2] as u32,
            bucket: bucket as u32,
            start_landed,
            landed: start_landed,
            hold: 0,
            event: 0,
            hint,
            hint_radius: claimed,
            observed_tiles: 0,
            rewarded_tiles: 0,
            seen: 0,
            reached: 0,
        };
        let body = BodyState {
            position: pack(position),
            velocity: pack(velocity),
            rotation: rot,
            angular_velocity: pack(rates),
        };
        unsafe {
            let mut word = 0usize;
            while word < 512 {
                *buffers.coverage_seen.add(i * 512 + word) = 0;
                word += 1;
            }
            *buffers.bodies.add(i) = body;
            *buffers.states.add(i) = state;
        }
    }

    // Where the camera looks at the target box: the anchor (box point nearest the view ray),
    // its NDC, whether it is inside the frustum, and whether it is also unoccluded. The
    // occlusion segment stops just before the box so the object's own mesh never occludes.
    #[inline(always)]
    unsafe fn anchor_view(buffers: Buffers, config: Config, eye: [f32; 3], forward: [f32; 3], up: [f32; 3],
        center: [f32; 3], half: [f32; 3]) -> ([f32; 2], bool, bool) {
        let anchor = box_anchor(eye, forward, center, half);
        let (ndc, in_frustum) = project(config, eye, forward, up, anchor);
        if !in_frustum {
            return (ndc, false, false);
        }
        let entry = segment_box_entry(eye, anchor, center, half);
        let span = length(sub(anchor, eye));
        let stop = maxf(maxf(entry, 0.0) - BOX_STANDOFF_EPS / maxf(span, 1.0e-6), 0.0);
        let visible = stop == 0.0 || !unsafe {
            sweep_hits(buffers.nodes, config.node_count, buffers.triangles, eye,
                add(eye, scale(sub(anchor, eye), stop)), 0.0, LANDABLE)
        };
        (ndc, true, visible)
    }

    // Screen-space bounds [x_min, y_min, x_max, y_max] (NDC, y up, clipped to [-1, 1]) of the
    // target box's projection: the image region the object occupies. A corner at or behind the
    // near plane means the camera is at the box, so the whole frame counts.
    #[inline(always)]
    fn box_screen(config: Config, eye: [f32; 3], forward: [f32; 3], up: [f32; 3], center: [f32; 3],
        half: [f32; 3]) -> [f32; 4] {
        let right = cross(forward, up);
        let mut bounds = [1.0f32, 1.0, -1.0, -1.0];
        let mut corner = 0u32;
        while corner < 8 {
            let sign = |bit: u32| if corner & bit == 0 { -1.0 } else { 1.0 };
            let point = [center[0] + sign(1) * half[0], center[1] + sign(2) * half[1], center[2] + sign(4) * half[2]];
            let d = sub(point, eye);
            let z = dot(d, forward);
            if z <= CAMERA_NEAR {
                return [-1.0, -1.0, 1.0, 1.0];
            }
            let x = dot(d, right) / (z * config.tan_half_vfov * config.aspect);
            let y = dot(d, up) / (z * config.tan_half_vfov);
            bounds = [if x < bounds[0] { x } else { bounds[0] }, if y < bounds[1] { y } else { bounds[1] },
                maxf(x, bounds[2]), maxf(y, bounds[3])];
            corner += 1;
        }
        [clampf(bounds[0], -1.0, 1.0), clampf(bounds[1], -1.0, 1.0), clampf(bounds[2], -1.0, 1.0), clampf(bounds[3], -1.0, 1.0)]
    }

    // Sensors, privileged state, diagnostics, camera matrices and per-env identity fields.
    #[inline(always)]
    unsafe fn observe_env(buffers: Buffers, config: Config, i: usize) {
        let st = unsafe { *buffers.states.add(i) };
        let body = unsafe { *buffers.bodies.add(i) };
        let p = xyz(body.position);
        let v = xyz(body.velocity);
        let q = body.rotation;
        let w = xyz(body.angular_velocity);
        let target = xyz(st.target);
        let half = xyz(st.target_half);
        let base = stream(config.seed, i as u32, st.episode, st.step);
        let w_b = rotate_inverse(q, w);
        let v_b = rotate_inverse(q, v);
        let g_b = rotate_inverse(q, [0.0, -1.0, 0.0]);
        let height_above = p[1] - config.landable_height;

        // Sensors.
        let s = unsafe { buffers.sensors.add(i * SENSOR_DIM) };
        unsafe {
            put3(s, 0, g_b);
            put3(s, 3, [
                (w_b[0] + gauss(base, LANE_GYRO) * NOISE_GYRO) / RATE_SCALE,
                (w_b[1] + gauss(base, LANE_GYRO + 1) * NOISE_GYRO) / RATE_SCALE,
                (w_b[2] + gauss(base, LANE_GYRO + 2) * NOISE_GYRO) / RATE_SCALE,
            ]);
            put3(s, 6, [
                (v_b[0] + gauss(base, LANE_VELOCITY) * NOISE_VELOCITY) / VELOCITY_SCALE,
                (v_b[1] + gauss(base, LANE_VELOCITY + 1) * NOISE_VELOCITY) / VELOCITY_SCALE,
                (v_b[2] + gauss(base, LANE_VELOCITY + 2) * NOISE_VELOCITY) / VELOCITY_SCALE,
            ]);
            put(s, 9, (height_above + gauss(base, LANE_HEIGHT) * NOISE_HEIGHT) / HEIGHT_SCALE);
            put4(s, 10, st.action);
        }
        let rel = sub(add(target, xyz(st.hint_offset)), p);
        let fwd_w = rotate(q, [0.0, 0.0, -1.0]);
        let fwd_h = [fwd_w[0], 0.0, fwd_w[2]];
        let forward = if length(fwd_h) > 1.0e-6 { normalize(fwd_h) } else { [0.0, 0.0, -1.0] };
        let right = [-forward[2], 0.0, forward[0]];
        unsafe {
            let hinted = st.hint != HINT_NONE;
            let hint = [dot(rel, right) / POSITION_SCALE, rel[1] / POSITION_SCALE, -dot(rel, forward) / POSITION_SCALE];
            put3(s, 14, if hinted { hint } else { [0.0; 3] });
            let remaining = if st.step >= HORIZON { 0.0 } else { (HORIZON - st.step) as f32 / HORIZON as f32 };
            put(s, 17, remaining);
            put(s, 18, if hinted { 1.0 } else { 0.0 });
            put(s, 19, if hinted { st.hint_radius / POSITION_SCALE } else { 0.0 });
        }

        // Privileged (critic-only) state.
        let (eye_off, cam_forward, cam_up) = camera_basis(q);
        let eye = add(p, eye_off);
        let (ndc, in_frustum, seen) = unsafe { anchor_view(buffers, config, eye, cam_forward, cam_up, target, half) };
        let rel_true = sub(target, p);
        let rel_body = rotate_inverse(q, rel_true);
        let pv = unsafe { buffers.privileged.add(i * PRIVILEGED_DIM) };
        unsafe {
            put3(pv, 0, scale(rel_body, 1.0 / POSITION_SCALE));
            put(pv, 3, box_distance(p, target, half) / POSITION_SCALE);
            put3(pv, 4, scale(v, 1.0 / VELOCITY_SCALE));
            put3(pv, 7, scale(w_b, 1.0 / RATE_SCALE));
            put4(pv, 10, q);
            put(pv, 14, if seen { 1.0 } else { 0.0 });
            put(pv, 15, if st.bucket == 0 { 1.0 } else { 0.0 });
            put(pv, 16, if st.bucket == 1 { 1.0 } else { 0.0 });
            put(pv, 17, if st.bucket == 2 { 1.0 } else { 0.0 });
            put(pv, 18, ndc[0]);
            put(pv, 19, ndc[1]);
            put(pv, 20, if in_frustum { 1.0 } else { 0.0 });
            put3(pv, 21, scale(half, 1.0 / POSITION_SCALE));
            put(pv, 24, st.hold as f32 / HOLD_STEPS as f32);
            let screen = if seen { box_screen(config, eye, cam_forward, cam_up, target, half) } else { [0.0; 4] };
            put4(pv, 25, screen);
        }

        // Diagnostics.
        let dg = unsafe { buffers.diagnostics.add(i * DIAGNOSTIC_DIM) };
        unsafe {
            put3(dg, 0, p);
            put3(dg, 3, target);
            put4(dg, 6, q);
            put3(dg, 10, v);
            put(dg, 13, st.thrust);
            put(dg, 14, st.class as f32);
            put(dg, 15, st.bucket as f32);
            put(dg, 16, st.catalog as f32);
            put(dg, 17, st.step as f32);
            put(dg, 18, st.landed as f32);
            put(dg, 19, st.start_landed as f32);
            put(dg, 20, st.initial_distance);
            put(dg, 21, st.visible_start);
            put(dg, 22, (st.episode & 0xFF_FFFF) as f32);
            put(dg, 23, st.hint as f32);
            put3(dg, 24, half);
            put(dg, 27, st.object as f32);
            *buffers.catalog_index.add(i) = st.catalog as i32;
            *buffers.episodes.add(i) = st.episode;
            *buffers.keys.add(i) = st.catalog;
        }

        // Camera: rigid view and projection, written for the renderer to read.
        let right_cam = cross(cam_forward, cam_up);
        unsafe {
            write_camera(buffers.cameras.add(i * CAMERA_FLOATS), config, eye, right_cam, cam_up, cam_forward);
        }
    }

    // Camera slot `i` shows catalog entry `first + i`'s goal reference view (level camera) with
    // that entry's appearance; its crop rectangle is copied out for the learner.
    #[inline(always)]
    unsafe fn goal_view_env(buffers: Buffers, config: Config, i: usize, first: u32) {
        let index = if first + (i as u32) < config.catalog_size { first + i as u32 } else { config.catalog_size - 1 };
        let spec = unsafe { *buffers.catalog.add(index as usize) };
        let eye = xyz(spec.reference_eye);
        let forward = normalize(xyz(spec.reference_forward));
        let right = normalize(cross(forward, [0.0, 1.0, 0.0]));
        let up = cross(right, forward);
        unsafe {
            write_camera(buffers.cameras.add(i * CAMERA_FLOATS), config, eye, right, up, forward);
            put4(buffers.crops.add(i * 4), 0, spec.crop);
            *buffers.keys.add(i) = index;
        }
    }

    // Yaw-only quaternion (rotation about world +y) of the horizontal heading of rotation `q`.
    // A vertical body has no horizontal heading and maps to identity.
    #[inline(always)]
    fn heading_quat(q: [f32; 4]) -> [f32; 4] {
        let fwd = rotate(q, [0.0, 0.0, -1.0]);
        let l = float::sqrt_rn_f32(fwd[0] * fwd[0] + fwd[2] * fwd[2]);
        if l < 1.0e-6 {
            return [0.0, 0.0, 0.0, 1.0];
        }
        let y = -fwd[0] / l;
        let w = 1.0 - fwd[2] / l;
        let n = float::sqrt_rn_f32(y * y + w * w);
        if n < 1.0e-6 {
            return [0.0, 1.0, 0.0, 0.0];
        }
        [0.0, y / n, 0.0, w / n]
    }

    // Minimal-arc quaternion taking world +y to unit vector `u` (requires u[1] > -1).
    #[inline(always)]
    fn tilt_quat(u: [f32; 3]) -> [f32; 4] {
        quat_normalize([u[2], 0.0, -u[0], 1.0 + u[1]])
    }

    // Velocity-mode outer loop for one physics substep. Actions are forward, right, up and yaw
    // rate in [-1, 1]. Returns the body-frame rate command and collective thrust that track the
    // velocity setpoint, and advances the yaw setpoint. Idle (on ground, a[2] <= 0) holds level.
    #[inline(always)]
    fn velocity_command(a: [f32; 4], st: &mut EnvState, q: [f32; 4], v: [f32; 3], on_ground: bool,
        dt: f32) -> ([f32; 3], f32) {
        if on_ground {
            st.yaw_setpoint = heading_quat(q);
        }
        let idle = on_ground && a[2] <= 0.0;
        if !idle {
            let yq = quat_normalize([0.0, 0.5 * YAW_RATE_MAX * a[3] * dt, 0.0, 1.0]);
            st.yaw_setpoint = quat_normalize(quat_mul(yq, st.yaw_setpoint));
        }

        let fwd_w = rotate(q, [0.0, 0.0, -1.0]);
        let fwd_h = [fwd_w[0], 0.0, fwd_w[2]];
        let forward = if length(fwd_h) > 1.0e-6 { normalize(fwd_h) } else { [0.0, 0.0, -1.0] };
        let right = [-forward[2], 0.0, forward[0]];
        let stick = float::sqrt_rn_f32(a[0] * a[0] + a[1] * a[1]);
        let gain = if stick > 1.0 { 1.0 / stick } else { 1.0 };
        let h = scale(add(scale(forward, a[0]), scale(right, a[1])), gain * VELOCITY_MAX_HORIZONTAL);
        let v_cmd = [h[0], a[2] * VELOCITY_MAX_VERTICAL, h[2]];

        let up_body = rotate(q, [0.0, 1.0, 0.0]);
        let a_tot = if idle {
            [0.0, GRAVITY, 0.0]
        } else {
            // Drag feedforward: the dynamics apply -DRAG * v_body (rotated to world) per unit mass,
            // so cancelling that at the commanded velocity needs +DRAG * v_cmd_body / MASS in world.
            let v_cmd_b = rotate_inverse(q, v_cmd);
            let drag_ff = rotate(q, [DRAG[0] * v_cmd_b[0], DRAG[1] * v_cmd_b[1], DRAG[2] * v_cmd_b[2]]);
            let a_des = add(scale(sub(v_cmd, v), VELOCITY_GAIN), scale(drag_ff, 1.0 / MASS));
            let up_acc = maxf(a_des[1] + GRAVITY, MIN_VERTICAL_ACCEL_FRACTION * GRAVITY);
            let h_lim = MAX_TILT_TAN * up_acc;
            let h_norm = float::sqrt_rn_f32(a_des[0] * a_des[0] + a_des[2] * a_des[2]);
            let k = if h_norm > h_lim { h_lim / h_norm } else { 1.0 };
            [a_des[0] * k, up_acc, a_des[2] * k]
        };
        let thrust = if idle { 0.0 } else { clampf(MASS * dot(a_tot, up_body), 0.0, MAX_THRUST) };

        // Attitude: tilt toward the desired specific-force direction, then yaw about the setpoint.
        let q_d = quat_mul(tilt_quat(normalize(a_tot)), st.yaw_setpoint);
        let q_e = quat_mul([-q[0], -q[1], -q[2], q[3]], q_d);
        // Small-angle rotation vector of the body-frame error (2 * vector part, shortest arc).
        let sign = if q_e[3] < 0.0 { -2.0 } else { 2.0 };
        let omega = [
            clampf(ATTITUDE_GAIN * sign * q_e[0], -MAX_RATES[0], MAX_RATES[0]),
            clampf(ATTITUDE_GAIN * sign * q_e[1], -MAX_RATES[1], MAX_RATES[1]),
            clampf(ATTITUDE_GAIN * sign * q_e[2], -MAX_RATES[2], MAX_RATES[2]),
        ];
        (omega, thrust)
    }

    // Advances env `i` by one control step (SUBSTEPS physics substeps), applying `action`.
    #[inline(always)]
    unsafe fn control_env(buffers: Buffers, config: Config, i: usize, actions: *const f32) {
        let mut st = unsafe { *buffers.states.add(i) };
        let mut body = unsafe { *buffers.bodies.add(i) };

        let rates = config.control_mode == CONTROL_RATES;
        let mut a = [0.0f32; 4];
        let mut k = 0usize;
        while k < 4 {
            let raw = unsafe { *actions.add(i * 4 + k) };
            a[k] = if raw != raw {
                // NaN: rates mode treats thrust as idle; velocity mode commands zero everywhere.
                if rates && k == 0 { -1.0 } else { 0.0 }
            } else {
                clampf(raw, -1.0, 1.0)
            };
            k += 1;
        }
        st.action = a;

        // Rates mode: body-rate and thrust commands are fixed for the whole control step.
        let rate_cmd = [MAX_RATES[0] * a[2], MAX_RATES[1] * a[3], -MAX_RATES[2] * a[1]];
        let rate_thrust = (a[0] + 1.0) * 0.5 * MAX_THRUST;
        let dt = 1.0 / PHYSICS_HZ as f32;

        let mut event = 0u32;
        let mut on_ground = st.landed != 0;
        let mut s = 0u32;
        while s < SUBSTEPS && event == 0 {
            let q = body.rotation;
            let (omega_cmd, thrust_cmd) = if rates {
                (rate_cmd, rate_thrust)
            } else {
                velocity_command(a, &mut st, q, xyz(body.velocity), on_ground, dt)
            };
            st.thrust += (thrust_cmd - st.thrust) * clampf(dt / THRUST_TAU, 0.0, 1.0);
            let w_world = xyz(body.angular_velocity);
            let w_b = rotate_inverse(q, w_world);
            let iw = [INERTIA[0] * w_b[0], INERTIA[1] * w_b[1], INERTIA[2] * w_b[2]];
            let gyro = cross(w_b, iw);
            let mut alpha = [0.0f32; 3];
            let mut ax = 0usize;
            while ax < 3 {
                let ctrl = clampf(INERTIA[ax] * (omega_cmd[ax] - w_b[ax]) / RATE_TAU, -MAX_TORQUE, MAX_TORQUE);
                alpha[ax] = (ctrl - gyro[ax]) / INERTIA[ax];
                ax += 1;
            }
            let w_b_next = add(w_b, scale(alpha, dt));
            let w_next = rotate(q, w_b_next);

            let v = xyz(body.velocity);
            let v_b = rotate_inverse(q, v);
            let up_body = rotate(q, [0.0, 1.0, 0.0]);
            let drag_w = rotate(q, [-DRAG[0] * v_b[0], -DRAG[1] * v_b[1], -DRAG[2] * v_b[2]]);
            let force = add(scale(up_body, st.thrust), drag_w);
            let accel = sub(scale(force, 1.0 / MASS), [0.0, GRAVITY, 0.0]);
            let mut v_next = add(v, scale(accel, dt));
            let p_old = xyz(body.position);
            let mut p_next = add(p_old, scale(v_next, dt));

            let dq = quat_mul(pack(w_next), q);
            let h = 0.5 * dt;
            let q_next = quat_normalize([q[0] + dq[0] * h, q[1] + dq[1] * h, q[2] + dq[2] * h, q[3] + dq[3] * h]);
            let mut w_final = w_next;

            if unsafe { sweep_hits(buffers.nodes, config.node_count, buffers.triangles, p_old, p_next, RADIUS,
                LANDABLE) } {
                event = EVENT_CRASH;
            }

            let mut contact = 0u32;
            if event == 0 && p_next[1] - RADIUS <= config.landable_height {
                let up_world = rotate(q_next, [0.0, 1.0, 0.0]);
                if -v_next[1] > LANDING_SPEED || up_world[1] < COS_LANDING_TILT {
                    event = EVENT_CRASH;
                } else {
                    contact = 1;
                    p_next[1] = config.landable_height + RADIUS;
                    if v_next[1] < 0.0 {
                        v_next[1] = 0.0;
                    }
                    let damp = clampf(1.0 - SETTLE_RATE * dt, 0.0, 1.0);
                    v_next[0] *= damp;
                    v_next[2] *= damp;
                    w_final = scale(w_final, damp);
                }
            }

            let inside = if config.free_flight != 0 {
                p_next[0] > -config.city_half_extent && p_next[0] < config.city_half_extent
                    && p_next[2] > -config.city_half_extent && p_next[2] < config.city_half_extent
                    && p_next[1] <= config.landable_height + FREE_CEILING
            } else {
                let in_district = unsafe { inside_district(buffers, config, p_next[0], p_next[2]) };
                in_district && p_next[1] <= CEILING
            };
            if event == 0 && !inside {
                event = EVENT_OOB;
            }

            body.position = pack(p_next);
            body.velocity = pack(v_next);
            body.rotation = q_next;
            body.angular_velocity = pack(w_final);
            on_ground = contact != 0;
            s += 1;
        }

        st.landed = if on_ground { 1 } else { 0 };
        st.event = event;
        st.step += 1;
        unsafe {
            *buffers.bodies.add(i) = body;
            *buffers.states.add(i) = st;
        }
    }

    // Search, reach, hold (sim_state::drone::goal_reward). Success is the first sustained
    // three-second hold and ends the episode; the 40 s deadline otherwise fails it. No action override.
    #[inline(always)]
    unsafe fn finish_env(buffers: Buffers, config: Config, i: usize) {
        let mut st = unsafe { *buffers.states.add(i) };
        let body = unsafe { *buffers.bodies.add(i) };
        let p = xyz(body.position);
        let v = xyz(body.velocity);
        let center = xyz(st.target);
        let half = xyz(st.target_half);
        let d = box_distance(p, center, half);
        let speed = length(v);

        let crash = st.event & EVENT_CRASH != 0;
        let oob = st.event & EVENT_OOB != 0;
        let (eye_off, cam_forward, cam_up) = camera_basis(body.rotation);
        let (_, _, facing) = unsafe { anchor_view(buffers, config, add(p, eye_off), cam_forward, cam_up, center, half) };
        let near = !crash && !oob && d <= STANDOFF_MAX && facing;
        let holding = near && speed <= HOLD_SPEED;
        let free = config.free_flight != 0;
        let outcome = hold_outcome(st.hold, st.step, holding, crash, oob, free);
        st.hold = outcome.hold;
        let success = outcome.success;
        let done = outcome.done;
        if free && st.step >= HORIZON {
            // No timeout in free flight; the observed clock wraps within its trained range.
            st.step = 0;
        }

        // Search until the target is first in view (or a precise hint locates it), then approach;
        // the reach bonus is paid once, on first entering the facing band.
        if facing {
            st.seen = 1;
        }
        let guided = st.seen != 0 || st.hint == HINT_PRECISE;
        let reached = near && st.reached == 0;
        if reached {
            st.reached = 1;
        }
        let new_tiles = st.observed_tiles - st.rewarded_tiles;
        st.rewarded_tiles = st.observed_tiles;
        let effective = maxf(d, STANDOFF_NEAR);
        let reward = goal_reward(st.previous_distance - effective, new_tiles, guided, reached, holding, success,
            done && !success);

        unsafe {
            *buffers.episode_diagnostics.add(i * 3 + 1) = if done { st.observed_tiles as f32 } else { -1.0 };
            *buffers.episode_diagnostics.add(i * 3 + 2) = if done { st.hold as f32 } else { -1.0 };
            *buffers.rewards.add(i) = reward;
            *buffers.done.add(i) = if done { 1.0 } else { 0.0 };
            *buffers.success.add(i) = if success { 1.0 } else { 0.0 };
            *buffers.collision.add(i) = if crash { 1.0 } else { 0.0 };
            *buffers.oob.add(i) = if oob { 1.0 } else { 0.0 };
            *buffers.bucket.add(i) = if done { st.bucket as f32 } else { -1.0 };
            *buffers.ended_hint.add(i) = if done { st.hint as f32 } else { -1.0 };
        }
        st.previous_distance = effective;
        unsafe { *buffers.states.add(i) = st };

        if done {
            unsafe { reset_env(buffers, config, i, st.episode + 1) };
        }
    }

    // Lane `lane` of the appearance stream of catalog entry `key`, optionally per material.
    #[inline(always)]
    fn appearance_unit(key: u32, material: u32, lane: u32) -> f32 {
        unit(mix(APPEARANCE_SALT ^ ((key as u64) << 8) ^ ((material as u64) << 40)), lane)
    }

    #[inline(always)]
    fn signed(u: f32) -> f32 {
        2.0 * u - 1.0
    }

    #[inline(always)]
    fn power(x: f32, y: f32) -> f32 {
        if x <= 0.0 { 0.0 } else { float::ex2_approx_f32(y * float::lg2_approx_f32(x)) }
    }

    #[inline(always)]
    fn lerp3(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
        [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t]
    }

    // Hue rotation, saturation and value scaling in YIQ, on display-referred RGB.
    #[inline(always)]
    fn recolor(c: [f32; 3], hue: f32, saturation: f32, value: f32) -> [f32; 3] {
        let y = 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2];
        let i = 0.596 * c[0] - 0.274 * c[1] - 0.322 * c[2];
        let q = 0.211 * c[0] - 0.523 * c[1] + 0.312 * c[2];
        let (s, k) = (float::sin_approx_f32(hue), float::cos_approx_f32(hue));
        let i2 = (i * k - q * s) * saturation;
        let q2 = (i * s + q * k) * saturation;
        let y2 = y * value;
        [y2 + 0.956 * i2 + 0.621 * q2, y2 - 0.272 * i2 - 0.647 * q2, y2 - 1.106 * i2 + 1.703 * q2]
    }

    // Lens: see the host-side struct (shared with the per-camera look table).

    #[inline(always)]
    unsafe fn lens(args: ImageArgs, camera: usize, s: f32) -> Lens {
        let key = unsafe { *args.keys.add(camera) };
        let cx = 0.5 * args.width as f32;
        let cy = 0.5 * args.height as f32;
        let (mut pan, mut tilt, mut roll) = (0.0, 0.0, 0.0);
        if args.moving != 0 && s > 0.0 {
            let body = unsafe { *args.bodies.add(camera) };
            let (_, forward, up) = camera_basis(body.rotation);
            let w = xyz(body.angular_velocity);
            let focal = cy / args.tan_half_vfov;
            pan = dot(w, up) * focal;
            tilt = dot(w, cross(forward, up)) * focal;
            roll = dot(w, forward);
        }
        Lens {
            k: s * LENS_DISTORTION * appearance_unit(key, 0, 10),
            vignette: s * LENS_VIGNETTE * appearance_unit(key, 0, 11),
            exposure: s * LENS_EXPOSURE * appearance_unit(key, 0, 12),
            readout: s * LENS_READOUT * appearance_unit(key, 0, 13),
            pan, tilt, roll, cx, cy,
            radius: float::sqrt_rn_f32(cx * cx + cy * cy),
        }
    }

    // Render pixel (camera-local index) seen by output pixel centre (u, v) at time t after the
    // render instant: radial distortion, then the first-order image motion of a rotating camera.
    #[inline(always)]
    fn lens_source(l: Lens, width: u32, height: u32, u: f32, v: f32, t: f32) -> usize {
        let du = u - l.cx;
        let dv = v - l.cy;
        let g = 1.0 + l.k * (du * du + dv * dv) / (l.radius * l.radius);
        let flow_u = -l.pan - l.roll * dv;
        let flow_v = -l.tilt + l.roll * du;
        let su = clampf(l.cx + du * g - flow_u * t, 0.0, width as f32 - 1.0) as usize;
        let sv = clampf(l.cy + dv * g - flow_v * t, 0.0, height as f32 - 1.0) as usize;
        sv * width as usize + su
    }

    // Rolling shutter: row v is read `readout` seconds apart from top to bottom.
    #[inline(always)]
    fn row_time(l: Lens, height: u32, v: f32) -> f32 {
        (v / height as f32 - 0.5) * l.readout
    }

    #[inline(always)]
    fn unpack_rgb(word: u32) -> [f32; 3] {
        [(word & 0xff) as f32 / 255.0, ((word >> 8) & 0xff) as f32 / 255.0, ((word >> 16) & 0xff) as f32 / 255.0]
    }

    /// Per-camera constants of one frame (see `Look`), one thread per camera; the image kernels
    /// of the same frame read them instead of rehashing them at every pixel.
    #[kernel]
    pub unsafe fn camera_look(args: ImageArgs) {
        let camera = thread::index_1d().get();
        let plane = (args.width * args.height) as usize;
        if camera >= args.pixels as usize / plane {
            return;
        }
        let raw = unsafe { *args.lanes.add(4) };
        let s = if raw != raw { 0.0 } else { clampf(raw, 0.0, 1.0) };
        let key = unsafe { *args.keys.add(camera) };
        // Sky: blue to overcast grey at the top, warmer and lighter toward the horizon.
        let top = lerp3([0.35, 0.55, 0.85], [0.70, 0.72, 0.75], appearance_unit(key, 0, 0));
        let horizon = lerp3([clampf(top[0] * 1.25, 0.0, 1.0), clampf(top[1] * 1.2, 0.0, 1.0), clampf(top[2] * 1.1, 0.0, 1.0)],
            [0.95, 0.65, 0.45], 0.6 * appearance_unit(key, 0, 1));
        let look = Look {
            strength: s,
            lens: unsafe { lens(args, camera, s) },
            sky_top: top,
            sky_horizon: horizon,
            fog_density: s * appearance_unit(key, 0, 2) * 0.02,
            exposure_gain: float::ex2_approx_f32(s * signed(appearance_unit(key, 0, 3)) * 0.6),
            contrast: 1.0 + s * signed(appearance_unit(key, 0, 7)) * 0.2,
            gamma: float::ex2_approx_f32(s * signed(appearance_unit(key, 0, 8)) * 0.3),
            sigma: s * appearance_unit(key, 0, 9) * 0.03,
            balance: [1.0 + s * signed(appearance_unit(key, 0, 4)) * 0.12,
                1.0 + s * signed(appearance_unit(key, 0, 5)) * 0.12,
                1.0 + s * signed(appearance_unit(key, 0, 6)) * 0.12],
        };
        unsafe { *args.looks.add(camera) = look };
    }

    // Per-pixel randomized appearance, keyed by the catalog entry so an episode and its goal
    // view share one look: monocular lens (radial distortion, vignette, motion blur and rolling
    // shutter from body rates), per-material hue/saturation/value, sky gradient, depth fog,
    // exposure, white balance, contrast, gamma and sensor noise. Strength 0 reproduces the plain
    // render. Per-camera constants come from `camera_look`.
    #[kernel]
    pub unsafe fn appearance(args: ImageArgs) {
        let index = thread::index_1d().get();
        if index >= args.pixels as usize {
            return;
        }
        let plane = (args.width * args.height) as usize;
        let camera = index / plane;
        let pixel = index - camera * plane;
        let row = pixel / args.width as usize;
        let look = unsafe { *args.looks.add(camera) };
        let s = look.strength;
        let l = look.lens;
        let u = (pixel - row * args.width as usize) as f32 + 0.5;
        let v = row as f32 + 0.5;
        let t_row = row_time(l, args.height, v);
        let source = camera * plane + lens_source(l, args.width, args.height, u, v, t_row);
        let mut c = unpack_rgb(unsafe { *args.rendered.add(source) });
        if l.exposure > 0.0 && (l.pan != 0.0 || l.tilt != 0.0 || l.roll != 0.0) {
            c = [0.0; 3];
            let mut tap = 0u32;
            while tap < LENS_BLUR_TAPS {
                let t = t_row + l.exposure * ((tap as f32 + 0.5) / LENS_BLUR_TAPS as f32 - 0.5);
                let sample = camera * plane + lens_source(l, args.width, args.height, u, v, t);
                let rgb = unpack_rgb(unsafe { *args.rendered.add(sample) });
                c = [c[0] + rgb[0], c[1] + rgb[1], c[2] + rgb[2]];
                tap += 1;
            }
            c = scale(c, 1.0 / LENS_BLUR_TAPS as f32);
        }
        if s > 0.0 {
            let depth = f32::from_bits(unsafe { *args.rendered.add(args.pixels as usize + source) });
            let id = unsafe { *args.rendered.add(2 * args.pixels as usize + source) };
            let key = unsafe { *args.keys.add(camera) };
            if id == 0 {
                let t = row as f32 / maxf((args.height - 1) as f32, 1.0);
                c = lerp3(c, lerp3(look.sky_top, look.sky_horizon, t), s);
            } else {
                let material = if id - 1 < args.mesh_count { 1 + unsafe { *args.materials.add((id - 1) as usize) } } else { 0 };
                c = recolor(c, s * signed(appearance_unit(key, material, 0)) * 0.35,
                    1.0 + s * signed(appearance_unit(key, material, 1)) * 0.4,
                    1.0 + s * signed(appearance_unit(key, material, 2)) * 0.25);
                let fog = 1.0 - float::ex2_approx_f32(-look.fog_density * depth * core::f32::consts::LOG2_E);
                c = lerp3(c, look.sky_horizon, fog);
            }
            let du = (u - l.cx) / l.radius;
            let dv = (v - l.cy) / l.radius;
            let shade = 1.0 - l.vignette * (du * du + dv * dv);
            let exposure = look.exposure_gain * shade;
            let noise = mix(((index as u64) << 20) ^ ((args.frame as u64) << 52) ^ key as u64);
            let mut k = 0usize;
            while k < 3 {
                let v = (c[k] * look.balance[k] * exposure - 0.5) * look.contrast + 0.5;
                let grain = (unit(noise, 3 * k as u32) + unit(noise, 3 * k as u32 + 1) + unit(noise, 3 * k as u32 + 2) - 1.5) * 2.0;
                c[k] = clampf(power(clampf(v, 0.0, 1.0), look.gamma) + look.sigma * grain, 0.0, 1.0);
                k += 1;
            }
        }
        let mut k = 0usize;
        while k < 3 {
            unsafe { *args.output.add((camera * 3 + k) * plane + pixel) = c[k] };
            k += 1;
        }
    }

    // Smallest layout object whose box (grown by LABEL_MARGIN) contains `hit`, or u32::MAX.
    #[inline(always)]
    unsafe fn object_at(labels: LabelArgs, config: Config, hit: [f32; 3]) -> u32 {
        let h = config.city_half_extent;
        if !(hit[0] >= -h && hit[0] < h && hit[2] >= -h && hit[2] < h) {
            return u32::MAX;
        }
        let cx = ((hit[0] + h) / LABEL_CELL) as u32;
        let cz = ((hit[2] + h) / LABEL_CELL) as u32;
        if cx >= labels.cells || cz >= labels.cells {
            return u32::MAX;
        }
        let cell = (cz * labels.cells + cx) as usize;
        let (start, end) = unsafe { (*labels.cell_start.add(cell), *labels.cell_start.add(cell + 1)) };
        let mut best = u32::MAX;
        let mut best_volume = f32::MAX;
        let mut j = start;
        while j < end {
            let object = unsafe { *labels.cell_objects.add(j as usize) };
            let b = unsafe { *labels.boxes.add(object as usize) };
            let inside = (hit[0] - b[0]).abs() <= b[4] + LABEL_MARGIN
                && (hit[1] - b[1]).abs() <= b[5] + LABEL_MARGIN
                && (hit[2] - b[2]).abs() <= b[6] + LABEL_MARGIN;
            let volume = b[4] * b[5] * b[6];
            if inside && volume < best_volume {
                best = object;
                best_volume = volume;
            }
            j += 1;
        }
        best
    }

    /// Per-patch labels of the current observation, one thread per (env, patch): LABEL_SIDE^2
    /// samples through the same lens mapping as the image (without blur) give the layout object
    /// most samples land on and the mean planar depth of that object's samples (of all hits when
    /// the patch names no object). Training labels only.
    #[kernel]
    pub unsafe fn patch_labels(buffers: Buffers, config: Config, args: ImageArgs, labels: LabelArgs) {
        let index = thread::index_1d().get();
        let columns = (args.width / PATCH) as usize;
        let per = columns * (args.height / PATCH) as usize;
        if index >= config.n as usize * per {
            return;
        }
        let i = index / per;
        let patch = index - i * per;
        let (px, py) = ((patch % columns) as u32, (patch / columns) as u32);
        let l = unsafe { (*args.looks.add(i)).lens };
        let body = unsafe { *buffers.bodies.add(i) };
        let (off, forward, up) = camera_basis(body.rotation);
        let eye = add(xyz(body.position), off);
        let right = cross(forward, up);
        let plane = (args.width * args.height) as usize;
        let step = PATCH / LABEL_SIDE;
        let mut ids = [u32::MAX; (LABEL_SIDE * LABEL_SIDE) as usize];
        let mut depths = [0.0f32; (LABEL_SIDE * LABEL_SIDE) as usize];
        let mut depth_sum = 0.0f32;
        let mut hits = 0u32;
        let mut sample = 0u32;
        while sample < LABEL_SIDE * LABEL_SIDE {
            let u = (px * PATCH + (sample % LABEL_SIDE) * step + step / 2) as f32 + 0.5;
            let v = (py * PATCH + (sample / LABEL_SIDE) * step + step / 2) as f32 + 0.5;
            let local = lens_source(l, args.width, args.height, u, v, row_time(l, args.height, v));
            let source = i * plane + local;
            let id = unsafe { *args.rendered.add(2 * args.pixels as usize + source) };
            let depth = f32::from_bits(unsafe { *args.rendered.add(args.pixels as usize + source) });
            if id != 0 && depth > 0.0 && depth <= CAMERA_FAR {
                let su = (local % args.width as usize) as f32 + 0.5;
                let sv = (local / args.width as usize) as f32 + 0.5;
                let nx = (2.0 * su / args.width as f32 - 1.0) * config.tan_half_vfov * config.aspect;
                let ny = (1.0 - 2.0 * sv / args.height as f32) * config.tan_half_vfov;
                // Planar depth along `forward`: the unnormalized ray has unit forward component.
                let hit = add(eye, scale(add(forward, add(scale(right, nx), scale(up, ny))), depth));
                ids[sample as usize] = unsafe { object_at(labels, config, hit) };
                depths[sample as usize] = depth;
                depth_sum += depth;
                hits += 1;
            }
            sample += 1;
        }
        let mut best = u32::MAX;
        let mut best_count = 0u32;
        let mut a = 0usize;
        while a < ids.len() {
            if ids[a] != u32::MAX {
                let mut count = 0u32;
                let mut b = 0usize;
                while b < ids.len() {
                    if ids[b] == ids[a] { count += 1; }
                    b += 1;
                }
                if count > best_count {
                    best = ids[a];
                    best_count = count;
                }
            }
            a += 1;
        }
        // A patch that names an object takes the depth of that object's samples only: averaging in
        // the background behind an edge would place the object metres too far away.
        let named = best_count >= LABEL_MIN_HITS;
        let mut object_depth = 0.0f32;
        let mut c = 0usize;
        while c < ids.len() {
            if named && ids[c] == best { object_depth += depths[c]; }
            c += 1;
        }
        unsafe {
            *labels.patch_depth.add(index) = if named { object_depth / best_count as f32 }
                else if hits > 0 { depth_sum / hits as f32 } else { 0.0 };
            *labels.patch_object.add(index) = if named { best as i32 } else { -1 };
        }
    }

    #[kernel]
    pub unsafe fn goal_view(buffers: Buffers, config: Config, first: u32) {
        let i = thread::index_1d().get();
        if i >= config.n as usize {
            return;
        }
        unsafe { goal_view_env(buffers, config, i, first) };
    }

    // Replaces env i's current-episode target where `targets[i*8 + 3] >= 0`. Row layout:
    // [centre x, y, z, layout object, half x, y, z, layout class]. Success, reward and labels then
    // refer to that box until the episode ends; autoreset returns to catalog targets.
    #[kernel]
    pub unsafe fn retarget(buffers: Buffers, config: Config, targets: *const f32) {
        let i = thread::index_1d().get();
        if i >= config.n as usize {
            return;
        }
        let row = unsafe { targets.add(i * 8) };
        let object = unsafe { *row.add(3) };
        if !(object >= 0.0) {
            return;
        }
        let center = unsafe { [*row, *row.add(1), *row.add(2)] };
        let half = unsafe { [*row.add(4), *row.add(5), *row.add(6)] };
        let mut st = unsafe { *buffers.states.add(i) };
        let p = xyz(unsafe { *buffers.bodies.add(i) }.position);
        let distance = box_distance(p, center, half);
        st.target = pack(center);
        st.target_half = pack(half);
        st.object = object as u32;
        st.class = unsafe { *row.add(7) } as u32;
        st.initial_distance = distance;
        st.previous_distance = maxf(distance, STANDOFF_NEAR);
        st.hold = 0;
        // A new target is a new task from here: the episode clock restarts.
        st.step = 0;
        st.observed_tiles = 0;
        st.rewarded_tiles = 0;
        st.seen = 0;
        st.reached = 0;
        let mut word = 0usize;
        while word < 512 {
            unsafe { *buffers.coverage_seen.add(i * 512 + word) = 0; }
            word += 1;
        }
        unsafe {
            *buffers.episode_diagnostics.add(i * 3 + 1) = -1.0;
            *buffers.episode_diagnostics.add(i * 3 + 2) = -1.0;
        }
        unsafe {
            *buffers.states.add(i) = st;
            observe_env(buffers, config, i);
        }
    }

    /// Occupancy of a voxel grid for planners and the solvability verifier: 1 where a sphere of
    /// `radius` at the voxel centre touches collision geometry (the control kernel's crash test).
    #[kernel]
    pub unsafe fn occupancy(buffers: Buffers, config: Config, args: OccupancyArgs) {
        let i = thread::index_1d().get();
        let (nx, ny, nz) = (args.dims[0] as usize, args.dims[1] as usize, args.dims[2] as usize);
        if i >= nx * ny * nz { return; }
        let p = [
            args.origin[0] + ((i % nx) as f32 + 0.5) * args.cell,
            args.origin[1] + (((i / nx) % ny) as f32 + 0.5) * args.cell,
            args.origin[2] + ((i / (nx * ny)) as f32 + 0.5) * args.cell,
        ];
        let hit = unsafe { sweep_hits(buffers.nodes, config.node_count, buffers.triangles, p, p, args.radius, LANDABLE) };
        unsafe { *args.out.add(i) = hit as u8 };
    }

    /// Actual-camera surface coverage: the goal reward's search term before the first sighting,
    /// the explore command's reward, and diagnostics.
    #[kernel]
    pub unsafe fn coverage(buffers: Buffers, config: Config, args: ImageArgs) {
        let i = thread::index_1d().get();
        if i >= config.n as usize { return; }
        let mut st = unsafe { *buffers.states.add(i) };
        let body = unsafe { *buffers.bodies.add(i) };
        let (off, forward, up) = camera_basis(body.rotation);
        let eye = add(xyz(body.position), off);
        let right = cross(forward, up);
        let plane = args.width as usize * args.height as usize;
        let mut newly_seen = 0u32;
        let mut sample = 0usize;
        while sample < 48 {
            let x = ((sample % 8) * 2 + 1) * args.width as usize / 16;
            let y = ((sample / 8) * 2 + 1) * args.height as usize / 12;
            let pixel = i * plane + y * args.width as usize + x;
            let id = unsafe { *args.rendered.add(2 * args.pixels as usize + pixel) };
            let depth = f32::from_bits(unsafe { *args.rendered.add(args.pixels as usize + pixel) });
            if id != 0 && depth > 0.0 && depth <= CAMERA_FAR {
                let nx = (2.0 * (x as f32 + 0.5) / args.width as f32 - 1.0)
                    * config.tan_half_vfov * config.aspect;
                let ny = (1.0 - 2.0 * (y as f32 + 0.5) / args.height as f32) * config.tan_half_vfov;
                let ray = normalize(add(forward, add(scale(right, nx), scale(up, ny))));
                let hit = add(eye, scale(ray, depth));
                let h = config.city_half_extent;
                if hit[0] >= -h && hit[0] < h && hit[2] >= -h && hit[2] < h {
                    let tx = ((hit[0] + h) * (128.0 / (2.0 * h))) as u32;
                    let tz = ((hit[2] + h) * (128.0 / (2.0 * h))) as u32;
                    if tx < 128 && tz < 128 {
                        let tile = tz * 128 + tx;
                        let address = unsafe { buffers.coverage_seen.add(i * 512 + (tile / 32) as usize) };
                        let old = unsafe { *address };
                        let bit = 1u32 << (tile % 32);
                        if old & bit == 0 {
                            unsafe { *address = old | bit; }
                            newly_seen += 1;
                        }
                    }
                }
            }
            sample += 1;
        }
        st.observed_tiles += newly_seen;
        unsafe {
            *buffers.episode_diagnostics.add(i * 3) = st.observed_tiles as f32;
            *buffers.states.add(i) = st;
        }
    }

    #[kernel]
    pub unsafe fn reset(buffers: Buffers, config: Config) {
        let i = thread::index_1d().get();
        if i >= config.n as usize {
            return;
        }
        unsafe {
            reset_env(buffers, config, i, 0);
            *buffers.rewards.add(i) = 0.0;
            *buffers.done.add(i) = 0.0;
            *buffers.success.add(i) = 0.0;
            *buffers.collision.add(i) = 0.0;
            *buffers.oob.add(i) = 0.0;
            *buffers.bucket.add(i) = -1.0;
            *buffers.ended_hint.add(i) = -1.0;
            *buffers.episode_diagnostics.add(i * 3) = 0.0;
            *buffers.episode_diagnostics.add(i * 3 + 1) = -1.0;
            *buffers.episode_diagnostics.add(i * 3 + 2) = -1.0;
            observe_env(buffers, config, i);
        }
    }

    #[kernel]
    pub unsafe fn control(buffers: Buffers, config: Config, actions: *const f32) {
        let i = thread::index_1d().get();
        if i >= config.n as usize {
            return;
        }
        unsafe { control_env(buffers, config, i, actions) };
    }

    #[kernel]
    pub unsafe fn finish(buffers: Buffers, config: Config) {
        let i = thread::index_1d().get();
        if i >= config.n as usize {
            return;
        }
        unsafe { finish_env(buffers, config, i) };
    }

    #[kernel]
    pub unsafe fn observe(buffers: Buffers, config: Config) {
        let i = thread::index_1d().get();
        if i >= config.n as usize {
            return;
        }
        unsafe { observe_env(buffers, config, i) };
    }
}
