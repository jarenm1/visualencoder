//! CUDA kernels for city-drone-goal-v1 (ABI v3).
//!
//! Each thread owns one environment `i`. It writes only drone body `i`, target
//! slots `n + i*K + k`, camera matrices `i`, and row `i` of every per-env field.

use cuda_device::cuda_module;
use sim_state::BodyState;
use sim_state::collision::{BvhNode, Triangle};
use sim_state::drone::EpisodeSpec;

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct Config {
    pub n: u32,
    /// Target designs for this split (K).
    pub designs: u32,
    pub catalog_size: u32,
    pub node_count: u32,
    pub seed: u64,
    pub bucket_first: [u32; 3],
    pub bucket_count: [u32; 3],
    pub exclusion_count: u32,
    pub district_min: [f32; 2],
    pub district_max: [f32; 2],
    pub landable_height: f32,
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
    /// Target centre (xyz), w unused.
    pub target: [f32; 4],
    /// Clamped action applied on the last step.
    pub action: [f32; 4],
    /// Clamped action applied on the step before.
    pub prev_action: [f32; 4],
    /// Per-episode GPS offset: noisy target = target + gps_offset.
    pub gps_offset: [f32; 4],
    pub initial_distance: f32,
    pub previous_distance: f32,
    pub visible_start: f32,
    pub thrust: f32,
    pub step: u32,
    pub catalog: u32,
    pub design: u32,
    pub bucket: u32,
    pub start_landed: u32,
    pub landed: u32,
    /// EVENT_* bits set by the control kernel.
    pub event: u32,
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
    pub design: *mut i32,
}

#[cuda_module]
pub(crate) mod device {
    use super::{Buffers, Config, EnvState};
    use crate::task::{
        COS_LANDING_TILT, CAMERA_FLOATS, DIAGNOSTIC_DIM, DISTRICT_MARGIN, EVENT_CRASH, EVENT_OOB,
        GPS_MIN_RADIUS, GPS_RADIUS_FRACTION, GPS_SCALE, GPS_VERTICAL, HEIGHT_SCALE, LANE_BUCKET,
        LANE_CATALOG, LANE_GPS_ANGLE, LANE_GPS_RADIUS, LANE_GPS_VERTICAL, LANE_GYRO, LANE_HEIGHT,
        LANE_VELOCITY, NOISE_GYRO, NOISE_HEIGHT, NOISE_VELOCITY, PRIVILEGED_DIM, RATE_SCALE,
        SENSOR_DIM, SETTLE_RATE, TARGET_PARK_Y, VELOCITY_SCALE, stream, unit,
    };
    use cuda_device::{float, kernel, thread};
    use sim_state::BodyState;
    use sim_state::collision::{LANDABLE, sweep_hits};
    use sim_state::drone::{
        CAMERA_FAR, CAMERA_NEAR, CEILING, DRAG, GRAVITY, HORIZON, INERTIA, LANDING_SPEED,
        MASS, MAX_RATES, MAX_THRUST, MAX_TORQUE, PHYSICS_HZ, RADIUS, RATE_TAU, SUBSTEPS,
        SUCCESS_RADIUS, SUCCESS_SPEED, THRUST_TAU, camera_basis,
    };

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

    // Target visible from the eye: inside the 4:3 frustum and unobstructed by non-landable geometry.
    #[inline(always)]
    unsafe fn visible(buffers: Buffers, config: Config, eye: [f32; 3], forward: [f32; 3],
        up: [f32; 3], target: [f32; 3]) -> bool {
        let d = sub(target, eye);
        let right = cross(forward, up);
        let z = dot(d, forward);
        let x = dot(d, right);
        let y = dot(d, up);
        if !(z > CAMERA_NEAR && z < CAMERA_FAR) {
            return false;
        }
        let half_y = z * config.tan_half_vfov;
        let half_x = half_y * config.aspect;
        if x < -half_x || x > half_x || y < -half_y || y > half_y {
            return false;
        }
        let blocked = unsafe {
            sweep_hits(buffers.nodes, config.node_count, buffers.triangles, eye, target, 0.0, LANDABLE)
        };
        !blocked
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
        let pick = unit(base, LANE_BUCKET) * sum;
        let bucket = if pick < w[0] { 0usize } else if pick < w[0] + w[1] { 1usize } else { 2usize };
        let count = config.bucket_count[bucket];
        let within = unit(base, LANE_CATALOG) * count as f32;
        let mut within = within as u32;
        if within >= count {
            within = count - 1;
        }
        let index = config.bucket_first[bucket] + within;
        let spec = unsafe { *buffers.catalog.add(index as usize) };

        let rot = spec.start_rotation;
        let position = xyz(spec.start_position);
        let velocity = xyz(spec.start_velocity);
        let rates = rotate(rot, xyz(spec.start_rates));
        let target = xyz(spec.target);
        let design = spec.meta[0];
        let start_landed = spec.meta[2];

        let d0 = spec.stats[1];
        let radius = maxf(GPS_MIN_RADIUS, unit(base, LANE_GPS_RADIUS) * GPS_RADIUS_FRACTION * d0);
        let angle = core::f32::consts::TAU * unit(base, LANE_GPS_ANGLE);
        let gps = [
            radius * float::cos_approx_f32(angle),
            (2.0 * unit(base, LANE_GPS_VERTICAL) - 1.0) * GPS_VERTICAL,
            radius * float::sin_approx_f32(angle),
        ];

        let thrust = if start_landed != 0 { 0.0 } else { MASS * GRAVITY };
        let state = EnvState {
            episode,
            target: spec.target,
            action: [0.0; 4],
            prev_action: [0.0; 4],
            gps_offset: pack(gps),
            initial_distance: d0,
            previous_distance: length(sub(target, position)),
            visible_start: spec.stats[0],
            thrust,
            step: 0,
            catalog: index,
            design,
            bucket: bucket as u32,
            start_landed,
            landed: start_landed,
            event: 0,
        };
        let body = BodyState {
            position: pack(position),
            velocity: pack(velocity),
            rotation: rot,
            angular_velocity: pack(rates),
        };
        let slot0 = config.n as usize + i * config.designs as usize;
        let mut k = 0u32;
        while k < config.designs {
            let slot = if k == design {
                BodyState { position: pack(target), velocity: [0.0; 4], rotation: [0.0, 0.0, 0.0, 1.0],
                    angular_velocity: [0.0; 4] }
            } else {
                BodyState { position: [0.0, TARGET_PARK_Y, 0.0, 0.0], velocity: [0.0; 4],
                    rotation: [0.0, 0.0, 0.0, 1.0], angular_velocity: [0.0; 4] }
            };
            unsafe { *buffers.bodies.add(slot0 + k as usize) = slot };
            k += 1;
        }
        unsafe {
            *buffers.bodies.add(i) = body;
            *buffers.states.add(i) = state;
        }
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
        let rel = sub(add(target, xyz(st.gps_offset)), p);
        let fwd_w = rotate(q, [0.0, 0.0, -1.0]);
        let fwd_h = [fwd_w[0], 0.0, fwd_w[2]];
        let forward = if length(fwd_h) > 1.0e-6 { normalize(fwd_h) } else { [0.0, 0.0, -1.0] };
        let right = [-forward[2], 0.0, forward[0]];
        unsafe {
            put3(s, 14, [dot(rel, right) / GPS_SCALE, rel[1] / GPS_SCALE, -dot(rel, forward) / GPS_SCALE]);
            let remaining = if st.step >= HORIZON { 0.0 } else { (HORIZON - st.step) as f32 / HORIZON as f32 };
            put(s, 17, remaining);
        }

        // Privileged (critic-only) state.
        let (eye_off, cam_forward, cam_up) = camera_basis(q);
        let eye = add(p, eye_off);
        let seen = unsafe { visible(buffers, config, eye, cam_forward, cam_up, target) };
        let rel_true = sub(target, p);
        let rel_body = rotate_inverse(q, rel_true);
        let pv = unsafe { buffers.privileged.add(i * PRIVILEGED_DIM) };
        unsafe {
            put3(pv, 0, scale(rel_body, 1.0 / GPS_SCALE));
            put(pv, 3, length(rel_true) / GPS_SCALE);
            put3(pv, 4, scale(v, 1.0 / VELOCITY_SCALE));
            put3(pv, 7, scale(w_b, 1.0 / RATE_SCALE));
            put4(pv, 10, q);
            put(pv, 14, if seen { 1.0 } else { 0.0 });
            put(pv, 15, if st.bucket == 0 { 1.0 } else { 0.0 });
            put(pv, 16, if st.bucket == 1 { 1.0 } else { 0.0 });
            put(pv, 17, if st.bucket == 2 { 1.0 } else { 0.0 });
        }

        // Diagnostics.
        let dg = unsafe { buffers.diagnostics.add(i * DIAGNOSTIC_DIM) };
        unsafe {
            put3(dg, 0, p);
            put3(dg, 3, target);
            put4(dg, 6, q);
            put3(dg, 10, v);
            put(dg, 13, st.thrust);
            put(dg, 14, st.design as f32);
            put(dg, 15, st.bucket as f32);
            put(dg, 16, st.catalog as f32);
            put(dg, 17, st.step as f32);
            put(dg, 18, st.landed as f32);
            put(dg, 19, st.start_landed as f32);
            put(dg, 20, st.initial_distance);
            put(dg, 21, st.visible_start);
            put(dg, 22, (st.episode & 0xFF_FFFF) as f32);
            put(dg, 23, 0.0);
            *buffers.design.add(i) = st.design as i32;
            *buffers.episodes.add(i) = st.episode;
        }

        // Camera: rigid view and projection, written for the renderer to read.
        let right_cam = cross(cam_forward, cam_up);
        unsafe {
            write_camera(buffers.cameras.add(i * CAMERA_FLOATS), config, eye, right_cam, cam_up, cam_forward);
        }
    }

    // Advances env `i` by one control step (SUBSTEPS physics substeps), applying `action`.
    #[inline(always)]
    unsafe fn control_env(buffers: Buffers, config: Config, i: usize, actions: *const f32) {
        let mut st = unsafe { *buffers.states.add(i) };
        let mut body = unsafe { *buffers.bodies.add(i) };

        let mut a = [0.0f32; 4];
        let mut k = 0usize;
        while k < 4 {
            let raw = unsafe { *actions.add(i * 4 + k) };
            a[k] = if raw != raw {
                if k == 0 { -1.0 } else { 0.0 }
            } else {
                clampf(raw, -1.0, 1.0)
            };
            k += 1;
        }
        st.prev_action = st.action;
        st.action = a;

        // Body-frame rate commands: x = pitch (a2), y = yaw (a3), z = roll (a1, sign flipped).
        let omega_cmd = [MAX_RATES[0] * a[2], MAX_RATES[1] * a[3], -MAX_RATES[2] * a[1]];
        let thrust_cmd = (a[0] + 1.0) * 0.5 * MAX_THRUST;
        let dt = 1.0 / PHYSICS_HZ as f32;

        let mut event = 0u32;
        let mut landed = 0u32;
        let mut s = 0u32;
        while s < SUBSTEPS && event == 0 {
            st.thrust += (thrust_cmd - st.thrust) * clampf(dt / THRUST_TAU, 0.0, 1.0);

            let q = body.rotation;
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

            if event == 0 && (!unsafe { inside_district(buffers, config, p_next[0], p_next[2]) }
                || p_next[1] > CEILING) {
                event = EVENT_OOB;
            }

            body.position = pack(p_next);
            body.velocity = pack(v_next);
            body.rotation = q_next;
            body.angular_velocity = pack(w_final);
            landed = contact;
            s += 1;
        }

        st.landed = landed;
        st.event = event;
        st.step += 1;
        unsafe {
            *buffers.bodies.add(i) = body;
            *buffers.states.add(i) = st;
        }
    }

    // Reward, termination flags, and autoreset for env `i` after a control step.
    #[inline(always)]
    unsafe fn finish_env(buffers: Buffers, config: Config, i: usize) {
        let mut st = unsafe { *buffers.states.add(i) };
        let body = unsafe { *buffers.bodies.add(i) };
        let p = xyz(body.position);
        let v = xyz(body.velocity);
        let w_b = rotate_inverse(body.rotation, xyz(body.angular_velocity));
        let d = length(sub(xyz(st.target), p));
        let speed = length(v);

        let crash = st.event & EVENT_CRASH != 0;
        let oob = st.event & EVENT_OOB != 0;
        let success = !crash && !oob && d <= SUCCESS_RADIUS && speed <= SUCCESS_SPEED;
        let done = crash || oob || success || st.step >= HORIZON;

        let mut da2 = 0.0f32;
        let mut k = 0usize;
        while k < 4 {
            let diff = st.action[k] - st.prev_action[k];
            da2 += diff * diff;
            k += 1;
        }
        let rate_sq = dot(w_b, w_b);
        let mut reward = (st.previous_distance - d) - 0.01 - 0.02 * da2 / 4.0
            - 0.005 * rate_sq / (MAX_RATES[0] * MAX_RATES[0]);
        if success {
            reward += 10.0;
        }
        if crash {
            reward -= 10.0;
        }
        if oob {
            reward -= 5.0;
        }

        unsafe {
            *buffers.rewards.add(i) = reward;
            *buffers.done.add(i) = if done { 1.0 } else { 0.0 };
            *buffers.success.add(i) = if success { 1.0 } else { 0.0 };
            *buffers.collision.add(i) = if crash { 1.0 } else { 0.0 };
            *buffers.oob.add(i) = if oob { 1.0 } else { 0.0 };
            *buffers.bucket.add(i) = if done { st.bucket as f32 } else { -1.0 };
        }
        st.previous_distance = d;
        unsafe { *buffers.states.add(i) = st };

        if done {
            unsafe { reset_env(buffers, config, i, st.episode + 1) };
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
