//! Quadrotor task constants, episode layout and the body-mounted camera definition.
//!
//! Units are SI: metres, seconds, kilograms, newtons, radians. Quaternions are xyzw and
//! must be unit length. Body frame: +x right, +y up, -z forward (camera looks along -z).
//! Thrust acts along body +y. Command sign conventions (roll, pitch, yaw) are owned by the
//! native controller; these constants are magnitudes only.

use bytemuck::{Pod, Zeroable};

pub const GRAVITY: f32 = 9.81;
pub const MASS: f32 = 1.0;
/// Principal inertias (x, y, z body axes), kg m^2.
pub const INERTIA: [f32; 3] = [0.010, 0.016, 0.010];
/// Collective thrust ceiling, newtons: 2.5 * m * g.
pub const MAX_THRUST: f32 = 2.5 * MASS * GRAVITY;
/// First-order lag of the collective thrust response, seconds.
pub const THRUST_TAU: f32 = 0.04;
/// Rate-loop time constant: torque = I * (omega_cmd - omega) / RATE_TAU, seconds.
pub const RATE_TAU: f32 = 0.05;
/// Body-rate command magnitude per body axis (x pitch, y yaw, z roll), rad/s.
pub const MAX_RATES: [f32; 3] = [3.5, 2.0, 3.5];
/// Per-axis torque saturation of the rate loop, N m. Plausible for a 1 kg quad with
/// roughly 0.12 m arms and about 6 N per motor.
pub const MAX_TORQUE: f32 = 0.8;
/// Linear body-frame drag coefficients (x, y, z), N s/m: force = -DRAG * v_body.
pub const DRAG: [f32; 3] = [0.35, 0.25, 0.35];
/// Collision sphere radius (prop guards), metres.
pub const RADIUS: f32 = 0.25;
pub const PHYSICS_HZ: u32 = 240;
pub const CONTROL_HZ: u32 = 30;
/// Physics substeps per control step.
pub const SUBSTEPS: u32 = PHYSICS_HZ / CONTROL_HZ;
/// Control steps per episode (40 s at 30 Hz): time to cross 60 m and search when the target is hidden.
pub const HORIZON: u32 = 1200;
/// Camera uptilt above the body x-axis, radians (15 deg).
pub const CAMERA_UPTILT_RAD: f32 = core::f32::consts::PI / 12.0;
/// Vertical field of view, radians (75 deg).
pub const CAMERA_VFOV_RAD: f32 = 5.0 * core::f32::consts::PI / 12.0;
/// Camera origin in body coordinates.
pub const CAMERA_OFFSET: [f32; 3] = [0.0, 0.05, -0.1];
pub const CAMERA_NEAR: f32 = 0.05;
pub const CAMERA_FAR: f32 = 120.0;
/// Sustained standoff: success is the first `HOLD_STEPS` consecutive controls within the target band,
/// slow and facing it; training episodes end there.
pub const STANDOFF_MAX: f32 = 6.0;
/// Progress reward stops below this box distance: closer brings no reward, only risk.
pub const STANDOFF_NEAR: f32 = 3.0;
pub const HOLD_SPEED: f32 = 1.0;
/// Three seconds at 30 Hz. Arrival diagnostics separately count an initial 0.5 s stop.
pub const HOLD_STEPS: u32 = 90;
pub const ARRIVAL_STEPS: u32 = 15;
/// Maximum downward speed for a landed (not crashed) contact with landable geometry, m/s.
pub const LANDING_SPEED: f32 = 2.0;
/// Maximum tilt for a landed contact, radians (30 deg).
pub const LANDING_TILT_RAD: f32 = core::f32::consts::PI / 6.0;
/// Maximum altitude above the landable plane, metres.
pub const CEILING: f32 = 40.0;
/// Visibility buckets: 0 visible, 1 partial, 2 hidden.
pub const BUCKETS: usize = 3;
/// Velocity mode (PX4-style offboard): heading-frame command at full stick, m/s.
pub const VELOCITY_MAX_HORIZONTAL: f32 = 4.0;
pub const VELOCITY_MAX_VERTICAL: f32 = 2.0;
/// Yaw-rate command at full stick, rad/s. Positive is nose left.
pub const YAW_RATE_MAX: f32 = 1.5;
/// Outer velocity P gain, 1/s: a_des = VELOCITY_GAIN * (v_cmd - v).
pub const VELOCITY_GAIN: f32 = 3.0;
/// Attitude P gain, 1/s: body rate command = ATTITUDE_GAIN * attitude error vector.
pub const ATTITUDE_GAIN: f32 = 6.0;
/// Maximum commanded tilt of the thrust vector from world up, radians (35 deg).
pub const MAX_TILT_RAD: f32 = 35.0 * core::f32::consts::PI / 180.0;
/// tan(MAX_TILT_RAD).
pub const MAX_TILT_TAN: f32 = 0.700_207_5;
/// Floor on the vertical specific force used for the tilt limit and attitude target,
/// as a fraction of g. Keeps the attitude target defined when the vertical command
/// would otherwise require downward thrust.
pub const MIN_VERTICAL_ACCEL_FRACTION: f32 = 0.1;

/// Shared CPU/device outcome logic; it never supplies or overrides a flight command.
#[derive(Clone, Copy, Debug)]
pub struct HoldOutcome {
    pub hold: u32,
    pub success: bool,
    pub done: bool,
}

#[inline(always)]
pub fn hold_outcome(previous_hold: u32, step: u32, holding: bool,
    crash: bool, oob: bool, free: bool) -> HoldOutcome {
    let hold = if holding && !crash && !oob {
        if previous_hold < HOLD_STEPS { previous_hold + 1 } else { HOLD_STEPS }
    } else { 0 };
    let deadline = step >= HORIZON;
    let success = hold >= HOLD_STEPS && previous_hold < HOLD_STEPS && !crash && !oob;
    HoldOutcome { hold, success, done: crash || oob || (!free && (success || deadline)) }
}

/// Reward per newly observed surface tile while searching (target not yet seen, no precise hint).
pub const SEARCH_TILE_REWARD: f32 = 0.02;
/// One-time bonus for first entering the standoff band while facing the target.
pub const REACH_BONUS: f32 = 3.0;
/// Per-step reward while holding (in the band, facing, at most `HOLD_SPEED`).
pub const HOLD_REWARD: f32 = 0.1;
pub const STEP_COST: f32 = 0.01;
/// Terminal success bonus and failure (crash, leaving the area, deadline) penalty.
pub const OUTCOME_REWARD: f32 = 10.0;

/// Search, reach, hold. Until the target has been seen (`guided` false, unless a precise hint
/// locates it), the policy cannot observe where it is, so box-distance progress would be noise to
/// it and newly observed surface tiles are rewarded instead; once guided, distance progress. A
/// one-time bonus on reaching the band (`reached`), a per-step bonus while `holding`, a step cost
/// and the terminal outcome.
#[inline(always)]
pub fn goal_reward(progress: f32, new_tiles: u32, guided: bool, reached: bool, holding: bool, success: bool,
    failed: bool) -> f32 {
    let search = if guided { progress } else { SEARCH_TILE_REWARD * new_tiles as f32 };
    search - STEP_COST + if reached { REACH_BONUS } else { 0.0 } + if holding { HOLD_REWARD } else { 0.0 }
        + if success { OUTCOME_REWARD } else if failed { -OUTCOME_REWARD } else { 0.0 }
}

/// One candidate episode. Written by host catalog construction; read by the native reset
/// and the goal-view renderer.
///
/// `target` / `target_half`: centre and half extents of the target object's world AABB.
/// `reference_eye` / `reference_forward`: the goal camera (roll-free, world up), whose
/// view of the object supplies the goal crop `crop = [x0, y0, x1, y1]` in normalized
/// image coordinates (top-left origin, square in normalized units so 4:3 is preserved).
/// `meta = [object index (city layout), bucket, landed (0/1), catalog index]`.
/// `stats = [visible_fraction, initial box distance, object class (layout index), reference visible fraction]`.
/// Start position is the body centre; landed starts sit at landable height + RADIUS.
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct EpisodeSpec {
    pub start_position: [f32; 4],
    pub start_rotation: [f32; 4],
    pub start_velocity: [f32; 4],
    pub start_rates: [f32; 4],
    pub target: [f32; 4],
    pub target_half: [f32; 4],
    pub reference_eye: [f32; 4],
    pub reference_forward: [f32; 4],
    pub crop: [f32; 4],
    pub meta: [u32; 4],
    pub stats: [f32; 4],
}

/// Euclidean distance from `p` to the axis-aligned box (`center`, `half`); 0 inside.
#[inline(always)]
pub fn box_distance(p: [f32; 3], center: [f32; 3], half: [f32; 3]) -> f32 {
    let mut sum = 0.0;
    let mut k = 0;
    while k < 3 {
        let offset = p[k] - center[k];
        let d = (if offset < 0.0 { -offset } else { offset }) - half[k];
        if d > 0.0 { sum += d * d; }
        k += 1;
    }
    if sum > 0.0 { sqrt(sum) } else { 0.0 }
}

/// Parameter in [0, 1] where segment `from -> to` enters the box, or a negative value if it
/// misses. 0 when `from` is inside.
#[inline(always)]
pub fn segment_box_entry(from: [f32; 3], to: [f32; 3], center: [f32; 3], half: [f32; 3]) -> f32 {
    let mut t0 = 0.0f32;
    let mut t1 = 1.0f32;
    let mut k = 0;
    while k < 3 {
        let d = to[k] - from[k];
        let lo = center[k] - half[k];
        let hi = center[k] + half[k];
        if d < 1.0e-9 && d > -1.0e-9 {
            if from[k] < lo || from[k] > hi { return -1.0; }
        } else {
            let a = (lo - from[k]) / d;
            let b = (hi - from[k]) / d;
            let (near, far) = if a < b { (a, b) } else { (b, a) };
            if near > t0 { t0 = near; }
            if far < t1 { t1 = far; }
            if t0 > t1 { return -1.0; }
        }
        k += 1;
    }
    t0
}

/// Point of the box nearest to the camera's forward ray: the part of the object the
/// camera is most directly looking at. Used for the facing test and saliency labels.
#[inline(always)]
pub fn box_anchor(eye: [f32; 3], forward: [f32; 3], center: [f32; 3], half: [f32; 3]) -> [f32; 3] {
    let rel = [center[0] - eye[0], center[1] - eye[1], center[2] - eye[2]];
    let along = rel[0] * forward[0] + rel[1] * forward[1] + rel[2] * forward[2];
    let t = if along > 0.0 { along } else { 0.0 };
    let mut anchor = [0.0; 3];
    let mut k = 0;
    while k < 3 {
        let p = eye[k] + forward[k] * t;
        let lo = center[k] - half[k];
        let hi = center[k] + half[k];
        anchor[k] = if p < lo { lo } else if p > hi { hi } else { p };
        k += 1;
    }
    anchor
}

// Newton iterations from a bit-level estimate; core has no sqrt in no_std device code.
#[inline(always)]
fn sqrt(x: f32) -> f32 {
    let mut y = f32::from_bits((x.to_bits() >> 1) + 0x1fbd_1df5);
    y = 0.5 * (y + x / y);
    y = 0.5 * (y + x / y);
    0.5 * (y + x / y)
}

// sin/cos of CAMERA_UPTILT_RAD; core has no trigonometry in no_std.
const UPTILT_SIN: f32 = 0.258_819_05;
const UPTILT_COS: f32 = 0.965_925_83;

/// Body-mounted camera definition shared by CPU catalog code and CUDA kernels.
/// Returns `(eye_offset_world, forward, up)` for a unit quaternion `rotation` (xyzw):
/// the eye is the body origin plus the rotated `CAMERA_OFFSET`; forward and up are unit
/// world-frame vectors, mutually orthogonal, with forward uptilted by `CAMERA_UPTILT_RAD`.
pub fn camera_basis(rotation: [f32; 4]) -> ([f32; 3], [f32; 3], [f32; 3]) {
    let forward = rotate(rotation, [0.0, UPTILT_SIN, -UPTILT_COS]);
    let up = rotate(rotation, [0.0, UPTILT_COS, UPTILT_SIN]);
    (rotate(rotation, CAMERA_OFFSET), forward, up)
}

// v' = v + 2w(u x v) + 2u x (u x v), u = (x, y, z), for unit quaternions.
fn rotate(q: [f32; 4], v: [f32; 3]) -> [f32; 3] {
    let u = [q[0], q[1], q[2]];
    let t = {
        let c = cross(u, v);
        [2.0 * c[0], 2.0 * c[1], 2.0 * c[2]]
    };
    let c = cross(u, t);
    [v[0] + q[3] * t[0] + c[0], v[1] + q[3] * t[1] + c[1], v[2] + q[3] * t[2] + c[2]]
}

#[inline(always)]
fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::mem::{align_of, offset_of, size_of};

    #[test]
    fn brief_arrival_does_not_end_and_drift_away_fails_at_deadline() {
        let mut hold = 0;
        for step in 1..HORIZON {
            let out = hold_outcome(hold, step, step % 60 < 30, false, false, false);
            assert!(!out.done && !out.success);
            hold = out.hold;
        }
        let out = hold_outcome(hold, HORIZON, false, false, false, false);
        assert!(out.done && !out.success);
        assert_eq!(out.hold, 0);
        assert!((goal_reward(0.0, 0, true, false, false, out.success, out.done && !out.success) + 10.01).abs() < 1e-5);
    }

    #[test]
    fn first_sustained_hold_succeeds_and_ends_and_every_failure_costs_the_same() {
        let held = hold_outcome(HOLD_STEPS - 1, 200, true, false, false, false);
        assert!(held.done && held.success);
        let too_short = hold_outcome(HOLD_STEPS - 2, HORIZON, true, false, false, false);
        assert!(too_short.done && !too_short.success);
        for (crash, oob, step) in [(true, false, 100), (false, true, 100), (false, false, HORIZON)] {
            // A crash or leaving bounds fails even mid-hold; the deadline fails without one.
            let out = hold_outcome(HOLD_STEPS - 1, step, crash || oob, crash, oob, false);
            assert!(out.done && !out.success);
            assert!((goal_reward(0.0, 0, true, false, false, false, true) + 10.01).abs() < 1e-5);
        }
    }

    #[test]
    fn search_reach_and_hold_terms() {
        // Before the target is seen, distance progress is ignored and new surface pays.
        assert!((goal_reward(5.0, 0, false, false, false, false, false) + STEP_COST).abs() < 1e-6);
        assert!((goal_reward(5.0, 10, false, false, false, false, false) - (10.0 * SEARCH_TILE_REWARD - STEP_COST)).abs() < 1e-6);
        // Once guided, progress pays and new surface does not.
        assert!((goal_reward(0.5, 10, true, false, false, false, false) - (0.5 - STEP_COST)).abs() < 1e-6);
        let base = goal_reward(0.0, 0, true, false, false, false, false);
        assert!((goal_reward(0.0, 0, true, true, false, false, false) - base - REACH_BONUS).abs() < 1e-6);
        assert!((goal_reward(0.0, 0, true, false, true, false, false) - base - HOLD_REWARD).abs() < 1e-6);
    }

    #[test]
    fn free_flight_reports_hold_once_without_stopping_the_controller() {
        let first = hold_outcome(HOLD_STEPS - 1, 200, true, false, false, true);
        assert!(first.success && !first.done);
        let next = hold_outcome(first.hold, 201, true, false, false, true);
        assert!(!next.success && !next.done);
        let deadline = hold_outcome(next.hold, HORIZON, true, false, false, true);
        assert!(!deadline.success && !deadline.done);
    }

    fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
        a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    }

    #[test]
    fn episode_layout_is_exact() {
        assert_eq!(size_of::<EpisodeSpec>(), 176);
        assert_eq!(align_of::<EpisodeSpec>(), 16);
        assert_eq!(offset_of!(EpisodeSpec, start_rotation), 16);
        assert_eq!(offset_of!(EpisodeSpec, target), 64);
        assert_eq!(offset_of!(EpisodeSpec, target_half), 80);
        assert_eq!(offset_of!(EpisodeSpec, reference_eye), 96);
        assert_eq!(offset_of!(EpisodeSpec, reference_forward), 112);
        assert_eq!(offset_of!(EpisodeSpec, crop), 128);
        assert_eq!(offset_of!(EpisodeSpec, meta), 144);
        assert_eq!(offset_of!(EpisodeSpec, stats), 160);
    }

    #[test]
    fn box_geometry_helpers() {
        let (c, h) = ([0.0, 1.0, 0.0], [1.0, 1.0, 2.0]);
        assert_eq!(box_distance([0.5, 1.5, 0.0], c, h), 0.0);
        assert!((box_distance([4.0, 1.0, 0.0], c, h) - 3.0).abs() < 1e-5);
        assert!((box_distance([4.0, 6.0, 0.0], c, h) - 5.0).abs() < 1e-5);
        assert!((segment_box_entry([-5.0, 1.0, 0.0], [5.0, 1.0, 0.0], c, h) - 0.4).abs() < 1e-6);
        assert_eq!(segment_box_entry([0.0, 1.0, 0.0], [5.0, 1.0, 0.0], c, h), 0.0);
        assert!(segment_box_entry([-5.0, 5.0, 0.0], [5.0, 5.0, 0.0], c, h) < 0.0);
        // Looking along +x past the box: the anchor is the box point nearest the ray.
        let anchor = box_anchor([-5.0, 3.0, 0.0], [1.0, 0.0, 0.0], c, h);
        assert_eq!(anchor, [0.0, 2.0, 0.0]);
        assert!((sqrt(2.0) - core::f32::consts::SQRT_2).abs() < 1e-6);
    }

    #[test]
    fn camera_basis_is_orthonormal_for_unit_rotations() {
        let rotations = [
            [0.0, 0.0, 0.0, 1.0],
            [0.6, 0.0, 0.0, 0.8],
            [0.5, 0.5, 0.5, 0.5],
            [-0.5, 0.5, -0.5, 0.5],
            [0.2, 0.4, 0.4, 0.8],
            [0.36, 0.48, 0.0, 0.8],
        ];
        for q in rotations {
            let (offset, forward, up) = camera_basis(q);
            assert!((dot(forward, forward) - 1.0).abs() < 1e-5);
            assert!((dot(up, up) - 1.0).abs() < 1e-5);
            assert!(dot(forward, up).abs() < 1e-5);
            assert!(dot(offset, offset).is_finite());
        }
    }

    #[test]
    fn identity_camera_is_uptilted_and_offset() {
        let (offset, forward, up) = camera_basis([0.0, 0.0, 0.0, 1.0]);
        assert_eq!(offset, CAMERA_OFFSET);
        assert!((forward[1] - UPTILT_SIN).abs() < 1e-6);
        assert!((forward[2] + UPTILT_COS).abs() < 1e-6);
        assert!((up[1] - UPTILT_COS).abs() < 1e-6);
        assert!((UPTILT_SIN * UPTILT_SIN + UPTILT_COS * UPTILT_COS - 1.0).abs() < 1e-6);
    }

    #[test]
    fn derived_constants_follow_contract() {
        assert_eq!(SUBSTEPS, 8);
        assert!((MAX_THRUST - 24.525).abs() < 1e-4);
    }
}
