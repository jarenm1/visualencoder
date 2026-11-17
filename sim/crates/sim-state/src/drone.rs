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
/// Control steps per episode (15 s at 30 Hz).
pub const HORIZON: u32 = 450;
/// Camera uptilt above the body x-axis, radians (15 deg).
pub const CAMERA_UPTILT_RAD: f32 = core::f32::consts::PI / 12.0;
/// Vertical field of view, radians (75 deg).
pub const CAMERA_VFOV_RAD: f32 = 5.0 * core::f32::consts::PI / 12.0;
/// Camera origin in body coordinates.
pub const CAMERA_OFFSET: [f32; 3] = [0.0, 0.05, -0.1];
pub const CAMERA_NEAR: f32 = 0.05;
pub const CAMERA_FAR: f32 = 120.0;
pub const SUCCESS_RADIUS: f32 = 1.5;
pub const SUCCESS_SPEED: f32 = 3.0;
/// Maximum downward speed for a landed (not crashed) contact with landable geometry, m/s.
pub const LANDING_SPEED: f32 = 2.0;
/// Maximum tilt for a landed contact, radians (30 deg).
pub const LANDING_TILT_RAD: f32 = core::f32::consts::PI / 6.0;
/// Maximum altitude above the landable plane, metres.
pub const CEILING: f32 = 40.0;
/// Visibility buckets: 0 visible, 1 partial, 2 hidden.
pub const BUCKETS: usize = 3;

/// One candidate episode. Written by host catalog construction; read by the native reset.
///
/// `meta = [design (split-local), bucket, landed (0/1), catalog index]`.
/// `stats = [visible_fraction, initial_distance, 0, 0]`.
/// Start position is the body centre; landed starts sit at landable height + RADIUS.
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct EpisodeSpec {
    pub start_position: [f32; 4],
    pub start_rotation: [f32; 4],
    pub start_velocity: [f32; 4],
    pub start_rates: [f32; 4],
    pub target: [f32; 4],
    pub meta: [u32; 4],
    pub stats: [f32; 4],
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

    fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
        a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    }

    #[test]
    fn episode_layout_is_exact() {
        assert_eq!(size_of::<EpisodeSpec>(), 112);
        assert_eq!(align_of::<EpisodeSpec>(), 16);
        assert_eq!(offset_of!(EpisodeSpec, start_rotation), 16);
        assert_eq!(offset_of!(EpisodeSpec, target), 64);
        assert_eq!(offset_of!(EpisodeSpec, meta), 80);
        assert_eq!(offset_of!(EpisodeSpec, stats), 96);
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
