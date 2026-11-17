//! Constants and counter-based hashing for city-drone-goal-v2 (ABI v5).
//!
//! Plain integer and constant definitions shared by the host and the CUDA kernels.

pub const TASK_NAME: &str = "city-drone-goal-v2";
pub const ABI_VERSION: u32 = 5;
pub const ACTION_DIM: usize = 4;
pub const SENSOR_DIM: usize = 19;
pub const PRIVILEGED_DIM: usize = 29;
pub const DIAGNOSTIC_DIM: usize = 28;
pub const CATALOG_SIZE: usize = 65536;
/// Floats per camera slot: view (16), view-projection (16), eye (4).
pub const CAMERA_FLOATS: usize = 36;
pub const FIELD_COUNT: usize = 16;
/// Curriculum lanes: bucket weights (3), no-GPS probability, appearance strength.
pub const CURRICULUM_LANES: usize = 5;

/// Keep-in margin applied to the district rectangle (metres).
pub const DISTRICT_MARGIN: f32 = 2.0;
/// Salt of the per-catalog-entry appearance stream (shared by episodes and their goal views).
pub const APPEARANCE_SALT: u64 = 0x5EED_A99E_A2A9_CE01;

pub const EVENT_CRASH: u32 = 1;
pub const EVENT_OOB: u32 = 2;

/// cos(pi/6): minimum upright component of body up for a landing.
pub const COS_LANDING_TILT: f32 = 0.866_025_4;
/// Landed-body velocity damping rate (1/s) applied while in contact.
pub const SETTLE_RATE: f32 = 8.0;

/// Metre-scale GPS error: horizontal radius is uniform in [min, max] per episode.
pub const GPS_MIN_RADIUS: f32 = 3.0;
pub const GPS_MAX_RADIUS: f32 = 15.0;
/// Vertical GPS error is uniform in [-GPS_VERTICAL, GPS_VERTICAL] per episode, metres.
pub const GPS_VERTICAL: f32 = 1.5;
/// Normalisation for metre-scale relative positions.
pub const GPS_SCALE: f32 = 32.0;
/// Normalisation for body rates (rad/s).
pub const RATE_SCALE: f32 = 3.5;
/// Normalisation for velocities (m/s).
pub const VELOCITY_SCALE: f32 = 5.0;
/// Normalisation for height above the landable surface (m).
pub const HEIGHT_SCALE: f32 = 10.0;

pub const NOISE_GYRO: f32 = 0.02;
pub const NOISE_VELOCITY: f32 = 0.1;
pub const NOISE_HEIGHT: f32 = 0.03;

// Gaussian lanes consume unit lanes 2*lane and 2*lane+1. Uniform lanes start at 16.
pub const LANE_GYRO: u32 = 0;
pub const LANE_VELOCITY: u32 = 3;
pub const LANE_HEIGHT: u32 = 6;
pub const LANE_BUCKET: u32 = 16;
pub const LANE_CATALOG: u32 = 17;
pub const LANE_GPS_RADIUS: u32 = 18;
pub const LANE_GPS_ANGLE: u32 = 19;
pub const LANE_GPS_VERTICAL: u32 = 20;
pub const LANE_GPS_AVAILABLE: u32 = 21;

/// Control modes, as passed to `sim_env_create`.
pub const CONTROL_VELOCITY: u32 = 0;
pub const CONTROL_RATES: u32 = 1;

const GOLDEN: u64 = 0x9e37_79b9_7f4a_7c15;
const LANE_MUL: u64 = 0xd1b5_4a32_d192_ed03;

/// splitmix64 finaliser.
#[inline(always)]
pub fn mix(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// Base hash for one (seed, env, episode, step) stream.
#[inline(always)]
pub fn stream(seed: u64, env: u32, episode: u64, step: u32) -> u64 {
    let h = mix(seed ^ GOLDEN);
    let h = mix(h ^ env as u64);
    let h = mix(h ^ episode);
    mix(h ^ step as u64)
}

/// Uniform in (0, 1) for `lane` of a stream base.
#[inline(always)]
pub fn unit(base: u64, lane: u32) -> f32 {
    let h = mix(base ^ (lane as u64).wrapping_mul(LANE_MUL));
    ((h >> 40) as f32 + 0.5) * (1.0 / 16_777_216.0)
}
