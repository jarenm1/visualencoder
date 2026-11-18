//! Constants and counter-based hashing for city-drone-goal-v5 (ABI v8).
//!
//! Plain integer and constant definitions shared by the host and the CUDA kernels.

pub const TASK_NAME: &str = "city-drone-goal-v5";
pub const ABI_VERSION: u32 = 8;
pub const ACTION_DIM: usize = 4;
pub const SENSOR_DIM: usize = 20;
pub const PRIVILEGED_DIM: usize = 29;
pub const DIAGNOSTIC_DIM: usize = 28;
/// Default (and largest) episode catalogue per environment; smaller ones down to the catalogue
/// builder's candidate minimum are allowed (multi-city training).
pub const CATALOG_SIZE: usize = 65536;
pub const MIN_CATALOG_SIZE: usize = 4096;
/// Floats per camera slot: view (16), view-projection (16), eye (4).
pub const CAMERA_FLOATS: usize = 36;
pub const FIELD_COUNT: usize = 20;
/// Patch size of the frozen encoders' token grid; per-patch labels use the same grid.
pub const PATCH: u32 = 16;
/// Per-patch label samples per side (LABEL_SIDE^2 pixels per patch).
pub const LABEL_SIDE: u32 = 4;
/// A patch names an object only if at least this many of its samples hit it.
pub const LABEL_MIN_HITS: u32 = 2;
/// Object boxes are grown by this margin (metres) when assigning depth hits to objects.
pub const LABEL_MARGIN: f32 = 0.2;
/// Cell size (metres) of the x/z grid that lists layout objects for label lookup.
pub const LABEL_CELL: f32 = 8.0;
/// Curriculum lanes: bucket weights (3), no-hint probability, appearance strength, and among
/// hinted episodes the probabilities of a coarse and of a wrong hint (the rest are precise).
pub const CURRICULUM_LANES: usize = 7;

/// Keep-in margin applied to the district rectangle (metres).
pub const DISTRICT_MARGIN: f32 = 2.0;
/// Salt of the per-catalog-entry appearance stream (shared by episodes and their goal views).
pub const APPEARANCE_SALT: u64 = 0x5EED_A99E_A2A9_CE01;

/// Monocular camera effects at full appearance strength (each scaled by a per-episode uniform):
/// radial distortion k (r normalized by the half-diagonal), vignette darkening at the corners,
/// exposure time (motion blur, s) and rolling-shutter readout time (top to bottom row, s).
pub const LENS_DISTORTION: f32 = 0.05;
pub const LENS_VIGNETTE: f32 = 0.35;
pub const LENS_EXPOSURE: f32 = 1.0 / 60.0;
pub const LENS_READOUT: f32 = 1.0 / 30.0;
pub const LENS_BLUR_TAPS: u32 = 5;

pub const EVENT_CRASH: u32 = 1;
pub const EVENT_OOB: u32 = 2;

/// cos(pi/6): minimum upright component of body up for a landing.
pub const COS_LANDING_TILT: f32 = 0.866_025_4;
/// Landed-body velocity damping rate (1/s) applied while in contact.
pub const SETTLE_RATE: f32 = 8.0;

/// Target hints: the commander's estimate of where the target is, fixed per episode.
/// Precise is GPS-like; coarse is a rough guess with its uncertainty radius; wrong is
/// confidently wrong (a large error behind a small claimed radius). The policy reads the hinted
/// offset and the claimed radius; progress reward before the first sighting needs a precise hint.
pub const HINT_NONE: u32 = 0;
pub const HINT_PRECISE: u32 = 1;
pub const HINT_COARSE: u32 = 2;
pub const HINT_WRONG: u32 = 3;
/// Claimed radius ranges (metres); precise and coarse errors are uniform in the claimed disc.
pub const HINT_PRECISE_RADIUS: [f32; 2] = [1.0, 5.0];
pub const HINT_COARSE_RADIUS: [f32; 2] = [10.0, 40.0];
/// Wrong hints: horizontal error (metres) while claiming a precise radius.
pub const HINT_WRONG_ERROR: [f32; 2] = [30.0, 80.0];
/// Vertical hint error is uniform in [-HINT_VERTICAL, HINT_VERTICAL] per episode, metres.
pub const HINT_VERTICAL: f32 = 1.5;
/// Normalisation for metre-scale relative positions and radii.
pub const POSITION_SCALE: f32 = 32.0;
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
pub const LANE_HINT_RADIUS: u32 = 18;
pub const LANE_HINT_ANGLE: u32 = 19;
pub const LANE_HINT_VERTICAL: u32 = 20;
pub const LANE_HINT_AVAILABLE: u32 = 21;
pub const LANE_HINT_MODE: u32 = 22;
pub const LANE_HINT_ERROR: u32 = 23;

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
