//! Shared, allocation-free city task constants and route layouts.
use bytemuck::{Pod, Zeroable};

pub const ALTITUDE: f32 = 6.0;
pub const SPEED: f32 = 3.0;
pub const DRONE_RADIUS: f32 = 0.35;
pub const PLANNING_RADIUS: f32 = 0.85;
pub const GOAL_RADIUS: f32 = 1.0;
pub const WAYPOINT_RADIUS: f32 = 0.25;
pub const HORIZON: u32 = 320;
pub const PHYSICS_HZ: u32 = 120;
pub const SUBSTEPS: u32 = 6;
pub const ROUTES_PER_SPLIT: usize = 128;
pub const GRID_SPACING: f32 = 2.0;
pub const DISTRICT_HALF_SIZE: f32 = 32.0;
pub const MIN_ROUTE_LENGTH: f32 = 12.0;
pub const MAX_ROUTE_LENGTH: f32 = 38.0;
pub const CATALOG_SEEDS: [u64; 3] = [
    0x65a4_947b_3a2d_7811, 0xb8d0_c5e4_109a_3267, 0x2f61_39ac_75e8_d403,
];

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct Route {
    pub first: u32,
    pub count: u32,
    pub length: f32,
    pub reserved: u32,
}

/// xyz position and exact suffix polyline distance to final goal in metres.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct Waypoint {
    pub position: [f32; 3],
    pub remaining: f32,
}

pub fn mix(mut x: u64) -> u64 {
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

pub fn episode_route(seed: u64, env: u32, episode: u64, count: u32) -> u32 {
    (mix(seed ^ (env as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)
        ^ episode.wrapping_mul(0xd1b5_4a32_d192_ed03)) % count as u64) as u32
}

/// (right, forward), normalized; the public action numbering is clockwise.
pub fn action_direction(action: i64) -> [f32; 2] {
    const D: f32 = core::f32::consts::FRAC_1_SQRT_2;
    match action {
        1 => [0.0, 1.0], 2 => [D, D], 3 => [1.0, 0.0],
        4 => [D, -D], 5 => [0.0, -1.0], 6 => [-D, -D],
        7 => [-1.0, 0.0], 8 => [-D, D], _ => [0.0, 0.0],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn commands_are_unit_and_clockwise() {
        assert_eq!(action_direction(0), [0.0, 0.0]);
        assert_eq!(action_direction(3), [1.0, 0.0]);
        assert_eq!(action_direction(7), [-1.0, 0.0]);
        for a in 1..=8 {
            let [r, f] = action_direction(a);
            assert!((r * r + f * f - 1.0).abs() < 1e-6);
        }
    }
    #[test]
    fn route_selection_is_episode_local_and_repeatable() {
        // Repeated routes are legal; test independent sequences, not uniqueness.
        assert!((0..128).any(|episode|
            episode_route(7, 2, episode, 128) != episode_route(7, 3, episode, 128)));
        assert!((0..128).any(|episode|
            episode_route(7, 2, episode, 128) != episode_route(8, 2, episode, 128)));
    }
}
