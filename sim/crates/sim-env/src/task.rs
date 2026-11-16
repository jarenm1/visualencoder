//! Pure task geometry used by both CUDA and the CPU regression tests.
pub const GAP_HALF_WIDTH: f32 = 0.8;
pub const DRONE_RADIUS: f32 = 0.18;
pub const WALL_Z: f32 = -6.0;
pub const GOAL_Z: f32 = -12.0;
pub const HORIZON: u32 = 150;

pub fn episode_gap(seed: u64, env: u32, episode: u64, style: u32) -> f32 {
    let mut z = seed ^ (env as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)
        ^ episode.wrapping_mul(0xd1b5_4a32_d192_ed03)
        ^ (style as u64).wrapping_mul(0x94d0_49bb_1331_11eb);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^= z >> 31;
    (z >> 40) as f32 * (5.0 / 16_777_216.0) - 2.5
}

/// Swept sphere against the wall's entire thickness, not just its centre plane.
pub fn hits_wall(old_x: f32, old_z: f32, x: f32, z: f32, gap: f32) -> bool {
    let front = WALL_Z + 0.15 + DRONE_RADIUS;
    let back = WALL_Z - 0.15 - DRONE_RADIUS;
    if (old_z > front && z > front) || (old_z < back && z < back) {
        return false;
    }
    let dz = z - old_z;
    let (mut a, mut b) = (0.0, 1.0);
    if dz != 0.0 {
        let t0 = (front - old_z) / dz;
        let t1 = (back - old_z) / dz;
        let (lo, hi) = if t0 < t1 { (t0, t1) } else { (t1, t0) };
        if lo > a { a = lo; }
        if hi < b { b = hi; }
    }
    if a > b { return false; }
    let xa = old_x + (x - old_x) * a - gap;
    let xb = old_x + (x - old_x) * b - gap;
    let clearance = GAP_HALF_WIDTH - DRONE_RADIUS;
    xa < -clearance || xa > clearance || xb < -clearance || xb > clearance
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn swept_wall_catches_tunnelling_and_side_entry() {
        assert!(!hits_wall(0.0, -5.0, 0.0, -7.0, 0.0));
        assert!(hits_wall(1.0, -5.0, 1.0, -7.0, 0.0));
        assert!(hits_wall(0.0, -6.0, 0.9, -6.1, 0.0));
        assert!(!hits_wall(2.0, -1.0, 2.0, -2.0, 0.0));
        assert!(!hits_wall(2.0, -8.0, 2.0, -9.0, 0.0));
        assert!(hits_wall(0.0, -5.6, 1.0, -5.8, 0.0));
    }

    #[test]
    fn gap_rng_is_reproducible_and_episode_local() {
        assert_eq!(episode_gap(7, 3, 9, 0), episode_gap(7, 3, 9, 0));
        assert_ne!(episode_gap(7, 3, 9, 0), episode_gap(7, 3, 10, 0));
        assert_ne!(episode_gap(7, 3, 9, 0), episode_gap(7, 4, 9, 0));
        assert_ne!(episode_gap(7, 3, 9, 0), episode_gap(7, 3, 9, 1));
        for episode in 0..1000 {
            let gap = episode_gap(7, 3, episode, 0);
            assert!((-2.5..2.5).contains(&gap));
        }
    }
}
