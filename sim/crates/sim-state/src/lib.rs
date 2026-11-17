#![no_std]

//! GPU representation contracts, independent of any graphics or compute runtime.
//!
//! World coordinates are right-handed, Y-up, in metres. Quaternions are xyzw;
//! linear and angular velocities are world-frame metres/s and radians/s.

use bytemuck::{Pod, Zeroable};

pub mod collision;

/// One rigid body. The fourth lane of each non-quaternion vector is reserved zero.
///
/// A zeroed allocation is not an initialized body: use [`Self::default`] or
/// [`seed_bodies`] to obtain an identity orientation.
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct BodyState {
    pub position: [f32; 4],
    pub velocity: [f32; 4],
    pub rotation: [f32; 4],
    pub angular_velocity: [f32; 4],
}

impl Default for BodyState {
    fn default() -> Self {
        Self {
            position: [0.0; 4],
            velocity: [0.0; 4],
            rotation: [0.0, 0.0, 0.0, 1.0],
            angular_velocity: [0.0; 4],
        }
    }
}

/// World-frame force (newtons) and torque (newton-metres), held for one step.
/// Fourth lanes are reserved zero. Bodies use a scalar, isotropic inertia.
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct BodyAction {
    pub force: [f32; 4],
    pub torque: [f32; 4],
}

/// Validated uniform integration parameters. Damping rates are in inverse seconds.
///
/// Velocity uses semi-implicit Euler with implicit linear drag:
/// `v_next = (v + (force / mass + gravity) * dt) / (1 + damping * dt)`.
/// Angular velocity uses the same rule with torque and isotropic inertia.
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct StepParams {
    gravity: [f32; 4],
    dt: f32,
    inverse_mass: f32,
    inverse_inertia: f32,
    linear_damping: f32,
    angular_damping: f32,
    reserved: [f32; 3],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InvalidStepParams;

impl core::fmt::Display for InvalidStepParams {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("step parameters must be finite, dt and damping nonnegative, and mass/inertia positive with finite nonzero reciprocals")
    }
}

impl core::error::Error for InvalidStepParams {}

impl StepParams {
    pub fn new(
        dt: f32,
        mass: f32,
        inertia: f32,
        linear_damping: f32,
        angular_damping: f32,
        gravity: [f32; 3],
    ) -> Result<Self, InvalidStepParams> {
        let inverse_mass = 1.0 / mass;
        let inverse_inertia = 1.0 / inertia;
        if !dt.is_finite()
            || dt < 0.0
            || !mass.is_finite()
            || mass <= 0.0
            || !inertia.is_finite()
            || inertia <= 0.0
            || !inverse_mass.is_finite()
            || inverse_mass <= 0.0
            || !inverse_inertia.is_finite()
            || inverse_inertia <= 0.0
            || !linear_damping.is_finite()
            || linear_damping < 0.0
            || !angular_damping.is_finite()
            || angular_damping < 0.0
            || !gravity.iter().all(|v| v.is_finite())
            || !(linear_damping * dt).is_finite()
            || !(angular_damping * dt).is_finite()
        {
            return Err(InvalidStepParams);
        }
        Ok(Self {
            gravity: [gravity[0], gravity[1], gravity[2], 0.0],
            dt,
            inverse_mass,
            inverse_inertia,
            linear_damping,
            angular_damping,
            reserved: [0.0; 3],
        })
    }

    pub const fn dt(self) -> f32 {
        self.dt
    }

    pub const fn gravity(self) -> [f32; 4] {
        self.gravity
    }

    pub const fn inverse_mass(self) -> f32 {
        self.inverse_mass
    }

    pub const fn inverse_inertia(self) -> f32 {
        self.inverse_inertia
    }

    pub const fn linear_damping(self) -> f32 {
        self.linear_damping
    }

    pub const fn angular_damping(self) -> f32 {
        self.angular_damping
    }
}

impl Default for StepParams {
    fn default() -> Self {
        Self {
            gravity: [0.0, -9.81, 0.0, 0.0],
            dt: 1.0 / 120.0,
            inverse_mass: 1.0,
            inverse_inertia: 1.0,
            linear_damping: 0.05,
            angular_damping: 0.05,
            reserved: [0.0; 3],
        }
    }
}

/// Host initialization only: deterministic positions, nonzero velocities, identity
/// orientations, and zero reserved lanes. Fills existing storage without allocation.
/// The same seed produces the same prefix regardless of the slice's length.
pub fn seed_bodies(states: &mut [BodyState], seed: u64) {
    let mut rng = seed;
    for state in states {
        *state = BodyState {
            position: [random_unit(&mut rng) * 8.0 - 4.0, random_unit(&mut rng) * 2.0 + 1.0, random_unit(&mut rng) * 8.0 - 4.0, 0.0],
            velocity: [random_unit(&mut rng) - 0.5, 0.0, random_unit(&mut rng) - 0.5, 0.0],
            rotation: [0.0, 0.0, 0.0, 1.0],
            angular_velocity: [0.0, random_unit(&mut rng) + 0.1, 0.0, 0.0],
        };
    }
}

// SplitMix64: the high 24 bits map exactly into the f32 unit interval.
fn random_unit(state: &mut u64) -> f32 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^= z >> 31;
    (z >> 40) as f32 * (1.0 / 16_777_216.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::mem::{align_of, offset_of, size_of};

    #[test]
    fn gpu_layouts_are_exact() {
        assert_eq!(size_of::<BodyState>(), 64);
        assert_eq!(align_of::<BodyState>(), 16);
        assert_eq!(offset_of!(BodyState, position), 0);
        assert_eq!(offset_of!(BodyState, velocity), 16);
        assert_eq!(offset_of!(BodyState, rotation), 32);
        assert_eq!(offset_of!(BodyState, angular_velocity), 48);
        assert_eq!(size_of::<BodyAction>(), 32);
        assert_eq!(align_of::<BodyAction>(), 16);
        assert_eq!(offset_of!(BodyAction, torque), 16);
        assert_eq!(size_of::<StepParams>(), 48);
        assert_eq!(align_of::<StepParams>(), 16);
        assert_eq!(offset_of!(StepParams, dt), 16);
        assert_eq!(offset_of!(StepParams, angular_damping), 32);
    }

    #[test]
    fn initialized_bodies_have_valid_orientations_and_reserved_lanes() {
        let mut first = [BodyState::default(); 7];
        let mut second = [BodyState::default(); 9];
        seed_bodies(&mut first, 42);
        seed_bodies(&mut second, 42);
        assert_eq!(&first[..], &second[..first.len()]);
        for state in first {
            assert_eq!(state.rotation, [0.0, 0.0, 0.0, 1.0]);
            assert_eq!(state.position[3], 0.0);
            assert_eq!(state.velocity[3], 0.0);
            assert_eq!(state.angular_velocity[3], 0.0);
            assert!(state.angular_velocity[1] > 0.0);
            assert!(bytemuck::cast_slice::<BodyState, f32>(&[state]).iter().all(|v| v.is_finite()));
        }
        assert_eq!(BodyState::default().rotation, [0.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn integration_parameters_validate_physical_inputs() {
        let parameters = StepParams::new(0.01, 2.0, 4.0, 0.1, 0.2, [0.0, -9.81, 0.0]).unwrap();
        assert_eq!(parameters.inverse_mass(), 0.5);
        assert_eq!(parameters.inverse_inertia(), 0.25);
        assert!(StepParams::new(0.01, 0.0, 1.0, 0.0, 0.0, [0.0; 3]).is_err());
        assert!(StepParams::new(-0.01, 1.0, 1.0, 0.0, 0.0, [0.0; 3]).is_err());
        assert!(StepParams::new(0.01, 1.0, 1.0, -1.0, 0.0, [0.0; 3]).is_err());
        assert!(StepParams::new(0.01, 1.0, 1.0, 0.0, 0.0, [f32::NAN; 3]).is_err());
        assert!(StepParams::new(0.0, 1.0, 1.0, 0.0, 0.0, [0.0; 3]).is_ok());
    }
}
