use glam::{Quat, Vec3};
use sim_render::{Instance, MeshData};
use sim_scene::CityScene;
use sim_state::BodyState;

use crate::recording::Pose;

/// Body 0 is the quadcopter. Placed city targets are part of the shared city
/// geometry, so no target body is needed.
pub const BODY_COUNT: usize = 1;

/// Appends replay-only quadcopter geometry on body 0, posed by the actual
/// simulated attitude. Rotor disks are static: rotor speed is not recorded.
pub fn append(scene: &mut CityScene) {
    let cube = scene.meshes.len();
    scene.meshes.push(MeshData::cube());
    let disk = scene.meshes.len();
    scene.meshes.push(MeshData::cylinder(24));
    let dark = [0.035, 0.055, 0.075, 1.0];
    let orange = [1.0, 0.27, 0.035, 1.0];
    let rotor = [0.14, 0.19, 0.24, 1.0];
    part(scene, cube, 0, Vec3::ZERO, Vec3::new(0.20, 0.09, 0.24), Quat::IDENTITY, orange);
    for angle in [-std::f32::consts::FRAC_PI_4, std::f32::consts::FRAC_PI_4] {
        part(scene, cube, 0, Vec3::ZERO, Vec3::new(0.43, 0.035, 0.035), Quat::from_rotation_y(angle), dark);
    }
    for x in [-0.145, 0.145] {
        for z in [-0.145, 0.145] {
            part(scene, disk, 0, Vec3::new(x, 0.025, z), Vec3::new(0.17, 0.018, 0.17), Quat::IDENTITY, rotor);
            part(scene, disk, 0, Vec3::new(x, 0.04, z), Vec3::new(0.038, 0.025, 0.038), Quat::IDENTITY, orange);
        }
    }
    // Bright nose marks body forward (-z), where the policy camera looks.
    part(scene, cube, 0, Vec3::new(0.0, 0.0, -0.128), Vec3::new(0.11, 0.05, 0.025), Quat::IDENTITY,
        [1.0, 0.85, 0.18, 1.0]);
    for x in [-0.09, 0.09] {
        part(scene, cube, 0, Vec3::new(x, -0.08, 0.0), Vec3::new(0.025, 0.035, 0.22), Quat::IDENTITY, dark);
    }
}

pub fn bodies(pose: &Pose, out: &mut Vec<BodyState>) {
    out.clear();
    out.push(BodyState { position: [pose.position[0], pose.position[1], pose.position[2], 0.0],
        rotation: pose.rotation, ..Default::default() });
}

fn part(scene: &mut CityScene, mesh: usize, body: u32, position: Vec3, scale: Vec3, rotation: Quat, color: [f32; 4]) {
    let mut instance = Instance::body(mesh, body);
    instance.position = position.to_array();
    instance.scale = scale.to_array();
    instance.rotation = rotation.to_array();
    instance.material.color = color;
    instance.material.specular = 0.45;
    instance.object_id = u32::MAX - 1;
    scene.instances.push(instance);
}

/// Horizontal heading of body forward (-z) under the recorded attitude.
pub fn heading(rotation: [f32; 4]) -> Vec3 {
    let forward = Quat::from_array(rotation) * Vec3::NEG_Z;
    Vec3::new(forward.x, 0.0, forward.z).try_normalize().unwrap_or(Vec3::NEG_Z)
}
