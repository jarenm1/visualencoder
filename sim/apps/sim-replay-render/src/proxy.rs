use std::f32::consts::TAU;

use glam::{Quat, Vec3};
use sim_render::{Instance, MeshData, MeshVertex};
use sim_scene::CityScene;
use sim_state::BodyState;

// Replay-only geometry: a 0.3m-radius level quadcopter, not an inferred
// aerodynamic state. The four fixed rotor disks deliberately do not fake RPM.
pub fn append(scene: &mut CityScene) {
    let cube = scene.meshes.len();
    scene.meshes.push(MeshData::cube());
    let disk = scene.meshes.len();
    scene.meshes.push(cylinder(24));
    let ring = scene.meshes.len();
    scene.meshes.push(goal_ring(40, 8));
    let dark = [0.035, 0.055, 0.075, 1.0];
    let orange = [1.0, 0.27, 0.035, 1.0];
    let rotor = [0.14, 0.19, 0.24, 1.0];
    let teal = [0.025, 1.0, 0.62, 1.0];
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
    // The bright nose shows the declared goal-facing visualization orientation.
    part(scene, cube, 0, Vec3::new(0.0, 0.0, -0.128), Vec3::new(0.11, 0.05, 0.025), Quat::IDENTITY,
        [1.0, 0.85, 0.18, 1.0]);
    for x in [-0.09, 0.09] {
        part(scene, cube, 0, Vec3::new(x, -0.08, 0.0), Vec3::new(0.025, 0.035, 0.22), Quat::IDENTITY, dark);
    }
    part(scene, ring, 1, Vec3::ZERO, Vec3::ONE, Quat::IDENTITY, teal);
    part(scene, cube, 1, Vec3::ZERO, Vec3::splat(0.13), Quat::from_rotation_z(std::f32::consts::FRAC_PI_4), teal);
    part(scene, cube, 1, Vec3::new(0.0, -0.80, 0.0), Vec3::new(0.035, 0.40, 0.035), Quat::IDENTITY, teal);
}

fn part(scene: &mut CityScene, mesh: usize, body: u32, position: Vec3, scale: Vec3, rotation: Quat, color: [f32; 4]) {
    let mut instance = Instance::body(mesh, body);
    instance.position = position.to_array();
    instance.scale = scale.to_array();
    instance.rotation = rotation.to_array();
    instance.material.color = color;
    instance.material.specular = 0.45;
    instance.object_id = if body == 0 { u32::MAX - 1 } else { u32::MAX };
    scene.instances.push(instance);
}

pub fn bodies(state: &[f32; 8]) -> [BodyState; 2] {
    let position = Vec3::new(state[0], state[1], state[2]);
    let goal = Vec3::new(state[3], state[4], state[5]);
    let forward = forward(position, goal);
    let rotation = Quat::from_rotation_y((-forward.x).atan2(-forward.z));
    [
        BodyState { position: position.extend(0.0).to_array(), rotation: rotation.to_array(), ..Default::default() },
        BodyState { position: goal.extend(0.0).to_array(), rotation: rotation.to_array(), ..Default::default() },
    ]
}

pub fn forward(position: Vec3, goal: Vec3) -> Vec3 {
    Vec3::new(goal.x - position.x, 0.0, goal.z - position.z).try_normalize().unwrap_or(Vec3::NEG_Z)
}

fn cylinder(sides: u32) -> MeshData {
    let mut mesh = MeshData { vertices: Vec::with_capacity((sides * 12) as usize), indices: Vec::with_capacity((sides * 12) as usize), material: None };
    for side in 0..sides {
        let a = TAU * side as f32 / sides as f32;
        let b = TAU * (side + 1) as f32 / sides as f32;
        let na = Vec3::new(a.cos(), 0.0, a.sin());
        let nb = Vec3::new(b.cos(), 0.0, b.sin());
        let a0 = na * 0.5 - Vec3::Y * 0.5;
        let a1 = na * 0.5 + Vec3::Y * 0.5;
        let b0 = nb * 0.5 - Vec3::Y * 0.5;
        let b1 = nb * 0.5 + Vec3::Y * 0.5;
        triangle(&mut mesh, [a0, a1, b1], [na, na, nb]);
        triangle(&mut mesh, [a0, b1, b0], [na, nb, nb]);
        triangle(&mut mesh, [Vec3::Y * 0.5, b1, a1], [Vec3::Y; 3]);
        triangle(&mut mesh, [-Vec3::Y * 0.5, a0, b0], [-Vec3::Y; 3]);
    }
    mesh
}

fn goal_ring(segments: u32, sides: u32) -> MeshData {
    let mut mesh = MeshData { vertices: Vec::with_capacity((segments * sides) as usize), indices: Vec::with_capacity((segments * sides * 6) as usize), material: None };
    for segment in 0..segments {
        let angle = TAU * segment as f32 / segments as f32;
        let radial = Vec3::new(angle.cos(), angle.sin(), 0.0);
        for side in 0..sides {
            let around = TAU * side as f32 / sides as f32;
            let normal = radial * around.cos() + Vec3::Z * around.sin();
            mesh.vertices.push(MeshVertex { position: (radial * 0.58 + normal * 0.035).to_array(), normal: normal.to_array(), uv: [0.0; 2] });
            let a = segment * sides + side;
            let b = ((segment + 1) % segments) * sides + side;
            let c = ((segment + 1) % segments) * sides + (side + 1) % sides;
            let d = segment * sides + (side + 1) % sides;
            mesh.indices.extend_from_slice(&[a, b, c, a, c, d]);
        }
    }
    mesh
}

fn triangle(mesh: &mut MeshData, positions: [Vec3; 3], normals: [Vec3; 3]) {
    for (position, normal) in positions.into_iter().zip(normals) {
        mesh.indices.push(mesh.vertices.len() as u32);
        mesh.vertices.push(MeshVertex { position: position.to_array(), normal: normal.to_array(), uv: [0.0; 2] });
    }
}
