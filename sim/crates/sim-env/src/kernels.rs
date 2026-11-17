use cuda_device::cuda_module;
use sim_state::{BodyState, collision::{BvhNode, Triangle}};
use crate::task::{Route, Waypoint};

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct Config {
    pub n: u32,
    pub route_first: u32,
    pub route_count: u32,
    pub node_count: u32,
    pub seed: u64,
    pub projection_x: f32,
    pub projection_y: f32,
    pub projection_z: f32,
    pub projection_w: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct TaskState {
    pub route: u32,
    pub waypoint: u32,
    pub clock: u32,
    pub collision: u32,
    pub previous_action: f32,
    pub remaining: f32,
}

/// Persistent device pointers. Only bodies/cameras are shared with Vulkan.
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct Buffers {
    pub bodies: *mut BodyState,
    pub task: *mut TaskState,
    pub cameras: *mut f32,
    pub nodes: *const BvhNode,
    pub triangles: *const Triangle,
    pub routes: *const Route,
    pub waypoints: *const Waypoint,
    pub proprio: *mut f32,
    pub rewards: *mut f32,
    pub done: *mut f32,
    pub success: *mut f32,
    pub privileged: *mut f32,
    pub collision: *mut f32,
    pub episodes: *mut u64,
    pub diagnostics: *mut f32,
}

#[cuda_module]
pub(crate) mod device {
    use cuda_device::{float, kernel, thread};
    use sim_state::{BodyState, collision::sweep_hits};
    use crate::kernels::{Buffers, Config, TaskState};
    use crate::task::*;

    fn distance(a: [f32; 3], b: [f32; 3]) -> f32 {
        let dx = a[0] - b[0];
        let dz = a[2] - b[2];
        float::sqrt_rn_f32(dx * dx + dz * dz)
    }

    fn frame(position: [f32; 4], goal: [f32; 3]) -> [f32; 4] {
        let dx = goal[0] - position[0];
        let dz = goal[2] - position[2];
        let length = float::sqrt_rn_f32(dx * dx + dz * dz);
        if length > 1e-6 {
            let fx = dx / length;
            let fz = dz / length;
            [-fz, fx, fx, fz] // right xz, forward xz
        } else { [1.0, 0.0, 0.0, -1.0] }
    }

    unsafe fn reset_one(i: usize, buffers: Buffers, config: Config, episode: u64) {
        let id = config.route_first + episode_route(config.seed, i as u32, episode, config.route_count);
        let route = unsafe { *buffers.routes.add(id as usize) };
        let start = unsafe { (*buffers.waypoints.add(route.first as usize)).position };
        unsafe {
            *buffers.bodies.add(i) = BodyState {
                position: [start[0], ALTITUDE, start[2], 0.0],
                velocity: [0.0; 4], rotation: [0.0, 0.0, 0.0, 1.0], angular_velocity: [0.0; 4],
            };
            *buffers.task.add(i) = TaskState { route: id, waypoint: 1, clock: 0,
                collision: 0, previous_action: 0.0, remaining: route.length };
        }
    }

    unsafe fn observe_one(i: usize, buffers: Buffers, config: Config) {
        let body = unsafe { *buffers.bodies.add(i) };
        let task = unsafe { *buffers.task.add(i) };
        let route = unsafe { *buffers.routes.add(task.route as usize) };
        let goal = unsafe { (*buffers.waypoints.add((route.first + route.count - 1) as usize)).position };
        let next = unsafe { (*buffers.waypoints.add((route.first + task.waypoint) as usize)).position };
        let [x, y, z, _] = body.position;
        let [rx, rz, fx, fz] = frame(body.position, goal);
        let clock = task.clock as f32 / HORIZON as f32;
        let goal_distance = distance([x, y, z], goal);
        let proprio = [0.0, ALTITUDE - y, -goal_distance,
            body.velocity[0] * rx + body.velocity[2] * rz, body.velocity[1],
            body.velocity[0] * fx + body.velocity[2] * fz, 1.0 - clock, task.previous_action];
        let dx = next[0] - x;
        let dz = next[2] - z;
        let privileged = [(dx * rx + dz * rz) / 32.0, (dx * fx + dz * fz) / 32.0,
            task.remaining / 48.0, clock];
        let diagnostics = [x, y, z, goal[0], goal[1], goal[2], task.route as f32, task.clock as f32];
        let mut j = 0;
        while j < 8 {
            unsafe {
                *buffers.proprio.add(i * 8 + j) = proprio[j];
                *buffers.diagnostics.add(i * 8 + j) = diagnostics[j];
            }
            j += 1;
        }
        j = 0;
        while j < 4 {
            unsafe { *buffers.privileged.add(i * 4 + j) = privileged[j]; }
            j += 1;
        }
        // Right-handed level look-at, followed by the renderer's RH projection.
        // CameraGpu is column-major view, view_projection, eye (36 floats).
        let rdot = rx * x + rz * z;
        let fdot = fx * x + fz * z;
        let a = config.projection_x;
        let b = config.projection_y;
        let c = config.projection_z;
        let d = config.projection_w;
        let camera = [
            rx, 0.0, -fx, 0.0, 0.0, 1.0, 0.0, 0.0,
            rz, 0.0, -fz, 0.0, -rdot, -y, fdot, 1.0,
            a * rx, 0.0, -c * fx, fx, 0.0, b, 0.0, 0.0,
            a * rz, 0.0, -c * fz, fz, -a * rdot, -b * y, c * fdot + d, -fdot,
            x, y, z, 1.0,
        ];
        j = 0;
        while j < 36 {
            unsafe { *buffers.cameras.add(i * 36 + j) = camera[j]; }
            j += 1;
        }
    }

    #[kernel]
    pub unsafe fn reset(buffers: Buffers, config: Config) {
        let i = thread::index_1d().get();
        if i >= config.n as usize { return; }
        unsafe {
            *buffers.episodes.add(i) = 0;
            *buffers.rewards.add(i) = 0.0;
            *buffers.done.add(i) = 0.0;
            *buffers.success.add(i) = 0.0;
            *buffers.collision.add(i) = 0.0;
            reset_one(i, buffers, config, 0);
            observe_one(i, buffers, config);
        }
    }

    #[kernel]
    pub unsafe fn control(buffers: Buffers, config: Config, actions: *const i64) {
        let i = thread::index_1d().get();
        if i >= config.n as usize { return; }
        let mut body = unsafe { *buffers.bodies.add(i) };
        let mut task = unsafe { *buffers.task.add(i) };
        let route = unsafe { *buffers.routes.add(task.route as usize) };
        let goal = unsafe { (*buffers.waypoints.add((route.first + route.count - 1) as usize)).position };
        let action = unsafe { *actions.add(i) };
        let action = if action >= 0 && action <= 8 { action } else { 0 };
        let [right, forward] = action_direction(action);
        let [rx, rz, fx, fz] = frame(body.position, goal);
        body.velocity = [SPEED * (right * rx + forward * fx), 0.0,
            SPEED * (right * rz + forward * fz), 0.0];
        task.previous_action = action as f32 / 8.0;
        task.collision = 0;
        let mut tick = 0;
        while tick < SUBSTEPS {
            let from = [body.position[0], ALTITUDE, body.position[2]];
            let to = [from[0] + body.velocity[0] / PHYSICS_HZ as f32, ALTITUDE,
                from[2] + body.velocity[2] / PHYSICS_HZ as f32];
            if unsafe { sweep_hits(buffers.nodes, config.node_count, buffers.triangles, from, to, DRONE_RADIUS) } {
                task.collision = 1;
                body.velocity = [0.0; 4];
                break;
            }
            body.position = [to[0], ALTITUDE, to[2], 0.0];
            tick += 1;
        }
        unsafe {
            *buffers.task.add(i) = task;
            *buffers.bodies.add(i) = body;
        }
    }

    #[kernel]
    pub unsafe fn finish(buffers: Buffers, config: Config) {
        let i = thread::index_1d().get();
        if i >= config.n as usize { return; }
        let body = unsafe { *buffers.bodies.add(i) };
        let mut task = unsafe { *buffers.task.add(i) };
        let route = unsafe { *buffers.routes.add(task.route as usize) };
        let goal = unsafe { (*buffers.waypoints.add((route.first + route.count - 1) as usize)).position };
        let position = [body.position[0], body.position[1], body.position[2]];
        let old_remaining = task.remaining;
        let mut next = unsafe { *buffers.waypoints.add((route.first + task.waypoint) as usize) };
        while task.waypoint + 1 < route.count && distance(position, next.position) <= WAYPOINT_RADIUS {
            task.waypoint += 1;
            next = unsafe { *buffers.waypoints.add((route.first + task.waypoint) as usize) };
        }
        task.remaining = distance(position, next.position) + next.remaining;
        let collision = task.collision != 0;
        let success = !collision && distance(position, goal) <= GOAL_RADIUS;
        task.clock += 1;
        let done = collision || success || task.clock >= HORIZON;
        let reward = 0.2 * (old_remaining - task.remaining) - 0.01
            + if success { 5.0 } else if collision { -5.0 } else { 0.0 };
        unsafe {
            *buffers.rewards.add(i) = reward;
            *buffers.done.add(i) = if done { 1.0 } else { 0.0 };
            *buffers.success.add(i) = if success { 1.0 } else { 0.0 };
            *buffers.collision.add(i) = if collision { 1.0 } else { 0.0 };
            *buffers.task.add(i) = task;
            if done {
                let episode = (*buffers.episodes.add(i)).wrapping_add(1);
                *buffers.episodes.add(i) = episode;
                reset_one(i, buffers, config, episode);
            }
            // Preserve transition outputs while publishing the reset observation.
            observe_one(i, buffers, config);
        }
    }
}
