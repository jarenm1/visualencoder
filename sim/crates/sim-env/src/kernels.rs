use cuda_device::cuda_module;
use sim_state::{BodyAction, BodyState};

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct Config {
    pub n: u32,
    pub style: u32,
    pub seed: u64,
    pub projection_x: f32,
    pub projection_y: f32,
    pub projection_z: f32,
    pub projection_w: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct TaskState {
    pub old_x: f32,
    pub old_z: f32,
    pub gap: f32,
    pub clock: u32,
    pub previous_action: f32,
}

/// All pointers are persistent device allocations. Only bodies/cameras are shared
/// with Vulkan, and these kernels run exclusively inside Exchange::cycle.
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct Buffers {
    pub bodies: *mut BodyState,
    pub forces: *mut BodyAction,
    pub task: *mut TaskState,
    pub cameras: *mut f32,
    pub proprio: *mut f32,
    pub rewards: *mut f32,
    pub done: *mut f32,
    pub success: *mut f32,
    pub privileged: *mut f32,
    pub collision: *mut f32,
    pub episodes: *mut u64,
}

#[cuda_module]
pub(crate) mod device {
    use cuda_device::{kernel, thread};
    use sim_state::{BodyAction, BodyState};
    use crate::kernels::{Buffers, Config, TaskState};
    use crate::task::{DRONE_RADIUS, GAP_HALF_WIDTH, GOAL_Z, HORIZON, WALL_Z, episode_gap, hits_wall};

    unsafe fn reset_one(i: usize, buffers: Buffers, config: Config, episode: u64) {
        let base = i as f32 * 32.0;
        let gap = episode_gap(config.seed, i as u32, episode, config.style);
        unsafe {
            *buffers.bodies.add(i) = BodyState {
                position: [base, 1.2, 0.0, 0.0],
                velocity: [0.0, 0.0, -2.0, 0.0],
                rotation: [0.0, 0.0, 0.0, 1.0],
                angular_velocity: [0.0; 4],
            };
            *buffers.task.add(i) = TaskState {
                old_x: base, old_z: 0.0, gap, clock: 0, previous_action: 0.0,
            };
            let wall = BodyState {
                position: [base + gap - GAP_HALF_WIDTH - 4.0, 1.5, WALL_Z, 0.0],
                velocity: [0.0; 4], rotation: [0.0, 0.0, 0.0, 1.0], angular_velocity: [0.0; 4],
            };
            *buffers.bodies.add(config.n as usize + 2 * i) = wall;
            let mut right = wall;
            right.position[0] = base + gap + GAP_HALF_WIDTH + 4.0;
            *buffers.bodies.add(config.n as usize + 2 * i + 1) = right;
        }
    }

    unsafe fn observe_one(i: usize, buffers: Buffers, config: Config) {
        let body = unsafe { *buffers.bodies.add(i) };
        let task = unsafe { *buffers.task.add(i) };
        let [x, y, z, _] = body.position;
        let base = i as f32 * 32.0;
        let clock = task.clock as f32 / HORIZON as f32;
        let proprio = [base - x, 1.2 - y, GOAL_Z - z,
            body.velocity[0], body.velocity[1], body.velocity[2], 1.0 - clock, task.previous_action];
        let privileged = [base + task.gap - x, WALL_Z - z, GAP_HALF_WIDTH, clock];
        let mut j = 0;
        while j < 8 {
            unsafe { *buffers.proprio.add(i * 8 + j) = proprio[j]; }
            j += 1;
        }
        j = 0;
        while j < 4 {
            unsafe { *buffers.privileged.add(i * 4 + j) = privileged[j]; }
            j += 1;
        }
        // Fixed heading -Z gives an identity view rotation. Matrices are exactly
        // sim-render CameraGpu: column-major view, view_projection, eye (36 f32).
        let a = config.projection_x;
        let b = config.projection_y;
        let c = config.projection_z;
        let d = config.projection_w;
        let camera = [
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0, -x, -y, -z, 1.0,
            a, 0.0, 0.0, 0.0, 0.0, b, 0.0, 0.0,
            0.0, 0.0, c, -1.0, -a * x, -b * y, d - c * z, z,
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
            *buffers.forces.add(i) = BodyAction { force: [0.0; 4], torque: [0.0; 4] };
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
        let body = unsafe { *buffers.bodies.add(i) };
        let action = unsafe { *actions.add(i) };
        let target = (action as f32 - 1.0) * 2.0;
        let mut task = unsafe { *buffers.task.add(i) };
        task.old_x = body.position[0];
        task.old_z = body.position[2];
        task.previous_action = target * 0.5;
        unsafe {
            *buffers.task.add(i) = task;
            *buffers.forces.add(i) = BodyAction {
                force: [8.0 * (target - body.velocity[0]),
                    9.81 + 20.0 * (1.2 - body.position[1]) - 8.0 * body.velocity[1],
                    8.0 * (-2.0 - body.velocity[2]), 0.0],
                torque: [0.0; 4],
            };
        }
    }

    #[kernel]
    pub unsafe fn finish(buffers: Buffers, config: Config) {
        let i = thread::index_1d().get();
        if i >= config.n as usize { return; }
        let body = unsafe { *buffers.bodies.add(i) };
        let mut task = unsafe { *buffers.task.add(i) };
        let local_x = body.position[0] - i as f32 * 32.0;
        let collision = local_x < -4.0 + DRONE_RADIUS || local_x > 4.0 - DRONE_RADIUS
            || hits_wall(task.old_x - i as f32 * 32.0, task.old_z,
                local_x, body.position[2], task.gap);
        let success = !collision && body.position[2] <= GOAL_Z;
        task.clock += 1;
        let done = collision || success || task.clock >= HORIZON;
        let reward = (task.old_z - body.position[2]) * 0.2 - 0.01
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
            // Terminal flags/rewards survive autoreset, while the actor receives
            // the reset state's camera, proprioception, and rendered RGB.
            observe_one(i, buffers, config);
        }
    }
}
