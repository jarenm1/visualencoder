//! Concrete kernels are embedded in this library's `.oxart` section by cuda-oxide.
//! No CUDA source files or precompiled PTX participate in this crate.

use cuda_device::cuda_module;

#[cuda_module]
pub(crate) mod device {
    use cuda_device::{SharedArray, float, kernel, thread};
    use sim_state::{BodyAction, BodyState, StepParams};

    use crate::ImageStatistics;

    /// One thread per body. Force and torque are world-frame; inertia is isotropic.
    ///
    /// # Safety
    /// The launch is one-dimensional. `states` and `actions` are disjoint,
    /// aligned device allocations containing `count` initialized elements.
    /// Each body has finite state, a nonzero quaternion, and finite actions;
    /// the chosen step must keep intermediate values representable as f32.
    #[kernel]
    pub unsafe fn integrate(
        states: *mut BodyState,
        actions: *const BodyAction,
        count: u32,
        params: StepParams,
    ) {
        let i = thread::index_1d().get();
        if i >= count as usize {
            return;
        }
        // SAFETY: bounds above and the kernel's allocation/aliasing contract.
        let mut state = unsafe { *states.add(i) };
        let action = unsafe { *actions.add(i) };
        let dt = params.dt();
        let gravity = params.gravity();
        let linear_decay = 1.0 / (1.0 + params.linear_damping() * dt);
        let angular_decay = 1.0 / (1.0 + params.angular_damping() * dt);
        let mut axis = 0;
        while axis < 3 {
            state.velocity[axis] = (state.velocity[axis]
                + (action.force[axis] * params.inverse_mass() + gravity[axis]) * dt)
                * linear_decay;
            state.position[axis] += state.velocity[axis] * dt;
            state.angular_velocity[axis] = (state.angular_velocity[axis]
                + action.torque[axis] * params.inverse_inertia() * dt)
                * angular_decay;
            axis += 1;
        }

        // World-frame angular velocity: q_dot = (omega, 0) * q / 2.
        let wx = state.angular_velocity[0];
        let wy = state.angular_velocity[1];
        let wz = state.angular_velocity[2];
        let half_dt = 0.5 * dt;
        let [qx, qy, qz, qw] = state.rotation;
        let x = qx + half_dt * (wx * qw + wy * qz - wz * qy);
        let y = qy + half_dt * (-wx * qz + wy * qw + wz * qx);
        let z = qz + half_dt * (wx * qy - wy * qx + wz * qw);
        let w = qw - half_dt * (wx * qx + wy * qy + wz * qz);
        let inverse_norm = float::rsqrt_approx_f32(x * x + y * y + z * z + w * w);
        state.rotation = [x * inverse_norm, y * inverse_norm, z * inverse_norm, w * inverse_norm];
        state.position[3] = 0.0;
        state.velocity[3] = 0.0;
        state.angular_velocity[3] = 0.0;
        // SAFETY: only this invocation writes body i.
        unsafe { *states.add(i) = state };
    }

    /// Convert only the leading RGBA8 region of the renderer's shared output.
    ///
    /// # Safety
    /// Launch is one-dimensional. `rgba` covers 4*pixel_count readable bytes;
    /// `rgb` covers 3*pixel_count writable f32 values, disjoint from input.
    /// `pixels_per_image` is positive and divides `pixel_count`.
    #[kernel]
    pub unsafe fn rgba8_to_nchw(
        rgba: *const u8,
        rgb: *mut f32,
        pixel_count: u32,
        pixels_per_image: u32,
    ) {
        let i = thread::index_1d().get();
        if i >= pixel_count as usize {
            return;
        }
        let plane = pixels_per_image as usize;
        let camera = i / plane;
        let pixel = i % plane;
        let output = camera * 3 * plane + pixel;
        let scale = 1.0 / 255.0;
        // SAFETY: validated shape and the kernel's disjoint allocation contract;
        // each thread owns one pixel in each of the three output planes.
        unsafe {
            *rgb.add(output) = *rgba.add(i * 4) as f32 * scale;
            *rgb.add(output + plane) = *rgba.add(i * 4 + 1) as f32 * scale;
            *rgb.add(output + 2 * plane) = *rgba.add(i * 4 + 2) as f32 * scale;
        }
    }

    /// One 256-thread block per camera, with a strided scan and tree reduction.
    ///
    /// # Safety
    /// Exactly 256 x-threads/block, one x-block/camera, all other dimensions 1.
    /// `rgb` holds camera-major NCHW initialized finite RGB in [0,1], with
    /// `pixels_per_image > 0`. `statistics` holds one aligned writable element
    /// per block and is disjoint from input. All threads participate in barriers.
    #[kernel]
    pub unsafe fn reduce_images(
        rgb: *const f32,
        statistics: *mut ImageStatistics,
        pixels_per_image: u32,
    ) {
        static mut RED: SharedArray<f32, 256> = SharedArray::UNINIT;
        static mut GREEN: SharedArray<f32, 256> = SharedArray::UNINIT;
        static mut BLUE: SharedArray<f32, 256> = SharedArray::UNINIT;
        static mut MINIMUM: SharedArray<f32, 256> = SharedArray::UNINIT;
        static mut MAXIMUM: SharedArray<f32, 256> = SharedArray::UNINIT;

        // SAFETY: raw receivers avoid overlapping mutable references to shared
        // arrays. Per-slot ownership and barriers below govern all accesses.
        let red_shared = unsafe { SharedArray::as_raw_mut_ptr(&raw mut RED) };
        let green_shared = unsafe { SharedArray::as_raw_mut_ptr(&raw mut GREEN) };
        let blue_shared = unsafe { SharedArray::as_raw_mut_ptr(&raw mut BLUE) };
        let minimum_shared = unsafe { SharedArray::as_raw_mut_ptr(&raw mut MINIMUM) };
        let maximum_shared = unsafe { SharedArray::as_raw_mut_ptr(&raw mut MAXIMUM) };

        let lane = thread::threadIdx_x() as usize;
        let camera = thread::blockIdx_x() as usize;
        let plane = pixels_per_image as usize;
        let base = camera * 3 * plane;
        let mut red = 0.0;
        let mut green = 0.0;
        let mut blue = 0.0;
        let mut minimum = f32::INFINITY;
        let mut maximum = f32::NEG_INFINITY;
        let mut pixel = lane;
        while pixel < plane {
            // SAFETY: each strided index is within this camera's three planes.
            let r = unsafe { *rgb.add(base + pixel) };
            let g = unsafe { *rgb.add(base + plane + pixel) };
            let b = unsafe { *rgb.add(base + 2 * plane + pixel) };
            let luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b;
            red += r;
            green += g;
            blue += b;
            minimum = minimum.min(luminance);
            maximum = maximum.max(luminance);
            pixel += 256;
        }
        // SAFETY: every thread writes only its own initialized shared slot.
        unsafe {
            *red_shared.add(lane) = red;
            *green_shared.add(lane) = green;
            *blue_shared.add(lane) = blue;
            *minimum_shared.add(lane) = minimum;
            *maximum_shared.add(lane) = maximum;
        }
        thread::sync_threads();
        let mut stride = 128;
        while stride > 0 {
            if lane < stride {
                // SAFETY: the previous barrier makes both slots visible. Active
                // threads own disjoint destinations; partners are read-only.
                unsafe {
                    *red_shared.add(lane) += *red_shared.add(lane + stride);
                    *green_shared.add(lane) += *green_shared.add(lane + stride);
                    *blue_shared.add(lane) += *blue_shared.add(lane + stride);
                    *minimum_shared.add(lane) =
                        (*minimum_shared.add(lane)).min(*minimum_shared.add(lane + stride));
                    *maximum_shared.add(lane) =
                        (*maximum_shared.add(lane)).max(*maximum_shared.add(lane + stride));
                }
            }
            thread::sync_threads();
            stride /= 2;
        }
        if lane == 0 {
            let reciprocal_count = 1.0 / pixels_per_image as f32;
            // SAFETY: reduction completed, and this block owns statistics[camera].
            unsafe {
                let r = *red_shared * reciprocal_count;
                let g = *green_shared * reciprocal_count;
                let b = *blue_shared * reciprocal_count;
                *statistics.add(camera) = ImageStatistics {
                    mean_rgb: [r, g, b, 0.0],
                    mean_luminance: 0.2126 * r + 0.7152 * g + 0.0722 * b,
                    min_luminance: *minimum_shared,
                    max_luminance: *maximum_shared,
                    pixel_count: pixels_per_image,
                };
            }
        }
    }
}
