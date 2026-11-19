use std::{cell::RefCell, ffi::{CStr, CString, c_char, c_void}, panic::{AssertUnwindSafe, catch_unwind}, path::Path};
use anyhow::{Result, ensure};
use crate::Env;

thread_local! {
    static ERROR: RefCell<CString> = RefCell::new(CString::new("").expect("empty CString"));
}

fn failure(message: impl std::fmt::Display) {
    let text = message.to_string().replace('\0', " ");
    ERROR.with(|slot| *slot.borrow_mut() = CString::new(text).expect("NUL bytes removed"));
}

fn boundary<T>(fallback: T, run: impl FnOnce() -> Result<T>) -> T {
    match catch_unwind(AssertUnwindSafe(run)) {
        Ok(Ok(value)) => value,
        Ok(Err(error)) => { failure(format_args!("{error:#}")); fallback }
        Err(payload) => {
            let message = payload.downcast_ref::<&str>().copied()
                .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
                .unwrap_or("unknown Rust panic");
            failure(format_args!("Rust panic: {message}"));
            fallback
        }
    }
}

unsafe fn environment<'a>(handle: *mut c_void) -> Result<&'a mut Env> {
    ensure!(!handle.is_null(), "environment handle must not be null");
    // SAFETY: C ABI caller guarantees a live, exclusively accessed Env handle.
    Ok(unsafe { &mut *handle.cast::<Env>() })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn sim_env_create(n: u32, seed: u64, width: u32, height: u32, style: u32,
    control_mode: u32, city_path: *const c_char) -> *mut c_void {
    boundary(std::ptr::null_mut(), || {
        ensure!(!city_path.is_null(), "city_path must not be null");
        // SAFETY: caller supplies a readable, NUL-terminated UTF-8 path.
        let path = unsafe { CStr::from_ptr(city_path) }.to_str()?;
        ensure!(!path.is_empty(), "city_path must not be empty");
        Ok(Box::into_raw(Box::new(Env::new(n, seed, width, height, style, control_mode, Path::new(path))?)).cast())
    })
}

/// `sim_env_create` with a start-distance range (metres from the target's box surface) and an
/// episode catalogue size (`MIN_CATALOG_SIZE..=CATALOG_SIZE`).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sim_env_create_with_options(n: u32, seed: u64, width: u32, height: u32,
    style: u32, control_mode: u32, city_path: *const c_char, min_distance: f32, max_distance: f32,
    catalog_size: u32) -> *mut c_void {
    boundary(std::ptr::null_mut(), || {
        ensure!(!city_path.is_null(), "city_path must not be null");
        let path = unsafe { CStr::from_ptr(city_path) }.to_str()?;
        ensure!(!path.is_empty(), "city_path must not be empty");
        Ok(Box::into_raw(Box::new(Env::new_with_options(n, seed, width, height, style, control_mode,
            Path::new(path), [min_distance, max_distance], catalog_size as usize)?)).cast())
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn sim_env_reset(handle: *mut c_void, seed: u64) -> i32 {
    boundary(-1, || {
        unsafe { environment(handle)? }.reset(seed)?;
        Ok(0)
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn sim_env_step(handle: *mut c_void, actions: *const f32) -> i32 {
    boundary(-1, || {
        unsafe { environment(handle)?.step(actions)?; }
        Ok(0)
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn sim_env_render_goals(handle: *mut c_void, first: u32) -> i32 {
    boundary(-1, || {
        unsafe { environment(handle)? }.render_goals(first)?;
        Ok(0)
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn sim_env_retarget(handle: *mut c_void, targets: *const f32) -> i32 {
    boundary(-1, || {
        unsafe { environment(handle)?.retarget(targets)?; }
        Ok(0)
    })
}

/// Collision occupancy of an nx x ny x nz voxel grid (origin = grid corner, metres) into `out`
/// (CUDA u8, x fastest), enqueued on the env stream.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sim_env_occupancy(handle: *mut c_void, origin_x: f32, origin_y: f32, origin_z: f32,
    cell: f32, nx: u32, ny: u32, nz: u32, radius: f32, out: *mut u8) -> i32 {
    boundary(-1, || {
        unsafe { environment(handle)?.occupancy([origin_x, origin_y, origin_z], cell, [nx, ny, nz], radius, out)?; }
        Ok(0)
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn sim_env_set_free_flight(handle: *mut c_void, enabled: u32) -> i32 {
    boundary(-1, || {
        unsafe { environment(handle)? }.set_free_flight(enabled != 0);
        Ok(0)
    })
}

/// Step profiler on/off (enabled != 0 starts a fresh window). While on, every step blocks
/// until its GPU work completes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sim_env_set_profiling(handle: *mut c_void, enabled: u32) -> i32 {
    boundary(-1, || {
        unsafe { environment(handle)? }.set_profiling(enabled != 0)?;
        Ok(0)
    })
}

/// Writes `len` (>= stage count) float64 stage means, `crate::STAGES` order, and starts a new window.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sim_env_profile(handle: *mut c_void, out: *mut f64, len: u32) -> i32 {
    boundary(-1, || {
        anyhow::ensure!(!out.is_null(), "profile output pointer must not be null");
        // SAFETY: the caller provides `len` writable float64 values.
        let out = unsafe { std::slice::from_raw_parts_mut(out, len as usize) };
        unsafe { environment(handle)? }.profile(out)?;
        Ok(0)
    })
}

/// Writes `crate::RENDER_STATS` float64 scene/culling statistics of the last render. Blocks.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sim_env_render_stats(handle: *mut c_void, out: *mut f64, len: u32) -> i32 {
    boundary(-1, || {
        anyhow::ensure!(!out.is_null() && len as usize >= crate::RENDER_STATS,
            "render stats need {} float64 values", crate::RENDER_STATS);
        let stats = unsafe { environment(handle)? }.render_stats()?;
        // SAFETY: the caller provides at least RENDER_STATS writable float64 values.
        unsafe { std::ptr::copy_nonoverlapping(stats.as_ptr(), out, stats.len()) };
        Ok(0)
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn sim_env_buffer(handle: *mut c_void, field: i32) -> u64 {
    boundary(0, || unsafe { environment(handle)? }.buffer(field))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn sim_env_stream(handle: *mut c_void) -> u64 {
    boundary(0, || Ok(unsafe { environment(handle)? }.stream()))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn sim_env_metadata(handle: *mut c_void) -> *const c_char {
    boundary(std::ptr::null(), || Ok(unsafe { environment(handle)? }.metadata()))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn sim_env_destroy(handle: *mut c_void) {
    boundary((), || {
        if !handle.is_null() {
            // SAFETY: caller relinquishes this unique live handle exactly once.
            drop(unsafe { Box::from_raw(handle.cast::<Env>()) });
        }
        Ok(())
    });
}

#[unsafe(no_mangle)]
pub extern "C" fn sim_env_error() -> *const c_char {
    ERROR.with(|slot| slot.borrow().as_ptr())
}
