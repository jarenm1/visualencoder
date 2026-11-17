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
