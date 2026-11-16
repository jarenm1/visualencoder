use std::{
    os::fd::{AsRawFd, FromRawFd, IntoRawFd, OwnedFd},
    panic::{AssertUnwindSafe, catch_unwind, resume_unwind},
    sync::{Arc, atomic::Ordering},
};

use anyhow::{Context as _, Result, ensure};
use ash::{khr, vk};

use crate::{Gpu, SharedBuffer, cuda_result};

/// A fixed set of buffers scheduled as one Vulkan → CUDA → Vulkan exchange.
///
/// The two ownership command buffers are recorded once with SIMULTANEOUS_USE,
/// never reset or reallocated per frame. One exported timeline is signaled to
/// 2n+1 by Vulkan and to 2n+2 by CUDA. The intervening acquire prevents Vulkan
/// from signaling the next odd value before CUDA has signaled the current even
/// value. Thus signals are strictly ordered even with many frames CPU-enqueued.
/// Dropping an exchange drains both APIs before releasing its fixed resources.
pub struct Exchange {
    enrollment: Enrollment,
    timeline: Timeline,
    _pool: CommandPool,
    release: vk::CommandBuffer,
    acquire: vk::CommandBuffer,
    values: Values,
    poisoned: bool,
    // Last: command pool and CUDA/Vulkan semaphore cleanup require a live device.
    gpu: Arc<Gpu>,
}

impl Exchange {
    pub fn new(gpu: Arc<Gpu>, buffers: Vec<Arc<SharedBuffer>>) -> Result<Self> {
        let enrollment = Enrollment::new(&gpu, buffers)?;
        let timeline = Timeline::new(gpu.clone())?;
        let pool = CommandPool::new(&gpu)?;
        let commands = pool.allocate(2)?;
        record_ownership(&gpu, commands[0], &enrollment.buffers, false)?;
        record_ownership(&gpu, commands[1], &enrollment.buffers, true)?;
        Ok(Self {
            gpu, enrollment, timeline, _pool: pool,
            release: commands[0], acquire: commands[1],
            values: Values::default(), poisoned: false,
        })
    }

    /// Submit renderer work, hand every listed buffer to CUDA, then return it to
    /// Vulkan. This schedules only GPU waits/signals; it does not wait on the host.
    /// An empty encoder is useful for the initial CUDA state initialization.
    ///
    /// The returned index covers renderer submission, NOT subsequent CUDA work.
    /// Use Gpu::wait_idle only at explicit startup/shutdown boundaries. A CUDA
    /// callback error still schedules signal/acquire, permitting a later cycle;
    /// driver handoff errors poison the exchange and its buffers instead.
    ///
    /// # Safety
    /// - Every Vulkan access to these buffers must be in this cycle's encoder.
    ///   Do not use Gpu::submit, queue writes or another queue to access them.
    /// - The callback may enqueue buffer accesses only on Gpu::cuda_stream().
    ///   Other streams must join this stream before the callback returns.
    /// - CUDA pointers are borrowed and must stay within each buffer's byte range.
    ///   The callback must not free mappings, retain pointers for later launches,
    ///   synchronize the stream/device, or reenter Gpu queue operations.
    /// - Do not manually destroy the wgpu buffers or clone/retain a raw queue.
    pub unsafe fn cycle(
        &mut self,
        mut encoder: wgpu::CommandEncoder,
        cuda: impl FnOnce() -> Result<()>,
    ) -> Result<wgpu::SubmissionIndex> {
        ensure!(!self.poisoned, "exchange is poisoned by a previous driver handoff error");
        let (released, returned) = self.values.next()?;
        // COPY_SRC is a real, supported boundary state. wgpu generates all
        // shader-write/read transitions before EXTERNAL release and remembers
        // this same state for its next frame. The raw acquire makes external
        // writes visible without lying to wgpu's internal state tracker.
        encoder.transition_resources(self.enrollment.buffers.iter().map(|buffer| {
            wgpu::BufferTransition { buffer: buffer.buffer(), state: wgpu::BufferUses::COPY_SRC }
        }), std::iter::empty());
        let command = encoder.finish();
        let gpu = self.gpu.clone();
        let _guard = gpu.lock_queue();
        gpu.cuda_context().bind_to_thread()?;
        let submission = gpu.queue_locked().submit([command]);
        if let Err(error) = self.submit_release(released) {
            self.poison();
            return Err(error);
        }
        if let Err(error) = self.timeline.cuda_wait(released) {
            // No Vulkan wait on an unsignalled CUDA value is submitted here.
            self.poison();
            return Err(error);
        }
        // Return ownership even when the callback errors or panics, so unwinding
        // cannot strand a pending Vulkan submission on an absent CUDA signal.
        let callback = catch_unwind(AssertUnwindSafe(cuda));
        if let Err(error) = self.timeline.cuda_signal(returned) {
            self.poison();
            if let Err(panic) = callback { resume_unwind(panic); }
            return Err(error);
        }
        if let Err(error) = self.submit_acquire(returned) {
            self.poison();
            if let Err(panic) = callback { resume_unwind(panic); }
            return Err(error);
        }
        self.values.completed(returned);
        match callback {
            Ok(result) => { result?; Ok(submission) }
            Err(panic) => resume_unwind(panic),
        }
    }

    fn submit_release(&self, value: u64) -> Result<()> {
        let values = [value];
        let semaphores = [self.timeline.vulkan];
        let commands = [self.release];
        let mut timeline = vk::TimelineSemaphoreSubmitInfo::default().signal_semaphore_values(&values);
        let submit = vk::SubmitInfo::default().command_buffers(&commands)
            .signal_semaphores(&semaphores).push_next(&mut timeline);
        // SAFETY: cycle holds the queue lock; immutable SIMULTANEOUS_USE command
        // buffers and monotonically increasing values are retained by Exchange.
        unsafe { self.gpu.raw.queue_submit(self.gpu.raw_queue, &[submit], vk::Fence::null()) }
            .context("submitting Vulkan EXTERNAL release")
    }

    fn submit_acquire(&self, value: u64) -> Result<()> {
        let values = [value];
        let semaphores = [self.timeline.vulkan];
        let stages = [vk::PipelineStageFlags::ALL_COMMANDS];
        let commands = [self.acquire];
        let mut timeline = vk::TimelineSemaphoreSubmitInfo::default().wait_semaphore_values(&values);
        let submit = vk::SubmitInfo::default().command_buffers(&commands)
            .wait_semaphores(&semaphores).wait_dst_stage_mask(&stages).push_next(&mut timeline);
        // SAFETY: the matching CUDA signal has already been enqueued. Waiting at
        // ALL_COMMANDS gates the acquire and subsequent queue work, not the host.
        unsafe { self.gpu.raw.queue_submit(self.gpu.raw_queue, &[submit], vk::Fence::null()) }
            .context("submitting Vulkan EXTERNAL acquire")
    }

    fn poison(&mut self) {
        self.poisoned = true;
        for buffer in &self.enrollment.buffers { buffer.poison(); }
    }
}

impl Drop for Exchange {
    fn drop(&mut self) {
        let _guard = self.gpu.lock_queue();
        // Shutdown only. Fields (semaphore, pool, buffer owners) drop afterward.
        let _ = self.gpu.drain_locked();
    }
}

struct Enrollment { buffers: Vec<Arc<SharedBuffer>> }

impl Enrollment {
    fn new(gpu: &Arc<Gpu>, buffers: Vec<Arc<SharedBuffer>>) -> Result<Self> {
        ensure!(!buffers.is_empty(), "exchange must contain at least one buffer");
        for (index, buffer) in buffers.iter().enumerate() {
            ensure!(Arc::ptr_eq(gpu, &buffer.gpu), "exchange buffer belongs to another Gpu");
            ensure!(!buffer.unusable.load(Ordering::Acquire), "exchange buffer was poisoned by a driver handoff failure");
            ensure!(!buffers[..index].iter().any(|previous| Arc::ptr_eq(previous, buffer)),
                "exchange contains a duplicate buffer");
        }
        for (index, buffer) in buffers.iter().enumerate() {
            if buffer.enrolled.compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire).is_err() {
                for previous in &buffers[..index] { previous.enrolled.store(false, Ordering::Release); }
                anyhow::bail!("buffer is already enrolled in another Exchange");
            }
        }
        // A previous owner may have poisoned a buffer between the first check
        // and releasing its enrollment. Our acquire claim observes that release.
        if buffers.iter().any(|buffer| buffer.unusable.load(Ordering::Acquire)) {
            for buffer in &buffers { buffer.enrolled.store(false, Ordering::Release); }
            anyhow::bail!("exchange buffer was poisoned by a driver handoff failure");
        }
        Ok(Self { buffers })
    }
}

impl Drop for Enrollment {
    fn drop(&mut self) {
        for buffer in &self.buffers { buffer.enrolled.store(false, Ordering::Release); }
    }
}

#[derive(Default)]
struct Values { last_returned: u64 }

impl Values {
    fn next(&self) -> Result<(u64, u64)> {
        let returned = self.last_returned.checked_add(2).context("external timeline value exhausted")?;
        Ok((returned - 1, returned))
    }
    fn completed(&mut self, returned: u64) { self.last_returned = returned; }
}

struct Timeline {
    gpu: Arc<Gpu>,
    vulkan: vk::Semaphore,
    cuda: cuda_bindings::CUexternalSemaphore,
}

// SAFETY: immutable handles remain owned by Exchange; all submissions are
// serialized under Gpu's queue lock and destruction follows a two-API drain.
unsafe impl Send for Timeline {}
unsafe impl Sync for Timeline {}

impl Timeline {
    fn new(gpu: Arc<Gpu>) -> Result<Self> {
        gpu.cuda_context().bind_to_thread()?;
        let mut export = vk::ExportSemaphoreCreateInfo::default()
            .handle_types(vk::ExternalSemaphoreHandleTypeFlags::OPAQUE_FD);
        let mut kind = vk::SemaphoreTypeCreateInfo::default()
            .semaphore_type(vk::SemaphoreType::TIMELINE).initial_value(0);
        let create = vk::SemaphoreCreateInfo::default().push_next(&mut export).push_next(&mut kind);
        // SAFETY: extension exportability and timeline feature were checked/enabled.
        let vulkan = unsafe { gpu.raw.create_semaphore(&create, None) }.context("creating external timeline semaphore")?;
        let mut timeline = Self { gpu, vulkan, cuda: std::ptr::null_mut() };
        let external = khr::external_semaphore_fd::Device::new(&timeline.gpu.instance, &timeline.gpu.raw);
        let fd = unsafe { external.get_semaphore_fd(&vk::SemaphoreGetFdInfoKHR::default()
            .semaphore(vulkan).handle_type(vk::ExternalSemaphoreHandleTypeFlags::OPAQUE_FD)) }
            .context("exporting timeline semaphore FD")?;
        // SAFETY: Vulkan returned a new descriptor owned by the caller.
        let fd = unsafe { OwnedFd::from_raw_fd(fd) };
        let mut import: cuda_bindings::CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC = unsafe { std::mem::zeroed() };
        // OPAQUE_FD describes the Vulkan transport. CUDA's import enum must be
        // TIMELINE_SEMAPHORE_FD, not its binary OPAQUE_FD semaphore enum.
        import.type_ = cuda_bindings::CUexternalSemaphoreHandleType_enum_CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_TIMELINE_SEMAPHORE_FD;
        import.handle.fd = fd.as_raw_fd();
        let mut imported = std::ptr::null_mut();
        cuda_result(unsafe { cuda_bindings::cuImportExternalSemaphore(&mut imported, &import) },
            "cuImportExternalSemaphore(TIMELINE_SEMAPHORE_FD)")?;
        // CUDA consumed it. Import failure instead lets OwnedFd close it.
        let _ = fd.into_raw_fd();
        timeline.cuda = imported;
        Ok(timeline)
    }

    fn cuda_wait(&self, value: u64) -> Result<()> {
        let mut parameters: cuda_bindings::CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS = unsafe { std::mem::zeroed() };
        parameters.params.fence.value = value;
        // SAFETY: owning context is current; one valid semaphore/parameter pair;
        // the Vulkan signal was submitted before enqueueing this stream wait.
        cuda_result(unsafe { cuda_bindings::cuWaitExternalSemaphoresAsync(
            &self.cuda, &parameters, 1, self.gpu.cuda_stream().cu_stream(),
        ) }, "cuWaitExternalSemaphoresAsync")
    }

    fn cuda_signal(&self, value: u64) -> Result<()> {
        let mut parameters: cuda_bindings::CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS = unsafe { std::mem::zeroed() };
        parameters.params.fence.value = value;
        // SAFETY: signal follows the wait and caller's work on this same stream.
        cuda_result(unsafe { cuda_bindings::cuSignalExternalSemaphoresAsync(
            &self.cuda, &parameters, 1, self.gpu.cuda_stream().cu_stream(),
        ) }, "cuSignalExternalSemaphoresAsync")
    }
}

impl Drop for Timeline {
    fn drop(&mut self) {
        let _ = self.gpu.cuda_context().bind_to_thread();
        // SAFETY: constructor failure has no submissions; otherwise Exchange
        // drained both APIs before dropping this field.
        unsafe {
            if !self.cuda.is_null() { let _ = cuda_bindings::cuDestroyExternalSemaphore(self.cuda); }
            self.gpu.raw.destroy_semaphore(self.vulkan, None);
        }
    }
}

pub(crate) struct CommandPool { device: ash::Device, pool: vk::CommandPool }

impl CommandPool {
    pub(crate) fn new(gpu: &Gpu) -> Result<Self> {
        let pool = unsafe { gpu.raw.create_command_pool(&vk::CommandPoolCreateInfo::default()
            .queue_family_index(gpu.family), None) }.context("creating interop command pool")?;
        Ok(Self { device: gpu.raw.clone(), pool })
    }
    pub(crate) fn allocate(&self, count: u32) -> Result<Vec<vk::CommandBuffer>> {
        unsafe { self.device.allocate_command_buffers(&vk::CommandBufferAllocateInfo::default()
            .command_pool(self.pool).level(vk::CommandBufferLevel::PRIMARY).command_buffer_count(count)) }
            .context("allocating reusable interop command buffers")
    }
}

impl Drop for CommandPool {
    fn drop(&mut self) {
        // SAFETY: initialization/Exchange owns the pool and drains before dropping.
        unsafe { self.device.destroy_command_pool(self.pool, None) };
    }
}

fn record_ownership(gpu: &Gpu, command: vk::CommandBuffer, buffers: &[Arc<SharedBuffer>], acquire: bool) -> Result<()> {
    let barriers: Vec<_> = buffers.iter().map(|buffer| {
        let (source, destination, source_access, destination_access) = if acquire {
            (vk::QUEUE_FAMILY_EXTERNAL, gpu.family, vk::AccessFlags::empty(),
                vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE)
        } else {
            (gpu.family, vk::QUEUE_FAMILY_EXTERNAL,
                vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE, vk::AccessFlags::empty())
        };
        vk::BufferMemoryBarrier::default().buffer(buffer.allocation.raw)
            .offset(0).size(buffer.len_bytes())
            .src_queue_family_index(source).dst_queue_family_index(destination)
            .src_access_mask(source_access).dst_access_mask(destination_access)
    }).collect();
    let (source_stage, destination_stage) = if acquire {
        (vk::PipelineStageFlags::TOP_OF_PIPE, vk::PipelineStageFlags::ALL_COMMANDS)
    } else {
        (vk::PipelineStageFlags::ALL_COMMANDS, vk::PipelineStageFlags::BOTTOM_OF_PIPE)
    };
    // SAFETY: fresh command buffer from the correct family; buffers retained for
    // its full lifetime; immutable contents permit simultaneous pending reuse.
    unsafe {
        gpu.raw.begin_command_buffer(command, &vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::SIMULTANEOUS_USE))?;
        gpu.raw.cmd_pipeline_barrier(command, source_stage, destination_stage,
            vk::DependencyFlags::empty(), &[], &barriers, &[]);
        gpu.raw.end_command_buffer(command)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alternating_values_are_strictly_monotonic() {
        let mut values = Values::default();
        for cycle in 0..1024 {
            let (release, returned) = values.next().unwrap();
            assert_eq!(release, cycle * 2 + 1);
            assert_eq!(returned, release + 1);
            values.completed(returned);
        }
    }

    #[test]
    fn overflow_is_rejected_before_scheduling() {
        let values = Values { last_returned: u64::MAX - 1 };
        assert!(values.next().is_err());
    }
}
