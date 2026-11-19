use std::sync::{Arc, Mutex, MutexGuard};

use anyhow::{Context as _, Result, ensure};
use ash::{khr, vk};
use wgpu_hal::api::Vulkan;

use crate::cuda_result;

/// One UUID-matched Vulkan device, queue, CUDA primary context and CUDA stream.
/// All queue submissions made by this crate share one external-synchronization lock.
pub struct Gpu {
    pub(crate) raw: ash::Device,
    pub(crate) instance: ash::Instance,
    pub(crate) physical: vk::PhysicalDevice,
    pub(crate) family: u32,
    pub(crate) raw_queue: vk::Queue,
    pub(crate) queue_lock: Mutex<()>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    cuda_stream: Arc<cuda_core::Stream>,
    cuda_context: Arc<cuda_core::Context>,
    uuid: [u8; 16],
    // HAL also retains the instance; keep its public owner explicit.
    _instance: wgpu::Instance,
}

impl Gpu {
    /// Select CUDA ordinal exactly, then find the Vulkan adapter with the same UUID.
    /// Unsupported external memory/timeline semaphore configurations are errors.
    pub fn new(cuda_ordinal: i32) -> Result<Arc<Self>> {
        let ordinal = usize::try_from(cuda_ordinal).context("CUDA ordinal must be nonnegative")?;
        let cuda_context = cuda_core::Context::new(ordinal).context("creating CUDA primary context")?;
        let cuda_stream = cuda_context.new_stream().context("creating CUDA interop stream")?;
        let mut cuda_uuid = std::mem::MaybeUninit::uninit();
        // SAFETY: the context owns this CUdevice and the output points to CUuuid storage.
        unsafe {
            cuda_result(
                cuda_bindings::cuDeviceGetUuid_v2(cuda_uuid.as_mut_ptr(), cuda_context.cu_device()),
                "cuDeviceGetUuid_v2",
            )?;
        }
        // SAFETY: CUDA wrote the entire UUID on success above.
        let uuid = unsafe { cuda_uuid.assume_init() }.bytes.map(|byte| byte as u8);
        let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
        descriptor.backends = wgpu::Backends::VULKAN;
        // Indirect commands come only from the renderer's cull pass, which copies CPU-validated
        // index ranges and writes instance counts of 0 or 1; wgpu's per-call indirect validation
        // (a validation dispatch plus host bookkeeping for every indirect draw) is redundant.
        descriptor.flags = wgpu::InstanceFlags::from_build_config() - wgpu::InstanceFlags::VALIDATION_INDIRECT_CALL;
        let instance = wgpu::Instance::new(descriptor);
        let adapters = pollster::block_on(instance.enumerate_adapters(wgpu::Backends::VULKAN));
        let mut matched = None;
        for adapter in adapters {
            // SAFETY: read-only physical-device queries; no handle is destroyed.
            let hal = unsafe { adapter.as_hal::<Vulkan>() }.context("adapter is not Vulkan")?;
            let raw_instance = hal.shared_instance().raw_instance();
            let mut id = vk::PhysicalDeviceIDProperties::default();
            let mut properties = vk::PhysicalDeviceProperties2::default().push_next(&mut id);
            // SAFETY: physical device comes from this instance; pNext storage is live.
            unsafe { raw_instance.get_physical_device_properties2(hal.raw_physical_device(), &mut properties) };
            let vendor = properties.properties.vendor_id;
            if id.device_uuid == uuid {
                ensure!(vendor == 0x10de, "UUID-matched Vulkan device is not NVIDIA");
                drop(hal);
                matched = Some(adapter);
                break;
            }
        }
        let adapter = matched.with_context(|| {
            format!("no NVIDIA Vulkan adapter matches CUDA ordinal {cuda_ordinal} UUID {uuid:02x?}; no fallback is permitted")
        })?;
        let limits = adapter.limits();
        // GPU-generated indexed draws address persistent instance ranges directly; the cull pass
        // compacts each camera's visible draws and supplies their count (drawIndirectCount).
        let required = wgpu::Features::INDIRECT_FIRST_INSTANCE | wgpu::Features::MULTI_DRAW_INDIRECT_COUNT;
        ensure!(adapter.features().contains(required),
            "matched Vulkan adapter lacks indirect first-instance or indirect-count draws");
        // Pass timestamps are optional: they only serve the native step profiler.
        let features = required | (adapter.features() & wgpu::Features::TIMESTAMP_QUERY);
        ensure!(
            adapter.get_downlevel_capabilities().flags.contains(wgpu::DownlevelFlags::INDIRECT_EXECUTION),
            "matched Vulkan adapter lacks indirect draw execution",
        );
        let memory_hints = wgpu::MemoryHints::MemoryUsage;
        // SAFETY: handles are only queried/used to create their own logical device.
        let hal = unsafe { adapter.as_hal::<Vulkan>() }.context("matched adapter is not Vulkan")?;
        let raw_instance = hal.shared_instance().raw_instance().clone();
        let physical = hal.raw_physical_device();
        let properties = unsafe { raw_instance.get_physical_device_properties(physical) };
        ensure!(properties.api_version >= vk::API_VERSION_1_2, "interop requires Vulkan 1.2 or newer");
        let capabilities = hal.physical_device_capabilities();
        let required_external = [
            khr::external_memory_fd::NAME,
            khr::external_semaphore_fd::NAME,
            khr::timeline_semaphore::NAME,
        ];
        let mut extensions = hal.required_device_extensions(features);
        for extension in required_external {
            ensure!(capabilities.supports_extension(extension), "required Vulkan device extension {extension:?} is unavailable");
            if !extensions.contains(&extension) {
                extensions.push(extension);
            }
        }
        let mut timeline = vk::PhysicalDeviceTimelineSemaphoreFeatures::default();
        let mut supported = vk::PhysicalDeviceFeatures2::default().push_next(&mut timeline);
        // SAFETY: all queried structures and the physical device belong to this instance.
        unsafe { raw_instance.get_physical_device_features2(physical, &mut supported) };
        ensure!(timeline.timeline_semaphore == vk::TRUE, "Vulkan timelineSemaphore feature is unavailable");
        verify_external_timeline(&raw_instance, physical)?;
        let families = unsafe { raw_instance.get_physical_device_queue_family_properties(physical) };
        let family = families.iter().position(|family| {
            family.queue_count != 0
                && family.queue_flags.contains(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE)
        }).context("matched Vulkan adapter has no graphics+compute queue family")? as u32;
        let priorities = [1.0];
        let queues = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(family)
            .queue_priorities(&priorities)];
        let extension_names: Vec<_> = extensions.iter().map(|extension| extension.as_ptr()).collect();
        // HAL's feature builder enables its timeline feature, already checked above.
        // Do not append a duplicate timeline feature structure to the pNext chain.
        let mut enabled_features = hal.physical_device_features(&extensions, features);
        let create = enabled_features.add_to_device_create(
            vk::DeviceCreateInfo::default()
                .queue_create_infos(&queues)
                .enabled_extension_names(&extension_names),
        );
        // SAFETY: extensions, features and queue family were queried from this device.
        let raw = unsafe { raw_instance.create_device(physical, &create, None) }
            .context("creating external-memory Vulkan device")?;
        let raw_queue = unsafe { raw.get_device_queue(family, 0) };
        // A callback owns the device even if HAL's early descriptor-layout
        // creation fails before it has constructed DeviceShared. Keep the instance
        // alive until vkDestroyDevice, including if wgpu buffer clones outlive Gpu.
        let destroy_device = raw.clone();
        let instance_owner = instance.clone();
        let destroy: wgpu_hal::DropCallback = Box::new(move || {
            // SAFETY: HAL calls this once, after destroying its device resources.
            unsafe { destroy_device.destroy_device(None) };
            drop(instance_owner);
        });
        // SAFETY: exactly HAL's required features/extensions and selected family;
        // destroy owns raw and runs on every HAL failure/success destruction path.
        let opened = unsafe {
            hal.device_from_raw(raw.clone(), Some(destroy), &extensions, features, &limits, &memory_hints, family, 0)
        }.context("wrapping Vulkan device in wgpu-hal")?;
        drop(hal);
        let descriptor = wgpu::DeviceDescriptor {
            label: Some("sim Vulkan/CUDA device"),
            required_features: features,
            required_limits: limits,
            memory_hints,
            ..Default::default()
        };
        // SAFETY: opened was created from this exact adapter with these features/limits.
        let (device, queue) = unsafe { adapter.create_device_from_hal::<Vulkan>(opened, &descriptor) }
            .context("wrapping Vulkan device in wgpu")?;
        Ok(Arc::new(Self {
            raw,
            instance: raw_instance,
            physical,
            family,
            raw_queue,
            queue_lock: Mutex::new(()),
            device,
            queue,
            cuda_stream,
            cuda_context,
            uuid,
            _instance: instance,
        }))
    }

    pub fn device(&self) -> &wgpu::Device { &self.device }
    pub fn cuda_context(&self) -> &Arc<cuda_core::Context> { &self.cuda_context }
    pub fn cuda_stream(&self) -> &Arc<cuda_core::Stream> { &self.cuda_stream }
    pub fn device_uuid(&self) -> [u8; 16] { self.uuid }

    /// Ordinary nonshared-buffer submissions, externally synchronized with interop.
    pub fn submit(&self, commands: impl IntoIterator<Item = wgpu::CommandBuffer>) -> wgpu::SubmissionIndex {
        let _guard = self.lock_queue();
        self.queue.submit(commands)
    }

    /// Upload ordinary (not exchange-owned) data, normally during startup.
    pub fn write_buffer(&self, buffer: &wgpu::Buffer, offset: u64, data: &[u8]) {
        let _guard = self.lock_queue();
        self.queue.write_buffer(buffer, offset, data);
    }

    /// Run a queue operation under the same lock used for raw Vulkan submissions.
    ///
    /// # Safety
    /// Do not clone, retain or return the queue. Do not submit exchange buffers or
    /// stage HAL semaphore operations here; use Exchange::cycle for shared work.
    /// Do not reenter this Gpu from the closure.
    pub unsafe fn with_queue<R>(&self, f: impl FnOnce(&wgpu::Queue) -> R) -> R {
        let _guard = self.lock_queue();
        f(&self.queue)
    }

    /// Explicit startup/shutdown drain. Never called by a frame handoff.
    pub fn wait_idle(&self) -> Result<()> {
        let _guard = self.lock_queue();
        self.drain_locked()
    }

    pub(crate) fn lock_queue(&self) -> MutexGuard<'_, ()> {
        self.queue_lock.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    pub(crate) fn queue_locked(&self) -> &wgpu::Queue { &self.queue }

    pub(crate) fn drain_locked(&self) -> Result<()> {
        // Submitted CUDA waits only target already-submitted Vulkan releases.
        let cuda = self.cuda_stream.synchronize().context("draining CUDA interop stream");
        // SAFETY: the queue lock excludes every queue operation exposed by this crate.
        let vulkan = unsafe { self.raw.device_wait_idle() }.context("draining Vulkan device");
        cuda.and(vulkan)
    }
}

impl Drop for Gpu {
    fn drop(&mut self) {
        // Resource owners retain Gpu, so their destruction has already completed.
        let _ = self.drain_locked();
    }
}

fn verify_external_timeline(instance: &ash::Instance, physical: vk::PhysicalDevice) -> Result<()> {
    let mut timeline = vk::SemaphoreTypeCreateInfo::default().semaphore_type(vk::SemaphoreType::TIMELINE);
    let info = vk::PhysicalDeviceExternalSemaphoreInfo::default()
        .handle_type(vk::ExternalSemaphoreHandleTypeFlags::OPAQUE_FD)
        .push_next(&mut timeline);
    let mut properties = vk::ExternalSemaphoreProperties::default();
    // SAFETY: valid physical device and Vulkan-defined timeline query chain.
    unsafe { instance.get_physical_device_external_semaphore_properties(physical, &info, &mut properties) };
    ensure!(properties.external_semaphore_features.contains(vk::ExternalSemaphoreFeatureFlags::EXPORTABLE),
        "Vulkan timeline semaphores cannot be exported as OPAQUE_FD");
    ensure!(properties.compatible_handle_types.contains(vk::ExternalSemaphoreHandleTypeFlags::OPAQUE_FD),
        "Vulkan timeline semaphore OPAQUE_FD handle type is incompatible");
    // CPU scheduling is deliberately unbounded and never polls a GPU counter to
    // throttle frames. Full-range differences make every nonoverflowing queued
    // odd/even value legal, without an implicit outstanding-frame limit.
    let mut limits = vk::PhysicalDeviceTimelineSemaphoreProperties::default();
    let mut device_properties = vk::PhysicalDeviceProperties2::default().push_next(&mut limits);
    // SAFETY: Vulkan 1.2 timeline properties are supported on the selected device.
    unsafe { instance.get_physical_device_properties2(physical, &mut device_properties) };
    ensure!(limits.max_timeline_semaphore_value_difference == u64::MAX,
        "interop requires full-range maxTimelineSemaphoreValueDifference for wait-free CPU scheduling");
    Ok(())
}
