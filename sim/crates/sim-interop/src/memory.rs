use std::{
    os::fd::{AsRawFd, FromRawFd, IntoRawFd, OwnedFd},
    sync::{Arc, atomic::{AtomicBool, Ordering}},
};

use anyhow::{Context as _, Result, ensure};
use ash::{khr, vk};
use wgpu_hal::api::Vulkan;

use crate::{Gpu, cuda_result};

/// Dedicated device-local Vulkan allocation, mapped once into CUDA's address space.
/// The returned CUDA pointer is borrowed; never pass it to a CUDA deallocator.
/// The HAL destroy callback retains the allocation even if a wgpu Buffer clone
/// outlives this wrapper. Both APIs see exactly the same physical memory.
pub struct SharedBuffer {
    buffer: wgpu::Buffer,
    pub(crate) allocation: Arc<Allocation>,
    pub(crate) enrolled: AtomicBool,
    pub(crate) unusable: AtomicBool,
    bytes: u64,
    // Last: the local allocation reference must drop while Gpu is still live.
    pub(crate) gpu: Arc<Gpu>,
}

pub(crate) struct Allocation {
    device: ash::Device,
    cuda_context: Arc<cuda_core::Context>,
    pub(crate) raw: vk::Buffer,
    memory: vk::DeviceMemory,
    external: cuda_bindings::CUexternalMemory,
    pointer: cuda_bindings::CUdeviceptr,
}

// SAFETY: all CUDA calls bind the owning context to the calling thread. Handles
// are immutable after construction; Exchange serializes GPU access and retains
// these owners until both APIs finish. Destruction has exclusive ownership.
unsafe impl Send for Allocation {}
unsafe impl Sync for Allocation {}

impl SharedBuffer {
    /// Allocate and initialize to zero entirely on the GPU. Startup synchronization
    /// is intentional: create_buffer_from_hal requires initialized contents.
    /// Sizes must be nonzero multiples of four. CPU mapping usages are forbidden.
    /// COPY_SRC/COPY_DST are added for initialization and the handoff state boundary.
    pub fn new(gpu: Arc<Gpu>, bytes: u64, usage: wgpu::BufferUsages, label: &str) -> Result<Arc<Self>> {
        ensure!(bytes != 0 && bytes % 4 == 0, "shared buffer size must be a nonzero multiple of four");
        ensure!(bytes <= gpu.device().limits().max_buffer_size, "shared buffer exceeds wgpu max_buffer_size");
        let usage = usage | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;
        let raw_usage = buffer_usage(usage)?;
        verify_external_buffer(&gpu, raw_usage)?;
        gpu.cuda_context().bind_to_thread()?;
        let mut external_info = vk::ExternalMemoryBufferCreateInfo::default()
            .handle_types(vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD);
        let create = vk::BufferCreateInfo::default()
            .size(bytes)
            .usage(raw_usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .push_next(&mut external_info);
        // SAFETY: validated usage and external handle support on this device.
        let raw = unsafe { gpu.raw.create_buffer(&create, None) }.context("creating exportable Vulkan buffer")?;
        let mut allocation = Allocation {
            device: gpu.raw.clone(), cuda_context: gpu.cuda_context().clone(),
            raw, memory: vk::DeviceMemory::null(),
            external: std::ptr::null_mut(), pointer: 0,
        };
        let mut dedicated_requirements = vk::MemoryDedicatedRequirements::default();
        let mut requirements = vk::MemoryRequirements2::default().push_next(&mut dedicated_requirements);
        // SAFETY: raw is a live buffer on gpu.raw, output pNext storage is valid.
        unsafe {
            gpu.raw.get_buffer_memory_requirements2(
                &vk::BufferMemoryRequirementsInfo2::default().buffer(raw), &mut requirements,
            );
        }
        let requirements = requirements.memory_requirements;
        let properties = unsafe { gpu.instance.get_physical_device_memory_properties(gpu.physical) };
        let memory_type = properties.memory_types_as_slice().iter().enumerate().find(|(index, memory)| {
            requirements.memory_type_bits & (1u32 << index) != 0
                && memory.property_flags.contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
                && !memory.property_flags.intersects(vk::MemoryPropertyFlags::PROTECTED | vk::MemoryPropertyFlags::LAZILY_ALLOCATED)
        }).context("exportable buffer has no device-local Vulkan memory type")?.0 as u32;
        // Always dedicate, including when Vulkan merely prefers it. CUDA's
        // DEDICATED flag therefore exactly matches the Vulkan allocation chain.
        let mut dedicated = vk::MemoryDedicatedAllocateInfo::default().buffer(raw);
        let mut export = vk::ExportMemoryAllocateInfo::default()
            .handle_types(vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD);
        let allocate = vk::MemoryAllocateInfo::default()
            .allocation_size(requirements.size)
            .memory_type_index(memory_type)
            .push_next(&mut dedicated)
            .push_next(&mut export);
        allocation.memory = unsafe { gpu.raw.allocate_memory(&allocate, None) }
            .context("allocating dedicated exportable Vulkan memory")?;
        unsafe { gpu.raw.bind_buffer_memory(raw, allocation.memory, 0) }.context("binding exportable Vulkan memory")?;
        let external_memory = khr::external_memory_fd::Device::new(&gpu.instance, &gpu.raw);
        let fd = unsafe {
            external_memory.get_memory_fd(&vk::MemoryGetFdInfoKHR::default()
                .memory(allocation.memory)
                .handle_type(vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD))
        }.context("exporting Vulkan memory FD")?;
        // SAFETY: successful vkGetMemoryFdKHR returns a newly owned descriptor.
        let fd = unsafe { OwnedFd::from_raw_fd(fd) };
        // Zero initializes all CUDA reserved fields, including union padding.
        let mut import: cuda_bindings::CUDA_EXTERNAL_MEMORY_HANDLE_DESC = unsafe { std::mem::zeroed() };
        import.type_ = cuda_bindings::CUexternalMemoryHandleType_enum_CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD;
        import.handle.fd = fd.as_raw_fd();
        import.size = requirements.size;
        import.flags = cuda_bindings::CUDA_EXTERNAL_MEMORY_DEDICATED;
        let mut imported = std::ptr::null_mut();
        // SAFETY: FD references this dedicated Vulkan allocation, and size is the
        // actual allocation size, not merely the buffer's requested logical size.
        cuda_result(unsafe { cuda_bindings::cuImportExternalMemory(&mut imported, &import) },
            "cuImportExternalMemory")?;
        // CUDA consumes FD only on success; OwnedFd closes it on every error path.
        let _ = fd.into_raw_fd();
        allocation.external = imported;
        let mut mapping: cuda_bindings::CUDA_EXTERNAL_MEMORY_BUFFER_DESC = unsafe { std::mem::zeroed() };
        mapping.offset = 0;
        mapping.size = bytes;
        let mut pointer = 0;
        // SAFETY: one nonoverlapping mapping at the Vulkan buffer's bound offset.
        cuda_result(unsafe {
            cuda_bindings::cuExternalMemoryGetMappedBuffer(&mut pointer, allocation.external, &mapping)
        }, "cuExternalMemoryGetMappedBuffer")?;
        allocation.pointer = pointer;
        initialize_zero(&gpu, raw, bytes)?;
        let allocation = Arc::new(allocation);
        let owner = allocation.clone();
        // SAFETY: callback retains both handles and frees exactly once when HAL
        // has finished using the buffer. HAL itself destroys neither handle.
        let hal = unsafe {
            wgpu_hal::vulkan::Buffer::from_raw_externally_owned(raw, Box::new(move || drop(owner)))
        };
        // SAFETY: usage/size match the allocation, initial GPU fill has completed,
        // and CPU mapping is forbidden. Core's Buffer retains its device while
        // invoking the callback; no callback→Gpu→wgpu resource ownership cycle.
        let buffer = unsafe {
            gpu.device().create_buffer_from_hal::<Vulkan>(hal, &wgpu::BufferDescriptor {
                label: Some(label), size: bytes, usage, mapped_at_creation: false,
            })
        };
        Ok(Arc::new(Self {
            buffer, allocation, bytes, gpu,
            enrolled: AtomicBool::new(false), unusable: AtomicBool::new(false),
        }))
    }

    pub fn buffer(&self) -> &wgpu::Buffer { &self.buffer }
    pub fn device_ptr(&self) -> u64 { self.allocation.pointer }
    pub fn len_bytes(&self) -> u64 { self.bytes }

    pub(crate) fn poison(&self) { self.unusable.store(true, Ordering::Release); }
}

impl Drop for Allocation {
    fn drop(&mut self) {
        // Exchange retains the SharedBuffer and drains both APIs before release.
        // HAL invokes its callback only after Vulkan is done. No waits here: this
        // callback may itself run inside a queue submission's resource cleanup.
        let _ = self.cuda_context.bind_to_thread();
        unsafe {
            // CUDA mapping is independent of the imported object's lifetime.
            // cuMemFree frees only this mapping, not Vulkan's allocation.
            if self.pointer != 0 { let _ = cuda_bindings::cuMemFree_v2(self.pointer); }
            if !self.external.is_null() { let _ = cuda_bindings::cuDestroyExternalMemory(self.external); }
            self.device.destroy_buffer(self.raw, None);
            if self.memory != vk::DeviceMemory::null() { self.device.free_memory(self.memory, None); }
        }
    }
}

fn buffer_usage(usage: wgpu::BufferUsages) -> Result<vk::BufferUsageFlags> {
    let supported = wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST
        | wgpu::BufferUsages::INDEX | wgpu::BufferUsages::VERTEX
        | wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::STORAGE
        | wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::QUERY_RESOLVE;
    ensure!(supported.contains(usage), "shared buffers support only unmapped copy, index, vertex, uniform, storage, indirect and query-resolve usages");
    let mut flags = vk::BufferUsageFlags::empty();
    for (wgpu, vulkan) in [
        (wgpu::BufferUsages::COPY_SRC, vk::BufferUsageFlags::TRANSFER_SRC),
        (wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::QUERY_RESOLVE, vk::BufferUsageFlags::TRANSFER_DST),
        (wgpu::BufferUsages::INDEX, vk::BufferUsageFlags::INDEX_BUFFER),
        (wgpu::BufferUsages::VERTEX, vk::BufferUsageFlags::VERTEX_BUFFER),
        (wgpu::BufferUsages::UNIFORM, vk::BufferUsageFlags::UNIFORM_BUFFER),
        (wgpu::BufferUsages::STORAGE, vk::BufferUsageFlags::STORAGE_BUFFER),
        (wgpu::BufferUsages::INDIRECT, vk::BufferUsageFlags::INDIRECT_BUFFER),
    ] {
        if usage.intersects(wgpu) { flags |= vulkan; }
    }
    Ok(flags)
}

fn verify_external_buffer(gpu: &Gpu, usage: vk::BufferUsageFlags) -> Result<()> {
    let info = vk::PhysicalDeviceExternalBufferInfo::default()
        .usage(usage).handle_type(vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD);
    let mut properties = vk::ExternalBufferProperties::default();
    unsafe { gpu.instance.get_physical_device_external_buffer_properties(gpu.physical, &info, &mut properties) };
    let properties = properties.external_memory_properties;
    ensure!(properties.external_memory_features.contains(vk::ExternalMemoryFeatureFlags::EXPORTABLE),
        "Vulkan buffer usage {usage:?} cannot be exported as OPAQUE_FD");
    ensure!(properties.compatible_handle_types.contains(vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD),
        "Vulkan buffer OPAQUE_FD handle type is incompatible");
    Ok(())
}

fn initialize_zero(gpu: &Gpu, buffer: vk::Buffer, bytes: u64) -> Result<()> {
    let _guard = gpu.lock_queue();
    let pool = crate::sync::CommandPool::new(gpu)?;
    let command = pool.allocate(1)?[0];
    unsafe {
        gpu.raw.begin_command_buffer(command, &vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT))?;
        gpu.raw.cmd_fill_buffer(command, buffer, 0, bytes, 0);
        let barrier = vk::BufferMemoryBarrier::default()
            .buffer(buffer).offset(0).size(bytes)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE);
        gpu.raw.cmd_pipeline_barrier(command, vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::ALL_COMMANDS, vk::DependencyFlags::empty(), &[], &[barrier], &[]);
        gpu.raw.end_command_buffer(command)?;
        let commands = [command];
        gpu.raw.queue_submit(gpu.raw_queue, &[vk::SubmitInfo::default().command_buffers(&commands)], vk::Fence::null())?;
        // Startup only; guarantees initialized contents before importing into wgpu.
        gpu.raw.queue_wait_idle(gpu.raw_queue)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mapping_usages_are_rejected() {
        assert!(buffer_usage(wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST).is_err());
        assert!(buffer_usage(wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC).is_err());
    }

    #[test]
    fn storage_and_copy_usages_match_vulkan() {
        assert_eq!(buffer_usage(wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST).unwrap(), vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST);
    }
}
