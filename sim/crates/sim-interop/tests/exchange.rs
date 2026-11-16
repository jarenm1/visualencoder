//! Opt-in NVIDIA GPU tests. Only the final test oracle uses a staging readback;
//! every frame handoff and both directions of shared-buffer access remain D2D.

use std::sync::Arc;

use anyhow::Result;
use sim_interop::{Exchange, Gpu, SharedBuffer};

fn driver(code: cuda_bindings::CUresult) -> Result<()> {
    anyhow::ensure!(code == cuda_bindings::cudaError_enum_CUDA_SUCCESS, "CUDA status {code}");
    Ok(())
}

#[test]
#[ignore = "requires Linux NVIDIA Vulkan/CUDA external FD and timeline semaphore support"]
fn repeated_bidirectional_handoffs_preserve_contents() -> Result<()> {
    let gpu = Gpu::new(0)?;
    let bytes = 4096;
    let state = SharedBuffer::new(gpu.clone(), bytes, wgpu::BufferUsages::STORAGE, "test state")?;
    let rendered = SharedBuffer::new(gpu.clone(), bytes, wgpu::BufferUsages::STORAGE, "test rendered")?;
    let consumed = SharedBuffer::new(gpu.clone(), bytes, wgpu::BufferUsages::STORAGE, "test consumed")?;
    let mut exchange = Exchange::new(gpu.clone(), vec![state.clone(), rendered.clone(), consumed.clone()])?;
    for generation in 1..=128u32 {
        let mut encoder = gpu.device().create_command_encoder(&Default::default());
        // Vulkan observes the previous CUDA-written generation and copies it to
        // simulated renderer output without touching host-visible memory.
        encoder.copy_buffer_to_buffer(state.buffer(), 0, rendered.buffer(), 0, bytes);
        // SAFETY: all accesses are on the interop stream and the owners outlive
        // the Exchange. No independent queue touches these three buffers.
        unsafe {
            exchange.cycle(encoder, || {
                driver(cuda_bindings::cuMemcpyDtoDAsync_v2(
                    consumed.device_ptr(), rendered.device_ptr(), bytes as usize,
                    gpu.cuda_stream().cu_stream(),
                ))?;
                driver(cuda_bindings::cuMemsetD32Async(
                    state.device_ptr(), generation, bytes as usize / 4,
                    gpu.cuda_stream().cu_stream(),
                ))
            })?;
        }
    }
    // One final scheduled Vulkan read verifies CUDA→Vulkan visibility, too.
    let readback = gpu.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("test-only final oracle"), size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = gpu.device().create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(consumed.buffer(), 0, &readback, 0, bytes);
    unsafe { exchange.cycle(encoder, || Ok(()))?; }
    gpu.wait_idle()?;
    let (send, receive) = std::sync::mpsc::channel();
    readback.slice(..).map_async(wgpu::MapMode::Read, move |result| { send.send(result).unwrap(); });
    gpu.device().poll(wgpu::PollType::wait_indefinitely())?;
    receive.recv()??;
    {
        let view = readback.slice(..).get_mapped_range()?;
        for word in view.chunks_exact(4) {
            assert_eq!(u32::from_ne_bytes(word.try_into().unwrap()), 127);
        }
    }
    readback.unmap();
    Ok(())
}

#[test]
#[ignore = "requires Linux NVIDIA GPU"]
fn buffer_enrollment_is_exclusive_and_released_after_drain() -> Result<()> {
    let gpu = Gpu::new(0)?;
    let buffer = SharedBuffer::new(gpu.clone(), 256, wgpu::BufferUsages::STORAGE, "enrollment")?;
    assert!(Exchange::new(gpu.clone(), vec![buffer.clone(), buffer.clone()]).is_err());
    let exchange = Exchange::new(gpu.clone(), vec![buffer.clone()])?;
    assert!(Exchange::new(gpu.clone(), vec![buffer.clone()]).is_err());
    drop(exchange);
    let replacement = Exchange::new(Arc::clone(&gpu), vec![buffer])?;
    drop(replacement);
    Ok(())
}
