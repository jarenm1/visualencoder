//! Optional per-stage step profiler. Off by default; while on, every step blocks until its GPU
//! work finishes so stage times can be read back, which slows the step itself.
//!
//! The CUDA <-> Vulkan exchange is serialized: CUDA waits for each Vulkan release before its
//! callback runs. CUDA events recorded at the start of a callback therefore mark the end of the
//! preceding Vulkan work, and one CUDA timeline covers the whole step:
//!
//! ```text
//! E0 control E1 finish E2 observe E3 | Vulkan render + handoff | E4 appearance E5 labels E6 coverage E7
//! ```
//!
//! Vulkan pass timestamps (TIMESTAMP_QUERY) split the render into cull, camera passes and pack.
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context as _, Result, ensure};
use cuda_core::{CudaContext, CudaEvent, CudaStream};
use sim_render::RENDER_TIMESTAMPS;

/// CUDA events per step (see the module timeline).
pub const MARKS: usize = 8;

/// Stage means reported by `Profiler::report`, in this order (milliseconds unless noted).
pub const STAGES: [&str; 17] = [
    "steps",                 // profiled steps in the window (count)
    "host_enqueue_ms",       // CPU time inside Env::step (kernel launches, encoding, submissions)
    "wall_ms",               // step call start to its GPU work complete
    "gpu_step_ms",           // E0 -> E7
    "control_ms",            // E0 -> E1: physics, collision sweep
    "finish_ms",             // E1 -> E2: task outcome, reward, autoreset
    "observe_ms",            // E2 -> E3: proprio, privileged, cameras
    "render_handoff_ms",     // E3 -> E4: Vulkan render plus both semaphore handoffs
    "appearance_ms",         // E4 -> E5: appearance randomization into the observation
    "labels_ms",             // E5 -> E6: per-patch depth / object labels
    "coverage_ms",           // E6 -> E7: observed surface tiles (search reward)
    "vk_cull_ms",            // cull compute pass
    "vk_camera_passes_ms",   // first camera pass begin -> last camera pass end
    "vk_pack_ms",            // texture-to-buffer pack
    "vk_total_ms",           // cull begin -> pack end
    "vk_gaps_ms",            // vk_total minus the three passes (pass transitions, barriers)
    "handoff_ms",            // render_handoff minus vk_total (semaphores, submission latency)
];

pub struct Profiler {
    marks: Vec<CudaEvent>,
    queries: Option<wgpu::QuerySet>,
    resolve: Option<wgpu::Buffer>,
    readback: Option<wgpu::Buffer>,
    period_ns: f64,
    started: Option<Instant>,
    enqueue_ms: f64,
    sums: [f64; STAGES.len()],
}

impl Profiler {
    pub fn new(device: &wgpu::Device, context: &Arc<CudaContext>, period_ns: f32) -> Result<Self> {
        let marks = (0..MARKS).map(|_| context.new_event().context("creating profiler event"))
            .collect::<Result<Vec<_>>>()?;
        let timed = device.features().contains(wgpu::Features::TIMESTAMP_QUERY);
        let bytes = u64::from(RENDER_TIMESTAMPS) * 8;
        let queries = timed.then(|| device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("step profiler render timestamps"), ty: wgpu::QueryType::Timestamp, count: RENDER_TIMESTAMPS,
        }));
        let buffer = |label, usage| device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label), size: bytes, usage, mapped_at_creation: false,
        });
        let resolve = timed.then(|| buffer("profiler timestamp resolve",
            wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC));
        let readback = timed.then(|| buffer("profiler timestamp readback",
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST));
        Ok(Self { marks, queries, resolve, readback, period_ns: f64::from(period_ns), started: None,
            enqueue_ms: 0.0, sums: [0.0; STAGES.len()] })
    }

    pub fn queries(&self) -> Option<&wgpu::QuerySet> { self.queries.as_ref() }

    pub fn begin(&mut self) { self.started = Some(Instant::now()); }

    /// Records timeline mark `index` on the env stream.
    pub fn mark(&self, index: usize, stream: &CudaStream) -> Result<()> {
        self.marks[index].record(stream).context("recording profiler event")
    }

    /// Appends the timestamp resolve and readback copy to the render encoder.
    pub fn resolve(&self, encoder: &mut wgpu::CommandEncoder) {
        if let (Some(queries), Some(resolve), Some(readback)) = (&self.queries, &self.resolve, &self.readback) {
            encoder.resolve_query_set(queries, 0..RENDER_TIMESTAMPS, resolve, 0);
            encoder.copy_buffer_to_buffer(resolve, 0, readback, 0, resolve.size());
        }
    }

    /// Host enqueue finished; waits for the step's GPU work and accumulates its stage times.
    pub fn end(&mut self, device: &wgpu::Device) -> Result<()> {
        let started = self.started.take().context("profiler end without begin")?;
        self.enqueue_ms = started.elapsed().as_secs_f64() * 1e3;
        self.marks[MARKS - 1].synchronize().context("waiting for the profiled step")?;
        let wall = started.elapsed().as_secs_f64() * 1e3;
        let span = |a: usize, b: usize| -> Result<f64> {
            Ok(f64::from(self.marks[a].elapsed_time(&self.marks[b]).context("reading profiler events")?))
        };
        let cuda = [span(0, 7)?, span(0, 1)?, span(1, 2)?, span(2, 3)?, span(3, 4)?, span(4, 5)?, span(5, 6)?,
            span(6, 7)?];
        let vk = self.read_timestamps(device)?;
        let mut sample = [0.0; STAGES.len()];
        sample[0] = 1.0;
        sample[1] = self.enqueue_ms;
        sample[2] = wall;
        sample[3..11].copy_from_slice(&cuda);
        if let Some(t) = vk {
            let ms = |a: usize, b: usize| (t[b].saturating_sub(t[a])) as f64 * self.period_ns * 1e-6;
            let (cull, cameras, pack, total) = (ms(0, 1), ms(2, 3), ms(4, 5), ms(0, 5));
            sample[11..16].copy_from_slice(&[cull, cameras, pack, total, total - cull - cameras - pack]);
            sample[16] = cuda[4] - total;
        }
        for (sum, value) in self.sums.iter_mut().zip(sample) { *sum += value; }
        Ok(())
    }

    fn read_timestamps(&self, device: &wgpu::Device) -> Result<Option<[u64; RENDER_TIMESTAMPS as usize]>> {
        let Some(readback) = &self.readback else { return Ok(None) };
        let slice = readback.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::PollType::wait_indefinitely()).context("waiting for profiler timestamps")?;
        let values = {
            let bytes = slice.get_mapped_range().map_err(|error| anyhow::anyhow!("mapping timestamps: {error:?}"))?;
            let mut values = [0u64; RENDER_TIMESTAMPS as usize];
            values.copy_from_slice(bytemuck::cast_slice(&bytes[..]));
            values
        };
        readback.unmap();
        Ok(Some(values))
    }

    /// Writes `[steps, stage means...]` (STAGES order) into `out` and starts a new window.
    /// Vulkan stages read zero when the device lacks TIMESTAMP_QUERY.
    pub fn report(&mut self, out: &mut [f64]) -> Result<()> {
        ensure!(out.len() >= STAGES.len(), "profile output needs {} values", STAGES.len());
        let steps = self.sums[0].max(1.0);
        out[0] = self.sums[0];
        for (slot, sum) in out[1..STAGES.len()].iter_mut().zip(&self.sums[1..]) { *slot = sum / steps; }
        self.sums = [0.0; STAGES.len()];
        Ok(())
    }
}

/// Scene and culling statistics of the last render, from the culled indirect commands:
/// `[draws per camera, scene triangles, cameras, mean visible draws per camera,
///   mean visible triangles per camera, max visible triangles in one camera,
///   indirect commands issued per frame]`.
pub const RENDER_STATS: usize = 7;

pub fn render_stats(commands: &[u32], draw_count: u32, cameras: u32) -> [f64; RENDER_STATS] {
    let draws = draw_count as usize;
    let mut scene_triangles = 0u64;
    let (mut visible_draws, mut visible_triangles, mut max_triangles) = (0u64, 0u64, 0u64);
    for camera in 0..cameras as usize {
        let mut triangles = 0u64;
        for draw in 0..draws {
            let record = &commands[(camera * draws + draw) * 5..][..5];
            if camera == 0 { scene_triangles += u64::from(record[0] / 3); }
            if record[1] != 0 {
                visible_draws += 1;
                triangles += u64::from(record[0] / 3);
            }
        }
        visible_triangles += triangles;
        max_triangles = max_triangles.max(triangles);
    }
    let per_camera = f64::from(cameras.max(1));
    [f64::from(draw_count), scene_triangles as f64, f64::from(cameras), visible_draws as f64 / per_camera,
        visible_triangles as f64 / per_camera, max_triangles as f64, f64::from(draw_count) * f64::from(cameras)]
}
