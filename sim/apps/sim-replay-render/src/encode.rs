use std::{fs::{self, OpenOptions}, path::{Path, PathBuf}, process::{Child, ChildStdin, Command, Stdio}};

use anyhow::{Context, Result, ensure};
use sim_scene::CityScene;

use crate::{capture::Capture, proxy::Proxy, recording::{Environment, Recording, StateColumns}};

// A failed encode never replaces a completed clip. Closing/killing/reaping the
// child and removing only this export's partial file is guaranteed on all paths.
struct Encoder {
    child: Child,
    input: Option<ChildStdin>,
    partial: PathBuf,
    reaped: bool,
}

impl Encoder {
    fn start(recording: &Recording, environment: &Environment, directory: &Path) -> Result<Self> {
        let raw = directory.join(environment.video.with_extension("rgb"));
        let partial = directory.join(environment.video.with_extension(format!("partial-{}.mp4", std::process::id())));
        // Reserve our partial name without following an existing symlink or
        // destroying leftovers from another exporter.
        OpenOptions::new().write(true).create_new(true).open(&partial)
            .with_context(|| format!("reserving partial video {}", partial.display()))?;
        let frame_size = format!("{}x{}", recording.frame_width, recording.frame_height);
        let rate = recording.control_hz.to_string();
        let child = Command::new("ffmpeg")
            .args(["-hide_banner", "-loglevel", "warning", "-nostdin", "-y", "-filter_complex_threads", "1"])
            .args(["-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", &frame_size, "-framerate", &rate, "-i"])
            .arg(&raw)
            .args(["-f", "rawvideo", "-pixel_format", "rgba", "-video_size", &frame_size, "-framerate", &rate, "-i", "pipe:0"])
            .args(["-filter_complex",
                "[0:v:0]format=rgb24,setpts=PTS-STARTPTS[left];[1:v:0]format=rgb24,setpts=PTS-STARTPTS[right];[left][right]hstack=inputs=2:shortest=1,format=yuv420p[video]",
                "-map", "[video]", "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                "-threads", "2", "-g", "1", "-bf", "0", "-fps_mode", "passthrough", "-frames:v"])
            .arg(environment.frame_count.to_string())
            .args(["-movflags", "+faststart", "-f", "mp4"])
            .arg(&partial)
            .stdin(Stdio::piped()).stdout(Stdio::null()).stderr(Stdio::inherit())
            .spawn();
        let child = match child {
            Ok(child) => child,
            Err(error) => {
                let _ = fs::remove_file(&partial);
                return Err(error).context("starting ffmpeg (requires libx264 support)");
            }
        };
        let mut encoder = Self { child, input: None, partial, reaped: false };
        encoder.input = Some(encoder.child.stdin.take().context("ffmpeg did not expose stdin")?);
        Ok(encoder)
    }

    fn finish(mut self, destination: &Path) -> Result<()> {
        drop(self.input.take());
        let status = self.child.wait().context("waiting for ffmpeg")?;
        self.reaped = true;
        ensure!(status.success(), "ffmpeg failed with {status}; see its stderr above");
        ensure!(fs::metadata(&self.partial)?.len() > 0, "ffmpeg produced an empty video");
        fs::rename(&self.partial, destination)
            .with_context(|| format!("installing completed video {}", destination.display()))?;
        Ok(())
    }
}

impl Drop for Encoder {
    fn drop(&mut self) {
        drop(self.input.take());
        if !self.reaped {
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
        let _ = fs::remove_file(&self.partial);
    }
}

pub fn export(capture: &Capture, scene: &CityScene, proxy: &Proxy, columns: &StateColumns,
    recording: &Recording, directory: &Path) -> Result<()> {
    let mut completed_frames = 0usize;
    for environment in &recording.environments {
        ensure!(proxy.validate_design(environment.design),
            "environment {} design {} is not a target design of split {}", environment.id, environment.design, recording.style);
        let destination = directory.join(&environment.video);
        let mut encoder = Encoder::start(recording, environment, directory)
            .with_context(|| format!("starting environment {} export", environment.id))?;
        for (frame, state) in environment.states.iter().enumerate() {
            let input = encoder.input.as_mut().context("ffmpeg input already closed")?;
            if let Err(error) = capture.write_frame(scene, proxy, &columns.pose(state), environment.design, input) {
                // Probe only this child, so an encoder that exited early gets
                // reported alongside the frame/transfer error. Drop reaps it.
                let status = encoder.child.try_wait().ok().flatten();
                return Err(error).with_context(|| format!(
                    "exporting environment {} frame {}/{} (ffmpeg status: {:?})",
                    environment.id, frame + 1, environment.frame_count, status));
            }
        }
        encoder.finish(&destination).with_context(|| format!("finishing environment {} export", environment.id))?;
        completed_frames += environment.frame_count;
        println!("Exported environment {}: {} synchronized frames, {:.3}s, {}x{}, {}",
            environment.id, environment.frame_count, environment.duration_seconds,
            recording.video_width, recording.video_height, destination.display());
    }
    println!("Completed {} environment videos ({} synchronized frame pairs). Offline Vulkan readback and CPU video encoding only.",
        recording.environments.len(), completed_frames);
    Ok(())
}
