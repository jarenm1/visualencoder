use std::{collections::HashSet, fs, path::{Component, Path, PathBuf}};

use anyhow::{Context, Result, ensure};
use serde::Deserialize;

#[derive(Deserialize)]
pub struct Recording {
    pub format: String,
    pub control_hz: f64,
    pub frame_width: u32,
    pub frame_height: u32,
    pub video_width: u32,
    pub video_height: u32,
    pub views: Views,
    pub city_path: PathBuf,
    pub city_fingerprint: String,
    pub environments: Vec<Environment>,
}

#[derive(Deserialize)]
pub struct Views {
    first_person: [u32; 4],
    third_person: [u32; 4],
}

#[derive(Deserialize)]
pub struct Environment {
    pub id: u32,
    pub route_id: u32,
    pub frame_count: usize,
    pub duration_seconds: f64,
    pub outcome: String,
    pub video: PathBuf,
    pub states: Vec<[f32; 8]>,
    pub transitions: Vec<[f64; 5]>,
}

impl Recording {
    pub fn load(path: &Path) -> Result<(Self, PathBuf)> {
        let path = path.canonicalize().context("locating recording manifest")?;
        let directory = path.parent().context("manifest has no parent directory")?.to_path_buf();
        let mut recording: Self = serde_json::from_slice(
            &fs::read(&path).with_context(|| format!("reading {}", path.display()))?,
        ).context("decoding replay manifest")?;
        recording.validate(&directory)?;
        if recording.city_path.is_relative() {
            recording.city_path = directory.join(&recording.city_path);
        }
        Ok((recording, directory))
    }

    fn validate(&self, directory: &Path) -> Result<()> {
        ensure!(self.format == "sim-city-replay-v1", "unsupported replay format {}", self.format);
        ensure!(self.control_hz.is_finite() && self.control_hz > 0.0, "control_hz must be positive and finite");
        ensure!(self.frame_width > 0 && self.frame_height > 0 && self.frame_height % 2 == 0,
            "frame dimensions must be positive, with even height for yuv420p");
        ensure!(self.frame_width.checked_mul(2) == Some(self.video_width)
            && self.video_height == self.frame_height, "video dimensions must contain exactly two side-by-side views");
        ensure!(self.views.first_person == [0, 0, self.frame_width, self.frame_height]
            && self.views.third_person == [self.frame_width, 0, self.frame_width, self.frame_height],
            "manifest view rectangles do not match side-by-side video layout");
        ensure!(!self.environments.is_empty(), "recording contains no environments");
        let mut ids = HashSet::new();
        let mut videos = HashSet::new();
        for environment in &self.environments {
            ensure!(ids.insert(environment.id), "duplicate environment ID {}", environment.id);
            ensure!(videos.insert(&environment.video), "duplicate video filename {}", environment.video.display());
            environment.validate(self.control_hz)
                .with_context(|| format!("invalid environment {}", environment.id))?;
            let frame_bytes = u64::from(self.frame_width).checked_mul(u64::from(self.frame_height))
                .and_then(|pixels| pixels.checked_mul(3)).context("raw RGB frame size overflows u64")?;
            let expected = frame_bytes.checked_mul(u64::try_from(environment.frame_count)?)
                .context("raw recording size overflows u64")?;
            let raw = directory.join(environment.video.with_extension("rgb"));
            let metadata = fs::metadata(&raw).with_context(|| format!("reading raw first-person recording {}", raw.display()))?;
            ensure!(metadata.is_file() && metadata.len() == expected,
                "{} must contain exactly {} RGB24 bytes (found {})", raw.display(), expected, metadata.len());
        }
        Ok(())
    }
}

impl Environment {
    fn validate(&self, control_hz: f64) -> Result<()> {
        let mut components = self.video.components();
        ensure!(matches!(components.next(), Some(Component::Normal(_))) && components.next().is_none()
            && self.video.extension().is_some_and(|extension| extension == "mp4"),
            "video must be a basename ending in .mp4");
        ensure!(self.frame_count > 0 && self.states.len() == self.frame_count
            && self.transitions.len() == self.frame_count, "frame_count must equal nonempty state and transition lengths");
        let duration = self.frame_count as f64 / control_hz;
        ensure!(self.duration_seconds.is_finite() && (self.duration_seconds - duration).abs() <= 1e-6 * duration.max(1.0),
            "duration_seconds must equal frame_count / control_hz");
        ensure!(matches!(self.outcome.as_str(), "success" | "collision" | "timeout"), "unknown terminal outcome");
        for (index, (state, transition)) in self.states.iter().zip(&self.transitions).enumerate() {
            ensure!(state.iter().all(|value| value.is_finite()) && transition.iter().all(|value| value.is_finite()),
                "non-finite value at frame {index}");
            ensure!(state[6] as f64 == f64::from(self.route_id) && state[7] as f64 == index as f64,
                "route/clock discontinuity at frame {index}; autoreset states must not be appended");
            ensure!(state[3..6] == self.states[0][3..6], "goal changes at frame {index}");
            ensure!(transition[0].fract() == 0.0 && (0.0..=8.0).contains(&transition[0]), "invalid action at frame {index}");
            ensure!(transition[2..].iter().all(|&value| value == 0.0 || value == 1.0),
                "terminal flags must be zero or one at frame {index}");
            let last = index + 1 == self.frame_count;
            ensure!((transition[2] == 1.0) == last, "only the last transition may be terminal");
            if !last {
                ensure!(transition[3] == 0.0 && transition[4] == 0.0, "early terminal outcome at frame {index}");
            } else {
                let expected = match self.outcome.as_str() {
                    "success" => [1.0, 0.0],
                    "collision" => [0.0, 1.0],
                    _ => [0.0, 0.0],
                };
                ensure!(transition[3..] == expected, "outcome disagrees with terminal transition");
            }
        }
        Ok(())
    }
}
