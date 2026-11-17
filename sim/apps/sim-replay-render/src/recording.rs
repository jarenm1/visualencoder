use std::{collections::HashSet, fs, path::{Component, Path, PathBuf}};

use anyhow::{Context, Result, ensure};
use serde::Deserialize;

pub const FORMAT: &str = "sim-city-replay-v3";
/// Four control channels (named by `action_names`) followed by these outcome fields.
const OUTCOME_FIELDS: [&str; 5] = ["reward", "done", "success", "collision", "out_of_bounds"];
const TRANSITION_WIDTH: usize = 4 + OUTCOME_FIELDS.len();
const STATE_FIELDS: [&str; 13] = ["world_x", "world_y", "world_z", "quat_x", "quat_y", "quat_z", "quat_w",
    "target_x", "target_y", "target_z", "catalog", "clock", "landed"];

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
    pub state_fields: Vec<String>,
    pub action_names: Vec<String>,
    pub transition_fields: Vec<String>,
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
    pub frame_count: usize,
    pub duration_seconds: f64,
    pub outcome: String,
    pub video: PathBuf,
    pub states: Vec<Vec<f32>>,
    pub transitions: Vec<Vec<f64>>,
}

/// Recorded pose needed for one third-person frame.
#[derive(Clone, Copy)]
pub struct Pose {
    pub position: [f32; 3],
    /// xyzw, actual simulated attitude.
    pub rotation: [f32; 4],
}

/// Column indices of the named state fields used by the renderer.
#[derive(Clone, Copy)]
pub struct StateColumns {
    position: [usize; 3],
    rotation: [usize; 4],
    target: [usize; 3],
    catalog: usize,
    clock: usize,
}

impl StateColumns {
    fn new(fields: &[String]) -> Result<Self> {
        let column = |name: &str| fields.iter().position(|field| field == name)
            .with_context(|| format!("manifest state_fields lacks {name}"));
        Ok(Self {
            position: [column("world_x")?, column("world_y")?, column("world_z")?],
            rotation: [column("quat_x")?, column("quat_y")?, column("quat_z")?, column("quat_w")?],
            target: [column("target_x")?, column("target_y")?, column("target_z")?],
            catalog: column("catalog")?,
            clock: column("clock")?,
        })
    }

    pub fn pose(&self, state: &[f32]) -> Pose {
        Pose { position: self.position.map(|i| state[i]), rotation: self.rotation.map(|i| state[i]) }
    }
}

impl Recording {
    pub fn load(path: &Path) -> Result<(Self, PathBuf, StateColumns)> {
        let path = path.canonicalize().context("locating recording manifest")?;
        let directory = path.parent().context("manifest has no parent directory")?.to_path_buf();
        let mut recording: Self = serde_json::from_slice(
            &fs::read(&path).with_context(|| format!("reading {}", path.display()))?,
        ).context("decoding replay manifest")?;
        let columns = recording.validate(&directory)?;
        if recording.city_path.is_relative() {
            recording.city_path = directory.join(&recording.city_path);
        }
        Ok((recording, directory, columns))
    }

    fn validate(&self, directory: &Path) -> Result<StateColumns> {
        ensure!(self.format == FORMAT, "unsupported replay format {} (expected {FORMAT})", self.format);
        ensure!(self.control_hz.is_finite() && self.control_hz > 0.0, "control_hz must be positive and finite");
        ensure!(self.frame_width > 0 && self.frame_height > 0 && self.frame_height % 2 == 0,
            "frame dimensions must be positive, with even height for yuv420p");
        ensure!(self.frame_width.checked_mul(2) == Some(self.video_width)
            && self.video_height == self.frame_height, "video dimensions must contain exactly two side-by-side views");
        ensure!(self.views.first_person == [0, 0, self.frame_width, self.frame_height]
            && self.views.third_person == [self.frame_width, 0, self.frame_width, self.frame_height],
            "manifest view rectangles do not match side-by-side video layout");
        ensure!(self.action_names.len() == 4, "action_names must name exactly four control channels");
        ensure!(self.transition_fields.len() == TRANSITION_WIDTH
            && self.transition_fields[..4] == self.action_names[..]
            && self.transition_fields[4..].iter().map(String::as_str).eq(OUTCOME_FIELDS),
            "transition_fields must be action_names followed by {OUTCOME_FIELDS:?}");
        ensure!(STATE_FIELDS.iter().all(|name| self.state_fields.iter().any(|field| field == name)),
            "state_fields must include {STATE_FIELDS:?}");
        let columns = StateColumns::new(&self.state_fields)?;
        ensure!(!self.environments.is_empty(), "recording contains no environments");
        let mut ids = HashSet::new();
        let mut videos = HashSet::new();
        for environment in &self.environments {
            ensure!(ids.insert(environment.id), "duplicate environment ID {}", environment.id);
            ensure!(videos.insert(&environment.video), "duplicate video filename {}", environment.video.display());
            environment.validate(self, &columns)
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
        Ok(columns)
    }
}

impl Environment {
    fn validate(&self, recording: &Recording, columns: &StateColumns) -> Result<()> {
        let mut components = self.video.components();
        ensure!(matches!(components.next(), Some(Component::Normal(_))) && components.next().is_none()
            && self.video.extension().is_some_and(|extension| extension == "mp4"),
            "video must be a basename ending in .mp4");
        ensure!(self.frame_count > 0 && self.states.len() == self.frame_count
            && self.transitions.len() == self.frame_count, "frame_count must equal nonempty state and transition lengths");
        let duration = self.frame_count as f64 / recording.control_hz;
        ensure!(self.duration_seconds.is_finite() && (self.duration_seconds - duration).abs() <= 1e-6 * duration.max(1.0),
            "duration_seconds must equal frame_count / control_hz");
        ensure!(matches!(self.outcome.as_str(), "success" | "collision" | "out_of_bounds" | "timeout"), "unknown terminal outcome");
        let first_target = columns.target.map(|i| self.states[0][i]);
        let first_catalog = self.states[0][columns.catalog];
        for (index, (state, transition)) in self.states.iter().zip(&self.transitions).enumerate() {
            ensure!(state.len() == recording.state_fields.len() && transition.len() == TRANSITION_WIDTH,
                "row width mismatch at frame {index}");
            ensure!(state.iter().all(|value| value.is_finite()) && transition.iter().all(|value| value.is_finite()),
                "non-finite value at frame {index}");
            ensure!(state[columns.clock] as f64 == index as f64,
                "clock discontinuity at frame {index}; autoreset states must not be appended");
            ensure!(state[columns.catalog] == first_catalog, "catalog changes at frame {index}");
            ensure!(columns.target.map(|i| state[i]) == first_target, "target moves at frame {index}");
            let rotation = columns.rotation.map(|i| state[i]);
            let norm = rotation.iter().map(|v| v * v).sum::<f32>().sqrt();
            ensure!((norm - 1.0).abs() < 1e-3, "attitude quaternion is not unit at frame {index}");
            ensure!(transition[..4].iter().all(|value| (-1.0..=1.0).contains(value)), "action outside [-1,1] at frame {index}");
            ensure!(transition[5..].iter().all(|&value| value == 0.0 || value == 1.0),
                "terminal flags must be zero or one at frame {index}");
            let last = index + 1 == self.frame_count;
            ensure!((transition[5] == 1.0) == last, "only the last transition may be terminal");
            let outcome = [transition[6], transition[7], transition[8]];
            if !last {
                ensure!(outcome == [0.0; 3], "early terminal outcome at frame {index}");
            } else {
                let expected = match self.outcome.as_str() {
                    "success" => [1.0, 0.0, 0.0],
                    "collision" => [0.0, 1.0, 0.0],
                    "out_of_bounds" => [0.0, 0.0, 1.0],
                    _ => [0.0; 3],
                };
                ensure!(outcome == expected, "outcome disagrees with terminal transition");
            }
        }
        Ok(())
    }
}
