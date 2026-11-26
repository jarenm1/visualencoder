//! Offline replay export from actual pre-action drone poses (real attitude) and native RGB.
//! Placed city targets are part of the shared city geometry and are not drawn separately.
//! No policy execution, simulation stepping, or trained-library changes occur.

mod capture;
mod encode;
mod proxy;
mod recording;

use std::{env, ffi::OsStr, io::{self, BufWriter, Write}, path::PathBuf};

use anyhow::{Context, Result, bail, ensure};
use sim_scene::CityScene;

use crate::{capture::Capture, recording::Recording};

fn main() -> Result<()> {
    let mut arguments = env::args_os().skip(1);
    let mut manifest = None;
    let mut chase_poses = false;
    while let Some(argument) = arguments.next() {
        if argument == OsStr::new("--help") || argument == OsStr::new("-h") {
            println!("Usage: sim-replay-render --recording <manifest.json> [--chase-poses]\n\nExports synchronized first-/third-person H.264 with Vulkan and ffmpeg, or prints chase camera positions as JSON lines.");
            return Ok(());
        } else if argument == OsStr::new("--recording") {
            ensure!(manifest.is_none(), "--recording may only be specified once");
            manifest = Some(PathBuf::from(arguments.next().context("--recording requires a manifest path")?));
        } else if argument == OsStr::new("--chase-poses") {
            chase_poses = true;
        } else {
            bail!("unknown argument {}; use --help", argument.to_string_lossy());
        }
    }
    let manifest = manifest.context("missing --recording <manifest.json>; use --help")?;
    let (recording, directory, columns) = Recording::load(&manifest, !chase_poses)?;
    let mut scene = CityScene::load(&recording.city_path)
        .with_context(|| format!("loading recorded city {}", recording.city_path.display()))?;
    ensure!(scene.fingerprint == recording.city_fingerprint,
        "city fingerprint mismatch: recording {}, loaded {}; refusing mismatched geometry",
        recording.city_fingerprint, scene.fingerprint);
    if chase_poses {
        let mut output = BufWriter::new(io::stdout().lock());
        for episode in &recording.environments {
            write!(output, "[{},[", episode.id)?;
            for (index, state) in episode.states.iter().enumerate() {
                if index != 0 {
                    output.write_all(b",")?;
                }
                let camera = capture::chase_camera(&scene, &columns.pose(state))?;
                serde_json::to_writer(&mut output, &camera.eye.to_array())?;
            }
            output.write_all(b"]]\n")?;
        }
        output.flush()?;
        return Ok(());
    }
    println!("Loaded one shared city: {} triangles, {} material chunks; {}",
        scene.triangles.len(), scene.meshes.len(), scene.fingerprint);
    proxy::append(&mut scene);
    let capture = Capture::new(&scene, &recording, &columns)?;
    encode::export(&capture, &scene, &columns, &recording, &directory)
}
