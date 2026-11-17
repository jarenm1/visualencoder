//! Offline replay export from actual pre-action drone poses (real attitude) and native RGB.
//! The split's actual target designs are drawn at recorded target positions.
//! No policy execution, simulation stepping, or trained-library changes occur.

mod capture;
mod encode;
mod proxy;
mod recording;

use std::{env, ffi::OsStr, path::PathBuf};

use anyhow::{Context, Result, bail, ensure};
use sim_scene::CityScene;

use crate::{capture::Capture, proxy::Proxy, recording::Recording};

fn main() -> Result<()> {
    let mut arguments = env::args_os().skip(1);
    let mut manifest = None;
    while let Some(argument) = arguments.next() {
        if argument == OsStr::new("--help") || argument == OsStr::new("-h") {
            println!("Usage: sim-replay-render --recording <manifest.json>\n\nExports synchronized first-/third-person all-intra H.264 videos using Vulkan and ffmpeg.\nRelative city_path values resolve against the manifest directory.");
            return Ok(());
        } else if argument == OsStr::new("--recording") {
            ensure!(manifest.is_none(), "--recording may only be specified once");
            manifest = Some(PathBuf::from(arguments.next().context("--recording requires a manifest path")?));
        } else {
            bail!("unknown argument {}; use --help", argument.to_string_lossy());
        }
    }
    let manifest = manifest.context("missing --recording <manifest.json>; use --help")?;
    let (recording, directory, columns) = Recording::load(&manifest)?;
    let mut scene = CityScene::load(&recording.city_path)
        .with_context(|| format!("loading recorded city {}", recording.city_path.display()))?;
    ensure!(scene.fingerprint == recording.city_fingerprint,
        "city fingerprint mismatch: recording {}, loaded {}; refusing mismatched geometry",
        recording.city_fingerprint, scene.fingerprint);
    println!("Loaded one shared city: {} triangles, {} material chunks; {}",
        scene.triangles.len(), scene.meshes.len(), scene.fingerprint);
    let proxy = Proxy::append(&mut scene, recording.style);
    let capture = Capture::new(&scene, &recording, &columns, &proxy)?;
    encode::export(&capture, &scene, &proxy, &columns, &recording, &directory)
}
