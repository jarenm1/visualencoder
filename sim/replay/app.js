"use strict";

const $ = (id) => document.getElementById(id);
const screen = $("screen");
const context = screen.getContext("2d");
const timeline = $("timeline");
const playButton = $("play-pause");
const playbackIds = ["play-pause", "rewind", "step-back", "step-forward", "timeline"];
const MODES = ["split", "first", "encoder", "third"];
const PLAY_ICON = "M4 2.5v11L13 8z";
const PAUSE_ICON = "M4 2.5h3v11H4zM9 2.5h3v11H9z";
const VIEW_ICONS = {
  split: "M5.8 4v8M10.2 4v8", first: "M3 6h2.5v4H3z", encoder: "M6.75 6h2.5v4h-2.5z", third: "M10.5 6H13v4h-2.5z",
};
const mapCanvas = document.createElement("canvas");
const mapContext = mapCanvas.getContext("2d");
let encoderMap = null;
let manifest = null;
let selected = -1;
let mode = "split";
let video = null;
let generation = 0;
let pendingTarget = null;
let displayedTime = 0;
let sourceReady = false;
let failed = false;

function environment() { return manifest.environments[selected]; }
function frameAt(time) { return Math.min(environment().frame_count - 1, Math.max(0, Math.floor(time * manifest.control_hz + 0.00001))); }
function setControls(enabled) { playbackIds.forEach((id) => { $(id).disabled = !enabled; }); }
function busy(value) { document.querySelector("main").setAttribute("aria-busy", String(value)); }

function panels() {
  const first = manifest.views.first_person;
  const third = manifest.views.third_person;
  const encoder = { encoder: true, width: first[2], height: first[3] };
  const video = (rect) => ({ rect, width: rect[2], height: rect[3] });
  if (mode === "first") return [video(first)];
  if (mode === "third") return [video(third)];
  if (mode === "encoder") return [encoder];
  return [video(first), encoder, video(third)];
}

function resize() {
  const views = panels();
  screen.width = views.reduce((sum, view) => sum + view.width, 0);
  screen.height = Math.max(...views.map((view) => view.height));
}

function playIcon() {
  const paused = !video || video.paused;
  $("play-icon").setAttribute("d", paused ? PLAY_ICON : PAUSE_ICON);
  playButton.setAttribute("aria-label", paused ? "Play" : "Pause");
}

function drawEncoder(frame, x, width, height) {
  const { width: columns, height: rows } = manifest.encoder_map;
  const size = columns * rows * 3;
  const pixels = mapContext.createImageData(columns, rows);
  for (let source = frame * size, target = 0; target < pixels.data.length; source += 3, target += 4) {
    pixels.data[target] = encoderMap[source];
    pixels.data[target + 1] = encoderMap[source + 1];
    pixels.data[target + 2] = encoderMap[source + 2];
    pixels.data[target + 3] = 255;
  }
  mapContext.putImageData(pixels, 0, 0);
  context.imageSmoothingEnabled = false;
  context.drawImage(mapCanvas, 0, 0, columns, rows, x, 0, width, height);
  context.imageSmoothingEnabled = true;
}

function present(source, time) {
  if (source !== video || source.seeking || source.readyState < 2 || pendingTarget !== null || failed || !encoderMap) return;
  const frame = frameAt(time);
  let x = 0;
  for (const panel of panels()) {
    if (panel.encoder) drawEncoder(frame, x, panel.width, panel.height);
    else context.drawImage(source, ...panel.rect, x, 0, panel.width, panel.height);
    x += panel.width;
  }
  displayedTime = Math.min(time, environment().duration_seconds);
  timeline.value = String(displayedTime);
  timeline.setAttribute("aria-valuetext", `frame ${frame + 1} of ${environment().frame_count}`);
  busy(false);
  playIcon();
}

function fail(message) {
  failed = true;
  video?.pause();
  context.clearRect(0, 0, screen.width, screen.height);
  setControls(false);
  busy(false);
  screen.setAttribute("aria-label", message);
  console.error(message);
}

function renderLoop(source, token) {
  if (source !== video || token !== generation || failed) return;
  source.requestVideoFrameCallback((_now, metadata) => {
    if (source !== video || token !== generation) return;
    present(source, source.paused ? source.currentTime : metadata.mediaTime);
    renderLoop(source, token);
  });
}

function seekTo(target) {
  if (!video || !sourceReady || failed) return;
  video.pause();
  const limit = Math.min(environment().duration_seconds, Number.isFinite(video.duration) ? video.duration : Infinity);
  const requested = Math.min(limit, Math.max(0, target));
  if (Math.abs(video.currentTime - requested) < 0.000001 && !video.seeking && video.readyState >= 2) {
    pendingTarget = null;
    present(video, requested);
  } else {
    pendingTarget = requested;
    video.currentTime = requested;
  }
}

function chooseEnvironment(index) {
  const count = manifest.environments.length;
  selected = ((index % count) + count) % count;
  const env = environment();
  const token = ++generation;
  if (video) {
    video.pause();
    video.removeAttribute("src");
    video.load();
    video.remove();
  }
  pendingTarget = null;
  failed = false;
  sourceReady = false;
  encoderMap = null;
  displayedTime = 0;
  setControls(false);
  busy(true);
  screen.setAttribute("aria-label", `Environment ${env.id}, route ${env.route_id}, ${env.outcome}`);
  timeline.max = String(env.duration_seconds);
  timeline.step = String(1 / manifest.control_hz);
  timeline.value = "0";

  const source = document.createElement("video");
  video = source;
  source.preload = "auto";
  source.muted = true;
  source.playsInline = true;
  $("video-source").append(source);
  const current = () => token === generation && source === video;
  let metadataReady = false;
  const ready = () => {
    if (!current() || !metadataReady || !encoderMap || failed) return;
    sourceReady = true;
    setControls(true);
    present(source, source.currentTime);
  };
  fetch(`/replay/${encodeURIComponent(env.encoder_map)}`, { cache: "no-store" })
    .then((response) => {
      if (!response.ok) throw new Error(`${env.encoder_map}: HTTP ${response.status}`);
      return response.arrayBuffer();
    })
    .then((buffer) => {
      if (!current()) return;
      const { width, height } = manifest.encoder_map;
      if (buffer.byteLength !== env.frame_count * width * height * 3) throw new Error(`${env.encoder_map}: wrong size`);
      encoderMap = new Uint8Array(buffer);
      ready();
    })
    .catch((error) => { if (current()) fail(error.message); });
  source.addEventListener("loadedmetadata", () => {
    if (!current()) return;
    if (source.videoWidth !== manifest.video_width || source.videoHeight !== manifest.video_height) return fail("Video dimensions do not match the recording");
    if (!Number.isFinite(source.duration) || Math.abs(source.duration - env.duration_seconds) > Math.max(0.1, 1 / manifest.control_hz)) return fail("Video duration does not match the recording");
    metadataReady = true;
    ready();
  });
  source.addEventListener("loadeddata", () => { if (current()) present(source, source.currentTime); });
  source.addEventListener("seeked", () => {
    if (!current() || source.seeking) return;
    if (pendingTarget !== null && Math.abs(source.currentTime - pendingTarget) > 0.001) {
      source.currentTime = pendingTarget;
      return;
    }
    pendingTarget = null;
    present(source, source.currentTime);
  });
  source.addEventListener("pause", () => { if (current()) { present(source, source.currentTime); playIcon(); } });
  source.addEventListener("play", () => { if (current()) playIcon(); });
  source.addEventListener("ended", () => { if (current()) { present(source, env.duration_seconds); playIcon(); } });
  source.addEventListener("error", () => { if (current()) fail(`${env.video}: ${source.error?.message || "media error"}`); });
  source.src = `/replay/${encodeURIComponent(env.video)}`;
  source.load();
  renderLoop(source, token);
}

async function togglePlay() {
  if (!video || !sourceReady || failed) return;
  const source = video;
  if (!source.paused) { source.pause(); return; }
  if (source.ended || source.currentTime >= environment().duration_seconds - 0.00001) {
    pendingTarget = null;
    source.currentTime = 0;
  }
  try { await source.play(); }
  catch (error) { if (source === video && error.name !== "AbortError") console.error(error); }
}

function stepFrame(delta) {
  if (!sourceReady || failed) return;
  const frame = Math.max(0, Math.min(environment().frame_count - 1, frameAt(pendingTarget ?? displayedTime) + delta));
  seekTo(frame / manifest.control_hz);
}

function randomEnvironment() {
  const count = manifest.environments.length;
  if (count > 1) chooseEnvironment(selected + 1 + Math.floor(Math.random() * (count - 1)));
}

function cycleView() {
  mode = MODES[(MODES.indexOf(mode) + 1) % MODES.length];
  $("view-icon").setAttribute("d", VIEW_ICONS[mode]);
  $("view").setAttribute("aria-label", `Camera view: ${mode}`);
  resize();
  if (video) present(video, video.currentTime);
}

function validateManifest(data) {
  const demand = (condition, message) => { if (!condition) throw new Error(message); };
  demand(data?.format === "sim-city-replay-v1", "Unsupported recording format");
  demand(Number.isFinite(data.control_hz) && data.control_hz > 0, "Invalid recording frame rate");
  demand(Number.isInteger(data.video_width) && Number.isInteger(data.video_height), "Missing video dimensions");
  for (const key of ["first_person", "third_person"]) {
    const rect = data.views?.[key];
    demand(Array.isArray(rect) && rect.length === 4 && rect.every(Number.isInteger) && rect[0] >= 0 && rect[1] >= 0 && rect[2] > 0 && rect[3] > 0
      && rect[0] + rect[2] <= data.video_width && rect[1] + rect[3] <= data.video_height, `Invalid ${key} crop`);
  }
  demand(Number.isInteger(data.encoder_map?.width) && data.encoder_map.width > 0
    && Number.isInteger(data.encoder_map?.height) && data.encoder_map.height > 0, "Missing encoder map layout");
  demand(Array.isArray(data.environments) && data.environments.length > 0, "Recording has no environments");
  for (const env of data.environments) {
    demand(Number.isInteger(env.frame_count) && env.frame_count > 0, `Invalid frame count for environment ${env.id}`);
    demand(Math.abs(env.duration_seconds - env.frame_count / data.control_hz) < 0.0001, `Invalid duration for environment ${env.id}`);
    demand(typeof env.video === "string" && /^[A-Za-z0-9][A-Za-z0-9_.-]*\.mp4$/.test(env.video), "Unsafe recording video filename");
    demand(typeof env.encoder_map === "string" && /^[A-Za-z0-9][A-Za-z0-9_.-]*\.encoder$/.test(env.encoder_map), "Unsafe encoder map filename");
  }
  return data;
}

$("play-pause").addEventListener("click", togglePlay);
$("rewind").addEventListener("click", () => seekTo(0));
$("step-back").addEventListener("click", () => stepFrame(-1));
$("step-forward").addEventListener("click", () => stepFrame(1));
$("previous-env").addEventListener("click", () => chooseEnvironment(selected - 1));
$("next-env").addEventListener("click", () => chooseEnvironment(selected + 1));
$("random-env").addEventListener("click", randomEnvironment);
$("view").addEventListener("click", cycleView);
timeline.addEventListener("input", () => seekTo(Number(timeline.value)));
document.addEventListener("keydown", (event) => {
  if (event.altKey || event.ctrlKey || event.metaKey || event.target.closest("input")) return;
  if (event.code === "Space" && event.target.closest("button")) return;
  const actions = {
    " ": togglePlay, ArrowLeft: () => stepFrame(-1), ArrowRight: () => stepFrame(1), Home: () => seekTo(0),
    "[": () => chooseEnvironment(selected - 1), "]": () => chooseEnvironment(selected + 1), r: randomEnvironment, v: cycleView,
  };
  const action = manifest && actions[event.key];
  if (action) { event.preventDefault(); action(); }
});

(async () => {
  try {
    const response = await fetch("/replay/manifest.json", { cache: "no-store" });
    if (!response.ok) throw new Error(`Manifest request returned HTTP ${response.status}`);
    manifest = validateManifest(await response.json());
    const multiple = manifest.environments.length > 1;
    $("previous-env").disabled = !multiple;
    $("next-env").disabled = !multiple;
    $("random-env").disabled = !multiple;
    $("view").disabled = false;
    mapCanvas.width = manifest.encoder_map.width;
    mapCanvas.height = manifest.encoder_map.height;
    resize();
    chooseEnvironment(0);
  } catch (error) {
    fail(error.message);
  }
})();
