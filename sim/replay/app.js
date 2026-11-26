"use strict";

const $ = (id) => document.getElementById(id);
const screen = $("screen");
const context = screen.getContext("2d");
const timeline = $("timeline");
const playButton = $("play-pause");
const playbackIds = ["play-pause", "rewind", "step-back", "step-forward", "timeline"];
const ALL_MODES = ["split", "first", "encoder", "third"];
let MODES = ALL_MODES;
const PLAY_ICON = "M4 2.5v11L13 8z";
const PAUSE_ICON = "M4 2.5h3v11H4zM9 2.5h3v11H9z";
const VIEW_ICONS = {
  split: "M5.8 4v8M10.2 4v8", first: "M3 6h2.5v4H3z", encoder: "M6.75 6h2.5v4h-2.5z", third: "M10.5 6H13v4h-2.5z",
};
const mapCanvas = document.createElement("canvas");
const mapContext = mapCanvas.getContext("2d");
let encoderMap = null;
let saliencyMap = null;
let memoryMap = null;
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
function replayUrl(name) { return `/replay/${encodeURIComponent(manifest.recording)}/${encodeURIComponent(name)}`; }
function frameAt(time) { return Math.min(environment().frame_count - 1, Math.max(0, Math.floor(time * manifest.control_hz + 0.00001))); }
function setControls(enabled) { playbackIds.forEach((id) => { $(id).disabled = !enabled; }); }
function busy(value) { document.querySelector("main").setAttribute("aria-busy", String(value)); }

function panels() {
  const first = manifest.views.first_person;
  const third = manifest.views.third_person;
  const encoder = { encoder: true, width: first[2], height: first[3] };
  const memory = { memory: true, width: first[3], height: first[3] };
  const video = (rect, overlay = false) => ({ rect, overlay, width: rect[2], height: rect[3] });
  if (mode === "first") return [video(first)];
  if (mode === "third") return [video(third)];
  if (mode === "encoder") return [encoder];
  const overlay = Boolean(manifest.saliency);
  return [video(first, overlay), ...(manifest.encoder_map ? [encoder] : []),
    ...(manifest.memory_map ? [memory] : []), video(third)];
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

// Policy saliency as a red heat layer (alpha = patch probability x P(in view)). When the
// policy believes the target is in view (P > 0.5) its most likely patch gets a solid red
// square outline. A thin white ring marks the TRUE target centre whenever it is visible
// (replay-only ground truth, never a policy input).
function drawSaliency(frame, x, width, height) {
  const { width: columns, height: rows } = manifest.saliency;
  const stride = columns * rows + 1;
  const base = frame * stride;
  const inView = saliencyMap[base + columns * rows] / 255;
  const pixels = mapContext.createImageData(columns, rows);
  for (let patch = 0; patch < columns * rows; patch += 1) {
    pixels.data[patch * 4] = 255;
    pixels.data[patch * 4 + 1] = 40;
    pixels.data[patch * 4 + 2] = 40;
    pixels.data[patch * 4 + 3] = Math.round(saliencyMap[base + patch] * inView * 0.7);
  }
  mapCanvas.width = columns;
  mapCanvas.height = rows;
  mapContext.putImageData(pixels, 0, 0);
  context.imageSmoothingEnabled = false;
  context.drawImage(mapCanvas, 0, 0, columns, rows, x, 0, width, height);
  context.imageSmoothingEnabled = true;
  if (inView > 0.5) {
    let best = 0;
    for (let patch = 1; patch < columns * rows; patch += 1) {
      if (saliencyMap[base + patch] > saliencyMap[base + best]) best = patch;
    }
    const cellW = width / columns;
    const cellH = height / rows;
    context.strokeStyle = "rgba(255,60,60,1)";
    context.lineWidth = 2;
    context.strokeRect(x + (best % columns) * cellW, Math.floor(best / columns) * cellH, cellW, cellH);
  }
  const state = environment().states[frame];
  const field = (name) => state[manifest.state_fields.indexOf(name)];
  if (field("target_visible") > 0.5) {
    // White outline: the target box's projection, the region the saliency label rewards.
    const left = x + (field("target_box_x0") + 1) / 2 * width;
    const right = x + (field("target_box_x1") + 1) / 2 * width;
    const top = (1 - field("target_box_y1")) / 2 * height;
    const bottom = (1 - field("target_box_y0")) / 2 * height;
    context.strokeStyle = "rgba(255,255,255,0.9)";
    context.lineWidth = 1.5;
    context.strokeRect(left, top, right - left, bottom - top);
  }
}

// The policy's spatial memory: top-down, forward up, drone at the centre (white triangle).
// Warm = goal evidence, dim blue = looked at, black = never seen. The white ring is the TRUE
// target position (replay-only ground truth, never a policy input).
function drawMemory(frame, x, width, height) {
  const { width: columns, height: rows, radius_m: radius } = manifest.memory_map;
  const plane = columns * rows;
  const base = frame * 2 * plane;
  mapCanvas.width = columns;
  mapCanvas.height = rows;
  const pixels = mapContext.createImageData(columns, rows);
  for (let cell = 0; cell < plane; cell += 1) {
    const evidence = memoryMap[base + cell] / 255;
    const seen = memoryMap[base + plane + cell] / 255;
    pixels.data[cell * 4] = Math.round(30 * seen + 225 * evidence);
    pixels.data[cell * 4 + 1] = Math.round(45 * seen + 170 * evidence);
    pixels.data[cell * 4 + 2] = Math.round(70 * seen * (1 - evidence));
    pixels.data[cell * 4 + 3] = 255;
  }
  mapContext.putImageData(pixels, 0, 0);
  context.imageSmoothingEnabled = false;
  context.drawImage(mapCanvas, 0, 0, columns, rows, x, 0, width, height);
  context.imageSmoothingEnabled = true;
  const cx = x + width / 2;
  const cy = height / 2;
  context.fillStyle = "rgba(255,255,255,0.95)";
  context.beginPath();
  context.moveTo(cx, cy - 7);
  context.lineTo(cx - 5, cy + 5);
  context.lineTo(cx + 5, cy + 5);
  context.fill();
  const state = environment().states[frame];
  const field = (name) => state[manifest.state_fields.indexOf(name)];
  const [qx, qy, qz, qw] = ["quat_x", "quat_y", "quat_z", "quat_w"].map(field);
  let fx = -2 * (qx * qz + qw * qy);
  let fz = -(1 - 2 * (qx * qx + qy * qy));
  const norm = Math.hypot(fx, fz) || 1;
  fx /= norm;
  fz /= norm;
  const dx = field("target_x") - field("world_x");
  const dz = field("target_z") - field("world_z");
  const forward = dx * fx + dz * fz;
  const right = -dx * fz + dz * fx;
  const scale = width / 2 / radius;
  context.strokeStyle = "rgba(255,255,255,0.9)";
  context.lineWidth = 1.5;
  context.beginPath();
  context.arc(cx + right * scale, cy - forward * scale, 6, 0, 2 * Math.PI);
  context.stroke();
  context.fillStyle = "rgba(255,255,255,0.75)";
  context.font = "11px sans-serif";
  context.fillText(`memory ±${radius} m`, x + 6, height - 8);
}

function present(source, time) {
  if (source !== video || source.seeking || source.readyState < 2 || pendingTarget !== null || failed
    || (manifest.encoder_map && !encoderMap) || (manifest.saliency && !saliencyMap)
    || (manifest.memory_map && !memoryMap)) return;
  const frame = frameAt(time);
  let x = 0;
  for (const panel of panels()) {
    if (panel.encoder) drawEncoder(frame, x, panel.width, panel.height);
    else if (panel.memory) drawMemory(frame, x, panel.width, panel.height);
    else context.drawImage(source, ...panel.rect, x, 0, panel.width, panel.height);
    if (panel.overlay) drawSaliency(frame, x, panel.width, panel.height);
    x += panel.width;
  }
  displayedTime = Math.min(time, environment().duration_seconds);
  drawOutcome(frame);
  showEpisodeStatus(frame);
  timeline.value = String(displayedTime);
  timeline.setAttribute("aria-valuetext", `frame ${frame + 1} of ${environment().frame_count}`);
  busy(false);
  playIcon();
}

// How each episode ended, in plain words. "out_of_bounds" is leaving the training district
// (an invisible boundary) or the ceiling; "collision" is touching geometry or a hard landing.
const OUTCOME_LABELS = {
  success: "SUCCESS", collision: "CRASH", out_of_bounds: "LEFT AREA", timeout: "TIMEOUT",
};
const OUTCOME_COLOURS = {
  success: "#4f9d4f", collision: "#c04848", out_of_bounds: "#c04848", timeout: "#b8932f",
};
const BANNER_SECONDS = 1.0;

function field(name) { return manifest.state_fields?.indexOf(name) ?? -1; }

// Distance from the drone to the target's box, height and speed at a recorded frame.
function flightAt(frame) {
  const states = environment().states;
  if (!Array.isArray(states) || states.length === 0) return null;
  const at = (k) => states[Math.min(k, states.length - 1)];
  const s = at(frame);
  const p = [field("world_x"), field("world_y"), field("world_z")].map((i) => s[i]);
  const c = [field("target_x"), field("target_y"), field("target_z")].map((i) => s[i]);
  const h = [field("target_half_x"), field("target_half_y"), field("target_half_z")].map((i) => s[i]);
  const gap = Math.hypot(...p.map((v, i) => Math.max(Math.abs(v - c[i]) - h[i], 0)));
  const q = frame > 0 ? at(frame - 1) : s;
  const prev = [field("world_x"), field("world_y"), field("world_z")].map((i) => q[i]);
  const speed = Math.hypot(...p.map((v, i) => v - prev[i])) * manifest.control_hz;
  return { gap, height: p[1], speed, landed: s[field("landed")] > 0.5, visible: s[field("target_visible")] > 0.5 };
}

function drawOutcome(frame) {
  const env = environment();
  if (env.duration_seconds - frame / manifest.control_hz > BANNER_SECONDS) return;
  const label = OUTCOME_LABELS[env.outcome] ?? env.outcome;
  const size = Math.max(16, Math.round(screen.height / 9));
  context.save();
  context.font = `600 ${size}px system-ui, sans-serif`;
  const width = context.measureText(label).width + size;
  context.globalAlpha = 0.85;
  context.fillStyle = OUTCOME_COLOURS[env.outcome] ?? "#555";
  context.fillRect((screen.width - width) / 2, size * 0.4, width, size * 1.5);
  context.globalAlpha = 1;
  context.fillStyle = "#fff";
  context.textAlign = "center";
  context.textBaseline = "middle";
  context.fillText(label, screen.width / 2, size * 1.15);
  context.restore();
}

function showEpisodeStatus(frame) {
  const env = environment();
  const status = $("status");
  status.replaceChildren();
  const outcome = document.createElement("b");
  outcome.className = env.outcome;
  outcome.textContent = OUTCOME_LABELS[env.outcome] ?? env.outcome;
  const visibility = ["visible", "partly hidden", "hidden"][env.bucket] ?? "?";
  const head = `env ${env.id} (${selected + 1}/${manifest.environments.length})  ends: `;
  const start = `  at ${env.duration_seconds.toFixed(1)} s · target ${env.target_class}, ${visibility} at start`
    + ` · ${env.gps_available ? "hint" : "no hint"} · ${env.start_landed ? "landed" : "flying"} start`;
  const flight = flightAt(frame);
  const now = flight
    ? `\nt ${(frame / manifest.control_hz).toFixed(1)} s · ${flight.gap.toFixed(1)} m to target`
      + ` · height ${flight.height.toFixed(1)} m · ${flight.speed.toFixed(1)} m/s`
      + `${flight.landed ? " · on ground" : ""}${flight.visible ? " · target in view" : ""}`
    : "";
  status.append(head, outcome, start + now);
  if (manifest.presentation_render) status.append(`\n${manifest.presentation_render}`);
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
  saliencyMap = null;
  memoryMap = null;
  displayedTime = 0;
  setControls(false);
  busy(true);
  const visibility = ["visible", "partly hidden", "hidden"][env.bucket] ?? "unknown visibility";
  screen.setAttribute("aria-label", `Environment ${env.id}, ${visibility} start, ${env.outcome}`);
  timeline.max = String(env.duration_seconds);
  timeline.step = String(1 / manifest.control_hz);
  timeline.value = "0";
  const goal = $("goal");
  goal.hidden = false;
  goal.src = replayUrl(env.goal);
  goal.alt = goal.title = env.grounding
    ? `Gemma box: ${env.grounding.asset}`
    : `Goal crop: ${env.target_class}`;

  const source = document.createElement("video");
  video = source;
  source.preload = "auto";
  source.muted = true;
  source.playsInline = true;
  $("video-source").append(source);
  const current = () => token === generation && source === video;
  let metadataReady = false;
  const ready = () => {
    if (!current() || !metadataReady || (manifest.encoder_map && !encoderMap)
      || (manifest.saliency && !saliencyMap) || (manifest.memory_map && !memoryMap) || failed) return;
    sourceReady = true;
    setControls(true);
    present(source, source.currentTime);
  };
  const load = (name, bytes, assign) => fetch(replayUrl(name), { cache: "no-store" })
    .then((response) => {
      if (!response.ok) throw new Error(`${name}: HTTP ${response.status}`);
      return response.arrayBuffer();
    })
    .then((buffer) => {
      if (!current()) return;
      if (buffer.byteLength !== bytes) throw new Error(`${name}: wrong size`);
      assign(new Uint8Array(buffer));
      ready();
    })
    .catch((error) => { if (current()) fail(error.message); });
  if (manifest.encoder_map) {
    const { width, height } = manifest.encoder_map;
    load(env.encoder_map, env.frame_count * width * height * 3, (data) => { encoderMap = data; });
  }
  if (manifest.saliency) {
    const { width, height } = manifest.saliency;
    load(env.saliency, env.frame_count * (width * height + 1), (data) => { saliencyMap = data; });
  }
  if (manifest.memory_map) {
    const { width, height, channels } = manifest.memory_map;
    load(env.memory_map, env.frame_count * channels * width * height, (data) => { memoryMap = data; });
  }
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
  source.src = replayUrl(env.video);
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
  demand(data?.format === "sim-city-replay-v3", "Unsupported recording format");
  demand(typeof data.recording === "string" && /^[A-Za-z0-9][A-Za-z0-9_.-]*$/.test(data.recording), "Missing recording id");
  demand(Number.isFinite(data.control_hz) && data.control_hz > 0, "Invalid recording frame rate");
  demand(Number.isInteger(data.video_width) && Number.isInteger(data.video_height), "Missing video dimensions");
  for (const key of ["first_person", "third_person"]) {
    const rect = data.views?.[key];
    demand(Array.isArray(rect) && rect.length === 4 && rect.every(Number.isInteger) && rect[0] >= 0 && rect[1] >= 0 && rect[2] > 0 && rect[3] > 0
      && rect[0] + rect[2] <= data.video_width && rect[1] + rect[3] <= data.video_height, `Invalid ${key} crop`);
  }
  demand(data.encoder_map == null || (Number.isInteger(data.encoder_map.width) && data.encoder_map.width > 0
    && Number.isInteger(data.encoder_map.height) && data.encoder_map.height > 0), "Invalid encoder map layout");
  demand(data.memory_map == null || (Number.isInteger(data.memory_map.width) && Number.isInteger(data.memory_map.height)
    && data.memory_map.channels === 2 && data.memory_map.radius_m > 0), "Invalid memory map layout");
  demand(Array.isArray(data.environments) && data.environments.length > 0, "Recording has no environments");
  for (const env of data.environments) {
    demand(Number.isInteger(env.frame_count) && env.frame_count > 0, `Invalid frame count for environment ${env.id}`);
    demand(Math.abs(env.duration_seconds - env.frame_count / data.control_hz) < 0.0001, `Invalid duration for environment ${env.id}`);
    demand(typeof env.video === "string" && /^[A-Za-z0-9][A-Za-z0-9_.-]*\.mp4$/.test(env.video), "Unsafe recording video filename");
    demand(data.encoder_map == null || (typeof env.encoder_map === "string" && /^[A-Za-z0-9][A-Za-z0-9_.-]*\.encoder$/.test(env.encoder_map)), "Unsafe encoder map filename");
    demand(data.memory_map == null || (typeof env.memory_map === "string" && /^[A-Za-z0-9][A-Za-z0-9_.-]*\.memory$/.test(env.memory_map)), "Unsafe memory map filename");
    demand(typeof env.goal === "string" && /^[A-Za-z0-9][A-Za-z0-9_.-]*-goal\.png$/.test(env.goal), "Unsafe goal image filename");
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

async function loadManifest() {
  const response = await fetch("/replay/manifest.json", { cache: "no-store" });
  if (!response.ok) throw new Error(`Manifest request returned HTTP ${response.status}`);
  manifest = validateManifest(await response.json());
  failed = false;
  const multiple = manifest.environments.length > 1;
  $("previous-env").disabled = !multiple;
  $("next-env").disabled = !multiple;
  $("random-env").disabled = !multiple;
  $("view").disabled = false;
  if (manifest.encoder_map) {
    mapCanvas.width = manifest.encoder_map.width;
    mapCanvas.height = manifest.encoder_map.height;
    MODES = ALL_MODES;
  } else {
    MODES = ALL_MODES.filter((name) => name !== "encoder");
    if (mode === "encoder") mode = "split";
  }
  resize();
  chooseEnvironment(0);
}

// Live mode: the server resolves typed instructions with Gemma, records the policy, then swaps recordings.
const instruction = $("instruction");

function showStatus(status) {
  const working = ["framing", "grounding", "flying"].includes(status.state);
  instruction.disabled = working;
  $("send").disabled = working;
  const found = `${status.found} views with a box, ${status.matched} matched to a city object`;
  if (status.state === "framing") instruction.placeholder = `${status.instruction}: rendering start views`;
  else if (status.state === "grounding") instruction.placeholder = `${status.instruction}: Gemma looking`;
  else if (status.state === "flying") instruction.placeholder = `${status.instruction}: ${found} - flying`;
  else if (status.state === "done") instruction.placeholder = `${status.instruction}: ${found}`;
  else if (status.state === "error") instruction.placeholder = status.message;
  if (working) instruction.value = "";
  return working;
}

async function pollStatus() {
  const response = await fetch("/api/status", { cache: "no-store" });
  if (!response.ok) return;
  const status = await response.json();
  if (!status.live) return;
  $("instruct").hidden = false;
  const working = showStatus(status);
  // The server's current recording is the source of truth; switch whenever it differs from the one shown.
  if (status.current !== manifest?.recording) await loadManifest();
  if (working) setTimeout(() => pollStatus().catch((error) => fail(error.message)), 1500);
}

$("instruct").addEventListener("submit", async (event) => {
  event.preventDefault();
  const text = instruction.value.trim();
  if (!text) return;
  const response = await fetch("/api/instruct", {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ instruction: text }),
  });
  if (!response.ok && response.status !== 409) return fail(`Instruction rejected: HTTP ${response.status}`);
  pollStatus().catch((error) => fail(error.message));
});

(async () => {
  try {
    await loadManifest();
    await pollStatus();
  } catch (error) {
    fail(error.message);
  }
})();
