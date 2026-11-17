"use strict";

const $ = (id) => document.getElementById(id);
const overlay = $("overlay");
const ctx = overlay.getContext("2d");
const map = $("map");
const mapCtx = map.getContext("2d");
const CLASS_COLOURS = { building: "#4a4a4a", vehicle: "#6a5a3a", vegetation: "#2f4a35", prop: "#3a3a52" };
let state = null;
let layout = null;
let drag = null;
let trail = [];

const post = (path, body = {}) => fetch(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });

// Drag a box on the camera view: its normalized corners become the goal.
function normalized(event) {
  const rect = overlay.getBoundingClientRect();
  return [Math.min(1, Math.max(0, (event.clientX - rect.left) / rect.width)),
          Math.min(1, Math.max(0, (event.clientY - rect.top) / rect.height))];
}
overlay.addEventListener("pointerdown", (event) => { drag = { from: normalized(event), to: normalized(event) }; overlay.setPointerCapture(event.pointerId); });
overlay.addEventListener("pointermove", (event) => { if (drag) drag.to = normalized(event); });
overlay.addEventListener("pointerup", () => {
  if (!drag) return;
  const [x0, x1] = [Math.min(drag.from[0], drag.to[0]), Math.max(drag.from[0], drag.to[0])];
  const [y0, y1] = [Math.min(drag.from[1], drag.to[1]), Math.max(drag.from[1], drag.to[1])];
  drag = null;
  if (x1 - x0 > 0.01 && y1 - y0 > 0.01) post("/box", { box: [x0, y0, x1, y1] });
});

$("pause").addEventListener("click", () => post("/pause"));
$("new").addEventListener("click", () => { trail = []; post("/new"); });
$("catalog").addEventListener("click", () => post("/catalog"));
$("instruct").addEventListener("submit", (event) => {
  event.preventDefault();
  const text = $("instruction").value.trim();
  if (text) { post("/instruct", { instruction: text }); $("instruction").value = ""; }
});

function drawOverlay() {
  const w = overlay.width, h = overlay.height;
  ctx.clearRect(0, 0, w, h);
  if (state) {
    const [rows, columns] = state.saliency.grid;
    if (state.saliency.in_view > 0.5) {
      const peak = state.saliency.peak;
      ctx.strokeStyle = "rgba(255,70,70,0.95)";
      ctx.lineWidth = 2;
      ctx.strokeRect((peak % columns) * w / columns, Math.floor(peak / columns) * h / rows, w / columns, h / rows);
      ctx.fillStyle = "rgba(255,100,100,0.95)";
      ctx.font = "12px system-ui";
      ctx.fillText("policy", Math.min(w - 42, (peak % columns) * w / columns),
                   Math.max(12, Math.floor(peak / columns) * h / rows - 4));
    }
    if (state.target.visible && state.goal.scored) {
      const [bx0, by0, bx1, by1] = state.target.box;
      ctx.strokeStyle = "rgba(255,255,255,0.9)";
      ctx.lineWidth = 1.5;
      ctx.strokeRect((bx0 + 1) / 2 * w, (1 - by1) / 2 * h, (bx1 - bx0) / 2 * w, (by1 - by0) / 2 * h);
      ctx.fillStyle = "#fff";
      ctx.font = "12px system-ui";
      ctx.fillText(`GT #${state.target.object_index}`, Math.max(0, (bx0 + 1) / 2 * w),
                   Math.max(12, (1 - by1) / 2 * h - 4));
    }
  }
  if (drag) {
    ctx.strokeStyle = "rgba(120,200,255,0.95)";
    ctx.setLineDash([6, 4]);
    ctx.strokeRect(drag.from[0] * w, drag.from[1] * h, (drag.to[0] - drag.from[0]) * w, (drag.to[1] - drag.from[1]) * h);
    ctx.setLineDash([]);
  }
  requestAnimationFrame(drawOverlay);
}

const MAP_SPAN = 160;   // metres shown around the drone

function drawMap() {
  if (!layout || !state) return;
  const [x, , z] = state.position;
  const s = map.width / MAP_SPAN;
  const px = (wx) => (wx - x) * s + map.width / 2, pz = (wz) => (wz - z) * s + map.height / 2;
  mapCtx.clearRect(0, 0, map.width, map.height);
  const edge = layout.city_half_extent;
  mapCtx.strokeStyle = "#5a3a3a";
  mapCtx.strokeRect(px(-edge), pz(-edge), 2 * edge * s, 2 * edge * s);
  const [minX, minZ, maxX, maxZ] = layout.district;
  mapCtx.strokeStyle = "#333";
  mapCtx.strokeRect(px(minX), pz(minZ), (maxX - minX) * s, (maxZ - minZ) * s);
  const reach = MAP_SPAN / 2 + 15;
  for (const [ox, oz, hx, hz, cls] of layout.objects) {
    if (Math.abs(ox - x) > reach || Math.abs(oz - z) > reach) continue;
    mapCtx.fillStyle = CLASS_COLOURS[cls] ?? "#444";
    mapCtx.fillRect(px(ox - hx), pz(oz - hz), 2 * hx * s, 2 * hz * s);
  }
  const t = state.target;
  mapCtx.strokeStyle = state.goal.scored ? "#e6e6e6" : "#555";
  mapCtx.lineWidth = 1.5;
  mapCtx.strokeRect(px(t.center[0] - t.half[0]), pz(t.center[2] - t.half[2]), 2 * t.half[0] * s, 2 * t.half[2] * s);
  mapCtx.strokeStyle = state.goal.scored ? "rgba(230,230,230,0.25)" : "#333";
  mapCtx.beginPath();
  mapCtx.arc(px(t.center[0]), pz(t.center[2]), (Math.max(t.half[0], t.half[2]) + 6) * s, 0, 2 * Math.PI);
  mapCtx.stroke();
  // Trail: only the flight since the last teleport (crash restart).
  trail.push([x, z]);
  if (trail.length > 900) trail.shift();
  mapCtx.strokeStyle = "rgba(120,200,255,0.5)";
  mapCtx.beginPath();
  trail.forEach(([tx, tz], i) => (i ? mapCtx.lineTo(px(tx), pz(tz)) : mapCtx.moveTo(px(tx), pz(tz))));
  mapCtx.stroke();
  const [hx, hz] = state.heading;
  mapCtx.fillStyle = "#78c8ff";
  mapCtx.beginPath();
  mapCtx.arc(px(x), pz(z), 4, 0, 2 * Math.PI);
  mapCtx.fill();
  mapCtx.strokeStyle = "#78c8ff";
  mapCtx.beginPath();
  mapCtx.moveTo(px(x), pz(z));
  mapCtx.lineTo(px(x + hx * 8), pz(z + hz * 8));
  mapCtx.stroke();
}

let goalRevision = null;
async function poll() {
  try {
    const previous = state;
    state = await (await fetch("/state", { cache: "no-store" })).json();
    if (previous) {
      const [ax, , az] = previous.position, [bx, , bz] = state.position;
      if (Math.hypot(bx - ax, bz - az) > 10) trail = [];
    }
    $("pause-icon").setAttribute("d", state.paused ? "M4 2.5v11L13 8z" : "M4 2.5h3v11H4zM9 2.5h3v11H9z");
    $("instruct").hidden = !state.gemma;
    const goal = state.goal || {};
    if (goal.revision !== goalRevision) {
      goalRevision = goal.revision;
      $("goal").src = `/goal.jpg?${goalRevision}`;
    }
    $("goal").title = goal.source || "";
    const target = state.target, prediction = state.saliency;
    const match = goal.match_iou == null ? "catalog reference" : `box match IoU ${goal.match_iou.toFixed(2)}`;
    $("goal-detail").textContent = `Goal: ${goal.source} · ${match} · crop age ${goal.age_s.toFixed(1)} s`;
    const dimensions = target.half.map((v) => (2 * v).toFixed(1)).join(" × ");
    const visibility = target.visible ? "anchor visible" : target.in_frustum ? "anchor occluded" : "anchor outside view";
    $("target-detail").textContent = `${goal.scored ? "Simulator GT" : "UNMATCHED GOAL — old, unrelated simulator target"}: ` +
      `${target.asset} #${target.object_index} · ${target.class} · ${dimensions} m · ${visibility}`;
    const agreement = prediction.on_target == null ? "not comparable" : prediction.on_target ? "on GT box" : "OFF GT box";
    $("perception-detail").textContent = `Policy: P(target in view) ${Math.round(prediction.in_view * 100)}% · ` +
      `predicted patch ${agreement} (patch-in-box check, not object identity)`;
    const distance = goal.scored
      ? `distance to box ${state.distance_m.toFixed(1)} m (selected at ${goal.distance_at_selection_m.toFixed(1)} m)`
      : "goal distance/hold unavailable; no simulator match";
    $("status").textContent = (state.error ? `${state.error} · ` : "") +
      `${distance} · ${state.speed_mps.toFixed(1)} m/s · alt ${state.altitude_m.toFixed(1)} m` +
      (goal.scored ? ` · hold ${state.hold >= 1 ? "complete" : `${Math.round(state.hold * 100)}%`}` : "") +
      (state.last_outcome ? ` · last outcome: ${state.last_outcome}` : "");
    const checkpoint = state.checkpoint;
    $("checkpoint-detail").textContent = `${(checkpoint.steps / 1e6).toFixed(2)}M checkpoint · ` +
      `fixed live goal (no automatic refresh) · ${checkpoint.path}`;
    $("checkpoint-detail").textContent += " · training success = final 3 s within 6 m, ≤1 m/s, facing";
    drawMap();
  } catch (_) { /* server restarting */ }
  setTimeout(poll, 100);
}

(async () => {
  layout = await (await fetch("/map")).json();
  poll();
  requestAnimationFrame(drawOverlay);
})();
