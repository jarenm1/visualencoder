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
    }
    if (state.target.visible) {
      const [bx0, by0, bx1, by1] = state.target.box;
      ctx.strokeStyle = state.goal && state.goal.scored ? "rgba(255,255,255,0.9)" : "rgba(255,255,255,0.35)";
      ctx.lineWidth = 1.5;
      ctx.strokeRect((bx0 + 1) / 2 * w, (1 - by1) / 2 * h, (bx1 - bx0) / 2 * w, (by1 - by0) / 2 * h);
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

function drawMap() {
  if (!layout || !state) return;
  const [minX, minZ, maxX, maxZ] = layout.district;
  const pad = 20, span = Math.max(maxX - minX, maxZ - minZ) + 2 * pad;
  const s = map.width / span;
  const px = (x) => (x - minX + pad) * s, pz = (z) => (z - minZ + pad) * s;
  mapCtx.clearRect(0, 0, map.width, map.height);
  mapCtx.strokeStyle = "#333";
  mapCtx.strokeRect(px(minX), pz(minZ), (maxX - minX) * s, (maxZ - minZ) * s);
  for (const [x, z, hx, hz, cls] of layout.objects) {
    mapCtx.fillStyle = CLASS_COLOURS[cls] ?? "#444";
    mapCtx.fillRect(px(x - hx), pz(z - hz), 2 * hx * s, 2 * hz * s);
  }
  const t = state.target;
  mapCtx.strokeStyle = "#e6e6e6";
  mapCtx.lineWidth = 1.5;
  mapCtx.strokeRect(px(t.center[0] - t.half[0]), pz(t.center[2] - t.half[2]), 2 * t.half[0] * s, 2 * t.half[2] * s);
  mapCtx.strokeStyle = "rgba(230,230,230,0.25)";
  mapCtx.beginPath();
  mapCtx.arc(px(t.center[0]), pz(t.center[2]), (Math.max(t.half[0], t.half[2]) + 6) * s, 0, 2 * Math.PI);
  mapCtx.stroke();
  const [x, , z] = state.position;
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

let goalSource = null;
async function poll() {
  try {
    const previous = state;
    state = await (await fetch("/state", { cache: "no-store" })).json();
    if (previous && state.clock_s < previous.clock_s) trail = [];
    $("pause-icon").setAttribute("d", state.paused ? "M4 2.5v11L13 8z" : "M4 2.5h3v11H4zM9 2.5h3v11H9z");
    $("instruct").hidden = !state.gemma;
    const goal = state.goal || {};
    if (goal.source !== goalSource) { goalSource = goal.source; $("goal").src = `/goal.jpg?${Date.now()}`; }
    $("goal").title = goal.source || "";
    $("status").textContent = state.error
      ? state.error
      : `${goal.source}${goal.scored ? "" : " (unscored)"} · ${state.target.class} ${state.distance_m.toFixed(1)} m · ` +
        `${state.speed_mps.toFixed(1)} m/s · alt ${state.altitude_m.toFixed(1)} m · hold ${Math.round(state.hold * 100)}% · ` +
        `${state.clock_s.toFixed(1)} s` + (state.last_outcome ? ` · last: ${state.last_outcome}` : "");
    drawMap();
  } catch (_) { /* server restarting */ }
  setTimeout(poll, 100);
}

(async () => {
  layout = await (await fetch("/map")).json();
  poll();
  requestAnimationFrame(drawOverlay);
})();
