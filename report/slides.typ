// Build from the repo root: typst compile --root . report/slides.typ runs/presentation/slides.pdf
#set document(title: "Visual encoders for sim-to-real transfer", author: "Jaren McHugh")
#let ink = rgb("152138")
#let muted = rgb("4C5C72")
#let accent = rgb("00665C")
#let tint = rgb("E3F3EF")
#let rule = rgb("D6DEE9")
#let slide-width = 856pt
#set page(width: 960pt, height: 540pt, margin: (x: 52pt, top: 42pt, bottom: 38pt),
  fill: rgb("F7F9FC"),
  footer: context align(right, text(12pt, fill: muted)[#counter(page).display()]))
#set text(font: ("Inter", "DejaVu Sans"), size: 20pt, fill: ink)
#set par(leading: 0.42em, spacing: 3pt)

#let kicker(body) = text(13pt, weight: "bold", tracking: 1.2pt, fill: accent, body)
#let secondary(body, size: 18pt) = text(size, fill: muted, body)
#let title(section, body) = [
  #kicker(section)
  #v(10pt)
  #text(38pt, weight: "bold", body)
  #v(21pt)
]
#let strip(path, height, aspect) = box(width: slide-width, height: height, clip: true,
  place(top + left, image(path, width: slide-width, height: slide-width / aspect)))
#let card(label, body, highlighted: false) = block(width: 100%, inset: 16pt, radius: 6pt, breakable: false,
  fill: if highlighted { tint } else { white }, stroke: 0.75pt + rule)[
  #kicker(label)
  #v(8pt)
  #body
]

#let refinement-plot() = {
  let checks = json("../reports/gate/presentation-retrain-seed2.json").checks
  let rows = (
    ([Simulation], [Shifted-look retrieval\@1], checks.at("sim/retrieval_at_1")),
    ([Real imagery · UAVid], [Car-patch average precision], checks.at("real/sim_to_real_vehicle_ap")),
    ([Task · held-out flight frames], [Goal-matching score], checks.at("failures/patch_head_score")),
  )
  let x(v) = 280pt + (v + 0.05) * 1350pt
  let score(v) = str(calc.round(v, digits: 3))
  block(width: slide-width, height: 226pt, breakable: false, {
    place(top + left, dx: 691pt, kicker[BEFORE → AFTER])
    for (tick, label) in ((-0.05, [−0.05]), (0.0, [0]), (0.05, [+0.05]), (0.10, [+0.10]), (0.15, [+0.15]), (0.20, [+0.20])) {
      let stroke = if tick == 0 { 1.5pt + muted } else { 0.75pt + rule }
      place(top + left, dx: x(tick), dy: 24pt, line(end: (0pt, 151pt), stroke: stroke))
      place(top + left, dx: x(tick) - 23pt, dy: 182pt, box(width: 46pt, align(center, secondary(label, size: 13pt))))
    }
    for (i, (label, metric, row)) in rows.enumerate() {
      let y = 48pt + i * 49pt
      let (lo, hi) = row.ci95
      let color = if i == 1 { accent } else { muted }
      place(top + left, dy: y - 18pt, text(17pt, weight: "bold", label))
      place(top + left, dy: y + 1pt, secondary(metric, size: 14pt))
      place(top + left, dx: x(lo), dy: y, line(end: (x(hi) - x(lo), 0pt), stroke: 2.5pt + color))
      place(top + left, dx: x(row.difference) - 4.5pt, dy: y - 4.5pt, circle(radius: 4.5pt, fill: color))
      place(top + left, dx: 691pt, dy: y - 8pt, text(16pt, weight: "medium")[#score(row.baseline) → #score(row.candidate)])
    }
    place(top + left, dx: 280pt, dy: 206pt, box(width: x(0.225) - 280pt, align(center, secondary([Score change (refined − original)], size: 15pt))))
  })
}

#kicker[01 · PROBLEM]
#v(45pt)
#text(64pt, weight: "bold")[Visual encoders for \ sim-to-real transfer]
#v(23pt)
#secondary(size: 25pt)[Can features learned in simulation stay useful \
on a real drone’s camera?]
#v(34pt)
#text(22pt, weight: "bold", fill: accent)[Physical AI · Drone perception]
#v(28pt)
#secondary[Jaren McHugh]

#pagebreak()
#title([01 · PROBLEM], [Appearance is one part of the sim-to-real gap])
#strip("figures/looks.png", 128pt, 1600 / 960)
#v(5pt)
#grid(columns: (1fr, 2fr, 2fr),
  secondary([Canonical], size: 15pt),
  secondary([Two training looks], size: 15pt),
  secondary([Two held-out looks], size: 15pt),
)
#v(25pt)
#text(24pt, weight: "bold")[Change lighting, materials, and fog. Keep the geometry fixed.]
#v(15pt)
#secondary[We isolate appearance in simulation, then test transfer on real drone imagery.]

#pagebreak()
#title([02 · SOLUTION], [Transfer the visual representation])
#image("figures/encoder-architecture.svg", width: slide-width)
#v(18pt)
#secondary[22M parameters. Downstream tasks use the student’s features, not the teacher.]

#pagebreak()
#title([02 · SOLUTION], [Features a drone can use])
#v(28pt)
#grid(columns: (1fr, 1fr), gutter: 16pt,
  card([1 CLS TOKEN], [A whole-frame feature for pose and retrieval.], highlighted: true),
  card([300 PATCH TOKENS], [Local features for depth, segmentation, and goal matching.]),
)
#v(25pt)
#secondary[The encoder supplies visual features. Task heads and the pilot decide how to use them.]

#pagebreak()
#title([03 · IMPLEMENTATION], [Learn features from a frozen teacher])
#v(22pt)
#grid(columns: (1fr, 1fr), gutter: 16pt,
  card([FROZEN TEACHER], [LingBot ViT-L/16 produces 1024-dimensional target features.]),
  card([TRAINABLE STUDENT], [Two linear heads project 384 → 1024: one for CLS, one shared across patches.], highlighted: true),
)
#v(25pt)
#text(23pt, weight: "bold")[Cosine loss on the global token and corresponding patches.]

#pagebreak()
#title([03 · IMPLEMENTATION], [What changes: the teacher target])
#v(18pt)
#grid(columns: (1fr, 1fr), gutter: 16pt,
  card([STANDARD DISTILLATION], [
    #grid(columns: (1fr, auto, 1fr), column-gutter: 9pt, row-gutter: 14pt,
      [Look A], [→], [Teacher A],
      [Look B], [→], [Teacher B],
    )
  ]),
  card([PAIRED-VIEW DISTILLATION], [
    #grid(columns: (1fr, auto, 1fr), column-gutter: 9pt, row-gutter: 14pt,
      [Look A], [↘], grid.cell(rowspan: 2, align: horizon)[Mean of \
        teacher A and B],
      [Look B], [↗],
    )
  ], highlighted: true),
)
#v(23pt)
#secondary[Same pose and shared crop, so patch locations line up.]

#pagebreak()
#title([03 · IMPLEMENTATION], [What do the frozen features retain?])
#v(28pt)
#card([LINEAR PROBES], [Fit on canonical renders. Read out pose, depth, and segmentation under shifted looks.])
#v(18pt)
#card([RETRIEVAL], [Given one view, find the same pose under a different look. No fitted probe.], highlighted: true)

#pagebreak()
#title([03 · IMPLEMENTATION · SIMULATOR RESULTS], [Paired targets improve robustness])
#set text(size: 18pt)
#table(
  columns: (2.1fr, 1fr, 1fr, 1.1fr, 1.2fr),
  align: (left, right, right, right, right),
  inset: (x: 12pt, y: 11pt),
  stroke: (top: none, right: none, bottom: 0.65pt + rule, left: none),
  fill: (_, y) => if y == 5 { tint } else { none },
  table.header([*Frozen encoder*], [*Pose R² ↑*], [*Depth err. ↓*], [*Seg. mIoU ↑*], [*Retrieval\@1 ↑*]),
  [LingBot ViT-S], [0.345], [0.277], [0.675], [0.660],
  [LingBot ViT-L (teacher)], [0.375], [0.228], [0.788], [0.700],
  [DINOv3 ViT-L], [0.323], [0.228], [*0.835*], [0.775],
  [Standard distillation], [0.465], [0.242], [0.788], [0.747],
  [*Paired-view distillation*], [*0.574*], [*0.201*], [0.821], [*0.951*],
)
#v(19pt)
#secondary[Held-out lighting and textures. Same probe protocol for all encoders; retrieval needs no fitted probe.]

#pagebreak()
#title([03 · IMPLEMENTATION], [Add unlabeled real training images])
#v(26pt)
#grid(columns: (1fr, 1fr), gutter: 16pt,
  card([SIMULATOR], [Keep the paired teacher target.]),
  card([UAVID TRAINING RGB], [Match the teacher on each unlabeled real frame.], highlighted: true),
)
#v(25pt)
#secondary[Update the last four blocks and final LayerNorm. Test on separate videos.]

#pagebreak()
#title([03 · IMPLEMENTATION · EVALUATION], [Refinement helped most on real images])
#secondary[Fresh training seed (2), same held-out data. Before and after adding unlabeled real images.]
#v(16pt)
#refinement-plot()
#v(12pt)
#secondary(size: 15pt)[Bars: paired 95% bootstrap intervals within this seed. Goal matching is a probe, not flight success.]

#pagebreak()
#title([04 · IMPLICATIONS], [Flight gives us harder camera views])
#image("figures/textured-flight.png", width: 550pt)
#v(18pt)
#text(22pt, weight: "bold")[Learned flight in simulation]
#v(11pt)
#secondary[Selected successes with hints. Re-rendered onboard view and original external view.]
#v(19pt)
#text(20pt, weight: "medium")[Recorded flight frames → fresh probe on frozen features]

#pagebreak()
#title([04 · IMPLICATIONS], [One encoder, three ways to use it])
#align(center, box(inset: (x: 18pt, y: 11pt), radius: 6pt, fill: tint,
  text(19pt, weight: "bold")[Camera RGB → Frozen ViT-S patch features]))
#block(width: slide-width, height: 34pt, {
  let first = 138pt
  let middle = slide-width / 2
  let last = slide-width - first
  let wire = 1.25pt + muted
  place(top + left, dx: first, dy: 17pt, line(end: (last - first, 0pt), stroke: wire))
  for x in (first, middle, last) {
    place(top + left, dx: x, dy: if x == middle { 0pt } else { 17pt },
      line(end: (0pt, if x == middle { 34pt } else { 17pt }), stroke: wire))
  }
})
#grid(columns: (1fr, 1fr, 1fr), gutter: 14pt,
  card([GEOMETRY IN SIM], [Predict depth and semantic masks.]),
  card([REAL-IMAGE TRANSFER], [Rank car patches with a readout fit in simulation.], highlighted: true),
  card([CLOSED LOOP IN SIM], [Match a goal crop and feed a learned pilot.]),
)
#v(26pt)
#secondary[Next: test whether memory helps across frames and measure latency on real hardware.]
