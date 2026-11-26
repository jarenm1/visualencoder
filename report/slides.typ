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

#let panels(path, labels, width: slide-width) = {
  image(path, width: width)
  v(6pt)
  grid(columns: labels.map(_ => width / labels.len()), ..labels.map(label => secondary(label, size: 15pt)))
}

#let three(..items) = grid(columns: (1fr, 1fr, 1fr), gutter: 14pt, ..items.pos())

#let refinement-plot() = {
  let checks = json("presentation-retrain-seed2.json").checks
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
#title([02 · SOLUTION], [A patch token knows more than its pixels])
#panels("figures/patch-pca.png", ([Onboard RGB · 16 × 16 pixel cells], [Same frame · token PCA, not class colors]), width: 740pt)
#v(12pt)
#secondary[Each grid cell has 384 features that can use context from the entire image. The CLS token summarizes the frame.]

#pagebreak()
#title([02 · SOLUTION], [Read geometry and semantics from patches])
#panels("figures/patch-readouts.png", ([Held-out simulator RGB], [Predicted depth · warm is near], [Predicted classes]))
#v(16pt)
#secondary[The encoder stays frozen. Separate linear heads predict distance and class from each patch token.]

#pagebreak()
#title([03 · IMPLEMENTATION], [Learn features from a frozen teacher])
#image("figures/paired-training-views.png", width: 370pt)
#v(10pt)
#grid(columns: (1fr, 1fr), gutter: 14pt,
  card([FROZEN LINGBOT-L], [1,024-dimensional teacher features.]),
  card([TRAINABLE LINGBOT-S], [384-dimensional features + a projection for each token type.], highlighted: true),
)
#v(12pt)
#text(23pt, weight: "bold")[Cosine loss on the global token and corresponding patches.]

#pagebreak()
#title([03 · IMPLEMENTATION], [What changes: the teacher target])
#image("figures/paired-training-views.png", width: 400pt)
#v(10pt)
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
#strip("figures/geometry.png", 129pt, 1600 / 530)
#v(9pt)
#secondary[Canonical views across city layouts. Probe fitting and testing use separate poses.]
#v(16pt)
#grid(columns: (1fr, 1fr), gutter: 16pt,
  card([LINEAR PROBES], [Fit on canonical renders. Test pose, depth, and labels under shifted looks.]),
  card([RETRIEVAL], [Find the same pose under a different look. No fitted probe.], highlighted: true),
)

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
#grid(columns: (2fr, 1fr), gutter: 0pt,
  panels("figures/paired-training-views.png", ([Sim look A], [Sim look B]), width: 544pt),
  panels("figures/uavid-training-view.png", ([Unlabeled UAVid RGB],), width: 272pt),
)
#v(17pt)
#secondary[Keep the paired simulator target. On each real frame, match the frozen teacher. Update only the last four blocks and final normalization.]

#pagebreak()
#title([03 · IMPLEMENTATION], [A simulator-trained readout sees real cars])
#panels("figures/real-car-readout.png", ([Held-out UAVid RGB], [Car scores · original], [Car scores · refined]))
#v(14pt)
#secondary[Same color scale. Brighter patches rank higher for cars. Scores are not probabilities.]

#pagebreak()
#title([03 · IMPLEMENTATION · EVALUATION], [Refinement helped most on real images])
#secondary[Fresh training seed (2), same held-out data. Before and after adding unlabeled real images.]
#v(16pt)
#refinement-plot()
#v(12pt)
#secondary(size: 15pt)[Bars: paired 95% bootstrap intervals within this seed. Goal matching is a probe, not flight success.]

#pagebreak()
#title([03 · IMPLEMENTATION · PILOT], [Find the goal in the current view])
#panels("figures/pilot-match.png", ([Reference crop], [Policy camera], [Match scores if in view]))
#v(16pt)
#secondary[Orange marks stronger relative matches. A separate output estimates whether the target is in view at all.]

#pagebreak()
#title([03 · IMPLEMENTATION · PILOT], [Remember where the camera looked])
#panels("figures/pilot-memory.png", ([0 s], [4 s], [8 s]), width: 745pt)
#v(16pt)
#secondary[Teal: looked-at space. Amber: goal evidence. The arrow is the drone, pointing forward.]

#pagebreak()
#title([03 · IMPLEMENTATION · PILOT], [The actor does not get the answer])
#grid(columns: (1fr, auto, 1fr, auto, 1fr), gutter: 10pt, align: horizon,
  card([OBSERVE], [Patch matches, depth, map, sensors, optional hint and command.]),
  text(24pt, fill: accent)[→],
  card([ACTOR], [256-unit GRU carries state from frame to frame.], highlighted: true),
  text(24pt, fill: accent)[→],
  card([ACT], [Forward, right and up velocity. Yaw rate.]),
)
#v(16pt)
#card([TRAINING ONLY], [A separate critic estimates returns from privileged simulator state. The actor runs without it.])
#v(13pt)
#secondary[The image encoder stays frozen. At 30 Hz, a low-level controller executes the actor’s commands.]

#pagebreak()
#title([03 · IMPLEMENTATION · PILOT], [Search, approach, then hold])
#panels("figures/pilot-rollout.png", ([0 s], [4 s], [11.5 s · successful hold]), width: 745pt)
#v(10pt)
#three(
  card([SEARCH], [Reward new visible surface before finding the target.]),
  card([APPROACH], [Reward distance progress once the target is found.]),
  card([HOLD], [Face the target from nearby. Avoid crashes and leaving bounds.]),
)
#v(8pt)
#secondary(size: 15pt)[Selected earlier visual-policy flight, not reward phase boundaries.]

#pagebreak()
#title([03 · IMPLEMENTATION · PILOT], [PPO learns from its own flights])
#three(
  card([1 · COLLECT], [32 parallel sim environments × 64 steps. Save actions, rewards and outcomes.]),
  card([2 · COMPARE], [Estimate whether actions did better or worse than the critic expected.]),
  card([3 · UPDATE], [Four PPO epochs. Clip the action-probability ratio in the objective; fit the critic and matcher.], highlighted: true),
)
#v(15pt)
#grid(columns: (190pt, 1fr), gutter: 18pt, align: horizon,
  image("figures/pilot-transition.png", width: 190pt),
  [
    #kicker[ONE TRANSITION · RECORDED EARLIER PILOT]
    #v(6pt)
    Camera at 5.5 s → action (+0.96, +0.08, −0.63, −0.21)
    #v(5pt)
    #secondary[Normalized forward, right, up, yaw rate. Reward +0.115. Episode continues.]
  ],
)
#v(22pt)
#secondary[Then collect fresh flights with the updated policy. The image encoder stays frozen throughout.]

#pagebreak()
#title([03 · IMPLEMENTATION · DATA LOOP], [Flights produce new vision examples])
#panels("figures/pilot-data-streams.png", ([Distillation RGB], [Simulator class labels], [Mined flight RGB]), width: 590pt)
#v(11pt)
#three(
  card([1 · RENDER], [Paired looks train the encoder. Simulator depth and classes train separate readouts.]),
  card([2 · FLY], [Freeze the encoder. Fit the pilot in sim, then fly it with fixed goal crops for mining.]),
  card([3 · LABEL], [Save RGB, goal crop, target box or absence, depth, and object IDs.], highlighted: true),
)
#v(11pt)
#text(20pt, weight: "bold")[Mined frames → fresh probes on held-out districts]
#v(7pt)
#secondary(size: 16pt)[Mined frames diagnose errors. The refinement shown here used sim and real anchors.]

#pagebreak()
#title([03 · IMPLEMENTATION · DATA LOOP], [A confident match can be wrong])
#grid(columns: (292pt, 1fr), gutter: 26pt,
  panels("figures/pilot-mined.png", ([Goal crop], [Recorded view]), width: 288pt),
  [
    #card([WHERE THE MODEL LOOKED], [Magenta: peak-scoring patch. Yellow: simulator target box, if visible.])
    #v(14pt)
    #card([WHAT THE LABELS REVEAL], [Top: wrong patch. Middle: false alarm when the goal is absent. Bottom: correct match.], highlighted: true)
    #v(15pt)
    #secondary[Examples, not error rates.]
  ],
)

#pagebreak()
#title([03 · IMPLEMENTATION · FLIGHT EVAL], [Better visual transfer did not fix flight])
#strip("figures/city-cameras.png", 104pt, 1280 / 530)
#v(13pt)
#let flight = json("presentation-flight-seed2.json")
#let count(arm, cohort, metric) = {
  let row = flight.at(arm).cohorts.at(cohort)
  str(row.at(metric)) + " / " + str(row.episodes)
}
#table(
  columns: (2fr, 1fr, 1fr),
  align: (left, right, right),
  inset: (x: 11pt, y: 7pt),
  stroke: (top: none, right: none, bottom: 0.65pt + rule, left: none),
  table.header([*Held-out simulator flights*], [*Original encoder*], [*Refined encoder*]),
  [No hint · success], count("mean", "standard_no_hint", "successes"), count("refined", "standard_no_hint", "successes"),
  [No hint · collision], count("mean", "standard_no_hint", "collisions"), count("refined", "standard_no_hint", "collisions"),
  [Hidden and far · success], count("mean", "hidden_far", "successes"), count("refined", "hidden_far", "successes"),
)
#v(12pt)
#secondary(size: 16pt)[Same PPO budget and warm start. One seed. Simulator-assisted goal-crop refresh still enabled.]

#pagebreak()
#title([04 · IMPLICATIONS], [Flight gives us harder camera views])
#panels("figures/textured-replay-flight.png", ([Onboard camera], [Chase camera]), width: 790pt)
#v(14pt)
#secondary[Both views re-rendered with Poly Haven textures and updated lighting.]
#v(9pt)
#secondary[Selected simulator success with a location hint. The flight path is unchanged.]

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

#pagebreak()
#title([APPENDIX · WAYPOINT EXPERIMENT], [Give the pilot a waypoint, not an image])
#align(center, box(inset: 13pt, radius: 6pt, fill: tint,
  text(20pt, weight: "bold")[Location command + estimated obstacle depth → PPO pilot]))
#v(18pt)
#table(
  columns: (2fr, 1fr, 1fr),
  align: (left, right, right),
  inset: (x: 11pt, y: 9pt),
  stroke: (top: none, right: none, bottom: 0.65pt + rule, left: none),
  table.header([*Held-out simulator flights*], [*Straight-line pursuit*], [*Waypoint PPO*]),
  [Reach and hold], [22 / 256], [*151 / 256*],
  [Collision], [204 / 256], [*74 / 256*],
)
#v(15pt)
#secondary[Same location hints and 8–30 m starts. One seed; not a test of object search or real flight.]
