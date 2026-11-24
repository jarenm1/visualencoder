// Build from the repo root: typst compile report/slides.typ runs/presentation/slides.pdf
#set document(title: "Appearance-robust vision for drones", author: "Jaren McHugh")
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

#kicker[VISUAL ENCODER RESEARCH]
#v(45pt)
#text(64pt, weight: "bold")[Appearance-robust \ vision for drones]
#v(23pt)
#secondary(size: 25pt)[Can a small encoder recognize the same scene \ through changes in lighting, textures, and fog?]
#v(34pt)
#text(22pt, weight: "bold", fill: accent)[300M-parameter teacher → 22M-parameter student]
#v(28pt)
#secondary[Jaren McHugh]

#pagebreak()
#title([01 · DATA], [One pose, different appearances])
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
#secondary[Train on paired renders. Test on unseen poses and appearances.]

#pagebreak()
#title([02 · ARCHITECTURE], [What runs on each frame])
#image("figures/encoder-architecture.svg", width: slide-width)
#v(18pt)
#secondary[22M parameters. No teacher or distillation heads at inference.]

#pagebreak()
#title([03 · OUTPUTS], [A scene summary and a spatial grid])
#v(28pt)
#grid(columns: (1fr, 1fr), gutter: 16pt,
  card([ONE GLOBAL TOKEN], [A summary of the frame. Useful for scene retrieval and pose probes.], highlighted: true),
  card([300 PATCH TOKENS], [One feature vector per patch location. Useful for depth, segmentation, and goal matching.]),
)
#v(25pt)
#secondary[Self-attention lets each token use context from the rest of the image.]

#pagebreak()
#title([04 · DISTILLATION], [Teach the student to match features])
#v(22pt)
#grid(columns: (1fr, 1fr), gutter: 16pt,
  card([FROZEN TEACHER], [LingBot ViT-L/16 produces 1024-dimensional target features.]),
  card([TRAINABLE STUDENT], [Two linear heads project 384 → 1024: one for CLS, one shared across patches.], highlighted: true),
)
#v(25pt)
#text(23pt, weight: "bold")[Cosine loss on the global token and corresponding patches.]

#pagebreak()
#title([05 · PAIRED TARGETS], [Same images. Different teacher target.])
#v(28pt)
#grid(columns: (1fr, 1fr), gutter: 16pt,
  card([STANDARD], [Look A matches teacher A. \ Look B matches teacher B.]),
  card([PAIRED], [Both looks match the mean of teacher A and teacher B.], highlighted: true),
)
#v(25pt)
#secondary[Same pose and shared crop, so patch locations line up across the pair.]

#pagebreak()
#title([06 · EVALUATION], [What do the frozen features retain?])
#v(28pt)
#card([LINEAR PROBES], [Fit on canonical renders. Read out pose, depth, and segmentation under shifted looks.])
#v(18pt)
#card([RETRIEVAL], [Given one view, find the same pose under a different look. No fitted probe.], highlighted: true)

#pagebreak()
#title([07 · SIMULATOR RESULTS], [Paired targets improve robustness])
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
#title([08 · REAL-DATA REFINEMENT], [Add real RGB without dropping simulator pairs])
#v(26pt)
#grid(columns: (1fr, 1fr), gutter: 16pt,
  card([SIMULATOR], [Keep the paired teacher target.]),
  card([UAVID TRAINING RGB], [Match the teacher on each unlabeled real frame.], highlighted: true),
)
#v(25pt)
#secondary[Update only the last four student blocks. Evaluate on separate UAVid videos.]

#pagebreak()
#title([09 · REAL-IMAGE RESULTS], [Unlabeled real images improved transfer])
#secondary(size: 22pt)[Paired student → refined student, matched run]
#v(36pt)
#text(76pt, weight: "bold", fill: accent)[0.604 → 0.749]
#v(7pt)
#text(26pt, weight: "bold")[Car-patch average precision on held-out UAVid videos]
#v(17pt)
#secondary[Paired 95% interval for the gain: +0.095 to +0.180. One refinement seed.]
#v(38pt)
#secondary[Simulator readouts stayed within prespecified margins. This is patch ranking, not box detection.]

#pagebreak()
#title([10 · LEARNED FLIGHT], [The pilot supplies harder camera views])
#image("figures/textured-flight.png", width: 550pt)
#v(18pt)
#text(22pt, weight: "bold")[Play the 27-second replay: `pilot-demo-textured.mp4`]
#v(11pt)
#secondary[Two selected approaches with location hints. Textured re-render and recorded external camera.]
#v(19pt)
#text(20pt, weight: "medium")[Frozen camera features + image goal + motion inputs → recurrent PPO pilot]
#v(8pt)
#secondary[Recorded flights supply test frames for a fresh goal-matching probe.]

#pagebreak()
#title([11 · TAKEAWAY], [What improved, and what didn’t])
#v(16pt)
#card([WHAT WORKED], [Paired targets improved simulator robustness. Real RGB improved transfer.], highlighted: true)
#v(18pt)
#card([WHAT’S OPEN], [Small targets and look-alikes still fail. Held-out navigation did not reliably improve.])
#v(22pt)
#secondary[No-hint city hard set: 28/256 vs. 24/256 successful flights (original vs. refined pilot).]
