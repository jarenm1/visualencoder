# Presentation guide — Appearance-robust vision for drones

**Five-minute recording, 12 slides.** Tell the story in four parts. Use the cues below to practice; don’t read them into the mic.

| Rubric question | Slides | One answer to land |
|---|---|---|
| What was the problem? | 1–2 | Lighting, materials, and fog can change the pixels without changing the scene. We wanted a small encoder whose features stay useful under that shift. |
| What’s the solution? | 3–4 | Use a 22M-parameter ViT that keeps a scene summary and a grid of local features. Train it with a large teacher; deploy only the student. |
| How did you solve it? | 5–10 | Average teacher features across aligned simulator looks, test against standard distillation, then add unlabeled real images and measure transfer separately. |
| What are the implications? | 11–12 | Better features for vision tasks at the same student size; simulated flight exposes failure cases, but doesn’t establish real-world navigation or onboard latency. |

## Before you start

- Open the [slides](../runs/presentation/slides.pdf) full-screen. Cue the silent [flight reel](../runs/presentation/pilot-demo-textured.mp4) for slide 11. Capture the whole display or splice in the MP4; a capture limited to the PDF window won’t show the player.
- Make a 15-second audio test in the same recorder. Play it back with headphones: listen for clipping, fan noise, and dropped words. Then do one timed take and trim the reel before speeding up the method or the results.
- For a three-minute cut, use slides 1, 3, 5–6, 8, 10 and 12. Skip the reel or splice in a short excerpt. No live server or model needs to run.
- Keep the [report](report.pdf) and [supplement](supplement.pdf) available for questions. To rebuild the deck from the repo root: `nix develop --command typst compile report/slides.typ runs/presentation/slides.pdf` (create the output directory first if needed).

## On-camera cues

### Problem · 0:00–0:38

- **1 · 0:00.** Open with the technical question: “Can a small encoder tell this is the same scene when the pixels look different?”
- **2 · 0:18.** Point to the same roof or road marking across the renders. Same pose and geometry; different lighting, materials, and fog. Transition: “That gives us a controlled way to train and test invariance.”

### Solution · 0:38–1:27

- **3 · 0:38.** Follow the architecture arrows: 320×240 RGB → 16×16 patches → twelve transformer blocks → normalized features. Say that the backbone already exists; the contribution is the training recipe.
- **4 · 1:08.** Contrast one 384-dimensional CLS token for the whole frame with 300 local tokens for spatial readouts. The encoder does not output boxes or flight controls. Transition: “We want the student to keep that information when appearance changes.”

### Implementation · 1:27–3:50

- **5 · 1:27.** Point to the frozen teacher and the trainable student. The two linear heads match the student’s 384-dimensional CLS and patch features to the teacher’s 1024-dimensional space for cosine loss. None of that runs at inference.
- **6 · 1:52.** Trace separate arrows, then converging arrows. Same images, different targets: average the teacher’s normalized features from the two aligned looks. Shared cropping keeps patches aligned.
- **7 · 2:18.** Freeze the encoders. Fit linear probes on canonical renders and test on shifted looks. Retrieval asks whether the same pose is the nearest match under a different look, with no trained probe.
- **8 · 2:40.** Point to the last two rows: retrieval 0.747 → 0.951; depth error 0.242 → 0.201. Don’t read the table. DINOv3-L still leads on segmentation.
- **9 · 3:04.** Transition: “But robustness in simulation didn’t guarantee transfer.” Keep the paired simulator target, add unlabeled UAVid RGB, and update the last four blocks and final LayerNorm.
- **10 · 3:25.** A simulator-fit probe ranks car patches on separate UAVid videos: AP 0.604 → 0.749, with a paired interval over videos. Patch ranking, not object detection.

### Implications · 3:50–5:00

- **11 · 3:50.** Explain what the reel shows, then play it (~27 seconds). These are selected successes with location hints. Left: Poly Haven re-render of the saved camera poses. Right: original external view. The first clip is held out; the second is from a training district. Flight frames also provide harder views for a fresh probe on frozen features.
- **12 · 4:36.** End on the distinction: paired views improved simulator robustness; real RGB improved transfer. The 22M-parameter encoder needs no teacher at runtime, but onboard latency was not measured and the new pilot didn’t reliably improve. Stop by 5:00.

The [original reel](../runs/presentation/pilot-demo.mp4) shows the policy’s actual input. To regenerate the textured visuals without changing benchmark renders: `nix develop --command .venv/bin/python scripts/render_presentation.py --figure report/figures/textured-flight.png --video --ffmpeg /path/to/ffmpeg` (requires `drawtext` and `libx264`).

## Likely questions

- **What’s new versus domain randomization?** Both students see randomized images. The key comparison holds the views fixed and changes the **teacher target** from each view’s own features to the average across paired looks.
- **Did you use real labels to train the encoder?** No. The refinement sees unlabeled UAVid training RGB and teacher features. UAVid labels are for evaluation readouts; the car AP readout is fit in simulation and scored on real validation videos.
- **Why AP rather than only AUROC?** Car patches are sparse (8.6% of eligible patches). AP makes false positives near the top of the ranking more visible. The interval reflects the seven available validation videos, not training-seed uncertainty; refinement used one seed.
- **Why did you adopt a checkpoint the gate rejected?** We adopted it as the base encoder because real-image readouts improved and simulator checks stayed within their margins. It did not pass as a *flight-failure fix*: the goal-matching score rose by 0.016, but its paired interval (−0.005 to +0.038) includes zero.
- **Why not just use the teacher?** The student is about 14× smaller by parameter count. The result is about feature quality at that size; latency on the embedded target was not measured.
- **Is the demo representative?** No. It shows selected successes from earlier policies. On the later no-hint hard set, collisions rose from 80/256 to 116/256; both policies went 0/256 on hidden-and-far goals. The two runs also differed in perception-head data and training setup, so this does not isolate the encoder’s effect on flight.
