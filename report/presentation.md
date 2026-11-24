# Presentation guide — Appearance-robust vision for drones

**Five-minute recording, 12 slides.** Walk through one frame from RGB to features, then show how those features were trained and tested. Keep the narration on the encoder; the flight reel is an example, not the headline result.

## Before you start

- Open the [slides](../runs/presentation/slides.pdf) full-screen and cue the silent [textured flight reel](../runs/presentation/pilot-demo-textured.mp4). Record the whole display or splice the MP4 into the recording at slide 11; a capture limited to the PDF window won’t show the player. Make a short test recording to check slide legibility, mic level, and the video cut.
- For a three-minute cut, use slides 1, 3, 5–6, 8, 10 and 12. Skip the reel or splice in a short excerpt. No live server or model needs to run.
- Keep the [report](report.pdf) and [supplement](supplement.pdf) available for questions. To rebuild the deck from the repo root: `nix develop --command typst compile report/slides.typ runs/presentation/slides.pdf` (create the output directory first if needed).

## What to say

### 1 · Question (0:00–0:18)

> A building is still the same building at noon and at dusk, but it looks different to the camera. We wanted a small encoder that keeps useful scene information when appearance changes, then tested whether that helped on real drone imagery.

### 2 · Data (0:18–0:38)

> These five images share one camera pose in our Blender city. We change lighting, materials and fog, not the geometry. The middle two columns show the kinds of looks we pair in training; the last two are held-out looks for evaluation.

### 3 · Architecture (0:38–1:08)

Trace the arrows left to right.

> A 320 by 240 frame gives us a grid of 20 by 15 patches, each 16 pixels square. Patch embedding turns those patches into vectors. Twelve transformer blocks let tokens exchange context across the image. At the output we keep one global token and 300 patch tokens, all 384-dimensional. This is the existing 22-million-parameter LingBot ViT-S backbone; our contribution is how we train it, not a new transformer architecture.

### 4 · Outputs (1:08–1:27)

> The CLS token is a summary of the frame. For global readouts we also pool the patch features. The full grid keeps the spatial layout, so a small probe can read depth or class at a particular location. The encoder itself doesn’t output boxes or flight controls.

### 5 · Distillation (1:27–1:52)

> The 300-million-parameter teacher stays frozen. Its features are 1024-dimensional, so two linear layers map the student’s outputs into the same space: one for the CLS token, one shared by all the patches. Cosine loss compares the direction of the projected features with the target. At inference we keep only the student backbone.

### 6 · Paired targets (1:52–2:18)

> Standard distillation matches each image to the teacher on that same image. Our variant averages the teacher’s normalized features across the two looks, then makes each student view match that shared target. The crop is shared so we average the same patch locations. Same data, same backbone, different supervision.

### 7 · Evaluation (2:18–2:40)

> We freeze the encoders. We fit small linear probes on canonical renders, then test what pose, depth and segmentation information survives under shifted looks. Retrieval asks a separate question: can a view find the same pose in another appearance, without fitting a readout?

### 8 · Simulator baselines (2:40–3:04)

> Follow the last two rows. Standard distillation already helps, but changing to paired targets raises retrieval from 0.747 to 0.951 and drops depth error from 0.242 to 0.201. DINOv3-L still leads on segmentation. The main improvement is robustness under new looks, not a sweep of every benchmark.

### 9 · Real-data refinement (3:04–3:25)

> The simulator gain didn’t automatically carry over to UAVid. We kept the paired simulator target and mixed in unlabeled real training images, matching each real frame to the teacher on that frame. We updated only the student’s last four blocks.

### 10 · UAVid results (3:25–3:50)

> A probe fit in simulation ranked car patches in separate UAVid validation videos. In the matched run, average precision rose from 0.604 to 0.749. The interval resamples videos, not patches, and the simulator checks stayed within the margins we set in advance. This is patch ranking, not bounding-box detection.

### 11 · Flight reel (3:50–4:36, including 27 seconds of video)

> A learned policy flies in simulation and supplies camera viewpoints for testing the encoder. These are selected successes with location hints. The left panel re-renders the recorded camera poses with Poly Haven textures; the right panel is the original external camera recording.

**Cut to `pilot-demo-textured.mp4`.** The first clip is held out; the memory-enabled second clip is from a training district. Then return to the deck. The [original reel](../runs/presentation/pilot-demo.mp4) shows the policy's actual camera input.

To regenerate the presentation visuals from the saved flights, run `nix develop --command .venv/bin/python scripts/render_presentation.py --figure report/figures/textured-flight.png --video --ffmpeg /path/to/ffmpeg` from the repo root with FFmpeg (including `drawtext` and `libx264`). This path doesn't change training or benchmark renders.

### 12 · Takeaway (4:36–5:00)

> Paired targets helped with unfamiliar simulator looks. Real RGB improved transfer. The flight frames exposed weaknesses in small-target and look-alike matching that the refinement didn’t fix. It didn’t establish a better pilot either. The result is a stronger visual representation, not reliable autonomous flight.

## Likely questions

- **What’s new versus domain randomization?** Both students see randomized images. The key comparison holds the views fixed and changes the **teacher target** from each view’s own features to the average across paired looks.
- **Did you use real labels to train the encoder?** No. The refinement sees unlabeled UAVid training RGB and teacher features. UAVid labels are for evaluation readouts; the car AP readout is fit in simulation and scored on real validation videos.
- **Why AP rather than only AUROC?** Car patches are sparse (8.6% of eligible patches). AP makes false positives near the top of the ranking more visible. The interval reflects the seven available validation videos, not training-seed uncertainty; refinement used one seed.
- **Why did you adopt a checkpoint the gate rejected?** We adopted it as the base encoder because real-image readouts improved and simulator checks stayed within their margins. It did not pass as a *flight-failure fix*: the goal-matching score rose by 0.016, but its paired interval (−0.005 to +0.038) includes zero.
- **Why not just use the teacher?** The student is about 14× smaller by parameter count. The result is about feature quality at that size; latency on the embedded target was not measured.
- **Is the demo representative?** No. It shows selected successes from earlier policies. On the later no-hint hard set, collisions rose from 80/256 to 116/256; both policies went 0/256 on hidden-and-far goals. The two runs also differed in perception-head data and training setup, so this does not isolate the encoder’s effect on flight.
