# Presentation guide — Visual encoders for sim-to-real transfer

**Five-minute recording, 12 slides.** Tell the story in four parts. Use the cues below to practice; don’t read them into the mic.

| Rubric question | Slides | One answer to land |
|---|---|---|
| What was the problem? | 1–2 | Vision learned in simulation may not transfer to real cameras. Appearance is one part of that gap we can isolate and measure. |
| What’s the solution? | 3–4 | Transfer the visual representation: a 22M-parameter encoder produces global and spatial features that perception heads and a pilot can use. |
| How did you solve it? | 5–10 | Average teacher features across aligned simulator looks, test against standard distillation, then add unlabeled real images and measure transfer separately. |
| What are the implications? | 11–12 | A compact encoder can supply visual information for physical AI. We measured transfer on real images, not autonomous flight on real hardware. |

## Before you start

- Open the [slides](../runs/presentation/slides.pdf) full-screen. Cue the silent [flight reel](../runs/presentation/pilot-demo-textured.mp4) for slide 11. Capture the whole display or splice in the MP4; a capture limited to the PDF window won’t show the player.
- Make a 15-second audio test in the same recorder. Play it back with headphones: listen for clipping, fan noise, and dropped words. Then do one timed take and trim the reel before speeding up the method or the results.
- For a three-minute cut, use slides 1, 3, 5–6, 8, 10 and 12. Skip the reel or splice in a short excerpt. No live server or model needs to run.
- Keep the [report](report.pdf) and [supplement](supplement.pdf) available for questions. To rebuild from the repo root: `nix develop --command typst compile --root . report/slides.typ runs/presentation/slides.pdf` (create the output directory first if needed).

## On-camera cues

### Problem · 0:00–0:38

- **1 · 0:00.** Open with sim-to-real: “Simulation gives us data we can label and control. The problem is that what a drone learns to see there may not carry over to a real camera. Can we transfer the visual features instead?”
- **2 · 0:18.** Point to the same roof across the renders. We isolate one piece of the gap: change appearance, keep geometry and pose fixed. We test on real imagery separately.

### Solution · 0:38–1:27

- **3 · 0:38.** Focus on the encoder, not the flight policy. Follow the arrows: 320×240 RGB → 16×16 patches → twelve transformer blocks → normalized features. The architecture already exists; the contribution is the training recipe.
- **4 · 1:08.** One 384-dimensional global token summarizes the frame; 300 local tokens retain spatial detail for depth, segmentation, and goal matching. This is the visual information a physical AI system can use; the encoder itself doesn’t choose actions.

### Implementation · 1:27–3:50

- **5 · 1:27.** Point to the frozen teacher and the trainable student. The two linear heads match the student’s 384-dimensional CLS and patch features to the teacher’s 1024-dimensional space for cosine loss. None of that runs at inference.
- **6 · 1:52.** Trace separate arrows, then converging arrows. Same images, different targets: average the teacher’s normalized features from the two aligned looks. Shared cropping keeps patches aligned.
- **7 · 2:18.** Freeze the encoders. Fit linear probes on canonical renders and test on shifted looks. Retrieval asks whether the same pose is the nearest match under a different look, with no trained probe.
- **8 · 2:40.** Point to the last two rows: retrieval 0.747 → 0.951; depth error 0.242 → 0.201. Don’t read the table. DINOv3-L still leads on segmentation.
- **9 · 3:04.** Transition: “But robustness in simulation didn’t guarantee transfer.” Keep the paired simulator target, add unlabeled UAVid RGB, and update the last four blocks and final LayerNorm.
- **10 · 3:25.** Point to the zero line. In the fresh seed, simulator retrieval stayed near 0.96 and held-out UAVid car-patch AP rose from 0.550 to 0.734. The earlier seed also improved (0.604 to 0.749). Neither seed showed a clear gain in goal matching. The last row tests frozen features, not whether the drone reached the goal.

### Implications · 3:50–5:00

- **11 · 3:50.** Explain what the reel shows, then play it (~27 seconds). These are selected successes with location hints. Left: Poly Haven re-render of the saved camera poses. Right: original external view. The first clip is held out; the second is from a training district. Flight frames also provide harder views for a fresh probe on frozen features.
- **12 · 4:36.** Trace the branches: “The same frozen patch features let us read out depth and semantics, transfer a sim-trained car readout to real video, and feed a goal-conditioned pilot in simulation. The real-video result is for the car readout, not the pilot. Next I’d test whether memory helps across frames and measure latency on hardware.” Stop by 5:00.

The [original reel](../runs/presentation/pilot-demo.mp4) shows the policy’s actual input. To regenerate the textured visuals without changing benchmark renders: `nix develop --command .venv/bin/python scripts/render_presentation.py --figure report/figures/textured-flight.png --video --ffmpeg /path/to/ffmpeg` (requires `drawtext` and `libx264`).

## Likely questions

- **What’s new versus domain randomization?** Both students see randomized images. The key comparison holds the views fixed and changes the **teacher target** from each view’s own features to the average across paired looks.
- **Did you use real labels to train the encoder?** No. The refinement sees unlabeled UAVid training RGB and teacher features. UAVid labels are for evaluation readouts; the car AP readout is fit in simulation and scored on real validation videos.
- **Why AP rather than only AUROC?** Car patches are sparse (8.6% of eligible patches). AP makes false positives near the top of the ranking more visible. The interval reflects the seven available validation videos, not training-seed uncertainty. Both seeds were evaluated on those same videos.
- **Did refinement pass the full gate?** No. Both seeds improved real-image readouts without a clear gain in the patch goal-matching score. The fresh seed also missed the guardrails for held-out-region depth and global goal matching. This is a transfer result for specific readouts, not a claim that every feature improved.
- **Why not just use the teacher?** The student is about 14× smaller by parameter count. The result is about feature quality at that size; latency on the embedded target was not measured.
- **Is the demo representative?** No. It shows selected successes from earlier policies. On the later no-hint hard set, collisions rose from 80/256 to 116/256; both policies went 0/256 on hidden-and-far goals. The two runs also differed in perception-head data and training setup, so this does not isolate the encoder’s effect on flight.

## Retraining comparison (not in the timed talk)

Run `nix develop --command bash scripts/retrain_presentation.sh` from the repo root. The seed-2 run writes to `runs/presentation-retrain-seed2/`; use a new output path as the script’s first argument for a rerun. To start PPO after both sets of heads are ready and before either policy has saved a best checkpoint, pass `--policy-only` before the output path.

- See the [seed-2 gate verdict](presentation-retrain-seed2.json) for all sim, real-image, and offline goal-matching checks. These are new checkpoints on the same held-out data, not a new test split.
- Compare `policy-mean/hard-set.json` and `policy-refined/hard-set.json` for no-hint success and collisions. Both start from the same policy checkpoint, train for 2M additional steps with fresh heads fit on the same frames, and use the same held-out episode cohorts. This compares the frozen encoders with their own perception heads under one PPO recipe; it is not evidence of real-world flight.
