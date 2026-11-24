# Presentation guide — Visual encoders for sim-to-real transfer

**22 main slides plus a waypoint-results appendix.** The full recording runs about 6:30. If the cap is five minutes, use the short cut below and keep the rest for questions. The encoder is the main result; use the policy to show where those features help and where they fail.

| Rubric question | Slides | One answer to land |
|---|---|---|
| What was the problem? | 1–2 | Vision learned in simulation may not transfer to real cameras. Appearance is one part of that gap we can isolate and measure. |
| What’s the solution? | 3–5 | A 22M-parameter encoder gives a global summary and spatial tokens. Frozen patch features support different readouts without retraining the encoder for every task. |
| How did you solve it? | 6–19 | Pair teacher targets across aligned looks, refine on unlabeled UAVid RGB, then use frozen features in PPO. Flown frames with simulator labels become new evaluation examples. |
| What are the implications? | 20–22 | The real-image readout transferred. Simulated navigation is a separate, harder test; it does not establish autonomy on hardware. |

## Before you start

- Open the [slides](../runs/presentation/slides.pdf) full-screen. Cue the silent [flight reel](../runs/presentation/pilot-demo-textured.mp4) for slide 21. Capture the whole display or splice in the MP4.
- Make a 15-second audio test in the same recorder. Play it back with headphones: listen for clipping, fan noise, and dropped words. Then do one timed take and trim the reel before speeding up the method or the results.
- **Five-minute cut:** Show slides 1–5, 7, 9–10, 12–13, 15, 17–20, and 22. Aim to reach slide 12 by 2:25 and slide 18 by 4:00. Skip the reel or run 8 seconds over the closing narration. The full deck also works offline.
- Keep the [report](report.pdf) and [supplement](supplement.pdf) available for questions. To rebuild from the repo root: `nix develop --command typst compile --root . report/slides.typ runs/presentation/slides.pdf` (create the output directory first if needed).

## On-camera cues

### Problem and solution · 0:00–1:18

- **1 · 0:00.** “Simulation gives us data we can control and label. The trouble is that what a drone learns to see there doesn’t necessarily carry over to a real camera.”
- **2 · 0:15.** Point to the same roof across the renders: appearance changes; geometry and camera pose don’t. This isolates one part of the gap.
- **3 · 0:30.** Follow 320×240 RGB → 16×16 patches → twelve transformer blocks. The contribution is the training recipe, not a new ViT.
- **4 · 0:48.** Point at one cell. The grid is 15×20, and each cell stores a 384-dimensional vector. Attention mixes whole-image context; it’s not a bag of isolated tiles. The right panel shows a three-component PCA projection, not segmentation. The CLS token is the global summary.
- **5 · 1:03.** Point out that near and far surfaces, and classes such as building and vegetation, are read from the same frozen vectors. Separate heads turn features into named quantities; individual token channels don’t have fixed names. This is a qualitative probe example, not an accuracy figure.

### Train and evaluate the encoder · 1:18–3:18

- **6 · 1:18.** The student has two training-only linear heads, one for CLS and one shared across patches. They align 384-dimensional features with the frozen teacher's 1024-dimensional targets by cosine loss.
- **7 · 1:33.** Trace the separate and converging arrows. Both students see the same rendered looks and crops. Only the teacher target changes: average normalized teacher features at the same position.
- **8 · 1:54.** Freeze the encoders. Fit simple readouts on canonical views, test other looks. Retrieval uses no fitted probe: return the same pose as the nearest feature match.
- **9 · 2:08.** Point to standard versus paired-view distillation: retrieval 0.747 → 0.951, depth error 0.242 → 0.201. DINOv3-L still leads on segmentation.
- **10 · 2:26.** Keep paired simulator anchors. Match the teacher on unlabeled UAVid training frames. Update the last four blocks and final LayerNorm.
- **11 · 2:41.** Point to a car location in the held-out UAVid example. Both heads were fit on sim; the color scale is identical. The scores rank car patches, not calibrated probabilities or precise boxes.
- **12 · 2:56.** Point to zero. Fresh-seed UAVid AP rose 0.550 → 0.734, echoing 0.604 → 0.749 on the earlier seed and same videos. Sim retrieval stayed roughly flat. Neither seed showed a clear gain on the patch goal probe.

### The RL policy and data loop · 3:18–5:35

- **13 · 3:18.** Follow the reference crop to the live frame to the heatmap. A shared 64-dimensional learned projection supports soft-max cosine matching against goal patches. Localization has 300 patch scores plus a not-in-view option. The example is a recorded policy frame, not a post-hoc attention map.
- **14 · 3:36.** Predicted depth projects matched patches into top-down memory. A separate channel tracks where the camera looked. The crop is centered on the drone, forward up, and aligned using dead-reckoned motion. Unlike the GRU’s hidden state, this memory has spatial coordinates.
- **15 · 3:55.** The visual actor combines pooled features and goal evidence, spatial memory, a coarse obstacle-depth grid, onboard state and optional location hint, plus goal/explore command. The 256-unit GRU outputs four velocity/yaw commands. The critic sees privileged simulator state only during training.
- **16 · 4:14.** Explain reward shaping: search for new surfaces, then make distance progress, then face and hold near the target. The goal is not to touch the object. Explore mode rewards coverage instead of reaching an object.
- **17 · 4:31.** Roll out concurrently, use generalized advantage estimation, and fit a clipped PPO actor plus value loss. An auxiliary box/visibility loss trains the matcher; when the look-alike mask is enabled, same-class mismatches skip policy and value loss but still train the matcher. The visual encoder stays frozen.
- **18 · 4:51.** Distinguish controlled pairs for distillation, labeled exploratory sim frames for perception heads, and policy-flown episodes for mining. The fixed-crop replay saved 6,144 labeled frames each from training and test districts, sampling every 25 steps and keeping normal cases too. Those data fed fresh probes; the showcased encoder refinement used sim plus real anchors, not the mined failures.
- **19 · 5:14.** Show the wrong peak and false-view example. A fresh probe on frozen features tests whether the visual representation supports the right answer, separately from whether a pilot can execute a maneuver.

### Implications · 5:35–6:30

- **20 · 5:35.** “Better real-image scores didn’t fix the pilot. With matched training budgets, the two policies reached the goal in only four and five of 256 no-hint episodes. Both collided in over half, and neither solved a hidden-and-far episode.” No hint removes the location hint, not the simulator-assisted crop refresh.
- **21 · 5:51.** Play roughly 20 seconds of the reel. These are selected successes with location hints, not a random sample. Left is a re-render of saved camera poses; right is the original external view.
- **22 · 6:14.** Trace the three uses. “One encoder gives geometry in simulation, useful car scores on real video, and a visual input to a simulated pilot. Transfer of a readout doesn’t imply transfer of a controller.” Close with better temporal association and hardware latency as next tests.

The [original reel](../runs/presentation/pilot-demo.mp4) shows the policy’s actual input. To regenerate the textured visuals without changing benchmark renders: `nix develop --command .venv/bin/python scripts/render_presentation.py --figure report/figures/textured-flight.png --video --ffmpeg /path/to/ffmpeg` (requires `drawtext` and `libx264`).

The patch PCA, saliency, memory and rollout stills come from `replay-drone11-train101`, episode 15, a selected v4 success in a training district. The saliency overlay is relative within the frame; the separate in-view estimate at 8 s was 0.824. The reward description is the newer v5 recipe, not a label on each v4 screenshot.

## Likely questions

- **What’s new versus domain randomization?** Both students see randomized images. The key comparison holds the views fixed and changes the **teacher target** from each view’s own features to the average across paired looks.
- **Did you use real labels to train the encoder?** No. The refinement sees unlabeled UAVid training RGB and teacher features. UAVid labels are for evaluation readouts; the car AP readout is fit in simulation and scored on real validation videos.
- **Why AP rather than only AUROC?** Car patches are sparse (8.6% of eligible patches). AP makes false positives near the top of the ranking more visible. The interval reflects the seven available validation videos, not training-seed uncertainty. Both seeds were evaluated on those same videos.
- **Did refinement pass the full gate?** No. Both seeds improved real-image readouts without a clear gain in the patch goal-matching score. The fresh seed also missed the guardrails for held-out-region depth and global goal matching. This is a transfer result for specific readouts, not a claim that every feature improved.
- **Why not just use the teacher?** The student is about 14× smaller by parameter count. The result is about feature quality at that size; latency on the embedded target was not measured.
- **Is the demo representative?** No. It shows selected successes from earlier policies. In the matched retrain, the no-hint collision counts were 143/256 and 140/256. The real-image AP result and the simulated flight outcomes measure different things.

## Retraining comparison (not in the timed talk)

Run `nix develop --command bash scripts/retrain_presentation.sh` from the repo root. The seed-2 run writes to `runs/presentation-retrain-seed2/`; use a new output path as the script’s first argument for a rerun. To start PPO after both sets of heads are ready and before either policy has saved a best checkpoint, pass `--policy-only` before the output path.

- See the [seed-2 gate verdict](presentation-retrain-seed2.json) for all sim, real-image, and offline goal-matching checks. These are new checkpoints on the same held-out data, not a new test split.
- See the [matched seed-2 flight outcomes](presentation-flight-seed2.json) for all nine 256-episode cohorts. The policies start from the same checkpoint, train for 2,000,896 additional steps with fresh heads fit on the same frames, and face the same held-out episode cohorts. This compares frozen encoders with their own perception heads under one PPO recipe. The one-success difference on the no-hint hard set is not evidence that refinement improved flight.

## Optional slide 23 · Waypoint experiment

Use this instead of the reel if you want to end the policy section with the new experiment. “We took object recognition out of the pilot’s goal. With the same noisy location command, waypoint PPO with estimated depth succeeded on 151/256 simulator episodes, versus 22/256 for straight-line pursuit.”

The [test output](../runs/waypoint-depth-seed2/final-test.json) covers 8–30 m starts and simulator-supplied hints with 1–5 m uncertainty; no goal image or crop refresh. Validation selected the checkpoint at 253,952 steps from a 1,003,520-step run. This is a single-seed comparison of two controllers, not an encoder ablation or a head-to-head comparison with slide 20’s no-hint image-goal task.
