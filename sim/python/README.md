# GPU-resident quadrotor visual-goal learning

Task `city-drone-goal-v4`, ABI v7. A quadrotor in the imported city must fly to an object shown only as a goal crop and hold a standoff, or (explore command) cover as much of the city as it can. Targets are the city's own buildings, vehicles, trees and props. Native CUDA owns dynamics, sensors, rendering, appearance randomization, monocular camera effects, training labels and goal reward; Python owns learning, perception heads, spatial memory and evaluation. Flight commands come from the policy, not scripted search or stopping.

Use the root Python environment (`nix develop` from the repository root installs it) and the simulator's CUDA/Vulkan shell when launching native learning.

```sh
# Build outside the timed training process.
cd sim
nix develop --command cargo oxide build --arch sm_120 \
  --device-codegen-crate sim_physics,sim_env -- -p sim-env -p sim-smoke --profile training
cd ..

nix develop path:sim --command .venv/bin/python sim/python/train.py profile \
  --encoder mean --num-envs 64 --width 320 --height 240 \
  --library .sim-target/training/libsim_env.so --out runs/drone-profile

nix develop path:sim --command .venv/bin/python sim/python/train.py benchmark \
  --encoder mean --batch-sizes 64 128 --width 320 --height 240 \
  --library .sim-target/training/libsim_env.so --out runs/drone-benchmark

nix develop path:sim --command .venv/bin/python sim/python/train.py train \
  --encoder mean --num-envs 128 --eval-envs 256 \
  --encoder-precision fp16 --goal-refresh-steps 30 \
  --total-steps 2000000 --validation-every 500000 --wall-seconds 0 \
  --out runs/drone8-mean-hold-seed0

nix develop path:sim --command .venv/bin/python sim/python/train.py eval \
  --checkpoint runs/drone8-mean-hold-seed0/best.pt --split test \
  --library .sim-target/training/libsim_env.so --out runs/drone8-mean-hold-test

nix develop path:sim --command .venv/bin/python sim/python/train.py probe --out runs/drone-probe

nix develop path:sim --command .venv/bin/python sim/python/train.py memory-check \
  --encoder mean --num-envs 64 --eval-envs 384 --repeats 3 \
  --library .sim-target/training/libsim_env.so --out runs/drone-memory

nix develop path:sim --command .venv/bin/python sim/python/train.py verify-trace \
  --trace runs/drone-profile/trace.json
```

`--control rates` selects the body-rate interface (thrust, roll, pitch, yaw); `velocity` is the default.

Unit tests need no simulator: `cd sim/python && python -m unittest`.

## Task contract

- Control: velocity setpoints by default (PX4 offboard style). Actions are `[forward, right, up, yaw_rate]`, scaled by the native `action_scales` (velocity: 4.0, 4.0, 2.0 m/s and 1.5 rad/s). The native environment clamps the horizontal command to the unit disc. `--control rates` uses `[thrust, roll_rate, pitch_rate, yaw_rate]` with native scales `[MAX_THRUST/2, 3.5, 3.5, 2.0]`. The learner reads scales from native metadata and hardcodes none. The policy is a tanh-squashed Gaussian: the simulator receives `tanh(raw)` and rollouts store the pre-squash sample, whose log-probability includes the tanh Jacobian. (The v4 policy clipped an unbounded mean; once the mean left [-1, 1] its gradient vanished and failures sat at full yaw rate.)
- Hover action: zeros in velocity mode; `(-0.2, 0, 0, 0)` in rates mode (a fixed mild thrust offset; hover thrust is set by native).
- Yaw sign: `yaw_rate` positive is nose left (native convention). The GPS heuristic turns toward the target with this sign.
- Starts are 50% landed and 50% flying. Diagnostics index 19 records the start-landed flag.
- GPS: proprioception lanes 14–16 hold the GPS hint to the target box centre (right, up, −forward) in metres / 32, with a per-episode 3–15 m error; lane 18 is `gps_available`. With GPS denied the hint is zero. `--no-gps-fraction` sets the probability that an episode denies GPS; evaluation reports GPS available and GPS denied separately.
- Curriculum buckets are `visible`, `partial` and `hidden`. Start weights are (0.8, 0.15, 0.05) with floors (0.10, 0.10, 0.05). Adaptive promotion and demotion use per-bucket EMA success. Evaluation always uses uniform weights. The curriculum tensor is `[w_visible, w_partial, w_hidden, p_no_gps, appearance]`.
- Targets: placed objects of `assets/city/layout.json` of class building, vehicle, vegetation or prop whose box lies inside the split's district; vegetation and props must stand ≥2 m (box gap) from any object of their class so a crop identifies one object. Episodes draw the class uniformly, then the object.
- Success: within 6 m of the target box, speed ≤1 m/s, facing its unoccluded anchor, for **90 consecutive steps (3 s)**. The first such hold succeeds and ends the episode; the 1200-step (40 s) horizon otherwise fails it. A 15-step (0.5 s) hold is recorded separately as arrival. (ABI v6 required the hold at the 15 s deadline; 50 m starts could not finish.)
- Reward: progress in box distance, capped below 3 m; −0.01 per step; near-and-facing braking credit `0.1 × clamp(1 − speed/3, 0, 1)`; +10 terminal success or −10 any failure, including timeout. No novelty, action-smoothness, body-rate or yaw-rate reward terms. Explore-command episodes (`--explore-fraction`) replace this with `0.02 × newly observed surface tiles − 0.01 per step − 10 on crash or leaving the area`, with no timeout penalty.
- Episodes end on success, collision, out-of-bounds or horizon. Outcomes are exclusive: success > collision > out-of-bounds > timeout. Interactive free flight remains continuous and reports a success pulse after a 3 s hold; no controller forces that hold.
- Start distance: the native default is 8–30 m from the target box; `--start-distance 8 60` sets the range for training, validation and test.
- Camera effects (appearance strength > 0, per episode): radial distortion, vignette, motion blur over the exposure from the body rates, rolling shutter. Monocular only.
- Training labels (never actor input): field 17 `patch_depth` float32[n, 300] (planar depth of the patch's object samples), field 18 `patch_object` int32[n, 300] (layout object most of the patch's 16 samples hit, from depth and layout boxes).
- Catalog: 65,536 episodes per split (each bucket ≥512); starts 8–30 m from the target box. Each episode fixes a goal reference camera: the (levelled) start view in half of the visible-start episodes — the operator taps the object in the first frame — otherwise a level camera 5–25 m from the box at 1.5–12 m height (drone-like; bird's-eye references made unrecognizable goals) that sees at least half of it. The goal crop is the box's projection grown 15% per side and squared; the object must span ≥8% of the frame and the crop ≤80%, otherwise the view is rejected, so every goal shows one recognizable object. Splits use styles `TRAIN=0`, `VALIDATION=1`, `TEST=2` (disjoint districts), with validation seed 10001 and test seed 20002.
- Appearance: a post-render CUDA kernel randomizes, per catalog entry, material hue/saturation/value, sky gradient, depth fog, exposure, white balance, contrast, gamma and sensor noise (`--appearance` strength, default 1; 0 is the plain render). An episode and its goal view share one appearance. Sun direction and shadows are fixed.

## Policy

Frozen encoder tokens are computed once per control tick and reused by every PPO epoch. Visual arms use `TargetSaliency` over patch tokens `[N, 300, D]` (320×240 input, 16 px patches): each patch scores a goal-free linear term plus a low-rank (rank 64) token × goal bilinear term, where the goal is `concat(CLS, mean patch)` of the episode's goal crop. Goal embeddings are built once per environment: the native `render_goals` renders every catalog entry's reference view, the learner crops it (`features.crop_goals`, the same crop any operator or detector box would use) and encodes it into a `[65536, 2D]` bf16 table (~50 s at 128 envs); each step gathers the row of `env.catalog_index`. A "not in view" logit from CLS and goal completes a softmax over 301 classes. The actor receives the saliency-expected target NDC (2), the peak patch logit (1), P(in view) (1), saliency-pooled tokens (64) and a projected CLS + 2×3 pooled context (128), then proprioception (19).

Optional `--goal-refresh-steps 30`: once per second, crop the current camera view at the simulator's visible target box and re-encode it. Keep the last goal when hidden or outside the catalog's recognizable crop-size range; reset to the new catalog goal at every episode boundary. This changes perception inputs only: no action override, scripted search, reward, horizon or curriculum changes. PPO stores each step's actual goal embedding so optimization replays the same inputs. The refresh preserves recurrent memory.

Training and validation with refresh use **oracle simulator boxes**, not Gemma. The interval and box source are recorded in checkpoints and evaluation results; catalog replay uses the recorded interval (its thumbnail is the initial goal). Grounded/operator replays keep their supplied goal fixed. Final test reports both refreshed-box and fixed-initial-goal performance; neither measures autonomous language search. Live deployment needs perception-produced boxes for refresh.

Fresh 2M-step hold trial: `nix develop path:sim --command .venv/bin/python sim/python/train.py train --encoder mean --encoder-precision fp16 --num-envs 128 --eval-envs 256 --goal-refresh-steps 30 --total-steps 2000000 --wall-seconds 0 --validation-every 500000 --out runs/drone8-mean-hold-seed0`. The step budget rounds up to complete PPO rollouts. Validation is persisted after each check. Controls, PPO hyperparameters, 8–30 m training starts and curriculum are unchanged.

Surface coverage is diagnostic only. A GPU kernel samples an 8×6 depth grid and marks visible world x/z hits in a 128×128 city bitset; sky, reference renders and revisits do not add tiles. ABI v6 field 16 is `episode_diagnostics`, `float32[n,3]`: current unique tiles, ended-episode unique tiles, and ended-episode consecutive hold steps. Terminal diagnostics survive autoreset. No coverage reward or exploration flag remains.

`train.py eval-cohorts --checkpoint <v3-policy.pt> --baseline-checkpoint <v2-policy.pt> --out <results.json>` compares identical seeded standard and hidden-only 8–30 m episodes, far 35–50 m episodes, and hidden-plus-far episodes (256 per GPS mode). All policies face the new final-3-second hold objective; old early-arrival scores are not comparable. Baseline transfer is explicit and visual-only, with encoder, sensors, observations and flight dynamics still validated. Fixed reference crops are the default; `--goal-refresh-steps 30` instead uses oracle boxes. Reports distinguish arrival, final success, post-arrival held-frame fraction, and losing a hold after arrival, alongside crashes, stationary-away fraction and coverage. Far starts are outside the training range but reachable within 15 s.

Arms (`--encoder`):

| Arm | Actor input | Notes |
|---|---|---|
| `mean` (default) | visual, `run:runs/mean` | paired-consistency encoder |
| `pretrained` | visual, `lingbot:small` | off-the-shelf encoder |
| `baseline` | visual, `run:runs/baseline` | standard-distilled encoder |
| `state` | privileged (29) + proprioception | privileged ceiling; not visual evidence |
| `hint` | proprioception only | no-vision baseline |

Encoder inference (`--encoder-precision`, recorded in the checkpoint's `preprocessing`; `eval` and replay read it back):

| Precision | Path | Encoder µs/img (batch 128, RTX 5070 Ti) | Patch cosine to FP32 (mean / min) |
|---|---|---|---|
| `fp16` (default) | TensorRT 11 FP16 engine, strongly typed ONNX export (RoPE stays FP32) | 190 | 1.00000 / 0.99966 |
| `bf16` | `torch.compile` under bf16 autocast; numerics of pre-TensorRT checkpoints | 278 | 0.99997 / 0.99948 |
| (previous eager bf16) | removed | 602 | 0.99997 / 0.99912 |

The engine is built once per encoder fingerprint, input size, GPU and TensorRT version (about 20 s) and cached in `$XDG_CACHE_HOME/visenc/tensorrt`; batches above 256 run in chunks. `torch.compile` needs the `sim` shell's `TRITON_*` paths.

`train.py benchmark --encoder mean` (64 rollout steps, collect + update):

| Path | Transitions/s at 64 envs | Transitions/s at 128 envs | Encoder ms/step at 128 |
|---|---|---|---|
| eager bf16 (previous) | 1,413 | 1,334 | 76.8 |
| `bf16` (`torch.compile`) | 2,164 | 2,391 | 34.8 |
| `fp16` (TensorRT) | 2,949 | 2,987 | 24.2 |

The critic always receives privileged (29) + proprioception (19). The actor is a GRU over the trunk output, with a tanh-squashed Gaussian head whose log-std is state-independent (initial -0.5). Memory resets at episode starts. Terminal GAE never bootstraps across a done flag. The entropy bonus is a reparameterized estimate of the squashed distribution's entropy; the pre-squash Gaussian entropy grows without bound and drove the first v5 run into saturated bang-bang actions.

The auxiliary loss (visual arms only, `--aux-coefficient`, default 0.5) is cross-entropy over those 301 classes against a soft target: uniform over the patches whose centres fall inside the target box's projection (privileged lanes 25–28, always including the anchor patch) when privileged lane 14 says the target is visible, otherwise "not in view". A single anchor patch was an arbitrary label for large objects (offline within-one-patch localization 0.38). The actor never receives these privileged labels. The learning curve logs `aux_patch_accuracy`: the fraction of visible frames whose peak patch lies on the target's projection.

Native tensors are borrowed through DLPack on the native CUDA stream. Reset, step, and rollout never call `.item()`, `.cpu()`, or boolean-mask indexing; readbacks happen only at end-of-rollout and evaluation boundaries.

## Evaluation

Validation and final tests run in both GPS modes: GPS available (`no_gps=0`) and GPS denied (`no_gps=1`). Each mode uses the first completed episode of each environment and must reach at least 256 completed episodes (`MIN_EPISODES`); `--eval-envs` defaults to 384. Reported per mode:

- success with a Wilson 95% interval, collision, out-of-bounds, timeout;
- `mean_steps` (all episodes) and `mean_success_steps` (conditional on success, null when none succeed);
- per-bucket and start-landed/flying breakdowns.

Blind controls run on the same test seed and style: `hover` and `random` (uniform [-1, 1]), plus `gps` (a pure-pursuit controller on the GPS hint) in velocity mode. The `gps` heuristic is not defined in rates mode. Validation selects `best.pt`: validation GPS success when `no_gps_fraction` is 0, otherwise the mean of both modes. Validation skips the blind controls. The reported test uses `best.pt`.

## Localization probe

`train.py probe` is the offline goal-localization gate run before RL. Each frame is the start view of an independent catalog episode (8,192 train-split, 4,096 test-split frames; test districts are held out), shared across encoders. Each frame's goal is its catalog entry's goal-crop embedding. A fresh `TargetSaliency` is trained on each frozen encoder with the policy's saliency loss (identical seed, 20 epochs, lr 1e-3). It reports visibility AUROC/AP, exact-patch and within-one-patch localization and pixel error of the predicted patch, overall and per target class; `--appearance` sets the randomization strength. Frames from correlated random-action rollouts let probes memorize (≈90% train vs 25% test patch accuracy) and are not used; an earlier learned-position query attention head also memorized and was replaced.

Measured on ABI v4 (designed 1.5–2.5 m shapes, held-out designs; `runs/drone-probe-saliency`, test split):

| Encoder | Visibility AUROC | Exact patch | Within 1 patch | Median error |
|---|---|---|---|---|
| `pretrained` (LingBot ViT-S) | 0.862 | 0.539 | 0.821 | 8.4 px |
| `baseline` (standard distilled) | 0.894 | 0.629 | 0.861 | 7.6 px |
| `mean` (paired consistency) | 0.897 | 0.620 | 0.856 | 7.7 px |

## Failure mining and perception post-train

1. Simple reward: `(d_prev − d) + 0.1·in_zone − 1·crash/OOB`, γ ≈ 0.997 for the 15 s horizon; sweep the three coefficients.
2. Wrong-target mining: a frozen policy in large env batches flags frames whose high-confidence saliency peak or approach lands on a non-target object (renderer object IDs), then snapshots sim state, frame, goal crop and distractor ID.
3. Perception post-train: hard-negative (target vs. mined distractor) training of `TargetSaliency`, optionally LoRA on the last ViT blocks anchored to the frozen encoder.
4. RL restarts from a fraction of mined states.

**Paper comparison (encoder study):** run step 3 identically for `pretrained`, `baseline` and `mean`, using the same mined failure set. Report before vs. after post-train on held-out mined confusions and the localization probe, plus out-of-domain retention (real-image segmentation, the paper's existing metrics), so the study shows whether hard-negative post-training helps each encoder equally and whether it costs appearance robustness. The mined set comes from one policy's failures, so report the mining policy and keep the evaluation confusions held out (different districts and seeds).

### Results (first round)

Mining (`train.py mine`, `runs/mined-drone6`): `drone6` best policy (`mean` encoder, fixed catalog goals), 256 envs, one labelled sample per 0.5 s. On the held-out validation and test districts (7,680 unbiased samples each), the policy was confidently "in view" on a wrong patch in 10% of samples and confidently "in view" while the target was not visible in 31–37%. Training set: 15,665 frames from train districts (12,992 failures, 2,673 correct).

Post-train (`train.py post-train`): `runs/mean` last 4 ViT blocks + the policy's saliency head (`runs/posttrain-mean-b4`), or the head alone (`runs/posttrain-mean-head`). Loss: policy saliency loss on mined frames (half failures) + cosine distillation to `lingbot:large` through the frozen projection heads on UAVid train and `train-v2` renders. 3,000 steps of 32 frames (~7 min); best validation at step 500 for both.

Held-out test (7,680 samples, separate district). Visible = peak patch on the target when visible; absent = P(in view) < 0.5 when not visible; score = their mean.

| Encoder + head | Visible on target | Absent correct | Score |
|---|---|---|---|
| `mean` + original policy head | 0.645 | 0.448 | 0.547 |
| `mean` + post-trained head | 0.650 | 0.836 | 0.743 |
| post-trained encoder + head | 0.641 | 0.833 | 0.737 |

Confusion probe (`train.py confusion-probe`; fresh head per frozen encoder, same 8,192 mined training frames, same seed and schedule, same test set). `global` = the policy's head (goal = one CLS + mean-patch vector); `patch` = `PatchMatchSaliency` (goal = CLS + the goal crop's patch grid pooled to 6×8; each frame patch scores by its soft-max cosine match to the goal patches, and "not in view" sees the best match). Visible on target / absent correct / score:

| Encoder | `global` head | `patch` head |
|---|---|---|
| `lingbot:small` (off the shelf) | 0.493 / 0.844 / 0.668 | 0.621 / 0.932 / 0.776 |
| `lingbot:large` (teacher) | 0.538 / 0.833 / 0.685 | 0.741 / 0.896 / 0.819 |
| `baseline` (standard distillation) | 0.610 / 0.811 / 0.711 | 0.722 / 0.915 / 0.819 |
| `mean` (paired consistency) | 0.624 / 0.836 / 0.730 | 0.706 / 0.915 / 0.810 |
| `mean` post-trained (4 blocks), first run | 0.624 / 0.766 / 0.695 | — |

UAVid retention (`visenc real-eval`, 7 val sequences): post-trained vs `mean` sim→real mIoU 0.511 vs 0.516, car AUROC 0.860 vs 0.903, car AP 0.533 vs 0.604 (wide intervals).

Findings:
- Most mined failures were the policy head's **calibration**, not the encoder: it reported the target in view when it was not. Retraining the head on mined frames moves absent-correct 0.45 → 0.84; visible localization barely moves (0.645 → 0.650).
- **Encoder post-training added nothing** with this recipe: the joint result matches head-only, the frozen-probe score drops, and real car metrics drop. Not adopted.
- **The goal representation was the bottleneck.** Patch matching lifts every encoder by 0.08–0.13 score, raising both visible localization and absent rejection. The policy now uses it (`--goal patch`, the default).
- With the single-vector goal, both distilled students looked better than their teacher (0.730 / 0.711 vs 0.685). **With patch matching that gap closes** (teacher 0.819 = `baseline` 0.819 ≥ `mean` 0.810; teacher best at visible on-target, 0.741). The earlier "student beats teacher" reading came from the students' summary vector, not finer discrimination. Single seed, no intervals; frames within an episode are correlated.

### Gemma grounding benchmark

`gemma_goal.py bench --mined runs/mined-drone6/eval-set/test`: 150 held-out frames with the target visible plus 50 without. Each frame gets "Fly to the <asset name>." for its true target (e.g. "Fly to the van.", "Fly to the building."); boxes are scored against the simulator's projected target box (`runs/gemma-bench/bench-x1.json`). Gemma 4 E4B-it, 320×240, 0.8 s per answer:

| Visible: boxed | Visible: IoU ≥ 0.3 | Mean IoU | Absent: false box |
|---|---|---|---|
| 127 / 150 | 94 / 150 | 0.38 | 29 / 50 |

By family, hits are cars 41/49, buildings and trees 49/81, street props 4/20 (lights and planters, often < 2% of the frame). Box centres are unbiased (mean offset +0.02, −0.01), so image coordinates are read correctly; boxes are ~0.55× the truth area because the truth is the whole object's projected bounds, including occluded parts. A 2× upscale changed nothing (94 / 150, 30 / 50 false). Misses are mostly another instance of the named class ("the building" with several in view, "the large tree"), and when the target is out of view Gemma still boxes a look-alike 58% of the time: one-frame grounding cannot say "not here".



## Perception, spatial memory, counting and the agent (ABI v7)

**Perception heads** (`sim_rl/perception.py`, `train.py perception`): a small conv head on the frozen 15×20 token grid predicts per-patch log depth, a "has geometry" logit and the object class (none, building, vegetation, vehicle, prop). Supervision is the native label fields (never actor input). A ground-plane prior from the drone's own altimeter height and gravity direction (each patch ray's distance to the ground, its elevation, whether it reaches the ground) is concatenated to the tokens. Frames come from exploratory Ornstein–Uhlenbeck flights at 8–60 m. `runs/perception-v4` (16,384 train / 3,072 held-out frames, 16 epochs):

| Encoder | Depth abs-rel | δ<1.25 | Class mIoU | Vehicle IoU |
|---|---|---|---|---|
| `lingbot:small` | 0.204 | 0.718 | 0.714 | 0.720 |
| `baseline` | 0.173 | 0.786 | 0.743 | 0.763 |
| `mean` | 0.178 | 0.775 | 0.732 | 0.747 |

Without the ground prior `mean` reaches 0.214 / 0.70.

**Spatial memory** (`sim_rl/memory.py`): a 128×128 top-down map at 1.5 m (±96 m) per environment, in the episode's start frame. Pose is dead-reckoned from the noisy gyro and body-velocity sensors (heading rate of the levelled forward axis, not ω·up, which drifted 0.2 rad in 10 s); measured position error 0.06 m median after 10 s. Each patch ray is cast to its predicted depth: the hit cell takes the goal evidence (the policy's saliency, P(this patch rather than not in view)), cells part-way along the ray are marked looked-at; the newest look at a cell replaces its evidence. With labelled depth a seen target lands within 3 m of its box every time; with predicted depth about half the time (range error, bearing exact). The actor reads a 32×32 heading-aligned crop (±48 m) through a small CNN plus the strongest candidate (forward, right, distance, evidence); an auxiliary loss decodes the target's body-frame offset from the memory features. `--memory-heads runs/perception-v4/mean.pt` enables it; the heads are frozen inside the checkpoint.

**Counting** (`sim_rl/tracks.py`, `train.py count-eval`): patches classified vehicle within 35 m are clustered per frame (≥2 patches), associated to tracks with a range-dependent gate (3 m + 0.25 × the larger of the observation's and the track's effective range; 1/range weights), merged when tracks converge, and confirmed after 3 frames. On 128 held-out 15 s exploratory flights the count is close to the number of cars seen in ≥3 frames (mean 1.70 vs 1.77), mean absolute error 0.76, exact 46%, within one 83%; flights that saw 6+ cars undercount (3.5 vs 6.5, two flights).

**Explore command** (`--explore-fraction`): a share of training episodes gets the explore command instead of the goal; the actor and critic take a one-hot command, the goal input and the memory's goal evidence are zero, and the reward is newly observed surface tiles (native coverage counter) minus a step cost and the crash/leave penalty. Validation reports the explore command's mean observed tiles alongside goal success.

**Agent** (`sim/python/agent.py`, live play with `--gemma` and a memory checkpoint): a typed instruction starts a loop in which Gemma sees the camera frame and a text summary of the memory (area looked at, confirmed cars with offsets, goal evidence) and answers `explore`, `go` (a box in the current frame, which becomes the goal crop) or `answer`. It replans every 5 simulated seconds or when a skill ends; simulated time waits while Gemma thinks. Switching goals keeps the map and the car tracks.

**Gemma grounding post-train** (`sim/python/gemma_tune.py`): exploratory-flight frames with per-patch object labels give "Fly to the [nearest] <name>." → box examples and absent names → `[]`; LoRA on the language model's projections; `eval` compares base and tuned hit and false-box rates on held-out districts.

## Patch-goal run (ABI v6)

`runs/drone9-mean-patch-seed0`: `drone8`'s exact settings (fresh seed 0, TensorRT FP16 `mean`, 128 envs, oracle refresh every 30 steps, 2M steps, sustained-hold task, unchanged reward) with one change: `--goal patch`, the patch-matching saliency head. 2,007,040 steps in 1409 s. Validation GPS success at 0.5M / 1M / 1.5M / 2M: 4.3% / 5.5% / 14.5% / 12.5%; `best.pt` = 1.5M.

Paired fixed-goal cohorts (`eval-cohorts`, test seed 20002, 256 episodes per cell, no oracle refresh), success % GPS / no GPS:

| Cohort | `drone9` patch (1.5M) | `drone8` global (1.0M) | `drone6` global, ABI 5 (3.0M) |
|---|---|---|---|
| standard | 14.5 / 12.1 | 2.7 / 2.3 | 12.1 / 10.9 |
| hidden | 2.0 / 0.8 | 0.4 / 0.4 | 2.7 / 0.8 |
| far | 1.6 / 2.7 | 0.4 / 0.4 | 1.6 / 0.4 |
| hidden+far | 0 / 0 | 0 / 0 | 0 / 0 |

At the same budget and task the patch goal is a 5× gain over `drone8` on standard episodes (the matched comparison). It matches `drone6`, which trained 3M steps on the easier early-arrival task. Hidden and hidden+far stay near zero: a target out of view at the start is a search-and-memory problem, not a matching one.

## Sustained-hold experiment (ABI v6)

`runs/drone8-mean-hold-seed0`: fresh seed-0 visual policy, frozen TensorRT FP16 `mean` encoder, 128 envs, oracle goal-box refresh every 30 steps, unchanged flight limits, PPO settings, 8–30 m starts and adaptive curriculum. Completed 2,007,040 steps in 1371 s (22.9 min including initialization, validation and standard test). Validation selected `best.pt` at 1,007,616 steps; `last.pt` retains the full-budget policy.

Scores require the new final-3-second hold, not ABI v5's early 0.5-second arrival. Validation uses oracle simulator boxes, not Gemma:

| Training steps | GPS success % | No-GPS success % |
|---|---|---|
| 507,904 | 0.39 | 0.00 |
| 1,007,616 | 6.25 | 6.25 |
| 1,507,328 | 2.34 | 3.52 |
| 2,007,040 | 3.12 | 3.52 |

The complete standard test, with refreshed-box and fixed-goal modes, is in `final-test.json`. Paired fixed-goal results use the selected policy and explicitly transfer both previous visual policies into the new task, preserving their encoder/observation/control contracts. Different selected training budgets and one seed make this a practical checkpoint comparison, not a causal reward ablation.

Paired fixed-goal test, 256 episodes per cell. All policies use the same new success rule. Percentages below are observed rates; Wilson intervals and complete diagnostics are in `cohorts-fixed-goal.json`.

| Cohort | GPS mode | Previous non-exploration, 3.01M | Exploration, 2.01M | New hold, selected 1.01M |
|---|---|---|---|---|
| Standard, 8–30 m | available | 10.94% | 1.17% | 2.73% |
| Standard, 8–30 m | denied | 8.59% | 0.39% | 2.73% |
| Hidden, 8–30 m | available | 1.95% | 0.39% | 0.78% |
| Hidden, 8–30 m | denied | 1.17% | 0.39% | 0.78% |
| Far, 35–50 m | available | 2.34% | 0.00% | 0.39% |
| Far, 35–50 m | denied | 0.39% | 0.00% | 0.00% |
| Hidden + far | available | 0.00% | 0.00% | 0.00% |
| Hidden + far | denied | 0.00% | 0.00% | 0.00% |

Standard-cohort GPS behavior. Arrival means a 0.5 s hold; held-frame fraction and ever-lost-hold rate are conditional on arrival. A policy may lose the hold, reacquire it, and still succeed at the deadline.

| Policy | Arrived / 256 | Post-arrival held frames | Ever lost hold | Stopped-away frames | Crash |
|---|---|---|---|---|---|
| Previous non-exploration | 30 | 94.7% | 6.7% | 26.9% | 38.7% |
| Exploration | 12 | 60.2% | 58.3% | 7.9% | 35.5% |
| New hold | 15 | 49.1% | 73.3% | 37.8% | 21.1% |

This retrain is **not a navigation or stopping improvement over the previous non-exploration policy**. It crashes less but stops away from the target more, and its post-arrival holding is worse. The small success increase over exploration is not evidence of a reliable fix. Hidden/far navigation remains poor, and these held-out cohorts still contain no skyscraper targets. Keep the previous non-exploration policy as the comparison baseline; retain this run's validation-selected `best.pt` for experimenting, not promotion.

The standalone standard test and separate paired invocation differed by one successful episode per GPS mode; the table reports the paired invocation. Do not infer a precise improvement from differences of a few episodes.

Verification: native CUDA build; 23 passing scene/state host tests (2 additional scene tests ignored); 34 existing Python tests and two task-transfer regressions; real 2,048-transition visual PPO collect/update with finite losses; 64 native hover episodes all timed out at step 450 with the failure penalty. The live trial was opened in a browser and its checkpoint/3-second criterion labels visually checked.

## Bounded exploration experiment (ABI v5)

`runs/drone7-mean-explore-seed0` (historical ABI v5): fresh seed-0 camera policy, TensorRT FP16 `mean` encoder, 128 envs, reference-goal refresh every 30 steps, unchanged 8–30 m training starts and adaptive curriculum. The removed novelty reward paid +0.05 per newly observed tile after the initial view, capped at +2 per episode. Completed 2,007,040 steps in 1452 s (24.2 min including initialization, validation and standard test). Standard validation GPS success at 0.5M / 1M / 1.5M / 2M was 0.8% / 10.2% / 8.6% / 3.1%; `best.pt` is the 1,007,616-step checkpoint, while the predeclared 2M cohort comparison uses `last.pt`. Standard fixed-goal test success of `best.pt` was 7.4%.

Paired test catalogs (seed 20002), 256 episodes per policy/cohort/GPS mode, fixed goal references and **no oracle refresh or novelty reward at evaluation**. Baseline is the previous 3,006,464-step `runs/drone6-mean-refresh-seed0/best.pt`; training budgets differ, so this is a practical checkpoint comparison, not a matched-budget causal ablation. Wilson 95% intervals:

| Cohort | GPS mode | Previous success % [95%] | Exploration success % [95%] |
|---|---|---|---|
| Hidden, 8–30 m | available | 1.95 [0.84, 4.49] | 1.17 [0.40, 3.39] |
| Hidden, 8–30 m | denied | 1.56 [0.61, 3.95] | 1.17 [0.40, 3.39] |
| Far, 35–50 m | available | 4.30 [2.42, 7.53] | 0.39 [0.07, 2.18] |
| Far, 35–50 m | denied | 3.52 [1.86, 6.55] | 0.39 [0.07, 2.18] |
| Hidden + far | available | 0.39 [0.07, 2.18] | 0.00 [0.00, 1.48] |
| Hidden + far | denied | 0.39 [0.07, 2.18] | 0.00 [0.00, 1.48] |

Behavioral diagnostics, GPS available. Stopped-away = fraction of active episode frames at speed <0.2 m/s and >6 m from the target box; new tiles exclude the initial view:

| Cohort | Previous stopped-away % | Exploration stopped-away % | Previous new tiles | Exploration new tiles |
|---|---|---|---|---|
| Hidden | 39.7 | 10.2 | 102.8 | 166.0 |
| Far | 25.8 | 8.4 | 144.5 | 233.0 |
| Hidden + far | 44.9 | 11.5 | 96.8 | 179.1 |

The intended behavioral change occurred: much less stopping and more observed area. Reaching success did not improve, and far-target success fell. Do not promote this checkpoint as a navigation improvement. One seed and unequal budgets limit attribution. These district-based cohorts still contain no skyscraper targets; they test hidden/range difficulty, not skyscraper-family generalization. Full metrics: `cohorts-fixed-goal.json`.

Verification: native CUDA build; 34 Python tests; 7 scene host tests (2 ignored); real GPU smoke verified initial/revisit/goal-render/terminal zero novelty, the +2 budget, finite PPO update, old-checkpoint base-contract loading, and hidden-only 35–50 m starts.

## Refreshed-goal run (ABI v5)

`runs/drone6-mean-refresh-seed0`: fresh camera policy, `mean` encoder with TensorRT FP16, seed 0, 128 envs, goal refresh every 30 steps. Reward, horizon, start-distance range and curriculum settings are unchanged from v5. Requested budget was 6M steps without a wall cutoff; stopped at 5,185,536 steps after validation regressed, retaining `best.pt` from 3,006,464 steps.

Validation GPS success at 1M / 2M / 3M / 4M / 5M was 6.6% / 14.5% / 15.6% / 14.8% / 8.6%. No-GPS was 4.7% / 12.1% / 12.9% / 12.9% / 11.3%. The curriculum remained at 85% visible starts. Later training did not improve held-out validation; this is one seed, not evidence for a universal 3M-step cutoff.

Held-out test of the selected 3M checkpoint, 256 episodes per GPS mode, uniform visibility buckets and randomized appearance:

| Goal input | GPS success [Wilson 95%] | No-GPS success [Wilson 95%] | GPS crash | GPS timeout |
|---|---|---|---|---|
| Refreshed, simulator boxes | 0.156 [0.117, 0.206] | 0.152 [0.113, 0.201] | 0.363 | 0.391 |
| Fixed initial reference | 0.117 [0.083, 0.162] | 0.090 [0.061, 0.131] | 0.348 | 0.438 |

These refresh results use oracle simulator boxes, not an LLM detector. They do not establish language-guided search when the object has never been seen. Fixed-goal evaluation removes the additional box observations. Compared with the earlier 2.54M-step v5 run, the training budget and encoder precision also differ; improvements cannot be attributed solely to refresh. Results are in `final-test.json`, validation history in `validation.json`.

## ABI v5 results (city-object targets)

Camera + `mean` encoder (bf16), velocity control, seed 0, 128 envs, 2.54M steps in the 3300 s budget (49.5 min incl. startup, validation and test; `runs/drone5-mean-seed0`). Held-out test district, 256 episodes per GPS mode, uniform visibility buckets, appearance randomization on; Wilson 95% intervals.

| Actor | GPS success | Visible / partial / hidden | No-GPS success | Crash (GPS) | Out of bounds | Timeout |
|---|---|---|---|---|---|---|
| Camera + `mean` encoder | 0.070 [0.045, 0.108] | 0.12 / 0.06 / 0.02 | 0.078 [0.051, 0.118] | 0.46 | 0.12 | 0.35 |
| Scripted GPS pursuit | 0.070 | — | 0.000 | 0.64 | — | — |
| Hover / random | 0.000 / 0.000 | — | 0.000 / 0.000 | 0.00 / 0.16 | — | — |

By class (GPS): buildings 0.086, vegetation 0.096, vehicles 0.059, props 0.033. Validation success rose 0.031 → 0.055 → 0.086 at 1.0M / 2.0M / 2.54M steps, so the run stopped at its budget, not a plateau. Training-episode success reached 0.36, but the adaptive curriculum never left 85% visible starts (promotion needs 0.7 visible success), while evaluation draws the buckets uniformly. With GPS the camera policy only ties the scripted pursuit; without GPS it keeps 0.078 where the pursuit scores 0, so its target finding is visual. Crashes are the dominant failure. No spinning remains: a 1M-step replay had no frames at full yaw rate (v4 failures: 82–98%).

Earlier v5 attempts, kept for the record: the Gaussian entropy bonus drove saturated bang-bang actions (entropy 3.7 → 4.1, success → 0); single-patch anchor labels and bird's-eye, full-frame goal crops kept localization near chance (offline within-one-patch 0.38; box-region labels and drone-like crops reach peak-on-target 0.58 offline, 0.85–0.9 in training); and an all-or-nothing hold bonus left the policy circling found targets at ~2.6 m/s (`runs/drone5-mean-seed0-binaryhold`, partial).

Live play: `nix develop --command nix develop path:sim --command .venv/bin/python sim/python/live.py --checkpoint runs/drone8-mean-hold-seed0/best.pt [--gemma]` serves http://127.0.0.1:8790/. Drag a box or type an instruction with `--gemma` to supply a goal crop. Free flight ignores district bounds, timeouts and success termination; a crash or leaving the city/100 m ceiling restarts. No automatic goal refresh runs in live play.

Live diagnostics: white camera outline = simulator target box (when its anchor is visible); red patch = the policy's localization guess, drawn when P(in-view) exceeds 0.5; thumbnail = actual goal image. Blue map marker = drone. The panel shows goal source/age, geometric box-match IoU, simulator asset/object ID/class/dimensions, selection/current box distance, anchor visibility, policy P(in-view), patch-in-GT-box agreement and checkpoint. A box (drawn or from Gemma) that frames no city object is rejected and the previous goal stays. IoU and patch-in-box agreement do not establish semantic object identity. Ground truth is diagnostic only, never an added input to the visual actor.

Skyscraper coverage limitation: applying the current district/object-pool rules to `assets/city/layout.json` yields 27 eligible skyscraper instances in training, zero in validation and zero in test. Eligibility is before start/reference-view rejection; it does not measure actual catalog frequency. The reported aggregate held-out success therefore does not evaluate skyscraper navigation, especially distant live selections outside the 8–30 m training-start range.

## ABI v4 results (designed targets; superseded task)

Velocity control, one seed, held-out test split. 256 episodes per GPS mode, uniform visibility buckets; Wilson 95% intervals. Wall times include initialization, training and evaluation on an RTX 5070 Ti shared with an unrelated GPU workload (not controlled). The camera arm stopped at its wall budget with fewer steps than the controls.

| Actor | Steps | Minutes | GPS success | Visible / partial / hidden | No-GPS success | Crash (GPS) |
|---|---|---|---|---|---|---|
| Camera + `mean` encoder | 1.98M | 49.5 | 0.449 [0.389, 0.510] | 0.55 / 0.61 / 0.22 | 0.430 [0.371, 0.491] | 0.25 |
| GPS hint + sensors only (learned) | 6.00M | 34.5 | 0.199 [0.155, 0.252] | 0.23 / 0.33 / 0.07 | 0.078 [0.051, 0.118] | 0.48 |
| Privileged target state (ceiling) | 6.00M | 34.7 | 0.805 [0.752, 0.849] | 0.91 / 0.87 / 0.64 | 0.746 [0.689, 0.796] | 0.18 |
| Scripted GPS pursuit | — | — | 0.043 | 0.02 / 0.09 / 0.03 | 0.000 | 0.42 |

The camera policy was trained only with GPS available; its no-GPS score is zero-shot. Results are single-seed and budget-limited, not convergence claims; the camera arm does not yet include the `pretrained`/`baseline` encoder comparison.

### Language goals via Gemma 4 on ABI v4 (no retraining)

Gemma 4 E4B-it chose the goal: it saw the 8 held-out reference images as numbered options plus one instruction and answered an option number (bf16 on GPU, 5.6 GB per-layer embedding table kept on CPU; ~0.75 s per answer). The chosen design's embedding replaced the true goal; the camera policy above flew 256 test episodes per condition. (Code: commit `b32da5c`, `sim/python/gemma_goal.py`; ABI v5 replaced design picking with grounding, below.)

| Goal source | Pick accuracy (8) | Success |
|---|---|---|
| True goal | 1.00 | 0.449 [0.389, 0.510] |
| Gemma, colour only ("Go to the green one") | 0.75 | 0.418 [0.359, 0.479] |
| Gemma, riddle ("…shaped like a donut") | 0.75 | 0.367 [0.311, 0.428] |
| Gemma, full name ("Fly to the checkered green octahedron") | 0.625 | 0.344 [0.288, 0.404] |
| Wrong design (ablation) | 0 | 0.121 |

Every Gemma miss confused two designs that share colour and pattern but differ in shape (octahedron/torus, cube/cylinder). Colour-only instructions are ambiguous in this split (two green, two blue designs) but scored higher than full names.

## Checkpoints

Format `sim-drone-ppo-v3`. A checkpoint records task/ABI, controls, encoder fingerprint and preprocessing, native task contract and library hash (provenance only), seeds and PPO settings. Normal loading requires task v3/ABI6. `eval-cohorts --baseline-checkpoint` alone permits explicit visual-actor comparison from task v2/ABI5: reward and success semantics may differ, but flight dynamics, observations and encoder configuration must match. Privileged-state actors cannot transfer because hold-progress normalization changed.

## Replay and reports

`scripts/render_navigation.py --checkpoint runs/drone-mean/best.pt [--no-gps-fraction 1]` renders one first-episode rollout (`rollout.gif`, `contact.png`, `path.png`, `path.json`, `rollout.json`). `rollout.json` records control mode, action names, GPS mode and outcome. `scripts/show_navigation.py --runs runs/drone-mean runs/drone-pretrained ...` writes `report/city-drone-goal.typ` and `.json` from completed runs only, with GPS-available and GPS-denied columns. Both reject results with fewer than 256 episodes per mode or a different task or ABI.

Replay export (`sim_rl.replay`, entry `sim/python/replay.py --checkpoint ... --out ... --num-envs 32 --split test`) writes format `sim-city-replay-v3`: a manifest (including `control_mode`, `action_names`, `no_gps_fraction`, `appearance`, and per environment `gps_available`, `target_class`, layout `object` and the goal crop `env-XXX-goal.png`) plus per-environment states (pose, target box centre and half extents, catalog index, target class and object, clock, landed, and the anchor labels) and transitions (`action_names` followed by `reward, done, success, collision, out_of_bounds`), trimmed to the first done frame. `--target-class NAME` keeps episodes whose target has that layout class. The site shows the goal crop beside the controls.

Language goals by grounding (`sim/python/gemma_goal.py`): `frames` renders the first-person start frames of a fresh held-out batch; `ground --instruction TEXT` asks Gemma 4 to box the instructed object in each frame and matches the box to the target-class city object whose projected bounding box overlaps it most (IoU ≥ 0.3; occlusion is not modelled). `replay.py --goal-boxes DIR/boxes.json` then flies the matched episodes: the goal is the crop of the start frame at Gemma's box (the crop an operator tap would give) and the native `retarget` call points GPS and success at the matched object. Boxes matching no city object are not flown (they would give the policy GPS to one object and an image of another). `bench` scores grounding on mined frames with known targets (see "Gemma grounding benchmark").

Typed goals in the replay site: start the server inside the root dev shell with `nix develop --command .venv/bin/python sim/replay/serve.py --data <recording> --live [--checkpoint runs/drone8-mean-hold-seed0/best.pt]`. Each instruction runs frames → ground → record on `--live-envs` (default 32) held-out environments with a fresh random seed and switches the page to the recording; episodes where Gemma boxed nothing are dropped. Recordings have their own `/replay/<recording>/` URLs.

## Resource notes

- Rollout token memory is about 944 MB at 64 environments, D=384, bf16.
- Replay token memory is about 450 × n × 300 × D × 2 bytes; host RGB adds about 1.66 GB at 16 environments.
- Profiling verifies zero host transfers and host waits inside `sim.collect` and `sim.ppo_update` scopes; `verification.json` records the verdict.
