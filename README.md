# visualencoder

Sim-to-real at the visual-encoder level: distil a large vision foundation model (LingBot-Vision by default, or DINOv3) into a small ViT. Paired simulated appearances isolate appearance robustness; independent experiments test geometry generalization, unlabelled-real adaptation, and downstream navigation.

## Setup

```sh
nix develop          # uv-managed venv (torch 2.14.1+cu130, sm_120), raylib, jupyterlab
hf auth login         # only for DINOv3 (gated: request access at huggingface.co/facebook); LingBot-Vision is ungated
```

## Pipeline

```sh
scene/fetch_assets.sh                                                    # CC0 Kenney kits + Poly Haven textures -> assets/
blender -b --python scene/build_city.py -- --assets assets/kenney --out assets/city
visenc data --out data/train --seed 1 --count 20000 --exclude-region 0 0 245 245
visenc data --out data/eval  --seed 2 --count 3000
visenc distill --data data/train --views random1 random2 --pair-target mean --out runs/mean
visenc eval --data data/eval --heldout-region 0 0 245 245 --out reports/x \
  --encoders pixels lingbot:small run:runs/mean
```

`scripts/baselines.sh` runs the full sweep. It adds the DINOv3 runs automatically once your account has access.

Encoder specs (teacher, baselines, eval): `pixels`, `lingbot:<small|base|large|giant>`, `lingbot-random:<variant>`, `hf:<id>`, `random:<hf id>`, `run:<dir>`. Students: `lingbot:<variant>` or `hf:<id>`.

### Scene

`scene/build_city.py` builds one city in Blender from CC0 Kenney kits (commercial, suburban, roads, nature, cars). The city is 490 m across: downtown towers, suburbs, parks, and a forest ring so cameras near the edge still see a horizon. It is exported as one GLB per semantic class (`ground road building vegetation vehicle prop`, plus `sky` = 0), along with `layout.json`, which holds the solid objects' AABBs. Geometry is fixed and seeded. Each object's vertex colour carries a per-object variant (R) and surface kind (G: foliage, bark, pavement), which the renderer uses to pick textures.

Surfaces get 1k CC0 Poly Haven PBR textures (albedo and normal), applied in world space (triplanar) because Kenney meshes only have palette UVs. Roofs, dirt and bark take the photo colour. Walls, roads and props keep their Kenney palette hue and gain the texture's pattern. Grass and foliage keep a natural green, because Poly Haven's grass and leaf photos are dry and brown. The sun casts shadows (a 4096² shadow map over the 320 m square ahead of the camera).

### Data

`visenc data` (`src/visenc/city.py`, raylib) varies only two things:

- **Pose**: position anywhere in the city at 2–45 m height, kept clear of solid objects by at least 1.5 m; any yaw; pitch −60°..15°; roll ±20°; vertical FOV 55–75°.
- **Appearance**: sun direction, colour and intensity; sky and fog; ambient; exposure; colour tint; which Poly Haven texture each surface role uses (wall, roof, asphalt, pavement, grass, dirt, foliage, bark, metal, concrete). `canonical` is one fixed look; `random1..K` are seeded per pose.

Each pose stores RGB per appearance view (2× supersampled), plus semantic classes, optical-Z depth and a state vector: position, yaw sin/cos, pitch, roll, vfov, clearance to the nearest solid object, and log depth straight ahead.

## Distillation

The teacher is a frozen encoder: [LingBot-Vision](https://huggingface.co/collections/robbyant/lingbot-vision) ViT-L/16 by default (Apache-2.0, a DINOv3-style ViT with registers and RoPE, pretrained with masked boundary modeling for dense spatial features), or DINOv3 via `--teacher hf:facebook/dinov3-…`. The student defaults to LingBot-Vision ViT-S/16, initialised either pretrained or random. The loss is cosine distillation on the CLS token and on every patch token. All views of a pose share one crop and flip, so the patch grids stay aligned. The student input can also get photometric corruption.

- `--pair-target own`: each view distils the teacher's features of itself.
- `--pair-target mean`: every view distils the teacher features averaged over views, so the student learns one representation per physical state.
- `--extra-images DIR_OR_NPY`: mixes in unlabelled real frames, distilled with `own` targets. Prepared `data/uavid/train_rgb.npy` uses the same 4:3 crop as evaluation and never opens masks.

## Evaluation

All probes are linear ridge regressions on frozen features. Each probe is fit on the `canonical` view and tested on every view of held-out poses. The gap between canonical and random views is the appearance-shift penalty, which is the in-sim proxy for sim-to-real.

| metric | from | measures |
|---|---|---|
| state R² (position / orientation / intrinsics / clearance) | CLS ++ mean patch | pose and control-relevant geometry |
| depth AbsRel, RMSE(log) | per-patch, non-sky | metric 3D structure |
| semantic mIoU | per-patch | scene content |
| retrieval@1, global/patch centred cosine | no fitting | same pose, different look → same features (retrieval rules out trivial collapse) |
| `--heldout-region` scores | poses in a region absent from training | spatial generalisation |
| `--real DIR` domain accuracy | linear sim-vs-real classifier | separability of real frames (read with probe scores) |

Baselines, from the floor upward:

1. `pixels`
2. Randomly initialised ViT-S (`lingbot-random:small`)
3. Off-the-shelf LingBot-Vision S, plus DINOv3 S when accessible
4. The LingBot-Vision B/L teachers, plus DINOv3 B/L when accessible

Ablations in `scripts/baselines.sh` each change one factor: pair target, student init, student augmentation, and training on canonical only.

### Real drone evaluation

```sh
visenc uavid --out data/uavid
visenc real-eval --encoders lingbot:small run:runs/baseline run:runs/mean \
  --baseline lingbot:small --out reports/real-with-uncertainty
# Recompute metrics or paired comparisons from saved predictions, without model inference:
python -m visenc.real rescore --reports reports/real-with-uncertainty/report.json \
  --baseline lingbot:small --out reports/real-rescored
```

UAVid train/validation videos must be disjoint. A simulator-fit segmentation probe is applied unchanged to 70 real validation frames; a separate UAVid-train probe measures real→real readout quality. Car detection is **patch ranking**, not bounding boxes: positives have ≥10% car pixels, negatives have none, and intermediate fractions are excluded. Both static and moving cars count. Reports include tie-correct AUROC, non-interpolated average precision, positive/negative counts and prevalence.

Confidence intervals resample whole video sequences (1,000 draws, shared across models and probe directions), not individual patches. Paired differences use those same draws. Only seven validation sequences are available; these intervals exclude training-seed and probe-fitting uncertainty. Cached per-frame predictions, protocol and fit indices accompany every report. The real→real minus sim→real gap must be read alongside both absolute scores: losing general real-image quality can also shrink that gap.

## Independent experiments

### Geometry × consistency

```sh
python -m visenc.geometry prepare --out data/geometry
python -m visenc.experiments geometry
```

The 2×2 design crosses fixed versus three-layout training with `own` versus `mean` targets. Each arm has 20,000 training poses, 10,000 optimizer steps and 64 images/step, with seeds 0 and 1. Matching completed fixed-city controls are reused. All cities retain the same 490 m extent and excluded quadrant; road spacing, districts, building shape/scale and object arrangements vary. Two separate seed-family layouts are held out.

**All encoders use identical canonical probe-fitting images and identical evaluation images.** Probes are never refit on held-out geometry. Reports separate familiar geometry under canonical/seen/novel appearance from held-out geometry under each appearance regime. Layout hashes, pose seeds, budgets and per-scene scores are saved. `--data` also accepts multiple split directories directly for concatenated distillation.

### Unlabelled-real adaptation

```sh
python -m visenc.experiments mixed
```

This independent 2×2 crosses simulation-only versus simulation-plus-real images with `own` versus `mean` targets. Only UAVid **training RGB** enters distillation; no real labels or validation frames are used. Both conditions process 64 images/step: 32 paired simulator poses versus 24 paired poses plus 16 real images. Steps, teacher, initialization and training seeds match. The fixed compute budget therefore reduces simulator exposure in the mixed condition; it is not an equal-simulator-exposure comparison.

Each treatment is evaluated independently on the existing simulator split and UAVid. This is **real-data-assisted adaptation**, not simulation-only transfer. Machine-readable results, uncertainty intervals and readable tables live under `reports/experiments/mixed`; geometry outputs are separate under `reports/experiments/geometry`. Incomplete training runs are never deleted or silently reused.

`python scripts/show_experiments.py --study mixed` generates that study's paper table without requiring geometry results; `--study geometry` selects the other study. The default requires both completed studies. Missing or incomplete-budget results are errors, not silently omitted.

### GPU-owned visual navigation

The separate [`sim/`](sim/) Cargo workspace supplies Vulkan/wgpu-hal rendering, NVIDIA Rust CUDA physics, external-memory/timeline synchronization, and a GPU-resident navigation task. The Python learner borrows CUDA tensor storage and runs the actual frozen visual encoder plus recurrent PPO on the native stream. The superseded host-staged prototype is removed.

```sh
cd sim && nix develop
cargo oxide build --arch sm_120 --device-codegen-crate sim_physics,sim_env -- -p sim-env --profile training
cd ..
nix develop path:sim --command .venv/bin/python sim/python/train.py profile \
  --encoder mean --width 320 --height 240 --library .sim-target/training/libsim_env.so \
  --out runs/gpu-profile
```

Actual collection and PPO-update traces pass the zero-host-staging/wait gate. Initialization, checkpoint export and post-rollout logging are explicit boundaries outside that claim. Learning success and wall-clock training time are separate measurements; an engineering oracle or profiler pass is not an RL result.

Completed visual PPO runs and learned blind/privileged controls are recorded in [`sim/verification.json`](sim/verification.json); the generated paper table is `report/navigation.typ`. Timings include initialization, training and evaluation, not compilation. This is a three-action lateral-gap task with stabilized forward speed and altitude, one PPO training seed, and 128 independent held-out episodes per policy—not full aerodynamic flight.


## Jetson Orin Nano deployment

Target: Jetson Orin Nano (JetPack 6, TensorRT 10) running the ViT-S student at the training resolution (320×240, 300 patches).

```sh
visenc deploy --encoder run:runs/mean-pretrained --calib data/train --out deploy/mean-pretrained
```

This writes:

- `encoder.onnx`: FP32 weights, static `[1,3,240,320]` float input in [0,1] (ImageNet normalisation is inside the graph). Outputs `cls [1,384]` and `patches [1,300,384]`.
- `encoder_int8.onnx`: explicit INT8 Q/DQ graph from NVIDIA ModelOpt. It uses SmoothQuant on every Linear/Conv layer, calibrated on 512 rendered views. LayerNorm inputs stay FP16, because the residual stream has outlier channels (final-norm input range ≈ ±400). With ModelOpt's default config, which also quantizes norm inputs, patch cosine to FP32 drops to 0.58.
- `int8_modelopt.pth`: the quantized model, so `int8:deploy/<name>` can be evaluated.
- `deploy.json`: cosine of each artefact against PyTorch FP32, plus the `trtexec` commands.

Build the engines on the Jetson; TensorRT engines are tied to the GPU and the TensorRT version:

```sh
trtexec --onnx=encoder.onnx      --fp16        --saveEngine=encoder_fp16.engine
trtexec --onnx=encoder_int8.onnx --int8 --fp16 --saveEngine=encoder_int8.engine
```

Precision is evaluated like any other encoder: `fp16:<spec>` runs the model purely in float16, and `int8:<deploy dir>` runs ModelOpt's simulated INT8. `scripts/baselines.sh` deploys the two pretrained-student runs and adds both precisions to the report.

On the smoke-test student, ONNX INT8 matches simulated INT8 (patch cosine to FP32 0.998 for both), FP16 is unchanged on every probe, and INT8 moves the probes by about 0.01–0.03; the largest change is retrieval@1 (0.705 → 0.670). On-device latency has not been measured yet.

## Report

Compile explicitly to the repository root so the main PDF is easy to find:

```sh
nix develop --command typst compile report/report.typ report.pdf
nix develop --command typst compile report/supplement.typ report/supplement.pdf
```

The supplement contains secondary ablations and real/layout/trained-policy visual checks. Table generators consume completed measured reports; they do not substitute plans or smoke results for full-budget experiments.

## Licences

The Kenney kits are CC0, fetched from OpenGameArt mirrors. Poly Haven textures are CC0. LingBot-Vision code and weights are Apache-2.0. DINOv3 weights are under the DINOv3 licence.
