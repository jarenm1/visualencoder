# visualencoder

Sim-to-real at the visual-encoder level: distil a large vision foundation model (LingBot-Vision by default, DINOv3 once access is granted) into a small ViT. The student is trained and evaluated on one large, static simulated city, where every image comes with exact ground truth.

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
- `--extra-images DIR`: mixes in unlabelled real frames, distilled with `own` targets.

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

## Licences

The Kenney kits are CC0, fetched from OpenGameArt mirrors. Poly Haven textures are CC0. LingBot-Vision code and weights are Apache-2.0. DINOv3 weights are under the DINOv3 licence.
