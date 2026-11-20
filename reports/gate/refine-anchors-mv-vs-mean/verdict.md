# Encoder gate: run:runs/refine-anchors-mv vs run:runs/mean

**REJECT** (failed: failures/patch_head_score, failures/global_head_score)

| metric | role | baseline | candidate | improvement [95% CI] | margin | pass |
|---|---|---|---|---|---|---|
| failures/patch_head_score | target | 0.6914 | 0.7077 | +0.0163 [-0.0047, +0.0383] | - | ✗ |
| failures/global_head_score | guard | 0.6689 | 0.6524 | -0.0165 [-0.0459, +0.0046] | 0.01 | ✗ |
| sim/pose_r2_shifted | guard | 0.5909 | 0.5884 | -0.0024 [-0.0061, +0.0013] | 0.02 | ✓ |
| sim/depth_absrel_shifted | guard | 0.1878 | 0.1910 | -0.0032 [-0.0053, -0.0015] | 0.01 | ✓ |
| sim/semantic_miou_shifted | guard | 0.8290 | 0.8274 | -0.0016 [-0.0029, -0.0002] | 0.01 | ✓ |
| sim/pose_r2_region | guard | -0.7951 | -0.8035 | -0.0085 [-0.0196, +0.0027] | 0.02 | ✓ |
| sim/depth_absrel_region | guard | 0.1894 | 0.1926 | -0.0032 [-0.0048, -0.0017] | 0.01 | ✓ |
| sim/semantic_miou_region | guard | 0.8227 | 0.8207 | -0.0019 [-0.0032, -0.0008] | 0.01 | ✓ |
| sim/retrieval_at_1 | guard | 0.9700 | 0.9678 | -0.0023 [-0.0062, +0.0023] | 0.02 | ✓ |
| sim/patch_cosine | guard | 0.9698 | 0.9668 | -0.0030 [-0.0032, -0.0029] | 0.01 | ✓ |
| real/sim_to_real_miou | guard | 0.5157 | 0.5366 | +0.0209 [-0.0050, +0.0338] | 0.01 | ✓ |
| real/sim_to_real_vehicle_auroc | guard | 0.9026 | 0.9444 | +0.0418 [+0.0253, +0.0518] | 0.01 | ✓ |
| real/sim_to_real_vehicle_ap | guard | 0.6039 | 0.7493 | +0.1454 [+0.0949, +0.1799] | 0.02 | ✓ |
| real/real_to_real_miou | guard | 0.6132 | 0.6486 | +0.0354 [+0.0080, +0.0451] | 0.01 | ✓ |
