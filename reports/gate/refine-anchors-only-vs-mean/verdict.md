# Encoder gate: run:runs/refine-anchors-only vs run:runs/mean

**REJECT** (failed: failures/patch_head_score, failures/global_head_score, sim/depth_absrel_shifted, sim/depth_absrel_region, sim/retrieval_at_1, sim/patch_cosine)

| metric | role | baseline | candidate | improvement [95% CI] | margin | pass |
|---|---|---|---|---|---|---|
| failures/patch_head_score | target | 0.6914 | 0.7142 | +0.0228 [-0.0064, +0.0538] | - | ✗ |
| failures/global_head_score | guard | 0.6689 | 0.6497 | -0.0193 [-0.0503, +0.0034] | 0.01 | ✗ |
| sim/pose_r2_shifted | guard | 0.5909 | 0.5802 | -0.0107 [-0.0151, -0.0065] | 0.02 | ✓ |
| sim/depth_absrel_shifted | guard | 0.1878 | 0.1965 | -0.0086 [-0.0106, -0.0069] | 0.01 | ✗ |
| sim/semantic_miou_shifted | guard | 0.8290 | 0.8254 | -0.0036 [-0.0051, -0.0020] | 0.01 | ✓ |
| sim/pose_r2_region | guard | -0.7951 | -0.7756 | +0.0194 [+0.0062, +0.0320] | 0.02 | ✓ |
| sim/depth_absrel_region | guard | 0.1894 | 0.1977 | -0.0083 [-0.0100, -0.0067] | 0.01 | ✗ |
| sim/semantic_miou_region | guard | 0.8227 | 0.8189 | -0.0038 [-0.0052, -0.0025] | 0.01 | ✓ |
| sim/retrieval_at_1 | guard | 0.9700 | 0.9231 | -0.0469 [-0.0594, -0.0351] | 0.02 | ✗ |
| sim/patch_cosine | guard | 0.9698 | 0.9563 | -0.0135 [-0.0145, -0.0126] | 0.01 | ✗ |
| real/sim_to_real_miou | guard | 0.5157 | 0.5335 | +0.0177 [-0.0056, +0.0305] | 0.01 | ✓ |
| real/sim_to_real_vehicle_auroc | guard | 0.9026 | 0.9421 | +0.0394 [+0.0231, +0.0496] | 0.01 | ✓ |
| real/sim_to_real_vehicle_ap | guard | 0.6039 | 0.7468 | +0.1429 [+0.0920, +0.1777] | 0.02 | ✓ |
| real/real_to_real_miou | guard | 0.6132 | 0.6512 | +0.0381 [+0.0086, +0.0490] | 0.01 | ✓ |
