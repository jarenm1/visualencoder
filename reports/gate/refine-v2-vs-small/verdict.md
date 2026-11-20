# Encoder gate: run:runs/refine-v2 vs lingbot:small

**REJECT** (failed: failures/patch_head_score, real/real_to_real_miou)

| metric | role | baseline | candidate | improvement [95% CI] | margin | pass |
|---|---|---|---|---|---|---|
| failures/patch_head_score | target | 0.6781 | 0.6881 | +0.0100 [-0.0339, +0.0529] | - | ✗ |
| failures/global_head_score | guard | 0.6241 | 0.6579 | +0.0338 [-0.0026, +0.0744] | 0.01 | ✓ |
| sim/pose_r2_shifted | guard | 0.3924 | 0.5792 | +0.1868 [+0.1645, +0.2125] | 0.02 | ✓ |
| sim/depth_absrel_shifted | guard | 0.2668 | 0.1943 | +0.0725 [+0.0632, +0.0842] | 0.01 | ✓ |
| sim/semantic_miou_shifted | guard | 0.7064 | 0.8271 | +0.1208 [+0.1099, +0.1316] | 0.01 | ✓ |
| sim/pose_r2_region | guard | -1.2123 | -0.8655 | +0.3468 [+0.2865, +0.4203] | 0.02 | ✓ |
| sim/depth_absrel_region | guard | 0.2635 | 0.1947 | +0.0687 [+0.0624, +0.0759] | 0.01 | ✓ |
| sim/semantic_miou_region | guard | 0.7153 | 0.8221 | +0.1068 [+0.0988, +0.1152] | 0.01 | ✓ |
| sim/retrieval_at_1 | guard | 0.7602 | 0.9338 | +0.1736 [+0.1505, +0.1963] | 0.02 | ✓ |
| sim/patch_cosine | guard | 0.7948 | 0.9602 | +0.1654 [+0.1612, +0.1694] | 0.01 | ✓ |
| real/sim_to_real_miou | guard | 0.4921 | 0.5220 | +0.0299 [-0.0038, +0.0746] | 0.01 | ✓ |
| real/sim_to_real_vehicle_auroc | guard | 0.8920 | 0.9420 | +0.0500 [+0.0372, +0.0698] | 0.01 | ✓ |
| real/sim_to_real_vehicle_ap | guard | 0.6404 | 0.7306 | +0.0902 [+0.0308, +0.1131] | 0.02 | ✓ |
| real/real_to_real_miou | guard | 0.6505 | 0.6448 | -0.0057 [-0.0226, +0.0119] | 0.01 | ✗ |
