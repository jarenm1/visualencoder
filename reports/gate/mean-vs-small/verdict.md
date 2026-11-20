# Encoder gate: run:runs/mean vs lingbot:small

**REJECT** (failed: real/sim_to_real_vehicle_ap, real/real_to_real_miou)

| metric | role | baseline | candidate | improvement [95% CI] | margin | pass |
|---|---|---|---|---|---|---|
| failures/patch_head_score | target | 0.7765 | 0.8101 | +0.0336 [+0.0018, +0.0636] | - | ✓ |
| failures/global_head_score | guard | 0.6684 | 0.7297 | +0.0613 [+0.0324, +0.0904] | 0.01 | ✓ |
| sim/pose_r2_shifted | guard | 0.3924 | 0.5909 | +0.1984 [+0.1768, +0.2256] | 0.02 | ✓ |
| sim/depth_absrel_shifted | guard | 0.2668 | 0.1878 | +0.0789 [+0.0692, +0.0911] | 0.01 | ✓ |
| sim/semantic_miou_shifted | guard | 0.7064 | 0.8290 | +0.1226 [+0.1122, +0.1332] | 0.01 | ✓ |
| sim/pose_r2_region | guard | -1.2123 | -0.7951 | +0.4172 [+0.3582, +0.4857] | 0.02 | ✓ |
| sim/depth_absrel_region | guard | 0.2635 | 0.1894 | +0.0741 [+0.0678, +0.0809] | 0.01 | ✓ |
| sim/semantic_miou_region | guard | 0.7153 | 0.8227 | +0.1074 [+0.0994, +0.1160] | 0.01 | ✓ |
| sim/retrieval_at_1 | guard | 0.7602 | 0.9700 | +0.2098 [+0.1867, +0.2330] | 0.02 | ✓ |
| sim/patch_cosine | guard | 0.7948 | 0.9698 | +0.1750 [+0.1703, +0.1794] | 0.01 | ✓ |
| real/sim_to_real_miou | guard | 0.4921 | 0.5157 | +0.0236 [-0.0082, +0.0691] | 0.01 | ✓ |
| real/sim_to_real_vehicle_auroc | guard | 0.8920 | 0.9026 | +0.0106 [+0.0016, +0.0314] | 0.01 | ✓ |
| real/sim_to_real_vehicle_ap | guard | 0.6404 | 0.6039 | -0.0365 [-0.0792, +0.0200] | 0.02 | ✗ |
| real/real_to_real_miou | guard | 0.6505 | 0.6132 | -0.0373 [-0.0542, -0.0053] | 0.01 | ✗ |
