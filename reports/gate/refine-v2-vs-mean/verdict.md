# Encoder gate: run:runs/refine-v2 vs run:runs/mean

**REJECT** (failed: failures/patch_head_score, failures/global_head_score, sim/pose_r2_region, sim/retrieval_at_1, sim/patch_cosine)

| metric | role | baseline | candidate | improvement [95% CI] | margin | pass |
|---|---|---|---|---|---|---|
| failures/patch_head_score | target | 0.6914 | 0.6881 | -0.0033 [-0.0304, +0.0233] | - | ✗ |
| failures/global_head_score | guard | 0.6689 | 0.6579 | -0.0110 [-0.0585, +0.0359] | 0.01 | ✗ |
| sim/pose_r2_shifted | guard | 0.5909 | 0.5792 | -0.0117 [-0.0171, -0.0068] | 0.02 | ✓ |
| sim/depth_absrel_shifted | guard | 0.1878 | 0.1943 | -0.0064 [-0.0087, -0.0043] | 0.01 | ✓ |
| sim/semantic_miou_shifted | guard | 0.8290 | 0.8271 | -0.0019 [-0.0044, +0.0005] | 0.01 | ✓ |
| sim/pose_r2_region | guard | -0.7951 | -0.8655 | -0.0704 [-0.0901, -0.0530] | 0.02 | ✗ |
| sim/depth_absrel_region | guard | 0.1894 | 0.1947 | -0.0053 [-0.0073, -0.0035] | 0.01 | ✓ |
| sim/semantic_miou_region | guard | 0.8227 | 0.8221 | -0.0006 [-0.0025, +0.0012] | 0.01 | ✓ |
| sim/retrieval_at_1 | guard | 0.9700 | 0.9338 | -0.0362 [-0.0464, -0.0260] | 0.02 | ✗ |
| sim/patch_cosine | guard | 0.9698 | 0.9602 | -0.0096 [-0.0103, -0.0090] | 0.01 | ✗ |
| real/sim_to_real_miou | guard | 0.5157 | 0.5220 | +0.0062 [-0.0041, +0.0171] | 0.01 | ✓ |
| real/sim_to_real_vehicle_auroc | guard | 0.9026 | 0.9420 | +0.0394 [+0.0241, +0.0494] | 0.01 | ✓ |
| real/sim_to_real_vehicle_ap | guard | 0.6039 | 0.7306 | +0.1268 [+0.0795, +0.1579] | 0.02 | ✓ |
| real/real_to_real_miou | guard | 0.6132 | 0.6448 | +0.0316 [+0.0104, +0.0405] | 0.01 | ✓ |
