# Encoder gate: run:runs/posttrain-mean-b4 vs run:runs/mean

**REJECT** (failed: failures/global_head_score, real/sim_to_real_miou, real/sim_to_real_vehicle_auroc, real/sim_to_real_vehicle_ap)

| metric | role | baseline | candidate | improvement [95% CI] | margin | pass |
|---|---|---|---|---|---|---|
| failures/patch_head_score | target | 0.8101 | 0.8267 | +0.0166 [+0.0047, +0.0317] | - | ✓ |
| failures/global_head_score | guard | 0.7297 | 0.7313 | +0.0016 [-0.0187, +0.0217] | 0.01 | ✗ |
| sim/pose_r2_shifted | guard | 0.5909 | 0.5896 | -0.0013 [-0.0037, +0.0012] | 0.02 | ✓ |
| sim/depth_absrel_shifted | guard | 0.1878 | 0.1907 | -0.0029 [-0.0037, -0.0021] | 0.01 | ✓ |
| sim/semantic_miou_shifted | guard | 0.8290 | 0.8268 | -0.0022 [-0.0038, -0.0006] | 0.01 | ✓ |
| sim/pose_r2_region | guard | -0.7951 | -0.7739 | +0.0212 [+0.0147, +0.0279] | 0.02 | ✓ |
| sim/depth_absrel_region | guard | 0.1894 | 0.1921 | -0.0027 [-0.0033, -0.0021] | 0.01 | ✓ |
| sim/semantic_miou_region | guard | 0.8227 | 0.8199 | -0.0028 [-0.0041, -0.0014] | 0.01 | ✓ |
| sim/retrieval_at_1 | guard | 0.9700 | 0.9695 | -0.0006 [-0.0040, +0.0023] | 0.02 | ✓ |
| sim/patch_cosine | guard | 0.9698 | 0.9693 | -0.0005 [-0.0005, -0.0004] | 0.01 | ✓ |
| real/sim_to_real_miou | guard | 0.5157 | 0.5106 | -0.0051 [-0.0131, +0.0001] | 0.01 | ✗ |
| real/sim_to_real_vehicle_auroc | guard | 0.9026 | 0.8600 | -0.0426 [-0.0571, -0.0118] | 0.01 | ✗ |
| real/sim_to_real_vehicle_ap | guard | 0.6039 | 0.5332 | -0.0707 [-0.0854, -0.0309] | 0.02 | ✗ |
| real/real_to_real_miou | guard | 0.6132 | 0.6176 | +0.0045 [-0.0006, +0.0149] | 0.01 | ✓ |
