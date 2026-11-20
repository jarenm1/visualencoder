# Encoder gate: run:runs/refine-no-real vs run:runs/mean

**REJECT** (failed: failures/patch_head_score, failures/global_head_score, sim/pose_r2_region, sim/retrieval_at_1, real/sim_to_real_miou, real/sim_to_real_vehicle_auroc, real/sim_to_real_vehicle_ap)

| metric | role | baseline | candidate | improvement [95% CI] | margin | pass |
|---|---|---|---|---|---|---|
| failures/patch_head_score | target | 0.6914 | 0.6980 | +0.0066 [-0.0195, +0.0333] | - | ✗ |
| failures/global_head_score | guard | 0.6689 | 0.6513 | -0.0176 [-0.0651, +0.0293] | 0.01 | ✗ |
| sim/pose_r2_shifted | guard | 0.5909 | 0.5816 | -0.0093 [-0.0135, -0.0052] | 0.02 | ✓ |
| sim/depth_absrel_shifted | guard | 0.1878 | 0.1936 | -0.0058 [-0.0075, -0.0040] | 0.01 | ✓ |
| sim/semantic_miou_shifted | guard | 0.8290 | 0.8259 | -0.0031 [-0.0056, -0.0008] | 0.01 | ✓ |
| sim/pose_r2_region | guard | -0.7951 | -0.8352 | -0.0401 [-0.0527, -0.0279] | 0.02 | ✗ |
| sim/depth_absrel_region | guard | 0.1894 | 0.1939 | -0.0045 [-0.0058, -0.0030] | 0.01 | ✓ |
| sim/semantic_miou_region | guard | 0.8227 | 0.8221 | -0.0006 [-0.0026, +0.0012] | 0.01 | ✓ |
| sim/retrieval_at_1 | guard | 0.9700 | 0.9350 | -0.0351 [-0.0458, -0.0249] | 0.02 | ✗ |
| sim/patch_cosine | guard | 0.9698 | 0.9614 | -0.0084 [-0.0090, -0.0078] | 0.01 | ✓ |
| real/sim_to_real_miou | guard | 0.5157 | 0.5059 | -0.0098 [-0.0294, -0.0035] | 0.01 | ✗ |
| real/sim_to_real_vehicle_auroc | guard | 0.9026 | 0.8662 | -0.0365 [-0.0486, -0.0087] | 0.01 | ✗ |
| real/sim_to_real_vehicle_ap | guard | 0.6039 | 0.5341 | -0.0697 [-0.0865, -0.0206] | 0.02 | ✗ |
| real/real_to_real_miou | guard | 0.6132 | 0.6091 | -0.0041 [-0.0093, +0.0086] | 0.01 | ✓ |
