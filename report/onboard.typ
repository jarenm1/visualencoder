== Onboard memory budget
The target platform has 8 GB of GPU memory for the whole onboard stack: frozen encoder, policy, and the small
language model (Gemma 4 E4B) that turns instructions into goals. @tab-budget measures each component's added
device memory in one process (NVML per-process usage, after one control step and one grounding answer).

#figure(
  table(
    columns: (auto, auto),
    align: (left, center),
    stroke: none, table.hline(),
    [*Component*], [*GPU memory (GiB)*],
    table.hline(stroke: 0.5pt),
    [CUDA context + runtime], [0.26],
    [Paired student ViT-S, TensorRT FP16], [0.50],
    [Policy with patch goal head (1.12M parameters)], [0.06],
    [Gemma 4 E4B weights, bf16 (per-layer embeddings in host memory)], [9.55],
    [Gemma grounding answer (activations, KV cache)], [0.12],
    table.hline(stroke: 0.5pt),
    [*Total*], [*10.49*],
    table.hline(),
  ),
  caption: [Measured on an RTX 5070 Ti with `sim/python/budget.py`. Everything except the language model
  uses 0.82 GiB; the bf16 language model alone exceeds the 8 GB budget, so it must be quantized.],
) <tab-budget>
