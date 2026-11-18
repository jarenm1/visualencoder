== Onboard memory budget
The target platform has 8 GB of GPU memory for the whole onboard stack: frozen encoder, policy, and the small
language model (Gemma 4 E4B) that turns instructions into goals. @tab-budget measures each component's added
device memory in one process (NVML per-process usage, after one control step and one grounding answer).

#figure(
  table(
    columns: (auto, auto, auto),
    align: (left, center, center),
    stroke: none, table.hline(),
    [*Component*], [*bf16 Gemma*], [*4-bit Gemma*],
    table.hline(stroke: 0.5pt),
    [CUDA context + runtime], [0.26], [0.26],
    [Paired student ViT-S, TensorRT FP16], [0.50], [0.50],
    [Policy (bf16 column: patch goal head, 1.12M parameters; 4-bit column: with spatial memory, perception heads and commands, 1.93M)], [0.06], [0.07],
    [Gemma 4 E4B weights (per-layer embeddings in host memory)], [9.55], [4.15],
    [Gemma grounding answer (activations, KV cache)], [0.12], [0.17],
    table.hline(stroke: 0.5pt),
    [*Total*], [*10.49*], [*5.15*],
    table.hline(),
  ),
  caption: [Device memory in GiB, measured on an RTX 5070 Ti with `sim/python/budget.py`. Everything except
  the language model uses under 0.9 GiB. The bf16 language model alone exceeds the 8 GB budget; storing its
  linear layers as 4-bit NF4 (bitsandbytes, bf16 compute; vision tower and embeddings stay bf16) brings the
  whole stack to 5.15 GiB. On the 200-frame grounding benchmark the 4-bit model keeps 94/150 visible hits
  and 29/50 absent false boxes, mean IoU 0.37 vs 0.38, at 1.19 s vs 0.82 s per answer.],
) <tab-budget>
