== Unlabelled-real adaptation
Only UAVid training RGB is added; labels and validation frames are excluded from distillation. Both conditions use 64 images/step: 32 paired simulator poses versus 24 paired poses plus 16 real frames. This matches compute and optimizer steps, not simulator exposure; it is real-data-assisted adaptation, not sim-only transfer.

#figure(table(
  columns: (auto, auto, auto, auto, auto, auto),
  align: (left, center, center, center, center, center),
  stroke: none, table.hline(),
  [*Training*], [*Sim→real IoU ↑*], [*Car AUROC ↑*], [*Car AP ↑*], [*Real→real IoU ↑*], [*Sim novel IoU ↑*],
  table.hline(stroke: 0.5pt),
  [sim only, standard], [0.539], [0.909], [0.649], [0.632], [0.788],
  [sim only, paired], [0.525], [0.905], [0.622], [0.615], [0.821],
  [sim + real, standard], [0.553], [0.898], [0.697], [0.689], [0.784],
  [sim + real, paired], [0.573], [0.896], [0.703], [0.690], [0.818],
  table.hline(),
), caption: [Independent real-data experiment, two training seeds per condition. Car AP uses ≥10% car versus car-free patches. Whole-video bootstrap intervals and paired differences are in the saved real-image evaluation; these means do not imply seed-level significance.])
