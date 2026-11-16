== Additional geometry readouts
#figure(table(
  columns: (auto, auto, auto, auto, auto, auto),
  align: (left, center, center, center, center, center),
  stroke: none, table.hline(),
  [*Training*], [*App. pose R²*], [*Geo. pose R²*], [*Both pose R²*], [*App. retrieval*], [*Both retrieval*],
  table.hline(stroke: 0.5pt),
  [fixed, standard], [0.363], [0.528], [0.353], [0.714], [0.725],
  [fixed, paired], [0.535], [0.528], [0.491], [0.953], [0.942],
  [multi, standard], [0.383], [0.498], [0.358], [0.699], [0.748],
  [multi, paired], [0.542], [0.495], [0.473], [0.944], [0.945],
  table.hline(),
), caption: [Same geometry probe protocol as the main table. Retrieval candidates: 750 within each layout. App., Geo., and Both denote the appearance-only, geometry-only, and combined shifts.])
#figure(table(
  columns: (auto, auto, auto, auto, auto),
  align: (left, center, center, center, center),
  stroke: none, table.hline(),
  [*Target*], [*Δ real IoU*], [*Δ car AUROC*], [*Δ car AP*], [*Δ real-fit IoU*],
  table.hline(stroke: 0.5pt),
  [standard], [-0.0026], [+0.0005], [-0.0010], [-0.0128],
  [paired], [-0.0106], [-0.0026], [-0.0138], [-0.0151],
  table.hline(),
), caption: [Geometry intervention on UAVid: three-layout minus fixed-layout training, averaged over matched seeds. These descriptive differences are not equivalence tests; real-image scores retain the same simulator-fit and real-fit probe protocols.])
