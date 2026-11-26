// Standalone research note. Build: typst compile report/temporal-encoder.typ /tmp/temporal-encoder.pdf
#set document(title: "Temporal Memory Inside the Visual Encoder")
#set page(paper: "us-letter", margin: 0.8in, numbering: "1")
#set text(font: "New Computer Modern", size: 10pt)
#set par(justify: true)
#align(center)[#text(size: 17pt, weight: "bold")[Temporal Memory Inside the Visual Encoder] \
Standalone research note · proposal and simulator pilots]

= What the Tesla post suggests
#link("https://x.com/elonmusk/status/2124458557525397780")[Musk's Nov. 22 Reality View post]
(#link("https://twiscan.com/en/x/elonmusk/2124458557525397780")[full-text mirror]) describes a shared Universal
Vision Encoder feeding both driving and a separate visualization model. The latter conditions on a recent sequence
of encoder features and feeds back its previous rendered output. The post does not say that the encoder itself
is recurrent. The useful idea here is to learn temporal features once and reuse them across tasks. Our hypothesis
is different: fold visual history into the encoder's output on every frame, rather than making each consumer
cache features or maintain a separate map.

The current LingBot-Vision ViT-S/16 student emits a 384-d CLS token and 300 patch tokens from one 320×240 image.
The current top-down memory instead stores ray-cast evidence from predicted depth and dead-reckoned pose for
policy and planning. The first test only asks whether the encoder can retain *short-horizon visual evidence*:
which vehicle was seen, roughly where it is now relative to the camera, and whether the current frame confirms it.
It does not test replacement of an explicit map for counting or revisits.

= Proposed encoder
#raw("step(rgb, state, reset) -> (cls, patches, next_state)")

Insert a recurrent adapter before the student's last two ViT blocks. Image tokens read the previous state with
cross-attention through a residual branch initialized near zero; the last blocks then emit the same CLS and patch
shapes as today. In parallel with that read, each previous state slot attends to the *current, pre-read* image
features for its update. A learned per-slot gate interpolates between the previous slot and its update. Both paths
use the same previous state; the write becomes visible on the next frame. Start with 16 slots × 256 channels:
8 KiB of FP16 persistent state per stream, excluding weights and temporary activations. Slots are learned latent
tokens, not patch coordinates or object IDs. Let the model learn correspondence across camera motion rather than
averaging features at the same image coordinates. Keep the first test RGB-only; if turns or dropped frames are
the dominant failure, test time gaps and onboard relative motion as a separate input ablation.

The caller carries one state tensor per stream, but only the encoder reads or updates it. Reset before processing
the first frame of each new clip or episode, independently per batch lane. Goal reference crops use a stateless
pass and never write flight state. No future frames, cross-episode state, policy recurrence or external map in the
primary comparison. Current-frame probes still consume the ordinary CLS or patch outputs; an occlusion probe may
also consume those outputs but must distinguish remembered evidence from current visibility.

#link("https://cut3r.github.io/")[CUT3R] is the closest precedent for learned read/write state in streaming visual
geometry; #link("https://openaccess.thecvf.com/content/CVPR2022/html/Wu_MeMViT_Memory-Augmented_Multiscale_Vision_Transformer_for_Efficient_Long-Term_Video_Recognition_CVPR_2022_paper.html")[MeMViT] motivates the competing design that caches compressed features instead of rewriting slots. Neither
establishes that this small student retains the right information under drone motion and appearance shift.

#pagebreak()
= Experiment: same frame, different history
Use the simulator's replay capture (`sim/python/sim_rl/replay.py`) as the starting point for contiguous RGB, pose,
target IDs and visibility diagnostics; add a controlled replay generator for paired histories. Begin with one
static vehicle and a moving camera. Use paired renders of the *same* camera path with the vehicle at different
hidden positions, or with the vehicle removed. Reveal it early, move the camera behind an occluder or turn away,
then evaluate on a suffix where the current RGB matches across pairs. Verify the match (including shadows) and
exclude leaky pairs. Vary the unobserved interval (0.25, 0.5, 1, 2 and 4 s at 8 Hz); include lookalikes on
re-entry, and clips where a formerly occupied location comes back into view empty. Treat occlusion and leaving the
frustum as separate slices. Pose and object labels are for supervision and scoring, not encoder inputs.
Split by layout and trajectory before deriving clips; render each path twice with different clip-consistent looks,
and reserve novel looks for evaluation.

Initialize all models from the same appearance-robust student. First freeze the original blocks and train the new
parameters on contiguous causal unrolls long enough to include the reveal and the scored gap. Keep the original
current-frame CLS/patch distillation loss on both looks to protect visible-frame quality. Train small identical
heads across variants: a camera-relative bearing/range and remembered-target confidence head on CLS, and a
visible-target patch head for re-identification. Regress position only for vehicles seen earlier in the clip;
never-seen and empty clips are confidence negatives. Train the heads jointly for this pilot. Then freeze the
encoders and refit the same limited-capacity probes on a disjoint probe-fit split; report held-out results from
those probes, not from the training heads.

#table(
  columns: (1fr, 2fr), align: left, inset: 5pt,
  [*Variant*], [*What it tests*],
  [Original student + reset-every-frame adapter], [Separate the frame-only floor from the gains of extra capacity.],
  [Recent-feature cache], [Causal FIFO of four frames × four compressed tokens; same state bytes, read insertion and probe capacity.],
  [Recurrent adapter after the last block], [Same slot dimensions and read/write budget, but temporal fusion happens after spatial encoding. Tests whether placement matters.],
  [Recurrent adapter before the last two blocks], [Proposed model. Repeat at inference with state zeroed or swapped between matched histories.],
)

Match training clips, optimizer steps and backbone freezing; match adapter parameters and per-frame compute as
closely as practical and report the residual differences. The cache is a fixed-byte comparator, not a claim about
all window sizes. Swap tests keep RGB fixed and should move predictions in the donor history's direction.

= Decision rule
Primary: target-position recall at a validation threshold calibrated to 5% false memory on negative clips (also
report the realized false-memory rate on test). A hit requires bearing within 10° and range within 5 m.
Preselect the 1–2 s occlusion slice; plot recall by gap length, report
reappearance top-1 ID among lookalikes, and measure stale evidence after a clear empty view. Use paired
trajectory-cluster bootstrap intervals on held-out clips and report each of at least two training seeds. Verify
prefix outputs are unchanged if future frames change and lane resets match a fresh stream. Report visible-frame
UAVid car AP and simulator novel-look depth error, plus batch-1 p50/p95 step latency and memory on the same device.

Advance only if the integrated model beats both the FIFO and the late recurrent adapter on the primary slice
with a positive paired interval, stays within 0.02 absolute car AP and 5% relative depth-error degradation of
the original student, and adds at most 10% p95 encoder latency at the same precision. These are proposed gates,
not results. A FIFO tie favors the simpler cache; a late-adapter tie says history helps but placement did not.
Use these gates to decide whether the recurrent encoder merits a closed-loop trial. Keep the map for coverage,
counting and long revisits until those capabilities are measured separately.

#pagebreak()
= What the flight policy sees
The goal is not to render a world view for the actor. It is to give the actor an estimate of its local
surroundings that does not disappear when the camera turns. For example, after the drone yaws away from a
parked vehicle, the actor should know the vehicle was last seen to its left, even while the current image shows
open space ahead. It should not treat that remembered location as a fresh detection or assume an unseen sector
is clear.

#raw("actor(goal-match(CLS_t, patches_t), history_t, proprio_t) -> action_t")

Use the existing goal-conditioned saliency path in `sim/python/sim_rl/model.py` with the encoder's latest CLS
and patch outputs. They are anchored to the current frame but already informed by history. Add a small projected
summary of the encoder's read from *previous* slots, after the goal-matching projection and before the actor
input projection. The read is conditioned on the current image; the actor gets neither the writable slots nor a
stack of old frames. The current-frame saliency head still predicts *visible now*. The history summary can
carry evidence of something last seen outside the frame. Keep proprioception and goal inputs unchanged, and
leave the critic unchanged.

First check whether a small probe can decode where the remembered evidence lies in the *current body frame*:
relative bearing and range to the previously seen goal or landmark, and whether an area is observed, remembered
or unknown. Score after yaw and lateral movement; an embedding that remembers *what* was there but cannot
recover *where it is relative to the drone now* is not spatial awareness. Keep RGB-only as the first condition.
If it cannot handle those turns reliably, feed measured onboard relative motion to the encoder's state update
and compare against RGB-only; do not give the encoder or actor simulator poses. Longer term, the same visual
history could support re-identification, steadier depth, or maintaining awareness of nearby obstacles; each needs
its own readout and failure tests.

== Closed-loop test
Freeze the encoders. Train an otherwise matched actor with a fixed, stateless goal reference, no external map
and no goal refresh. Start with a feed-forward actor to isolate the encoder's memory from the existing policy
GRU. Compare frame-only, recent-feature cache, recurrent tokens alone, and recurrent tokens plus the history
summary. Give the summary branch the same input width and projection budget across variants, using pooled
current-frame features for the frame-only control. Then repeat the best comparison with the normal policy GRU
to find out whether temporal vision adds anything to the deployed policy architecture.

Use paired held-out starts with a guaranteed initial sighting followed by a turn or occlusion. Report target
reacquisition success and time, task success, collision rate and wrong-target pursuit; stratify by time out of
view and turn magnitude. On matched replay frames, swap the history while keeping the current image and
proprioception fixed: the actor's preferred search direction should follow the prior evidence while the visible
saliency remains tied to the current image. Compare closed-loop results against the existing map policy as a
separate reference, not as an input to the recurrent encoder. A gain without a map shows that encoder state
can supply short-horizon spatial context; it does not show that it replaces a persistent map.

#pagebreak()
= Simulator pilots: remembering a side is not localizing a car
The matched turn-away benchmark uses rendered left, right and never-seen histories whose scored frames
are pixel-identical. The frozen patch probe found the reveal side on 24 of 24 held-out clips while the car was
visible. On hidden frames, the initial recurrent visual head stayed at chance. Adding direct slot supervision
made the original side decodable after two seconds: 83–100% across the two seeds and two supervised in-ViT
settings. A nonzero reader gate without slot supervision did not help.

The geometry pilot used the same direct readout and 16 × 256 state values for both the recurrent writer and
strided FIFO, matched trajectory draws and two seeds. It compared RGB-only localization against explicit
transport of a predicted reveal-frame point using exact simulator relative motion. Scores are on 24 paths across
two test layouts, separate from adaptation and calibration. Four-second results include a stationary hold using
the final rendered image. This is a test of retention, not continued flight or a real odometry test.

#table(
  columns: (2fr, 1fr, 1fr), inset: 5pt,
  [*Two-second recall*], [*FIFO*], [*Recurrent*],
  [Narrow, RGB only], [0% / 0%], [0% / 0%],
  [Narrow, exact motion; fresh paths], [37.5% / 81.2%], [8.3% / 47.9%],
  [Wider reveal geometry, exact motion], [0% / 0%], [0% / 2.1%],
)
Numbers are ordered by seed (17 / 29). Recall requires 10° bearing and 5 m range accuracy and a confidence
threshold calibrated on absent validation paths. The small validation set makes the nominal 5% false-memory
threshold coarse. On the fresh narrow run at four seconds, FIFO presence AUROC fell to 0.50 in both seeds,
while recurrent presence AUROC was 0.55 and 0.98. Only one seed clearly retained useful presence information.

A pose-only guess of a point 15 m ahead at reveal met the two-second geometry tolerance on 100% of positives
in the narrow control, and on 25% in the wide control (8–27 m reveal range with lateral offset). It cannot
distinguish present from absent. On the wide control, ignoring confidence, FIFO localized 40–46% of positives;
recurrence localized 6–15%. Presence AUROC was only 0.56–0.63 for FIFO and 0.56–0.58 for recurrence.
These results do not justify a policy trial. Separate visible-frame bearing, range and confidence calibration
from retention by transporting one supervised sighting through the same turn-away clips before adding recurrence.

Run with `scripts/temporal_geometry.py`; `--wide --transport-only --scene-seed 5831` reproduces the wide arm.
Local results: `runs/temporal-geometry-01/geometry.json`, `runs/temporal-geometry-02/geometry.json`, and
`runs/temporal-geometry-wide/geometry.json`.

#pagebreak()
= Single-sighting baseline
Fit ridge readouts on the frozen encoder's first-frame CLS token or 3 × 4 pooled patch features.
The position readout trains on positives only; the confidence readout also sees never-seen negatives.
Cache one predicted position and confidence; use exact simulator relative motion to update the coordinate
frame. Neither readout sees any RGB after the first frame. Both runs use the wider reveal geometry and
24 held-out paths; the second draws fresh paths from the same test layouts and refits the readouts.

#table(
  columns: (2fr, 1fr, 1fr, 1.4fr), inset: 5pt,
  [*Readout / path seed*], [*Reveal recall*], [*2 s recall*], [*False memory paths*],
  [CLS / 5831], [81.2%], [81.2%], [0/24],
  [Patches / 5831], [91.7%], [93.8%], [3/24],
  [CLS / 9613], [87.5%], [83.3%], [0/24],
  [Patches / 9613], [95.8%], [95.8%], [1/24],
)
Recall uses the same thresholds as the geometry pilot. At two seconds the patch readout had 3.4–3.7°
mean bearing error and 1.9–2.0 m mean range error. At four seconds, both exact-motion patch scores were unchanged.
Synthetic signed drift (2°/s yaw and 0.25 m/s lateral bias, accumulated from the first sighting) lowered patch
recall to 62.5–87.5% at two seconds and 22.9–60.4% at four seconds. These are sensitivity bounds across the
four tested combinations of path seed and drift sign, not estimates of real onboard odometry performance.

The previous learned memories reached 0–2.1% at two seconds on the same paths as the first sighting run.
For a static car with a clean initial sighting, carrying a point with ego-motion beats relearning its
position from a latent history. One locked sighting cannot correct a wrong initial observation or track a
moving target. A next comparison should corrupt or obscure the first view and test recovery from subsequent
glimpses under the same noisy motion, against a simple multi-sighting filter.

Run with `scripts/temporal_sighting.py`; local results are `runs/temporal-sighting-01/sighting.json`
and `runs/temporal-sighting-02/sighting.json`.
