# Presentation guide — Appearance-robust vision for drones

**Five minutes, six slides.** The encoder is the contribution; the learned pilot is the demo and a way to collect challenging camera views. Do not lead with the pilot’s success rate.

## Before you start

- Open [slides.pdf](../runs/presentation/slides.pdf) in full-screen mode and cue [pilot-demo.mp4](../runs/presentation/pilot-demo.mp4) at 0:00 in a separate player. Switch to the player on slide 5 and back to the deck when it ends. The video is silent, 27 seconds, and plays offline. No server, model loading, or training is needed.
- On the presentation machine, rehearse that window switch once. If playback fails, stay on slide 5 and explain the two camera views and the flight-to-probe loop. If you only have three minutes, keep the method, the retrieval result, and the UAVid result; play only the first 15 seconds of the reel.
- Keep the [report](report.pdf) and [supplement](supplement.pdf) available for questions. To rebuild the deck from the repo root: `nix develop --command typst compile report/slides.typ runs/presentation/slides.pdf` (create the output directory first if needed).

## What to say

### 1 · Question (0:00–0:30)

> A drone can fly past the same building at noon and at dusk. The building hasn’t changed, but almost every pixel has. Can a small vision model learn what stays the same, and does that help on real drone imagery? We distilled a 300-million-parameter teacher into a 22-million-parameter student and tested those two questions separately.

### 2 · Method (0:30–1:30)

Point to the first row of renders: one pose, five looks. Only the two training looks enter the paired loss. The two novel looks illustrate the kind of shift used for evaluation.

> Our Blender city gives us the same camera pose under different lighting, materials and fog. Standard distillation asks the student to copy the frozen teacher separately on each look. Instead, we average the teacher’s features from two aligned looks, then ask the student to match that average from either image. We apply cosine loss to the global feature and the local patches. A shared random crop keeps those patch targets aligned.

### 3 · Simulator baselines (1:30–2:20)

> We freeze each encoder and use the same evaluation: linear probes for pose, depth and segmentation, and nearest-neighbor retrieval without a fitted probe. This table is for held-out looks. Standard distillation helps, but changing the target from per-look to paired raises retrieval from 0.747 to 0.951. Depth error also drops from 0.242 to 0.201. DINOv3-L still leads on segmentation. On the canonical look the distilled models already score similarly, so the big change is robustness to appearance.

### 4 · Transfer to real drone imagery (2:20–3:10)

> The simulator gain didn’t automatically carry over to UAVid. We refined the last four blocks using unlabeled real training images, while keeping the paired simulator objective. A simulator-fit probe ranked car patches in held-out real videos: average precision rose from 0.604 to 0.749 in this matched run. The paired interval resamples videos, not individual patches. This is car-patch ranking, not a bounding-box detector. The simulator readouts stayed within the margins we set in advance.

### 5 · Learned flight demo (3:10–4:15, including playback)

> A recurrent PPO policy combines frozen camera features, an image goal and onboard motion inputs. The first clip is a selected held-out takeoff and car approach. The second shows the memory-enabled pilot in a training district. Both have location hints. Onboard view is on the left; the external camera on the right is only for the viewer.

**Play `pilot-demo.mp4`.** After playback: “The flights also give us camera viewpoints for stress-testing the encoder. We fit a fresh matcher on frozen features so we don’t confuse a bad turn by the pilot with a failure of the representation.”

### 6 · Takeaway (4:15–5:00)

> Paired targets improved robustness to unseen simulator appearance. Unlabeled real imagery improved transfer while the paired simulator objective retained that robustness. Mining flight failures was useful for diagnosis, but the losses we tried didn’t establish a fix for small targets or look-alikes. And a better encoder didn’t establish a better pilot: on the standard no-hint hard set, the old policy succeeded 28 times out of 256 and the new one 24. The claim is about the representation, not reliable autonomous flight.

## Likely questions

- **What’s new versus domain randomization?** Both students see randomized images. The key comparison holds the views fixed and changes the **teacher target** from each view’s own features to the average across paired looks.
- **Did you use real labels to train the encoder?** No. The refinement sees unlabeled UAVid training RGB and teacher features. UAVid labels are for evaluation readouts; the car AP readout is fit in simulation and scored on real validation videos.
- **Why AP rather than only AUROC?** Car patches are sparse (8.6% of eligible patches). AP makes false positives near the top of the ranking more visible. The interval reflects the seven available validation videos, not training-seed uncertainty; refinement used one seed.
- **Why did you adopt a checkpoint the gate rejected?** We adopted it as the base encoder because real-image readouts improved and simulator checks stayed within their margins. It did not pass as a *flight-failure fix*: the goal-matching score rose by 0.016, but its paired interval (−0.005 to +0.038) includes zero.
- **Why not just use the teacher?** The student is about 14× smaller by parameter count. The result is about feature quality at that size; latency on the embedded target was not measured.
- **Is the demo representative?** No. It shows selected successes from earlier policies. On the later no-hint hard set, collisions rose from 80/256 to 116/256; both policies went 0/256 on hidden-and-far goals. The two runs also differed in perception-head data and training setup, so this does not isolate the encoder’s effect on flight.
