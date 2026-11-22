== Pilot post-training
The commander (Gemma) decides and the pilot (the policy) flies, so the pilot must be reliable at a fixed
interface: an image goal (an instance crop, or a type reference), an optional location hint, and the
explore command. Task v5 aligns the task with that interface and tests whether replaying verified failures
improves the pilot.

*Search, reach, hold.* Distance progress toward a target the pilot has never seen is unobservable to it,
so before the first sighting the reward pays newly observed surface instead. After the sighting it pays
progress, then a one-time bonus on entering the facing standoff band and a per-step bonus while holding.
The last replaces the old braking term, which paid for slow flight in the band even while facing away.

*Hints.* The GPS lane becomes a commander hint fixed per episode: none, precise (1–5 m), coarse
(10–40 m) or wrong (30–80 m off while claiming 1–5 m). The pilot reads the hinted offset and the claimed
radius, so it learns to use hints without trusting them over its camera.

*Look-alike steps.* A goal step where the saliency believes the target is in view but its peak patch
shows another object of the target's class is a goal-matcher confusion, not direct evidence of a flight error. Such steps are
logged per catalog entry and given zero weight in the policy and value losses, while the supervised
saliency loss keeps training the matcher on them.

*A verifier first.* Hard-case training samples what the pilot fails, and an impossible episode fails
forever. A geometric verifier uses the simulator's own crash test at 1 m voxels to check that the start's
free space reaches the standoff band with a line of sight to the target. Its first pass found that 17%
of test episodes started inside hollow suburban houses, 80% of them landed starts. Those episodes pass
the start clearance test, but no flight leaves them, and most were labelled "hidden". Every earlier
evaluation included them. The catalogue now rejects enclosed starts (all of nine probe rays blocked
within 25 m), leaving 99.7% of test and 99.3% of training episodes solvable.

*Replay and the hard set.* In the first post-training run, half the environments start from solvable
catalogue entries sampled by the pilot's own failure rate; the other half keep the curriculum. The held-out
hard set fixes 256 solvable test episodes per cohort: standard without a hint, standard with a precise hint,
hidden, far, hidden and far, coarse hint, wrong hint, look-alike nearby, and category goals. Category goals
start from another same-type object's reference, and success remains the actual instance.

*Does the refined encoder improve flight?* On the same hard set, the original pilot (`drone13`, paired student)
and the warm-started refined pilot (`drone15`, @sec-refine) score 28/256 vs 24/256 on standard no-hint goals,
and 16/256 vs 23/256 with precise hints. No-hint collisions rise from 80/256 to 116/256.
Hidden-and-far goals score 0/256 for both. This is a single-seed system comparison, not an encoder ablation:
the refined run also used separately fit perception heads with half as many training frames and a different
training budget. The encoder's UAVid gains did not translate into a demonstrated navigation gain.
