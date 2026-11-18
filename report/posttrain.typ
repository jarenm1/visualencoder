== Pilot post-training
The commander (Gemma) decides and the pilot (the policy) flies, so the pilot must be reliable at a fixed
interface: an image goal (an instance crop, or a type reference), an optional location hint, and the
explore command. Task v5 aligns the task with that interface, and a post-training stage trains on the
pilot's verified failures.

*Search, reach, hold.* Distance progress toward a target the pilot has never seen is unobservable to it,
so before the first sighting the reward pays newly observed surface instead. After the sighting it pays
progress, then a one-time bonus on entering the facing standoff band and a per-step bonus while holding.
The last replaces the old braking term, which paid for slow flight in the band even while facing away.

*Hints.* The GPS lane becomes a commander hint fixed per episode: none, precise (1–5 m), coarse
(10–40 m) or wrong (30–80 m off while claiming 1–5 m). The pilot reads the hinted offset and the claimed
radius, so it learns to use hints without trusting them over its camera.

*Look-alike steps.* A goal step where the saliency believes the target is in view but its peak patch
shows another object of the target's class is an encoder error, not a flight error. Such steps are
logged per catalog entry and given zero weight in the policy and value losses, while the supervised
saliency loss keeps training the matcher on them.

*A verifier first.* Hard-case training samples what the pilot fails, and an impossible episode fails
forever. A geometric verifier uses the simulator's own crash test at 1 m voxels to check that the start's
free space reaches the standoff band with a line of sight to the target. Its first pass found that 17%
of test episodes started inside hollow suburban houses, 80% of them landed starts. Those episodes pass
the start clearance test, but no flight leaves them, and most were labelled "hidden". Every earlier
evaluation included them. The catalogue now rejects enclosed starts (all of nine probe rays blocked
within 25 m), leaving 99.7% of test and 99.3% of training episodes solvable.

*Replay and the hard set.* During post-training, half the environments start their next episode from a
solvable catalog entry drawn in proportion to the pilot's own failure rate there (prioritized level
replay). The other half keep the curriculum. A held-out hard set fixes 256 solvable test episodes per
cohort: standard, hidden, far, hidden and far, coarse hint, wrong hint, look-alike nearby, and category
goals. Category goals start from another same-type object's reference, and success remains the actual
instance.
