== Spatial memory, counting and language planning
A reactive policy forgets a target once it leaves the view and cannot count. We add a GPU-resident
top-down memory built only from onboard, monocular quantities, and a small language model that plans
over it.

*Memory.* Each environment keeps a 128×128 map at 1.5 m around its start. The pose is dead-reckoned
from the gyro and body-velocity sensors. Integrating the gyro about the gravity axis drifted 0.2 rad in
10 s whenever the body tilted; integrating the heading rate of the levelled forward axis,
$dot(psi) = bold(omega) dot bold(u) - f_y (bold(omega) dot hat(bold(f))) \/ |bold(h)|$, brings the median position error after 10 s to 0.06 m. Every patch ray is
cast to its predicted depth (@tab-perception): the hit cell takes the policy's own goal evidence and
cells along the ray are marked looked-at, so evidence is replaced when the drone looks again. The actor
reads a heading-aligned 32×32 crop (±48 m) through a small CNN plus the strongest candidate; an
auxiliary loss decodes the target's offset from the memory features. With labelled depth a seen target
is placed within 3 m of its box in every case; with predicted depth in about half, the remaining error
being range rather than bearing.

*Counting.* Patches the class head calls vehicle (within 35 m) are clustered per frame, associated to
tracks with a range-dependent gate (3 m + 0.25 × the larger of the observation's and the track's range),
merged when tracks converge and confirmed after three frames. On 128 held-out 15 s exploratory flights
the mean count is 1.70 against 1.77 cars actually seen, with mean absolute error 0.76; 46% of flights are
exact and 83% within one.

*Language planning.* Gemma 4 E4B sees the camera frame and a text summary of the memory (area looked
at, confirmed cars and their offsets, goal evidence) and chooses the next skill: explore, go to an object
it boxes in the frame, or answer. The learned policy executes every skill; nothing is scripted.
