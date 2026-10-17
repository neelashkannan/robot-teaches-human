# Judge Q&A (v1, 12:25, update after the 14:43 checkpoint)

Answers are 1 to 2 lines. Say only what is true in the repo at demo time; edit anything that changed.

## Required

**Did the robot really learn this?**
No. The robot hand's poses come from a table of signs (`data/signs.json`), not a trained policy. A
reinforcement-learning teacher is the next step, not something we built.

**Is this medical?**
No. It is a demo of the idea that a robot can be the teacher. It makes no health or therapy claim.

**What happens to the camera images?**
They stay in memory for the live check and are never saved or sent anywhere. The app makes no network
calls at runtime.

**Is this real BSL?**
It is an approximation of a small subset. The hand has 4 fingers, shows one hand, and we check
handshape only, not location, palm direction or facial expression. Each sign has a cited source
(BSL SignBank or the BDA / Signature chart) and an `exact` or `approximated` label. A fluent signer
has not reviewed it yet.

## Likely

**What is the flip?**
Robots spent a decade learning from humans. Here the same pipeline, human motion turned into robot
commands, runs backwards: the robot shows the sign and examines the human.

**How does the marking work?**
Both hands, yours and the robot's, are reduced to the same 8 numbers: curl and spread for 4 fingers.
Marking compares them per finger and names the biggest fix, for example "Ring finger: curl 20 degrees
more". You pass by holding the shape for 3 seconds.

**Why a robot hand and not just a video of the sign?**
Honest answer: a video would teach too. The robot is the demonstration of the flip, and the same 8
numbers make the comparison direct. We have not tested whether a robot teaches better than a video.

**Why only 4 fingers and 5 to 6 signs?**
That is what the Pollen AmazingHand has (4 fingers, 8 servos) and what we could verify against sources
in 6 hours. Signs we could not represent honestly are marked `not_representable` and left out.

**Does it work for different hand sizes and lighting?**
We calibrate each person with a 10 second open-hand and fist warm-up. We have tested it on a small
number of hands, not systematically; a fairness test across many hands is something we would do next.
(Update with what we actually tested.)

**Who is it for?**
Today it is a demo for learners and curious visitors. We have no users and no partnerships. Whether it
helps people learn is untested; testing it with deaf-led organisations and signers is the next step.

**What is next?**
In order: have a fluent signer review and correct the sign set; train the teacher with reinforcement
learning in the simulator; add the left hand, movement and two-handed signs.

**Is the robot real?**
It is the simulated Pollen AmazingHand (open-source model, Apache-2.0) in MuJoCo. No real hardware is
involved today.

**Why does it say "right" or "wrong" if it can only check handshape?**
Because "correct" here means "matches what this hand and camera can check", and the screen says so.
It is feedback on the shape, not a judgement of fluent signing.

## If asked and we are unsure

Say: "We have not tested that." Do not guess numbers, accuracy, or user counts.
