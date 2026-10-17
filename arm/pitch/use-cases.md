# Use cases: Robot Teaches Human (BSL)

Flip: in robotics the human teaches the robot. Here the robot hand teaches the human, and marks them.
Both hands become the same 8 numbers (curl and spread for 4 fingers), so marking is a direct comparison.

## Built today (safe to demo and to claim)
1. **Learn BSL handshapes with instant feedback.** A learner sees the robot hand show a sign, copies
   it in front of a webcam, and each finger is marked right or wrong with a one-line fix
   ("Ring finger: curl 20 degrees more"). They hold it for 3 seconds to pass, then move on.
2. **Show-and-copy teaching at events, libraries and classrooms.** One laptop, no instructor. A visitor
   tries 3 to 4 signs in about two minutes and gets a scorecard of their weakest finger.
3. **Browse the sign set.** The learner can step through all signs, see the robot demonstrate each,
   and see the source and an honest label (exact or approximated) for every one.

## Next (say "next", never "done")
4. **Hand-exercise drills.** The same loop with drills instead of signs, for people who want their form
   checked at home. Needs input from a clinician before anyone calls it rehabilitation.
5. **Movement-based signs.** Add the wrist path (for example "sorry", a circle) to the handshape check.
6. **Robot as examiner for demonstration data.** Check which human gestures this 4-finger hand can
   actually reproduce, before they are used to teach a robot.
7. **A trained teacher.** Replace scripted poses with a policy learned in simulation.

## Not claimed
- Not an accredited BSL course. No fluent signer has reviewed the signs; handshapes come from a
  commercial dictionary site, not SignBank or the BDA.
- Marks handshape only: not hand location, palm direction, movement or facial expression.
- A subset: one hand, four fingers, no little finger. Most of BSL (including fingerspelling) is out of reach.
- Not medical. Camera frames stay in memory and are never saved or sent.
- The teacher poses are scripted, not RL-trained. No drill recogniser or adaptive curriculum exists.
