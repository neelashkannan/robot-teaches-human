# Pitch: Robot Teaches Human

The video runs **1:59** with its own voiceover and music, so nobody needs to talk over it. Press
play, then take the minute of Q&A. The voice is OpenAI `gpt-audio` (voice "cedar", directed to talk
like someone telling a friend) through OpenRouter, the same setup as the AI PCB promo videos. The
music is Google Lyria 3. Both are disclosed on the tech stack card.

To change a line, edit `src/voice/lines.json` and run `python3 scripts/voiceover.py` (it needs
`OPENROUTER_API_KEY` set). Only changed lines are re-voiced, and the scenes stretch to fit.

## Voiceover

Regenerate this table with `python3 scripts/vo_table.py`.

| Time | Scene | Voiceover |
| --- | --- | --- |
| 0:00 | The flip | For years, robots have learned by copying us. |
| 0:05 | The flip | So we flipped it. |
| 0:07 | The flip | Now the robot teaches, and you learn. |
| 0:11 | Title | This is Robot Teaches Human. |
| 0:16 | Why a robot | Why a robot? It knows every joint angle exactly. It never gets tired, and it never judges you. |
| 0:23 | No robot? Virtual dojo | One problem: we couldn't get access to a real robot here. |
| 0:27 | No robot? Virtual dojo | So we built a virtual dojo, with real physics. |
| 0:32 | Teacher 1: robot hand | We started small: a robot hand, teaching sign language. |
| 0:36 | Teacher 1: robot hand | It shows a sign, you copy it, and every finger gets checked. |
| 0:41 | Teacher 1: robot hand | And it's honest. When four fingers can't make the real sign, it tells you. |
| 0:46 | Hand → whole body | That was one hand. |
| 0:47 | Hand → whole body | Now... the whole body. |
| 0:51 | Teacher 2: Unitree G1 | This is a real run at our table: a Unitree G1, teaching kung fu. |
| 0:57 | Teacher 2: Unitree G1 | See the red? It tells you exactly which limb to fix. |
| 1:03 | Teacher 2: Unitree G1 | Every limb green, hold for three seconds, and you've passed. |
| 1:09 | Teacher 2: Unitree G1 | Then you get your scorecard. |
| 1:12 | How it marks you fairly | But the robot is shorter than you. So is that fair? |
| 1:16 | How it marks you fairly | We turn both bodies into the same eight arrows. |
| 1:20 | How it marks you fairly | We ignore their length, and only compare which way they point. |
| 1:24 | How it marks you fairly | The biggest miss becomes one simple tip. |
| 1:27 | Tech stack | Under the hood: MuJoCo, MediaPipe, our own stance recogniser, and Nebius for the AI sensei. |
| 1:35 | Honest limits | To be honest: the robots are virtual for now, and the stances are set by hand. |
| 1:40 | What's next (concept) | Now picture a real robot. |
| 1:43 | What's next (concept) | You get it wrong, and it walks over, and gently moves your arm into place. |
| 1:48 | What's next (concept) | Just like a real sensei. |
| 1:51 | Close | Robots spent years learning from us. |
| 1:54 | Close | Now, they can teach us back. |
| 1:56 | Close | Come and try it. |

## Q&A cheat sheet (1 minute)

Short and true. Change anything that's different on the day.

**How does the marking work?**
Both bodies become 8 limb directions (upper arm, forearm, thigh, shin on each side) measured in
their own torso frame, so height and where you stand don't matter. Each of your directions is
compared with the robot's. Within 30 degrees, the limb is green. Hold every limb green for the hold
time (2 to 3 s) and the stance counts. Code: `src/rth/limbs.py`.

**Why only mostly the front view?**
A webcam's depth guess is unreliable. Where a limb lies across the camera's view, we score it
face-on. We only use full 3D where the limb points towards or away from the camera.

**Why not a real robot?**
We couldn't get robot access at the hackathon. The simulator uses the real G1's published model and
real physics, so the next step is running the same lessons on hardware.

**How does the robot hand mark you?**
Your hand and the robot's both become 8 numbers: how bent and how spread each of 4 fingers is.
Each finger is marked; hold the shape 3 seconds to pass. Signs it can only approximate are labelled
"closest this hand can do", and it checks handshape only, not location, palm direction or face.

**Where do the robot's stances come from?**
We set the joint angles by hand in `config/stances.yaml` and show them on the open MuJoCo
Menagerie model of the Unitree G1. A physics mode exists (`rth sim`, with a safety rope); the booth
poses the robot directly. The robot did not learn the stances.

**What does Nebius do?**
Three roles, each called once per request or per stance, never per frame:
- an embedding model (Qwen3-Embedding-8B) matches what you type ("something for balance") to a stance;
- an LLM (Qwen3-235B) is the sensei: it picks the next stance and writes one line, median 0.8 s in our logs;
- an optional vision tip, switched off today.
With no key, a timeout or a bad reply, the rule-based sensei takes over and the demo carries on.

**What ML did you build yourselves?**
A stance recogniser: pose embeddings plus k-nearest neighbours, running every frame on the laptop.
So far it has only been tested on synthetic poses. *(Update this if the team recordings are done.)*

**What's faked or mocked?**
We couldn't get access to a real robot at the venue, so both robots run in a virtual world (MuJoCo,
with the real robots' open models). The recordings in the video are real runs: a real person, a real
webcam, live scoring. The stances are hand-posed. Nothing in the demo video is
staged: it's a screen recording of a real run at 13:49.

**Do you save camera images?**
No. Frames stay in memory on the laptop. The vision tip, the only thing that would send a still to
Nebius, is switched off.

**Which AI tools did you use?**
Claude Code helped write the code and this video (Remotion). The video's voice is OpenAI gpt-audio
and its music is Google Lyria 3, both through OpenRouter. Nebius models run inside the product.

**Is this medical or physio?**
No. It's exercise coaching only, with no medical claims.

**What's next?**
A real robot in the room: when you get a stance wrong, it walks over and guides your arm into place,
like a sensei would (the video shows this as a concept animation, clearly labelled "not built yet").
Before that: more stances, recording more people to test the recogniser on people it hasn't seen,
and letting the robot learn the stances in simulation instead of being posed by hand.

**Is the robot walking over to you real?**
No. That scene is a concept animation of the next step, and it says so on screen. Everything in the
two demo recordings is real.
