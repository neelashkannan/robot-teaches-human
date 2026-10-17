"""Print the voiceover transcript with video timestamps (markdown table) for SCRIPT.md.

    python3 scripts/vo_table.py

Scene lengths come from `npx remotion compositions`, so they match the render.
"""
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FPS = 30
IN_T = {"Hype": 12, "UnitreeDemo": 8}  # crossfade into each scene (frames); default 15
NAMES = {
    "Flip": "The flip", "Title": "Title", "Why": "Why a robot", "Virtual": "No robot? Virtual dojo",
    "ArmDemo": "Teacher 1: robot hand", "Hype": "Hand → whole body", "UnitreeDemo": "Teacher 2: Unitree G1",
    "Trick": "How it marks you fairly", "Stack": "Tech stack", "Limits": "Honest limits",
    "Next": "What's next (concept)", "Close": "Close",
}

out = subprocess.run(["npx", "remotion", "compositions", "src/index.ts"], cwd=ROOT, capture_output=True, text=True).stdout
frames = {m[0]: int(m[1]) for m in re.findall(r"^(\w+)\s+30\s+1920x1080\s+(\d+)", out, re.M)}
spec = json.loads((ROOT / "src/voice/lines.json").read_text())["scenes"]
dur = json.loads((ROOT / "src/voice/durations.json").read_text())
order = [s for s in NAMES if s in frames]

t, rows = 0, ["| Time | Scene | Voiceover |", "| --- | --- | --- |"]
for i, scene in enumerate(order):
    if i:
        t += frames[order[i - 1]] - IN_T.get(scene, 15)
    end = 0.0
    for line in spec.get(scene, []):
        start = max(line["at"], 0 if end == 0 else end + 0.3)
        end = start + dur[line["id"]]["seconds"]
        g = t / FPS + start
        rows.append(f"| {int(g // 60)}:{int(g % 60):02d} | {NAMES[scene]} | {line['text']} |")
total = (t + frames[order[-1]]) / FPS
print("\n".join(rows))
print(f"\nTotal: {int(total // 60)}:{total % 60:04.1f}")
