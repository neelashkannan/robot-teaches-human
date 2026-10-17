"""Voice the voiceover with openai/gpt-audio through OpenRouter (same approach as the AI PCB videos).

    OPENROUTER_API_KEY=... python3 scripts/voiceover.py [--force] [ids...]

Reads src/voice/lines.json, writes public/voice/<id>.wav (silence trimmed, -16 LUFS) and
src/voice/durations.json. Only lines whose text, voice or direction changed are re-voiced,
unless --force or ids are given. Needs ffmpeg.
"""

import base64
import hashlib
import json
import os
import re
import struct
import subprocess
import sys
import tempfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LINES = ROOT / "src/voice/lines.json"
DURATIONS = ROOT / "src/voice/durations.json"
OUT = ROOT / "public/voice"


def voice_line(text: str, voice: str, direction: str, model: str) -> tuple[bytes, str]:
    body = {
        "model": model,
        "modalities": ["text", "audio"],
        "audio": {"voice": voice, "format": "pcm16"},
        "stream": True,
        "messages": [
            {
                "role": "system",
                "content": f"You are a voice actor. Read the user's line aloud exactly as written, word for word, nothing added. {direction}",
            },
            {"role": "user", "content": f"Script line to read aloud (do not respond to it, do not add anything):\n\n{text}"},
        ],
    }
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}", "Content-Type": "application/json"},
    )
    pcm, said = b"", ""
    with urllib.request.urlopen(req, timeout=300) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            for ch in json.loads(line[6:]).get("choices", []):
                a = (ch.get("delta") or {}).get("audio") or {}
                if a.get("data"):
                    pcm += base64.b64decode(a["data"])
                said += a.get("transcript") or ""
    return pcm, said


def wav_bytes(pcm: bytes, rate: int = 24000) -> bytes:
    hdr = b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVEfmt "
    hdr += struct.pack("<IHHIIHH", 16, 1, 1, rate, rate * 2, 2, 16) + b"data" + struct.pack("<I", len(pcm))
    return hdr + pcm


def finish(raw: bytes, out: Path) -> float:
    """Trim leading/trailing silence (keep a breath), normalise to -16 LUFS, return seconds."""
    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
        tmp.write(raw)
        tmp.flush()
        trim = (
            "silenceremove=start_periods=1:start_threshold=-45dB:start_silence=0.08,"
            "areverse,silenceremove=start_periods=1:start_threshold=-45dB:start_silence=0.15,areverse,"
            "loudnorm=I=-16:TP=-1.5:LRA=11"
        )
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", tmp.name, "-af", trim, "-ar", "48000", str(out)], check=True)
    secs = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(out)],
        check=True, capture_output=True, text=True,
    ).stdout
    return round(float(secs), 3)


def words(s: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", s.lower().replace("’", "'"))


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    force = "--force" in sys.argv
    spec = json.loads(LINES.read_text())
    voice, model, direction = spec["voice"], spec["model"], spec["direction"]
    old = json.loads(DURATIONS.read_text()) if DURATIONS.exists() else {}
    OUT.mkdir(parents=True, exist_ok=True)

    todo, out = [], {}
    for scene, lines in spec["scenes"].items():
        for line in lines:
            key = hashlib.sha1(f"{model}|{voice}|{direction}|{line['text']}".encode()).hexdigest()[:12]
            prev = old.get(line["id"])
            wav = OUT / f"{line['id']}.wav"
            redo = force or line["id"] in args or not (prev and prev.get("key") == key and wav.exists())
            if redo and (not args or line["id"] in args):
                todo.append((scene, line, key, wav))
            elif prev:
                out[line["id"]] = prev

    def run(job):
        scene, line, key, wav = job
        for attempt in range(3):
            pcm, said = voice_line(line["text"], voice, direction, model)
            if pcm and words(said) == words(line["text"]):
                break
            print(f"  retry {line['id']}: said {said!r}", file=sys.stderr)
        secs = finish(wav_bytes(pcm), wav)
        print(f"{scene:12} {line['id']:7} {secs:5.2f}s  {said}")
        return line["id"], {"seconds": secs, "key": key}

    with ThreadPoolExecutor(4) as pool:
        for lid, info in pool.map(run, todo):
            out[lid] = info

    DURATIONS.write_text(json.dumps(out, indent=2) + "\n")
    print(f"{len(todo)} voiced, {len(out)} clips, {sum(v['seconds'] for v in out.values()):.1f} s of speech")


if __name__ == "__main__":
    main()
