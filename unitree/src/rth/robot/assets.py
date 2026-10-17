"""Download the G1 model and retargeted LAFAN1 clips into assets/ (idempotent).

Sources
-------
* Unitree G1 MJCF + meshes from MuJoCo Menagerie (google-deepmind/mujoco_menagerie,
  folder ``unitree_g1``, BSD-3-Clause). Only that folder is fetched.
* LAFAN1 motion retargeted to the G1, from the Hugging Face dataset
  ``lvhaidong/LAFAN1_Retargeting_Dataset``. The motion data derives from Ubisoft
  LAFAN1, licensed CC BY-NC-ND 4.0: non-commercial use only, no redistribution of
  modified clips. Fine for a hackathon booth; don't ship it in a product.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import urllib.request
from pathlib import Path

from rth.config import project_path

MENAGERIE_REPO = "https://github.com/google-deepmind/mujoco_menagerie.git"
MENAGERIE_API = "https://api.github.com/repos/google-deepmind/mujoco_menagerie"
MENAGERIE_RAW = "https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie"
MENAGERIE_DIR = "assets/mujoco_menagerie"
G1_FOLDER = "unitree_g1"

LAFAN_REPO = "lvhaidong/LAFAN1_Retargeting_Dataset"
LAFAN_DIR = "assets/lafan1"
# ~2.5 MB each. Fighting and dance clips hold the martial-arts-like frames; a walk
# gives plain standing frames.
LAFAN_CLIPS = [
    "fight1_subject2",
    "fight1_subject3",
    "fight1_subject5",
    "fightAndSports1_subject1",
    "fightAndSports1_subject4",
    "dance1_subject1",
    "dance2_subject1",
    "walk1_subject1",
    "jumps1_subject1",
]


def _g1_present(root: Path) -> bool:
    g1 = root / G1_FOLDER
    return (g1 / "scene.xml").exists() and (g1 / "g1.xml").exists() and (g1 / "assets").is_dir()


def _fetch_menagerie_git(root: Path) -> None:
    """Shallow, blob-less, sparse clone: only unitree_g1 blobs are downloaded."""
    if root.exists():
        shutil.rmtree(root)
    root.parent.mkdir(parents=True, exist_ok=True)
    clone = ["git", "clone", "--depth=1", "--filter=blob:none", "--sparse", MENAGERIE_REPO]
    subprocess.run([*clone, str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "sparse-checkout", "set", G1_FOLDER], check=True)


def _fetch_menagerie_http(root: Path) -> None:
    """Fallback without git: list the tree via the GitHub API and fetch each file."""
    with urllib.request.urlopen(f"{MENAGERIE_API}/git/trees/main?recursive=1", timeout=30) as r:
        tree = json.load(r)["tree"]
    files = [
        t["path"] for t in tree if t["type"] == "blob" and t["path"].startswith(G1_FOLDER + "/")
    ]
    for rel in files:
        dest = root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(f"{MENAGERIE_RAW}/main/{rel}", dest)
    print(f"  downloaded {len(files)} files")


def fetch_g1(force: bool = False) -> Path:
    root = project_path(MENAGERIE_DIR)
    if _g1_present(root) and not force:
        print(f"G1 model: already in {root / G1_FOLDER}")
        return root / G1_FOLDER
    print(f"G1 model: fetching {G1_FOLDER} from MuJoCo Menagerie into {root}")
    try:
        _fetch_menagerie_git(root)
    except (OSError, subprocess.CalledProcessError) as e:
        print(f"  git sparse checkout failed ({e}); falling back to per-file download")
        _fetch_menagerie_http(root)
    if not _g1_present(root):
        raise RuntimeError(f"G1 model missing after download: {root / G1_FOLDER}")
    return root / G1_FOLDER


def fetch_clips(force: bool = False, clips: list[str] | None = None) -> Path:
    from huggingface_hub import hf_hub_download

    root = project_path(LAFAN_DIR)
    out = root / "g1"
    out.mkdir(parents=True, exist_ok=True)
    for name in clips or LAFAN_CLIPS:
        dest = out / f"{name}.csv"
        if dest.exists() and not force:
            print(f"LAFAN1 clip: already have {dest.name}")
            continue
        print(f"LAFAN1 clip: downloading g1/{name}.csv")
        with tempfile.TemporaryDirectory() as tmp:
            src = hf_hub_download(LAFAN_REPO, f"g1/{name}.csv", repo_type="dataset", local_dir=tmp)
            shutil.move(src, dest)
    return out


def fetch(force: bool = False) -> None:
    """Download the G1 model and the LAFAN1 G1 clips if they're not already present."""
    fetch_g1(force)
    fetch_clips(force)
