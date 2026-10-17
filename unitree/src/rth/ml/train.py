"""`rth train-recognizer`: evaluate the stance recogniser on unseen people, then fit it on everyone.

Evaluation is leave-one-person-out (LOPO): train on everyone else, test on the
held-out person, rotate. That is the honest number for a booth where every
visitor is a stranger. With a single person there is nobody to hold out, so it
falls back to a time split inside each recording and says so in the report.

Frames from one recording are strongly correlated, so the report also gives
per-recording accuracy (majority vote over the recording's frames), which
counts each performance once.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from rth.config import RecognizerSettings, load_settings, project_path
from rth.ml.dataset import NONE_LABEL, Dataset, load_dataset, summary_table
from rth.ml.embedding import PAIRS, embed_batch
from rth.ml.knn import KNNStanceClassifier

NO_STANCE = "no stance"  # predicted column: rejected, or voted into the 'none' class
TIME_SPLIT_TRAIN = 0.7
SINGLE_PERSON_MARGIN = 2.0  # between-person / within-person spread, roughly, on synthetic data
SEGMENT_MS = 2000  # with one person, the rejection threshold is learned across ~2 s chunks


def make_classifier(rs: RecognizerSettings) -> KNNStanceClassifier:
    return KNNStanceClassifier(
        k=rs.k,
        top_n_by_max=rs.top_n_by_max,
        forward_weight=rs.forward_weight,
        reject_quantile=rs.reject_quantile,
        reject_margin=rs.reject_margin,
    )


def cap_per_recording(ds: Dataset, max_frames: int) -> np.ndarray:
    """Mask keeping at most ``max_frames`` evenly spaced frames of each recording."""
    keep = np.zeros(len(ds), bool)
    for r in np.unique(ds.recording):
        idx = np.flatnonzero(ds.recording == r)
        if len(idx) > max_frames:
            idx = idx[np.linspace(0, len(idx) - 1, max_frames).round().astype(int)]
        keep[idx] = True
    return keep


def time_split(ds: Dataset, train_frac: float = TIME_SPLIT_TRAIN) -> tuple[np.ndarray, np.ndarray]:
    """(train, test) masks: the first part of every recording trains, the rest tests."""
    train = np.zeros(len(ds), bool)
    for r in np.unique(ds.recording):
        idx = np.flatnonzero(ds.recording == r)
        idx = idx[np.argsort(ds.timestamps_ms[idx], kind="stable")]
        train[idx[: round(len(idx) * train_frac)]] = True
    return train, ~train


def segment_groups(ds: Dataset) -> np.ndarray:
    """Split each recording into 2-10 consecutive chunks of about SEGMENT_MS each."""
    out = np.empty(len(ds), dtype=object)
    for r in np.unique(ds.recording):
        idx = np.flatnonzero(ds.recording == r)
        idx = idx[np.argsort(ds.timestamps_ms[idx], kind="stable")]
        span = int(ds.timestamps_ms[idx[-1]] - ds.timestamps_ms[idx[0]])
        n_chunks = int(np.clip(round(span / SEGMENT_MS), 2, 10))
        out[idx] = [f"{r}:{c}" for c in np.arange(len(idx)) * n_chunks // len(idx)]
    return out.astype(str)


def threshold_groups(ds: Dataset) -> np.ndarray:
    """Groups for learning the rejection threshold: people, or 2 s segments if only one person."""
    return ds.persons if len(ds.people) >= 2 else segment_groups(ds)


def fit(ds: Dataset, rs: RecognizerSettings) -> KNNStanceClassifier:
    """Fit on a dataset; with one person, widen the threshold learned within their recordings.

    Within one person the same stance varies much less than between people, so a
    threshold learned that way would reject most strangers.
    """
    clf = make_classifier(rs).fit(ds.landmarks, ds.labels, threshold_groups(ds))
    if len(ds.people) < 2:
        clf.threshold_ *= SINGLE_PERSON_MARGIN
    return clf


def predict_labels(clf: KNNStanceClassifier, landmarks: np.ndarray, min_share: float) -> np.ndarray:
    """Per-frame label: a stance id, or NO_STANCE when rejected or voted 'none'."""
    preds = clf.predict_batch(landmarks, min_share=min_share)
    return np.array([NO_STANCE if p in (None, NONE_LABEL) else p for p in preds], dtype=object)


def _correct(true: np.ndarray, pred: np.ndarray) -> np.ndarray:
    return (pred == true) | ((true == NONE_LABEL) & (pred == NO_STANCE))


def _majority(preds: np.ndarray) -> str:
    return Counter(preds.tolist()).most_common(1)[0][0]


def _acc(correct: np.ndarray) -> float | None:
    return float(correct.mean()) if len(correct) else None


def evaluate(ds: Dataset, rs: RecognizerSettings) -> dict:
    """Run LOPO (or the single-person time split) and return all metrics as plain data."""
    people = ds.people
    lopo = len(people) >= 2
    pred = np.full(len(ds), "", dtype=object)
    tested = np.zeros(len(ds), bool)
    folds = []
    if lopo:
        for person in people:
            test = ds.persons == person
            train = ~test
            clf = fit(ds.subset(train), rs)
            pred[test] = predict_labels(clf, ds.landmarks[test], rs.min_confidence)
            tested |= test
            folds.append({"held_out": person, "train_people": len(people) - 1,
                          "train_frames": int(train.sum()),
                          "test_frames": int(test.sum()), "threshold": clf.threshold_})
    else:
        train, test = time_split(ds)
        clf = fit(ds.subset(train), rs)
        pred[test] = predict_labels(clf, ds.landmarks[test], rs.min_confidence)
        tested = test
        folds.append({"held_out": f"last {1 - TIME_SPLIT_TRAIN:.0%} of each recording",
                      "train_people": 1,
                      "train_frames": int(train.sum()), "test_frames": int(test.sum()),
                      "threshold": clf.threshold_})

    t = ds.subset(tested)
    p = pred[tested]
    ok = _correct(t.labels, p)
    rows = t.stances
    cols = [s for s in rows if s != NONE_LABEL] + [NO_STANCE]
    confusion = np.zeros((len(rows), len(cols)), int)
    for true_label, pl in zip(t.labels, p):
        confusion[rows.index(true_label), cols.index(pl)] += 1

    recs = []
    for r in np.unique(t.recording):
        m = t.recording == r
        label = str(t.labels[m][0])
        vote = _majority(p[m])
        recs.append({"recording": ds.recordings[r], "person": str(t.persons[m][0]), "stance": label,
                     "frames": int(m.sum()), "frame_accuracy": float(ok[m].mean()), "vote": vote,
                     "correct": bool(_correct(np.array([label]), np.array([vote]))[0])})

    def rec_acc(sel: list[dict]) -> float | None:
        return float(np.mean([x["correct"] for x in sel])) if sel else None

    per_person = {
        person: {"frame_accuracy": _acc(ok[t.persons == person]), "frames": int((t.persons == person).sum()),
                 "recording_accuracy": rec_acc([x for x in recs if x["person"] == person]),
                 "recordings": sum(x["person"] == person for x in recs)}
        for person in t.people
    }
    per_stance = {}
    for s in rows:
        m = t.labels == s
        col = cols.index(NO_STANCE if s == NONE_LABEL else s)
        predicted = confusion[:, col].sum()
        per_stance[s] = {
            "frame_accuracy": _acc(ok[m]), "frames": int(m.sum()),
            "precision": float(confusion[rows.index(s), col] / predicted) if predicted else None,
            "recording_accuracy": rec_acc([x for x in recs if x["stance"] == s]),
            "recordings": sum(x["stance"] == s for x in recs),
        }
    n_stances = len([s for s in rows if s != NONE_LABEL])
    return {
        "protocol": "leave-one-person-out" if lopo else "time-split (single person)",
        "people": people,
        "stances": rows,
        "frames_tested": len(t),
        "overall": {
            "frame_accuracy": _acc(ok),
            "recording_accuracy": rec_acc(recs),
            "recordings_correct": sum(x["correct"] for x in recs),
            "recordings": len(recs),
            "no_stance_rate": float(np.mean(p == NO_STANCE)) if len(p) else None,
            "chance": 1.0 / max(len(rows), 1),
            "n_stances": n_stances,
        },
        "per_person": per_person,
        "per_stance": per_stance,
        "confusion": {"rows": rows, "cols": cols, "counts": confusion.tolist()},
        "recordings": recs,
        "folds": folds,
    }


# ------------------------------------------------------------------------- reports


def _pct(x: float | None) -> str:
    return "-" if x is None else f"{100 * x:.1f}%"


def write_markdown(res: dict, data_table: str, rs: RecognizerSettings, path: Path) -> None:
    o = res["overall"]
    lopo = res["protocol"] == "leave-one-person-out"
    lines = [
        "# Stance recogniser evaluation",
        "",
        f"Generated {res['generated_utc']}.",
        "",
        *(
            ["**These results are on SYNTHETIC poses, not real recordings. Not for the poster.**", ""]
            if "synthetic" in res.get("sources", {}) else []
        ),
        "## Headline",
        "",
        f"- Protocol: **{res['protocol']}**"
        + (f" over {len(res['people'])} people." if lopo else "."),
        (f"- Frame accuracy: **{_pct(o['frame_accuracy'])}** on {res['frames_tested']} test frames "
        f"(chance {_pct(o['chance'])} for {len(res['stances'])} classes)."),
        (f"- Recording accuracy (majority vote per recording): **{_pct(o['recording_accuracy'])}** "
        f"({o['recordings_correct']}/{o['recordings']})."),
        (f"- Frames where the recogniser said \"no stance\": {_pct(o['no_stance_rate'])} "
        "(counted as wrong unless the true label is `none`)."),
        "",
    ]
    if lopo:
        lines += ["Every test frame comes from a person the model never saw during training.", ""]
    else:
        lines += [
            ("**Only one person recorded, so this is NOT a test on unseen people.** The model trained "
            f"on the first {TIME_SPLIT_TRAIN:.0%} of each recording and was tested on the rest, from "
            "the same person. Expect lower accuracy on strangers; record more people for a real test."),
            "",
        ]
    lines += [
        "## Method",
        "",
        ("Each frame's 33 MediaPipe world landmarks become an embedding of "
        f"{len(PAIRS)} landmark-pair difference vectors in the body's own torso frame, divided "
        "by torso length (after MediaPipe's pose-classification embedder). Every training pose is "
        "also added mirrored (left/right swapped), so a stance done on the other side gets the same "
        f"label. k-NN as in MediaPipe: shortlist the {rs.top_n_by_max} training poses with the smallest "
        f"largest-difference, then the {rs.k} of those with the smallest mean difference vote. A "
        "neighbour only votes if it is within a distance threshold learned from how far the same stance "
        "sits between different training people; poses unlike any stance get no votes. A frame counts "
        f"as recognised when at least {rs.min_confidence:.0%} of the votes agree (the booth applies "
        "the same bar after smoothing over time)."),
        "",
        f"Training uses at most {rs.max_frames_per_recording} evenly spaced frames per recording.",
        "",
    ]
    if any(f["train_people"] < 2 for f in res["folds"]):
        lines += [
            ("Training data from a single person can't show how much a stance varies between people, "
            "so the rejection threshold learned across chunks of their recordings was widened "
            f"{SINGLE_PERSON_MARGIN:g}x. Record a third person to remove this guess."),
            "",
        ]
    lines += [
        "## Data",
        "",
        data_table,
        "",
        "## Per person" + (" (held out in turn)" if lopo else ""),
        "",
        "| person | frame accuracy | frames | recording accuracy | recordings |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for person, m in res["per_person"].items():
        lines.append(f"| {person} | {_pct(m['frame_accuracy'])} | {m['frames']} | "
                     f"{_pct(m['recording_accuracy'])} | {m['recordings']} |")
    lines += [
        "",
        "## Per stance",
        "",
        "| stance | recall (frame accuracy) | precision | frames | recording accuracy | recordings |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for s, m in res["per_stance"].items():
        lines.append(f"| {s} | {_pct(m['frame_accuracy'])} | {_pct(m['precision'])} | {m['frames']} | "
                     f"{_pct(m['recording_accuracy'])} | {m['recordings']} |")
    c = res["confusion"]
    lines += [
        "",
        "## Confusion matrix (frames; rows = true, columns = predicted)",
        "",
        "| true \\ predicted | " + " | ".join(c["cols"]) + " |",
        "| --- | " + " | ".join("---:" for _ in c["cols"]) + " |",
    ]
    for row, counts in zip(c["rows"], c["counts"]):
        lines.append(f"| {row} | " + " | ".join(str(n) for n in counts) + " |")
    lines += [
        "",
        "![confusion matrix](recognizer_confusion.png)",
        "",
        "## Caveats",
        "",
        ("- Frames within a recording are not independent; the recording-level number is the "
        "conservative one, but it rests on few recordings."),
        f"- {len(res['people'])} people is a small sample of body shapes and styles.",
        "- \"No stance\" rejection is tuned on the training people; random movement was "
        + ("included as the `none` class." if NONE_LABEL in res["stances"]
           else "not recorded, so its false-accept rate is not measured here."),
        "",
    ]
    path.write_text("\n".join(lines))


def _cell_pct(x: float) -> str:
    # Never round a non-zero cell to 0% or a non-perfect one to 100%.
    if 0 < x < 0.005:
        return "<1%"
    if 0.995 <= x < 1:
        return ">99%"
    return f"{100 * x:.0f}%"


def plot_confusion(res: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    c = res["confusion"]
    counts = np.array(c["counts"], float)
    share = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)
    cmap = LinearSegmentedColormap.from_list(
        "blue", ["#f7f9fc", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
    )
    n_rows, n_cols = share.shape
    fig, ax = plt.subplots(figsize=(1.0 * n_cols + 2.6, 0.8 * n_rows + 2.0), dpi=150)
    ax.imshow(share, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for i in range(n_rows):
        for j in range(n_cols):
            if counts[i, j] == 0:
                continue
            ink = "white" if share[i, j] > 0.55 else "#1f2328"
            ax.text(j, i, f"{_cell_pct(share[i, j])}\n{int(counts[i, j])}", ha="center",
                    va="center", fontsize=8, color=ink)
    ax.set_xticks(range(n_cols), c["cols"], rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(n_rows), c["rows"], fontsize=9)
    ax.set_xlabel("predicted", fontsize=9, color="#57606a")
    ax.set_ylabel("true stance", fontsize=9, color="#57606a")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(which="both", length=0)
    ax.set_xticks(np.arange(-0.5, n_cols), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    o = res["overall"]
    who = f"{len(res['people'])} people, leave-one-person-out" if res["protocol"].startswith(
        "leave") else "1 person, time split (not unseen people)"
    if "synthetic" in res.get("sources", {}):
        who += " - SYNTHETIC DATA"
    ax.set_title(f"Stance recogniser: {_pct(o['frame_accuracy'])} of frames, "
                 f"{_pct(o['recording_accuracy'])} of recordings\n{who}", fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _known_stances(stances_path: str) -> list[str] | None:
    import yaml

    from rth.stances import load_stances

    try:
        return load_stances(stances_path).ids()
    except (OSError, ValueError, yaml.YAMLError):
        return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="rth train-recognizer", description=__doc__.splitlines()[0])
    ap.add_argument("--recordings", help="recordings folder (default: settings.recognizer.recordings_dir)")
    ap.add_argument("--model", help="where to save the model (default: settings.recognizer.model_path)")
    ap.add_argument("--report-dir", help="report folder (default: settings.recognizer.report_dir)")
    ap.add_argument("--settings", default="config/settings.yaml")
    ap.add_argument("--no-save", action="store_true", help="evaluate only; don't write the model")
    args = ap.parse_args(argv)

    settings = load_settings(args.settings)
    rs = settings.recognizer
    rec_dir = project_path(args.recordings or rs.recordings_dir)
    model_path = project_path(args.model or rs.model_path)
    report_dir = project_path(args.report_dir or rs.report_dir)

    known = _known_stances(settings.stances_path)
    ds = load_dataset(rec_dir)
    if known is not None:
        unknown = sorted(set(ds.stances) - set(known) - {NONE_LABEL})
        if unknown:
            print(f"skipping recordings of unknown stances {unknown} (not in {settings.stances_path})")
            ds = ds.subset(np.isin(ds.labels, [*known, NONE_LABEL]))
    if len(ds):
        ds = ds.subset(~np.isnan(embed_batch(ds.landmarks)).any(axis=1))
    if len(ds) == 0:
        print(f"no usable recordings under {rec_dir}; record some with `rth record --person NAME --all`")
        return 1
    if len([s for s in ds.stances if s != NONE_LABEL]) < 2:
        print(f"need at least 2 stances to train, found {ds.stances}")
        return 1
    ds = ds.subset(cap_per_recording(ds, rs.max_frames_per_recording))

    table = summary_table(ds)
    print(f"{len(ds)} frames, {len(ds.recordings)} recordings, people: {', '.join(ds.people)}")
    print(table)
    if len(ds.people) < 2:
        print("only one person: falling back to a time-split hold-out (not a test on unseen people)")

    res = evaluate(ds, rs)
    used = np.unique(ds.recording)
    res["sources"] = dict(Counter(ds.sources[r].split(":")[0] for r in used)) if ds.sources else {}
    res["generated_utc"] = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    res["settings"] = rs.model_dump()
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "recognizer.json").write_text(json.dumps(res, indent=2))
    write_markdown(res, table, rs, report_dir / "recognizer.md")
    plot_confusion(res, report_dir / "recognizer_confusion.png")

    o = res["overall"]
    print(f"\n{res['protocol']}: frame accuracy {_pct(o['frame_accuracy'])}, recording accuracy "
          f"{_pct(o['recording_accuracy'])} ({o['recordings_correct']}/{o['recordings']}), "
          f"no-stance {_pct(o['no_stance_rate'])}")
    for person, m in res["per_person"].items():
        print(f"  {person:>12}: {_pct(m['frame_accuracy'])} of {m['frames']} frames")
    print(f"reports written to {report_dir}")

    if not args.no_save:
        clf = fit(ds, rs)
        meta = {"trained_utc": res["generated_utc"], "people": ds.people, "stances": ds.stances,
                "protocol": res["protocol"], "frame_accuracy": o["frame_accuracy"],
                "recording_accuracy": o["recording_accuracy"]}
        clf.save(model_path, meta)
        print(f"model saved to {model_path} ({clf.n_train} poses incl. mirrored, "
              f"threshold {clf.threshold_:.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
