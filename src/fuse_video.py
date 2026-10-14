"""Fold the Kaggle-trained action scores into the structural model.

Uses the kernel's OUT-OF-FOLD training scores, never in-sample ones: the fusion
weight is fitted against them, and in-sample base-model outputs would make the
video look far more reliable than it is and push the weight too high.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

import common
from pipeline import Pipeline
from take_model import load_features

SCORES = common.ROOT / "data" / "video_scores_ens"


def load_scores_from(dirname: str, split: str) -> dict[str, dict[str, float]]:
    """Load a specific run's scores. HARn `single` and HAU fusion use different
    runs (ensemble vs v5), so the directory is explicit."""
    root = common.ROOT / "data" / dirname
    name = "oof" if split == "training" else "test"
    matrix = np.load(root / f"{name}_scores.npy")
    actions = json.loads((common.ROOT / "data" / "video_scores" / "actions.json").read_text())
    index = json.loads((common.ROOT / "data" / "clips" /
                        f"{split}_Depth_index.json").read_text())
    return {clip: {a: float(matrix[i][j]) for j, a in enumerate(actions)}
            for i, clip in enumerate(index)}


def load_scores(split: str) -> dict[str, dict[str, float]]:
    name = "oof" if split == "training" else "test"
    matrix = np.load(SCORES / f"{name}_scores.npy")
    actions = json.loads((SCORES / "actions.json").read_text())
    index = json.loads((common.ROOT / "data" / "clips" /
                        f"{split}_Depth_index.json").read_text())
    return {clip: {a: float(matrix[i][j]) for j, a in enumerate(actions)}
            for i, clip in enumerate(index)}


def cross_validate(weight: float, video: dict, feats, train) -> tuple[dict, dict]:
    hits, total = defaultdict(int), defaultdict(int)
    for tr, va in common.subject_folds(train):
        model = Pipeline(video_scores=video, video_weight=weight).fit(tr, feats)
        for (_, r), p in zip(va.iterrows(), model.predict(va, feats)):
            for key in (f"{r['source']}/{r['category']}", r["category"]):
                total[key] += 1
                hits[key] += p == r["answer"]
    return {k: hits[k] / total[k] for k in total}, total


if __name__ == "__main__":
    train = common.load("training")
    feats = load_features("training")
    video = load_scores("training")
    best = None
    for weight in (0.5, 0.75, 1.0, 1.25):
        acc, total = cross_validate(weight, video, feats, train)
        score = common.lb_estimate({c: acc[c] for c in common.TEST_COUNTS})
        print(f"  weight {weight:4.2f}  CV {score:.4f}   "
              f"multi {acc['multi']:.3f}  single {acc['single']:.3f}  "
              f"obj {acc['object_interaction']:.3f}", flush=True)
        if best is None or score > best[0]:
            best = (score, weight, acc)
    print(f"\nbest weight {best[1]} -> CV {best[0]:.4f}")
    for k in sorted(best[2]):
        print(f"  {k:22s} {best[2][k]:.4f}")
    (common.ROOT / "data" / "video_weight.json").write_text(json.dumps({"weight": best[1]}))
