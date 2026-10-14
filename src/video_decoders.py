"""Decoders for the second kernel's outputs: a video take head and windowed scores.

Both target slices the structural model cannot reach.

* `video_take` replaces the duration/|S| take estimate. Duration recovers the take
  at only 0.43 while the oracle take is worth 0.723 on emotion, so a learned pace
  head has a lot of headroom.
* `sequence_order` orders a question's four options by *when* each action peaks
  across sliding windows. The structural precedence prior reaches 0.438 using only
  which actions tend to precede which; window peaks say what actually happened in
  this clip.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np

import actions as action_vocab
import common

SCORES = common.ROOT / "data" / "video_scores"
WINDOWS = common.ROOT / "data" / "video_windows"
TAKES = ("1", "2", "3")


def load_take_probs(split: str) -> dict[str, dict[str, float]]:
    """clip_id -> {take: P(take)} from the video head."""
    name = "oof_take" if split == "training" else "test_take"
    path = WINDOWS / f"{name}.npy"
    if not path.is_file():
        return {}
    matrix = np.load(path)
    index = json.loads((common.ROOT / "data" / "clips" /
                        f"{split}_Depth_index.json").read_text())
    return {clip: {t: float(matrix[i][j]) for j, t in enumerate(TAKES)}
            for i, clip in enumerate(index)}


def load_windows(split: str) -> tuple[dict[str, np.ndarray], list[str]]:
    """clip_id -> (n_windows, n_actions) probabilities."""
    name = "train_windows" if split == "training" else "test_windows"
    path = WINDOWS / f"{name}.npy"
    if not path.is_file():
        return {}, []
    matrix = np.load(path)
    meta = json.loads((WINDOWS / "meta_out.json").read_text())
    index = json.loads((common.ROOT / "data" / "clips" /
                        f"{split}_Depth_index.json").read_text())
    return {clip: matrix[i] for i, clip in enumerate(index)}, meta["actions"]


def peak_positions(row, windows: np.ndarray, action_list: list[str]) -> dict[str, float]:
    """Soft temporal centroid of each option's action across windows.

    A centroid weighted by per-window probability is steadier than an argmax when
    an action is detected weakly in several adjacent windows.
    """
    out = {}
    n_windows = windows.shape[0]
    for L in common.options(row):
        names = [a for a in action_vocab.canonical_set(row[L]) if a in action_list]
        if not names:
            continue
        profile = np.max([windows[:, action_list.index(a)] for a in names], axis=0)
        weight = profile.sum()
        out[L] = (float((profile * np.arange(n_windows)).sum() / weight)
                  if weight > 1e-6 else (n_windows - 1) / 2)
    return out


def sequence_order(row, windows: np.ndarray, action_list: list[str],
                   prior_score, blend: float = 0.5) -> str:
    """Rank permutations by agreement with window peaks, blended with the prior."""
    peaks = peak_positions(row, windows, action_list)
    live = common.options(row)
    if len(peaks) < len(live):
        return prior_score(row)
    best = None
    for perm in itertools.permutations(live):
        # penalise orderings that disagree with the observed temporal centroids
        agree = -sum(abs(peaks[L] - i * (len(perm) - 1) / max(1, len(perm) - 1))
                     for i, L in enumerate(perm))
        ranks = {L: i for i, L in enumerate(perm)}
        inversions = sum(1 for a, b in itertools.combinations(live, 2)
                         if (peaks[a] - peaks[b]) * (ranks[a] - ranks[b]) < 0)
        score = -inversions + blend * agree
        cand = (score, "".join(perm))
        if best is None or cand > best:
            best = cand
    return best[1]
