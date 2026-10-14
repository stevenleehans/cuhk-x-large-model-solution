"""Recover the HAU *take* from Depth motion, and use it to decode `emotion`.

HAU paths are `HAU/<user>/<a>-<b>-<take>` and the take digit encodes performance
speed: take 1 slow (FAST manner 0.7%), take 2 normal, take 3 fast (65.6%).
Conditioning the emotion prior on the true take lifts emotion 0.315 -> 0.723, and
the option text does not leak it (0.315 vs 0.334 chance). Test paths hide the
take, so we predict it from motion statistics instead.

The usable signal is *duration per action*, not motion energy: takes differ in
pace, but absolute duration confounds with scene complexity, so a clip's length is
only meaningful relative to how many actions it contains. duration/|S| alone
predicts the take at 0.485 (chance 0.334) and beats all 15 motion statistics
combined (0.404) — adding them back makes it worse. |S| is taken from the
structural posterior, which is available at test time.

Predictions are kept soft: emotion is scored under P(take=k) rather than a hard
argmax, so a shaky take estimate degrades gracefully to the plain text prior.
End to end this lifts emotion 0.336 -> 0.446 (oracle take would give 0.723).
"""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

import common
TAKES = ("1", "2", "3")
COLS = ["dur_per_action"]


def take_of(path: str) -> str | None:
    """Take digit from a HAU training path; None for HARn and for test clips."""
    parts = str(path).replace("\\", "/").split("/")
    if parts[0] != "HAU" or len(parts) < 3:
        return None
    bits = parts[2].split("-")
    return bits[2] if len(bits) == 3 and bits[2] in TAKES else None


def load_features(split: str) -> pd.DataFrame:
    path = common.ROOT / "data" / f"depth_features_{split}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"{path} - run: python src/depth_features.py {split}")
    return pd.read_csv(path)


class TakeModel:
    """P(take | Depth motion features), trained on HAU training clips."""

    def __init__(self):
        self.model = None

    @staticmethod
    def _table(feats: pd.DataFrame, n_actions: dict[str, float]) -> pd.DataFrame:
        usable = feats[feats["found"]].copy()
        counts = usable["clip_id"].map(lambda c: max(1.0, n_actions.get(c, 3.8)))
        usable["dur_per_action"] = usable["duration_s"] / counts
        return usable

    def fit(self, train: pd.DataFrame, feats: pd.DataFrame,
            n_actions: dict[str, float]) -> "TakeModel":
        clips = train[train["source"] == "HAU"][["clip_id", "path"]].drop_duplicates()
        clips["take"] = clips["path"].map(take_of)
        clips = clips.dropna(subset=["take"])
        merged = clips.merge(self._table(feats, n_actions), on="clip_id")
        if merged.empty:
            return self
        self.model = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.06, max_depth=4, random_state=0
        ).fit(merged[COLS].values, merged["take"].values)
        return self

    def predict_proba(self, feats: pd.DataFrame,
                      n_actions: dict[str, float]) -> dict[str, dict[str, float]]:
        if self.model is None:
            return {}
        usable = self._table(feats, n_actions)
        if usable.empty:
            return {}
        p = self.model.predict_proba(usable[COLS].values)
        classes = list(self.model.classes_)
        return {clip: {c: float(row[i]) for i, c in enumerate(classes)}
                for clip, row in zip(usable["clip_id"], p)}


class EmotionModel:
    """P(option correct | take, actions present), marginalised over the take.

    The take (pace) is the dominant signal, but the actions carry more on top of
    it: conditioning on both lifts the oracle-take ceiling from 0.723 to 0.760.
    Backs off action -> take -> global frequency, so thin cells degrade safely.
    """

    SMOOTHING = 4.0

    def __init__(self):
        self.hit = Counter()
        self.seen = Counter()
        self.hit0 = Counter()
        self.seen0 = Counter()
        self.hit_a = Counter()
        self.seen_a = Counter()

    def fit(self, train: pd.DataFrame,
            actions_by_clip: dict[str, set[str]] | None = None) -> "EmotionModel":
        actions_by_clip = actions_by_clip or {}
        rows = train[train["category"] == "emotion"]
        for row in rows.itertuples(index=False):
            r = row._asdict()
            take = take_of(r["path"])
            present = actions_by_clip.get(r["clip_id"], ())
            for L in common.options(r):
                correct = L in r["answer"]
                self.seen0[r[L]] += 1
                if correct:
                    self.hit0[r[L]] += 1
                if take:
                    self.seen[(take, r[L])] += 1
                    if correct:
                        self.hit[(take, r[L])] += 1
                    for a in present:
                        self.seen_a[(take, a, r[L])] += 1
                        if correct:
                            self.hit_a[(take, a, r[L])] += 1
        return self

    def _backoff(self, text: str) -> float:
        return (self.hit0[text] + self.SMOOTHING * 0.25) / (self.seen0[text] + self.SMOOTHING)

    def score(self, r, take_probs: dict[str, float] | None,
              present: set[str] | None = None) -> dict[str, float]:
        out = {}
        for L in common.options(r):
            base = self._backoff(r[L])
            if not take_probs:
                out[L] = base
                continue
            total = 0.0
            for k, w in take_probs.items():
                p = ((self.hit[(k, r[L])] + self.SMOOTHING * base)
                     / (self.seen[(k, r[L])] + self.SMOOTHING))
                if present:
                    num = sum(self.hit_a[(k, a, r[L])] for a in present)
                    den = sum(self.seen_a[(k, a, r[L])] for a in present)
                    p = (num + self.SMOOTHING * p) / (den + self.SMOOTHING)
                total += w * p
            out[L] = total
        return out
