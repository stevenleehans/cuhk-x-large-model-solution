"""Fuse video evidence into the structural action-set posterior.

The structural model already recovers S at per-action AUC 0.956, and oracle S
decodes single/multi/combination at 1.00/1.00/0.999. So the video's whole job is
to sharpen P(a in S) — it does not need to answer questions directly.

Video evidence enters as a log-odds increment on each candidate action, which
keeps the constraint propagation and the answer grammar untouched.
"""
from __future__ import annotations

import math

import pandas as pd

import actions
from action_set import ActionSetModel
from common import options
from take_model import EmotionModel


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class FusedModel(ActionSetModel):
    """ActionSetModel + a per-clip, per-action video presence score.

    `video_scores` maps clip_id -> {canonical_action: P(action present from video)};
    `take_probs` maps clip_id -> {take: P(take)} and drives `emotion`.
    Absent clips or actions fall back to the structural model alone.
    """

    def __init__(self, video_scores: dict[str, dict[str, float]] | None = None,
                 take_probs: dict[str, dict[str, float]] | None = None,
                 weight: float = 1.0, C: float = 1.0):
        super().__init__(C=C)
        self.video_scores = video_scores or {}
        self.weight = weight
        if not math.isfinite(weight) or weight < 0:
            raise ValueError("video weight must be finite and nonnegative")
        for clip, scores in self.video_scores.items():
            for action, p in scores.items():
                if action not in actions.INDEX or not math.isfinite(p) or not 0 <= p <= 1:
                    raise ValueError(f"Invalid video score for {clip}/{action}")
        self.take_probs = take_probs or {}
        self.emotion = EmotionModel()

    def fit(self, train):
        super().fit(train)
        self.emotion.fit(train)
        return self

    def posteriors(self, group: pd.DataFrame) -> dict[str, float]:
        post = super().posteriors(group)
        clip_id = group["clip_id"].iloc[0]
        return self._fuse(clip_id, post)

    def _video_score(self, clip_id: str, text: str) -> float | None:
        scores = self.video_scores.get(clip_id, {})
        names = actions.canonical_set(text)
        if not names or any(name not in scores for name in names):
            return None
        # A phrase may denote either smartphone use or gaming. Use strongest
        # compatible evidence without assuming independent action probabilities.
        return max(scores[name] for name in names)

    def _fuse(self, clip_id: str, post: dict[str, float]) -> dict[str, float]:
        if not post or self.weight == 0:
            return post
        out = {}
        for text, p in post.items():
            video = self._video_score(clip_id, text)
            out[text] = p if video is None else _sigmoid(
                max(-700.0, min(700.0, _logit(p) + self.weight * _logit(video))))
        return out

    def posteriors_batch(self, frame: pd.DataFrame) -> dict[str, dict[str, float]]:
        return {clip: self._fuse(clip, post)
                for clip, post in super().posteriors_batch(frame).items()}

    def predict(self, frame: pd.DataFrame) -> list[str]:
        """Decode HAU with fused evidence and HARn single directly from video."""
        base = super().predict(frame)
        out = list(base)
        for i, row in enumerate(frame.itertuples(index=False)):
            r = row._asdict()
            if r["category"] == "single" and r["source"] == "HARn" and self.weight > 0:
                scores = {L: self._video_score(r["clip_id"], r[L]) for L in options(r)}
                if all(p is not None for p in scores.values()):
                    out[i] = max(scores, key=scores.get)
            elif r["category"] == "emotion" and self.take_probs.get(r["clip_id"]):
                scores = self.emotion.score(r, self.take_probs[r["clip_id"]])
                out[i] = max(scores, key=scores.get)
        return out
