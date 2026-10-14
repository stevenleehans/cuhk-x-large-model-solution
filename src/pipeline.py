"""Full model: structural action-set decoding + Depth-video take for `emotion`.

Fit order matters. The take model needs duration normalised by how many actions a
clip contains, and that count comes from the structural posterior — so the
action-set model is fitted first and its E[|S|] feeds the take model.
"""
from __future__ import annotations

from collections import defaultdict

import pandas as pd

import common
import actions as action_vocab
from action_set import ActionSetModel
from common import options
from take_model import EmotionModel, TakeModel, load_features


class _VideoPosterior:
    """Pickle-safe callable; always reads the pipeline's current score mapping."""

    def __init__(self, pipeline):
        self.pipeline = pipeline

    def __call__(self, frame):
        import math
        owner = self.pipeline
        # Call the class implementation, avoiding recursion through this instance hook.
        out = ActionSetModel.posteriors_batch(owner.structural, frame)

        def logit(p):
            p = min(max(float(p), 1e-6), 1-1e-6)
            return math.log(p/(1-p))

        for clip_id, post in out.items():
            vid = owner.video_scores.get(clip_id)
            if not vid:
                continue
            for text in list(post):
                name = action_vocab.canonical(text)
                if name in vid:
                    z = logit(post[text]) + owner.video_weight * logit(vid[name])
                    post[text] = 1.0/(1.0+math.exp(-z))
        return out


class Pipeline:
    """Structural model, optionally sharpened by video action scores, plus the
    Depth take model for `emotion`.

    `video_scores` maps clip_id -> {canonical_action: P(present)}. It enters as a
    log-odds increment on the action posterior, so the constraint propagation and
    answer grammar are untouched; `video_weight` scales how much it is trusted.
    """

    def __init__(self, video_scores: dict[str, dict[str, float]] | None = None,
                 video_weight: float = 1.0,
                 harn_scores: dict[str, dict[str, float]] | None = None):
        self.structural = ActionSetModel()
        self.take = TakeModel()
        self.emotion = EmotionModel()
        self.video_scores = video_scores or {}
        # HARn `single` and HAU fusion want *different* video models: the 3-run
        # ensemble is best at ranking three offered options (0.786 vs 0.744) but
        # dilutes the HAU action posterior (multi 0.829 vs 0.854). Keep them apart.
        self.harn_scores = harn_scores or video_scores or {}
        self.video_weight = video_weight
        self.train_feats: pd.DataFrame | None = None
        if self.video_scores and self.video_weight > 0:
            self._install_video_hook()

    def _install_video_hook(self):
        """Wrap the structural posterior so video evidence is folded in before the
        constraint propagation and decoding run."""
        self.structural.posteriors_batch = _VideoPosterior(self)

    @staticmethod
    def _present(structural, frame, threshold: float = 0.5) -> dict[str, set[str]]:
        """Actions the structural model believes are in each clip, as canonical ids."""
        import actions as action_vocab
        out = {}
        for clip_id, post in structural.posteriors_batch(frame).items():
            names = set()
            for text, p in post.items():
                if p > threshold:
                    name = action_vocab.canonical(text)
                    if name:
                        names.add(name)
            out[clip_id] = names
        return out

    def fit(self, train: pd.DataFrame, feats: pd.DataFrame | None = None) -> "Pipeline":
        self.structural.fit(train)
        # Emotion conditions on the take ONLY. Adding the inferred action set looked
        # better in CV (+0.0015) but cost 4 LB questions (0.74269 -> 0.73099): the
        # (take, action, adverb) cells are too thin once the take itself is only
        # 0.485 accurate. Revisit if the video model sharpens the take.
        self.emotion.fit(train)
        if feats is not None:
            sizes = self.structural.expected_sizes(train)
            self.take.fit(train, feats, sizes)
            self.train_feats = feats
        return self

    def _harn_single(self, r) -> str | None:
        """HARn `single` from the video scores.

        The structural model is helpless here (0.364, at the 1/3 floor) because the
        43 distractors are sampled near-uniformly under one question template. The
        video model's raw 44-way top-1 is only 0.262, but the question just needs
        the best of *three* offered options, and there it scores 0.744.
        """
        vid = self.harn_scores.get(r["clip_id"])
        if not vid:
            return None
        best = None
        for L in options(r):
            names = action_vocab.canonical_set(r[L])
            score = max((vid[a] for a in names if a in vid), default=-1.0)
            if best is None or score > best[0]:
                best = (score, L)
        return best[1] if best and best[0] >= 0 else None

    def predict(self, frame: pd.DataFrame,
                feats: pd.DataFrame | None = None) -> list[str]:
        preds = self.structural.predict(frame)
        if self.harn_scores:
            preds = list(preds)
            for i, row in enumerate(frame.itertuples(index=False)):
                r = row._asdict()
                if r["source"] == "HARn" and r["category"] == "single":
                    got = self._harn_single(r)
                    if got:
                        preds[i] = got
        if feats is None or self.take.model is None:
            return preds
        sizes = self.structural.expected_sizes(frame)
        take_probs = self.take.predict_proba(feats, sizes)
        out = list(preds)
        for i, row in enumerate(frame.itertuples(index=False)):
            r = row._asdict()
            if r["category"] != "emotion":
                continue
            scores = self.emotion.score(r, take_probs.get(r["clip_id"]))
            out[i] = max(scores, key=scores.get)
        return out


def cross_validate(n_folds: int = 6):
    train = common.load("training")
    feats = load_features("training")
    hits, total = defaultdict(int), defaultdict(int)
    for tr, va in common.subject_folds(train, n_folds):
        model = Pipeline().fit(tr, feats)
        preds = model.predict(va, feats)
        for (_, r), p in zip(va.iterrows(), preds):
            for key in (f"{r['source']}/{r['category']}", r["category"]):
                total[key] += 1
                hits[key] += p == r["answer"]
    return {k: hits[k] / total[k] for k in total}, total


if __name__ == "__main__":
    acc, total = cross_validate()
    for k in sorted(total, key=lambda k: -total[k]):
        if "/" in k:
            print(f"  {k:28s} {acc[k]:.4f}  (n={total[k]})")
    per = {c: acc[c] for c in common.TEST_COUNTS}
    e = common.lb_estimate(per)
    print(f"\n  CV {e:.4f}")
