"""Clip-graph decoder: the strongest media-free model.

The 208 test clips carry 682 questions, so most clips have 4-5 linked questions
about the same footage. Every question's option list is (clip's real actions) +
freshly sampled distractors, so an action that recurs across a clip's *other*
questions is very likely real: mean recurrence is 3.15 for correct options
against 1.69 for distractors.

Scoring is a logistic model over per-option features (global text prior +
within-clip recurrence), fitted subject-disjoint; decoding then applies the
per-category answer grammar.
"""
from __future__ import annotations

import itertools
import math
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from common import LABELS, options
from text_prior import TextPrior

FEATURES = [
    "prior_logit", "mentions_other", "mention_frac", "mentions_othercat",
    "n_questions", "is_multi_action", "clip_actions", "share_max",
]


def _atoms(value: str) -> list[str]:
    """A combination option is a comma-joined action tuple; others are one action."""
    return [a.strip() for a in str(value).split(",") if a.strip()]


def _clip_mentions(group: pd.DataFrame):
    """Per-clip: how many questions mention each action, and in which categories."""
    per_question = defaultdict(set)
    per_category = defaultdict(set)
    for row in group.itertuples(index=False):
        r = row._asdict()
        if r["category"] == "emotion":       # adverbs share no vocabulary with actions
            continue
        for L in options(r):
            for a in _atoms(r[L]):
                per_question[a].add(r["qa_id"])
                per_category[a].add(r["category"])
    return per_question, per_category


class ClipGraph:
    def __init__(self, C: float = 1.0):
        self.prior = TextPrior()
        self.C = C
        self.model: LogisticRegression | None = None

    # ---------- features ----------
    def _rows(self, frame: pd.DataFrame, labelled: bool):
        X, y, index = [], [], []
        for clip_id, group in frame.groupby("clip_id"):
            per_question, per_category = _clip_mentions(group)
            n_q = len(group)
            for row in group.itertuples(index=False):
                r = row._asdict()
                probs = self.prior.option_probs(r)
                live = options(r)
                raw = {}
                for L in live:
                    atoms = _atoms(r[L])
                    other = np.mean([len(per_question[a] - {r["qa_id"]}) for a in atoms])
                    cats = np.mean([len(per_category[a] - {r["category"]}) for a in atoms])
                    raw[L] = (other, cats)
                mx = max((v[0] for v in raw.values()), default=0.0) or 1.0
                for L in live:
                    other, cats = raw[L]
                    p = probs[L]
                    X.append([
                        math.log(p / (1 - p)),
                        other,
                        other / max(1, n_q - 1),
                        cats,
                        n_q,
                        float(r["source"] == "HAU"),
                        len(per_question),
                        other / mx,
                    ])
                    index.append((r["qa_id"], L))
                    if labelled:
                        y.append(int(L in r["answer"]))
        return np.asarray(X, dtype=float), np.asarray(y), index

    # ---------- fit / predict ----------
    def fit(self, train: pd.DataFrame) -> "ClipGraph":
        self.prior.fit(train)
        X, y, _ = self._rows(train, labelled=True)
        self.model = LogisticRegression(C=self.C, max_iter=2000).fit(X, y)
        return self

    def option_scores(self, frame: pd.DataFrame) -> dict[tuple[str, str], float]:
        X, _, index = self._rows(frame, labelled=False)
        p = self.model.predict_proba(X)[:, 1]
        return {k: v for k, v in zip(index, p)}

    def predict(self, frame: pd.DataFrame) -> list[str]:
        scores = self.option_scores(frame)
        out = []
        for row in frame.itertuples(index=False):
            r = row._asdict()
            live = options(r)
            s = {L: scores[(r["qa_id"], L)] for L in live}
            cat = r["category"]
            if cat == "sequence":
                out.append(self.prior._sequence(r))
            elif cat == "multi":
                chosen = [L for L in live if s[L] > 0.5] or [max(live, key=s.get)]
                out.append("".join(sorted(chosen)))
            else:
                out.append(max(live, key=s.get))
        return out
