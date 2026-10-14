"""Latent action-set model — the core structural engine.

Each HAU clip has a latent set S of actions that actually occur. Every question's
options are (some members of S) + distractors resampled from the 40-action
vocabulary. Two facts verified on training data make S the right latent variable:

  * all four `sequence` options are members of S (308/308 clips);
  * a `single` question's correct option is in S 93.9% of the time, while its
    distractors are in S 0/2427 times.

So we infer P(a in S) per clip from the option lists alone, then decode every
question of that clip from the shared posterior. Evidence is pooled across the
clip instead of being spent per-question, which is what makes this far stronger
than scoring options independently.
"""
from __future__ import annotations

import itertools
import math
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier


def _make_model():
    """Gradient boosting beats logistic regression here (CV 0.6927 vs 0.6787, multi
    0.810 vs 0.753): the count features interact non-linearly -- e.g. recurrence
    only means something relative to how many questions the clip has."""
    return HistGradientBoostingClassifier(
        max_iter=250, learning_rate=0.08, max_depth=4,
        l2_regularization=1.0, random_state=0)

from common import options
from text_prior import TextPrior

# Categories whose options name actions (emotion options are manner adverbs).
ACTION_CATS = ("single", "multi", "combination", "sequence")


def atoms(value: str) -> list[str]:
    return [a.strip() for a in str(value).split(",") if a.strip()]


def _norm(value: str) -> str:
    return " ".join(str(value).strip().lower().split())


def clip_evidence(group: pd.DataFrame):
    """Per-action evidence within one clip: which questions and categories name it."""
    questions = defaultdict(set)
    categories = defaultdict(set)
    in_combination = Counter()
    for row in group.itertuples(index=False):
        r = row._asdict()
        if r["category"] not in ACTION_CATS:
            continue
        for L in options(r):
            names = atoms(r[L])
            for a in names:
                questions[a].add(r["qa_id"])
                categories[a].add(r["category"])
                if r["category"] == "combination":
                    in_combination[a] += 1
    return questions, categories, in_combination


def true_action_set(group: pd.DataFrame) -> set[str]:
    """S, recovered from the clip's answers. Training supervision only."""
    S = set()
    for row in group.itertuples(index=False):
        r = row._asdict()
        if r["category"] not in ACTION_CATS:
            continue
        for L in r["answer"]:
            S |= set(atoms(r[L]))
    return S


class ActionSetModel:
    """P(a in S) per clip-action, plus grammar-aware decoding of each question."""

    def __init__(self, C: float = 1.0):
        self.prior = TextPrior()
        self.C = C
        self.model = None
        self.global_freq = Counter()
        self.n_rows = 1
        # Scene coherence: HAU scenes are room-themed, so a bathroom action is
        # implausible in a kitchen clip. PMI over co-occurrence within S separates
        # real actions from distractors at AUC 0.930.
        self.co = Counter()
        self.uni = Counter()
        self.n_clips = 1
        self.stage2 = None
        # object_interaction: 90% of test option-sets recur in training, and the
        # option set determines the object, so memorising sets beats the text
        # prior (0.902 vs 0.797 subject-disjoint).
        self.object_memory = defaultdict(Counter)
        # HARn `single` has 43 near-uniformly sampled distractors under a single
        # question template, so there is no text signal: the prior scores 0.310,
        # *below* the 1/3 floor. Abstain to a constant letter until video lands.
        self.harn_single_letter = "A"

    # ---------- features ----------
    def _features(self, group: pd.DataFrame):
        questions, categories, in_comb = clip_evidence(group)
        n_q = sum(1 for row in group.itertuples(index=False)
                  if row.category in ACTION_CATS)
        seq_opts = set()
        for row in group.itertuples(index=False):
            if row.category == "sequence":
                r = row._asdict()
                seq_opts |= {a for L in options(r) for a in atoms(r[L])}
        rows, names = [], []
        for a, qs in questions.items():
            cats = categories[a]
            rows.append([
                len(qs),                                   # questions naming a
                len(qs) / max(1, n_q),                     # ... as a fraction
                len(cats),                                 # distinct categories
                float("sequence" in cats),                 # sequence options are all real
                float("multi" in cats),
                float("combination" in cats),
                float("single" in cats),
                in_comb[a],
                n_q,
                len(questions),                            # candidate pool size
                self.global_freq[a] / self.n_rows,         # how often a is used anywhere
                float(a in seq_opts),
            ])
            names.append(a)
        return np.asarray(rows, dtype=float), names

    def _pmi(self, a: str, b: str) -> float:
        key = (a, b) if a < b else (b, a)
        p_ab = self.co[key] / self.n_clips
        p_a = self.uni[a] / self.n_clips
        p_b = self.uni[b] / self.n_clips
        return math.log((p_ab + 1e-4) / (p_a * p_b + 1e-6))

    def _coherence(self, names, weights):
        """Mean PMI of each candidate against the other candidates, weighted by
        their stage-1 posterior. Using posteriors rather than a hard set keeps
        this usable at inference time, where S is unknown."""
        out = []
        for i, a in enumerate(names):
            num = den = 0.0
            for j, b in enumerate(names):
                if i == j:
                    continue
                w = weights[j]
                num += w * self._pmi(a, b)
                den += w
            out.append(num / den if den > 1e-9 else 0.0)
        return np.asarray(out, dtype=float).reshape(-1, 1)

    # ---------- fit / predict ----------
    def fit(self, train: pd.DataFrame) -> "ActionSetModel":
        self.prior.fit(train)
        self.n_rows = max(1, len(train))
        letters = Counter()
        for row in train.itertuples(index=False):
            r = row._asdict()
            if r["category"] in ACTION_CATS:
                for L in options(r):
                    for a in atoms(r[L]):
                        self.global_freq[a] += 1
            if r["category"] == "object_interaction":
                key = tuple(sorted(_norm(r[L]) for L in options(r)))
                self.object_memory[key][_norm(r[r["answer"]])] += 1
            elif r["category"] == "single" and r["source"] == "HARn":
                letters[r["answer"]] += 1
        if letters:
            self.harn_single_letter = letters.most_common(1)[0][0]
        groups, X, y = [], [], []
        for _, group in train.groupby("clip_id"):
            feats, names = self._features(group)
            if not names:
                continue
            S = true_action_set(group)
            if group["source"].iloc[0] == "HAU":
                self.n_clips += 1
                for a in S:
                    self.uni[a] += 1
                for a in S:
                    for b in S:
                        if a < b:
                            self.co[(a, b)] += 1
            groups.append((feats, names))
            X.append(feats)
            y.extend(int(a in S) for a in names)
        X = np.vstack(X)
        y = np.asarray(y)
        self.model = _make_model().fit(X, y)

        # Stage 2 refits on base features + coherence computed from stage-1 output.
        X2 = []
        for feats, names in groups:
            p1 = self.model.predict_proba(feats)[:, 1]
            X2.append(np.hstack([feats, self._coherence(names, p1)]))
        self.stage2 = _make_model().fit(np.vstack(X2), y)
        return self

    def posteriors_batch(self, frame: pd.DataFrame) -> dict[str, dict[str, float]]:
        """Posteriors for every clip in one pair of model calls.

        Scoring clips one at a time means thousands of tiny GBM predicts, which is
        dominated by per-call overhead; stacking them is ~40x faster and identical.
        """
        blocks, names_per_clip, clip_ids = [], [], []
        for clip_id, group in frame.groupby("clip_id"):
            feats, names = self._features(group)
            if not names:
                continue
            blocks.append(feats)
            names_per_clip.append(names)
            clip_ids.append(clip_id)
        if not blocks:
            return {}
        stacked = np.vstack(blocks)
        p1 = self.model.predict_proba(stacked)[:, 1]
        rows, offset = [], 0
        for feats, names in zip(blocks, names_per_clip):
            n = len(names)
            rows.append(np.hstack([feats, self._coherence(names, p1[offset:offset + n])]))
            offset += n
        p2 = self.stage2.predict_proba(np.vstack(rows))[:, 1]
        out, offset = {}, 0
        for clip_id, names in zip(clip_ids, names_per_clip):
            n = len(names)
            out[clip_id] = dict(zip(names, p2[offset:offset + n]))
            offset += n
        return out

    def expected_sizes(self, frame: pd.DataFrame) -> dict[str, float]:
        """E[|S|] per clip: the sum of action posteriors. Feeds the take model,
        which needs duration normalised by how many actions a clip contains."""
        batch = self.posteriors_batch(frame)
        return {c: (sum(batch[c].values()) if batch.get(c) else 3.8)
                for c in frame["clip_id"].unique()}

    def posteriors(self, group: pd.DataFrame) -> dict[str, float]:
        feats, names = self._features(group)
        if not names:
            return {}
        p1 = self.model.predict_proba(feats)[:, 1]
        X2 = np.hstack([feats, self._coherence(names, p1)])
        p = self.stage2.predict_proba(X2)[:, 1]
        return dict(zip(names, p))

    def _propagate(self, group: pd.DataFrame, post: dict[str, float],
                   rounds: int = 2) -> dict[str, float]:
        """Enforce the per-question count constraints on the action posterior.

        Verified on training data:
          * every `sequence` option is in S (308/308 clips) -> hard evidence;
          * exactly one `single` option is in S (correct 93.9%, distractors 0/2427)
            -> the four options' posteriors must sum to one;
          * exactly one `combination` option's whole tuple is in S -> the winning
            tuple's members are reinforced by its normalised share.

        Constraints are applied as repeated projections, which sharpens actions that
        several questions agree on and suppresses one-off distractors.
        """
        post = dict(post)
        rows = [r._asdict() for r in group.itertuples(index=False)]
        for _ in range(rounds):
            for r in rows:
                live = options(r)
                if r["category"] == "sequence":
                    for L in live:
                        for a in atoms(r[L]):
                            post[a] = max(post.get(a, 0.5), 0.99)
                elif r["category"] == "single":
                    vals = {L: post.get(r[L], 0.05) for L in live if r[L] in post}
                    total = sum(vals.values())
                    if total > 1e-9 and len(vals) > 1:
                        for L, v in vals.items():
                            post[r[L]] = v / total
                elif r["category"] == "combination":
                    scores = {}
                    for L in live:
                        names = atoms(r[L])
                        scores[L] = float(np.prod([post.get(a, 0.05) for a in names])) \
                            if names else 0.0
                    total = sum(scores.values())
                    if total > 1e-9:
                        win = max(scores, key=scores.get)
                        for a in atoms(r[win]):
                            post[a] = max(post.get(a, 0.05), scores[win] / total)
        return {a: min(max(v, 1e-6), 1 - 1e-6) for a, v in post.items()}

    def _object(self, r) -> str:
        """Exact option-set lookup, falling back to the text prior when unseen."""
        live = options(r)
        key = tuple(sorted(_norm(r[L]) for L in live))
        memo = self.object_memory.get(key)
        if memo:
            want = memo.most_common(1)[0][0]
            for L in live:
                if _norm(r[L]) == want:
                    return L
        return self.prior.predict_row(r)

    # ---------- decoding ----------
    def _option_score(self, r, L, post) -> float:
        """Combination options are action tuples; require every member to be in S."""
        names = atoms(r[L])
        if not names:
            return 0.0
        return float(np.mean([math.log(max(post.get(a, 0.05), 1e-6)) for a in names]))

    def predict(self, frame: pd.DataFrame) -> list[str]:
        out = {}
        batch = self.posteriors_batch(frame)
        for clip_id, group in frame.groupby("clip_id"):
            post = batch.get(clip_id) or {}
            if post:
                post = self._propagate(group, post)
            for row in group.itertuples(index=False):
                r = row._asdict()
                live = options(r)
                cat = r["category"]
                # HARn clips carry only 1-2 questions, so there is no clip graph to
                # exploit; and emotion options share no vocabulary with actions.
                if cat == "object_interaction":
                    out[r["qa_id"]] = self._object(r)
                    continue
                if cat == "single" and r["source"] == "HARn":
                    out[r["qa_id"]] = (self.harn_single_letter
                                       if self.harn_single_letter in live else live[0])
                    continue
                if cat == "emotion" or r["source"] == "HARn" or not post:
                    out[r["qa_id"]] = self.prior.predict_row(r)
                    continue
                if cat == "sequence":
                    out[r["qa_id"]] = self.prior._sequence(r)
                elif cat == "multi":
                    chosen = [L for L in live if post.get(r[L], 0.0) > 0.5]
                    if not chosen:
                        chosen = [max(live, key=lambda L: post.get(r[L], 0.0))]
                    out[r["qa_id"]] = "".join(sorted(chosen))
                else:
                    out[r["qa_id"]] = max(
                        live, key=lambda L: self._option_score(r, L, post))
        return [out[q] for q in frame["qa_id"]]
