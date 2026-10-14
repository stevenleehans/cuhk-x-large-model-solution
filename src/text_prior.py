"""Media-free structural baseline.

Establishes the floor and, more importantly, the *prior* that the video experts
later override. Three signals, all fitted subject-disjoint:

  1. Option-text priors  P(option is in the answer | source, category, option text).
     Distractors are drawn from a wider vocabulary than correct answers, so this
     alone is far above chance (emotion: 0.32 vs 0.25; sequence: 0.42 vs 0.042).
  2. Pairwise temporal precedence counts plus a mean-position prior, blended for
     exact permutation decoding of `sequence` (0.406 -> 0.442 subject-disjoint).
     Room conditioning was tried and dropped: it helps given the true room (0.455)
     but the room is only 44.7% recoverable from the inferred action set, which the
     test paths do not expose.
"""
from __future__ import annotations

import itertools
import math
from collections import Counter, defaultdict

import pandas as pd

from common import LABELS, options

SMOOTHING = 16.0


def _empty_counts():
    """Module-level factory so trained priors can be serialized portably."""
    return [0, 0]


class TextPrior:
    def __init__(self, smoothing: float = SMOOTHING):
        self.smoothing = smoothing
        self.hit = Counter()
        self.seen = Counter()
        self.base = defaultdict(_empty_counts)
        self.sizes = defaultdict(Counter)
        self.before = Counter()
        self.position = defaultdict(list)
        # Emotion distractors are sampled *near* the correct adverb: the answer sits
        # in a shared semantic bucket far more than chance (singleton-bucket rate
        # 0.335 vs a 0.666 baseline). Option co-occurrence measures that affinity
        # without hand-built buckets, and lifts emotion 0.315 -> 0.334.
        self.co_option = Counter()
        self.uni_option = Counter()

    def fit(self, train: pd.DataFrame) -> "TextPrior":
        for row in train.itertuples(index=False):
            r = row._asdict()
            live = options(r)
            slice_key = (r["source"], r["category"])
            for L in live:
                self.seen[(slice_key, r[L])] += 1
                if L in r["answer"]:
                    self.hit[(slice_key, r[L])] += 1
            b = self.base[slice_key]
            b[0] += len(r["answer"])
            b[1] += len(live)
            if r["category"] == "emotion":
                texts = [r[L] for L in live]
                for a in texts:
                    self.uni_option[a] += 1
                    for c in texts:
                        if a != c:
                            self.co_option[(a, c)] += 1
            if r["category"] == "multi":
                self.sizes[slice_key][len(r["answer"])] += 1
            elif r["category"] == "sequence":
                ordered = [r[L] for L in r["answer"]]
                for i, a in enumerate(ordered):
                    self.position[a].append(i / max(1, len(ordered) - 1))
                    for c in ordered[i + 1:]:
                        self.before[(a, c)] += 1
        self.mean_position = {a: sum(v) / len(v) for a, v in self.position.items()}
        return self

    def option_probs(self, r) -> dict[str, float]:
        """P(option letter is part of the answer), smoothed toward the slice base rate."""
        slice_key = (r["source"], r["category"])
        b = self.base[slice_key]
        p0 = b[0] / b[1] if b[1] else 0.25
        out = {}
        for L in options(r):
            k = (slice_key, r[L])
            p = (self.hit[k] + self.smoothing * p0) / (self.seen[k] + self.smoothing)
            out[L] = min(max(p, 1e-6), 1 - 1e-6)
        return out

    POSITION_BLEND = 0.4

    def _sequence(self, r) -> str:
        live = options(r)
        w = self.POSITION_BLEND
        best = None
        for perm in itertools.permutations(live):
            texts = [r[L] for L in perm]
            s = 0.0
            for i, a in enumerate(texts):
                for c in texts[i + 1:]:
                    x, y = self.before[(a, c)], self.before[(c, a)]
                    s += (1 - w) * math.log((x + 1.5) / (x + y + 3.0))
                target = i / max(1, len(texts) - 1)
                s -= w * abs(self.mean_position.get(a, 0.5) - target)
            cand = (s, "".join(perm))
            if best is None or cand > best:
                best = cand
        return best[1]

    def _multi(self, r, probs) -> str:
        """Argmax over subsets of P(exact set) under an independent-option model.

        With no size prior this reduces to {L : p_L > 0.5}. A cardinality prior was
        tried and *hurt* (0.235 vs 0.258 subject-disjoint): it is a global prior
        fighting weak per-option evidence, so it overrides correct singletons.
        """
        live = options(r)
        chosen = [L for L in live if probs[L] > 0.5]
        if not chosen:                       # never emit an empty answer
            chosen = [max(live, key=lambda L: probs[L])]
        return "".join(sorted(chosen))

    AFFINITY = 2.0

    def _emotion(self, r, probs) -> str:
        """Reweight the frequency prior by how strongly each option co-occurs with
        the other options offered here -- the correct adverb tends to be the one
        the distractor sampler drew its neighbours around."""
        live = options(r)
        scored = {}
        for L in live:
            others = [r[M] for M in live if M != L]
            affinity = sum(self.co_option[(r[L], o)] for o in others) / (
                max(1, self.uni_option[r[L]]) * max(1, len(others)))
            scored[L] = probs[L] * (1 + self.AFFINITY * affinity)
        return max(scored, key=scored.get)

    def predict_row(self, r) -> str:
        cat = r["category"]
        if cat == "sequence":
            return self._sequence(r)
        probs = self.option_probs(r)
        if cat == "multi":
            return self._multi(r, probs)
        if cat == "emotion":
            return self._emotion(r, probs)
        return max(probs, key=probs.get)

    def predict(self, frame: pd.DataFrame) -> list[str]:
        return [self.predict_row(row._asdict()) for row in frame.itertuples(index=False)]
