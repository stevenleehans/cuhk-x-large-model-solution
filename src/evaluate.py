"""Subject-disjoint CV harness. Reports per-category accuracy and an LB estimate
reweighted by the test-set category mix."""
from __future__ import annotations

import sys
from collections import defaultdict

import pandas as pd

import common
from text_prior import TextPrior


def cv(model_factory, df: pd.DataFrame, n_folds: int = 6):
    hits = defaultdict(int)
    total = defaultdict(int)
    oof = {}
    for train, val in common.subject_folds(df, n_folds):
        model = model_factory().fit(train)
        preds = model.predict(val)
        for (_, r), p in zip(val.iterrows(), preds):
            oof[r["qa_id"]] = p
            total[r["category"]] += 1
            hits[r["category"]] += p == r["answer"]
    per_cat = {c: hits[c] / total[c] for c in total}
    return per_cat, total, oof


def report(name, per_cat, total):
    print(f"\n=== {name} — subject-disjoint {len(total)} categories ===")
    for c in sorted(per_cat, key=lambda c: -common.TEST_COUNTS[c]):
        print(f"  {c:20s} {per_cat[c]:.4f}   (cv n={total[c]:4d}, test n={common.TEST_COUNTS[c]:3d})")
    overall = sum(per_cat[c] * total[c] for c in per_cat) / sum(total.values())
    print(f"  {'-- train mix':20s} {overall:.4f}")
    print(f"  {'-- LB ESTIMATE':20s} {common.lb_estimate(per_cat):.4f}")
    return common.lb_estimate(per_cat)


if __name__ == "__main__":
    train = common.load("training")
    per_cat, total, oof = cv(TextPrior, train)
    report("TextPrior", per_cat, total)
    pd.DataFrame({"qa_id": list(oof), "pred": list(oof.values())}).to_csv(
        common.ROOT / "data" / "oof_text_prior.csv", index=False)
    print(f"\nOOF written to data/oof_text_prior.csv")
