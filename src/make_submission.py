"""Fit on all training data and write a validated submission.

Usage: python src/make_submission.py [name] [video_weight]
Video scores (from the Kaggle kernel) are used for HARn `single` whenever present;
`video_weight` additionally folds them into the HAU action posterior.
"""
from __future__ import annotations

import json
import sys

import common
from fuse_video import load_scores, load_scores_from
from pipeline import Pipeline
from take_model import load_features

if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "pipeline"
    weight = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
    train, test = common.load("training"), common.load("test")
    tr_feats, te_feats = load_features("training"), load_features("test")
    # Fit on OOF scores, predict with the fold-averaged test scores.
    model = Pipeline(video_scores=load_scores_from("video_scores", "training"),
                     video_weight=weight,
                     harn_scores=load_scores_from("video_scores_ens", "training"))
    model.fit(train, tr_feats)
    model.video_scores = load_scores_from("video_scores", "test")
    model.harn_scores = load_scores_from("video_scores_ens", "test")
    preds = model.predict(test, te_feats)
    out = common.write_submission(test, preds, common.ROOT / "submissions" / f"{name}.csv")
    print(f"Wrote {out} — {len(preds)} rows, all valid (video_weight={weight})")
