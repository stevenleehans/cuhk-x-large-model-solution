"""Shared loading, answer-grammar and validation helpers for CUHK-X Large Model Track.

Task shape (derived from the released CSVs):
  682 test questions over 208 test clips; 4087 train questions over 1333 train clips.
  Two sources: HARn (single-action clips, action label in the path) and
  HAU (multi-action scripted scenes, 40-action vocabulary).
  Test media exposes only the Depth modality.
"""
from __future__ import annotations

import itertools
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
LABELS = "ABCD"

# Answer grammar, verified against training_qa.csv.
SINGLE_LETTER = {"single", "combination", "emotion", "object_interaction"}
CATEGORIES = SINGLE_LETTER | {"multi", "sequence"}

# Test-set composition, used to weight CV estimates into an LB estimate.
TEST_COUNTS = {
    "single": 195, "multi": 144, "emotion": 144,
    "combination": 139, "sequence": 39, "object_interaction": 21,
}


def load(split: str) -> pd.DataFrame:
    df = pd.read_csv(RAW / f"{split}_qa.csv", dtype=str,
                     keep_default_na=False, encoding="utf-8-sig")
    df["clip_id"] = df["path"].map(clip_key)
    df["subject"] = df["path"].map(subject_of)
    return df


def clip_key(path: str) -> str:
    """Collapse a row path to its clip directory.

    Train paths are already clip dirs ("HAU/user1/1-1-1"); test paths point at a
    modality file ("large_model_track_test/LM_test_0066/Depth/Depth.mp4").
    """
    parts = [p for p in str(path).replace("\\", "/").split("/") if p]
    if parts[-1].lower().endswith(".mp4"):
        parts = parts[:-2]
    return "/".join(parts)


def subject_of(path: str) -> str:
    """Recording subject, for subject-disjoint CV. Test clips are anonymised."""
    parts = str(path).replace("\\", "/").split("/")
    if parts[0] == "HAU":
        return parts[1]
    if parts[0] == "HARn":
        return parts[2]
    return "test"


def options(row) -> list[str]:
    """Live option letters. HARn `single` questions have only A/B/C (D is blank)."""
    return [L for L in LABELS if str(row[L]) != ""]


def valid(pred: str, row) -> bool:
    """Answer grammar enforced by the scorer."""
    pred = str(pred).strip().upper()
    live = set(options(row))
    if not pred or any(c not in live for c in pred):
        return False
    if len(set(pred)) != len(pred):
        return False
    cat = row["category"]
    if cat in SINGLE_LETTER:
        return len(pred) == 1
    if cat == "multi":
        return 1 <= len(pred) <= len(live) and pred == "".join(sorted(pred))
    if cat == "sequence":
        return set(pred) == live and len(pred) == len(live)
    return False


def subject_folds(df: pd.DataFrame, n_folds: int = 6):
    """Subject-disjoint folds. Never random-split rows: clips share subjects and
    questions share clips, so row-wise CV is optimistic by a wide margin."""
    subjects = sorted(df["subject"].unique())
    for i in range(n_folds):
        held = set(subjects[i::n_folds])
        yield df[~df["subject"].isin(held)], df[df["subject"].isin(held)]


def write_submission(test: pd.DataFrame, preds, path: Path) -> Path:
    """Write and hard-validate a submission in official test order."""
    sub = pd.DataFrame({"qa_id": test["qa_id"].astype(str), "prediction": list(preds)})
    bad = [(q, p) for (q, p), (_, r) in zip(sub.itertuples(index=False, name=None),
                                            test.iterrows()) if not valid(p, r)]
    if bad:
        raise ValueError(f"{len(bad)} invalid predictions, e.g. {bad[:5]}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "qa_id,prediction\n" + "".join(f"{q},{p}\n" for q, p in
                                             sub.itertuples(index=False, name=None))
    path.write_bytes(payload.encode("utf-8"))
    return path


def score(preds, truth) -> float:
    """Exact-match accuracy; matches the observed 1/342 public-LB granularity."""
    return sum(p == t for p, t in zip(preds, truth)) / len(truth)


def lb_estimate(per_category: dict[str, float]) -> float:
    """Reweight per-category CV accuracy by the test-set category mix."""
    total = sum(TEST_COUNTS.values())
    return sum(per_category.get(c, 0.25) * n for c, n in TEST_COUNTS.items()) / total
