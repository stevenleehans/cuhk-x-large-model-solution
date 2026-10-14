"""Motion features from Depth video.

Purpose: recover the HAU *take* digit, which encodes performance speed
(take 1 slow, take 2 normal, take 3 fast). Conditioning the emotion prior on the
true take lifts emotion from 0.315 to 0.723, and the option text alone does not
leak it -- so this is the single highest-value thing the video buys us.

The features are deliberately cheap and model-free: per-frame motion energy from
depth frame differencing, summarised into speed/rhythm statistics. A learned video
model is overkill for "how fast is this person moving".
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import common

MEDIA = common.ROOT / "data" / "media"
FEATURE_NAMES = [
    "n_frames", "duration_s", "motion_mean", "motion_std", "motion_p10",
    "motion_p50", "motion_p90", "motion_max", "motion_iqr", "zero_frac",
    "peak_rate", "spec_centroid", "autocorr_1", "energy_slope", "active_frac",
]


def find_clip_video(clip_id: str, modality: str = "Depth") -> Path | None:
    """Locate a clip's video under data/media.

    Matching is on the FULL relative path, never the basename: HARn trial dirs
    reuse HAU trial names ("1-1-2"), so a basename glob silently returns another
    source's footage and every clip looks "found".
    """
    stem = clip_id.replace("\\", "/")
    root = MEDIA / stem / modality
    if root.is_dir():
        for f in sorted(root.iterdir()):
            if f.suffix.lower() == ".mp4":
                return f
    return None


FALLBACK_MODALITIES = ("Depth", "IR", "Depth_Color")


def find_any_video(clip_id: str) -> tuple[Path | None, str | None]:
    """Depth if present, else another modality — two test clips have empty Depth."""
    for modality in FALLBACK_MODALITIES:
        found = find_clip_video(clip_id, modality)
        if found is not None and found.stat().st_size > 1024:
            return found, modality
    return None, None


def motion_signal(path: Path, max_side: int = 160) -> tuple[np.ndarray, float]:
    """Per-frame mean absolute depth difference, plus fps."""
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    prev, sig = None, []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = frame.shape
        scale = max_side / max(h, w)
        if scale < 1:
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
        cur = frame.astype(np.float32) / 255.0
        if prev is not None:
            sig.append(float(np.abs(cur - prev).mean()))
        prev = cur
    cap.release()
    return np.asarray(sig, dtype=np.float32), float(fps)


def summarise(sig: np.ndarray, fps: float) -> dict[str, float]:
    if sig.size < 4:
        return {k: 0.0 for k in FEATURE_NAMES}
    n = sig.size
    p10, p50, p90 = np.percentile(sig, [10, 50, 90])
    q1, q3 = np.percentile(sig, [25, 75])
    centred = sig - sig.mean()
    denom = float((centred ** 2).sum()) or 1.0
    spec = np.abs(np.fft.rfft(centred))
    freqs = np.fft.rfftfreq(n, d=1.0 / fps)
    thresh = sig.mean()
    crossings = int(((sig[:-1] < thresh) & (sig[1:] >= thresh)).sum())
    return {
        "n_frames": float(n),
        "duration_s": n / fps,
        "motion_mean": float(sig.mean()),
        "motion_std": float(sig.std()),
        "motion_p10": float(p10),
        "motion_p50": float(p50),
        "motion_p90": float(p90),
        "motion_max": float(sig.max()),
        "motion_iqr": float(q3 - q1),
        "zero_frac": float((sig < 1e-4).mean()),
        # rhythm: how often motion energy crosses its own mean, per second
        "peak_rate": crossings / (n / fps),
        "spec_centroid": float((spec * freqs).sum() / (spec.sum() + 1e-9)),
        "autocorr_1": float((centred[:-1] * centred[1:]).sum() / denom),
        "energy_slope": float(np.polyfit(np.arange(n), sig, 1)[0] * n),
        "active_frac": float((sig > thresh).mean()),
    }


def build(split: str, modality: str = "Depth") -> pd.DataFrame:
    df = common.load(split)
    rows = []
    clips = sorted(df["clip_id"].unique())
    for i, clip_id in enumerate(clips, 1):
        if modality == "Depth":
            video, used = find_any_video(clip_id)
        else:
            video, used = find_clip_video(clip_id, modality), modality
        rec = {"clip_id": clip_id, "found": video is not None, "modality": used}
        rec.update(summarise(*motion_signal(video)) if video else
                   {k: 0.0 for k in FEATURE_NAMES})
        rows.append(rec)
        if i % 50 == 0 or i == len(clips):
            print(f"  {split}: {i}/{len(clips)}", flush=True)
    out = pd.DataFrame(rows)
    suffix = "" if modality == "Depth" else f"_{modality.lower()}"
    dest = common.ROOT / "data" / f"depth_features_{split}{suffix}.csv"
    out.to_csv(dest, index=False)
    print(f"wrote {dest}  ({out['found'].sum()}/{len(out)} videos found)")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("splits", nargs="*", default=["training", "test"])
    ap.add_argument("--modality", default="Depth")
    a = ap.parse_args()
    for split in a.splits:
        build(split, a.modality)
