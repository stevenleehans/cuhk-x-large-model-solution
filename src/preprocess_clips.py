"""Decode clips to fixed-size uint8 frame tensors for GPU training elsewhere.

Training runs on Kaggle GPU, not locally, so the videos have to get there. Shipping
the raw archives means an 8 GB upload; decoding first to 32x112x112 grayscale uint8
is ~400 KB per clip, so the whole corpus (1541 clips) fits in ~600 MB.

Decoding is CPU-only and streams one clip at a time into a memmap, so peak RAM
stays at a single clip.
"""
from __future__ import annotations

import argparse
import json

import cv2
import numpy as np

import common
from depth_features import find_any_video

FRAMES = 32
SIZE = 112


def sample_clip(path, n_frames: int = FRAMES, size: int = SIZE) -> np.ndarray:
    """Uniformly sample n_frames, grayscale, resized. Short clips repeat the last."""
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA))
    cap.release()
    if not frames:
        return np.zeros((n_frames, size, size), dtype=np.uint8)
    idx = np.linspace(0, len(frames) - 1, n_frames).round().astype(int)
    return np.stack([frames[i] for i in idx]).astype(np.uint8)


def build(split: str, modality: str = "Depth"):
    df = common.load(split)
    clips = sorted(df["clip_id"].unique())
    out_dir = common.ROOT / "data" / "clips"
    out_dir.mkdir(parents=True, exist_ok=True)
    array = np.lib.format.open_memmap(
        out_dir / f"{split}_{modality}.npy", mode="w+",
        dtype=np.uint8, shape=(len(clips), FRAMES, SIZE, SIZE))
    missing = []
    for i, clip_id in enumerate(clips):
        video = find_clip_or_any(clip_id, modality)
        if video is None:
            missing.append(clip_id)
        else:
            array[i] = sample_clip(video)
        if (i + 1) % 100 == 0 or i + 1 == len(clips):
            print(f"  {split}/{modality}: {i+1}/{len(clips)}", flush=True)
    array.flush()
    (out_dir / f"{split}_{modality}_index.json").write_text(json.dumps(clips))
    print(f"wrote {split}_{modality}.npy  shape={array.shape}  missing={len(missing)}")
    return missing


def find_clip_or_any(clip_id: str, modality: str):
    from depth_features import find_clip_video
    found = find_clip_video(clip_id, modality)
    if found is not None and found.stat().st_size > 1024:
        return found
    return find_any_video(clip_id)[0]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="*", default=["training", "test"])
    ap.add_argument("--modality", default="Depth")
    a = ap.parse_args()
    for split in a.splits:
        build(split, a.modality)
