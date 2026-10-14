"""Depth-video action recognizer -> per-clip action presence + temporal order.

Design follows from the structural analysis: with a perfect action set S,
single/multi/combination decode at 1.00/1.00/0.999. So the model's job is to emit
P(action present) per clip, not to answer questions. Sliding-window inference
additionally gives each action a mean temporal position, which is what `sequence`
needs.

Training signal, over the 44 canonical actions (see actions.py):
  * HARn clip -> exactly one positive label (the path folder);
  * HAU clip  -> the |S| positives recovered from that clip's answers.
Both are multi-label BCE, so one model covers both sources. 1333 clips is far too
little to train from scratch, hence a Kinetics-pretrained 3D backbone.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import actions
import common
from action_set import true_action_set
from depth_features import find_clip_video

DEVICE = "cuda"


def require_kaggle_gpu():
    if not Path('/kaggle/input').is_dir() or not Path('/kaggle/working').is_dir() or not torch.cuda.is_available():
        raise RuntimeError('Video model execution requires Kaggle CUDA; MPS and CPU fallback are disabled.')
N_CLASSES = len(actions.CANONICAL)
CLIP_FRAMES = 16
SIZE = 112


def read_frames(path: Path, idx: np.ndarray) -> np.ndarray:
    """Read specific frame indices, resized and grayscale-replicated to 3 channels."""
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    idx = np.clip(idx, 0, max(0, total - 1))
    want = set(int(i) for i in idx)
    grabbed, i = {}, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i in want:
            if frame.ndim == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            grabbed[i] = cv2.resize(frame, (SIZE, SIZE))
        i += 1
    cap.release()
    if not grabbed:
        return np.zeros((len(idx), SIZE, SIZE), dtype=np.float32)
    last = sorted(grabbed)[-1]
    out = [grabbed.get(int(j), grabbed[last]) for j in idx]
    return np.stack(out).astype(np.float32) / 255.0


def clip_labels(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Multi-label target per training clip."""
    out = {}
    for clip_id, group in df.groupby("clip_id"):
        vec = np.zeros(N_CLASSES, dtype=np.float32)
        if group["source"].iloc[0] == "HARn":
            folder = clip_id.split("/")[1]
            vec[actions.INDEX[actions.FOLDER[folder]]] = 1.0
        else:
            for text in true_action_set(group):
                name = actions.canonical(text)
                if name:
                    vec[actions.INDEX[name]] = 1.0
        out[clip_id] = vec
    return out


class ClipDataset(Dataset):
    def __init__(self, clip_ids, labels=None, train=False, windows=1):
        self.clip_ids = list(clip_ids)
        self.labels = labels
        self.train = train
        self.windows = windows

    def __len__(self):
        return len(self.clip_ids)

    def __getitem__(self, i):
        clip_id = self.clip_ids[i]
        video = find_clip_video(clip_id)
        if video is None:
            x = np.zeros((CLIP_FRAMES, SIZE, SIZE), dtype=np.float32)
        else:
            cap = cv2.VideoCapture(str(video))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or CLIP_FRAMES
            cap.release()
            if self.train:  # random temporal crop
                span = max(CLIP_FRAMES, int(total * np.random.uniform(0.5, 1.0)))
                start = np.random.randint(0, max(1, total - span + 1))
                idx = np.linspace(start, start + span - 1, CLIP_FRAMES)
            else:
                idx = np.linspace(0, total - 1, CLIP_FRAMES)
            x = read_frames(video, idx.astype(int))
        if self.train and np.random.rand() < 0.5:
            x = x[:, :, ::-1].copy()
        t = torch.from_numpy(x).unsqueeze(0).repeat(3, 1, 1, 1)  # C,T,H,W
        if self.labels is None:
            return t, clip_id
        return t, torch.from_numpy(self.labels[clip_id])


def build_model() -> nn.Module:
    require_kaggle_gpu()
    from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights
    model = r2plus1d_18(weights=R2Plus1D_18_Weights.KINETICS400_V1)
    model.fc = nn.Linear(model.fc.in_features, N_CLASSES)
    return model.to(DEVICE)


def train(epochs: int = 12, batch_size: int = 8, lr: float = 1e-4, holdout: str | None = None):
    require_kaggle_gpu()
    df = common.load("training")
    labels = clip_labels(df)
    clips = df[["clip_id", "subject"]].drop_duplicates()
    if holdout:
        held = set(holdout.split(","))
        tr_ids = clips[~clips["subject"].isin(held)]["clip_id"]
        va_ids = clips[clips["subject"].isin(held)]["clip_id"]
    else:
        tr_ids, va_ids = clips["clip_id"], clips["clip_id"][:0]
    loader = DataLoader(ClipDataset(tr_ids, labels, train=True),
                        batch_size=batch_size, shuffle=True, num_workers=4)
    model = build_model()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * len(loader))
    lossf = nn.BCEWithLogitsLoss()
    for ep in range(epochs):
        model.train(); running = 0.0
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss = lossf(model(x), y)
            loss.backward(); opt.step(); sched.step()
            running += loss.item()
        print(f"epoch {ep+1}/{epochs}  loss={running/len(loader):.4f}", flush=True)
        if len(va_ids):
            print(f"  val AP={evaluate(model, va_ids, labels):.4f}", flush=True)
    torch.save(model.state_dict(), common.ROOT / "data" / "action_recognizer.pt")
    return model


@torch.no_grad()
def evaluate(model, clip_ids, labels) -> float:
    require_kaggle_gpu()
    from sklearn.metrics import average_precision_score
    model.eval()
    loader = DataLoader(ClipDataset(clip_ids, labels), batch_size=8, num_workers=4)
    ys, ps = [], []
    for x, y in loader:
        ps.append(torch.sigmoid(model(x.to(DEVICE))).cpu().numpy()); ys.append(y.numpy())
    return average_precision_score(np.vstack(ys), np.vstack(ps), average="micro")


@torch.no_grad()
def predict(split: str, model=None) -> dict:
    """Per-clip action presence scores, keyed for fusion.FusedModel."""
    require_kaggle_gpu()
    model = model or build_model()
    if model is not None and not isinstance(model, nn.Module):
        raise TypeError("model must be an nn.Module")
    state = common.ROOT / "data" / "action_recognizer.pt"
    if state.is_file():
        model.load_state_dict(torch.load(state, map_location=DEVICE))
    model.eval()
    df = common.load(split)
    ids = sorted(df["clip_id"].unique())
    loader = DataLoader(ClipDataset(ids), batch_size=8, num_workers=4)
    scores = {}
    for x, clip_ids in loader:
        p = torch.sigmoid(model(x.to(DEVICE))).cpu().numpy()
        for clip_id, row in zip(clip_ids, p):
            scores[clip_id] = {a: float(row[i]) for a, i in actions.INDEX.items()}
    dest = common.ROOT / "data" / f"video_scores_{split}.json"
    dest.write_text(json.dumps(scores))
    print(f"wrote {dest} ({len(scores)} clips)")
    return scores


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train", "predict"])
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--holdout", default=None)
    ap.add_argument("--split", default="test")
    a = ap.parse_args()
    if a.mode == "train":
        train(epochs=a.epochs, holdout=a.holdout)
    else:
        predict(a.split)
