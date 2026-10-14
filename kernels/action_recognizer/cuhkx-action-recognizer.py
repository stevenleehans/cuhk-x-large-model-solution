"""CUHK-X LMT — Depth action recogniser (Kaggle GPU).

Emits P(action present) per clip over the 44 canonical actions. That is the whole
job: decoding from the true action set S gives single 1.000, multi 1.000 and
combination 0.999, so on HAU the task reduces to recovering S.

Outputs, consumed by the local fusion step:
  oof_scores.npy  (n_train, 44)  subject-disjoint out-of-fold, for honest fitting
  test_scores.npy (n_test, 44)   mean over folds
In-sample base-model outputs would bias the fusion, hence OOF rather than refit.
"""
import json
import os
import subprocess
import sys

import numpy as np
import torch

# Kaggle hands out either a T4 (sm_75) or a P100 (sm_60), and the preinstalled
# torch is built for sm_70+ only -- on a P100 every forward pass dies. The
# accelerator field in kernel-metadata is ignored, so detect and self-heal.
if torch.cuda.is_available() and not os.environ.get("TORCH_REINSTALLED"):
    major, minor = torch.cuda.get_device_capability(0)
    supported = torch.cuda.get_arch_list()
    if f"sm_{major}{minor}" not in supported:
        print(f"GPU sm_{major}{minor} unsupported by torch {torch.__version__} "
              f"({supported}); installing a Pascal-capable build", flush=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                        "torch==2.5.1", "torchvision==0.20.1",
                        "--index-url", "https://download.pytorch.org/whl/cu121"],
                       check=True)
        os.environ["TORCH_REINSTALLED"] = "1"
        os.execv(sys.executable, [sys.executable] + sys.argv)
import torch.nn as nn
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader, Dataset

def _input_dir() -> str:
    """Kaggle sometimes mounts a dataset under a slug that differs from the id, so
    locate it by content rather than assuming the path."""
    import glob
    hits = glob.glob("/kaggle/input/**/training_Depth.npy", recursive=True)
    if not hits:
        listing = [os.path.join(r, f)
                   for r, _, fs in os.walk("/kaggle/input") for f in fs][:40]
        raise SystemExit(f"dataset not mounted; saw {listing}")
    return os.path.dirname(hits[0])


IN = _input_dir()
OUT = "/kaggle/working"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FRAMES_STORED, FRAMES_USED, SIZE = 32, 16, 112
FOLDS, EPOCHS, BATCH, LR = 6, 22, 16, 3e-4
TTA = 3  # temporal offsets averaged at inference

train_x = np.load(f"{IN}/training_Depth.npy", mmap_mode="r")
test_x = np.load(f"{IN}/test_Depth.npy", mmap_mode="r")
train_y = np.load(f"{IN}/training_labels.npy").astype(np.float32)
meta = json.load(open(f"{IN}/meta.json"))
subjects = np.array(meta["subjects"])
sources = np.array(meta["sources"])
ACTIONS = meta["actions"]
print(f"train {train_x.shape} test {test_x.shape} labels {train_y.shape} "
      f"actions {len(ACTIONS)} device {DEVICE}", flush=True)


class Clips(Dataset):
    def __init__(self, x, y=None, train=False, offset=None):
        self.x, self.y, self.train, self.offset = x, y, train, offset

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        v = np.asarray(self.x[i], dtype=np.float32) / 255.0
        if self.train:
            # temporal jitter: pick FRAMES_USED of the FRAMES_STORED with a random phase
            start = np.random.randint(0, FRAMES_STORED - FRAMES_USED + 1)
            idx = np.arange(start, start + FRAMES_USED)
        elif self.offset is not None:
            start = self.offset
            idx = np.arange(start, start + FRAMES_USED)
        else:
            idx = np.linspace(0, FRAMES_STORED - 1, FRAMES_USED).round().astype(int)
        v = v[idx]
        if self.train:
            if np.random.rand() < 0.5:
                v = v[:, :, ::-1]
            v = np.clip(v * np.random.uniform(0.85, 1.15), 0, 1)
        t = torch.from_numpy(np.ascontiguousarray(v)).unsqueeze(0).repeat(3, 1, 1, 1)
        if self.y is None:
            return t
        return t, torch.from_numpy(self.y[i])


class ClipsWithFlag(Clips):
    """Training set that also yields a HARn indicator, for the extra softmax term."""

    def __init__(self, x, y, flags):
        super().__init__(x, y, train=True)
        self.flags = flags

    def __getitem__(self, i):
        t, y = super().__getitem__(i)
        return t, y, float(self.flags[i])


def build():
    from torchvision.models.video import R2Plus1D_18_Weights, r2plus1d_18
    m = r2plus1d_18(weights=R2Plus1D_18_Weights.KINETICS400_V1)
    m.fc = nn.Linear(m.fc.in_features, len(ACTIONS))
    return m.to(DEVICE)


oof = np.zeros((len(train_x), len(ACTIONS)), dtype=np.float32)
test_pred = np.zeros((len(test_x), len(ACTIONS)), dtype=np.float32)
subject_list = sorted(set(subjects))

for fold in range(FOLDS):
    held = set(subject_list[fold::FOLDS])
    va = np.isin(subjects, list(held))
    tr = ~va
    model = build()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    is_harn = torch.from_numpy((sources[tr] == "HARn").astype(np.float32))
    loader = DataLoader(ClipsWithFlag(train_x[tr], train_y[tr], is_harn.numpy()),
                        batch_size=BATCH, shuffle=True, num_workers=2, drop_last=True)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR,
                                               total_steps=EPOCHS * len(loader))
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE == "cuda")
    bce = nn.BCEWithLogitsLoss()
    ce = nn.CrossEntropyLoss()
    for ep in range(EPOCHS):
        model.train()
        total = 0.0
        for xb, yb, hb in loader:
            xb, yb = xb.to(DEVICE, non_blocking=True), yb.to(DEVICE, non_blocking=True)
            hb = hb.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=DEVICE == "cuda"):
                logits = model(xb)
                loss = bce(logits, yb)
                # HARn clips carry exactly one action, so a softmax term suits them
                # far better than 44 independent sigmoids with a single positive.
                if hb.sum() > 0:
                    m = hb > 0.5
                    loss = loss + 0.5 * ce(logits[m], yb[m].argmax(1))
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.item()
        print(f"  fold {fold} ep {ep+1}/{EPOCHS} loss {total/len(loader):.4f}", flush=True)

    model.eval()

    def infer(x):
        """Average over TTA temporal offsets and a horizontal flip."""
        acc = np.zeros((len(x), len(ACTIONS)), dtype=np.float32)
        offsets = np.linspace(0, FRAMES_STORED - FRAMES_USED, TTA).round().astype(int)
        for off in offsets:
            preds = []
            for xb in DataLoader(Clips(x, offset=int(off)), batch_size=BATCH,
                                 num_workers=2):
                xb = xb.to(DEVICE)
                with torch.amp.autocast("cuda", enabled=DEVICE == "cuda"):
                    p = torch.sigmoid(model(xb)).float()
                    p = p + torch.sigmoid(model(torch.flip(xb, dims=[4]))).float()
                preds.append((p / 2).cpu().numpy())
            acc += np.vstack(preds) / len(offsets)
        return acc

    with torch.no_grad():
        oof[va] = infer(train_x[va])
        test_pred += infer(test_x) / FOLDS
    ap = average_precision_score(train_y[va], oof[va], average="micro")
    print(f"fold {fold}: val micro-AP {ap:.4f}", flush=True)

print(f"OOF micro-AP {average_precision_score(train_y, oof, average='micro'):.4f}")
np.save(f"{OUT}/oof_scores.npy", oof)
np.save(f"{OUT}/test_scores.npy", test_pred)
json.dump(ACTIONS, open(f"{OUT}/actions.json", "w"))
print("done")
