"""Download the gated CUHK-X Large Model Track media.

Plain single-stream HTTP from the HF CDN runs at ~0.25 MB/s here and the hf_hub
xet backend stalls outright, so this fetches each file as parallel byte ranges
(~1.8 MB/s with 8 workers). Chunks are written directly into a preallocated file
and a sidecar records which are done, so an interrupted run resumes.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BASE = ("https://huggingface.co/datasets/Kevin-Pal/CUHK-X_Large_Model_Track"
        "/resolve/main")
# Test set first: nothing can be scored without it.
FILES = [
    "Large-Model-Track/Testing/data/large_model_track_test.zip",
    "Large-Model-Track/Training/data/HAU.zip",
    "Large-Model-Track/Training/data/HARn.zip",
    "Large-Model-Track/Training/modality_list.csv",
]
CHUNK = 8 << 20
WORKERS = 8
DEST = Path(__file__).resolve().parent.parent / "data" / "media"


def _headers() -> dict:
    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("export HF_TOKEN=hf_... first")
    return {"Authorization": f"Bearer {token}", "User-Agent": "curl/8"}


def size_of(url: str) -> int:
    req = urllib.request.Request(url, headers=_headers(), method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers["Content-Length"])


def fetch(remote: str) -> Path:
    url = f"{BASE}/{remote}"
    out = DEST / Path(remote).name
    state_path = out.with_suffix(out.suffix + ".chunks")
    total = size_of(url)
    if out.is_file() and out.stat().st_size == total and not state_path.exists():
        print(f"    already complete: {out.name}")
        return out

    n_chunks = (total + CHUNK - 1) // CHUNK
    done = set(json.loads(state_path.read_text())) if state_path.is_file() else set()
    if not out.is_file() or out.stat().st_size != total:
        with open(out, "wb") as f:
            f.truncate(total)
        done = set()

    lock = threading.Lock()
    started = time.time()
    counter = {"n": len(done)}

    def one(i: int):
        if i in done:
            return
        start, end = i * CHUNK, min(total, (i + 1) * CHUNK) - 1
        for attempt in range(6):
            try:
                req = urllib.request.Request(
                    url, headers={**_headers(), "Range": f"bytes={start}-{end}"})
                data = urllib.request.urlopen(req, timeout=120).read()
                with lock:
                    with open(out, "r+b") as f:
                        f.seek(start)
                        f.write(data)
                    done.add(i)
                    counter["n"] += 1
                    state_path.write_text(json.dumps(sorted(done)))
                    frac = counter["n"] / n_chunks
                    rate = counter["n"] * CHUNK / 1e6 / max(1e-6, time.time() - started)
                    print(f"\r    {out.name} {frac:6.1%}  {rate:5.2f} MB/s",
                          end="", flush=True)
                return
            except Exception:
                time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"chunk {i} of {remote} failed")

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        list(pool.map(one, range(n_chunks)))
    print()
    state_path.unlink(missing_ok=True)
    return out


def main(only_test: bool = False):
    DEST.mkdir(parents=True, exist_ok=True)
    targets = FILES[:1] if only_test else FILES
    for remote in targets:
        print(f"--> {remote}", flush=True)
        path = fetch(remote)
        if path.suffix == ".zip":
            print(f"    extracting {path.name}", flush=True)
            with zipfile.ZipFile(path) as z:
                z.extractall(DEST)
            path.unlink()
    print("done:", sorted(p.name for p in DEST.iterdir()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only-test", action="store_true")
    main(ap.parse_args().only_test)
