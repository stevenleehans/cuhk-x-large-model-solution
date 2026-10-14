"""Ask Gemini the questions directly, with the clip's Depth video attached.

This is the track's namesake approach and it is complementary to the structural
model: it sees motion, which the text can never recover. Two things learned the
hard way and encoded here:

  * thinking must stay ON. The same sequence prompt scores 0.267 with thinking and
    0.037 (chance) with `thinkingBudget: 0`.
  * Depth clips are ~4 MB, so they go inline as base64 rather than through the
    Files API, which avoids the resumable-upload handshake entirely.
  * a clip's questions go in ONE call. They share footage, and the clip-graph
    result says answering them jointly is far stronger than independently.

Answers are written per clip as JSON so a rate-limited run resumes, and the caller
folds them into the structural posterior rather than trusting them outright.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import common
from depth_features import find_clip_video

# Optional video prompting. Models can be supplied explicitly through the environment.
MODELS = [m.strip() for m in os.environ.get("GEMINI_MODELS", "gemini-2.5-flash").split(",") if m.strip()]
API = "https://generativelanguage.googleapis.com/v1beta"
CACHE = common.ROOT / "data" / "gemini"

GRAMMAR = {
    "single": "exactly one letter",
    "emotion": "exactly one letter",
    "combination": "exactly one letter",
    "object_interaction": "exactly one letter",
    "multi": "one or more letters, sorted, e.g. AC",
    "sequence": "all four letters in temporal order, e.g. BADC",
}


def _key() -> str:
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise SystemExit("Set GEMINI_API_KEY in your environment.")
    return key


def build_prompt(rows: list[dict]) -> str:
    lines = [
        "This is a DEPTH video (grayscale depth map, 10 fps) of one person doing "
        "everyday activities at home. Nearer surfaces are brighter.",
        "Answer every question below about THIS video. The questions describe the "
        "same clip, so keep your answers consistent with each other.",
        "",
    ]
    for i, r in enumerate(rows, 1):
        opts = "  ".join(f"{L}) {r[L]}" for L in common.options(r))
        lines.append(f"{i}. [{r['category']}] {r['question']}")
        lines.append(f"   {opts}")
        lines.append(f"   answer format: {GRAMMAR[r['category']]}")
    lines += ["", "Reply ONLY with lines '<n>: <letters>'. No explanation."]
    return "\n".join(lines)


MAX_INLINE = 18 << 20


def ask(video: Path, prompt: str, model: str) -> dict[int, str]:
    import base64
    import urllib.request
    blob = base64.b64encode(video.read_bytes()).decode()
    body = json.dumps({
        "contents": [{"parts": [
            {"inline_data": {"mime_type": "video/mp4", "data": blob}},
            {"text": prompt}]}],
        "generationConfig": {"temperature": 0.0},
    }).encode()
    req = urllib.request.Request(f"{API}/models/{model}:generateContent",
                                 data=body, headers={"Content-Type": "application/json", "x-goog-api-key": _key()})
    with urllib.request.urlopen(req, timeout=600) as r:
        cand = json.load(r)["candidates"][0]
    parts = cand.get("content", {}).get("parts")
    text = parts[0]["text"] if parts else ""
    out = {}
    for line in text.strip().splitlines():
        if ":" in line:
            n, v = line.split(":", 1)
            n = n.strip().rstrip(".")
            v = "".join(c for c in v.strip().upper() if c in "ABCD")
            if n.isdigit() and v:
                out[int(n)] = v
    return out


def run(split: str, limit: int | None = None, sleep: float = 4.0):
    CACHE.mkdir(parents=True, exist_ok=True)
    live = list(MODELS)
    df = common.load(split)
    clips = sorted(df["clip_id"].unique())
    if limit:
        clips = clips[:limit]
    for n, clip_id in enumerate(clips, 1):
        dest = CACHE / (clip_id.replace("/", "__") + ".json")
        if dest.is_file():
            continue
        video = find_clip_video(clip_id)
        if video is None:
            continue
        rows = [r._asdict() for r in df[df["clip_id"] == clip_id].itertuples(index=False)]
        if video.stat().st_size > MAX_INLINE:
            print(f"  {clip_id}: {video.stat().st_size/1e6:.0f} MB too large, skipped")
            continue
        answers, used = None, None
        for model in list(live):
            try:
                answers, used = ask(video, build_prompt(rows), model), model
                break
            except Exception as exc:
                if getattr(exc, "code", None) == 429:
                    live.remove(model)               # day's quota gone for this one
                    print(f"    {model} exhausted ({len(live)} left)", flush=True)
                else:
                    print(f"    {model}: {type(exc).__name__} {str(exc)[:60]}", flush=True)
                    time.sleep(5)
        if answers is None:
            print("  all models exhausted for today; stopping", flush=True)
            return
        dest.write_text(json.dumps(
            {"_model": used, **{r["qa_id"]: answers.get(i + 1)
                                for i, r in enumerate(rows)}}))
        print(f"  [{n}/{len(clips)}] {clip_id} -> {len(answers)} via {used}", flush=True)
        time.sleep(sleep)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--sleep", type=float, default=4.0)
    a = ap.parse_args()
    run(a.split, a.limit, a.sleep)
