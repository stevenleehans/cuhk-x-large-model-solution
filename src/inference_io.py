"""Fresh-question/media input, deterministic CPU decode, and strict score alignment."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path, PurePosixPath
import cv2
import numpy as np
import pandas as pd
import common


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def safe_relative(value: str) -> str:
    value = str(value).replace('\\', '/')
    path = PurePosixPath(value)
    if path.is_absolute() or '..' in path.parts or not path.parts or ':' in path.parts[0]:
        raise ValueError(f'Expected a relative media path inside the input directory: {value}')
    return str(path)


def read_questions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding='utf-8-sig')
    needed = {'qa_id', 'source', 'category', 'path', 'A', 'B', 'C', 'D'}
    if not needed.issubset(df.columns):
        raise ValueError(f'Missing question columns: {sorted(needed-set(df.columns))}')
    if df.empty or not df.qa_id.is_unique or (df.qa_id == '').any():
        raise ValueError('Question IDs must be unique and nonempty.')
    if not set(df.source).issubset({'HAU', 'HARn'}) or not set(df.category).issubset(common.CATEGORIES):
        raise ValueError('Unrecognized question source/category.')
    df['path'] = df.path.map(safe_relative)
    df['clip_id'] = df.path.map(common.clip_key)
    if (df.clip_id == '').any() or any(not common.options(r) for r in df.to_dict('records')):
        raise ValueError('Question needs a clip path and live options.')
    if (df.groupby('clip_id').source.nunique() != 1).any():
        raise ValueError('A clip cannot belong to multiple sources.')
    # Labels or previous predictions never enter prediction, even in replay audits.
    return df.drop(columns=['answer', 'prediction'], errors='ignore')


def resolve_video(root: Path, clip: str) -> tuple[Path, str]:
    root = root.resolve()
    for modality in ('Depth', 'IR', 'Depth_Color'):
        directory = (root / safe_relative(clip) / modality).resolve()
        if not directory.is_relative_to(root):
            raise ValueError('Media path escapes the input directory.')
        files = sorted(p for p in directory.glob('*') if p.suffix.lower() == '.mp4' and p.stat().st_size > 1024)
        if len(files) > 1:
            raise ValueError(f'Ambiguous video files for {clip}/{modality}')
        if files:
            if not files[0].resolve().is_relative_to(root):
                raise ValueError('Media symlink escapes the input directory.')
            return files[0], modality
    raise FileNotFoundError(f'No usable non-RGB video for {clip}')


def decode_video(path: Path) -> tuple[np.ndarray, dict]:
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError(f'Cannot open video: {path}')
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        declared = float(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError(f'Missing/invalid frame rate: {path}')
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame.ndim == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(cv2.resize(frame, (112, 112), interpolation=cv2.INTER_AREA))
    finally:
        cap.release()
    if not frames:
        raise ValueError(f'No decoded frames: {path}')
    if np.isfinite(declared) and declared > 0 and abs(declared-len(frames)) > 1:
        raise ValueError(f'Incomplete decode: reported {declared}, decoded {len(frames)}: {path}')
    idx = np.linspace(0, len(frames)-1, 32).round().astype(int)
    array = np.stack([frames[i] for i in idx]).astype(np.uint8)
    if not array.any():
        raise ValueError(f'All-zero decoded clip: {path}')
    # Retain the training feature contract: motion differences number N-1.
    # The legacy feature extractor returns all zeros below four differences.
    # Valid very short clips still produce repeated sampled frames, never zero clips.
    return array, dict(decoded_frames=len(frames), declared_frames=declared, fps=fps,
                       decoded_duration_s=len(frames)/fps,
                       duration_s=(len(frames)-1)/fps if len(frames)>=5 else 0.0)


def prepare_media(questions: pd.DataFrame, media_root: Path, output: Path) -> pd.DataFrame:
    output.mkdir(parents=True, exist_ok=False)
    clips = sorted(questions.clip_id.unique())
    array = np.lib.format.open_memmap(output/'test_Depth.npy', mode='w+', dtype=np.uint8,
                                     shape=(len(clips), 32, 112, 112))
    records = []
    for i, clip in enumerate(clips):
        video, modality = resolve_video(media_root, clip)
        array[i], info = decode_video(video)
        records.append(dict(clip_id=clip, found=True, modality=modality,
                            source_relative_path=str(video.relative_to(media_root.resolve())),
                            source_sha256=sha256(video), **info))
        if (i+1) % 25 == 0 or i+1 == len(clips):
            print(f'Decoded {i+1}/{len(clips)} clips', flush=True)
    array.flush()
    (output/'test_Depth_index.json').write_text(json.dumps(clips))
    feats = pd.DataFrame(records)
    feats.to_csv(output/'depth_features.csv', index=False)
    (output/'decode_manifest.json').write_text(json.dumps(dict(clips=len(clips),
        arrays_sha256=sha256(output/'test_Depth.npy'),
        index_sha256=sha256(output/'test_Depth_index.json'),
        feature_sha256=sha256(output/'depth_features.csv'),
        modality_counts=feats.modality.value_counts().to_dict()), indent=2))
    return feats


def read_scores(root: Path, expected_clips: list[str], expected_actions: list[str], split='test') -> dict:
    matrix = np.load(root/f'{split}_scores.npy', allow_pickle=False)
    index_name = 'training_index.json' if split == 'oof' else 'test_index.json'
    index = json.loads((root/index_name).read_text())
    actions = json.loads((root/'actions.json').read_text())
    if len(index) != len(set(index)) or set(index) != set(expected_clips):
        raise ValueError('Score clip IDs do not match the input questions exactly.')
    if actions != expected_actions or len(actions) != len(set(actions)):
        raise ValueError('Score action columns do not match the model bundle.')
    if matrix.shape != (len(index), len(actions)) or not np.isfinite(matrix).all():
        raise ValueError('Invalid score shape or nonfinite scores.')
    if not ((matrix >= 0) & (matrix <= 1)).all():
        raise ValueError('Scores must be probabilities.')
    return {c: dict(zip(actions, map(float, row))) for c, row in zip(index, matrix)}
