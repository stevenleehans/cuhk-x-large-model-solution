"""One-entry fresh-media inference. Run only with a bundle created by this project."""
from __future__ import annotations
import argparse
import importlib.metadata
import json
from pathlib import Path
import pickle
import subprocess
import sys
import time
import common
from inference_io import prepare_media, read_questions, read_scores, safe_relative, sha256


def verify_bundle(bundle: Path) -> dict:
    import cv2
    manifest = json.loads((bundle/'bundle.json').read_text())
    if manifest.get('schema_version') != 1:
        raise ValueError('Unsupported bundle schema.')
    if cv2.__version__ != manifest['opencv_runtime']:
        raise RuntimeError('OpenCV runtime differs from the bundle decode contract.')
    for relative, expected in manifest['files'].items():
        path = (bundle/safe_relative(relative)).resolve()
        if not path.is_relative_to(bundle.resolve()) or sha256(path) != expected:
            raise ValueError(f'Bundle file hash mismatch: {relative}')
    for name, expected in manifest['cpu_dependencies'].items():
        actual = importlib.metadata.version(name)
        if actual != expected:
            raise RuntimeError(f'{name} version {actual}; bundle requires {expected}. Install requirements-cpu.txt.')
    return manifest


def main(argv=None):
    ap=argparse.ArgumentParser()
    ap.add_argument('--input',type=Path,required=True,help='Directory with test_qa.csv and media paths')
    ap.add_argument('--bundle',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--work',type=Path,required=True,help='New scratch directory; existing directories are rejected')
    args=ap.parse_args(argv)
    started=time.time()
    manifest=verify_bundle(args.bundle)
    args.work.mkdir(parents=True,exist_ok=False)
    questions=read_questions(args.input/'test_qa.csv')
    root=args.input/'media' if (args.input/'media').is_dir() else args.input
    features=prepare_media(questions,root,args.work/'decoded')
    decode_seconds=time.time()-started
    # The video executable enforces Kaggle + CUDA and verifies all checkpoint hashes.
    subprocess.run([sys.executable,str(args.bundle/'video_model.py'),'--mode','infer',
                    '--input',str((args.work/'decoded').resolve()),
                    '--checkpoints',str((args.bundle/'video').resolve()),
                    '--output',str((args.work/'scores').resolve())],check=True)
    scores=read_scores(args.work/'scores',questions.clip_id.unique().tolist(),manifest['actions'])
    with (args.bundle/'pipeline.pkl').open('rb') as stream:
        model=pickle.load(stream)
    model.video_scores=scores
    model.harn_scores=scores
    predictions=model.predict(questions,features)
    if len(predictions)!=len(questions):
        raise ValueError('Prediction count does not match question count.')
    common.write_submission(questions,predictions,args.output)
    record=dict(bundle_manifest_sha256=sha256(args.bundle/'bundle.json'),
                questions_sha256=sha256(args.input/'test_qa.csv'),
                output_sha256=sha256(args.output),questions=len(questions),
                clips=int(questions.clip_id.nunique()),decode_seconds=decode_seconds,
                total_seconds=time.time()-started)
    (args.work/'inference_report.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))


if __name__=='__main__':
    main()
