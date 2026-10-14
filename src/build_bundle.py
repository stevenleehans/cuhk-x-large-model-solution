"""Export an unpromoted replay bundle from verified six-fold video checkpoints."""
from __future__ import annotations
import argparse
import importlib.metadata
import json
from pathlib import Path
import pickle
import shutil
import sys
import numpy as np
import cv2
import common
from inference_io import read_scores, safe_relative, sha256
from pipeline import Pipeline
from take_model import load_features


def verify_video(root: Path, training) -> tuple[dict, dict]:
    manifest=json.loads((root/'manifest.json').read_text())
    actions=manifest['actions']
    if len(actions)!=44 or len(set(actions))!=44:
        raise ValueError('Unexpected action vocabulary.')
    if len(manifest['checkpoints'])!=6 or len(manifest['folds'])!=6 or manifest['config']['folds']!=6:
        raise ValueError('Six completed folds and checkpoints are required.')
    for name,expected in manifest['output_hashes'].items():
        if sha256(root/safe_relative(name))!=expected:
            raise ValueError(f'Video output hash mismatch: {name}')
    for entry in manifest['checkpoints']:
        if sha256(root/safe_relative(entry['file']))!=entry['sha256']:
            raise ValueError('Checkpoint hash mismatch.')
    if len({entry['file'] for entry in manifest['checkpoints']})!=6:
        raise ValueError('Duplicate checkpoint files.')
    index=json.loads((root/'training_index.json').read_text())
    fold_ids=[fold['fold'] for fold in manifest['folds']]
    if sorted(fold_ids)!=list(range(6)):
        raise ValueError('Missing or duplicate fold IDs.')
    coverage=np.zeros(len(index),dtype=int)
    for fold in manifest['folds']:
        ids=np.asarray(fold['validation_indices'],dtype=int)
        if len(ids)!=len(set(ids.tolist())) or (ids<0).any() or (ids>=len(index)).any():
            raise ValueError('Invalid held indices.')
        expected_subjects=set(fold['held_subjects'])
        configured_subjects=set(sorted(training.subject.unique())[fold['fold']::6])
        if expected_subjects!=configured_subjects:
            raise ValueError('Video fold assignment differs from the frozen QA folds.')
        observed={common.subject_of(index[i]) for i in ids}
        if observed!=expected_subjects or set(index[i] for i in ids)!={c for c in index if common.subject_of(c) in expected_subjects}:
            raise ValueError('Held subject/index mismatch.')
        if not 0<=fold['checkpoint_roundtrip_max_abs']<=1e-6:
            raise ValueError('Checkpoint roundtrip check failed.')
        coverage[ids]+=1
    if not (coverage==1).all():
        raise ValueError('OOF clip coverage must equal one.')
    if manifest['input_hashes']['training_Depth_index.json']!=sha256(common.ROOT/'data/clips/training_Depth_index.json'):
        raise ValueError('Training index identity mismatch.')
    for name in ['training_labels.npy','meta.json']:
        if manifest['input_hashes'][name]!=sha256(common.ROOT/'data/kaggle_upload'/name):
            raise ValueError(f'Training data identity mismatch: {name}')
    scores=read_scores(root,training.clip_id.unique().tolist(),actions,split='oof')
    return manifest,scores


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--video',type=Path,required=True)
    ap.add_argument('--video-source',type=Path,default=common.ROOT/'kernels/action_repro/train.py')
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--weight',type=float,default=.75)
    args=ap.parse_args()
    if not np.isfinite(args.weight) or args.weight<0:
        raise ValueError('Video weight must be finite and nonnegative.')
    training=common.load('training')
    manifest,scores=verify_video(args.video,training)
    if manifest.get('source_kind')=='python_file' and manifest['source_sha256']!=sha256(args.video_source):
        raise ValueError('Video inference source does not match training source.')
    args.output.mkdir(parents=True,exist_ok=False)
    model=Pipeline(video_scores=scores,video_weight=args.weight,harn_scores=scores).fit(training,load_features('training'))
    # Export fitted parameters only; fresh inference must not depend on old clip IDs.
    model.video_scores={}; model.harn_scores={}; model.train_feats=None
    (args.output/'pipeline.pkl').write_bytes(pickle.dumps(model,protocol=4))
    (args.output/'src').mkdir()
    for name in ['common.py','actions.py','action_set.py','text_prior.py','take_model.py','pipeline.py','inference_io.py','infer.py']:
        shutil.copy2(common.ROOT/'src'/name,args.output/'src'/name)
    shutil.copy2(args.video_source,args.output/'video_model.py')
    video_out=args.output/'video';video_out.mkdir()
    for name in ['manifest.json',*[entry['file'] for entry in manifest['checkpoints']]]:
        shutil.copy2(args.video/name,video_out/name)
    entry="""from pathlib import Path
import sys
root=Path(__file__).resolve().parent
sys.path.insert(0,str(root/'src'))
from infer import main
main(['--bundle',str(root),*sys.argv[1:]])
"""
    (args.output/'run.py').write_text(entry)
    dependencies={name:importlib.metadata.version(name) for name in ['numpy','pandas','scikit-learn','scipy','joblib','threadpoolctl']}
    try:
        opencv_package=importlib.metadata.version('opencv-python-headless')
    except importlib.metadata.PackageNotFoundError:
        opencv_package=importlib.metadata.version('opencv-python')
    (args.output/'requirements-cpu.txt').write_text('\n'.join(f'{k}=={v}' for k,v in dependencies.items())+f'\nopencv-python-headless=={opencv_package}\n')
    readme='''# Fresh-video inference bundle (unpromoted)

This bundle uses the new checkpoint model for both HAU and HARn. It does not reproduce the legacy v13 ensemble and has no score claim until separately evaluated.

Run on Kaggle GPU with the CPU dependencies in requirements-cpu.txt installed. The input directory contains test_qa.csv and either media/<question-relative paths> or the paths directly beneath the input directory. No training CSV, cached video scores, or old clip IDs are needed at inference.

    python run.py --input /kaggle/input/fresh-data --output /kaggle/working/predictions.csv --work /kaggle/working/fresh-run

The work directory must be new. Missing/corrupt media, mismatched indices, invalid output grammar, absent GPU, changed weights, or dependency mismatch stops execution. Input annotations, if present during replay, are removed. Decode and total timing plus hashes are saved to the work directory. Only load this locally created bundle: pipeline.pkl is a trusted Python model serialization.

The video script may install torch 2.5.1 / torchvision 0.20.1 if the supplied GPU is unsupported by the preinstalled build. Fresh inference uses local checkpoints and does not download pretrained weights. Internet may still be needed for that dependency installation.
'''
    (args.output/'README.md').write_text(readme)
    files={str(p.relative_to(args.output)):sha256(p) for p in args.output.rglob('*') if p.is_file()}
    bundle=dict(schema_version=1,status='unpromoted replay bundle',actions=manifest['actions'],
                weight=args.weight,video_manifest_sha256=sha256(args.video/'manifest.json'),
                training_qa_sha256=sha256(common.RAW/'training_qa.csv'),
                training_features_sha256=sha256(common.ROOT/'data/depth_features_training.csv'),
                cpu_dependencies=dependencies,opencv_runtime=cv2.__version__,files=files)
    (args.output/'bundle.json').write_text(json.dumps(bundle,indent=2)+'\n')
    print(f'Exported unpromoted bundle to {args.output}')


if __name__=='__main__':
    main()
