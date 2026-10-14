"""Kaggle-only checkpoint-producing full-clip action training and inference.

Train: python train.py
Fresh preprocessed arrays: python train.py --mode infer --input DIR --checkpoints DIR --output DIR
The inference input directory must contain test_Depth.npy and test_Depth_index.json.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

CONFIG = dict(folds=6, epochs=16, batch=16, lr=3e-4, weight_decay=1e-4,
              seed=20260908, frames_stored=32, frames_used=16,
              weights='R2Plus1D_18_Weights.KINETICS400_V1',
              temporal='one frame per chronological two-frame bin',
              inference='mean of phases 0/1 and horizontal flips',
              transform='official KINETICS400_V1 transforms')


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2) + '\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['train', 'infer'], default='train')
    ap.add_argument('--input', type=Path)
    ap.add_argument('--checkpoints', type=Path)
    ap.add_argument('--resume', type=Path, help='Directory of repaired final-epoch checkpoints')
    ap.add_argument('--output', type=Path, default=Path('/kaggle/working'))
    # Kaggle converts scripts to notebooks; ignore the IPython launcher's flags.
    args = ap.parse_args([] if 'ipykernel' in sys.modules else None)
    # Fail before any model construction or training on a local/non-GPU host.
    if not Path('/kaggle/input').is_dir() or not Path('/kaggle/working').is_dir():
        raise RuntimeError('This experiment requires Kaggle GPU execution.')
    import numpy as np
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('Kaggle CUDA GPU required; CPU fallback is prohibited.')
    cap = 'sm_%d%d' % torch.cuda.get_device_capability(0)
    if cap not in torch.cuda.get_arch_list():
        if os.environ.get('CUHKX_TORCH_REINSTALLED'):
            raise RuntimeError('GPU still unsupported after pinned dependency installation.')
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                        'torch==2.5.1', 'torchvision==0.20.1',
                        '--index-url', 'https://download.pytorch.org/whl/cu121'], check=True)
        os.environ['CUHKX_TORCH_REINSTALLED'] = '1'
        if '__file__' in globals() and Path(__file__).is_file():
            restart_file = Path(__file__)
        else:
            restart_file = Path('/kaggle/working/restart_action_repro.py')
            restart_file.write_text('\n'.join(get_ipython().history_manager.input_hist_raw))
        extra = [] if 'ipykernel' in sys.modules else sys.argv[1:]
        os.execv(sys.executable, [sys.executable, str(restart_file), *extra])
    import torchvision
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights
    from sklearn.metrics import average_precision_score
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(2)
    torch.ones(1, device='cuda').add_(1)
    started = time.time()
    if args.input is None:
        hits = list(Path('/kaggle/input').rglob('training_Depth.npy'))
        if len(hits) != 1:
            raise RuntimeError(f'Expected one input dataset, got {len(hits)}.')
        args.input = hits[0].parent
    args.output.mkdir(parents=True, exist_ok=True)
    weights = R2Plus1D_18_Weights.KINETICS400_V1
    transform = weights.transforms()
    if '__file__' in globals() and Path(__file__).is_file():
        source_text = Path(__file__).read_bytes()
        source_kind = 'python_file'
    else:
        source_text = '\n'.join(get_ipython().history_manager.input_hist_raw).encode()
        source_kind = 'executed_ipython_cells'
    provenance = dict(config=CONFIG, mode=args.mode, device=torch.cuda.get_device_name(),
                      capability=cap, torch=str(torch.__version__), torchvision=str(torchvision.__version__),
                      numpy=np.__version__, python=sys.version,
                      source_sha256=hashlib.sha256(source_text).hexdigest(), source_kind=source_kind,
                      transform=repr(transform))
    print(json.dumps(provenance), flush=True)

    def load_array(split):
        p = args.input / f'{split}_Depth.npy'
        idxpath = args.input / f'{split}_Depth_index.json'
        x = np.load(p, mmap_mode='r', allow_pickle=False)
        index = json.loads(idxpath.read_text())
        assert x.dtype == np.uint8 and x.shape == (len(index), 32, 112, 112)
        assert len(index) == len(set(index)) and len(index) > 0
        assert all(np.any(x[i]) for i in range(len(x))), 'All-zero clip detected'
        return x, index, {p.name: sha(p), idxpath.name: sha(idxpath)}

    test_x, test_index, hashes = load_array('test')

    class Clips(Dataset):
        def __init__(self, x, indices, labels=None, flags=None, training=False, phase=0):
            self.x, self.indices = x, np.asarray(indices)
            self.labels, self.flags, self.training, self.phase = labels, flags, training, phase

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, i):
            j = self.indices[i]
            offsets = np.random.randint(0, 2, size=16) if self.training else self.phase
            idx = np.arange(16) * 2 + offsets
            v = np.array(self.x[j][idx], dtype=np.float32) / 255.0
            if self.training:
                if np.random.rand() < .5:
                    v = v[:, :, ::-1]
                v = np.clip(v * np.random.uniform(.85, 1.15), 0, 1)
            # Official transform receives (T,C,H,W); apply on GPU by batch below.
            t = torch.from_numpy(np.ascontiguousarray(v)).unsqueeze(1).repeat(1, 3, 1, 1)
            if self.labels is None:
                return t
            return t, torch.from_numpy(self.labels[j]), bool(self.flags[j])

    def loader(dataset, training=False, seed=0):
        generator = torch.Generator().manual_seed(seed)
        # Main-process decoding keeps random sampling reproducible across platforms.
        return DataLoader(dataset, batch_size=CONFIG['batch'], shuffle=training,
                          num_workers=0, drop_last=False, pin_memory=True, generator=generator)

    def build(n_actions, pretrained):
        net = r2plus1d_18(weights=weights if pretrained else None)
        net.fc = nn.Linear(net.fc.in_features, n_actions)
        return net.to('cuda')

    @torch.inference_mode()
    def infer(model, x, indices, n_actions):
        model.eval()
        result = np.zeros((len(indices), n_actions), dtype=np.float32)
        for phase in [0, 1]:
            offset = 0
            for raw in loader(Clips(x, indices, phase=phase)):
                batch = transform(raw.to('cuda', non_blocking=True))
                with torch.amp.autocast('cuda'):
                    p = model(batch).sigmoid().float()
                    p = (p + model(batch.flip([4])).sigmoid().float()) / 2
                p = p.cpu().numpy()
                result[offset:offset + len(p)] += p / 2
                offset += len(p)
            assert offset == len(indices)
        assert np.isfinite(result).all() and ((result >= 0) & (result <= 1)).all()
        return result

    if args.mode == 'infer':
        if args.checkpoints is None:
            raise ValueError('--checkpoints is required for inference.')
        manifest = json.loads((args.checkpoints / 'manifest.json').read_text())
        assert manifest['config'] == CONFIG, 'Inference recipe differs from trained recipe'
        actions = manifest['actions']
        checkpoints = manifest['checkpoints']
        assert len(checkpoints) == CONFIG['folds']
        scores = np.zeros((len(test_x), len(actions)), dtype=np.float32)
        for entry in checkpoints:
            p = args.checkpoints / entry['file']
            assert sha(p) == entry['sha256']
            ckpt = torch.load(p, map_location='cpu', weights_only=True)
            assert ckpt['actions'] == actions and ckpt['config'] == CONFIG
            model = build(len(actions), False)
            model.load_state_dict(ckpt['state_dict'], strict=True)
            scores += infer(model, test_x, np.arange(len(test_x)), len(actions)) / len(checkpoints)
            del model
            torch.cuda.empty_cache()
        np.save(args.output / 'test_scores.npy', scores)
        dump(args.output / 'test_index.json', test_index)
        dump(args.output / 'actions.json', actions)
        dump(args.output / 'inference_manifest.json', provenance | dict(input_hashes=hashes,
             elapsed_seconds=time.time()-started, checkpoint_manifest_sha256=sha(args.checkpoints/'manifest.json')))
        return

    train_x, train_index, train_hashes = load_array('training')
    hashes.update(train_hashes)
    meta = json.loads((args.input / 'meta.json').read_text())
    y = np.load(args.input / 'training_labels.npy', allow_pickle=False).astype(np.float32)
    hashes.update({name: sha(args.input / name) for name in ['meta.json', 'training_labels.npy']})
    actions, subjects, sources = meta['actions'], np.asarray(meta['subjects']), np.asarray(meta['sources'])
    assert len(actions) == len(set(actions)) == 44
    assert y.shape == (len(train_x), len(actions)) and np.isfinite(y).all()
    assert ((y == 0) | (y == 1)).all() and (y.sum(1) >= 1).all()
    assert len(subjects) == len(sources) == len(train_x)
    assert not set(train_index) & set(test_index)
    for c, s, src in zip(train_index, subjects, sources):
        parts = c.split('/')
        assert parts[0] == src and parts[1 if src == 'HAU' else 2] == s
    assert (y[sources == 'HARn'].sum(1) == 1).all()
    oof = np.full_like(y, np.nan)
    coverage = np.zeros(len(y), dtype=int)
    test_scores = np.zeros((len(test_x), len(actions)), dtype=np.float32)
    folds = []
    checkpoints = []
    dump(args.output / 'run_started.json', provenance | dict(input_hashes=hashes, actions=actions))
    # Plain Python strings keep checkpoint metadata compatible with weights_only=True.
    subject_list = sorted(set(map(str, subjects)))
    # Reuse only verified, final-epoch checkpoints; there is no partial-epoch resume.
    if args.resume is None:
        candidates = list(Path('/kaggle/input').rglob('recovery_manifest.json'))
        if len(candidates) > 1:
            raise RuntimeError('Ambiguous checkpoint recovery inputs.')
        if candidates:
            args.resume = candidates[0].parent
    recovery_entries = {}
    if args.resume is not None:
        recovery_path = args.resume / 'recovery_manifest.json'
        recovery = json.loads(recovery_path.read_text())
        assert recovery['config'] == CONFIG and recovery['input_hashes'] == hashes
        for entry in recovery['checkpoints']:
            assert type(entry['fold']) is int and 0 <= entry['fold'] < CONFIG['folds']
            assert entry['fold'] not in recovery_entries
            assert Path(entry['file']).name == entry['file']
            assert entry['tensor_values_unchanged'] is True
            assert sha(args.resume / entry['file']) == entry['sha256']
            recovery_entries[entry['fold']] = entry
        provenance['recovery_manifest_sha256'] = sha(recovery_path)
        provenance['recovery'] = recovery
    for fold in range(CONFIG['folds']):
        seed = CONFIG['seed'] + fold
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        held = set(subject_list[fold::CONFIG['folds']])
        va = np.flatnonzero(np.isin(subjects, list(held)))
        tr = np.flatnonzero(~np.isin(subjects, list(held)))
        assert len(tr) and len(va) and not set(subjects[tr]) & set(subjects[va])
        recovered = recovery_entries.get(fold)
        model = build(len(actions), recovered is None)
        if recovered is not None:
            recovered_state = torch.load(args.resume / recovered['file'], map_location='cpu', weights_only=True)
            assert recovered_state['config'] == CONFIG and recovered_state['input_hashes'] == hashes
            assert recovered_state['fold'] == fold and recovered_state['seed'] == seed
            assert recovered_state['actions'] == actions and recovered_state['held_subjects'] == sorted(held)
            model.load_state_dict(recovered_state['state_dict'], strict=True)
            print(f'fold {fold}: resumed verified final-epoch checkpoint; training skipped', flush=True)
        opt = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])
        batches = loader(Clips(train_x, tr, y, sources == 'HARn', training=True), True, seed)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=CONFIG['lr'], total_steps=CONFIG['epochs']*len(batches))
        scaler = torch.amp.GradScaler('cuda')
        bce, ce = nn.BCEWithLogitsLoss(), nn.CrossEntropyLoss()
        for epoch in range(0 if recovered is not None else CONFIG['epochs']):
            model.train()
            losses = []
            for raw, target, harn in batches:
                batch = transform(raw.to('cuda', non_blocking=True))
                target, harn = target.to('cuda'), harn.to('cuda')
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda'):
                    logits = model(batch)
                    loss = bce(logits, target)
                    if harn.any():
                        loss = loss + .5 * ce(logits[harn], target[harn].argmax(1))
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite training loss')
                scaler.scale(loss).backward()
                scaler.step(opt); scaler.update(); sched.step()
                losses.append(loss.item())
            print(f'fold {fold} epoch {epoch+1}/{CONFIG["epochs"]} loss {np.mean(losses):.5f}', flush=True)
        ckpt_path = args.output / f'fold_{fold}.pt'
        if ckpt_path.exists():
            raise FileExistsError(f'Refusing to overwrite {ckpt_path}')
        torch.save(dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},
                        actions=actions, config=CONFIG, seed=seed, fold=fold,
                        held_subjects=sorted(held), input_hashes=hashes), ckpt_path)
        t0 = time.time()
        before = infer(model, train_x, va[:min(16, len(va))], len(actions))
        restored = build(len(actions), False)
        saved = torch.load(ckpt_path, map_location='cpu', weights_only=True)
        restored.load_state_dict(saved['state_dict'], strict=True)
        after = infer(restored, train_x, va[:min(16, len(va))], len(actions))
        max_error = float(np.abs(before-after).max())
        assert np.allclose(before, after, rtol=0, atol=1e-6), 'Checkpoint roundtrip failed'
        del model, opt, sched, scaler
        torch.cuda.empty_cache()
        oof[va] = infer(restored, train_x, va, len(actions))
        coverage[va] += 1
        fold_test = infer(restored, test_x, np.arange(len(test_x)), len(actions))
        test_scores += fold_test / CONFIG['folds']
        np.save(args.output / f'fold_{fold}_test_scores.npy', fold_test)
        ap_value = float(average_precision_score(y[va], oof[va], average='micro'))
        folds.append(dict(fold=fold, seed=seed, held_subjects=sorted(held),
                          validation_indices=va.tolist(), action_micro_ap=ap_value,
                          checkpoint_roundtrip_max_abs=max_error, inference_seconds=time.time()-t0,
                          resumed_from_sha256=recovered['sha256'] if recovered else None))
        checkpoints.append(dict(file=ckpt_path.name, sha256=sha(ckpt_path)))
        np.save(args.output / 'oof_partial.npy', oof)
        dump(args.output / 'progress.json', dict(folds=folds, checkpoints=checkpoints))
        print(f'fold {fold} COMPLETE action AP {ap_value:.5f}; checkpoint verified', flush=True)
        del restored
        torch.cuda.empty_cache()
    assert (coverage == 1).all() and np.isfinite(oof).all()
    np.save(args.output / 'oof_scores.npy', oof)
    np.save(args.output / 'test_scores.npy', test_scores)
    dump(args.output / 'training_index.json', train_index)
    dump(args.output / 'test_index.json', test_index)
    dump(args.output / 'actions.json', actions)
    manifest = provenance | dict(actions=actions, input_hashes=hashes, folds=folds,
                 checkpoints=checkpoints, elapsed_seconds=time.time()-started,
                 action_micro_ap=float(average_precision_score(y, oof, average='micro')))
    manifest['output_hashes'] = {n:sha(args.output/n) for n in ['oof_scores.npy','test_scores.npy','training_index.json','test_index.json','actions.json']}
    dump(args.output / 'manifest.json', manifest)
    print('COMPLETE ' + json.dumps(dict(action_micro_ap=manifest['action_micro_ap'],elapsed_seconds=manifest['elapsed_seconds'])), flush=True)

if __name__ == '__main__':
    main()
