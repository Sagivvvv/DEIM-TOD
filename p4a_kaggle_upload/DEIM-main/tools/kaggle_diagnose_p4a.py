"""P4-A: full-training gradient census, frozen P4 checkpoints, no training."""
import argparse
from dataclasses import asdict
import gzip
import hashlib
import io
import json
from pathlib import Path
import platform
import sys
import traceback
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))
import numpy as np
import torch
from kaggle_train_p4 import P3Reference
from kaggle_diagnose_p1 import compressed_json
from TileStitchingNet.p2 import atomic_json
from TileStitchingNet.p3 import PairConfig, pair_tensors
from TileStitchingNet.p4 import RoiRescorer, annotation_audit
from TileStitchingNet.p4a import gradient_census, check_original_probe
from TileStitchingNet.pipeline_deim_tilestitching import OfflineDatasetIO
from TileStitchingNet.stage2 import QualityConfig, QualityTargets, prepare


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


class P4Reference(P3Reference):
    def __init__(self, path=None):
        folder = ROOT / 'TileStitchingNet/reports/p4_20261117'
        requested = Path(path) if path else folder / 'p4_reports.zip'
        candidates = [requested, requested.with_suffix(''), folder] if path is None else [requested, requested.with_suffix('')]
        self.path = next((p for p in candidates if p.is_file() or (p / 'p4_summary.json').is_file()), None)
        if self.path is None:
            raise FileNotFoundError('Missing P4 report ZIP or expanded folder')
        if self.path.name == 'p4_summary.json':
            self.path = self.path.parent
        self.archive = None

    def __enter__(self):
        try:
            if self.path.is_file():
                self.archive = zipfile.ZipFile(self.path)
            self.config = self.json('run_config.json')
            self.summary = self.json('p4_summary.json')
            if self.summary['status'] != 'P4_ROI_COMPLETED' or self.config['smoke']:
                raise ValueError('A completed full P4 report is required')
            return self
        except Exception:
            self.__exit__()
            raise


def jobs_for(ref, smoke):
    jobs = []
    for method, result in ref.summary['results'].items():
        history = ref.json(method + '/history.json')
        best = result['best_epoch']
        epochs = [best] if smoke else sorted({0, min(10, ref.config['config']['epochs']), ref.config['config']['epochs'], best})
        for epoch in epochs:
            is_best = epoch == best
            name = result['best_checkpoint'].replace('\\', '/') if is_best else f'{method}/epoch_{epoch:02d}.pth'
            raw = ref.read(name)
            if is_best and sha_bytes(raw) != result['best_checkpoint_sha256']:
                raise ValueError('P4 best checkpoint hash mismatch: ' + name)
            expected = ref.json(method + '_best_gradients.json')['diagnostics'] if is_best else history[epoch]['validation'][method]['gradient_probe']
            jobs.append(dict(method=method, epoch=epoch, name=name, sha256=sha_bytes(raw),
                             expected_probe=expected, roi_mode=result['roi_mode'], pair_weight=result['pair_weight']))
    return jobs


def execute(args, output, report):
    if args.threads < 1:
        raise ValueError('threads must be positive')
    torch.set_num_threads(args.threads)
    with P4Reference(args.p4_report) as ref:
        cfg = ref.config
        config = QualityConfig(**cfg['config'])
        pair_config = PairConfig(**cfg['pair_config'])
        if config.device != 'cpu' or config.rank_weight != 0:
            raise ValueError('Unexpected P4 training configuration')
        files = {'cache': Path(args.train_cache), 'gt': Path(args.train_gt),
                 'manifest': Path(args.manifest or args.train_cache + '.manifest.json')}
        input_hashes = {key: sha_file(path) for key, path in files.items()}
        for key, value in input_hashes.items():
            if value != cfg[key + '_sha256']:
                raise ValueError('P4 input changed: ' + key)
        sources = {path: sha_file(ROOT / path) for path in cfg['source_hashes']}
        if sources != cfg['source_hashes']:
            raise ValueError('Reused P4 sources changed')
        for path in ['TileStitchingNet/p4a.py', 'tools/kaggle_diagnose_p4a.py', 'tools/kaggle_run_p4a.py']:
            sources[path] = sha_file(ROOT / path)
        jobs = jobs_for(ref, args.smoke)
        members = ['p4_summary.json', 'run_config.json', 'split.json', 'gradient_probe_selection.json',
                   'train_pairs.json.gz', 'train_pair_geometry.json.gz', 'validation_pair_geometry.json.gz']
        reference_hashes = {name: sha_bytes(ref.read(name)) for name in members}
        signature = dict(schema='p4a-v1', inputs=input_hashes, reference=reference_hashes, jobs=jobs,
                         source_hashes=sources, config=asdict(config), pair_config=asdict(pair_config),
                         smoke=args.smoke, threads=args.threads, torch=str(torch.__version__),
                         numpy=np.__version__, python=platform.python_version())
        config_path = output / 'run_config.json'
        if config_path.exists():
            if not args.resume or json.loads(config_path.read_text(encoding='utf8')) != signature:
                raise ValueError('Existing output requires --resume with identical inputs/code/runtime')
        atomic_json(signature, config_path)
        for name in sources:
            target = output / 'sources' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((ROOT / name).read_bytes())
        split = ref.json('split.json')
        if set(split['train_ids']) & set(split['val_ids']):
            raise ValueError('Training/validation overlap')
        original_ids = ref.json('gradient_probe_selection.json')['image_ids']
        if not set(original_ids) <= set(split['train_ids']):
            raise ValueError('P4 probe leaks validation images')
        samples = OfflineDatasetIO.load_samples(str(files['cache']))
        by_id = {s.image_id: s for s in samples}
        if len(by_id) != len(samples) or set(by_id) != set(split['train_ids']) | set(split['val_ids']):
            raise ValueError('Cache image set differs from P4')
        geometry = {**ref.json('train_pair_geometry.json.gz'), **ref.json('validation_pair_geometry.json.gz')}
        targets = QualityTargets(files['gt'], config)
        annotation = annotation_audit(json.loads(files['gt'].read_text(encoding='utf8')), targets, split, geometry)
        atomic_json(annotation, output / 'annotation_audit.json')
        if annotation['status'] != 'PASS':
            raise ValueError('Annotation audit failed')
        pairs = ref.json('train_pairs.json.gz')
        if pairs['signature'] != ref.summary['pairs']['train']['signature']:
            raise ValueError('Frozen pair signature differs')
        ids = original_ids if args.smoke else split['train_ids']
        batches, pair_lists = [], []
        for image_id in ids:
            sample = by_id[image_id]
            for attribute, field in [('boxes_global', 'boxes'), ('scores', 'scores'), ('patch_ids', 'patch_ids')]:
                value = getattr(sample, attribute)
                if not np.array_equal(value, np.asarray(geometry[image_id][field], dtype=value.dtype).reshape(value.shape)):
                    raise ValueError('Frozen candidate data changed: ' + image_id)
            if sample.node_features.shape != (len(sample.scores), 76) or not np.isfinite(sample.node_features).all():
                raise ValueError('Invalid input features: ' + image_id)
            rows = pairs['pairs'][image_id]
            batches.append(dict(prepare(sample, config, False, targets), **pair_tensors(rows)))
            pair_lists.append(rows)
        atomic_json(dict(image_ids=ids, original_probe_ids=original_ids,
            pair_counts=[len(p) for p in pair_lists],
            rule='All training images; exclude empty/no-valid-label images exactly as P4 training. '
                 'Smoke uses original training probe only; no new selection by validation metrics.'), output / 'selection.json')
        report.update(input=dict(training_images_requested=len(ids), original_training_images=len(split['train_ids']),
                                 new_training_runs=0, test_evaluated=False, validation_evaluated=False), results={})
        for job in jobs:
            method, epoch = job['method'], job['epoch']
            key = f'{method}_epoch_{epoch:02d}'
            path = output / (key + '.json')
            detail_path = output / (key + '_images.json.gz')
            if args.resume and path.exists():
                result = json.loads(path.read_text(encoding='utf8'))
                if result['checkpoint_sha256'] != job['sha256'] or sha_file(detail_path) != result['detail_sha256']:
                    raise ValueError('Resume diagnostic artifacts changed: ' + key)
                report['results'][key] = result
                print('[resume]', key, flush=True)
                continue
            raw = ref.read(job['name'])
            payload = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)
            if payload['schema'] != 'p4-v1' or payload['config'] != asdict(config) or payload['epoch'] != epoch:
                raise ValueError('Checkpoint metadata differs: ' + key)
            if payload['roi_mode'] != job['roi_mode'] or payload['pair_weight'] != job['pair_weight'] or payload['pair_config'] != asdict(pair_config):
                raise ValueError('Checkpoint supervision differs: ' + key)
            model = RoiRescorer(payload['feature_dim'], config, payload['roi_mode'])
            model.load_state_dict(payload['model'], strict=True)
            print('[diagnose]', key, 'images=', len(ids), flush=True)
            result, rows = gradient_census(model, batches, pair_lists, ids, config, pair_config,
                job['pair_weight'], original_ids,
                progress=lambda done, total: print(f'  {key}: {done}/{total}', flush=True))
            result['original_probe_reproduction'] = check_original_probe(result['cohorts']['original_probe'], job['expected_probe'])
            if any(not torch.equal(v, payload['model'][k]) for k, v in model.state_dict().items()):
                raise AssertionError('Diagnostic mutated checkpoint parameters')
            compressed_json(detail_path, rows)
            result.update(method=method, epoch=epoch, roi_mode=job['roi_mode'],
                          checkpoint_sha256=job['sha256'], detail_sha256=sha_file(detail_path))
            atomic_json(result, path)
            report['results'][key] = result
            total = result['cohorts']['all_training']['pair_components']['all']
            print('  pair/QFL=', total['weighted_to_qfl_norm'], 'cos=', total['cosine_to_qfl'], flush=True)
        if any(sha_file(path) != input_hashes[key] for key, path in files.items()):
            raise AssertionError('Inputs changed during diagnosis')
        if any(sha_bytes(ref.read(job['name'])) != job['sha256'] for job in jobs):
            raise AssertionError('Reference weights changed during diagnosis')
        report['status'] = 'P4A_SMOKE_COMPLETED' if args.smoke else 'P4A_DIAGNOSTICS_COMPLETED'
        report['notes'] = ['No optimizer, training, AP evaluation or new checkpoint selection.',
            'Full run enumerates the original training split; P4 validation-selected checkpoints remain frozen.',
            'Same P3 GT-group normalization within images; gradient vectors averaged across eligible images.',
            'Raw gradients before clipping; neither Adam steps nor proof of causal loss conflict.',
            'Existing P4 8-image probe must reproduce within stated floating point tolerances.']


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-cache', required=True)
    parser.add_argument('--train-gt', required=True)
    parser.add_argument('--manifest')
    parser.add_argument('--p4-report')
    parser.add_argument('--output', default='/kaggle/working/p4a_full')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    report = dict(status='RUNNING', arguments=vars(args))
    try:
        execute(args, output, report)
    except Exception:
        report.update(status='ERROR', traceback=traceback.format_exc())
        raise
    finally:
        atomic_json(report, output / 'p4a_summary.json')
        archive = output / 'p4a_reports.zip'
        with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
            for path in sorted(output.rglob('*')):
                if path.is_file() and path != archive and path.suffix != '.tmp':
                    z.write(path, path.relative_to(output).as_posix())
        print('Reports:', archive, flush=True)


if __name__ == '__main__':
    main()
