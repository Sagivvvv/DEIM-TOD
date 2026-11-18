"""Verify P4-A provenance and scalar/per-image consistency, without rerunning gradients."""
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import zipfile
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ARCHIVE = HERE / 'p4a_reports.zip'
if not ARCHIVE.exists():
    shutil.copy2(Path(r'C:\Users\19333\Downloads\p4a_reports.zip'), ARCHIVE)
sha = lambda raw: hashlib.sha256(raw).hexdigest()
evidence = dict(zip_sha256=sha(ARCHIVE.read_bytes()),
    limitations=['No new AP evaluation or independent parameter-gradient rerun.',
                 'Global gradient directions cannot be reconstructed from exported per-image norms/cosines alone.'])
with zipfile.ZipFile(ARCHIVE) as z, zipfile.ZipFile(HERE.parent / 'p4_20261117/p4_reports.zip') as prior:
    assert z.testzip() is None
    def read(name):
        return json.loads(gzip.decompress(z.read(name)) if name.endswith('.gz') else z.read(name))
    report, config = read('p4a_summary.json'), read('run_config.json')
    assert report['status'] == 'P4A_DIAGNOSTICS_COMPLETED'
    assert not config['smoke'] and len(config['jobs']) == 13
    for name, digest in config['source_hashes'].items():
        assert sha((ROOT / name).read_bytes()) == digest, name
        assert sha(z.read('sources/' + name)) == digest, name
    for name, digest in config['reference'].items():
        raw = prior.read(name)
        assert sha(gzip.decompress(raw) if name.endswith('.gz') else raw) == digest
    selection = read('selection.json')
    old_split = json.loads(prior.read('split.json'))
    assert selection['image_ids'] == old_split['train_ids']
    assert not set(selection['image_ids']) & set(old_split['val_ids'])
    geometry = json.loads(gzip.decompress(prior.read('train_pair_geometry.json.gz')))
    evidence['jobs'] = {}
    for job in config['jobs']:
        key = f"{job['method']}_epoch_{job['epoch']:02d}"
        result = read(key + '.json')
        assert result == report['results'][key]
        assert sha(prior.read(job['name'])) == result['checkpoint_sha256'] == job['sha256']
        assert result['original_probe_reproduction']['status'] == 'PASS'
        for image_id in result['skipped_image_ids']:
            assert not geometry[image_id]['scores'] and not geometry[image_id]['normal_boxes']
        detail_name = key + '_images.json.gz'
        assert sha(z.read(detail_name)) == result['detail_sha256']
        rows = read(detail_name)
        ids = [r['image_id'] for r in rows]
        assert len(ids) == len(set(ids)) == 592
        assert set(ids) | set(result['skipped_image_ids']) == set(selection['image_ids'])
        for cohort, aggregate in result['cohorts'].items():
            selected = [r for r in rows if cohort == 'all_training' or
                (cohort == 'pair_images' and r['pairs']) or
                (cohort == 'original_probe' and r['image_id'] in selection['original_probe_ids'])]
            assert aggregate['images'] == len(selected)
            assert aggregate['pairs'] == sum(r['pairs'] for r in selected)
            assert aggregate['gt_groups'] == sum(r['gt_groups'] for r in selected)
            assert np.isclose(aggregate['qfl_loss'], np.mean([r['losses']['qfl'] for r in selected]))
            for component, values in aggregate['pair_components'].items():
                assert np.isclose(values['loss'], np.mean([r['losses'][component] for r in selected]))
                cosines = [r['components'][component]['cosine_to_qfl'] for r in selected
                           if r['components'][component]['cosine_to_qfl'] is not None]
                stats = values['per_image_cosine']
                assert stats['defined'] == len(cosines)
                assert stats['negative'] == sum(c < 0 for c in cosines)
                if cosines:
                    assert np.allclose(stats['quantiles'], np.quantile(cosines, [0, .1, .25, .5, .75, .9, 1]))
                    assert np.isclose(stats['negative_fraction'], np.mean(np.array(cosines) < 0))
                assert np.isclose(values['weighted_gradient_norm'], aggregate['pair_weight'] * values['raw_gradient_norm'])
                if values['weighted_to_qfl_norm'] is not None:
                    assert np.isclose(values['weighted_to_qfl_norm'], values['weighted_gradient_norm'] / aggregate['qfl_gradient_norm'])
            assert np.isclose(aggregate['pair_components']['all']['loss'], sum(
                aggregate['pair_components'][c]['loss'] for c in ['base_correct', 'base_wrong', 'base_tie']))
        for component, expected in job['expected_probe']['pair_components'].items():
            actual = result['cohorts']['original_probe']['pair_components'][component]
            for field in ['loss', 'raw_gradient_norm', 'cosine_to_qfl']:
                if expected[field] is None:
                    assert actual[field] is None
                else:
                    assert np.isclose(actual[field], expected[field], rtol=3e-4, atol=2e-6)
        total = result['cohorts']['all_training']
        pc = total['pair_components']
        evidence['jobs'][key] = dict(cohorts=result['cohorts'], skipped=result['skipped_image_ids'],
            pair_partition_resultant_ratio=pc['all']['raw_gradient_norm'] / sum(
                pc[c]['raw_gradient_norm'] for c in ['base_correct', 'base_wrong', 'base_tie']))
    evidence.update(status='PASS', source_hashes_verified=len(config['source_hashes']),
                    checkpoint_hashes_verified=len(config['jobs']), detail_hashes_verified=len(config['jobs']),
                    per_image_records_verified=13 * 592, training_images=597, eligible_images=592,
                    pair_images=451, pairs=14837, gt_groups=7639)
(HERE / 'review_evidence.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding='utf8')
print(json.dumps({k: v for k, v in evidence.items() if k != 'jobs'}, ensure_ascii=False, indent=2))
