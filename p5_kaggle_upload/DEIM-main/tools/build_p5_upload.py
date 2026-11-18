"""Package the runnable P5 notebook, exact P4 controls and P0-R input fingerprints."""
import gzip
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT.parent


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


NOTEBOOK = """from pathlib import Path
import os, subprocess, sys, zipfile

inputs = Path('/kaggle/input')
markers = list(inputs.rglob('P5_UPLOAD.json'))
if not markers:
    archives = list(inputs.rglob('p5_kaggle_upload.zip'))
    if len(archives) != 1:
        raise RuntimeError(f'Attach one p5_kaggle_upload code dataset. Found {archives}')
    destination = Path('/kaggle/working/p5_code')
    with zipfile.ZipFile(archives[0]) as z:
        z.extractall(destination)
    markers = list(destination.rglob('P5_UPLOAD.json'))
if len(markers) != 1:
    raise RuntimeError(f'Expected one P5 code dataset. Found {markers}')
root = markers[0].parent
subprocess.run([sys.executable, '-B', str(root/'tools/kaggle_run_p5.py')],
               cwd=root, env=dict(os.environ, PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1'), check=True)
"""


def main():
    report_dir = ROOT/'TileStitchingNet/reports'
    reference_dir = Path('TileStitchingNet/reports/p4_20261117/p4_reports')
    payload = {}
    with zipfile.ZipFile(WORK/'p4d_kaggle_upload.zip') as prior:
        for original in prior.namelist():
            name = original.removeprefix('DEIM-main/')
            if '/reports/' not in name and name!='P4D_UPLOAD.json':
                payload[name] = (ROOT/name).read_bytes()
    added = ['TileStitchingNet/p5.py','tools/kaggle_train_p5.py','tools/kaggle_run_p5.py']
    for name in [*added,'TileStitchingNet/test_p5.py','tools/build_p5_upload.py']:
        payload[name] = (ROOT/name).read_bytes()
    with zipfile.ZipFile(report_dir/'p4_20261117/p4_reports.zip') as p4:
        cfg, summary = [json.loads(p4.read(n)) for n in ('run_config.json','p4_summary.json')]
        names = ['run_config.json','p4_summary.json','split.json','train_pairs.json.gz','p0r_traditional_reference.json']
        for family in ('A_full','C_full'):
            names += [family+'/history.json',summary['results'][family]['best_checkpoint'].replace(chr(92),'/')]
        ref = {}
        for name in names:
            raw = p4.read(name)
            logical = gzip.decompress(raw) if name.endswith('.gz') else raw
            ref[name] = digest(logical)
            # Store logical JSON directly for portability under Kaggle archive expansion.
            payload[(reference_dir/name.removesuffix('.gz')).as_posix()] = logical
        split = json.loads(p4.read('split.json'))
    sources = {name:digest(payload[name]) for name in cfg['source_hashes']}
    if sources!=cfg['source_hashes']:
        raise ValueError('Current source differs from the code used in P4')
    sources.update({name:digest(payload[name]) for name in [*added,'tools/kaggle_diagnose_p4a.py','TileStitchingNet/trainEval.py']})
    anchor = dict(epochs=cfg['config']['epochs'],split_sizes=[len(split[k]) for k in ('train_ids','val_ids')],
                  p4=ref,source_hashes=sources)
    with zipfile.ZipFile(report_dir/'p0r_20261116/p0r_reports.zip') as p0:
        status = json.loads(p0.read('p0r_summary.json'))
        for side in ('train','test'):
            raw = p0.read(f'p0r_{side}.pt.manifest.json')
            manifest = json.loads(raw)
            anchor[side] = dict(images=len(status[side+'_refresh']),manifest_sha256=digest(raw),
                                **{k:manifest[k] for k in ('cache_sha256','gt_sha256')})
    if (anchor['epochs']!=18 or anchor['split_sizes']!=[597,149] or anchor['test']['images']!=786
            or any(anchor['train'][k]!=cfg[k] for k in ('cache_sha256','manifest_sha256','gt_sha256'))):
        raise ValueError('Historical P5 input contract failed')
    anchor_bytes = json.dumps(anchor,indent=2).encode()
    (ROOT/'P5_REFERENCE.json').write_bytes(anchor_bytes)
    payload['P5_REFERENCE.json'] = anchor_bytes
    payload['P5_UPLOAD.json'] = json.dumps(dict(stage='P5',files={n:digest(b) for n,b in payload.items()}),indent=2).encode()
    output = WORK/'p5_kaggle_upload.zip'
    temp = output.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED) as z:
        for name,raw in sorted(payload.items()):
            z.writestr('DEIM-main/'+name,raw)
    with zipfile.ZipFile(temp) as z:
        if z.testzip() or any(z.read('DEIM-main/'+name)!=raw for name,raw in payload.items()):
            raise ValueError('Upload verification failed')
    temp.replace(output)
    compile(NOTEBOOK,'<P5 notebook>','exec')
    notebook = dict(nbformat=4,nbformat_minor=5,
        metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3')),
        cells=[dict(cell_type='code',id='p5-run',metadata={},execution_count=None,outputs=[],source=NOTEBOOK.splitlines(keepends=True))])
    (WORK/'p5_kaggle.ipynb').write_text(json.dumps(notebook,indent=2),encoding='utf8')
    print(output,output.stat().st_size,digest(output.read_bytes()))


if __name__=='__main__':
    main()
