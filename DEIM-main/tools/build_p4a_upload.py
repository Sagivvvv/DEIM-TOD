"""Build the self-contained P4-A code/reference bundle and one-cell notebook."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT.parent


def main():
    payloads = {}
    with zipfile.ZipFile(WORK / 'p4_kaggle_upload.zip') as prior:
        for name in prior.namelist():
            if '/reports/' in name or name.endswith('P4_UPLOAD.json'):
                continue
            local = WORK / name
            if not local.is_file():
                raise FileNotFoundError(local)
            payloads[name] = local.read_bytes()
    for name in ['TileStitchingNet/p4a.py', 'TileStitchingNet/test_p4a.py',
                 'tools/kaggle_diagnose_p4a.py', 'tools/kaggle_run_p4a.py', 'tools/build_p4a_upload.py',
                 'TileStitchingNet/reports/p4_20261117/p4_reports.zip']:
        payloads['DEIM-main/' + name] = (ROOT / name).read_bytes()
    manifest = dict(stage='P4A', operation='Read-only full-training gradient diagnostics; no training',
                    files={name: hashlib.sha256(raw).hexdigest() for name, raw in payloads.items()})
    payloads['DEIM-main/P4A_UPLOAD.json'] = json.dumps(manifest, indent=2).encode()
    archive = WORK / 'p4a_kaggle_upload.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for name, raw in sorted(payloads.items()):
            z.writestr(name, raw)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for name, digest in manifest['files'].items():
            assert hashlib.sha256(z.read(name)).hexdigest() == digest
    code = '''from pathlib import Path
import os, subprocess, sys, zipfile

markers = list(Path('/kaggle/input').rglob('P4A_UPLOAD.json'))
if not markers:
    archives = list(Path('/kaggle/input').rglob('p4a_kaggle_upload.zip'))
    if len(archives) != 1:
        raise RuntimeError(f'Please attach one P4-A upload dataset; archives={archives}')
    destination = Path('/kaggle/working/p4a_code')
    with zipfile.ZipFile(archives[0]) as z:
        z.extractall(destination)
    markers = list(destination.rglob('P4A_UPLOAD.json'))
if len(markers) != 1:
    raise RuntimeError(f'Expected one P4-A code dataset; found {markers}')
root = markers[0].parent
print('P4-A code:', root)
subprocess.run([sys.executable, '-B', str(root/'tools/kaggle_run_p4a.py')],
               cwd=root, env=dict(os.environ, PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1'), check=True)
'''
    notebook = dict(nbformat=4, nbformat_minor=5,
        metadata=dict(kernelspec=dict(display_name='Python 3', language='python', name='python3')),
        cells=[dict(cell_type='code', id='p4a-run', metadata={}, execution_count=None,
                    outputs=[], source=code.splitlines(keepends=True))])
    (WORK / 'p4a_kaggle.ipynb').write_text(json.dumps(notebook, indent=2), encoding='utf8')
    print(json.dumps(dict(archive=str(archive), files=len(payloads), bytes=archive.stat().st_size,
                         sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                         notebook=str(WORK / 'p4a_kaggle.ipynb')), indent=2))


if __name__ == '__main__':
    main()
