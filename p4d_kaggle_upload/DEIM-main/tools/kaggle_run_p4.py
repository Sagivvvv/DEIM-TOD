"""Run P4 smoke then full training on Kaggle using the existing cache and GT."""
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
INPUT = Path('/kaggle/input')


def locate(preferred, filename):
    preferred = Path(preferred)
    if preferred.is_file():
        return preferred
    matches = sorted(p for p in INPUT.rglob(filename) if p.is_file())
    if len(matches) != 1:
        raise FileNotFoundError(f'Expected one {filename}; found {len(matches)}: {matches}. '
                                'Use kaggle_train_p4.py with explicit paths if several copies are mounted.')
    return matches[0]


def main():
    cache = locate('/kaggle/input/datasets/xyjxyjxyj55/p0back/p02/p0r_train.pt', 'p0r_train.pt')
    gt = locate('/kaggle/input/datasets/xyjxyjxyj55/tinyperson/datasets/Tinyperson/annotations/tiny_set_train_all.json',
                'tiny_set_train_all.json')
    manifest = Path(str(cache) + '.manifest.json')
    if not manifest.is_file():
        raise FileNotFoundError('Missing cache manifest: ' + str(manifest))
    print('P4 code:', ROOT, '\nCache:', cache, '\nGT:', gt, flush=True)
    command = [sys.executable, '-B', str(ROOT/'tools/kaggle_train_p4.py'),
               '--train-cache', str(cache), '--manifest', str(manifest), '--train-gt', str(gt),
               '--epochs', '18', '--threads', '4', '--resume']
    env = dict(os.environ, PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1')
    for folder, extra in [('p4_smoke', ['--smoke']), ('p4_full', [])]:
        subprocess.run(command + ['--output', '/kaggle/working/' + folder] + extra,
                       cwd=ROOT, env=env, check=True)
    print('Send back: /kaggle/working/p4_full/p4_reports.zip', flush=True)


if __name__ == '__main__':
    main()
