"""Run P4-B smoke and the full controlled margin-loss experiment."""
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from kaggle_run_p4 import locate


def main():
    cache = locate('/kaggle/input/datasets/xyjxyjxyj55/p0back/p02/p0r_train.pt', 'p0r_train.pt')
    gt = locate('/kaggle/input/datasets/xyjxyjxyj55/tinyperson/datasets/Tinyperson/annotations/tiny_set_train_all.json', 'tiny_set_train_all.json')
    command = [sys.executable, '-B', str(ROOT / 'tools/kaggle_train_p4b.py'),
               '--train-cache', str(cache), '--train-gt', str(gt), '--epochs', '18', '--threads', '4', '--resume']
    env = dict(os.environ, PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1')
    for folder, extra in [('p4b_smoke', ['--smoke']), ('p4b_full', [])]:
        subprocess.run(command + ['--output', '/kaggle/working/' + folder] + extra, cwd=ROOT, env=env, check=True)
    print('Send back: /kaggle/working/p4b_full/p4b_reports.zip', flush=True)


if __name__ == '__main__':
    main()
