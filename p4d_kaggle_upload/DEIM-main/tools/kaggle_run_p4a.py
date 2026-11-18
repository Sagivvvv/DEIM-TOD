"""Run smoke and full P4-A diagnostics. No training and no new detector extraction."""
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
    command = [sys.executable, '-B', str(ROOT / 'tools/kaggle_diagnose_p4a.py'),
               '--train-cache', str(cache), '--train-gt', str(gt), '--threads', '4', '--resume']
    env = dict(os.environ, PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1')
    for folder, extra in [('p4a_smoke', ['--smoke']), ('p4a_full', [])]:
        subprocess.run(command + ['--output', '/kaggle/working/' + folder] + extra,
                       cwd=ROOT, env=env, check=True)
    print('Send back: /kaggle/working/p4a_full/p4a_reports.zip', flush=True)


if __name__ == '__main__':
    main()
