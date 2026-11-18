"""Kaggle entry point. A completed validation report can be mounted in a later session."""
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
import zipfile

ROOT = Path(__file__).resolve().parents[1]
INPUT = Path('/kaggle/input')
WORK = Path('/kaggle/working')


def locate(name, preferred):
    if (INPUT/preferred/name).is_file():
        return INPUT/preferred/name
    paths = sorted(INPUT.rglob(name))
    if len(paths)>1:
        raise ValueError(f'Multiple {name}: {paths}. Use kaggle_train_p5.py with explicit paths.')
    return paths[0] if paths else None


def frozen_report():
    name = 'p5_validation_reports.zip'
    local = WORK/'p5_validation'/name
    if local.is_file():
        return local
    archives = list(INPUT.rglob(name))
    if archives:
        if len(archives)!=1:
            raise ValueError(f'Multiple P5 validation archives: {archives}')
        return archives[0]
    folders = []
    for marker in INPUT.rglob('frozen_config.json'):
        status = marker.with_name('status.json')
        if status.exists() and json.loads(status.read_text())['status']=='P5_VALIDATION_FROZEN':
            folders.append(marker.parent)
    if len(folders)>1:
        raise ValueError(f'Multiple P5 validation folders: {folders}')
    return folders[0] if folders else None


def run(*args):
    subprocess.run([sys.executable,'-B',str(ROOT/'tools/kaggle_train_p5.py'),*map(str,args)],
                   cwd=ROOT,env=dict(os.environ,PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1'),check=True)


def main():
    cache_dir = Path('datasets/xyjxyjxyj55/p0back/p02')
    gt_dir = Path('datasets/xyjxyjxyj55/tinyperson/datasets/Tinyperson/annotations')
    freeze = frozen_report()
    if freeze is None:
        train = locate('p0r_train.pt',cache_dir)
        gt = locate('tiny_set_train_all.json',gt_dir)
        if train is None or gt is None or not Path(str(train)+'.manifest.json').is_file():
            raise FileNotFoundError('Attach p0r_train.pt with its adjacent manifest and tiny_set_train_all.json')
        run('validate','--train-cache',train,'--train-gt',gt,'--output',WORK/'p5_smoke','--smoke')
        run('validate','--train-cache',train,'--train-gt',gt,'--output',WORK/'p5_validation')
        freeze = WORK/'p5_validation/p5_validation_reports.zip'
    test = locate('p0r_test.pt',cache_dir)
    gt = locate('tiny_set_test_all.json',gt_dir)
    if test is None or gt is None or not Path(str(test)+'.manifest.json').is_file():
        print('Validation frozen:',freeze,flush=True)
        print('For final test, attach this validation report, p0r_test.pt with its adjacent manifest, and tiny_set_test_all.json.',flush=True)
        return
    run('test','--frozen',freeze,'--test-cache',test,'--test-gt',gt,'--output',WORK/'p5_final')
    print('Send back:',WORK/'p5_final/p5_reports.zip',flush=True)


if __name__=='__main__':
    try:
        main()
    except Exception:
        with zipfile.ZipFile(WORK/'p5_error.zip','w',zipfile.ZIP_DEFLATED) as archive:
            archive.writestr('launcher.txt',traceback.format_exc())
            for folder in ('p5_smoke','p5_validation','p5_final'):
                error = WORK/folder/'error.json'
                if error.is_file():
                    archive.write(error,folder+'/error.json')
        print('Send back:',WORK/'p5_error.zip',flush=True)
        raise
