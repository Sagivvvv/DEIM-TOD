"""Fetch the pinned authors' source trees into the local benchmark source cache."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
from urllib.request import urlopen
import zipfile


REPOS = {
    'ta_tm': ('mohammad-AJP/PCB-Defect-Detection','5536f1867feb4d24bc7574b954afdd2f1afada75'),
    'cluster': ('Zzh-tju/DIoU-SSD-pytorch','cec038bc1057f0cd532752413b24924fde427f09'),
    'diou': ('Zzh-tju/DIoU-darknet','257ba175cb626e6d2b10da25f49332c71696550a'),
    'soft': ('bharatsingh430/soft-nms','b8e69bdf8df2ad53025c9d198ded909b50471d4f'),
    'wbf': ('ZFTurbo/Weighted-Boxes-Fusion','651884c5c7df13a920d286404b47c6fa86944782'),
    'sahi': ('obss/sahi','68e61981e1f79c56299c208df1d5bac03a01811d'),
}
CAR_SHA256 = '6e2eaa2f4c24b903765f33fe210b57281229709559c6c9baefe1c428a8fdc1a0'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sources',type=Path,default=Path(__file__).resolve().parents[1]/'output/postprocess_sources')
    directory = p.parse_args().sources.resolve()
    directory.mkdir(exist_ok=True,parents=True)
    manifest = {}
    for name,(repo,sha) in REPOS.items():
        path = directory/name
        url = 'https://github.com/'+repo+'.git'
        if not path.exists():
            subprocess.run(['git','init',str(path)],check=True)
            subprocess.run(['git','-C',str(path),'remote','add','origin',url],check=True)
        git = ['git','-C',str(path)]
        head = subprocess.run(git+['rev-parse','--verify','HEAD'],capture_output=True,text=True)
        if head.returncode:
            subprocess.run(git+['fetch','--depth','1',url,sha],check=True)
            subprocess.run(git+['checkout','--detach',sha],check=True)
        elif head.stdout.strip()!=sha:
            raise ValueError(f'{path} contains a different revision')
        subprocess.run(git+['diff','--quiet','HEAD','--'],check=True)
        licenses = [f.name for f in path.iterdir() if f.is_file() and f.name.lower().startswith(('license','copying'))]
        manifest[name] = dict(url=url,commit=sha,license_files=licenses)
    url = 'https://zenodo.org/records/23956349/files/CAR_code.zip'
    archive = directory/'CAR_code.zip'
    if not archive.exists():
        with urlopen(url,timeout=90) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest()!=CAR_SHA256:
            raise ValueError('Downloaded archive does not match v1.3.0')
        tmp = archive.with_suffix('.part')
        tmp.write_bytes(data)
        os.replace(tmp,archive)
    if hashlib.sha256(archive.read_bytes()).hexdigest()!=CAR_SHA256:
        raise ValueError('Unexpected CAR_code.zip checksum')
    with zipfile.ZipFile(archive) as z:
        root = directory/'car'
        for member in z.infolist():
            target = (root/member.filename).resolve()
            if not target.is_relative_to(root):
                raise ValueError('Invalid archive path')
            if member.is_dir():
                continue
            raw = z.read(member)
            if target.exists():
                if target.read_bytes()!=raw:
                    raise ValueError(f'Unexpected CAR file contents: {target}')
            else:
                target.parent.mkdir(parents=True,exist_ok=True)
                target.write_bytes(raw)
    manifest['car'] = dict(url=url,version='1.3.0',sha256=CAR_SHA256,license_files=['code/LICENSE','code/THIRD_PARTY_NOTICE.md'])
    (directory/'sources.lock.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
    print('Verified source trees and CAR v1.3.0 in',directory)


if __name__=='__main__':
    main()
