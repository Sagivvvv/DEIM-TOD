"""Build the original Soft-NMS Cython function and Darknet box.c on Windows/Linux."""
import argparse
import os
from pathlib import Path
import sys

import numpy as np
from Cython.Build import cythonize
from setuptools import Distribution, Extension
from setuptools._distutils.ccompiler import new_compiler
from setuptools._distutils.sysconfig import customize_compiler
from tile_postprocess_adapters import checked_repo


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('sources', type=Path)
    args = p.parse_args()
    root = args.sources.resolve()
    out = root/'compiled'
    out.mkdir(exist_ok=True)
    soft = checked_repo(root,'soft','b8e69bdf8df2ad53025c9d198ded909b50471d4f')
    # The subsequent, unrelated hard-NMS function uses removed NumPy 1.x types.
    # Build the untouched soft function, helpers and copyright header only.
    raw = (soft/'lib/nms/cpu_nms.pyx').read_text().split('def cpu_nms(')[0]
    pyx = out/'original_soft_nms.pyx'
    pyx.write_text(raw.expandtabs(8), encoding='utf8')
    ext = Extension('original_soft_nms',[str(pyx)],include_dirs=[np.get_include()])
    dist = Distribution({'ext_modules':cythonize([ext], compiler_directives={'language_level':3})})
    cmd = dist.get_command_obj('build_ext')
    cmd.build_lib, cmd.build_temp = str(out), str(out/'build')
    dist.run_command('build_ext')

    diou = checked_repo(root,'diou','257ba175cb626e6d2b10da25f49332c71696550a')
    cc = new_compiler()
    customize_compiler(cc)
    include = [str(diou/'include'),str(diou/'src')]
    if sys.platform == 'win32':
        # box.c does not use pthreads. darknet.h declares unrelated loader functions.
        (out/'pthread.h').write_text('#include <time.h>\ntypedef void *pthread_t;\n')
        include.insert(0,str(out))
    objects = cc.compile([str(diou/'src/box.c')], output_dir=str(out/'build'), include_dirs=include,
                         extra_postargs=['/O2'] if os.name=='nt' else ['-O2','-fPIC'])
    cc.link_shared_object(objects,str(out/cc.library_filename('darknet_box',lib_type='shared')),
                          export_symbols=['diounms_sort'], libraries=[] if os.name=='nt' else ['m'])


if __name__ == '__main__':
    main()
