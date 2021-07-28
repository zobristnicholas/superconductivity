import os
import sys
from setuptools import find_packages
from distutils.util import get_platform
from numpy.distutils.core import setup, Extension


def get_version(path):
    with open(path, "r") as f:
        for line in f.readlines():
            if line.startswith('__version__'):
                sep = '"' if '"' in line else "'"
                return line.split(sep)[1]
        else:
            raise RuntimeError("Unable to find version string.")


# In order to properly include the .mod fortran files we have to include a
# temporary directory that gets created during the build. The directory's
# name is platform dependent, so our workaround is to use the same code to
# generate the name as in numpy's distutils. This will break if numpy ever
# changes their specification. It works as of 1.19.2 (line 34 of build.py):
# https://github.com/numpy/numpy/blob/v1.19.2/numpy/distutils/command/build.py
platform = ".{}-{}.{}".format(get_platform(), *sys.version_info[:2])
extension = Extension(name="superconductivity.multilayer.usadel",
                      sources=["src/superconductivity/multilayer/usadel.f90"],
                      libraries=["bvp_m-2", "bvp_la-2", "pchip"],
                      include_dirs=["build/temp" + platform],
                      extra_f90_compile_args=["-fopenmp"],
                      extra_link_args=['-lgomp'])

setup(name='superconductivity',
      description='Tools for computing the properties of superconductors',
      version=get_version("src/superconductivity/__init__.py"),
      author='Nicholas Zobrist',
      license='GPLv3',
      url='https://github.com/zobristnicholas/superconductivity',
      packages=find_packages('src'),
      package_dir={'': 'src'},
      long_description=open('README.md').read(),
      install_requires=['numpy',
                        'scipy',
                        'numba'],
      classifiers=['Development Status :: 1 - Planning',
                   'Intended Audience :: Science/Research',
                   'License :: OSI Approved :: GNU General Public License v3 '
                   '(GPLv3)',
                   'Programming Language :: Python',
                   'Topic :: Scientific/Engineering :: Physics'],
      include_package_data=True,
      package_data={'': ['*.npz']},
      libraries=[('bvp_m-2', dict(sources=['external/bvp_m-2.f90'],
                                  extra_compile_args=["-std=f95"])),
                 ('bvp_la-2', dict(sources=['external/bvp_la-2.f'],
                                   extra_compile_args=["-std=legacy"])),
                 ('pchip', dict(sources=['external/pchip.f90'],
                                extra_compile_args=["-std=legacy"]))],
      ext_modules=[extension])
