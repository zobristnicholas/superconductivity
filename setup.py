from setuptools import setup, find_packages

setup(name='superconductivity',
      description='tools for computing the properties of superconductors',
      version=version_number,
      author='Nicholas Zobrist',
      license='GPLv3',
      url='http://github.com/zobristnicholas/superconductivity',
      packages=find_packages(),
      long_description=open('README.rst').read(),
      install_requires=['numpy',
                        'scipy',
                        'numba'],
      classifiers=['Development Status :: 1 - Planning',
                   'Intended Audience :: Science/Research',
                   'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
                   'Programming Language :: Python',
                   'Topic :: Scientific/Engineering :: Physics'])
