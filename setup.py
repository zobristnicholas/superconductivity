from setuptools import setup, find_packages

version_number = '0.0'


setup(name='superconductivity',
      description='Tools for computing the properties of superconductors',
      version=version_number,
      author='Nicholas Zobrist',
      license='GPLv3',
      url='http://github.com/zobristnicholas/superconductivity',
      packages=find_packages(),
      long_description=open('README.rst').read(),
      install_requires=['numpy',
                        'scipy'],
      classifiers=['Development Status :: 1 - Planning',
                   'Intended Audience :: Science/Research',
                   'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
                   'Programming Language :: Python',
                   'Topic :: Scientific/Engineering :: Physics'])
