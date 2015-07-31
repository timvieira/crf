#import numpy as np
from setuptools import setup
#from Cython.Build import cythonize

setup(name='crf',
      author='Tim Vieira',
      description='Simple implementation of a linear chain conditional random field.',
      version='1.0',
      install_requires=[],
      packages=['crf'],
#      include_dirs=[np.get_include()],
#      ext_modules = cythonize(['transdeuce/**/*.pyx'])
)
