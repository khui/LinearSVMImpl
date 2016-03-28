from distutils.core import setup
from Cython.Build import cythonize

setup(
        name = "lossfuncs",
        ext_modules = cythonize('lossfuncs.pyx'),  # accepts a glob pattern
)
