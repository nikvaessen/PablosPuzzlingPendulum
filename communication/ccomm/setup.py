#!/usr/bin/python3
from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension

# compile with: python3 setup_test.py build_ext --inplace
# clean dir with: rm -r build ClassWrapper.cpp ClassWrapper.cpython-35m-x86_64-linux-gnu.so
ext=[Extension('*',
			sources=['cppcomm.pyx', 'comm.cpp'],
			library_dirs=['/usr/local/lib/'],
			libraries=['serial', 'pthread'],
			language='c++')]

setup(ext_modules=cythonize(ext))

