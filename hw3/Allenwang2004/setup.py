from setuptools import setup, Extension
import pybind11

module = Extension('_matrix', 
                   sources=['matrix.cpp'],
                   include_dirs=[pybind11.get_include()],
                   extra_compile_args=['-std=c++17', '-O3', '-march=native', '-ffast-math', '-DNDEBUG'])
setup(name='matrix',
      version='1.0',
      description='Matrix multiplication module',
      ext_modules=[module])