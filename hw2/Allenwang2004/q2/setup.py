from setuptools import setup, Extension
import pybind11

module = Extension('_vector', 
                   sources=['vector.cpp'],
                   include_dirs=[pybind11.get_include()], 
                   language='c++',
                   extra_compile_args=['-std=c++17']
                   )

setup(
    name='_vector',
    version='0.1',
    ext_modules=[module],
)