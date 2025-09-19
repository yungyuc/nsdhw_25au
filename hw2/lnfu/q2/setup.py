import sys

from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext
from pybind11 import get_cmake_dir

__version__ = "0.0.1"

ext_modules = [
    Pybind11Extension(
        "_vector",
        ["vector.cpp"],
        define_macros=[("VERSION_INFO", '"{}"'.format(__version__))],
    ),
]

setup(
    name="_vector",
    version=__version__,
    author="Enfu Liao",
    author_email="enfu.liao.cs14@nycu.edu.tw",
    ext_modules=ext_modules,
    extras_require={"test": "pytest"},
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)
