import sys

from pybind11.setup_helpers import Pybind11Extension
from setuptools import setup

if sys.platform == 'win32':
    compile_args = ['/std:c++17', '/openmp']
    link_args = ['/openmp']
else:
    compile_args = ['-std=c++17', '-fopenmp']
    link_args = ['-fopenmp']

# See https://conda-forge.org/docs/maintainer/knowledge_base.html#newer-c-features-with-old-sdk
if sys.platform == 'darwin':
    compile_args.append('-D_LIBCPP_DISABLE_AVAILABILITY')

setup(
    ext_modules=[
        Pybind11Extension(
            '_azint',
            ['azint.cpp'],
            depends=['azint.hpp'],
            extra_compile_args=compile_args,
            extra_link_args=link_args,
        ),
    ],
)
