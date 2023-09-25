import sys
from setuptools import setup, find_packages, Extension
from pybind11.setup_helpers import Pybind11Extension

if sys.platform == 'win32':
    compile_args = ['/std:c++17', '/openmp']
    link_ars = ['/openmp']
else:
    compile_args = ['-std=c++17', '-fopenmp']
    link_ars = ['-fopenmp']

ext_modules = [
    Pybind11Extension(
        '_azint',
        ['azint.cpp'],
        extra_compile_args=compile_args,
        extra_link_args=link_ars
    ),
]

setup(
    name = 'azint',
    description = 'Azimthual Integration',
    ext_modules = ext_modules,
    packages=find_packages(),
    package_data={'azint': ['benchmark/bench.poni']},
) 
