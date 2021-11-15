import sys
from setuptools import setup, find_packages, Extension
from pybind11.setup_helpers import Pybind11Extension

if sys.platform == 'win32':
    openmp_flag = '/openmp'
else:
    openmp_flag = '-fopenmp'

ext_modules = [
    Pybind11Extension(
        'sparse',
        ['azint.cpp'],
        extra_compile_args=[openmp_flag],
        extra_link_args=[openmp_flag]
    ),
]

setup(
    name = 'azint',
    version = '0.8.1',
    description = 'Azimthual Integration',
    ext_modules = ext_modules,
    packages=find_packages(),
) 
