import numpy as np
from setuptools import setup, find_packages, Extension
from pybind11.setup_helpers import Pybind11Extension

ext_modules = [
    Pybind11Extension(
        'sparse',
        ['azint.cpp'],
    ),
]

setup(
    name = 'azint',
    version = '0.5.5',
    description = 'Azimthual Integration',
    ext_modules = ext_modules,
    packages=find_packages(),
) 
