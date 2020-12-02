import numpy as np
from setuptools import setup, find_packages, Extension

extentions = Extension('_azint',
                    language='c++',
                    sources = ['azint.cpp'],
                    include_dirs=[np.get_include()])

setup(
    name = 'azint',
    version = '0.1',
    description = 'Azimthual Integration',
    ext_modules = [extentions],
    packages=find_packages(),
) 
