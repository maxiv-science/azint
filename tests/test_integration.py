# ===========================================================================================
# For using azint.py in this repo instead of an installed azint conda package
# However, select the python interpreter from the azint conda package to fix module imports
#       ---> REMOVE THE FOLLOWING WHEN FINALIZING MERGE REQUEST TO MASTER BRANCH <---
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
print( f"[DEV LOG]: system path is {sys.path}" )
import importlib.util
azint_path = importlib.util.find_spec("azint").origin
print( f"[DEV LOG]: azint path is {azint_path}" )
# ===========================================================================================

import numpy as np
from azint import AzimuthalIntegrator

poni_file = 'tests/test.poni'
img = np.ones((1000, 1000), dtype=np.uint16)

def safe_divide(a, b):
    return np.divide(a, b, out=np.zeros_like(a), where=b!=0.0)

def test1d():
    ai = AzimuthalIntegrator(poni_file, 4, 1000, solid_angle=False) 
    I, error_1d, _, _ = ai.integrate(img)
    assert(np.allclose(I, 1.0))
    
    ai = AzimuthalIntegrator(poni_file, 1, 1000, error_model='poisson', solid_angle=False) 
    I, error_1d, _, _ = ai.integrate(img)
    assert(np.allclose(I, 1.0))
    
def test2d():
    ai = AzimuthalIntegrator(poni_file, 4, 1000, 360, solid_angle=False, normalized=False) 
    I, error_1d, I_2d, error_2d = ai.integrate(img)
    projection = safe_divide(np.sum(I_2d, axis=0), np.sum(ai.norm_2d, axis=0))
    assert(np.allclose(projection, 1.0))
    
def test_custom_ranges():
    q = [0.0, 0.1, 0.2, 0.3, 0.4]
    phi = [0.0, 90.0, 180.0, 270.0, 360]
    ai = AzimuthalIntegrator(poni_file, 4, q, phi, solid_angle=False)
    q = np.array(q)
    phi = np.array(phi)
    assert(np.allclose(0.5*(q[1:] + q[:-1]), ai.radial_axis))
    assert(np.allclose(0.5*(phi[1:] + phi[:-1]), ai.azimuth_axis))