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

# ===========================================================================================
# Dev playground with data from:
#
# PONI: /data/visitors/danmax/20250946/2025052008/process/setup_Si_135mm/Si_135mm.poni
# raw: /data/visitors/danmax/20250946/2025052008/raw/Si_135mm/scan-1737_pilatus.h5
# mask: /data/visitors/danmax/20250946/2025052008/process/hot_px_bs_mask.npy
# config: /data/visitors/danmax/20250946/2025052008/process/config1.json
#
#       ---> REMOVE THE FOLLOWING WHEN FINALIZING MERGE REQUEST TO MASTER BRANCH <---
import h5py
import matplotlib.pyplot as plt
from azint import Poni

h5name = "tests/scan-1737_pilatus.h5"
h5file = h5py.File(h5name, 'r')
img    = h5file['/entry/instrument/pilatus/data'][10]
poni   = 'tests/Si_135mm.poni'
mask   = 'tests/hot_px_bs_mask.npy'

if isinstance(poni, str):
    poni_class_instance = Poni.from_file(poni)
        
if isinstance(poni, dict):
    poni_class_instance = Poni.from_dict(poni)

pixel_width  = poni_class_instance.det.pixel1
pixel_height = poni_class_instance.det.pixel2 

poni_x = poni_class_instance.poni2 / pixel_width  # [m] -> [px]
poni_y = poni_class_instance.poni1 / pixel_height # [m] -> [px]
poni_point = ( poni_x, poni_y )

# Some basic plotting ---
fig, ax = plt.subplots(1,2)
ax[0].imshow(img, vmin=0, vmax=1200)
ax[0].plot( poni_x, poni_y, "x", color="tab:orange" )

mask_data = np.load(mask)
masked_img = img*(1-mask_data) # as done in azint.py
ax[1].imshow(masked_img, vmin=0, vmax=1200)
ax[1].plot( poni_x, poni_y, "x", color="tab:orange" )

plt.show()

# Setting up the integrator ---
config = {
    'poni': poni,                  # Path to the PONI file
    'mask': mask,                  # Mask to ignore bad pixels
    'radial_bins': 3000,           # Number of radial bins for integration
    'azimuth_bins': 180, # or None # Number of azimuthal bins (for 2D integration)
    'n_splitting': 21,             # Number of subdivisions per pixel (for precision)
    'error_model': 'poisson',      # Error propagation model
    'solid_angle': True,           # Apply solid angle correction
    'polarization_factor': 0.965,  # Correction for polarization effects
    'normalized': True,            # Normalize the output intensities
    'unit': '2th',   # or q        # Output units (e.g., 2θ, q)
}

ai = AzimuthalIntegrator(**config)

# ===========================================================================================
