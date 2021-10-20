import numpy as np
from azint import AzimuthalIntegrator

poni_file = 'tests/test.poni'
pixel_size = 1.0e-4
img = np.ones((1000, 1000), dtype=np.uint16)

def test1d():
    ai = AzimuthalIntegrator(poni_file, img.shape, pixel_size, 4, 1000, solid_angle=False) 
    res, _ = ai.integrate(img)
    assert(np.allclose(res, 1.0))
    
def test2d():
    ai = AzimuthalIntegrator(poni_file, img.shape, pixel_size, 4, 1000, 360, solid_angle=False) 
    res, _ = ai.integrate(img)
    # set emtpy bins to nan to conserve the mean
    res[res==0.0] = np.nan
    assert(np.allclose(np.nanmean(res, axis=0), 1.0))
