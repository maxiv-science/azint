# azint 

## Installation
``` bash
conda install -c maxiv azint
```

## Getting started
``` python
import fabio
import numpy as np
from azint import AzimuthalIntegrator

img = fabio.open('Eiger4M_Al2O3_13.45keV.edf').data
mask = fabio.open('mask.tif').data
ai = AzimuthalIntegrator('test.poni', mask.shape, 
                          75.0e-6, 4, [2000,], 
                          solid_angle=True) 
res = ai.integrate(img)
```
