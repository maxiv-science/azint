# azint: Azimuthal Integration

azint is a python library for azimuthal integration of area detectors. The azimuthal integration is transformed into a sparse matrix vector multiplication for performance. Pixel splitting is done by subdividing each pixels into subpixels and assigning bins and weights for the individual subpixels. This can increase the precision of the transformation but also introduces correlation between neighboring bins.


---
**Documentation**: <a href="https://maxiv-science.github.io/azint" target="_blank">https://maxiv-science.github.io/azint</a>

---
## Installation
``` bash
conda install -c maxiv azint
```