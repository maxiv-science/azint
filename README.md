# azint: Azimuthal Integration

[![DOI](https://zenodo.org/badge/315677387.svg)](https://doi.org/10.5281/zenodo.16480726)

**azint** is a Python library for efficient azimuthal integration of area detector data.  
The integration process is transformed into a sparse matrix–vector multiplication for performance.  

For increased precision, pixel splitting is done by subdividing each pixel into subpixels, assigning bins and weights to each.  
This improves the accuracy of the transformation but may introduce correlation between neighboring bins.

---

## 📖 Documentation

Full documentation is available at:  
👉 [https://maxiv-science.github.io/azint](https://maxiv-science.github.io/azint)

---

## 💾 Installation

Install via `conda` from the MAX IV channel:

```bash
conda install -c maxiv azint
```

---
## 📤 NXazint Output

To write azimuthal integration results in the [NeXus](https://www.nexusformat.org/) data format,  
you can use the [`azint_writer`](https://github.com/maxiv-science/azint_writer) package in combination with `azint`.

This enables standardized output for integration results, compatible with NeXus-based data workflows and analysis tools.
