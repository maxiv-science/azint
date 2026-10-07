"""Minimal pytest-benchmark for CI history (github-action-benchmark)."""
from pathlib import Path

import numpy as np

from azint import AzimuthalIntegrator

_PONI = Path(__file__).resolve().parents[1] / "azint" / "benchmark" / "bench.poni"


def test_eiger4m_integrate(benchmark):
    ai = AzimuthalIntegrator(str(_PONI), 4, 2000, normalized=False)
    img = np.ones((2167, 2070), dtype=np.uint32)
    benchmark(ai.integrate, img)
