"""Spectral metrics shared by the brute-force baseline and re-simulation scoring.

Dip metrics follow the original Fig. 3 analysis: the dip is the minimum of R inside a search region, and its
prominence and full width at half maximum (FWHM) come from scipy's peak_prominences / peak_widths on 1 - R
(rel_height = 0.5, i.e. half of the prominence).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import peak_prominences, peak_widths

THZ_TO_CM = 33.35641  # 1 THz in cm^-1


@dataclass
class DipMetrics:
    index: int            # grid index of the dip
    freq: float           # THz
    fwhm: float           # THz
    prominence: float     # in reflectance units
    half_height: float    # reflectance level at which the FWHM is measured
    left: float           # THz, interpolated half-maximum crossings
    right: float

    @property
    def fwhm_cm(self) -> float:
        return self.fwhm * THZ_TO_CM


def dip_metrics(freq, R, region=None) -> DipMetrics:
    """Metrics of the deepest dip of R inside `region` (boolean mask; None or empty = whole spectrum).

    R keeps its dtype (float32 dataset spectra give exactly the numbers in the report)."""
    freq, R = np.asarray(freq), np.asarray(R)
    idx = np.arange(len(R)) if region is None or not np.any(region) else np.flatnonzero(region)
    pk = int(idx[np.argmin(R[idx])])
    inv = 1 - R
    prom = peak_prominences(inv, [pk])
    widths, heights, left_ips, right_ips = peak_widths(inv, [pk], rel_height=0.5, prominence_data=prom)
    step = freq[1] - freq[0]
    grid = np.arange(len(freq))
    return DipMetrics(index=pk, freq=float(freq[pk]), fwhm=float(widths[0] * step), prominence=float(prom[0][0]),
                      half_height=float(1 - heights[0]), left=float(np.interp(left_ips[0], grid, freq)),
                      right=float(np.interp(right_ips[0], grid, freq)))


def mse(a, b, mask=None) -> np.ndarray:
    """Mean squared error along the last axis, optionally only where `mask` is True."""
    a, b = np.asarray(a), np.asarray(b)
    if mask is not None:
        a, b = a[..., mask], b[..., mask]
    return np.mean((a - b) ** 2, axis=-1)
