"""Design targets: Gaussian (or super-Gaussian) reflection dips.

`FIG3` is the target set behind report Fig. 3: 18 Gaussian dips (FWHM 0.2 THz, depth 0.8, baseline 0.95) with centres
spaced evenly from 25.975 to 27.625 THz. Matching is scored only inside 25.875-27.725 THz (indices 35:110 of the 161-point
25-29 THz grid). These settings come from the archived notebook data_looker.ipynb, cell 12, and reproduce the report.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


def gaussian_dip(freq, centre, fwhm, depth, order=2.0, baseline=1.0):
    """baseline - depth * exp(-|(f - centre) / (sqrt(2) sigma)|^order), with sigma taken from the FWHM of a Gaussian.

    order=2 is a Gaussian; higher orders give a flatter bottom (port of create_reflectance_target)."""
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    return baseline - depth * np.exp(-(np.abs((np.asarray(freq) - centre) / (np.sqrt(2) * sigma))) ** order)


@dataclass(frozen=True)
class DipTargetSet:
    fwhm: float = 0.2          # THz
    depth: float = 0.8
    baseline: float = 0.95     # the original target was gaussian_dip(...) - 0.05
    order: float = 2.0
    f_lo: float = 25.875       # THz, edges of the scoring window
    f_hi: float = 27.725
    margin: float = 0.1        # THz between the window edges and the first/last centre
    n: int = 18
    region_level: float = 0.9  # the dip of a matched spectrum is searched where target < region_level

    def centres(self) -> np.ndarray:
        return np.linspace(self.f_lo + self.margin, self.f_hi - self.margin, self.n)

    def spectra(self, freq) -> np.ndarray:
        """(n, F) target spectra on `freq`."""
        return np.stack([gaussian_dip(freq, c, self.fwhm, self.depth, self.order, self.baseline)
                         for c in self.centres()])

    def window(self, freq) -> np.ndarray:
        """Boolean mask of the scoring window (tolerant to float32 frequencies)."""
        freq = np.asarray(freq)
        tol = 1e-4
        return (freq >= self.f_lo - tol) & (freq <= self.f_hi + tol)

    def to_dict(self) -> dict:
        return asdict(self)


FIG3 = DipTargetSet()
TARGET_SETS = {"fig3": FIG3}
