"""Micropillar geometry: derived quantities and validity checks.

Columns: h_pill, d_pill, w_pill, d_sep, w_sep (um) and angle (rad).
"""
from __future__ import annotations

import numpy as np

# Sampling bounds used to build the dataset (MINAS_param_maker.py)
BOUNDS = {
    "h_pill": (0.5, 1.0), "d_pill": (0.5, 2.0), "w_pill": (0.5, 2.0),
    "d_sep": (0.25, 2.0), "w_sep": (0.25, 2.0), "angle": (0.0, np.pi / 4),
}
MAX_ASPECT = 2.0
MIN_GAP = 0.5  # um, intended fabrication limit


def derived(geo: np.ndarray) -> dict:
    """Unit cell, aspect ratios and pillar gaps for an (N, 6) array of physical geometries.

    The gap uses the angle in radians. The original sampler applied deg2rad to an angle that was already in
    radians, so 2.5 % of the dataset has a gap below MIN_GAP (minimum 0.37 um); see data/DATA.md.
    """
    h, d, w, d_sep, w_sep, th = np.asarray(geo, dtype=float).T
    w_sub = np.cos(np.pi / 2 - th) * d + np.cos(th) * w + w_sep
    d_sub = np.cos(np.pi / 2 - th) * w + np.cos(th) * d + d_sep
    return {
        "w_sub": w_sub, "d_sub": d_sub,
        "w_aspect": h / w, "d_aspect": h / d,
        "d_gap": np.cos(th) * (d_sub - d), "w_gap": np.cos(th) * (w_sub - w),
    }


def check(geo: np.ndarray, min_gap: float = MIN_GAP, tol: float = 0.0) -> dict:
    """Per-design flags. `in_bounds`: inside the dataset's sampling box (the model was never trained outside it)."""
    geo = np.asarray(geo, dtype=float)
    lo = np.array([b[0] for b in BOUNDS.values()]) - tol
    hi = np.array([b[1] for b in BOUNDS.values()]) + tol
    q = derived(geo)
    return {
        "in_bounds": np.all((geo >= lo) & (geo <= hi), axis=1),
        "aspect_ok": (q["w_aspect"] <= MAX_ASPECT) & (q["d_aspect"] <= MAX_ASPECT),
        "gap_ok": (q["d_gap"] >= min_gap) & (q["w_gap"] >= min_gap),
        "positive": np.all(geo[:, :5] > 0, axis=1),
        "min_gap": np.minimum(q["d_gap"], q["w_gap"]),
    }
