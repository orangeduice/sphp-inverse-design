"""Dataset loading, scaling and splitting.

`reference_split` reproduces exactly what the working Keras notebook did
(helper.load_data_from_h5 + Keras `validation_split`), so PyTorch results can be compared with it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

GEO_NAMES = ("h_pill", "d_pill", "w_pill", "d_sep", "w_sep", "angle")
GEO_UNITS = ("um", "um", "um", "um", "um", "rad")


@dataclass
class Dataset:
    geo: np.ndarray    # (N, 6) physical units
    R: np.ndarray      # (N, F) reflection spectra
    freq: np.ndarray   # (F,) THz (stored as 'wavelengths' in the h5)


def load_dataset(path: str | Path) -> Dataset:
    with h5py.File(path, "r") as f:
        return Dataset(
            geo=np.asarray(f["geo"], dtype=np.float32),
            R=np.asarray(f["R"], dtype=np.float32),
            freq=np.asarray(f["wavelengths"], dtype=np.float32),
        )


@dataclass
class Split:
    train: np.ndarray  # row indices into the dataset
    val: np.ndarray
    test: np.ndarray


def reference_split(n: int, test_fraction: float = 0.05, val_fraction: float = 0.3,
                    random_state: int = 42) -> Split:
    """Index split identical to the Keras reference.

    1. sklearn `train_test_split(test_size=test_fraction, random_state=random_state)`. The shuffle depends only on
       n and the seed, so splitting indices gives the same permutation as splitting the arrays.
    2. Keras `fit(validation_split=v)` keeps the *last* fraction of the (already shuffled) training arrays as
       validation, without reshuffling, using split_at = floor(n_train * (1 - v)).
    """
    idx = np.arange(n)
    trainval, test = train_test_split(idx, test_size=test_fraction, random_state=random_state)
    split_at = int(math.floor(len(trainval) * (1.0 - val_fraction)))
    return Split(train=trainval[:split_at], val=trainval[split_at:], test=test)


def fit_geo_scaler(geo: np.ndarray, rows: np.ndarray | None = None) -> StandardScaler:
    """StandardScaler for the geometry. The reference fits it on *all* rows (rows=None)."""
    return StandardScaler().fit(geo if rows is None else geo[rows])


def ensure_geometry_columns(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim != 2 or arr.shape[1] != len(GEO_NAMES):
        raise ValueError(f"expected (N, {len(GEO_NAMES)}) geometry array, got {arr.shape}")
    return arr
