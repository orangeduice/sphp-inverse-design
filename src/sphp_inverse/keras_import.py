"""Load weights exported from the Keras 2.15 reference (parity/export_keras_reference.py) into GeometryCVAE.

No TensorFlow needed: reads weights.npz + layers.json. Layers are matched by *creation order* (the numeric
suffix Keras gives each layer name), per layer type, which mirrors the module creation order in models/cvae.py.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .models.cvae import GeometryCVAE


def _suffix(name):
    m = re.search(r"_(\d+)$", name)
    return int(m.group(1)) if m else 0


def _keras_layers(rows, cls):
    """Layers of one class in creation order. Auto-named ones (conv1d_17) are sorted by suffix; explicitly
    named ones (e.g. 'design_out') keep their topological position, which for this model is last."""
    auto = re.compile(rf"^{ {'Conv1D': 'conv1d', 'BatchNormalization': 'batch_normalization', 'Dense': 'dense'}[cls]}(_\d+)?$")
    sel = [r for r in rows if r["class"] == cls]
    named = [r for r in sel if auto.match(r["name"])]
    other = [r for r in sel if not auto.match(r["name"])]
    return sorted(named, key=lambda r: _suffix(r["name"])) + other


def _torch_modules(module, cls):
    return [m for m in module.modules() if isinstance(m, cls)]


def load_keras_weights(model: GeometryCVAE, ref_dir: str | Path) -> dict:
    ref_dir = Path(ref_dir)
    w = np.load(ref_dir / "weights.npz")
    table = json.load(open(ref_dir / "layers.json"))
    report = {}
    with torch.no_grad():
        for part in ("encoder", "decoder"):
            rows, tmod = table[part], getattr(model, part)
            pairs = [("Conv1D", nn.Conv1d), ("BatchNormalization", nn.BatchNorm1d), ("Dense", nn.Linear)]
            for kcls, tcls in pairs:
                k_layers, t_layers = _keras_layers(rows, kcls), _torch_modules(tmod, tcls)
                if len(k_layers) != len(t_layers):
                    raise ValueError(f"{part}: {len(k_layers)} Keras {kcls} vs {len(t_layers)} torch {tcls.__name__}")
                for kl, tl in zip(k_layers, t_layers):
                    arrs = [w[f"{part}/{kl['name']}/{i}"] for i in range(len(kl["weight_shapes"]))]
                    if kcls == "Conv1D":      # Keras (k, in, out) -> torch (out, in, k)
                        _copy(tl.weight, arrs[0].transpose(2, 1, 0), kl)
                        if tl.bias is not None:
                            _copy(tl.bias, arrs[1], kl)
                        elif len(arrs) > 1:
                            raise ValueError(f"{kl['name']} has a bias but torch layer does not")
                    elif kcls == "Dense":     # Keras (in, out) -> torch (out, in)
                        _copy(tl.weight, arrs[0].T, kl)
                        _copy(tl.bias, arrs[1], kl)
                    else:                     # gamma, beta, moving_mean, moving_variance
                        _copy(tl.weight, arrs[0], kl)
                        _copy(tl.bias, arrs[1], kl)
                        _copy(tl.running_mean, arrs[2], kl)
                        _copy(tl.running_var, arrs[3], kl)
                report[f"{part}/{kcls}"] = len(k_layers)
    return report


def _copy(param, arr, layer):
    if tuple(param.shape) != tuple(arr.shape):
        raise ValueError(f"shape mismatch for Keras layer {layer['name']}: torch {tuple(param.shape)} vs {arr.shape}")
    param.copy_(torch.from_numpy(np.ascontiguousarray(arr)))
