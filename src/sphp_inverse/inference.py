"""Inverse design: target spectrum -> candidate geometries -> files for COMSOL re-simulation.

    uv run python -m sphp_inverse.inference --model <run dir or Keras export dir> --data <h5> \
        --test-idx 138 169 190 --n 5 --out <dir> --category 5

Each target gets a folder  <out>/run_<ID>/  with the same files as the original Keras workflow:
  <ID>_params.csv                     physical designs, with the 2-line header COMSOL's set_params_call expects
  <ID>_suggested_designs.csv          scaled designs (decoder output)
  <ID>_suggested_designs_physical.csv
  <ID>_target_spectra.csv             frequency, target reflection
  <ID>_metadata.json                  model, z samples, validity flags
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from . import geometry
from .data import GEO_NAMES, GEO_UNITS, fit_geo_scaler, load_dataset, reference_split
from .ids import encode_id
from .keras_import import load_keras_weights
from .models.cvae import CVAEConfig, GeometryCVAE

PARAMS_HEADER = ", ".join(GEO_NAMES) + "\n" + ", ".join(GEO_UNITS)


def load_model(model_dir, device="cpu", n_blocks=None, n_freq=None):
    """Load a trained model. `model_dir` is either a PyTorch run folder (model.pt) or a Keras export folder
    (weights.npz from parity/export_keras_reference.py). Returns (model, scaler or None)."""
    model_dir = Path(model_dir)
    if (model_dir / "model.pt").exists():
        ck = torch.load(model_dir / "model.pt", map_location=device, weights_only=False)
        cfg = CVAEConfig(**{**ck["model_config"], "spec_filters": tuple(ck["model_config"]["spec_filters"])})
        model = GeometryCVAE(cfg)
        model.load_state_dict(ck["model"])
        scaler = StandardScaler()
        scaler.mean_, scaler.scale_ = np.asarray(ck["scaler_mean"]), np.asarray(ck["scaler_scale"])
        scaler.var_, scaler.n_features_in_ = scaler.scale_ ** 2, len(scaler.mean_)
    elif (model_dir / "weights.npz").exists():
        table = json.load(open(model_dir / "layers.json"))
        # each residual block with BatchNorm has 3 BN layers; the decoder has 5 block sequences
        n_bn = sum(r["class"] == "BatchNormalization" for r in table["decoder"])
        cfg = CVAEConfig(n_freq=n_freq or 161, n_blocks=n_blocks or n_bn // 15)
        model = GeometryCVAE(cfg)
        load_keras_weights(model, model_dir)
        scaler = None
    else:
        raise FileNotFoundError(f"no model.pt or weights.npz in {model_dir}")
    return model.to(device).eval(), scaler


@torch.no_grad()
def suggest_designs(model, target, n=5, z=None, seed=0):
    """Decode n latent samples with one target spectrum. Returns (scaled designs (n, 6), z (n, latent))."""
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    if z is None:
        z = np.random.default_rng(seed).standard_normal((n, model.cfg.latent_dim))
    z_t = torch.as_tensor(np.asarray(z), dtype=dtype, device=device)
    y = torch.as_tensor(np.asarray(target), dtype=dtype, device=device).expand(len(z_t), -1)
    return model.decoder(z_t, y).cpu().numpy(), np.asarray(z)


def save_inference_run(out_dir, run_id, designs_scaled, scaler, target, freq, meta):
    run = Path(out_dir) / f"run_{run_id}"
    run.mkdir(parents=True, exist_ok=True)
    phys = scaler.inverse_transform(designs_scaled)
    flags = geometry.check(phys)
    np.savetxt(run / f"{run_id}_suggested_designs.csv", designs_scaled, delimiter=",")
    np.savetxt(run / f"{run_id}_suggested_designs_physical.csv", phys, delimiter=",")
    np.savetxt(run / f"{run_id}_target_spectra.csv", np.column_stack([freq, target]), delimiter=",")
    np.savetxt(run / f"{run_id}_params.csv", phys, delimiter=",", header=PARAMS_HEADER, comments="")
    meta = {**meta, "inference_run_id": str(run_id),
            "checks": {k: np.asarray(v).tolist() for k, v in flags.items()}}
    json.dump(meta, open(run / f"{run_id}_metadata.json", "w"), indent=1)
    return run, flags


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--test-idx", type=int, nargs="+", required=True, help="targets = test-set spectra at these positions")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--category", type=int, default=5, help="ID category digit (keep one per model/experiment)")
    ap.add_argument("--z-from", default=None,
                    help="folder of earlier runs: reuse z_samples from run_*<idx>/..._metadata.json for the same test index")
    ap.add_argument("--freq-slice", type=int, nargs=2, default=None)
    ap.add_argument("--n-blocks", type=int, default=None, help="only for Keras exports")
    a = ap.parse_args()

    ds = load_dataset(a.data)
    R, freq = ds.R, ds.freq
    if a.freq_slice:
        R, freq = R[:, a.freq_slice[0]:a.freq_slice[1]], freq[a.freq_slice[0]:a.freq_slice[1]]
    split = reference_split(len(ds.geo))
    model, scaler = load_model(a.model, n_blocks=a.n_blocks, n_freq=R.shape[1])
    scaler = scaler or fit_geo_scaler(ds.geo)  # Keras reference: scaler fitted on all rows

    for idx in a.test_idx:
        target = R[split.test[idx]]
        z, z_source = None, f"default_rng({a.seed})"
        if a.z_from:
            hits = sorted(Path(a.z_from).glob(f"run_*/*_metadata.json"))
            hits = [h for h in hits if json.load(open(h)).get("target_id") == str(idx)]
            if not hits:
                raise FileNotFoundError(f"no earlier run for test index {idx} in {a.z_from}")
            z = np.array(json.load(open(hits[0]))["inference_params"]["z_samples"])
            z_source = f"reused from {hits[0].name}"
        designs, z = suggest_designs(model, target, a.n, z, a.seed)
        run_id = encode_id(a.category, idx)
        run, flags = save_inference_run(a.out, run_id, designs, scaler, target, freq, {
            "target_id": str(idx), "target_source": "test split (reference_split)", "dataset_row": int(split.test[idx]),
            "model": str(Path(a.model).resolve()),
            "inference_params": {"num_predictions": len(z), "z_source": z_source, "z_samples": z.tolist()}})
        print(f"run {run_id}  test idx {idx:3d}  in_bounds {int(flags['in_bounds'].sum())}/{len(z)}  "
              f"gap>=0.5 {int(flags['gap_ok'].sum())}/{len(z)}  min gap {flags['min_gap'].min():.2f} um  -> {run}")


if __name__ == "__main__":
    main()
