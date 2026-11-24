"""Model selection without COMSOL: score a trained cVAE's *sampled designs* with the forward surrogate.

    uv run python -m sphp_inverse.selection --surrogate <dir> --data <h5> --runs <run dir> [<run dir> ...] --out <csv>

For each model and each latent prior it reports, over two target sets:
  val      300 validation-split spectra (in distribution, never used for training)
  ood_val  12 Gaussian dips (FWHM 0.3 THz, depth 0.7) that are NOT the dataset's spectra and NOT the final OOD
           test targets (those are FIG3 dips: FWHM 0.2 THz, depth 0.8)
  best_mse   best-of-n (n = 5) MSE between surrogate-predicted spectrum and target (OOD: inside 25.875-27.725 THz)
  mean_mse   mean over the n designs (how good a *typical* sample is)
  in_bounds  fraction of designs inside the training parameter box;  gap_ok  fraction with pillar gap >= 0.5 um

Latent priors:
  normal     z ~ N(0, I) (what the reference workflow does)
  aggregate  z ~ N(m, C): mean and covariance of the encoder's posterior over the training set. Uses the same
             standard-normal draws, mapped through the Cholesky factor, so priors are compared on equal footing.

The test split and the final OOD targets are never touched here.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from . import geometry
from .data import load_dataset, reference_split
from .inference import load_model
from .surrogate import load_surrogate, predict_physical
from .targets import DipTargetSet

OOD_VAL = DipTargetSet(fwhm=0.3, depth=0.7, n=12)
N_VAL, N_SAMPLES = 300, 5


@torch.no_grad()
def aggregate_posterior(model, geo_scaled, R, batch_size=1024):
    """Mean and covariance of q(z) = average over training samples of q(z | x, y)."""
    mus, vars_ = [], []
    for i in range(0, len(R), batch_size):
        mu, logvar = model.encoder(torch.as_tensor(geo_scaled[i:i + batch_size]), torch.as_tensor(R[i:i + batch_size]))
        mus.append(mu.numpy()), vars_.append(logvar.exp().numpy())
    mu, var = np.concatenate(mus), np.concatenate(vars_)
    return mu.mean(0), np.cov(mu, rowvar=False) + np.diag(var.mean(0))


def evaluate_model(run_dir, surrogate, ds, split, n_val=N_VAL, n=N_SAMPLES, seed=0):
    model, scaler = load_model(run_dir)
    model = model.cpu().float().eval()
    surr, s_mean, s_scale = surrogate
    geo_s = scaler.transform(ds.geo).astype(np.float32)
    m, C = aggregate_posterior(model, geo_s[split.train], ds.R[split.train])
    L = np.linalg.cholesky(C + 1e-9 * np.eye(len(m)))

    rng = np.random.default_rng(seed)
    val_rows = rng.choice(split.val, size=min(n_val, len(split.val)), replace=False)
    window = OOD_VAL.window(ds.freq)
    target_sets = {"val": (ds.R[val_rows], None), "ood_val": (OOD_VAL.spectra(ds.freq).astype(np.float32), window)}
    eps = np.random.default_rng(seed + 1).standard_normal((max(len(v[0]) for v in target_sets.values()), n, model.cfg.latent_dim))

    out = {"run": str(run_dir), "latent_mean_norm": float(np.linalg.norm(m)),
           "latent_cov_trace": float(np.trace(C)), "latent_dim": model.cfg.latent_dim}
    for prior in ("normal", "aggregate"):
        for name, (targets, mask) in target_sets.items():
            T = len(targets)
            z = eps[:T] if prior == "normal" else m + eps[:T] @ L.T
            with torch.no_grad():
                y = torch.as_tensor(np.repeat(targets, n, axis=0))
                designs = model.decoder(torch.as_tensor(z.reshape(T * n, -1), dtype=torch.float32), y).numpy()
            phys = scaler.inverse_transform(designs)
            pred = predict_physical(surr, s_mean, s_scale, phys).reshape(T, n, -1)
            diff = pred - targets[:, None, :]
            if mask is not None:
                diff = diff[..., mask]
            mse = (diff ** 2).mean(-1)
            flags = geometry.check(phys)
            key = f"{name}_{prior}"
            out[f"{key}_best_mse"] = float(mse.min(1).mean())
            out[f"{key}_mean_mse"] = float(mse.mean())
            out[f"{key}_in_bounds"] = float(flags["in_bounds"].mean())
            out[f"{key}_gap_ok"] = float(flags["gap_ok"].mean())
    metrics = json.load(open(Path(run_dir) / "metrics.json")) if (Path(run_dir) / "metrics.json").exists() else {}
    out["test_recon_zmu"] = metrics.get("test_recon_zmu")
    out["best_val_loss"] = metrics.get("best_val_loss")
    json.dump({"latent_mean": m.tolist(), "latent_cov": C.tolist()}, open(Path(run_dir) / "latent_prior.json", "w"))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--surrogate", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", default=None, help="CSV with one row per run")
    a = ap.parse_args()
    ds = load_dataset(a.data)
    split = reference_split(len(ds.geo))
    surrogate = load_surrogate(a.surrogate)
    rows = []
    for r in a.runs:
        if not (Path(r) / "model.pt").exists():
            continue
        row = evaluate_model(r, surrogate, ds, split)
        rows.append(row)
        print(f"{Path(r).name:36s} val best/mean {row['val_normal_best_mse']:.4f}/{row['val_normal_mean_mse']:.4f} "
              f"(agg {row['val_aggregate_best_mse']:.4f}/{row['val_aggregate_mean_mse']:.4f})  "
              f"ood best {row['ood_val_normal_best_mse']:.4f} (agg {row['ood_val_aggregate_best_mse']:.4f})  "
              f"recon {row['test_recon_zmu']:.4f}", flush=True)
    if a.out and rows:
        with open(a.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
