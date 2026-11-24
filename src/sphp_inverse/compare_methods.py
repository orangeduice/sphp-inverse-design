"""Compare design *methods* on validation targets, judged by an independent surrogate.

    uv run python -m sphp_inverse.compare_methods --model <run dir> --chooser <surrogate A> --judge <surrogate B> --data <h5>

Methods (all return 5 designs per target):
  lookup        the 5 training-set geometries whose (COMSOL) spectra best match the target: the dataset baseline
  cvae          5 decoder samples, z ~ N(0, I) (the reference workflow)
  cvae_agg      5 decoder samples, z from the fitted aggregate posterior
  filter        2000 aggregate-prior samples, best 5 distinct valid designs by the *chooser* surrogate
  filter+refine as filter, plus 100 gradient steps on z through decoder + chooser surrogate
  nn-cond       as filter, but decoding with the 20 nearest training spectra as conditions (100 z each)
  nn-cond+refine

Scores use the *judge* surrogate (different architecture and seed), except lookup, which uses the dataset's own
COMSOL spectra. Targets: 50 validation spectra and the 12 OOD-validation dips (see selection.py). The test split
and the final OOD targets are not used.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from . import geometry
from .data import load_dataset, reference_split
from .designer import DesignConfig, design
from .inference import load_model
from .selection import OOD_VAL, aggregate_posterior
from .surrogate import load_surrogate, predict_physical


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--chooser", required=True)
    ap.add_argument("--judge", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--n-val", type=int, default=50)
    ap.add_argument("--methods", nargs="+", default=["lookup", "cvae", "cvae_agg", "filter", "filter+refine"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ds = load_dataset(a.data)
    split = reference_split(len(ds.geo))
    model, scaler = load_model(a.model)
    model = model.float().to(dev).eval()
    chooser = load_surrogate(a.chooser, device=dev)
    judge = load_surrogate(a.judge)
    geo_s = scaler.transform(ds.geo).astype(np.float32)
    lat = aggregate_posterior(model.cpu(), geo_s[split.train], ds.R[split.train])
    model.to(dev)

    rng = np.random.default_rng(0)
    val_rows = rng.choice(split.val, a.n_val, replace=False)
    window = OOD_VAL.window(ds.freq)
    sets = {"val": (ds.R[val_rows], None), "ood_val": (OOD_VAL.spectra(ds.freq).astype(np.float32), window)}
    R_train, geo_train = ds.R[split.train], ds.geo[split.train]

    results = {}
    for method in a.methods:
        for name, (targets, mask) in sets.items():
            best, mean, gap_ok, t0 = [], [], [], time.time()
            for i, t in enumerate(targets):
                sel = slice(None) if mask is None else mask
                if method == "lookup":
                    e = ((R_train[:, sel] - t[sel]) ** 2).mean(1)
                    top = np.argsort(e)[:5]
                    errs, phys = e[top], geo_train[top]
                else:
                    if method in ("cvae", "cvae_agg"):
                        cfg = DesignConfig(n_candidates=5, prior="normal" if method == "cvae" else "aggregate",
                                           min_distance=0.0, require_gap=False, seed=1000 + i)
                    elif method in ("filter", "filter+refine", "nn-cond", "nn-cond+refine"):
                        cfg = DesignConfig(n_candidates=2000, prior="aggregate", seed=1000 + i,
                                           refine_steps=100 if "refine" in method else 0,
                                           n_conditions=20 if method.startswith("nn-cond") else 0)
                    else:
                        raise ValueError(f"unknown method {method}")
                    g, _, _, _ = design(model, chooser, scaler, t, mask, cfg, lat, reference_spectra=R_train)
                    phys = scaler.inverse_transform(g)
                    pred = predict_physical(*judge, phys)
                    errs = ((pred[:, sel] - t[sel]) ** 2).mean(1)
                best.append(errs.min()), mean.append(errs.mean())
                gap_ok.append(geometry.check(phys)["gap_ok"].mean())
            results[f"{method}/{name}"] = {"best_mse": float(np.mean(best)), "mean_mse": float(np.mean(mean)),
                                           "gap_ok": float(np.mean(gap_ok)), "seconds": time.time() - t0}
            r = results[f"{method}/{name}"]
            print(f"{method:14s} {name:8s} best-of-5 {r['best_mse']:.4f}  mean {r['mean_mse']:.4f}  "
                  f"gap_ok {r['gap_ok']:.2f}  ({r['seconds']:.0f}s)", flush=True)
    if a.out:
        json.dump({"model": str(Path(a.model).resolve()), "results": results}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
