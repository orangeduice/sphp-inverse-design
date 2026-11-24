"""Evaluate "match here, ignore elsewhere" design on Gaussian dip targets.

    uv run python -m sphp_inverse.masked_eval --models <run dir> [...] --judge <surrogate B> --chooser <surrogate A> \
        --data <h5> --out <json>

Target sets (Gaussian dips, baseline 0.95, depth 0.8 unless noted):
  local   FIG3 centres, care window = centre +/- 0.5 THz  (a 1 THz spectral filter around the dip)
  fig3    FIG3 centres, care window = 25.875-27.725 THz  (the report / brute-force window)
  oodval  FWHM 0.3 THz, depth 0.7, 12 other centres, local window  (used to *choose* between settings)

Per target and method we keep 5 designs and report, inside the care window (independent "judge" surrogate):
  best_mse       best-of-5 MSE against the target
  hit            deepest dip of the best design within 0.1 THz of the target centre
  isolated       best design has exactly one dip (prominence > 0.1) inside the care window
  fwhm_err_cm    |FWHM(best design) - FWHM(target)| in cm^-1

Methods: lookup (training spectra, true COMSOL), sample (5 decoder samples), guided (2000 samples + surrogate pick
+ 100 refinement steps; unmasked models also condition on the 20 nearest training spectra).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.signal import find_peaks

from .data import load_dataset, reference_split
from .designer import DesignConfig, design
from .inference import load_model
from .metrics import THZ_TO_CM, dip_metrics
from .selection import aggregate_posterior
from .surrogate import load_surrogate, predict_physical
from .targets import FIG3, DipTargetSet, gaussian_dip

LOCAL_HALF_WIDTH = 0.5  # THz


def target_sets(freq):
    f = np.asarray(freq)
    out = {}
    cen = FIG3.centres()
    T = FIG3.spectra(f).astype(np.float32)
    out["local"] = [(t, np.abs(f - c) <= LOCAL_HALF_WIDTH + 1e-4, c, FIG3.fwhm) for t, c in zip(T, cen)]
    out["fig3"] = [(t, FIG3.window(f), c, FIG3.fwhm) for t, c in zip(T, cen)]
    ov = DipTargetSet(fwhm=0.3, depth=0.7, n=12, margin=0.15)
    out["oodval"] = [(t, np.abs(f - c) <= LOCAL_HALF_WIDTH + 1e-4, c, ov.fwhm)
                     for t, c in zip(ov.spectra(f).astype(np.float32), ov.centres())]
    return out


def score_designs(spectra, target, mask, centre, fwhm, freq):
    """Metrics of the best (lowest in-window MSE) of the given spectra."""
    e = ((spectra[:, mask] - target[mask]) ** 2).mean(1)
    b = int(np.argmin(e))
    best = spectra[b]
    dm = dip_metrics(freq, best, mask)
    n_dips = len(find_peaks(1 - best[mask], prominence=0.1)[0])
    hits = [abs(dip_metrics(freq, s, mask).freq - centre) <= 0.1 for s in spectra]
    return {"best_mse": float(e[b]), "mean_mse": float(e.mean()), "hit": bool(hits[b]), "hit_any": float(np.mean(hits)),
            "isolated": n_dips == 1, "n_dips": n_dips, "fwhm_err_cm": abs(dm.fwhm - fwhm) * THZ_TO_CM}


def summarise(rows):
    return {"best_mse": float(np.mean([r["best_mse"] for r in rows])),
            "mean_mse": float(np.mean([r["mean_mse"] for r in rows])),
            "hit": float(np.mean([r["hit"] for r in rows])), "isolated": float(np.mean([r["isolated"] for r in rows])),
            "fwhm_err_cm": float(np.median([r["fwhm_err_cm"] for r in rows])), "n": len(rows)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--judge", required=True)
    ap.add_argument("--chooser", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--sets", nargs="+", default=["oodval", "local", "fig3"])
    ap.add_argument("--methods", nargs="+", default=["sample", "guided"])
    ap.add_argument("--masked-nn-cond", action="store_true",
                    help="masked models also condition on the 20 nearest training spectra (masked to the care window)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ds = load_dataset(a.data)
    split = reference_split(len(ds.geo))
    F, Rtr = ds.freq, ds.R[split.train]
    judge, chooser = load_surrogate(a.judge), load_surrogate(a.chooser, device=dev)
    sets = {k: v for k, v in target_sets(F).items() if k in a.sets}
    results = {}

    # lookup baseline: top-5 training spectra by in-window MSE, judged on their true (COMSOL) spectra
    for sname, targets in sets.items():
        rows = []
        for t, m, c, fw in targets:
            top = np.argsort(((Rtr[:, m] - t[m]) ** 2).mean(1))[:5]
            rows.append(score_designs(Rtr[top], t, m, c, fw, F))
        results[f"lookup/{sname}"] = summarise(rows)
        print(f"{'lookup':34s} {sname:7s} {json.dumps({k: round(v, 4) for k, v in results[f'lookup/{sname}'].items()})}", flush=True)

    for mdir in a.models:
        model, scaler = load_model(mdir)
        model = model.float().cpu().eval()
        geo_s = scaler.transform(ds.geo).astype(np.float32)
        latent = aggregate_posterior(model, geo_s[split.train], Rtr)
        masked = getattr(model.cfg, "cond_mask", False)
        model = model.to(dev)
        name = Path(mdir).name
        for method in a.methods:
            for sname, targets in sets.items():
                rows = []
                for i, (t, m, c, fw) in enumerate(targets):
                    if method == "sample":
                        cfg = DesignConfig(n_candidates=5, prior="normal", min_distance=0.0, require_gap=False, seed=100 + i)
                    else:
                        cfg = DesignConfig(n_candidates=2000, prior="aggregate", refine_steps=100, seed=100 + i,
                                           n_conditions=0 if (masked and not a.masked_nn_cond) else 20)
                    g, _, _, _ = design(model, chooser, scaler, t, m, cfg, latent, reference_spectra=Rtr)
                    rows.append(score_designs(predict_physical(*judge, scaler.inverse_transform(g)), t, m, c, fw, F))
                key = f"{name}/{method}/{sname}"
                results[key] = summarise(rows)
                print(f"{name[:26] + '/' + method:34s} {sname:7s} "
                      f"{json.dumps({k: round(v, 4) for k, v in results[key].items()})}", flush=True)
    if a.out:
        json.dump(results, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
