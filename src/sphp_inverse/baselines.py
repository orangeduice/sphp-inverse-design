"""Baselines for inverse design.

Brute force (report Fig. 3): for every target spectrum, return the dataset spectrum with the lowest MSE inside the
scoring window. This shows which targets the simulated dataset can already reach.

    uv run python -m sphp_inverse.baselines bruteforce --data <h5> --out <dir> [--top-k 1] [--min-prominence 0]

Writes to <dir>:
  candidates.csv   one row per (target, rank): centre, dataset row, MSE, dip frequency, FWHM, prominence, geometry
  spectra.npz      freq, targets, matched spectra, scoring window (input for figures/fig3_bruteforce.py)
  summary.json     target settings and the mean FWHM / prominence / linewidth ratio over the rank-1 candidates
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .data import GEO_NAMES, load_dataset
from .metrics import THZ_TO_CM, dip_metrics, mse
from .targets import FIG3, DipTargetSet


def brute_force_search(R, targets, mask=None, top_k=1):
    """For each target, the `top_k` dataset rows with the lowest MSE (inside `mask`).

    Returns (rows (n_targets, top_k), mse (n_targets, top_k)). Ties go to the lowest row index, as in the original
    loop that only replaced the best on a strictly lower loss."""
    R = np.asarray(R)
    targets = np.atleast_2d(targets)
    rows = np.empty((len(targets), top_k), dtype=int)
    errs = np.empty((len(targets), top_k))
    for i, t in enumerate(targets):
        e = mse(R, t, mask)
        order = np.argsort(e, kind="stable")[:top_k]
        rows[i], errs[i] = order, e[order]
    return rows, errs


def run_bruteforce(ds, spec: DipTargetSet = FIG3, top_k=1, min_prominence=0.0):
    """Brute-force search of the dataset `ds` for every target of `spec`. Returns (records, arrays, summary)."""
    freq = ds.freq
    centres, targets, window = spec.centres(), spec.spectra(freq), spec.window(freq)
    rows, errs = brute_force_search(ds.R, targets, window, top_k)
    records = []
    for i, (c, t) in enumerate(zip(centres, targets)):
        region = t < spec.region_level
        for k in range(top_k):
            r = int(rows[i, k])
            m = dip_metrics(freq, ds.R[r], region)
            records.append({"target": i, "rank": k + 1, "centre_THz": float(c), "row": r, "mse": float(errs[i, k]),
                            "dip_THz": m.freq, "dip_err_cm": abs(m.freq - c) * THZ_TO_CM, "fwhm_THz": m.fwhm,
                            "fwhm_cm": m.fwhm_cm, "prominence": m.prominence, "half_height": m.half_height,
                            "left_THz": m.left, "right_THz": m.right,
                            "accepted": bool(m.prominence >= min_prominence),
                            **{name: float(v) for name, v in zip(GEO_NAMES, ds.geo[r])}})
    best = [r for r in records if r["rank"] == 1]
    kept = [r for r in best if r["accepted"]]
    span = float(freq.max() - freq.min())
    mean_fwhm = float(np.mean([r["fwhm_THz"] for r in kept])) if kept else float("nan")
    summary = {
        "targets": spec.to_dict(), "centres_THz": centres.tolist(), "top_k": top_k, "min_prominence": min_prominence,
        "n_targets": len(best), "n_accepted": len(kept),
        "mean_fwhm_THz": mean_fwhm,
        "mean_prominence": float(np.mean([r["prominence"] for r in kept])) if kept else float("nan"),
        "mean_mse": float(np.mean([r["mse"] for r in kept])) if kept else float("nan"),
        # report: mean FWHM over the full simulated range (25-29 THz)
        "linewidth_ratio": mean_fwhm / span, "freq_span_THz": span,
    }
    arrays = {"freq": freq, "targets": targets, "window": window, "rows": rows, "mse": errs,
              "matched": ds.R[rows[:, 0]], "centres": centres}
    return records, arrays, summary


def save_bruteforce(out, records, arrays, summary):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "candidates.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0]))
        w.writeheader()
        w.writerows(records)
    np.savez(out / "spectra.npz", **arrays)
    json.dump(summary, open(out / "summary.json", "w"), indent=1)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    bf = sub.add_parser("bruteforce", help="best-matching dataset spectra for Gaussian dip targets")
    bf.add_argument("--data", required=True)
    bf.add_argument("--out", required=True)
    bf.add_argument("--top-k", type=int, default=1)
    bf.add_argument("--min-prominence", type=float, default=0.0,
                    help="mark rank-1 candidates below this prominence as not accepted (excluded from the means)")
    for name, default in FIG3.to_dict().items():
        bf.add_argument(f"--{name.replace('_', '-')}", type=type(default), default=default)
    a = ap.parse_args()

    spec = DipTargetSet(**{k: getattr(a, k) for k in FIG3.to_dict()})
    records, arrays, summary = run_bruteforce(load_dataset(a.data), spec, a.top_k, a.min_prominence)
    out = save_bruteforce(a.out, records, arrays, summary)
    for r in records:
        if r["rank"] == 1:
            print(f"{r['centre_THz']:.4f} THz  row {r['row']:5d}  mse {r['mse']:.5f}  dip {r['dip_THz']:.3f}  "
                  f"fwhm {r['fwhm_THz']:.4f}  prom {r['prominence']:.4f}" + ("" if r["accepted"] else "  (rejected)"))
    print(f"{summary['n_accepted']}/{summary['n_targets']} accepted  mean FWHM {summary['mean_fwhm_THz']:.4f} THz  "
          f"mean prominence {summary['mean_prominence']:.4f}  linewidth ratio {summary['linewidth_ratio']:.5f}  -> {out}")


if __name__ == "__main__":
    main()
