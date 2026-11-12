"""Score COMSOL re-simulations of suggested designs against their target spectra.

    uv run python -m sphp_inverse.resim <folder containing run_*/> [more folders] [--csv out.csv]

A run folder needs <ID>_dataout.csv (COMSOL export: 6 geometry columns, freq (THz), reflection; one block of rows
per design) and <ID>_target_spectra.csv.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

THZ_TO_CM = 33.35641  # 1 THz in cm^-1


def read_dataout(path):
    """-> list of (geometry (6,), freq (F,), reflection (F,)), one per design, in file order."""
    rows = np.loadtxt(path, delimiter=",", comments="%", ndmin=2)
    designs, order = {}, []
    for r in rows:
        key = tuple(r[:6])
        if key not in designs:
            designs[key] = []
            order.append(key)
        designs[key].append((r[6], r[7]))
    out = []
    for key in order:
        fr = np.array(sorted(designs[key]))
        out.append((np.array(key), fr[:, 0], fr[:, 1]))
    return out


def score_run(run_dir):
    """Metrics for one run folder, or None if it has no results yet."""
    run_dir = Path(run_dir)
    run_id = run_dir.name.replace("run_", "")
    out, tgt = run_dir / f"{run_id}_dataout.csv", run_dir / f"{run_id}_target_spectra.csv"
    if not (out.exists() and tgt.exists()):
        return None
    target = np.loadtxt(tgt, delimiter=",")
    t_freq, t_R = target[:, 0], target[:, 1]
    t_dip = t_freq[np.argmin(t_R)]
    mse, dip_err = [], []
    for _, f, R in read_dataout(out):
        R_on_target = np.interp(t_freq, f, R)
        mse.append(float(np.mean((R_on_target - t_R) ** 2)))
        dip_err.append(float(abs(f[np.argmin(R)] - t_dip) * THZ_TO_CM))
    best = int(np.argmin(mse))
    return {"run": run_id, "n_designs": len(mse), "best_mse": mse[best], "mean_mse": float(np.mean(mse)),
            "target_dip_THz": float(t_dip), "best_dip_err_cm": dip_err[best], "mse_per_design": mse,
            "dip_err_per_design_cm": dip_err}


def score_folder(folder):
    return [s for s in (score_run(d) for d in sorted(Path(folder).glob("run_*"))) if s]


def summarise(scores, kpi_cm=4.0):
    if not scores:
        return {"targets": 0}
    return {"targets": len(scores), "mean_best_mse": float(np.mean([s["best_mse"] for s in scores])),
            "median_best_mse": float(np.median([s["best_mse"] for s in scores])),
            "mean_mse_all_designs": float(np.mean([s["mean_mse"] for s in scores])),
            f"dip_within_{kpi_cm:g}cm": int(sum(s["best_dip_err_cm"] <= kpi_cm for s in scores))}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folders", nargs="+")
    ap.add_argument("--csv", default=None)
    a = ap.parse_args()
    rows = []
    for folder in a.folders:
        scores = score_folder(folder)
        pending = len(list(Path(folder).glob("run_*"))) - len(scores)
        print(f"{folder}: {summarise(scores)}" + (f"  ({pending} runs without results yet)" if pending else ""))
        rows += [{"folder": str(folder), **{k: v for k, v in s.items() if not isinstance(v, list)}} for s in scores]
    if a.csv and rows:
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
