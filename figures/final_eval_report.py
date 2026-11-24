"""Score the COMSOL re-simulations of the 2026-11-23 final evaluation and draw the results.

    uv run python figures/final_eval_report.py --runs ../runs --data ../data/2026_04_15_full_pisa_data.h5

Reads runs/final_eval_2026-11-23/{test_sample,test_guided,ood_sample,ood_guided_local}/run_*/ (with <ID>_dataout.csv)
and writes to runs/final_eval_2026-11-23/:
  scores_per_target.csv    one row per target and method
  summary.md               tables used in the report
  8_resim_test.png         9 test targets: target vs COMSOL spectrum of the best plain and best guided design
  9_resim_ood.png          3 OOD dips: same, with the local window

Scoring:
  test targets   MSE over the whole spectrum (as in the archived Keras runs and the lookup baseline)
  OOD targets    inside centre +/- 0.5 THz ("local", the design goal) and inside 25.875-27.725 THz ("report window")
  dip metrics    deepest dip in the scoring window: centre error and FWHM error (cm^-1), min R, dips with
                 prominence > 0.1 (isolated = exactly 1)
Baselines: train-set lookup (true COMSOL spectra of the dataset) and, for the test targets, the archived Keras
runs 123xxx (same 9 targets; the plain-sampling runs reuse their z).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

import sys
sys.path.insert(0, str(Path(__file__).parent))
from training_report import C, GRID, INK, INK2  # noqa: E402  (same style and palette)

from sphp_inverse.data import load_dataset, reference_split  # noqa: E402
from sphp_inverse.metrics import THZ_TO_CM, dip_metrics  # noqa: E402
from sphp_inverse.resim import read_dataout  # noqa: E402
from sphp_inverse.surrogate import load_surrogate, predict_physical  # noqa: E402

F = np.linspace(25, 29, 161)
REPORT_WINDOW = (25.875, 27.725)


def comsol_spectra(run_dir):
    rid = run_dir.name[4:]
    designs = read_dataout(run_dir / f"{rid}_dataout.csv")
    return np.array([d[0] for d in designs]), np.array([np.interp(F, f, R) for _, f, R in designs])


def dip_stats(spec, target, mask, centre):
    dm, tm = dip_metrics(F, spec, mask), dip_metrics(F, target, mask)
    return {"centre_err_cm": abs(dm.freq - centre) * THZ_TO_CM, "fwhm_err_cm": abs(dm.fwhm - tm.fwhm) * THZ_TO_CM,
            "min_R": float(spec[mask].min()), "n_dips": len(find_peaks(1 - spec[mask], prominence=0.1)[0])}


def score(spectra, target, mask, centre):
    e = ((spectra[:, mask] - target[mask]) ** 2).mean(1)
    b = int(np.argmin(e))
    out = {"best_mse": float(e[b]), "mean_mse": float(e.mean()), "best": b, **dip_stats(spectra[b], target, mask, centre)}
    out["n_isolated_designs"] = int(sum(len(find_peaks(1 - s[mask], prominence=0.1)[0]) == 1 for s in spectra))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="../runs")
    ap.add_argument("--data", default="../data/2026_04_15_full_pisa_data.h5")
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # the tables contain cm⁻¹ etc.; Windows consoles default to cp1252
    runs = Path(a.runs).resolve()
    base = runs / "final_eval_2026-11-23"
    ds = load_dataset(a.data)
    Rtr = ds.R[reference_split(len(ds.geo)).train]
    judge = load_surrogate(runs / "surrogate_resmlp_v1")
    archive = runs.parent / "archive" / "v3_inverse-design_2026-07" / "keras" / "geomerty_cVAE_best" / "inference_data" / "results"

    rows, curves = [], {}
    for kind in ("test", "ood"):
        methods = [("sample", "plain cVAE"), ("guided" if kind == "test" else "guided_local", "guided")]
        idx_dirs = sorted((base / f"{kind}_{methods[0][0]}").glob("run_*"), key=lambda d: int(d.name[-3:]))
        for d0 in idx_dirs:
            idx = int(d0.name[-3:])
            tgt = np.loadtxt(d0 / f"{d0.name[4:]}_target_spectra.csv", delimiter=",")[:, 1]
            meta = json.load(open(d0 / f"{d0.name[4:]}_metadata.json"))
            centre = meta.get("centre_THz", float(F[np.argmin(tgt)]))
            windows = {"full": np.ones_like(F, bool)} if kind == "test" else {
                "local": np.abs(F - centre) <= 0.5 + 1e-4,
                "report": (F >= REPORT_WINDOW[0] - 1e-4) & (F <= REPORT_WINDOW[1] + 1e-4)}
            label = f"test #{idx}" if kind == "test" else f"FIG3 #{idx} ({centre:.3f} THz)"
            for folder, mname in methods:
                d = next((base / f"{kind}_{folder}").glob(f"run_*{idx:03d}"))
                geo, spec = comsol_spectra(d)
                pred = predict_physical(*judge, geo)
                for wname, m in windows.items():
                    s = score(spec, tgt, m, centre)
                    sp = ((pred[:, m] - tgt[m]) ** 2).mean(1)
                    rows.append({"set": kind, "target": label, "method": mname, "window": wname, **s,
                                 "surrogate_pred_best_mse": float(sp.min()),
                                 "design_flags_in_bounds": int(np.sum(json.load(open(d / f"{d.name[4:]}_metadata.json"))["checks"]["in_bounds"]))})
                curves[(kind, idx, mname)] = (spec, tgt, centre)
            for wname, m in windows.items():  # lookup baseline (true COMSOL spectra)
                top = np.argsort(((Rtr[:, m] - tgt[m]) ** 2).mean(1))[:5]
                rows.append({"set": kind, "target": label, "method": "lookup", "window": wname,
                             **score(Rtr[top], tgt, m, centre)})
            if kind == "test":  # archived Keras run on the same target
                ad = archive / f"run_123{idx}" if idx >= 100 else archive / f"run_123{idx:02d}"
                if (ad / f"{ad.name[4:]}_dataout.csv").exists():
                    _, aspec = comsol_spectra(ad)
                    rows.append({"set": kind, "target": label, "method": "archived Keras (May 2026)", "window": "full",
                                 **score(aspec, tgt, np.ones_like(F, bool), centre)})
                    curves[(kind, idx, "keras")] = (aspec, tgt, centre)

    with open(base / "scores_per_target.csv", "w", newline="") as f:
        keys = sorted({k for r in rows for k in r}, key=lambda k: list(rows[0]).index(k) if k in rows[0] else 99)
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    # ---- summary tables
    def agg(kind, window, method, key, fn=np.mean):
        v = [r[key] for r in rows if r["set"] == kind and r["window"] == window and r["method"] == method]
        return fn(v) if v else float("nan")

    lines = ["# Final evaluation: COMSOL re-simulation results (2026-11-24)\n",
             "Best-of-5 = the best of the 5 designs per target. Dip metrics are for that best design.\n",
             "## Test spectra (9 targets, whole spectrum)\n",
             "| Method | mean best-of-5 MSE | median | mean of all 5 | surrogate-predicted best | dip centre within 4 cm⁻¹ |",
             "|---|---|---|---|---|---|"]
    for m in ("plain cVAE", "guided", "lookup", "archived Keras (May 2026)"):
        pred = agg("test", "full", m, "surrogate_pred_best_mse") if m in ("plain cVAE", "guided") else float("nan")
        hits = sum(r["centre_err_cm"] <= 4 for r in rows if r["set"] == "test" and r["method"] == m)
        n = sum(1 for r in rows if r["set"] == "test" and r["method"] == m)
        lines.append(f"| {m} | {agg('test', 'full', m, 'best_mse'):.4f} | {agg('test', 'full', m, 'best_mse', np.median):.4f} | "
                     f"{agg('test', 'full', m, 'mean_mse'):.4f} | {'' if np.isnan(pred) else f'{pred:.4f}'} | {hits}/{n} |")
    lines += ["\n## Out-of-distribution Gaussian dips (3 targets)\n",
              "| Target | Method | MSE ± 0.5 THz | centre err (cm⁻¹) | FWHM err (cm⁻¹) | min R | dips in ± 0.5 THz | isolated designs (of 5) | MSE report window | surrogate-predicted (± 0.5 THz) |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if r["set"] == "ood" and r["window"] == "local":
            rep = next(x for x in rows if x["set"] == "ood" and x["target"] == r["target"] and x["method"] == r["method"] and x["window"] == "report")
            sp = r.get("surrogate_pred_best_mse")
            lines.append(f"| {r['target']} | {r['method']} | {r['best_mse']:.4f} | {r['centre_err_cm']:.1f} | {r['fwhm_err_cm']:.1f} | "
                         f"{r['min_R']:.2f} | {r['n_dips']} | {r['n_isolated_designs'] if r['method'] != 'lookup' else '–'} | "
                         f"{rep['best_mse']:.4f} | {'' if sp is None else f'{sp:.4f}'} |")
    (base / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

    # ---- figures: COMSOL spectra
    for kind, fname, ncols in (("test", "8_resim_test.png", 3), ("ood", "9_resim_ood.png", 3)):
        idxs = sorted({k[1] for k in curves if k[0] == kind})
        nrows = int(np.ceil(len(idxs) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.3 * nrows + 0.6), sharex=True, sharey=True, squeeze=False)
        for ax, idx in zip(axes.flat, idxs):
            spec_s, tgt, centre = curves[(kind, idx, "plain cVAE")]
            spec_g = curves[(kind, idx, "guided")][0]
            m = np.ones_like(F, bool) if kind == "test" else np.abs(F - centre) <= 0.5 + 1e-4
            for spec, col, lab in ((spec_s, C["orange"], "plain cVAE"), (spec_g, C["blue"], "guided")):
                e = ((spec[:, m] - tgt[m]) ** 2).mean(1)
                ax.plot(F, spec[np.argmin(e)], color=col, lw=1.7, label=f"{lab} ({e.min():.4f})", zorder=3)
            ax.plot(F, tgt, color=INK, lw=1.5, ls="--", label="target", zorder=4)
            if kind == "ood":
                ax.axvspan(centre - 0.5, centre + 0.5, color=GRID, alpha=0.8, zorder=0)
                ax.set_title(f"Gaussian dip FIG3 #{idx} ({centre:.3f} THz)", loc="left", fontsize=10)
            else:
                ax.set_title(f"Test spectrum #{idx}", loc="left", fontsize=10)
            ax.legend(loc="lower right", fontsize=7.5)
            ax.set_ylim(0, 1.02)
        for ax in axes[-1]:
            ax.set_xlabel("frequency (THz)")
        for ax in axes[:, 0]:
            ax.set_ylabel("reflectance (COMSOL)")
        title = ("COMSOL re-simulation, test spectra: best of 5 designs per method (MSE over the whole spectrum)"
                 if kind == "test" else
                 "COMSOL re-simulation, Gaussian dips: best of 5 designs per method (MSE within ± 0.5 THz, shaded)")
        fig.suptitle(title, x=0.01, ha="left", fontsize=11, color=INK)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(base / fname)
        plt.close(fig)


if __name__ == "__main__":
    main()
