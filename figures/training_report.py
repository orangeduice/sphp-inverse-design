"""Figures summarising the 2026-11-23 cVAE training / model-selection study.

    uv run python figures/training_report.py --runs ../runs --data ../data/2026_04_15_full_pisa_data.h5 --out ../runs/figures_2026-11-23

Writes six PNGs (see FIGURES below). Everything is read from run folders; nothing is retrained.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# reference categorical palette (fixed order) + text/grid tokens
C = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100", "magenta": "#e87ba4",
     "green": "#008300", "violet": "#6250d6", "red": "#e34948"}
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"

plt.rcParams.update({
    "font.size": 9.5, "axes.titlesize": 10.5, "axes.titleweight": "bold", "axes.labelcolor": INK2,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.8, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False, "lines.linewidth": 2.0, "figure.dpi": 140,
    "savefig.bbox": "tight", "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
})

CHOSEN = "eps_sample_beta1e-2_long"


def read_history(run):
    rows = list(csv.DictReader(open(Path(run) / "history.csv")))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


def cfg_of(run):
    import yaml
    return yaml.safe_load(open(Path(run) / "config.yaml"))


def fig_curves(runs, out):
    """1: training curves, reference recipe vs chosen recipe."""
    groups = {"Reference recipe (β 0.001, 37 epochs)": (sorted(p for p in glob.glob(str(runs / "torch_parity_seed*")) if os.path.isdir(p)), C["orange"]),
              "Chosen recipe (β 0.01, 120 epochs)": (sorted(p for p in glob.glob(str(runs / "exp" / f"{CHOSEN}_seed*")) if os.path.isdir(p)), C["blue"])}
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for label, (paths, col) in groups.items():
        for j, p in enumerate(paths):
            h, c = read_history(p), cfg_of(p)
            ep = h["epoch"]
            kl_nats = h["val_kl_loss"] * c["latent_dim"] / c["beta"]   # undo beta and the 1/latent_dim factor
            lab = label if j == 0 else None
            axes[0].plot(ep, h["reconstr_loss"], color=col, alpha=0.8, lw=1.3, label=lab)
            axes[1].plot(ep, h["val_reconstr_loss"], color=col, alpha=0.8, lw=1.3, label=lab)
            axes[2].plot(ep, kl_nats, color=col, alpha=0.8, lw=1.3, label=lab)
    for ax, t, yl in zip(axes, ["Train reconstruction", "Validation reconstruction", "Validation KL divergence"],
                         ["MSE (scaled geometry)", "MSE (scaled geometry)", "KL (nats, summed over 8 dims)"]):
        ax.set_title(t, loc="left")
        ax.set_xlabel("epoch")
        ax.set_ylabel(yl)
        ax.set_yscale("log")
    axes[1].set_ylim(3e-3, 1.5)
    axes[0].legend(loc="upper right")
    fig.suptitle("Training curves: the chosen recipe gives up some reconstruction for a latent much closer to the prior "
                 "(KL ≈ 7 vs ≈ 30 nats)",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.savefig(out / "1_training_curves.png")
    plt.close(fig)


def load_selection(runs):
    rows = []
    for f in glob.glob(str(runs / "exp" / "selection_*.csv")):
        for r in csv.DictReader(open(f)):
            r = {k: (float(v) if k != "run" and v not in ("", "None") else v) for k, v in r.items()}
            name = Path(r["run"]).name
            if name.startswith("torch_parity"):
                beta, sched, eps = 1e-3, "37", "shared"
            else:
                c = cfg_of(r["run"])
                beta, sched, eps = c["beta"], str(sum(e for _, e in c["rounds"])), "shared" if c["shared_eps"] else "sample"
            m = json.load(open(Path(r["run"]) / "metrics.json"))
            rows.append({**r, "name": name, "beta": beta, "sched": sched, "eps": eps,
                         "kl_nats": m["test_kl_sum"], "mu_var": m["test_mu_var_per_dim"]})
    return rows


def fig_beta(rows, out):
    """2: beta sweep."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    styles = {("37", "shared"): ("Reference (shared ε), 37 ep", C["orange"], "s"),
              ("37", "sample"): ("Per-sample ε, 37 ep", C["blue"], "o"),
              ("120", "sample"): ("Per-sample ε, 120 ep", C["aqua"], "D")}
    seen = set()
    for r in rows:
        lab, col, mk = styles[(r["sched"], r["eps"])]
        kw = dict(color=col, marker=mk, s=42, edgecolor="#fcfcfb", linewidth=1.2, zorder=3,
                  label=lab if lab not in seen else None)
        seen.add(lab)
        axes[0].scatter(r["beta"], r["val_normal_best_mse"], **kw)
        axes[1].scatter(r["beta"], r["val_normal_mean_mse"], **{**kw, "label": None})
        axes[2].scatter(r["beta"], max(r["kl_nats"], 1e-3), **{**kw, "label": None})
    chosen = [r for r in rows if r["name"].startswith(CHOSEN)]
    for ax, key in zip(axes[:2], ["val_normal_best_mse", "val_normal_mean_mse"]):
        for r in chosen:
            ax.annotate("chosen", (r["beta"], r[key]), xytext=(-48, 14), textcoords="offset points", color=INK,
                        fontsize=8.5, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
            break
    axes[2].axvspan(0.07, 1.5, color=GRID, alpha=0.7, zorder=0)
    axes[2].text(0.085, 2.0, "posterior\ncollapse", color=INK2, fontsize=8.5, va="top")
    titles = ["Best of 5 sampled designs", "Mean of 5 sampled designs", "Latent information (test KL)"]
    ylabels = ["surrogate MSE vs target (validation)", "surrogate MSE vs target (validation)", "KL (nats)"]
    for ax, t, yl in zip(axes, titles, ylabels):
        ax.set_xscale("log")
        ax.set_xlabel("β (KL weight)")
        ax.set_title(t, loc="left")
        ax.set_ylabel(yl)
    axes[2].set_yscale("log")
    axes[2].set_ylim(5e-4, 80)
    axes[0].legend(loc="upper left")
    fig.suptitle("β sweep: 0.01-0.03 gives the best sampled designs; β ≥ 0.1 collapses (KL → 0, all samples identical)",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.savefig(out / "2_beta_sweep.png")
    plt.close(fig)


def fig_latent(rows, out, runs, data):
    """3: per-dimension spread of the aggregate posterior (Var of μ + mean σ²) vs the N(0, 1) prior.

    At inference z is drawn from N(0, 1). The decoder only saw z from the aggregate posterior during training, whose
    variance per dimension is Var(μ) + E[σ²]; a good match is a total near 1. A dimension with Var(μ) ≈ 0 carries
    no information (the encoder just returns the prior there): inactive, but harmless.
    """
    import torch

    from sphp_inverse.data import load_dataset, reference_split
    from sphp_inverse.inference import load_model

    ds = load_dataset(data)
    split = reference_split(len(ds.geo))
    pick = {"Reference recipe (β 0.001)": (runs / "torch_parity_seed2", C["orange"], "#f6b39a"),
            "Chosen recipe (β 0.01, 120 ep)": (runs / "exp" / f"{CHOSEN}_seed0", C["blue"], "#9fc3ee")}
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.9), sharey=True)
    for ax, (label, (path, col, light)) in zip(axes, pick.items()):
        model, scaler = load_model(path)
        model = model.float().cpu().eval()
        with torch.no_grad():
            mu, logvar = model.encoder(torch.tensor(scaler.transform(ds.geo[split.train]), dtype=torch.float32),
                                       torch.tensor(ds.R[split.train]))
        var_mu, mean_s2 = mu.var(0).numpy(), logvar.exp().mean(0).numpy()
        x = np.arange(8)
        ax.bar(x, var_mu, width=0.62, color=col, label="Var(μ): information carried", zorder=3)
        ax.bar(x, mean_s2, bottom=var_mu, width=0.62, color=light, edgecolor="#fcfcfb", linewidth=2,
               label="mean σ²: encoder noise", zorder=3)
        ax.axhline(1.0, color=INK, lw=1.2, ls="--", zorder=4)
        ax.text(7.45, 1.1, "prior", ha="right", color=INK, fontsize=8.5)
        informative = int((var_mu > 0.1).sum())
        ax.set_title(f"{label}: {informative}/8 dims with Var(μ) > 0.1, total spread {np.mean(var_mu + mean_s2):.1f}× prior",
                     loc="left", fontsize=10)
        ax.set_xticks(x, [f"z{i + 1}" for i in x])
        ax.legend(loc="upper right")
    axes[0].set_ylabel("variance per latent dim (training set)")
    fig.suptitle("Latent spread vs the N(0, 1) prior used for sampling: the reference model's latent is 2-5× too wide; "
                 "the chosen one matches it, using about half its dims", x=0.01, ha="left", fontsize=11, color=INK)
    fig.savefig(out / "3_latent_variance.png")
    plt.close(fig)


def fig_methods(runs, out):
    """4: design methods (validation targets, judged by the independent surrogate)."""
    ref = json.load(open(runs / "exp" / "methods_torch_parity_seed2.json"))["results"]
    best = json.load(open(runs / "exp" / f"methods_{CHOSEN}_seed0.json"))["results"]
    items = [("Train-set lookup", ref["lookup/val"], ref["lookup/ood_val"], MUTED),
             ("cVAE, 5 samples\n(reference recipe)", ref["cvae/val"], ref["cvae/ood_val"], C["orange"]),
             ("cVAE, 5 samples\n(chosen recipe)", best["cvae/val"], best["cvae/ood_val"], C["blue"]),
             ("+ 2000 samples,\nsurrogate pick", ref["filter/val"], ref["filter/ood_val"], C["violet"]),
             ("+ nearest-spectrum\nconditioning", best["nn-cond/val"], best["nn-cond/ood_val"], C["aqua"]),
             ("+ refinement\n(guided, used)", best["nn-cond+refine/val"], best["nn-cond+refine/ood_val"], C["green"])]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    for ax, j, title in zip(axes, (1, 2), ["In distribution: 50 validation spectra", "Out of distribution: 12 Gaussian dips (FWHM 0.3 THz)"]):
        vals = [it[j]["best_mse"] for it in items]
        y = np.arange(len(items))[::-1]
        ax.barh(y, vals, color=[it[3] for it in items], height=0.62, zorder=3)
        for yi, v in zip(y, vals):
            ax.text(v * 1.03, yi, f"{v:.4f}", va="center", color=INK, fontsize=8.5)
        ax.set_yticks(y, [it[0] for it in items])
        ax.set_xlabel("best-of-5 MSE vs target (independent surrogate; lookup uses true COMSOL spectra)")
        ax.set_title(title, loc="left")
        ax.set_xlim(0, max(vals) * 1.22)
        ax.grid(axis="y", visible=False)
    axes[1].set_yticklabels([])
    fig.suptitle("A plain cVAE is worse than looking the target up in the dataset; surrogate-guided design closes the gap",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.savefig(out / "4_design_methods.png")
    plt.close(fig)


def fig_surrogate(runs, out):
    """5: surrogate vs COMSOL on archived re-simulations."""
    from sphp_inverse.resim import read_dataout
    from sphp_inverse.surrogate import load_surrogate, predict_physical

    roots = [runs.parent / "archive" / "v3_inverse-design_2026-07" / "keras" / f / "inference_data" / "results"
             for f in ("geomerty_cVAE_best", "geomerty_cVAE_reduced_range", "geomerty_cVAE_reduced_range_2")]
    roots += [runs.parent / "archive" / "v2_tensorflow_2026-05" / f / "inference_data" / "results"
              for f in ("geomerty_cVAE_best", "old_geomerty_cVAE")]
    F = np.linspace(25, 29, 161)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    for ax, (sname, col) in zip(axes, [("surrogate_mlp_v1", C["blue"]), ("surrogate_resmlp_v1", C["violet"])]):
        surr = load_surrogate(runs / sname)
        xs, ys = [], []
        for root in roots:
            for d in sorted(glob.glob(str(root / "run_*"))):
                rid = os.path.basename(d)[4:]
                outf = Path(d) / f"{rid}_dataout.csv"
                if not outf.exists():
                    continue
                tgt = np.loadtxt(Path(d) / f"{rid}_target_spectra.csv", delimiter=",")
                for geo, f, Rc in read_dataout(outf):
                    Rs = np.interp(tgt[:, 0], F, predict_physical(*surr, geo[None])[0])
                    xs.append(np.mean((np.interp(tgt[:, 0], f, Rc) - tgt[:, 1]) ** 2))
                    ys.append(np.mean((Rs - tgt[:, 1]) ** 2))
        xs, ys = np.array(xs), np.array(ys)
        from scipy.stats import spearmanr
        rho = spearmanr(xs, ys).correlation
        lim = [min(xs.min(), ys.min()) * 0.7, max(xs.max(), ys.max()) * 1.4]
        ax.plot(lim, lim, color=MUTED, lw=1, ls="--", zorder=2)
        ax.scatter(xs, ys, s=20, color=col, alpha=0.75, edgecolor="#fcfcfb", linewidth=0.6, zorder=3)
        ax.set_xscale("log"), ax.set_yscale("log"), ax.set_xlim(lim), ax.set_ylim(lim)
        ax.set_xlabel("COMSOL: MSE of re-simulated spectrum vs target")
        ax.set_title(f"{'MLP (chooses designs)' if 'mlp_v1' in sname and 'res' not in sname else 'Residual MLP (judges designs)'}"
                     f"   Spearman ρ = {rho:.3f}", loc="left")
    axes[0].set_ylabel("surrogate-predicted MSE vs target")
    fig.suptitle(f"Surrogates rank designs like COMSOL does ({len(xs)} archived suggested designs, never in the dataset)",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.savefig(out / "5_surrogate_vs_comsol.png")
    plt.close(fig)


def fig_final(runs, out):
    """6: preview of the final designs (surrogate prediction of the best design per method)."""
    from sphp_inverse.surrogate import load_surrogate, predict_physical

    judge = load_surrogate(runs / "surrogate_resmlp_v1")
    base = runs / "final_eval_2026-11-23"
    F = np.linspace(25, 29, 161)
    targets = [("test", d.name[4:][-3:].lstrip("0") or "0") for d in sorted((base / "test_guided").glob("run_*"))]
    targets += [("ood", d.name[4:][-3:].lstrip("0") or "0") for d in sorted((base / "ood_guided").glob("run_*"))]
    fig, axes = plt.subplots(3, 4, figsize=(14, 8.4), sharex=True, sharey=True)
    for ax, (kind, idx) in zip(axes.flat, targets):
        curves = {}
        for method, col in (("sample", C["orange"]), ("guided", C["blue"])):
            d = next((base / f"{kind}_{method}").glob(f"run_*{int(idx):03d}"))
            rid = d.name[4:]
            tgt = np.loadtxt(d / f"{rid}_target_spectra.csv", delimiter=",")
            phys = np.loadtxt(d / f"{rid}_suggested_designs_physical.csv", delimiter=",")
            pred = predict_physical(*judge, phys)
            win = (F >= 25.875 - 1e-4) & (F <= 27.725 + 1e-4) if kind == "ood" else np.ones_like(F, bool)
            e = ((pred[:, win] - tgt[win, 1]) ** 2).mean(1)
            curves[method] = (pred[np.argmin(e)], e.min(), col)
        ax.plot(tgt[:, 0], tgt[:, 1], color=INK, lw=1.6, ls="--", label="target", zorder=4)
        for method, (spec, err, col) in curves.items():
            lab = {"sample": "plain cVAE", "guided": "guided"}[method]
            ax.plot(F, spec, color=col, lw=1.6, label=f"{lab} ({err:.4f})", zorder=3)
        if kind == "ood":
            ax.axvspan(25.875, 27.725, color=GRID, alpha=0.5, zorder=0)
        ax.set_title(("Test spectrum #" if kind == "test" else "Gaussian dip FIG3 #") + idx, loc="left", fontsize=9.5)
        ax.legend(loc="lower right", fontsize=7.5, handlelength=1.4)
        ax.set_ylim(0, 1.02)
    for ax in axes[-1]:
        ax.set_xlabel("frequency (THz)")
    for ax in axes[:, 0]:
        ax.set_ylabel("reflectance")
    fig.suptitle("Final designs, preview: surrogate-predicted spectrum of the best of 5 designs per method "
                 "(MSE in legend; COMSOL will give the real result)", x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out / "6_final_designs_preview.png")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="../runs")
    ap.add_argument("--data", default="../data/2026_04_15_full_pisa_data.h5")
    ap.add_argument("--out", default="../runs/figures_2026-11-23")
    a = ap.parse_args()
    runs, out = Path(a.runs).resolve(), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = load_selection(runs)
    fig_curves(runs, out)
    fig_beta(rows, out)
    fig_latent(rows, out, runs, a.data)
    fig_methods(runs, out)
    fig_surrogate(runs, out)
    fig_final(runs, out)
    table = fig_ood_local(runs, out, a.data)
    with open(out / "7_ood_local_window.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
    print("wrote", sorted(p.name for p in out.glob("*.png")))



def fig_ood_local(runs, out, data):
    """7: OOD guided designs chosen for a local window (centre +/- 0.5 THz) vs for the report window.

    Returns a table of metrics (judged by the independent surrogate inside the local window)."""
    from scipy.signal import find_peaks

    from sphp_inverse.data import load_dataset, reference_split
    from sphp_inverse.metrics import THZ_TO_CM, dip_metrics
    from sphp_inverse.surrogate import load_surrogate, predict_physical

    judge = load_surrogate(runs / "surrogate_resmlp_v1")
    base = runs / "final_eval_2026-11-23"
    ds = load_dataset(data)
    Rtr = ds.R[reference_split(len(ds.geo)).train]
    F = np.linspace(25, 29, 161)
    variants = [("ood_guided_local", "guided, local window (new)", C["blue"]),
                ("ood_guided_reportwindow", "guided, report window (old)", C["orange"])]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), sharey=True)
    table = []
    for ax, idx in zip(axes, (0, 8, 17)):
        for k, (folder, label, col) in enumerate(variants):
            d = next((base / folder).glob(f"run_*{idx:03d}"))
            rid = d.name[4:]
            meta = json.load(open(d / f"{rid}_metadata.json"))
            tgt = np.loadtxt(d / f"{rid}_target_spectra.csv", delimiter=",")
            c = meta["centre_THz"]
            m = np.abs(F - c) <= 0.5 + 1e-4
            pred = predict_physical(*judge, np.loadtxt(d / f"{rid}_suggested_designs_physical.csv", delimiter=","))
            e = ((pred[:, m] - tgt[m, 1]) ** 2).mean(1)
            b = int(np.argmin(e))
            if k == 0:
                for p in pred:
                    ax.plot(F, p, color=col, lw=0.8, alpha=0.35, zorder=2)
            ax.plot(F, pred[b], color=col, lw=2, label=f"{label}", zorder=3)
            dm, tm = dip_metrics(F, pred[b], m), dip_metrics(F, tgt[:, 1], m)
            table.append({"target": f"FIG3 #{idx} ({c:.3f} THz)", "design set": label, "best MSE (±0.5 THz)": e[b],
                          "centre error (cm-1)": abs(dm.freq - c) * THZ_TO_CM, "FWHM error (cm-1)": abs(dm.fwhm - tm.fwhm) * THZ_TO_CM,
                          "min R": float(pred[b][m].min()),
                          "dips in window": len(find_peaks(1 - pred[b][m], prominence=0.1)[0]),
                          "designs isolated": int(sum(len(find_peaks(1 - p[m], prominence=0.1)[0]) == 1 for p in pred))})
        # lookup reference
        top = np.argmin(((Rtr[:, m] - tgt[m, 1]) ** 2).mean(1))
        lm = dip_metrics(F, Rtr[top], m)
        table.append({"target": f"FIG3 #{idx} ({c:.3f} THz)", "design set": "train-set lookup (COMSOL)",
                      "best MSE (±0.5 THz)": float(((Rtr[top, m] - tgt[m, 1]) ** 2).mean()),
                      "centre error (cm-1)": abs(lm.freq - c) * THZ_TO_CM, "FWHM error (cm-1)": abs(lm.fwhm - tm.fwhm) * THZ_TO_CM,
                      "min R": float(Rtr[top][m].min()), "dips in window": len(find_peaks(1 - Rtr[top][m], prominence=0.1)[0]),
                      "designs isolated": None})
        ax.axvspan(c - 0.5, c + 0.5, color=GRID, alpha=0.8, zorder=0)
        ax.text(c, 0.03, "±0.5 THz filter", ha="center", color=INK2, fontsize=8)
        ax.plot(tgt[:, 0], tgt[:, 1], color=INK, lw=1.6, ls="--", label="target", zorder=4)
        ax.set_title(f"Gaussian dip FIG3 #{idx}  ({c:.3f} THz)", loc="left")
        ax.set_xlabel("frequency (THz)")
        ax.set_ylim(0, 1.02)
    axes[0].set_ylabel("reflectance (surrogate prediction)")
    h, l = axes[2].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("OOD designs re-chosen for a locally isolated dip: best design (thick) and all 5 new designs (thin)",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.savefig(out / "7_ood_local_window.png")
    plt.close(fig)
    return table


if __name__ == "__main__":
    main()
