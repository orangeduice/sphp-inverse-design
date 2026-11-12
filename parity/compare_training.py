"""Compare Keras and PyTorch training runs (Phase 3a step 4: training parity).

    uv run python parity/compare_training.py --runs ../runs [--notebook ../runs/keras_reference/notebook_history.csv]

Reads <runs>/keras_train_seed*/ (history.json, metrics.json) and <runs>/torch_parity_seed*/ (history.csv,
metrics.json), prints a table and writes <runs>/training_parity.md and training_parity.png.
"""
import argparse
import csv
import glob
import json
from pathlib import Path

import numpy as np

KEYS = ["loss", "reconstr_loss", "kl_loss", "val_loss", "val_reconstr_loss", "val_kl_loss", "lr"]


def load(runs):
    out = {"keras": [], "torch": []}
    for d in sorted(glob.glob(str(runs / "keras_train_seed*/"))):
        d = Path(d)
        if (d / "metrics.json").exists():
            out["keras"].append((json.load(open(d / "history.json")), json.load(open(d / "metrics.json"))))
    for d in sorted(glob.glob(str(runs / "torch_parity_seed*/"))):
        d = Path(d)
        if (d / "metrics.json").exists():
            rows = list(csv.DictReader(open(d / "history.csv")))
            out["torch"].append(({k: [float(r[k]) for r in rows] for k in KEYS}, json.load(open(d / "metrics.json"))))
    return out


def stat(vals):
    v = np.asarray(vals, dtype=float)
    return f"{v.mean():.4f} ± {v.std():.4f} [{v.min():.4f}, {v.max():.4f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="../runs")
    ap.add_argument("--notebook", default=None)
    a = ap.parse_args()
    runs = Path(a.runs)
    data = load(runs)
    nb = None
    if a.notebook and Path(a.notebook).exists():
        rows = list(csv.DictReader(open(a.notebook)))
        nb = {k: [float(r[k]) for r in rows] for k in KEYS}

    lines = [f"# Training parity: Keras 2.15 vs PyTorch\n",
             f"Runs: {len(data['keras'])} Keras, {len(data['torch'])} PyTorch"
             + (", plus the original notebook output (1 run)" if nb else "") + ". Values are mean ± std [min, max] over seeds.\n",
             "| Metric | Keras 2.15 | PyTorch |" + (" Notebook |" if nb else ""),
             "|---|---|---|" + ("---|" if nb else "")]

    def row(name, fk, ft, fn=None):
        k = stat([fk(h, m) for h, m in data["keras"]]) if data["keras"] else "-"
        t = stat([ft(h, m) for h, m in data["torch"]]) if data["torch"] else "-"
        n = f" {fn(nb):.4f} |" if (nb and fn) else (" - |" if nb else "")
        lines.append(f"| {name} | {k} | {t} |{n}")

    last = lambda key: (lambda h, m: h[key][-1])
    row("epochs run", lambda h, m: len(h["loss"]), lambda h, m: len(h["loss"]), lambda n: len(n["loss"]))
    for key in ["reconstr_loss", "kl_loss", "val_reconstr_loss", "val_kl_loss"]:
        row(f"final {key}", last(key), last(key), lambda n, key=key: n[key][-1])
    row("best val_loss", lambda h, m: min(h["val_loss"]), lambda h, m: min(h["val_loss"]), lambda n: min(n["val_loss"]))
    row("mean val_reconstr_loss, last 5 epochs", lambda h, m: np.mean(h["val_reconstr_loss"][-5:]),
        lambda h, m: np.mean(h["val_reconstr_loss"][-5:]), lambda n: np.mean(n["val_reconstr_loss"][-5:]))
    row("val_loss at epoch 3 (BN warm-up)", lambda h, m: h["val_loss"][2], lambda h, m: h["val_loss"][2], lambda n: n["val_loss"][2])
    row("final lr (x1e4)", lambda h, m: h["lr"][-1] * 1e4, lambda h, m: h["lr"][-1] * 1e4, lambda n: n["lr"][-1] * 1e4)
    for key in ["test_recon_zmu", "test_recon_zprior", "test_kl_sum", "active_units"]:
        row(key, lambda h, m, key=key: m[key], lambda h, m, key=key: m[key])
    row("min Var(mu) over latent dims (test)", lambda h, m: min(m["test_mu_var_per_dim"]), lambda h, m: min(m["test_mu_var_per_dim"]))
    row("mean Var(mu) over latent dims (test)", lambda h, m: np.mean(m["test_mu_var_per_dim"]), lambda h, m: np.mean(m["test_mu_var_per_dim"]))
    row("training time (s)", lambda h, m: m["train_seconds"], lambda h, m: m["train_seconds"])

    text = "\n".join(lines) + "\n"
    print(text)
    (runs / "training_parity.md").write_text(text, encoding="utf-8")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
        for ax, key, title in zip(axes, ["reconstr_loss", "val_reconstr_loss", "val_kl_loss"],
                                  ["Train reconstruction", "Validation reconstruction", "Validation KL (x beta)"]):
            for fw, colour in (("keras", "tab:orange"), ("torch", "tab:blue")):
                for j, (h, _) in enumerate(data[fw]):
                    ax.plot(range(1, len(h[key]) + 1), h[key], color=colour, alpha=0.7, lw=1.2,
                            label={"keras": "Keras 2.15", "torch": "PyTorch"}[fw] if j == 0 else None)
            if nb:
                ax.plot(range(1, len(nb[key]) + 1), nb[key], color="k", ls="--", lw=1.2, label="original notebook")
            ax.set_yscale("log"), ax.set_xlabel("epoch"), ax.set_title(title), ax.grid(alpha=0.3)
        axes[0].legend()
        fig.tight_layout()
        fig.savefig(runs / "training_parity.png", dpi=130)
    except Exception as e:  # plotting is optional
        print("plot skipped:", e)


if __name__ == "__main__":
    main()
