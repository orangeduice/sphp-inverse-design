"""Report Fig. 3: best-matching dataset spectra for the 18 Gaussian dip targets (brute-force search).

    uv run python -m sphp_inverse.baselines bruteforce --data <h5> --out <dir>
    uv run python figures/fig3_bruteforce.py <dir> [--out fig3_bruteforce.png]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot(result_dir, out=None, dpi=300):
    result_dir = Path(result_dir)
    arr = np.load(result_dir / "spectra.npz")
    summary = json.load(open(result_dir / "summary.json"))
    rec = [r for r in csv.DictReader(open(result_dir / "candidates.csv")) if r["rank"] == "1"]
    freq, targets, matched = arr["freq"], arr["targets"], arr["matched"]

    n = len(rec)
    cols = math.ceil(n / 2)
    fig, axes = plt.subplots(2, cols, figsize=(3.3 * cols, 10), sharex=True, sharey=True, squeeze=False)
    fig.subplots_adjust(hspace=0, wspace=0)
    for i, ax in enumerate(axes.flat):
        if i >= n:
            ax.set_visible(False)
            continue
        r = rec[i]
        k = int(np.argmin(np.abs(freq - float(r["dip_THz"]))))
        ax.plot(freq, matched[i], label=f"MSE: {float(r['mse']):.4f}")
        ax.plot(freq, targets[i], "--", label=f"dip at {float(r['centre_THz']):.4f} THz")
        ax.plot(freq[k], matched[i][k], "x")
        ax.hlines(float(r["half_height"]), float(r["left_THz"]), float(r["right_THz"]), color="C2",
                  label=f"FWHM: {float(r['fwhm_THz']):.4f} THz")
        ax.set_ylim(-0.2, 1)
        ax.legend(loc="lower center", fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("Reflectance")
        ax.set_yticks(np.arange(0, 1.01, 0.1))
    for ax in axes[-1]:
        ax.set_xlabel("Frequency (THz)")
    fig.suptitle(f"mean FWHM {summary['mean_fwhm_THz']:.4f} THz, mean prominence {summary['mean_prominence']:.4f}, "
                 f"linewidth ratio {summary['linewidth_ratio']:.5f}", y=0.92)
    out = Path(out or result_dir / "fig3_bruteforce.png")
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("result_dir")
    ap.add_argument("--out", default=None)
    ap.add_argument("--dpi", type=int, default=300)
    a = ap.parse_args()
    print(plot(a.result_dir, a.out, a.dpi))


if __name__ == "__main__":
    main()
