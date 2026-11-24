"""Forward surrogate: geometry -> reflection spectrum.

Used to *rank* inverse-design models without COMSOL (a cheap stand-in for re-simulation during model selection).
It is never used to report final results: those come from COMSOL re-simulation.

    uv run python -m sphp_inverse.surrogate --data <h5> --out <dir> [--epochs 400]

Trained on the reference *train* split only, so validation and test spectra stay unseen.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .data import fit_geo_scaler, load_dataset, reference_split


class ResBlock(nn.Module):
    def __init__(self, d, p):
        super().__init__()
        self.f = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Dropout(p), nn.Linear(d, d))

    def forward(self, x):
        return x + self.f(x)


class ForwardResMLP(nn.Module):
    """Residual MLP with LayerNorm and dropout (a different architecture, used as an independent second opinion)."""

    def __init__(self, n_geo=6, n_freq=161, width=512, depth=6, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_geo, width), *[ResBlock(width, dropout) for _ in range(depth)],
                                 nn.LayerNorm(width), nn.Linear(width, n_freq))
        self.hidden, self.n_freq = ("res", width, depth, dropout), n_freq

    def forward(self, geo_scaled):
        return self.net(geo_scaled)


def build(hidden, n_freq):
    if hidden and hidden[0] == "res":
        return ForwardResMLP(n_freq=n_freq, width=hidden[1], depth=hidden[2], dropout=hidden[3])
    return ForwardMLP(n_freq=n_freq, hidden=hidden)


class ForwardMLP(nn.Module):
    def __init__(self, n_geo=6, n_freq=161, hidden=(512, 1024, 1024, 512), slope=0.1):
        super().__init__()
        layers, d = [], n_geo
        for h in hidden:
            layers += [nn.Linear(d, h), nn.LeakyReLU(slope)]
            d = h
        layers.append(nn.Linear(d, n_freq))
        self.net = nn.Sequential(*layers)
        self.hidden, self.n_freq = tuple(hidden), n_freq

    def forward(self, geo_scaled):
        return self.net(geo_scaled)


def train_surrogate(data_path, out_dir, epochs=400, batch_size=128, lr=1e-3, seed=0, device=None, verbose=True,
                    arch="mlp"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(seed)
    ds = load_dataset(data_path)
    split = reference_split(len(ds.geo))
    scaler = fit_geo_scaler(ds.geo)  # same scaler as the cVAE reference
    geo = torch.tensor(scaler.transform(ds.geo), dtype=torch.float32, device=device)
    R = torch.tensor(ds.R, device=device)
    tr, va, te = (torch.as_tensor(s, device=device) for s in (split.train, split.val, split.test))

    model = (ForwardResMLP(n_freq=R.shape[1]) if arch == "resmlp" else ForwardMLP(n_freq=R.shape[1])).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * ((len(tr) + batch_size - 1) // batch_size))
    best, best_state, hist = float("inf"), None, []
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        perm = tr[torch.randperm(len(tr), device=device)]
        for i in range(0, len(perm), batch_size):
            idx = perm[i:i + batch_size]
            loss = nn.functional.mse_loss(model(geo[idx]), R[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
        model.eval()
        with torch.no_grad():
            v = nn.functional.mse_loss(model(geo[va]), R[va]).item()
        hist.append(v)
        if v < best:
            best, best_state = v, {k: x.clone() for k, x in model.state_dict().items()}
        if verbose and (ep % 50 == 0 or ep == epochs - 1):
            print(f"ep {ep + 1:4d}  val MSE {v:.2e}  best {best:.2e}", flush=True)
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        per_spec = ((model(geo[te]) - R[te]) ** 2).mean(1).cpu().numpy()
    metrics = {"arch": arch, "seed": seed, "val_mse_best": best, "test_mse": float(per_spec.mean()), "test_mse_median": float(np.median(per_spec)),
               "test_mse_p95": float(np.percentile(per_spec, 95)), "epochs": epochs, "seconds": time.time() - t0}
    torch.save({"model": model.state_dict(), "hidden": model.hidden, "n_freq": model.n_freq,
                "scaler_mean": scaler.mean_, "scaler_scale": scaler.scale_}, out_dir / "surrogate.pt")
    json.dump({**metrics, "val_history": hist}, open(out_dir / "metrics.json", "w"))
    return model, metrics


def load_surrogate(path, device="cpu"):
    ck = torch.load(Path(path) / "surrogate.pt", map_location=device, weights_only=False)
    model = build(tuple(ck["hidden"]), ck["n_freq"]).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, np.asarray(ck["scaler_mean"]), np.asarray(ck["scaler_scale"])


@torch.no_grad()
def predict_physical(model, mean, scale, geo_physical, device="cpu"):
    """Spectra (N, F) for physical geometries (N, 6)."""
    g = torch.tensor((np.asarray(geo_physical) - mean) / scale, dtype=torch.float32, device=device)
    return model(g).cpu().numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arch", choices=["mlp", "resmlp"], default="mlp")
    a = ap.parse_args()
    _, m = train_surrogate(a.data, a.out, epochs=a.epochs, seed=a.seed, arch=a.arch)
    print(json.dumps(m))


if __name__ == "__main__":
    main()
