"""Train the geometry cVAE.

    uv run python -m sphp_inverse.train --config configs/geometry_cvae_reference.yaml --data <h5> --out <run dir> --seed 0

Every run folder gets: config.yaml (resolved), history.csv, metrics.json, model.pt, env.json.
The defaults reproduce the working Keras notebook (see configs/geometry_cvae_reference.yaml).
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import yaml

from .callbacks import EarlyStopping, ReduceLROnPlateau
from .data import fit_geo_scaler, load_dataset, reference_split
from .models.cvae import CVAEConfig, GeometryCVAE, reference_loss


@dataclass
class TrainConfig:
    # data
    test_fraction: float = 0.05
    val_fraction: float = 0.3
    split_seed: int = 42
    scaler_fit: str = "all"            # "all" (reference) or "train"
    freq_slice: list | None = None     # e.g. [35, 110] for the reduced range
    # model
    latent_dim: int = 8
    n_blocks: int = 1
    shared_eps: bool = True            # reference: one epsilon per batch
    # loss / optimiser (Keras AdamW defaults)
    beta: float = 1e-3
    lr: float = 1e-4
    weight_decay: float = 0.004
    adam_eps: float = 1e-7
    # schedule: rounds of (batch_size, epochs); callbacks persist across rounds
    rounds: list = field(default_factory=lambda: [[64, 25], [128, 12]])
    reduce_lr_factor: float = 0.66
    reduce_lr_patience: int = 5
    reduce_lr_min_delta: float = 1e-4
    early_stop_patience: int = 15
    restore_best: bool = True
    # numerics
    allow_tf32: bool = False


def load_config(path) -> TrainConfig:
    return TrainConfig(**(yaml.safe_load(open(path)) or {}))


def _git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


@torch.no_grad()
def epoch_eval(model, x, y, beta, batch_size):
    """Validation as Keras does it: BN in inference mode, z still sampled, batch-size-weighted mean."""
    model.eval()
    tot = np.zeros(3)
    for i in range(0, len(x), batch_size):
        xb, yb = x[i:i + batch_size], y[i:i + batch_size]
        pred, mu, logvar = model(xb, yb)
        tot += np.array([v.item() for v in reference_loss(xb, pred, mu, logvar, beta)]) * len(xb)
    return tot / len(x)


@torch.no_grad()
def evaluate(model, x, y, batch_size=256, prior_seed=0):
    """Deterministic test metrics (same definitions as parity/train_keras_reference.py)."""
    model.eval()
    mus, logvars, rec_mu, rec_prior = [], [], [], []
    z_prior = torch.from_numpy(np.random.default_rng(prior_seed).standard_normal(
        (len(x), model.cfg.latent_dim)).astype(np.float32)).to(x.device)
    for i in range(0, len(x), batch_size):
        xb, yb = x[i:i + batch_size], y[i:i + batch_size]
        mu, logvar = model.encoder(xb, yb)
        mus.append(mu), logvars.append(logvar)
        rec_mu.append(model.decoder(mu, yb))
        rec_prior.append(model.decoder(z_prior[i:i + batch_size], yb))
    mu, logvar = torch.cat(mus), torch.cat(logvars)
    kl_dim = (-0.5 * (1 + logvar - mu ** 2 - logvar.exp())).mean(0)
    return {
        "test_recon_zmu": torch.mean((x - torch.cat(rec_mu)) ** 2).item(),
        "test_recon_zprior": torch.mean((x - torch.cat(rec_prior)) ** 2).item(),
        "test_kl_per_dim": kl_dim.tolist(), "test_kl_sum": kl_dim.sum().item(),
        "test_mu_var_per_dim": mu.var(0, unbiased=False).tolist(),
        "active_units": int((mu.var(0, unbiased=False) > 0.01).sum().item()),
    }


def train(cfg: TrainConfig, data_path, out_dir, seed=0, device=None, verbose=True):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = cfg.allow_tf32
    torch.manual_seed(seed)
    np.random.seed(seed)

    # ---- data (kept on the device; it is small)
    ds = load_dataset(data_path)
    R = ds.R if cfg.freq_slice is None else ds.R[:, cfg.freq_slice[0]:cfg.freq_slice[1]]
    split = reference_split(len(ds.geo), cfg.test_fraction, cfg.val_fraction, cfg.split_seed)
    scaler = fit_geo_scaler(ds.geo, None if cfg.scaler_fit == "all" else split.train)
    geo = scaler.transform(ds.geo).astype(np.float32)
    to = lambda a: torch.from_numpy(a).to(device)
    x_tr, y_tr = to(geo[split.train]), to(R[split.train])
    x_va, y_va = to(geo[split.val]), to(R[split.val])
    x_te, y_te = to(geo[split.test]), to(R[split.test])

    # ---- model, optimiser, callbacks
    model = GeometryCVAE(CVAEConfig(n_freq=R.shape[1], latent_dim=cfg.latent_dim, n_blocks=cfg.n_blocks,
                                    shared_eps=cfg.shared_eps)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, eps=cfg.adam_eps)
    reduce_lr = ReduceLROnPlateau(opt, cfg.reduce_lr_factor, cfg.reduce_lr_patience, cfg.reduce_lr_min_delta)
    early = EarlyStopping(model, cfg.early_stop_patience, restore_best=cfg.restore_best)
    gen = torch.Generator(device=device).manual_seed(seed)

    history, epoch_total, t0 = [], 0, time.time()
    for rnd, (batch_size, epochs) in enumerate(cfg.rounds, start=1):
        for ep in range(epochs):
            t_ep = time.time()
            model.train()
            perm = torch.randperm(len(x_tr), device=device, generator=gen)  # Keras fit(shuffle=True)
            tot = np.zeros(3)
            for i in range(0, len(perm), batch_size):
                idx = perm[i:i + batch_size]
                xb, yb = x_tr[idx], y_tr[idx]
                pred, mu, logvar = model(xb, yb)
                loss, recon, kl = reference_loss(xb, pred, mu, logvar, cfg.beta)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                tot += np.array([loss.item(), recon.item(), kl.item()]) * len(idx)
            tr = tot / len(perm)                      # running mean over the epoch, as Keras reports it
            va = epoch_eval(model, x_va, y_va, cfg.beta, batch_size)
            epoch_total += 1
            lr_used = reduce_lr.lr
            reduced = reduce_lr.step(va[0])
            stop = early.step(va[0], ep, epoch_total)
            history.append(dict(epoch=epoch_total, round=rnd, batch_size=batch_size, loss=tr[0], reconstr_loss=tr[1],
                                kl_loss=tr[2], val_loss=va[0], val_reconstr_loss=va[1], val_kl_loss=va[2],
                                lr=lr_used, epoch_seconds=time.time() - t_ep))
            if verbose:
                print(f"r{rnd} ep{epoch_total:3d} loss {tr[0]:.4f} rec {tr[1]:.4f} kl {tr[2]:.4f} | "
                      f"val {va[0]:.4f} rec {va[1]:.4f} kl {va[2]:.4f} | lr {lr_used:.2e}"
                      + ("  -> lr reduced" if reduced else "") + ("  -> early stop" if stop else ""), flush=True)
            if stop:
                break  # Keras stops this fit() call; the notebook then still starts the next round
    train_seconds = time.time() - t0

    # ---- outputs
    val_losses = [h["val_loss"] for h in history]
    metrics = {"seed": seed, "epochs": len(history), "train_seconds": train_seconds, "final": history[-1],
               "best_val_loss": float(np.min(val_losses)), "best_val_epoch": int(np.argmin(val_losses)) + 1,
               **evaluate(model, x_te, y_te)}
    with open(out_dir / "history.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0]))
        w.writeheader()
        w.writerows(history)
    json.dump(metrics, open(out_dir / "metrics.json", "w"), indent=1)
    yaml.safe_dump({**asdict(cfg), "seed": seed, "data": str(data_path)}, open(out_dir / "config.yaml", "w"))
    json.dump({"git": _git_hash(), "torch": torch.__version__, "device": str(device),
               "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
               "python": platform.python_version()}, open(out_dir / "env.json", "w"), indent=1)
    torch.save({"model": model.state_dict(), "model_config": asdict(model.cfg),
                "scaler_mean": scaler.mean_, "scaler_scale": scaler.scale_}, out_dir / "model.pt")
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    m = train(load_config(a.config), a.data, a.out, a.seed, a.device, verbose=not a.quiet)
    print(json.dumps({k: m[k] for k in ("epochs", "train_seconds", "best_val_loss", "test_recon_zmu",
                                         "test_recon_zprior", "test_kl_sum", "active_units")}))


if __name__ == "__main__":
    main()
