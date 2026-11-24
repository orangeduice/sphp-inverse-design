"""Surrogate-guided inverse design on top of the cVAE ("cVAE + surrogate").

The cVAE proposes many candidate geometries for a target (z drawn from its latent prior). A forward surrogate
predicts each candidate's spectrum, optionally refines z by gradient descent (neural-adjoint style, staying in the
decoder's latent space so designs remain on the learned design manifold), and the best n *distinct, valid* designs
are kept. This is the hybrid "optimisation + deep-learning surrogate" route of grant Activity 4.1.

The surrogate used here to choose designs must not be the one used to judge them: final judgement is COMSOL.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from . import geometry


@dataclass
class DesignConfig:
    n_out: int = 5                 # designs returned per target
    n_candidates: int = 2000       # z samples decoded per target
    prior: str = "aggregate"       # "normal" or "aggregate" (fitted posterior mean/covariance)
    refine_steps: int = 0          # gradient steps on z through decoder + surrogate (0 = sampling only)
    refine_top: int = 64           # how many of the best candidates to refine
    refine_lr: float = 0.05
    prior_weight: float = 1e-3     # keeps refined z near the prior: weight * mean(|L^-1 (z - m)|^2)
    min_distance: float = 0.25     # minimum L2 distance between returned designs (scaled geometry units)
    require_gap: bool = True       # only return designs with pillar gap >= 0.5 um (fabrication limit)
    seed: int = 0
    n_conditions: int = 0          # >0: decode with the k nearest *training* spectra as conditions instead of the
                                   # target itself (helps when the target is unlike any dataset spectrum); designs
                                   # are still scored against the target


def _prior(model, cfg, latent_prior):
    d = model.cfg.latent_dim
    if cfg.prior == "aggregate":
        if latent_prior is None:
            raise ValueError("aggregate prior needs latent_prior (mean, cov); see selection.aggregate_posterior")
        m = np.asarray(latent_prior[0], dtype=np.float64)
        L = np.linalg.cholesky(np.asarray(latent_prior[1], dtype=np.float64) + 1e-9 * np.eye(d))
    else:
        m, L = np.zeros(d), np.eye(d)
    return m, L


def design(model, surrogate, scaler, target, mask=None, cfg: DesignConfig = DesignConfig(), latent_prior=None,
           reference_spectra=None):
    """Return (designs_scaled (n_out, 6), z (n_out, d), predicted_mse (n_out,), info dict) for one target spectrum.

    surrogate = (model, mean, scale) from surrogate.load_surrogate. mask restricts the spectral match (e.g. the
    FIG3 window); None = whole spectrum. reference_spectra (N, F): training spectra, needed when n_conditions > 0.
    """
    surr, s_mean, s_scale = surrogate
    device = next(model.parameters()).device
    dt = torch.float32
    m, L = _prior(model, cfg, latent_prior)
    rng = np.random.default_rng(cfg.seed)
    z0 = m + rng.standard_normal((cfg.n_candidates, len(m))) @ L.T

    # geometry scaling chain: cVAE-scaled -> physical -> surrogate-scaled (both StandardScalers, so affine)
    c_mean = torch.tensor(scaler.mean_, dtype=dt, device=device)
    c_scale = torch.tensor(scaler.scale_, dtype=dt, device=device)
    sm = torch.tensor(s_mean, dtype=dt, device=device)
    ss = torch.tensor(s_scale, dtype=dt, device=device)
    tgt = torch.as_tensor(np.asarray(target), dtype=dt, device=device)
    if cfg.n_conditions > 0:
        sel = slice(None) if mask is None else np.asarray(mask)
        ref = np.asarray(reference_spectra)
        near = np.argsort(((ref[:, sel] - np.asarray(target)[sel]) ** 2).mean(1))[: cfg.n_conditions]
        cond_bank = torch.as_tensor(ref[near], dtype=dt, device=device)
        cond_idx = torch.as_tensor(np.arange(cfg.n_candidates) % cfg.n_conditions, device=device)
    else:
        cond_bank, cond_idx = tgt[None], torch.zeros(cfg.n_candidates, dtype=torch.long, device=device)
    msk = torch.ones_like(tgt, dtype=torch.bool) if mask is None else torch.as_tensor(np.asarray(mask), device=device)
    if getattr(model.cfg, "cond_mask", False) and mask is not None:
        # masked-condition model: tell the decoder which part of the spectrum must be matched
        from .models.cvae import masked_condition
        cond_bank = masked_condition(cond_bank, msk.to(dt))
    lo = torch.tensor([b[0] for b in geometry.BOUNDS.values()], dtype=dt, device=device)
    hi = torch.tensor([b[1] for b in geometry.BOUNDS.values()], dtype=dt, device=device)
    Linv = torch.tensor(np.linalg.inv(L), dtype=dt, device=device)
    m_t = torch.tensor(m, dtype=dt, device=device)
    surr = surr.to(device)

    def forward(z, ci):
        g_scaled = model.decoder(z, cond_bank[ci])
        phys = g_scaled * c_scale + c_mean
        spec = surr((phys - sm) / ss)
        err = ((spec - tgt)[:, msk] ** 2).mean(1)
        out_of_box = (torch.relu(lo - phys) + torch.relu(phys - hi)).pow(2).sum(1)
        return g_scaled, phys, err, out_of_box

    model.eval()
    with torch.no_grad():
        z = torch.tensor(z0, dtype=dt, device=device)
        g, phys, err, oob = forward(z, cond_idx)

    if cfg.refine_steps > 0:
        top = torch.argsort(err + 10 * oob)[: cfg.refine_top]
        zr, ci_r = z[top].clone().requires_grad_(True), cond_idx[top]
        opt = torch.optim.Adam([zr], lr=cfg.refine_lr)
        for _ in range(cfg.refine_steps):
            _, _, e, o = forward(zr, ci_r)
            prior_pen = (((zr - m_t) @ Linv.T) ** 2).mean(1)
            loss = (e + 10 * o + cfg.prior_weight * prior_pen).sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            z, cond_idx = torch.cat([z, zr.detach()]), torch.cat([cond_idx, ci_r])
            g, phys, err, oob = forward(z, cond_idx)

    g, phys, err, z = g.cpu().numpy(), phys.cpu().numpy(), err.cpu().numpy(), z.cpu().numpy()
    flags = geometry.check(phys)
    valid = flags["in_bounds"] & flags["aspect_ok"] & (flags["gap_ok"] if cfg.require_gap else True)
    order = [i for i in np.argsort(err) if valid[i]]
    chosen = []
    for i in order:
        if all(np.linalg.norm(g[i] - g[j]) >= cfg.min_distance for j in chosen):
            chosen.append(i)
        if len(chosen) == cfg.n_out:
            break
    if len(chosen) < cfg.n_out:  # not enough distinct valid designs: fill with the next best valid ones
        chosen += [i for i in order if i not in chosen][: cfg.n_out - len(chosen)]
    if len(chosen) < cfg.n_out:  # still short (e.g. plain sampling with no valid sample): best invalid ones,
        # which keep their validity flags in the run metadata
        chosen += [i for i in np.argsort(err) if i not in chosen][: cfg.n_out - len(chosen)]
    chosen = np.array(chosen, dtype=int)
    info = {"n_valid": int(valid.sum()), "n_total": int(len(err)), "best_predicted_mse": float(err[chosen].min())}
    return g[chosen], z[chosen], err[chosen], info
