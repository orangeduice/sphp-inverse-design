"""Surrogate-guided design (designer.py) and the surrogate itself, on small untrained models (no data needed)."""
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from sphp_inverse import geometry
from sphp_inverse.designer import DesignConfig, design
from sphp_inverse.models import GeometryCVAE
from sphp_inverse.surrogate import ForwardMLP, ForwardResMLP, build, predict_physical


def setup():
    torch.manual_seed(0)
    model = GeometryCVAE().eval()
    rng = np.random.default_rng(0)
    lo = np.array([b[0] for b in geometry.BOUNDS.values()])
    hi = np.array([b[1] for b in geometry.BOUNDS.values()])
    geo = rng.uniform(lo, hi, (500, 6))
    scaler = StandardScaler().fit(geo)
    surr = ForwardMLP(hidden=(32, 32)).eval()
    surrogate = (surr, scaler.mean_, scaler.scale_)
    target = np.linspace(0.9, 0.3, 161).astype(np.float32)
    return model, scaler, surrogate, target


def test_surrogate_shapes_and_build():
    m = build(("res", 64, 2, 0.0), 161)
    assert isinstance(m, ForwardResMLP) and m(torch.zeros(3, 6)).shape == (3, 161)
    assert isinstance(build((32, 32), 161), ForwardMLP)
    _, scaler, surrogate, _ = setup()
    assert predict_physical(*surrogate, np.full((4, 6), 1.0)).shape == (4, 161)


def test_guided_design_returns_distinct_designs_ranked_by_surrogate():
    model, scaler, surrogate, target = setup()
    cfg = DesignConfig(n_out=5, n_candidates=300, prior="normal", min_distance=0.1, require_gap=False, seed=1)
    g, z, err, info = design(model, surrogate, scaler, target, cfg=cfg)
    assert g.shape == (5, 6) and z.shape == (5, 8) and err.shape == (5,)
    assert info["n_total"] == 300
    # chosen designs are the surrogate's best among valid candidates, in ascending order
    assert np.all(np.diff(err) >= -1e-7)
    d = np.linalg.norm(g[:, None] - g[None], axis=-1)[np.triu_indices(5, 1)]
    if info["n_valid"] >= 5:
        assert d.min() >= 0.1 - 1e-6


def test_refinement_does_not_make_the_best_design_worse():
    model, scaler, surrogate, target = setup()
    base = DesignConfig(n_out=5, n_candidates=200, prior="normal", require_gap=False, seed=2)
    _, _, err0, _ = design(model, surrogate, scaler, target, cfg=base)
    refined = DesignConfig(**{**base.__dict__, "refine_steps": 20, "refine_top": 16})
    _, _, err1, info = design(model, surrogate, scaler, target, cfg=refined)
    assert info["n_total"] == 216            # 200 samples + 16 refined
    assert err1.min() <= err0.min() + 1e-7


def test_nearest_spectrum_conditioning_and_fallback():
    model, scaler, surrogate, target = setup()
    ref = np.random.default_rng(3).uniform(0, 1, (50, 161)).astype(np.float32)
    cfg = DesignConfig(n_out=5, n_candidates=40, prior="normal", n_conditions=4, require_gap=False, seed=3)
    g, _, _, _ = design(model, surrogate, scaler, target, cfg=cfg, reference_spectra=ref)
    assert g.shape == (5, 6)
    # impossible validity (every design rejected): still returns n_out designs instead of failing
    impossible = DesignConfig(n_out=5, n_candidates=5, prior="normal", require_gap=True, min_distance=100.0, seed=4)
    old = geometry.MIN_GAP
    try:
        g, _, _, _ = design(model, surrogate, scaler, target, cfg=impossible)
    finally:
        geometry.MIN_GAP = old
    assert g.shape == (5, 6)
