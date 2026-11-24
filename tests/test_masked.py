"""Masked conditioning ("match here, ignore elsewhere")."""
import torch

from sphp_inverse.models.cvae import CVAEConfig, GeometryCVAE, masked_condition
from sphp_inverse.train import TrainConfig, random_care_masks


def test_masked_condition_layout():
    y = torch.rand(3, 161)
    m = torch.zeros(3, 161)
    m[:, 40:80] = 1
    c = masked_condition(y, m)
    assert c.shape == (3, 2, 161)
    assert torch.all(c[:, 0, :40] == 0) and torch.allclose(c[:, 0, 40:80], y[:, 40:80])
    assert torch.equal(c[:, 1], m)
    assert torch.equal(masked_condition(y)[:, 1], torch.ones_like(y))   # no mask = care everywhere


def test_masked_model_accepts_plain_and_masked_conditions():
    model = GeometryCVAE(CVAEConfig(cond_mask=True, shared_eps=False)).eval()
    y = torch.rand(4, 161)
    z = torch.randn(4, 8)
    with torch.no_grad():
        plain = model.decoder(z, y)
        full = model.decoder(z, masked_condition(y))
    assert torch.allclose(plain, full)            # a plain spectrum means "whole spectrum matters"


def test_random_care_masks():
    y = torch.rand(500, 161)
    cfg = TrainConfig(cond_mask=True, mask_prob=0.6, mask_width_thz=[0.5, 1.0])
    c = random_care_masks(y, cfg, 0.025, torch.Generator().manual_seed(0))
    care = c[:, 1].sum(1)
    windowed = care < 161
    assert 0.5 < windowed.float().mean() < 0.7                      # about mask_prob get a window
    assert care[windowed].max() <= 1.0 / 0.025 + 2
    # windows away from the spectrum edges keep their full width (near an edge they are clipped)
    inner = windowed & (c[:, 1, 0] == 0) & (c[:, 1, -1] == 0)
    assert care[inner].min() >= 0.5 / 0.025 - 2
    # each window is one contiguous block
    for row in c[windowed][:50, 1]:
        idx = torch.nonzero(row).flatten()
        assert idx.numel() == 0 or idx.max() - idx.min() + 1 == idx.numel()
