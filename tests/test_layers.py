"""Unit tests for the Keras-compatible building blocks (no reference export needed)."""
import torch

from sphp_inverse.models.cvae import CVAEConfig, GeometryCVAE, SameConv1d, keras_conv_out_len


def test_same_padding_follows_keras_rule():
    """Keras 'same': out = ceil(L/s), extra padding goes on the RIGHT. Matters for even lengths
    (e.g. the reduced-range spectra: 75 -> 38 -> 19 -> 10 -> 5), not for the full 161-point range."""
    conv = SameConv1d(1, 1, 3, stride=2, bias=False)
    torch.nn.init.ones_(conv.weight)
    x = torch.arange(1.0, 11.0).view(1, 1, 10)  # L = 10 -> pad (0, 1)
    out = conv(x).flatten().tolist()
    assert out == [1 + 2 + 3, 3 + 4 + 5, 5 + 6 + 7, 7 + 8 + 9, 9 + 10 + 0]


def test_output_lengths():
    assert keras_conv_out_len(161, 4) == 11
    assert keras_conv_out_len(75, 4) == 5


def test_reduced_range_model_builds():
    m = GeometryCVAE(CVAEConfig(n_freq=75))
    geo_pred, mu, logvar = m(torch.randn(4, 6), torch.rand(4, 75))
    assert geo_pred.shape == (4, 6) and mu.shape == (4, 8)


def test_shared_eps_matches_keras_sampling():
    m = GeometryCVAE()
    mu, logvar = torch.zeros(5, 8), torch.zeros(5, 8)
    z = m.reparameterise(mu, logvar)
    assert torch.allclose(z, z[0].expand_as(z))  # one epsilon for the whole batch, as in the Keras SamplingRNG
    m.cfg.shared_eps = False
    z = m.reparameterise(mu, logvar)
    assert not torch.allclose(z, z[0].expand_as(z))
