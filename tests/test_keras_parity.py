"""Phase 3a step 3: architecture parity with the Keras 2.15 reference.

Needs the export from parity/export_keras_reference.py (default ../runs/keras_reference, or set SPHP_KERAS_REF).
"""
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from sphp_inverse.keras_import import load_keras_weights
from sphp_inverse.models import GeometryCVAE, reference_loss

REF = Path(os.environ.get("SPHP_KERAS_REF", Path(__file__).resolve().parents[2] / "runs" / "keras_reference"))
pytestmark = pytest.mark.skipif(not (REF / "weights.npz").exists(), reason=f"no Keras reference export at {REF}")

ATOL = 1e-4


@pytest.fixture(scope="module")
def ref():
    return np.load(REF / "batch.npz")


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    m = GeometryCVAE().double()  # float64 so differences come from the model, not from rounding
    load_keras_weights(m, REF)
    return m


def t(a):
    return torch.from_numpy(np.asarray(a, dtype=np.float64))


def test_parameter_counts_match_keras():
    m = GeometryCVAE()
    summ = {p: (REF / f"{p}_summary.txt").read_text() for p in ("encoder", "decoder")}
    for part in ("encoder", "decoder"):
        mod = getattr(m, part)
        n = sum(p.numel() for p in mod.parameters()) + sum(b.numel() for b in mod.buffers() if b.dtype.is_floating_point)
        line = [l for l in summ[part].splitlines() if l.startswith("Total params")][0]
        assert n == int(line.split(":")[1].split("(")[0].replace(",", "").strip()), part


def test_encoder_inference_mode(model, ref):
    model.eval()
    with torch.no_grad():
        mu, logvar = model.encoder(t(ref["x"][..., 0]), t(ref["y"][..., 0]))
    np.testing.assert_allclose(mu.numpy(), ref["mu"], atol=ATOL)
    np.testing.assert_allclose(logvar.numpy(), ref["logvar"], atol=ATOL)


def test_decoder_inference_mode(model, ref):
    model.eval()
    with torch.no_grad():
        out_mu = model.decoder(t(ref["mu"]), t(ref["y"][..., 0]))
        out_fixed = model.decoder(t(ref["z_fixed"]), t(ref["y"][..., 0]))
    np.testing.assert_allclose(out_mu.numpy(), ref["dec_mu"][..., 0], atol=ATOL)
    np.testing.assert_allclose(out_fixed.numpy(), ref["dec_fixed"][..., 0], atol=ATOL)


def test_batchnorm_training_mode(model, ref):
    """BN with batch statistics (training=True in Keras). Running stats are restored afterwards."""
    state = {k: v.clone() for k, v in model.state_dict().items()}
    model.train()
    with torch.no_grad():
        mu, logvar = model.encoder(t(ref["x"][..., 0]), t(ref["y"][..., 0]))
        dec = model.decoder(t(ref["z_fixed"]), t(ref["y"][..., 0]))
    model.load_state_dict(state)
    model.eval()
    np.testing.assert_allclose(mu.numpy(), ref["mu_bn_train"], atol=ATOL)
    np.testing.assert_allclose(logvar.numpy(), ref["logvar_bn_train"], atol=ATOL)
    np.testing.assert_allclose(dec.numpy(), ref["dec_fixed_bn_train"][..., 0], atol=ATOL)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA GPU")
def test_float32_gpu_without_tf32(ref):
    """float32 on the GPU matches Keras only with TF32 off (cuDNN enables TF32 convolutions by default,
    which gives ~1e-3 relative differences: fine for training, too loose for parity checks)."""
    old = torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32
    torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = False
    try:
        m = GeometryCVAE().cuda()
        load_keras_weights(m, REF)
        m.eval()
        with torch.no_grad():
            mu, _ = m.encoder(torch.tensor(ref["x"][..., 0]).cuda(), torch.tensor(ref["y"][..., 0]).cuda())
        np.testing.assert_allclose(mu.cpu().numpy(), ref["mu"], atol=1e-4)
    finally:
        torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = old


def test_loss_matches_keras_formula(model, ref):
    model.eval()
    with torch.no_grad():
        x, y = t(ref["x"][..., 0]), t(ref["y"][..., 0])
        mu, logvar = model.encoder(x, y)
        _, recon, kl = reference_loss(x, model.decoder(mu, y), mu, logvar, beta=1e-3)
    assert np.isclose(recon.item(), float(ref["recon_loss_zmu"]), rtol=1e-4)
    assert np.isclose(kl.item(), float(ref["kl_loss"]), rtol=1e-4)


def test_split_matches_keras():
    from sphp_inverse.data import reference_split

    k = json.load(open(REF / "split.json"))
    s = reference_split(9800)
    for name in ("train", "val", "test"):
        np.testing.assert_array_equal(getattr(s, name), np.array(k[name]))
