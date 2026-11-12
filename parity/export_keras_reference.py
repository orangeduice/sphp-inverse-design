"""Export the working Keras 2.15 cVAE as a frozen reference for the PyTorch port.

Runs in its OWN environment. TF 2.15 has no Windows GPU support and can't use the RTX 5070 Ti; CPU is enough here:

    uv run --no-project --python 3.11 --with tensorflow==2.15.0 --with h5py --with scikit-learn \
        python parity/export_keras_reference.py --models <dir with encoder/decoder .h5> --data <dataset.h5> --out <dir>

Writes to --out:
  weights.npz      every layer weight, keyed  <model>/<layer_name>/<i>
  layers.json      per model: layer name, class, config summary, weight shapes (creation order = name suffix)
  batch.npz        a fixed batch (scaled geo, spectra) and Keras outputs on it (inference + BN-training mode)
  split.json       train/val/test indices as Keras itself computes them
  env.json         versions
"""
import argparse
import json
import math
import os
import pickle
import re
import sys

import h5py
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras


class SamplingRNG(keras.layers.Layer):
    """Same as the reference model_architecture.SamplingRNG (needed to deserialise the encoder)."""

    def call(self, inputs):
        z_mean, z_var_log = inputs
        epsilon = keras.backend.random_normal(shape=tf.shape(z_mean)[1:])
        return z_mean + keras.backend.exp(0.5 * z_var_log) * epsilon


def creation_index(name):
    m = re.search(r"_(\d+)$", name)
    return int(m.group(1)) if m else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--latent", type=int, default=8)
    ap.add_argument("--blocks", type=int, default=1)
    ap.add_argument("--n-batch", type=int, default=64)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    tag = f"latent{a.latent}_blocks{a.blocks}"
    enc = keras.models.load_model(os.path.join(a.models, f"encoder_cVAE_geomerty_encoder_{tag}.h5"),
                                  custom_objects={"SamplingRNG": SamplingRNG}, compile=False)
    dec = keras.models.load_model(os.path.join(a.models, f"decoder_cVAE_geomerty_decoder_{tag}.h5"),
                                  custom_objects={"SamplingRNG": SamplingRNG}, compile=False)

    # ---- weights + layer table
    weights, table = {}, {}
    for mname, model in (("encoder", enc), ("decoder", dec)):
        rows = []
        for layer in model.layers:
            ws = layer.get_weights()
            cfg = layer.get_config()
            summ = {k: cfg[k] for k in ("filters", "kernel_size", "strides", "padding", "use_bias", "units",
                                         "momentum", "epsilon", "alpha", "target_shape") if k in cfg}
            rows.append({"name": layer.name, "class": layer.__class__.__name__,
                         "creation_index": creation_index(layer.name), "config": summ,
                         "weight_names": [w.name for w in layer.weights],
                         "weight_shapes": [list(w.shape) for w in ws]})
            for i, w in enumerate(ws):
                weights[f"{mname}/{layer.name}/{i}"] = w
        table[mname] = rows
        with open(os.path.join(a.out, f"{mname}_summary.txt"), "w") as f:
            model.summary(print_fn=lambda s: f.write(s + "\n"))
    np.savez(os.path.join(a.out, "weights.npz"), **weights)
    json.dump(table, open(os.path.join(a.out, "layers.json"), "w"), indent=1)

    # ---- data exactly as helper.load_data_from_h5(input_name='geo', condition_name='R')
    with h5py.File(a.data, "r") as f:
        geo = np.array(f["geo"], dtype=np.float32)
        R = np.array(f["R"], dtype=np.float32)
    scaler = StandardScaler().fit(geo)
    geo_s = scaler.transform(geo)[..., None].astype(np.float32)
    R_c = R[..., None]
    idx = np.arange(len(geo))
    x_tr, x_te, y_tr, y_te, i_tr, i_te = train_test_split(geo_s, R_c, idx, test_size=0.05, random_state=42)

    # Keras' own validation split of the training arrays
    from keras.src.engine import data_adapter
    (i_fit,), (i_val,) = data_adapter.train_validation_split((i_tr,), validation_split=0.3)
    i_fit, i_val = np.asarray(i_fit), np.asarray(i_val)
    assert len(i_fit) == int(math.floor(len(i_tr) * (1.0 - 0.3)))
    json.dump({"train": i_fit.tolist(), "val": i_val.tolist(), "test": i_te.tolist()},
              open(os.path.join(a.out, "split.json"), "w"))

    # ---- fixed batch: first n test samples
    xb, yb = x_te[: a.n_batch], y_te[: a.n_batch]
    rng = np.random.default_rng(0)
    z_fixed = rng.standard_normal((a.n_batch, a.latent)).astype(np.float32)

    mu, logvar, _ = enc([xb, yb], training=False)
    mu_bn_train, logvar_bn_train, _ = enc([xb, yb], training=True)  # BN uses batch statistics; no weight update
    dec_mu = dec([mu, yb], training=False)
    dec_fixed = dec([z_fixed, yb], training=False)
    dec_fixed_bn_train = dec([z_fixed, yb], training=True)

    # reference losses with z = mu (deterministic): same formulas as the notebook
    beta, latent = 0.001, a.latent
    recon = float(np.mean((xb.reshape(-1) - dec_mu.numpy().reshape(-1)) ** 2))
    kl = float(beta * np.mean(-0.5 / latent * np.sum(1 + logvar - mu ** 2 - np.exp(logvar), axis=-1)))

    np.savez(os.path.join(a.out, "batch.npz"),
             x=xb, y=yb, test_idx=i_te[: a.n_batch], z_fixed=z_fixed,
             mu=mu.numpy(), logvar=logvar.numpy(),
             mu_bn_train=mu_bn_train.numpy(), logvar_bn_train=logvar_bn_train.numpy(),
             dec_mu=dec_mu.numpy(), dec_fixed=dec_fixed.numpy(), dec_fixed_bn_train=dec_fixed_bn_train.numpy(),
             recon_loss_zmu=recon, kl_loss=kl)

    json.dump({"python": sys.version, "tensorflow": tf.__version__, "keras": tf.keras.__version__ if hasattr(tf.keras, "__version__") else __import__("keras").__version__,
               "numpy": np.__version__, "models": os.path.abspath(a.models), "data": os.path.abspath(a.data),
               "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist(),
               "recon_loss_zmu": recon, "kl_loss": kl},
              open(os.path.join(a.out, "env.json"), "w"), indent=1)
    print(f"exported {len(weights)} weight arrays; split {len(i_fit)}/{len(i_val)}/{len(i_te)}; "
          f"recon(z=mu)={recon:.5f} kl={kl:.6f} -> {a.out}")


if __name__ == "__main__":
    main()
