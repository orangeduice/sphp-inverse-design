"""Re-train the original Keras 2.15 geometry cVAE, as a training reference for the PyTorch port.

The model, loss, callbacks and training loop are copied from the working notebook
(cVAE_geomerty_train_IMPORTANT.ipynb) and its model_architecture.py / helper.py. Only the seed handling,
the output paths and the final evaluation are new. Runs on CPU in its own environment:

    uv run --no-project --python 3.11 --with tensorflow==2.15.0 --with h5py --with scikit-learn \
        python parity/train_keras_reference.py --data <dataset.h5> --out ../runs/keras_train_seed0 --seed 0
"""
import argparse
import json
import os
import time

import h5py
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


# ----------------------------------------------------------------------------- model_architecture.py (verbatim)
def resblock_sequence_down(x_in, N_filter, N_blocks):
    x = x_in
    for i in range(N_blocks):
        x = residual_block(x, N_filter, kernel_size=3, strides=1, with_BN=True)
    x = residual_block(x, N_filter, kernel_size=3, strides=2)
    return x


def resblock_sequence_id(x_in, N_filter, N_blocks):
    x = x_in
    for i in range(N_blocks):
        x = residual_block(x, N_filter, kernel_size=3, strides=1, with_BN=True)
    return x


def residual_block(x_in, N_filter, kernel_size=3, strides=1,
                   conv_layer=keras.layers.Conv1D, alpha=0.3,
                   with_BN=False):
    if x_in.shape[-1] != N_filter or strides != 1:
        conv_shortcut = conv_layer(
            filters=N_filter, kernel_size=1, strides=strides, padding='same')(x_in)
    else:
        conv_shortcut = x_in
    x = x_in
    x = conv_layer(filters=N_filter, kernel_size=1, strides=1,
                   padding='same', use_bias=not with_BN)(x)
    if with_BN:
        x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU(alpha)(x)
    x = conv_layer(filters=N_filter, kernel_size=kernel_size,
                   strides=strides, padding='same', use_bias=not with_BN)(x)
    if with_BN:
        x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU(alpha)(x)
    x = conv_layer(filters=N_filter, kernel_size=1, strides=1,
                   padding='same', use_bias=not with_BN)(x)
    if with_BN:
        x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Add()([x, conv_shortcut])
    x = keras.layers.LeakyReLU(alpha)(x)
    return x


class SamplingRNG(keras.layers.Layer):
    def call(self, inputs):
        z_mean, z_var_log = inputs
        epsilon = keras.backend.random_normal(shape=tf.shape(z_mean)[1:])
        z = z_mean + keras.backend.exp(0.5 * z_var_log) * epsilon
        return z


# ----------------------------------------------------------------------------- notebook callbacks (verbatim)
class PersistentEarlyStopping(EarlyStopping):
    def __init__(self, *args, **kwargs):
        super(PersistentEarlyStopping, self).__init__(*args, **kwargs)
        self.best = np.Inf if self.monitor_op == np.less else -np.Inf
        self.wait = 0
        self._first_run = True

    def on_train_begin(self, logs=None):
        if self._first_run:
            self.wait = 0
            self.stopped_epoch = 0
            self.best = np.Inf if self.monitor_op == np.less else -np.Inf
            self._first_run = False


class PersistentReduceLROnPlateau(ReduceLROnPlateau):
    def __init__(self, *args, **kwargs):
        super(PersistentReduceLROnPlateau, self).__init__(*args, **kwargs)
        self._first_run = True

    def on_train_begin(self, logs=None):
        if self._first_run:
            super().on_train_begin(logs)
            self._first_run = False
        else:
            self.model.optimizer.lr.assign(self.model.optimizer.lr)


class EpochTimer(keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        self.t = time.time()

    def on_epoch_end(self, epoch, logs=None):
        logs["epoch_seconds"] = time.time() - self.t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    keras.utils.set_random_seed(a.seed)

    # ---- helper.load_data_from_h5(input_name='geo', condition_name='R')
    with h5py.File(a.data) as f_read:
        inputs = np.array(f_read['geo'], dtype=np.float32)
        conditions = np.array(f_read['R'], dtype=np.float32)
    scaler_geo = StandardScaler().fit(inputs)
    inputs = scaler_geo.transform(inputs)
    inputs = np.expand_dims(inputs, -1)
    conditions = np.expand_dims(conditions, -1)
    x_train, x_test, y_train, y_test = train_test_split(
        inputs, conditions, test_size=0.05, random_state=42)

    # ---- notebook cells 2-5
    keras.backend.clear_session()
    latent_dim = 8
    N_blocks = 1

    condition_target_spec_input = keras.layers.Input(shape=y_train.shape[1:], name='condition_in')
    design_in = keras.Input(shape=x_train.shape[1:], name='design_in')

    x = condition_target_spec_input
    x = resblock_sequence_down(x, N_filter=32, N_blocks=N_blocks)
    x = resblock_sequence_down(x, N_filter=64, N_blocks=N_blocks)
    x = resblock_sequence_down(x, N_filter=128, N_blocks=N_blocks)
    x = resblock_sequence_down(x, N_filter=256, N_blocks=N_blocks)
    x_spec = keras.layers.Flatten()(x)

    x = design_in
    x = resblock_sequence_id(x, N_filter=128, N_blocks=N_blocks)
    x_design = keras.layers.Flatten()(x)

    x = keras.layers.Concatenate()([x_spec, x_design])
    x = keras.layers.Dense(256)(x)
    x = keras.layers.LeakyReLU(alpha=1e-1)(x)

    z_mean = keras.layers.Dense(latent_dim)(x)
    z_var_log = keras.layers.Dense(latent_dim)(x)
    z = SamplingRNG()([z_mean, z_var_log])
    encoder = keras.Model([design_in, condition_target_spec_input], [z_mean, z_var_log, z], name="encoder")

    latent_in = keras.Input(shape=(latent_dim,), name='latent_in')
    x = condition_target_spec_input
    x = resblock_sequence_down(x, N_filter=32, N_blocks=N_blocks)
    x = resblock_sequence_down(x, N_filter=64, N_blocks=N_blocks)
    x = resblock_sequence_down(x, N_filter=128, N_blocks=N_blocks)
    x = resblock_sequence_down(x, N_filter=256, N_blocks=N_blocks)
    x_spec = keras.layers.Flatten()(x)
    x = keras.layers.Concatenate()([x_spec, latent_in])
    x = keras.layers.Dense(6*128)(x)
    x = keras.layers.Reshape((6, 128))(x)
    x = resblock_sequence_id(x, N_filter=128, N_blocks=N_blocks)
    out_design = keras.layers.Conv1D(filters=x_train.shape[-1], kernel_size=1, name='design_out')(x)
    decoder = keras.Model([latent_in, condition_target_spec_input], out_design, name="decoder")

    z_mean_out, z_var_log_out, z_rnd_out = encoder([design_in, condition_target_spec_input])
    design_pred = decoder([z_rnd_out, condition_target_spec_input])
    cVAE = keras.Model([design_in, condition_target_spec_input], design_pred, name="cVAE")

    beta = 0.001
    reconstruction_loss = keras.losses.mse(keras.backend.flatten(design_in), keras.backend.flatten(design_pred))
    cVAE.add_loss(reconstruction_loss)
    cVAE.add_metric(reconstruction_loss, name='reconstr_loss', aggregation='mean')
    kl_loss = keras.backend.sum(
        1 + z_var_log_out - keras.backend.square(z_mean_out) - keras.backend.exp(z_var_log_out), axis=-1)
    kl_loss = beta * keras.backend.mean(-0.5 / latent_dim * kl_loss)
    cVAE.add_loss(kl_loss)
    cVAE.add_metric(kl_loss, name='kl_loss', aggregation='mean')

    # ---- notebook cell 7
    lr = 0.0001
    optimizer = keras.optimizers.AdamW(learning_rate=lr)
    cVAE.compile(optimizer=optimizer)
    reduce_lr = PersistentReduceLROnPlateau(monitor='val_loss', factor=0.66, patience=5, verbose=1)
    early_stopping = PersistentEarlyStopping(monitor='val_loss', patience=15, verbose=1, restore_best_weights=True)

    hist = None
    t0 = time.time()
    for i, o in zip([1, 2], [1, 0.5]):
        print(f"\n--- Starting Training Round {i+1} with batch_size={32 * 2**i} ---", flush=True)
        _h = cVAE.fit(
            x=[x_train, y_train], y=[x_train],
            validation_split=.3,
            batch_size=32 * 2**i,
            epochs=int(25*o),
            callbacks=[reduce_lr, early_stopping, EpochTimer()],
            verbose=2,
        )
        if hist is None:
            hist = {k: list(map(float, v)) for k, v in _h.history.items()}
        else:
            for k in hist:
                hist[k] += list(map(float, _h.history[k]))
    train_seconds = time.time() - t0

    # ---- new: deterministic evaluation on the held-out test set (inference mode)
    mu, logvar, _ = encoder.predict([x_test, y_test], batch_size=256, verbose=0)
    rec_mu = decoder.predict([mu, y_test], batch_size=256, verbose=0)
    rng = np.random.default_rng(0)
    z_prior = rng.standard_normal(mu.shape).astype(np.float32)
    rec_prior = decoder.predict([z_prior, y_test], batch_size=256, verbose=0)
    kl_dim = np.mean(-0.5 * (1 + logvar - mu ** 2 - np.exp(logvar)), axis=0)
    metrics = {
        "seed": a.seed, "epochs": len(hist["loss"]), "train_seconds": train_seconds,
        "final": {k: v[-1] for k, v in hist.items()},
        "best_val_loss": float(np.min(hist["val_loss"])), "best_val_epoch": int(np.argmin(hist["val_loss"])) + 1,
        "test_recon_zmu": float(np.mean((x_test - rec_mu) ** 2)),
        "test_recon_zprior": float(np.mean((x_test - rec_prior) ** 2)),
        "test_kl_per_dim": kl_dim.tolist(), "test_kl_sum": float(kl_dim.sum()),
        "test_mu_var_per_dim": np.var(mu, axis=0).tolist(),
        "active_units": int(np.sum(np.var(mu, axis=0) > 0.01)),
        "tensorflow": tf.__version__,
    }
    json.dump(hist, open(os.path.join(a.out, "history.json"), "w"))
    json.dump(metrics, open(os.path.join(a.out, "metrics.json"), "w"), indent=1)
    encoder.save(os.path.join(a.out, "encoder_cVAE_geomerty_encoder_latent8_blocks1.h5"))
    decoder.save(os.path.join(a.out, "decoder_cVAE_geomerty_decoder_latent8_blocks1.h5"))
    print(json.dumps({k: metrics[k] for k in ("epochs", "train_seconds", "best_val_loss", "test_recon_zmu",
                                               "test_recon_zprior", "test_kl_sum", "active_units")}))


if __name__ == "__main__":
    main()
