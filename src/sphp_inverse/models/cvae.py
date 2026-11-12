"""Geometry cVAE: PyTorch translation of the working Keras 2.15 model.

Spec: docs/INVENTORY.md §6 (parent folder) and the saved Keras config. Keras conventions that are reproduced
deliberately (each is a known source of silent Keras -> PyTorch differences):

* Tensors are (N, C, L) internally, but every Flatten/Reshape uses Keras' channels-last order (N, L, C).
* Conv1D padding='same' uses Keras' rule (extra padding on the right) via `SameConv1d`.
* BatchNormalization: Keras momentum 0.99 / epsilon 1e-3  ->  PyTorch momentum 0.01 / eps 1e-3.
* Glorot-uniform kernels with zero biases (PyTorch's default is Kaiming-uniform).
* LeakyReLU slope 0.3 inside residual blocks and 0.1 after the encoder Dense layer.
* Module *creation order* matches the Keras layer creation order (shortcut conv first in each residual block),
  so weights can be mapped across frameworks in order (see `load_keras_weights`).
* SamplingRNG in the reference draws ONE epsilon vector per batch (shape = z_mean.shape[1:]) and broadcasts it.
  `shared_eps=True` reproduces that; set False for the standard per-sample reparameterisation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------- building blocks
class SameConv1d(nn.Conv1d):
    """Conv1d with Keras padding='same' semantics (pad_total split as floor left / ceil right)."""

    def __init__(self, in_ch, out_ch, kernel_size, stride=1, bias=True):
        super().__init__(in_ch, out_ch, kernel_size, stride=stride, padding=0, bias=bias)

    def forward(self, x):
        L, k, s = x.shape[-1], self.kernel_size[0], self.stride[0]
        out = math.ceil(L / s)
        pad = max((out - 1) * s + k - L, 0)
        if pad:
            x = F.pad(x, (pad // 2, pad - pad // 2))
        return super().forward(x)


def keras_bn(ch):
    return nn.BatchNorm1d(ch, eps=1e-3, momentum=0.01)


class ResidualBlock(nn.Module):
    """model_architecture.residual_block (kernel 1 -> k -> 1 bottleneck, shortcut conv if shape changes)."""

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, with_bn=False, alpha=0.3):
        super().__init__()
        # creation order mirrors Keras: shortcut conv is built first
        self.shortcut = SameConv1d(in_ch, out_ch, 1, stride) if (in_ch != out_ch or stride != 1) else None
        self.conv1 = SameConv1d(in_ch, out_ch, 1, 1, bias=not with_bn)
        self.bn1 = keras_bn(out_ch) if with_bn else None
        self.conv2 = SameConv1d(out_ch, out_ch, kernel_size, stride, bias=not with_bn)
        self.bn2 = keras_bn(out_ch) if with_bn else None
        self.conv3 = SameConv1d(out_ch, out_ch, 1, 1, bias=not with_bn)
        self.bn3 = keras_bn(out_ch) if with_bn else None
        self.alpha = alpha

    def forward(self, x):
        sc = x if self.shortcut is None else self.shortcut(x)
        h = self.conv1(x)
        if self.bn1 is not None:
            h = self.bn1(h)
        h = F.leaky_relu(h, self.alpha)
        h = self.conv2(h)
        if self.bn2 is not None:
            h = self.bn2(h)
        h = F.leaky_relu(h, self.alpha)
        h = self.conv3(h)
        if self.bn3 is not None:
            h = self.bn3(h)
        return F.leaky_relu(h + sc, self.alpha)


def resblock_sequence_down(in_ch, out_ch, n_blocks):
    """n_blocks BN blocks at stride 1, then one block (no BN) at stride 2."""
    layers, ch = [], in_ch
    for _ in range(n_blocks):
        layers.append(ResidualBlock(ch, out_ch, 3, 1, with_bn=True))
        ch = out_ch
    layers.append(ResidualBlock(ch, out_ch, 3, 2, with_bn=False))
    return nn.Sequential(*layers)


def resblock_sequence_id(in_ch, out_ch, n_blocks):
    layers, ch = [], in_ch
    for _ in range(n_blocks):
        layers.append(ResidualBlock(ch, out_ch, 3, 1, with_bn=True))
        ch = out_ch
    return nn.Sequential(*layers)


def flatten_channels_last(x):
    """(N, C, L) -> (N, L*C) in Keras Flatten order."""
    return x.permute(0, 2, 1).reshape(x.shape[0], -1)


def keras_conv_out_len(L, n_down):
    for _ in range(n_down):
        L = math.ceil(L / 2)
    return L


# ----------------------------------------------------------------------------- model
@dataclass
class CVAEConfig:
    n_geo: int = 6
    n_freq: int = 161
    latent_dim: int = 8
    n_blocks: int = 1
    spec_filters: tuple = (32, 64, 128, 256)
    design_filters: int = 128
    dense_units: int = 256
    shared_eps: bool = True  # reproduce the Keras SamplingRNG (one epsilon per batch)


class SpectrumBranch(nn.Module):
    def __init__(self, cfg: CVAEConfig):
        super().__init__()
        chs = (1,) + tuple(cfg.spec_filters)
        self.blocks = nn.Sequential(*[resblock_sequence_down(chs[i], chs[i + 1], cfg.n_blocks)
                                      for i in range(len(cfg.spec_filters))])
        self.out_features = keras_conv_out_len(cfg.n_freq, len(cfg.spec_filters)) * cfg.spec_filters[-1]

    def forward(self, spectrum):  # (N, F) -> (N, features)
        return flatten_channels_last(self.blocks(spectrum.unsqueeze(1)))


class Encoder(nn.Module):
    def __init__(self, cfg: CVAEConfig):
        super().__init__()
        self.spec = SpectrumBranch(cfg)
        self.design = resblock_sequence_id(1, cfg.design_filters, cfg.n_blocks)
        self.dense = nn.Linear(self.spec.out_features + cfg.n_geo * cfg.design_filters, cfg.dense_units)
        self.z_mean = nn.Linear(cfg.dense_units, cfg.latent_dim)
        self.z_logvar = nn.Linear(cfg.dense_units, cfg.latent_dim)

    def forward(self, geo, spectrum):  # geo (N, 6) scaled, spectrum (N, F)
        h = torch.cat([self.spec(spectrum), flatten_channels_last(self.design(geo.unsqueeze(1)))], dim=1)
        h = F.leaky_relu(self.dense(h), 0.1)
        return self.z_mean(h), self.z_logvar(h)


class Decoder(nn.Module):
    def __init__(self, cfg: CVAEConfig):
        super().__init__()
        self.cfg = cfg
        self.spec = SpectrumBranch(cfg)
        self.dense = nn.Linear(self.spec.out_features + cfg.latent_dim, cfg.n_geo * cfg.design_filters)
        self.design = resblock_sequence_id(cfg.design_filters, cfg.design_filters, cfg.n_blocks)
        self.out = SameConv1d(cfg.design_filters, 1, 1)

    def forward(self, z, spectrum):  # -> (N, 6) scaled geometry
        h = self.dense(torch.cat([self.spec(spectrum), z], dim=1))
        h = h.view(-1, self.cfg.n_geo, self.cfg.design_filters).permute(0, 2, 1)  # Keras Reshape((6, 128))
        return self.out(self.design(h)).squeeze(1)


class GeometryCVAE(nn.Module):
    def __init__(self, cfg: CVAEConfig | None = None):
        super().__init__()
        self.cfg = cfg or CVAEConfig()
        self.encoder = Encoder(self.cfg)
        self.decoder = Decoder(self.cfg)
        self.reset_parameters()

    def reset_parameters(self):
        """Keras defaults: glorot_uniform kernels, zero biases, BN gamma=1 beta=0."""
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def reparameterise(self, mu, logvar):
        shape = mu.shape[1:] if self.cfg.shared_eps else mu.shape
        eps = torch.randn(shape, device=mu.device, dtype=mu.dtype)
        return mu + torch.exp(0.5 * logvar) * eps

    def forward(self, geo, spectrum):
        mu, logvar = self.encoder(geo, spectrum)
        z = self.reparameterise(mu, logvar)
        return self.decoder(z, spectrum), mu, logvar

    @torch.no_grad()
    def sample(self, spectrum, n_samples=1, generator=None):
        """Inverse design: z ~ N(0, I), decode with the target spectrum. Returns (n_samples, N, 6) scaled geometry."""
        self.eval()
        out = []
        for _ in range(n_samples):
            z = torch.randn(spectrum.shape[0], self.cfg.latent_dim, device=spectrum.device, generator=generator)
            out.append(self.decoder(z, spectrum))
        return torch.stack(out)


def reference_loss(geo, geo_pred, mu, logvar, beta=1e-3):
    """Loss exactly as in the Keras notebook: MSE over all elements + beta * KL normalised by latent_dim."""
    recon = F.mse_loss(geo_pred.reshape(-1), geo.reshape(-1))
    latent = mu.shape[1]
    kl = beta * torch.mean(-0.5 / latent * torch.sum(1 + logvar - mu ** 2 - torch.exp(logvar), dim=-1))
    return recon + kl, recon, kl
