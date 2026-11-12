"""Keras-2.15-compatible training callbacks.

These follow keras.callbacks.ReduceLROnPlateau / EarlyStopping line by line, and keep their state across
several training rounds, like the notebook's PersistentReduceLROnPlateau / PersistentEarlyStopping.
"""
from __future__ import annotations

import copy


class ReduceLROnPlateau:
    """mode='min'. An epoch improves if value < best - min_delta. Otherwise wait += 1 and, at wait >= patience,
    lr *= factor and wait is reset."""

    def __init__(self, optimizer, factor=0.66, patience=5, min_delta=1e-4, cooldown=0, min_lr=0.0):
        self.optimizer, self.factor, self.patience = optimizer, factor, patience
        self.min_delta, self.cooldown, self.min_lr = min_delta, cooldown, min_lr
        self.best, self.wait, self.cooldown_counter = float("inf"), 0, 0

    @property
    def lr(self):
        return self.optimizer.param_groups[0]["lr"]

    def step(self, value) -> bool:
        """Call at the end of each epoch. Returns True if the learning rate was reduced."""
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            self.wait = 0
        if value < self.best - self.min_delta:
            self.best, self.wait = value, 0
        elif self.cooldown_counter <= 0:
            self.wait += 1
            if self.wait >= self.patience:
                old = self.lr
                if old > self.min_lr:
                    new = max(old * self.factor, self.min_lr)
                    for g in self.optimizer.param_groups:
                        g["lr"] = new
                    self.cooldown_counter, self.wait = self.cooldown, 0
                    return True
        return False


class EarlyStopping:
    """mode='min', min_delta=0, as in Keras 2.15: wait is incremented first, then reset on improvement;
    stop when wait >= patience (never on the first epoch of a fit call). With restore_best, the best weights are
    restored only when training is actually stopped early (Keras 2.15 behaviour)."""

    def __init__(self, model, patience=15, min_delta=0.0, restore_best=True):
        self.model, self.patience, self.min_delta, self.restore_best = model, patience, min_delta, restore_best
        self.best, self.wait, self.best_state, self.best_epoch = float("inf"), 0, None, 0

    def step(self, value, epoch_in_round: int, epoch_total: int) -> bool:
        """Returns True if training should stop."""
        self.wait += 1
        if value < self.best - self.min_delta:
            self.best, self.best_epoch, self.wait = value, epoch_total, 0
            if self.restore_best:
                self.best_state = copy.deepcopy(self.model.state_dict())
            return False
        if self.wait >= self.patience and epoch_in_round > 0:
            if self.restore_best and self.best_state is not None:
                self.model.load_state_dict(self.best_state)
            return True
        return False
