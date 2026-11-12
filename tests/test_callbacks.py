"""The callbacks must behave like keras.callbacks.ReduceLROnPlateau / EarlyStopping (Keras 2.15)."""
import torch

from sphp_inverse.callbacks import EarlyStopping, ReduceLROnPlateau


def make():
    model = torch.nn.Linear(1, 1)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    return model, opt


def test_reduce_lr_after_patience_epochs_without_improvement():
    _, opt = make()
    cb = ReduceLROnPlateau(opt, factor=0.66, patience=5, min_delta=1e-4)
    assert cb.step(1.0) is False                      # first value becomes the best
    reduced = [cb.step(1.0) for _ in range(5)]        # no improvement beyond min_delta
    assert reduced == [False, False, False, False, True]
    assert abs(cb.lr - 0.66e-4) < 1e-12
    assert cb.wait == 0


def test_reduce_lr_needs_improvement_larger_than_min_delta():
    _, opt = make()
    cb = ReduceLROnPlateau(opt, patience=2, min_delta=1e-4)
    cb.step(1.0)
    cb.step(1.0 - 5e-5)   # too small to count
    assert cb.wait == 1
    cb.step(0.9)          # real improvement
    assert cb.wait == 0 and cb.best == 0.9


def test_state_persists_across_rounds():
    """The notebook's 'Persistent' callbacks keep wait/best between fit() calls: here it is one object."""
    _, opt = make()
    cb = ReduceLROnPlateau(opt, patience=5)
    for v in (1.0, 1.0, 1.0, 1.0):   # round 1 ends with wait = 3
        cb.step(v)
    assert cb.wait == 3
    assert [cb.step(1.0), cb.step(1.0)] == [False, True]   # round 2 continues the count


def test_early_stopping_restores_best_weights():
    model, _ = make()
    cb = EarlyStopping(model, patience=3, restore_best=True)
    with torch.no_grad():
        model.weight.fill_(1.0)
    assert cb.step(0.5, 0, 1) is False               # best, weights (=1) stored
    with torch.no_grad():
        model.weight.fill_(2.0)
    assert cb.step(0.6, 1, 2) is False
    assert cb.step(0.6, 2, 3) is False
    assert cb.step(0.6, 3, 4) is True                # wait reaches 3 -> stop and restore
    assert model.weight.item() == 1.0 and cb.best_epoch == 1


def test_early_stopping_never_stops_on_first_epoch_of_a_round():
    model, _ = make()
    cb = EarlyStopping(model, patience=1)
    cb.step(0.5, 0, 1)
    assert cb.step(0.9, 0, 2) is False   # epoch_in_round == 0, like Keras' `epoch > 0` condition
    assert cb.step(0.9, 1, 3) is True
