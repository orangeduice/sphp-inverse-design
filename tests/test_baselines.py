"""Brute-force baseline (report Fig. 3), dip targets and dip metrics."""
import json
from pathlib import Path

import numpy as np
import pytest

from sphp_inverse.baselines import brute_force_search, run_bruteforce, save_bruteforce
from sphp_inverse.data import Dataset, load_dataset
from sphp_inverse.metrics import dip_metrics
from sphp_inverse.targets import FIG3, gaussian_dip

REFERENCE_CSV = Path(__file__).parent / "data" / "fig3_reference.csv"  # best_geometries.csv from the original run
# Rows picked by the original notebook (archive/v3_inverse-design_2026-07/keras/data_looker.ipynb, cell 12)
FIG3_ROWS = [3282, 4951, 7855, 5033, 430, 3005, 8505, 2105, 5201, 1222, 7421, 417, 3008, 8528, 46, 9480, 1072, 4170]
FREQ = np.linspace(25, 29, 161)


# ----------------------------------------------------------------------------- targets / metrics
def test_fig3_target_set():
    assert np.allclose(FIG3.centres()[[0, -1]], [25.975, 27.625]) and len(FIG3.centres()) == 18
    w = FIG3.window(FREQ.astype(np.float32))
    assert np.flatnonzero(w)[[0, -1]].tolist() == [35, 109]          # the original [35:110] slice
    t = FIG3.spectra(FREQ)
    assert t.shape == (18, 161) and np.isclose(t.min(), 0.15, atol=1e-3) and np.isclose(t.max(), 0.95)


def test_dip_metrics_recovers_gaussian_fwhm():
    f = np.linspace(25, 29, 4001)
    m = dip_metrics(f, gaussian_dip(f, 26.5, fwhm=0.3, depth=0.6, baseline=0.95))
    assert np.isclose(m.freq, 26.5) and np.isclose(m.prominence, 0.6, atol=1e-3)
    assert np.isclose(m.fwhm, 0.3, atol=2e-3) and m.left < 26.5 < m.right
    assert np.isclose(m.half_height, 0.95 - 0.3, atol=1e-3)


def test_dip_metrics_region_picks_the_dip_inside_it():
    R = gaussian_dip(FREQ, 26.0, 0.2, 0.8) * gaussian_dip(FREQ, 28.0, 0.2, 0.5)
    assert np.isclose(dip_metrics(FREQ, R).freq, 26.0)
    assert np.isclose(dip_metrics(FREQ, R, FREQ > 27).freq, 28.0)


# ----------------------------------------------------------------------------- search (synthetic)
def test_brute_force_finds_planted_spectrum():
    rng = np.random.default_rng(0)
    R = rng.uniform(0.5, 1.0, (500, 161))
    targets = FIG3.spectra(FREQ)[[3, 10]]
    R[123], R[400] = targets[0], targets[1] + 1e-3
    rows, err = brute_force_search(R, targets, FIG3.window(FREQ), top_k=3)
    assert rows[:, 0].tolist() == [123, 400] and err[0, 0] == 0 and np.isclose(err[1, 0], 1e-6)
    assert np.all(np.diff(err, axis=1) >= 0)


def test_brute_force_ties_go_to_first_row():
    R = np.ones((4, 161)) * 0.5
    rows, _ = brute_force_search(R, np.ones(161), top_k=2)
    assert rows.tolist() == [[0, 1]]


def test_run_and_save(tmp_path):
    rng = np.random.default_rng(1)
    centres = FIG3.centres()
    R = np.stack([gaussian_dip(FREQ, rng.choice(centres), 0.25, rng.uniform(0.2, 0.8), baseline=0.95)
                  for _ in range(60)]).astype(np.float32)
    ds = Dataset(geo=rng.uniform(0.5, 1, (60, 6)).astype(np.float32), R=R, freq=FREQ.astype(np.float32))
    records, arrays, summary = run_bruteforce(ds, top_k=2, min_prominence=0.3)
    assert len(records) == 36 and summary["n_targets"] == 18
    assert summary["n_accepted"] == sum(r["rank"] == 1 and r["prominence"] >= 0.3 for r in records)
    out = save_bruteforce(tmp_path, records, arrays, summary)
    assert (out / "candidates.csv").read_text().splitlines()[0].startswith("target,rank,centre_THz,row")
    assert np.load(out / "spectra.npz")["matched"].shape == (18, 161)
    assert json.load(open(out / "summary.json"))["targets"]["n"] == 18


# ----------------------------------------------------------------------------- reproduction of report Fig. 3
@pytest.fixture(scope="module")
def fig3(data_path):
    return load_dataset(data_path), run_bruteforce(load_dataset(data_path))


def test_fig3_rows_match_original(fig3):
    _, (records, _, _) = fig3
    assert [r["row"] for r in records] == FIG3_ROWS


def test_fig3_matches_reference_csv(fig3):
    ds, (records, _, _) = fig3
    ref = np.loadtxt(REFERENCE_CSV, delimiter=",", skiprows=1)
    np.testing.assert_array_equal([r["centre_THz"] for r in records], ref[:, 0])
    np.testing.assert_array_equal(ds.geo[[r["row"] for r in records]], ref[:, 1:].astype(np.float32))


def test_fig3_report_numbers(fig3):
    """Report: mean FWHM 0.2250 THz, mean prominence 0.6643, linewidth ratio 0.05624 (notebook output, full precision)."""
    _, (_, _, s) = fig3
    assert np.isclose(s["mean_fwhm_THz"], 0.2249575548376445, rtol=0, atol=1e-9)
    assert np.isclose(s["mean_prominence"], 0.664251039425532, rtol=0, atol=1e-9)
    assert np.isclose(s["linewidth_ratio"], 0.05623938870941113, rtol=0, atol=1e-10)
