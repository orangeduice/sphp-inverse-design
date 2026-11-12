"""Inference -> COMSOL files -> submission commands -> scoring (no cluster or COMSOL needed)."""
import json
from datetime import date
from pathlib import Path

import numpy as np
import pytest
import torch

from sphp_inverse import geometry, hpc, resim
from sphp_inverse.data import fit_geo_scaler
from sphp_inverse.ids import decode_id, encode_id
from sphp_inverse.inference import PARAMS_HEADER, load_model, save_inference_run, suggest_designs
from sphp_inverse.models import GeometryCVAE

CFG = {"hostname": "cluster.example.org", "username": "ab1c23", "remote_dir": "/home/ab1c23/micropillar_resim",
       "results_dirs": ["/home/ab1c23/micropillar_resim", "/scratch/ab1c23/out"], "mail_user": ""}


# ----------------------------------------------------------------------------- geometry
def test_gap_uses_radians():
    # 45 degree pillar: gap = cos(45 deg) * (sub - pillar), clearly different from the unrotated value
    g = np.array([[0.8, 1.0, 1.0, 0.5, 0.5, np.pi / 4]])
    q = geometry.derived(g)
    d_sub = np.sin(np.pi / 4) * 1.0 + np.cos(np.pi / 4) * 1.0 + 0.5
    assert np.isclose(q["d_gap"][0], np.cos(np.pi / 4) * (d_sub - 1.0))
    assert np.isclose(q["w_aspect"][0], 0.8)


def test_check_flags():
    ok = [0.8, 1.0, 1.0, 1.0, 1.0, 0.1]
    too_tall = [1.0, 0.45, 1.0, 1.0, 1.0, 0.1]      # outside bounds and aspect > 2
    tight = [0.8, 1.0, 1.0, 0.3, 0.3, 0.0]           # gap 0.3 < 0.5
    f = geometry.check(np.array([ok, too_tall, tight]))
    assert f["in_bounds"].tolist() == [True, False, True]
    assert f["aspect_ok"].tolist() == [True, False, True]
    assert f["gap_ok"].tolist() == [True, True, False]


# ----------------------------------------------------------------------------- ids
def test_id_roundtrip_matches_original_scheme():
    rid = encode_id(1, 0, date(2026, 2, 17))
    assert rid == 602171000                      # first datagen batch id
    assert decode_id(rid) == {"year": 2026, "month": 2, "day": 17, "category": 1, "sequence": 0}
    assert encode_id(9, 999, date(2040, 12, 31)) < 2 ** 31
    with pytest.raises(ValueError):
        encode_id(1, 1000)


# ----------------------------------------------------------------------------- inference files
def test_inference_run_files(tmp_path):
    torch.manual_seed(0)
    model = GeometryCVAE().eval()
    geo = np.random.default_rng(0).uniform([0.5, 0.5, 0.5, 0.25, 0.25, 0], [1, 2, 2, 2, 2, 0.78], (200, 6))
    scaler = fit_geo_scaler(geo)
    target, freq = np.linspace(0.9, 0.2, 161), np.linspace(25, 29, 161)
    designs, z = suggest_designs(model, target, n=5, seed=1)
    assert designs.shape == (5, 6) and z.shape == (5, 8)
    again, _ = suggest_designs(model, target, z=z)
    np.testing.assert_allclose(designs, again)                       # same z -> same designs
    run, flags = save_inference_run(tmp_path, 611125138, designs, scaler, target, freq, {"target_id": "138"})
    params = (run / "611125138_params.csv").read_text().splitlines()
    assert "\n".join(params[:2]) == PARAMS_HEADER                      # the header COMSOL expects
    assert params[0] == "h_pill, d_pill, w_pill, d_sep, w_sep, angle" and params[1] == "um, um, um, um, um, rad"
    assert len(params) == 7
    meta = json.load(open(run / "611125138_metadata.json"))
    assert len(meta["checks"]["in_bounds"]) == 5 and meta["inference_run_id"] == "611125138"
    assert np.loadtxt(run / "611125138_target_spectra.csv", delimiter=",").shape == (161, 2)


def test_reproduces_archived_keras_designs():
    """With the saved reduced-range Keras weights and the stored z samples, the PyTorch decoder must return the
    designs the Keras workflow produced (75-point spectra, so Keras' asymmetric padding is exercised)."""
    root = Path(__file__).resolve().parents[2]
    ref = root / "runs" / "keras_reference_reduced_range"
    runs = root / "archive" / "v3_inverse-design_2026-07" / "keras" / "geomerty_cVAE_reduced_range" / "inference_data" / "results"
    if not (ref / "weights.npz").exists() or not runs.exists():
        pytest.skip("archived reduced-range model/runs not available")
    model, _ = load_model(ref, n_freq=75)
    model = model.double()
    checked = 0
    for d in sorted(runs.glob("run_*")):
        rid = d.name[4:]
        target = np.loadtxt(d / f"{rid}_target_spectra.csv", delimiter=",")[:, 1]
        z = json.load(open(d / f"{rid}_metadata.json"))["inference_params"]["z_samples"]
        designs, _ = suggest_designs(model, target, z=z)
        np.testing.assert_allclose(designs, np.loadtxt(d / f"{rid}_suggested_designs.csv", delimiter=","), atol=2e-3)
        checked += 1
    assert checked >= 5


# ----------------------------------------------------------------------------- scoring
def _write_run(folder, rid, shift):
    run = folder / f"run_{rid}"
    run.mkdir()
    f = np.linspace(25, 29, 161)
    dip = lambda f0: 0.95 - 0.7 * np.exp(-((f - f0) / 0.1) ** 2)
    np.savetxt(run / f"{rid}_target_spectra.csv", np.column_stack([f, dip(27.0)]), delimiter=",")
    rows = []
    for k, f0 in enumerate([27.0 + shift, 27.5]):                     # design 0 is the good one
        g = [0.8, 1.0 + k, 1.0, 1.0, 1.0, 0.1]
        rows += [g + [fi, ri] for fi, ri in zip(f, dip(f0))]
    with open(run / f"{rid}_dataout.csv", "w") as fh:
        fh.write("% Model,x.mph\n% h_pill (um),d_pill (um),w_pill (um),d_sep (um),w_sep (um),angle (rad),freq (THz),Reflection (1)\n")
        np.savetxt(fh, np.array(rows), delimiter=",")
    return run


def test_resim_scoring(tmp_path):
    _write_run(tmp_path, 1, shift=0.0)
    _write_run(tmp_path, 2, shift=0.05)     # 0.05 THz = 1.67 cm^-1
    (tmp_path / "run_3").mkdir()             # no results yet
    scores = resim.score_folder(tmp_path)
    assert [s["run"] for s in scores] == ["1", "2"]
    assert scores[0]["best_mse"] < 1e-12 and scores[0]["best_dip_err_cm"] == 0
    assert np.isclose(scores[1]["best_dip_err_cm"], 0.05 * resim.THZ_TO_CM, atol=1e-6)
    assert scores[1]["n_designs"] == 2 and scores[1]["mean_mse"] > scores[1]["best_mse"]
    assert resim.summarise(scores)["dip_within_4cm"] == 2


# ----------------------------------------------------------------------------- cluster commands (dry run)
@pytest.mark.parametrize("rsync", [True, False])
def test_submit_and_fetch_commands(tmp_path, monkeypatch, capsys, rsync):
    monkeypatch.setattr(hpc, "have_rsync", lambda: rsync)
    run = tmp_path / "run_611125138"
    run.mkdir()
    (run / "611125138_params.csv").write_text(PARAMS_HEADER + "\n")
    assert hpc.submit_resim(CFG, [run], "model_rr.mph", dry_run=True) == {}
    out = capsys.readouterr().out
    assert ("rsync -av -e ssh" in out) == rsync and ("scp -p" in out) == (not rsync)
    assert "ab1c23@cluster.example.org:/home/ab1c23/micropillar_resim/" in out
    assert "js_micropillar_resim.sh" in out and "611125138_params.csv" in out   # scripts + parameters in one transfer
    assert 'cd /home/ab1c23/micropillar_resim && for id in 611125138; do sbatch js_micropillar_resim.sh "$id" model_rr.mph; done' in out
    hpc.fetch_results(CFG, [run], dry_run=True)
    out = capsys.readouterr().out
    assert "611125138_dataout.csv" in out or "--files-from" in out
    assert "/scratch/ab1c23/out" in out                                # falls through to the second results dir
    (run / "611125138_dataout.csv").write_text("x")
    assert hpc.fetch_results(CFG, [run], dry_run=True) == {}           # nothing left to fetch
    assert hpc.submit_resim(CFG, [run], "model_rr.mph", dry_run=True) == {}
    assert "already present" in capsys.readouterr().out


def test_fetch_extracts_tar_stream(tmp_path, monkeypatch):
    """No rsync: results arrive as a tar stream over ssh and land in the matching run folders."""
    import io, tarfile, types
    monkeypatch.setattr(hpc, "have_rsync", lambda: False)
    runs = []
    for rid in ("1", "2"):
        d = tmp_path / f"run_{rid}"
        d.mkdir()
        runs.append(d)
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:   # only run 2 has finished on the "cluster"
        data = b"% header" + bytes([10]) + b"1,2" + bytes([10])
        info = tarfile.TarInfo("2_dataout.csv")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    calls = []

    def fake_run(cmd, dry_run, **kw):
        calls.append(cmd)
        return types.SimpleNamespace(stdout=buf.getvalue() if len(calls) == 1 else b"")

    monkeypatch.setattr(hpc, "_run", fake_run)
    got = hpc.fetch_results(CFG, runs)
    assert list(got) == ["2"] and (tmp_path / "run_2" / "2_dataout.csv").read_bytes().startswith(b"% header")
    assert not (tmp_path / "run_1" / "1_dataout.csv").exists()
    assert len(calls) == 2                             # second results dir tried for the missing run


def test_config_validation(tmp_path):
    p = tmp_path / "config.json"
    p.write_text(json.dumps({"hostname": "h", "username": "your_username", "remote_dir": "/home/your_username/x"}))
    with pytest.raises(ValueError):
        hpc.load_config(p)
    p.write_text(json.dumps({"hostname": "h", "username": "u", "remote_dir": "/home/u/micropillar_resim"}))
    assert hpc.load_config(p)["results_dirs"] == ["/home/u/micropillar_resim"]
