"""Send re-simulation jobs to the HPC cluster (Iridis) and bring the results back.

    uv run python -m sphp_inverse.hpc submit <inference folder> --mph 2026_03_02_micropillar_resim_v64.mph [--dry-run]
    uv run python -m sphp_inverse.hpc fetch  <inference folder> [--dry-run]
    uv run python -m sphp_inverse.hpc status

Needs passwordless SSH (keys) and a config.json next to pyproject.toml (copy config.example.json).

Transfers use rsync when it is installed. Without it (e.g. plain Windows), uploads fall back to scp and downloads
to a single `ssh ... tar` stream, so every step is still one SSH connection.

Where files go is fixed by the COMSOL models themselves (paths are hard-coded in their methods):
  resim model reads   <remote_dir>/<ID>_params.csv        (remote_dir must be ~/micropillar_resim)
  results are written <one of results_dirs>/<ID>_dataout.csv
"""
from __future__ import annotations

import argparse
import io
import json
import re
import shlex
import shutil
import subprocess
import tarfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
COMSOL_DIR = REPO_ROOT / "comsol"
RESIM_SCRIPT = "js_micropillar_resim.sh"


# ----------------------------------------------------------------------------- config
def load_config(path=None) -> dict:
    path = Path(path) if path else REPO_ROOT / "config.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: copy config.example.json to config.json and fill it in")
    cfg = json.load(open(path))
    for key in ("hostname", "username", "remote_dir"):
        if not cfg.get(key) or "your_username" in str(cfg[key]):
            raise ValueError(f"'{key}' is not set in {path}")
    cfg.setdefault("results_dirs", [cfg["remote_dir"]])
    return cfg


def _host(cfg):
    return f"{cfg['username']}@{cfg['hostname']}"


def have_rsync():
    return shutil.which("rsync") is not None


def _run(cmd, dry_run, **kw):
    print("   $ " + " ".join(shlex.quote(str(c)) for c in cmd))
    if dry_run:
        return None
    return subprocess.run([str(c) for c in cmd], check=True, **kw)


# ----------------------------------------------------------------------------- transfers
def push(cfg, files, remote_dir=None, dry_run=False):
    """Copy local files into remote_dir (rsync if available, else scp). One connection."""
    remote_dir = remote_dir or cfg["remote_dir"]
    dest = f"{_host(cfg)}:{remote_dir.rstrip('/')}/"
    files = [Path(f) for f in files]
    if have_rsync():
        return _run(["rsync", "-av", "-e", "ssh", *files, dest], dry_run)
    return _run(["scp", "-p", *files, dest], dry_run)


def fetch_results(cfg, run_dirs, dry_run=False):
    """Download <ID>_dataout.csv for every run folder that does not have it yet.

    Looks in each of cfg['results_dirs'] in turn. Returns {run_id: local path} for the files retrieved."""
    wanted = {}
    for d in map(Path, run_dirs):
        run_id = d.name.replace("run_", "")
        if not (d / f"{run_id}_dataout.csv").exists():
            wanted[run_id] = d
    got = {}
    if not wanted:
        print("nothing to fetch: every run already has its dataout.csv")
        return got
    for remote_dir in cfg["results_dirs"]:
        todo = {rid: d for rid, d in wanted.items() if rid not in got}
        if not todo:
            break
        names = [f"{rid}_dataout.csv" for rid in todo]
        print(f"--> looking for {len(names)} result file(s) in {_host(cfg)}:{remote_dir}")
        tmp = Path(next(iter(todo.values()))).parent / ".fetch_tmp"
        if have_rsync():
            tmp.mkdir(exist_ok=True)
            listing = tmp / "files.txt"
            listing.write_text("\n".join(names) + "\n")
            _run(["rsync", "-av", "-e", "ssh", "--ignore-missing-args", f"--files-from={listing}",
                  f"{_host(cfg)}:{remote_dir.rstrip('/')}/", f"{tmp}/"], dry_run)
            if not dry_run:
                for rid, d in todo.items():
                    src = tmp / f"{rid}_dataout.csv"
                    if src.exists():
                        shutil.move(str(src), d / src.name)
                        got[rid] = d / src.name
                shutil.rmtree(tmp, ignore_errors=True)
        else:
            # no rsync: one ssh connection streaming a tar of the files that exist
            remote = (f"cd {shlex.quote(remote_dir)} && for f in {' '.join(map(shlex.quote, names))}; "
                      f"do [ -f \"$f\" ] && echo \"$f\"; done | tar cf - -T -")
            res = _run(["ssh", _host(cfg), remote], dry_run, capture_output=True)
            if res is not None and res.stdout:
                try:
                    tar = tarfile.open(fileobj=io.BytesIO(res.stdout))
                except tarfile.ReadError:   # empty archive: none of the files exist there yet
                    continue
                with tar:
                    for member in tar.getmembers():
                        name = Path(member.name).name
                        rid = name.replace("_dataout.csv", "")
                        if rid in todo and member.isfile():
                            (todo[rid] / name).write_bytes(tar.extractfile(member).read())
                            got[rid] = todo[rid] / name
    missing = sorted(set(wanted) - set(got))
    if not dry_run:
        print(f"retrieved {len(got)} file(s)" + (f"; not found yet: {', '.join(missing)}" if missing else ""))
    return got


# ----------------------------------------------------------------------------- jobs
def comsol_files(with_models=False):
    """Job scripts (and optionally the local .mph models), so the cluster runs what is in this repo."""
    return sorted(COMSOL_DIR.glob("*.sh")) + (sorted(COMSOL_DIR.glob("*.mph")) if with_models else [])


def submit_resim(cfg, run_dirs, mph, with_models=False, dry_run=False, script=RESIM_SCRIPT, pause=1.0):
    """Upload the job scripts and every <ID>_params.csv, then submit one re-simulation job per run.

    Two SSH connections in total, however many runs: one transfer, one command that calls sbatch for each ID,
    waiting `pause` seconds after each submission (submitting many jobs with no pause gave errors on Iridis).
    Runs that already have results are skipped. Returns {run_id: SLURM job id} (empty in dry-run)."""
    todo = {}
    for d in map(Path, run_dirs):
        run_id = d.name.replace("run_", "")
        if (d / f"{run_id}_dataout.csv").exists():
            print(f"run {run_id}: results already present, not submitting")
            continue
        if not (d / f"{run_id}_params.csv").exists():
            raise FileNotFoundError(d / f"{run_id}_params.csv")
        todo[run_id] = d
    if not todo:
        return {}
    files = comsol_files(with_models) + [d / f"{rid}_params.csv" for rid, d in todo.items()]
    print(f"--> uploading {len(files)} file(s) to {cfg['remote_dir']} (job scripts + parameters for {len(todo)} run(s))")
    push(cfg, files, dry_run=dry_run)

    sbatch = ["sbatch"]
    if cfg.get("mail_user"):
        sbatch += [f"--mail-user={cfg['mail_user']}", "--mail-type=END,FAIL"]
    sbatch = " ".join(shlex.quote(x) for x in sbatch + [script])
    remote = (f"cd {shlex.quote(cfg['remote_dir'])} && for id in {' '.join(todo)}; "
              f"do {sbatch} \"$id\" {shlex.quote(mph)}; sleep {float(pause):g}; done")
    print(f"--> submitting {len(todo)} job(s), {float(pause):g} s apart")
    res = _run(["ssh", _host(cfg), remote], dry_run, capture_output=True, text=True)
    if res is None:
        return {}
    print("    " + res.stdout.strip().replace(chr(10), chr(10) + "    "))
    job_ids = re.findall(r"Submitted batch job (\d+)", res.stdout)
    jobs = dict(zip(todo, job_ids))
    for rid, d in todo.items():
        json.dump({"run_id": rid, "slurm_job_id": jobs.get(rid), "mph": mph, "script": script,
                   "submitted": time.strftime("%Y-%m-%dT%H:%M:%S"), "remote_dir": cfg["remote_dir"]},
                  open(d / f"{rid}_submission.json", "w"), indent=1)
    if len(job_ids) != len(todo):
        print(f"WARNING: {len(todo)} runs but {len(job_ids)} job ids returned; check `hpc status`")
    return jobs


def queue_status(cfg, dry_run=False):
    res = _run(["ssh", _host(cfg), f"squeue -u {shlex.quote(cfg['username'])}"], dry_run, capture_output=True, text=True)
    if res is not None:
        print(res.stdout)


def _run_dirs(paths):
    out = []
    for p in map(Path, paths):
        out += [p] if p.name.startswith("run_") else sorted(d for d in p.glob("run_*") if d.is_dir())
    if not out:
        raise FileNotFoundError(f"no run_* folders in {paths}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("submit", help="upload parameters and submit re-simulation jobs")
    s.add_argument("paths", nargs="+", help="run_<ID> folders, or a folder containing them")
    s.add_argument("--mph", required=True, help="resim model file name (in remote_dir), full-range or reduced-range")
    s.add_argument("--sync-models", action="store_true", help="also upload the local .mph files")
    s.add_argument("--pause", type=float, default=1.0, help="seconds to wait between job submissions (default 1)")
    s.add_argument("--dry-run", action="store_true", help="print the commands without connecting")
    f = sub.add_parser("fetch", help="download <ID>_dataout.csv for runs that do not have it yet")
    f.add_argument("paths", nargs="+")
    f.add_argument("--dry-run", action="store_true")
    st = sub.add_parser("status", help="show your SLURM queue")
    st.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.config)
    print(f"cluster: {_host(cfg)}   transfer: {'rsync' if have_rsync() else 'scp / ssh+tar (rsync not found)'}")
    if a.cmd == "submit":
        submit_resim(cfg, _run_dirs(a.paths), a.mph, with_models=a.sync_models, dry_run=a.dry_run, pause=a.pause)
    elif a.cmd == "fetch":
        fetch_results(cfg, _run_dirs(a.paths), dry_run=a.dry_run)
    else:
        queue_status(cfg, dry_run=a.dry_run)


if __name__ == "__main__":
    main()
