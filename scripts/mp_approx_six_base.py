"""MP-Approx six-dataset base, track MP-ORACLE (configs/mp_approx_six_base.yaml): the matched non-message-passing twin
of universal-v2 stage 2's joint six-dataset GNN, fitted under stage 2's frozen training rule, then (work 2, its own
later commit) level 0's scoring pass on the six datasets.

    python scripts/mp_approx_six_base.py --stage fit --seed 0 --threads 8   # work 1: one joint twin fit (laptop CPU)
    python scripts/mp_approx_six_base.py --stage fits                       # work 1: seeds 0, 1, 2 in order, each a fresh process
    python scripts/mp_approx_six_base.py --stage status                     # what exists, the fit-hours spent

An analysis base only: the twin is the ladder's reference model, as the trio twin was at level 0, and it is not a
Universal-MLP candidate. universal_v2_run.py, universal_v2_six.py and the M3B code are imported and never edited: the
contract, the carves, the model, the sampler, the training rule, the loss and the checkpointing are theirs. What this
file adds is the arm (u_mlp_v2_mix in place of stage 2's u_gnn_v2_ef), its own output directory, its own parameter pin
and its own fit-hour ceiling.
"""

from __future__ import annotations

import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):   # as universal_v2_six.py, before numpy loads
    os.environ.setdefault(_var, os.environ.get("M3B_BLAS_THREADS", "2"))

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from dataclasses import asdict  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402
import yaml  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "src", ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import universal_v2_run as V2  # noqa: E402  (imported, never edited)
import universal_v2_six as U6  # noqa: E402  (imported, never edited)
from mp_retrieval.m3b_models import parameter_count  # noqa: E402
from mp_retrieval.m3b_train import fit_model  # noqa: E402

CONFIG = ROOT / "configs" / "mp_approx_six_base.yaml"
OUT = ROOT / "outputs" / "mp_approx_six_base"
FITS = OUT / "fits"
ARM = "u_mlp_v2_mix"
PARAMETERS = 330955
SEEDS = (0, 1, 2)
FIT_THREADS = 8
CEILING_HOURS = 45.0
FIRST_PROJECTION_HOURS = 15.0
FIT_CONFIG = {"H": V2.HIDDEN, "stage": "six", "file": "mp_approx_six_base"}
DATASETS = U6.DATASETS


def log_utc(msg: str) -> None:
    print(f"[{V2.utc()}] {msg}", flush=True)


def lf_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_declaration(path: Path = CONFIG) -> dict:
    decl = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if decl.get("phase") != "MP_APPROX_SIX_BASE":
        raise SystemExit(f"{path}: not the six-dataset base declaration")
    w1 = decl["work_1_twin_fits"]
    if w1["arm"] != ARM or int(w1["parameters"]) != PARAMETERS or list(w1["seeds"]) != list(SEEDS):
        raise SystemExit("work_1_twin_fits: the arm, the parameter pin or the seeds differ from this script's; refusing")
    return decl


def pin_problems(decl: dict, root: Path = ROOT) -> list[str]:
    """inputs: the frozen code (LF), the stage-2 checkpoints, the six caches' meta.json and the contract files."""
    inp, bad = decl["inputs"], []
    for rel, want in inp["frozen_code_lf"].items():
        p = root / rel
        if not p.exists() or lf_sha256(p) != want:
            bad.append(rel)
    for key, pins in inp["gnn_checkpoints"].items():
        for kind, suffix in (("weights", ".pt"), ("record", ".json")):
            p = root / "outputs" / "universal_v2" / "six" / "fits" / f"{key}{suffix}"
            if not p.exists() or sha256_file(p) != pins[kind]:
                bad.append(str(p.relative_to(root)))
    for name, kinds in inp["caches"].items():
        for kind, want in kinds.items():
            p = root / "outputs" / "universal_v2" / "cache" / name / kind / "meta.json"
            if not p.exists() or sha256_file(p) != want:
                bad.append(str(p.relative_to(root)))
    for spec in inp["contract"].values():
        p = root / spec["path"]
        if not p.exists() or sha256_file(p) != spec["sha256"]:
            bad.append(spec["path"])
    return bad


def verify_pins(decl: dict) -> None:
    bad = pin_problems(decl)
    if bad:
        hard_stop(f"pinned inputs differ from configs/mp_approx_six_base.yaml: {bad}")


def hard_stop(message: str, **evidence) -> None:
    """hard_stops: the evidence is written to hard_stop.json and nothing further runs."""
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "hard_stop.json").write_text(json.dumps({"utc": V2.utc(), "message": message, **evidence}, indent=1), encoding="utf-8")
    raise SystemExit(f"HARD STOP: {message}")


def fit_key(seed: int) -> str:
    return f"{ARM}__H{V2.HIDDEN}__six__s{seed}"


def module_shas() -> dict:
    """The LF sha256 of every repository module this process imported."""
    out = {}
    for mod in list(sys.modules.values()):
        f = getattr(mod, "__file__", None)
        if not isinstance(f, str) or not os.path.isabs(f):   # torch.ops carries __file__ = "_ops.py", which names no file (amendment 1)
            continue
        p = Path(f).resolve()
        try:
            rel = p.relative_to(ROOT)
        except ValueError:
            continue
        if p.suffix == ".py":
            out[rel.as_posix()] = lf_sha256(p)
    return dict(sorted(out.items()))


def git_head() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:   # the record says so rather than failing a multi-hour fit
        return None


def filed_fits(fits_dir: Path = FITS) -> dict:
    out = {}
    for s in SEEDS:
        p = fits_dir / f"{fit_key(s)}.json"
        if p.exists():
            out[s] = json.loads(p.read_text(encoding="utf-8"))
    return out


def ceiling_guard(seed: int, fits_dir: Path = FITS) -> dict:
    """compute.ceiling: the seconds of the fits already filed plus this fit's projection (the longest filed fit, or 15
    hours before any is filed) must stay within 45 fit-hours; otherwise ceiling_breach.json and no fit starts."""
    filed = filed_fits(fits_dir)
    spent = sum(float(r.get("seconds", 0.0)) for r in filed.values()) / 3600
    projected = max((float(r.get("seconds", 0.0)) for r in filed.values()), default=FIRST_PROJECTION_HOURS * 3600) / 3600
    out = {"seed": seed, "fit_hours_spent": round(spent, 3), "projected_fit_hours": round(projected, 3),
           "ceiling_fit_hours": CEILING_HOURS, "within_ceiling": bool(spent + projected <= CEILING_HOURS)}
    if not out["within_ceiling"]:
        fits_dir.mkdir(parents=True, exist_ok=True)
        (fits_dir / "ceiling_breach.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
        raise SystemExit(f"compute ceiling: {spent:.1f} + {projected:.1f} > {CEILING_HOURS:.0f} fit-hours; seed {seed} not started")
    return out


def build_twin(inputs: dict, bank=None) -> torch.nn.Module:
    """work_1.architecture: the frozen make_model, and the parameter pin."""
    model = V2.make_model(ARM, inputs, bank)
    params = parameter_count(model)
    if params != PARAMETERS:
        hard_stop(f"{ARM}: {params} parameters, the declaration pins {PARAMETERS}")
    return model


def stage_fit(seed: int, log=log_utc) -> dict:
    """work 1, one seed: u_mlp_v2_mix jointly on the six datasets under stage 2's frozen rule, from scratch, resumable
    through fit_model's checkpoint; the record is written once the best weights are saved."""
    if seed not in SEEDS:
        raise SystemExit(f"seed {seed}: the file fits seeds {list(SEEDS)}")
    decl = load_declaration()
    key = fit_key(seed)
    FITS.mkdir(parents=True, exist_ok=True)
    rec_path = FITS / f"{key}.json"
    if rec_path.exists():
        log(f"   {key}: record exists, not repeated")
        return json.loads(rec_path.read_text(encoding="utf-8"))
    verify_pins(decl)
    guard = ceiling_guard(seed)
    cfg, cfg_m3b, _cfg_h = V2.load_configs()
    inputs, freeze, _contexts, _handles, _pkg, bank, carves, _ = U6.open_six(cfg, cfg_m3b)
    training = U6.training_rule_six(cfg, cfg_m3b)
    torch.manual_seed(seed)
    model = build_twin(inputs, bank)
    log(f"== fit {key}: {PARAMETERS} parameters, {len(carves['fit'])} datasets, seed {seed}, {torch.get_num_threads()} threads; "
        f"guard {guard['fit_hours_spent']:.2f} h spent, {guard['projected_fit_hours']:.2f} h projected")
    t0 = time.time()
    model, record = fit_model(model, carves["fit"], carves["select"], seed=seed, arm=ARM, config=dict(FIT_CONFIG),
                              max_epochs=training["max_epochs"], batches_per_epoch=training["batches_per_epoch"],
                              batch_size=training["batch_size"], patience=training["patience"], lr=training["lr"],
                              weight_decay=training["weight_decay"], clip=training["clip"], dataset_draw=training["dataset_draw"],
                              epoch_limit_s=training["epoch_limit_s"], pack_workers=V2.PACK["workers"],
                              prefetch_depth=V2.PACK["depth"], checkpoint=FITS / f"{key}.ckpt", log=log)
    weights = FITS / f"{key}.pt"
    tmp = FITS / f"{key}.pt.tmp"
    torch.save(model.state_dict(), tmp)
    os.replace(tmp, weights)
    out = {**asdict(record), "key": key, "file": "configs/mp_approx_six_base.yaml", "work": "work_1_twin_fits", "arm": ARM,
           "seed": seed, "hidden": V2.HIDDEN, "parameters": PARAMETERS, "datasets": sorted(carves["fit"]),
           "sampler": "uniform over the six, then a query of that fit carve (dataset_draw per_query)",
           "early_stopping": "macro select recall@5 over the six select carves", "warm_start": None,
           "contract_block": inputs["contract_block"], "columns": inputs["n_scalars"], "core_sha256": inputs["core_sha256"],
           "base": inputs["base"], "freeze_checked": freeze, "relation_bank": {"rows": bank.n_rows, "sha256": bank.sha256},
           "training": training, "utc": V2.utc(), "wall_seconds_this_process": round(time.time() - t0, 1),
           "threads": torch.get_num_threads(), "blas_threads": {v: os.environ.get(v) for v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
           "pack_workers": V2.PACK["workers"], "prefetch_depth": V2.PACK["depth"], "peak_rss_bytes": V2.M3B_RUN.peak_rss_bytes(),
           "placement": {"host": platform.node(), "machine": platform.machine(), "python": platform.python_version(),
                         "torch": torch.__version__, "device": "cpu"},
           "git_head": git_head(), "module_sha256": module_shas(), "state_sha256": sha256_file(weights), "ceiling_guard": guard,
           "select_numbers": "the early-stopping trace of this fit only; never read as a result (role)"}
    tmp = FITS / f"{key}.json.tmp"
    tmp.write_text(json.dumps(out, indent=1), encoding="utf-8")
    os.replace(tmp, rec_path)
    (FITS / f"{key}.ckpt").unlink(missing_ok=True)
    verify_pins(decl)   # frozen code: checked at the start and the end of every stage
    log(f"   {key}: best epoch {record.best_epoch} select macro R@5 {record.best_select_macro_recall5:.4f} in {record.seconds:.0f}s")
    return out


def stage_fits(log=log_utc) -> None:
    """The three fits in order, each in a fresh process at 8 threads (placement.one_at_a_time)."""
    for seed in SEEDS:
        if (FITS / f"{fit_key(seed)}.json").exists():
            log(f"   {fit_key(seed)}: record exists")
            continue
        cmd = [sys.executable, str(Path(__file__).resolve()), "--stage", "fit", "--seed", str(seed), "--threads", str(FIT_THREADS)]
        log(f"== seed {seed}: {' '.join(cmd[1:])}")
        rc = subprocess.run(cmd, cwd=ROOT).returncode
        if rc != 0:
            raise SystemExit(f"seed {seed}: the fit process exited {rc}; the next seed is not started")


def stage_status(log=print) -> None:
    filed = filed_fits()
    for s in SEEDS:
        key = fit_key(s)
        if s in filed:
            r = filed[s]
            log(f"{key}: best epoch {r['best_epoch']} of {r['epochs_run']}, {float(r['seconds']) / 3600:.2f} h, state {r['state_sha256'][:12]}")
        elif (FITS / f"{key}.ckpt").exists():
            log(f"{key}: running or interrupted (checkpoint present)")
        else:
            log(f"{key}: not started")
    spent = sum(float(r.get("seconds", 0.0)) for r in filed.values()) / 3600
    log(f"fit-hours spent {spent:.2f} of {CEILING_HOURS:.0f}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="MP-Approx six-dataset base: the matched twin and the scored base")
    ap.add_argument("--stage", required=True, choices=["fit", "fits", "status"])
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--threads", type=int, default=None)
    args = ap.parse_args(argv)
    if args.stage == "fit":
        if args.seed is None:
            raise SystemExit("--stage fit needs --seed")
        if args.threads != FIT_THREADS:
            raise SystemExit(f"--stage fit runs at --threads {FIT_THREADS} (placement.threads)")
        torch.set_num_threads(args.threads)
        stage_fit(args.seed)
    elif args.stage == "fits":
        stage_fits()
    else:
        stage_status()


if __name__ == "__main__":
    main()
