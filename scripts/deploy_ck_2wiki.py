"""The deployable 2wiki compiled-kernel model, track MP-APPROX (configs/deploy_ck_2wiki.yaml): u_mlp_v2_mix plus one hop
of the compiled factorised kernel (src/mp_retrieval/compiled_kernel.py), fitted from the gold labels alone on 2wiki's own
carves, against its matched twin, the kernel's self-entry control and the GNN, every one fitted and scored on host_gpu_det.

    python scripts/deploy_ck_2wiki.py --stage pins                      # laptop: the input pins, printed for the declaration
    python scripts/deploy_ck_2wiki.py --stage transfer                  # laptop: the pinned caches and M3B arrays pushed by rx
    python scripts/deploy_ck_2wiki.py --stage reference                 # laptop: amendment 2's reference, the first 16 batches packed here
    python scripts/deploy_ck_2wiki.py --stage verify                    # host: pins, the mirror, the CSR stores built from it and checked
    python scripts/deploy_ck_2wiki.py --stage fit --arm ck_qi --seed 0  # host_gpu_det: one fit in a fresh process (--repeat for the repeat)
    python scripts/deploy_ck_2wiki.py --stage eval                      # host_gpu_det: the eval population compiled per query, every fit scored
    python scripts/deploy_ck_2wiki.py --stage read                      # host: contrasts, shares, the band and the flags -> read.json
    python scripts/deploy_ck_2wiki.py --stage doc                       # laptop: record.json and docs/DEPLOY_CK_2WIKI.md
    python scripts/deploy_ck_2wiki.py --stage file --date 2026_09_30 --commit <sha> --extra run_extra.json

No GNN output, probe or kernel of the MP-Approx ladder enters any arm: every arm is fitted on the gold labels under
universal-v2's frozen training rule. The contract, the carves, the population, the sampler, the loss, the packing and the
metrics are universal_v2_run.py's and the M3B code's, imported unchanged; the device path is device_placement.py's. What
this file adds is the kernel arm and its self-entry control, a device-aware copy of the eval pass (the compiled form scored
beside the forward, batch-one latency on the GPU and the host CPU), its own reading and its own output directory.
"""

from __future__ import annotations

import os
import sys

HOST_STAGES = ("verify", "fit", "eval", "read")


def _stage_in_argv() -> str | None:
    if "--stage" in sys.argv:
        i = sys.argv.index("--stage")
        return sys.argv[i + 1] if i + 1 < len(sys.argv) else None
    return None


if __name__ == "__main__":   # placement: BLAS and OpenMP pools, and cuBLAS's workspace, are fixed before numpy and torch load (level 3's rule)
    _HOST = _stage_in_argv() in HOST_STAGES
    _THREADS = "8" if _HOST or _stage_in_argv() == "reference" else "6"   # amendment 2: the reference packs at the host stages' count
    for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[_var] = _THREADS
    if _HOST:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import argparse  # noqa: E402
import copy  # noqa: E402
import dataclasses  # noqa: E402
import gc  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from dataclasses import asdict  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "src", ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import universal_v2_run as V2  # noqa: E402  (imported, never edited)
from mp_retrieval import compiled_kernel as CK  # noqa: E402
from mp_retrieval import m3b_pools  # noqa: E402
from mp_retrieval.device_placement import batch_to, fit_model_placed, model_to  # noqa: E402
from mp_retrieval.m3b_features import QueryInputs  # noqa: E402
from mp_retrieval.m3b_models import parameter_count  # noqa: E402
from mp_retrieval.m3b_train import METRIC_NAMES, draw_indices, pack_parts, rank_metrics  # noqa: E402
from mp_retrieval.universal_v2_features import IDX, compile_query_v2  # noqa: E402
from mp_retrieval.universal_v2_models import FAMILIES, CarveDataV2, pack_queries_v2  # noqa: E402

CONFIG = ROOT / "configs" / "deploy_ck_2wiki.yaml"
MIRROR_CONFIG = ROOT / "configs" / "host_mirror_six.yaml"
OUT = ROOT / "outputs" / "deploy_ck_2wiki"
FITS = OUT / "fits"
EVAL = OUT / "eval"
RECORD = OUT / "record.json"
DOC = ROOT / "docs" / "DEPLOY_CK_2WIKI.md"
REFERENCE = ROOT / "outputs" / "deploy_ck_2wiki_reference" / "batches_laptop.json"   # amendment 2: packed on the laptop, pushed
AMENDMENT_2 = "amendment_2_2026_09_30_batch_check"
CHECK_BATCHES = 16
NAME = "2wiki"
PHASE = "DEPLOY_CK_2WIKI"
SPEC = {"host": "host", "device": "cuda", "threads": 8, "mode": "det", "tf32": False, "env": "mpr-cu128@62fc45e9e1ba"}   # host_gpu_det, level 3's
ARM_SPECS = {   # how each arm key is built; the declaration's arms must be these, at these parameter counts
    "twin": {"arm": "u_mlp_v2_mix", "parameters": 330_955},
    "ck_qi": {"side": "qi", "form": "KERN", "parameters": 377_803},
    "ck_qw": {"side": "qw", "form": "KERN", "parameters": 377_995},
    "ck_full": {"side": "full", "form": "KERN", "parameters": 390_475},
    "ck_self": {"side": "qi", "form": "SELF", "parameters": 343_883},
    "gnn": {"arm": "u_gnn_v2_ef", "parameters": 420_932},
}
SEEDS = (0, 1, 2)
HALVES = ("V2_HELD_CONFIRMATION", "V2_GATE", "whole")        # the primary half first; the whole population descriptive
READ_METRICS = ("recall@5", "full_coverage@5", "hit@1")      # the reading's metric, its co-read, and the descriptive one
BOOT = {"resamples": 1000, "seed": 0, "level": 95}           # universal_v2_run.BOOTSTRAP's procedure, one matrix per half
READINGS = ("GNN_GAIN_ABSENT", "CK_KEEPS", "CK_PARTIAL", "CK_NO_GAIN")
FLAGS = ("CK_NOT_BELOW_GNN", "EDGES_CARRY", "SELF_CARRIES", "COMPILED_COST", "COMPILED_DIFFERS", "REPEAT_DIFFERS",
         "GATE_HALF_DIFFERS", "FC5_DIFFERS")
KEEP = {"share": 0.75, "share_low": 0.50}
LATENCY_QUERIES = 500
CHUNK_NODES = 24_000
LF = chr(10)


def log_utc(msg: str) -> None:
    print(f"[{V2.utc()}] {msg}", flush=True)


sha256_file = V2.sha256_file
lf_sha256 = V2.lf_sha256


def atomic_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(clean(obj), indent=1), encoding="utf-8")
    os.replace(tmp, path)


def clean(v):
    """NaN and infinities become null in a record; numpy scalars become Python ones."""
    if isinstance(v, dict):
        return {k: clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [clean(x) for x in v]
    if isinstance(v, (float, np.floating)):
        return float(v) if math.isfinite(float(v)) else None
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return v


def read_json(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


# ── the declaration and its pins ─────────────────────────────────────────────


def arm_spec(spec: dict) -> dict:
    return {k: spec[k] for k in ("arm", "side", "form") if k in spec}


def load_declaration(path: Path = CONFIG) -> dict:
    decl = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if decl.get("phase") != PHASE:
        raise SystemExit(f"{path}: not the {PHASE} declaration")
    for key, spec in decl["arms"].items():
        want = ARM_SPECS.get(key)
        if want is None or arm_spec(spec) != arm_spec(want) or int(spec["parameters"]) != want["parameters"]:
            raise SystemExit(f"arms.{key}: not an arm this script builds, or its construction or parameter pin differs")
    fitting = decl["fitting"]
    if list(fitting["seeds"]) != list(SEEDS):
        raise SystemExit(f"fitting.seeds: {fitting['seeds']}, this script fits {list(SEEDS)}")
    rep = fitting["repeat"]
    if rep["arm"] not in decl["arms"] or int(rep["seed"]) not in SEEDS:
        raise SystemExit("fitting.repeat names an arm or a seed the file does not fit")
    r = decl["readings"]
    need = {"twin", "gnn", "ck_self", r["primary_kernel"]} | ({r["per_query_reference"]} if r.get("per_query_reference") else set())
    if not need <= set(decl["arms"]):
        raise SystemExit(f"readings: {sorted(need - set(decl['arms']))} are not declared arms")
    if list(r["bands"]) != list(READINGS) or list(r["flags"]) != list(FLAGS):
        raise SystemExit("readings: the bands or the flags are not this script's")
    return decl


def primary_kernel(decl: dict) -> str:
    return decl["readings"]["primary_kernel"]


def reference_kernel(decl: dict) -> str | None:
    return decl["readings"].get("per_query_reference")


def pushed_paths(decl: dict) -> dict[str, str]:
    return dict(decl["inputs"]["pushed_sha256"])


def pin_problems(decl: dict, root: Path = ROOT, pushed: bool = True) -> dict:
    """inputs: the frozen code (LF sha256) and, with ``pushed``, every file the transfer carries (byte sha256)."""
    bad = {}
    for rel, want in decl["inputs"]["frozen_code_lf"].items():
        p = root / rel
        got = lf_sha256(p) if p.exists() else None
        if got != want:
            bad[rel] = {"declared": want, "found": got}
    if pushed:
        for rel, want in pushed_paths(decl).items():
            p = root / rel
            got = sha256_file(p) if p.exists() else None
            if got != want:
                bad[rel] = {"declared": want, "found": got}
    return bad


def hard_stop(message: str, **evidence) -> None:
    """hard_stops: the evidence is written to hard_stop.json and nothing further runs in this process."""
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "hard_stop.json").write_text(json.dumps(clean({"utc": V2.utc(), "message": message, **evidence}), indent=1), encoding="utf-8")
    raise SystemExit(f"HARD STOP: {message}")


def verify_pins(decl: dict, pushed: bool = True) -> None:
    bad = pin_problems(decl, pushed=pushed)
    if bad:
        hard_stop("pinned inputs differ from configs/deploy_ck_2wiki.yaml#inputs", files=bad)


def store_digest(store) -> str:
    """The content of one family store: every field's name and, for an array, its dtype, shape and bytes, in field order.
    The npz file's bytes carry zip timestamps, so two machines' files are compared by content."""
    h = hashlib.sha256()
    for f in dataclasses.fields(store):
        v = getattr(store, f.name)
        h.update(f.name.encode("utf-8"))
        if isinstance(v, np.ndarray):
            a = np.ascontiguousarray(v)
            h.update(f"{a.dtype.str}{a.shape}".encode("utf-8"))
            h.update(a)
        else:
            h.update(repr(v).encode("utf-8"))
    return h.hexdigest()


def store_digests(csr_dir: Path | None = None) -> dict[str, str]:
    """Each 2wiki family store as the code loads it (m3b_pools.FamilyStore.load), by content."""
    d = csr_dir or (V2.M3B_OUT / "csr")
    out = {}
    for fam in FAMILIES:
        p = d / f"{NAME}_{fam}.npz"
        out[fam] = store_digest(m3b_pools.FamilyStore.load(p, fam)) if p.exists() else None
    return out


def store_problems(decl: dict, csr_dir: Path | None = None) -> dict:
    got = store_digests(csr_dir)
    want = decl["inputs"]["csr_content_sha256"]
    return {f: {"declared": want[f], "found": got[f]} for f in FAMILIES if got[f] != want[f]}


# ── placement, code identity ─────────────────────────────────────────────────


def host_placement() -> tuple[dict, list]:
    """placement.settings_applied_by: host_gpu_det's settings applied through the equivalence file's functions, read back
    and checked by its environment_problems; any difference is a hard stop. The driver is recorded (a deviation, not a stop)."""
    import cpu_gpu_equivalence as CGE   # imported, never edited; pinned in inputs.frozen_code_lf

    if not torch.cuda.is_available():
        hard_stop("placement: CUDA is not available to a host_gpu_det stage", CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES"))
    CGE.placement_settings(SPEC["mode"], SPEC["threads"])
    place = CGE.placement_block(SPEC["device"])
    bad = CGE.environment_problems(SPEC, place)
    if bad:
        hard_stop("placement: host_gpu_det's settings read back differently from host_gpu_det_as_tested", problems=bad, placement=place)
    deviations = []
    if place.get("driver") != CGE.GPU_DRIVER:
        deviations.append(f"driver {place.get('driver')}, the qualification names {CGE.GPU_DRIVER}")
    return place, deviations


def warning_summary(caught) -> list:
    seen: dict = {}
    for w in caught:
        text = f"{w.category.__name__}: {w.message}"
        seen[text] = seen.get(text, 0) + 1
    return [{"warning": k, "count": v, "determinism": "determinis" in k.lower()} for k, v in sorted(seen.items())]


def module_shas() -> dict:
    """placement.host_native_protocol.identical_code: the LF sha256 of every module this process imported from the repository."""
    root = ROOT.resolve()
    out = {}
    for m in list(sys.modules.values()):
        f = getattr(m, "__file__", None)
        if not isinstance(f, str) or not os.path.isabs(f):
            continue
        p = Path(f).resolve()
        if p.suffix == ".py" and root in p.parents:
            out[p.relative_to(root).as_posix()] = lf_sha256(p)
    return dict(sorted(out.items()))


def git_head() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None


def shown(p: Path) -> str:
    """A path for a log line: relative to the repository when it lies inside it (the tests write under a temporary directory)."""
    return p.relative_to(ROOT).as_posix() if p.is_relative_to(ROOT) else str(p)


def committed_lf_sha(commit: str, path: str) -> str | None:
    try:
        blob = subprocess.run(["git", "show", f"{commit}:{path}"], cwd=ROOT, capture_output=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    return hashlib.sha256(blob.replace(b"\r\n", b"\n")).hexdigest()


def code_problems(jobs: dict, committed) -> dict:
    """identical_code: across the host jobs' records, every module path has one sha256, the committed file's."""
    seen: dict = {}
    for job, shas in jobs.items():
        for path, sha in shas.items():
            seen.setdefault(path, {})[job] = sha
    bad = {}
    for path, by_job in sorted(seen.items()):
        want = committed(path)
        if len(set(by_job.values())) != 1 or want not in set(by_job.values()):
            bad[path] = {"jobs": by_job, "committed": want}
    return bad


def state_digest(state: dict) -> str:
    """The weights by content (key, dtype, shape, bytes in key order): what the repeat compares."""
    h = hashlib.sha256()
    for k in sorted(state):
        t = state[k].detach().to("cpu").contiguous()
        h.update(k.encode("utf-8"))
        h.update(f"{t.dtype}{tuple(t.shape)}".encode("utf-8"))
        h.update(t.numpy().tobytes() if t.dtype != torch.bfloat16 else t.view(torch.int16).numpy().tobytes())
    return h.hexdigest()


# ── the host: the mirror, the contexts, the models ───────────────────────────


def mirror_root(decl: dict) -> Path:
    """The host mirror named by configs/host_mirror_six.yaml#host.mirror_root, which must be the one this file declares."""
    mirror_cfg = yaml.safe_load(MIRROR_CONFIG.read_text(encoding="utf-8"))
    root = mirror_cfg["host"]["mirror_root"]
    if root != decl["inputs"]["mirror"]["root"]:
        hard_stop("the host mirror root differs from the declared one", declared=decl["inputs"]["mirror"]["root"], config=root)
    return Path(root)


def mirror_record(decl: dict) -> tuple[dict, str]:
    """The mirror's own verify record for 2wiki (scripts/host_mirror_hf.py verify): VERIFIED, the served freeze equal to
    the declared one, the loader imported from the mirror."""
    path = ROOT / decl["inputs"]["mirror"]["verify_record"]
    rec = read_json(path)
    if rec is None:
        hard_stop("the host mirror holds no verify record for 2wiki; the fits wait for it", path=str(path))
    served = (Path(decl["inputs"]["mirror"]["root"]) / "data" / "final_canonical").as_posix()
    ok = (rec.get("status") == "VERIFIED" and rec.get("freeze_matches_declared") is True and rec.get("loader_imported_from_mirror") is True
          and NAME in rec.get("datasets", []) and Path(rec.get("mirror", "")).as_posix() == served)
    if not ok:
        hard_stop("the host mirror's 2wiki verify record is not VERIFIED at the declared root", record=rec)
    return rec, sha256_file(path)


def open_2wiki(decl: dict, host: bool, log=log_utc):
    """configs, the v2 core, the 2wiki context (family stores, dense nodes, relation table) and the served bank, through
    universal_v2_run.open_contexts_v2 unchanged. On the host substrate.package_root is replaced in memory by the mirror
    (configs/host_mirror_six.yaml: no config file is edited); the freeze check of m3b_compile.open_package still runs."""
    cfg, cfg_m3b, cfg_h = V2.load_configs()
    if host:
        cfg_m3b["substrate"]["package_root"] = str(mirror_root(decl))    # in memory only
    m3b_compile = V2.M3B_RUN.load_script("m3b_compile")
    t0 = time.time()
    contexts, handles, pkg, bank = V2.open_contexts_v2(cfg_m3b, [NAME], m3b_compile)
    inputs = V2.model_inputs(cfg, cfg_m3b)
    if pkg[3]["RECORD_SHA256"] != decl["inputs"]["freeze_RECORD_SHA256"]:
        hard_stop("the served freeze is not the declared one", found=pkg[3]["RECORD_SHA256"])
    log(f"   {NAME}: contexts opened in {time.time() - t0:.0f}s from {pkg[2]}; bank rows {bank.n_rows}")
    return cfg, cfg_m3b, cfg_h, inputs, contexts, handles, pkg, bank, m3b_compile


def build_model(arm_key: str, inputs: dict, bank) -> torch.nn.Module:
    """arms: the twin and the GNN by the frozen make_model, the kernel arms by compiled_kernel.build_ck; the parameter pin."""
    spec = ARM_SPECS[arm_key]
    if "arm" in spec:
        model = V2.make_model(spec["arm"], inputs, bank)
    else:
        model = CK.build_ck(inputs, hidden=V2.HIDDEN, dropout=V2.DROPOUT, side=spec["side"], form=spec["form"])
    params = parameter_count(model)
    if params != spec["parameters"]:
        hard_stop(f"{arm_key}: {params} parameters, the declaration pins {spec['parameters']}")
    if params > V2.PARAMETER_BUDGET["gnn"]:
        hard_stop(f"{arm_key}: {params} parameters, over the GNN budget {V2.PARAMETER_BUDGET['gnn']}")
    return model


def fit_key(arm_key: str, seed: int, repeat: bool = False) -> str:
    return f"{arm_key}__H{V2.HIDDEN}__{NAME}__s{seed}" + ("__repeat" if repeat else "")


def declared_fits(decl: dict) -> list[tuple[str, int, bool]]:
    rep = decl["fitting"]["repeat"]
    return [(a, s, False) for a in decl["arms"] for s in SEEDS] + [(rep["arm"], int(rep["seed"]), True)]


def training_rule(decl: dict, cfg: dict, cfg_m3b: dict) -> dict:
    """fitting.rule: universal_v2_run.training_rule_v2 unchanged, which must equal the values this file quotes."""
    rule = V2.training_rule_v2(cfg, cfg_m3b)
    want = decl["fitting"]["rule"]
    differs = {k: {"declared": v, "found": rule.get(k)} for k, v in want.items() if rule.get(k) != v}
    if differs:
        hard_stop("the frozen training rule differs from the declared one", differs=differs)
    return rule


def ceiling_guard(decl: dict, key: str) -> dict:
    """compute.fit_hours_ceiling: the filed fits' seconds plus this fit's projection (the longest filed fit, or the
    declared first projection) must stay within the ceiling; otherwise ceiling_breach.json and no fit starts."""
    c = decl["compute"]
    filed = [read_json(p) for p in FITS.glob("*.json")] if FITS.exists() else []
    spent = sum(float(r.get("seconds", 0.0)) for r in filed if r) / 3600
    projected = max((float(r.get("seconds", 0.0)) for r in filed if r), default=float(c["first_projection_hours"]) * 3600) / 3600
    out = {"fit": key, "fit_hours_spent": round(spent, 3), "projected_fit_hours": round(projected, 3),
           "ceiling_fit_hours": float(c["fit_hours_ceiling"]), "within_ceiling": bool(spent + projected <= float(c["fit_hours_ceiling"]))}
    if not out["within_ceiling"]:
        atomic_json(FITS / "ceiling_breach.json", out)
        raise SystemExit(f"compute ceiling: {spent:.1f} + {projected:.1f} fit-hours over {c['fit_hours_ceiling']}; {key} not started")
    return out


def host_open(decl: dict, log=log_utc):
    """The preamble of the fit and eval stages: host_gpu_det applied and read back, then host_inputs."""
    place, deviations = host_placement()
    opened, checked = host_inputs(decl, log)
    return place, deviations, opened, checked


def host_inputs(decl: dict, log=log_utc):
    """Every host stage's inputs: the pins (code and pushed files), the verify record of this file, the contexts from
    the mirror, and the stores by content."""
    verify_pins(decl)
    ver = read_json(OUT / "verify.json")
    if ver is None or ver.get("status") != "VERIFIED":
        hard_stop("the verify stage has not passed on the host (outputs/deploy_ck_2wiki/verify.json)")
    mrec, mrec_sha = mirror_record(decl)
    opened = open_2wiki(decl, host=True, log=log)
    bad = store_problems(decl)
    if bad:
        hard_stop("the 2wiki family stores differ by content from the laptop's", stores=bad)
    return opened, {"verify_json_sha256": sha256_file(OUT / "verify.json"), "mirror_verify_record_sha256": mrec_sha}


# ── amendment 2: the 16-batch check (configs/host_mirror_six.yaml#verification) ─


def batch_pins(batch) -> dict:
    """A packed batch field by field as the equivalence file pins it (cpu_gpu_equivalence.field_pins: the dtype, the shape
    and the sha256 of the bytes), with the NaN count of every floating field beside it: equal pins are byte identity,
    which is torch.equal wherever a field holds no NaN."""
    import cpu_gpu_equivalence as CGE   # imported, never edited; pinned in inputs.frozen_code_lf

    pins = CGE.field_pins(batch)
    for f, p in pins.items():
        t = getattr(batch, f)
        p["nan"] = int(torch.isnan(t).sum()) if t.is_floating_point() else 0
    return pins


def parts_list(parts) -> list:
    return [[name, [int(i) for i in idx]] for name, idx in parts]


def fit_draws(fit, seed: int, rule: dict, n: int = CHECK_BATCHES) -> list:
    """The first ``n`` draws of fit_model_placed for ``seed``: its generator and cursors, the pinned draw_indices called in
    the order its loop calls it (packing ahead, epochs and batches skipped for want of gold change no draw)."""
    rng = np.random.default_rng(seed)
    fits = {NAME: fit}
    names = sorted(fits)
    cursors = {k: [rng.permutation(fits[k].trainable), 0] for k in names}
    return [draw_indices(fits, names, cursors, rng, int(rule["batch_size"]), rule["dataset_draw"]) for _ in range(n)]


def fit_batch_pins(fit, seed: int, rule: dict, n: int = CHECK_BATCHES) -> list[dict]:
    """A fit's first ``n`` batches, packed by the pinned pack_parts as fit_model_placed packs them, pinned."""
    return [{"batch": k, "parts": parts_list(parts), "fields": batch_pins(pack_parts({NAME: fit}, parts, FAMILIES))}
            for k, parts in enumerate(fit_draws(fit, seed, rule, n))]


def eval_chunks(n: int, chunk: int) -> list[np.ndarray]:
    """The query blocks eval_placed packs, in its order."""
    return [np.arange(start, min(start + chunk, n)) for start in range(0, n, chunk)]


def eval_chunk_batch(ev: dict, context, columns, idx: np.ndarray, m3b_compile):
    """One block of the eval pass compiled and packed exactly as eval_placed compiles and packs it."""
    qds = []
    for i in idx:
        E, compiled = compile_eval_query(ev["prep"], context, int(i))
        gold_local = m3b_compile.gold_local_of(ev["prep"].pools[i], ev["pop"].golds[i])
        qds.append(eval_entry(compiled, E, ev["prep"], ev["pop"], int(i), gold_local, columns))
    return pack_queries_v2(qds, context)


def eval_batch_pins(ev: dict, context, columns, m3b_compile, n: int = CHECK_BATCHES) -> list[dict]:
    """The eval pass's first ``n`` packed batches, pinned."""
    return [{"batch": k, "queries": [int(idx[0]), int(idx[-1]) + 1], "fields": batch_pins(eval_chunk_batch(ev, context, columns, idx, m3b_compile))}
            for k, idx in enumerate(eval_chunks(ev["n"], ev["chunk"])[:n])]


def batch_problems(want: list[dict], got: list[dict]) -> dict:
    """Every batch of ``want`` against ``got``, in order: the draw (or the query block) and each field's pin."""
    bad = {}
    if len(want) != len(got):
        bad["count"] = {"reference": len(want), "here": len(got)}
    for w, g in zip(want, got):
        for key in ("batch", "parts", "queries"):
            if w.get(key) != g.get(key):
                bad[f"batch {w.get('batch')}: {key}"] = {"reference": w.get(key), "here": g.get(key)}
        for f in sorted(set(w["fields"]) | set(g["fields"])):
            if w["fields"].get(f) != g["fields"].get(f):
                bad[f"batch {w.get('batch')}: {f}"] = {"reference": w["fields"].get(f), "here": g["fields"].get(f)}
    return bad


def load_reference(decl: dict) -> tuple[dict, str]:
    """The laptop's batches (amendment 2), refused unless the file is the one the amendment pins."""
    pin = (decl.get(AMENDMENT_2) or {}).get("reference_sha256")
    if pin is None:
        hard_stop(f"{AMENDMENT_2} is not filed; no host stage reads data without the 16-batch check")
    got = sha256_file(REFERENCE) if REFERENCE.exists() else None
    if got != pin:
        hard_stop("the laptop's batch reference is not the file amendment 2 pins", path=str(REFERENCE), declared=pin, found=got)
    return read_json(REFERENCE), got


# ── stage: pins, transfer (laptop) ───────────────────────────────────────────


def pinned_now(decl_code: list[str]) -> dict:
    """What this file pins, as found on this machine: the code (LF), the pushed files (bytes) and the stores (content)."""
    pushed = {}
    for kind in ("fit", "select"):
        for p in sorted((V2.CACHE / NAME / kind).iterdir()):
            if p.is_file():
                pushed[p.relative_to(ROOT).as_posix()] = sha256_file(p)
    m3b = V2.M3B_OUT / "eval" / f"{NAME}.npz"
    pushed[m3b.relative_to(ROOT).as_posix()] = sha256_file(m3b)
    return {"frozen_code_lf": {rel: lf_sha256(ROOT / rel) for rel in decl_code}, "pushed_sha256": pushed,
            "csr_content_sha256": store_digests()}


def stage_pins(decl_code: list[str]) -> None:
    print(yaml.safe_dump(pinned_now(decl_code), sort_keys=False, width=200))


def stage_transfer(decl: dict, log=log_utc) -> None:
    """transfer: the pinned files, checked here first, pushed into the host workspace by rx (read in place)."""
    verify_pins(decl)
    rels = list(pushed_paths(decl))
    total = sum((ROOT / r).stat().st_size for r in rels)
    cmd = [sys.executable, str(ROOT / "tools" / "rx" / "rx.py"), "push", "--force"]
    for r in rels:
        cmd += ["--inputs", r]
    log(f"transfer: {len(rels)} files, {total / 1e6:.1f} MB")
    t0 = time.time()
    rc = subprocess.run(cmd, cwd=ROOT).returncode
    if rc != 0:
        raise SystemExit(f"rx push exited {rc}")
    log(f"transfer: pushed in {time.time() - t0:.0f}s")


def stage_reference(decl: dict, log=log_utc) -> dict:
    """Amendment 2, the laptop's half: from the laptop's package and stores, the first 16 batches of each seed's fit draw
    under the frozen rule and the first 16 batches of the eval pass, packed by the code the host runs, pinned field by
    field. No model is built and no number is made. compile_query_v2's features depend on the BLAS thread count, so the
    reference is packed at the host stages' count and refused at any other."""
    if REFERENCE.exists():
        raise SystemExit(f"{REFERENCE.relative_to(ROOT)} exists; it is made once")
    if os.environ.get("OPENBLAS_NUM_THREADS") != str(SPEC["threads"]):
        raise SystemExit(f"the reference packs at the host stages' BLAS thread count ({SPEC['threads']}), not "
                         f"{os.environ.get('OPENBLAS_NUM_THREADS')}: the packed features depend on it")
    t0 = time.time()
    verify_pins(decl)
    cfg, cfg_m3b, cfg_h, inputs, contexts, handles, pkg, bank, m3b_compile = open_2wiki(decl, host=False, log=log)
    bad = store_problems(decl)
    if bad:
        hard_stop("reference: the laptop's 2wiki family stores differ from their pins", stores=bad)
    rule = training_rule(decl, cfg, cfg_m3b)
    fit = CarveDataV2(V2.CACHE / NAME / "fit", contexts[NAME], columns=inputs["column_indices"])
    fits = {}
    for s in SEEDS:
        fits[str(s)] = fit_batch_pins(fit, s, rule)
        log(f"   fit seed {s}: {CHECK_BATCHES} batches packed, {time.time() - t0:.0f}s")
    m3b_contract = V2.M3B_RUN.load_script("m3b_contract")
    ev = eval_population(decl, cfg, cfg_m3b, cfg_h, contexts[NAME], handles[NAME], pkg, m3b_compile, m3b_contract, log=log)
    t_eval = time.time()
    batches = eval_batch_pins(ev, contexts[NAME], inputs["column_indices"], m3b_compile)
    log(f"   eval: {len(batches)} batches of {ev['chunk']} queries packed in {time.time() - t_eval:.0f}s")
    import cpu_gpu_equivalence as CGE
    out = {"file": "configs/deploy_ck_2wiki.yaml", "amendment": AMENDMENT_2, "utc": V2.utc(), "where": "laptop",
           "what": "the first 16 packed batches of each seed's fit draw and of the eval pass, packed from the laptop's package and stores",
           "pins": "cpu_gpu_equivalence.field_pins (dtype, shape, sha256 of the bytes) and each floating field's NaN count",
           "rule": {k: rule[k] for k in ("batch_size", "dataset_draw")}, "fit_queries": fit.n_queries, "fit": fits,
           "eval": {"queries": ev["n"], "ids_sha256": ev["pop"].digest, "chunk": ev["chunk"], "batches": batches},
           "freeze_RECORD_SHA256": pkg[3]["RECORD_SHA256"], "csr_content_sha256": store_digests(), "placement": CGE.placement_block(None),
           "threads": torch.get_num_threads(), "blas_threads": int(os.environ["OPENBLAS_NUM_THREADS"]),
           "seconds": round(time.time() - t0, 1), "peak_rss_bytes": V2.M3B_RUN.peak_rss_bytes(),
           "git_head": git_head(), "module_sha256": module_shas()}
    atomic_json(REFERENCE, out)
    log(f"reference: {REFERENCE.relative_to(ROOT)} ({sha256_file(REFERENCE)}), {out['seconds']}s")
    return out


def reference_check(ref: dict, fit, rule: dict, ev: dict | None, context, columns, m3b_compile, seeds=SEEDS) -> dict:
    """The host's half: the same batches packed here and compared with the laptop's; the problems by seed and for the eval."""
    problems = {f"fit seed {s}": batch_problems(ref["fit"][str(s)], fit_batch_pins(fit, s, rule)) for s in seeds}
    if ev is not None:
        if ev["chunk"] != ref["eval"]["chunk"] or ev["pop"].digest != ref["eval"]["ids_sha256"]:
            problems["eval blocks"] = {"reference": {"chunk": ref["eval"]["chunk"], "ids_sha256": ref["eval"]["ids_sha256"]},
                                       "here": {"chunk": ev["chunk"], "ids_sha256": ev["pop"].digest}}
        problems["eval"] = batch_problems(ref["eval"]["batches"], eval_batch_pins(ev, context, columns, m3b_compile))
    return {k: v for k, v in problems.items() if v}


# ── stage: verify (host) ─────────────────────────────────────────────────────


def stage_verify(decl: dict, log=log_utc) -> dict:
    """transfer.verification, on the host: every pushed file's sha256, the frozen code, the mirror's 2wiki verify record,
    the contexts opened from the mirror (the family stores built from its graph by m3b_pools.load_or_build_store when
    absent) and every store's content equal to the laptop's. verify.json is what the fit and eval stages require."""
    t0 = time.time()
    bad = pin_problems(decl)
    if bad:
        hard_stop("verify: pinned inputs differ on the host", files=bad)
    mrec, mrec_sha = mirror_record(decl)
    csr = V2.M3B_OUT / "csr"
    built = {f: not (csr / f"{NAME}_{f}.npz").exists() for f in FAMILIES}
    ref, ref_sha = load_reference(decl)
    cfg, cfg_m3b, cfg_h, inputs, contexts, handles, pkg, bank, m3b_compile = open_2wiki(decl, host=True, log=log)
    stores_bad = store_problems(decl)
    if stores_bad:
        hard_stop("verify: the 2wiki family stores built on the host differ by content from the laptop's", stores=stores_bad, built=built)
    fit = CarveDataV2(V2.CACHE / NAME / "fit", contexts[NAME], columns=inputs["column_indices"])
    sel = CarveDataV2(V2.CACHE / NAME / "select", contexts[NAME], columns=inputs["column_indices"])
    rule = training_rule(decl, cfg, cfg_m3b)
    m3b_contract = V2.M3B_RUN.load_script("m3b_contract")
    t_check = time.time()
    ev = eval_population(decl, cfg, cfg_m3b, cfg_h, contexts[NAME], handles[NAME], pkg, m3b_compile, m3b_contract, log=log)
    problems = reference_check(ref, fit, rule, ev, contexts[NAME], inputs["column_indices"], m3b_compile)
    blas = {"reference": ref.get("blas_threads"), "here": os.environ.get("OPENBLAS_NUM_THREADS")}
    if problems:
        hard_stop("verify: batches packed on the host differ from the laptop's (amendment 2)", problems=problems, reference_sha256=ref_sha,
                  blas_threads=blas)
    check = {"reference": REFERENCE.relative_to(ROOT).as_posix(), "reference_sha256": ref_sha, "fit_seeds": list(SEEDS),
             "fit_batches_per_seed": CHECK_BATCHES, "eval_batches": len(ref["eval"]["batches"]), "eval_chunk": ev["chunk"],
             "blas_threads": blas, "all_equal": True, "seconds": round(time.time() - t_check, 1)}
    out = {"utc": V2.utc(), "status": "VERIFIED", "pushed_files": len(pushed_paths(decl)), "frozen_code_files": len(decl["inputs"]["frozen_code_lf"]),
           "mirror_verify_record": decl["inputs"]["mirror"]["verify_record"], "mirror_verify_record_sha256": mrec_sha,
           "mirror_verify_utc": mrec.get("utc"), "served": str(pkg[2]), "freeze_RECORD_SHA256": pkg[3]["RECORD_SHA256"],
           "stores_built_here": built, "csr_content_sha256": store_digests(), "fit_queries": fit.n_queries, "select_queries": sel.n_queries,
           "bank_rows": bank.n_rows, "batch_check": check, "seconds": round(time.time() - t0, 1), "git_head": git_head(),
           "module_sha256": module_shas()}
    atomic_json(OUT / "verify.json", out)
    log(f"verify: {out['pushed_files']} pushed files and {out['frozen_code_files']} code pins equal; mirror VERIFIED; stores equal by content "
        f"(built here: {[f for f, b in built.items() if b]}); fit {fit.n_queries}, select {sel.n_queries}; the first {CHECK_BATCHES} batches "
        f"of seeds {list(SEEDS)} and {check['eval_batches']} eval batches equal the laptop's -> VERIFIED")
    return out


# ── stage: fit (host_gpu_det) ────────────────────────────────────────────────


def stage_fit(decl: dict, arm_key: str, seed: int, repeat: bool = False, log=log_utc) -> dict:
    """One fit under the frozen rule on 2wiki's fit carve, early-stopped on its select carve, on host_gpu_det, resumable
    through its checkpoint; the weights and the record are written atomically once the best weights are restored."""
    if arm_key not in decl["arms"] or seed not in SEEDS:
        raise SystemExit(f"{arm_key} seed {seed}: not a declared fit")
    rep = decl["fitting"]["repeat"]
    if repeat and (arm_key, seed) != (rep["arm"], int(rep["seed"])):
        raise SystemExit(f"--repeat: the declared repeat is {rep['arm']} seed {rep['seed']}")
    key = fit_key(arm_key, seed, repeat)
    rec_path = FITS / f"{key}.json"
    if rec_path.exists():
        log(f"   {key}: record exists, not repeated")
        return read_json(rec_path)
    FITS.mkdir(parents=True, exist_ok=True)
    place, deviations, (cfg, cfg_m3b, cfg_h, inputs, contexts, handles, pkg, bank, _), checked = host_open(decl, log)
    cuda = torch.device(SPEC["device"]).type == "cuda"
    guard = ceiling_guard(decl, key)
    training = training_rule(decl, cfg, cfg_m3b)
    fit = CarveDataV2(V2.CACHE / NAME / "fit", contexts[NAME], columns=inputs["column_indices"])
    sel = CarveDataV2(V2.CACHE / NAME / "select", contexts[NAME], columns=inputs["column_indices"])
    ref, ref_sha = load_reference(decl)   # amendment 2: this fit's own first batches, packed here, before any step
    problems = reference_check(ref, fit, training, None, None, None, None, seeds=(seed,))
    if problems:
        hard_stop(f"{key}: its first {CHECK_BATCHES} batches packed on the host differ from the laptop's (amendment 2)", problems=problems)
    batch_check = {"reference_sha256": ref_sha, "seed": seed, "batches": CHECK_BATCHES, "equal": True}
    torch.manual_seed(seed)
    model = build_model(arm_key, inputs, bank)
    params = parameter_count(model)
    log(f"== fit {key}: {params} parameters, fit {fit.n_queries} / select {sel.n_queries} queries, {place['device_name']}, "
        f"{torch.get_num_threads()} threads; guard {guard['fit_hours_spent']:.2f} h spent")
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    config = {"H": V2.HIDDEN, "file": "deploy_ck_2wiki", "arm_key": arm_key, "repeat": bool(repeat)}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model, record = fit_model_placed(model, {NAME: fit}, {NAME: sel}, seed=seed, arm=arm_key, config=config,
                                         max_epochs=training["max_epochs"], batches_per_epoch=training["batches_per_epoch"],
                                         batch_size=training["batch_size"], patience=training["patience"], lr=training["lr"],
                                         weight_decay=training["weight_decay"], clip=training["clip"], dataset_draw=training["dataset_draw"],
                                         epoch_limit_s=training["epoch_limit_s"], pack_workers=V2.PACK["workers"],
                                         prefetch_depth=V2.PACK["depth"], checkpoint=FITS / f"{key}.ckpt", device=SPEC["device"], log=log)
    state = {k: v.detach().to("cpu") for k, v in model.state_dict().items()}
    weights, tmp = FITS / f"{key}.pt", FITS / f"{key}.pt.tmp"
    torch.save(state, tmp)
    os.replace(tmp, weights)
    out = {**asdict(record), "key": key, "file": "configs/deploy_ck_2wiki.yaml", "arm_key": arm_key, "arm": getattr(model, "arm", arm_key),
           "construction": arm_spec(ARM_SPECS[arm_key]), "seed": seed, "repeat": bool(repeat), "hidden": V2.HIDDEN, "parameters": params,
           "parameter_budget": V2.PARAMETER_BUDGET["gnn"], "datasets": [NAME], "fit_queries": fit.n_queries, "select_queries": sel.n_queries,
           "sampler": "a query of 2wiki's fit carve (dataset_draw per_query over one dataset)",
           "early_stopping": "2wiki select recall@5 (the macro over one dataset)", "warm_start": None,
           "contract_block": inputs["contract_block"], "columns": inputs["n_scalars"], "core_sha256": inputs["core_sha256"], "base": inputs["base"],
           "relation_bank": {"rows": bank.n_rows, "sha256": bank.sha256}, "training": training, "utc": V2.utc(),
           "wall_seconds_this_process": round(time.time() - t0, 1), "threads": torch.get_num_threads(),
           "pack_workers": V2.PACK["workers"], "prefetch_depth": V2.PACK["depth"], "peak_rss_bytes": V2.M3B_RUN.peak_rss_bytes(),
           "cuda_peak_bytes": int(torch.cuda.max_memory_allocated()) if cuda else None, "placement": place, "deviations": deviations,
           "warnings": warning_summary(caught), "inputs_checked": checked, "batch_check": batch_check, "ceiling_guard": guard, "git_head": git_head(),
           "module_sha256": module_shas(), "state_sha256": sha256_file(weights), "state_content_sha256": state_digest(state),
           "select_numbers": "the early-stopping trace of this fit only; never read as a result"}
    atomic_json(rec_path, out)
    (FITS / f"{key}.ckpt").unlink(missing_ok=True)
    log(f"   {key}: best epoch {record.best_epoch} of {record.epochs_run}, select R@5 {record.best_select_macro_recall5:.4f}, "
        f"{record.seconds:.0f}s, weights {out['state_content_sha256'][:12]}")
    return out


# ── stage: eval (host_gpu_det) ───────────────────────────────────────────────


def load_fits(decl: dict, inputs: dict, bank) -> dict[str, torch.nn.Module]:
    """Every declared fit and the repeat, each checked against its record's weights sha256 and content, on the CPU."""
    models = {}
    for arm_key, seed, repeat in declared_fits(decl):
        key = fit_key(arm_key, seed, repeat)
        rec, pt = read_json(FITS / f"{key}.json"), FITS / f"{key}.pt"
        if rec is None or not pt.exists():
            hard_stop(f"{key}: no fit record or weights; the eval pass follows every declared fit")
        if sha256_file(pt) != rec["state_sha256"]:
            hard_stop(f"{key}: the weights are not the file its record pins")
        state = torch.load(pt, map_location="cpu")
        if state_digest(state) != rec["state_content_sha256"]:
            hard_stop(f"{key}: the weights' content is not the record's")
        model = build_model(arm_key, inputs, bank)
        model.load_state_dict(state)
        model.eval()
        models[key] = model
    return models


def is_kernel(model) -> bool:
    return isinstance(model, CK.CKv2)


def _sync(device) -> None:
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()


def compile_eval_query(prep, context, i: int):
    """One eval query's dense rows and its compile under the frozen v2 contract (compile_query_v2, unchanged)."""
    E = context.nodes.read(prep.pools[i])
    inp = QueryInputs(prep.qemb[i], prep.dense_ids[i], prep.dense_scores[i], prep.splade_ids[i], prep.splade_scores[i])
    return E, compile_query_v2(inp, prep.pools[i], prep.seeds[i], context.stores, context.nodes, context.rel_table, embeddings=E)


def eval_entry(compiled, E, prep, pop, i: int, gold_local, columns) -> dict:
    """One compiled eval query as pack_queries_v2 reads it."""
    return {"pool": compiled.pool, "x": compiled.scalars[:, columns], "seedw": compiled.seedw, "qemb": prep.qemb[i],
            "seeds": compiled.seeds_local, "gold": gold_local, "gold_total": int(pop.golds[i].size), "emb": E}


def eval_population(decl: dict, cfg: dict, cfg_m3b: dict, cfg_h: dict, context, ds, pkg, m3b_compile, m3b_contract, shard=None,
                    log=log_utc) -> dict:
    """eval_placed's population, before any query is compiled: 2wiki's M3B eval population (its digest, split and halves
    checked against the declaration and universal-v2's filed values), prepared under the frozen M3B contract, with the
    headroom cell and the block size the pass packs by. ``shard`` (k, N) keeps every N-th query from the k-th after the
    population checks: the laptop test's smoke only; the host pass reads the whole population."""
    m3a, canonical, served, freeze = pkg
    key_m3b, frozen = m3b_compile.frozen_contract(cfg_m3b)
    construction = frozen["per_dataset"][NAME]["construction"]
    split = cfg_m3b["populations"]["eval_splits"][NAME]
    declared = decl["population"]
    filed = cfg["m3b_incumbents"]["eval_populations_reused_here"][NAME]
    if "test" in str(split):
        hard_stop(f"{NAME}: split {split} is a test split; no test split is read by this file")
    t0 = time.time()
    positions = m3a.node_position_map(ds)
    pop = m3b_compile.population(ds, NAME, "eval", cfg_m3b, cfg_h, m3a, positions)
    del positions
    gc.collect()
    if (pop.digest != declared["ids_sha256"] or pop.digest != filed["ids_sha256"] or pop.idx.size != int(declared["queries"])
            or split != declared["split"]):
        hard_stop(f"{NAME}: not the declared eval population", queries=int(pop.idx.size), digest=pop.digest, split=split)
    half = V2.half_labels(NAME, ds, split, pop.ids)
    counts = {"V2_GATE": int(half.sum()), "V2_HELD_CONFIRMATION": int((~half).sum())}
    if counts != {k: int(v) for k, v in declared["halves"].items()} or counts != V2.declared_half_counts(cfg, NAME):
        hard_stop(f"{NAME}: the halves are not the declared counts", found=counts, declared=declared["halves"])
    full_n = int(pop.idx.size)
    if shard is not None:
        k, N = shard
        pop.ids, pop.idx, pop.golds = pop.ids[k::N], pop.idx[k::N], pop.golds[k::N]
        half = half[k::N]
    prep = m3b_compile.prepare(ds, [pop], construction, cfg_h, context.stores, m3a, m3b_contract)[0]
    sizes = np.asarray([q.size for q in prep.pools])
    n = int(pop.idx.size)
    log(f"   {NAME}: {n} eval queries ({pop.zero_gold_excluded} zero-gold excluded), V2_GATE {counts['V2_GATE']}, pools mean {sizes.mean():.0f}, "
        f"prepared in {time.time() - t0:.0f}s")
    ks = tuple(int(k) for k in cfg_h["retrieval_pools"]["ks"])
    ceiling, _ = m3a.cell(prep.pools, m3a.golds_ragged(pop.golds), int(ds.n_nodes), ks, dataset=NAME, population="eval",
                          pool=frozen["per_dataset"][NAME]["pool"])
    chunk = max(1, int(CHUNK_NODES // max(sizes.mean(), 1)))
    return {"pop": pop, "half": half, "counts": counts, "full_n": full_n, "prep": prep, "sizes": sizes, "n": n, "ceiling": ceiling,
            "chunk": chunk, "construction": construction, "key_m3b": key_m3b, "frozen": frozen, "freeze": freeze, "t0": t0}


@torch.no_grad()   # scoring only, as universal_v2_six.eval_dataset_six: no autograd graph
def eval_placed(decl: dict, cfg: dict, cfg_m3b: dict, cfg_h: dict, inputs: dict, models: dict, cpu_models: dict, context, ds, pkg,
                m3b_compile, m3b_contract, device, shard=None, check: list | None = None, log=log_utc) -> dict:
    """universal_v2_six.eval_dataset_six for this file, with a device: the population of eval_population, compiled per
    query on the host CPU under the frozen v2 contract, packed on the CPU, scored on ``device`` by every fit and on the
    CPU by the fixed scorers; every kernel arm is also scored by its compiled form; batch-one latency on the first
    queries, on the GPU and the host CPU. ``check`` (amendment 2): the laptop's pins of the pass's first batches; each of
    those batches is compared field by field once packed and before any fit scores it, and a difference is a hard stop
    (nothing is written)."""
    ev = eval_population(decl, cfg, cfg_m3b, cfg_h, context, ds, pkg, m3b_compile, m3b_contract, shard=shard, log=log)
    pop, half, prep, sizes, n, chunk, t0 = ev["pop"], ev["half"], ev["prep"], ev["sizes"], ev["n"], ev["chunk"], ev["t0"]
    columns = inputs["column_indices"]
    fixed = list(V2.FIXED_SCORERS)
    kernels = [k for k, m in models.items() if is_kernel(m)]
    scorers = list(models) + [f"{k}@compiled" for k in kernels] + [f"fixed:{c}" for c in fixed]
    arrays = {f"{s}/{m}": np.zeros(n) for s in scorers for m in METRIC_NAMES}
    arrays["gold_dist_struct"] = np.full(n, -1, dtype=np.int64)
    gap = {k: 0.0 for k in kernels}
    latency = {"compile": [], "pack": [], "to_device": [], **{f"{where}:{k}": [] for k in cpu_models for where in ("gpu", "cpu")}}
    checked = 0
    t_loop = time.time()
    for k_block, idx in enumerate(eval_chunks(n, chunk)):
        start = int(idx[0])
        qds, gold_locals = [], []
        for i in idx:
            t = time.perf_counter()
            E, compiled = compile_eval_query(prep, context, i)
            if i < LATENCY_QUERIES:
                latency["compile"].append(time.perf_counter() - t)
            gold_local = m3b_compile.gold_local_of(prep.pools[i], pop.golds[i])
            qds.append(eval_entry(compiled, E, prep, pop, i, gold_local, columns))
            gold_locals.append(gold_local)
            arrays["gold_dist_struct"][i] = V2.gold_distance_struct(compiled.scalars, gold_local)
            for c in fixed:
                r = rank_metrics(compiled.scalars[:, IDX[c]], gold_local, int(pop.golds[i].size))
                for m in METRIC_NAMES:
                    arrays[f"fixed:{c}/{m}"][i] = r[m]
        if start < LATENCY_QUERIES:   # cold per-query latency, a batch of one, on the first queries (systems only)
            for j, i in enumerate(idx):
                if i >= LATENCY_QUERIES:
                    break
                t = time.perf_counter()
                single = pack_queries_v2([qds[j]], context)
                latency["pack"].append(time.perf_counter() - t)
                t = time.perf_counter()
                on_dev = batch_to(single, device)
                _sync(device)
                latency["to_device"].append(time.perf_counter() - t)
                for key, cpu_model in cpu_models.items():
                    model = models[key]
                    for where, m, b in (("gpu", model, on_dev), ("cpu", cpu_model, single)):
                        t = time.perf_counter()
                        m(V2.arm_view(m, b, inputs))
                        if where == "gpu":
                            _sync(device)
                        latency[f"{where}:{key}"].append(time.perf_counter() - t)
        batch = pack_queries_v2(qds, context)
        if check is not None and k_block < len(check):
            here = [{"batch": k_block, "queries": [start, int(idx[-1]) + 1], "fields": batch_pins(batch)}]
            bad = batch_problems([check[k_block]], here)
            if bad:
                hard_stop(f"eval: batch {k_block} packed on the host differs from the laptop's (amendment 2)", problems=bad)
            checked += 1
        ptr = batch.qptr.numpy()
        on_dev = batch_to(batch, device)
        for key, model in models.items():
            view = V2.arm_view(model, on_dev, inputs)
            scores_t = model(view)
            scores = scores_t.cpu().numpy()
            for j, i in enumerate(idx):
                r = rank_metrics(scores[ptr[j]:ptr[j + 1]], gold_locals[j], int(pop.golds[i].size))
                for m in METRIC_NAMES:
                    arrays[f"{key}/{m}"][i] = r[m]
            for mname, values in V2.mechanism(model, view, scores_t, ptr).items():
                arrays.setdefault(f"{key}/{mname}", np.zeros(n))[idx] = values
            if key in gap:
                comp = model.forward_compiled(view).cpu().numpy()
                gap[key] = max(gap[key], float(np.max(np.abs(comp - scores))) if comp.size else 0.0)
                for j, i in enumerate(idx):
                    r = rank_metrics(comp[ptr[j]:ptr[j + 1]], gold_locals[j], int(pop.golds[i].size))
                    for m in METRIC_NAMES:
                        arrays[f"{key}@compiled/{m}"][i] = r[m]
        if (start // chunk) % 10 == 0:
            done = min(start + chunk, n)
            log(f"      {NAME}: {done}/{n} queries, {(time.time() - t_loop) / done * 1000:.0f} ms/query")
    arrays["pool_size"] = sizes.astype(np.int64)
    arrays["half"] = half.astype(bool)
    rec = write_eval_record(arrays, scorers, pop, prep, ev["ceiling"], latency, gap, ev["key_m3b"], ev["frozen"], ev["construction"], chunk,
                            t0, t_loop, ev["freeze"], inputs, shard, ev["full_n"], log)
    if check is not None:
        rec["batch_check"] = {"batches_compared": checked, "reference_batches": len(check), "equal": checked == min(len(check), len(eval_chunks(n, chunk)))}
    return rec


def write_eval_record(arrays, scorers, pop, prep, ceiling, latency, gap, key_m3b, frozen, construction, chunk, t0, t_loop, freeze,
                      inputs, shard=None, full_n=None, log=log_utc) -> dict:
    """universal_v2_six.write_eval_record_six for this file: the ceiling check against the headroom cell, the MRR audit on
    every scorer, the agreement with the frozen M3B fixed rrf, and the record, summarised on each half and the whole."""
    EVAL.mkdir(parents=True, exist_ok=True)
    n = int(pop.idx.size)
    sizes = arrays["pool_size"]
    first = scorers[0]
    from_arrays = V2.M3B_RUN.ceiling_from_arrays(arrays[f"{first}/gold_in_pool"], arrays[f"{first}/gold_total"], sizes)
    if abs(from_arrays["recall_ceiling@5"] - float(ceiling["recall_ceiling@5"])) > 1e-9:
        hard_stop(f"{NAME}: the ceiling from the arrays is not the headroom cell's", arrays=from_arrays["recall_ceiling@5"],
                  cell=float(ceiling["recall_ceiling@5"]))
    audits = V2.audit_scorers(arrays, scorers, NAME)
    agreement = V2.m3b_fixed_rrf_agreement(NAME, {m: arrays[f"fixed:rrf/{m}"] for m in ("recall@5", "hit@1", "gold_in_pool", "gold_total", "pool_size")},
                                           shard, n if full_n is None else full_n)
    if agreement is None:
        hard_stop(f"{NAME}: the frozen M3B eval arrays are not on the host (outputs/m3b/eval/{NAME}.npz)")
    half = arrays["half"]
    masks = {"V2_HELD_CONFIRMATION": ~half, "V2_GATE": half, "whole": np.ones(n, dtype=bool)}
    record = {"dataset": NAME, "utc": V2.utc(), "file": "configs/deploy_ck_2wiki.yaml", "queries": n, "zero_gold_excluded": pop.zero_gold_excluded,
              "ids_sha256": pop.digest, "shard": None if shard is None else {"k": shard[0], "N": shard[1], "population_queries": full_n}, "contract_block": inputs["contract_block"], "core_sha256": inputs["core_sha256"],
              "m3b_contract_block": key_m3b, "pool": frozen["per_dataset"][NAME]["pool"], "construction": construction,
              "ceiling_as_compiled": {k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v) for k, v in ceiling.items()},
              "any_gold_at_pool": round(float((arrays[f"{first}/gold_in_pool"] > 0).mean()), 6),
              "seeds_added_mean": float(prep.seeds_added.mean()), "scorers": scorers, "chunk_queries": chunk,
              "halves": {"V2_GATE": int(half.sum()), "V2_HELD_CONFIRMATION": int((~half).sum()), "label_array": "half (True = V2_GATE)",
                         "rule": "scripts/universal_v2_split_audit.py::in_v2_gate (configs/universal_v2.yaml check_1_the_halves.split_rule_amended)"},
              "compiled_form": {k: {"max_abs_score_difference": gap[k]} for k in gap},
              "seconds": round(time.time() - t0, 1), "ms_per_query": round(1000 * (time.time() - t_loop) / max(n, 1), 2),
              "latency": {k: V2.M3B_RUN.percentiles(v) for k, v in latency.items()}, "peak_rss_bytes": V2.M3B_RUN.peak_rss_bytes(),
              "cuda_peak_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() and torch.cuda.is_initialized() else None,
              "threads": torch.get_num_threads(), "freeze_RECORD_SHA256": freeze["RECORD_SHA256"], "mrr_audit": audits,
              "m3b_fixed_rrf_agreement": agreement,
              "summary": {h: V2.gate_summary(arrays, scorers, m) for h, m in masks.items()}}
    np.savez_compressed(EVAL / f"{NAME}.npz", **arrays)
    (EVAL / f"{NAME}_query_ids.json").write_text(json.dumps(pop.ids), encoding="utf-8")
    atomic_json(EVAL / f"{NAME}.json", record)
    log(f"   {NAME}: done in {record['seconds']}s; held " +
        "; ".join(f"{s.split('__')[0]} R@5={v['recall@5']:.3f}" for s, v in record["summary"]["V2_HELD_CONFIRMATION"].items()
                  if "__s0" in s and "@" not in s))
    return record


def stage_eval(decl: dict, log=log_utc) -> dict:
    """The one eval pass: every declared fit and the repeat on 2wiki's eval population, host_gpu_det."""
    if (EVAL / f"{NAME}.json").exists():
        log("eval record exists, not repeated")
        return read_json(EVAL / f"{NAME}.json")
    place, deviations, (cfg, cfg_m3b, cfg_h, inputs, contexts, handles, pkg, bank, m3b_compile), checked = host_open(decl, log)
    ref, ref_sha = load_reference(decl)   # amendment 2: the pass's first batches, compared as they are packed
    m3b_contract = V2.M3B_RUN.load_script("m3b_contract")
    models = load_fits(decl, inputs, bank)
    cpu_models = {fit_key(a, 0): copy.deepcopy(models[fit_key(a, 0)]) for a in decl["arms"]}   # seed 0 of each arm, timed on the CPU
    for m in models.values():
        model_to(m, SPEC["device"])
    if torch.device(SPEC["device"]).type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    log(f"== eval: {len(models)} fits on {place['device_name']}, {len(cpu_models)} also timed on the host CPU")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rec = eval_placed(decl, cfg, cfg_m3b, cfg_h, inputs, models, cpu_models, contexts[NAME], handles[NAME], pkg, m3b_compile, m3b_contract,
                          SPEC["device"], check=ref["eval"]["batches"], log=log)
    rec["batch_check"]["reference_sha256"] = ref_sha
    rec.update({"placement": place, "deviations": deviations, "warnings": warning_summary(caught), "inputs_checked": checked,
                "git_head": git_head(), "module_sha256": module_shas()})
    atomic_json(EVAL / f"{NAME}.json", rec)
    return rec


# ── stage: read (host) ───────────────────────────────────────────────────────


def resample_matrix(n: int) -> np.ndarray:
    """universal_v2_run.paired_bootstrap's resamples (default_rng(0), one draw of n per resample, 1000 of them) as one matrix."""
    rng = np.random.default_rng(BOOT["seed"])
    return np.stack([rng.integers(n, size=n) for _ in range(BOOT["resamples"])]) if n else np.zeros((BOOT["resamples"], 0), dtype=np.int64)


def interval(values: np.ndarray) -> list[float]:
    lo, hi = np.percentile(values, [(100 - BOOT["level"]) / 2, 100 - (100 - BOOT["level"]) / 2])
    return [round(float(lo), 4), round(float(hi), 4)]


def seed_mean(arrays: dict, arm_key: str, metric: str, mask: np.ndarray) -> np.ndarray:
    return np.mean([arrays[f"{fit_key(arm_key, s)}/{metric}"][mask] for s in SEEDS], axis=0)


def contrast_pairs(decl: dict) -> dict:
    """readings.contrasts: each a paired difference of per-query seed means, (a, b) = a - b."""
    ck, ref = primary_kernel(decl), reference_kernel(decl)
    out = {"gnn_gain": ("gnn", "twin"), "ck_gain": (ck, "twin"), "edge_gain": (ck, "ck_self"), "self_gain": ("ck_self", "twin"),
           "ck_vs_gnn": (ck, "gnn")}
    if ref:
        out.update({"compiled_cost": (ck, ref), "ref_gain": (ref, "twin"), "ref_vs_gnn": (ref, "gnn")})
    return out


def share_pairs(decl: dict) -> dict:
    """readings.shares: a ratio of two contrasts over the same resamples, read only where the denominator's interval lies above 0."""
    ck, ref = primary_kernel(decl), reference_kernel(decl)
    out = {"ck_share": ((ck, "twin"), ("gnn", "twin")), "edge_share": ((ck, "ck_self"), ("gnn", "twin"))}
    if ref:
        out["ref_share"] = ((ref, "twin"), ("gnn", "twin"))
    return out


def read_half(decl: dict, arrays: dict, mask: np.ndarray, metric: str) -> dict:
    """Per arm the seed mean per query; the contrasts (paired differences) and the shares over one resample matrix."""
    n = int(mask.sum())
    R = resample_matrix(n)
    means = {a: seed_mean(arrays, a, metric, mask) for a in decl["arms"]}
    boot = {a: v[R].mean(axis=1) if n else np.zeros(BOOT["resamples"]) for a, v in means.items()}
    out = {"queries": n, "arms": {a: round(float(v.mean()), 4) for a, v in means.items()},
           "per_seed": {a: [round(float(arrays[f"{fit_key(a, s)}/{metric}"][mask].mean()), 4) for s in SEEDS] for a in decl["arms"]},
           "contrasts": {}, "shares": {}}
    for c, (a, b) in contrast_pairs(decl).items():
        out["contrasts"][c] = {"point": round(float(means[a].mean() - means[b].mean()), 4), "ci": interval(boot[a] - boot[b]), "of": [a, b]}
    for s, ((a, b), (c, d)) in share_pairs(decl).items():
        den_b = boot[c] - boot[d]
        den = float(means[c].mean() - means[d].mean())
        readable = interval(den_b)[0] > 0
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = (boot[a] - boot[b]) / den_b
        out["shares"][s] = {"point": round(float(means[a].mean() - means[b].mean()) / den, 4) if den else None,
                            "ci": interval(ratio[np.isfinite(ratio)]) if readable and np.isfinite(ratio).any() else None,
                            "readable": bool(readable), "of": [[a, b], [c, d]]}
    return out


def band_of(h: dict, gain: str = "ck_gain", share: str = "ck_share") -> str:
    """readings.bands, exactly one: from gnn_gain, a kernel's gain over the twin and its share of the GNN's, on one half
    and metric (the primary kernel's by default; the per-query reference's, descriptive, with ref_gain and ref_share)."""
    if h["contrasts"]["gnn_gain"]["ci"][0] <= 0:
        return "GNN_GAIN_ABSENT"
    s = h["shares"][share]
    if s["readable"] and s["point"] is not None and s["ci"] is not None and s["point"] >= KEEP["share"] and s["ci"][0] >= KEEP["share_low"]:
        return "CK_KEEPS"
    if h["contrasts"][gain]["ci"][0] > 0:
        return "CK_PARTIAL"
    return "CK_NO_GAIN"


def compiled_check(arrays: dict, kernels: list[str]) -> dict:
    out = {}
    for k in kernels:
        differ = np.zeros(arrays["half"].size, dtype=bool)
        for m in METRIC_NAMES:
            differ |= arrays[f"{k}/{m}"] != arrays[f"{k}@compiled/{m}"]
        out[k] = {"queries_where_a_metric_differs": int(differ.sum())}
    return out


def trace(history: list) -> list:
    """A fit's early-stopping trace without its wall seconds: what two runs of one fit can share."""
    return [{k: v for k, v in e.items() if k != "seconds"} for e in history]


def repeat_check(decl: dict, arrays: dict) -> dict:
    rep = decl["fitting"]["repeat"]
    a, b = fit_key(rep["arm"], int(rep["seed"])), fit_key(rep["arm"], int(rep["seed"]), True)
    ra, rb = read_json(FITS / f"{a}.json"), read_json(FITS / f"{b}.json")
    metrics_equal = all(np.array_equal(arrays[f"{a}/{m}"], arrays[f"{b}/{m}"]) for m in METRIC_NAMES)
    diffs = {m: float(np.max(np.abs(arrays[f"{a}/{m}"] - arrays[f"{b}/{m}"]))) for m in METRIC_NAMES}
    same_weights = ra["state_content_sha256"] == rb["state_content_sha256"]
    return {"fit": a, "repeat": b, "weights_bit_identical": bool(same_weights), "metrics_identical": bool(metrics_equal),
            "largest_metric_difference": max(diffs.values()), "trace_identical": trace(ra["history"]) == trace(rb["history"])}


def stage_read(decl: dict, log=log_utc) -> dict:
    """readings: per half and metric the arms' seed means, the contrasts and shares with their intervals; the band on the
    primary half and metric; the flags; the compiled-form and repeat checks; the per-seed spread; the mechanism readouts."""
    rec = read_json(EVAL / f"{NAME}.json")
    if rec is None:
        hard_stop("read: no eval record")
    with np.load(EVAL / f"{NAME}.npz") as z:
        arrays = {k: z[k] for k in z.files}
    half = arrays["half"]
    masks = {"V2_HELD_CONFIRMATION": ~half, "V2_GATE": half, "whole": np.ones(half.size, dtype=bool)}
    r = decl["readings"]
    primary_half, metric, co = r["primary_half"], r["primary_metric"], r["co_read_metric"]
    halves = {h: {m: read_half(decl, arrays, mask, m) for m in READ_METRICS} for h, mask in masks.items()}
    bands = {h: {m: band_of(halves[h][m]) for m in READ_METRICS} for h in halves}
    reference_bands = ({h: {m: band_of(halves[h][m], "ref_gain", "ref_share") for m in READ_METRICS} for h in halves}
                       if reference_kernel(decl) else None)
    reading = bands[primary_half][metric]
    p = halves[primary_half][metric]
    kernels = [k for k in rec["compiled_form"]]
    comp = compiled_check(arrays, kernels)
    for k in comp:
        comp[k]["max_abs_score_difference"] = rec["compiled_form"][k]["max_abs_score_difference"]
    rep = repeat_check(decl, arrays)
    flags = []
    if p["contrasts"]["ck_vs_gnn"]["ci"][1] >= 0:
        flags.append("CK_NOT_BELOW_GNN")
    if p["contrasts"]["edge_gain"]["ci"][0] > 0:
        flags.append("EDGES_CARRY")
    if p["contrasts"]["self_gain"]["ci"][0] > 0:
        flags.append("SELF_CARRIES")
    if "compiled_cost" in p["contrasts"] and p["contrasts"]["compiled_cost"]["ci"][1] < 0:
        flags.append("COMPILED_COST")
    if any(v["queries_where_a_metric_differs"] for v in comp.values()):
        flags.append("COMPILED_DIFFERS")
    if not (rep["weights_bit_identical"] and rep["metrics_identical"]):
        flags.append("REPEAT_DIFFERS")
    other = [h for h in ("V2_GATE", "V2_HELD_CONFIRMATION") if h != primary_half][0]
    if bands[other][metric] != reading:
        flags.append("GATE_HALF_DIFFERS")
    if bands[primary_half][co] != reading:
        flags.append("FC5_DIFFERS")
    mech = {}
    for a in decl["arms"]:
        keys = [fit_key(a, s) for s in SEEDS]
        mech[a] = {m: round(float(np.mean([arrays[f"{k}/{m}"][masks[primary_half]].mean() for k in keys])), 4)
                   for m in ("delta_ratio", "top1_changed") if all(f"{k}/{m}" in arrays for k in keys)}
    fixed = {h: {c: round(float(arrays[f"fixed:{c}/{metric}"][mask].mean()), 4) for c in ("rrf",)} for h, mask in masks.items()}
    out = {"utc": V2.utc(), "dataset": NAME, "reading": reading, "primary": {"half": primary_half, "metric": metric, "co_read": co},
           "primary_kernel": primary_kernel(decl), "per_query_reference": reference_kernel(decl),
           "flags": flags, "bands": bands, "reference_bands": reference_bands, "halves": halves, "compiled_form": comp, "repeat": rep,
           "mechanism": mech, "fixed_rrf": fixed,
           "latency": rec["latency"], "eval_sha256": sha256_file(EVAL / f"{NAME}.json"), "arrays_sha256": sha256_file(EVAL / f"{NAME}.npz"),
           "git_head": git_head(), "module_sha256": module_shas()}
    atomic_json(OUT / "read.json", out)
    log(f"read: {reading} on {primary_half} {metric}; flags {flags}")
    return out


# ── stage: doc, file (laptop) ────────────────────────────────────────────────


def fmt_ci(c: dict) -> str:
    return f"{c['point']:+.4f} [{c['ci'][0]:+.4f}, {c['ci'][1]:+.4f}]"


def fmt_share(s: dict) -> str:
    if not s["readable"] or s["ci"] is None:
        return f"{s['point']:.3f} (not read: the denominator's interval reaches 0)" if s["point"] is not None else "not read"
    return f"{s['point']:.3f} [{s['ci'][0]:.3f}, {s['ci'][1]:.3f}]"


def stage_doc(decl: dict, log=log_utc) -> Path:
    """record.json from the host's read.json, verify.json, eval record and fit records (no arithmetic), then the document."""
    read, ver, ev = read_json(OUT / "read.json"), read_json(OUT / "verify.json"), read_json(EVAL / f"{NAME}.json")
    if read is None or ver is None or ev is None:
        raise SystemExit("doc: read.json, verify.json and the eval record are fetched from the host first")
    fits = {}
    for a, s, rep in declared_fits(decl):
        f = read_json(FITS / f"{fit_key(a, s, rep)}.json")
        if f is None:
            raise SystemExit(f"doc: {fit_key(a, s, rep)}.json is missing")
        fits[f["key"]] = {"arm_key": a, "seed": s, "repeat": rep, "parameters": f["parameters"], "best_epoch": f["best_epoch"],
                          "epochs_run": f["epochs_run"], "select_recall@5": f["best_select_macro_recall5"], "seconds": f["seconds"],
                          "state_content_sha256": f["state_content_sha256"], "cuda_peak_bytes": f.get("cuda_peak_bytes"),
                          "determinism_warnings": sum(w["count"] for w in f["warnings"] if w.get("determinism")),
                          "deviations": f["deviations"], "batch_check": f.get("batch_check")}
    record = {"file": "configs/deploy_ck_2wiki.yaml", "utc": V2.utc(), "reading": read["reading"], "flags": read["flags"],
              "primary": read["primary"], "primary_kernel": read["primary_kernel"], "per_query_reference": read["per_query_reference"],
              "bands": read["bands"], "reference_bands": read["reference_bands"], "halves": read["halves"], "compiled_form": read["compiled_form"],
              "repeat": read["repeat"], "mechanism": read["mechanism"], "fixed_rrf": read["fixed_rrf"], "latency": read["latency"],
              "fits": fits, "verify": {k: ver[k] for k in ("status", "mirror_verify_record_sha256", "freeze_RECORD_SHA256", "stores_built_here",
                                                            "csr_content_sha256", "fit_queries", "select_queries", "bank_rows")}
                         | {"batch_check": ver.get("batch_check")},
              "eval": {"queries": ev["queries"], "ids_sha256": ev["ids_sha256"], "halves": ev["halves"], "ceiling_recall@5": ev["ceiling_as_compiled"].get("recall_ceiling@5"),
                       "any_gold_at_pool": ev["any_gold_at_pool"], "m3b_fixed_rrf_agreement": ev["m3b_fixed_rrf_agreement"]["ok"],
                       "mrr_audit_ok": all(v["ok"] for v in ev["mrr_audit"].values()), "seconds": ev["seconds"],
                       "placement": {k: ev["placement"].get(k) for k in ("host", "env", "device_name", "driver", "torch", "cuda")},
                       "deviations": ev["deviations"], "determinism_warnings": sum(w["count"] for w in ev["warnings"] if w.get("determinism")),
                       "batch_check": ev.get("batch_check")},
              "read_sha256": sha256_file(OUT / "read.json"), "eval_sha256": sha256_file(EVAL / f"{NAME}.json")}
    atomic_json(RECORD, record)
    DOC.write_text(LF.join(doc_lines(decl, record)) + LF, encoding="utf-8")
    log(f"doc: {shown(RECORD)} and {shown(DOC)}")
    return DOC


def doc_lines(decl: dict, rec: dict) -> list[str]:
    p = rec["primary"]
    ph, pm = p["half"], p["metric"]
    arms = list(decl["arms"])
    ref = rec.get("per_query_reference")
    L = [f"# Deployable 2wiki compiled-kernel model (configs/deploy_ck_2wiki.yaml)", "",
         f"**Reading: `{rec['reading']}`** for the primary kernel `{rec['primary_kernel']}` on {ph} {pm} (co-read {p['co_read']}); flags: "
         + (", ".join(f"`{f}`" for f in rec["flags"]) or "none") + ".", ""]
    if ref:
        L += [f"The per-query reference `{ref}` (descriptive, the same bands): `{rec['reference_bands'][ph][pm]}` on {ph} {pm}.", ""]
    L += [
         "Every arm was fitted from the gold labels on 2wiki's own fit carve under universal-v2's frozen rule, early-stopped on its select "
         "carve, and scored on the M3B eval population, all on host_gpu_det. No GNN output, probe or kernel of the MP-Approx ladder enters "
         "any arm. The kernel is one hop of learned propagation (an MP arm by configs/universal_v2.yaml's definition), not QLS-U.", "",
         "## Arms (seed means per query, then the mean)", ""]
    for h in ("V2_HELD_CONFIRMATION", "V2_GATE", "whole"):
        L += [f"### {h} ({rec['halves'][h][pm]['queries']} queries)", "", "| arm | parameters | " + " | ".join(READ_METRICS) + " | per-seed R@5 |",
              "|---|---:|" + "---:|" * len(READ_METRICS) + "---|"]
        for a in arms:
            L.append(f"| {a} | {decl['arms'][a]['parameters']:,} | " + " | ".join(f"{rec['halves'][h][m]['arms'][a]:.4f}" for m in READ_METRICS) +
                     " | " + ", ".join(f"{v:.4f}" for v in rec["halves"][h]["recall@5"]["per_seed"][a]) + " |")
        L += ["", f"| contrast | " + " | ".join(READ_METRICS) + " |", "|---|" + "---|" * len(READ_METRICS)]
        for c in rec["halves"][h][pm]["contrasts"]:
            of = rec["halves"][h][pm]["contrasts"][c]["of"]
            L.append(f"| {c} ({of[0]} - {of[1]}) | " + " | ".join(fmt_ci(rec["halves"][h][m]["contrasts"][c]) for m in READ_METRICS) + " |")
        for s in rec["halves"][h][pm]["shares"]:
            L.append(f"| {s} | " + " | ".join(fmt_share(rec["halves"][h][m]["shares"][s]) for m in READ_METRICS) + " |")
        L += ["", "Bands: " + ", ".join(f"{m} `{rec['bands'][h][m]}`" for m in READ_METRICS) + "."
              + ("" if not ref else " Reference: " + ", ".join(f"{m} `{rec['reference_bands'][h][m]}`" for m in READ_METRICS) + "."), ""]
    L += ["## Checks", ""]
    for k, v in rec["compiled_form"].items():
        L.append(f"- Compiled form of `{k}`: largest score difference from the forward {v['max_abs_score_difference']:.2e}; "
                 f"queries where a metric differs: {v['queries_where_a_metric_differs']}.")
    r = rec["repeat"]
    L.append(f"- Repeat of `{r['fit']}` in a fresh process: weights bit-identical {r['weights_bit_identical']}, metrics identical {r['metrics_identical']}.")
    e = rec["eval"]
    L.append(f"- Eval population {e['queries']} queries (ids {e['ids_sha256'][:12]}), halves {e['halves']['V2_GATE']} / {e['halves']['V2_HELD_CONFIRMATION']}; "
             f"recall ceiling@5 {e['ceiling_recall@5']:.4f}; M3B fixed-rrf agreement {e['m3b_fixed_rrf_agreement']}; MRR audit {e['mrr_audit_ok']}.")
    v = rec["verify"]
    L.append(f"- Inputs: the mirror's 2wiki record VERIFIED (sha256 {v['mirror_verify_record_sha256'][:12]}), freeze {v['freeze_RECORD_SHA256'][:12]}; "
             f"family stores equal by content to the laptop's (built on the host: {[f for f, b in v['stores_built_here'].items() if b]}).")
    bc = v.get("batch_check")
    if bc:
        own = all((f.get("batch_check") or {}).get("equal") for f in rec["fits"].values()) and bool((e.get("batch_check") or {}).get("equal"))
        L.append(f"- Batches (amendment 2): the first {bc['fit_batches_per_seed']} batches of seeds {bc['fit_seeds']} and the first "
                 f"{bc['eval_batches']} eval batches, packed on the host, equal the laptop's field by field (reference "
                 f"{bc['reference_sha256'][:12]}); each fit and the eval pass compared their own again before use: {own}.")
    L += ["", "## Fits", "", "| fit | best epoch | epochs | select R@5 | minutes | weights |", "|---|---:|---:|---:|---:|---|"]
    for k, f in rec["fits"].items():
        L.append(f"| {k} | {f['best_epoch']} | {f['epochs_run']} | {f['select_recall@5']:.4f} | {f['seconds'] / 60:.1f} | {f['state_content_sha256'][:12]} |")
    L += ["", "The select numbers are each fit's early-stopping trace, never a result.", "", "## Latency (systems only)", "",
          "Batch of one on the first 500 eval queries, host_gpu_det (synchronised) and the host CPU at 8 threads, seed 0 of each arm. "
          "Compile is the unchanged per-query feature compile; the definitive end-to-end timing is the fast-compiler step's, not this file's.", "",
          "| where | p50 ms | p95 ms | p99 ms |", "|---|---:|---:|---:|"]
    for k, v in rec["latency"].items():
        if v:
            L.append(f"| {k} | {v['p50_ms']} | {v['p95_ms']} | {v['p99_ms']} |")
    L += ["", "## Mechanism (primary half, seed means)", "", "| arm | delta ratio | top-1 changed |", "|---|---:|---:|"]
    for a, m in rec["mechanism"].items():
        L.append(f"| {a} | {m.get('delta_ratio', float('nan')):.4f} | {m.get('top1_changed', float('nan')):.4f} |")
    L += ["", "## Wording", "", str(decl["readings"]["wording"]).strip(), ""]
    return L


def stage_file(decl: dict, date: str, commit: str, extra: dict | None = None, log=log_utc) -> None:
    """run_record_deploy_ck_2wiki_<date> appended after the code-identity check; status DECLARED_NOT_RUN -> RUN."""
    text = CONFIG.read_text(encoding="utf-8")
    key = f"run_record_deploy_ck_2wiki_{date}"
    if key in decl:
        raise SystemExit(f"{key} exists")
    if decl["status"] != "DECLARED_NOT_RUN":
        raise SystemExit(f"status is {decl['status']}, not DECLARED_NOT_RUN")
    if committed_lf_sha(commit, "scripts/deploy_ck_2wiki.py") is None:
        raise SystemExit(f"{commit} does not hold scripts/deploy_ck_2wiki.py")
    rec = read_json(RECORD)
    jobs = {"verify": read_json(OUT / "verify.json")["module_sha256"], "eval": read_json(EVAL / f"{NAME}.json")["module_sha256"],
            "read": read_json(OUT / "read.json")["module_sha256"]}
    for a, s, rep in declared_fits(decl):
        jobs[f"fit/{fit_key(a, s, rep)}"] = read_json(FITS / f"{fit_key(a, s, rep)}.json")["module_sha256"]
    bad = code_problems(jobs, lambda path: committed_lf_sha(commit, path))
    if bad:
        hard_stop("identical_code: the host jobs' recorded module sha256 values are not one set equal to the committed files", problems=bad)
    primary = rec["halves"][rec["primary"]["half"]][rec["primary"]["metric"]]
    run = {"utc": V2.utc(), "status_moves": "DECLARED_NOT_RUN -> RUN", "terminal": "STOP_FOR_REVIEW", "code_commit": commit,
           "reading": rec["reading"], "flags": rec["flags"], "primary": rec["primary"],
           "arms_primary": primary["arms"], "contrasts_primary": {c: {"point": v["point"], "ci": v["ci"]} for c, v in primary["contrasts"].items()},
           "shares_primary": {s: {"point": v["point"], "ci": v["ci"], "readable": v["readable"]} for s, v in primary["shares"].items()},
           "bands": rec["bands"], "compiled_form": rec["compiled_form"], "repeat": rec["repeat"],
           "identical_code": f"{len({p for s in jobs.values() for p in s})} module paths over {len(jobs)} host processes, one sha256 each, "
                             "equal to the committed files",
           "placement": "verify, fits, eval and read host-native (fits and eval on host_gpu_det); doc and file on the laptop",
           "fits": {k: {"best_epoch": f["best_epoch"], "epochs_run": f["epochs_run"], "minutes": round(f["seconds"] / 60, 1)} for k, f in rec["fits"].items()},
           "record_sha256": sha256_file(RECORD), "document": DOC.relative_to(ROOT).as_posix(), "document_sha256": lf_sha256(DOC)}
    if extra:
        run.update(extra)
    block = yaml.safe_dump(clean({key: run}), sort_keys=False, width=160, allow_unicode=True)
    text = text.replace("status: DECLARED_NOT_RUN", "status: RUN", 1)
    CONFIG.write_text(text.rstrip(LF) + LF + LF + block, encoding="utf-8")
    log(f"filed {key}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="The deployable 2wiki compiled-kernel model (configs/deploy_ck_2wiki.yaml)")
    ap.add_argument("--stage", required=True, choices=["pins", "transfer", "reference", "verify", "fit", "eval", "read", "doc", "file"])
    ap.add_argument("--arm", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--repeat", action="store_true")
    ap.add_argument("--date", default=None)
    ap.add_argument("--commit", default=None)
    ap.add_argument("--extra", default=None, help="a JSON file of run-record fields (deviations, job ids)")
    ap.add_argument("--code", nargs="*", default=None, help="pins: the frozen code paths to pin")
    args = ap.parse_args(argv)
    if args.stage == "pins":
        stage_pins(args.code or list(load_declaration()["inputs"]["frozen_code_lf"]))
        return
    decl = load_declaration()
    if args.stage == "transfer":
        stage_transfer(decl)
    elif args.stage == "reference":
        stage_reference(decl)
    elif args.stage == "verify":
        stage_verify(decl)
    elif args.stage == "fit":
        if args.arm is None or args.seed is None:
            raise SystemExit("--stage fit needs --arm and --seed")
        stage_fit(decl, args.arm, args.seed, args.repeat)
    elif args.stage == "eval":
        stage_eval(decl)
    elif args.stage == "read":
        stage_read(decl)
    elif args.stage == "doc":
        stage_doc(decl)
    else:
        if not args.date or not args.commit:
            raise SystemExit("--stage file needs --date and --commit")
        extra = json.loads(Path(args.extra).read_text(encoding="utf-8")) if args.extra else None
        stage_file(decl, args.date, args.commit, extra)


if __name__ == "__main__":
    main()
