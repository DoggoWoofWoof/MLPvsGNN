"""The CPU-GPU equivalence test (configs/cpu_gpu_equivalence.yaml): on inputs that are bit-identical by construction, do
the lab host's CPU and GPU compute the frozen universal-v2 models' per-query scores and training gradients within the
repository's own tolerance of the laptop CPU, the machine that produced every filed number?

Stages, in the declared order (execution.order):
  bundle  laptop: the first 16 batches of the frozen stage-2 seed-0 draw, packed by the frozen code, with the relation
          bank, the model inputs and a manifest of the sha256 of every file and every field (outputs/cpu_gpu_equivalence/bundle)
  arm     one placement arm in this process: its settings, its environment, the integrity checks, then its tiers --
          T1 forward, T2 gradient, T3 trajectory, T4 clock -- under outputs/cpu_gpu_equivalence/arms/<arm>/
  arms    several arms in the order given, one fresh process each
  read    the floor first, then every candidate against laptop_cpu_t8: the cells, the verdicts, the reported statistics
  doc     docs/CPU_GPU_EQUIVALENCE.md from record.json
  file    run_record_cpu_gpu_equivalence_<date> appended to the declaration; status DECLARED_NOT_RUN -> RUN

It fits nothing, selects nothing and reads no retrieval number: metric values on fit-carve queries are computed only to
compare one placement with another, and their levels are never tabulated.
"""

from __future__ import annotations

import os
import sys

# placements: every arm is one process that sets its settings, reads them back and records them. The thread count and,
# for the determinism mode, the cuBLAS workspace must reach the process before numpy, torch and CUDA start.
_ALL = ("T1", "T2", "T3", "T4")
ARMS = {
    "laptop_cpu_t8": {"role": "reference", "host": "laptop", "device": "cpu", "threads": 8, "mode": None, "env": None, "tiers": _ALL},
    "laptop_cpu_t8_r2": {"role": "repeat", "of": "laptop_cpu_t8", "host": "laptop", "device": "cpu", "threads": 8, "mode": None, "env": None,
                         "tiers": ("T1",)},
    "laptop_cpu_t4": {"role": "floor", "host": "laptop", "device": "cpu", "threads": 4, "mode": None, "env": None, "tiers": _ALL},
    "host_cpu_t8": {"role": "candidate", "host": "host", "device": "cpu", "threads": 8, "mode": None, "env": "mpr-cpu@31803e6457ab", "tiers": _ALL},
    "host_gpu_det": {"role": "candidate", "host": "host", "device": "cuda", "threads": 8, "mode": "det", "env": "mpr-cu128@62fc45e9e1ba",
                     "tiers": _ALL},
    "host_gpu_det_r2": {"role": "repeat", "of": "host_gpu_det", "host": "host", "device": "cuda", "threads": 8, "mode": "det",
                        "env": "mpr-cu128@62fc45e9e1ba", "tiers": ("T1",)},
    "host_gpu_default": {"role": "candidate", "host": "host", "device": "cuda", "threads": 8, "mode": "default", "env": "mpr-cu128@62fc45e9e1ba",
                         "tiers": _ALL},
    "host_gpu_default_r2": {"role": "repeat", "of": "host_gpu_default", "host": "host", "device": "cuda", "threads": 8, "mode": "default",
                            "env": "mpr-cu128@62fc45e9e1ba", "tiers": ("T1",)},
}


def _arm_in_argv() -> dict | None:
    if "--arm" in sys.argv:
        i = sys.argv.index("--arm")
        return ARMS.get(sys.argv[i + 1]) if i + 1 < len(sys.argv) else None
    return None


_EARLY = _arm_in_argv()
if _EARLY is not None:
    for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[_var] = str(_EARLY["threads"])
    if _EARLY["mode"] == "det":
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    else:
        os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "src", ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from mp_retrieval.device_placement import (CUBLAS_WORKSPACE, batch_to, model_to, on_device, placement_block,  # noqa: E402
                                           placement_settings)
from mp_retrieval.m3b_features import FAMILIES  # noqa: E402
from mp_retrieval.m3b_models import PackedBatch, listwise_loss, parameter_count  # noqa: E402
from mp_retrieval.m3b_train import METRIC_NAMES, draw_indices, pack_parts, rank_metrics  # noqa: E402

CONFIG = ROOT / "configs" / "cpu_gpu_equivalence.yaml"
OUT = ROOT / "outputs" / "cpu_gpu_equivalence"
BUNDLE = OUT / "bundle"
ARMS_DIR = OUT / "arms"
RECORD = OUT / "record.json"
DOC = ROOT / "docs" / "CPU_GPU_EQUIVALENCE.md"
SIX_FITS = ROOT / "outputs" / "universal_v2" / "six" / "fits"
TRIO_FITS = ROOT / "outputs" / "universal_v2" / "fits"
V2_CACHE = ROOT / "outputs" / "universal_v2" / "cache"
MODULE = ROOT / "src" / "mp_retrieval" / "device_placement.py"

FIELDS = ("x", "qptr", "node_query", "emb", "qemb", "seedw", "seed_nodes", "edge_index", "edge_attr", "gold")
SEED, BATCH_SIZE, DATASET_DRAW = 0, 16, "per_query"       # probe.batches: the frozen stage-2 draw of seed 0
N_BATCHES, MIN_BATCHES = 16, 8
BUNDLE_CAP_BYTES = 1.0e9                                   # probe.size_cap "1.0 GB", read as 10^9 bytes
DRAW_COUNTS = {"2wiki": 46, "hotpotqa": 43, "metaqa": 35, "musique": 52, "squad": 40, "webqsp": 40}
GNN_ARM, TWIN_ARM = "u_gnn_v2_ef", "u_mlp_v2_mix"
PARAMETERS = {GNN_ARM: 420_932, TWIN_ARM: 330_955}
N_COLUMNS = 129
CORE_SHA256 = "8d1da88b14dfdee13dd32ee6a05f6d9733d14a9899d5e011c41b0cce53000bf9"
BANK_ROWS, BANK_SHA256 = 7_067, "57d34d3688602e634b92d401dfad9333cd1cfae448d8dffe2cee1be34f10a2ed"
REFERENCE, FLOOR = "laptop_cpu_t8", "laptop_cpu_t4"
CANDIDATES = ("host_cpu_t8", "host_gpu_det", "host_gpu_default")
ATOL = RTOL = 1e-5                          # tolerances.forward_inherited
LOSS_REL, LOSS_ABS = 1e-5, 1e-6             # tolerances.gradient.loss
GRAD_REL, GRAD_ABS = 1e-4, 1e-6             # tolerances.gradient.global and per_tensor
T3_AFTER = (1, 2, 4, 8, 16)
LR, WEIGHT_DECAY, CLIP = 1e-3, 1e-4, 1.0    # the frozen training rule, for T3
LAPTOP = {"python": "3.13.5", "torch": "2.8.0+cpu", "numpy": "2.3.2", "cpu": "Family 6 Model 186", "logical_cpus": 12}
HOST_TORCH = {"cpu": "2.8.0+cpu", "cuda": "2.8.0+cu128"}
HOST_NUMPY = "2.3.2"
GPU_NAME, GPU_DRIVER = "NVIDIA RTX 4500 Ada Generation", "596.71"


# ── small things ─────────────────────────────────────────────────────────────


def utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log_utc(msg: str) -> None:
    print(f"[{utc()}] {msg}", flush=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def lf_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def tensor_sha256(t: torch.Tensor) -> str:
    """sha256 of a tensor's dtype, shape and bytes in C order: equal on two machines iff all three are."""
    t = t.detach().to("cpu").contiguous()
    h = hashlib.sha256(f"{t.dtype}|{tuple(t.shape)}|".encode())
    h.update(t.numpy().tobytes())
    return h.hexdigest()


def git_head() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None if out.returncode == 0 else None


def jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return jsonable(obj.tolist())
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def load_declaration() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


class HardStop(SystemExit):
    pass


def hard_stop(what: str, evidence: dict, where: str) -> None:
    """hard_stops: the evidence is written beside the outputs (fetched with an arm's outputs), the file stage refuses
    while any exists, and the status line stays where it is; the hard_stop_<date> block is filed from it."""
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"hard_stop_{where}.json"
    path.write_text(json.dumps({"utc": utc(), "what": what, "evidence": jsonable(evidence)}, indent=1, default=str), encoding="utf-8")
    raise HardStop(f"HARD STOP ({where}): {what} -- the evidence is in {path.relative_to(ROOT).as_posix()}")


def v2():
    import universal_v2_run as V2  # noqa: E402  (imported, never edited)
    return V2


# ── pins ─────────────────────────────────────────────────────────────────────


def frozen_code_problems(decl: dict) -> dict:
    """frozen_code: every pinned file's LF-normalised sha256; the mismatches (None: the file is missing)."""
    return {rel: (lf_sha256(ROOT / rel) if (ROOT / rel).exists() else None) for rel, want in decl["frozen_code"].items()
            if not (ROOT / rel).exists() or lf_sha256(ROOT / rel) != want}


def model_specs(decl: dict) -> list[dict]:
    """object_under_test.models in the declaration's order, with their files (object_under_test.paths)."""
    out = []
    for key, pins in decl["object_under_test"]["models"].items():
        arm = GNN_ARM if key.startswith(GNN_ARM + "__") else TWIN_ARM if key.startswith(TWIN_ARM + "__") else None
        if arm is None:
            raise SystemExit(f"{key}: neither {GNN_ARM} nor {TWIN_ARM}")
        d = SIX_FITS if arm == GNN_ARM else TRIO_FITS
        out.append({"key": key, "arm": arm, "seed": int(key.rsplit("__s", 1)[1]), "pt": d / f"{key}.pt", "json": d / f"{key}.json",
                    "weights": pins["weights"], "record": pins["record"]})
    return out


def checkpoint_problems(decl: dict) -> dict:
    """Every checkpoint and record against object_under_test.models; the record's own state_sha256 too."""
    bad = {}
    for s in model_specs(decl):
        got = {"weights": sha256_file(s["pt"]) if s["pt"].exists() else None, "record": sha256_file(s["json"]) if s["json"].exists() else None}
        state = json.loads(s["json"].read_text(encoding="utf-8")).get("state_sha256") if s["json"].exists() else None
        if got["weights"] != s["weights"] or got["record"] != s["record"] or state != s["weights"]:
            bad[s["key"]] = {**got, "record_state_sha256": state}
    return bad


def frozen_inputs(where: str) -> dict:
    """UNIVERSAL_V2_CORE_CONTRACT as the arms read it, from the configs alone (universal_v2_run.model_inputs, whose
    load_configs also checks the M3B pins)."""
    V2 = v2()
    cfg, cfg_m3b, _ = V2.load_configs()
    inputs = V2.model_inputs(cfg, cfg_m3b)
    if inputs["core_sha256"] != CORE_SHA256 or int(inputs["n_scalars"]) != N_COLUMNS:
        hard_stop("the frozen contract", {"core_sha256": inputs["core_sha256"], "n_scalars": inputs["n_scalars"]}, where)
    return inputs


def load_models(decl: dict, inputs: dict, bank_emb: torch.Tensor, where: str, keys=None) -> dict:
    """The six checkpoints on the CPU in eval mode, built by the frozen make_model (the six-dataset bank set on the GNN),
    each parameter count checked before its weights are loaded."""
    V2 = v2()
    models = {}
    for s in model_specs(decl):
        if keys is not None and s["key"] not in keys:
            continue
        torch.manual_seed(0)                   # the initial draw is overwritten by the checkpoint
        model = V2.make_model(s["arm"], inputs, bank_emb)
        n = parameter_count(model)
        if n != PARAMETERS[s["arm"]]:
            hard_stop("a parameter count", {"model": s["key"], "parameters": n, "declared": PARAMETERS[s["arm"]]}, where)
        model.load_state_dict(torch.load(s["pt"], map_location="cpu", weights_only=True))
        model.eval()
        models[s["key"]] = model
    return models


# ── the bundle ───────────────────────────────────────────────────────────────


def save_batch(batch: PackedBatch, path: Path) -> None:
    torch.save({f: getattr(batch, f) for f in FIELDS}, path)


def read_batch(path: Path) -> PackedBatch:
    d = torch.load(path, map_location="cpu", weights_only=True)
    return PackedBatch(**{f: d[f] for f in FIELDS})


def field_pins(batch: PackedBatch) -> dict:
    return {f: {"dtype": str(getattr(batch, f).dtype), "shape": list(getattr(batch, f).shape), "sha256": tensor_sha256(getattr(batch, f))}
            for f in FIELDS}


def replay_draw(fits: dict, n_batches: int) -> list:
    """The first ``n_batches`` batches of fit_model's draw for seed 0: the same generator, the same cursors and the
    pinned draw_indices, called in the order the fit loop calls it (packing ahead changes no draw)."""
    rng = np.random.default_rng(SEED)
    names = sorted(fits)
    cursors = {n: [rng.permutation(fits[n].trainable), 0] for n in names}
    return [draw_indices(fits, names, cursors, rng, BATCH_SIZE, DATASET_DRAW) for _ in range(n_batches)]


def draw_counts(parts: list) -> dict:
    counts: dict = {}
    for batch_parts in parts:
        for name, idx in batch_parts:
            counts[name] = counts.get(name, 0) + int(len(idx))
    return dict(sorted(counts.items()))


def stage_bundle(log=log_utc) -> dict:
    """probe.bundle: packed on the laptop by the frozen code (open_six, fit kind; CarveDataV2.pack) -- fit-carve rows
    only, each carve checked against its declared sha256 before a row is read."""
    if (BUNDLE / "manifest.json").exists():
        log("bundle/manifest.json exists, not rebuilt")
        return json.loads((BUNDLE / "manifest.json").read_text(encoding="utf-8"))
    decl = load_declaration()
    bad = frozen_code_problems(decl)
    if bad:
        hard_stop("frozen_code", bad, "bundle")
    V2 = v2()
    import universal_v2_six as SIX  # noqa: E402  (imported, never edited)
    t0 = time.time()
    cfg, cfg_m3b, _ = V2.load_configs()
    inputs, _, _, _, _, bank, carves, _ = SIX.open_six(cfg, cfg_m3b, kinds=("fit",))
    fits = carves["fit"]
    carves_seen = {n: fits[n].meta.get("carve_sha256_declared") for n in sorted(fits)}
    if carves_seen != dict(sorted(decl["probe"]["fit_carves_sha256"].items())):
        hard_stop("the fit carves", {"declared": decl["probe"]["fit_carves_sha256"], "caches": carves_seen}, "bundle")
    if inputs["core_sha256"] != CORE_SHA256 or int(inputs["n_scalars"]) != N_COLUMNS:
        hard_stop("the frozen contract", {"core_sha256": inputs["core_sha256"], "n_scalars": inputs["n_scalars"]}, "bundle")
    emb = bank.embeddings.detach().to(torch.float32).contiguous()
    bank_sha = hashlib.sha256(np.ascontiguousarray(emb.numpy()).tobytes()).hexdigest()
    if bank.n_rows != BANK_ROWS or bank.sha256 != BANK_SHA256 or bank_sha != BANK_SHA256:
        hard_stop("the relation bank", {"rows": bank.n_rows, "sha256": bank.sha256, "recomputed": bank_sha}, "bundle")
    parts = replay_draw(fits, N_BATCHES)
    counts = draw_counts(parts)
    if counts != DRAW_COUNTS:
        hard_stop("the probe draw", {"declared": DRAW_COUNTS, "replayed": counts}, "bundle")
    BUNDLE.mkdir(parents=True, exist_ok=True)
    fields, draws = {}, []
    for k, batch_parts in enumerate(parts):
        batch = pack_parts(fits, batch_parts, FAMILIES)
        if not bool(batch.gold.any()):
            hard_stop("a probe batch without gold", {"batch": k}, "bundle")   # the fit record skipped no batch
        ptr = batch.qptr.numpy()
        rows, j = [], 0
        for name, idx in batch_parts:
            for i in idx:
                qd = fits[name].query(int(i))
                local = np.flatnonzero(batch.gold[int(ptr[j]):int(ptr[j + 1])].numpy())
                if not np.array_equal(local, np.unique(qd["gold"])):
                    raise SystemExit(f"batch {k} query {j}: the packed gold is not the carve's gold")
                rows.append({"dataset": name, "fit_index": int(i), "qrow": int(fits[name].qrow[int(i)]), "gold": qd["gold"].tolist(),
                             "gold_total": int(qd["gold_total"]), "candidates": int(ptr[j + 1] - ptr[j])})
                j += 1
        if j != batch.n_queries:
            raise SystemExit(f"batch {k}: {j} draws for {batch.n_queries} packed queries")
        name_k = f"batch_{k:02d}.pt"
        save_batch(batch, BUNDLE / name_k)
        fields[name_k] = field_pins(batch)
        draws.append({"batch": k, "qptr": ptr.tolist(), "queries": rows})
        log(f"   batch {k}: {batch.n_queries} draws, {int(ptr[-1])} candidates, {int(batch.edge_index.shape[1])} edges, "
            f"{(BUNDLE / name_k).stat().st_size / 1e6:.1f} MB")
    torch.save({"embeddings": emb.clone(), "offsets": {str(k): int(v) for k, v in bank.offsets.items()}, "sha256": bank.sha256}, BUNDLE / "bank.pt")
    (BUNDLE / "inputs.json").write_text(json.dumps(jsonable(inputs), indent=1), encoding="utf-8")
    sizes = [(BUNDLE / f"batch_{k:02d}.pt").stat().st_size for k in range(N_BATCHES)]
    fixed = (BUNDLE / "bank.pt").stat().st_size + (BUNDLE / "inputs.json").stat().st_size
    B = N_BATCHES
    while fixed + sum(sizes[:B]) > BUNDLE_CAP_BYTES and B > MIN_BATCHES:
        B -= 1
    if fixed + sum(sizes[:B]) > BUNDLE_CAP_BYTES:
        raise SystemExit(f"the bundle of {MIN_BATCHES} batches is {(fixed + sum(sizes[:B])) / 1e9:.2f} GB, over the 1.0 GB cap: "
                         "probe.size_cap does not cover it; stop for review")
    for k in range(B, N_BATCHES):
        (BUNDLE / f"batch_{k:02d}.pt").unlink()
    (BUNDLE / "draws.json").write_text(json.dumps({"seed": SEED, "batch_size": BATCH_SIZE, "dataset_draw": DATASET_DRAW, "batches": draws[:B]},
                                                  indent=1), encoding="utf-8")
    names = [f"batch_{k:02d}.pt" for k in range(B)] + ["bank.pt", "inputs.json", "draws.json"]
    files = {n: {"bytes": (BUNDLE / n).stat().st_size, "sha256": sha256_file(BUNDLE / n)} for n in names}
    manifest = {"phase": "CPU_GPU_EQUIVALENCE", "stage": "bundle", "utc": utc(), "git_head": git_head(), "batches": B,
                "batches_declared": N_BATCHES, "draw_counts_declared_16": counts, "draw_counts": draw_counts(parts[:B]),
                "bytes_total": int(sum(f["bytes"] for f in files.values())), "cap_bytes": BUNDLE_CAP_BYTES, "files": files,
                "fields": {n: fields[n] for n in names if n in fields}, "bank": {"rows": bank.n_rows, "sha256": bank.sha256},
                "core_sha256": inputs["core_sha256"], "fit_carves_sha256": carves_seen, "placement": placement_block(None),
                "seconds": round(time.time() - t0, 1)}
    (BUNDLE / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    log(f"bundle: {B} batches, {manifest['bytes_total'] / 1e9:.3f} GB, draws {manifest['draw_counts']}, {manifest['seconds']}s")
    return manifest


# ── an arm ───────────────────────────────────────────────────────────────────


def load_manifest() -> dict:
    p = BUNDLE / "manifest.json"
    if not p.exists():
        raise SystemExit("no bundle/manifest.json: the bundle precedes every arm (execution.order)")
    return json.loads(p.read_text(encoding="utf-8"))


def bundle_problems(manifest: dict) -> tuple[dict, list]:
    """integrity_in_every_arm, the bundle: every file, then every reconstructed field, against the manifest."""
    bad = {}
    for fname, pin in manifest["files"].items():
        p = BUNDLE / fname
        got = sha256_file(p) if p.exists() else None
        if got != pin["sha256"]:
            bad[fname] = got
    batches = []
    if bad:
        return bad, batches
    for k in range(int(manifest["batches"])):
        name = f"batch_{k:02d}.pt"
        batch = read_batch(BUNDLE / name)
        got = field_pins(batch)
        for f in FIELDS:
            if got[f] != manifest["fields"][name][f]:
                bad[f"{name}:{f}"] = got[f]
        batches.append(batch)
    return bad, batches


def load_bank() -> tuple[torch.Tensor, str]:
    d = torch.load(BUNDLE / "bank.pt", map_location="cpu", weights_only=True)
    emb = d["embeddings"]
    return emb, hashlib.sha256(np.ascontiguousarray(emb.numpy()).tobytes()).hexdigest()


def environment_problems(spec: dict, place: dict) -> dict:
    """hard_stops: an arm's environment is not the one declared for it (env name@hash, torch version, device name);
    the settings the arm set are read back and must be the declared ones."""
    bad = {}

    def need(key, want, got):
        if got != want:
            bad[key] = {"declared": want, "found": got}

    if spec["host"] == "laptop":
        need("python", LAPTOP["python"], place["python"])
        need("torch", LAPTOP["torch"], place["torch"])
        need("numpy", LAPTOP["numpy"], place["numpy"])
        if LAPTOP["cpu"] not in str(place["cpu"]) or place["logical_cpus"] != LAPTOP["logical_cpus"]:
            bad["machine"] = {"declared": f"{LAPTOP['cpu']}, {LAPTOP['logical_cpus']} logical CPUs", "found": f"{place['cpu']}, {place['logical_cpus']}"}
    else:
        need("env", spec["env"], place["env"])
        need("torch", HOST_TORCH[spec["device"]], place["torch"])
        need("numpy", HOST_NUMPY, place["numpy"])
    need("threads", spec["threads"], place["threads"])
    if spec["device"] == "cuda":
        need("device_name", GPU_NAME, place.get("device_name"))
        need("cuda_matmul_allow_tf32", False, place["cuda_matmul_allow_tf32"])
        need("cudnn_allow_tf32", False, place["cudnn_allow_tf32"])
        need("float32_matmul_precision", "highest", place["float32_matmul_precision"])
        need("cudnn_benchmark", False, place["cudnn_benchmark"])
        need("deterministic_algorithms", spec["mode"] == "det", place["deterministic_algorithms"])
        need("cublas_workspace_config", CUBLAS_WORKSPACE if spec["mode"] == "det" else None, place["cublas_workspace_config"])
    return bad


def sync(device) -> None:
    if on_device(device):
        torch.cuda.synchronize()


def centre(s: torch.Tensor, batch: PackedBatch) -> torch.Tensor:
    """tests/test_m3b_models.py::_centred_scores on scores already on the CPU: each score minus its query's mean."""
    ones = torch.ones_like(s)
    n = torch.zeros(batch.n_queries).index_add_(0, batch.node_query, ones)
    tot = torch.zeros(batch.n_queries).index_add_(0, batch.node_query, s)
    return s - (tot / n)[batch.node_query]


@torch.no_grad()
def tier_forward(models: dict, batches: list, placed: list, draws: list, device) -> tuple[dict, dict]:
    """T1: every model in eval() under no_grad on every probe batch; per draw the raw and centred scores and the 14
    METRIC_NAMES of rank_metrics on the raw scores. The forward is timed for T4 (CUDA synchronised around it)."""
    arrays, clock = {}, {}
    for key, model in models.items():
        model.eval()
        raw, cen, rows, ms = [], [], [], []
        for k, (batch, on) in enumerate(zip(batches, placed)):
            sync(device)
            t0 = time.perf_counter()
            s = model(on)
            sync(device)
            ms.append(round((time.perf_counter() - t0) * 1e3, 3))
            s = s.detach().to("cpu")
            raw.append(s)
            cen.append(centre(s, batch))
            ptr, sn = batch.qptr.numpy(), s.numpy()
            for j, q in enumerate(draws[k]["queries"]):
                m = rank_metrics(sn[ptr[j]:ptr[j + 1]], np.asarray(q["gold"], dtype=np.int64), int(q["gold_total"]))
                rows.append([m[name] for name in METRIC_NAMES])
        arrays[f"raw__{key}"] = torch.cat(raw).numpy()
        arrays[f"centred__{key}"] = torch.cat(cen).numpy()
        arrays[f"metrics__{key}"] = np.asarray(rows, dtype=np.float64)
        clock[key] = ms
    return arrays, clock


def tier_gradient(models: dict, placed: list, device, out: Path) -> dict:
    """T2: every model in eval() (dropout off) with gradients on, on every probe batch separately, from the loaded
    weights: listwise_loss, backward, nothing stepped. The loss and every parameter tensor's gradient are kept; the
    forward+backward is timed for T4 and the peak GPU memory read."""
    clock = {}
    for key, model in models.items():
        model.eval()
        params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        losses, grads, absent, ms = [], {n: [] for n, _ in params}, set(), []
        if on_device(device):
            torch.cuda.reset_peak_memory_stats()
        for on in placed:
            model.zero_grad(set_to_none=True)
            sync(device)
            t0 = time.perf_counter()
            with torch.enable_grad():
                loss = listwise_loss(model(on), on)
                loss.backward()
            sync(device)
            ms.append(round((time.perf_counter() - t0) * 1e3, 3))
            losses.append(loss.detach().to("cpu").reshape(()))
            for n, p in params:
                if p.grad is None:
                    absent.add(n)
                    grads[n].append(torch.zeros(p.shape, dtype=p.dtype))
                else:
                    grads[n].append(p.grad.detach().to("cpu", copy=True))
        model.zero_grad(set_to_none=True)
        torch.save({"loss": torch.stack(losses), "names": [n for n, _ in params], "grads": {n: torch.stack(v) for n, v in grads.items()},
                    "grad_none_in_some_batch": sorted(absent)}, out / f"t2__{key}.pt")
        clock[key] = {"ms": ms, "peak_gpu_bytes": int(torch.cuda.max_memory_allocated()) if on_device(device) else None}
    return clock


def tier_trajectory(decl: dict, inputs: dict, bank_emb: torch.Tensor, batches: list, placed: list, device, where: str) -> tuple[dict, dict]:
    """T3 (descriptive, never a gate): seed 0 of each family, dropout off, a fresh AdamW at the frozen lr, weight decay
    and clip, taking the probe batches as consecutive steps; centred scores on every probe draw after steps 1, 2, 4, 8
    and 16. The optimiser is the one fit_model builds, with torch's defaults for the device."""
    keys = [s["key"] for s in model_specs(decl) if s["seed"] == 0]
    models = load_models(decl, inputs, bank_emb, where, keys=keys)
    after = tuple(s for s in T3_AFTER if s <= len(placed))
    arrays, losses = {}, {}
    for key, model in models.items():
        if on_device(device):
            model_to(model, device)
        model.eval()
        optimiser = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        losses[key] = []
        for step, on in enumerate(placed, start=1):
            optimiser.zero_grad(set_to_none=True)
            loss = listwise_loss(model(on), on)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            optimiser.step()
            losses[key].append(float(loss.detach()))
            if step in after:
                with torch.no_grad():
                    arrays[f"centred__{key}__s{step}"] = torch.cat([centre(model(o).detach().to("cpu"), b) for b, o in zip(batches, placed)]).numpy()
    return arrays, losses


def tier_clock(t1_clock: dict, t2_clock: dict) -> dict:
    """T4 (systems, never a gate): per probe batch the forward wall clock (the T1 pass) and the forward+backward wall
    clock (the T2 pass), packing excluded and CUDA synchronised around each, and the peak GPU memory of the T2 pass."""
    out = {}
    for key, fwd in t1_clock.items():
        fb = t2_clock.get(key, {}).get("ms")
        out[key] = {"forward_ms": fwd, "forward_ms_median": float(np.median(fwd)), "forward_ms_first": float(fwd[0]),
                    "forward_backward_ms": fb, "forward_backward_ms_median": float(np.median(fb)) if fb else None,
                    "forward_backward_ms_first": float(fb[0]) if fb else None, "peak_gpu_bytes": t2_clock.get(key, {}).get("peak_gpu_bytes")}
    return out


def warning_summary(caught) -> list:
    seen: dict = {}
    for w in caught:
        text = f"{w.category.__name__}: {w.message}"
        seen[text] = seen.get(text, 0) + 1
    return [{"warning": k, "count": v, "determinism": "determinis" in k.lower()} for k, v in sorted(seen.items())]


def stage_arm(arm: str, log=log_utc) -> dict:
    """One placement arm in this process: settings first (before any CUDA work), then the environment, the integrity
    checks, and the arm's tiers. Any failed pin is a hard stop, never a verdict."""
    spec = ARMS[arm]
    out = ARMS_DIR / arm
    if (out / "done.json").exists():
        log(f"{arm}: done.json exists, not rerun")
        return json.loads((out / "done.json").read_text(encoding="utf-8"))
    t_start, u_start = time.time(), utc()
    decl = load_declaration()
    device = spec["device"]
    if device == "cuda" and not torch.cuda.is_available():
        hard_stop("the arm's device", {"arm": arm, "cuda_available": False, "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES")}, arm)
    placement_settings(spec["mode"], spec["threads"])
    place = placement_block(device)
    bad = environment_problems(spec, place)
    if bad:
        hard_stop("the arm's environment", {"arm": arm, "problems": bad, "placement": place}, arm)
    deviations = []
    if device == "cuda" and place.get("driver") != GPU_DRIVER:
        deviations.append(f"driver {place.get('driver')}, the declaration names {GPU_DRIVER}")
    manifest = load_manifest()
    bad = frozen_code_problems(decl)
    if bad:
        hard_stop("frozen_code", bad, arm)
    bad = checkpoint_problems(decl)
    if bad:
        hard_stop("the checkpoints", bad, arm)
    bad, batches = bundle_problems(manifest)
    if bad:
        hard_stop("the bundle", bad, arm)
    bank_emb, bank_sha = load_bank()
    if bank_sha != BANK_SHA256 or int(bank_emb.shape[0]) != BANK_ROWS:
        hard_stop("the relation bank", {"rows": int(bank_emb.shape[0]), "sha256": bank_sha}, arm)
    inputs = frozen_inputs(arm)
    shipped = json.loads((BUNDLE / "inputs.json").read_text(encoding="utf-8"))
    if jsonable(inputs) != shipped:
        hard_stop("the model inputs", {"differ": sorted(k for k in shipped if jsonable(inputs).get(k) != shipped[k])}, arm)
    draws = json.loads((BUNDLE / "draws.json").read_text(encoding="utf-8"))["batches"]
    models = load_models(decl, inputs, bank_emb, arm)
    integrity = {"bundle_files": len(manifest["files"]), "bundle_fields": len(batches) * len(FIELDS), "checkpoints": len(model_specs(decl)),
                 "bank_sha256": bank_sha, "frozen_code_files": len(decl["frozen_code"]), "core_sha256": inputs["core_sha256"],
                 "parameters": {k: parameter_count(m) for k, m in models.items()}, "all_equal": True}
    out.mkdir(parents=True, exist_ok=True)
    log(f"{arm}: integrity holds ({len(batches)} batches, {len(models)} models); device {device}, {spec['threads']} threads, mode {spec['mode']}")
    seconds, t2_clock = {}, {}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        placed = [batch_to(b, device) for b in batches] if on_device(device) else batches
        if on_device(device):
            for m in models.values():
                model_to(m, device)
        t = time.time()
        t1, t1_clock = tier_forward(models, batches, placed, draws, device)
        np.savez(out / "t1.npz", **t1)
        seconds["T1"] = round(time.time() - t, 1)
        log(f"   T1 {seconds['T1']}s")
        if "T2" in spec["tiers"]:
            t = time.time()
            t2_clock = tier_gradient(models, placed, device, out)
            seconds["T2"] = round(time.time() - t, 1)
            log(f"   T2 {seconds['T2']}s")
        if "T3" in spec["tiers"]:
            t = time.time()
            t3, t3_losses = tier_trajectory(decl, inputs, bank_emb, batches, placed, device, arm)
            np.savez(out / "t3.npz", **t3)
            (out / "t3_losses.json").write_text(json.dumps(t3_losses, indent=1), encoding="utf-8")
            seconds["T3"] = round(time.time() - t, 1)
            log(f"   T3 {seconds['T3']}s")
        if "T4" in spec["tiers"]:
            (out / "t4.json").write_text(json.dumps(tier_clock(t1_clock, t2_clock), indent=1), encoding="utf-8")
    done = {"arm": arm, "spec": {k: list(v) if isinstance(v, tuple) else v for k, v in spec.items()}, "utc_start": u_start, "utc_end": utc(),
            "seconds_total": round(time.time() - t_start, 1), "seconds": seconds, "placement": place, "deviations": deviations,
            "warnings": warning_summary(caught), "integrity": integrity, "bundle_manifest_sha256": sha256_file(BUNDLE / "manifest.json"),
            "batches": len(batches), "script_sha256": lf_sha256(Path(__file__)), "module_sha256": lf_sha256(MODULE), "git_head": git_head(),
            "argv": sys.argv[1:], "rx_job_id": os.environ.get("RX_JOB_ID")}
    (out / "done.json").write_text(json.dumps(done, indent=1, default=str), encoding="utf-8")
    log(f"{arm}: done in {done['seconds_total']}s; {sum(w['count'] for w in done['warnings'])} warnings recorded")
    return done


def stage_arms(arms: list, log=log_utc) -> None:
    """Several arms, one fresh process each, in the order given (one_process_per_arm); a failed arm stops the rest."""
    unknown = [a for a in arms if a not in ARMS]
    if unknown:
        raise SystemExit(f"unknown arms {unknown}; one of {sorted(ARMS)}")
    for arm in arms:
        log(f"== {arm} (fresh process)")
        rc = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--stage", "arm", "--arm", arm], cwd=ROOT).returncode
        if rc != 0:
            raise SystemExit(f"{arm}: exited {rc}; the later arms were not started")


# ── the read ─────────────────────────────────────────────────────────────────


def draw_pointer(draws: list) -> np.ndarray:
    """The candidate offsets of every query draw over the batch-concatenated score vectors."""
    ptr, off = [0], 0
    for b in draws:
        q = np.asarray(b["qptr"], dtype=np.int64)
        ptr.extend((off + q[1:]).tolist())
        off += int(q[-1])
    return np.asarray(ptr, dtype=np.int64)


def forward_cells(a: np.ndarray, b: np.ndarray, ptr: np.ndarray) -> dict:
    """tolerances.forward_inherited, one cell per query draw: |a - b| <= ATOL + RTOL |b| elementwise with b the
    reference (torch.testing.assert_close(a, b, atol, rtol), evaluated in float64); bit: equal as torch.equal is."""
    a64, b64 = a.astype(np.float64), b.astype(np.float64)
    diff = np.abs(a64 - b64)
    tol = ATOL + RTOL * np.abs(b64)
    starts = ptr[:-1]
    excess = np.maximum.reduceat(diff - tol, starts)
    return {"ok": ~(excess > 0) & ~np.isnan(excess), "excess": excess, "ratio": np.maximum.reduceat(diff / tol, starts),
            "max_abs": np.maximum.reduceat(diff, starts), "bit": np.asarray([np.array_equal(a[s:e], b[s:e]) for s, e in zip(ptr[:-1], ptr[1:])])}


def gradient_cells(c: dict, r: dict) -> dict:
    """tolerances.gradient, per batch: the loss, the global gradient norm and every parameter tensor, in float64."""
    names = list(r["names"])
    if list(c["names"]) != names:
        raise SystemExit("the gradient files name different parameters")
    B = int(r["loss"].shape[0])
    L, Lr = c["loss"].double(), r["loss"].double()
    diff_sq, ref_sq, bit = [], [], (c["loss"] == r["loss"]).numpy()
    for n in names:
        g, gr = c["grads"][n].double().reshape(B, -1), r["grads"][n].double().reshape(B, -1)
        diff_sq.append(((g - gr) ** 2).sum(1))
        ref_sq.append((gr ** 2).sum(1))
        bit &= np.asarray([torch.equal(c["grads"][n][k], r["grads"][n][k]) for k in range(B)])
    diff_sq, ref_sq = torch.stack(diff_sq), torch.stack(ref_sq)          # (tensors, B)
    g_ref = ref_sq.sum(0).sqrt()
    out = {"names": names, "bit": bit,
           "loss_lhs": (L - Lr).abs().numpy(), "loss_rhs": (LOSS_REL * Lr.abs() + LOSS_ABS).numpy(),
           "global_lhs": diff_sq.sum(0).sqrt().numpy(), "global_rhs": (GRAD_REL * g_ref).numpy(),
           "tensor_lhs": diff_sq.sqrt().numpy(), "tensor_rhs": (GRAD_REL * ref_sq.sqrt() + GRAD_ABS * g_ref.unsqueeze(0)).numpy(),
           "g_ref": g_ref.numpy(), "loss_ref": Lr.numpy()}
    out["loss_ok"] = out["loss_lhs"] <= out["loss_rhs"]
    out["global_ok"] = out["global_lhs"] <= out["global_rhs"]
    out["tensor_ok"] = out["tensor_lhs"] <= out["tensor_rhs"]
    return out


def compare_forward(run: str, t1c, t1r, keys: list, ptr: np.ndarray) -> dict:
    fails, failing, bit, stats = set(), [], True, {}
    for key in keys:
        cells = forward_cells(np.asarray(t1c[f"centred__{key}"]), np.asarray(t1r[f"centred__{key}"]), ptr)
        bit = bit and bool(cells["bit"].all())
        for q in np.flatnonzero(~cells["ok"]):
            fails.add(("T1", key, int(q)))
            failing.append({"tier": "T1", "run": run, "model": key, "draw": int(q), "excess": float(cells["excess"][q])})
        stats[key] = {"draws": int(ptr.size - 1), "draws_failing": int((~cells["ok"]).sum()), "draws_bit_identical": int(cells["bit"].sum()),
                      "max_abs_centred_difference": float(np.max(cells["max_abs"])), "max_tolerance_ratio": float(np.max(cells["ratio"]))}
    return {"fails": fails, "failing": failing, "bit": bit, "stats": stats, "cells": len(keys) * int(ptr.size - 1)}


def compare_gradient(run: str, t2c: dict, t2r: dict, keys: list) -> dict:
    fails, failing, bit, stats, cells = set(), [], True, {}, 0
    for key in keys:
        g = gradient_cells(t2c[key], t2r[key])
        bit = bit and bool(g["bit"].all())
        B = g["loss_ok"].size
        cells += B * (2 + len(g["names"]))
        for b in range(B):
            for check, ok, lhs, rhs in (("loss", g["loss_ok"][b], g["loss_lhs"][b], g["loss_rhs"][b]),
                                        ("global", g["global_ok"][b], g["global_lhs"][b], g["global_rhs"][b])):
                if not ok:
                    fails.add(("T2", key, b, check))
                    failing.append({"tier": "T2", "run": run, "model": key, "batch": b, "check": check, "excess": float(lhs - rhs)})
            for t, n in enumerate(g["names"]):
                if not g["tensor_ok"][t, b]:
                    fails.add(("T2", key, b, f"tensor:{n}"))
                    failing.append({"tier": "T2", "run": run, "model": key, "batch": b, "check": f"tensor:{n}",
                                    "excess": float(g["tensor_lhs"][t, b] - g["tensor_rhs"][t, b])})
        loss_rel = g["loss_lhs"] / np.maximum(np.abs(g["loss_ref"]), 1e-300)
        grad_rel = g["global_lhs"] / np.maximum(g["g_ref"], 1e-300)
        stats[key] = {"loss_relative_error_max": float(loss_rel.max()), "loss_relative_error_median": float(np.median(loss_rel)),
                      "gradient_relative_error_max": float(grad_rel.max()), "gradient_relative_error_median": float(np.median(grad_rel)),
                      "batches_bit_identical": int(g["bit"].sum()), "batches": int(B)}
    return {"fails": fails, "failing": failing, "bit": bit, "stats": stats, "cells": cells}


def verdict(fails: set, floor_fails: set, bit_identical: bool) -> str:
    """readings.exactly_one: the first of the four that holds."""
    if bit_identical and not fails:
        return "EQUIVALENT_BIT_IDENTICAL"
    if not fails:
        return "EQUIVALENT_WITHIN_TOLERANCE"
    if fails - floor_fails:
        return "NOT_EQUIVALENT"
    return "INCONCLUSIVE_FLOOR"


def has_near_tie(scores: np.ndarray) -> bool:
    """Two candidates whose scores differ by less than twice the forward tolerance at their size."""
    s = np.sort(np.asarray(scores, dtype=np.float64))
    if s.size < 2:
        return False
    tol = 2 * (ATOL + RTOL * np.maximum(np.abs(s[1:]), np.abs(s[:-1])))
    return bool((np.diff(s) < tol).any())


def decision_agreement(t1c, t1r, keys: list, ptr: np.ndarray, ok_by_model: dict) -> dict:
    """reported_beside_never_gating.decision agreement: per model, the query draws whose 14 metrics differ from the
    reference in any place, how many of them hold forward_inherited and sit on a near tie of the reference, and the
    largest |difference| of any metric's probe mean."""
    out = {}
    for key in keys:
        a, b = np.asarray(t1c[f"metrics__{key}"]), np.asarray(t1r[f"metrics__{key}"])
        same = (a == b) | (np.isnan(a) & np.isnan(b))
        differ = np.flatnonzero(~same.all(1))
        raw = np.asarray(t1r[f"raw__{key}"])
        tie = [int(q) for q in differ if ok_by_model[key][q] and has_near_tie(raw[ptr[q]:ptr[q + 1]])]
        with np.errstate(invalid="ignore"):
            gap = np.abs(np.nanmean(a, 0) - np.nanmean(b, 0))
        out[key] = {"draws_differing": int(differ.size), "differing_on_a_near_tie_within_tolerance": len(tie),
                    "largest_probe_mean_difference": float(np.nanmax(gap)) if np.isfinite(gap).any() else 0.0,
                    "metric_of_largest": METRIC_NAMES[int(np.nanargmax(gap))] if np.isfinite(gap).any() and np.nanmax(gap) > 0 else None}
    return out


def load_t2(arm: str, keys: list) -> dict:
    return {k: torch.load(ARMS_DIR / arm / f"t2__{k}.pt", map_location="cpu", weights_only=True) for k in keys}


GATE_DIR = OUT / "device_path_tests"
CUDA_TESTS = ("test_a_cuda_fit_checkpoints_cpu_tensors_and_resumes_to_the_uninterrupted_fit",
              "test_the_cuda_evaluation_is_the_cpu_evaluation_away_from_near_ties")


def junit_outcomes(path: Path) -> dict:
    import xml.etree.ElementTree as ET  # noqa: E402
    root = ET.parse(path).getroot()
    out = {}
    for tc in root.iter("testcase"):
        out[tc.get("name")] = ("failed" if tc.find("failure") is not None or tc.find("error") is not None else
                               "skipped" if tc.find("skipped") is not None else "passed")
    return out


def device_path_gate() -> dict:
    """device_path.gate: the device-path tests passed on the laptop and on the host GPU (its CUDA tests run, not
    skipped), on the module and test file every arm ran with."""
    files = {"laptop": (GATE_DIR / "laptop_cpu.xml", GATE_DIR / "Inspiron-14__cpu.json"),
             "host_gpu": (GATE_DIR / "host_gpu.xml", GATE_DIR / "DESKTOP-SLQMEQH__cuda.json")}
    want = {"tests/test_cpu_gpu_equivalence.py": lf_sha256(ROOT / "tests" / "test_cpu_gpu_equivalence.py"),
            "src/mp_retrieval/device_placement.py": lf_sha256(MODULE), "scripts/cpu_gpu_equivalence.py": lf_sha256(Path(__file__))}
    gate = {}
    for where, (xml, rep) in files.items():
        if not xml.exists() or not rep.exists():
            raise SystemExit(f"device_path.gate: no {where} report ({xml.name}, {rep.name}); the tests precede every candidate arm")
        outcomes = junit_outcomes(xml)
        report = json.loads(rep.read_text(encoding="utf-8"))
        counts = {k: sum(v == k for v in outcomes.values()) for k in ("passed", "failed", "skipped")}
        stale = sorted(k for k, v in want.items() if report.get("sha256_lf", {}).get(k) != v)
        gate[where] = {**counts, "utc": report["utc"], "stale_files": stale, "seen": report.get("seen", {}),
                       "cuda_tests": {t: outcomes.get(t) for t in CUDA_TESTS}}
        if counts["failed"] or stale:
            raise SystemExit(f"device_path.gate: {where} has {counts['failed']} failed tests, stale files {stale}")
    if any(v != "passed" for v in gate["host_gpu"]["cuda_tests"].values()):
        raise SystemExit(f"device_path.gate: the host GPU's CUDA tests did not all pass: {gate['host_gpu']['cuda_tests']}")
    return gate


def stage_read(log=log_utc) -> dict:
    """readings: floor_first, then each candidate from its first run (T1, T2) and its _r2 repeat (T1) against
    laptop_cpu_t8 over all six models and all probe batches; the reported statistics beside, never gating."""
    decl = load_declaration()
    manifest = load_manifest()
    msha = sha256_file(BUNDLE / "manifest.json")
    draws = json.loads((BUNDLE / "draws.json").read_text(encoding="utf-8"))["batches"]
    ptr = draw_pointer(draws)
    keys = [s["key"] for s in model_specs(decl)]
    done = {a: json.loads((ARMS_DIR / a / "done.json").read_text(encoding="utf-8")) for a in ARMS if (ARMS_DIR / a / "done.json").exists()}
    missing = [a for a in ARMS if a not in done]
    if missing:
        raise SystemExit(f"arms not run: {missing}; the read follows every arm")
    code = {(d["script_sha256"], d["module_sha256"]) for d in done.values()}
    if len(code) != 1:
        raise SystemExit(f"the arms ran different code: {sorted(code)}; one script, identical in every arm")
    stale = [a for a, d in done.items() if d["bundle_manifest_sha256"] != msha or int(d["batches"]) != int(manifest["batches"])]
    if stale:
        raise SystemExit(f"arms {stale} read another bundle than bundle/manifest.json")
    early = [a for a, d in done.items() if d["utc_start"] < manifest["utc"]]
    if early:
        raise SystemExit(f"arms {early} started before the bundle's B was fixed ({manifest['utc']}); probe.size_cap files B before any arm")
    gate = device_path_gate()
    first_candidate = min(done[a]["utc_start"] for a in ARMS if ARMS[a]["host"] == "host")
    if max(g["utc"] for g in gate.values()) > first_candidate:
        raise SystemExit("device_path.gate: a device-path report is later than the first host arm")
    t1 = {a: np.load(ARMS_DIR / a / "t1.npz") for a in ARMS}
    ref_t2 = load_t2(REFERENCE, keys)
    log(f"read: {len(keys)} models, {ptr.size - 1} draws, {manifest['batches']} batches; floor first")
    floor_f = compare_forward(FLOOR, t1[FLOOR], t1[REFERENCE], keys, ptr)
    floor_g = compare_gradient(FLOOR, load_t2(FLOOR, keys), ref_t2, keys)
    floor = {"arm": FLOOR, "cells": {"T1": floor_f["cells"], "T2": floor_g["cells"]},
             "failing": floor_f["failing"] + floor_g["failing"], "n_failing": len(floor_f["fails"] | floor_g["fails"]),
             "bit_identical": floor_f["bit"] and floor_g["bit"], "T1": floor_f["stats"], "T2": floor_g["stats"]}
    floor_fails = floor_f["fails"] | floor_g["fails"]
    log(f"   floor {FLOOR}: {floor['n_failing']} failing cells of {floor_f['cells'] + floor_g['cells']}")
    candidates, reported = {}, {}
    for cand in (FLOOR,) + CANDIDATES:
        f = floor_f if cand == FLOOR else compare_forward(cand, t1[cand], t1[REFERENCE], keys, ptr)
        g = floor_g if cand == FLOOR else compare_gradient(cand, load_t2(cand, keys), ref_t2, keys)
        ok_by_model = {k: forward_cells(np.asarray(t1[cand][f"centred__{k}"]), np.asarray(t1[REFERENCE][f"centred__{k}"]), ptr)["ok"] for k in keys}
        reported[cand] = {"forward": f["stats"], "gradient": g["stats"], "decision_agreement": decision_agreement(t1[cand], t1[REFERENCE], keys, ptr, ok_by_model)}
        if cand == FLOOR:
            continue
        runs = [cand] + [a for a, s in ARMS.items() if s.get("of") == cand]
        fails, failing, bit, cells = set(f["fails"]) | g["fails"], f["failing"] + g["failing"], f["bit"] and g["bit"], f["cells"] + g["cells"]
        for rep in runs[1:]:
            r = compare_forward(rep, t1[rep], t1[REFERENCE], keys, ptr)
            fails |= r["fails"]
            failing += r["failing"]
            bit = bit and r["bit"]
            cells += r["cells"]
            reported[cand].setdefault("repeat_forward", {})[rep] = r["stats"]
        v = verdict(fails, floor_fails, bit)
        candidates[cand] = {"verdict": v, "runs": runs, "cells": cells, "n_failing": len(fails), "bit_identical": bit,
                            "also_failing_for_the_floor": len(fails & floor_fails),
                            "largest_excess": max((c["excess"] for c in failing), default=None), "failing": failing}
        log(f"   {cand}: {v} ({len(fails)} failing cells of {cells})")
    determinism = {}
    for rep, s in ARMS.items():
        if s["role"] != "repeat":
            continue
        first = s["of"]
        determinism[rep] = {"of": first, **{k: {"raw_bit_identical": bool(np.array_equal(t1[rep][f"raw__{k}"], t1[first][f"raw__{k}"])),
                                                "centred_bit_identical": bool(np.array_equal(t1[rep][f"centred__{k}"], t1[first][f"centred__{k}"]))}
                                            for k in keys}}
    trajectory = {}
    ref_t3 = np.load(ARMS_DIR / REFERENCE / "t3.npz")
    for arm in (FLOOR,) + CANDIDATES:
        t3 = np.load(ARMS_DIR / arm / "t3.npz")
        trajectory[arm] = {}
        for name in ref_t3.files:
            key, step = name[len("centred__"):].rsplit("__s", 1)
            step = int(step)
            trajectory[arm].setdefault(key, {})[step] = float(np.max(np.abs(t3[name].astype(np.float64) - ref_t3[name].astype(np.float64))))
    clocks = {a: json.loads((ARMS_DIR / a / "t4.json").read_text(encoding="utf-8")) for a in ARMS if (ARMS_DIR / a / "t4.json").exists()}
    record = {"phase": "CPU_GPU_EQUIVALENCE", "utc": utc(), "git_head": git_head(), "batches": int(manifest["batches"]), "draws": int(ptr.size - 1),
              "models": keys, "reference": REFERENCE, "bundle": {"manifest_sha256": msha, "bytes_total": manifest["bytes_total"],
                                                                "draw_counts": manifest["draw_counts"], "utc": manifest["utc"],
                                                                "seconds": manifest["seconds"]},
              "device_path_gate": gate, "floor": floor, "candidates": candidates, "reported": reported, "determinism": determinism, "trajectory": trajectory,
              "clocks": {a: {k: {kk: v for kk, v in c.items() if not isinstance(v, list)} for k, c in cl.items()} for a, cl in clocks.items()},
              "arms": {a: {k: d[k] for k in ("utc_start", "utc_end", "seconds_total", "seconds", "placement", "deviations", "warnings",
                                             "integrity", "script_sha256", "module_sha256", "git_head", "rx_job_id")} for a, d in done.items()}}
    RECORD.write_text(json.dumps(jsonable(record), indent=1), encoding="utf-8")
    log("record.json written: " + ", ".join(f"{c} {v['verdict']}" for c, v in candidates.items()))
    return record


# ── the document ─────────────────────────────────────────────────────────────


def sentence(text) -> str:
    t = " ".join(str(text).split())
    return t[:1].upper() + t[1:]


def e2(x) -> str:
    return "—" if x is None else f"{x:.1e}" if x != 0 else "0"


def gate_paragraph(gate: dict) -> str:
    lap, gpu = gate["laptop"], gate["host_gpu"]
    seen = gpu["seen"]
    resume, ev = seen.get("cuda_fit_resume", {}), seen.get("cuda_evaluation", {})
    compared = sum(ev[c]["compared"] for c in ("select", "fit") if c in ev)
    ties = sum(ev[c]["near_ties"] for c in ("select", "fit") if c in ev)
    queries = sum(ev[c]["queries"] for c in ("select", "fit") if c in ev)
    warned = sorted(set(resume.get("determinism_warnings", [])) | set(ev.get("determinism_warnings", [])))
    return (f"`tests/test_cpu_gpu_equivalence.py` passed on the laptop ({lap['passed']} passed, {lap['skipped']} skipped: the two CUDA tests, "
            f"no CUDA device) and on the host GPU in `mpr-cu128` under the `host_gpu_det` settings ({gpu['passed']} passed, {gpu['skipped']} "
            "skipped) before any candidate arm ran. On the CPU the device-path copies give the pinned fit and evaluation exactly (weights "
            "`torch.equal`, records equal apart from wall-clock fields, every metric array equal). On CUDA a synthetic GAT fit wrote CPU tensors "
            "to its checkpoint and resumed to the uninterrupted fit " + ("bit for bit (weights `torch.equal`)" if resume.get("weights_torch_equal")
                                                                          else "within tolerances.forward_inherited") +
            f", and the CUDA evaluation gave the CPU evaluation's metrics on every compared synthetic query ({compared} of {queries} compared; "
            f"{ties} with a near tie at twice the forward tolerance were counted and left out). Determinism warnings: "
            + ("; ".join(warned) if warned else "none") + ".")


def render_doc(rec: dict, decl: dict) -> str:
    cands, floor, rep = rec["candidates"], rec["floor"], rec["reported"]
    keys = rec["models"]
    short = {k: k.replace("__H128", "").replace("u_gnn_v2_ef__six", "GNN six").replace("u_mlp_v2_mix", "twin") for k in keys}
    arms = rec["arms"]
    L = ["# CPU–GPU equivalence", "",
         "Declaration: `configs/cpu_gpu_equivalence.yaml` (declared at `1ee9e48`). Script: `scripts/cpu_gpu_equivalence.py`; device path: "
         "`src/mp_retrieval/device_placement.py`. Sidecars: `outputs/cpu_gpu_equivalence/` (git-ignored).", "",
         "> " + " ".join(str(decl["equivalence_question"]).split()), "",
         "This decides only **where** later stages may run. Nothing was fitted or selected, and no retrieval number was read: the metrics "
         "below are computed on fit-carve queries only to compare one placement with another.", "",
         "## Verdicts", "",
         "| candidate | verdict | failing cells | cells checked | bit-identical to the reference | largest excess over the tolerance |",
         "|---|---|---:|---:|---|---:|"]
    for c, v in cands.items():
        L.append(f"| `{c}` | **{v['verdict']}** | {v['n_failing']} | {v['cells']:,} | {'yes' if v['bit_identical'] else 'no'} | {e2(v['largest_excess'])} |")
    L += ["", f"The floor, `{FLOOR}` against `{REFERENCE}` (the same laptop at 4 threads), fails {floor['n_failing']} of "
          f"{floor['cells']['T1'] + floor['cells']['T2']:,} cells and is {'' if floor['bit_identical'] else 'not '}bit-identical to the reference. "
          "A candidate's failing cell counts against it only where the floor passes that cell (readings.NOT_EQUIVALENT, INCONCLUSIVE_FLOOR).", "",
          "Cells: T1, one per (model, query draw), the centred scores under `assert_close(atol=1e-5, rtol=1e-5)`; T2, one per (model, batch) for "
          "the loss (`|L - L_ref| <= 1e-5|L_ref| + 1e-6`) and the global gradient (`||g - g_ref|| <= 1e-4||g_ref||`), and one per (model, batch, "
          "parameter tensor) (`||g_p - g_ref_p|| <= 1e-4||g_ref_p|| + 1e-6||g_ref||`). A candidate's `_r2` repeat is held to the T1 gates too.", "",
          "**Wording.** " + sentence(decl["readings"]["wording"]), "",
          "**What a pass opens.** " + sentence(decl["what_a_verdict_opens"]["pass"]), "",
          "**What a fail keeps.** " + sentence(decl["what_a_verdict_opens"]["fail"]), "",
          "## The device-path gate", "", gate_paragraph(rec["device_path_gate"]), "",
          "## The probe", "",
          f"{rec['batches']} batches of the frozen stage-2 draw for seed 0 (batch 16, `dataset_draw per_query`), {rec['draws']} query draws from the "
          "six fit carves: " + ", ".join(f"{n} {c}" for n, c in rec["bundle"]["draw_counts"].items()) +
          f". Packed once on the laptop by the frozen code and shipped as a {rec['bundle']['bytes_total'] / 1e9:.2f} GB bundle (manifest sha256 "
          f"`{rec['bundle']['manifest_sha256'][:16]}…`). Every arm checked every bundle file and every reconstructed field against the manifest, the "
          "six checkpoints and records against the declaration, the six-dataset relation bank (7,067 rows), the frozen contract (129 columns) and "
          "the twelve frozen-code files before it computed anything.", "",
          "## The arms", "",
          "| arm | host | env | torch | device | threads | settings | seconds | warnings |", "|---|---|---|---|---|---:|---|---:|---:|"]
    for a, d in arms.items():
        p = d["placement"]
        dev = p.get("device_name") or p["device"]
        mode = ARMS[a]["mode"] or "—"
        sets = (f"mode {mode}; TF32 {'off' if not p['cuda_matmul_allow_tf32'] else 'on'}; deterministic "
                f"{'on (warn_only)' if p['deterministic_algorithms'] else 'off'}" if ARMS[a]["device"] == "cuda" else "—")
        L.append(f"| `{a}` | {p['host']} | `{p['env']}` | {p['torch']} | {dev}{' (driver ' + str(p.get('driver')) + ')' if p.get('driver') else ''} | "
                 f"{p['threads']} | {sets} | {d['seconds_total']:.0f} | {sum(w['count'] for w in d['warnings'])} |")
    L += ["", "## Reported beside, never gating", "", "### Forward (T1): largest |centred difference| and largest ratio to the tolerance", "",
          "| model | " + " | ".join(f"`{c}`" for c in rep) + " |", "|---|" + "---:|" * len(rep)]
    for k in keys:
        L.append(f"| {short[k]} | " + " | ".join(f"{e2(rep[c]['forward'][k]['max_abs_centred_difference'])} "
                                                f"({rep[c]['forward'][k]['max_tolerance_ratio']:.2f})" for c in rep) + " |")
    L += ["", "### Gradient (T2): loss and global-gradient relative errors, largest (median) over batches", "",
          "| model | " + " | ".join(f"`{c}` loss" for c in rep) + " | " + " | ".join(f"`{c}` gradient" for c in rep) + " |",
          "|---|" + "---:|" * (2 * len(rep))]
    for k in keys:
        L.append(f"| {short[k]} | " + " | ".join(f"{e2(rep[c]['gradient'][k]['loss_relative_error_max'])} ({e2(rep[c]['gradient'][k]['loss_relative_error_median'])})"
                                                for c in rep) + " | " +
                 " | ".join(f"{e2(rep[c]['gradient'][k]['gradient_relative_error_max'])} ({e2(rep[c]['gradient'][k]['gradient_relative_error_median'])})"
                            for c in rep) + " |")
    L += ["", "### Decision agreement", "",
          f"Query draws (of {rec['draws']}) whose 14 metrics differ from the reference in any place; in brackets, how many of those hold the forward "
          "tolerance and sit on a near tie of the reference (two candidates within twice the tolerance).", "",
          "| model | " + " | ".join(f"`{c}`" for c in rep) + " |", "|---|" + "---:|" * len(rep)]
    for k in keys:
        L.append(f"| {short[k]} | " + " | ".join(f"{rep[c]['decision_agreement'][k]['draws_differing']} "
                                                f"({rep[c]['decision_agreement'][k]['differing_on_a_near_tie_within_tolerance']})" for c in rep) + " |")
    L += ["", "### Determinism: is each `_r2` run bit-identical to its first run?", "",
          "| repeat | " + " | ".join(short[k] for k in keys) + " |", "|---|" + "---|" * len(keys)]
    for r, d in rec["determinism"].items():
        L.append(f"| `{r}` | " + " | ".join("yes" if d[k]["raw_bit_identical"] else "no" for k in keys) + " |")
    warned = {a: [w for w in d["warnings"] if w["determinism"]] for a, d in arms.items()}
    L += ["", "Warnings raised by the determinism request: " +
          ("; ".join(f"`{a}`: " + "; ".join(f"{w['warning'][:160]} (×{w['count']})" for w in ws) for a, ws in warned.items() if ws) or "none") + ".", "",
          "### Trajectory (T3, descriptive): largest |centred difference| from the reference's own trajectory", "",
          "Seed 0 of each family, dropout off, a fresh AdamW (lr 1e-3, weight decay 1e-4, clip 1.0) over the probe batches as consecutive steps; "
          "how fast rounding differences grow, beside the same growth for the floor. Context for `placement_rules.new_seeds` only.", ""]
    steps = sorted({int(s) for a in rec["trajectory"].values() for m in a.values() for s in m})
    L += ["| arm | model | " + " | ".join(f"after {s}" for s in steps) + " |", "|---|---|" + "---:|" * len(steps)]
    for a, models in rec["trajectory"].items():
        for k, curve in models.items():
            L.append(f"| `{a}` | {short.get(k, k)} | " + " | ".join(e2(curve.get(str(s), curve.get(s))) for s in steps) + " |")
    L += ["", "### Clock (T4, systems)", "",
          "Milliseconds per probe batch, timed on the T1 forward pass and the T2 gradient pass themselves (no separate pass), packing "
          "excluded, CUDA synchronised around each batch: the median over the family's three checkpoints of each checkpoint's median "
          "over the probe batches, and the first batch (warm-up included). Peak GPU memory is the gradient pass's, with the probe "
          "batches resident on the device.", "",
          "| arm | family | forward median | forward first | forward+backward median | peak GPU MB |", "|---|---|---:|---:|---:|---:|"]
    for a, cl in rec["clocks"].items():
        for fam, prefix in (("GNN", GNN_ARM), ("twin", TWIN_ARM)):
            cs = [c for k, c in cl.items() if k.startswith(prefix + "__")]
            if not cs:
                continue
            peak = max((c.get("peak_gpu_bytes") or 0) for c in cs)
            fb = [c["forward_backward_ms_median"] for c in cs if c.get("forward_backward_ms_median") is not None]
            L.append(f"| `{a}` | {fam} | {np.median([c['forward_ms_median'] for c in cs]):.1f} | "
                     f"{np.median([c['forward_ms_first'] for c in cs]):.1f} | {np.median(fb) if fb else float('nan'):.1f} | "
                     + (f"{peak / 2**20:.0f} |" if peak else "— |"))
    devs = [f"`{a}`: {x}" for a, d in arms.items() for x in d["deviations"]]
    L += ["", "## Placement rules (filed with the declaration, binding every stage placed off the laptop)", ""]
    for name, text in decl["placement_rules"].items():
        L.append(f"- **{name}**: " + " ".join(str(text).split()))
    L += ["", "## Deviations", "", ("; ".join(devs) + ".") if devs else "None recorded by the arms.", ""]
    return "\n".join(L)


def stage_doc(log=log_utc) -> Path:
    if not RECORD.exists():
        raise SystemExit("no record.json: the read precedes the document")
    rec = json.loads(RECORD.read_text(encoding="utf-8"))
    DOC.write_text(render_doc(rec, load_declaration()), encoding="utf-8")
    log(f"wrote {DOC.relative_to(ROOT).as_posix()}")
    return DOC


# ── the run record ───────────────────────────────────────────────────────────


def stage_file(date: str, log=log_utc, extra: dict | None = None) -> None:
    """outputs.run_record: run_record_cpu_gpu_equivalence_<date> appended to the declaration, the status line moved
    DECLARED_NOT_RUN -> RUN, each candidate's verdict on its own line. Refused after a hard stop, or twice."""
    key = f"run_record_cpu_gpu_equivalence_{date}"
    text = CONFIG.read_text(encoding="utf-8")
    decl = yaml.safe_load(text)
    if key in decl:
        raise SystemExit(f"{key} exists; a run record is filed once")
    if decl.get("status") != "DECLARED_NOT_RUN":
        raise SystemExit(f"status is {decl.get('status')}, not DECLARED_NOT_RUN")
    stops = sorted(OUT.glob("hard_stop_*.json"))
    if stops:
        raise SystemExit(f"hard stops were written ({[p.name for p in stops]}); file hard_stop_<date> instead of a run record")
    if not RECORD.exists() or not DOC.exists():
        raise SystemExit("the read and the document precede the run record")
    rec = json.loads(RECORD.read_text(encoding="utf-8"))
    run = {"utc": utc(), "status_moves": "DECLARED_NOT_RUN -> RUN"}
    for cand in CANDIDATES:
        run[f"verdict_{cand}"] = rec["candidates"][cand]["verdict"]
    run["floor"] = (f"{FLOOR} against {REFERENCE}: {rec['floor']['n_failing']} failing cells of "
                    f"{rec['floor']['cells']['T1'] + rec['floor']['cells']['T2']}; bit-identical {rec['floor']['bit_identical']}")
    run["cells"] = {c: {"failing": v["n_failing"], "checked": v["cells"], "bit_identical": v["bit_identical"],
                        "also_failing_for_the_floor": v["also_failing_for_the_floor"], "largest_excess": v["largest_excess"]}
                    for c, v in rec["candidates"].items()}
    run["device_path_gate"] = {w: {k: g[k] for k in ("passed", "failed", "skipped", "utc", "cuda_tests")} for w, g in rec["device_path_gate"].items()}
    run["probe"] = {"batches": rec["batches"], "b_fixed_utc": rec["bundle"]["utc"], "draws": rec["draws"], "draw_counts": rec["bundle"]["draw_counts"],
                    "bundle_bytes": rec["bundle"]["bytes_total"], "bundle_manifest_sha256": rec["bundle"]["manifest_sha256"]}
    run["arms"] = {a: {"host": d["placement"]["host"], "env": d["placement"]["env"], "torch": d["placement"]["torch"],
                       "device": d["placement"].get("device_name") or d["placement"]["device"], "driver": d["placement"].get("driver"),
                       "cuda": d["placement"].get("cuda"), "threads": d["placement"]["threads"], "mode": ARMS[a]["mode"],
                       "deterministic_algorithms": d["placement"]["deterministic_algorithms"],
                       "tf32": d["placement"]["cuda_matmul_allow_tf32"] or d["placement"]["cudnn_allow_tf32"] if ARMS[a]["device"] == "cuda" else None,
                       "float32_matmul_precision": d["placement"]["float32_matmul_precision"],
                       "cublas_workspace_config": d["placement"]["cublas_workspace_config"], "seconds": d["seconds_total"],
                       "determinism_warnings": sum(w["count"] for w in d["warnings"] if w["determinism"]), "utc_end": d["utc_end"],
                       "rx_job_id": d["rx_job_id"]}
                   for a, d in rec["arms"].items()}
    run["determinism_r2_bit_identical"] = {r: all(v["raw_bit_identical"] for k, v in d.items() if k != "of") for r, d in rec["determinism"].items()}
    run["record"] = "outputs/cpu_gpu_equivalence/record.json"
    run["record_sha256"] = sha256_file(RECORD)
    run["document"] = DOC.relative_to(ROOT).as_posix()
    run["document_sha256_lf"] = lf_sha256(DOC)
    run["what_it_opens"] = ("a passing verdict makes that placement NAMEABLE in a later dated authorization block of a declared stage, which "
                            "quotes the tested settings verbatim and runs mirror_verification first; no stage is placed by this record")
    if extra:
        run.update(extra)
    run["terminal"] = "STOP_FOR_REVIEW"
    block = yaml.safe_dump(jsonable({key: run}), sort_keys=False, width=160, allow_unicode=True)
    new = text.replace("status: DECLARED_NOT_RUN", "status: RUN", 1)
    CONFIG.write_text(new.rstrip("\n") + "\n\n" + block, encoding="utf-8")
    log(f"{key} appended; status RUN; " + ", ".join(f"{c} {run['verdict_' + c]}" for c in CANDIDATES))


# ── main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage", required=True, choices=("bundle", "arm", "arms", "read", "doc", "file"))
    ap.add_argument("--arm", choices=sorted(ARMS), help="arm: the placement arm this process runs")
    ap.add_argument("--arms", help="arms: comma-separated arms, one fresh process each, in this order")
    ap.add_argument("--date", help="file: the run record's date, e.g. 2026_09_29")
    ap.add_argument("--extra", type=Path, default=None, help="file: a JSON object of fields added to the run record")
    args = ap.parse_args()
    if (args.stage == "arm") != (args.arm is not None):
        ap.error("--arm goes with --stage arm, and only there (it sets the process's threads before torch starts)")
    if args.stage == "bundle":
        stage_bundle()
    elif args.stage == "arm":
        stage_arm(args.arm)
    elif args.stage == "arms":
        if not args.arms:
            ap.error("--stage arms needs --arms")
        stage_arms([a.strip() for a in args.arms.split(",") if a.strip()])
    elif args.stage == "read":
        stage_read()
    elif args.stage == "doc":
        stage_doc()
    else:
        if not args.date:
            ap.error("--stage file needs --date")
        stage_file(args.date, extra=json.loads(args.extra.read_text(encoding="utf-8")) if args.extra else None)


if __name__ == "__main__":
    main()
