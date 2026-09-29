"""The GPU task-level qualification (configs/gpu_task_qualification.yaml): on fresh probe draws, does the lab host's
GPU in the tested determinism mode make the laptop CPU's retrieval decisions for the frozen universal-v2 models -- the
same top-1 and top-5 per query, hence the same R@5, FullCov@5 and Hit@1 -- and make them reproducibly?

Stages, in the declared order (execution.order):
  bundle  laptop: batches 11-26 of the frozen stage-2 seed-0 draw (the ones after the equivalence probe's 11), packed
          by the frozen code under the 1.0 GB cap, with draws.json and a manifest of every file's and field's sha256
  arm     one placement arm in this process: settings, environment, inventory, integrity, T1 (and T5 on the GPU det
          arms) under outputs/gpu_task_qualification/arms/<arm>/
  arms    several arms in the order given, one fresh process each
  read    the floor first, then host_gpu_det and its repeat against laptop_cpu_t8; the verdicts; the beside readings
  doc     docs/GPU_TASK_QUALIFICATION.md from record.json
  file    run_record_gpu_task_qualification_<date> appended to the declaration; status DECLARED_NOT_RUN -> RUN

scripts/cpu_gpu_equivalence.py is imported unchanged (its LF sha256 is pinned and checked) for its pins, packing, T1
forward and helpers. It fits nothing and selects nothing; metric values on fit-carve queries are computed only to
compare placements, and their levels are never tabulated.
"""

from __future__ import annotations

import os
import sys

# the arms (placements): settings of configs/cpu_gpu_equivalence.yaml#placements, quoted; host_gpu_tf32 is diagnosis only
ARMS = {
    "laptop_cpu_t8": {"role": "reference", "host": "laptop", "device": "cpu", "threads": 8, "mode": None, "tf32": False, "env": None,
                      "tiers": ("T1",)},
    "laptop_cpu_t4": {"role": "floor", "host": "laptop", "device": "cpu", "threads": 4, "mode": None, "tf32": False, "env": None,
                      "tiers": ("T1",)},
    "host_gpu_det": {"role": "candidate", "host": "host", "device": "cuda", "threads": 8, "mode": "det", "tf32": False,
                     "env": "mpr-cu128@62fc45e9e1ba", "tiers": ("T1", "T5")},
    "host_gpu_det_r2": {"role": "repeat", "of": "host_gpu_det", "host": "host", "device": "cuda", "threads": 8, "mode": "det", "tf32": False,
                        "env": "mpr-cu128@62fc45e9e1ba", "tiers": ("T1", "T5")},
    "host_cpu_t8": {"role": "beside", "host": "host", "device": "cpu", "threads": 8, "mode": None, "tf32": False, "env": "mpr-cpu@31803e6457ab",
                    "tiers": ("T1",)},
    "host_gpu_tf32": {"role": "diagnosis", "host": "host", "device": "cuda", "threads": 8, "mode": "det", "tf32": True,
                      "env": "mpr-cu128@62fc45e9e1ba", "tiers": ("T1",)},
}


def _arm_in_argv() -> dict | None:
    if "--arm" in sys.argv:
        i = sys.argv.index("--arm")
        return ARMS.get(sys.argv[i + 1]) if i + 1 < len(sys.argv) else None
    return None


# the thread count and, for the determinism mode, the cuBLAS workspace must reach the process before numpy, torch and CUDA
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
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "src", ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import cpu_gpu_equivalence as CGE  # noqa: E402  (imported, never edited; its bytes are pinned below)
from mp_retrieval.m3b_models import listwise_loss  # noqa: E402

CONFIG = ROOT / "configs" / "gpu_task_qualification.yaml"
OUT = ROOT / "outputs" / "gpu_task_qualification"
BUNDLE = OUT / "bundle"
ARMS_DIR = OUT / "arms"
RECORD = OUT / "record.json"
DOC = ROOT / "docs" / "GPU_TASK_QUALIFICATION.md"
EQ_SCRIPT = ROOT / "scripts" / "cpu_gpu_equivalence.py"
EQ_SCRIPT_SHA256 = "71ffa5d89619e02f50618287c3891d7fefb0893b693c97c6dc1e8c6ae7a9a390"   # object_under_test.frozen_code
EQ_BATCHES = 11                                  # the equivalence probe's B (its run record: probe.batches 11)
FIRST, N_BATCHES, MIN_BATCHES = EQ_BATCHES, 16, 8  # probe.batches: batches 11-26, then probe.size_cap
BUNDLE_CAP_BYTES = 1.0e9
REFERENCE, FLOOR, CANDIDATE, REPEAT = "laptop_cpu_t8", "laptop_cpu_t4", "host_gpu_det", "host_gpu_det_r2"
SAME_HOST_CPU, TF32_ARM = "host_cpu_t8", "host_gpu_tf32"
ATOL = RTOL = CGE.ATOL                            # criteria.score_cells: forward_inherited, unchanged
TASK_METRICS = ("recall@5", "full_coverage@5", "hit@1")
TASK_IDX = tuple(CGE.METRIC_NAMES.index(m) for m in TASK_METRICS)

utc, log_utc, sha256_file, lf_sha256, tensor_sha256, jsonable, git_head = (CGE.utc, CGE.log_utc, CGE.sha256_file, CGE.lf_sha256,
                                                                            CGE.tensor_sha256, CGE.jsonable, CGE.git_head)


class HardStop(SystemExit):
    pass


def hard_stop(what: str, evidence: dict, where: str) -> None:
    """hard_stops: the evidence is written under this phase's outputs (fetched with an arm's outputs); the file stage
    refuses while any exists, and the status line stays where it is."""
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"hard_stop_{where}.json"
    path.write_text(json.dumps({"utc": utc(), "what": what, "evidence": jsonable(evidence)}, indent=1, default=str), encoding="utf-8")
    raise HardStop(f"HARD STOP ({where}): {what} -- the evidence is in {path.relative_to(ROOT).as_posix()}")


def bind_stops() -> None:
    """The imported helpers that stop (frozen_inputs, load_models) stop into THIS phase's outputs, never the
    equivalence file's. Bound by main(), so importing this module changes nothing."""
    CGE.hard_stop = hard_stop


def eq_script_problem() -> str | None:
    got = lf_sha256(EQ_SCRIPT)
    return None if got == EQ_SCRIPT_SHA256 else got


def load_declaration() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def pins_or_stop(where: str) -> dict:
    """The equivalence declaration's pins (models, frozen_code, fit carves) and the imported script's bytes."""
    got = eq_script_problem()
    if got is not None:
        hard_stop("the imported script", {"declared": EQ_SCRIPT_SHA256, "found": got}, where)
    decl_eq = CGE.load_declaration()
    bad = CGE.frozen_code_problems(decl_eq)
    if bad:
        hard_stop("frozen_code", bad, where)
    return decl_eq


# ── the bundle ───────────────────────────────────────────────────────────────


def draw_keys(parts: list) -> list:
    """(dataset, fit_index) of every draw, batch by batch, in packing order."""
    return [[(name, int(i)) for name, idx in batch_parts for i in idx] for batch_parts in parts]


def overlap_with(parts: list, earlier: list) -> list:
    """probe.fresh: the draws of ``parts`` that are also draws of an earlier probe's draws.json batches."""
    seen = {(q["dataset"], int(q["fit_index"])) for b in earlier for q in b["queries"]}
    return sorted({k for batch in draw_keys(parts) for k in batch} & seen)


def replay_problem(parts_all: list, earlier: list) -> dict | None:
    """The replay's first batches must be the equivalence probe's batches, draw for draw -- the same draw, replayed."""
    for k, b in enumerate(earlier):
        want = [(q["dataset"], int(q["fit_index"])) for q in b["queries"]]
        if k >= len(parts_all) or draw_keys([parts_all[k]])[0] != want:
            return {"batch": k, "equivalence_probe": want[:4], "replayed": draw_keys([parts_all[k]])[0][:4] if k < len(parts_all) else None}
    return None


def keep_under_cap(sizes: list, fixed: int, cap: float = BUNDLE_CAP_BYTES, least: int = MIN_BATCHES) -> int | None:
    """probe.size_cap: the largest B >= least whose bundle fits (None if not even ``least`` batches fit)."""
    B = len(sizes)
    while fixed + sum(sizes[:B]) > cap and B > least:
        B -= 1
    return B if fixed + sum(sizes[:B]) <= cap else None


def stage_bundle(log=log_utc) -> dict:
    """probe.bundle: batches FIRST..FIRST+15 of the seed-0 stage-2 draw, packed on the laptop by the frozen code
    (open_six, fit kind; pack_parts) -- fit-carve rows only, each carve checked against its declared sha256."""
    if (BUNDLE / "manifest.json").exists():
        log("bundle/manifest.json exists, not rebuilt")
        return json.loads((BUNDLE / "manifest.json").read_text(encoding="utf-8"))
    decl_eq = pins_or_stop("bundle")
    earlier = json.loads((CGE.BUNDLE / "draws.json").read_text(encoding="utf-8"))["batches"]
    if len(earlier) != EQ_BATCHES:
        hard_stop("the equivalence probe", {"batches": len(earlier), "declared": EQ_BATCHES}, "bundle")
    V2 = CGE.v2()
    import universal_v2_six as SIX  # noqa: E402  (imported, never edited)
    t0 = time.time()
    cfg, cfg_m3b, _ = V2.load_configs()
    inputs, _, _, _, _, bank, carves, _ = SIX.open_six(cfg, cfg_m3b, kinds=("fit",))
    fits = carves["fit"]
    carves_seen = {n: fits[n].meta.get("carve_sha256_declared") for n in sorted(fits)}
    if carves_seen != dict(sorted(decl_eq["probe"]["fit_carves_sha256"].items())):
        hard_stop("the fit carves", {"declared": decl_eq["probe"]["fit_carves_sha256"], "caches": carves_seen}, "bundle")
    if inputs["core_sha256"] != CGE.CORE_SHA256 or int(inputs["n_scalars"]) != CGE.N_COLUMNS:
        hard_stop("the frozen contract", {"core_sha256": inputs["core_sha256"], "n_scalars": inputs["n_scalars"]}, "bundle")
    emb = bank.embeddings.detach().to(torch.float32).contiguous()
    bank_sha = hashlib.sha256(np.ascontiguousarray(emb.numpy()).tobytes()).hexdigest()
    if bank.n_rows != CGE.BANK_ROWS or bank.sha256 != CGE.BANK_SHA256 or bank_sha != CGE.BANK_SHA256:
        hard_stop("the relation bank", {"rows": bank.n_rows, "sha256": bank.sha256, "recomputed": bank_sha}, "bundle")
    parts_all = CGE.replay_draw(fits, FIRST + N_BATCHES)
    if CGE.draw_counts(parts_all[:16]) != CGE.DRAW_COUNTS:
        hard_stop("the replayed draw", {"declared_first_16": CGE.DRAW_COUNTS, "replayed": CGE.draw_counts(parts_all[:16])}, "bundle")
    bad = replay_problem(parts_all, earlier)
    if bad:
        hard_stop("the replayed draw is not the equivalence probe's", bad, "bundle")
    parts = parts_all[FIRST:]
    seen = overlap_with(parts, earlier)
    if seen:
        hard_stop("a draw of this probe is a draw of the equivalence probe", {"overlap": seen[:20], "n": len(seen)}, "bundle")
    BUNDLE.mkdir(parents=True, exist_ok=True)
    fields, draws = {}, []
    for k, batch_parts in enumerate(parts):
        batch = CGE.pack_parts(fits, batch_parts, CGE.FAMILIES)
        if not bool(batch.gold.any()):
            hard_stop("a probe batch without gold", {"batch": FIRST + k}, "bundle")
        ptr = batch.qptr.numpy()
        rows, j = [], 0
        for name, idx in batch_parts:
            for i in idx:
                qd = fits[name].query(int(i))
                local = np.flatnonzero(batch.gold[int(ptr[j]):int(ptr[j + 1])].numpy())
                if not np.array_equal(local, np.unique(qd["gold"])):
                    raise SystemExit(f"batch {FIRST + k} query {j}: the packed gold is not the carve's gold")
                rows.append({"dataset": name, "fit_index": int(i), "qrow": int(fits[name].qrow[int(i)]), "gold": qd["gold"].tolist(),
                             "gold_total": int(qd["gold_total"]), "candidates": int(ptr[j + 1] - ptr[j])})
                j += 1
        if j != batch.n_queries:
            raise SystemExit(f"batch {FIRST + k}: {j} draws for {batch.n_queries} packed queries")
        name_k = f"batch_{k:02d}.pt"
        CGE.save_batch(batch, BUNDLE / name_k)
        fields[name_k] = CGE.field_pins(batch)
        draws.append({"batch": k, "stage2_batch": FIRST + k, "qptr": ptr.tolist(), "queries": rows})
        log(f"   batch {k} (stage-2 batch {FIRST + k}): {batch.n_queries} draws, {int(ptr[-1])} candidates, "
            f"{int(batch.edge_index.shape[1])} edges, {(BUNDLE / name_k).stat().st_size / 1e6:.1f} MB")
    torch.save({"embeddings": emb.clone(), "offsets": {str(k): int(v) for k, v in bank.offsets.items()}, "sha256": bank.sha256}, BUNDLE / "bank.pt")
    (BUNDLE / "inputs.json").write_text(json.dumps(jsonable(inputs), indent=1), encoding="utf-8")
    sizes = [(BUNDLE / f"batch_{k:02d}.pt").stat().st_size for k in range(N_BATCHES)]
    fixed = (BUNDLE / "bank.pt").stat().st_size + (BUNDLE / "inputs.json").stat().st_size
    B = keep_under_cap(sizes, fixed)
    if B is None:
        raise SystemExit(f"the bundle of {MIN_BATCHES} batches is over the 1.0 GB cap: probe.size_cap does not cover it; stop for review")
    for k in range(B, N_BATCHES):
        (BUNDLE / f"batch_{k:02d}.pt").unlink()
    (BUNDLE / "draws.json").write_text(json.dumps({"seed": CGE.SEED, "batch_size": CGE.BATCH_SIZE, "dataset_draw": CGE.DATASET_DRAW,
                                                   "first_stage2_batch": FIRST, "batches": draws[:B]}, indent=1), encoding="utf-8")
    names = [f"batch_{k:02d}.pt" for k in range(B)] + ["bank.pt", "inputs.json", "draws.json"]
    files = {n: {"bytes": (BUNDLE / n).stat().st_size, "sha256": sha256_file(BUNDLE / n)} for n in names}
    manifest = {"phase": "GPU_TASK_QUALIFICATION", "stage": "bundle", "utc": utc(), "git_head": git_head(), "batches": B,
                "first_stage2_batch": FIRST, "batches_packed": N_BATCHES, "draw_counts": CGE.draw_counts(parts[:B]),
                "draw_counts_packed": CGE.draw_counts(parts), "overlap_with_equivalence_probe": 0,
                "bytes_total": int(sum(f["bytes"] for f in files.values())), "cap_bytes": BUNDLE_CAP_BYTES, "files": files,
                "fields": {n: fields[n] for n in names if n in fields}, "bank": {"rows": bank.n_rows, "sha256": bank.sha256},
                "core_sha256": inputs["core_sha256"], "fit_carves_sha256": carves_seen, "placement": CGE.placement_block(None),
                "seconds": round(time.time() - t0, 1)}
    (BUNDLE / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    log(f"bundle: {B} batches (stage-2 {FIRST}-{FIRST + B - 1}), {manifest['bytes_total'] / 1e9:.3f} GB, draws {manifest['draw_counts']}, "
        f"{manifest['seconds']}s")
    return manifest


# ── an arm ───────────────────────────────────────────────────────────────────


def load_manifest() -> dict:
    p = BUNDLE / "manifest.json"
    if not p.exists():
        raise SystemExit("no bundle/manifest.json: the bundle precedes every arm (execution.order)")
    return json.loads(p.read_text(encoding="utf-8"))


def bundle_problems(manifest: dict) -> tuple[dict, list]:
    """Every bundle file, then every reconstructed field, against the manifest."""
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
        batch = CGE.read_batch(BUNDLE / name)
        got = CGE.field_pins(batch)
        for f in CGE.FIELDS:
            if got[f] != manifest["fields"][name][f]:
                bad[f"{name}:{f}"] = got[f]
        batches.append(batch)
    return bad, batches


def load_bank() -> tuple[torch.Tensor, str]:
    d = torch.load(BUNDLE / "bank.pt", map_location="cpu", weights_only=True)
    emb = d["embeddings"]
    return emb, hashlib.sha256(np.ascontiguousarray(emb.numpy()).tobytes()).hexdigest()


def apply_settings(spec: dict) -> dict:
    """The equivalence file's placement_settings for the arm's mode; host_gpu_tf32 then turns TF32 on."""
    CGE.placement_settings(spec["mode"], spec["threads"])
    if spec["tf32"]:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    return CGE.placement_block(spec["device"])


def environment_problems(spec: dict, place: dict) -> dict:
    """The equivalence file's environment check; for the TF32 arm the three TF32 settings must read back ON instead."""
    bad = CGE.environment_problems(spec, place)
    if spec["tf32"]:
        for k in ("cuda_matmul_allow_tf32", "cudnn_allow_tf32", "float32_matmul_precision"):
            bad.pop(k, None)
        for k, want in (("cuda_matmul_allow_tf32", True), ("cudnn_allow_tf32", True), ("float32_matmul_precision", "high")):
            if place.get(k) != want:
                bad[k] = {"declared": want, "found": place.get(k)}
    return bad


def autocast_on() -> dict:
    out = {}
    for dev in ("cpu", "cuda"):
        try:
            out[dev] = bool(torch.is_autocast_enabled(dev))
        except TypeError:
            out[dev] = bool(torch.is_autocast_enabled()) if dev == "cuda" else bool(torch.is_autocast_cpu_enabled())
    return out


def inventory(device: str) -> dict:
    """tiers.inventory: the build, the libraries and the settings the arm computed with."""
    inv = {"torch": torch.__version__, "numpy": np.__version__, "cuda_build": torch.version.cuda, "mkl": torch.backends.mkl.is_available(),
           "mkldnn": torch.backends.mkldnn.is_available(), "openmp": torch.backends.openmp.is_available(), "autocast": autocast_on(),
           "torch_config": torch.__config__.show(), "parallel_info": torch.__config__.parallel_info()}
    try:
        import torch_geometric
        inv["torch_geometric"] = torch_geometric.__version__
    except ImportError:
        inv["torch_geometric"] = None
    if device == "cuda":
        inv.update({"cudnn": torch.backends.cudnn.version(), "device_name": torch.cuda.get_device_name(0),
                    "capability": list(torch.cuda.get_device_capability(0)), "driver": CGE.placement_block("cuda").get("driver")})
    return inv


def state_sha256(model: torch.nn.Module) -> str:
    """sha256 over the state_dict's names and each tensor's dtype, shape and bytes (tensor_sha256), in name order."""
    pins = {k: tensor_sha256(v) for k, v in model.state_dict().items()}
    return hashlib.sha256(json.dumps(pins, sort_keys=True).encode()).hexdigest()


def tier_train_repeat(decl_eq: dict, inputs: dict, bank_emb: torch.Tensor, batches: list, placed: list, device, where: str) -> tuple[dict, dict]:
    """T5: seed 0 of each family, model.train() (dropout on), torch.manual_seed(0), a fresh AdamW at the frozen lr,
    weight decay and clip, the probe batches as consecutive steps; the loss at every step, the final state_dict's
    sha256 and the final eval-mode centred scores on every probe draw. The weights are discarded."""
    keys = [s["key"] for s in CGE.model_specs(decl_eq) if s["seed"] == 0]
    models = CGE.load_models(decl_eq, inputs, bank_emb, where, keys=keys)
    arrays, rows = {}, {}
    for key, model in models.items():
        if CGE.on_device(device):
            CGE.model_to(model, device)
        torch.manual_seed(0)
        model.train()
        optimiser = torch.optim.AdamW(model.parameters(), lr=CGE.LR, weight_decay=CGE.WEIGHT_DECAY)
        losses = []
        for on in placed:
            optimiser.zero_grad(set_to_none=True)
            loss = listwise_loss(model(on), on)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CGE.CLIP)
            optimiser.step()
            losses.append(loss.detach().to("cpu").reshape(()))
        model.eval()
        with torch.no_grad():
            arrays[f"centred__{key}"] = torch.cat([CGE.centre(model(o).detach().to("cpu"), b) for b, o in zip(batches, placed)]).numpy()
        arrays[f"loss__{key}"] = torch.stack(losses).numpy()
        rows[key] = {"steps": len(placed), "state_sha256": state_sha256(model), "losses": [float(x) for x in losses]}
    return arrays, rows


def stage_arm(arm: str, log=log_utc) -> dict:
    """One arm in this process: settings first (before any CUDA work), then the environment, the inventory, the
    integrity checks and the arm's tiers. Any failed pin is a hard stop, never a verdict."""
    spec = ARMS[arm]
    out = ARMS_DIR / arm
    if (out / "done.json").exists():
        log(f"{arm}: done.json exists, not rerun")
        return json.loads((out / "done.json").read_text(encoding="utf-8"))
    t_start, u_start = time.time(), utc()
    device = spec["device"]
    if device == "cuda" and not torch.cuda.is_available():
        hard_stop("the arm's device", {"arm": arm, "cuda_available": False, "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES")}, arm)
    place = apply_settings(spec)
    bad = environment_problems(spec, place)
    if bad:
        hard_stop("the arm's environment", {"arm": arm, "problems": bad, "placement": place}, arm)
    inv = inventory(device)
    if any(inv["autocast"].values()):
        hard_stop("autocast is active", {"arm": arm, "autocast": inv["autocast"]}, arm)
    deviations = []
    if device == "cuda" and place.get("driver") != CGE.GPU_DRIVER:
        deviations.append(f"driver {place.get('driver')}, the declaration names {CGE.GPU_DRIVER}")
    manifest = load_manifest()
    decl_eq = pins_or_stop(arm)
    bad = CGE.checkpoint_problems(decl_eq)
    if bad:
        hard_stop("the checkpoints", bad, arm)
    bad, batches = bundle_problems(manifest)
    if bad:
        hard_stop("the bundle", bad, arm)
    bank_emb, bank_sha = load_bank()
    if bank_sha != CGE.BANK_SHA256 or int(bank_emb.shape[0]) != CGE.BANK_ROWS:
        hard_stop("the relation bank", {"rows": int(bank_emb.shape[0]), "sha256": bank_sha}, arm)
    inputs = CGE.frozen_inputs(arm)
    shipped = json.loads((BUNDLE / "inputs.json").read_text(encoding="utf-8"))
    if jsonable(inputs) != shipped:
        hard_stop("the model inputs", {"differ": sorted(k for k in shipped if jsonable(inputs).get(k) != shipped[k])}, arm)
    draws = json.loads((BUNDLE / "draws.json").read_text(encoding="utf-8"))["batches"]
    models = CGE.load_models(decl_eq, inputs, bank_emb, arm)
    out.mkdir(parents=True, exist_ok=True)
    log(f"{arm}: integrity holds ({len(batches)} batches, {len(models)} models); device {device}, {spec['threads']} threads, "
        f"mode {spec['mode']}, tf32 {spec['tf32']}")
    seconds, t5_rows, t1_clock = {}, None, {}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        placed = [CGE.batch_to(b, device) for b in batches] if CGE.on_device(device) else batches
        if CGE.on_device(device):
            for m in models.values():
                CGE.model_to(m, device)
        t = time.time()
        t1, t1_clock = CGE.tier_forward(models, batches, placed, draws, device)
        np.savez(out / "t1.npz", **t1)
        seconds["T1"] = round(time.time() - t, 1)
        log(f"   T1 {seconds['T1']}s")
        if "T5" in spec["tiers"]:
            del models
            t = time.time()
            t5, t5_rows = tier_train_repeat(decl_eq, inputs, bank_emb, batches, placed, device, arm)
            np.savez(out / "t5.npz", **t5)
            seconds["T5"] = round(time.time() - t, 1)
            log(f"   T5 {seconds['T5']}s")
    done = {"arm": arm, "spec": {k: list(v) if isinstance(v, tuple) else v for k, v in spec.items()}, "utc_start": u_start, "utc_end": utc(),
            "seconds_total": round(time.time() - t_start, 1), "seconds": seconds, "placement": place, "inventory": inv,
            "deviations": deviations, "warnings": CGE.warning_summary(caught), "t5": t5_rows,
            "t4_forward_ms_median": {k: float(np.median(v)) for k, v in t1_clock.items()},
            "bundle_manifest_sha256": sha256_file(BUNDLE / "manifest.json"), "batches": len(batches), "script_sha256": lf_sha256(Path(__file__)),
            "eq_script_sha256": lf_sha256(EQ_SCRIPT), "git_head": git_head(), "argv": sys.argv[1:], "rx_job_id": os.environ.get("RX_JOB_ID")}
    (out / "done.json").write_text(json.dumps(done, indent=1, default=str), encoding="utf-8")
    log(f"{arm}: done in {done['seconds_total']}s; {sum(w['count'] for w in done['warnings'])} warnings recorded")
    return done


def stage_arms(arms: list, log=log_utc) -> None:
    """Several arms, one fresh process each, in the order given; a failed arm stops the rest."""
    unknown = [a for a in arms if a not in ARMS]
    if unknown:
        raise SystemExit(f"unknown arms {unknown}; one of {sorted(ARMS)}")
    for arm in arms:
        log(f"== {arm} (fresh process)")
        rc = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--stage", "arm", "--arm", arm], cwd=ROOT).returncode
        if rc != 0:
            raise SystemExit(f"{arm}: exited {rc}; the later arms were not started")


# ── the criteria ─────────────────────────────────────────────────────────────


def order_of(raw: np.ndarray) -> np.ndarray:
    """criteria.order: rank_metrics' order -- descending score, ties broken by pool position."""
    return np.argsort(-np.asarray(raw), kind="stable")


def top_k(raw: np.ndarray, k: int) -> frozenset:
    """The first k candidates of criteria.order (the whole pool when it has k or fewer)."""
    return frozenset(int(i) for i in order_of(raw)[:k])


def near_tie_pair(a: float, b: float) -> bool:
    """Two reference centred scores within twice the forward tolerance at their size."""
    return bool(a - b < 2 * (ATOL + RTOL * max(abs(a), abs(b))))


def boundary(raw_x: np.ndarray, raw_r: np.ndarray, cen_r: np.ndarray, k: int) -> tuple[bool, bool]:
    """criteria.near_tie_swap at k: (the top-k sets differ, the difference is excused). Excused when every candidate
    that left the reference's top-k and every one that entered it are near-tied in the reference's centred scores."""
    X, R = top_k(raw_x, k), top_k(raw_r, k)
    if X == R:
        return False, False
    return True, all(near_tie_pair(float(cen_r[i]), float(cen_r[j])) for i in R - X for j in X - R)


def draw_reading(raw_x, raw_r, cen_r, m_x, m_r) -> dict:
    """One (model, draw) under Q_RANK and Q_METRIC: arm x against the reference r."""
    d1, e1 = boundary(raw_x, raw_r, cen_r, 1)
    d5, e5 = boundary(raw_x, raw_r, cen_r, 5)
    rank_diff = bool((d1 and not e1) or (d5 and not e5))
    excused = bool((d1 or d5) and not rank_diff)
    metric_diff = bool(any(not (m_x[i] == m_r[i] or (np.isnan(m_x[i]) and np.isnan(m_r[i]))) for i in TASK_IDX))
    return {"top1_differs": bool(d1), "top5_differs": bool(d5), "rank_difference": rank_diff, "excused": excused,
            "metric_differs": metric_diff, "metric_violation": bool(metric_diff and not excused)}


def compare(t1x: dict, t1r: dict, keys: list, ptr: np.ndarray, datasets: list) -> dict:
    """Arm x against reference r over every model and draw: the score cells (forward_inherited, the equivalence file's
    forward_cells), Q_RANK and Q_METRIC, and per model and dataset the difference of the task metrics' means."""
    out = {"models": {}, "score_fail": set(), "rank_fail": set(), "excused": set(), "bit": True}
    for key in keys:
        cx, cr = np.asarray(t1x[f"centred__{key}"]), np.asarray(t1r[f"centred__{key}"])
        rx, rr = np.asarray(t1x[f"raw__{key}"]), np.asarray(t1r[f"raw__{key}"])
        mx, mr = np.asarray(t1x[f"metrics__{key}"]), np.asarray(t1r[f"metrics__{key}"])
        cells = CGE.forward_cells(cx, cr, ptr)
        out["bit"] = out["bit"] and bool(np.array_equal(rx, rr))
        rows = []
        for q in range(ptr.size - 1):
            s, e = ptr[q], ptr[q + 1]
            r = draw_reading(rx[s:e], rr[s:e], cr[s:e], mx[q], mr[q])
            r["score_ok"] = bool(cells["ok"][q])
            r["ratio"] = float(cells["ratio"][q])
            rows.append(r)
            if not r["score_ok"]:
                out["score_fail"].add((key, q))
            if r["rank_difference"] or r["metric_violation"]:
                out["rank_fail"].add((key, q))
            elif r["excused"]:
                out["excused"].add((key, q))
        deltas = {}
        for ds in sorted(set(datasets)) + ["all"]:
            sel = np.asarray([ds == "all" or d == ds for d in datasets])
            deltas[ds] = {m: float(np.mean(mx[sel, i]) - np.mean(mr[sel, i])) for m, i in zip(TASK_METRICS, TASK_IDX)}
            deltas[ds]["draws_differing"] = int(sum(rows[q]["metric_differs"] for q in np.flatnonzero(sel)))
            deltas[ds]["draws"] = int(sel.sum())
        failing = [{"draw": q, "dataset": datasets[q], "ratio": rows[q]["ratio"], "top1_kept": not rows[q]["top1_differs"],
                    "top5_kept": not rows[q]["top5_differs"]} for q in range(len(rows)) if not rows[q]["score_ok"]]
        out["models"][key] = {"draws": len(rows), "score_cells_failing": failing, "max_ratio": float(np.max(cells["ratio"])),
                              "max_abs_centred_difference": float(np.max(cells["max_abs"])),
                              "top1_differs": int(sum(r["top1_differs"] for r in rows)), "top5_differs": int(sum(r["top5_differs"] for r in rows)),
                              "rank_differences": [q for q, r in enumerate(rows) if r["rank_difference"]],
                              "metric_violations": [q for q, r in enumerate(rows) if r["metric_violation"]],
                              "excused": [q for q, r in enumerate(rows) if r["excused"]],
                              "metric_deltas": deltas, "raw_bit_identical": bool(np.array_equal(rx, rr))}
    return out


def evaluation_verdict(q_det: bool, score_ok: bool, rank_fail: set, floor_rank_fail: set) -> str:
    """verdicts.evaluation: the first that holds."""
    if not q_det:
        return "NOT_REPRODUCIBLE"
    if score_ok:
        if rank_fail:
            raise SystemExit("every score cell holds but a ranking differs away from a near tie: impossible by construction -- a bug; stop")
        return "SCORE_EQUIVALENT"
    if not rank_fail:
        return "TASK_EQUIVALENT"
    if rank_fail - floor_rank_fail:
        return "RANKING_DIFFERS"
    return "INCONCLUSIVE_FLOOR"


def training_verdict(t5a: dict, t5b: dict, rows_a: dict, rows_b: dict) -> tuple[str, dict]:
    """verdicts.training: Q_TRAIN_DET -- every step loss, the final state_dict sha256 and the final centred scores equal."""
    detail = {}
    for key in rows_a:
        detail[key] = {"losses_equal": bool(np.array_equal(t5a[f"loss__{key}"], t5b[f"loss__{key}"])),
                       "state_sha256_equal": rows_a[key]["state_sha256"] == rows_b[key]["state_sha256"],
                       "final_scores_equal": bool(np.array_equal(t5a[f"centred__{key}"], t5b[f"centred__{key}"]))}
    ok = bool(detail) and set(rows_a) == set(rows_b) and all(all(v.values()) for v in detail.values())
    return ("TRAINING_REPRODUCIBLE" if ok else "TRAINING_NOT_REPRODUCIBLE"), detail


# ── the read ─────────────────────────────────────────────────────────────────


def load_t1(arm: str) -> dict:
    with np.load(ARMS_DIR / arm / "t1.npz") as z:
        return {k: z[k] for k in z.files}


def load_t5(arm: str) -> dict:
    with np.load(ARMS_DIR / arm / "t5.npz") as z:
        return {k: z[k] for k in z.files}


def summary(c: dict) -> dict:
    return {"score_cells_failing": len(c["score_fail"]), "rank_or_metric_differences": len(c["rank_fail"]), "excused_near_tie_draws": len(c["excused"]),
            "raw_bit_identical": c["bit"], "models": c["models"]}


def stage_read(log=log_utc) -> dict:
    """criteria.floor_first, then the candidate and its repeat, then the beside and diagnosis arms."""
    manifest = load_manifest()
    msha = sha256_file(BUNDLE / "manifest.json")
    missing = [a for a in ARMS if not (ARMS_DIR / a / "done.json").exists()]
    if missing:
        raise SystemExit(f"arms without done.json: {missing}")
    done = {a: json.loads((ARMS_DIR / a / "done.json").read_text(encoding="utf-8")) for a in ARMS}
    for a, d in done.items():
        if d["bundle_manifest_sha256"] != msha:
            raise SystemExit(f"{a} ran on another bundle")
        if d["utc_start"] < manifest["utc"]:
            raise SystemExit(f"{a} started before the manifest was written")
        if d["eq_script_sha256"] != EQ_SCRIPT_SHA256:
            raise SystemExit(f"{a} ran with other equivalence-script bytes")
    draws = json.loads((BUNDLE / "draws.json").read_text(encoding="utf-8"))["batches"]
    ptr = CGE.draw_pointer(draws)
    datasets = [q["dataset"] for b in draws for q in b["queries"]]
    keys = [s["key"] for s in CGE.model_specs(CGE.load_declaration())]
    t1 = {a: load_t1(a) for a in ARMS}
    ref = t1[REFERENCE]
    floor = compare(t1[FLOOR], ref, keys, ptr, datasets)
    log(f"floor {FLOOR}: {len(floor['score_fail'])} score cells failing, {len(floor['rank_fail'])} rank/metric differences, "
        f"{len(floor['excused'])} excused")
    runs = {r: compare(t1[r], ref, keys, ptr, datasets) for r in (CANDIDATE, REPEAT)}
    q_det = all(np.array_equal(t1[CANDIDATE][f"raw__{k}"], t1[REPEAT][f"raw__{k}"]) for k in keys)
    score_ok = all(not runs[r]["score_fail"] for r in runs)
    rank_fail = runs[CANDIDATE]["rank_fail"] | runs[REPEAT]["rank_fail"]
    verdict_eval = evaluation_verdict(q_det, score_ok, rank_fail, floor["rank_fail"])
    verdict_train, train_detail = training_verdict(load_t5(CANDIDATE), load_t5(REPEAT), done[CANDIDATE]["t5"], done[REPEAT]["t5"])
    log(f"{CANDIDATE}: evaluation {verdict_eval}, training {verdict_train}")
    beside = {SAME_HOST_CPU: compare(t1[SAME_HOST_CPU], ref, keys, ptr, datasets),
              f"{CANDIDATE}_vs_{SAME_HOST_CPU}": compare(t1[CANDIDATE], t1[SAME_HOST_CPU], keys, ptr, datasets),
              TF32_ARM: compare(t1[TF32_ARM], ref, keys, ptr, datasets)}
    inventory_keys = ("torch", "numpy", "cuda_build", "mkl", "mkldnn", "openmp", "torch_geometric", "cudnn", "device_name", "capability", "driver")
    rec = {"phase": "GPU_TASK_QUALIFICATION", "utc": utc(), "git_head": git_head(), "verdict_evaluation": verdict_eval, "verdict_training": verdict_train,
           "q_det": q_det, "score_cells_hold_on_both_runs": score_ok, "rank_or_metric_differences": sorted([list(x) for x in rank_fail]),
           "floor": summary(floor), "candidate": {r: summary(c) for r, c in runs.items()}, "training": train_detail,
           "beside": {k: summary(c) for k, c in beside.items()},
           "bundle": {"batches": manifest["batches"], "first_stage2_batch": manifest["first_stage2_batch"], "draw_counts": manifest["draw_counts"],
                      "bytes_total": manifest["bytes_total"], "utc": manifest["utc"], "manifest_sha256": msha},
           "draws": int(ptr.size - 1),
           "arms": {a: {"placement": d["placement"], "utc_start": d["utc_start"], "utc_end": d["utc_end"], "seconds": d["seconds"],
                        "seconds_total": d["seconds_total"], "warnings": d["warnings"], "deviations": d["deviations"], "rx_job_id": d["rx_job_id"],
                        "git_head": d["git_head"], "t4_forward_ms_median": d["t4_forward_ms_median"],
                        "inventory": {k: d["inventory"].get(k) for k in inventory_keys}, "autocast": d["inventory"]["autocast"]}
                    for a, d in done.items()}}
    RECORD.write_text(json.dumps(jsonable(rec), indent=1, default=str), encoding="utf-8")
    log(f"wrote {RECORD.relative_to(ROOT).as_posix()}")
    return rec


# ── the document ─────────────────────────────────────────────────────────────


def e2(x) -> str:
    return "0" if x == 0 else f"{x:.2e}"


def model_table(s: dict) -> list:
    lines = ["| model | score cells failing | largest ratio | top-1 differs | top-5 set differs | excused (near tie) | ranking or metric differences |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for key, m in s["models"].items():
        lines.append(f"| `{key}` | {len(m['score_cells_failing'])} | {m['max_ratio']:.2f} | {m['top1_differs']} | {m['top5_differs']} | "
                     f"{len(m['excused'])} | {len(m['rank_differences']) + len([q for q in m['metric_violations'] if q not in m['rank_differences']])} |")
    return lines


def delta_table(s: dict) -> list:
    lines = ["| model | draws with a task metric differing | largest |Δ| of a dataset mean (R@5, FullCov@5, Hit@1) |", "|---|---:|---:|"]
    for key, m in s["models"].items():
        d = m["metric_deltas"]
        largest = max(abs(v[k]) for ds, v in d.items() if ds != "all" for k in TASK_METRICS)
        lines.append(f"| `{key}` | {d['all']['draws_differing']} | {e2(largest)} |")
    return lines


def render_doc(rec: dict) -> str:
    cand = rec["candidate"][CANDIDATE]
    a = rec["arms"]
    L = [f"# GPU task-level qualification", "",
         f"Declaration: [`configs/gpu_task_qualification.yaml`](../configs/gpu_task_qualification.yaml). Record: "
         f"`outputs/gpu_task_qualification/record.json`. Read {rec['utc']} at `{(rec['git_head'] or '')[:7]}`.", "",
         "## Verdicts", "",
         f"- **host_gpu_det, evaluation: {rec['verdict_evaluation']}.**",
         f"- **host_gpu_det, training: {rec['verdict_training']}.**", "",
         "The question: on fresh probe draws, does the host GPU in the tested determinism mode make the laptop CPU's retrieval "
         "decisions for the frozen universal-v2 models, and make them reproducibly? Metric levels on these fit-carve queries are "
         "never reported; only differences between placements are.", "",
         "## Probe", "",
         f"{rec['bundle']['batches']} batches (stage-2 seed-0 batches {rec['bundle']['first_stage2_batch']}–"
         f"{rec['bundle']['first_stage2_batch'] + rec['bundle']['batches'] - 1}), {rec['draws']} query draws "
         f"({', '.join(f'{k} {v}' for k, v in rec['bundle']['draw_counts'].items())}), {rec['bundle']['bytes_total'] / 1e9:.3f} GB. "
         "None of them is a draw of the equivalence probe.", "",
         "## The candidate against the laptop reference", "",
         f"Q_DET (repeat bit-identical): {rec['q_det']}. Score cells hold on both runs: {rec['score_cells_hold_on_both_runs']}. "
         f"Ranking or metric differences away from a near tie, both runs: {len(rec['rank_or_metric_differences'])}.", ""]
    L += model_table(cand) + [""]
    L += ["Task-metric differences (candidate minus reference; the per-dataset means of recall@5, full_coverage@5 and hit@1):", ""]
    L += delta_table(cand) + [""]
    L += ["## The floor (laptop, 4 threads against 8)", ""] + model_table(rec["floor"]) + [""]
    L += ["## Training reproducibility (T5)", "",
          "| model | step losses equal | final state sha256 equal | final scores equal |", "|---|---|---|---|"]
    for key, d in rec["training"].items():
        L.append(f"| `{key}` | {d['losses_equal']} | {d['state_sha256_equal']} | {d['final_scores_equal']} |")
    L += ["", "## Beside the verdicts (never gating)", ""]
    for name, title in ((SAME_HOST_CPU, "host CPU (8 threads) against the laptop reference"),
                        (f"{CANDIDATE}_vs_{SAME_HOST_CPU}", "host GPU against host CPU, one machine"),
                        (TF32_ARM, "host GPU with TF32 on (diagnosis) against the laptop reference")):
        s = rec["beside"][name]
        L += [f"### {title}", "",
              f"Score cells failing: {s['score_cells_failing']}; ranking or metric differences away from a near tie: "
              f"{s['rank_or_metric_differences']}; excused near-tie draws: {s['excused_near_tie_draws']}; raw bit-identical: {s['raw_bit_identical']}.", ""]
        L += model_table(s) + [""]
    L += ["## Placements and inventory", "", "| arm | host | env | torch | device | TF32 | deterministic | threads | seconds |",
          "|---|---|---|---|---|---|---|---:|---:|"]
    for arm, d in a.items():
        p = d["placement"]
        tf32 = (p.get("cuda_matmul_allow_tf32"), p.get("cudnn_allow_tf32")) if "cuda" in str(p.get("device")) else "–"
        L.append(f"| {arm} | {p['host']} | {p['env']} | {p['torch']} | {p.get('device_name') or p['device']} | {tf32} | "
                 f"{p['deterministic_algorithms']} | {p['threads']} | {d['seconds_total']} |")
    L += ["", "Inventory per arm (versions, libraries, autocast): `record.json`, `arms.<arm>.inventory`. Autocast was off in every arm "
          "(an arm with autocast on is a hard stop).", "",
          "## What this opens", "",
          "Read `configs/gpu_task_qualification.yaml#what_a_verdict_opens` and `#host_native_protocol`. A verdict opens no stage by "
          "itself: a later stage names the placement in a dated authorization block that quotes the tested settings verbatim and "
          "runs mirror_verification. A GPU fit is a new draw, never paired with a laptop seed.", ""]
    return "\n".join(L)


def stage_doc(log=log_utc) -> Path:
    if not RECORD.exists():
        raise SystemExit("no record.json: the read precedes the document")
    DOC.write_text(render_doc(json.loads(RECORD.read_text(encoding="utf-8"))), encoding="utf-8")
    log(f"wrote {DOC.relative_to(ROOT).as_posix()}")
    return DOC


# ── the run record ───────────────────────────────────────────────────────────


def stage_file(date: str, log=log_utc, extra: dict | None = None) -> None:
    """outputs.run_record: run_record_gpu_task_qualification_<date> appended; status DECLARED_NOT_RUN -> RUN."""
    key = f"run_record_gpu_task_qualification_{date}"
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
    run = {"utc": utc(), "status_moves": "DECLARED_NOT_RUN -> RUN", "verdict_host_gpu_det_evaluation": rec["verdict_evaluation"],
           "verdict_host_gpu_det_training": rec["verdict_training"], "q_det": rec["q_det"],
           "score_cells_hold_on_both_runs": rec["score_cells_hold_on_both_runs"],
           "rank_or_metric_differences": len(rec["rank_or_metric_differences"]),
           "floor": {k: rec["floor"][k] for k in ("score_cells_failing", "rank_or_metric_differences", "excused_near_tie_draws")},
           "candidate": {r: {k: s[k] for k in ("score_cells_failing", "rank_or_metric_differences", "excused_near_tie_draws", "raw_bit_identical")}
                         for r, s in rec["candidate"].items()},
           "beside": {r: {k: s[k] for k in ("score_cells_failing", "rank_or_metric_differences", "excused_near_tie_draws", "raw_bit_identical")}
                      for r, s in rec["beside"].items()},
           "probe": rec["bundle"], "draws": rec["draws"],
           "arms": {arm: {"host": d["placement"]["host"], "env": d["placement"]["env"], "torch": d["placement"]["torch"],
                          "device": d["placement"].get("device_name") or d["placement"]["device"], "threads": d["placement"]["threads"],
                          "seconds": d["seconds_total"], "utc_end": d["utc_end"], "rx_job_id": d["rx_job_id"]} for arm, d in rec["arms"].items()},
           "record": "outputs/gpu_task_qualification/record.json", "record_sha256": sha256_file(RECORD),
           "document": DOC.relative_to(ROOT).as_posix(), "document_sha256_lf": lf_sha256(DOC)}
    if extra:
        run.update(extra)
    run["terminal"] = "STOP_FOR_REVIEW"
    block = yaml.safe_dump(jsonable({key: run}), sort_keys=False, width=160, allow_unicode=True)
    new = text.replace("status: DECLARED_NOT_RUN", "status: RUN", 1)
    CONFIG.write_text(new.rstrip("\n") + "\n\n" + block, encoding="utf-8")
    log(f"{key} appended; status RUN; evaluation {rec['verdict_evaluation']}, training {rec['verdict_training']}")


# ── main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage", required=True, choices=("bundle", "arm", "arms", "read", "doc", "file"))
    ap.add_argument("--arm", choices=sorted(ARMS), help="arm: the placement arm this process runs")
    ap.add_argument("--arms", help="arms: comma-separated arms, one fresh process each, in this order")
    ap.add_argument("--date", help="file: the run record's date, e.g. 2026_09_29")
    ap.add_argument("--extra", type=Path, default=None, help="file: a JSON object of fields added to the run record")
    args = ap.parse_args()
    bind_stops()
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
