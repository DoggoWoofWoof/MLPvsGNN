"""MP-Approx level 14, track MP-APPROX: the population module of configs/mp_approx_l14.yaml.

This module holds the read carve r's rule, the pins, r's scoring pass and the check. scripts/mp_approx_l14_fit.py holds
the deploy views (full and b0), the fits, the repeat, the read, the doc and the file stage, and imports this module
unchanged. This file is committed first and is not edited after that commit, and the score, assemble and check jobs
never import the fit module (placement.identical_code). The training rows are level 12's carve sidecars, read in place
on the host and pinned.

On the host (placement): the verified mirror stands in for the package, in memory only.

    python scripts/mp_approx_l14.py --host --stage score --shard 0/12             # with 1/12 to 11/12, at once: r's rows
    python scripts/mp_approx_l14.py --host --stage assemble                       # scores any chunk no shard wrote, assembles
    python scripts/mp_approx_l14.py --host --stage check                          # r's sidecar's check -> check.json

A systems smoke, which makes no number of the file: --stage score --limit N --out DIR (never under outputs/mp_approx_l14).

Level 13's population module is imported unchanged, and through it the modules of levels 0 and 8 to 12. Level 12's
carve pass runs on r with four of its module names rebound inside this process only (CARVES, carve_ids_of,
verify_inputs and CONFIG), and they are restored when the pass ends. Nothing is written to the files or outputs of
levels 8 to 13 or of the pilot.
"""

from __future__ import annotations

import os
import sys

if __name__ == "__main__":   # placement.threads: the pools are fixed before numpy and torch load
    _THREADS = "6" if any(s in sys.argv for s in ("score", "assemble")) else "4"
    for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[_var] = _THREADS

import argparse  # noqa: E402
import contextlib  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import yaml  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "src", ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import torch  # noqa: E402

import mp_approx_l13 as P13  # noqa: E402  (level 13's population module, imported unchanged)
from mp_retrieval import m3b_pools  # noqa: E402

P12 = P13.P12
L0, L3, L8, L9, L10, L11 = P12.L0, P12.L3, P12.L8, P12.L9, P12.L10, P12.L11

CONFIG = ROOT / "configs" / "mp_approx_l14.yaml"
OUT = ROOT / "outputs" / "mp_approx_l14"
NAME = L8.NAME
DATA = OUT / NAME
SCRIPT_REL = "scripts/mp_approx_l14.py"
LF = L8.LF

READ_CARVE = "r"
R_OFFSETS = (8, 9)                       # population.offsets: r is the fit stride's offsets 8 and 9
FAMILIES = P12.FAMILIES
LEVELS = ("level0", "level8", "level9", "level10", "level11", "level12", "level13")
L12_CARVES = P13.L12_CARVES              # inputs.level12_carves: the carve sidecars this file trains and selects on
L12_OTHER_CARVES = ("x4", "x5", "x6", "x7")   # inputs.level12_carves_x4_x7: read for r's disjointness check only
SHARDS = 12
HARD_STOP_DIR = [OUT]                    # a smoke run and the tests point it at their own directory

_L13_VERIFY = P13.verify_inputs
_L12_CARVES = P12.CARVES                 # level 12's nine carves and its own carve_ids_of, captured before any rebinding
_L12_CARVE_IDS = P12.carve_ids_of
CARVES = (*_L12_CARVES, READ_CARVE)      # every carve the rebound population check recomputes and keeps disjoint


# ── small helpers ────────────────────────────────────────────────────────────


log_utc = L8.log_utc
write_json, read_json, shown = L8.write_json, L8.read_json, L8.shown
is_train_id, hop_counts = P12.is_train_id, P12.hop_counts


def route_stops() -> None:
    """The hard stops of levels 0, 3 and 8 to 13 land beside this file's."""
    P13.HARD_STOP_DIR[0] = HARD_STOP_DIR[0]
    P13.route_stops()


def hard_stop(message: str, **evidence) -> None:
    """hard_stops: the evidence goes to outputs/mp_approx_l14/hard_stops.json; the status line is left alone."""
    route_stops()
    L0.hard_stop(message, **evidence)


def load_declaration() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


# ── pins ─────────────────────────────────────────────────────────────────────


def verify_inputs(decl: dict) -> None:
    """inputs: level 13's check on this file's copy (levels 0 and 8 to 12, their pinned qids.json, the pilot's training
    caches and five of level 12's carve sidecars), then level 13's own files and its pinned qids.json, the meta.json and
    check.json of level 12's other four carve sidecars, level 12's loopcheck, which must be on file as equal, and the two
    earlier read.json files."""
    route_stops()
    _L13_VERIFY(decl)
    route_stops()
    inp = decl["inputs"]
    lv = inp["level13"]
    pins = [(lv[k]["path"], lv[k]["sha256"], True) for k in ("declaration_lf", "script_lf", "tests_lf", "fit_script_lf", "fit_tests_lf")]
    pins.append((lv["excluded_rows"]["qids"]["path"], lv["excluded_rows"]["qids"]["sha256"], False))
    cv = inp["level12_carves_x4_x7"]
    for carve in L12_OTHER_CARVES:
        pins += [((Path(cv["dir"]) / carve / f).as_posix(), digest, False) for f, digest in cv[carve].items()]
    pins.append((inp["level12_loopcheck"]["path"], inp["level12_loopcheck"]["sha256"], False))
    pins += [(e["path"], e["sha256"], False) for e in inp["earlier_reads"].values()]
    for rel, digest, lf in pins:
        p = ROOT / rel
        found = (L0.lf_sha256(p) if lf else L0.sha256_file(p)) if p.exists() else "missing"
        if found != digest:
            hard_stop(f"{rel}{' (LF)' if lf else ''} is not its pinned sha256", path=rel, pinned=digest, found=found)
    loop = read_json(ROOT / inp["level12_loopcheck"]["path"])
    if loop.get("equal") is not True or loop.get("differences"):
        hard_stop("level 12's loopcheck is not on file as equal", equal=loop.get("equal"), differences=loop.get("differences"))


# ── the population rule (r) ──────────────────────────────────────────────────


def r_ids_of(rule: dict) -> list[str]:
    """population.rule: remaining[i] for every position i with i mod s_fit in R_OFFSETS, in position order."""
    s_fit = rule["s_fit"]
    if not all(0 <= j < s_fit for j in R_OFFSETS):
        raise ValueError(f"r: the fit stride {s_fit} has no offset {max(R_OFFSETS)}")
    return [q for i, q in enumerate(rule["remaining"]) if i % s_fit in R_OFFSETS]


def carve_ids_of(carve: str, rule: dict) -> list[str]:
    """carves.rule: r by population.rule, and every other carve by level 12's own carve_ids_of."""
    if carve == READ_CARVE:
        return r_ids_of(rule)
    return _L12_CARVE_IDS(carve, rule)


# ── stage: score / assemble (r; level 12's carve pass, names rebound) ────────


@contextlib.contextmanager
def level12_rebound():
    """scoring_pass.path: the four names of level 12's module that differ here, rebound for the pass and restored after it."""
    names = {"CARVES": CARVES, "carve_ids_of": carve_ids_of, "verify_inputs": verify_inputs, "CONFIG": CONFIG}
    saved = {k: getattr(P12, k) for k in names}
    for k, v in names.items():
        setattr(P12, k, v)
    try:
        yield
    finally:
        for k, v in saved.items():
            setattr(P12, k, v)


def stage_score(decl: dict, log=print, shard: tuple[int, int] | None = None, limit: int | None = None,
                out_dir: Path | None = None) -> None:
    """scoring_pass: level 12's stage_carve on r, with this file's carves, rule, pins check and declaration; this file's
    verify_inputs runs inside it, before the population is read and again at the end. With a shard the process scores
    only its chunks; a run without one scores any chunk not written and assembles."""
    route_stops()
    with level12_rebound():
        P12.stage_carve(decl, READ_CARVE, log, shard, limit, out_dir or DATA)


# ── stage: check (r's sidecar) ───────────────────────────────────────────────


def stage_check(decl: dict, log=print) -> dict:
    """check: level 9's check_family on both views of r's sidecar (the direction check is a hard stop) and level 12's
    anchors, before any fit. A smoke, a sidecar that is not r's, a non-train id, ids other than r's pin, a q_row out of
    order, any query id of levels 0 and 8 to 13 and any id of level 12's nine carve sidecars (each must be assembled)
    are refused; verify_inputs refuses a loopcheck not on file as equal."""
    t0 = time.time()
    torch.set_num_threads(L8.FIT_THREADS)
    verify_inputs(decl)
    vs = L9.views(DATA)
    base = vs["std"]
    meta = base.meta
    if meta["limit"] is not None:
        raise SystemExit("r's sidecar is a smoke run")
    pin = decl["carves"]["pins"][READ_CARVE]
    if meta.get("carve") != READ_CARVE or meta.get("carve_ids_sha256") != pin["ids_sha256"]:
        hard_stop(f"check: the sidecar at {shown(DATA)} is not r's", carve=meta.get("carve"), digest=meta.get("carve_ids_sha256"))
    ids = list(base.qids)
    bad = [q for q in ids if not is_train_id(q)]
    if bad:
        hard_stop("check: an id of r's sidecar is not a metaqa train-split id", queries=bad[:10])
    got = {"queries": len(ids), "ids_sha256": m3b_pools.ids_digest(ids), "hops": hop_counts(ids)}
    want = {"queries": int(pin["queries"]), "ids_sha256": pin["ids_sha256"], "hops": {int(h): int(v) for h, v in pin["hops"].items()}}
    if got != want:
        hard_stop("check: r's ids are not its pin", found=got, pinned=want)
    if not np.array_equal(base.q_row, np.arange(base.n_q)):
        hard_stop("check: r's q_row is not each query's position in r")
    inp = decl["inputs"]
    earlier = set()
    for level in LEVELS:
        earlier |= set(read_json(ROOT / inp[level]["excluded_rows"]["qids"]["path"]))
    shared = earlier & set(ids)
    if shared:
        hard_stop("check: a query id of level 0 or levels 8 to 13 is in r", queries=sorted(shared)[:10])
    compared = {}
    for carve in _L12_CARVES:
        theirs = P12.assembled_ids(P12.CARVES_DIR / carve)
        if theirs is None:
            hard_stop(f"check: level 12's carve {carve} is not assembled; r's disjointness check reads its ids")
        shared = set(ids) & set(theirs)
        if shared:
            hard_stop(f"check: r and level 12's carve {carve} share ids", queries=sorted(shared)[:10])
        compared[carve] = len(theirs)
    route_stops()
    chains, qt_of = P12.chains_of(base, meta)
    found = P12.anchors(vs, chains, qt_of)
    out = {"stage": "check", "carve": READ_CARVE, "queries": base.n_q, "qtypes": len(meta["qtypes"]), "hops": got["hops"],
           "ids_sha256": got["ids_sha256"], "train_split_ids": base.n_q, "earlier_level_ids_shared": 0,
           "carves_compared": compared, "loopcheck_equal": True, **found,
           "carve_metrics": P12.carve_metrics(base.q_metrics, base.q_hop),
           "carve_metrics_note": "descriptive; r is train-split rows that the twin and the GNN never trained or selected on",
           "meta_sha256": L0.sha256_file(DATA / "meta.json"), **L8.job_fields(t0)}
    write_json(DATA / "check.json", out)
    route_stops()
    P12.direction_stops(found["families"])
    verify_inputs(decl)
    fam = found["families"]
    log(f"check: r holds {base.n_q} queries ({out['rows_with_gold_in_pool']['all']} with an in-pool gold); the chain reaches a "
        f"gold on {fam['std']['direction_check']['declared_share']:.3f} (std) and {fam['nb']['direction_check']['declared_share']:.3f} "
        f"(nb); compared with level 12's carves {sorted(compared)}")
    return out


# ── main ─────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=("score", "assemble", "check"))
    ap.add_argument("--host", action="store_true", help="on the host: the verified mirror in place of the package, in memory")
    ap.add_argument("--shard", default=None, help="score: i/n, the chunks ci with ci mod n = i; a run without it assembles")
    ap.add_argument("--limit", type=int, default=None, help="score, smoke only: N queries spread over r, written to --out")
    ap.add_argument("--out", type=Path, default=None, help="score, smoke only: a directory outside outputs/mp_approx_l14")
    args = ap.parse_args(argv)
    if not args.host:
        ap.error(f"--stage {args.stage} runs on the host (--host), placement")
    if args.shard is not None and args.stage != "score":
        ap.error("--shard is for --stage score")
    if (args.limit is None) != (args.out is None) or (args.limit is not None and args.stage != "score"):
        ap.error("--limit and --out go together, with --stage score (a smoke run)")
    if args.limit is not None and args.shard is not None:
        ap.error("a smoke run has no --shard")
    decl = load_declaration()
    if args.limit is not None:
        out = args.out.resolve()
        if OUT.resolve() in (out, *out.parents):
            ap.error("a smoke run never writes under outputs/mp_approx_l14")
        HARD_STOP_DIR[0] = out
    route_stops()
    L8.host_mode(decl, log_utc)
    shard = L8.parse_shard(args.shard)
    if args.stage in ("score", "assemble"):
        if args.limit is not None:
            stage_score(decl, log_utc, None, args.limit, out / NAME)
        else:
            stage_score(decl, log_utc, shard if args.stage == "score" else None)
    else:
        stage_check(decl, log_utc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
