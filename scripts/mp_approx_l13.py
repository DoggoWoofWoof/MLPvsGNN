"""MP-Approx level 13, track MP-APPROX: the population module of configs/mp_approx_l13.yaml.

This module holds the population rule, the pins, the dev scoring pass and the check. scripts/mp_approx_l13_fit.py holds
the deploy views, the fits, the repeat, the read, the doc and the file stage, and imports this module unchanged. This
file is committed first and is not edited after that commit, and the score jobs never import the fit module
(placement.identical_code). The training rows are level 12's carve sidecars, read in place on the host and pinned.

On the host (placement): the verified mirror stands in for the package, in memory only.

    python scripts/mp_approx_l13.py --host --stage score --shard 0/2              # with 1/2, at once: the dev rows
    python scripts/mp_approx_l13.py --host --stage assemble                       # scores any chunk no shard wrote, assembles
    python scripts/mp_approx_l13.py --host --stage check                          # the dev sidecar's check -> check.json

A systems smoke, which makes no number of the file: --stage score --limit N --out DIR (never under outputs/mp_approx_l13).

Level 12's population module is imported unchanged, and through it the modules of levels 0 and 8 to 11. Level 8's
scoring pass runs on the dev rows with some of its module names rebound inside this process only (the population rule,
PER_HOP, the walk programme, the pins, the declaration and the output paths), and they are restored when the pass
ends. Nothing is written to the files or outputs of levels 8 to 12 or of the pilot.
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

import mp_approx_l12 as P12  # noqa: E402  (level 12's population module, imported unchanged)

L0, L3, L8, L9, L10, L11 = P12.L0, P12.L3, P12.L8, P12.L9, P12.L10, P12.L11

CONFIG = ROOT / "configs" / "mp_approx_l13.yaml"
OUT = ROOT / "outputs" / "mp_approx_l13"
NAME = L8.NAME
DATA = OUT / NAME
SCRIPT_REL = "scripts/mp_approx_l13.py"
LF = L8.LF

PER_HOP = {1: 0, 2: 1000, 3: 1000}
SALT = "mp_approx_l13|"
AVAILABLE_PER_HOP = {1: 0, 2: 1644, 3: 1473}
FAMILIES = P12.FAMILIES
LEVELS = ("level0", "level8", "level9", "level10", "level11", "level12")
L12_CARVES = ("fit", "select", "x1", "x2", "x3")   # inputs.level12_carves: the carve sidecars this file trains and selects on
DEV_SHARDS = 4
HARD_STOP_DIR = [OUT]                    # a smoke run and the tests point it at their own directory

_L12_ROWS = P12.l12_rows                 # level 12's own rule, captured before any rebinding
_L12_VERIFY = P12.verify_inputs


# ── small helpers ────────────────────────────────────────────────────────────


log_utc = L8.log_utc
write_json, read_json, shown = L8.write_json, L8.read_json, L8.shown
is_train_id, hop_counts = P12.is_train_id, P12.hop_counts


def route_stops() -> None:
    """The hard stops of levels 0, 3 and 8 to 12 land beside this file's."""
    P12.HARD_STOP_DIR[0] = HARD_STOP_DIR[0]
    P12.route_stops()


def hard_stop(message: str, **evidence) -> None:
    """hard_stops: the evidence goes to outputs/mp_approx_l13/hard_stops.json; the status line is left alone."""
    route_stops()
    L0.hard_stop(message, **evidence)


def load_declaration() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


# ── pins ─────────────────────────────────────────────────────────────────────


def verify_inputs(decl: dict) -> None:
    """inputs: level 12's check on this file's copy (levels 0 and 8 to 11, their pinned qids.json and the pilot's
    training caches), then level 12's own files and its pinned qids.json, then the meta.json and check.json of each of
    level 12's carve sidecars this file reads (inputs.level12_carves)."""
    route_stops()
    _L12_VERIFY(decl)
    route_stops()
    lv = decl["inputs"]["level12"]
    pins = [(lv[k]["path"], lv[k]["sha256"], True) for k in ("declaration_lf", "script_lf", "tests_lf", "fit_script_lf", "fit_tests_lf")]
    pins.append((lv["excluded_rows"]["qids"]["path"], lv["excluded_rows"]["qids"]["sha256"], False))
    cv = decl["inputs"]["level12_carves"]
    for carve in L12_CARVES:
        pins += [((Path(cv["dir"]) / carve / f).as_posix(), digest, False) for f, digest in cv[carve].items()]
    for rel, digest, lf in pins:
        p = ROOT / rel
        found = (L0.lf_sha256(p) if lf else L0.sha256_file(p)) if p.exists() else "missing"
        if found != digest:
            hard_stop(f"{rel}{' (LF)' if lf else ''} is not its pinned sha256", path=rel, pinned=digest, found=found)


# ── the population rule (the dev rows) ───────────────────────────────────────


def l13_rows(ids: list[str], gate: np.ndarray, excluded: np.ndarray, per_hop: dict = PER_HOP, level8_ids: list[str] | None = None,
             level9_ids: list[str] | None = None, level10_ids: list[str] | None = None, level11_ids: list[str] | None = None,
             level12_ids: list[str] | None = None, level8_per_hop: int = L8.PER_HOP, level9_per_hop: int = L9.PER_HOP,
             level10_per_hop: int = L10.PER_HOP, level11_per_hop: int = L11.PER_HOP,
             level12_per_hop: dict = P12.PER_HOP) -> tuple[np.ndarray, dict]:
    """population.rule: level 12's rows by level 12's own rule (l12_rows, which recomputes levels 8 to 11 and checks
    level 11's against its pinned qids.json), and level 12's must be its pinned qids.json. Levels 8 to 12 leave with
    level 0's, and the rest are sorted per hop by sha256(SALT + id), the first per_hop[h] kept, in population order."""
    excluded = np.asarray(excluded, dtype=np.int64)
    route_stops()
    l12, _available12 = _L12_ROWS(ids, gate, excluded, level12_per_hop, level8_ids=level8_ids, level9_ids=level9_ids,
                                  level10_ids=level10_ids, level11_ids=level11_ids, level8_per_hop=level8_per_hop,
                                  level9_per_hop=level9_per_hop, level10_per_hop=level10_per_hop, level11_per_hop=level11_per_hop)
    route_stops()
    if level12_ids is not None and [ids[i] for i in l12] != list(level12_ids):
        hard_stop("level 12's recomputed rows are not its pinned qids.json")
    l8, _a8 = P12._L8_ROWS(ids, gate, excluded, level8_per_hop)
    l9, _a9 = L9.l9_rows(ids, gate, excluded, level9_per_hop, level8_ids=level8_ids, level8_per_hop=level8_per_hop)
    l10, _a10 = L10.l10_rows(ids, gate, excluded, level10_per_hop, level8_ids=level8_ids, level9_ids=level9_ids,
                             level8_per_hop=level8_per_hop, level9_per_hop=level9_per_hop)
    route_stops()
    l11, _a11 = L11.l11_rows(ids, gate, excluded, level11_per_hop, level8_ids=level8_ids, level9_ids=level9_ids,
                             level10_ids=level10_ids, level8_per_hop=level8_per_hop, level9_per_hop=level9_per_hop,
                             level10_per_hop=level10_per_hop)
    route_stops()
    out = excluded
    for earlier in (l8, l9, l10, l11, l12):
        out = np.union1d(out, earlier)
    rows, available = P12.rows_by_salt(ids, gate, out, SALT, per_hop)
    if np.isin(rows, out).any():
        hard_stop("a level 0 or level 8 to 12 row would be scored")
    return rows, available


def level_ids(decl: dict) -> dict:
    """The pinned qids.json of levels 8 to 12, keyed as l13_rows takes them."""
    inp = decl["inputs"]
    return {f"{lv}_ids": read_json(ROOT / inp[lv]["excluded_rows"]["qids"]["path"]) for lv in LEVELS[1:]}


# ── stage: score / assemble (the dev rows; level 8's pass, names rebound) ────


@contextlib.contextmanager
def level8_rebound(out_dir: Path, rows_rule):
    """scoring_pass.path: level 8's module names that differ here, rebound for the pass and restored after it."""
    names = {"l8_rows": rows_rule, "walk_entries": L9.walk_entries_both, "verify_inputs": verify_inputs, "CONFIG": CONFIG,
             "OUT": OUT, "DATA": out_dir, "AVAILABLE_PER_HOP": dict(AVAILABLE_PER_HOP), "PER_HOP": dict(PER_HOP)}
    saved = {k: getattr(L8, k) for k in names}
    for k, v in names.items():
        setattr(L8, k, v)
    try:
        yield
    finally:
        for k, v in saved.items():
            setattr(L8, k, v)


def stage_score(decl: dict, log=print, shard: tuple[int, int] | None = None, limit: int | None = None,
                out_dir: Path | None = None) -> None:
    """scoring_pass: level 8's stage_score with this file's rule and per_hop, level 9's combined walks, this file's pins
    and paths; this file's verify_inputs runs inside it, before the population is read and again at the end."""
    out_dir = out_dir or DATA
    pinned = level_ids(decl)

    def rows_rule(ids, gate, excluded, per_hop=PER_HOP):
        return l13_rows(ids, gate, excluded, per_hop, **pinned)

    route_stops()
    with level8_rebound(out_dir, rows_rule):
        L8.stage_score(decl, log, shard, limit, out_dir)


# ── stage: check (the dev sidecar) ───────────────────────────────────────────


def stage_check(decl: dict, log=print) -> dict:
    """check: level 9's check_family on both views of the dev sidecar and level 12's anchors, before any fit; any level 0
    or level 8 to 12 query id, any held row and any train-split id in the sidecar is refused (the fit module's deploy
    view checks the hops against population.per_hop)."""
    t0 = time.time()
    torch.set_num_threads(L8.FIT_THREADS)
    verify_inputs(decl)
    vs = L9.views(DATA)
    base = vs["std"]
    meta = base.meta
    if meta["limit"] is not None:
        raise SystemExit("the sidecar is a smoke run")
    inp = decl["inputs"]
    earlier = set()
    for level in LEVELS:
        earlier |= set(read_json(ROOT / inp[level]["excluded_rows"]["qids"]["path"]))
    if earlier & set(base.qids):
        hard_stop("a level 0 or level 8 to 12 query id is in the sidecar")
    train = [q for q in base.qids if is_train_id(q)]
    if train:
        hard_stop("a train-split query id is in the dev sidecar", queries=train[:10])
    half, _hop, _stored = L0.load_stored(L0.load_declaration(), NAME)
    if not half[base.q_row].all():
        hard_stop("a held row is in the sidecar")
    route_stops()
    chains, qt_of = P12.chains_of(base, meta)
    found = P12.anchors(vs, chains, qt_of)
    out = {"stage": "check", "queries": base.n_q, "qtypes": len(meta["qtypes"]), "hops": hop_counts(base.qids), **found,
           "meta_sha256": L0.sha256_file(DATA / "meta.json"), **L8.job_fields(t0)}
    write_json(DATA / "check.json", out)
    route_stops()
    P12.direction_stops(found["families"])
    verify_inputs(decl)
    fam = found["families"]
    log(f"check: the chain reaches a gold on {fam['std']['direction_check']['declared_share']:.3f} (std) and "
        f"{fam['nb']['direction_check']['declared_share']:.3f} (nb) of the dev rows; NB removes "
        f"{out['nb_trim']['r_star_share_removed']['all']:.3f} of R* and {out['nb_trim']['gold_share_removed']['all']:.4f} of the golds; "
        f"{out['gold_unreached_nb']['all']:.4f} of the in-pool golds lie in no nb reach set")
    return out


# ── main ─────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=("score", "assemble", "check"))
    ap.add_argument("--host", action="store_true", help="on the host: the verified mirror in place of the package, in memory")
    ap.add_argument("--shard", default=None, help="score: i/n, the chunks ci with ci mod n = i; a run without it assembles")
    ap.add_argument("--limit", type=int, default=None, help="score, smoke only: N queries spread over the rows, written to --out")
    ap.add_argument("--out", type=Path, default=None, help="score, smoke only: a directory outside outputs/mp_approx_l13")
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
            ap.error("a smoke run never writes under outputs/mp_approx_l13")
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
