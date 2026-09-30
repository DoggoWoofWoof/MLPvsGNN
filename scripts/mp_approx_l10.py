"""MP-Approx level 10, track MP-APPROX: level 9's non-message-passing typed-walk model on metaqa, fitted with a set
likelihood and scored by coverage (configs/mp_approx_l10.yaml).

This module holds the population rule, the pins, the scoring pass and the check. scripts/mp_approx_l10_fit.py holds the
fits, the repeat, the read, the doc and the file stage, and imports this module unchanged. This file is committed first
and is not edited after that commit, so the score jobs and the fit jobs run one file (placement.identical_code).

On the host (placement): the verified mirror stands in for the package, in memory only.

    python scripts/mp_approx_l10.py --host --stage score --shard 0/3   # with 1/3 and 2/3, at once: scoring pass + both walk families
    python scripts/mp_approx_l10.py --host --stage assemble            # scores any chunk no shard wrote, then assembles
    python scripts/mp_approx_l10.py --host --stage check               # chain map, direction checks, anchors -> check.json

A systems smoke, which makes no number of the file: --stage score --limit N --out DIR (never under outputs/mp_approx_l10).

Level 8's and level 9's scripts are imported unchanged. Level 8's scoring pass runs with some of its module names rebound
inside this process only (the population rule, the walk programme, the pins, the declaration and the output paths), and
they are restored when the pass ends. Nothing is written to level 8's or level 9's files or outputs.
"""

from __future__ import annotations

import os
import sys

if __name__ == "__main__":   # placement.threads: the pools are fixed before numpy and torch load
    _THREADS = "6" if ("score" in sys.argv or "assemble" in sys.argv) else "4"
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

import mp_approx_l0 as L0  # noqa: E402  (level 0, imported unchanged)
import mp_approx_l3 as L3  # noqa: E402  (level 3, imported unchanged: hard-stop routing only)
import mp_approx_l8 as L8  # noqa: E402  (level 8, imported unchanged)
import mp_approx_l9 as L9  # noqa: E402  (level 9, imported unchanged)

CONFIG = ROOT / "configs" / "mp_approx_l10.yaml"
OUT = ROOT / "outputs" / "mp_approx_l10"
NAME = L8.NAME
DATA = OUT / NAME
SCRIPT_REL = "scripts/mp_approx_l10.py"
LF = L8.LF

PER_HOP = 1000
SALT = "mp_approx_l10|"
AVAILABLE_PER_HOP = {1: 2221, 2: 4644, 3: 4473}
FAMILIES = L9.FAMILIES
HARD_STOP_DIR = [OUT]                    # a smoke run and the tests point it at their own directory

_L8_ROWS = L9._L8_ROWS                   # level 8's own rule, captured by level 9 before any rebinding
_L9_VERIFY = L9.verify_inputs


# ── small helpers ────────────────────────────────────────────────────────────


log_utc = L8.log_utc
write_json, read_json, shown = L8.write_json, L8.read_json, L8.shown


def route_stops() -> None:
    """Level 0's, level 3's, level 8's and level 9's hard stops land beside this file's."""
    L9.HARD_STOP_DIR[0] = L8.HARD_STOP_DIR[0] = L0.HARD_STOP_DIR[0] = L3.HARD_STOP_DIR[0] = HARD_STOP_DIR[0]


def hard_stop(message: str, **evidence) -> None:
    """hard_stops: the evidence goes to outputs/mp_approx_l10/hard_stops.json; the status line is left alone."""
    route_stops()
    L0.hard_stop(message, **evidence)


def load_declaration() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


# ── pins ─────────────────────────────────────────────────────────────────────


def verify_inputs(decl: dict) -> None:
    """inputs: level 9's check on this file's copy (level 8's pins, level 0's verify_pins on metaqa, level 8's own files
    and its pinned qids.json), then level 9's own files and its pinned qids.json."""
    route_stops()
    _L9_VERIFY(decl)
    route_stops()
    lv = decl["inputs"]["level9"]
    pins = [(lv[k]["path"], lv[k]["sha256"], True) for k in ("declaration_lf", "script_lf", "tests_lf")]
    pins.append((lv["excluded_rows"]["qids"]["path"], lv["excluded_rows"]["qids"]["sha256"], False))
    for rel, digest, lf in pins:
        p = ROOT / rel
        found = (L0.lf_sha256(p) if lf else L0.sha256_file(p)) if p.exists() else "missing"
        if found != digest:
            hard_stop(f"{rel}{' (LF)' if lf else ''} is not its pinned sha256", path=rel, pinned=digest, found=found)


# ── the population rule ──────────────────────────────────────────────────────


def l10_rows(ids: list[str], gate: np.ndarray, excluded: np.ndarray, per_hop: int = PER_HOP, level8_ids: list[str] | None = None,
             level9_ids: list[str] | None = None, level8_per_hop: int = L8.PER_HOP,
             level9_per_hop: int = L9.PER_HOP) -> tuple[np.ndarray, dict]:
    """population.rule: level 8's rows by level 8's own rule and salt, from the rows level 8 excluded (level 0's); level
    9's rows by level 9's own rule (l9_rows, which checks level 8's rows against their pinned ids), from the rows level 9
    excluded; level 9's must be its pinned qids.json. All three leave, and the rest are sorted per hop by
    sha256(SALT + id), the first per_hop kept, in population order."""
    excluded = np.asarray(excluded, dtype=np.int64)
    l8, _available8 = _L8_ROWS(ids, gate, excluded, level8_per_hop)
    l9, _available9 = L9.l9_rows(ids, gate, excluded, level9_per_hop, level8_ids=level8_ids, level8_per_hop=level8_per_hop)
    if level9_ids is not None and [ids[i] for i in l9] != list(level9_ids):
        hard_stop("level 9's recomputed rows are not its pinned qids.json")
    out = np.union1d(np.union1d(excluded, l8), l9)
    rows, available = L9.rows_by_salt(ids, gate, out, SALT, per_hop)
    if np.isin(rows, out).any():
        hard_stop("a level 0, level 8 or level 9 row would be scored")
    return rows, available


# ── stage: score / assemble (level 8's pass, names rebound in this process) ──


@contextlib.contextmanager
def level8_rebound(out_dir: Path, rows_rule):
    """scoring_pass.path: level 8's module names that differ here, rebound for the pass and restored after it."""
    names = {"l8_rows": rows_rule, "walk_entries": L9.walk_entries_both, "verify_inputs": verify_inputs, "CONFIG": CONFIG,
             "OUT": OUT, "DATA": out_dir, "AVAILABLE_PER_HOP": dict(AVAILABLE_PER_HOP)}
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
    """scoring_pass: level 8's stage_score with this file's rule, level 9's combined walks, this file's pins and paths;
    this file's verify_inputs runs inside it, before the population is read and again at the end."""
    out_dir = out_dir or DATA
    inp = decl["inputs"]
    level8_ids = read_json(ROOT / inp["level8"]["excluded_rows"]["qids"]["path"])
    level9_ids = read_json(ROOT / inp["level9"]["excluded_rows"]["qids"]["path"])

    def rows_rule(ids, gate, excluded, per_hop=PER_HOP):
        return l10_rows(ids, gate, excluded, per_hop, level8_ids=level8_ids, level9_ids=level9_ids)

    route_stops()
    with level8_rebound(out_dir, rows_rule):
        L8.stage_score(decl, log, shard, limit, out_dir)


# ── stage: check ─────────────────────────────────────────────────────────────


def stage_check(decl: dict, log=print) -> dict:
    """check: level 9's check_family on both views (level 8's direction check and chain anchors, and level 9's), the
    chain map and the anchors that need no fit, before any fit."""
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
    for level in ("level0", "level8", "level9"):
        earlier |= set(read_json(ROOT / inp[level]["excluded_rows"]["qids"]["path"]))
    if earlier & set(base.qids):
        hard_stop("a level 0, level 8 or level 9 query id is in the sidecar")
    half, _hop, _stored = L0.load_stored(L0.load_declaration(), NAME)
    if not half[base.q_row].all():
        hard_stop("a held row is in the sidecar")
    chains = {}
    for qt in meta["qtypes"]:
        try:
            chains[qt] = L8.true_chain(qt)
        except ValueError as err:
            hard_stop(f"qtype {qt} does not parse to a chain", detail=str(err))
    qt_of = [meta["qtypes"][i] for i in base.q_qtype]
    bad = [q for q, qt, h in zip(base.qids, qt_of, base.q_hop) if len(chains[qt]) != int(h)]
    if bad:
        hard_stop("a qtype does not parse to a chain of its hop's length", queries=bad[:10])
    fam, rstar = {}, {}
    for f, v in vs.items():
        fam[f], rstar[f] = L9.check_family(v, chains, qt_of)
    n, hop = base.n_q, base.q_hop
    trim, lost, has_std = np.zeros(n), np.zeros(n), np.zeros(n, dtype=bool)
    unreached = np.zeros(n)
    for q in range(n):
        rs, rn = rstar["std"][q], rstar["nb"][q]
        g = base.gold_local(q)
        gone = np.setdiff1d(rs, rn)
        if rs.size:
            has_std[q] = True
            trim[q] = gone.size / rs.size
        if g.size:
            lost[q] = np.intersect1d(gone, g).size / g.size
            node, _count, _tl = vs["nb"].entries(q)
            unreached[q] = np.setdiff1d(g, node).size / g.size
    te = base.q_te_local
    te_in_seeds = np.asarray([bool(te[q] >= 0 and (base.q_seed_local[q] == te[q]).any()) for q in range(n)])
    te_in_b0 = np.asarray([bool(te[q] >= 0 and ((base.q_seed_local[q] == te[q]) & (base.q_seed_bucket[q] == 0)).any())
                           for q in range(n)])
    has_gold = base.q_gold_in_pool > 0
    out = {"stage": "check", "queries": n, "qtypes": len(meta["qtypes"]),
           "qtypes_per_hop": {f"hop={h}": len({qt_of[q] for q in range(n) if hop[q] == h}) for h in (1, 2, 3)},
           "chain_map": {qt: [[L8.REL_ORDER[r], "fwd" if d == 0 else "bwd"] for r, d in steps] for qt, steps in chains.items()},
           "families": fam,
           "nb_trim": {"r_star_share_removed": L8.by_hop(trim, hop, has_std), "gold_share_removed": L8.by_hop(lost, hop, has_gold)},
           "gold_unreached_nb": L8.by_hop(unreached, hop, has_gold),
           "topic_entity": {"in_pool": L8.by_hop(te >= 0, hop), "in_seeds": L8.by_hop(te_in_seeds, hop), "in_b0": L8.by_hop(te_in_b0, hop),
                            "queries_without_topic_entity_in_pool": int((te < 0).sum())},
           "gold_in_pool": L8.by_hop(has_gold, hop),
           "pool": {"mean": float(base.q_pool_size.mean()), "seeds_b0_mean": float((base.q_seed_bucket == 0).sum(1).mean()),
                    "seeds_b1_mean": float((base.q_seed_bucket == 1).sum(1).mean())},
           "edges": {"structural_mean": float(base.q_struct_edges.mean()), "tokenised_mean": float(base.q_token_edges.mean()),
                     "direction_classes_total": {c: int(base.q_dir_class[:, i].sum()) for i, c in enumerate(("fwd", "bwd", "both"))}},
           "meta_sha256": L0.sha256_file(DATA / "meta.json"), **L8.job_fields(t0)}
    write_json(DATA / "check.json", out)
    for f in FAMILIES:
        dc = fam[f]["direction_check"]
        if not dc["passes"]:
            hard_stop(f"direction_check ({f}): the swapped chain reaches an in-pool gold on more queries than the declared chain",
                      declared=dc["declared_share"], swapped=dc["swapped_share"])
    verify_inputs(decl)
    log(f"check: the chain reaches a gold on {fam['std']['direction_check']['declared_share']:.3f} (std) and "
        f"{fam['nb']['direction_check']['declared_share']:.3f} (nb) of queries; NB removes "
        f"{out['nb_trim']['r_star_share_removed']['all']:.3f} of R* and {out['nb_trim']['gold_share_removed']['all']:.4f} of the golds; "
        f"{out['gold_unreached_nb']['all']:.4f} of the in-pool golds lie in no nb reach set")
    return out


# ── main ─────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=("score", "assemble", "check"))
    ap.add_argument("--host", action="store_true", help="on the host: the verified mirror in place of the package, in memory")
    ap.add_argument("--shard", default=None, help="score: i/n, the chunks ci with ci mod n = i; a run without it assembles")
    ap.add_argument("--limit", type=int, default=None, help="score, smoke only: N queries spread over the population, written to --out")
    ap.add_argument("--out", type=Path, default=None, help="score, smoke only: a directory outside outputs/mp_approx_l10")
    args = ap.parse_args(argv)
    if not args.host:
        ap.error(f"--stage {args.stage} runs on the host (--host), placement")
    if args.shard is not None and args.stage != "score":
        ap.error("--shard is for --stage score")
    if (args.limit is None) != (args.out is None) or (args.limit is not None and args.stage != "score"):
        ap.error("--limit and --out go together, with --stage score (a smoke run)")
    decl = load_declaration()
    if args.limit is not None:
        out = args.out.resolve()
        if OUT.resolve() in (out, *out.parents):
            ap.error("a smoke run never writes under outputs/mp_approx_l10")
        HARD_STOP_DIR[0] = out
    route_stops()
    L8.host_mode(decl, log_utc)
    if args.stage in ("score", "assemble"):
        if args.limit is not None:
            stage_score(decl, log_utc, None, args.limit, out / NAME)
        else:
            stage_score(decl, log_utc, L8.parse_shard(args.shard) if args.stage == "score" else None)
    else:
        stage_check(decl, log_utc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
