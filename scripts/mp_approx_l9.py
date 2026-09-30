"""MP-Approx level 9, track MP-APPROX: level 8's non-message-passing typed-walk model pushed at its weak points on metaqa
(configs/mp_approx_l9.yaml). The walks are non-backtracking, the type vector gains an exact per-sequence residual
beside the rotary text composition, the grid is wider and EM runs more rounds, with level 8's protocol nested in every
unit.

On the host (placement): the verified mirror stands in for the package, in memory only.

    python scripts/mp_approx_l9.py --host --stage score --shard 0/3   # with 1/3 and 2/3, at once: scoring pass + both walk families
    python scripts/mp_approx_l9.py --host --stage assemble            # scores any chunk no shard wrote, then assembles
    python scripts/mp_approx_l9.py --host --stage check               # chain map, direction checks, anchors -> check.json
    python scripts/mp_approx_l9.py --host --stage fit --arm NB-hyb    # one job per arm: 3 seeds x 5 folds
    python scripts/mp_approx_l9.py --host --stage repeat              # the unit (NB-hyb, k 0, fold 0) again in a fresh process
    python scripts/mp_approx_l9.py --host --stage read                # rho, bootstrap, bands, contrasts -> read.json

On the laptop, from the fetched read.json, check.json and meta.json:

    python scripts/mp_approx_l9.py --stage doc                        # record.json and docs/MP_APPROX_L9.md, no arithmetic
    python scripts/mp_approx_l9.py --stage file --date 2026_10_01 --commit <sha> [--extra run_extra.json]

A systems smoke, which makes no number of the file: --stage score --limit N --out DIR (never under outputs/mp_approx_l9).

Level 8's script is imported unchanged. Its scoring pass runs with some of its module names rebound inside this process
only (the population rule, the walk programme, the pins, the declaration and the output paths), and they are restored
when the pass ends. Nothing is written to level 8's files or outputs. Measurement only: every learned quantity is a
function of the query embedding and a discrete walk type, applied once to counts compiled before any fit (boundary).
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
import copy  # noqa: E402
import hashlib  # noqa: E402
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

CONFIG = ROOT / "configs" / "mp_approx_l9.yaml"
OUT = ROOT / "outputs" / "mp_approx_l9"
NAME = L8.NAME
DATA = OUT / NAME
RECORD = OUT / "record.json"
DOC = ROOT / "docs" / "MP_APPROX_L9.md"
SCRIPT_REL = "scripts/mp_approx_l9.py"
LF = L8.LF

SEEDS, FOLDS, FUNCS = L8.SEEDS, L8.FOLDS, L8.FUNCS
METRIC_NAMES, rank_metrics = L8.METRIC_NAMES, L8.rank_metrics
PER_HOP = 1000
SALT = "mp_approx_l9|"
AVAILABLE_PER_HOP = {1: 3221, 2: 5644, 3: 5473}
TB, MAX_L, D, N_DIR = L8.TB, L8.MAX_L, L8.D, L8.N_DIR
NB_SHIFT = 2 * TB ** MAX_L               # walks.one_sidecar: an NB type of bucket b is stored under bucket b + 2
FAMILIES = ("std", "nb")
HIDDEN = 256                             # arms.query_map.mlp
KAPPAS = (1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2, 1.0, 2.0, 4.0, 8.0, 16.0)
ETAS = (0.01, 0.1, 1.0, 10.0, 100.0)
EM_ROUNDS = 10                           # the nested protocol reads level 8's own EM_ROUNDS, KAPPAS and ETAS
ARM_SPEC = {"NB-hyb": ("nb", "hyb", "linear"), "NB-hyb-mlp": ("nb", "hyb", "mlp"), "NB-text": ("nb", "text", "linear"),
            "NB-id": ("nb", "id", "linear"), "NB-hash": ("nb", "hash", "linear"), "STD-hyb": ("std", "hyb", "linear"),
            "STD-text": ("std", "text", "linear"), "STD-hash": ("std", "hash", "linear")}
ARMS = tuple(ARM_SPEC)
NESTED = "@L8"
REFERENCES = {"NB-oracle": "nb", "STD-oracle": "std"}
PRIMARY = "NB-hyb"
REPEAT_UNIT = ("NB-hyb", 0, 0)
RETRIEVAL = L8.RETRIEVAL
CONTRASTS = {"nb_adds": ("NB-hyb", "STD-hyb"), "nb_adds_hash": ("NB-hash", "STD-hash"), "id_residual_adds": ("NB-hyb", "NB-text"),
             "text_residual_adds": ("NB-hyb", "NB-id"), "id_vs_hash": ("NB-id", "NB-hash"), "query_mlp_adds": ("NB-hyb-mlp", "NB-hyb"),
             "protocol_adds": ("NB-hyb", "NB-hyb@L8"), "over_level8_primary": ("NB-hyb", "STD-text@L8"),
             "ceiling_gap": ("NB-oracle", "NB-hyb"), "nb_ceiling": ("NB-oracle", "STD-oracle")}
INTERPRET_CONTRAST = {"nb_adds": ("nb_adds", "nb_hurts"), "id_residual_adds": ("id_residual_adds", None),
                      "text_residual_adds": ("text_residual_adds", None), "id_vs_hash": ("id_beats_hash", None),
                      "query_mlp_adds": ("query_mlp_adds", None), "protocol_adds": ("protocol_adds", None),
                      "over_level8_primary": ("above_level8_primary", None), "nb_ceiling": ("nb_raises_ceiling", None)}
HARD_STOP_DIR = [OUT]                    # a smoke run and the tests point it at their own directory

_L8_WALKS = L8.walk_entries              # level 8's programme, rule and pins, captured before any rebinding
_L8_ROWS = L8.l8_rows
_L8_VERIFY = L8.verify_inputs


# ── small helpers ────────────────────────────────────────────────────────────


log_utc = L8.log_utc
write_json, read_json, shown = L8.write_json, L8.read_json, L8.shown


def route_stops() -> None:
    """Level 0's, level 3's and level 8's hard stops land beside this file's."""
    L8.HARD_STOP_DIR[0] = L0.HARD_STOP_DIR[0] = L3.HARD_STOP_DIR[0] = HARD_STOP_DIR[0]


def hard_stop(message: str, **evidence) -> None:
    """hard_stops: the evidence goes to outputs/mp_approx_l9/hard_stops.json; the status line is left alone."""
    route_stops()
    L0.hard_stop(message, **evidence)


def load_declaration() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


# ── pins ─────────────────────────────────────────────────────────────────────


def verify_inputs(decl: dict) -> None:
    """inputs: level 8's pins in this file's copy (pilot, frozen code, level 0, level 4, relations, boundary, host, and level
    0's verify_pins on metaqa), then level 8's own files and its pinned qids.json."""
    route_stops()
    _L8_VERIFY(decl)
    lv = decl["inputs"]["level8"]
    pins = [(lv[k]["path"], lv[k]["sha256"], True) for k in ("declaration_lf", "script_lf", "tests_lf")]
    pins.append((lv["excluded_rows"]["qids"]["path"], lv["excluded_rows"]["qids"]["sha256"], False))
    for rel, digest, lf in pins:
        p = ROOT / rel
        found = (L0.lf_sha256(p) if lf else L0.sha256_file(p)) if p.exists() else "missing"
        if found != digest:
            hard_stop(f"{rel}{' (LF)' if lf else ''} is not its pinned sha256", path=rel, pinned=digest, found=found)


# ── the population rule ──────────────────────────────────────────────────────


def rows_by_salt(ids: list[str], gate: np.ndarray, excluded: np.ndarray, salt: str, per_hop: int) -> tuple[np.ndarray, dict]:
    """Level 8's rule (l8_rows) with the salt as an argument: per hop, the gate's rows outside `excluded` sorted by
    sha256(salt + id), the first per_hop kept, returned in population order with the rows available per hop."""
    gate = np.asarray(gate, dtype=bool)
    out = np.zeros(gate.size, dtype=bool)
    out[np.asarray(excluded, dtype=np.int64)] = True
    keep, available = [], {}
    for h in (1, 2, 3):
        rows = [int(i) for i in np.flatnonzero(gate & ~out) if L0.hop_from_id(ids[i]) == h]
        available[h] = len(rows)
        if len(rows) < per_hop:
            raise SystemExit(f"hop {h}: {len(rows)} rows available, fewer than {per_hop}")
        rows.sort(key=lambda i: hashlib.sha256((salt + ids[i]).encode("utf-8")).hexdigest())
        keep.extend(rows[:per_hop])
    return np.sort(np.asarray(keep, dtype=np.int64)), available


def l9_rows(ids: list[str], gate: np.ndarray, excluded: np.ndarray, per_hop: int = PER_HOP, level8_ids: list[str] | None = None,
            level8_per_hop: int = L8.PER_HOP) -> tuple[np.ndarray, dict]:
    """population.rule: level 8's rows, recomputed by level 8's own rule and salt from the rows it excluded (level 0's),
    must be its pinned qids.json; they leave with level 0's, and the rest are sorted per hop by sha256(SALT + id)."""
    l8, _available8 = _L8_ROWS(ids, gate, excluded, level8_per_hop)
    if level8_ids is not None and [ids[i] for i in l8] != list(level8_ids):
        hard_stop("level 8's recomputed rows are not its pinned qids.json")
    rows, available = rows_by_salt(ids, gate, np.union1d(np.asarray(excluded, dtype=np.int64), l8), SALT, per_hop)
    if np.isin(rows, l8).any() or np.isin(rows, excluded).any():
        hard_stop("a level 0 or level 8 row would be scored")
    return rows, available


# ── the walks ────────────────────────────────────────────────────────────────


def _empty() -> tuple[np.ndarray, np.ndarray]:
    z = np.zeros(0, dtype=np.int64)
    return z, z.copy()


def _expand(pk, node, cnt, indptr, dst, tok, n: int, block: int, keep_prev: bool) -> tuple[np.ndarray, np.ndarray]:
    """Every (state, out-edge) pair of the states (pk, node) with counts cnt, blocked by out-degree as level 8's
    walk_entries is: the key (pk * TB + t + 1, node, dst) with keep_prev, else (pk * TB + t + 1, dst), and the int64
    counts of equal keys summed. The keys come back sorted."""
    deg = indptr[node + 1] - indptr[node]
    cum = np.cumsum(deg)
    keys, sums, start = [], [], 0
    while start < node.size:
        base = int(cum[start - 1]) if start else 0
        stop = max(int(np.searchsorted(cum, base + block, side="right")), start + 1)
        d = deg[start:stop]
        m = int(d.sum())
        if m:
            rep = np.repeat(np.arange(stop - start), d)
            eidx = np.repeat(indptr[node[start:stop]] - (np.cumsum(d) - d), d) + np.arange(m)
            npk = pk[start:stop][rep] * TB + tok[eidx] + 1
            key = (npk * n + node[start:stop][rep]) * n + dst[eidx] if keep_prev else npk * n + dst[eidx]
            k_, s_ = L8._aggregate(key, cnt[start:stop][rep])
            keys.append(k_)
            sums.append(s_)
        start = stop
    if not keys:
        return _empty()
    return L8._aggregate(np.concatenate(keys), np.concatenate(sums)) if len(keys) > 1 else (keys[0], sums[0])


def _step_backs(pk, prev, node, cnt, pair_key, pair_tok, n: int, keep_prev: bool) -> tuple[np.ndarray, np.ndarray]:
    """The walks that step straight back: for each state (pk, prev, node) and each edge node -> prev, the key
    (pk * TB + t + 1, node, prev) with keep_prev, else (pk * TB + t + 1, prev), with the state's counts summed."""
    want = node * n + prev
    lo = np.searchsorted(pair_key, want, side="left")
    d = np.searchsorted(pair_key, want, side="right") - lo
    m = int(d.sum())
    if not m:
        return _empty()
    rep = np.repeat(np.arange(want.size), d)
    eidx = np.repeat(lo - (np.cumsum(d) - d), d) + np.arange(m)
    npk = pk[rep] * TB + pair_tok[eidx] + 1
    key = (npk * n + node[rep]) * n + prev[rep] if keep_prev else npk * n + prev[rep]
    return L8._aggregate(key, cnt[rep])


def walk_entries_nb(src, dst, tok, n: int, buckets, block: int = L8.EXPAND_BLOCK, cap: int = L8.MAX_Q_ENTRIES):
    """walks.nb: for each bucket b and L = 1..3, the number of walks v_0 .. v_L of each type (b, t1..tL) from the
    bucket's seeds to every node over the tokenised edges, with v_{i+1} != v_{i-1} (never straight back); an edge with
    two relations is two parallel edges, and a step back is excluded whichever relation it uses.

    An exact int64 programme over (prefix, previous node, node) states. At length 1 the states are the edges from the
    seeds, with the seed as the previous node. At each later length they are the previous length's entries (its states
    summed over the previous node) expanded over the edges sorted by source, as level 8's walk_entries expands its own,
    less, for every state (prefix, u, v) and every edge v -> u, the walks that would step back to u. The last length
    keeps no previous node, and each stored entry sums its states over it. Returns (code, node, count) sorted by code,
    then node, in level 8's form."""
    src = np.asarray(src, dtype=np.int64)
    dst = np.asarray(dst, dtype=np.int64)
    tok = np.asarray(tok, dtype=np.int64)
    order = np.argsort(src, kind="stable")
    src, dst, tok = src[order], dst[order], tok[order]
    indptr = np.zeros(n + 1, dtype=np.int64)
    indptr[1:] = np.cumsum(np.bincount(src, minlength=n))
    pair = np.lexsort((dst, src))        # the edges by (source, destination), for the steps back
    pair_key, pair_tok = src[pair] * n + dst[pair], tok[pair]
    codes, nodes, counts, total = [], [], [], 0
    for b, seeds in enumerate(buckets):
        seed = np.unique(np.asarray(seeds, dtype=np.int64))
        if seed.size == 0:
            continue
        if seed.min() < 0 or seed.max() >= n:
            raise ValueError("a seed outside the pool")
        for length in range(1, MAX_L + 1):
            keep = length < MAX_L
            if length == 1:
                key, cnt = _expand(np.full(seed.size, b, dtype=np.int64), seed, np.ones(seed.size, dtype=np.int64),
                                   indptr, dst, tok, n, block, keep)
            else:
                k_main, c_main = _expand(epk, enode, ecnt, indptr, dst, tok, n, block, keep)
                k_back, c_back = _step_backs(spk, sprev, snode, scnt, pair_key, pair_tok, n, keep)
                key, cnt = L8._aggregate(np.concatenate([k_main, k_back]), np.concatenate([c_main, -c_back]))
                if cnt.size and int(cnt.min()) < 0:
                    raise ValueError("a negative non-backtracking count")
                live = cnt > 0
                key, cnt = key[live], cnt[live]
            if key.size == 0:
                break
            if keep:
                snode, rest = key % n, key // n
                sprev, spk, scnt = rest % n, rest // n, cnt
                ekey, ecnt = L8._aggregate(spk * n + snode, cnt)
            else:
                ekey, ecnt = key, cnt
            epk, enode = ekey // n, ekey % n
            total += ekey.size
            if total > cap:
                raise L8.WalkCeiling(total)
            if int(ecnt.max()) >= 2 ** 32:
                raise ValueError("a walk count >= 2^32")
            codes.append(epk * TB ** (MAX_L - length))
            nodes.append(enode)
            counts.append(ecnt)
    if not codes:
        z = np.zeros(0, dtype=np.int64)
        return z, z.copy(), z.copy()
    code, node, count = np.concatenate(codes), np.concatenate(nodes), np.concatenate(counts)
    order = np.lexsort((node, code))
    return code[order], node[order], count[order]


def nb_within_std(c1, v1, k1, c2, v2, k2, n: int) -> None:
    """walks.checks_per_query: every NB entry is a standard entry with a count no larger, and the length-1 entries of
    the two families are equal. Raises ValueError otherwise."""
    c1, v1, k1 = (np.asarray(x, dtype=np.int64) for x in (c1, v1, k1))
    c2, v2, k2 = (np.asarray(x, dtype=np.int64) for x in (c2, v2, k2))
    key1, key2 = c1 * n + v1, c2 * n + v2
    i = np.searchsorted(key1, key2)
    found = i < key1.size
    found[found] = key1[i[found]] == key2[found]
    if not found.all():
        raise ValueError(f"{int((~found).sum())} non-backtracking entries are not standard entries")
    if (k2 > k1[i]).any():
        raise ValueError("a non-backtracking count exceeds the standard count")
    s1, s2 = L8.decode_types(c1)[2] == 1, L8.decode_types(c2)[2] == 1
    if not (np.array_equal(c1[s1], c2[s2]) and np.array_equal(v1[s1], v2[s2]) and np.array_equal(k1[s1], k2[s2])):
        raise ValueError("the non-backtracking length-1 entries differ from the standard ones")


def walk_entries_both(src, dst, tok, n: int, buckets, block: int = L8.EXPAND_BLOCK, cap: int = L8.MAX_Q_ENTRIES):
    """walks.one_sidecar: level 8's entries, from level 8's walk_entries unchanged, then the NB entries under buckets 2
    and 3, checked against each other. The codes stay sorted, and every query's NB entries follow its standard ones."""
    c1, v1, k1 = _L8_WALKS(src, dst, tok, n, buckets, block, cap)
    c2, v2, k2 = walk_entries_nb(src, dst, tok, n, buckets, block, cap)
    if c1.size + c2.size > cap:
        raise L8.WalkCeiling(int(c1.size + c2.size))
    nb_within_std(c1, v1, k1, c2, v2, k2, n)
    return np.concatenate([c1, c2 + NB_SHIFT]), np.concatenate([v1, v2]), np.concatenate([k1, k2])


class View(L8.Data):
    """walks.one_sidecar: one family of the combined sidecar, read back under buckets 0 and 1. The per-query and
    per-node arrays are the sidecar's, the per-type arrays are the family's rows, and the entries are read in place."""

    def __init__(self, d: Path, family: str, check: bool = True):
        if family not in FAMILIES:
            raise ValueError(family)
        super().__init__(d, check)
        self.family = family
        stored = np.asarray(self.t_code, dtype=np.int64)
        bucket = stored // TB ** MAX_L
        if bucket.size and (bucket.min() < 0 or bucket.max() > 3):
            raise SystemExit(f"{self.dir}: a stored type is not under buckets 0 to 3")
        keep = bucket >= 2 if family == "nb" else bucket < 2
        owner = np.repeat(np.arange(self.n_q, dtype=np.int64), self.q_types)[keep]
        self.t_code = stored[keep] - (NB_SHIFT if family == "nb" else 0)
        self.t_size = np.asarray(self.t_size)[keep]
        self.t_gold = np.asarray(self.t_gold)[keep]
        self.t_first = self.t_first[keep]
        self.q_types = np.bincount(owner, minlength=self.n_q).astype(np.int64)
        self.q_entries = np.zeros(self.n_q, dtype=np.int64)
        np.add.at(self.q_entries, owner, self.t_size.astype(np.int64))
        self.type_ptr = np.r_[0, np.cumsum(self.q_types)].astype(np.int64)
        self.entry_start = np.zeros(self.n_q, dtype=np.int64)
        has = self.q_types > 0
        self.entry_start[has] = self.t_first[self.type_ptr[:-1][has]]
        same = owner[1:] == owner[:-1]
        if not np.array_equal(self.t_first[1:][same], (self.t_first + self.t_size)[:-1][same]):
            raise SystemExit(f"{self.dir}: the {family} entries of a query are not one block")
        self.entry_ptr = None             # level 8's pointer spans both families; entries() reads the family's block

    def entries(self, q: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        s = slice(int(self.entry_start[q]), int(self.entry_start[q] + self.q_entries[q]))
        tl = np.repeat(np.arange(int(self.q_types[q]), dtype=np.int64), self.t_size[self.type_rows(q)])
        return np.asarray(self.e_node[s], dtype=np.int64), np.asarray(self.e_count[s], dtype=np.int64), tl


def views(d: Path) -> dict:
    """Both families of the sidecar; its files are checked against meta.json once."""
    return {"std": View(d, "std"), "nb": View(d, "nb", check=False)}


# ── stage: score / assemble (level 8's pass, names rebound in this process) ──


@contextlib.contextmanager
def level8_rebound(out_dir: Path, rows_rule):
    """scoring_pass.path: level 8's module names that differ here, rebound for the pass and restored after it."""
    names = {"l8_rows": rows_rule, "walk_entries": walk_entries_both, "verify_inputs": verify_inputs, "CONFIG": CONFIG, "OUT": OUT,
             "DATA": out_dir, "AVAILABLE_PER_HOP": dict(AVAILABLE_PER_HOP)}
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
    """scoring_pass: level 8's stage_score with this file's rule, walks, pins and paths; this file's verify_inputs runs
    inside it, before the population is read and again at the end."""
    out_dir = out_dir or DATA
    level8_ids = read_json(ROOT / decl["inputs"]["level8"]["excluded_rows"]["qids"]["path"])

    def rows_rule(ids, gate, excluded, per_hop=PER_HOP):
        return l9_rows(ids, gate, excluded, per_hop, level8_ids=level8_ids)

    route_stops()
    with level8_rebound(out_dir, rows_rule):
        L8.stage_score(decl, log, shard, limit, out_dir)


# ── stage: check ─────────────────────────────────────────────────────────────


def check_family(view: View, chains: dict, qt_of: list[str]) -> tuple[dict, list[np.ndarray]]:
    """walks.direction_check and level 8's chain anchors on one family, plus self_reach; also R* per query."""
    n = view.n_q
    reach = {key: np.zeros(n, dtype=bool) for key in ("b0", "b1", "swapped_b0", "swapped_b1")}
    rec, prec, bucket_star, size = np.zeros(n), np.zeros(n), np.zeros(n, dtype=np.int64), np.zeros(n, dtype=np.int64)
    self_seed, self_gold, rstars = np.zeros(n, dtype=bool), np.zeros(n, dtype=bool), []
    for q in range(n):
        steps = chains[qt_of[q]]
        g = view.gold_local(q)
        for swap, tag in ((False, ""), (True, "swapped_")):
            r0, r1 = L8.chain_reach(view, q, steps, swap)
            reach[f"{tag}b0"][q] = bool(np.isin(g, r0).any())
            reach[f"{tag}b1"][q] = bool(np.isin(g, r1).any())
        rs, bstar = L8.r_star(view, q, steps)
        rstars.append(rs)
        bucket_star[q], size[q] = bstar, rs.size
        if g.size:
            hit = np.intersect1d(rs, g).size
            rec[q], prec[q] = hit / g.size, (hit / rs.size if rs.size else 0.0)
        seeds, buckets = view.q_seed_local[q], view.q_seed_bucket[q]
        s_in = np.intersect1d(seeds[(buckets == 0) & (seeds >= 0)], rs)
        self_seed[q], self_gold[q] = s_in.size > 0, bool(np.isin(s_in, g).any())
    either = reach["b0"] | reach["b1"]
    swapped = reach["swapped_b0"] | reach["swapped_b1"]
    hop = view.q_hop
    has_gold = view.q_gold_in_pool > 0
    out = {"direction_check": {"declared_share": L8.share(either), "swapped_share": L8.share(swapped), "queries": n,
                               "passes": bool(swapped.sum() <= either.sum())},
           "chain_reach": {"b0": L8.by_hop(reach["b0"], hop), "b1": L8.by_hop(reach["b1"], hop), "either": L8.by_hop(either, hop),
                           "swapped_either": L8.by_hop(swapped, hop)},
           "chain_fit": {"recall": L8.by_hop(rec, hop, has_gold), "precision": L8.by_hop(prec, hop, has_gold),
                         "r_star_from_b1": L8.by_hop(bucket_star == 1, hop), "r_star_size_mean": float(size.mean()),
                         "r_star_empty": L8.by_hop(size == 0, hop)},
           "self_reach": {"b0_seed_in_r_star": L8.by_hop(self_seed, hop), "gold_b0_seed_in_r_star": L8.by_hop(self_gold, hop)},
           "sizes": {"types_mean": float(view.q_types.mean()), "types_max": int(view.q_types.max()),
                     "entries_mean": float(view.q_entries.mean()), "entries_max": int(view.q_entries.max()),
                     "entries_total": int(view.q_entries.sum())}}
    return out, rstars


def stage_check(decl: dict, log=print) -> dict:
    """walks.true_chain, walks.direction_check on both families, and the anchors that need no fit, before any fit."""
    t0 = time.time()
    torch.set_num_threads(L8.FIT_THREADS)
    verify_inputs(decl)
    vs = views(DATA)
    base = vs["std"]
    meta = base.meta
    if meta["limit"] is not None:
        raise SystemExit("the sidecar is a smoke run")
    inp = decl["inputs"]
    l0_ids = set(read_json(ROOT / inp["level0"]["excluded_rows"]["qids"]["path"]))
    l8_ids = set(read_json(ROOT / inp["level8"]["excluded_rows"]["qids"]["path"]))
    if (l0_ids | l8_ids) & set(base.qids):
        hard_stop("a level 0 or level 8 query id is in the sidecar")
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
        fam[f], rstar[f] = check_family(v, chains, qt_of)
    n, hop = base.n_q, base.q_hop
    trim, lost, has_std = np.zeros(n), np.zeros(n), np.zeros(n, dtype=bool)
    for q in range(n):
        rs, rn = rstar["std"][q], rstar["nb"][q]
        g = base.gold_local(q)
        gone = np.setdiff1d(rs, rn)
        if rs.size:
            has_std[q] = True
            trim[q] = gone.size / rs.size
        if g.size:
            lost[q] = np.intersect1d(gone, g).size / g.size
    te = base.q_te_local
    te_in_seeds = np.asarray([bool(te[q] >= 0 and (base.q_seed_local[q] == te[q]).any()) for q in range(n)])
    te_in_b0 = np.asarray([bool(te[q] >= 0 and ((base.q_seed_local[q] == te[q]) & (base.q_seed_bucket[q] == 0)).any())
                           for q in range(n)])
    out = {"stage": "check", "queries": n, "qtypes": len(meta["qtypes"]),
           "qtypes_per_hop": {f"hop={h}": len({qt_of[q] for q in range(n) if hop[q] == h}) for h in (1, 2, 3)},
           "chain_map": {qt: [[L8.REL_ORDER[r], "fwd" if d == 0 else "bwd"] for r, d in steps] for qt, steps in chains.items()},
           "families": fam,
           "nb_trim": {"r_star_share_removed": L8.by_hop(trim, hop, has_std),
                       "gold_share_removed": L8.by_hop(lost, hop, base.q_gold_in_pool > 0)},
           "topic_entity": {"in_pool": L8.by_hop(te >= 0, hop), "in_seeds": L8.by_hop(te_in_seeds, hop), "in_b0": L8.by_hop(te_in_b0, hop),
                            "queries_without_topic_entity_in_pool": int((te < 0).sum())},
           "gold_in_pool": L8.by_hop(base.q_gold_in_pool > 0, hop),
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
        f"{out['nb_trim']['r_star_share_removed']['all']:.3f} of R* and {out['nb_trim']['gold_share_removed']['all']:.4f} of the golds")
    return out


# ── the models ───────────────────────────────────────────────────────────────


def sequence_index(table: L8.TypeTable) -> tuple[np.ndarray, np.ndarray]:
    """arms.type_vector.id: the token sequences of the table (bucket dropped), ascending, and each type's sequence."""
    seq = np.asarray(table.codes, dtype=np.int64) % TB ** MAX_L
    uniq, inv = np.unique(seq, return_inverse=True)
    return uniq, inv.astype(np.int64).ravel()


class ResidualModel(torch.nn.Module):
    """arms.type_vector hyb and id, with the linear or mlp query map, in level 8's weight form: w(q, tau) = <A(q), e(tau)
    + beta_b + lambda_L> + c_{b,L} and the null logit <A(q), nu> + c_null. hyb creates A, P, delta and theta in level 8's
    TP order and E last at 0, so it starts exactly at level 8's TP. Its inputs are the query embedding and discrete type
    indices; its only buffers are the relation text (hyb) and the type table. No edge list, neighbour embedding or
    neighbour score reaches it."""

    def __init__(self, composition: str, query_map: str, rel_emb: np.ndarray, table: L8.TypeTable, q_dim: int = 1536):
        super().__init__()
        if composition not in ("hyb", "id") or query_map not in ("linear", "mlp"):
            raise ValueError(f"{composition}, {query_map}")
        self.composition, self.query_map = composition, query_map
        if query_map == "linear":
            self.A = torch.nn.Linear(q_dim, D, bias=False)
        else:
            self.A = torch.nn.Sequential(torch.nn.Linear(q_dim, HIDDEN), torch.nn.GELU(), torch.nn.Linear(HIDDEN, D))
        if composition == "hyb":
            self.P = torch.nn.Linear(rel_emb.shape[1], D, bias=False)
            self.delta = torch.nn.Parameter(torch.zeros(N_DIR, D))
            self.theta = torch.nn.Parameter(L8.THETA_BASE ** (-torch.arange(L8.PLANES, dtype=torch.float32) / L8.PLANES))
        self.beta = torch.nn.Parameter(torch.zeros(2, D))
        self.lam = torch.nn.Parameter(torch.zeros(MAX_L, D))
        self.c = torch.nn.Parameter(torch.zeros(2, MAX_L))
        self.nu = torch.nn.Parameter(torch.zeros(D))
        self.c_null = torch.nn.Parameter(torch.zeros(()))
        _seqs, seq_of = sequence_index(table)
        self.E = torch.nn.Parameter(torch.zeros(_seqs.size, D))
        if composition == "hyb":
            self.register_buffer("rel_emb", torch.as_tensor(np.asarray(rel_emb, dtype=np.float32)))
        self.register_buffer("tok", torch.as_tensor(table.toks, dtype=torch.long))
        self.register_buffer("tb", torch.as_tensor(table.b, dtype=torch.long))
        self.register_buffer("tl", torch.as_tensor(table.L, dtype=torch.long))
        self.register_buffer("seq", torch.as_tensor(seq_of, dtype=torch.long))

    def compose(self) -> torch.Tensor:
        """e(tau) for every type of the table: E[s(tau)], plus level 8's rotary composition for hyb."""
        e = self.E[self.seq]
        if self.composition == "hyb":
            rho = (self.P(self.rel_emb)[:, None, :] + self.delta[None, :, :]).reshape(-1, D)
            for pos in range(MAX_L):
                live = (self.tok[:, pos] >= 0).to(torch.float32)[:, None]
                e = e + L8.rotate(rho[self.tok[:, pos].clamp(min=0)] * live, (pos + 1) * self.theta)
        return e

    def weights(self, qemb: torch.Tensor, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        aq = self.A(qemb)
        e = self.compose() + self.beta[self.tb] + self.lam[self.tl - 1]
        w_all = aq @ e.T + self.c[self.tb, self.tl - 1][None, :]
        return torch.gather(w_all, 1, idx), aq @ self.nu + self.c_null


def make_model(arm: str, rel_emb: np.ndarray, table: L8.TypeTable) -> torch.nn.Module:
    """arms.fitted: text and hash are level 8's TypeModel "TP" and "TP-hash", called unchanged."""
    _family, comp, qmap = ARM_SPEC[arm]
    if comp == "text":
        return L8.TypeModel("TP", rel_emb, table)
    if comp == "hash":
        return L8.TypeModel("TP-hash", rel_emb, table)
    return ResidualModel(comp, qmap, rel_emb, table)


# ── fitting ──────────────────────────────────────────────────────────────────


def score_protocol(fx: L8.Fitter, model, states: dict, rounds: list, last: int, kappas, etas, k: int, inner_q, score_q) -> dict:
    """One protocol's selection and scores, as level 8's fit_unit makes them: the kept round over rounds 0..last (ties:
    the earliest), then kappa and eta on the inner queries (ties: the smaller kappa, then the larger eta), then the
    fold's scores."""
    data = fx.data
    kept = int(np.argmax([rd["inner_mll"] for rd in rounds[:last + 1]]))
    final = None if kept == 0 else model
    if kept:
        model.load_state_dict(states[kept])
    lp_inner = fx.logp(final, inner_q)
    grid = {}
    base = {int(q): (fx.mixture(int(q), lp), data.gold_local(int(q)), int(data.q_gold_total[q]), data.z(int(q), k))
            for q, lp in zip(inner_q, lp_inner)}
    for kappa in kappas:
        for eta in etas:
            grid[f"{kappa}|{eta}"] = L8.mean3([rank_metrics(z + kappa * np.log(nm + eta), g, gt) for nm, g, gt, z in base.values()])
    order = sorted(((-v if np.isfinite(v) else float("inf"), kappa, -eta) for kappa in kappas for eta in etas
                    for v in [grid[f"{kappa}|{eta}"]]))
    kappa, eta = order[0][1], -order[0][2]
    del base
    lp_score = fx.logp(final, score_q)
    scores, metrics, argmax = [], [], []
    for q, lp in zip(score_q, lp_score):
        q = int(q)
        s = data.z(q, k) + kappa * np.log(fx.mixture(q, lp) + eta)
        scores.append(s)
        metrics.append([rank_metrics(s, data.gold_local(q), int(data.q_gold_total[q]))[m] for m in METRIC_NAMES])
        t = int(data.q_types[q])
        a = int(np.argmax(lp))
        argmax.append(-1 if (kept == 0 or a == t) else int(data.t_code[data.type_rows(q)][a]))   # a uniform p chooses nothing
    return {"kept_round": kept, "kappa": kappa, "eta": eta, "grid": grid, "scores": scores,
            "metrics": np.asarray(metrics, dtype=np.float64).reshape(-1, len(METRIC_NAMES)), "argmax": np.asarray(argmax, dtype=np.int64)}


def fit_unit(fx: L8.Fitter, arm: str, k: int, fold: int, log=print) -> tuple[dict, dict]:
    """One cross-fitted unit (em, cross_fitting): level 8's EM loop for EM_ROUNDS rounds, then level 8's nested protocol
    (rounds 0..L8.EM_ROUNDS, level 8's grid) and this file's (rounds 0..EM_ROUNDS, the wide grid). Returns (arrays, fit
    log)."""
    t0 = time.time()
    seed = 1000 + 10 * k + fold
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    data = fx.data
    fit_q, inner_q, score_q = L8.unit_queries(data, fold)
    model = make_model(arm, fx.rel_emb, fx.table)
    flog = {"arm": arm, "family": ARM_SPEC[arm][0], "k": k, "fold": fold, "seed": seed, "fit_queries": int(fit_q.size),
            "inner_queries": int(inner_q.size), "scored_queries": int(score_q.size), "parameters": L8.n_params(model),
            "walk_types": int(fx.table.codes.size)}
    states, rounds = {0: None}, []
    lp_fit, lp_inner = fx.logp(None, fit_q), fx.logp(None, inner_q)
    _g, mll_fit = fx.e_step(lp_fit, fit_q)
    _g, mll_inner = fx.e_step(lp_inner, inner_q)
    rounds.append({"round": 0, "p": "uniform", "fit_mll": float(mll_fit.sum()), "inner_mll": float(mll_inner.sum())})
    for r in range(1, EM_ROUNDS + 1):
        gam_fit, _m = fx.e_step(lp_fit, fit_q)
        gam_inner, _m = fx.e_step(lp_inner, inner_q)
        gof = {int(q): g for q, g in zip(fit_q, gam_fit)}
        opt = torch.optim.AdamW(model.parameters(), lr=L8.LR, weight_decay=L8.WD)
        best, best_state, epochs = float("inf"), copy.deepcopy(model.state_dict()), []
        for epoch in range(1, L8.M_EPOCHS + 1):
            model.train()
            tot, cnt = 0.0, 0
            for bq in L8.minibatches(fit_q, rng):
                opt.zero_grad()
                loss = fx.soft_ce(model, bq, [gof[int(q)] for q in bq], grad=True)
                loss.backward()
                opt.step()
                tot, cnt = tot + float(loss.detach()) * bq.size, cnt + bq.size
            model.eval()
            val = float(fx.soft_ce(model, inner_q, gam_inner, grad=False)) if inner_q.size else float("nan")
            epochs.append({"epoch": epoch, "train_soft_ce": tot / max(cnt, 1), "inner_soft_ce": val})
            if val < best:
                best, best_state = val, copy.deepcopy(model.state_dict())
        model.load_state_dict(best_state)
        states[r] = copy.deepcopy(best_state)
        lp_fit, lp_inner = fx.logp(model, fit_q), fx.logp(model, inner_q)
        _g, mll_fit = fx.e_step(lp_fit, fit_q)
        _g, mll_inner = fx.e_step(lp_inner, inner_q)
        rounds.append({"round": r, "epochs": epochs, "best_inner_soft_ce": best, "fit_mll": float(mll_fit.sum()),
                       "inner_mll": float(mll_inner.sum())})
    nested = score_protocol(fx, model, states, rounds, L8.EM_ROUNDS, L8.KAPPAS, L8.ETAS, k, inner_q, score_q)
    main = score_protocol(fx, model, states, rounds, EM_ROUNDS, KAPPAS, ETAS, k, inner_q, score_q)
    mono = [r for r in range(1, len(rounds)) if rounds[r]["fit_mll"] < rounds[r - 1]["fit_mll"]]
    flog.update({"rounds": rounds, "kept_round": main["kept_round"], "kappa": main["kappa"], "eta": main["eta"],
                 "grid_inner_mean3": main["grid"], "kept_round_l8": nested["kept_round"], "kappa_l8": nested["kappa"],
                 "eta_l8": nested["eta"], "grid_inner_mean3_l8": nested["grid"], "em_not_monotone_rounds": mono})
    sizes = np.asarray([s.size for s in main["scores"]], dtype=np.int64)
    arrays = {"q": score_q.astype(np.int64), "metrics": main["metrics"],
              "score": np.concatenate(main["scores"]) if main["scores"] else np.zeros(0),
              "score_ptr": np.r_[0, np.cumsum(sizes)].astype(np.int64), "argmax": main["argmax"],
              "metrics_l8": nested["metrics"], "argmax_l8": nested["argmax"]}
    flog["timing"] = {"seconds": round(time.time() - t0, 1), "utc": L0.utc()}
    log(f"   {arm} k{k} f{fold}: {flog['timing']['seconds']:.0f}s, kept round {main['kept_round']} (level 8's protocol "
        f"{nested['kept_round']}), kappa {main['kappa']} ({nested['kappa']}), eta {main['eta']} ({nested['eta']})")
    return arrays, flog


def unit_paths(arm: str, k: int, fold: int, root: Path | None = None) -> tuple[Path, Path]:
    d = root or DATA / "units" / arm
    return d / f"k{k}_f{fold}.npz", d / f"k{k}_f{fold}.json"


def check_passed() -> None:
    path = DATA / "check.json"
    if not path.exists() or not all(read_json(path)["families"][f]["direction_check"]["passes"] for f in FAMILIES):
        raise SystemExit("the check stage has not passed; no fit runs before the direction checks")


def stage_fit(decl: dict, arm: str, log=print) -> None:
    t0 = time.time()
    L8.fit_process()
    verify_inputs(decl)
    check_passed()
    view = View(DATA, ARM_SPEC[arm][0])
    fx = L8.Fitter(view, L8.load_rel_emb(decl))
    log(f"{arm}: {view.n_q} queries, {fx.table.codes.size} {view.family} walk types in the population")
    for k in SEEDS:
        for fold in range(FOLDS):
            npz, js = unit_paths(arm, k, fold)
            if js.exists():
                continue
            arrays, flog = fit_unit(fx, arm, k, fold, log)
            L8.save_unit(arrays, flog, npz, js, L8.job_fields(t0))
    verify_inputs(decl)
    log(f"{arm}: {len(SEEDS) * FOLDS} units filed")


def stage_repeat(decl: dict, log=print) -> None:
    """cross_fitting.repeat: the unit (NB-hyb, k 0, fold 0) again in a fresh process, into repeat/."""
    t0 = time.time()
    L8.fit_process()
    verify_inputs(decl)
    check_passed()
    arm, k, fold = REPEAT_UNIT
    npz, js = unit_paths(arm, k, fold, DATA / "repeat")
    if js.exists():
        log(f"{js} exists")
        return
    view = View(DATA, ARM_SPEC[arm][0])
    arrays, flog = fit_unit(L8.Fitter(view, L8.load_rel_emb(decl)), arm, k, fold, log)
    L8.save_unit(arrays, flog, npz, js, L8.job_fields(t0))
    verify_inputs(decl)


# ── stage: read ──────────────────────────────────────────────────────────────


def band(point: float, interval, readable: bool) -> str:
    """readings.bands, the first that applies (level 8's thresholds)."""
    if not readable or point is None or not np.isfinite(point):
        return "NOT_READ"
    if interval[0] > 1:
        return "L9_ABOVE_GNN"
    if point >= 0.75 and interval[0] >= 0.50:
        return "L9_HIGH"
    if point <= 0.25 and interval[1] <= 0.50:
        return "L9_LOW"
    return "L9_MID"


def read_arm(Mx, T, G, dens, readable, W, mask=None) -> dict:
    """Level 8's read_arm, with this file's band."""
    entry = L8.read_arm(Mx, T, G, dens, readable, W, mask)
    entry["band"] = band(entry["rho_bar"]["point"], entry["rho_bar"]["ci"] or [0, 0], bool(readable))
    return entry


def grid_edges(units: dict) -> dict:
    """quantities.anchors.grid_edges: per protocol, the units whose kappa or eta sits at an end of its grid, and those
    whose kept round is the last."""
    out = {}
    vals = list(units.values())
    for tag, kap, et, last, suffix in (("wide", KAPPAS, ETAS, EM_ROUNDS, ""), ("level8", L8.KAPPAS, L8.ETAS, L8.EM_ROUNDS, "_l8")):
        k_end = [v[f"kappa{suffix}"] in (kap[0], kap[-1]) for v in vals]
        e_end = [v[f"eta{suffix}"] in (et[0], et[-1]) for v in vals]
        out[tag] = {"units": len(vals), "kappa_at_edge": int(sum(k_end)), "eta_at_edge": int(sum(e_end)),
                    "either_at_edge": int(sum(a or b for a, b in zip(k_end, e_end))),
                    "kept_last_round": int(sum(v[f"kept_round{suffix}"] == last for v in vals)),
                    "kappa_at_low_end": int(sum(v[f"kappa{suffix}"] == kap[0] for v in vals)),
                    "kappa_at_high_end": int(sum(v[f"kappa{suffix}"] == kap[-1] for v in vals)),
                    "eta_at_low_end": int(sum(v[f"eta{suffix}"] == et[0] for v in vals)),
                    "eta_at_high_end": int(sum(v[f"eta{suffix}"] == et[-1] for v in vals))}
    return out


def stage_read(decl: dict, log=print) -> dict:
    t0 = time.time()
    torch.set_num_threads(L8.FIT_THREADS)
    verify_inputs(decl)
    vs = views(DATA)
    base = vs["std"]
    check = read_json(DATA / "check.json")
    n = base.n_q
    ri = [METRIC_NAMES.index(m) for m in RETRIEVAL]
    qm = base.q_metrics
    T = qm[:, [FUNCS.index(f"twin{k}") for k in SEEDS]][:, :, ri]
    G = qm[:, [FUNCS.index(f"gnn{k}") for k in SEEDS]][:, :, ri]
    values, units, argmax, code = {}, {}, {}, {}
    for arm in ARMS:
        Mx = np.full((n, len(SEEDS), len(RETRIEVAL)), np.nan)
        M8 = np.full((n, len(SEEDS), len(RETRIEVAL)), np.nan)
        seen = np.zeros((n, len(SEEDS)), dtype=np.int64)
        units[arm], argmax[arm] = {}, np.full((n, len(SEEDS)), -3, dtype=np.int64)
        for k in SEEDS:
            for fold in range(FOLDS):
                npz, js = unit_paths(arm, k, fold)
                flog = read_json(js)
                if L0.sha256_file(npz) != flog["arrays_sha256"]:
                    raise SystemExit(f"{npz}: not the arrays its log records")
                with np.load(npz) as z:
                    q = z["q"]
                    if not np.array_equal(q, np.flatnonzero(base.q_fold == fold)):
                        raise SystemExit(f"{npz}: not fold {fold}'s queries")
                    Mx[q, k] = z["metrics"][:, ri]
                    M8[q, k] = z["metrics_l8"][:, ri]
                    argmax[arm][q, k] = z["argmax"]
                    seen[q, k] += 1
                units[arm][f"k{k}_f{fold}"] = {key: flog.get(key) for key in (
                    "kept_round", "kappa", "eta", "kept_round_l8", "kappa_l8", "eta_l8", "parameters", "walk_types", "fit_queries",
                    "inner_queries", "em_not_monotone_rounds")}
                units[arm][f"k{k}_f{fold}"]["seconds"] = flog["timing"]["seconds"]
                code[f"fit/{arm}/k{k}_f{fold}"] = flog["module_sha256"]
        if not (seen == 1).all():
            raise SystemExit(f"{arm}: a query is not scored exactly once per seed out of fold")
        values[arm], values[arm + NESTED] = Mx, M8
    chains = {qt: L8.true_chain(qt) for qt in base.meta["qtypes"]}
    for ref, fam in REFERENCES.items():
        v = vs[fam]
        Mo = np.zeros((n, len(SEEDS), len(RETRIEVAL)))
        for q in range(n):
            rs, _b = L8.r_star(v, q, chains[base.meta["qtypes"][base.q_qtype[q]]])
            bonus = np.zeros(int(v.q_pool_size[q]))
            bonus[rs] = L8.ORACLE_BONUS
            g, gt = v.gold_local(q), int(v.q_gold_total[q])
            for k in SEEDS:
                r = rank_metrics(v.z(q, k) + bonus, g, gt)
                Mo[q, k] = [r[m] for m in RETRIEVAL]
        values[ref] = Mo
    read_names = list(ARMS) + [a + NESTED for a in ARMS] + list(REFERENCES)
    W = L0.boot_weights(n)
    den_out, dens, readable = L8.denominators(T, G, W)
    arms_out = {a: read_arm(values[a], T, G, dens, readable, W) for a in read_names}
    contrasts = {}
    for c_name, (a, b) in CONTRASTS.items():
        if readable:
            contrasts[c_name] = {"of": f"rho_bar({a}) - rho_bar({b})",
                                 "point": arms_out[a]["rho_bar"]["point"] - arms_out[b]["rho_bar"]["point"],
                                 "ci": L0.ci(arms_out[a]["_boot"] - arms_out[b]["_boot"])}
        else:
            contrasts[c_name] = {"of": f"rho_bar({a}) - rho_bar({b})", "point": None, "ci": None}
    strata = {}
    for h in (1, 2, 3):
        mask = base.q_hop == h
        s_den, s_dens, s_read = L8.denominators(T, G, W, mask)
        strata[f"hop={h}"] = {"queries": int(mask.sum()), "denominators": s_den, "readable_metrics": s_read,
                              "arms": {a: {key: v for key, v in read_arm(values[a], T, G, s_dens, s_read, W, mask).items() if key != "_boot"}
                                       for a in read_names}}
    qt_names = [base.meta["qtypes"][i] for i in base.q_qtype]
    truth = [tuple(L8.chain_tokens(chains[qt])) for qt in qt_names]

    def agree(arm: str, sel=None) -> list[float]:
        qs = np.arange(n) if sel is None else np.flatnonzero(sel)
        return [float(np.mean([L8.token_sequence(int(argmax[arm][q, k])) == truth[q] for q in qs])) if qs.size else float("nan")
                for k in SEEDS]

    anchors = {"check": {key: check[key] for key in ("families", "nb_trim", "topic_entity", "gold_in_pool", "pool", "edges")}}
    anchors["agreement"] = {arm: {"per_k": agree(arm), "mean": float(np.mean(agree(arm))),
                                  "per_hop_mean": {f"hop={h}": float(np.mean(agree(arm, base.q_hop == h))) for h in (1, 2, 3)}}
                            for arm in ARMS}
    anchors["nmi_argmax_vs_qtype_k0"] = L8.nmi([L8.token_sequence(int(c)) for c in argmax[PRIMARY][:, 0]], qt_names)
    anchors["grid_edges"] = {arm: grid_edges(units[arm]) for arm in ARMS}
    flags = []
    orc = arms_out["NB-oracle"]["rho_bar"]
    if orc["ci"] is not None and orc["ci"][1] <= 0.50:
        flags.append("CEILING_LOW")
    rep_first = unit_paths(*REPEAT_UNIT)
    rep_again = unit_paths(*REPEAT_UNIT, DATA / "repeat")
    repeat = L8.compare_repeat(rep_first, rep_again)
    code["repeat"] = read_json(rep_again[1])["module_sha256"]
    if not repeat["bit_identical"]:
        flags.append("REPEAT_DIFFERS")
    not_mono = sorted(f"{a}/{u}" for a in ARMS for u, v in units[a].items() if v["em_not_monotone_rounds"])
    if not_mono:
        flags.append("EM_NOT_MONOTONE")
    edge = anchors["grid_edges"][PRIMARY]["wide"]
    if edge["either_at_edge"] * 2 > edge["units"]:
        flags.append("GRID_EDGE")
    reading = arms_out[PRIMARY]["band"]
    interp = {"L9_ABOVE_GNN": ["l9_above_gnn"], "L9_HIGH": ["l9_high"], "L9_MID": ["l9_mid"], "L9_LOW": ["l9_low"]}.get(reading, [])
    for c_name, (above, below) in INTERPRET_CONTRAST.items():
        ci_ = contrasts[c_name]["ci"]
        if ci_ is not None and ci_[0] > 0:
            interp.append(above)
        elif ci_ is not None and below is not None and ci_[1] < 0:
            interp.append(below)
    cg = contrasts["ceiling_gap"]["ci"]
    if anchors["agreement"][PRIMARY]["mean"] < 0.5 and cg is not None and cg[0] > 0:
        interp.append("chain_not_identified")
    for a in arms_out.values():
        a.pop("_boot", None)
    code["check"] = check["module_sha256"]
    code["meta"] = base.meta["module_sha256"]
    for s_name, s_rec in base.meta.get("shards", {}).items():
        code[f"score/{s_name}"] = s_rec["module_sha256"]
    out = {"stage": "read", "queries": n, "readable_metrics": readable, "denominators": den_out, "arms": arms_out,
           "primary": PRIMARY, "reading": reading, "flags": flags, "interpretation": interp, "contrasts": contrasts,
           "strata": strata, "anchors": anchors, "units": units, "em_not_monotone_units": not_mono, "repeat": repeat,
           "code": code, "check_sha256": L0.sha256_file(DATA / "check.json"), "meta_sha256": L0.sha256_file(DATA / "meta.json"),
           "resamples": L0.RESAMPLES, "boot_seed": L0.BOOT_SEED, **L8.job_fields(t0)}
    write_json(DATA / "read.json", out)
    verify_inputs(decl)
    log(f"read: {PRIMARY} {reading}, rho_bar {arms_out[PRIMARY]['rho_bar']['point']}, flags {flags}")
    return out


# ── stage: doc (laptop) ──────────────────────────────────────────────────────


f3, fci = L8.f3, L8.fci


def render_doc(rec: dict) -> str:
    rd, ck, mt = rec["read"], rec["check"], rec["meta"]
    an = rd["anchors"]
    fam = ck["families"]
    L = ["# MP-Approx level 9: non-backtracking typed walks and an exact residual, without message passing, on metaqa", "",
         f"Declaration: `configs/mp_approx_l9.yaml`. The script is `{SCRIPT_REL}`. The record is `outputs/mp_approx_l9/record.json`.",
         "", "## What was measured", "",
         ("On metaqa, level 8's non-message-passing typed-walk model was pushed at its weak points. The walks are "
          "non-backtracking (a walk never steps straight back to the node it came from). The type vector gains an exact "
          "learned residual per token sequence beside level 8's rotary text composition. The kappa and eta grid is wider, "
          "and EM runs 10 rounds. Every change has a paired contrast on the same queries, and level 8's own protocol is read "
          f"nested inside every unit as the @L8 arms. Every arm is cross-fitted on {rd['queries']} fresh metaqa V2_GATE "
          "queries, disjoint from levels 0 to 8. rho is level 8's quantity on a fresh population. It is not the "
          "within-U_q oracle rho of levels 0 to 7, and the two are never one quantity."), "",
         "## Reading", "",
         f"- The primary arm is {rd['primary']}, and its band is **{rd['reading']}**.",
         f"- Readable metrics: {', '.join(rd['readable_metrics']) or 'none'}.",
         f"- Flags: {', '.join(rd['flags']) or 'none'}.",
         f"- The interpretation map entries that apply: {', '.join(rd['interpretation']) or 'none'}.", "",
         "## Arms", "",
         ("rho_bar is the mean of rho over the readable metrics. Intervals are 95% bootstrap intervals over 1,000 query "
          "resamples. An @L8 arm is the same fit read under level 8's protocol (rounds 0 to 5, level 8's grid)."), "",
         "| arm | rho_bar | band | rho recall@5 | rho full_coverage@5 | rho hit@1 |", "|---|---|---|---|---|---|"]
    for a, v in rd["arms"].items():
        L.append(f"| {a} | {fci(v['rho_bar'])} | {v['band']} | " + " | ".join(fci(v["rho"][m]) for m in RETRIEVAL) + " |")
    L += ["", "The gap to the GNN is the mean over seeds and queries of M(arm) - M(G).", "",
          "| arm | recall@5 | full_coverage@5 | hit@1 |", "|---|---|---|---|"]
    for a, v in rd["arms"].items():
        L.append(f"| {a} | " + " | ".join(f"{fci(v['gap_to_gnn'][m])} {v['gap_to_gnn'][m]['flag'] or ''}".strip() for m in RETRIEVAL) + " |")
    L += ["", "Descriptive means over seeds and queries (the twin and the GNN are the stored values):", "",
          "| arm | recall@5 (arm / twin / GNN) | full_coverage@5 | hit@1 |", "|---|---|---|---|"]
    for a, v in rd["arms"].items():
        L.append(f"| {a} | " + " | ".join(f"{f3(v['mean'][m]['arm'])} / {f3(v['mean'][m]['twin'])} / {f3(v['mean'][m]['gnn'])}"
                                          for m in RETRIEVAL) + " |")
    L += ["", "## Denominators", "", "| metric | mean M(G) - M(T) | readable |", "|---|---|---|"]
    for m, v in rd["denominators"].items():
        L.append(f"| {m} | {f3(v['gap'])} [{f3(v['ci'][0])}, {f3(v['ci'][1])}] | {v['readable']} |")
    L += ["", "## Contrasts", "", "| contrast | of | paired difference |", "|---|---|---|"]
    for c, v in rd["contrasts"].items():
        L.append(f"| {c} | {v['of']} | {fci(v)} |")
    shown_arms = list(ARMS) + ["STD-text" + NESTED, PRIMARY + NESTED] + list(REFERENCES)
    L += ["", "## By hop", "", "rho_bar and band per hop, each hop read on its own readable metrics.", "",
          "| arm | " + " | ".join(f"{s} ({v['queries']} queries; {', '.join(v['readable_metrics']) or 'none'})"
                                  for s, v in rd["strata"].items()) + " |",
          "|---|" + "---|" * len(rd["strata"])]
    for a in shown_arms:
        L.append(f"| {a} | " + " | ".join(f"{fci(v['arms'][a]['rho_bar'])} {v['arms'][a]['band']}" for v in rd["strata"].values()) + " |")
    L += ["", "## Anchors (descriptive)", "",
          "| family | chain reaches a gold (either bucket) | swapped | R* recall | R* precision | b0 seed in R* | gold b0 seed in R* | types / query | entries / query |",
          "|---|---|---|---|---|---|---|---|---|"]
    for f in FAMILIES:
        a = fam[f]
        L.append(f"| {f} | {f3(a['direction_check']['declared_share'])} | {f3(a['direction_check']['swapped_share'])} | "
                 f"{f3(a['chain_fit']['recall']['all'])} | {f3(a['chain_fit']['precision']['all'])} | "
                 f"{f3(a['self_reach']['b0_seed_in_r_star']['all'])} | {f3(a['self_reach']['gold_b0_seed_in_r_star']['all'])} | "
                 f"{f3(a['sizes']['types_mean'])} | {f3(a['sizes']['entries_mean'])} |")
    trim = ck["nb_trim"]
    L += ["",
          f"- The non-backtracking rule removes {f3(trim['r_star_share_removed']['all'])} of the standard R* on average (by hop: "
          f"{', '.join(f3(trim['r_star_share_removed'][f'hop={h}']) for h in (1, 2, 3))}), and "
          f"{f3(trim['gold_share_removed']['all'])} of the in-pool golds (by hop: "
          f"{', '.join(f3(trim['gold_share_removed'][f'hop={h}']) for h in (1, 2, 3))}).",
          f"- Topic entity: in the pool {f3(ck['topic_entity']['in_pool']['all'])}, among the seeds {f3(ck['topic_entity']['in_seeds']['all'])}, "
          f"in b0 {f3(ck['topic_entity']['in_b0']['all'])}. Queries with an in-pool gold: {f3(ck['gold_in_pool']['all'])}.",
          "- The argmax type has the true chain's tokens on this share of queries (the mean over seeds; then hop 1, 2, 3):"]
    for arm, v in an["agreement"].items():
        L.append(f"  - {arm}: {f3(v['mean'])} ({', '.join(f3(v['per_hop_mean'][f'hop={h}']) for h in (1, 2, 3))})")
    L += [f"- NMI between {rd['primary']}'s argmax token sequence and the qtype (k = 0): {f3(an['nmi_argmax_vs_qtype_k0'])}.",
          "- Grid edges and kept rounds per arm, under the wide protocol / level 8's protocol:"]
    for arm, v in an["grid_edges"].items():
        w, l8 = v["wide"], v["level8"]
        L.append(f"  - {arm} ({w['units']} units): kappa at an end {w['kappa_at_edge']} / {l8['kappa_at_edge']} (low end "
                 f"{w['kappa_at_low_end']} / {l8['kappa_at_low_end']}), eta at an end {w['eta_at_edge']} / {l8['eta_at_edge']} "
                 f"(high end {w['eta_at_high_end']} / {l8['eta_at_high_end']}), the last round kept {w['kept_last_round']} / "
                 f"{l8['kept_last_round']}")
    L += ["", "## Checks", "",
          f"- Scoring integrity: {mt['mismatches']} mismatches against the stored per-query metrics on {mt['queries']} queries.",
          f"- Direction checks: std {f3(fam['std']['direction_check']['declared_share'])} against swapped "
          f"{f3(fam['std']['direction_check']['swapped_share'])}; nb {f3(fam['nb']['direction_check']['declared_share'])} against "
          f"swapped {f3(fam['nb']['direction_check']['swapped_share'])}.",
          f"- Repeat unit bit-identical: {rd['repeat']['bit_identical']}.",
          f"- Units whose fit-set marginal log-likelihood fell between rounds: {len(rd['em_not_monotone_units'])}.", "",
          "## What this does not say", "",
          ("Every arm is a cross-fitted measurement model on metaqa V2_GATE queries with gold labels. It is never a deployable "
           "or selected model, and it says nothing about another dataset. Nothing here enters QLS-U, the twin, a feature "
           "contract, M3, M4 or any selection. Level 8's numbers were measured on a different population, and the paired "
           "comparisons with level 8's procedure are the @L8 arms on this file's queries."), ""]
    return LF.join(L)


def stage_doc(log=print) -> None:
    """outputs.record and outputs.document: assembled from the fetched read.json, check.json and meta.json, without
    arithmetic."""
    rd, ck, mt = read_json(DATA / "read.json"), read_json(DATA / "check.json"), read_json(DATA / "meta.json")
    if rd["meta_sha256"] != L0.sha256_file(DATA / "meta.json") or rd["check_sha256"] != L0.sha256_file(DATA / "check.json"):
        raise SystemExit("read.json was not made from these check.json and meta.json")
    keep = ("queries", "population_queries", "available_per_hop", "per_hop", "chunks", "chunk_queries", "entries", "types", "qtypes",
            "mismatches", "held_rows_scored", "level0_rows_recomputed_equal", "placement", "seconds", "threads", "peak_rss_bytes",
            "arrays_sha256", "qids_sha256", "integrity")
    rec = {"phase": "MP_APPROX_L9", "declaration_lf_sha256": L0.lf_sha256(CONFIG), "read": rd, "check": ck,
           "meta": {k: mt.get(k) for k in keep},
           "sources_sha256": {"read.json": L0.sha256_file(DATA / "read.json"), "check.json": L0.sha256_file(DATA / "check.json"),
                              "meta.json": L0.sha256_file(DATA / "meta.json")}}
    write_json(RECORD, rec)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text(render_doc(rec), encoding="utf-8", newline=LF)
    log(f"wrote {shown(RECORD)} and {shown(DOC)}")


# ── stage: file (laptop) ─────────────────────────────────────────────────────


def code_problems(code: dict, commit: str) -> list[str]:
    """placement.identical_code: level 8's check (one set of sha256 values, equal to the files at the commit), with this
    script in it."""
    problems = L8.code_problems(code, commit)
    if not any(SCRIPT_REL in shas for shas in code.values()):
        problems.append(f"{SCRIPT_REL} is in no job's record")
    return problems


def append_block(key: str, block: dict, status_to: str | None = None) -> None:
    text = CONFIG.read_text(encoding="utf-8")
    decl = yaml.safe_load(text)
    if key in decl:
        raise SystemExit(f"{key} exists")
    if status_to is not None:
        if decl["status"] != "DECLARED_NOT_RUN":
            raise SystemExit(f"status is {decl['status']}, not DECLARED_NOT_RUN")
        text = text.replace("status: DECLARED_NOT_RUN", f"status: {status_to}", 1)
    dumped = yaml.safe_dump(L0.clean({key: block}), sort_keys=False, width=160, allow_unicode=True)
    CONFIG.write_text(text.rstrip(LF) + LF + LF + dumped, encoding="utf-8", newline=LF)
    if yaml.safe_load(CONFIG.read_text(encoding="utf-8"))[key] != L0.clean(block):
        raise SystemExit(f"{key}: the appended block does not read back identically")


def stage_file(date: str, commit: str, log=print, extra: dict | None = None) -> None:
    """run_record_mp_approx_l9_<date> after the code check; status DECLARED_NOT_RUN -> RUN."""
    rec = read_json(RECORD)
    rd = rec["read"]
    if rec["sources_sha256"]["read.json"] != L0.sha256_file(DATA / "read.json"):
        raise SystemExit("record.json was not made from the fetched read.json")
    code = dict(rd["code"])
    code["read"] = rd["module_sha256"]
    problems = code_problems(code, commit)
    if problems:
        hard_stop("identical_code: the host jobs did not run one committed set of files", problems=problems[:20])
    run = {"utc": L0.utc(), "status_moves": "DECLARED_NOT_RUN -> RUN", "terminal": "STOP_FOR_REVIEW", "code_commit": commit,
           "placement": "host CPU (rx env mpr-cpu), the verified mirror in memory; scoring at 6 threads, check, fits and read at 4",
           "reading": rd["reading"], "primary": rd["primary"], "primary_rho_bar": rd["arms"][rd["primary"]]["rho_bar"],
           "bands": {a: v["band"] for a, v in rd["arms"].items()},
           "rho_bar": {a: v["rho_bar"] for a, v in rd["arms"].items()},
           "readable_metrics": rd["readable_metrics"], "flags": rd["flags"], "interpretation": rd["interpretation"],
           "contrasts": {c: {"point": v["point"], "ci": v["ci"]} for c, v in rd["contrasts"].items()},
           "queries": rd["queries"], "scoring_mismatches": rec["meta"]["mismatches"], "held_rows_read": False, "test_rows_read": False,
           "checkpoints_updated": 0, "jobs_checked_for_identical_code": len(code), "record_sha256": L0.sha256_file(RECORD),
           "document": shown(DOC), "document_sha256": L0.lf_sha256(DOC)}
    if extra:
        run.update(extra)
    append_block(f"run_record_mp_approx_l9_{date}", run, status_to="RUN")
    log(f"filed run_record_mp_approx_l9_{date}; status RUN")


# ── main ─────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=("score", "assemble", "check", "fit", "repeat", "read", "doc", "file"))
    ap.add_argument("--host", action="store_true", help="on the host: the verified mirror in place of the package, in memory")
    ap.add_argument("--shard", default=None, help="score: i/n, the chunks ci with ci mod n = i; a run without it assembles")
    ap.add_argument("--arm", choices=ARMS, help="fit: the arm whose 15 units this job fits")
    ap.add_argument("--limit", type=int, default=None, help="score, smoke only: N queries spread over the population, written to --out")
    ap.add_argument("--out", type=Path, default=None, help="score, smoke only: a directory outside outputs/mp_approx_l9")
    ap.add_argument("--date", help="file: the run record's date, e.g. 2026_10_01")
    ap.add_argument("--commit", help="file: the commit every host job ran from")
    ap.add_argument("--extra", type=Path, default=None, help="file: a JSON object of fields added to the run record")
    args = ap.parse_args(argv)
    host_stages = ("score", "assemble", "check", "fit", "repeat", "read")
    if args.stage in host_stages and not args.host:
        ap.error(f"--stage {args.stage} runs on the host (--host), placement")
    if args.stage in ("doc", "file") and args.host:
        ap.error(f"--stage {args.stage} runs on the laptop")
    if args.shard is not None and args.stage != "score":
        ap.error("--shard is for --stage score")
    if args.stage == "fit" and args.arm is None:
        ap.error("--stage fit needs --arm")
    if (args.limit is None) != (args.out is None) or (args.limit is not None and args.stage != "score"):
        ap.error("--limit and --out go together, with --stage score (a smoke run)")
    decl = load_declaration()
    if args.limit is not None:
        out = args.out.resolve()
        if OUT.resolve() in (out, *out.parents):
            ap.error("a smoke run never writes under outputs/mp_approx_l9")
        HARD_STOP_DIR[0] = out
    route_stops()
    if args.host:
        L8.host_mode(decl, log_utc)
    if args.stage in ("score", "assemble"):
        if args.limit is not None:
            stage_score(decl, log_utc, None, args.limit, out / NAME)
        else:
            stage_score(decl, log_utc, L8.parse_shard(args.shard) if args.stage == "score" else None)
    elif args.stage == "check":
        stage_check(decl, log_utc)
    elif args.stage == "fit":
        stage_fit(decl, args.arm, log_utc)
    elif args.stage == "repeat":
        stage_repeat(decl, log_utc)
    elif args.stage == "read":
        stage_read(decl, log_utc)
    elif args.stage == "doc":
        stage_doc(log_utc)
    else:
        if not (args.date and args.commit):
            ap.error("--stage file needs --date and --commit")
        stage_file(args.date, args.commit, log_utc, read_json(args.extra) if args.extra else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
