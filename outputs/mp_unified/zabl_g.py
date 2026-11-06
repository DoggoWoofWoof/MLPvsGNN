"""ABL slot graphs (docs/ABL_LINK_FAMILY_ABLATION.md, sections 2 and 4): what fills the `structural` slot of an arm on a
dataset, and the walk's types on it. Pools and the models are not touched here.

    python outputs/mp_unified/zabl_g.py build DATASET [--host]       -> outputs/abl/g/<D>/SR.npz and SR.json
    python outputs/mp_unified/zabl_g.py types DATASET SLOT [--host]  -> outputs/abl/g/<D>/<SLOT>/{em/,score.npz,types.json}
    python outputs/mp_unified/zabl_g.py all DATASET [--host]         -> the union, then the types of S, R and SR
    python outputs/mp_unified/zabl_g.py --selftest

Slots: S the original structural edges the store ships with (graph "today" on the passage datasets, the KB store on the
KB datasets), R U2b's rule graph W2_S (the record's "chosen" graph on the passage datasets, ABL-U's build on the KB
datasets: outputs/abl/u/<D>/W2_S.npz), SR their union (undirected neighbours, rows sorted by node id).

Types are fit on each slot graph (section 4, check 8) by U2m (a)'s types stage, unchanged (EM centers on 200,000 pairs,
every entry's class); the record's own files serve where the slot is the record's graph (R on the passage datasets, S on
the KB). On the KB a slot that holds the KB's typed rows keeps them and each entry's block is its facts, empty for an
entry only R's graph supplies (the carried walk, RWe, weights an entry by its class, not by its facts). The passage
score classes exist on the rule graph's pairs only (U2k scores a mention link or a sibling pair and refuses any other
neighbour), so a slot's class for a pair is the record's class where the rule graph holds the pair and the null class
elsewhere (S alone: all null); the equal-count cuts over the scored entries are R's in every slot that holds R.
"""
from __future__ import annotations

import os
import sys

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
sys.dont_write_bytecode = True

import argparse
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (ROOT / "scripts", ROOT / "src", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

DECLARED = "docs/ABL_LINK_FAMILY_ABLATION.md"
OUT = ROOT / "outputs" / "abl" / "g"
U_OUT = ROOT / "outputs" / "abl" / "u"
PASSAGE = ("musique", "squad", "hotpotqa", "2wiki")
KB = ("webqsp", "metaqa")
SLOTS = ("S", "R", "SR")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def rel(p):
    return str(Path(p).resolve().relative_to(ROOT)).replace("\\", "/")


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def sha_arr(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def write_json(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1), encoding="utf-8")
    os.replace(tmp, p)


def save_npz(p, **arrs):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.stem + ".tmp.npz")
    np.savez(tmp, **arrs)
    os.replace(tmp, p)
    return sha_file(p)


# ── the arithmetic (no dataset) ────────────────────────────────────────────────


def entry_keys(n, ip, col):
    """Each stored entry u -> v as the int64 key u * n + v, in CSR order."""
    return np.repeat(np.arange(n, dtype=np.int64), np.diff(ip)) * np.int64(n) + np.asarray(col, np.int64)


def union_csr(n, ip1, col1, ip2, col2):
    """The union of two undirected neighbour lists over the same n nodes, rows sorted by node id."""
    k = np.union1d(entry_keys(n, ip1, col1), entry_keys(n, ip2, col2))
    a, b = k // n, k % n
    ip = np.zeros(n + 1, np.int64)
    np.cumsum(np.bincount(a, minlength=n), out=ip[1:])
    return ip, b.astype(np.int32)


def lookup_classes(n, ip, col, ref_ip, ref_col, ref_cls, null):
    """Per entry of (ip, col): the class of the same pair in the reference graph, or `null` when it has none."""
    rk = entry_keys(n, ref_ip, ref_col)
    kk = entry_keys(n, ip, col)
    pos = np.searchsorted(rk, kk)
    pos = np.minimum(pos, max(rk.size - 1, 0))
    hit = (rk[pos] == kk) if rk.size else np.zeros(kk.size, bool)
    out = np.full(kk.size, null, ref_cls.dtype)
    out[hit] = ref_cls[pos[hit]]
    return out, int(hit.sum())


def blocks_allow_empty(ip, col, typed):
    """(start, length) of each entry u -> v's facts in u's typed row, as U2m (a)'s blocks, except that an entry with no
    fact (one only R's graph supplies) has length 0. The blocks still cover every typed entry once, in order."""
    tip, tcol = typed[0], typed[1]
    n = ip.size - 1
    if tip.size != n + 1:
        raise SystemExit("the typed rows are not the store's nodes")
    ukey = entry_keys(n, ip, col)
    tkey = entry_keys(n, tip, tcol)
    if ukey.size > 1 and (np.diff(ukey) <= 0).any():
        raise SystemExit("the collapsed rows are not sorted by neighbour without repeats")
    if tkey.size > 1 and (np.diff(tkey) < 0).any():
        raise SystemExit("the typed rows are not sorted by neighbour")
    bst = np.searchsorted(tkey, ukey, side="left").astype(np.int64)
    bln = (np.searchsorted(tkey, ukey, side="right") - bst).astype(np.int32)
    if int(bln.sum()) != tkey.size:
        raise SystemExit("a typed pair is missing from the slot's graph: the blocks do not cover every typed entry")
    return bst, bln


# ── the slot graphs ───────────────────────────────────────────────────────────


def open_record(name):
    """(AT, ds, desc, ip, col, typed, nodes): U2m (a)'s opening of the record's structural family for the dataset."""
    import zu2m_a_types as AT
    ds, desc, ip, col, typed, nodes = AT.open_structural(name)
    return (AT, ds, desc, ip, col, typed, nodes)


def graph_of(name, slot, rec):
    """(ip, col, typed or None, description) of a slot on a dataset; `rec` is open_record's tuple. S on a passage
    dataset and R on a KB dataset are read from their files; the record's own graph is the other one."""
    import zu2c as U2C
    _AT, _ds, desc, ip, col, typed, _nodes = rec
    n = ip.size - 1
    passage = name in PASSAGE
    if slot == "S":
        if passage:
            sip, scol, sdesc = U2C.graph_csr(name, "today", n)
            return np.asarray(sip, np.int64), np.asarray(scol, np.int32), None, sdesc
        return ip, col, typed, desc
    if slot == "R":
        if passage:
            return ip, col, None, desc
        p = U_OUT / name / "W2_S.npz"
        bj = json.loads((U_OUT / name / "build.json").read_text(encoding="utf-8"))
        if sha_file(p) != bj["npz_sha256"]:
            raise SystemExit(f"{p}: not the file ABL-U's build record names")
        with np.load(p) as z:
            rip, rcol = z["indptr"].astype(np.int64), z["col"].astype(np.int32)
        if rip.size != n + 1:
            raise SystemExit(f"{p}: {rip.size - 1} nodes, not {n}")
        return rip, rcol, None, {"graph": "W2_S", "npz": bj["npz"], "npz_sha256": bj["npz_sha256"], "abl": "ABL-U"}
    if slot == "SR":
        p = OUT / name / "SR.npz"
        bj = json.loads((OUT / name / "SR.json").read_text(encoding="utf-8"))
        if sha_file(p) != bj["npz_sha256"]:
            raise SystemExit(f"{p}: not the file SR.json names")
        with np.load(p) as z:
            uip, ucol = z["indptr"].astype(np.int64), z["col"].astype(np.int32)
        return uip, ucol, (typed if not passage else None), {"graph": "SR", "npz": bj["npz"],
                                                             "npz_sha256": bj["npz_sha256"], "parts": bj["parts"]}
    raise SystemExit(f"slot {slot}: one of {SLOTS}")


def build_stage(name, rec=None):
    rec = rec or open_record(name)
    n = rec[3].size - 1
    S = graph_of(name, "S", rec)
    R = graph_of(name, "R", rec)
    t0 = time.time()
    uip, ucol = union_csr(n, S[0], S[1], R[0], R[1])
    rng = np.random.default_rng(0)
    for u in rng.choice(n, min(2000, n), replace=False).tolist():     # check 6, sampled rows; the keys are checked whole
        want = np.union1d(S[1][S[0][u]:S[0][u + 1]], R[1][R[0][u]:R[0][u + 1]])
        if not np.array_equal(want, ucol[uip[u]:uip[u + 1]]):
            raise SystemExit(f"{name}: row {u} of the union is not the union of the rows")
    if uip[-1] != np.union1d(entry_keys(n, S[0], S[1]), entry_keys(n, R[0], R[1])).size:
        raise SystemExit(f"{name}: the union's entries are not the union of the keys")
    od = OUT / name
    sha = save_npz(od / "SR.npz", indptr=uip, col=ucol)
    parts = {"S": {"entries": int(S[1].size), "desc": S[3]}, "R": {"entries": int(R[1].size), "desc": R[3]}}
    write_json(od / "SR.json", {"declared_in": DECLARED, "stage": "ABL slot graphs", "dataset": name, "slot": "SR",
                                "nodes": n, "entries": int(ucol.size), "parts": parts,
                                "entries_in_both": int(S[1].size + R[1].size - ucol.size),
                                "npz": rel(od / "SR.npz"), "npz_sha256": sha,
                                "seconds": round(time.time() - t0, 1)})
    log(f"{name}: S {S[1].size:,} + R {R[1].size:,} entries -> SR {ucol.size:,} "
        f"({S[1].size + R[1].size - ucol.size:,} in both)")
    return 0


# ── the types ─────────────────────────────────────────────────────────────────


def fit_em(rec, name, slot, ip, col, typed, sdesc):
    """U2m (a)'s types stage, unchanged, on the slot's graph: the EM fit on 200,000 of its pairs, every entry's class,
    and on a typed slot the facts' blocks (an entry with no fact, one only R's graph supplies, has an empty block). Its
    outputs go to outputs/abl/g/<D>/<slot>/em/ instead of U2m's folder."""
    AT, ds, _desc, _rip, _rcol, _rtyped, nodes = rec
    keep = (AT.open_structural, AT.blocks, AT.OUT)

    def open_slot(name_):
        if name_ != name:
            raise SystemExit(f"ABL slot types: asked for {name_}, not {name}")
        return ds, sdesc, ip, col, typed, nodes

    try:
        AT.open_structural, AT.OUT = open_slot, OUT / name / slot / "em"
        if typed is not None:
            AT.blocks = blocks_allow_empty
        AT.types_stage(name)
    finally:
        AT.open_structural, AT.blocks, AT.OUT = keep


def types_stage(name, slot, rec=None):
    import zu2p as P
    rec = rec or open_record(name)
    AT = rec[0]
    rdesc, rip, rcol = rec[2], rec[3], rec[4]
    n = rip.size - 1
    ip, col, typed, sdesc = graph_of(name, slot, rec)
    t0 = time.time()
    own = (slot == "R" and name in PASSAGE) or (slot == "S" and name in KB)
    if not own:
        fit_em(rec, name, slot, ip, col, typed, sdesc)
    od = OUT / name
    out = {"declared_in": DECLARED, "stage": "ABL slot types", "dataset": name, "slot": slot, "graph": sdesc,
           "entries": int(col.size), "typed": typed is not None, "own_record_graph": own,
           "em_dir": rel(AT.OUT) if own else rel(od / slot / "em")}
    if name in PASSAGE:
        cls_s, zbar, srec = P.load_types(name, rdesc)
        cls_s = np.asarray(cls_s)
        if slot == "R":
            cls_slot, hit = cls_s, int(cls_s.size)
        elif slot == "S":
            cls_slot, hit = np.full(col.size, P.NULL, cls_s.dtype), 0
        else:
            cls_slot, hit = lookup_classes(n, ip, col, rip, rcol, cls_s, cls_s.dtype.type(P.NULL))
        sha_sc = save_npz(od / slot / "score.npz", cls=cls_slot, zbar=np.asarray(zbar))
        out.update(score_record_file_sha256=srec["file_sha256"], score_cls_sha256=sha_arr(cls_slot),
                   score_npz=rel(od / slot / "score.npz"), score_file_sha256=sha_sc,
                   score_entries_with_a_class=int(hit), score_null=int((cls_slot == P.NULL).sum()))
    out["seconds"] = round(time.time() - t0, 1)
    write_json(od / slot / "types.json", out)
    log(f"{name}/{slot}: {col.size:,} entries, typed {typed is not None}, own record types {own}"
        + (f", score null {out['score_null']:,}" if name in PASSAGE else "") + f" ({out['seconds']} s)")
    return 0


def load_slot(name, slot, rec=None):
    """What an arm swaps in for a slot on a dataset: SimpleNamespace(ip, col, typed, desc, at_load, p_load, info), every
    file checked against its record. at_load(name, desc) and p_load(name, desc) have U2m (a)'s and U2p's load_types
    signatures and results, so an arm patches them in. `slot` "" is the empty slot (no entries)."""
    import zu2p as P
    rec = rec or open_record(name)
    AT, rdesc, rip = rec[0], rec[2], rec[3]
    n = rip.size - 1
    orig_at = AT.load_types
    if slot == "":
        mu = np.asarray(orig_at(name, rdesc)[0]["mu"], np.float64)
        e8 = np.zeros(0, np.int8)
        edesc = {"graph": "empty", "nodes": int(n)}
        sha = sha_arr(mu)

        def at_load(name_, desc=None):
            if name_ != name or (desc is not None and desc != edesc):
                raise SystemExit(f"ABL: the empty slot's types asked for {name_} on {desc}")
            return {"mu": mu, "cls": e8}, {"file_sha256": sha, "null": int(mu.shape[0]), "entries": 0, "typed": False,
                                           "graph_detail": edesc}

        def p_load(d, desc=None):
            if d != name or (desc is not None and desc != edesc):
                raise SystemExit(f"ABL: the empty slot's score types asked for {d} on {desc}")
            return e8.copy(), np.zeros(P.NT), {"file_sha256": sha_arr(np.zeros(P.NT)), "entries": 0}

        return SimpleNamespace(ip=np.zeros(n + 1, np.int64), col=np.zeros(0, np.int32), typed=None, desc=edesc,
                               at_load=at_load, p_load=p_load if name in PASSAGE else None,
                               info={"slot": "", "entries": 0})
    ip, col, typed, sdesc = graph_of(name, slot, rec)
    tj = json.loads((OUT / name / slot / "types.json").read_text(encoding="utf-8"))
    if tj["graph"] != sdesc or tj["entries"] != int(col.size) or tj["typed"] != (typed is not None):
        raise SystemExit(f"{name}/{slot}: the types were fit for another graph")
    em_dir = ROOT / tj["em_dir"]

    def at_load(name_, desc=None):
        if name_ != name or (desc is not None and desc != sdesc):
            raise SystemExit(f"ABL: the types of {name}/{slot} asked for {name_} on {desc}")
        keep = AT.OUT
        try:
            AT.OUT = em_dir
            arr, arec = orig_at(name, sdesc)
        finally:
            AT.OUT = keep
        if int(arec["entries"]) != int(col.size) or int(np.asarray(arr["cls"]).size) != int(col.size):
            raise SystemExit(f"ABL: the EM types of {name}/{slot} do not fit its graph")
        return arr, arec

    p_load = None
    if name in PASSAGE:
        ps = ROOT / tj["score_npz"]
        if sha_file(ps) != tj["score_file_sha256"]:
            raise SystemExit(f"{ps}: not the file {name}/{slot}/types.json names")
        with np.load(ps) as z:
            sc = (np.asarray(z["cls"], np.int8), np.asarray(z["zbar"]))
        if sha_arr(sc[0]) != tj["score_cls_sha256"] or sc[0].size != col.size:
            raise SystemExit(f"{ps}: the score classes are not the record's description")

        def p_load(d, desc=None):
            if d != name or (desc is not None and desc != sdesc):
                raise SystemExit(f"ABL: the score types of {name}/{slot} asked for {d} on {desc}")
            return sc[0], sc[1], {"file_sha256": tj["score_file_sha256"], "entries": int(col.size),
                                  "graph_detail": sdesc}

    return SimpleNamespace(ip=ip, col=col, typed=typed, desc=sdesc, at_load=at_load, p_load=p_load,
                           info={"slot": slot, "entries": int(col.size), "graph": sdesc,
                                 "types_record": rel(OUT / name / slot / "types.json"),
                                 "score_file_sha256": tj.get("score_file_sha256")})


# ── self-test ───────────────────────────────────────────────────────────────


def _selftest():
    rng = np.random.default_rng(1)
    n = 40

    def rand_graph():
        m = int(rng.integers(0, 120))
        a, b = rng.integers(0, n, m), rng.integers(0, n, m)
        k = a != b
        a, b = a[k], b[k]
        key = np.unique(np.concatenate([a * n + b, b * n + a]))
        ip = np.zeros(n + 1, np.int64)
        np.cumsum(np.bincount(key // n, minlength=n), out=ip[1:])
        return ip, (key % n).astype(np.int32)

    for case in range(20):
        (ip1, c1), (ip2, c2) = rand_graph(), rand_graph()
        uip, ucol = union_csr(n, ip1, c1, ip2, c2)
        for u in range(n):
            want = sorted(set(c1[ip1[u]:ip1[u + 1]].tolist()) | set(c2[ip2[u]:ip2[u + 1]].tolist()))
            if ucol[uip[u]:uip[u + 1]].tolist() != want:
                raise SystemExit(f"self-test case {case}: row {u} of the union")
        cls = rng.integers(0, 64, c2.size).astype(np.int8)
        got, hit = lookup_classes(n, uip, ucol, ip2, c2, cls, np.int8(64))
        k2 = {int(a): int(b) for a, b in zip(entry_keys(n, ip2, c2), cls)}
        want = np.array([k2.get(int(k), 64) for k in entry_keys(n, uip, ucol)], np.int8)
        if not np.array_equal(got, want) or hit != c2.size:
            raise SystemExit(f"self-test case {case}: the class lookup")
        # typed rows over the first graph: 1 to 3 facts per pair; the union adds pairs with no fact
        reps = rng.integers(1, 4, c1.size)
        tcol = np.repeat(c1, reps)
        tip = np.zeros(n + 1, np.int64)
        np.cumsum(np.bincount(np.repeat(entry_keys(n, ip1, c1) // n, reps), minlength=n), out=tip[1:])
        _bst, bln = blocks_allow_empty(uip, ucol, (tip, tcol))
        in1 = np.isin(entry_keys(n, uip, ucol), entry_keys(n, ip1, c1))
        if (bln[in1] < 1).any() or (bln[~in1] != 0).any() or int(bln.sum()) != tcol.size:
            raise SystemExit(f"self-test case {case}: the blocks")
        _fb, full_len = blocks_allow_empty(ip1, c1, (tip, tcol))
        if not np.array_equal(full_len, reps):
            raise SystemExit(f"self-test case {case}: the blocks of the typed graph itself")
        if np.setdiff1d(entry_keys(n, ip1, c1), entry_keys(n, ip2, c2)).size:
            try:
                blocks_allow_empty(ip2, c2, (tip, tcol))
            except SystemExit:
                pass
            else:
                raise SystemExit(f"self-test case {case}: a typed pair outside the slot passed")
    log("zabl_g self-test: the union equals a brute force on 20 random graph pairs; the class lookup gives the "
        "reference's class and the null elsewhere; the blocks give every fact once, empty for a pair with no fact, and "
        "refuse a slot that lacks a typed pair")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", nargs="?", choices=("build", "types", "all"))
    ap.add_argument("dataset", nargs="?")
    ap.add_argument("slot", nargs="?", choices=SLOTS)
    ap.add_argument("--host", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return _selftest()
    if a.dataset not in PASSAGE + KB:
        ap.error("a dataset of the six")
    if a.cmd == "build":
        return build_stage(a.dataset)
    if a.cmd == "types" and a.slot:
        return types_stage(a.dataset, a.slot)
    if a.cmd == "all":
        rec = open_record(a.dataset)
        build_stage(a.dataset, rec)
        for slot in SLOTS:
            types_stage(a.dataset, slot, rec)
        return 0
    ap.error("build DATASET, types DATASET SLOT, all DATASET, or --selftest")


if __name__ == "__main__":
    sys.exit(main())
