"""PL1 step S5 (docs/PL1_POOLS_ON_UL2_LAYOUT.md sections 4, 6, 7 and 8): the grade that reads the pools PL1 builds.

    python outputs/mp_unified/zpl1g.py selftest
    python outputs/mp_unified/zpl1g.py grade  --ds squad --version RK [--host] [--record-dir DIR] [--threads 8]
    python outputs/mp_unified/zpl1g.py report                    # the nine arms' grade.json -> outputs/pl1/report.{json,md}

One (dataset, version) per `grade` call; the SAME code and constants for every dataset and both versions (section 2a). Every grade
records one hash of the constants and the code path (`code_hash`); `report` refuses to file when two arms' hashes differ.

Reads (never writes anywhere but outputs/pl1/<d>/<v>/grade.json and outputs/pl1/report.{json,md}):
  pools   outputs/pl1/<d>/<v>/pools/parts/<d>__s1eval__<i>of<k>.npz (+ .json head), zpko's own part format
  places  outputs/pl1/<d>/<v>/places/<d>__s1eval__<i>of<k>.npz: per question ids, keep, kt_len, g_ptr; per gold g_node, pos, in_I, one_link
  golds   outputs/enc/e0/gold/<d>.npz (ENC E0's gold cache), first stage outputs/enc/e0/e3v/<d>.npz and the served SPLADE list
  record  the pools of record's parts (PKO; ABL's SRK for the passage datasets' SRK arm), read by the same part reader
  labels  the layout's pairs with their rule labels (zul2p's families + zul2c's carried.npz), checked against zpl1m's layout.json

Measures (section 4):
  M1  COV's coverage and rows (covdiag.carve_measures, diff_block, ratio_ci, versus) of the new pools beside the pools of record, paired
      by question; MultiHop-RAG (no record) beside the e3v first stage's own order at the same rows.
  M2  gold pairs GP (zul1.gp_slots) and MuSiQue's chain slots CH by hops: the share of slots with a layout link, with both ends in the
      pool, and with a link inside the pool graph. A dataset with no pair stands on M3 alone (SQuAD).
  M3  the reach diagnosis from the places file: IN_B / KEPT / BEYOND / OUTSIDE of the golds outside the first stage's top 5, per
      dataset, per hop (MuSiQue) and per label of the layout pairs that join the gold to a B node ('deeper' when there is none).
  M4  AP4's ledger (zap4.unit_block, import only) on the pools: anchors = the golds in the top 5 ('found') and the first row ('top1'),
      candidates = their linked pool rows outside the top 5; and the same ledger on the pairs from B into the expansion rows, whose
      `reach` is the share of KEPT golds each label placed.
Then the call (section 6) and the section 7 inputs (amendment 2d: arms freeze one by one, on evidence, with no cap on rounds). Per arm grade.json
holds the residual by label, the M4 reading of the pairs that placed KEPT golds, `candidates` (layout labels with POINTS in an M4 read that
hold residual golds; a list of names, no decision) and `open_flag` (OUTSIDE share > 5%: a flag, not a trigger). report.{json,md} count how
many of the nine arms have an empty candidate list. No number is chosen here: constants are fixed in this file before any pool exists.
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import zul2c as L2C  # noqa: E402  (first: imports zul2u -> zul2 -> zmhb -> zenc, so the BLAS threads come from --threads before numpy loads)

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import covdiag as CV  # noqa: E402
import zap4 as AP  # noqa: E402
import zenc as E  # noqa: E402
import zpko  # noqa: E402
import zul1 as UL  # noqa: E402

L2 = L2C.L2
ROOT = HERE.parents[1]
OUT = ROOT / "outputs" / "pl1"
DECLARED = "docs/PL1_POOLS_ON_UL2_LAYOUT.md"
MHR = "multihoprag"
SIX = UL.SIX
KB = ("metaqa", "webqsp")
DATASETS = SIX + (MHR,)
ARMS = (("musique", "RK"), ("squad", "RK"), ("2wiki", "RK"), ("2wiki", "SRK"), ("hotpotqa", "RK"), ("hotpotqa", "SRK"),
        ("metaqa", "SRK"), ("webqsp", "SRK"), (MHR, "RK"))                          # section 7's nine arms

# ── constants (fixed here, before any pool exists; every dataset and version reads these) ──────────────────────────────
TOP = AP.TOP                                   # the first stage's top 5: M3's denominator, M4's anchors (AP4's TOP)
BOOT, SEED = CV.BOOT, CV.SEED                  # COV's paired bootstrap: 2000 resamples, seed 0
OUTSIDE_FLAG = 0.05                            # section 4/7: an OUTSIDE share above this is flagged, not a trigger (amendment 2d)
MIN_T = 20                                     # UL2P's: fewer targets than this and a label's lift is reported as thin
HOPS = (2, 3, 4)                               # MuSiQue's hops
CLASSES = ("IN_B", "KEPT", "BEYOND", "OUTSIDE")
BASE_LABELS = ("R.mention", "R.name_comention", "R.title_comention", "K.knn")        # zul2p.BASE_BITS
CALL_NAME = {"MORE": "ABOVE", "SAME": "LEVEL", "LESS": "BELOW"}
CONSTANTS = {"top": TOP, "boot": BOOT, "seed": SEED, "outside_flag": OUTSIDE_FLAG, "min_targets": MIN_T, "hops": list(HOPS),
             "classes": list(CLASSES), "base_labels": list(BASE_LABELS), "kb_golds": list(UL.KB_GOLDS), "arms": [list(a) for a in ARMS],
             "rrf_c": int(E.RRF_C), "rank1": "position in the e3v dense list (proxy: pool rows carry no RRF score)",
             "adopt": "no arm BELOW on gold recall; every arm whose rows-ratio upper end is above 1.00 is ABOVE on gold recall"}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] zpl1g: {msg}", flush=True)


def stop(msg):
    raise SystemExit(f"ZPL1G: {msg}")


def rel(p):
    try:
        return str(Path(p).resolve().relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(p)


def sha_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def code_hash():
    """One hash of the constants and the code path the grade runs: this file and the modules whose functions it calls, LF-normalised."""
    files = {}
    for name in ("zpl1g.py", "covdiag.py", "zap4.py", "zul1.py", "zenc.py", "zpko.py"):
        files[name] = hashlib.sha256((HERE / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    blob = json.dumps({"constants": CONSTANTS, "files": files}, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest(), files


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def wjson(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(jsonable(obj), indent=1, sort_keys=True), encoding="utf-8", newline="\n")
    os.replace(tmp, path)


# ── the label table: the layout's pairs, each with the labels of the rules that made it ──────────────────────────────

class LabelGraph:
    """Sorted unique pair keys a*n+b (a < b) with a bit set of labels per pair; the neighbours of a node are read in both directions."""

    def __init__(self, n, keys, bits, names):
        keys, bits = np.asarray(keys, np.int64), np.asarray(bits, np.int64)
        if keys.size > 1 and np.any(np.diff(keys) <= 0):
            stop("label keys are not sorted unique")
        if keys.size and ((keys // n) >= (keys % n)).any():
            stop("a label key is not a < b")
        if len(names) > 60:
            stop("more labels than bits")
        self.n, self.names = int(n), tuple(names)
        self.k, self.b = keys, bits
        rk = (keys % n) * n + keys // n
        o = np.argsort(rk, kind="stable")
        self.rk, self.rb = rk[o], bits[o]

    def nbrs(self, x):
        """(node, bits) of every pair that has x as an end."""
        n, x = self.n, int(x)
        lo, hi = np.searchsorted(self.k, [x * n, (x + 1) * n])
        lo2, hi2 = np.searchsorted(self.rk, [x * n, (x + 1) * n])
        return (np.concatenate((self.k[lo:hi] % n, self.rk[lo2:hi2] % n)).astype(np.int64),
                np.concatenate((self.b[lo:hi], self.rb[lo2:hi2])))

    def set_name(self, bits):
        return "+".join(self.names[i] for i in range(len(self.names)) if int(bits) >> i & 1) or "deeper"


def label_names(version, forms):
    """S (SRK only), the four base labels, then each carried form's label (zul2c.LABEL)."""
    return (("S",) if version == "SRK" else ()) + BASE_LABELS + tuple(L2C.LABEL.get(f, f) for f in forms)


def label_parts(version, F, carried_keys=None, carried_code=None, forms=()):
    """The (name, sorted unique keys) of every label of one version, from the families as zul2p.base_units reads them: S, W2_S
    (R.mention), CN_all (R.name_comention), F0T minus CN_all (R.title_comention), K_e3v (K.knn), and each carried form's kept pairs."""
    parts = []
    if version == "SRK":
        parts.append(("S", np.asarray(F["S"], np.int64)))
    cn = np.asarray(F["CN_all"], np.int64)
    parts += [("R.mention", np.asarray(F["W2_S"], np.int64)), ("R.name_comention", cn),
              ("R.title_comention", np.setdiff1d(np.asarray(F["F0T"], np.int64), cn, assume_unique=True)),
              ("K.knn", np.asarray(F["Ke"], np.int64))]
    for i, f in enumerate(forms):
        sel = ((np.asarray(carried_code) >> np.uint8(i)) & 1) == 1
        parts.append((L2C.LABEL.get(f, f), np.asarray(carried_keys, np.int64)[sel]))
    return parts


def label_graph(n, names_keys):
    """LabelGraph of [(label, sorted unique keys)]: one bit per entry; two forms with one label name share its bit."""
    names = []
    for nm, _k in names_keys:
        if nm not in names:
            names.append(nm)
    allk = np.unique(np.concatenate([k for _n, k in names_keys])) if names_keys else np.zeros(0, np.int64)
    bits = np.zeros(allk.size, np.int64)
    for nm, k in names_keys:
        bits[np.searchsorted(allk, k)] |= 1 << names.index(nm)
    return LabelGraph(n, allk, bits, names)


# ── part readers ─────────────────────────────────────────────────────────────

def find_parts(dirp, d):
    """The complete set of one dataset's s1eval parts in dirp: [(i, k, npz path)] in order, or [] when there is none."""
    found = []
    for p in sorted(Path(dirp).glob(f"{d}__s1eval__*of*.npz")):
        m = re.fullmatch(rf"{re.escape(d)}__s1eval__(\d+)of(\d+)\.npz", p.name)
        if m:
            found.append((int(m.group(1)), int(m.group(2)), p))
    if not found:
        return []
    ks = {k for _i, k, _p in found}
    if len(ks) != 1 or sorted(i for i, _k, _p in found) != list(range(next(iter(ks)))):
        stop(f"{rel(dirp)}: the {d} parts are not one complete set ({[(i, k) for i, k, _ in found]})")
    return sorted(found)


def read_part(npz):
    """One pools part: ids, per question B (sorted), the expansion rows in kt_cos order, the sorted pool, the json head."""
    npz = Path(npz)
    head_p = npz.with_name(npz.stem + ".json")
    head = json.loads(head_p.read_text(encoding="utf-8")) if head_p.is_file() else None
    if head is not None and head.get("file_sha256") and sha_of(npz) != head["file_sha256"]:
        stop(f"{rel(npz)}: the file's sha256 is not its record's")
    with np.load(npz) as z:
        filed = {k: z[k] for k in z.files}
    zpko.check_filed(filed)
    ids = [str(x) for x in filed["ids"]]
    bp, bn, ep, en = filed["bs_ptr"], filed["bs_nodes"], filed["ptr"], filed["nodes"]
    B = [np.asarray(bn[bp[i]:bp[i + 1]], np.int64) for i in range(len(ids))]
    exp = [np.asarray(en[ep[i]:ep[i + 1]], np.int64) for i in range(len(ids))]
    pools = zpko.pools_of(filed)
    sizes = np.asarray(zpko.pool_sizes(filed), np.int64)
    if len(pools) != len(ids) or any(p.size != int(s) for p, s in zip(pools, sizes)):
        stop(f"{rel(npz)}: a pool's rows differ from pool_sizes")
    return {"ids": ids, "B": B, "exp": exp, "pools": [np.asarray(p, np.int64) for p in pools], "sizes": sizes, "head": head,
            "file": rel(npz), "sha256": sha_of(npz), "count": np.asarray(filed["count"], np.int64)}


def read_parts(dirp, d):
    """All parts of a dataset, questions in part order: the same dict, with ids/B/exp/pools/sizes concatenated."""
    ps = find_parts(dirp, d)
    if not ps:
        return None
    out = {"ids": [], "B": [], "exp": [], "pools": [], "sizes": [], "parts": []}
    for i, k, p in ps:
        r = read_part(p)
        for key in ("ids", "B", "exp", "pools"):
            out[key] += r[key]
        out["sizes"].append(r["sizes"])
        out["parts"].append({"i": i, "k": k, "file": r["file"], "sha256": r["sha256"], "questions": len(r["ids"]),
                             "rows": int(r["sizes"].sum()), "head_rows": None if r["head"] is None else r["head"].get("rows"),
                             "head_questions": None if r["head"] is None else r["head"].get("questions")})
    out["sizes"] = np.concatenate(out["sizes"])
    return out


def read_places(dirp, d, k_want=None):
    """All places parts of a dataset concatenated in part order (g_ptr re-based)."""
    ps = find_parts(dirp, d)
    if not ps:
        return None
    cols = {"ids": [], "keep": [], "kt_len": [], "g_node": [], "pos": [], "in_I": [], "one_link": []}
    ptr, base, files = [np.zeros(1, np.int64)], 0, []
    for i, k, p in ps:
        with np.load(p, allow_pickle=False) as z:
            a = {x: z[x] for x in z.files}
        if a["g_ptr"][-1] != a["g_node"].size or a["pos"].size != a["g_node"].size or a["ids"].size + 1 != a["g_ptr"].size:
            stop(f"{rel(p)}: the places file is not self-consistent")
        for key in cols:
            cols[key].append(a[key])
        ptr.append(a["g_ptr"][1:] + base)
        base += int(a["g_ptr"][-1])
        files.append({"file": rel(p), "sha256": sha_of(p), "questions": int(a["ids"].size), "golds": int(a["g_node"].size)})
    out = {key: np.concatenate(v) for key, v in cols.items()}
    out["ids"] = [str(x) for x in out["ids"]]
    out["g_ptr"] = np.concatenate(ptr)
    out["files"] = files
    return out


# ── the first stage: RRF orders, top 5, dense ranks ──────────────────────────

def rrf_order(dense_ids, splade_ids):
    """One question's whole RRF order: zenc.rrf_top's scores (float32 1/(60 + rank), first occurrences, dense then SPLADE) and its tie
    rule (score, then node id), every candidate listed. The first k entries equal rrf_top(k)."""
    sc = {}
    for ids in (dense_ids, splade_ids):
        w = E.F1 / (E.RRF_C + np.arange(1, ids.size + 1, dtype=np.float32))
        seen = set()
        for i, g in enumerate(ids.tolist()):
            if g < 0 or g in seen:
                continue
            seen.add(g)
            sc[g] = np.float32(sc.get(g, np.float32(0.0)) + w[i])
    return np.asarray([g for g, _ in sorted(sc.items(), key=lambda t: (-float(t[1]), t[0]))], np.int64)


def first_orders(dense, splade):
    return [rrf_order(dense[q], splade[q]) for q in range(dense.shape[0])]


def top_of(orders, k=TOP):
    out = np.full((len(orders), k), -1, np.int64)
    for q, o in enumerate(orders):
        out[q, :min(k, o.size)] = o[:k]
    return out


def ranks_in(row, nodes):
    """1-based positions of nodes in a list row (a proxy rank); rows outside the list get len(row) + 1."""
    row = np.asarray(row, np.int64)
    nodes = np.asarray(nodes, np.int64)
    if row.size == 0:
        return np.ones(nodes.size, np.int64)
    o = np.argsort(row, kind="stable")
    sr = row[o]
    i = np.minimum(np.searchsorted(sr, nodes), sr.size - 1)
    return np.where(sr[i] == nodes, o[i] + 1, row.size + 1).astype(np.int64)


def golds_of(flat, off):
    flat, off = np.asarray(flat, np.int64), np.asarray(off, np.int64)
    return [np.unique(flat[off[q]:off[q + 1]]) for q in range(off.size - 1)]


# ── M1: coverage and rows ────────────────────────────────────────────────────

def held_counts(pools, golds):
    return (np.asarray([g.size for g in golds], np.int64),
            np.asarray([CV.held_in(p, g) for p, g in zip(pools, golds)], np.int64))


def m1_block(golds, pools, rows, n_corpus, rec_pools=None, orders=None):
    """COV's measures of the new pools; beside the pools of record (paired by question), and beside the first stage's own order read at the
    pools' own rows (COV's equal-rows form, covdiag.versus). Questions with no gold are dropped inside covdiag's functions."""
    n_gold, held = held_counts(pools, golds)
    rows = np.asarray(rows, np.int64)
    keep = n_gold > 0
    out = {"new": CV.carve_measures(n_gold, held, rows, n_corpus)}
    if rec_pools is not None:
        _g2, held_r = held_counts(rec_pools, golds)
        rows_r = np.asarray([p.size for p in rec_pools], np.int64)
        blk = CV.diff_block(n_gold[keep], held[keep], held_r[keep])
        blk["calls"] = {k: CALL_NAME[v] for k, v in blk["calls"].items()}
        lo, hi = CV.ratio_ci(rows[keep], rows_r[keep])
        blk["rows_ratio"] = {"pt": CV.rnd(rows[keep].sum() / rows_r[keep].sum(), 5), "ci": [CV.rnd(lo, 5), CV.rnd(hi, 5)]}
        blk["record"] = CV.carve_measures(n_gold, held_r, rows_r, n_corpus)
        out["vs_record"] = blk
    if orders is not None:
        held_x = np.zeros(len(golds), np.int64)
        fits = np.zeros(len(golds), bool)
        for q, (g, o) in enumerate(zip(golds, orders)):
            pos = np.full(g.size, -1, np.int64)
            if g.size and o.size:
                srt = np.argsort(o, kind="stable")
                i = np.minimum(np.searchsorted(o[srt], g), o.size - 1)
                hit = o[srt][i] == g
                pos = np.where(hit, srt[i], -1)
            held_x[q] = CV.held_at(pos, o.size, rows[q])
            fits[q] = rows[q] <= o.size
        blk = CV.versus(n_gold, held, held_x, fits)
        blk["calls"] = {k: CALL_NAME[v] for k, v in blk["calls"].items()}
        blk["rows_ratio"] = {"pt": 1.0, "ci": [1.0, 1.0], "by_construction": "the first stage is read at the pool's own rows"}
        out["vs_first_stage"] = blk
    return out


# ── M2: gold pairs and chain slots ───────────────────────────────────────────

def slot_reads(n, qi, sk, nq, pools, rec_pools, link_keys):
    """Per question: slots, slots with a layout link, with both ends in the pool, with a link inside the pool graph (link and both ends),
    and (when the record's pools are given) both ends in the record's pool. Arrays over the nq questions."""
    qi, sk = np.asarray(qi, np.int64), np.asarray(sk, np.int64)
    o = np.argsort(qi, kind="stable")
    qi, sk = qi[o], sk[o]
    x, y = sk // n, sk % n
    link = UL.member(sk, link_keys)
    both = np.zeros(sk.size, bool)
    both_r = np.zeros(sk.size, bool)
    if sk.size:
        st = np.flatnonzero(np.r_[True, qi[1:] != qi[:-1]])
        en = np.r_[st[1:], sk.size]
        for s, e in zip(st, en):
            j = int(qi[s])
            both[s:e] = UL.member(x[s:e], pools[j]) & UL.member(y[s:e], pools[j])
            if rec_pools is not None:
                both_r[s:e] = UL.member(x[s:e], rec_pools[j]) & UL.member(y[s:e], rec_pools[j])
    bc = lambda m: np.bincount(qi[m], minlength=nq).astype(float)              # noqa: E731
    out = {"slots": np.bincount(qi, minlength=nq).astype(float), "link": bc(link), "both": bc(both), "link_pool": bc(link & both)}
    if rec_pools is not None:
        out["both_rec"] = bc(both_r)
    return out


def m2_cell(R, qmask=None):
    """One cell's shares of slots, with paired question-level intervals, over the questions with a slot (and in qmask)."""
    sel = R["slots"] > 0
    if qmask is not None:
        sel &= qmask
    c = {k: v[sel] for k, v in R.items()}
    out = {"questions": int(sel.sum()), "slots": int(c["slots"].sum())}
    for k in ("link", "both", "link_pool"):
        out[k] = AP.boot(lambda s, k=k: AP.ratio(s[k], s["slots"]), {k: c[k], "slots": c["slots"]})
    if "both_rec" in c:
        out["both_record"] = AP.boot(lambda s: AP.ratio(s["both_rec"], s["slots"]), {"both_rec": c["both_rec"], "slots": c["slots"]})
        d = AP.boot(lambda s: AP.ratio(s["both"], s["slots"]) - AP.ratio(s["both_rec"], s["slots"]),
                    {"both": c["both"], "both_rec": c["both_rec"], "slots": c["slots"]})
        out["both_minus_record"] = {"diff": d, "call": AP.call(d)}
    return out


def m2_block(n, link_keys, pools, rec_pools, slots, hop_q, nq):
    """slots: {kind: (qi, sk)}. MuSiQue's CH is also read by hops; a dataset with no slot of any kind stands on M3."""
    out = {"cells": {}, "applicable": False}
    for kind, (qi, sk) in slots.items():
        if np.asarray(sk).size == 0:
            continue
        out["applicable"] = True
        R = slot_reads(n, qi, sk, nq, pools, rec_pools, link_keys)
        cell = m2_cell(R)
        if kind == "CH":
            cell["hops"] = {str(h): m2_cell(R, hop_q == h) for h in HOPS if ((hop_q == h) & (R["slots"] > 0)).any()}
        out["cells"][kind] = cell
    if not out["applicable"]:
        out["note"] = "no oracle pair (one gold per question): the oracle comparison is M3 alone (section 4)"
    return out


# ── M3: the reach diagnosis ──────────────────────────────────────────────────

def reach_classes(pos, keep, g_ptr):
    """0 IN_B (pos 0), 1 KEPT (1 <= pos <= keep), 2 BEYOND (pos > keep), 3 OUTSIDE (pos -1): one code per gold."""
    pos, keep, g_ptr = np.asarray(pos), np.asarray(keep), np.asarray(g_ptr)
    qi = np.repeat(np.arange(keep.size), np.diff(g_ptr))
    k = keep[qi]
    c = np.full(pos.size, 3, np.int8)
    c[pos == 0] = 0
    c[(pos >= 1) & (pos <= k)] = 1
    c[pos > k] = 2
    return c


def gold_label_bits(G, g_node, qi, B):
    """Per gold: the OR of the label bits of the layout pairs joining it to a node of its question's B (0 = no one-hop link to B)."""
    out = np.zeros(g_node.size, np.int64)
    for i in range(g_node.size):
        nodes, bits = G.nbrs(g_node[i])
        if nodes.size:
            m = UL.member(nodes, B[qi[i]])
            if m.any():
                out[i] = np.bitwise_or.reduce(bits[m])
    return out


def m3_block(P, top5, hop_q, G, B):
    """P: the places arrays. top5 (nq, 5). Shares are of the golds outside the first stage's top 5."""
    nq = P["keep"].size
    g_ptr = np.asarray(P["g_ptr"], np.int64)
    qi = np.repeat(np.arange(nq), np.diff(g_ptr))
    g_node, pos = np.asarray(P["g_node"], np.int64), np.asarray(P["pos"], np.int64)
    cls = reach_classes(pos, P["keep"], g_ptr)
    in_top = np.zeros(g_node.size, bool)
    for q in range(nq):
        s, e = g_ptr[q], g_ptr[q + 1]
        in_top[s:e] = np.isin(g_node[s:e], top5[q][top5[q] >= 0])
    out_m = ~in_top
    names = G.names if G is not None else ()
    bits = gold_label_bits(G, g_node[out_m], qi[out_m], B) if G is not None else np.zeros(int(out_m.sum()), np.int64)
    cq, bq, cls_o = qi[out_m], bits, cls[out_m]
    tot_q = np.bincount(cq, minlength=nq).astype(float)

    def share(mask_o, qsel):
        cnt = np.bincount(cq[mask_o], minlength=nq).astype(float)
        return AP.boot(lambda s: AP.ratio(s["c"], s["t"]), {"c": cnt[qsel], "t": tot_q[qsel]}) if qsel.sum() else None

    def block(qsel):
        sel_o = qsel[cq]
        total = int(sel_o.sum())
        rec = {"golds_outside_top5": total, "questions": int(qsel.sum()), "classes": {}, "shares": {}}
        for ci, c in enumerate(CLASSES):
            m = sel_o & (cls_o == ci)
            rec["classes"][c] = int(m.sum())
            rec["shares"][c] = share(cls_o == ci, qsel)
        rec["reached"] = share((cls_o == 0) | (cls_o == 1), qsel)
        rec["outside_above_flag"] = bool(total and rec["classes"]["OUTSIDE"] / total > OUTSIDE_FLAG)
        return rec

    allq = tot_q > 0
    out = {"all": block(allq), "top5_golds": int(in_top.sum()), "labels": list(names)}
    if hop_q is not None and (hop_q > 0).any():
        out["by_hop"] = {str(h): block(allq & (hop_q == h)) for h in HOPS if (allq & (hop_q == h)).any()}
    if G is not None:
        by_label, by_set = {}, {}
        for ci, c in enumerate(CLASSES):
            m = cls_o == ci
            by_label[c] = {nm: int((m & (((bq >> i) & 1) == 1)).sum()) for i, nm in enumerate(names)}
            by_label[c]["deeper"] = int((m & (bq == 0)).sum())
            u, k = np.unique(bq[m], return_counts=True)
            order = np.argsort(-k, kind="stable")[:12]
            by_set[c] = {G.set_name(u[j]): int(k[j]) for j in order}
        total = max(int(out_m.sum()), 1)
        out["by_label"] = by_label
        out["by_label_share_of_outside_top5"] = {c: {k: round(v / total, 6) for k, v in d.items()} for c, d in by_label.items()}
        out["by_set"] = by_set
        out["residual"] = {c: by_label[c] for c in ("BEYOND", "OUTSIDE")}
        ol = np.asarray(P["one_link"], bool)[out_m]
        nb = cls_o != 0                                              # B nodes are not their own one-link neighbours
        out["one_link_vs_labels"] = {"golds_not_in_B": int(nb.sum()), "agree": int((nb & ((bq != 0) == ol)).sum()),
                                     "label_only": int((nb & (bq != 0) & ~ol).sum()), "build_only": int((nb & (bq == 0) & ol).sum())}
    out["in_I"] = {c: int(np.asarray(P["in_I"], bool)[out_m][cls_o == ci].sum()) for ci, c in enumerate(CLASSES)}
    return out


def paired_reach(golds, pools, rec_pools, top5, rec_top5):
    """The golds outside BOTH first stages' top 5, held by the new pool and by the pool of record; paired by question."""
    g_all, h_new, h_rec = [], [], []
    for q, g in enumerate(golds):
        gc = g[~np.isin(g, top5[q]) & ~np.isin(g, rec_top5[q])]
        g_all.append(gc.size)
        h_new.append(CV.held_in(pools[q], gc))
        h_rec.append(CV.held_in(rec_pools[q], gc))
    g_all, h_new, h_rec = (np.asarray(x, np.int64) for x in (g_all, h_new, h_rec))
    keep = g_all > 0
    if not keep.any():
        return None
    blk = CV.diff_block(g_all[keep], h_new[keep], h_rec[keep])
    blk["calls"] = {k: CALL_NAME[v] for k, v in blk["calls"].items()}
    blk.update({"golds": int(g_all.sum()), "new_held": int(h_new.sum()), "record_held": int(h_rec.sum())})
    return blk


# ── M4: AP4's ledger on the pools ────────────────────────────────────────────

def ledger_arrays(G, pools, B, exp, golds, top5, hop_q, dense=None):
    """The ledger arrays in zap4's own layout (Q_*, P_*) for the anchor read and the placing read; candidate and anchor columns hold
    pool-local indices (zap4's per_question needs them below Q_n)."""
    nq = len(pools)
    QK, PK = AP.QK, AP.PK
    zq = {k: [] for k in QK}
    zp = {k: [] for k in QK}
    PA = {k: [] for k in PK}
    PP = {k: [] for k in PK}

    def add(P, qi, kind, a_loc, a_rank, c_loc, bits, tgt, rk):
        m = c_loc.size
        for k, x in (("q", np.full(m, qi)), ("kind", np.full(m, kind)), ("a", np.full(m, a_loc)), ("arank", np.full(m, a_rank)),
                     ("c", c_loc), ("bits", bits), ("tgt", tgt), ("rk", rk), ("rkc", rk)):
            P[k].append(np.asarray(x))

    for q in range(nq):
        pool, g, t5 = pools[q], golds[q], top5[q][top5[q] >= 0]
        n = int(pool.size)
        gold_in = np.isin(pool, g)
        top_in = np.isin(pool, t5)
        found_a = [int(a) for a in t5 if a in set(g.tolist())]
        qrec = {"n": n, "gt": int(g.size), "hop": int(hop_q[q]), "found": len(found_a), "missed": int((gold_in & ~top_in).sum()),
                "out": int(n - top_in.sum()), "inpool": int(gold_in.sum()), "top1gold": int(bool(t5.size) and bool(t5[0] in g))}
        for k in QK:
            zq[k].append(qrec[k])
        anchors = [(0, a) for a in found_a] + ([(1, int(t5[0]))] if t5.size else [])
        drow = dense[q] if dense is not None else None
        for kind, a in anchors:
            nodes, bits = G.nbrs(a)
            sel = UL.member(nodes, pool) & ~np.isin(nodes, t5)
            c, b = nodes[sel], bits[sel]
            rk = ranks_in(drow, c) if drow is not None else np.ones(c.size, np.int64)
            add(PA, q, kind, int(np.searchsorted(pool, a)), int(ranks_in(drow, [a])[0]) if drow is not None else 1,
                np.searchsorted(pool, c), b, np.isin(c, g), rk)
        # the placing read: the pairs from B into the expansion rows
        e = exp[q]
        ge = np.isin(e, g)
        prec = {"n": n, "gt": int(g.size), "hop": int(hop_q[q]), "found": int(B[q].size), "missed": int(ge.sum()), "out": int(e.size),
                "inpool": int(gold_in.sum()), "top1gold": 0}
        for k in QK:
            zp[k].append(prec[k])
        es = np.sort(e)                                          # |B| is ~20x smaller than |exp|: walk B's pairs, keep those landing in exp
        for b in B[q]:
            nodes, bits = G.nbrs(b)
            m = UL.member(nodes, es)
            if m.any():
                c = nodes[m]
                rk = ranks_in(drow, c) if drow is not None else np.ones(c.size, np.int64)
                add(PP, q, 0, np.full(c.size, np.searchsorted(pool, b)), 1, np.searchsorted(pool, c), bits[m], np.isin(c, g), rk)

    def pack(Q, P):
        z = {f"Q_{k}": np.asarray(v, np.int64) for k, v in Q.items()}
        for k in PK:
            z[f"P_{k}"] = np.concatenate(P[k]) if P[k] else np.zeros(0, np.int64)
        for k in ("q", "a", "c", "kind", "arank", "rk", "rkc", "bits"):
            z[f"P_{k}"] = z[f"P_{k}"].astype(np.int64)
        z["P_tgt"] = z["P_tgt"].astype(bool)
        return z

    return pack(zq, PA), pack(zp, PP)


def unit_calls(z, names, kind, hop=None, with_rank=True):
    """AP4's unit_block for the any-link unit and each label, on one kind's anchors (zap4's own measures and calls)."""
    qs = AP.qsel_of(z, kind, hop)
    allm = (1 << len(names)) - 1
    anys = AP.per_question(z, allm, kind)[0]
    out = {"questions": int(qs.sum()), "any": AP.unit_block(z, allm, kind, qs, None), "labels": {}}
    for i, nm in enumerate(names):
        r = AP.unit_block(z, 1 << i, kind, qs, anys)
        r["thin"] = bool(r["targets"] < MIN_T)
        out["labels"][nm] = r
    if not with_rank:
        for r in [out["any"]] + list(out["labels"].values()):
            for k in ("rank1", "top2", "rank1_cos", "best_median"):
                r.pop(k, None)
    return out


def m4_block(G, pools, B, exp, golds, top5, hop_q, dense=None):
    names = G.names
    z, zp = ledger_arrays(G, pools, B, exp, golds, top5, hop_q, dense)
    out = {"labels": list(names), "rank1_source": "e3v dense list (proxy)" if dense is not None else "none",
           "anchors": {"found": unit_calls(z, names, 0), "top1": unit_calls(z, names, 1)},
           "placing": unit_calls(zp, names, 0, with_rank=False),
           "pairs": {"anchor_pairs": int(z["P_q"].size), "placing_pairs": int(zp["P_q"].size)}}
    if hop_q is not None and (hop_q > 0).any():
        out["by_hop"] = {str(h): {"found": unit_calls(z, names, 0, h), "placing": unit_calls(zp, names, 0, h, with_rank=False)}
                         for h in HOPS if (hop_q == h).any()}
    pts = {}
    for key, blk in (("found", out["anchors"]["found"]), ("top1", out["anchors"]["top1"]), ("placing", out["placing"])):
        pts[key] = sorted(nm for nm, r in blk["labels"].items() if r.get("lift_call") == "POINTS")
    out["points"] = pts
    out["calls"] = {key: {nm: r.get("lift_call") for nm, r in blk["labels"].items()}
                    for key, blk in (("found", out["anchors"]["found"]), ("top1", out["anchors"]["top1"]), ("placing", out["placing"]))}
    return out


# ── section 7's inputs: the candidate list and the report's count ────────────

def candidates_of(m3, m4):
    """An arm's candidate list (section 7, amendment 2d): the layout labels that M4 calls POINTS in any of its three reads (the anchors found,
    the first row's anchor, the pairs that placed KEPT golds) AND that hold residual golds (BEYOND or OUTSIDE golds with a one-hop layout
    pair of that label to a B node: M3's `residual`). Label names in the layout table's order; no decision, the round that picks one declares
    it. Only labels of the layout are visible here: a form no layout carries, or a pool change for BEYOND golds with no label (they are the
    'deeper' column of `residual`), cannot be named by the grade. Returns (names, basis), or (None, None) with no label table (the list is
    unknown then, never 'empty')."""
    if m4 is None or m3.get("residual") is None:
        return None, None
    reads = {"found": m4["anchors"]["found"], "top1": m4["anchors"]["top1"], "placing": m4["placing"]}
    cand, basis = [], {}
    for nm in m3["labels"]:
        pts = [k for k, blk in reads.items() if blk["labels"][nm].get("lift_call") == "POINTS"]
        res = {c: int(m3["residual"][c][nm]) for c in ("BEYOND", "OUTSIDE")}
        if pts and res["BEYOND"] + res["OUTSIDE"] > 0:
            cand.append(nm)
            basis[nm] = {"points_in": pts, "residual_golds": res, "thin_in": [k for k in pts if reads[k]["labels"][nm].get("thin")]}
    return cand, basis


def section7_summary(arms):
    """The report's section 7 line over the graded arms: how many of the nine have an empty candidate list. An arm with no grade, or with no
    label table (candidates None), is undetermined, not empty. `outside_flag_arms` is the flag (OUTSIDE > 5%), never a trigger."""
    by = {f"{a['dataset']}/{a['version']}": a for a in arms}
    order = [f"{d}/{v}" for d, v in ARMS]
    empty = [k for k in order if k in by and by[k].get("candidates") == []]
    withc = [k for k in order if k in by and by[k].get("candidates")]
    flag = [k for k in order if k in by and by[k].get("open_flag")]
    return {"arms": len(ARMS), "graded": len(by), "empty_candidates": len(empty), "empty_arms": empty, "arms_with_candidates": withc,
            "undetermined": [k for k in order if k not in by or by[k].get("candidates") is None],
            "outside_flag_arms": flag, "empty_and_flagged": [k for k in empty if k in flag]}


# ── the grade of one (dataset, version) on in-memory arrays ──────────────────

def gates_of(src, P):
    """Array-level gates: ids, golds, counts (G-C), the top 5 inside B and consistent with the places and E0's h_rrf5."""
    g = {}
    nq = len(src["ids"])
    g["ids_pools_places"] = src["ids"] == P["ids"]
    g["ids_golds"] = src["gold_ids"] is None or src["ids"] == list(src["gold_ids"])
    ptr = np.asarray(P["g_ptr"], np.int64)
    gold_n = np.asarray([x.size for x in src["golds"]], np.int64)
    g["G_C"] = {"pool_rows_grade": int(sum(p.size for p in src["pools"])),
                "pool_rows_build": int(np.sum(src["sizes"])),
                "golds_grade": int(gold_n.sum()), "golds_places": int(P["g_node"].size),
                "golds_equal_per_question": bool(np.array_equal(np.diff(ptr), gold_n)) if len(ptr) == nq + 1 else False}
    ok_g = g["G_C"]["golds_equal_per_question"] and all(
        np.array_equal(np.sort(P["g_node"][ptr[q]:ptr[q + 1]]), src["golds"][q]) for q in range(nq)) if len(ptr) == nq + 1 else False
    g["G_C"]["golds_same_nodes"] = bool(ok_g)
    g["G_C"]["equal"] = bool(g["G_C"]["pool_rows_grade"] == g["G_C"]["pool_rows_build"] and ok_g
                             and (src.get("head_rows") is None or src["head_rows"] == g["G_C"]["pool_rows_grade"]))
    g["pool_is_B_union_exp"] = all(np.array_equal(src["pools"][q], np.union1d(src["B"][q], src["exp"][q])) and
                                   src["pools"][q].size == src["B"][q].size + src["exp"][q].size for q in range(nq))
    exp_n = np.asarray([e.size for e in src["exp"]], np.int64)
    g["keep_equals_expansion"] = bool(np.array_equal(np.asarray(P["keep"], np.int64), exp_n))
    if len(ptr) == nq + 1:
        qix = np.repeat(np.arange(nq), np.diff(ptr))
        g["pos_within_kt_len"] = bool(((P["pos"] >= -1) & (P["pos"] <= np.asarray(P["kt_len"])[qix])).all())
    else:
        g["pos_within_kt_len"] = False
    t5 = src["top5"]
    g["top5_in_B"] = all(UL.member(t5[q][t5[q] >= 0], src["B"][q]).all() for q in range(nq))
    cls = reach_classes(P["pos"], P["keep"], P["g_ptr"])
    in_top = np.concatenate([np.isin(P["g_node"][ptr[q]:ptr[q + 1]], t5[q]) for q in range(nq)]) if nq else np.zeros(0, bool)
    g["top5_golds_are_in_B"] = bool(((cls == 0) | ~in_top).all())
    g["top5_golds_total"] = int(in_top.sum())
    if src.get("h_rrf5") is not None:
        hit = np.asarray([np.isin(src["golds"][q], t5[q]).sum() for q in range(nq)], np.int64)
        want = np.minimum(np.asarray(src["h_rrf5"], np.int64), gold_n)
        g["top5_equals_e0_h_rrf5"] = bool(np.array_equal(hit, want))
    hard = ["ids_pools_places", "ids_golds", "pool_is_B_union_exp", "keep_equals_expansion", "pos_within_kt_len", "top5_in_B",
            "top5_golds_are_in_B"]
    ok = all(g[k] for k in hard) and g["G_C"]["equal"] and g.get("top5_equals_e0_h_rrf5", True)
    g["verdict"] = "PASS" if ok else "FAIL"
    return g


def arm_summary(m1):
    """What the call (section 6) reads of one arm: recall and ALL calls, the rows ratio, and which comparator."""
    which = "vs_record" if "vs_record" in m1 else "vs_first_stage"
    b = m1[which]
    return {"comparator": "pools of record" if which == "vs_record" else "e3v first stage at equal rows",
            "recall_call": b["calls"]["recall"], "ALL_call": b["calls"]["ALL"], "d_recall": b["d_recall"], "d_recall_ci": b["d_recall_ci"],
            "rows_ratio": b["rows_ratio"]}


def grade_arrays(src):
    """The whole grade of one arm from arrays. src (see cmd_grade): ids, golds, pools, B, exp, sizes, places, top5, n_corpus, hop_q,
    optional record_pools, rec_top5, orders (first-stage RRF orders), dense (e3v dense lists), G (LabelGraph), slots."""
    t0 = time.time()
    P = src["places"]
    d, v = src["dataset"], src["version"]
    h, files = code_hash()
    res = {"stage": "PL1 S5", "declared_in": DECLARED, "dataset": d, "version": v, "questions": len(src["ids"]), "code_hash": h,
           "code_files": files, "constants": CONSTANTS}
    res["gates"] = gates_of(src, P)
    if res["gates"]["verdict"] != "PASS":
        res["verdict"] = "GATE_FAIL"
        log(f"{d}/{v}: gates FAIL {json.dumps(jsonable(res['gates']))[:500]}")
        return res
    rows = np.asarray([p.size for p in src["pools"]], np.int64)
    res["m1"] = m1_block(src["golds"], src["pools"], rows, src["n_corpus"], src.get("record_pools"), src.get("orders"))
    res["m1"]["call_inputs"] = arm_summary(res["m1"]) if ("vs_record" in res["m1"] or "vs_first_stage" in res["m1"]) else None
    G = src["G"]
    res["m2"] = m2_block(src["n_corpus"], G.k, src["pools"], src.get("record_pools"), src["slots"], src["hop_q"], len(src["ids"])) \
        if G is not None else {"applicable": False, "note": "no label table"}
    res["m3"] = m3_block(P, src["top5"], src["hop_q"], G, src["B"])
    if src.get("record_pools") is not None and src.get("rec_top5") is not None:
        res["m3"]["paired_reach_vs_record"] = paired_reach(src["golds"], src["pools"], src["record_pools"], src["top5"], src["rec_top5"])
    res["m4"] = m4_block(G, src["pools"], src["B"], src["exp"], src["golds"], src["top5"], src["hop_q"], src.get("dense")) \
        if G is not None else None
    o = res["m3"]["all"]
    pts = res["m4"]["points"] if res["m4"] else {}
    tot = o["golds_outside_top5"]
    # section 7 (amendment 2d): the arm's candidate list (label names; None = no label table, unknown and never "empty") and the OUTSIDE flag
    res["candidates"], res["candidate_basis"] = candidates_of(res["m3"], res["m4"])
    res["open_flag"] = bool(o["outside_above_flag"])             # OUTSIDE share of the golds outside the top 5 > OUTSIDE_FLAG: a flag only
    res["section7"] = {"outside_share": None if not tot else round(o["classes"]["OUTSIDE"] / tot, 6),
                       "beyond_share": None if not tot else round(o["classes"]["BEYOND"] / tot, 6),
                       "residual_golds": {c: o["classes"][c] for c in ("BEYOND", "OUTSIDE")},
                       "residual_by_label": res["m3"].get("residual"), "points_labels": pts,
                       "kept_placed_by": None if res["m4"] is None else {nm: r.get("reach") for nm, r in res["m4"]["placing"]["labels"].items()}}
    res["verdict"] = "GRADED"
    res["seconds"] = round(time.time() - t0, 1)
    return res


# ── the call (section 6) over every arm ──────────────────────────────────────

def call_pools(arms):
    """ONE call over the arms' call inputs. ADOPT_POOLS when no arm is BELOW on gold recall and every arm whose rows-ratio upper end is above
    1.00 is ABOVE on gold recall (section 6's two clauses, which this one condition states); otherwise NOT_ADOPTED. An arm with no record
    comparison, or a gate failure, makes the call INCOMPLETE."""
    rows, bad = [], []
    for a in arms:
        ci = a.get("m1", {}).get("call_inputs") if a.get("verdict") == "GRADED" else None
        if ci is None:
            bad.append({"arm": [a.get("dataset"), a.get("version")], "why": a.get("verdict", "no grade")})
            continue
        hi = ci["rows_ratio"]["ci"][1]
        rows.append({"arm": [a["dataset"], a["version"]], "comparator": ci["comparator"], "recall": ci["recall_call"], "ALL": ci["ALL_call"],
                     "rows_ratio": ci["rows_ratio"]["pt"], "rows_ratio_hi": hi, "rows_above_1": bool(hi > 1.0),
                     "clause1": bool(ci["recall_call"] != "BELOW" and hi <= 1.0),
                     "price_named": bool(hi > 1.0 and ci["recall_call"] == "ABOVE")})
    res = {"arms": rows, "incomplete": bad}
    if bad:
        res["call"] = "INCOMPLETE"
        return res
    no_below = all(r["recall"] != "BELOW" for r in rows)
    res["clause1"] = all(r["clause1"] for r in rows)
    res["clause2"] = bool(no_below and all((not r["rows_above_1"]) or r["price_named"] for r in rows))
    res["call"] = "ADOPT_POOLS" if res["clause1"] or res["clause2"] else "NOT_ADOPTED"
    res["failing"] = [r["arm"] for r in rows if r["recall"] == "BELOW" or (r["rows_above_1"] and not r["price_named"])]
    return res


# ── the host: files in, grade.json out ───────────────────────────────────────

def record_dir(d, v, override=None):
    if override:
        return Path(override)
    native = (d in KB and v == "SRK") or (d not in KB and v == "RK")
    return ROOT / "outputs" / ("pko" if native else "abl/SRK") / "pools" / "parts"


def load_e0(enc, d):
    p = E.OUT / "e0" / enc / f"{d}.npz"
    if not p.is_file():
        return None
    with np.load(p, allow_pickle=False) as z:
        return {"ids": [str(x) for x in z["ids"]], "top_ids": z["top_ids"].astype(np.int64), "gold_total": z["gold_total"].astype(np.int64),
                "h_rrf5": z["h_rrf5"].astype(np.int64), "file": rel(p), "sha256": sha_of(p)}


def load_gold_cache(d):
    p = E.OUT / "e0" / "gold" / f"{d}.npz"
    if not p.is_file():
        stop(f"{rel(p)} is missing: the gold cache of {d}")
    with np.load(p, allow_pickle=False) as z:
        return [str(x) for x in z["ids"]], golds_of(z["flat"], z["off"]), {"file": rel(p), "sha256": sha_of(p)}


def load_splade(d, ids):
    root = E.served_root()
    qids = json.loads((root / d / "queries" / "query_ids.json").read_text(encoding="utf-8"))
    qmap = {q: j for j, q in enumerate(qids)}
    rows = np.asarray([qmap[q] for q in ids], np.int64)
    return np.asarray(np.load(root / d / "retrieval_cache" / "splade_top1000.npz", mmap_mode="r")["ids"][rows], np.int64)


def align(ids_want, ids_have, what):
    at = {q: j for j, q in enumerate(ids_have)}
    miss = [q for q in ids_want if q not in at]
    if miss:
        stop(f"{what}: {len(miss)} of the build's questions are not in the source (first: {miss[0]})")
    return np.asarray([at[q] for q in ids_want], np.int64)


def build_labels(d, v, host):
    """The label table of one arm from UL2's families and carried.npz, checked against zpl1m's layout.json (union count, G-L PASS)."""
    import types
    import zul2p as P2
    lay = json.loads((OUT / d / v / "layout.json").read_text(encoding="utf-8"))
    if lay["gate_GL"]["verdict"] != "PASS":
        stop(f"{d}/{v}: the layout's gate G-L is not PASS")
    D = P2.load_mhr() if d == MHR else P2.load_host(d, host)
    cdir = L2C.OUT / d
    if d == MHR:
        import zul2m as M2
        cdir = M2.DOUT
    cp = cdir / "carried.npz"
    if not cp.is_file():
        stop(f"{d}: {rel(cp)} is not filed (the carried pairs' labels)")
    with np.load(cp) as z:
        ck, code, forms = z["keys"].astype(np.int64), z["code"], [str(x) for x in z["forms"]]
    parts = label_parts(v, D.F, ck, code, forms)
    G = label_graph(D.n, parts)
    if G.k.size != int(lay["counts"]["union"]):
        stop(f"{d}/{v}: the label table holds {G.k.size:,} pairs, the layout's union is {int(lay['counts']['union']):,}")
    return G, {"pairs": int(G.k.size), "layout_union": int(lay["counts"]["union"]), "labels": list(G.names), "carried_npz": rel(cp),
               "carried_sha256": sha_of(cp), "layout_json": rel(OUT / d / v / "layout.json"), "n": int(D.n)}, D


def slots_for(d, ids, golds, n, served):
    """GP slots from the gold sets; MuSiQue's CH slots from UL2's own reader; none for a dataset with one gold per question."""
    kb = UL.KB_GOLDS if d in KB else None
    out = {"GP": UL.gp_slots(golds, n, kb)}
    if d == "musique":
        recs = UL.musique_queries(served, ids)
        qi, keys, _h = UL.chain_slots(recs, ids, golds, UL.node_positions(served, d), n)
        out["CH"] = (qi, keys)
    return out


def cmd_grade(a):
    d, v = a.ds, a.version
    if d not in DATASETS or (d, v) not in ARMS:
        stop(f"the arms are {ARMS}")
    t0 = time.time()
    base = OUT / d / v
    pr = read_parts(base / "pools" / "parts", d)
    pl = read_places(base / "places", d)
    if pr is None or pl is None:
        stop(f"{d}/{v}: the pools parts or the places parts are not under {rel(base)}")
    gold_ids, gold_sets, gsrc = load_gold_cache(d)
    at = align(pr["ids"], gold_ids, "the gold cache")
    golds = [gold_sets[j] for j in at]
    e3 = load_e0("e3v", d)
    if e3 is None:
        stop(f"E0's e3v list of {d} is missing")
    ae = align(pr["ids"], e3["ids"], "E0's e3v list")
    dense = e3["top_ids"][ae]
    splade = load_splade(d, pr["ids"])
    orders = first_orders(dense, splade)
    top5 = top_of(orders, TOP)
    G, lrec, D = build_labels(d, v, a.host)
    n = int(D.n)
    import u1b
    served, _freeze = u1b.package_root(a.host)
    slots = slots_for(d, pr["ids"], golds, n, served)
    hop_q = np.asarray([AP.hop_of(d, q) for q in pr["ids"]], np.int64)
    rd = record_dir(d, v, a.record_dir)
    rec = None if d == MHR else read_parts(rd, d)
    rec_pools = rec_top5 = None
    rec_note = None
    if d != MHR:
        if rec is None:
            rec_note = f"no parts of the pools of record in {rel(rd)}"
        else:
            ar = align(pr["ids"], rec["ids"], "the pools of record")
            rec_pools = [rec["pools"][j] for j in ar]
            fr = load_e0("frozen", d)
            if fr is not None:
                af = align(pr["ids"], fr["ids"], "E0's frozen list")
                rec_top5 = top_of(first_orders(fr["top_ids"][af], splade), TOP)
    src = {"dataset": d, "version": v, "ids": pr["ids"], "gold_ids": pr["ids"], "golds": golds, "pools": pr["pools"], "B": pr["B"],
           "exp": pr["exp"], "sizes": pr["sizes"], "places": pl, "top5": top5, "n_corpus": n, "hop_q": hop_q, "record_pools": rec_pools,
           "rec_top5": rec_top5, "orders": orders, "dense": dense, "G": G, "slots": slots, "h_rrf5": e3["h_rrf5"][ae],
           "head_rows": None if any(p["head_rows"] is None for p in pr["parts"]) else int(sum(p["head_rows"] for p in pr["parts"]))}
    res = grade_arrays(src)
    res["inputs"] = {"pools": pr["parts"], "places": pl["files"], "golds": gsrc, "e0_e3v": {"file": e3["file"], "sha256": e3["sha256"]},
                     "labels": lrec, "record_dir": rel(rd), "record_note": rec_note,
                     "record_parts": None if rec is None else rec.get("parts"), "n_corpus": n}
    theirs = {}
    for p in pr["parts"]:                                        # the build's own count of each class (zpl1w's .view.json), beside the grade's
        f = (ROOT / p["file"]).with_name(Path(p["file"]).stem + ".view.json")
        if f.is_file():
            for k_, c_ in json.loads(f.read_text(encoding="utf-8")).get("places", {}).get("classes", {}).items():
                theirs[k_] = theirs.get(k_, 0) + int(c_)
    mine = dict(zip(CLASSES, np.bincount(reach_classes(pl["pos"], pl["keep"], pl["g_ptr"]), minlength=4).astype(int).tolist()))
    res["inputs"]["view_json_classes"] = {"build": theirs or None, "grade": mine, "equal": (theirs == mine) if theirs else None}
    if res["gates"]["verdict"] == "PASS" and res.get("m1") is not None and rec_note and d != MHR:
        res["verdict"] = "NO_RECORD"
    res["seconds"] = round(time.time() - t0, 1)
    res["utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    wjson(base / "grade.json", res)
    log(f"{d}/{v}: {res['verdict']} ({res['seconds']} s) -> {rel(base / 'grade.json')}")
    return 0 if res["verdict"] == "GRADED" else 3


# ── the report ───────────────────────────────────────────────────────────────

def pct(x, k=2):
    return "n/a" if x is None else f"{100 * x:.{k}f}"


def ci_txt(c, k=2, sign=False):
    if c is None:
        return "n/a"
    lo, hi = (c["ci"] if isinstance(c, dict) else c)
    pt = c["pt"] if isinstance(c, dict) else None
    f = (lambda x: f"{100 * x:+.{k}f}") if sign else (lambda x: f"{100 * x:.{k}f}")
    return f"{f(pt)} [{f(lo)}, {f(hi)}]" if pt is not None else f"[{f(lo)}, {f(hi)}]"


def report_md(rep):
    L = ["# PL1 report: the pools on the UL2 layout (S5 grade and S6 call)", "",
         f"Declared in {DECLARED}. Code hash `{rep['code_hash'][:16]}`. Call: **{rep['call']['call']}**.", "",
         "## The call (section 6), one call over the nine arms", "",
         "| arm | comparator | recall | ALL | rows ratio (upper end) | clause 1 | price named |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for r in rep["call"]["arms"]:
        L.append(f"| {r['arm'][0]}/{r['arm'][1]} | {r['comparator']} | {r['recall']} | {r['ALL']} | {r['rows_ratio']:.4f} "
                 f"({r['rows_ratio_hi']:.4f}) | {r['clause1']} | {r['price_named']} |")
    if rep["call"]["incomplete"]:
        L += ["", "Incomplete arms: " + json.dumps(rep["call"]["incomplete"])]
    L += ["", "## M1, coverage and rows (x100)", "", "| arm | gold recall new | record / first stage | d_recall [95%] | ALL new | mean rows new | share of corpus |",
          "| --- | --- | --- | --- | --- | --- | --- |"]
    for a in rep["arms"]:
        m1 = a["m1"]
        vs = m1.get("vs_record") or m1.get("vs_first_stage")
        base = (vs.get("record") or {"gold_recall": vs.get("gold_recall_x")})["gold_recall"]
        L.append(f"| {a['dataset']}/{a['version']} | {pct(m1['new']['gold_recall'])} | {pct(base)} | {ci_txt(vs['d_recall_ci'] and {'pt': vs['d_recall'], 'ci': vs['d_recall_ci']}, sign=True)} "
                 f"| {pct(m1['new']['ALL'])} | {m1['new']['rows_mean']} | {pct(m1['new']['share_mean'], 3)}% |")
    L += ["", "## M3, the reach diagnosis (shares of golds outside the first stage's top 5, x100)", "",
          "| arm | golds | IN_B | KEPT | BEYOND | OUTSIDE | OUTSIDE > 5% |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for a in rep["arms"]:
        m = a["m3"]["all"]
        t = max(m["golds_outside_top5"], 1)
        L.append(f"| {a['dataset']}/{a['version']} | {m['golds_outside_top5']} | " + " | ".join(pct(m["classes"][c] / t) for c in CLASSES)
                 + f" | {m['outside_above_flag']} |")
    s7 = rep["section7"]
    L += ["", "## Section 7, the closing rule's inputs per arm", "",
          f"**{s7['empty_candidates']} of {s7['arms']} arms have an empty candidate list** ({s7['graded']} of {s7['arms']} graded"
          + (f"; undetermined, not counted as empty: {', '.join(s7['undetermined'])}" if s7["undetermined"] else "") + "). "
          "A candidate is a label of the layout that M4 calls POINTS in some read and that holds residual (BEYOND or OUTSIDE) golds; the list "
          "names no decision, and an arm with candidates is OPEN. The OUTSIDE flag (> 5%) is a flag only. "
          f"Flagged: {', '.join(s7['outside_flag_arms']) or 'none'}.", "",
          "| arm | OUTSIDE share | BEYOND share | OUTSIDE > 5% | POINTS labels (found / top1 / placing) | candidates (label: BEYOND+OUTSIDE golds, POINTS reads) |",
          "| --- | --- | --- | --- | --- | --- |"]
    for a in rep["arms"]:
        s = a["section7"]
        p = s["points_labels"] or {}
        cb = a.get("candidate_basis") or {}
        cand = "n/a (no label table)" if a.get("candidates") is None else (
            "; ".join(f"{nm}: {sum(cb[nm]['residual_golds'].values())}, {'+'.join(cb[nm]['points_in'])}" for nm in a["candidates"]) or "none (empty)")
        L.append(f"| {a['dataset']}/{a['version']} | {pct(s['outside_share'])} | {pct(s['beyond_share'])} | {a['open_flag']} | "
                 f"{', '.join(p.get('found', [])) or '-'} / {', '.join(p.get('top1', [])) or '-'} / {', '.join(p.get('placing', [])) or '-'} | {cand} |")
    return "\n".join(L) + "\n"


def cmd_report(_a):
    h, files = code_hash()
    arms, miss = [], []
    for d, v in ARMS:
        p = OUT / d / v / "grade.json"
        if p.is_file():
            arms.append(json.loads(p.read_text(encoding="utf-8")))
        else:
            miss.append([d, v])
    hashes = sorted({x.get("code_hash") for x in arms})
    rep = {"declared_in": DECLARED, "code_hash": h, "code_files": files, "constants": CONSTANTS, "missing": miss}
    if len(hashes) > 1 or (hashes and hashes[0] != h):
        rep.update(verdict="REFUSED", why="the arms' code hashes differ from each other or from this code (section 2a: one rule set, one call)",
                   hashes={f"{x['dataset']}/{x['version']}": x.get("code_hash") for x in arms})
        wjson(OUT / "report.json", rep)
        log(f"REFUSED: {len(hashes)} hashes among the grades")
        return 1
    rep["call"] = call_pools(arms + [{"dataset": d, "version": v, "verdict": "MISSING"} for d, v in miss])
    rep["arms"] = [a for a in arms if a.get("verdict") == "GRADED"]
    rep["section7"] = section7_summary(rep["arms"])
    rep["verdict"] = "FILED" if not miss and rep["call"]["call"] != "INCOMPLETE" else "INCOMPLETE"
    wjson(OUT / "report.json", rep)
    if rep["arms"]:
        (OUT / "report.md").write_text(report_md(rep), encoding="utf-8", newline="\n")
    log(f"report {rep['verdict']}: call {rep['call']['call']}, {len(rep['arms'])} graded, missing {miss}; "
        f"empty candidate lists {rep['section7']['empty_candidates']} of {len(ARMS)}")
    return 0 if rep["verdict"] == "FILED" else 2


# ── self-test (real-shape arrays; brute force with python sets) ──────────────

def selftest():
    # 1. rrf_order equals zenc.rrf_top on a real e3v dense list and a shuffled stand-in for SPLADE
    rng = np.random.default_rng(0)
    p = E.OUT / "e0" / "e3v" / "squad.npz"
    if p.is_file():
        with np.load(p) as z:
            dense = z["top_ids"][:60].astype(np.int64)
        spl = np.stack([np.concatenate([rng.permutation(dense[q][:400]), rng.integers(0, 9_000_000, 600)]) for q in range(dense.shape[0])])
        ord_ = first_orders(dense, spl)
        assert np.array_equal(top_of(ord_, 5), E.rrf_top(dense, spl, 5)), "rrf_order is not zenc.rrf_top"
        assert np.array_equal(top_of(ord_, 20), E.rrf_top(dense, spl, 20))
        r = ranks_in(dense[0], np.r_[dense[0][3], dense[0][990], 99999999])
        assert list(r) == [4, 991, 1001], r
    # 2. the real-shape stand-in: PKO's local webqsp part as the new pools, the E0 gold cache as golds
    part = ROOT / "outputs" / "pko" / "pools" / "parts" / "webqsp__s1eval__0of2.npz"
    gp = E.OUT / "e0" / "gold" / "webqsp.npz"
    if not (part.is_file() and gp.is_file()):
        log("selftest: the webqsp part or the gold cache is not on this machine; array-level tests only")
        part = None
    toy_tests()
    if part is None:
        log("selftest: OK (toy only)")
        return 0
    pr = read_part(part)
    for key in ("ids", "B", "exp", "pools"):                      # 240 of the part's questions: real shapes, a test that finishes
        pr[key] = pr[key][:240]
    pr["sizes"] = pr["sizes"][:240]
    tm = [time.time()]

    def lap(what):
        now = time.time()
        log(f"selftest: {what} ({now - tm[0]:.1f} s)")
        tm[0] = now
    gids, gsets, _ = load_gold_cache("webqsp")
    at = align(pr["ids"], gids, "gold cache")
    golds = [gsets[j] for j in at]
    nq = len(golds)
    n = int(max(int(p.max()) for p in pr["pools"] if p.size) + 1000)
    # (a) the part reads as zpko's format: B inside the pool, expansion disjoint from B, pool = B + exp
    for q in range(nq):
        assert np.array_equal(pr["pools"][q], np.union1d(pr["B"][q], pr["exp"][q]))
        assert pr["B"][q].size + pr["exp"][q].size == pr["pools"][q].size and np.all(np.diff(pr["B"][q]) > 0)
    # (b) places with known classes: gold in B -> 0, in exp -> its position in exp order, others: BEYOND for even ids, OUTSIDE for odd
    P = synthetic_places(pr, golds)
    keep = P["keep"].astype(np.int64)
    cls = reach_classes(P["pos"], P["keep"], P["g_ptr"])
    k = 0
    for q in range(nq):
        for g in golds[q]:
            want = 0 if g in pr["B"][q] else 1 if g in pr["exp"][q] else 2 if g % 2 == 0 else 3
            assert cls[k] == want, (q, g, cls[k], want)
            k += 1
    # (c) a first stage: top 5 = the golds in B first, then the rest of B (so anchors exist); labels: random pairs, plus every B node -> gold
    top5, dense, _splade = synthetic_first_stage(pr, golds, n, rng)
    G, parts = toy_labels(pr, golds, n, rng)
    assert G.k.size == np.unique(np.concatenate([k_ for _, k_ in parts])).size
    # label bits brute force
    bf = {}
    for i, (_nm, k_) in enumerate(parts):
        for kk in k_.tolist():
            bf[kk] = bf.get(kk, 0) | (1 << i)
    assert dict(zip(G.k.tolist(), G.b.tolist())) == bf
    for x in rng.choice(np.concatenate(pr["B"][:5]), 25):
        nb, bits = G.nbrs(x)
        want = {}
        for kk, bb in bf.items():
            if kk // n == x:
                want[kk % n] = bb
            elif kk % n == x:
                want[kk // n] = bb
        assert dict(zip(nb.tolist(), bits.tolist())) == want
    hop_q = np.zeros(nq, np.int64)
    slots = {"GP": UL.gp_slots(golds, n, UL.KB_GOLDS)}
    src = {"dataset": "webqsp", "version": "SRK", "ids": pr["ids"], "gold_ids": pr["ids"], "golds": golds, "pools": pr["pools"], "B": pr["B"],
           "exp": pr["exp"], "sizes": pr["sizes"], "places": P, "top5": top5, "n_corpus": n, "hop_q": hop_q, "G": G, "slots": slots,
           "record_pools": None, "orders": None, "dense": dense, "head_rows": None}
    gt = gates_of(src, P)
    assert gt["verdict"] == "PASS", gt
    # a corrupted place or a dropped gold is caught by the gates
    bad = dict(P, g_node=P["g_node"].copy())
    bad["g_node"][0] += 1
    assert gates_of(src, bad)["verdict"] == "FAIL"
    assert gates_of(src, dict(P, keep=P["keep"] + 1))["verdict"] == "FAIL"                      # the cut is not the expansion's size
    bad = dict(P, pos=P["pos"].copy())
    bad["pos"][bad["pos"] > 0] += 10_000                                                          # a place past |kt_cos|
    assert gates_of(src, bad)["verdict"] == "FAIL"
    assert gates_of(dict(src, top5=np.where(src["top5"] >= 0, src["top5"] + 1, -1)), P)["verdict"] == "FAIL"   # a top 5 outside B
    # (d) M3 by brute force
    lap("setup, places, labels")
    m3 = m3_block(P, top5, hop_q, G, pr["B"])
    lap("m3_block")
    cnt = {c: 0 for c in CLASSES}
    lab = {c: {nm: 0 for nm in G.names} | {"deeper": 0} for c in CLASSES}
    kg = 0
    for q in range(nq):
        for g in golds[q]:
            if g in top5[q]:
                kg += 1
                continue
            c = CLASSES[0 if g in pr["B"][q] else 1 if g in pr["exp"][q] else 2 if g % 2 == 0 else 3]
            cnt[c] += 1
            b = 0
            for x in pr["B"][q].tolist():
                kk = min(x, g) * n + max(x, g)
                b |= bf.get(kk, 0)
            if b == 0:
                lab[c]["deeper"] += 1
            for i, nm in enumerate(G.names):
                lab[c][nm] += b >> i & 1
    assert m3["all"]["classes"] == cnt and m3["top5_golds"] == kg, (m3["all"]["classes"], cnt)
    assert m3["by_label"] == lab
    tot = sum(cnt.values())
    assert abs(m3["all"]["shares"]["KEPT"]["pt"] - cnt["KEPT"] / tot) < 1e-5
    lap("m3 brute force")
    # (e) M1 on the stand-in, the stand-in beside itself (SAME, ratio 1) and beside a pool with its expansion cut in half
    pools_h = [np.union1d(pr["B"][q], pr["exp"][q][:pr["exp"][q].size // 2]) for q in range(nq)]
    m1 = m1_block(golds, pr["pools"], pr["sizes"], n, pools_h)
    ng, hd = held_counts(pr["pools"], golds)
    assert m1["new"]["golds"] == int(ng[ng > 0].sum()) and m1["new"]["golds_held"] == int(hd[ng > 0].sum())
    brute = sum(len(set(g.tolist()) & set(p.tolist())) for g, p in zip(golds, pr["pools"]) if len(g))
    assert m1["new"]["golds_held"] == brute
    assert m1["vs_record"]["calls"]["recall"] in ("ABOVE", "LEVEL") and m1["vs_record"]["rows_ratio"]["pt"] > 1.0
    m1s = m1_block(golds, pr["pools"], pr["sizes"], n, pr["pools"])
    assert m1s["vs_record"]["calls"] == {"recall": "LEVEL", "ALL": "LEVEL"} and m1s["vs_record"]["rows_ratio"]["ci"] == [1.0, 1.0]
    # first stage at equal rows: an order = the pool itself (gold hits equal the pool's) is LEVEL
    m1o = m1_block(golds, pr["pools"], pr["sizes"], n, None, [np.asarray(p, np.int64) for p in pr["pools"]])
    assert m1o["vs_first_stage"]["calls"]["recall"] == "LEVEL" and m1o["vs_first_stage"]["past_length"] == 0
    lap("m1")
    # (f) M2 by brute force
    qi, sk = slots["GP"]
    R = slot_reads(n, qi, sk, nq, pr["pools"], pools_h, G.k)
    kset = set(G.k.tolist())
    link = both = lp = bothr = 0
    for j, key in zip(qi.tolist(), sk.tolist()):
        x, y = key // n, key % n
        l_ = key in kset
        b_ = x in pr["pools"][j] and y in pr["pools"][j]
        link += l_
        both += b_
        lp += l_ and b_
        bothr += x in pools_h[j] and y in pools_h[j]
    assert (R["link"].sum(), R["both"].sum(), R["link_pool"].sum(), R["both_rec"].sum()) == (link, both, lp, bothr)
    m2 = m2_block(n, G.k, pr["pools"], pools_h, slots, hop_q, nq)
    assert m2["applicable"] and m2["cells"]["GP"]["slots"] == sk.size
    assert m2_block(n, G.k, pr["pools"], None, {"GP": (np.zeros(0, np.int64), np.zeros(0, np.int64))}, hop_q, nq)["applicable"] is False
    lap("m2")
    # (g) M4: the ledger arrays by brute force on a few questions, then the units run
    z, zp = ledger_arrays(G, pr["pools"], pr["B"], pr["exp"], golds, top5, hop_q, dense)
    lap("ledger_arrays")
    for q in range(0, nq, 37):
        pool, g, t5 = pr["pools"][q], golds[q], top5[q][top5[q] >= 0]
        gs = set(g.tolist())
        found = [a for a in t5.tolist() if a in gs]
        assert z["Q_found"][q] == len(found) and z["Q_missed"][q] == len(gs & set(pool.tolist()) - set(t5.tolist()))
        assert z["Q_out"][q] == pool.size - len(set(t5.tolist()) & set(pool.tolist()))
        for kind, anchors in ((0, found), (1, [t5[0]] if t5.size else [])):
            sel = (z["P_q"] == q) & (z["P_kind"] == kind)
            got = sorted(zip(z["P_a"][sel].tolist(), z["P_c"][sel].tolist(), z["P_bits"][sel].tolist(), z["P_tgt"][sel].tolist()))
            want = []
            for a in anchors:
                for kk, bb in bf.items():
                    if kk // n == a or kk % n == a:
                        c = kk % n if kk // n == a else kk // n
                        if c in set(pool.tolist()) and c not in set(t5.tolist()):
                            want.append((int(np.searchsorted(pool, a)), int(np.searchsorted(pool, c)), bb, c in gs))
            assert got == sorted(want), (q, kind)
        # placing pairs: (B node, expansion row) pairs
        sel = zp["P_q"] == q
        got = sorted(zip(zp["P_a"][sel].tolist(), zp["P_c"][sel].tolist(), zp["P_bits"][sel].tolist(), zp["P_tgt"][sel].tolist()))
        want = []
        for e in pr["exp"][q].tolist():
            for x in pr["B"][q].tolist():
                kk = min(x, e) * n + max(x, e)
                if kk in bf:
                    want.append((int(np.searchsorted(pool, x)), int(np.searchsorted(pool, e)), bf[kk], e in gs))
        assert got == sorted(want), q
    lap("ledger brute force")
    m4 = m4_block(G, pr["pools"], pr["B"], pr["exp"], golds, top5, hop_q, dense)
    lap("m4_block")
    assert m4["pairs"]["anchor_pairs"] == z["P_q"].size and set(m4["points"]) == {"found", "top1", "placing"}
    for nm in G.names:
        assert m4["anchors"]["found"]["labels"][nm]["lift_call"] in (None, "POINTS", "NO_LIFT", "BELOW_BASE")
    # (h) the whole grade, twice (the same bytes), then the call
    res = grade_arrays(src)
    lap("grade_arrays")
    assert res["verdict"] == "GRADED" and res["section7"]["points_labels"] is not None
    res2 = grade_arrays(src)
    assert json.dumps(jsonable({k: v for k, v in res.items() if k != "seconds"}), sort_keys=True) == \
        json.dumps(jsonable({k: v for k, v in res2.items() if k != "seconds"}), sort_keys=True), "the grade is not deterministic"
    src2 = dict(src, record_pools=pools_h, orders=[np.asarray(p, np.int64) for p in pr["pools"]], rec_top5=top5)
    res3 = grade_arrays(src2)
    assert res3["m1"]["call_inputs"]["comparator"] == "pools of record" and res3["m3"]["paired_reach_vs_record"] is not None
    assert "vs_first_stage" in res3["m1"] and res3["m2"]["cells"]["GP"]["both_minus_record"] is not None
    call = call_pools([res3])
    s7 = section7_summary([res3])
    md = report_md({"code_hash": code_hash()[0], "call": call, "arms": [res3], "section7": s7})
    assert call["call"] in ("ADOPT_POOLS", "NOT_ADOPTED") and "M3, the reach diagnosis" in md, call
    assert f"**{s7['empty_candidates']} of 9 arms have an empty candidate list**" in md and "candidates (label:" in md
    # section 7: the candidate list equals its definition read straight off M3's residual and M4's calls, the flag equals the share test
    want = [nm for nm in G.names if res3["m4"]["points"]["found"].count(nm) + res3["m4"]["points"]["top1"].count(nm)
            + res3["m4"]["points"]["placing"].count(nm) > 0
            and res3["m3"]["by_label"]["BEYOND"][nm] + res3["m3"]["by_label"]["OUTSIDE"][nm] > 0]
    assert res3["candidates"] == want and set(res3["candidate_basis"]) == set(want), (res3["candidates"], want)
    o3 = res3["m3"]["all"]
    assert res3["open_flag"] == bool(o3["classes"]["OUTSIDE"] / o3["golds_outside_top5"] > 0.05) and res3["candidates"] == res["candidates"]
    assert res3["section7"]["residual_golds"] == {"BEYOND": o3["classes"]["BEYOND"], "OUTSIDE": o3["classes"]["OUTSIDE"]}
    log(f"selftest: section 7 on the stand-in: candidates {res3['candidates']}, open_flag {res3['open_flag']}")
    section7_test()
    lap("grade with the record, the call, the report text")
    report_test()
    grade_cmd_test()
    lap("report refusal, cmd_grade on files")
    log("selftest: OK (stand-in webqsp part: gates, M1-M4, the grade, the call and the report equal the brute force)")
    return 0


def synthetic_places(pr, golds):
    """A places dict (the contract's arrays) whose classes are known from the pool: a gold in B -> pos 0, in the expansion -> its 1-based place
    in expansion order, otherwise past the cut for an even node id and in no order (-1) for an odd one."""
    nq = len(golds)
    keep = np.asarray([e.size for e in pr["exp"]], np.int64)
    pos, g_ptr = [], [0]
    for q in range(nq):
        for g in golds[q]:
            if g in pr["B"][q]:
                pos.append(0)
            elif g in pr["exp"][q]:
                pos.append(int(np.flatnonzero(pr["exp"][q] == g)[0]) + 1)
            else:
                pos.append(int(keep[q]) + 1 + int(g % 7) if g % 2 == 0 else -1)
        g_ptr.append(len(pos))
    return {"ids": pr["ids"], "keep": keep.astype(np.int32), "kt_len": (keep + 20).astype(np.int32), "g_ptr": np.asarray(g_ptr, np.int64),
            "g_node": np.concatenate(golds).astype(np.int64), "pos": np.asarray(pos, np.int32),
            "in_I": np.zeros(len(pos), bool), "one_link": np.zeros(len(pos), bool)}


def synthetic_first_stage(pr, golds, n, rng):
    """A top 5 per question (the golds inside B first, then B's other nodes: inside B, so the anchors exist), and dense and SPLADE lists
    that open with it, so their RRF top 5 is that top 5 in that order (2/(60+r) beats any single list's 1/(60+6))."""
    nq = len(golds)
    top5 = np.full((nq, 5), -1, np.int64)
    for q in range(nq):
        gb = [g for g in golds[q] if g in pr["B"][q]]
        t = list(gb[:3]) + [b for b in pr["B"][q] if b not in gb[:3]]
        top5[q, :min(5, len(t))] = t[:5]
    dense = np.stack([np.concatenate([top5[q], rng.integers(0, n, 995)]) for q in range(nq)])
    splade = np.stack([np.concatenate([top5[q], rng.integers(0, n, 995)]) for q in range(nq)])
    return top5, dense, splade


def toy_labels(pr, golds, n, rng):
    """A label graph over real node ids: random pairs, plus links the walk could use (B <-> gold, B <-> expansion row), each pair in some
    labels. Returns the graph and the (label, keys) parts it was built from."""
    nq = len(golds)
    keys = set()
    for _ in range(40000):
        a_, b_ = sorted(rng.integers(0, n, 2).tolist())
        if a_ != b_:
            keys.add(a_ * n + b_)
    for q in range(0, nq, 3):
        for g in golds[q][:2]:
            for b in pr["B"][q][:2]:
                if b != g:
                    keys.add(min(b, g) * n + max(b, g))
        for e in pr["exp"][q][:6]:
            b = pr["B"][q][int(e) % pr["B"][q].size]
            keys.add(min(b, e) * n + max(b, e))
    keys = np.asarray(sorted(keys), np.int64)
    names = ("S",) + BASE_LABELS
    parts = [(nm, np.sort(rng.choice(keys, size=keys.size // (2 + i), replace=False))) for i, nm in enumerate(names)]
    return label_graph(n, parts), parts


def grade_cmd_test():
    """cmd_grade end to end on files: the local webqsp part as the new pools and (renamed 0of1) as the pools of record, synthetic places and
    first-stage lists with known classes, the real gold cache. Nothing is read from the host."""
    import shutil
    import types
    import u1b
    global OUT, load_e0, load_splade, build_labels
    part = ROOT / "outputs" / "pko" / "pools" / "parts" / "webqsp__s1eval__0of2.npz"
    if not (part.is_file() and (E.OUT / "e0" / "gold" / "webqsp.npz").is_file()):
        return
    saved = (OUT, load_e0, load_splade, build_labels, u1b.package_root)
    rng = np.random.default_rng(1)
    try:
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            OUT = td / "pl1"
            base = OUT / "webqsp" / "SRK"
            for sub in (base / "pools" / "parts", td / "record"):
                sub.mkdir(parents=True)
                shutil.copy(part, sub / "webqsp__s1eval__0of1.npz")
                shutil.copy(part.with_suffix(".json"), sub / "webqsp__s1eval__0of1.json")
            pr = read_parts(base / "pools" / "parts", "webqsp")
            gids, gsets, _ = load_gold_cache("webqsp")
            golds = [gsets[j] for j in align(pr["ids"], gids, "gold cache")]
            n = int(max(int(p.max()) for p in pr["pools"] if p.size) + 1000)
            P = synthetic_places(pr, golds)
            top5, dense, splade = synthetic_first_stage(pr, golds, n, rng)
            G, _parts = toy_labels(pr, golds, n, rng)
            h5 = np.asarray([np.isin(g, t).sum() for g, t in zip(golds, top5)], np.int64)
            (base / "places").mkdir()
            np.savez(base / "places" / "webqsp__s1eval__0of1.npz", **{k: (np.asarray(v, dtype=str) if k == "ids" else v)
                                                                       for k, v in P.items()})
            load_e0 = lambda enc, d: {"ids": pr["ids"], "top_ids": dense, "gold_total": np.asarray([g.size for g in golds]),   # noqa: E731
                                      "h_rrf5": h5, "file": "synthetic", "sha256": "0"}
            load_splade = lambda d, ids: splade                                                                          # noqa: E731
            build_labels = lambda d, v, host: (G, {"pairs": int(G.k.size), "labels": list(G.names)}, types.SimpleNamespace(n=n))   # noqa: E731
            u1b.package_root = lambda host: (None, None)
            a = types.SimpleNamespace(ds="webqsp", version="SRK", host=False, record_dir=str(td / "record"))
            assert cmd_grade(a) == 0
            res = json.loads((base / "grade.json").read_text(encoding="utf-8"))
            assert res["verdict"] == "GRADED" and res["gates"]["verdict"] == "PASS" and res["gates"]["top5_equals_e0_h_rrf5"]
            assert res["gates"]["G_C"]["equal"] and res["code_hash"] == code_hash()[0]
            assert res["m1"]["vs_record"]["calls"] == {"recall": "LEVEL", "ALL": "LEVEL"} and res["m1"]["call_inputs"]["comparator"] == "pools of record"
            assert res["m1"]["vs_first_stage"]["past_length"] > 0 and res["m3"]["paired_reach_vs_record"] is not None
            assert res["m2"]["cells"]["GP"]["slots"] > 0 and res["inputs"]["view_json_classes"]["equal"] is None
            assert isinstance(res["open_flag"], bool) and (res["candidates"] is None or isinstance(res["candidates"], list))
            assert cmd_report(None) == 2                                                  # one arm graded, eight missing: INCOMPLETE, and the files are written
            rj = json.loads((OUT / "report.json").read_text(encoding="utf-8"))
            assert rj["section7"]["graded"] == 1 and rj["section7"]["arms"] == 9 and len(rj["section7"]["undetermined"]) >= 8
            assert "of 9 arms have an empty candidate list" in (OUT / "report.md").read_text(encoding="utf-8")
            a.record_dir = str(td / "nowhere")                                            # no record parts: refused as NO_RECORD, exit 3
            assert cmd_grade(a) == 3 and json.loads((base / "grade.json").read_text(encoding="utf-8"))["verdict"] == "NO_RECORD"
    finally:
        OUT, load_e0, load_splade, build_labels, u1b.package_root = saved


def section7_test():
    """candidates_of and section7_summary on hand-built blocks: POINTS without residual is not listed, residual without POINTS is not listed,
    a POINTS read of any of the three counts, no label table is None (never empty), and the count of empty lists over the nine arms."""
    def blk(**calls):
        return {"labels": {nm: {"lift_call": calls.get(nm, "NO_LIFT"), "thin": nm == "B"} for nm in ("A", "B", "C")}}
    m3 = {"labels": ["A", "B", "C"], "residual": {"BEYOND": {"A": 2, "B": 0, "C": 1, "deeper": 3}, "OUTSIDE": {"A": 0, "B": 0, "C": 0, "deeper": 1}}}
    m4 = {"anchors": {"found": blk(A="POINTS", B="POINTS"), "top1": blk()}, "placing": blk(C="POINTS", B="POINTS")}
    cand, basis = candidates_of(m3, m4)
    assert cand == ["A", "C"], cand                              # B points but holds no residual gold
    assert basis["A"] == {"points_in": ["found"], "residual_golds": {"BEYOND": 2, "OUTSIDE": 0}, "thin_in": []}, basis
    assert basis["C"]["points_in"] == ["placing"] and basis["C"]["residual_golds"] == {"BEYOND": 1, "OUTSIDE": 0}
    m3_none = {"labels": ["A", "B", "C"], "residual": {"BEYOND": {"A": 0, "B": 0, "C": 0, "deeper": 4}, "OUTSIDE": {"A": 0, "B": 0, "C": 0, "deeper": 0}}}
    assert candidates_of(m3_none, m4) == ([], {})                # the residual holds no gold at a layout label: empty
    assert candidates_of({"labels": [], "residual": None}, None) == (None, None)
    assert candidates_of(m3, None) == (None, None)
    names = [f"{d}/{v}" for d, v in ARMS]
    arms = [{"dataset": d, "version": v, "candidates": c, "open_flag": f} for (d, v), c, f in zip(
        ARMS, ([], [], ["F"], None, [], ["A", "C"]), (False, True, False, True, False, True))]
    s = section7_summary(arms)
    assert s["arms"] == 9 and s["graded"] == 6 and s["empty_candidates"] == 3 and s["empty_arms"] == [names[0], names[1], names[4]], s
    assert s["arms_with_candidates"] == [names[2], names[5]] and s["undetermined"] == [names[3]] + names[6:], s
    assert s["outside_flag_arms"] == [names[1], names[3], names[5]] and s["empty_and_flagged"] == [names[1]], s
    assert section7_summary([])["empty_candidates"] == 0 and len(section7_summary([])["undetermined"]) == 9


def report_test():
    """cmd_report refuses when two arms' hashes differ (section 2a), and files INCOMPLETE when an arm is missing."""
    global OUT
    saved = OUT
    try:
        with tempfile.TemporaryDirectory() as td:
            OUT = Path(td)
            h0 = code_hash()[0]
            for (d_, v_), hh in ((("musique", "RK"), h0), (("squad", "RK"), "0" * 64)):
                wjson(OUT / d_ / v_ / "grade.json", {"dataset": d_, "version": v_, "verdict": "GATE_FAIL", "code_hash": hh})
            assert cmd_report(None) == 1 and json.loads((OUT / "report.json").read_text(encoding="utf-8"))["verdict"] == "REFUSED"
            (OUT / "squad" / "RK" / "grade.json").unlink()
            assert cmd_report(None) == 2 and json.loads((OUT / "report.json").read_text(encoding="utf-8"))["verdict"] == "INCOMPLETE"
    finally:
        OUT = saved


def toy_tests():
    """The call and the reach classes on toy arms; the label graph on a toy; no data needed."""
    # reach classes: the page's table
    pos = np.asarray([0, 1, 5, 6, -1, 3, 9, -1, 0])
    keep = np.asarray([5, 4])
    g_ptr = np.asarray([0, 5, 9])
    assert list(reach_classes(pos, keep, g_ptr)) == [0, 1, 1, 2, 3, 1, 2, 3, 0]

    def arm(d, v, recall, hi, pt=1.0):
        return {"dataset": d, "version": v, "verdict": "GRADED",
                "m1": {"call_inputs": {"comparator": "x", "recall_call": recall, "ALL_call": "LEVEL", "d_recall": 0, "d_recall_ci": [0, 0],
                                       "rows_ratio": {"pt": pt, "ci": [pt, hi]}}}}
    ok = [arm("a", "RK", "LEVEL", 1.0), arm("b", "RK", "ABOVE", 0.9)]
    assert call_pools(ok)["call"] == "ADOPT_POOLS"
    assert call_pools(ok + [arm("c", "RK", "BELOW", 0.8)])["call"] == "NOT_ADOPTED"
    assert call_pools(ok + [arm("c", "RK", "ABOVE", 1.2, 1.1)])["call"] == "ADOPT_POOLS"        # a price named by a gain
    assert call_pools(ok + [arm("c", "RK", "LEVEL", 1.2, 1.1)])["call"] == "NOT_ADOPTED"        # a price with no gain
    assert call_pools(ok + [{"dataset": "d", "version": "RK", "verdict": "NO_RECORD"}])["call"] == "INCOMPLETE"
    # the label graph on a toy, both directions
    n = 50
    G = label_graph(n, [("A", np.asarray([1 * n + 2, 3 * n + 4])), ("B", np.asarray([1 * n + 2, 2 * n + 9]))])
    nb, bits = G.nbrs(2)
    assert dict(zip(nb.tolist(), bits.tolist())) == {1: 3, 9: 2}
    assert G.set_name(3) == "A+B" and G.set_name(0) == "deeper"
    # code hash is one value for one code
    assert code_hash()[0] == code_hash()[0]
    log("selftest: toy tests OK")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")
    g = sub.add_parser("grade")
    g.add_argument("--ds", required=True)
    g.add_argument("--version", required=True)
    g.add_argument("--host", action="store_true")
    g.add_argument("--record-dir", dest="record_dir", default=None)
    g.add_argument("--threads", type=int, default=0)
    sub.add_parser("report")
    a = ap.parse_args(argv)
    return {"selftest": lambda: selftest(), "grade": lambda: cmd_grade(a), "report": lambda: cmd_report(a)}[a.cmd]()


if __name__ == "__main__":
    sys.exit(main())
