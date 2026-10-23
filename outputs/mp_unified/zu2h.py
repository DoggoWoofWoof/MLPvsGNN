"""U2h (docs/U2H_W2S_BESIDE.md): W2_S beside today's graph, as an input to both lean models. U2d's base build stays as
filed (today's graph, P1c2's pools); KT_W is P1c3's eight KT columns computed from Y's orders on W2_S (zu2g's carve
opener, then zu2d_pools.question, unchanged) for the base build's rows, in the base caches' row order:
    KT_W     outputs/u2h/ktw/<D>/<carve>/ (ktw.npy, n.npy, ids.json, ktw.json); parts under outputs/u2h/ktw/parts
    arms     zrckpw  zrckp's pointwise head widened to the row's KT and KT_W columns (16 inputs, 289 weights)
             zspkw   zspk's head widened to the KT and KT_W columns and the row's z-scored score (17 inputs, 305)
    fits     scr-u2h-<arm>[-hp]: seed 0 on U2d's card, every carve on U2d's base build, on Modal's L40S
squad, musique, hotpotqa and 2wiki get KT_W from Y's orders; metaqa and webqsp run on the KB in both builds, so their
KT_W is base's KT, byte for byte.

    python outputs/mp_unified/zu2h.py ktw D --carve C [--shard I/K] [--threads T] [--limit Q]
    python outputs/mp_unified/zu2h.py ktw-merge D --carve C
    python outputs/mp_unified/zu2h.py ktw-linked
    python outputs/mp_unified/zu2h.py train --split L-hotpotqa --name scr-u2h-zspkw-hp --arm zspkw --device cuda \\
        --cache-root outputs/u2d/base/cache --modal
    python outputs/mp_unified/zu2h.py read --name scr-u2h-zspkw-hp --device cuda --cache-root outputs/u2d/base/cache \\
        --modal
On the laptop, from the fetched reads (section 5's decision, section 6's readings):
    python outputs/mp_unified/zu2h.py compare-all
    python outputs/mp_unified/zu2h.py decide
    python outputs/mp_unified/zu2h.py --selftest
"""
import os
import sys

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import contextlib  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import zu2g as ZG  # noqa: E402  (Y registered in zu2d_pools: its roots, construction and carve opener)

ZP = ZG.ZP                                   # zu2d_pools (in zp1c3, "ZP" is zprop: here it is ZPROP)
KT3, U2C, PGB, C4S = ZP.KT3, ZP.U2C, ZP.PGB, ZP.C4S
Z = ZG.Zm()                                  # zu2d, under U2d's declaration and builds

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402

Z3, Z3P, Z2, ZM, ZPROP, ZL, RM = Z.Z3, Z.Z3P, Z.Z2, Z.ZM, Z.ZPROP, Z.ZL, Z.RM
S, S2, LG, LC, SR, R, G6, SP = Z.S, Z.S2, Z.LG, Z.LC, Z.SR, Z.R, Z.G6, Z.SP

DECLARED = "docs/U2H_W2S_BESIDE.md"
OUT = ROOT / "outputs" / "u2h"
KTW = OUT / "ktw"
PARTS = KTW / "parts"
SCREEN = ROOT / "outputs" / "screen"
VERDICT = OUT / "screen-verdict"
BUILD = "base"                               # every carve on U2d's base build
U2D_DECLARED = ZG.U2D_DECLARED
KTW_KEY = "_ktw"
KT_COLS = tuple(Z3.KT_COLS)
assert KT_COLS == tuple(KT3.COLS)
KT_KEY, NKT, KT_HID, KT_SEED_OFF, KT, ZCLIP = Z3.KT_KEY, Z3.NKT, Z3.KT_HID, Z3.KT_SEED_OFF, Z3.KT, Z3.ZCLIP
KT_DT = np.float16
NPW_IN = 2 * NKT                             # zrckpw: the KT and KT_W columns
NW_IN = 2 * NKT + 1                          # zspkw: and the row's z-scored score
N_PW = KT_HID * NPW_IN + KT_HID + KT_HID + 1  # 289
N_W = KT_HID * NW_IN + KT_HID + KT_HID + 1    # 305
OWN, LINKED = ZG.OWN, ZG.LINKED              # KT_W from Y's orders; base's KT byte for byte
CARVES_W = ZG.CARVES_Y
LINKED_CARVES = tuple((d, c) for d in LINKED for c in ZP.PC2.carves_for(d))
assert len(LINKED_CARVES) == 3 and set(CARVES_W) | set(LINKED_CARVES) == set(ZP.SCREEN)
ARMS_H = ("zrckpw", "zspkw")
BASE_ARM = {"zrckpw": "zrckp", "zspkw": "zspk"}
NULL_ARM = dict(BASE_ARM)
SPLITS, SUF = ZG.SPLITS, ZG.SUF
GRAPH_W = "W2_S's KT columns (U2b's chosen LLM-free links, P1c2's rule: Y's orders), beside today's graph"
FLOAT16_DEEP = ("float16 places deeper than about 6,985 of P1d's 7,000 read as 1, as a node outside the order does; "
                "the in-order shares below count a row as in KIND_T's order when its column is below 1")
log, utc = ZP.log, ZP.utc


def sha_src(p):
    return hashlib.sha256(Path(p).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def rel(p):
    p = Path(p)
    try:
        return p.relative_to(ROOT).as_posix()
    except ValueError:
        return p.as_posix()


def fit_name(arm, split):
    return f"scr-u2h-{arm}{SUF[split]}"


FITS = {fit_name(a, sp): (a, sp) for a in ARMS_H for sp in SPLITS}
assert len(FITS) == 4


# ── KT_W: one shard of a carve (section 2) ───────────────────────────────────


def ktw_question(q, y_B, y_exp, y_kt, b_B, b_exp, who):
    """One question's KT_W rows: P1c3's columns from Y's recomputed orders, over base's pool (sorted by node), fill = Y's
    expansion, entity = Y's kind. Stops unless B is Y's and base's, the expansion is Y's filed one, and Y's own pool's
    columns from these orders are Y's filed KT bit for bit. Returns (rows, counts)."""
    B = np.asarray(q["B"], np.int64)
    if not np.array_equal(B, np.asarray(y_B, np.int64)):
        raise SystemExit(f"zu2h: {who}: B is not Y's")
    if not np.array_equal(B, np.asarray(b_B, np.int64)):
        raise SystemExit(f"zu2h: {who}: B is not base's")
    y_exp = np.asarray(y_exp, np.int64)
    if not np.array_equal(np.asarray(q["exp"], np.int64), y_exp):
        raise SystemExit(f"zu2h: {who}: the expansion on W2_S is not Y's filed one")
    y_pool = np.union1d(B, y_exp)
    y_rows = KT3.columns(q["orders"], y_pool, y_exp, q["entity"]).astype(KT_DT)
    y_kt = np.asarray(y_kt)
    if y_kt.dtype != KT_DT or y_rows.shape != y_kt.shape or not np.array_equal(y_rows.view(np.uint16),
                                                                                 y_kt.view(np.uint16)):
        raise SystemExit(f"zu2h: {who}: Y's rows from the recomputed orders are not Y's filed KT columns bit for bit")
    b_exp = np.asarray(b_exp, np.int64)
    pool = np.union1d(B, b_exp)
    if pool.size != B.size + b_exp.size:
        raise SystemExit(f"zu2h: {who}: base's expansion repeats a node or holds one of B")
    rows = KT3.columns(q["orders"], pool, y_exp, q["entity"]).astype(KT_DT)
    counts = {"rows": int(pool.size), "y_rows": int(y_pool.size), "in_y_pool": int(np.isin(pool, y_pool).sum()),
              "in_kind_t_w2s": int((KT3.place(q["orders"]["KIND_T"], pool) >= 0).sum()),
              "fill_w2s": int(np.isin(pool, y_exp).sum())}
    return rows, counts


COUNT_KEYS = ("rows", "y_rows", "in_y_pool", "in_kind_t_w2s", "fill_w2s", "questions", "entity_differs_base",
              "route_differs_base")


def y_part(name, carve, shard):
    """Y's filed part of the shard (zu2g.py pools): its record (U2g's, P1c2's rule on W2_S, every check held, the files
    its record names), its manifest entry, its arrays and its KT columns. Returns (record, arrays, kt, record sha256,
    manifest sha256)."""
    i, k = shard
    stem = ZP.stem_of(name, carve, shard)
    ry = ZG.roots_y(ZG.BUILD)
    p = ry["parts"] / f"{stem}.json"
    if not p.is_file():
        raise SystemExit(f"zu2h: {rel(p)} missing (Y's part; zu2g.py pools {name} --carve {carve})")
    rec = json.loads(p.read_text(encoding="utf-8"))
    got = tuple(rec.get(x) for x in ("declared_in", "stage", "build", "construction", "dataset", "carve",
                                     "script_sha256", "part", "parts"))
    want = (ZG.DECLARED, "U2d", ZG.BUILD, ZG.CONSTRUCTION_Y, name, carve, ZP.script_sha256(), i, k)
    if got != want:
        raise SystemExit(f"zu2h: {rel(p)} is not Y's part {i}/{k} of {name}/{carve} ({got})")
    nq = int(rec["questions_all"])
    if (rec.get("q0"), rec.get("q1")) != U2C.shard_range(nq, i, k):
        raise SystemExit(f"zu2h: {rel(p)} is not the {i}-th of {k} question ranges")
    if not rec.get("checks") or not all(rec["checks"].values()):
        raise SystemExit(f"zu2h: {rel(p)} holds a failed check: {rec.get('checks')}")
    for key, suf in (("npz", ".npz"), ("kt", ".kt.npy")):
        f = ry["parts"] / f"{stem}{suf}"
        if not f.is_file() or ZP.sha_file(f) != rec["files"].get(key):
            raise SystemExit(f"zu2h: {f.name}: not its Y part record's file")
    mp = ry["pools"] / name / "pools.json"
    m = json.loads(mp.read_text(encoding="utf-8")) if mp.is_file() else {}
    if (m.get("declared_in"), m.get("build"), m.get("construction")) != (ZG.DECLARED, ZG.BUILD, ZG.CONSTRUCTION_Y):
        raise SystemExit(f"zu2h: {rel(mp)} is not Y's manifest of {name}")
    ent = (m.get("files") or {}).get(f"{name}__{carve}") or {}
    parts = ent.get("parts") or []
    mine = [x for x in parts if x.get("stem") == stem]
    if len(parts) != k or len(mine) != 1 or (mine[0].get("npz"), mine[0].get("kt")) != (rec["files"]["npz"],
                                                                                        rec["files"]["kt"]):
        raise SystemExit(f"zu2h: {rel(mp)}: {name}/{carve}'s entry does not name this part once among {k}")
    with np.load(ry["parts"] / f"{stem}.npz") as z:
        ya = {x: z[x] for x in z.files}
    ykt = np.load(ry["parts"] / f"{stem}.kt.npy")
    rows = int(PGB.expected_rows(ya).sum())
    if ykt.dtype != KT_DT or ykt.shape != (rows, NKT) or rows != int(rec["rows"]):
        raise SystemExit(f"zu2h: {stem}.kt.npy is {ykt.dtype} {ykt.shape}, not float16 ({rows}, {NKT})")
    return rec, ya, ykt, ZP.sha_file(p), ZP.sha_file(mp)


def ktw_part(name, carve, threads=None, limit=0, shard=None):
    """KT_W of one shard: Y's carve opened as zu2g opened it (W2_S swapped in, P1c2's rule), each question's orders
    from zu2d_pools.question, the columns over base's pool. Y's shard and threads (its part record's). --limit Q: the
    shard's first Q questions, every check, nothing written."""
    ZG.own(name, carve)
    i, k = shard or (0, 1)
    stem = ZP.stem_of(name, carve, (i, k))
    rp = PARTS / f"{stem}.json"
    if rp.exists() and not limit:
        raise SystemExit(f"zu2h: {rel(rp)} exists; U2h never overwrites a KT_W part")
    t0 = time.time()
    yrec, ya, ykt, ysha, ymsha = y_part(name, carve, (i, k))
    threads = int(yrec["threads"]) if threads is None else int(threads)
    snsha = ZG.same_as_new_ok(name)
    filed, ent, msha = ZP.load_filed(BUILD, name, carve)
    e2 = ZP.p1c2_entry(name, carve)
    cx = ZP.open_carve(ZG.BUILD, name, carve, threads)
    nq = len(cx.bs)
    ids = [str(q) for q in cx.pop.ids]
    if e2["carve_ids_sha256"] != cx.pop.digest or int(e2["questions"]) != nq:
        raise SystemExit(f"zu2h: {name}/{carve}: not P1c2's questions")
    if (yrec.get("carve_ids_sha256"), int(yrec["questions_all"])) != (cx.pop.digest, nq):
        raise SystemExit(f"zu2h: {name}/{carve}: Y's part holds other questions than the carve")
    if json.loads(json.dumps(cx.desc)) != yrec.get("graph_detail"):
        raise SystemExit(f"zu2h: {name}/{carve}: the graph is not the one Y's pools ran on")
    if [str(q) for q in filed["ids"].tolist()] != ids:
        raise SystemExit(f"zu2h: {name}/{carve}: base's pools file holds other questions than the carve")
    yq0, q1 = int(yrec["q0"]), int(yrec["q1"])
    if [str(q) for q in ya["ids"].tolist()] != ids[yq0:q1]:
        raise SystemExit(f"zu2h: {name}/{carve}: Y's part holds other questions than its range")
    if limit:
        q1 = min(q1, yq0 + limit)
    log(f"u2h ktw {name}/{carve} part {i}/{k}: prepared ({time.time() - t0:.0f}s); questions {yq0}-{q1 - 1} of {nq}")
    yp = ZP.ptr_of(PGB.expected_rows(ya))
    bp, ep = filed["bs_ptr"], filed["ptr"]
    out = []
    counts = {x: 0 for x in COUNT_KEYS}
    S4, K, D = cx.S4, cx.K, cx.D
    t = time.time()
    for lo in range(yq0, q1, S4.BATCH):
        js = list(range(lo, min(q1, lo + S4.BATCH)))
        srcs = D.sources_batch(cx.prep, js, cx.linked, cx.unions, cx.constant, threads)
        steered = K.steered_batch(cx.prep, js, cx.linked, cx.unions, srcs, cx.constant, threads)
        for n_, j in enumerate(js):
            q = ZP.question(cx, j, srcs[n_], steered[n_])
            u = j - yq0
            who = f"{name}/{carve} question {j} ({ids[j]})"
            inc = np.asarray(q["inc"], np.int64)
            sha = ya["inc_sha"][u]
            sha = sha.decode() if isinstance(sha, bytes) else str(sha)
            if inc.size != int(ya["inc_size"][u]) or C4S.pool_sha(inc) != sha:
                raise SystemExit(f"zu2h: {who}: I_q on W2_S is not Y's filed one")
            if bool(q["entity"]) != bool(ya["entity"][u]) or bool(q["route"]) != bool(ya["route"][u]):
                raise SystemExit(f"zu2h: {who}: the kind or route is not Y's filed one")
            rows, c = ktw_question(q, ya["bs_nodes"][ya["bs_ptr"][u]:ya["bs_ptr"][u + 1]],
                                   ya["nodes"][ya["ptr"][u]:ya["ptr"][u + 1]], ykt[yp[u]:yp[u + 1]],
                                   filed["bs_nodes"][bp[j]:bp[j + 1]], filed["nodes"][ep[j]:ep[j + 1]], who)
            out.append(rows)
            for x, v in c.items():
                counts[x] += v
            counts["questions"] += 1
            counts["entity_differs_base"] += int(bool(q["entity"]) != bool(filed["entity"][j]))
            counts["route_differs_base"] += int(bool(q["route"]) != bool(filed["route"][j]))
        if (lo // S4.BATCH) % 10 == 9:
            log(f"u2h ktw {name}/{carve}: {js[-1] + 1 - yq0}/{q1 - yq0} ({time.time() - t:.0f}s)")
    if limit:
        log(f"u2h ktw {name}/{carve} development run: questions {yq0}-{q1 - 1}, every check held "
            f"({time.time() - t:.0f}s); nothing written")
        return 0
    ktw = np.concatenate(out) if out else np.zeros((0, NKT), KT_DT)
    want = int(PGB.expected_rows(filed)[yq0:q1].sum())
    if ktw.shape != (want, NKT) or counts["rows"] != want:
        raise SystemExit(f"zu2h: {name}/{carve} part {i}/{k}: {ktw.shape[0]} rows, base's pools hold {want}")
    PARTS.mkdir(parents=True, exist_ok=True)
    fsha = ZP.save_npy(PARTS / f"{stem}.ktw.npy", ktw)
    ZP.write_json(rp, {
        "declared_in": DECLARED, "stage": "U2h", "build": BUILD, "dataset": name, "carve": carve, "part": i, "parts": k,
        "q0": yq0, "q1": q1, "questions_all": nq, "rows": int(ktw.shape[0]), "columns": list(KT_COLS),
        "dtype": "float16", "files": {"ktw": fsha},
        "y": {"record": rel(ZG.roots_y(ZG.BUILD)["parts"] / f"{stem}.json"), "record_sha256": ysha,
              "manifest_sha256": ymsha, "files": yrec["files"], "same_as_new_sha256": snsha,
              "construction": ZG.CONSTRUCTION_Y},
        "base": {"manifest_sha256": msha, "file_sha256": ent["sha256"], "construction": ent["construction"]},
        "checks": {"B_is_y_and_base": True, "expansion_is_y": True, "y_kt_bit_for_bit": True, "inc_is_y": True,
                   "kind_route_are_y": True, "questions_are_y_and_base": True, "graph_is_y": True},
        "counts": counts, "graph_detail": cx.desc, "carve_ids_sha256": cx.pop.digest, "script_sha256": sha_src(__file__),
        "zu2g_sha256": sha_src(ZG.__file__), "zu2d_pools_sha256": ZP.script_sha256(),
        "p1c3_sha256": KT3.script_sha256(), "p1e_sha256": K.script_sha256(), "threads": threads,
        "seconds": round(time.time() - t0, 1), "utc": utc()})
    log(f"u2h ktw {name}/{carve} part {i}/{k} (questions {yq0}-{q1 - 1}, {ktw.shape[0]} rows, "
        f"{time.time() - t0:.0f}s): written")
    return 0


# ── KT_W in the base caches' row order ───────────────────────────────────────


def base_kt(name, carve):
    """U2d's base KT record of the carve (zu2d_pools.py kt --build base), its arrays and ids by their sha256."""
    d = ZP.roots(BUILD)["kt"] / name / carve
    rp = d / "record.json"
    if not rp.is_file():
        raise SystemExit(f"zu2h: {rel(rp)} missing (zu2d_pools.py kt --build base {name} --carve {carve})")
    rec = json.loads(rp.read_text(encoding="utf-8"))
    got = tuple(rec.get(x) for x in ("declared_in", "build", "columns", "dataset", "carve"))
    if got != (U2D_DECLARED, BUILD, list(KT3.COLS), name, carve):
        raise SystemExit(f"zu2h: {rel(rp)} is not U2d's KT record of {name}/{carve} on the base build ({got})")
    for x in ("kt", "n"):
        if ZP.sha_file(d / f"{x}.npy") != rec["arrays"][x]["sha256"]:
            raise SystemExit(f"zu2h: {rel(d / f'{x}.npy')}: sha256 is not its KT record's")
    if ZP.sha_file(d / "ids.json") != rec.get("ids_sha256"):
        raise SystemExit(f"zu2h: {rel(d / 'ids.json')}: sha256 is not its KT record's")
    return rec, d, ZP.sha_file(rp)


def beside_kt(kt, ktw):
    """Beside, per carve: the base rows in KIND_T's order on each graph, how often their places differ, fill and kind."""
    a, b = np.asarray(kt, np.float32), np.asarray(ktw, np.float32)
    n = int(a.shape[0])

    def share(x):
        return round(float(x) / n, 4) if n else None

    t, w = a[:, 0] < 1.0, b[:, 0] < 1.0
    both = t & w
    return {"rows": n, "in_kind_t": {"today": int(t.sum()), "w2s": int(w.sum()), "both": int(both.sum())},
            "share_in_kind_t": {"today": share(t.sum()), "w2s": share(w.sum()), "both": share(both.sum())},
            "share_place_differs": {c: share((a[:, x] != b[:, x]).sum()) for x, c in enumerate(KT3.ORDERED)},
            "share_kind_t_place_differs_where_in_both": round(float((a[both, 0] != b[both, 0]).mean()), 4)
            if both.any() else None,
            "share_fill": {"today": share(a[:, 6].sum()), "w2s": share(b[:, 6].sum()),
                           "both": share(((a[:, 6] > 0) & (b[:, 6] > 0)).sum())},
            "rows_kind_differs": int((a[:, 7] != b[:, 7]).sum()), "caveat": FLOAT16_DEEP}


def ktw_merge(name, carve):
    """The carve's KT_W parts joined (every part U2h's, on one Y manifest and one base pools file) and put in the base
    caches' row order (zu2d_pools.to_cache_order: each question's cache rows are its base pool, each node once); the
    questions, their order and the rows per question are base's KT record's. Never over a filed one."""
    ZG.own(name, carve)
    d = KTW / name / carve
    rp = d / "ktw.json"
    if rp.exists():
        raise SystemExit(f"zu2h: {rel(rp)} exists; U2h never overwrites a KT_W input")
    t0 = time.time()
    filed, ent, msha = ZP.load_filed(BUILD, name, carve)
    recs = ZP.find_parts(PARTS, name, carve)
    k = len(recs)
    mp = ZG.roots_y(ZG.BUILD)["pools"] / name / "pools.json"
    m = json.loads(mp.read_text(encoding="utf-8")) if mp.is_file() else {}
    ny = len(((m.get("files") or {}).get(f"{name}__{carve}") or {}).get("parts") or [])
    if ny != k:
        raise SystemExit(f"zu2h: {name}/{carve}: {k} KT_W parts, Y's manifest names {ny}")
    ymsha = ZP.sha_file(mp)
    sizes = PGB.expected_rows(filed)
    flat, counts, parts = [], {x: 0 for x in COUNT_KEYS}, []
    for i, r in enumerate(recs):
        got = tuple(r.get(x) for x in ("declared_in", "stage", "build", "dataset", "carve", "part", "parts",
                                       "script_sha256"))
        if got != (DECLARED, "U2h", BUILD, name, carve, i, k, sha_src(__file__)):
            raise SystemExit(f"zu2h: {name}/{carve} part {i}: not U2h's part {i}/{k} by this zu2h.py ({got})")
        nq = int(r["questions_all"])
        if nq != int(filed["ids"].size) or (r["q0"], r["q1"]) != U2C.shard_range(nq, i, k):
            raise SystemExit(f"zu2h: {name}/{carve} part {i}: not the {i}-th of {k} question ranges")
        if not all(r["checks"].values()) or (r["base"]["manifest_sha256"], r["base"]["file_sha256"]) != (
                msha, ent["sha256"]) or r["y"]["manifest_sha256"] != ymsha:
            raise SystemExit(f"zu2h: {name}/{carve} part {i}: a failed check, or other pools than base's and Y's")
        stem = ZP.stem_of(name, carve, (i, k))
        f = PARTS / f"{stem}.ktw.npy"
        if ZP.sha_file(f) != r["files"]["ktw"]:
            raise SystemExit(f"zu2h: {f.name}: not its record's file")
        x = np.load(f)
        if x.dtype != KT_DT or x.shape != (int(r["rows"]), NKT) or int(r["rows"]) != int(sizes[r["q0"]:r["q1"]].sum()):
            raise SystemExit(f"zu2h: {f.name} is {x.dtype} {x.shape}, not float16 over base's pools of its range")
        flat.append(x)
        for c in COUNT_KEYS:
            counts[c] += int(r["counts"][c])
        parts.append({"stem": stem, "q0": r["q0"], "q1": r["q1"], "rows": r["rows"], "ktw": r["files"]["ktw"]})
    flat = np.concatenate(flat) if flat else np.zeros((0, NKT), KT_DT)
    if flat.shape[0] != int(sizes.sum()):
        raise SystemExit(f"zu2h: {name}/{carve}: the parts hold {flat.shape[0]} rows, base's pools {int(sizes.sum())}")
    krec, kd, ksha = base_kt(name, carve)
    croot = ZP.roots(BUILD)["cache"]
    c_ids, c_n, c_node, part_shas = KT3.cache_rows(croot, name, carve)
    if part_shas != krec.get("cache_parts") or rel(croot) != krec.get("cache_root"):
        raise SystemExit(f"zu2h: {name}/{carve}: base's KT record sits on other cache parts or another root")
    if (krec.get("pools_manifest_sha256"), krec.get("pools_file_sha256")) != (msha, ent["sha256"]):
        raise SystemExit(f"zu2h: {name}/{carve}: base's KT record was written on other pools")
    if [str(q) for q in json.loads((kd / "ids.json").read_text(encoding="utf-8"))] != list(c_ids):
        raise SystemExit(f"zu2h: {name}/{carve}: base's KT record holds other questions than the cache")
    ktw = ZP.to_cache_order(filed, flat, c_ids, c_n, c_node)
    kt = np.load(kd / "kt.npy")
    if ktw.shape != kt.shape or ktw.dtype != kt.dtype:
        raise SystemExit(f"zu2h: {name}/{carve}: KT_W is {ktw.dtype} {ktw.shape}, base's KT {kt.dtype} {kt.shape}")
    d.mkdir(parents=True, exist_ok=True)
    asha = ZP.save_npy(d / "ktw.npy", ktw)
    nsha = ZP.save_npy(d / "n.npy", np.asarray(c_n, np.int32))
    ZP.write_json(d / "ids.json", list(c_ids))
    isha = ZP.sha_file(d / "ids.json")
    if (nsha, isha) != (krec["arrays"]["n"]["sha256"], krec["ids_sha256"]):
        raise SystemExit(f"zu2h: {name}/{carve}: the rows per question or the questions are not base's KT record's")
    bs = beside_kt(kt, ktw)
    ZP.write_json(rp, {
        "declared_in": DECLARED, "stage": "U2h", "build": BUILD, "columns_declared_in": KT3.DECLARED,
        "columns": list(KT_COLS), "nmax": KT3.NMAX, "dtype": "float16",
        "arrays": {"ktw": {"sha256": asha, "shape": list(ktw.shape)}, "n": {"sha256": nsha, "shape": [len(c_ids)]}},
        "dataset": name, "carve": carve, "questions": len(c_ids), "rows": int(ktw.shape[0]), "ids_sha256": isha,
        "cache_root": krec["cache_root"], "cache_parts": part_shas, "pools_manifest_sha256": msha,
        "pools_file_sha256": ent["sha256"], "construction": ent["construction"], "base_kt_record_sha256": ksha,
        "source": "Y's orders on W2_S (P1c2's rule) over the base build's rows", "y_manifest_sha256": ymsha,
        "y_construction": ZG.CONSTRUCTION_Y, "parts": parts, "counts": counts, "beside": bs,
        "checks": {"rows_are_pool": True, "parts_sha256": True, "n_and_ids_are_base_kts": True},
        "script_sha256": sha_src(__file__), "p1c3_script_sha256": KT3.script_sha256(),
        "seconds": round(time.time() - t0, 1), "utc": utc()})
    log(f"u2h ktw-merge {name}/{carve}: {len(c_ids)} questions, {ktw.shape[0]} rows in the cache's order; in KIND_T's "
        f"order today {bs['share_in_kind_t']['today']}, on W2_S {bs['share_in_kind_t']['w2s']}")
    return 0


def _copy(src, dst):
    tmp = dst.with_name(dst.name + ".tmp")
    tmp.write_bytes(Path(src).read_bytes())
    os.replace(tmp, dst)
    return ZP.sha_file(dst)


def ktw_linked():
    """metaqa's and webqsp's KT_W: base's KT, byte for byte (the KB in both builds); a filed one is kept."""
    done = []
    for name, carve in LINKED_CARVES:
        d = KTW / name / carve
        if (d / "ktw.json").exists():
            log(f"u2h ktw-linked {name}/{carve}: filed before, kept")
            continue
        krec, kd, ksha = base_kt(name, carve)
        d.mkdir(parents=True, exist_ok=True)
        got = (_copy(kd / "kt.npy", d / "ktw.npy"), _copy(kd / "n.npy", d / "n.npy"), _copy(kd / "ids.json", d / "ids.json"))
        if got != (krec["arrays"]["kt"]["sha256"], krec["arrays"]["n"]["sha256"], krec["ids_sha256"]):
            raise SystemExit(f"zu2h: {name}/{carve}: the copy is not base's KT byte for byte")
        ZP.write_json(d / "ktw.json", {
            "declared_in": DECLARED, "stage": "U2h", "build": BUILD, "columns_declared_in": krec.get("columns_declared_in"),
            "columns": list(KT_COLS), "nmax": krec.get("nmax"), "dtype": "float16",
            "arrays": {"ktw": dict(krec["arrays"]["kt"]), "n": dict(krec["arrays"]["n"])}, "dataset": name,
            "carve": carve, "questions": krec.get("questions"), "rows": krec.get("rows"), "ids_sha256": krec["ids_sha256"],
            "cache_root": krec.get("cache_root"), "cache_parts": krec.get("cache_parts"),
            "pools_manifest_sha256": krec.get("pools_manifest_sha256"), "pools_file_sha256": krec.get("pools_file_sha256"),
            "construction": krec.get("construction"), "base_kt_record_sha256": ksha,
            "source": "base's KT, byte for byte: metaqa and webqsp run on the KB in both builds",
            "script_sha256": sha_src(__file__), "utc": utc()})
        done.append(f"{name}/{carve}")
    log(f"u2h ktw-linked: {', '.join(done) or 'nothing'} copied")
    return 0


# ── KT_W beside a carve, and the arms (section 3) ────────────────────────────


def load_ktw(c, root, cache_root, verify=True):
    """KT_W of c's dataset and carve on c's device as float16: U2h's, beside the KT record c loaded, on c's cache parts
    and root, for c's questions in c's order, with c's rows, every value finite."""
    d = Path(root) / c.ds / c.carve
    rp = d / "ktw.json"
    if not rp.exists():
        raise SystemExit(f"zu2h: {rp} missing (zu2h.py ktw-merge {c.ds} --carve {c.carve}, or ktw-linked)")
    rec = json.loads(rp.read_text(encoding="utf-8"))
    got = tuple(rec.get(x) for x in ("declared_in", "stage", "columns", "build"))
    if got != (DECLARED, "U2h", list(KT_COLS), BUILD):
        raise SystemExit(f"zu2h: {rp} is not U2h's KT_W input ({got})")
    if rec.get("base_kt_record_sha256") != getattr(c, "kt_record", None):
        raise SystemExit(f"zu2h: {rp} was not written beside this carve's KT record")
    if verify:
        for k, f in (("ktw", "ktw.npy"), ("n", "n.npy")):
            if LC.sha_file(d / f) != rec["arrays"][k]["sha256"]:
                raise SystemExit(f"zu2h: {d / f}: sha256 is not its KT_W record's")
    if rec.get("cache_parts") != c.part_shas:
        raise SystemExit(f"zu2h: {rp} was written on other cache parts than {c.ds}/{c.carve}'s")
    if Path(cache_root).resolve() != (ROOT / rec.get("cache_root", "")).resolve():
        raise SystemExit(f"zu2h: {rp} was written on the cache root {rec.get('cache_root')}, not {cache_root}")
    n = np.load(d / "n.npy").astype(np.int64)
    if not np.array_equal(n, c.n_np):
        raise SystemExit(f"zu2h: {d / 'n.npy'}: not {c.ds}/{c.carve}'s rows per question")
    if LC.sha_file(d / "ids.json") != rec.get("ids_sha256"):
        raise SystemExit(f"zu2h: {d / 'ids.json'}: sha256 is not its KT_W record's")
    ids = [str(q) for q in json.loads((d / "ids.json").read_text(encoding="utf-8"))]
    if getattr(c, "ids", None) is not None and ids != [str(q) for q in c.ids]:
        raise SystemExit(f"zu2h: {d / 'ids.json'}: not {c.ds}/{c.carve}'s questions in its order")
    ktw = np.load(d / "ktw.npy")
    if ktw.dtype != np.float16 or ktw.shape != (int(c.off_np[-1]), NKT):
        raise SystemExit(f"zu2h: {d / 'ktw.npy'} is {ktw.dtype} {ktw.shape}, not float16 ({int(c.off_np[-1])}, {NKT})")
    if not bool(np.isfinite(ktw).all()):
        raise SystemExit(f"zu2h: {d / 'ktw.npy'} holds a value that is not finite")
    return torch.from_numpy(ktw).to(c.device), LC.sha_file(rp)


def ktw_batch(c, qs):
    """The KT_W rows of questions qs, in lean_gpu.CacheCarve.batch's row order, float32 on c's device."""
    qs = np.asarray(qs, np.int64)
    cnt = c.n_np[qs]
    idx = np.repeat(c.off_np[qs] - (np.cumsum(cnt) - cnt), cnt) + np.arange(int(cnt.sum()))
    return c.ktw[torch.from_numpy(idx).to(c.device)].to(torch.float32)


def ktw_carve(base, root=None):
    """A KT carve class with KT_W beside it, read from KTW_ROOT (U2h's KTW when None): batch() adds it under KTW_KEY;
    every block, row, side key, KT row and gold is base's, unchanged."""

    class KTWCarve(base):
        KTW_ROOT = root

        def __init__(self, ds, carve, basis, device, cache_root=LC.OUT, verify=True, score2=False):
            super().__init__(ds, carve, basis, device, cache_root, verify, score2)
            self.ktw, self.ktw_record = load_ktw(self, type(self).KTW_ROOT or KTW, cache_root, verify)

        def nbytes(self):
            return super().nbytes() + self.ktw.numel() * self.ktw.element_size()

        def batch(self, qs, blocks):
            feats, nq, base_z, gold = super().batch(qs, blocks)
            feats[KTW_KEY] = ktw_batch(self, qs)
            return feats, nq, base_z, gold

    KTWCarve.__name__ = KTWCarve.__qualname__ = f"KTW{base.__name__}"
    return KTWCarve


KTWChainCarveZRC = ktw_carve(Z3.KTChainCarveZRC)
KTWLinkCarveBase = ktw_carve(Z3.KTLinkCarveBase)


class KTMixPW(ZM.ZRM):
    """zrm's score plus zrckp's pointwise head (zero at the start) widened to the row's own KT and KT_W columns."""

    def __init__(self, blocks, widths, hidden=128, dropout=0.1, seed=0, arm="ctl", ctx="none"):
        super().__init__(blocks, widths, hidden, dropout, seed, arm, ctx)
        g = torch.Generator().manual_seed(int(seed) + KT_SEED_OFF)
        self.kt_w1 = nn.Parameter((torch.rand(KT_HID, NPW_IN, generator=g) * 2 - 1) / NPW_IN ** 0.5)
        self.kt_b1 = nn.Parameter(torch.zeros(KT_HID))
        self.kt_w2 = nn.Parameter(torch.zeros(1, KT_HID))
        self.kt_b2 = nn.Parameter(torch.zeros(1))

    def kt_head(self, x):
        return Fn.linear(Fn.gelu(Fn.linear(x, self.kt_w1, self.kt_b1)), self.kt_w2, self.kt_b2).squeeze(-1)

    def forward(self, feats, keep, nq, B, base_z):
        s = super().forward(feats, keep, nq, B, base_z)
        if KT_KEY not in feats:
            return s
        if KTW_KEY not in feats:
            raise SystemExit("zu2h: a KT_W arm's batch holds the KT input without the KT_W input")
        return s + self.kt_head(torch.cat([feats[KT_KEY], feats[KTW_KEY]], 1))


class ZRCKPW(KTMixPW):
    """zrckpw: zrc's model (zrm.ZRM) plus the pointwise head over the KT and KT_W columns (the MLP: no score input)."""


class KTMixW(ZM.ZRM):
    """zrm's score plus zspk's head (zero at the start) widened to the row's KT and KT_W columns and its z-scored
    score."""

    def __init__(self, blocks, widths, hidden=128, dropout=0.1, seed=0, arm="ctl", ctx="none"):
        super().__init__(blocks, widths, hidden, dropout, seed, arm, ctx)
        g = torch.Generator().manual_seed(int(seed) + KT_SEED_OFF)
        self.kt_w1 = nn.Parameter((torch.rand(KT_HID, NW_IN, generator=g) * 2 - 1) / NW_IN ** 0.5)
        self.kt_b1 = nn.Parameter(torch.zeros(KT_HID))
        self.kt_w2 = nn.Parameter(torch.zeros(1, KT_HID))
        self.kt_b2 = nn.Parameter(torch.zeros(1))

    def kt_head(self, x):
        return Fn.linear(Fn.gelu(Fn.linear(x, self.kt_w1, self.kt_b1)), self.kt_w2, self.kt_b2).squeeze(-1)

    @staticmethod
    @torch.no_grad()
    def kt_inputs(s, kt, ktw, nq, B):
        z = LG.seg_zscore8D(s.detach().unsqueeze(1), nq, B).squeeze(1).clamp(-ZCLIP, ZCLIP)
        return torch.cat([kt, ktw, z.unsqueeze(1)], 1)

    def forward(self, feats, keep, nq, B, base_z):
        s = super().forward(feats, keep, nq, B, base_z)
        if KT_KEY not in feats:
            return s
        if KTW_KEY not in feats:
            raise SystemExit("zu2h: a KT_W arm's batch holds the KT input without the KT_W input")
        return s + self.kt_head(self.kt_inputs(s, feats[KT_KEY], feats[KTW_KEY], nq, B))


class ZSPKW(ZPROP.ZProp, KTMixW):
    """zspkw: zsp's model plus the widened head, added before zsp's propagation reads the score (MRO: ZProp, KTMixW,
    ZRM)."""


ARMS = {"zrckpw": (ZRCKPW, KTWChainCarveZRC), "zspkw": (ZSPKW, KTWLinkCarveBase)}
for _a in ARMS_H:                            # za6lw.py names its own arms zrckpw and zspkw
    if S.ARMS.get(_a) not in (None, ARMS[_a]):
        raise SystemExit(f"zu2h: lean_screen's arm {_a} is already {S.ARMS[_a]}; U2h's arms cannot share a process "
                         f"with it")
S.ARMS.update(ARMS)


def head_of(arm):
    kt_root = rel(Z.roots(BUILD)["kt"])
    cols = [*KT_COLS, *(f"{c} on W2_S" for c in KT_COLS)]
    if arm == "zrckpw":
        return {"columns": cols, "inputs": NPW_IN, "score_input": False, "hidden": KT_HID, "seed_offset": KT_SEED_OFF,
                "parameters": N_PW, "kt_root": kt_root, "ktw_root": rel(KTW)}
    if arm == "zspkw":
        return {"columns": cols + ["z-scored score"], "inputs": NW_IN, "score_input": True, "z_clip": ZCLIP,
                "hidden": KT_HID, "seed_offset": KT_SEED_OFF, "parameters": N_W, "kt_root": kt_root,
                "ktw_root": rel(KTW)}
    raise SystemExit(f"zu2h: no arm {arm}")


def ktw_records(carves, root=None):
    """Each carve's KT_W record sha256; stops when one is missing or not U2h's."""
    out = {}
    for ds, cv in carves:
        rp = Path(root or KTW) / ds / cv / "ktw.json"
        if not rp.exists():
            raise SystemExit(f"zu2h: {rp} missing (zu2h.py ktw-merge {ds} --carve {cv}, or ktw-linked)")
        rec = json.loads(rp.read_text(encoding="utf-8"))
        if (rec.get("declared_in"), rec.get("stage"), rec.get("build"), rec.get("columns")) != (
                DECLARED, "U2h", BUILD, list(KT_COLS)):
            raise SystemExit(f"zu2h: {rp} is not U2h's KT_W input")
        out[f"{ds}/{cv}"] = LC.sha_file(rp)
    return out


# ── train and read (section 4) ───────────────────────────────────────────────


def code_shas():
    return {**Z.code_shas(), "zu2h_sha256": sha_src(__file__), "zu2g_sha256": sha_src(ZG.__file__)}


def where_ok(where, env, what):
    if not Z.ok_where(where, env):
        raise SystemExit(f"zu2h: U2h {what}s on Modal's {Z.GPU} from the image {Z.IMAGE_TAG} (section 4), not on "
                         f"{where} with {Z.device_of(env)}")


def train(argv, split, where="laptop", env=None):
    """One of U2h's four fits: its arm and split from its name, seed 0, U2d's card on the GPU, every carve on U2d's base
    build with KT_W beside its KT; the base gate PASS; stamped with the device it ran on."""
    arm = R.arm_of(argv)
    name, _ = S2.where(argv)
    if name not in FITS:
        raise SystemExit(f"zu2h: {name} is not one of U2h's fits (scr-u2h-<arm>[-hp], section 4)")
    farm, fsplit = FITS[name]
    if arm != farm or split != fsplit:
        raise SystemExit(f"zu2h: {name} is {farm}'s fit of {fsplit}, not {arm}'s of {split}")
    if tuple(Z.SPLITS) != SPLITS:
        raise SystemExit(f"zu2h: zu2d's splits are {Z.SPLITS}, not {SPLITS}")
    if S.ARMS.get(arm) != ARMS[arm]:
        raise SystemExit(f"zu2h: the arm {arm} is {S.ARMS.get(arm)}, not U2h's {ARMS[arm]}")
    bad = G6.fixed_flags(argv, Z.REFUSED_TRAIN)
    if bad:
        raise SystemExit(f"zu2h: U2h trains on U2d's card (seed 0, hidden {LG.HIDDEN}, config {LG.CONFIG}, variant p, "
                         f"the split's fit carves and basis, threads 2, outputs/screen/fits); {bad} would change that")
    env = Z.device_env() if env is None else env
    where_ok(where, env, "train")
    with Z.on_u2d(BUILD) as r:
        Z2.cache_root_of(argv)
        argv = Z2.on_card(argv) + ["--seed", "0"]
        g = Z.gate_passed(BUILD)
        kt = Z3.kt_records(Z3.train_carves(split), r["kt"])
        ktw = ktw_records(Z3.train_carves(split))
        rc = RM.train(argv, split)
        stamp_fit(argv, name, env, g, kt, ktw)
    return rc


def stamp_fit(argv, name, env, g, kt, ktw):
    """The fit's screen.json stamped with U2h's record; called inside on_u2d (the chains' and links' records)."""
    arm, split = FITS[name]
    _, out_root = S2.where(argv)
    sj = Path(out_root) / name / "screen.json"
    if not sj.exists():
        return
    rec = json.loads(sj.read_text(encoding="utf-8"))
    r = Z.roots(BUILD)
    rec.update({**code_shas(), "u2h": {
        "declared_in": DECLARED, "fit": name, "build": BUILD, "arm": arm, "base_arm": BASE_ARM[arm], "seed": 0,
        "split": split, "device": Z.device_of(env), "env": env, "graph": Z.GRAPH[BUILD], "beside": GRAPH_W,
        "pools": g.get("construction"), "rule": ZP.RULE[BUILD], "cache_root": rel(r["cache"]),
        "gate": LC.sha_file(r["gate"]), "head": head_of(arm), "kt_records": kt, "ktw_records": ktw,
        "rmatch_records": RM.built_records(), "zrc_records": Z2._BUILT(r["ch_zrc"]),
        "zlink_records": ZL.built_records()}})
    LC.write_json(sj, rec)


def fit_stamp(name, out_root=None):
    """A U2h fit's screen.json and its u2h record, checked against the fit's name."""
    arm, split = FITS[name]
    sj = Path(out_root or S.OUT / "fits") / name / "screen.json"
    rec = json.loads(sj.read_text(encoding="utf-8")) if sj.exists() else {}
    u = rec.get("u2h") or {}
    if (rec.get("arm"), u.get("declared_in"), u.get("fit"), u.get("build"), u.get("arm"), u.get("seed"),
            u.get("split")) != (arm, DECLARED, name, BUILD, arm, 0, split):
        raise SystemExit(f"zu2h: {sj} is not U2h's {arm} fit of {split}")
    return rec, u


def read(argv, where="laptop", env=None):
    """The fit read with p@swa on the six s1eval carves of the base build, KT_W beside, on the device it was fitted
    on."""
    name, out_root = S2.where(argv)
    if name not in FITS:
        raise SystemExit(f"zu2h: {name} is not one of U2h's fits")
    arm, split = FITS[name]
    bad = G6.fixed_flags(argv, Z.REFUSED_READ)
    if bad:
        raise SystemExit(f"zu2h: U2h reads p's fits on the six s1eval carves of the base build; {bad} would change "
                         f"that")
    _rec, u = fit_stamp(name, out_root)
    env = Z.device_env() if env is None else env
    where_ok(where, env, "read")
    if Z.device_of(env) != u.get("device"):
        raise SystemExit(f"zu2h: {name} was fitted on {u.get('device')}; this read runs on {Z.device_of(env)}")
    with Z.on_u2d(BUILD) as r:
        Z2.cache_root_of(argv)
        argv = Z2.on_card(argv)
        g = Z.gate_passed(BUILD)
        kt = Z3.kt_records(Z3.eval_carves(), r["kt"])
        ktw = ktw_records(Z3.eval_carves())
        rc = RM.read(argv)
    rj = Path(out_root) / name / "read.json"
    if rj.exists():
        rec = json.loads(rj.read_text(encoding="utf-8"))
        rec.update({**code_shas(), "u2h": {
            "declared_in": DECLARED, "fit": name, "build": BUILD, "arm": arm, "seed": 0, "split": split,
            "device": Z.device_of(env), "env": env, "gate": LC.sha_file(Z.roots(BUILD)["gate"]),
            "pools": g.get("construction"), "kt_records": kt, "ktw_records": ktw}})
        LC.write_json(rj, rec)
    return rc


# ── the compares ─────────────────────────────────────────────────────────────


def fit_device(name, out_root=None):
    """A fetched fit's device: a U2h fit's from its u2h stamps (the read on the fit's device), U2g's and U2d's through
    zu2g."""
    if name not in FITS:
        return ZG.fit_device(name, out_root)
    _rec, u = fit_stamp(name, out_root)
    rj = Path(out_root or S.OUT / "fits") / name / "read.json"
    r = json.loads(rj.read_text(encoding="utf-8")) if rj.exists() else {}
    ru = r.get("u2h") or {}
    if ru.get("declared_in") != DECLARED or ru.get("fit") != name or ru.get("device") != u.get("device"):
        raise SystemExit(f"zu2h: {rj} is not U2h's read of {name} on the device it was fitted on")
    return u["device"]


def compares():
    """The eight compares: each arm against its base arm's seed-0 fit on the base build (dh, the decision), and against
    Y's fit of the base arm (vy, readings across workspaces). The flag: one device required."""
    out = {}
    for sp in SPLITS:
        s = SUF[sp]
        for arm in ARMS_H:
            out[f"scr-u2h-dh-{arm}{s}"] = (fit_name(arm, sp), [Z.fit_name("base", BASE_ARM[arm], 0, sp)], True)
            out[f"scr-u2h-vy-{arm}{s}"] = (fit_name(arm, sp), [ZG.fit_name(BASE_ARM[arm], sp)], False)
    return out


COMPARES = compares()
assert len(COMPARES) == 8


def devices_of(names, one=True, out_root=None):
    devs = {n: fit_device(n, out_root) for n in names}
    same = len({json.dumps(d, sort_keys=True) for d in devs.values()}) == 1
    if one and not same:
        raise SystemExit(f"zu2h: the fits ran on different devices or images: {devs}")
    return devs, same


def compare_cmd(cname, new, bases, one=True, out_root=None, screen=None):
    """lean_screen's compare of fit new against bases (the first decides), the paths relative to the repository; the
    comparison stamped with U2h's record and the fits' devices."""
    devs, same = devices_of([new, *bases], one, out_root)
    fits = Path(out_root or S.OUT / "fits")
    out = Path(screen or SCREEN) / cname

    def r_(p):
        p = Path(p)
        return p.relative_to(ROOT).as_posix() if p.is_absolute() and p.is_relative_to(ROOT) else str(p)

    with Z.at_root():
        rc = S2.main(["compare", "--new", r_(fits / new), "--base", ",".join(r_(fits / b) for b in bases),
                      "--out", r_(out)])
    js = out.with_suffix(".json")
    rec = json.loads(js.read_text(encoding="utf-8"))
    rec["u2h"] = {"declared_in": DECLARED, "compare": cname, "new": new, "bases": list(bases), "devices": devs,
                  "one_device": same, "zu2h_sha256": sha_src(__file__)}
    LC.write_json(js, rec)
    return rc


def compare_all(only=None, out_root=None, screen=None):
    """Every compare not yet filed (a filed one with the same fits and U2h's record is kept); a compare whose fits are
    not fetched is named and skipped. 1 when one is skipped."""
    done, ran, skipped = [], [], []
    for cname, (new, bases, one) in COMPARES.items():
        if only and cname not in only:
            continue
        js = Path(screen or SCREEN) / f"{cname}.json"
        if js.exists():
            u = json.loads(js.read_text(encoding="utf-8")).get("u2h") or {}
            if u.get("declared_in") == DECLARED and u.get("new") == new and u.get("bases") == list(bases):
                done.append(cname)
                continue
        try:
            compare_cmd(cname, new, bases, one, out_root, screen)
            ran.append(cname)
        except SystemExit as e:
            skipped.append(f"{cname}: {e}")
    log(f"zu2h compare-all: {len(ran)} run, {len(done)} filed before, {len(skipped)} skipped"
        + (f" ({'; '.join(skipped)})" if skipped else ""))
    return 1 if skipped else 0


# ── the decision (section 5) ─────────────────────────────────────────────────

DECISIONS = {"H": ARMS_H}
ASKS = {"H": "W2_S's KT columns beside today's graph, as an input to the lean pair"}
CMP = {"H": "dh"}
RULE = ("ADOPT_FOR_FULL_RUN when no arm's re-call under U2d's seed null has a LOSS on any of its twelve reads and the "
        "arms together have at least one GAIN; NO_LOSS when no read is a LOSS and none a GAIN; NOT_ADOPTED when any read "
        "is a LOSS; INCOMPLETE when a re-call is missing, is not U2h's, does not hold twelve reads, or its fits ran on "
        "different devices (docs/U2H_W2S_BESIDE.md, section 5)")
FULL = ("the lean pair's full run with KT_W (six splits per arm, under grade-nullx), declared in its own file, with zg6 "
        "on the same input (zg6w) beside it")
FULL_RUN = {"ADOPT_FOR_FULL_RUN": FULL, "NO_LOSS": "none: KT_W neither gains nor loses on this screen",
            "NOT_ADOPTED": "none: the lean pair stays on today's graph without KT_W",
            "INCOMPLETE": "none until the missing re-call is run"}
NEXT = {"ADOPT_FOR_FULL_RUN": FULL,
        "NO_LOSS": "the rest of W2_S's signal, as the next input: the look block's graph columns computed on W2_S for "
                   "the base rows (topo_STRUCT, depth_STRUCT, walk, walkf and dists), beside today's, in its own file",
        "NOT_ADOPTED": "a per-question gate on the KT_W head, its weight per question taken from inputs computed before "
                       "any row is scored, in its own file",
        "always": "U2g's section 7 stands: the links chosen by the lean pair's R@5, then X, when credit or the host allows",
        "INCOMPLETE": "the missing re-call is run, nothing else changes"}


def pair_base_of(arm, sp):
    return fit_name(arm, sp), Z.fit_name("base", BASE_ARM[arm], 0, sp)


def decision_cmp(decision, arm, sp):
    return f"scr-u2h-{CMP[decision]}-{arm}{SUF[sp]}"


def against(decision, arm):
    return (f"{arm} (KT_W beside KT, on U2d's base build) against {BASE_ARM[arm]} on U2d's base build (today's graph, "
            f"P1c2's rule)")


def u2h_loader(decision, arm):
    """screen_pair.load's checks for one of the decision's compares: U2h's, its new fit the arm's U2h fit and its
    deciding base the base arm's seed-0 fit of the split on the base build."""

    def load(fn):
        p = Path(fn)
        if not p.exists():
            raise SystemExit(f"zu2h: {p} missing")
        rec = json.loads(p.read_text(encoding="utf-8"))
        if (rec.get("u2h") or {}).get("declared_in") != DECLARED:
            raise SystemExit(f"zu2h: {p} is not U2h's compare")
        nn_ = SR.parts_of(rec["new"])[-1]
        if nn_ not in FITS or FITS[nn_][0] != arm:
            raise SystemExit(f"zu2h: {p} reads {nn_}, not one of U2h's {arm} fits")
        split = FITS[nn_][1]
        want_new, want_base = pair_base_of(arm, split)
        if nn_ != want_new or SR.parts_of(rec["bases"][0])[-1] != want_base:
            raise SystemExit(f"zu2h: {p} compares {nn_} with {rec['bases'][0]}, not decision {decision}'s {want_new} "
                             f"with {want_base}")
        if sorted(rec["trained_on"]) != sorted(LG.FITS[split]):
            raise SystemExit(f"zu2h: {p} trained on {rec['trained_on']}, not {split}'s {sorted(LG.FITS[split])}")
        if rec["candidate"] != "p@swa" or rec["carve"] != "s1eval":
            raise SystemExit(f"zu2h: {p} reads {rec['candidate']} on {rec['carve']}, not p@swa on s1eval")
        rows = rec["by_base"][rec["bases"][0]]["rows"]
        if sorted(r["dataset"] for r in rows) != sorted(LG.EVAL_ORDER):
            raise SystemExit(f"zu2h: {p} reads {[r['dataset'] for r in rows]}, not the six datasets")
        return split, rows

    return load


def pair_out(decision, arm, screen=None):
    return Path(screen or SCREEN) / f"scr-u2h-{CMP[decision]}-{arm}-pair"


def recall_out(decision, arm, screen=None):
    p = pair_out(decision, arm, screen)
    return p.with_name(p.name + "-recall.json")


def restamp(out, decision, arm, kind):
    out = Path(out)
    js = out.with_suffix(".json")
    if js.exists():
        rec = json.loads(js.read_text(encoding="utf-8"))
        rec.update({"stage": "U2h", "declared_in": DECLARED, "decision": decision, "asks": ASKS[decision], "arm": arm,
                    "against": against(decision, arm), "null_arm": NULL_ARM[arm],
                    "null": f"U2d's seed null ({U2D_DECLARED}, section 6)", "rebased_through": {},
                    "zu2h_sha256": sha_src(__file__)})
        LC.write_json(js, rec)
    md = out.with_suffix(".md")
    if md.exists():
        t = md.read_text(encoding="utf-8")
        for a, b in Z.FIX:
            t = t.replace(a, b)
        head = f"U2h ({DECLARED}), decision {decision} ({ASKS[decision]}): {against(decision, arm)}"
        if kind == "recall":
            head += f", re-called under U2d's {NULL_ARM[arm]} seed null"
        md.write_text(head + ".\n\n" + t, encoding="utf-8")


def pair(decision, arm, screens=None, out=None, screen=None):
    """screen_pair.pair over the decision's two compares of the arm (L-musique and L-hotpotqa)."""
    screens = screens or [Path(screen or SCREEN) / f"{decision_cmp(decision, arm, sp)}.json" for sp in SPLITS]
    out = Path(out or pair_out(decision, arm, screen))
    saved = SP.load
    SP.load = u2h_loader(decision, arm)
    try:
        rec = SP.pair(screens, out)
    finally:
        SP.load = saved
    restamp(out, decision, arm, "pair")
    return rec


def recall(decision, arm, pair_file=None, out=None, nulls=None, screen=None):
    """screen_recall.pair of the decision's pair under U2d's seed null of the base arm (zu2d.u2d_null); the pair's base
    is the null's own seed-0 fit, so nothing is re-based."""
    na = NULL_ARM[arm]
    files = nulls or Z.null_files(na, screen)
    null, sources = Z.u2d_null(na, files)
    for sp in SPLITS:
        if pair_base_of(arm, sp)[1] != Z.fit_name("base", na, 0, sp):
            raise SystemExit(f"zu2h: {arm}'s pair base of {sp} is not {na}'s seed-0 fit; U2h re-bases nothing")
    pf = Path(pair_file or pair_out(decision, arm, screen).with_suffix(".json"))
    pr = json.loads(pf.read_text(encoding="utf-8")) if pf.exists() else {}
    if pr.get("stage") != "U2h" or pr.get("decision") != decision or pr.get("arm") != arm:
        raise SystemExit(f"zu2h: {pf} is not U2h's pair of {arm} for decision {decision}")
    out = Path(out or recall_out(decision, arm, screen).with_suffix(""))
    saved = SR.load_null
    SR.load_null = lambda _files: (null, sources)
    try:
        rec = SR.pair(files, pf, out)
    finally:
        SR.load_null = saved
    restamp(out, decision, arm, "recall")
    return rec


def decision_fits(decision):
    """Every fit a decision's re-calls read: the pairs' fits and the null's."""
    names = set()
    for arm in DECISIONS[decision]:
        for sp in SPLITS:
            names |= set(pair_base_of(arm, sp))
            names |= {Z.fit_name("base", NULL_ARM[arm], s, sp) for s in (0, *SR.SEEDS)}
    return sorted(names)


def decision_of(decision, recalls=None, devices=None, screen=None):
    """The decision's verdict from its arms' re-calls (section 5's rule); devices maps fit names to their devices (the
    fetched fits' stamps when None)."""
    problems, rows = [], {}
    for arm in DECISIONS[decision]:
        rp = Path((recalls or {}).get(arm) or recall_out(decision, arm, screen))
        if not rp.exists():
            problems.append(f"{rp.name} missing")
            continue
        rec = json.loads(rp.read_text(encoding="utf-8"))
        if (rec.get("stage"), rec.get("declared_in"), rec.get("decision"), rec.get("arm")) != (
                "U2h", DECLARED, decision, arm):
            problems.append(f"{rp.name} is not U2h's re-call of {arm} for decision {decision}")
        rs = rec.get("rows", [])
        if sorted((r.get("split"), r.get("dataset")) for r in rs) != sorted(
                (sp, ds) for sp in SPLITS for ds in LG.EVAL_ORDER):
            problems.append(f"{rp.name}: {len(rs)} reads, not the twelve of {', '.join(SPLITS)} on the six datasets")
        rows[arm] = rs
    try:
        devs = devices if devices is not None else {n: fit_device(n) for n in decision_fits(decision)}
        if len({json.dumps(d, sort_keys=True) for d in devs.values()}) > 1:
            problems.append(f"the fits ran on different devices: {devs}")
    except SystemExit as e:
        problems.append(str(e))
    gains = [f"{arm} {r['split']} {r['dataset']}" for arm, rs in rows.items() for r in rs if r.get("call") == "GAIN"]
    losses = [f"{arm} {r['split']} {r['dataset']}" for arm, rs in rows.items() for r in rs if r.get("call") == "LOSS"]
    v = "INCOMPLETE" if problems else "NOT_ADOPTED" if losses else "ADOPT_FOR_FULL_RUN" if gains else "NO_LOSS"
    return {"decision": decision, "asks": ASKS[decision], "verdict": v, "gains": gains, "losses": losses,
            "problems": problems, "arms": {a: against(decision, a) for a in DECISIONS[decision]},
            "calls": {f"{arm} {r['split']} {r['dataset']}": r.get("call") for arm, rs in rows.items() for r in rs},
            "delta_r5": {f"{arm} {r['split']} {r['dataset']}": r.get("delta") for arm, rs in rows.items() for r in rs}}


def next_forms(v):
    """Section 7, fixed before any number."""
    h = v["H"]["verdict"]
    out = [NEXT[h]] if h in ("ADOPT_FOR_FULL_RUN", "NO_LOSS", "NOT_ADOPTED") else [f"H: {NEXT['INCOMPLETE']}"]
    return out + [NEXT["always"]]


# ── beside, deciding nothing (section 6) ─────────────────────────────────────


def fits_metrics(screen=None):
    """R@5, FC@5 and hit@1 per fit and read from U2h's compares (U2h's fits as new, the base and Y fits as bases), and
    rrf per split from the decision's compares (the base build's reads)."""
    fit, rrf = {}, {}
    for cname, (new, bases, _one) in COMPARES.items():
        for r in Z.cmp_rows(cname, screen) or []:
            fit.setdefault(new, {}).setdefault(r["dataset"], r["new"])
            fit.setdefault(bases[0], {}).setdefault(r["dataset"], r["base"])
            if cname.startswith("scr-u2h-dh-"):
                rrf.setdefault(FITS[new][1], {})[r["dataset"]] = r["rrf"]
    return fit, rrf


def vy_readings(screen=None):
    """Each U2h fit against Y's fit of its base arm, each read called under U2d's null of the base arm: readings, not
    decisions (the fits ran in different workspaces on different builds)."""
    floors, out = {}, {}
    for na in sorted(set(NULL_ARM.values())):
        try:
            null, _src = Z.u2d_null(na, Z.null_files(na, screen))
            floors[na] = {(sp, ds): v["floor"] for sp in SPLITS for ds, v in null[sp].items()}
        except SystemExit as e:
            floors[na] = str(e)
    for arm in ARMS_H:
        fl = floors[NULL_ARM[arm]]
        for sp in SPLITS:
            for r in Z.cmp_rows(f"scr-u2h-vy-{arm}{SUF[sp]}", screen) or []:
                f = fl.get((sp, r["dataset"])) if isinstance(fl, dict) else None
                out[f"{arm} {sp} {r['dataset']}"] = {
                    "delta": r["delta"][0], "lo": r["lo"][0], "hi": r["hi"][0], "call": r["call"],
                    "floor": round(f, 4) if f is not None else None,
                    "reading": SR.recall(r["call"], r["delta"][0], f) if f is not None else None}
    return out, {k: v if isinstance(v, str) else "U2d's null" for k, v in floors.items()}


def ktw_beside(root=None):
    out = {}
    for rp in sorted(Path(root or KTW).glob("*/*/ktw.json")):
        rec = json.loads(rp.read_text(encoding="utf-8"))
        out[f"{rp.parent.parent.name}/{rp.parent.name}"] = {"source": rec.get("source"), "counts": rec.get("counts"),
                                                            "beside": rec.get("beside")}
    return out


def beside(screen=None):
    fit, rrf = fits_metrics(screen)
    caps = Z.pools_stats()
    share, head = {}, {}
    for sp in SPLITS:
        for ds in LG.EVAL_ORDER:
            rr = (rrf.get(sp) or {}).get(ds)
            for tag, m_, g_ in (("u2h", fit_name("zrckpw", sp), fit_name("zspkw", sp)),
                                ("base", Z.fit_name("base", "zrckp", 0, sp), Z.fit_name("base", "zspk", 0, sp))):
                m = (fit.get(m_) or {}).get(ds)
                gg = (fit.get(g_) or {}).get(ds)
                share[f"{tag} {sp} {ds}"] = G6.mlp_share(m and m[0], gg and gg[0], rr and rr[0])
    for ds in LINKED:
        cv = (caps.get(BUILD) or {}).get(f"{ds}__s1eval") or {}
        row = {"hit1_cap": ((cv.get("caps") or {}).get("pool") or {}).get("hit1"),
               "r5_cap": ((cv.get("caps") or {}).get("pool") or {}).get("r5")}
        for sp in SPLITS:
            for arm in ARMS_H:
                for n in (fit_name(arm, sp), Z.fit_name("base", BASE_ARM[arm], 0, sp)):
                    x = (fit.get(n) or {}).get(ds)
                    row[n] = {"r5": x[0], "hit1": x[2]} if x else None
        head[ds] = row
    readings, floors = vy_readings(screen)
    return {"metrics": fit, "rrf": rrf, "mlp_share": share, "share_min": G6.SHARE_MIN, "headline_hit1": head,
            "vy_readings": readings, "vy_floors": floors, "ktw": ktw_beside()}


def decide(screen=None, out=None):
    """The pairs and re-calls of decision H's arms, the decision, what follows and the beside report:
    outputs/u2h/screen-verdict.json and .md."""
    fmt = Z.fmt
    ran = []
    for d, arms in DECISIONS.items():
        for arm in arms:
            try:
                pair(d, arm, screen=screen)
                recall(d, arm, screen=screen)
                ran.append(f"{d} {arm}")
            except SystemExit as e:
                log(f"zu2h decide {d} {arm}: {e}")
    v = {d: decision_of(d, screen=screen) for d in DECISIONS}
    verdicts = {d: v[d]["verdict"] for d in v}
    nxt = next_forms(v)
    bs = beside(screen)
    rec = {"stage": "U2h", "declared_in": DECLARED, "rule": RULE, "verdicts": verdicts,
           "full_run": FULL_RUN[verdicts["H"]], "next_forms": nxt, "decisions": v, "recalled": ran, "beside": bs,
           "zu2h_sha256": sha_src(__file__), "utc": utc()}
    md = [f"# U2h's screen verdict: H **{verdicts['H']}**", "", f"The lean pair's full run: {FULL_RUN[verdicts['H']]}.",
          "", f"Rule: {RULE}.", ""]
    for d in DECISIONS:
        x = v[d]
        md.append(f"- **{d}** ({ASKS[d]}; " + "; ".join(x["arms"].values()) + f"): **{x['verdict']}**, "
                  f"{len(x['gains'])} GAIN ({', '.join(x['gains']) or 'none'}), {len(x['losses'])} LOSS "
                  f"({', '.join(x['losses']) or 'none'})" + (f"; problems: {'; '.join(x['problems'])}"
                                                             if x["problems"] else "") + ".")
    md += ["", "What follows (section 7, fixed before any number):", ""] + [f"- {x}" for x in nxt]
    md += ["", "## Beside, deciding nothing", "",
           "R@5 / FC@5 / hit@1 per fit (seed 0) and read, every fit on the base build except Y's (W2_S in place of "
           "today's graph, another workspace); rrf from the base build's reads:", "",
           "| split | dataset | rrf | zrckp | zspk | zrckpw | zspkw | Y zrckp | Y zspk |",
           "|---|---|---|---|---|---|---|---|---|"]
    for sp in SPLITS:
        for ds in LG.EVAL_ORDER:
            cells = []
            for n in (Z.fit_name("base", "zrckp", 0, sp), Z.fit_name("base", "zspk", 0, sp), fit_name("zrckpw", sp),
                      fit_name("zspkw", sp), ZG.fit_name("zrckp", sp), ZG.fit_name("zspk", sp)):
                x = (bs["metrics"].get(n) or {}).get(ds)
                cells.append(" / ".join(fmt(y) for y in x) if x else "")
            rr = (bs["rrf"].get(sp) or {}).get(ds)
            md.append(f"| {sp} | {ds} | {' / '.join(fmt(y) for y in rr) if rr else ''} | " + " | ".join(cells) + " |")
    md += ["", "W2_S beside today's graph (U2h) against W2_S in place of it (Y), read under U2d's null (readings, not "
               "decisions):", "", "| arm | split | dataset | delta R@5 [95% CI] | floor | call at 0.0075 | reading |",
           "|---|---|---|---|---:|---|---|"]
    for k, x in bs["vy_readings"].items():
        arm, sp, ds = k.split()
        md.append(f"| {arm} | {sp} | {ds} | {x['delta']:+.4f} [{x['lo']:+.4f}, {x['hi']:+.4f}] | {fmt(x['floor'])} | "
                  f"{x['call']} | {x['reading']} |")
    md += ["", f"The MLP share, (MLP - rrf) / (GNN - rrf) on R@5 (zrckpw of zspkw; zrckp of zspk on the base build), "
               f"where the GNN's gain over rrf exceeds {G6.SHARE_MIN}:", "", "| read | share |", "|---|---:|"]
    md += [f"| {k} | {fmt(x)} |" for k, x in bs["mlp_share"].items()]
    md += ["", "metaqa and webqsp: hit@1 is the headline; R@5 against its cap (the base build's s1eval pools):", "",
           "| dataset | hit@1 cap | R@5 cap | fits (R@5, hit@1) |", "|---|---:|---:|---|"]
    for k, row in bs["headline_hit1"].items():
        fits_ = "; ".join(f"{a}: {fmt(x['r5'])}, {fmt(x['hit1'])}" for a, x in row.items()
                          if a not in ("hit1_cap", "r5_cap") and x)
        md.append(f"| {k} | {fmt(row['hit1_cap'])} | {fmt(row['r5_cap'])} | {fits_} |")
    md += ["", f"KT_W beside KT per carve (shares of the base rows; {FLOAT16_DEEP}):", "",
           "| carve | rows | in KIND_T today | on W2_S | both | KIND_T place differs | fill today | fill W2_S | source |",
           "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for k, x in bs["ktw"].items():
        b = x.get("beside") or {}
        si, sf = b.get("share_in_kind_t") or {}, b.get("share_fill") or {}
        md.append(f"| {k} | {b.get('rows', '')} | {fmt(si.get('today'))} | {fmt(si.get('w2s'))} | {fmt(si.get('both'))} "
                  f"| {fmt((b.get('share_place_differs') or {}).get('KIND_T'))} | {fmt(sf.get('today'))} | "
                  f"{fmt(sf.get('w2s'))} | {x.get('source')} |")
    out = Path(out or VERDICT)
    out.parent.mkdir(parents=True, exist_ok=True)
    LC.write_json(out.with_suffix(".json"), rec)
    out.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    log(f"zu2h decide: {verdicts}; full run: {FULL_RUN[verdicts['H']]}")
    return rec


# ── selftest (toy data only; run it in its own process) ──────────────────────


def stops(fn, *args, has=None, **kw):
    """fn(*args) must stop with SystemExit, its message holding has."""
    try:
        fn(*args, **kw)
    except SystemExit as e:
        assert has is None or has in str(e), (has, str(e))
        return str(e)
    raise AssertionError(f"{getattr(fn, '__name__', fn)}{args} did not stop")


@contextlib.contextmanager
def patched_globals(**kw):
    g = globals()
    saved = {k: g[k] for k in kw}
    g.update(kw)
    try:
        yield
    finally:
        g.update(saved)


def toy_questions(rng, nq):
    """Toy questions: B, Y's orders on W2_S, Y's I_q and expansion (P1c2's rule), base's expansion, kind and route."""
    qs = []
    for j in range(nq):
        B = np.unique(rng.integers(0, 40, int(rng.integers(1, 6)))).astype(np.int64)
        orders = {c: rng.permutation(400)[:int(rng.integers(60, 300))].astype(np.int64) for c in KT3.ORDERED}
        full = orders["KIND_T"][~np.isin(orders["KIND_T"], B)]
        inc = np.unique(np.concatenate([B, rng.integers(40, 400, int(rng.integers(0, 12)))])).astype(np.int64)
        y_exp, _w = ZP.base_expansion(full, inc.size, B.size)
        b_exp = rng.permutation(np.setdiff1d(np.arange(400), B))[:int(rng.integers(0, 14))].astype(np.int64)
        qs.append({"B": B, "orders": orders, "inc": inc, "exp": y_exp, "b_exp": b_exp, "entity": bool(j % 2),
                   "route": bool(j % 3 == 0)})
    return qs


def toy_arrays(qs, ids, exps, incs):
    per = {k: [] for k in ZP.PER_Q if k not in ("inc_size", "bs_size", "count")}
    for q, e, inc in zip(qs, exps, incs):
        for k, x in (("today_count", e.size), ("fill_want", e.size), ("fill_k", e.size), ("x_count", 0),
                     ("today_inc_size", inc.size), ("budget", e.size), ("kt_len", e.size + 3), ("entity", q["entity"]),
                     ("route", q["route"]), ("changed", True), ("n_gold", 0), ("n_held", 0), ("n_held_I", 0)):
            per[k].append(x)
    return ZP.part_arrays(ids, [q["B"] for q in qs], incs, exps, per, np.zeros(len(qs) + 1, np.int64),
                          np.zeros(0, bool))


def toy_y_parts(qs, name, carve, k, digest, desc):
    """Y's parts of a toy carve as zu2g.py pools files them (records, npz, KT columns), its manifest entry and its
    same_as_new PASS."""
    nq = len(qs)
    ids = [f"q{j}" for j in range(nq)]
    ry = ZG.roots_y(ZG.BUILD)
    ry["parts"].mkdir(parents=True, exist_ok=True)
    entry = []
    for i in range(k):
        q0, q1 = U2C.shard_range(nq, i, k)
        sub = qs[q0:q1]
        arrays = toy_arrays(sub, ids[q0:q1], [q["exp"] for q in sub], [q["inc"] for q in sub])
        kt = np.concatenate([KT3.columns(q["orders"], np.union1d(q["B"], q["exp"]), q["exp"], q["entity"]).astype(KT_DT)
                             for q in sub]) if sub else np.zeros((0, NKT), KT_DT)
        stem = ZP.stem_of(name, carve, (i, k))
        files = {"npz": ZP.save_npz(ry["parts"] / f"{stem}.npz", arrays),
                 "kt": ZP.save_npy(ry["parts"] / f"{stem}.kt.npy", kt)}
        ZP.write_json(ry["parts"] / f"{stem}.json", {
            "declared_in": ZG.DECLARED, "stage": "U2d", "build": ZG.BUILD, "dataset": name, "carve": carve,
            "construction": ZG.CONSTRUCTION_Y, "graph_detail": desc, "carve_ids_sha256": digest, "questions_all": nq,
            "script_sha256": ZP.script_sha256(), "threads": 2, "part": i, "parts": k, "q0": q0, "q1": q1,
            "rows": int(kt.shape[0]), "files": files, "checks": {"bs_step4d": True, "seeds_same_both_graphs": True},
            "seconds": 0.1, "utc": "x"})
        entry.append({"stem": stem, "q0": q0, "q1": q1, "rows": int(kt.shape[0]), **files})
    yd = ry["pools"] / name
    ZP.write_json(yd / "pools.json", {"declared_in": ZG.DECLARED, "stage": "U2d", "build": ZG.BUILD, "dataset": name,
                                      "construction": ZG.CONSTRUCTION_Y, "files": {f"{name}__{carve}": {"parts": entry}}})
    ZP.write_json(yd / "same_as_new.json", {"verdict": "PASS", "declared_in": ZG.DECLARED, "dataset": name})


def toy_base(qs, name, carve, rng):
    """The base build's pools file of a toy carve (one merged file and its manifest), its cache (the questions and
    each pool's rows shuffled) and its KT record. Returns (cache ids, cache rows per question, cache nodes, kt)."""
    ids = [f"q{j}" for j in range(len(qs))]
    a = toy_arrays(qs, ids, [q["b_exp"] for q in qs], [q["inc"] for q in qs])
    d = ZP.roots(BUILD)["pools"] / name
    d.mkdir(parents=True, exist_ok=True)
    fname = f"{name}__{carve}.npz"
    fsha = ZP.save_npz(d / fname, a)
    ZP.write_json(d / "pools.json", {"files": {f"{name}__{carve}": {"file": fname, "sha256": fsha,
                                                                    "construction": ZP.construction_of(BUILD)}}})
    perm = rng.permutation(len(qs))
    c_ids = [ids[p] for p in perm]
    nodes = [rng.permutation(np.union1d(qs[p]["B"], qs[p]["b_exp"])) for p in perm]
    n = [x.size for x in nodes]
    croot = ZP.roots(BUILD)["cache"]
    pd = croot / name / carve / "part_0of1"
    pd.mkdir(parents=True, exist_ok=True)
    np.save(pd / "n.npy", np.asarray(n, np.int32))
    np.save(pd / "node.npy", np.concatenate(nodes).astype(np.int32))
    ZP.write_json(pd / "record.json", {"query_range": [0, len(qs)], "look": {"carve_queries": len(qs)}, "ids": c_ids,
                                       "arrays": {"n": {"sha256": ZP.sha_file(pd / "n.npy")},
                                                  "node": {"sha256": ZP.sha_file(pd / "node.npy")}}})
    parts = {pd.name: ZP.sha_file(pd / "record.json")}
    kt = toy_base_kt(name, carve, c_ids, n, rng, parts, pools_manifest_sha256=ZP.sha_file(d / "pools.json"),
                     pools_file_sha256=fsha)
    return c_ids, n, nodes, kt


def toy_base_kt(name, carve, c_ids, n, rng, parts, **over):
    d = ZP.roots(BUILD)["kt"] / name / carve
    d.mkdir(parents=True, exist_ok=True)
    kt = rng.random((int(sum(n)), NKT)).astype(KT_DT)
    ksha = ZP.save_npy(d / "kt.npy", kt)
    nsha = ZP.save_npy(d / "n.npy", np.asarray(n, np.int32))
    ZP.write_json(d / "ids.json", list(c_ids))
    rec = {"declared_in": U2D_DECLARED, "columns_declared_in": KT3.DECLARED, "columns": list(KT3.COLS),
           "nmax": KT3.NMAX, "dtype": "float16", "arrays": {"kt": {"sha256": ksha, "shape": list(kt.shape)},
                                                            "n": {"sha256": nsha, "shape": [len(c_ids)]}},
           "dataset": name, "carve": carve, "questions": len(c_ids), "rows": int(kt.shape[0]),
           "ids_sha256": ZP.sha_file(d / "ids.json"), "cache_root": rel(ZP.roots(BUILD)["cache"]), "cache_parts": parts,
           "construction": ZP.construction_of(BUILD), "build": BUILD}
    rec.update(over)
    ZP.write_json(d / "record.json", rec)
    return kt


def toy_ktw(root, c, rng, kt_root, **over):
    """A KT_W input for toy carve c in ktw-merge's form (the fields this file reads); over replaces fields."""
    d = Path(root) / c.ds / c.carve
    d.mkdir(parents=True, exist_ok=True)
    ktw = rng.random((int(c.off_np[-1]), NKT)).astype(np.float16)
    np.save(d / "ktw.npy", ktw)
    if not (d / "n.npy").exists():
        np.save(d / "n.npy", c.n_np.astype(np.int32))
    if not (d / "ids.json").exists():
        LC.write_json(d / "ids.json", list(c.ids))
    rec = {"declared_in": DECLARED, "stage": "U2h", "build": BUILD, "columns": list(KT_COLS), "dtype": "float16",
           "arrays": {"ktw": {"sha256": LC.sha_file(d / "ktw.npy"), "shape": list(ktw.shape)},
                      "n": {"sha256": LC.sha_file(d / "n.npy"), "shape": [int(c.rows)]}},
           "dataset": c.ds, "carve": c.carve, "questions": int(c.rows), "rows": int(c.off_np[-1]),
           "ids_sha256": LC.sha_file(d / "ids.json"), "cache_root": over.pop("cache_root"),
           "cache_parts": dict(c.part_shas),
           "base_kt_record_sha256": LC.sha_file(Path(kt_root) / c.ds / c.carve / "record.json")}
    rec.update(over)
    LC.write_json(d / "ktw.json", rec)
    return ktw


def fake_cmp_h(td, cname, new, base, deltas, base_r5=0.6, stamp=True):
    """A compare in lean_screen's form of fit new against fit base, new's screen.json stamped (U2h's, U2g's or U2d's
    fit)."""
    td = Path(td)
    dev = Z.device_of(Z.TOY_ENV)
    if new in FITS:
        arm, split = FITS[new]
        key, u = "u2h", {"declared_in": DECLARED, "fit": new, "build": BUILD, "arm": arm, "seed": 0, "split": split,
                         "device": dev}
    elif new in ZG.FITS:
        arm, split = ZG.FITS[new]
        key, u = "u2g", {"declared_in": ZG.DECLARED, "fit": new, "arm": arm, "seed": 0, "split": split, "device": dev}
    else:
        b, arm, seed, split = Z.FITS[new]
        key, u = "u2d", {"declared_in": U2D_DECLARED, "fit": new, "build": b, "arm": arm, "seed": seed, "split": split,
                         "device": dev}
    fit = td / "fits" / new
    fit.mkdir(parents=True, exist_ok=True)
    (fit / "screen.json").write_text(json.dumps({"arm": arm, "split": split, "seed": 0, key: u}), encoding="utf-8")
    rows = []
    for ds in LG.EVAL_ORDER:
        d = deltas.get(ds, 0.0)
        lo, hi = round(d - 0.004, 4), round(d + 0.004, 4)
        rows.append({"dataset": ds, "read": "in-domain" if ds in LG.FITS[split] else "zero-shot", "questions": 100,
                     "base": [base_r5, 0.5, 0.4], "new": [round(base_r5 + d, 4), 0.5, 0.4], "rrf": [0.4, 0.3, 0.2],
                     "delta": [d, 0, 0], "lo": [lo, 0, 0], "hi": [hi, 0, 0], "call": S.call(d, lo, hi)})
    bp = f"outputs/screen/fits/{base}"
    v = S.verdict([r["call"] for r in rows])
    rec = {"new": str(fit), "bases": [bp], "candidate": "p@swa", "carve": "s1eval", "trained_on": sorted(LG.FITS[split]),
           "by_base": {bp: {"rows": rows, "verdict": v, "decides": True}}, "verdict": v}
    if stamp:
        rec["u2h"] = {"declared_in": DECLARED, "compare": cname, "new": new, "bases": [base]}
    p = td / f"{cname}.json"
    p.write_text(json.dumps(rec), encoding="utf-8")
    return p


def fake_recall_h(path, decision, arm, calls, over=None):
    rows = [{"split": sp, "dataset": ds, "read": "x", "base": 0.6, "new": 0.61, "rrf": 0.4, "delta": 0.01,
             "call": calls.get((sp, ds), "WITHIN")} for sp in SPLITS for ds in LG.EVAL_ORDER]
    rec = {"stage": "U2h", "declared_in": DECLARED, "decision": decision, "arm": arm, "verdict": "x", "rows": rows}
    rec.update(over or {})
    LC.write_json(path, rec)
    return path


def selftest():
    import shutil
    import tempfile
    from types import SimpleNamespace
    t0 = time.time()
    LG.bind_device_ops()
    # 1. the constants: Y registered (zu2g), za6lw's namesakes not in this process, U2h's arms on the KT carves with
    #    KT_W beside, four fits and eight compares on U2d's and U2g's names
    assert ZP.open_carve is ZG.open_carve_y and ZP.DECLARED == U2D_DECLARED and ZP.BUILDS == ZG.U2D_BUILDS
    assert "za6lw" not in sys.modules
    assert (N_PW, N_W, NPW_IN, NW_IN) == (289, 305, 16, 17)
    assert S.ARMS["zrckpw"] == (ZRCKPW, KTWChainCarveZRC) and S.ARMS["zspkw"] == (ZSPKW, KTWLinkCarveBase)
    assert S.ARMS["zrckp"] == Z3P.ARMS["zrckp"] and S.ARMS["zspk"] == Z3.ARMS["zspk"]
    assert ZRCKPW.__mro__[1:3] == (KTMixPW, ZM.ZRM) and ZSPKW.__mro__[1:4] == (ZPROP.ZProp, KTMixW, ZM.ZRM)
    assert KTWChainCarveZRC.__mro__[1] is Z3.KTChainCarveZRC and KTWLinkCarveBase.__mro__[1] is Z3.KTLinkCarveBase
    assert "KT_ROOT" not in vars(KTWChainCarveZRC) and "KT_ROOT" not in vars(KTWLinkCarveBase)
    assert KTWChainCarveZRC.KTW_ROOT is None and KTWLinkCarveBase.KTW_ROOT is None
    with Z.on_u2d(BUILD) as r:
        assert KTWChainCarveZRC.KT_ROOT == KTWLinkCarveBase.KT_ROOT == r["kt"]
        assert KTWLinkCarveBase.LK_ROOT == r["links"] and KTWChainCarveZRC.CH_ROOT == r["ch_zrc"]
    assert list(FITS) == ["scr-u2h-zrckpw", "scr-u2h-zrckpw-hp", "scr-u2h-zspkw", "scr-u2h-zspkw-hp"]
    named = {n for new, bs, _o in COMPARES.values() for n in [new, *bs]} | set(decision_fits("H"))
    assert named - set(FITS) <= set(Z.FITS) | set(ZG.FITS) and set(FITS) <= named
    assert COMPARES["scr-u2h-dh-zspkw-hp"] == ("scr-u2h-zspkw-hp", ["scr-u2d-base-zspk-hp"], True)
    assert COMPARES["scr-u2h-vy-zrckpw"] == ("scr-u2h-zrckpw", ["scr-u2g-y-zrckp"], False)
    assert {"scr-u2d-base-zspk-s2-hp", "scr-u2d-base-zrckp-s1", "scr-u2h-zspkw"} <= set(decision_fits("H"))
    assert all(NULL_ARM[a] == Z.NULL_ARM[BASE_ARM[a]] for a in ARMS_H) and SPLITS == tuple(Z.SPLITS)
    assert set(LINKED_CARVES) == {("metaqa", "fit"), ("metaqa", "s1eval"), ("webqsp", "s1eval")}
    assert head_of("zspkw")["parameters"] == N_W and len(head_of("zrckpw")["columns"]) == NPW_IN
    rng = np.random.default_rng(37)
    # 2. one question: KT_W is P1c3's columns from Y's orders over base's pool (fill Y's expansion, Y's kind); its
    #    refusals: B against Y's and base's, the expansion, Y's KT one unit in the last place off, base's expansion
    q = toy_questions(rng, 3)[1]
    ykt = KT3.columns(q["orders"], np.union1d(q["B"], q["exp"]), q["exp"], q["entity"]).astype(KT_DT)
    rows, cnt = ktw_question(q, q["B"], q["exp"], ykt, q["B"], q["b_exp"], "toy")
    pool = np.union1d(q["B"], q["b_exp"])
    assert np.array_equal(rows.view(np.uint16), KT3.columns(q["orders"], pool, q["exp"], q["entity"]).astype(
        KT_DT).view(np.uint16)) and rows.dtype == KT_DT
    assert cnt["rows"] == pool.size and cnt["fill_w2s"] == int(np.isin(pool, q["exp"]).sum())
    assert cnt["in_y_pool"] == int(np.isin(pool, np.union1d(q["B"], q["exp"])).sum()) >= q["B"].size
    stops(ktw_question, q, np.r_[q["B"], 399], q["exp"], ykt, q["B"], q["b_exp"], "toy", has="B is not Y's")
    stops(ktw_question, q, q["B"], q["exp"], ykt, q["B"][1:], q["b_exp"], "toy", has="B is not base's")
    stops(ktw_question, q, q["B"], q["exp"][::-1] if q["exp"].size > 1 else np.r_[q["exp"], 398], ykt, q["B"],
          q["b_exp"], "toy", has="not Y's filed one")
    k1 = ykt.copy()
    k1.view(np.uint16)[0, 0] += 1
    stops(ktw_question, q, q["B"], q["exp"], k1, q["B"], q["b_exp"], "toy", has="bit for bit")
    stops(ktw_question, q, q["B"], q["exp"], ykt.astype(np.float32), q["B"], q["b_exp"], "toy", has="bit for bit")
    stops(ktw_question, q, q["B"], q["exp"], ykt, q["B"], np.r_[q["b_exp"], q["B"][:1]], "toy", has="repeats a node")
    # 3. a carve's shards end to end on toy files: Y's parts, manifest and same_as_new; base's pools, cache and KT;
    #    Y's carve opener and zu2d_pools.question faked; a development run writes nothing; the parts, then the merge
    #    into the cache's order (each row's columns checked one node at a time); nothing overwritten; the refusals
    tmp = Path(tempfile.mkdtemp(prefix="zu2h_"))
    saved = (ZP.OUT, ZG._ORIG["open_carve"], ZP.question, ZP.p1c2_entry)
    try:
        name, carve, nq = "hotpotqa", "s1eval", 5
        qs = toy_questions(rng, nq)
        desc = {"graph": "W2_S (toy)", "npz": "toy"}
        ids = [f"q{j}" for j in range(nq)]
        cx = SimpleNamespace(K=SimpleNamespace(steered_batch=lambda prep, js, *a: [None] * len(js),
                                               script_sha256=lambda: "e" * 64),
                             D=SimpleNamespace(sources_batch=lambda prep, js, *a: [None] * len(js)),
                             S4=SimpleNamespace(BATCH=2), pop=SimpleNamespace(ids=ids, digest="toy-digest"), prep=None,
                             linked=None, unions=None, constant=60, bs=[q_["B"] for q_ in qs], desc=desc, threads=2,
                             build="new", nodes="dense", name=name, carve=carve)
        opened = []

        def fake_open(build, n_, c_, t_):
            opened.append((build, n_, c_, t_))
            return cx

        tamper = {}

        def fake_question(cx_, j, src, steered):
            assert cx_ is cx and cx_.build == "base" and cx_.nodes is None
            x = qs[j]
            out = {"B": x["B"].copy(), "route": x["route"], "entity": x["entity"], "inc": x["inc"],
                   "orders": x["orders"], "exp": x["exp"].copy(), "k": int(x["exp"].size), "want": int(x["exp"].size),
                   "today": int(x["exp"].size), "kt_len": 9}
            for k_, f in tamper.items():
                out[k_] = f(out[k_])
            return out

        ZP.OUT = tmp / "u2d"
        ZG._ORIG["open_carve"] = fake_open
        ZP.question = fake_question
        ZP.p1c2_entry = lambda n_, c_: {"carve_ids_sha256": "toy-digest", "questions": nq}
        with ZG.patched_globals(OUT=tmp / "u2g"), patched_globals(OUT=tmp / "u2h", KTW=tmp / "u2h" / "ktw",
                                                                  PARTS=tmp / "u2h" / "ktw" / "parts"):
            toy_y_parts(qs, name, carve, 2, "toy-digest", desc)
            c_ids, c_n, c_nodes, kt = toy_base(qs, name, carve, rng)
            stops(ktw_part, name, carve, None, 0, (0, 3), has="missing (Y's part")
            stops(ktw_part, "metaqa", "s1eval", has="Y builds")
            assert ktw_part(name, carve, limit=1, shard=(1, 2)) == 0 and not PARTS.exists()
            assert opened == [("new", name, carve, 2)]
            for k_, f, has in (("B", lambda b: np.r_[b, 399], "B is not Y's"),
                               ("exp", lambda e: np.r_[e, 397], "not Y's filed one"),
                               ("orders", lambda o: {**o, "KIND_T": o["KIND_T"][::-1]}, "bit for bit"),
                               ("inc", lambda i_: np.r_[i_, 396], "I_q on W2_S"),
                               ("entity", lambda e: not e, "kind or route")):
                tamper[k_] = f
                stops(ktw_part, name, carve, None, 2, (1, 2), has=has)
                tamper.clear()
            assert ktw_part(name, carve, shard=(0, 2)) == 0 and ktw_part(name, carve, shard=(1, 2)) == 0
            stops(ktw_part, name, carve, None, 0, (0, 2), has="never overwrites")
            r0 = json.loads((PARTS / f"{name}__{carve}__0of2.json").read_text(encoding="utf-8"))
            assert (r0["stage"], r0["q0"], r0["q1"], r0["threads"], r0["counts"]["questions"]) == ("U2h", 0, 2, 2, 2)
            assert all(r0["checks"].values()) and r0["y"]["construction"] == ZG.CONSTRUCTION_Y
            assert ktw_merge(name, carve) == 0
            d = KTW / name / carve
            got = np.load(d / "ktw.npy")
            assert got.dtype == KT_DT and got.shape == kt.shape
            at = 0
            for cid, nodes in zip(c_ids, c_nodes):
                x = qs[int(cid[1:])]
                for r_ in nodes:
                    want = KT3.columns(x["orders"], np.asarray([r_], np.int64), x["exp"], x["entity"]).astype(KT_DT)
                    assert np.array_equal(got[at].view(np.uint16), want[0].view(np.uint16)), (cid, r_)
                    at += 1
            kd = ZP.roots(BUILD)["kt"] / name / carve
            assert (d / "n.npy").read_bytes() == (kd / "n.npy").read_bytes()
            assert (d / "ids.json").read_bytes() == (kd / "ids.json").read_bytes()
            rec = json.loads((d / "ktw.json").read_text(encoding="utf-8"))
            assert (rec["stage"], rec["build"], rec["rows"], rec["questions"]) == ("U2h", BUILD, int(sum(c_n)), nq)
            assert rec["base_kt_record_sha256"] == ZP.sha_file(kd / "record.json") and len(rec["parts"]) == 2
            assert rec["counts"]["questions"] == nq and rec["counts"]["rows"] == int(sum(c_n))
            assert rec["beside"]["rows"] == int(sum(c_n)) and set(rec["beside"]["share_place_differs"]) == set(
                KT3.ORDERED)
            stops(ktw_merge, name, carve, has="never overwrites")
            # the merge's refusals: a part rewritten by hand, a base KT record on other pools
            shutil.rmtree(d)
            pj = PARTS / f"{name}__{carve}__1of2.json"
            good = pj.read_text(encoding="utf-8")
            pj.write_text(good.replace('"stage": "U2h"', '"stage": "U2g"'), encoding="utf-8")
            stops(ktw_merge, name, carve, has="not U2h's part 1/2")
            pj.write_text(good, encoding="utf-8")
            np.save(PARTS / f"{name}__{carve}__1of2.ktw.npy", np.zeros((3, NKT), KT_DT))
            stops(ktw_merge, name, carve, has="not its record's file")
            krp = kd / "record.json"
            kgood = krp.read_text(encoding="utf-8")
            krp.write_text(kgood.replace(U2D_DECLARED, Z3P.DECLARED), encoding="utf-8")
            stops(base_kt, name, carve, has="not U2d's KT record")
            krp.write_text(kgood, encoding="utf-8")
            assert not d.exists()
            # 4. metaqa's and webqsp's KT_W: base's KT byte for byte; a filed one kept; a base KT record not U2d's
            for ln, lc in LINKED_CARVES:
                toy_base_kt(ln, lc, ["a", "b", "c"], [2, 0, 3], rng, {"part_0of1": "p" * 64})
            assert ktw_linked() == 0
            for ln, lc in LINKED_CARVES:
                kd_, d_ = ZP.roots(BUILD)["kt"] / ln / lc, KTW / ln / lc
                assert (d_ / "ktw.npy").read_bytes() == (kd_ / "kt.npy").read_bytes()
                assert (d_ / "n.npy").read_bytes() == (kd_ / "n.npy").read_bytes()
                lr = json.loads((d_ / "ktw.json").read_text(encoding="utf-8"))
                assert (lr["stage"], lr["build"], lr["cache_parts"]) == ("U2h", BUILD, {"part_0of1": "p" * 64})
                assert lr["base_kt_record_sha256"] == ZP.sha_file(kd_ / "record.json") and "byte for byte" in lr["source"]
            before = (KTW / "metaqa" / "fit" / "ktw.json").read_bytes()
            assert ktw_linked() == 0 and (KTW / "metaqa" / "fit" / "ktw.json").read_bytes() == before
            with patched_globals(KTW=tmp / "u2h" / "ktw2"):
                krp = ZP.roots(BUILD)["kt"] / "webqsp" / "s1eval" / "record.json"
                krp.write_text(krp.read_text(encoding="utf-8").replace('"build": "base"', '"build": "new"'),
                               encoding="utf-8")
                stops(ktw_linked, has="not U2d's KT record")
    finally:
        ZP.OUT, ZG._ORIG["open_carve"], ZP.question, ZP.p1c2_entry = saved
        shutil.rmtree(tmp, ignore_errors=True)
    assert ZP.OUT == ROOT / "outputs" / "u2d" and ZP.question.__module__ == "zu2d_pools"
    assert ZG._ORIG["open_carve"].__module__ == "zu2d_pools" and ZG.OUT == ROOT / "outputs" / "u2g"
    assert (OUT, KTW, PARTS) == (ROOT / "outputs" / "u2h", ROOT / "outputs" / "u2h" / "ktw",
                                 ROOT / "outputs" / "u2h" / "ktw" / "parts")
    selftest_fits(rng)
    log(f"zu2h selftest ({time.time() - t0:.1f}s): ok")
    return 0


def selftest_fits(rng):
    import shutil
    import tempfile
    t0 = time.time()
    tmp = Path(tempfile.mkdtemp(prefix="zu2h_fits_"))
    try:
        RM.toy_roots(tmp, rng, chunks=(12, 12))
        RM.build("metaqa", "toy", out_root=tmp / "ch", cache_root=tmp / "cache", look_root=tmp / "look",
                 rel_dir=tmp / "rel")
        ZL.build("metaqa", "toy", out_root=tmp / "lk", cache_root=tmp / "cache", look_root=tmp / "look")

        class ToyIds(ZM.ToyZ):
            def __init__(self, *a, **k):
                super().__init__(*a, **k)
                self.ids = [f"q{i}" for i in range(self.rows)]

        plain = ZL.link_carve(RM.chain_carve(ToyIds, tmp / "ch", tmp / "rel"), tmp / "lk")
        ktc = Z3.kt_carve(plain, tmp / "kt")
        wc = ktw_carve(ktc, tmp / "ktw")
        c0 = plain("metaqa", "toy", "2wiki", "cpu", tmp / "cache")
        croot = str(tmp / "cache")                       # absolute: ROOT / an absolute path is that path
        with Z.declared():
            Z3.toy_kt(tmp / "kt", c0, rng, cache_root=croot)
            ktw = toy_ktw(tmp / "ktw", c0, rng, tmp / "kt", cache_root=croot)
            c = wc("metaqa", "toy", "2wiki", "cpu", tmp / "cache")
            ck = ktc("metaqa", "toy", "2wiki", "cpu", tmp / "cache")
        blocks = ["rank", "SEMB"]
        qs = np.r_[np.arange(c.rows)[::-1][:5], np.arange(c.rows)[:4]].astype(np.int64)
        B = qs.size
        feats, nq, bz, gold = c.batch(qs, blocks)
        fk, nqk, bzk, gk = ck.batch(qs, blocks)
        # 5. the KT_W carve's batch is the KT carve's plus KT_W's rows of qs in its row order; load_ktw's refusals
        assert set(feats) == set(fk) | {KTW_KEY} and KTW_KEY not in fk
        assert torch.equal(nq, nqk) and torch.equal(bz, bzk) and torch.equal(gold, gk)
        assert all(torch.equal(feats[k], fk[k]) for k in fk if isinstance(fk[k], torch.Tensor))
        rows = np.concatenate([np.arange(c.off_np[q], c.off_np[q + 1]) for q in qs])
        assert torch.equal(feats[KTW_KEY], torch.from_numpy(ktw[rows].astype(np.float32)))
        assert c.nbytes() == ck.nbytes() + ktw.size * 2 and c.ktw_record == LC.sha_file(
            tmp / "ktw" / "metaqa" / "toy" / "ktw.json")
        for i, (over, damage, has) in enumerate((
                ({"declared_in": U2D_DECLARED}, None, "not U2h's KT_W input"),
                ({"stage": "U2d"}, None, "not U2h's KT_W input"),
                ({"columns": list(KT_COLS[::-1])}, None, "not U2h's KT_W input"),
                ({"build": "new"}, None, "not U2h's KT_W input"),
                ({"base_kt_record_sha256": "0" * 64}, None, "beside this carve's KT record"),
                ({}, "ktw", "sha256 is not"), ({"cache_parts": {"part_0of2": "x"}}, None, "other cache parts"),
                ({"cache_root": "outputs/u2d/base/cache"}, None, "cache root"), ({}, "n", "rows per question"),
                ({}, "ids", "questions in its order"), ({}, "ids_sha", "ids.json"), ({}, "dtype", "not float16"),
                ({}, "inf", "not finite"), ({}, "record", "missing"))):
            root = tmp / f"ktw_bad{i}"
            toy_ktw(root, c0, rng, tmp / "kt", **{"cache_root": croot, **over})
            d = root / "metaqa" / "toy"
            rp = d / "ktw.json"
            rec = json.loads(rp.read_text(encoding="utf-8"))
            if damage == "ktw":
                np.save(d / "ktw.npy", np.zeros((int(c0.off_np[-1]), NKT), np.float16))
            elif damage == "n":
                n2 = c0.n_np.copy()
                n2[[0, 1]] = n2[[1, 0]] if n2[0] != n2[1] else (n2[0] + 1, n2[1] - 1)
                np.save(d / "n.npy", n2.astype(np.int32))
                rec["arrays"]["n"]["sha256"] = LC.sha_file(d / "n.npy")
            elif damage == "ids":
                LC.write_json(d / "ids.json", list(c0.ids)[::-1])
                rec["ids_sha256"] = LC.sha_file(d / "ids.json")
            elif damage == "ids_sha":
                rec["ids_sha256"] = "0" * 64
            elif damage in ("dtype", "inf"):
                x = np.load(d / "ktw.npy")
                if damage == "dtype":
                    x = x.astype(np.float32)
                else:
                    x[3, 2] = np.inf
                np.save(d / "ktw.npy", x)
                rec["arrays"]["ktw"]["sha256"] = LC.sha_file(d / "ktw.npy")
            if damage != "record":
                LC.write_json(rp, rec)
            else:
                rp.unlink()
            stops(load_ktw, ck, root, tmp / "cache", has=has)
        assert torch.equal(load_ktw(ck, tmp / "ktw", tmp / "cache")[0].float(), torch.from_numpy(ktw.astype(np.float32)))
        SR.must_stop(ktw_records, [("metaqa", "toy")], tmp / "ktw_bad13")
        SR.must_stop(ktw_records, [("metaqa", "toy")], tmp / "ktw_bad1")
        assert list(ktw_records([("metaqa", "toy")], tmp / "ktw")) == ["metaqa/toy"]
        # 6. from the same seed each arm's state is its base arm's with the first layer widened (the same names in the
        #    same order, every other tensor equal), the first layer from Generator(seed + 3301), the last zero; its
        #    forward is zrckp's or zspk's bit for bit in eval and with the same dropout draws in training; the global
        #    generator is where they leave it
        keep = torch.ones(B, len(blocks))
        for new, old, filed_arm, nin, npar in ((ZRCKPW, ZM.ZRM, Z3P.ZRCKP, NPW_IN, N_PW),
                                               (ZSPKW, ZPROP.ZProp, Z3.ZSPK, NW_IN, N_W)):
            nets, after = {}, {}
            for k, cl in (("old", old), ("filed", filed_arm), ("new", new)):
                torch.manual_seed(0)
                nets[k] = cl(blocks, c.widths, 16, dropout=0.1, seed=0)
                after[k] = torch.rand(4)
            assert torch.equal(after["old"], after["new"]) and torch.equal(after["filed"], after["new"]), new.__name__
            so, sf, sn = (nets[k].state_dict() for k in ("old", "filed", "new"))
            assert list(sn) == list(sf) and sorted(set(sn) - set(so)) == sorted(KT)
            assert all(torch.equal(so[k], sn[k]) for k in so)
            assert sum(int(sn[k].numel()) for k in KT) == npar and tuple(sn["kt_w1"].shape) == (KT_HID, nin)
            g = torch.Generator().manual_seed(0 + KT_SEED_OFF)
            assert torch.equal(sn["kt_w1"], (torch.rand(KT_HID, nin, generator=g) * 2 - 1) / nin ** 0.5)
            assert float(sn["kt_w2"].abs().sum()) == 0.0 and float(sn["kt_b2"].abs().sum()) == 0.0
            with torch.no_grad():
                for mode in (False, True):
                    out = {}
                    for k, m in nets.items():
                        m.train(mode)
                        torch.manual_seed(5)
                        out[k] = m(feats, keep, nq, B, bz)
                    assert torch.equal(out["old"], out["new"]) and torch.equal(out["filed"], out["new"]), (new, mode)
        # 7. with the head set: zrckpw is zrm plus the head on (KT, KT_W); zspkw is zsp's propagation over zrm plus the
        #    head on (KT, KT_W, the clipped z-score of zrm's score); KT without KT_W stops; no KT is the filed model's
        with torch.no_grad():
            gen = torch.Generator().manual_seed(3)
            nets = {}
            for k, cl in (("zrm", ZM.ZRM), ("zrckpw", ZRCKPW), ("zsp", ZPROP.ZProp), ("zspkw", ZSPKW)):
                torch.manual_seed(0)
                nets[k] = cl(blocks, c.widths, 16, dropout=0.1, seed=0).eval()
            for k in ("zrckpw", "zspkw"):
                for p in KT:
                    getattr(nets[k], p).copy_(torch.randn(getattr(nets[k], p).shape, generator=gen))
            nets["zspkw"].sp_w2.copy_(torch.randn(nets["zspkw"].sp_w2.shape, generator=gen))
            nets["zsp"].load_state_dict({k: v for k, v in nets["zspkw"].state_dict().items() if k not in KT})
            s = nets["zrm"](feats, keep, nq, B, bz)
            nw = nets["zrckpw"]
            x = torch.cat([feats[KT_KEY], feats[KTW_KEY]], 1)
            hd = Fn.linear(Fn.gelu(Fn.linear(x, nw.kt_w1, nw.kt_b1)), nw.kt_w2, nw.kt_b2).squeeze(-1)
            assert torch.equal(nw(feats, keep, nq, B, bz), s + hd) and float(hd.abs().sum()) > 0
            ns = nets["zspkw"]
            z = LG.seg_zscore8D(s.unsqueeze(1), nq, B).squeeze(1).clamp(-ZCLIP, ZCLIP)
            xs = torch.cat([feats[KT_KEY], feats[KTW_KEY], z.unsqueeze(1)], 1)
            sk = s + Fn.linear(Fn.gelu(Fn.linear(xs, ns.kt_w1, ns.kt_b1)), ns.kt_w2, ns.kt_b2).squeeze(-1)
            want = sk + ns.prop_head(ZPROP.prop_inputs(sk, feats[ZL.LK_KEY], nq, B))
            assert torch.equal(ns(feats, keep, nq, B, bz), want)
            f_kt = {k: v for k, v in feats.items() if k != KTW_KEY}
            stops(nw, f_kt, keep, nq, B, bz, has="without the KT_W input")
            stops(ns, f_kt, keep, nq, B, bz, has="without the KT_W input")
            f0 = {k: v for k, v in feats.items() if k not in (KT_KEY, KTW_KEY)}
            assert torch.equal(nw(f0, keep, nq, B, bz), s)
            assert torch.equal(ns(f0, keep, nq, B, bz), nets["zsp"](f0, keep, nq, B, bz))
            big = KTMixW.kt_inputs(s * 1e4, feats[KT_KEY], feats[KTW_KEY], nq, B)
            assert bool(torch.isfinite(big).all()) and float(big[:, -1].abs().max()) <= ZCLIP
            assert big.shape[1] == NW_IN and torch.equal(big[:, NKT:2 * NKT], feats[KTW_KEY])
        # 8. a toy fit through lean_gpu's loop for each arm: the head moves, and a repeat is identical
        cfg = {"lr": 2e-2, "wd": 1e-4, "dropout": 0.1, "epochs": 3, "swa_from": 1, "cos": False, "adamw": False,
               "drop": False}
        for arm in ARMS_H:
            with S.patched(arm):
                f1 = LG.fit_variant([c], blocks, cfg, 0, 16, "none", "cpu", tag=f"toy-{arm}")
                f2 = LG.fit_variant([c], blocks, cfg, 0, 16, "none", "cpu", tag=f"toy-{arm} repeat")
            assert float(f1["swa"]["kt_w2"].abs().sum()) > 0, arm
            assert not LG.same_state(f1["swa"], f2["swa"]), arm
        # 9. train's and read's refusals: the fit's name, arm and split; the card's flags (and abbreviations); the
        #    place, GPU and image; the cache root; the device; the gate; the KT and KT_W records; a read of a fit that
        #    is not U2h's or on another device; the devices
        saved_out, saved_s, saved_root = ZP.OUT, S.OUT, Z2.ROOT
        ZP.OUT, S.OUT, Z2.ROOT = tmp / "u2d", tmp / "screen", tmp       # zp1c2 names the cache root relative to ROOT
        try:
            with patched_globals(KTW=tmp / "u2h" / "ktw"):
                env = Z.TOY_ENV

                def argv_t(name, arm, *extra, cache="base"):
                    return ["train", "--name", name, "--arm", arm, "--device", "cuda", "--cache-root",
                            str(tmp / "u2d" / cache / "cache"), *extra]
                nm = fit_name("zspkw", "L-musique")
                stops(train, argv_t("scr-x", "zspkw"), "L-musique", "modal", env, has="not one of U2h's fits")
                stops(train, argv_t(Z.fit_name("base", "zspk", 0, "L-musique"), "zspk"), "L-musique", "modal", env,
                      has="not one of U2h's fits")
                stops(train, argv_t(nm, "zrckpw"), "L-musique", "modal", env, has="not zrckpw's")
                stops(train, argv_t(nm, "zspkw"), "L-hotpotqa", "modal", env, has="not zspkw's of L-hotpotqa")
                for extra in (["--seed", "1"], ["--se", "1"], ["--hidden", "64"], ["--config", "x"],
                              ["--variants", "p,q"], ["--train", "metaqa=fit"], ["--basis", "x"], ["--repeat", "2"],
                              ["--no-verify"], ["--threads", "4"], ["--out-root", str(tmp)], ["--host"], ["--seed=1"]):
                    stops(train, argv_t(nm, "zspkw", *extra), "L-musique", "modal", env, has="would change that")
                stops(train, argv_t(nm, "zspkw"), "L-musique", "laptop", env, has="on Modal's")
                for bad_env in ({**env, "gpu": "NVIDIA H100"}, {**env, "image": "other"}, {**env, "where": "elsewhere"}):
                    stops(train, argv_t(nm, "zspkw"), "L-musique", "modal", bad_env, has="on Modal's")
                stops(train, argv_t(nm, "zspkw", cache="new"), "L-musique", "modal", env, has="--cache-root")
                dev_cpu = [x if x != "cuda" else "cpu" for x in argv_t(nm, "zspkw")]
                stops(train, dev_cpu, "L-musique", "modal", env, has="on the cuda")
                stops(train, argv_t(nm, "zspkw"), "L-musique", "modal", env, has="gate.json missing")
                g = Z.roots(BUILD)["gate"]
                LC.write_json(g, {"verdict": "FAIL", "declared_in": U2D_DECLARED, "build": BUILD})
                stops(train, argv_t(nm, "zspkw"), "L-musique", "modal", env, has="not U2d's PASS")
                LC.write_json(g, {"verdict": "PASS", "declared_in": U2D_DECLARED, "build": BUILD, "construction": ["x"]})
                stops(train, argv_t(nm, "zspkw"), "L-musique", "modal", env, has="record.json missing")
                for ds, cv in Z3.train_carves("L-musique"):
                    LC.write_json(Z.roots(BUILD)["kt"] / ds / cv / "record.json", {"declared_in": U2D_DECLARED})
                stops(train, argv_t(nm, "zspkw"), "L-musique", "modal", env, has="ktw.json missing")
                for ds, cv in Z3.train_carves("L-musique"):
                    LC.write_json(KTW / ds / cv / "ktw.json", {"declared_in": DECLARED, "stage": "U2g", "build": BUILD,
                                                               "columns": list(KT_COLS)})
                stops(train, argv_t(nm, "zspkw"), "L-musique", "modal", env, has="not U2h's KT_W input")
                fd = tmp / "screen" / "fits" / nm
                fd.mkdir(parents=True)

                def argv_r(name, *extra):
                    return ["--name", name, "--device", "cuda", "--cache-root", str(tmp / "u2d" / "base" / "cache"),
                            *extra]
                stops(read, argv_r("scr-y"), "modal", env, has="not one of U2h's fits")
                stops(read, argv_r(nm), "modal", env, has="is not U2h's zspkw fit")
                good = {"declared_in": DECLARED, "fit": nm, "build": BUILD, "arm": "zspkw", "seed": 0,
                        "split": "L-musique", "device": Z.device_of(env)}
                for bad_u in ({**good, "split": "L-hotpotqa"}, {**good, "declared_in": U2D_DECLARED},
                              {**good, "fit": "scr-u2h-zrckpw"}, {**good, "seed": 1}, {**good, "arm": "zspk"},
                              {**good, "build": "new"}):
                    LC.write_json(fd / "screen.json", {"arm": "zspkw", "u2h": bad_u})
                    stops(read, argv_r(nm), "modal", env, has="is not U2h's zspkw fit")
                LC.write_json(fd / "screen.json", {"arm": "zspkw", "u2h": good})
                for extra in (["--read", "metaqa=s1eval"], ["--threads", "4"], ["--out-root", str(tmp)], ["--host"],
                              ["--variants", "p,q"], ["--basis", "x"], ["--no-verify"], ["--rea", "x"]):
                    stops(read, argv_r(nm, *extra), "modal", env, has="would change that")
                stops(read, argv_r(nm), "laptop", env, has="on Modal's")
                other = {**env, "torch": "2.8.0+cu126", "cuda": "12.6"}
                stops(read, argv_r(nm), "modal", other, has="was fitted on")
                stops(read, argv_r(nm), "modal", env, has="record.json missing")
                stops(fit_device, nm, has="is not U2h's read")
                LC.write_json(fd / "read.json", {"u2h": {"declared_in": DECLARED, "fit": nm, "device": Z.device_of(other)}})
                stops(fit_device, nm, has="is not U2h's read")
                LC.write_json(fd / "read.json", {"u2h": {"declared_in": DECLARED, "fit": nm, "device": Z.device_of(env)}})
                assert fit_device(nm) == Z.device_of(env)
                b0 = Z.fit_name("base", "zspk", 0, "L-musique")
                (tmp / "screen" / "fits" / b0).mkdir(parents=True)
                ub = {"declared_in": U2D_DECLARED, "fit": b0, "build": "base", "arm": "zspk", "seed": 0,
                      "split": "L-musique", "device": Z.device_of(other)}
                LC.write_json(tmp / "screen" / "fits" / b0 / "screen.json", {"arm": "zspk", "u2d": ub})
                LC.write_json(tmp / "screen" / "fits" / b0 / "read.json",
                              {"u2d": {"declared_in": U2D_DECLARED, "fit": b0, "device": Z.device_of(other)}})
                assert fit_device(b0) == Z.device_of(other)
                stops(devices_of, [nm, b0], has="different devices")
                stops(compare_cmd, "scr-u2h-dh-zspkw", nm, [b0], True, None, tmp, has="different devices")
                devs, same = devices_of([nm, b0], one=False)
                assert not same and set(devs) == {nm, b0}
        finally:
            ZP.OUT, S.OUT, Z2.ROOT = saved_out, saved_s, saved_root
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    # 10. the decision on toy re-calls: ADOPT_FOR_FULL_RUN, NO_LOSS, NOT_ADOPTED and each INCOMPLETE; section 7
    with tempfile.TemporaryDirectory() as td:
        T = Path(td)
        one = {n: Z.device_of(Z.TOY_ENV) for n in decision_fits("H")}
        G_, L_ = "GAIN", "LOSS"
        mk = lambda a, calls_, **o: fake_recall_h(T / f"rc-{a}-{len(o)}.json", "H", a, calls_, o)  # noqa: E731
        assert decision_of("H", {"zrckpw": mk("zrckpw", {("L-musique", "musique"): G_}),
                                 "zspkw": mk("zspkw", {})}, one)["verdict"] == "ADOPT_FOR_FULL_RUN"
        assert decision_of("H", {"zrckpw": mk("zrckpw", {}), "zspkw": mk("zspkw", {})}, one)["verdict"] == "NO_LOSS"
        nl = decision_of("H", {"zrckpw": mk("zrckpw", {("L-musique", "musique"): G_}),
                               "zspkw": mk("zspkw", {("L-hotpotqa", "2wiki"): L_})}, one)
        assert nl["verdict"] == "NOT_ADOPTED" and nl["losses"] == ["zspkw L-hotpotqa 2wiki"]
        ok = mk("zrckpw", {})
        assert decision_of("H", {"zrckpw": ok, "zspkw": T / "none.json"}, one)["verdict"] == "INCOMPLETE"
        for over in ({"stage": "U2g"}, {"decision": "Y"}, {"declared_in": U2D_DECLARED}, {"rows": []},
                     {"arm": "zspk"}):
            assert decision_of("H", {"zrckpw": ok, "zspkw": mk("zspkw", {}, **over)}, one)["verdict"] == "INCOMPLETE", over
        two = {**one, Z.fit_name("base", "zspk", 2, "L-hotpotqa"): Z.device_of({**Z.TOY_ENV, "gpu": "NVIDIA A10G"})}
        assert decision_of("H", {"zrckpw": ok, "zspkw": mk("zspkw", {})}, two)["verdict"] == "INCOMPLETE"
        for vv in ("ADOPT_FOR_FULL_RUN", "NO_LOSS", "NOT_ADOPTED"):
            assert next_forms({"H": {"verdict": vv}}) == [NEXT[vv], NEXT["always"]]
        assert next_forms({"H": {"verdict": "INCOMPLETE"}}) == [f"H: {NEXT['INCOMPLETE']}", NEXT["always"]]
        assert "zg6w" in NEXT["ADOPT_FOR_FULL_RUN"] and "gate" in NEXT["NOT_ADOPTED"] and "walkf" in NEXT["NO_LOSS"]
    # 11. the pairs and re-calls under U2d's null (no re-base); the loaders' refusals and restoration; the readings
    #     against Y; the metrics; decide on toys (INCOMPLETE: the toy base fits carry no stamp)
    with tempfile.TemporaryDirectory() as td:
        T = Path(td)
        nz = Z.fake_null(T / "z", "zspk", {("L-hotpotqa", "hotpotqa"): (-0.010, 0.012)})
        Z.fake_null(T / "z", "zrckp")
        hm = fake_cmp_h(T / "z", "scr-u2h-dh-zspkw", fit_name("zspkw", "L-musique"),
                        Z.fit_name("base", "zspk", 0, "L-musique"), {"musique": 0.02, "metaqa": -0.003})
        hh = fake_cmp_h(T / "z", "scr-u2h-dh-zspkw-hp", fit_name("zspkw", "L-hotpotqa"),
                        Z.fit_name("base", "zspk", 0, "L-hotpotqa"), {"hotpotqa": -0.0147})
        stops(pair, "H", "zrckpw", [hm, hh], T / "bad", has="not one of U2h's zrckpw fits")
        unst = fake_cmp_h(T / "u", "scr-u2h-dh-zspkw", fit_name("zspkw", "L-musique"),
                          Z.fit_name("base", "zspk", 0, "L-musique"), {}, stamp=False)
        stops(pair, "H", "zspkw", [unst, hh], T / "bad", has="not U2h's compare")
        wrong = fake_cmp_h(T / "w", "scr-u2h-dh-zspkw", fit_name("zspkw", "L-musique"), ZG.fit_name("zspk", "L-musique"),
                           {})
        stops(pair, "H", "zspkw", [wrong, hh], T / "bad", has="not decision H's")
        pr = pair("H", "zspkw", [hm, hh], T / "ph")
        assert pr["verdict"] == "MIXED" and len(pr["rows"]) == 12
        pj = json.loads((T / "ph.json").read_text(encoding="utf-8"))
        assert (pj["stage"], pj["decision"], pj["arm"], pj["null_arm"], pj["declared_in"]) == (
            "U2h", "H", "zspkw", "zspk", DECLARED)
        rc_ = recall("H", "zspkw", T / "ph.json", T / "rh", nz)
        assert rc_["verdict"] == "PROMISING" and rc_["changed"] == ["L-hotpotqa hotpotqa LOSS -> WITHIN"], rc_["changed"]
        rj = json.loads((T / "rh.json").read_text(encoding="utf-8"))
        assert rj["stage"] == "U2h" and rj["rebased_through"] == {} and rj["null_arm"] == "zspk"
        t = (T / "rh.md").read_text(encoding="utf-8")
        assert t.startswith(f"U2h ({DECLARED}), decision H") and "U2d's own seeds 1 and 2" in t
        stops(recall, "H", "zrckpw", T / "ph.json", T / "rz", Z.null_files("zrckp", T / "z"),
              has="not U2h's pair of zrckpw")
        assert SP.load.__module__ == "screen_pair" and SR.load_null.__module__ == "screen_recall"
        for sp, dd in (("L-musique", {"musique": -0.02}), ("L-hotpotqa", {"hotpotqa": -0.010, "2wiki": -0.03})):
            fake_cmp_h(T / "z", f"scr-u2h-vy-zspkw{SUF[sp]}", fit_name("zspkw", sp), ZG.fit_name("zspk", sp), dd)
        readings, floors = vy_readings(T / "z")
        assert readings["zspkw L-hotpotqa hotpotqa"]["call"] == "LOSS"
        assert readings["zspkw L-hotpotqa hotpotqa"]["reading"] == "WITHIN"
        assert readings["zspkw L-hotpotqa 2wiki"]["reading"] == "LOSS" and floors["zspk"] == "U2d's null"
        assert readings["zspkw L-musique musique"]["floor"] == 0.0075 and "zrckpw L-musique metaqa" not in readings
        fit, rrf = fits_metrics(T / "z")
        assert fit[fit_name("zspkw", "L-musique")]["musique"][0] == 0.62 and rrf["L-musique"]["musique"][0] == 0.4
        assert fit[Z.fit_name("base", "zspk", 0, "L-hotpotqa")]["hotpotqa"][0] == 0.6
        assert ZG.fit_name("zspk", "L-musique") in fit and set(rrf) == set(SPLITS)
        for sp in SPLITS:
            fake_cmp_h(T / "z", f"scr-u2h-dh-zrckpw{SUF[sp]}", fit_name("zrckpw", sp),
                       Z.fit_name("base", "zrckp", 0, sp), {"musique": 0.01})
        saved_s = S.OUT
        S.OUT = T / "z"
        try:
            v = decide(T / "z", T / "verdict")
        finally:
            S.OUT = saved_s
        assert v["verdicts"] == {"H": "INCOMPLETE"} and v["recalled"] == ["H zrckpw", "H zspkw"]
        probs = v["decisions"]["H"]["problems"]
        assert any("is not U2d's zrckp fit" in p for p in probs), probs
        assert (T / "verdict.md").read_text(encoding="utf-8").startswith("# U2h's screen verdict: H **INCOMPLETE**")
        assert SP.load.__module__ == "screen_pair" and SR.load_null.__module__ == "screen_recall"
    log(f"zu2h selftest (fits): the KT_W carve's batch and fourteen bad KT_W inputs refused; at the start each arm is "
        f"zrckp's or zspk's bit for bit (eval and training draws, the global generator unmoved), with {NPW_IN} and "
        f"{NW_IN} inputs and {N_PW} and {N_W} weights; the heads and zspkw's propagation by hand; toy fits move the head "
        f"and repeat; train's and read's refusals (name, arm, split, card flags, place, GPU, image, cache root, device, "
        f"gate, KT and KT_W records, another device); the decision (ADOPT, NO_LOSS, NOT_ADOPTED, seven INCOMPLETE) and "
        f"section 7; the pair and re-call under U2d's null; the readings against Y; decide on toys "
        f"({time.time() - t0:.1f}s): ok")
    return 0


# ── main ─────────────────────────────────────────────────────────────────────


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["--selftest"]:
        return selftest()
    cmd, rest = (argv[0], argv[1:]) if argv else (None, [])
    if cmd in ("ktw", "ktw-merge"):
        ap = argparse.ArgumentParser(prog=f"zu2h.py {cmd}", allow_abbrev=False)
        ap.add_argument("dataset", choices=OWN)
        ap.add_argument("--carve", required=True)
        if cmd == "ktw":
            ap.add_argument("--shard")
            ap.add_argument("--threads", type=int)
            ap.add_argument("--limit", type=int, default=0)
        a = ap.parse_args(rest)
        if cmd == "ktw-merge":
            return ktw_merge(a.dataset, a.carve)
        shard = tuple(int(x) for x in a.shard.split("/")) if a.shard else None
        return ktw_part(a.dataset, a.carve, a.threads, a.limit, shard)
    if cmd == "ktw-linked" and not rest:
        return ktw_linked()
    if cmd in ("train", "read"):
        ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
        ap.add_argument("--split", default="L-musique", choices=SPLITS)
        ap.add_argument("--modal", action="store_true")
        k, rest = ap.parse_known_args(rest)
        where = "modal" if k.modal else "laptop"
        return train(["train"] + rest, k.split, where) if cmd == "train" else read(rest, where)
    if cmd == "compare-all" and not rest:
        with Z.at_root():
            return compare_all(screen=Path("outputs/screen"))
    if cmd == "decide" and not rest:
        with Z.at_root():
            decide(screen=Path("outputs/screen"))
        return 0
    raise SystemExit("zu2h: ktw, ktw-merge, ktw-linked, train, read, compare-all, decide, or --selftest")


if __name__ == "__main__":
    sys.exit(main())
