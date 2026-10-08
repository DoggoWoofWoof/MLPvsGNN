"""Screens, twelfth round (docs/SCREENS.md): one idea, a learned question-relation match along the graph's own typed
relation chains, screened on one base. The base is rel's if rel's full run is ADOPTED under the seed null over every
split (outputs/full_rel/grade-nullx), step 1's if it is NOT_ADOPTED; `base` is the gate that says which (exit 0 for the
base that runs).

    python outputs/mp_unified/rmatch.py build --dataset metaqa --carve fit --host
    python outputs/mp_unified/rmatch.py smoke --device cuda --host
    python outputs/mp_unified/rmatch.py base --rel-grade outputs/full_rel/grade-nullx.json --want step1
    python outputs/mp_unified/rmatch.py train --name scr-rmatch --arm rmatch --device cuda --host
    python outputs/mp_unified/rmatch.py train --split L-hotpotqa --name scr-relrm-hp --arm relrm --device cuda --host
    python outputs/mp_unified/rmatch.py read --name scr-rmatch --device cuda --host
    python outputs/mp_unified/rmatch.py compare --new outputs/screen/fits/scr-rmatch \\
        --base outputs/step1/fits/L-musique --out outputs/screen/scr-rmatch
    python outputs/mp_unified/rmatch.py pair-rel --screens outputs/screen/scr-relrm.json,outputs/screen/scr-relrm-hp.json \\
        --out outputs/screen/scr-relrm-pair
    python outputs/mp_unified/rmatch.py recall-rel --null N1,N2,N3,N4 --pair outputs/screen/scr-relrm-pair.json \\
        --out outputs/screen/scr-relrm-pair-recall
    python outputs/mp_unified/rmatch.py grade-rel --full-root outputs/full_relrm --reuse L-musique=A,L-hotpotqa=B
    python outputs/mp_unified/rmatch.py --selftest
On step 1's base the pair and its re-call are screen_pair.py's and screen_recall.py's, unchanged. On rel's they are
relz.py's (each read decided against rel's screen fit of its split), run under relrm's name.

The arms (each changes every training dataset and every read alike):
  rmatch   lean_gpu's model plus a learned match of the question to each hop of the typed relation chains that reach
           a row from the question's seeds, added to every row's score:
               s(v) = lean_gpu's s(v) + g_m log(1 + S_m(v) / 0.01) + g_r log(1 + S_r(v))
               S_m(v) = sum over the entries (c, v) of w_q(c) m_c(v),   S_r(v) = the same sum of w_q(c)
           An entry is a chain c = (z_1 .. z_L), L <= 3 typed steps z = 2 r + d (r the stored relation, d 0 from a
           triple's head to its tail, 1 back), whose walk from one bucket of the question's seeds reaches row v with
           mass m_c(v): chainpop17's exact walk (mass 1/|S_b| on each seed, sent equally over each node's type-z
           out-edges in the pool, renormalised over the chain's non-seed rows). Per row and bucket the 64 heaviest
           entries are kept. `build` writes them once per carve, from the look's chunks, beside step 1's cache.
               log w_q(c) = log pi_q(L) + beta_b + sum over k of log sigmoid(h_qk(z_k))
               h_qk(2 r + d) = kappa_k zcos_q(r) + (A_k q) . (B e~_r + D_d) + dir_kd + bias_k
           q is the question's unit embedding (SEMB's input) and e_r relation r's name embedding by the same frozen
           encoder (outputs/m3b/relations). zcos_q(r) is their cosine, z-scored over the relations on the question's
           own pool; e~_r is e_r's unit vector less the graph's mean. pi_q = softmax(q W + c) over L in 1..3. A
           (3 x 8 x 1536) starts random, kappa at 1, and B, D, dir, bias, W, c, beta and the gates g_m, g_r at zero:
           the arm starts as its base model, bit for bit, from the same initialisation and batches; the listwise loss
           alone trains it, with the model's Adam, learning rate and weight decay. No new column, block or
           hyperparameter of the fit, and nothing reads a label at read time. A graph without typed relations (squad,
           musique, hotpotqa, 2wiki) has no entries, and there the arm is its base model; a question whose SEMB block
           is dropped takes no match. 53,795 parameters.
  relrm    the same on rel's carve and blocks (relcols.py: step 1's nine, then typed_rel, typed_v2, ordered).
  Why (docs/SCREENS.md, 'Within the depth that holds a gold'): on rel's metaqa 3-hop misses at a depth that holds a
  gold, the gold and the top-1 never share their relation columns, yet no fixed question-relation match column favours
  the gold (the best, 0.533), while the GNN prefers it on 0.743. The MLP reads a fixed cosine between the question and
  each relation name; the KB systems learn which relation each hop of the question asks for. This learns that match
  from the frozen embeddings, along the existing graph's chains.
"""
import os
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import contextlib  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402

import relcols as R  # noqa: E402
import relz as Z  # noqa: E402

S2, S = R.S2, R.S
LG, LC, LM = S.LG, S.LC, S.LM
log = S.log
ARM, REL_ARM, REL = "rmatch", "relrm", "rel"
CH_OUT = ROOT / "outputs" / "rmatch" / "cache"
REL_DIR = ROOT / "outputs" / "m3b" / "relations"
TYPED = ("metaqa", "webqsp")
HOPS, MAX_CHAINS, K_ROW, BUCKETS = 3, 20000, 64, 2
RANK, TAU, Q_DIM = 8, 0.01, 1536
CH_KEY = "_chain"
CHUNK = 1 << 21                                          # entries moved to the device at a time
KEYS = ("q_pool_size", "q_edges", "q_seed_local", "q_seed_bucket", "e_u", "e_v", "e_fam", "e_fwd", "e_bwd", "e_rel")
ARRAYS = ("q_ent", "ent_row", "ent_z", "ent_b", "ent_m", "q_nrel", "q_rel")
N_PARAMS = HOPS * RANK * Q_DIM + RANK * Q_DIM + 2 * RANK + 3 * HOPS + 2 * HOPS + Q_DIM * HOPS + BUCKETS + 2
SETTINGS = {"hops": HOPS, "max_chains": MAX_CHAINS, "k_row": K_ROW, "buckets": BUCKETS, "rank": RANK, "tau": TAU,
            "zcos": "over the relations on the question's own pool", "parameters": N_PARAMS,
            "init": "A random (generator seed + 1201), kappa 1, the rest 0"}
SMOKE_CARVES = R.SMOKE_CARVES


# ── the chains (chainpop17's exact walk, z = 2 r + d) ────────────────────────


def row_triples(c, ea, eb):
    """A question's structural stored triples (pool-local head, tail, relation slots), from either message of a pair
    (chainpop17.row_triples)."""
    s = c["e_fam"][ea:eb] == 0
    u = c["e_u"][ea:eb][s].astype(np.int64)
    v = c["e_v"][ea:eb][s].astype(np.int64)
    fw = c["e_fwd"][ea:eb][s] == 1
    bw = c["e_bwd"][ea:eb][s] == 1
    rel = c["e_rel"][ea:eb][s].astype(np.int64)
    return np.r_[u[fw], v[bw]], np.r_[v[fw], u[bw]], np.r_[rel[fw], rel[bw]]


def typed_graph(n, hl, tl, sl, kz):
    """Typed out-edges (src, z, dst), one per distinct (src, z, dst), sorted; each edge's (src, z) out-degree and the
    per-src offsets (chainpop17.typed_graph at the exact level)."""
    ok = sl >= 0
    k = ok.sum(1)
    h, t, r = np.repeat(hl, k), np.repeat(tl, k), sl[ok]
    src = np.r_[h, t]
    dst = np.r_[t, h]
    z = np.r_[2 * r, 2 * r + 1]
    key = np.unique((src * kz + z) * n + dst)
    dst = key % n
    sz = key // n
    z = sz % kz
    src = sz // kz
    if src.size:
        first = np.r_[True, sz[1:] != sz[:-1]]
        starts = np.flatnonzero(first)
        cnt = np.diff(np.r_[starts, src.size])
        deg = np.repeat(cnt, cnt).astype(np.float64)
    else:
        deg = np.zeros(0)
    indptr = np.searchsorted(src, np.arange(n + 1))
    return src, z, dst, deg, indptr


def walk_ref(n, g, seeds, hops, cap):
    """chainpop17.walk, chain by chain: every chain of 1..hops types from the seeds, in prefix order, as (types, nodes,
    mass); it returns after the hop level that takes the count above cap. The selftest's reference for walk."""
    src, z, dst, deg, indptr = g
    out = []
    if seeds.size == 0 or src.size == 0:
        return out, False
    front = [((), seeds, np.full(seeds.size, 1.0 / seeds.size))]
    for _ in range(hops):
        nxt = []
        for pre, idx, w in front:
            lens = indptr[idx + 1] - indptr[idx]
            tot = int(lens.sum())
            if tot == 0:
                continue
            sel = np.repeat(indptr[idx] - np.r_[0, np.cumsum(lens)[:-1]], lens) + np.arange(tot)
            ww = np.repeat(w, lens) / deg[sel]
            key = z[sel] * n + dst[sel]
            uq, inv = np.unique(key, return_inverse=True)
            m = np.bincount(inv, weights=ww)
            zz, vv = uq // n, uq % n
            st = np.flatnonzero(np.r_[True, zz[1:] != zz[:-1]])
            en = np.r_[st[1:], zz.size]
            for a_, b_ in zip(st, en):
                nxt.append((pre + (int(zz[a_]),), vv[a_:b_], m[a_:b_]))
        out.extend(nxt)
        front = nxt
        if len(out) > cap:
            return out, True
    return out, False


def walk(n, g, seeds, hops, cap, batch=1 << 22):
    """walk_ref vectorised over each level's chains, with its chains, order and masses bit for bit: a list of levels,
    each (types (J, l), offsets (J + 1,), nodes, masses), a level's chains in walk_ref's order and each chain's nodes
    sorted; it returns after the level that takes the count above cap. A level expands groups of whole chains of at
    most `batch` out-edges at a time (one chain alone may exceed it)."""
    src, z, dst, deg, indptr = g
    levels = []
    if seeds.size == 0 or src.size == 0:
        return levels, False
    K = int(z.max()) + 1
    types = np.zeros((1, 0), np.int64)
    off = np.array([0, seeds.size], np.int64)
    node = seeds.astype(np.int64)
    mass = np.full(seeds.size, 1.0 / seeds.size)
    total = 0
    for _ in range(hops):
        J = off.size - 1
        lens = indptr[node + 1] - indptr[node]
        cs = np.r_[0, np.cumsum(lens)]
        ccs = np.cumsum(cs[off[1:]] - cs[off[:-1]])
        bounds = [0]
        while bounds[-1] < J:
            base = int(ccs[bounds[-1] - 1]) if bounds[-1] else 0
            bounds.append(max(int(np.searchsorted(ccs, base + batch, side="right")), bounds[-1] + 1))
        nt, no, nv, nm = [], [], [], []
        at = 0
        for j0, j1 in zip(bounds[:-1], bounds[1:]):
            a, b_ = int(off[j0]), int(off[j1])
            ln = lens[a:b_]
            tot = int(ln.sum())
            if tot == 0:
                continue
            pc = np.repeat(np.arange(j0, j1), np.diff(off[j0:j1 + 1]))
            sel = np.repeat(indptr[node[a:b_]] - np.r_[0, np.cumsum(ln)[:-1]], ln) + np.arange(tot)
            ww = np.repeat(mass[a:b_], ln) / deg[sel]
            key = (np.repeat(pc, ln) * K + z[sel]) * n + dst[sel]
            uq, inv = np.unique(key, return_inverse=True)
            m = np.bincount(inv, weights=ww)
            pz = uq // n
            st = np.flatnonzero(np.r_[True, pz[1:] != pz[:-1]])
            nt.append(np.concatenate([types[pz[st] // K], (pz[st] % K)[:, None]], axis=1))
            no.append(st + at)
            nv.append(uq % n)
            nm.append(m)
            at += uq.size
        if not nv:
            break
        types, off = np.concatenate(nt), np.r_[np.concatenate(no), at]
        node, mass = np.concatenate(nv), np.concatenate(nm)
        levels.append((types, off, node, mass))
        total += types.shape[0]
        if total > cap:
            return levels, True
    return levels, False


def candidates_ref(chains, seeds):
    """chainpop17's candidates: each chain's non-seed nodes, its mass renormalised over them; a chain that reaches only
    seeds is dropped. The selftest's reference for question_entries."""
    out = []
    for types, nodes, mass in chains:
        keep = ~np.isin(nodes, seeds)
        if keep.any():
            m = mass[keep]
            out.append((types, nodes[keep], m / m.sum()))
    return out


def question_entries(n, hl, tl, sl, seeds_by_bucket, kz, hops=HOPS, cap=MAX_CHAINS, k_row=K_ROW):
    """One question's entries over its seed buckets, ((row, z (k, hops) padded with -1, bucket, mass), per-bucket
    statistics): each bucket's chains, every (chain, row) one entry; per row and bucket the k_row heaviest are kept
    (ties to the earlier chain), in chain order."""
    g = typed_graph(n, hl, tl, sl, kz)
    rows, zs, bs, ms, stats = [], [], [], [], []
    for b, seeds in enumerate(seeds_by_bucket):
        st = {"seeds": int(seeds.size), "capped": 0, "chains": 0, "before_cap": 0, "entries": 0, "mass": 0.0,
              "dropped_mass": 0.0}
        levels, capped = walk(n, g, seeds, hops, cap)
        st["capped"] = int(capped)
        J = sum(t.shape[0] for t, _o, _v, _m in levels)
        if J:
            zc = np.full((J, hops), -1, np.int64)
            at = 0
            for t, _o, _v, _m in levels:
                zc[at:at + t.shape[0], :t.shape[1]] = t
                at += t.shape[0]
            cid = np.repeat(np.arange(J), np.concatenate([np.diff(o) for _t, o, _v, _m in levels]))
            node = np.concatenate([v for _t, _o, v, _m in levels])
            mass = np.concatenate([m for _t, _o, _v, m in levels])
            seed = np.zeros(n, bool)
            seed[seeds] = True
            keep = ~seed[node]
            ck = cid[keep]
            live = np.bincount(ck, minlength=J) > 0
            row = node[keep]
            mass = mass[keep] / np.bincount(ck, weights=mass[keep], minlength=J)[ck]
            cid = (np.cumsum(live) - 1)[ck]
            zc = zc[live]
            st["chains"] = int(live.sum())
        if J and row.size:
            order = np.lexsort((cid, -mass, row))
            rs = row[order]
            first = np.r_[True, rs[1:] != rs[:-1]]
            start = np.maximum.accumulate(np.where(first, np.arange(rs.size), 0))
            keep = np.zeros(row.size, bool)
            keep[order[np.arange(rs.size) - start < k_row]] = True
            st.update({"before_cap": int(row.size), "entries": int(keep.sum()), "mass": float(mass.sum()),
                       "dropped_mass": float(mass[~keep].sum())})
            rows.append(row[keep])
            zs.append(zc[cid[keep]])
            bs.append(np.full(int(keep.sum()), b, np.int64))
            ms.append(mass[keep])
        stats.append(st)
    if not rows:
        return (np.zeros(0, np.int64), np.zeros((0, hops), np.int64), np.zeros(0, np.int64), np.zeros(0)), stats
    return (np.concatenate(rows), np.concatenate(zs), np.concatenate(bs), np.concatenate(ms)), stats


# ── build ────────────────────────────────────────────────────────────────────


def look_relations(d):
    """The look's relations record; every record of the carve must agree, and its relations must be typed."""
    rels = [json.loads(p.read_text(encoding="utf-8")).get("relations") for p in sorted(Path(d).glob("record*.json"))]
    if not rels or any(r != rels[0] for r in rels) or not (rels[0] or {}).get("typed"):
        raise SystemExit(f"{d}: the look's records do not agree on typed relations ({rels[:1]})")
    return rels[0]


def build(ds, carve, out_root=CH_OUT, cache_root=LC.OUT, look_root=LM.LOOK, rel_dir=REL_DIR, placement=None):
    t0 = time.time()
    if ds not in TYPED:
        raise SystemExit(f"rmatch build: {ds} has no typed relations (the arm builds {TYPED})")
    d, files, _ids, head = LC.look_chunks(ds, carve, look_root)
    rel = look_relations(d)
    n_rel, k_rel = int(rel["n_relations"]), int(rel["k_rel"])
    ef = Path(rel_dir) / f"{ds}_rel_embeddings.npy"
    shape = np.load(ef, mmap_mode="r").shape
    if shape != (n_rel, Q_DIM):
        raise SystemExit(f"{ef}: {shape}, the look has {n_rel} relations")
    relations = {"n_relations": n_rel, "kz": 2 * n_rel, "offset": rel.get("offset"), "k_rel": k_rel,
                 "embeddings": ef.name, "embeddings_sha256": LC.sha_file(ef)}
    parts, recs = LC.part_dirs(ds, carve, cache_root)
    summary = {}
    for p, r in zip(parts, recs):
        if r["look"]["carve_ids_sha256"] != head["carve_ids_sha256"] or r["look"]["records"] != head["records"]:
            raise SystemExit(f"{p}: built from another look than {d}")
        tp = time.time()
        lo, hi = r["chunks"]
        acc = {k: [] for k in ARRAYS}
        n_all = []
        tot = [{"questions_with_seeds": 0, "capped": 0, "no_chain": 0, "chains": 0, "before_cap": 0, "entries": 0,
                "mass": 0.0, "dropped_mass": 0.0} for _ in range(BUCKETS)]
        for ch in range(lo, hi):
            with np.load(files[ch]) as zf:
                c = {k: zf[k] for k in KEYS}
            qps = c["q_pool_size"].astype(np.int64)
            eo = np.r_[0, np.cumsum(c["q_edges"].astype(np.int64))]
            if int(eo[-1]) != c["e_u"].size or c["e_rel"].shape != (c["e_u"].size, k_rel):
                raise SystemExit(f"{files[ch]}: {c['e_u'].size} edges and e_rel {c['e_rel'].shape} against q_edges "
                                 f"{int(eo[-1])} and k_rel {k_rel}")
            for i in range(qps.size):
                n = int(qps[i])
                if n > np.iinfo(np.int16).max:
                    raise SystemExit(f"{files[ch]}: question {i} has a pool of {n} rows, above int16")
                hl, tl, sl = row_triples(c, int(eo[i]), int(eo[i + 1]))
                if sl.size and int(sl.max()) >= n_rel:
                    raise SystemExit(f"{files[ch]}: question {i} has relation {int(sl.max())} of {n_rel}")
                sl0, bk = c["q_seed_local"][i], c["q_seed_bucket"][i]
                ok = sl0 >= 0
                if bool(((bk[ok] < 0) | (bk[ok] >= BUCKETS)).any()) or bool((sl0[ok] >= n).any()):
                    raise SystemExit(f"{files[ch]}: question {i} has seeds {sl0[ok]} in buckets {bk[ok]}")
                seeds = [np.unique(sl0[ok & (bk == b)]).astype(np.int64) for b in range(BUCKETS)]
                (row, zz, bb, mm), qst = question_entries(n, hl, tl, sl, seeds, 2 * n_rel)
                acc["q_ent"].append(row.size)
                acc["ent_row"].append(row.astype(np.int16))
                acc["ent_z"].append(zz.astype(np.int16))
                acc["ent_b"].append(bb.astype(np.int8))
                acc["ent_m"].append(mm.astype(np.float16))
                qr = np.unique(sl[sl >= 0])
                acc["q_nrel"].append(qr.size)
                acc["q_rel"].append(qr.astype(np.int16))
                for b, s_ in enumerate(qst):
                    t_ = tot[b]
                    t_["questions_with_seeds"] += int(s_["seeds"] > 0)
                    t_["no_chain"] += int(s_["seeds"] > 0 and s_["chains"] == 0)
                    for k in ("capped", "chains", "before_cap", "entries", "mass", "dropped_mass"):
                        t_[k] += s_[k]
            n_all.append(qps)
        arrays = {"q_ent": np.asarray(acc["q_ent"], np.int64), "q_nrel": np.asarray(acc["q_nrel"], np.int64)}
        for k in ("ent_row", "ent_z", "ent_b", "ent_m", "q_rel"):
            arrays[k] = np.concatenate(acc[k])
        n_all = np.concatenate(n_all)
        if n_all.size != r["queries"] or int(n_all.sum()) != r["rows"]:
            raise SystemExit(f"{p}: {n_all.size} queries and {int(n_all.sum())} rows against {r['queries']} and {r['rows']}")
        if bool((arrays["ent_row"].astype(np.int64) >= np.repeat(n_all, arrays["q_ent"])).any()):
            raise SystemExit(f"{p}: an entry's row is outside its pool")
        if (p / "n.npy").exists():
            if not np.array_equal(np.asarray(LC.load_array(p, "n", r, verify=True)).astype(np.int64), n_all):
                raise SystemExit(f"{p}: the chunks' pool sizes are not step 1's cached n")
            against = "IDENTICAL"
        else:
            against = "step 1's arrays are not on disk here"
        out = Path(out_root) / ds / carve / p.name
        out.mkdir(parents=True, exist_ok=True)
        shas = {k: R.save_npy(out, k, v) for k, v in arrays.items()}
        for t_ in tot:
            t_["dropped_mass_share"] = round(t_["dropped_mass"] / t_["mass"], 6) if t_["mass"] else 0.0
            t_["mass"], t_["dropped_mass"] = round(t_["mass"], 3), round(t_["dropped_mass"], 3)
        qe = arrays["q_ent"]
        rec = {"dataset": ds, "carve": carve, "part": p.name, "step1_record_sha256": LC.sha_file(p / "record.json"),
               "chunks": [lo, hi], "queries": int(n_all.size), "rows": r["rows"], "entries": int(qe.sum()),
               "rel_entries": int(arrays["q_nrel"].sum()), "look": head["records"],
               "carve_ids_sha256": head["carve_ids_sha256"], "relations": relations,
               "walk": {k: SETTINGS[k] for k in ("hops", "max_chains", "k_row", "buckets")},
               "buckets": tot, "entries_per_question": {"mean": round(float(qe.mean()), 1) if qe.size else 0.0,
                                                        "p50": int(np.percentile(qe, 50)) if qe.size else 0,
                                                        "p90": int(np.percentile(qe, 90)) if qe.size else 0,
                                                        "max": int(qe.max()) if qe.size else 0},
               "n_against_step1": against,
               "arrays": {k: {"dtype": v.dtype.str, "shape": list(v.shape), "sha256": shas[k]} for k, v in arrays.items()},
               "placement": placement or {"where": "laptop"}, "script_sha256": LC.sha_src(__file__),
               "seconds": round(time.time() - tp, 1), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        R.write_json(out / "record.json", rec)
        summary[p.name] = {"queries": int(n_all.size), "entries": int(qe.sum()), "against": against}
        log(f"  {ds}/{carve} {p.name}: {n_all.size} queries, {int(qe.sum())} entries (p50 {rec['entries_per_question']['p50']} "
            f"a question); n {against}; bucket 0 capped {tot[0]['capped']}, dropped mass {tot[0]['dropped_mass_share']}; "
            f"bucket 1 capped {tot[1]['capped']}, dropped mass {tot[1]['dropped_mass_share']} ({time.time() - t0:.0f}s)")
    log(f"rmatch build {ds}/{carve}: {len(parts)} part(s) ({time.time() - t0:.0f}s)")
    return summary


# ── the carves ───────────────────────────────────────────────────────────────


def load_chains(c, ch_root=CH_OUT, rel_dir=REL_DIR, cache_root=LC.OUT, verify=True):
    """A typed carve's entries in host memory (its parts in order, each tied to its step-1 part's record, every entry
    checked in range) and its relation tables on the carve's device; None for a graph without typed relations."""
    if c.ds not in TYPED:
        return None
    t0 = time.time()
    parts, recs = LC.part_dirs(c.ds, c.carve, cache_root)
    hdrs = []
    for p, r in zip(parts, recs):
        cp = Path(ch_root) / c.ds / c.carve / p.name
        h = json.loads((cp / "record.json").read_text(encoding="utf-8"))
        if h["step1_record_sha256"] != c.part_shas[p.name] or h["queries"] != r["queries"] or h["rows"] != r["rows"]:
            raise SystemExit(f"{cp}: not built from {p}")
        hdrs.append((cp, h))
    rel = hdrs[0][1]["relations"]
    if any(h["relations"] != rel for _cp, h in hdrs):
        raise SystemExit(f"{c.ds}/{c.carve}: the parts were built on different relations")
    ef = Path(rel_dir) / rel["embeddings"]
    if LC.sha_file(ef) != rel["embeddings_sha256"]:
        raise SystemExit(f"{ef}: sha256 is not the build's")
    n_rel = int(rel["n_relations"])
    E = np.load(ef).astype(np.float64)
    if E.shape != (n_rel, Q_DIM):
        raise SystemExit(f"{ef}: {E.shape}, the build's relations are {n_rel}")
    U = E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)
    Q = c.n_np.size
    ne, nr = sum(h["entries"] for _cp, h in hdrs), sum(h["rel_entries"] for _cp, h in hdrs)
    ch = {"q_ent": np.empty(Q, np.int64), "q_nrel": np.empty(Q, np.int64), "ent_row": np.empty(ne, np.int16),
          "ent_z": np.empty((ne, HOPS), np.int16), "ent_b": np.empty(ne, np.int8), "ent_m": np.empty(ne, np.float16),
          "q_rel": np.empty(nr, np.int16)}
    qa = ea = ra = 0
    for (cp, h), r in zip(hdrs, recs):
        if r["query_range"][0] != qa:
            raise SystemExit(f"{cp}: its questions do not follow {qa}")
        arr = {}
        for k in ARRAYS:
            f = cp / f"{k}.npy"
            if verify and LC.sha_file(f) != h["arrays"][k]["sha256"]:
                raise SystemExit(f"{f}: sha256 is not its record's")
            arr[k] = np.load(f)
            if arr[k].dtype.str != h["arrays"][k]["dtype"] or list(arr[k].shape) != h["arrays"][k]["shape"]:
                raise SystemExit(f"{f}: {arr[k].dtype} {arr[k].shape}, its record says otherwise")
        q, e_, r_ = arr["q_ent"].size, int(arr["q_ent"].sum()), int(arr["q_nrel"].sum())
        if q != r["queries"] or e_ != arr["ent_row"].size or e_ != h["entries"] or r_ != arr["q_rel"].size:
            raise SystemExit(f"{cp}: {q} questions, {e_} entries and {r_} relations do not match its arrays")
        z = arr["ent_z"]
        pad = z < 0
        bad = [bool((arr["ent_row"] < 0).any()),
               bool((arr["ent_row"].astype(np.int64) >= np.repeat(c.n_np[qa:qa + q], arr["q_ent"])).any()),
               z.shape[1:] != (HOPS,), bool((z >= 2 * n_rel).any()), bool((z < -1).any()), bool(pad[:, 0].any()),
               bool((pad[:, :-1] & ~pad[:, 1:]).any()), bool(((arr["ent_b"] < 0) | (arr["ent_b"] >= BUCKETS)).any()),
               not bool(np.isfinite(arr["ent_m"]).all()), bool((arr["ent_m"] < 0).any()),
               bool((arr["q_rel"] < 0).any()), bool((arr["q_rel"] >= n_rel).any())]
        if any(bad):
            raise SystemExit(f"{cp}: an entry is out of range (checks {bad})")
        ch["q_ent"][qa:qa + q], ch["q_nrel"][qa:qa + q] = arr["q_ent"], arr["q_nrel"]
        for k in ("ent_row", "ent_z", "ent_b", "ent_m"):
            ch[k][ea:ea + e_] = arr[k]
        ch["q_rel"][ra:ra + r_] = arr["q_rel"]
        qa, ea, ra = qa + q, ea + e_, ra + r_
    if qa != Q:
        raise SystemExit(f"{c.ds}/{c.carve}: the built parts hold {qa} of {Q} questions")
    ch["e_off"] = np.r_[0, np.cumsum(ch["q_ent"])]
    ch["r_off"] = np.r_[0, np.cumsum(ch["q_nrel"])]
    ch["rel_unit"] = torch.from_numpy(U.astype(np.float32)).to(c.device)
    ch["rel_centered"] = torch.from_numpy((U - U.mean(0, keepdims=True)).astype(np.float32)).to(c.device)
    ch["n_rel"] = n_rel
    ch["records"] = {cp.name: LC.sha_file(cp / "record.json") for cp, _h in hdrs}
    host_gb = sum(ch[k].nbytes for k in ARRAYS) / 1e9
    log(f"  {c.ds}/{c.carve}: {ne} chain entries ({host_gb:.2f} GB in host memory), {n_rel} relations, from "
        f"{Path(ch_root) / c.ds / c.carve} ({len(parts)} part(s), {time.time() - t0:.0f}s)")
    return ch


def chain_batch(c, qs):
    """The entries of questions qs, in qs's order, for c.batch: each entry's batch question and batch row (its question's
    rows lie where lean_gpu.CacheCarve.batch puts them), chain, bucket and mass (host arrays; the model moves them to the
    device a chunk at a time); each question's own relations; and the graph's relation tables."""
    ch = c.chains
    qs = np.asarray(qs, np.int64)
    B = qs.size
    cnt = c.n_np[qs]
    rseg = np.cumsum(cnt) - cnt
    ne = ch["q_ent"][qs]
    idx = np.repeat(ch["e_off"][qs] - (np.cumsum(ne) - ne), ne) + np.arange(int(ne.sum()))
    eq = np.repeat(np.arange(B), ne)
    nr = ch["q_nrel"][qs]
    ridx = np.repeat(ch["r_off"][qs] - (np.cumsum(nr) - nr), nr) + np.arange(int(nr.sum()))
    dev = c.device
    return {"eq": eq, "row": rseg[eq] + ch["ent_row"][idx].astype(np.int64), "z": ch["ent_z"][idx],
            "b": ch["ent_b"][idx], "m": ch["ent_m"][idx],
            "qr_q": torch.from_numpy(np.repeat(np.arange(B), nr)).to(dev),
            "qr_r": torch.from_numpy(ch["q_rel"][ridx].astype(np.int64)).to(dev),
            "rel_unit": ch["rel_unit"], "rel_centered": ch["rel_centered"]}


def chain_carve(base, ch_root=CH_OUT, rel_dir=REL_DIR):
    """base's carve class, with a typed graph's chain entries (build) beside it: batch() adds them to the features under
    CH_KEY; every block, row and gold is base's, unchanged."""

    class ChainCarve(base):
        CH_ROOT, REL_DIR = ch_root, rel_dir

        def __init__(self, ds, carve, basis, device, cache_root=LC.OUT, verify=True, score2=False):
            super().__init__(ds, carve, basis, device, cache_root, verify, score2)
            self.chains = load_chains(self, type(self).CH_ROOT, type(self).REL_DIR, cache_root, verify)

        def nbytes(self):
            tabs = 0 if self.chains is None else sum(
                self.chains[k].numel() * self.chains[k].element_size() for k in ("rel_unit", "rel_centered"))
            return super().nbytes() + tabs

        def batch(self, qs, blocks):
            feats, nq, base_z, gold = super().batch(qs, blocks)
            if self.chains is not None:
                feats[CH_KEY] = chain_batch(self, qs)
            return feats, nq, base_z, gold

    ChainCarve.__name__ = ChainCarve.__qualname__ = f"Chain{base.__name__}"
    return ChainCarve


ChainCarveBase = chain_carve(S.ARMS["base"][1])
ChainCarveRel = chain_carve(R.RelCarveRel)


# ── the match ────────────────────────────────────────────────────────────────


class ChainMatch(LG.LeanMLP8D):
    """rmatch and relrm: lean_gpu's forward plus the gated chain features of each row (zero gates at the start)."""

    def __init__(self, blocks, widths, hidden=128, dropout=0.1, seed=0, arm="ctl", ctx="none"):
        if ctx != "none":
            raise SystemExit("rmatch takes ctx none only")
        if "SEMB" not in blocks:
            raise SystemExit("rmatch needs the SEMB block (the question's embedding)")
        super().__init__(blocks, widths, hidden, dropout, seed, arm, ctx)
        self.j_semb = self.blocks.index("SEMB")
        g = torch.Generator().manual_seed(int(seed) + 1201)
        self.cm_a = nn.Parameter(torch.randn(HOPS, RANK, Q_DIM, generator=g) / Q_DIM ** 0.5)
        self.cm_b = nn.Parameter(torch.zeros(RANK, Q_DIM))
        self.cm_d = nn.Parameter(torch.zeros(2, RANK))
        self.cm_kappa = nn.Parameter(torch.ones(HOPS))
        self.cm_dir = nn.Parameter(torch.zeros(HOPS, 2))
        self.cm_bias = nn.Parameter(torch.zeros(HOPS))
        self.cm_pi_w = nn.Parameter(torch.zeros(Q_DIM, HOPS))
        self.cm_pi_b = nn.Parameter(torch.zeros(HOPS))
        self.cm_beta = nn.Parameter(torch.zeros(BUCKETS))
        self.cm_gate = nn.Parameter(torch.zeros(2))

    @staticmethod
    @torch.no_grad()
    def zcos(q, ch):
        """(B, n_rel): each question's cosine with each relation, z-scored over the relations on its own pool (0 for a
        question with fewer than two, or none that vary)."""
        cos = q @ ch["rel_unit"].T
        qq, rr = ch["qr_q"], ch["qr_r"]
        B = q.shape[0]
        v = cos[qq, rr]
        cnt = torch.zeros(B, dtype=cos.dtype, device=cos.device).index_add_(0, qq, torch.ones_like(v))
        mean = torch.zeros(B, dtype=cos.dtype, device=cos.device).index_add_(0, qq, v) / cnt.clamp_min(1.0)
        dv = v - mean[qq]
        var = torch.zeros(B, dtype=cos.dtype, device=cos.device).index_add_(0, qq, dv * dv) / cnt.clamp_min(1.0)
        ok = (cnt >= 2) & (var > 1e-12)
        z = (cos - mean[:, None]) / var.clamp_min(1e-12).sqrt()[:, None]
        return torch.where(ok[:, None], z, torch.zeros_like(z))

    def step_logp(self, q, ch):
        """(B, 3, 2 n_rel): log sigmoid of each hop position's match to each typed step z = 2 r + d."""
        zc = self.zcos(q, ch)
        P = torch.einsum("khd,bd->bkh", self.cm_a, q)
        Gr = torch.einsum("bkh,rh->bkr", P, ch["rel_centered"] @ self.cm_b.T)
        Gd = torch.einsum("bkh,dh->bkd", P, self.cm_d)
        h = (self.cm_kappa[None, :, None, None] * zc[:, None, :, None] + Gr[..., None] + Gd[:, :, None, :]
             + self.cm_dir[None, :, None, :] + self.cm_bias[None, :, None, None])
        return -Fn.softplus(-h).reshape(q.shape[0], HOPS, -1)

    def chain_feats(self, feats, N):
        """(N,) each: log(1 + S_m / TAU) and log(1 + S_r) of every batch row."""
        ch = feats[CH_KEY]
        q = Fn.normalize(feats["SEMB"][0], dim=1)
        lp = self.step_logp(q, ch)
        Kz = lp.shape[2]
        flat = lp.reshape(-1)
        lpi = Fn.log_softmax(q @ self.cm_pi_w + self.cm_pi_b, dim=1)
        dev = q.device
        k = torch.arange(HOPS, device=dev)
        Sm = torch.zeros(N, dtype=q.dtype, device=dev)
        Sr = torch.zeros(N, dtype=q.dtype, device=dev)
        for s in range(0, ch["eq"].size, CHUNK):
            e = min(ch["eq"].size, s + CHUNK)
            eq = torch.from_numpy(ch["eq"][s:e]).to(dev)
            row = torch.from_numpy(ch["row"][s:e]).to(dev)
            z = torch.from_numpy(ch["z"][s:e]).to(dev).long()
            b = torch.from_numpy(ch["b"][s:e]).to(dev).long()
            m = torch.from_numpy(ch["m"][s:e]).to(dev).to(q.dtype)
            valid = z >= 0
            idx = (eq[:, None] * HOPS + k[None, :]) * Kz + z.clamp_min(0)
            lw = (flat[idx] * valid.to(q.dtype)).sum(1) + lpi[eq, valid.sum(1) - 1] + self.cm_beta[b]
            w = lw.exp()
            Sm = Sm.index_add(0, row, w * m)
            Sr = Sr.index_add(0, row, w)
        return torch.log1p(Sm / TAU), torch.log1p(Sr)

    def forward(self, feats, keep, nq, B, base_z):
        s = super().forward(feats, keep, nq, B, base_z)
        if CH_KEY not in feats:
            return s
        fm, fr = self.chain_feats(feats, nq.numel())
        return s + (self.cm_gate[0] * fm + self.cm_gate[1] * fr) * keep[nq, self.j_semb]


S.ARMS.update({ARM: (ChainMatch, ChainCarveBase), REL_ARM: (ChainMatch, ChainCarveRel)})


# ── train and read (lean_screen2's; relrm under rel's block sets) ────────────


def sets_of(arm):
    """rel's block sets for relrm (relcols.with_sets), lean_gpu's own otherwise."""
    return R.with_sets(REL) if arm == REL_ARM else contextlib.nullcontext()


def built_records(root=CH_OUT):
    return {str(f.parent.relative_to(root)).replace("\\", "/"): LC.sha_file(f)
            for f in sorted(Path(root).glob("*/*/part_*/record.json"))}


def train(argv, split):
    arm = R.arm_of(argv)
    with sets_of(arm):
        rc = S2.train(argv, split)
    name, out_root = S2.where(argv)
    sj = out_root / name / "screen.json"
    if sj.exists():
        rec = json.loads(sj.read_text(encoding="utf-8"))
        rec["rmatch_sha256"] = LC.sha_src(__file__)
        rec["rmatch"] = SETTINGS
        rec["rmatch_records"] = built_records()
        if arm == REL_ARM:
            rec["relcols_sha256"] = LC.sha_src(R.__file__)
            rec["relcols_blocks"] = list(R.ARM_BLOCKS[REL])
            rec["relcols_records"] = R.built_records()
        LC.write_json(sj, rec)
    return rc


def read(argv):
    name, out_root = S2.where(argv)
    arm = json.loads((out_root / name / "screen.json").read_text(encoding="utf-8"))["arm"]
    with sets_of(arm):
        return S2.main(["read"] + argv)


# ── the base gate, and relz.py's rel-based records under relrm's name ────────


def base_gate(rel_grade, want):
    """Exit 0 when rel's re-grade names this base: want 'rel' passes on ADOPT, want 'step1' on NOT_ADOPTED. A missing
    or INCOMPLETE re-grade passes neither (exit 1), and neither screen runs."""
    if want not in ("rel", "step1"):
        raise SystemExit(f"rmatch base: --want rel or step1, not {want}")
    p = Path(rel_grade)
    v = json.loads(p.read_text(encoding="utf-8")).get("verdict") if p.exists() else None
    ok = (want == "rel" and v == "ADOPT") or (want == "step1" and v == "NOT_ADOPTED")
    log(f"rmatch base gate: {p} {v}; the {want} base {'runs' if ok else 'does not run'}")
    return 0 if ok else 1


@contextlib.contextmanager
def as_relrm():
    """relz.py's pair, re-call and grade against rel's fits, under relrm's name; restored on the way out."""
    saved = Z.ARM
    Z.ARM = REL_ARM
    try:
        yield
    finally:
        Z.ARM = saved


FIX = (("the tenth round", "the twelfth round"), ("docs/FULL_ROUND10.md", "docs/FULL_ROUND12.md"))


def restamp(out):
    """A record relz.py wrote under relrm's name: this round's documents named in its md, this file's sha added."""
    out = Path(out)
    md = out.with_suffix(".md")
    if md.exists():
        t = md.read_text(encoding="utf-8")
        for a, b in FIX:
            t = t.replace(a, b)
        md.write_text(t, encoding="utf-8")
    js = out.with_suffix(".json")
    if js.exists():
        rec = json.loads(js.read_text(encoding="utf-8"))
        rec["rmatch_sha256"] = LC.sha_src(__file__)
        LC.write_json(js, rec)


# ── smoke ────────────────────────────────────────────────────────────────────


def carve_check(ds, carve, device):
    """One carve through rmatch's carve against lean_gpu.CacheCarve: the base matrix, golds and rows unchanged; on a
    typed graph its entries loaded (load_chains checks each in range); on its first questions the arm at its start
    scores as its base model bit for bit, with finite chain features."""
    # Its forwards run before train or read has bound lean_mlp's z-score to the device (8 Oct fix: the smoke failed in
    # 4 s on the card, before any training).
    LG.bind_device_ops()
    a = LG.CacheCarve(ds, carve, "2wiki", device)
    b = ChainCarveBase(ds, carve, "2wiki", device)
    rec = {"base_equal": bool(torch.equal(a.X, b.X) and torch.equal(a.gold, b.gold) and torch.equal(a.row, b.row)),
           "typed": b.chains is not None}
    ok = rec["base_equal"] and rec["typed"] == (ds in TYPED)
    blocks = [x for x in LG.SETS["pick"] if x in b.widths]
    widths = {x: b.widths[x] for x in blocks}
    qs = np.arange(min(16, b.rows))
    feats, nq, base_z, _gold = b.batch(qs, blocks)
    keep = torch.ones((qs.size, len(blocks)), dtype=torch.float32, device=b.device)
    torch.manual_seed(0)
    base = S.ARMS["base"][0](blocks, widths, 32, seed=0).to(b.device).eval()
    torch.manual_seed(0)
    arm = ChainMatch(blocks, widths, 32, seed=0).to(b.device).eval()
    with torch.no_grad():
        rec["identity_at_start"] = bool(torch.equal(base(feats, keep, nq, qs.size, base_z),
                                                    arm(feats, keep, nq, qs.size, base_z)))
        if b.chains is not None:
            fm, fr = arm.chain_feats(feats, nq.numel())
            rec["entries"] = int(b.chains["q_ent"].sum())
            rec["entries_first_questions"] = int(feats[CH_KEY]["eq"].size)
            rec["features_finite"] = bool(torch.isfinite(fm).all() and torch.isfinite(fr).all())
            rec["rows_reached_first_questions"] = round(float((fr > 0).to(torch.float32).mean()), 4)
            ok = ok and rec["features_finite"] and rec["entries_first_questions"] > 0
    ok = ok and rec["identity_at_start"]
    del a, b
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return ok, rec


def head_norms(models_pt):
    """{candidate: (|gates|, |B|, |D|, |W|)} of a fit's saved states (None where a state holds no match)."""
    blob = torch.load(models_pt, map_location="cpu", weights_only=False)
    out = {}
    for name, st in blob["states"].items():
        out[name] = [float(st[k].norm()) for k in ("cm_gate", "cm_b", "cm_d", "cm_pi_w")] if "cm_gate" in st else None
    return out


def moved(norms):
    return bool(norms) and all(v is not None and all(np.isfinite(v)) and v[0] > 0 and v[1] > 0 for v in norms.values())


def smoke(device, host, out_root=None):
    """The carve check on metaqa select, webqsp s1eval and 2wiki select; then step 1's base arm once, rmatch twice (the
    repeat must be IDENTICAL) and relrm once, one epoch each on metaqa's select carve, each then read on it. rmatch's and
    relrm's states must hold gates and a relation map B that moved from zero and are finite; rmatch's scores must differ
    from the base arm's (the same init and batches, so a difference is the match); relrm's fit must hold rel's blocks
    live. A crash fails it (exit 1)."""
    t0 = time.time()
    root = Path(out_root or S.OUT / "smoke12")
    h = ["--host"] if host else []
    rec, ok = {"carves": {}}, True
    for ds, cv in SMOKE_CARVES:
        good, chk = carve_check(ds, cv, device)
        rec["carves"][f"{ds}={cv}"] = chk
        ok = ok and good
    common = ["--train", "metaqa=select", "--basis", "2wiki", "--variants", "p", "--config", "2e-3:1e-4:0.1:1:0",
              "--device", device, "--out-root", str(root)] + h
    for arm, rep in (("base", "1"), (ARM, "2"), (REL_ARM, "1")):
        nm = f"smoke-{arm}"
        rc = main(["train", "--name", nm, "--arm", arm, "--repeat", rep] + common)
        rr = main(["read", "--name", nm, "--read", "metaqa=select", "--device", device, "--out-root", str(root)] + h)
        sj = json.loads((root / nm / "screen.json").read_text(encoding="utf-8"))
        tj = json.loads((root / nm / "train.json").read_text(encoding="utf-8"))
        blocks = tj["variants"]["p"]["blocks"]
        rec[arm] = {"train_rc": rc, "repeats": int(rep), "read_rc": rr, "screen_arm": sj.get("arm"), "blocks": blocks,
                    "head_norms": head_norms(root / nm / "models.pt"), "peak_gpu_gb": tj.get("peak_gpu_gb")}
        ok = ok and rc == 0 and rr == 0 and sj.get("arm") == arm
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
    rec[ARM]["repeat"] = "IDENTICAL" if rec[ARM]["train_rc"] == 0 else "DIFFERENT"
    rec["match_moved"] = {a: moved(rec[a]["head_norms"]) for a in (ARM, REL_ARM)}
    rec["base_has_no_match"] = all(v is None for v in rec["base"]["head_norms"].values())
    rec["relrm_rel_blocks_live"] = all(b in rec[REL_ARM]["blocks"] for b in R.ARM_BLOCKS[REL])
    rec["rmatch_blocks_are_base"] = rec[ARM]["blocks"] == rec["base"]["blocks"]
    ok = (ok and all(rec["match_moved"].values()) and rec["base_has_no_match"] and rec["relrm_rel_blocks_live"]
          and rec["rmatch_blocks_are_base"])
    fb = root / "smoke-base" / "reads" / "metaqa__select.npz"
    fq = root / f"smoke-{ARM}" / "reads" / "metaqa__select.npz"
    fr = root / f"smoke-{REL_ARM}" / "reads" / "metaqa__select.npz"
    if fb.exists() and fq.exists() and fr.exists():
        a, b, c = np.load(fb), np.load(fq), np.load(fr)
        rec["same_questions"] = [str(x) for x in a["ids"]] == [str(x) for x in b["ids"]] == [str(x) for x in c["ids"]]
        rec["scores_differ_from_base"] = bool(not np.array_equal(a["scores64"], b["scores64"]))
        k = [str(x) for x in a["candidates"]].index("p@ep0")
        rec["p@ep0_hit@1"] = {"base": float(a["hit"][k].mean()), ARM: float(b["hit"][k].mean()),
                              REL_ARM: float(c["hit"][k].mean())}
        ok = ok and rec["same_questions"] and rec["scores_differ_from_base"]
    else:
        rec["scores_differ_from_base"] = None
        ok = False
    rec.update({"ok": ok, "seconds": time.time() - t0, "script_sha256": LC.sha_src(__file__)})
    LC.write_json(root / "smoke.json", rec)
    log(f"smoke12: {json.dumps(rec)}; {'ok' if ok else 'FAILED'}")
    return 0 if ok else 1


# ── selftest ─────────────────────────────────────────────────────────────────


def brute_chains(n, src, z, dst, Kz, seeds, hops):
    """Dense per-type transition matrices and every type sequence: the chains with mass, by brute force
    (chainpop17.brute_chains)."""
    A = np.zeros((Kz, n, n))
    for s_, z_, d_ in zip(src, z, dst):
        A[z_, s_, d_] = 1.0
    deg = A.sum(2, keepdims=True)
    T = np.divide(A, deg, out=np.zeros_like(A), where=deg > 0)
    m0 = np.zeros(n)
    m0[seeds] = 1.0 / seeds.size
    out = {}
    front = {(): m0}
    for _ in range(hops):
        nxt = {}
        for pre, m in front.items():
            for z_ in range(Kz):
                m2 = m @ T[z_]
                if (m2 > 0).any():
                    nxt[pre + (z_,)] = m2
        out.update(nxt)
        front = nxt
    return out


def brute_entries(n, hl, tl, sl, seeds_by_bucket, kz, hops):
    """{(bucket, chain, row): mass}: brute_chains, each chain renormalised over its non-seed rows."""
    ok = sl >= 0
    k = ok.sum(1)
    h, t, r = np.repeat(hl, k), np.repeat(tl, k), sl[ok]
    src, dst, z = np.r_[h, t], np.r_[t, h], np.r_[2 * r, 2 * r + 1]
    out = {}
    for b, seeds in enumerate(seeds_by_bucket):
        if seeds.size == 0:
            continue
        for types, mv in brute_chains(n, src, z, dst, kz, seeds, hops).items():
            mv = mv.copy()
            mv[seeds] = 0.0
            if mv.sum() <= 0:
                continue
            mv /= mv.sum()
            for v in np.flatnonzero(mv > 0):
                out[(b, types, int(v))] = float(mv[v])
    return out


def toy_question(rng, n, n_rel, rels, k_rel=2):
    """One question's chunk arrays: n rows, messages among them (most structural, each read forward, backward or both)
    with one or two relation slots drawn from rels, and seeds in two buckets."""
    E = int(rng.integers(n, 3 * n))
    u, v = rng.integers(0, n, E), rng.integers(0, n, E)
    u, v = u[u != v], v[u != v]
    E = u.size
    fam = np.where(rng.random(E) < 0.8, 0, rng.integers(1, 3, E)).astype(np.int8)
    fw = (rng.random(E) < 0.6).astype(np.int8)
    bw = ((rng.random(E) < 0.5) | (fw == 0)).astype(np.int8)
    rel = np.full((E, k_rel), -1, np.int16)
    rel[:, 0] = rng.choice(rels, E)
    two = rng.random(E) < 0.3
    rel[two, 1] = rng.choice(rels, int(two.sum()))
    rel[rng.random(E) < 0.1, 0] = -1
    sl0, bk = np.full(10, -1, np.int64), np.full(10, -1, np.int64)
    k0, k1 = int(rng.integers(0, 3)), int(rng.integers(0, 4))
    sl0[:k0 + k1] = rng.integers(0, n, k0 + k1)
    bk[:k0], bk[k0:k0 + k1] = 0, 1
    return {"q_pool_size": n, "q_edges": E, "q_seed_local": sl0, "q_seed_bucket": bk, "e_u": u.astype(np.int16),
            "e_v": v.astype(np.int16), "e_fam": fam, "e_fwd": fw, "e_bwd": bw, "e_rel": rel}


def toy_roots(tmp, rng, n_rel=5, chunks=(3, 3)):
    """A typed toy look (metaqa/toy; two chunks), step 1's cache records for it (two parts, a chunk each, with n.npy),
    and its relation embeddings, under tmp. Returns the questions' chunk arrays."""
    look = tmp / "look" / "metaqa" / "toy"
    (look / "chunks").mkdir(parents=True)
    qs = []
    for ci, nqc in enumerate(chunks):
        qq = [toy_question(rng, int(rng.integers(3, 10)), n_rel,
                           rng.choice(n_rel, int(rng.integers(2, n_rel + 1)), replace=False)) for _ in range(nqc)]
        A = {k: np.asarray([q[k] for q in qq], np.int64) for k in ("q_pool_size", "q_edges")}
        A.update({k: np.stack([q[k] for q in qq]) for k in ("q_seed_local", "q_seed_bucket")})
        A.update({k: np.concatenate([q[k] for q in qq]) for k in ("e_u", "e_v", "e_fam", "e_fwd", "e_bwd", "e_rel")})
        np.savez(look / "chunks" / f"c{ci:05d}.npz", **A)
        qs += qq
    Q = len(qs)
    rec = {"chunks": list(range(len(chunks))), "n_chunks": len(chunks), "carve_ids_sha256": "toy", "full": True,
           "limit": None, "carve_queries": Q, "chunk_queries": max(chunks), "columns": ["rrf"], "column_blocks": {},
           "relations": {"typed": True, "offset": 0, "n_relations": n_rel, "k_rel": 2}}
    R.write_json(look / "record_0of1.json", rec)
    R.write_json(look / "ids.json", [f"q{i}" for i in range(Q)])
    head_records = {"record_0of1.json": LC.sha_file(look / "record_0of1.json")}
    q0 = r0 = 0
    for pi, nqc in enumerate(chunks):
        p = tmp / "cache" / "metaqa" / "toy" / f"part_{pi}of{len(chunks)}"
        p.mkdir(parents=True)
        n = np.asarray([q["q_pool_size"] for q in qs[q0:q0 + nqc]], np.int64)
        sha = R.save_npy(p, "n", n)
        R.write_json(p / "record.json", {"query_range": [q0, q0 + nqc], "row_range": [r0, r0 + int(n.sum())],
                                         "look": {"carve_queries": Q, "carve_ids_sha256": "toy", "records": head_records},
                                         "chunks": [pi, pi + 1], "queries": nqc, "rows": int(n.sum()),
                                         "arrays": {"n": {"sha256": sha}}})
        q0, r0 = q0 + nqc, r0 + int(n.sum())
    (tmp / "rel").mkdir()
    np.save(tmp / "rel" / "metaqa_rel_embeddings.npy",
            rng.standard_normal((n_rel, Q_DIM)).astype(np.float16))
    return qs


class ToyBase:
    """lean_gpu.CacheCarve's interface on the toy cache, on the CPU: a rank block (5 columns) and SEMB."""

    def __init__(self, ds, carve, basis, device, cache_root=LC.OUT, verify=True, score2=False):
        parts, _recs = LC.part_dirs(ds, carve, cache_root)
        self.ds, self.carve, self.device = ds, carve, torch.device(device)
        self.part_shas = {p.name: LC.sha_file(p / "record.json") for p in parts}
        self.n_np = np.concatenate([np.load(p / "n.npy") for p in parts]).astype(np.int64)
        self.off_np = np.r_[0, np.cumsum(self.n_np)]
        self.rows = self.n_np.size
        N = int(self.off_np[-1])
        g = torch.Generator().manual_seed(5)
        self.q_emb = torch.randn(self.rows, Q_DIM, generator=g)
        self.X = torch.rand(N, 5, generator=g) * 0.03 + 0.001
        self.P = torch.randn(N, LG.L3.STORE_DIM, generator=g)
        self.gold = torch.rand(N, generator=g) < 0.3
        self.widths = {"rank": 5, "SEMB": LM.SEMB_DIM}

    def nbytes(self):
        return 0

    def batch(self, qs, blocks):
        qs = np.asarray(qs, np.int64)
        cnt = self.n_np[qs]
        B = qs.size
        idx = torch.from_numpy(np.repeat(self.off_np[qs] - (np.cumsum(cnt) - cnt), cnt) + np.arange(int(cnt.sum())))
        nq = torch.from_numpy(np.repeat(np.arange(B), cnt))
        feats = {"rank": self.X[idx], "SEMB": (self.q_emb[torch.from_numpy(qs)], self.P[idx])}
        feats = {b: feats[b] for b in blocks}
        return feats, nq, LG.seg_zscore8D(self.X[idx][:, 1:2], nq, B).squeeze(1), self.gold[idx]


def hand_feats(m, ch, c, qs, keep_q):
    """float64 by hand, from the carve's own arrays (not chain_batch): each batch row's gate-weighted match."""
    P = {k: v.detach().double().numpy() for k, v in m.named_parameters() if k.startswith("cm_")}
    U = ch["rel_unit"].double().numpy()
    C = ch["rel_centered"].double().numpy()
    out = []
    for j, qi in enumerate(qs):
        n = int(c.n_np[qi])
        q = c.q_emb[qi].double().numpy()
        q = q / np.linalg.norm(q)
        cos = U @ q
        own = ch["q_rel"][ch["r_off"][qi]:ch["r_off"][qi + 1]].astype(np.int64)
        mu, var = cos[own].mean() if own.size else 0.0, cos[own].var() if own.size else 0.0
        zc = (cos - mu) / np.sqrt(max(var, 1e-12)) if own.size >= 2 and var > 1e-12 else np.zeros_like(cos)
        Pq = np.einsum("khd,d->kh", P["cm_a"], q)
        h = (P["cm_kappa"][:, None, None] * zc[None, :, None] + (Pq @ (C @ P["cm_b"].T).T)[:, :, None]
             + (Pq @ P["cm_d"].T)[:, None, :] + P["cm_dir"][:, None, :] + P["cm_bias"][:, None, None])
        lp = -np.logaddexp(0.0, -h).reshape(HOPS, -1)
        a = q @ P["cm_pi_w"] + P["cm_pi_b"]
        lpi = a - a.max() - np.log(np.exp(a - a.max()).sum())
        Sm, Sr = np.zeros(n), np.zeros(n)
        for e in range(ch["e_off"][qi], ch["e_off"][qi + 1]):
            zz = [int(x) for x in ch["ent_z"][e] if x >= 0]
            lw = lpi[len(zz) - 1] + P["cm_beta"][int(ch["ent_b"][e])] + sum(lp[k_, z_] for k_, z_ in enumerate(zz))
            Sm[int(ch["ent_row"][e])] += np.exp(lw) * float(ch["ent_m"][e])
            Sr[int(ch["ent_row"][e])] += np.exp(lw)
        f = P["cm_gate"][0] * np.log1p(Sm / TAU) + P["cm_gate"][1] * np.log1p(Sr)
        out.append(f * keep_q[j])
    return np.concatenate(out)


def selftest():
    t0 = time.time()
    LG.bind_device_ops()
    rng = np.random.default_rng(12)
    # 1. the walk's entries are the brute force's, every (bucket, chain, row) with its mass; the cap keeps the k
    #    heaviest per row and bucket (ties to the earlier chain), in chain order; a chain cap returns after its level
    n_rel = 4
    for trial in range(40):
        n = int(rng.integers(3, 9))
        tq = toy_question(rng, n, n_rel, rng.choice(n_rel, int(rng.integers(1, n_rel + 1)), replace=False))
        c = {k: np.asarray(v) for k, v in tq.items()}
        hl, tl, sl = row_triples(c, 0, int(c["q_edges"]))
        sl0, bk = c["q_seed_local"], c["q_seed_bucket"]
        seeds = [np.unique(sl0[(sl0 >= 0) & (bk == b)]).astype(np.int64) for b in range(BUCKETS)]
        (row, zz, bb, mm), st = question_entries(n, hl, tl, sl, seeds, 2 * n_rel, k_row=10 ** 9)
        got = {(int(b), tuple(int(x) for x in z if x >= 0), int(r)): float(m) for r, z, b, m in zip(row, zz, bb, mm)}
        want = brute_entries(n, hl, tl, sl, seeds, 2 * n_rel, HOPS)
        assert len(got) == row.size and set(got) == set(want), (trial, len(got), len(want))
        assert all(abs(got[k] - want[k]) < 1e-12 for k in want), trial
        assert sum(s_["entries"] for s_ in st) == row.size and all(s_["dropped_mass"] == 0 for s_ in st)
        for k_row in (1, 2):
            (r2, z2, b2, m2), st2 = question_entries(n, hl, tl, sl, seeds, 2 * n_rel, k_row=k_row)
            keep = np.zeros(row.size, bool)
            for key in set(zip(bb.tolist(), row.tolist())):
                at = [j for j in range(row.size) if (int(bb[j]), int(row[j])) == key]
                keep[sorted(at, key=lambda j: (-mm[j], j))[:k_row]] = True
            assert np.array_equal(r2, row[keep]) and np.array_equal(z2, zz[keep]) and np.array_equal(b2, bb[keep])
            assert np.array_equal(m2, mm[keep]) and sum(s_["entries"] for s_ in st2) == int(keep.sum())
            assert abs(sum(s_["dropped_mass"] for s_ in st2) - float(mm[~keep].sum())) < 1e-9
        g = typed_graph(n, hl, tl, sl, 2 * n_rel)
        for b in range(BUCKETS):
            # the vectorised walk is chainpop17's chain by chain, bit for bit, at every cap and expansion group size
            full, _cap = walk_ref(n, g, seeds[b], HOPS, MAX_CHAINS)
            caps = sorted({MAX_CHAINS, 0, 1, 2, len(full) // 2, max(len(full) - 1, 0)})
            for cap_ in caps:
                ref, rc_ = walk_ref(n, g, seeds[b], HOPS, cap_)
                for batch in (1, 3, 1 << 22):
                    lv, vc_ = walk(n, g, seeds[b], HOPS, cap_, batch)
                    got_c = [(tuple(int(x) for x in t[j]), v[o[j]:o[j + 1]], m[o[j]:o[j + 1]])
                             for t, o, v, m in lv for j in range(t.shape[0])]
                    assert vc_ == rc_ and len(got_c) == len(ref), (trial, cap_, batch)
                    assert all(a[0] == r_[0] and np.array_equal(a[1], r_[1]) and np.array_equal(a[2], r_[2])
                               for a, r_ in zip(got_c, ref)), (trial, cap_, batch)
            one = [x for x in full if len(x[0]) == 1]
            if len(one) >= 2 and len(full) > len(one):
                part, capped = walk_ref(n, g, seeds[b], HOPS, len(one) - 1)
                assert capped and [x[0] for x in part] == [x[0] for x in one]
            # question_entries is chainpop17's candidates, every row in its order (masses to rounding)
            cand = candidates_ref(full, seeds[b])
            (r1, z1, b1, m1), _s = question_entries(n, hl, tl, sl, [seeds[b] if k == b else np.zeros(0, np.int64)
                                                                    for k in range(BUCKETS)], 2 * n_rel, k_row=10 ** 9)
            want_r = np.concatenate([nd for _t, nd, _m in cand]) if cand else np.zeros(0, np.int64)
            want_m = np.concatenate([m_ for _t, _n, m_ in cand]) if cand else np.zeros(0)
            want_z = [t for t, nd, _m in cand for _ in range(nd.size)]
            assert np.array_equal(r1, want_r) and (b1 == b).all() and np.allclose(m1, want_m, rtol=1e-12, atol=0)
            assert [tuple(int(x) for x in z_ if x >= 0) for z_ in z1] == want_z
    # 2. build a toy carve; load it through a chain carve; each batch's entries sit on their question's rows
    tmp = Path(tempfile.mkdtemp(prefix="rmatch_"))
    try:
        qs_arr = toy_roots(tmp, rng)
        roots = {"out_root": tmp / "ch", "cache_root": tmp / "cache", "look_root": tmp / "look", "rel_dir": tmp / "rel"}
        build("metaqa", "toy", **roots)
        recs = [json.loads((tmp / "ch" / "metaqa" / "toy" / f"part_{k}of2" / "record.json").read_text(encoding="utf-8"))
                for k in range(2)]
        assert all(r["n_against_step1"] == "IDENTICAL" and r["relations"]["n_relations"] == 5 for r in recs)
        Cls = chain_carve(ToyBase, tmp / "ch", tmp / "rel")
        c = Cls("metaqa", "toy", "2wiki", "cpu", tmp / "cache")
        ch = c.chains
        for qi, tq in enumerate(qs_arr):
            cq = {k: np.asarray(v) for k, v in tq.items()}
            hl, tl, sl = row_triples(cq, 0, int(cq["q_edges"]))
            sl0, bk = cq["q_seed_local"], cq["q_seed_bucket"]
            seeds = [np.unique(sl0[(sl0 >= 0) & (bk == b)]).astype(np.int64) for b in range(BUCKETS)]
            (row, zz, bb, mm), _st = question_entries(int(cq["q_pool_size"]), hl, tl, sl, seeds, 10)
            e0, e1 = ch["e_off"][qi], ch["e_off"][qi + 1]
            assert np.array_equal(ch["ent_row"][e0:e1], row) and np.array_equal(ch["ent_z"][e0:e1], zz)
            assert np.array_equal(ch["ent_b"][e0:e1], bb) and np.array_equal(ch["ent_m"][e0:e1], mm.astype(np.float16))
            assert np.array_equal(ch["q_rel"][ch["r_off"][qi]:ch["r_off"][qi + 1]], np.unique(sl[sl >= 0]))
        assert int(ch["q_ent"].sum()) > 0 and torch.allclose(ch["rel_unit"].norm(dim=1), torch.ones(5))
        qs = np.array([4, 0, 2, 5])
        feats, nq, bz, gold = c.batch(qs, ["rank", "SEMB"])
        cb = feats[CH_KEY]
        cnt = c.n_np[qs]
        rseg = np.cumsum(cnt) - cnt
        want_rows, want_eq = [], []
        for j, qi in enumerate(qs):
            e0, e1 = ch["e_off"][qi], ch["e_off"][qi + 1]
            want_rows.append(rseg[j] + ch["ent_row"][e0:e1].astype(np.int64))
            want_eq.append(np.full(e1 - e0, j))
        assert np.array_equal(cb["row"], np.concatenate(want_rows)) and np.array_equal(cb["eq"], np.concatenate(want_eq))
        assert all(int(cb["row"][i]) < int(rseg[e] + cnt[e]) for i, e in enumerate(cb["eq"]))
        # 3. at the start the arm is its base model bit for bit; with every parameter set it adds the gated match by
        #    hand; a question whose SEMB is dropped takes none; a batch without entries (an untyped graph) is the base
        blocks = ["rank", "SEMB"]
        B = qs.size
        keep = torch.ones(B, 2)
        torch.manual_seed(0)
        base = LG.LeanMLP8D(blocks, c.widths, 16, dropout=0.1, seed=0).eval()
        torch.manual_seed(0)
        cm = ChainMatch(blocks, c.widths, 16, dropout=0.1, seed=0).eval()
        shared = {k: v for k, v in cm.state_dict().items() if not k.startswith("cm_")}
        assert all(torch.equal(v, base.state_dict()[k]) for k, v in shared.items()) and len(shared) == len(base.state_dict())
        n_cm = sum(p.numel() for k, p in cm.named_parameters() if k.startswith("cm_"))
        assert n_cm == N_PARAMS == 53795, n_cm
        with torch.no_grad():
            for p in cm.out.parameters():
                p.normal_()
            base.load_state_dict({k: v for k, v in cm.state_dict().items() if not k.startswith("cm_")})
            s0 = base(feats, keep, nq, B, bz)
            assert torch.equal(cm(feats, keep, nq, B, bz), s0)
            gen = torch.Generator().manual_seed(3)
            for k, p in cm.named_parameters():
                if k.startswith("cm_"):
                    p.copy_(torch.randn(p.shape, generator=gen) * (0.02 if k in ("cm_b", "cm_pi_w") else 0.5))
            got = (cm(feats, keep, nq, B, bz) - s0).double().numpy()
            want = hand_feats(cm, ch, c, qs, np.ones(B))
            assert np.allclose(got, want, atol=2e-4, rtol=1e-4), np.abs(got - want).max()
            k2 = keep.clone()
            k2[1, blocks.index("SEMB")] = 0.0
            got2 = (cm(feats, k2, nq, B, bz) - base(feats, k2, nq, B, bz)).double().numpy()
            assert np.allclose(got2, hand_feats(cm, ch, c, qs, np.array([1.0, 0.0, 1.0, 1.0])), atol=2e-4, rtol=1e-4)
            assert (got2[nq.numpy() == 1] == 0).all() and np.abs(got[nq.numpy() == 1]).max() > 0
            plain = {k: v for k, v in feats.items() if k != CH_KEY}
            assert torch.equal(cm(plain, keep, nq, B, bz), base(plain, keep, nq, B, bz))
        # 4. from the start only the gates take a gradient; with open gates B, kappa, beta, W, D take finite ones
        torch.manual_seed(0)
        cm0 = ChainMatch(blocks, c.widths, 16, dropout=0.1, seed=0)
        LG.listwiseD(cm0(feats, keep, nq, B, bz), gold, nq, B).backward()
        grads = {k: p.grad for k, p in cm0.named_parameters() if k.startswith("cm_")}
        assert float(grads["cm_gate"].abs().sum()) > 0
        assert all(g_ is None or float(g_.abs().sum()) == 0 for k, g_ in grads.items() if k != "cm_gate")
        cm0.zero_grad()
        with torch.no_grad():
            cm0.cm_gate.fill_(1.0)
            cm0.cm_b.normal_(0, 0.02)
        LG.listwiseD(cm0(feats, keep, nq, B, bz), gold, nq, B).backward()
        assert all(torch.isfinite(p.grad).all() for p in cm0.parameters() if p.grad is not None)
        for k in ("cm_a", "cm_b", "cm_d", "cm_kappa", "cm_pi_w", "cm_beta", "cm_dir", "cm_bias"):
            assert float(dict(cm0.named_parameters())[k].grad.abs().sum()) > 0, k
        # 5. refusals: an untyped graph builds nothing, a carve of it loads none; a build and a load on other relations
        for bad in (lambda: build("2wiki", "toy", **roots), lambda: ChainMatch(["rank"], {"rank": 5}, 16),
                    lambda: ChainMatch(blocks, c.widths, 16, ctx="film")):
            try:
                bad()
                raise AssertionError("a refusal was accepted")
            except SystemExit:
                pass
        shutil.copytree(tmp / "cache" / "metaqa", tmp / "cache" / "2wiki")
        assert Cls("2wiki", "toy", "2wiki", "cpu", tmp / "cache").chains is None
        f = tmp / "rel" / "metaqa_rel_embeddings.npy"
        good = f.read_bytes()
        np.save(f, rng.standard_normal((5, Q_DIM)).astype(np.float16))
        try:
            Cls("metaqa", "toy", "2wiki", "cpu", tmp / "cache")
            raise AssertionError("a load on other relation embeddings was accepted")
        except SystemExit:
            pass
        np.save(f, rng.standard_normal((6, Q_DIM)).astype(np.float16))
        try:
            build("metaqa", "toy", **roots)
            raise AssertionError("a build on six relation embeddings for five relations was accepted")
        except SystemExit:
            pass
        f.write_bytes(good)
        rp = tmp / "cache" / "metaqa" / "toy" / "part_0of2" / "record.json"
        rr = json.loads(rp.read_text(encoding="utf-8"))
        rr["note"] = "another step-1 part"
        R.write_json(rp, rr)
        try:
            Cls("metaqa", "toy", "2wiki", "cpu", tmp / "cache")
            raise AssertionError("a chain part tied to another step-1 part was accepted")
        except SystemExit:
            pass
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    # 6. load_models builds the arm's class under patching, its scores unchanged; the arms and patching restore
    blob = {"candidates": ["p@ep0"], "variants": {"p": {"blocks": blocks, "widths": c.widths, "ctx": "none"}},
            "hidden": 16, "states": {"p@ep0": cm.state_dict()}}
    assert S.ARMS[ARM] == (ChainMatch, ChainCarveBase) and S.ARMS[REL_ARM] == (ChainMatch, ChainCarveRel)
    assert issubclass(ChainCarveBase, S.ARMS["base"][1]) and issubclass(ChainCarveRel, R.RelCarveRel)
    assert S.ARMS[REL] == (S.ARMS["base"][0], R.RelCarveRel)
    for arm in (ARM, REL_ARM):
        with S.patched(arm):
            assert LG.LeanMLP8D is ChainMatch and LG.CacheCarve is S.ARMS[arm][1]
            (name, m, bl), = LG.load_models(blob, "cpu")
            assert isinstance(m, ChainMatch) and bl == blocks
            with torch.no_grad():
                assert torch.equal(m(feats, keep, nq, B, bz), cm(feats, keep, nq, B, bz))
    assert LG.LeanMLP8D is S.ARMS["base"][0] and LG.CacheCarve is S.ARMS["base"][1]
    # 7. relrm trains and reads under rel's block sets, rmatch under lean_gpu's own; both restored after
    saved = {k: list(v) for k, v in LG.SETS.items()}
    with sets_of(REL_ARM):
        assert all(LG.SETS[k] == saved[k] + [b for b in R.ARM_BLOCKS[REL] if b not in saved[k]] for k in saved)
        assert "SEMB" in LG.SETS["pick"]
    assert {k: list(v) for k, v in LG.SETS.items()} == saved
    with sets_of(ARM):
        assert {k: list(v) for k, v in LG.SETS.items()} == saved
    # 8. the base gate: ADOPT runs rel's base, NOT_ADOPTED step 1's; INCOMPLETE or a missing re-grade runs neither
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "g.json"
        for v, want in (("ADOPT", {"rel": 0, "step1": 1}), ("NOT_ADOPTED", {"rel": 1, "step1": 0}),
                        ("INCOMPLETE", {"rel": 1, "step1": 1}), (None, {"rel": 1, "step1": 1})):
            if v is None:
                f.unlink()
            else:
                f.write_text(json.dumps({"verdict": v}), encoding="utf-8")
            for w, rc in want.items():
                assert base_gate(f, w) == rc, (v, w)
        try:
            base_gate(f, "rel2")
            raise AssertionError("the base gate took want rel2")
        except SystemExit:
            pass
        # 9. relz.py's records under relrm's name, restored after (an error too); restamp names this round
        assert Z.ARM == "relz"
        with as_relrm():
            assert Z.ARM == REL_ARM and Z.tag({})["arm"] == REL_ARM
        assert Z.ARM == "relz"
        try:
            with as_relrm():
                raise KeyError("x")
        except KeyError:
            pass
        assert Z.ARM == "relz"
        o = Path(td) / "r"
        o.with_suffix(".md").write_text("the tenth round; docs/FULL_ROUND10.md\n", encoding="utf-8")
        LC.write_json(o.with_suffix(".json"), {"arm": REL_ARM})
        restamp(o)
        assert o.with_suffix(".md").read_text(encoding="utf-8") == "the twelfth round; docs/FULL_ROUND12.md\n"
        assert json.loads(o.with_suffix(".json").read_text(encoding="utf-8"))["rmatch_sha256"] == LC.sha_src(__file__)
    print("selftest: the vectorised walk is chainpop17's, bit for bit, at every cap and group size; the entries are "
          "its candidates' and the brute force's (every bucket, chain and row, with its mass); the row cap "
          "keeps the heaviest per row and bucket, ties to the earlier chain, in chain order; a chain cap returns after "
          "its level; a toy carve builds, loads through a chain carve and puts each entry on its question's batch row; "
          "at the start the arm is its base model bit for bit; with its parameters set it adds the gated match as "
          "computed by hand, none for a question whose SEMB is dropped, none on an untyped graph; from the start only "
          "the gates take a gradient, then every part a finite one; refusals; load_models under patching for both "
          "arms; relrm under rel's block sets; the base gate; relz's records under relrm's name, restored; restamp. "
          f"all checks passed ({time.time() - t0:.0f}s)")
    return 0


# ── main ─────────────────────────────────────────────────────────────────────


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("cmd", nargs="?")
    ap.add_argument("--split", default="L-musique", choices=S2.SPLITS)
    ap.add_argument("--selftest", action="store_true")
    k, rest = ap.parse_known_args(argv)
    if k.selftest:
        return selftest()
    if k.cmd == "build":
        bp = argparse.ArgumentParser()
        bp.add_argument("--dataset", required=True)
        bp.add_argument("--carve", required=True)
        bp.add_argument("--out-root")
        bp.add_argument("--host", action="store_true")
        b = bp.parse_args(rest)
        build(b.dataset, b.carve, Path(b.out_root) if b.out_root else CH_OUT,
              placement={"where": "host" if b.host else "laptop"})
        return 0
    if k.cmd == "smoke":
        sp = argparse.ArgumentParser()
        sp.add_argument("--device", default="cpu")
        sp.add_argument("--host", action="store_true")
        sp.add_argument("--out-root")
        s = sp.parse_args(rest)
        return smoke(s.device, s.host, s.out_root)
    if k.cmd == "base":
        gp = argparse.ArgumentParser()
        gp.add_argument("--rel-grade", required=True)
        gp.add_argument("--want", required=True, choices=("rel", "step1"))
        g = gp.parse_args(rest)
        return base_gate(g.rel_grade, g.want)
    if k.cmd == "train":
        return train([k.cmd] + rest, k.split)
    if k.cmd == "read":
        return read(rest)
    if k.cmd == "compare":
        return S2.main([k.cmd] + rest)
    if k.cmd in ("pair-rel", "recall-rel", "grade-rel"):
        gp = argparse.ArgumentParser()
        gp.add_argument("--screens", default="")
        gp.add_argument("--null", default="")
        gp.add_argument("--pair")
        gp.add_argument("--full-root")
        gp.add_argument("--reuse", default="")
        gp.add_argument("--out")
        g = gp.parse_args(rest)
        null = [x for x in g.null.split(",") if x]
        need = {"pair-rel": g.screens and g.out, "recall-rel": g.pair and len(null) == 4 and g.out,
                "grade-rel": g.full_root}[k.cmd]
        if not need:
            gp.error(f"{k.cmd}: pair-rel needs --screens A,B and --out; recall-rel --pair, the four --null files and "
                     "--out; grade-rel --full-root")
        out = g.out or str(Path(g.full_root) / "grade")
        try:
            with as_relrm():
                if k.cmd == "pair-rel":
                    Z.pair([x for x in g.screens.split(",") if x], out)
                elif k.cmd == "recall-rel":
                    Z.recall(null, g.pair, out)
                else:
                    reuse = dict(x.split("=", 1) for x in g.reuse.split(",") if x)
                    v = Z.grade(g.full_root, reuse, out)["verdict"]
                    restamp(out)
                    return 1 if v == "INCOMPLETE" else 0
        except SystemExit as e:
            log(f"rmatch {k.cmd}: {e}")
            return 2
        restamp(out)
        return 0
    raise SystemExit("rmatch: build, train, read, compare, base, pair-rel, recall-rel, grade-rel, smoke, or --selftest")


if __name__ == "__main__":
    sys.exit(main())
