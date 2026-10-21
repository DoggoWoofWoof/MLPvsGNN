"""zu2b.link_pass with identical outputs, faster (docs/U2B_LINK_RULE.md, section 5).

- choose_targets reads each variant's kinds through a boolean mask in place of np.isin, and computes a hit's title
  groups once per distinct mask.
- The context choice scores each (passage, title group) once across the variants, with zu2b's own expression
  (`float((vectors @ source).max())` on a fresh float32 array of the group's rows), and reads a chunk's missing group
  vectors in one call.

Nothing else changes: the scan, the gates, the AMBIG guard, the tie order (the first group with the highest score) and
the edge order are zu2b's. `python zu2b_fast.py --selftest` checks it equal to zu2b.link_pass on toy passages;
`check DATASET N` does so on a dataset's first N passages under both rules (zu2b_modal.py runs it before using it).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import u1b  # noqa: E402
import u1d  # noqa: E402
import zu2b  # noqa: E402


def kind_masks(variants):
    return {v: np.isin(np.arange(4), spec[0]) for v, spec in variants.items()}


def choose_targets(S, i, hits, common, need, variants, capmid, km):
    """zu2b.choose_targets, same results."""
    res = {v: [] for v in variants}
    for k, ok in hits:
        if not ok:
            continue
        com = bool(common[S.lowid[k]])
        if com and k not in capmid:
            continue
        nodes, kinds = S.owners(k)
        keep = nodes != i
        nodes, kinds = nodes[keep], kinds[keep]
        if nodes.size == 0:
            continue
        by_mask = {}
        for v, spec in variants.items():
            exact_first = spec[1]
            if com and spec[2:] != ("capmid",):
                continue
            m = km[v][kinds]
            if exact_first and (kinds[m] == 0).any():
                m &= kinds == 0
            if not m.any():
                continue
            key = m.tobytes()
            gs = by_mask.get(key)
            if gs is None:
                gs = by_mask[key] = np.unique(S.group[nodes[m]])
            if gs.size > zu2b.AMBIG:
                continue
            if gs.size == 1:
                res[v].append(int(gs[0]))
            else:
                need.append((i, v, tuple(gs.tolist())))
    return res


def link_pass(S, texts, titles_lo, rule, common, dense_get, variants, lo=0, chunk=u1d.CHUNK, cache_rows=400000,
              progress=None):
    """zu2b.link_pass, same results."""
    edges = {v: ([], []) for v in variants}
    stats = {v: {"surface_links": 0, "ambiguous": 0} for v in variants}
    km = kind_masks(variants)
    gcache, cached = {}, 0
    for c0 in range(0, len(texts), chunk):
        chosen, need = {}, []
        for off in range(c0, min(c0 + chunk, len(texts))):
            i = lo + off
            if rule.descriptive and not zu2b.is_descriptive(titles_lo[off], texts[off]):
                chosen[i] = {v: [] for v in variants}
                continue
            hits, _st, cm = zu2b.scan(S, texts[off], rule)
            chosen[i] = choose_targets(S, i, hits, common, need, variants, cm, km)
        if need:
            src = np.asarray(sorted({x[0] for x in need}), np.int64)
            sv = dict(zip(src.tolist(), dense_get(src)))
            used = {g for _i, _v, gs in need for g in gs}
            miss = sorted(used - gcache.keys())
            if miss:
                rows = [S.group_nodes(g) for g in miss]
                nrow = sum(r.size for r in rows)
                if cached + nrow > cache_rows:
                    gcache = {g: a for g, a in gcache.items() if g in used}
                    cached = sum(a.shape[0] for a in gcache.values())
                vec = dense_get(np.concatenate(rows))
                o = 0
                for g, r in zip(miss, rows):
                    gcache[g] = vec[o:o + r.size].copy()
                    o += r.size
                cached += nrow
                del vec
            score, best_of = {}, {}
            for i, v, gs in need:
                bg = best_of.get((i, gs))
                if bg is None:
                    best, bg = -9.0, -1
                    for g in gs:
                        s = score.get((i, g))
                        if s is None:
                            s = score[(i, g)] = float((gcache[g] @ sv[i]).max())
                        if s > best:
                            best, bg = s, g
                    best_of[(i, gs)] = bg
                chosen[i][v].append(bg)
                stats[v]["ambiguous"] += 1
        for i, per in chosen.items():
            for v, gl in per.items():
                stats[v]["surface_links"] += len(gl)
                tg = set()
                for g in gl:
                    tg.update(S.group_nodes(g).tolist())
                tg.discard(i)
                for j in sorted(tg):
                    edges[v][0].append(i)
                    edges[v][1].append(j)
        if progress is not None:
            progress(min(c0 + chunk, len(texts)), {v: len(e[0]) for v, e in edges.items()})
    return ({v: (np.asarray(s_, np.int32), np.asarray(d_, np.int32)) for v, (s_, d_) in edges.items()}, stats)


def same(a, b):
    ea, sa = a
    eb, sb = b
    return (sa == sb and ea.keys() == eb.keys()
            and all(np.array_equal(ea[v][0], eb[v][0]) and np.array_equal(ea[v][1], eb[v][1]) for v in ea))


def check(served, d, n_check, rules=("new", "u1d")):
    """zu2b.link_pass and this one on d's first n_check passages, under each rule: True when equal."""
    titles, texts, n, _sha = u1b.read_nodes(served, d, 0, n_check)
    D = u1d.Dense(served, d, n)
    out = {}
    for name in rules:
        rule = zu2b.NEW_RULE if name == "new" else zu2b.U1D_RULE
        variants = zu2b.VARIANTS if name == "new" else u1d.VARIANTS
        S = zu2b.Surfaces(titles, rule)
        cap, low = zu2b.stat_pass(S, texts, titles[:n_check], rule)
        common = low > cap
        t0 = time.time()
        a = zu2b.link_pass(S, texts, titles[:n_check], rule, common, D.get, variants)
        t1 = time.time()
        b = link_pass(S, texts, titles[:n_check], rule, common, D.get, variants)
        t2 = time.time()
        out[name] = {"equal": same(a, b), "secs_zu2b": round(t1 - t0, 1), "secs_fast": round(t2 - t1, 1),
                     "edges": {v: int(e[0].size) for v, e in a[0].items()}}
    return out


def _selftest():
    titles = ["Apple", "Orchard", "Apple Records", "Leeds United F.C.", "Leeds", "Apple"]
    texts = ["Apple is a company.", "In the orchard an apple fell, and Apple sued Apple Records.",
             "Apple Records is a label; Leeds United played Leeds.", "Leeds United F.C. is a club in Leeds.",
             "Leeds is a city. Apple and Orchard.", "Apple, the fruit."]
    rng = np.random.default_rng(0)
    vec = rng.standard_normal((len(titles), 1536)).astype(np.float32)

    def get(r):
        return vec[np.asarray(r, np.int64)].copy()

    for rule, variants in ((zu2b.NEW_RULE, zu2b.VARIANTS), (zu2b.U1D_RULE, u1d.VARIANTS)):
        S = zu2b.Surfaces(titles, rule)
        cap, low = zu2b.stat_pass(S, texts, titles, rule)
        for common in (low > cap, np.zeros_like(low, bool), np.ones_like(low, bool)):
            for ch in (1, 2, 1000):
                for cr in (1, 400000):
                    a = zu2b.link_pass(S, texts, titles, rule, common, get, variants, chunk=ch)
                    b = link_pass(S, texts, titles, rule, common, get, variants, chunk=ch, cache_rows=cr)
                    assert same(a, b), (rule, ch, cr)
    print("zu2b_fast selftest: OK")


if __name__ == "__main__":
    if sys.argv[1:2] == ["--selftest"]:
        _selftest()
    elif sys.argv[1:2] == ["check"]:
        import json
        served_, _f = u1b.package_root(False)
        print(json.dumps(check(served_, sys.argv[2], int(sys.argv[3]))))
    else:
        raise SystemExit(__doc__)
