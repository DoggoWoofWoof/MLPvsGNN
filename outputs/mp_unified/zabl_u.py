"""ABL-U (docs/ABL_LINK_FAMILY_ABLATION.md, section 8): U2b's link rule on the two KB datasets, exactly the code that
built W2_S on the passage datasets, run on the only text a KB entity has: its name and its KB facts.

    python outputs/mp_unified/zabl_u.py build DATASET [--host]      -> outputs/abl/u/<D>/W2_S.npz and build.json
    python outputs/mp_unified/zabl_u.py --selftest

Node text for the rule (the encoder's node vectors are not touched): the names of the entity's distinct KB neighbours,
both directions, in node-id order, joined by "; " (its own name is left out, so that no name is read as a mention of
a shorter name inside it). An entity with no neighbour keeps its name alone (a node whose text is its title names
nothing, U2b switch 4). Titles, rule (U2b's NEW_RULE), variant (W2), the
sibling links (form S), the AMBIG guard, the capitalization statistic and the context choice by the frozen dense
vectors are zu2b's and zu2b_fast's, unchanged. Writes only under outputs/abl/u/.
"""
from __future__ import annotations

import os
import sys

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
sys.dont_write_bytecode = True

import argparse  # noqa: E402
from array import array  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import u1b  # noqa: E402
import u1d  # noqa: E402
import zu2b  # noqa: E402
import zu2b_fast  # noqa: E402

DECLARED = "docs/ABL_LINK_FAMILY_ABLATION.md, section 8"
OUT = ROOT / "outputs" / "abl" / "u"
KB = ("metaqa", "webqsp")
SEP = "; "
VARIANT = {"W2": ((0,), False)}                       # the chosen graph W2_S is variant W2 with the sibling links


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write_json(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1), encoding="utf-8")
    os.replace(tmp, p)


def kb_neighbours(n, src, dst):
    """Distinct undirected KB neighbours of each node (no self), rows sorted by node id."""
    s, d = np.asarray(src, np.int64), np.asarray(dst, np.int64)
    keep = s != d
    return zu2b.undirected_csr(n, s[keep], d[keep])


def texts_of(titles, indptr, col):
    """The rule's text of every node: its neighbours' names joined by SEP (a neighbour with an empty name is left
    out); the node's own name alone when it has none."""
    names = [t.strip() for t in titles]
    out = []
    for i, nm in enumerate(names):
        nb = [names[j] for j in col[indptr[i]:indptr[i + 1]].tolist() if names[j]]
        out.append(SEP.join(nb) if nb else nm)
    return out


# ── amendment A4 (7 Nov 2026): the same scan and the same CSR, linear in a node's text and lean in memory ──────────────
# U2b's scan reads a node's text once per matched surface (occ_look's full-text find, bounded_look's) and copies text[:p]
# per occurrence (u1d.sent_start), so a node costs about degree^2.6; WebQSP has hubs of 130k and 121k neighbours and the
# first build ran 10 h without finishing its first pass. The functions below return exactly what zu2b.scan returns
# (same hits, same counts, same capmid) and exactly what zu2b.undirected_csr(sibling_pairs) returns; `check` proves it.

SCAN_SLOW = zu2b.scan                                  # U2b's own, kept for the check and for the u1d boundary


def sent_start_fast(text, p):
    """u1d.sent_start without the copy: no non-space character before p, or the last one ends a sentence."""
    i = p
    while i > 0 and text[i - 1].isspace():
        i -= 1
    return i == 0 or text[i - 1] in u1d.SENT_END


def pair_index(T):
    """(single, pair) of the table's run tuples: the first runs of the one-run titles, and {(run 1, run 2): ascending lengths}
    of the longer ones. A title that occurs at run k starts with runs k and k+1, so trying only the lengths listed under that
    pair finds every title the first-run lengths found (U2b tried every length listed under run 1 alone: 382 lengths, 92,000
    tuple elements per 'award' in WebQSP's titles, the cost of the first pass there)."""
    pi = T.__dict__.get("_abl_pairs")
    if pi is None:
        single, pair = set(), {}
        for tup in T.idx:
            if len(tup) == 1:
                single.add(tup[0])
            else:
                pair.setdefault(tup[:2], set()).add(len(tup))
        pi = T.__dict__["_abl_pairs"] = (single, {k: tuple(sorted(v)) for k, v in pair.items()})
    return pi


def occurrences(T, tl):
    """{surface k: ascending positions} of every bounded occurrence (zu2b.occ_look's test) of the lowercased surfaces of the
    table T in the lowercased text tl, for exactly the k that zu2b.Table.match returns. One pass over the text's runs: a
    bounded occurrence of a title is a run of whole word runs, so it starts at a run whose tuple the table indexes."""
    spans = [(m.start(), m.group()) for m in u1b.WORD.finditer(tl)]
    runs = [g for _s, g in spans]
    nr, m, isw = len(runs), len(tl), u1b._isw
    single, pair = pair_index(T)
    idx, low, key = T.idx, T.low, T.key
    off = T.__dict__.setdefault("_abl_off", {})
    toks = set(runs)
    out = {}
    for k, r in enumerate(runs):
        cand = []
        if r in single:
            cand.append(idx[(r,)])
        if k + 1 < nr:
            ls = pair.get((r, runs[k + 1]))
            if ls:
                for L in ls:
                    if k + L > nr:
                        break
                    js = idx.get(tuple(runs[k:k + L]))
                    if js:
                        cand.append(js)
        if not cand:
            continue
        start = spans[k][0]
        for js in cand:
            for j in js:
                if key[j] not in toks:
                    continue
                lw = low[j]
                o = off.get(j)
                if o is None:
                    o = off[j] = u1b.WORD.search(lw).start()
                p, n = start - o, len(lw)
                if p < 0 or not tl.startswith(lw, p):
                    continue
                if (p == 0 or not isw(tl[p - 1])) and (p + n >= m or not isw(tl[p + n])):
                    out.setdefault(j, []).append(p)
    return out


def scan_fast(S, text, rule):
    """zu2b.scan's (hits, counts, capmid), same values; the u1d boundary keeps U2b's own function."""
    if rule.boundary == "u1d":
        return SCAN_SLOW(S, text, rule)
    tl = text.lower()
    same = len(tl) == len(text)
    occ = occurrences(S.table, tl)
    out, stat, capmid, done_low = [], {}, set(), set()
    for k in sorted(occ):
        sf = S.surf[k]
        low = S.table.low[k]
        lid = int(S.lowid[k])
        ok = False
        count = lid not in done_low
        done_low.add(lid)
        for p in occ[k]:
            if not same:
                ok = True
                break
            sl = text[p:p + len(low)]
            if sl[1:] == sf[1:] and sl[0].lower() == sf[0].lower():
                ok = True
            mid = not sent_start_fast(text, p)
            if mid and sl == sf and sf[:1].isupper():
                capmid.add(k)
            if count and mid:
                c = stat.setdefault(lid, [0, 0])
                if sl[0].isupper():
                    c[0] += 1
                elif sl == low:
                    c[1] += 1
        out.append((k, ok))
    return out, stat, capmid


def scan_memo(S, text, rule):
    """scan_fast, remembered by text (amendment A4, third part). scan is a pure function of (S, rule, text), WebQSP's node texts
    repeat (1,109,157 distinct of 2,592,894) and the link pass scans the same texts as the first pass, so each distinct text is
    scanned once. The memo holds the three outputs as plain integers (hit k*2+ok, capmid, the counts as lid, c, low triples);
    every call returns fresh objects equal to scan_fast's."""
    memo = S.__dict__.setdefault("_abl_memo", {}).setdefault(id(rule), {})
    cnt = S.__dict__.setdefault("_abl_memo_n", [0, 0])
    cnt[0] += 1
    v = memo.get(text)
    if v is None:
        hits, st, cm = scan_fast(S, text, rule)
        flat = []
        for lid, (c, l_) in st.items():
            flat += (lid, c, l_)
        memo[text] = (array("q", [2 * k + int(ok) for k, ok in hits]), array("q", flat), tuple(cm))
        cnt[1] += 1
        return hits, st, cm
    hk, fl, cmt = v
    fl = fl.tolist()
    return ([(x >> 1, bool(x & 1)) for x in hk.tolist()],
            {fl[a]: [fl[a + 1], fl[a + 2]] for a in range(0, len(fl), 3)}, set(cmt))


def install_fast_scan():
    """zu2b.stat_pass, zu2b.link_pass and zu2b_fast.link_pass all call zu2b.scan by name at run time."""
    zu2b.scan = scan_memo


def lean_csr(n, s_, t_, group, desc):
    """zu2b.undirected_csr(n, mention + sibling sources, mention + sibling targets) and the number of sibling pairs
    (zu2b.sibling_pairs' size), without holding the pair lists: the keys a*n+b of the mention edges both ways and of each
    title group's clique (every ordered pair of distinct descriptive nodes), sorted once in place and de-duplicated."""
    a = np.concatenate([s_, t_]).astype(np.int64)
    b = np.concatenate([t_, s_]).astype(np.int64)
    pieces = [a * n + b]
    del a, b
    g = np.where(desc, group, -1)
    o = np.argsort(g, kind="stable")
    gs = g[o]
    starts = np.flatnonzero(np.r_[True, gs[1:] != gs[:-1]])
    ends = np.r_[starts[1:], gs.size]
    pairs = 0
    for x, y in zip(starts, ends):
        if gs[x] < 0 or y - x < 2:
            continue
        mem = o[x:y].astype(np.int64)
        c = y - x
        k = np.delete((mem[:, None] * n + mem[None, :]).ravel(), np.arange(c, dtype=np.int64) * (c + 1))
        pairs += int(k.size)
        pieces.append(k)
    keys = np.concatenate(pieces)
    del pieces
    keys.sort()
    if keys.size:
        keep = np.empty(keys.size, bool)
        keep[0] = True
        np.not_equal(keys[1:], keys[:-1], out=keep[1:])
        keys = keys[keep]
        del keep
    a, b = keys // n, keys % n
    del keys
    indptr = np.zeros(n + 1, np.int64)
    np.cumsum(np.bincount(a, minlength=n), out=indptr[1:])
    return indptr, b.astype(np.int32), pairs


def stat_pass_logged(S, texts, titles, rule, progress=None, every=100000):
    """zu2b.stat_pass, the same loop and the same counts, with a progress call every `every` nodes (the first pass of
    WebQSP's first build ran 10 h without a line)."""
    cap = np.zeros(S.n_low, np.int64)
    low = np.zeros(S.n_low, np.int64)
    for off, text in enumerate(texts):
        if progress is not None and off % every == 0:
            progress(off)
        if rule.descriptive and not zu2b.is_descriptive(titles[off], text):
            continue
        _h, st, _c = zu2b.scan(S, text, rule)
        for lid, (c, l_) in st.items():
            cap[lid] += c
            low[lid] += l_
    return cap, low


def build_graph(titles, texts, dense_get, rule=None, lo=0, progress=None, lean=True, stat_progress=None):
    """(indptr, col, record) of W2_S on the given titles and texts: the mention links of variant W2 (the context choice
    by dense_get), plus the sibling links, as an undirected CSR. lean: lean_csr (A4) in place of sibling_pairs +
    undirected_csr, the same arrays."""
    rule = rule or zu2b.NEW_RULE
    n = len(titles)
    S = zu2b.Surfaces(titles, rule)
    desc = np.asarray([zu2b.is_descriptive(t, x) for t, x in zip(titles, texts)], bool)
    cap, low = stat_pass_logged(S, texts, titles, rule, stat_progress)
    common = low > cap
    edges, stats = zu2b_fast.link_pass(S, texts, titles, rule, common, dense_get, VARIANT, lo=lo, progress=progress)
    calls, distinct = S.__dict__.pop("_abl_memo_n", [0, 0])
    S.__dict__.pop("_abl_memo", None)
    if calls:
        log(f"  scan memo: {distinct} texts scanned for {calls} scans")
    s_, t_ = (a.astype(np.int64) for a in edges["W2"])
    key = s_ * n + t_
    if np.unique(key).size != key.size or (s_ == t_).any():
        raise SystemExit("a mention pair repeats or loops")
    if lean:
        ip, col, n_sib = lean_csr(n, s_, t_, S.group, desc)
    else:
        sib_s, sib_d = zu2b.sibling_pairs(S.group, desc)
        ip, col = zu2b.undirected_csr(n, np.concatenate([s_, sib_s]), np.concatenate([t_, sib_d]))
        n_sib = int(sib_s.size)
    gs = np.bincount(S.group[desc]) if desc.any() else np.zeros(1, np.int64)
    rec = {"nodes": n, "descriptive_nodes": int(desc.sum()), "surfaces": len(S.surf), "common_forms": int(common.sum()),
           "forms_seen": int(((cap + low) > 0).sum()), "mention_directed_edges": int(s_.size),
           "sibling_pairs": n_sib, "groups_with_siblings": int((gs > 1).sum()), "largest_group": int(gs.max()),
           "undirected_neighbours": int(col.size), "mean_degree": round(float(np.diff(ip).mean()), 3),
           "link_stats": stats["W2"]}
    return ip, col, rec


def overlap(n, ip, col, kip, kcol):
    """Shares of R's graph entries that are KB neighbour entries, and of the KB's that R's graph holds."""
    def keys(a, b):
        return np.repeat(np.arange(n, dtype=np.int64), np.diff(a)) * n + b.astype(np.int64)
    r, k = keys(ip, col), keys(kip, kcol)
    both = np.intersect1d(r, k, assume_unique=True).size
    return {"r_entries": int(r.size), "kb_entries": int(k.size), "both": int(both),
            "share_of_r_in_kb": round(both / max(1, r.size), 4), "share_of_kb_in_r": round(both / max(1, k.size), 4)}


def code_shas():
    return {x: u1b.sha_src(f) for x, f in (("zabl_u", __file__), ("zu2b", zu2b.__file__),
                                          ("zu2b_fast", zu2b_fast.__file__), ("u1d", u1d.__file__),
                                          ("u1b", u1b.__file__))}


def build_stage(name, host=False, out=None):
    if name not in KB:
        raise SystemExit(f"a KB dataset of {KB}")
    served, freeze = u1b.package_root(host)
    titles, _names, n, nodes_sha = u1b.read_nodes(served, name)
    src, dst, _rel, _fam = u1b.package_structural(served, name)
    kip, kcol = kb_neighbours(n, src, dst)
    log(f"{name}: {n} nodes, {kcol.size} KB neighbour entries")
    install_fast_scan()                                # amendment A4: the same scan, linear in a node's text
    texts = texts_of(titles, kip, kcol)
    recipe_sha = hashlib.sha256((SEP + "|" + json.dumps(texts_of(["a", "b", "c"], *kb_neighbours(
        3, [0, 1], [1, 2])))).encode()).hexdigest()
    dense = u1d.Dense(served, name, n)
    t0 = time.time()
    ip, col, rec = build_graph(titles, texts, dense.get, progress=lambda i, e: log(f"  {i}/{n} nodes, edges {e}"),
                               stat_progress=lambda i: log(f"  first pass {i}/{n} nodes"))
    rec["overlap"] = overlap(n, ip, col, kip, kcol)
    od = (ROOT / out if out else OUT) / name
    od.mkdir(parents=True, exist_ok=True)
    p = od / "W2_S.npz"
    tmp = od / "W2_S.tmp.npz"
    np.savez(tmp, indptr=ip, col=col)
    os.replace(tmp, p)
    rec.update({"declared_in": DECLARED, "stage": "ABL-U", "dataset": name, "graph": "W2_S",
                "rule": "NEW_RULE, variant W2, form S", "freeze_record_sha256": freeze, "nodes_sha256": nodes_sha,
                "text_recipe": {"sep": SEP, "own_name": False, "sha256": recipe_sha}, "code_sha256": code_shas(),
                "dense_manifest_sha256": dense.man_sha, "npz": str(p.relative_to(ROOT)).replace("\\", "/"),
                "npz_sha256": sha_file(p), "secs": round(time.time() - t0, 1), "utc": time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime())})
    write_json(od / "build.json", rec)
    if out:                                            # amendment A4's rebuild check against the filed build
        fz = OUT / name
        if (fz / "W2_S.npz").exists() and (fz / "build.json").exists():
            a, b, ref = np.load(fz / "W2_S.npz"), np.load(p), json.loads((fz / "build.json").read_text())
            same_rec = True
            for key in ("descriptive_nodes", "surfaces", "common_forms", "forms_seen", "mention_directed_edges", "sibling_pairs",
                        "groups_with_siblings", "largest_group", "undirected_neighbours"):
                if ref.get(key) != rec.get(key):
                    same_rec = False
                    log(f"{name}: A4 check: {key} {rec.get(key)} here, {ref.get(key)} in the filed record")
            same_arr = (np.array_equal(a["indptr"], b["indptr"]) and np.array_equal(a["col"], b["col"])
                        and a["col"].dtype == b["col"].dtype and a["indptr"].dtype == b["indptr"].dtype)
            log(f"{name}: A4 rebuild check against the filed {fz.relative_to(ROOT)}: arrays " + ("IDENTICAL" if same_arr else "DIFFERENT")
                + ", record counts " + ("identical" if same_rec else "DIFFERENT"))
            if not (same_arr and same_rec):
                return 1
    log(f"{name}: W2_S {rec['undirected_neighbours']} entries (mean degree {rec['mean_degree']}), "
        f"{rec['overlap']['share_of_r_in_kb']} of them KB neighbours, {rec['overlap']['share_of_kb_in_r']} of the KB's in it; "
        f"common forms {rec['common_forms']}, largest title group {rec['largest_group']}; {rec['secs']} s")
    return 0


# ── amendment A4: the fast scan and the lean CSR against U2b's own ─────────────────────────────────────────────────────


def same_scan(a, b):
    """zu2b.scan's three outputs equal: the hit list in order, the counts, the capmid set."""
    return a[0] == b[0] and a[1] == b[1] and set(a[2]) == set(b[2])


def toy_titles_texts(rng, n_titles=400, n_texts=300):
    """Titles and texts that stress the boundary tests: punctuation, parentheses, commas, digits, underscores, non-ASCII,
    repeated words, mixed case, sentence ends, newlines and no-break spaces."""
    pool = ["alpha", "Bravo", "charlie", "Delta", "echo", "Foxtrot", "golf", "Hotel", "india", "Juliet", "kilo", "Lima",
            "mike", "Novem", "oscar", "papa", "quebec", "Romeo", "sierra", "tango", "9th", "x1", "\u00e9cole", "Stra\u00dfe",
            "\u0130stanbul", "caf\u00e9", "new_york", "St", "the", "of", "and"]
    glue = [" ", " ", " ", ", ", " - ", ": ", " (film)", " (1999 film)", "_", "'s ", " & ", ". "]

    def title():
        w = [pool[int(i)] for i in rng.integers(0, len(pool), int(rng.integers(1, 4)))]
        s = w[0]
        for x in w[1:]:
            s += glue[int(rng.integers(0, len(glue)))] + x
        return s

    titles = [title() for _ in range(n_titles)] + ["Alpha Bravo", "alpha bravo", "Alpha Bravo (film)", "Bravo, Alpha"]
    seps = ["; ", ". ", ".  ", "\n", " \u00a0", "! ", "? ", ", ", ": ", "\" ", " ", "\t", ""]
    texts = []
    for _ in range(n_texts):
        parts = []
        for _k in range(int(rng.integers(1, 14))):
            x = titles[int(rng.integers(0, len(titles)))]
            c = int(rng.integers(0, 5))
            parts.append(x.lower() if c == 0 else x.upper() if c == 1 else x)
            parts.append(seps[int(rng.integers(0, len(seps)))])
        texts.append("".join(parts))
    texts += ["", "alpha", " Alpha Bravo", "x Alpha Bravo", "Alpha Bravox", "xAlpha Bravo.", "\u0130stanbul is", "i\u0307stanbul. \u0130stanbul"]
    return titles, texts


def compare_scans(S, texts, rule, what):
    for i, tx in enumerate(texts):
        a, b = SCAN_SLOW(S, tx, rule), scan_fast(S, tx, rule)
        m1, m2 = scan_memo(S, tx, rule), scan_memo(S, tx, rule)        # the first call fills the memo, the second reads it
        if not (same_scan(a, m1) and same_scan(a, m2) and type(m2[0]) is list and all(type(h[1]) is bool for h in m2[0])):
            raise SystemExit(f"{what}: text {i} differs through the memo ({len(tx)} chars)")
        if not same_scan(a, b):
            raise SystemExit(f"{what}: text {i} differs ({len(tx)} chars): slow {a[0][:3]} {a[1]} fast {b[0][:3]} {b[1]}")


def compare_csr(rng, trials=40):
    for _ in range(trials):
        n = int(rng.integers(4, 60))
        group = rng.integers(0, max(2, n // 3), n).astype(np.int64)
        desc = rng.random(n) < 0.8
        m = int(rng.integers(0, 80))
        s_ = rng.integers(0, n, m).astype(np.int64)
        t_ = rng.integers(0, n, m).astype(np.int64)
        sib_s, sib_d = zu2b.sibling_pairs(group, desc)
        want = zu2b.undirected_csr(n, np.concatenate([s_, sib_s]), np.concatenate([t_, sib_d]))
        got = lean_csr(n, s_, t_, group, desc)
        if not (np.array_equal(want[0], got[0]) and np.array_equal(want[1], got[1]) and want[1].dtype == got[1].dtype
                and want[0].dtype == got[0].dtype and int(sib_s.size) == got[2]):
            raise SystemExit("lean_csr differs from sibling_pairs + undirected_csr on a random graph")


def check(name=None, hub=0, limit=0):
    rule = zu2b.NEW_RULE
    rng = np.random.default_rng(7)
    titles, texts = toy_titles_texts(rng)
    S = zu2b.Surfaces(titles, rule)
    compare_scans(S, texts, rule, "random texts")
    for rl in (zu2b.U1D_RULE,):
        compare_scans(zu2b.Surfaces(titles, rl), texts, rl, "u1d rule")
    compare_csr(rng)
    log(f"check: scan_fast == zu2b.scan on {len(texts)} random texts ({len(S.surf)} surfaces) and lean_csr == "
        f"sibling_pairs + undirected_csr on 40 random graphs")
    if not name:
        return 0
    served, _freeze = u1b.package_root(False)
    ttl, _names, n, _sha = u1b.read_nodes(served, name)
    src, dst, _rel, _fam = u1b.package_structural(served, name)
    kip, kcol = kb_neighbours(n, src, dst)
    names = [x.strip() for x in ttl]
    if hub:
        h = int(np.argmax(np.diff(kip)))
        nb = [names[j] for j in kcol[kip[h]:kip[h + 1]].tolist() if names[j]][:hub]
        Sh = zu2b.Surfaces(nb, rule)
        tx = SEP.join(nb)
        t0 = time.time()
        a = SCAN_SLOW(Sh, tx, rule)
        t1 = time.time()
        b = scan_fast(Sh, tx, rule)
        t2 = time.time()
        if not same_scan(a, b):
            raise SystemExit(f"hub of {len(nb)} neighbours: scan_fast differs from zu2b.scan")
        log(f"check: hub {names[h][:30]!r} first {len(nb)} neighbours ({len(tx)} chars, {len(a[0])} surfaces matched): "
            f"identical; U2b {t1 - t0:.1f} s, fast {t2 - t1:.1f} s")
        return 0
    S = zu2b.Surfaces(ttl, rule)
    texts = texts_of(ttl, kip, kcol)
    take = len(texts) if not limit else min(limit, len(texts))
    t0 = time.time()
    compare_scans(S, texts[:take], rule, f"{name} texts")
    log(f"check: {name}: scan_fast == zu2b.scan on {take} of {len(texts)} node texts ({time.time() - t0:.0f} s for both)")
    if take == len(texts):
        sub = ttl[:take]
        t0 = time.time()
        want = zu2b.stat_pass(S, texts, sub, rule)
        t1 = time.time()
        install_fast_scan()
        got = stat_pass_logged(S, texts, sub, rule, lambda i: log(f"  check pass {i}/{len(texts)}"), 20000)
        t2 = time.time()
        zu2b.scan = SCAN_SLOW
        if not (np.array_equal(want[0], got[0]) and np.array_equal(want[1], got[1])):
            raise SystemExit(f"{name}: stat_pass cap/low differ")
        log(f"check: {name}: stat_pass cap and low identical over {S.n_low} lowercase forms (U2b {t1 - t0:.0f} s, fast {t2 - t1:.0f} s)")
    return 0


# ── self-test ───────────────────────────────────────────────────────────────


def _selftest():
    # a toy KB: 0 Alphaville - 1 Bravo Hotel - 2 Charlie Delta - 3 Echo Foxtrot, node 4 repeats node 0's name and
    # hangs off 3; node 5 has no neighbour
    titles = ["Alphaville", "Bravo Hotel", "Charlie Delta", "Echo Foxtrot", "Alphaville", "Golf Island"]
    src = np.array([0, 1, 2, 3, 3])
    dst = np.array([1, 2, 3, 4, 3])                       # the last pair is a self loop and is dropped
    ip, col = kb_neighbours(6, src, dst)
    nb = {i: col[ip[i]:ip[i + 1]].tolist() for i in range(6)}
    if nb != {0: [1], 1: [0, 2], 2: [1, 3], 3: [2, 4], 4: [3], 5: []}:
        raise SystemExit(f"self-test: neighbours {nb}")
    texts = texts_of(titles, ip, col)
    want = ["Bravo Hotel", "Alphaville; Charlie Delta", "Bravo Hotel; Echo Foxtrot", "Charlie Delta; Alphaville",
            "Echo Foxtrot", "Golf Island"]
    if texts != want:
        raise SystemExit(f"self-test: texts {texts}")
    rng = np.random.default_rng(0)
    vec = rng.normal(size=(6, 8)).astype(np.float32)
    rip, rcol, rec = build_graph(titles, texts, lambda rows: vec[np.asarray(rows, np.int64)])
    r = {i: sorted(set(rcol[rip[i]:rip[i + 1]].tolist())) for i in range(6)}
    # a mention of "Alphaville" reaches both owners of that title (one title group); the siblings 0 and 4 are linked;
    # the isolated node names nothing and is named by no neighbour
    want_r = {0: [1, 3, 4], 1: [0, 2, 4], 2: [1, 3], 3: [0, 2, 4], 4: [0, 1, 3], 5: []}
    if r != want_r:
        raise SystemExit(f"self-test: graph {r} != {want_r}")
    if rec["descriptive_nodes"] != 5 or rec["sibling_pairs"] != 2 or rec["largest_group"] != 2:
        raise SystemExit(f"self-test: record {rec}")
    ov = overlap(6, rip, rcol, ip, col)
    if ov["share_of_kb_in_r"] != 1.0 or ov["both"] != ip[-1] or ov["r_entries"] != rcol.size:
        raise SystemExit(f"self-test: overlap {ov}")
    check()
    log(f"zabl_u self-test: the recipe's texts, the mention links (a duplicated name reaches both owners), the sibling "
        f"links, the isolated node and the overlap check agree on a toy KB ({rcol.size} entries)")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", nargs="?", choices=("build", "check"))
    ap.add_argument("dataset", nargs="?")
    ap.add_argument("--host", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--out", help="build: write under this directory instead of outputs/abl/u (the A4 rebuild check)")
    ap.add_argument("--hub", type=int, default=0, help="check: compare on the first N neighbours of the biggest hub")
    ap.add_argument("--limit", type=int, default=0, help="check: only the first N node texts")
    a = ap.parse_args(argv)
    if a.selftest:
        return _selftest()
    if a.cmd == "check" and a.dataset:
        return check(a.dataset, a.hub, a.limit)
    if a.cmd != "build" or not a.dataset:
        ap.error("build DATASET, check DATASET, or --selftest")
    return build_stage(a.dataset, a.host, a.out)


if __name__ == "__main__":
    sys.exit(main())
