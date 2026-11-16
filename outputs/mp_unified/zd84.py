"""D84: MultiHop-RAG, the rule-built NER store graded against the oracle pairs off Wikipedia, graph level, no encoder
(docs/D84_MULTIHOPRAG_LINK_ORACLE.md).

    python outputs/mp_unified/zd84.py selftest
    python outputs/mp_unified/zd84.py grade     # G1-G3, the four forms, the measures -> outputs/d84/grade.json (laptop)
    python outputs/mp_unified/zd84.py report    # section 5's call and the beside rows -> outputs/d84/report.md

Nodes are the corpus's articles; the golds are the articles a query's evidence names (labels grade only; no rule reads one).
The forms of R's NER slot: none, N_sp (b1_build.ner_edges, the package's spaCy store), CN_all and F0T (zul1's build, D83's
R.name_comention + R.title_comention). Nothing here reads an embedding. Pair arrays stay in memory; the outputs hold counts.
"""
import os
import sys

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (HERE, ROOT / "scripts", ROOT / "src", ROOT / "src" / "mp_retrieval"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import zap1 as Z  # noqa: E402
import zul1 as UL  # noqa: E402

DECLARED = "docs/D84_MULTIHOPRAG_LINK_ORACLE.md"
RAW = ROOT / "data" / "raw" / "multihoprag"
OUT = ROOT / "outputs" / "d84"
REVISION = "71ac0d0bd1f951d2d6b70311f7d2ae404e1ffa82"
FILES = {"corpus.json": (6785567, "20b61b5ab84de84a927420c5d265b7ec8d859ae49980699958a787ade9e4d28f"),
         "MultiHopRAG.json": (5171312, "03cfb4926461f868684903aadc8024447bdda5bb3f6804741424cce338515bff")}
FORMS = ("none", "N_sp", "CN_all", "F0T")
BOUND = 1.30                                   # UL1's: a form's pairs at most 1.30 times N_sp's
GOLD_COUNTS = (2, 3, 4)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def script_sha256():
    return hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def norm(s):
    return " ".join(str(s or "").split())


# ── G1, G2: the files and the golds ─────────────────────────────────────────


def load(raw=RAW, files=FILES):
    """G1: the pinned bytes and the shapes. Returns (articles, queries, {file: sha256})."""
    shas = {}
    for f, (size, sha) in files.items():
        p = raw / f
        if not p.exists() or p.stat().st_size != size or Z.sha_file(p) != sha:
            raise SystemExit(f"gate G1: {p} is missing or is not the pinned file ({size} bytes, sha256 {sha[:12]})")
        shas[f] = sha
    arts = json.loads((raw / "corpus.json").read_text(encoding="utf-8"))
    qs = json.loads((raw / "MultiHopRAG.json").read_text(encoding="utf-8"))
    if not isinstance(arts, list) or not all(isinstance(a, dict) and "title" in a and "body" in a and "url" in a for a in arts):
        raise SystemExit("gate G1: the corpus is not a list of articles with title, body and url")
    if not isinstance(qs, list) or not all(isinstance(q, dict) and {"query", "question_type", "evidence_list"} <= set(q) for q in qs):
        raise SystemExit("gate G1: the queries are not a list with query, question_type and evidence_list")
    return arts, qs, shas


def golds_of(arts, qs):
    """G2: each kept query's (non-empty evidence) golds = the distinct articles its items name, by url, the title agreeing.
    Returns (ids, golds, types, n_items)."""
    at, dup = {}, []
    for i, a in enumerate(arts):
        u = norm(a["url"])
        if u in at:
            dup.append(u)
        at[u] = i
    if dup:
        raise SystemExit(f"gate G2: {len(dup)} corpus urls name two articles (e.g. {dup[:3]})")
    ids, golds, types, items, bad = [], [], [], [], []
    for j, q in enumerate(qs):
        ev = q["evidence_list"] or []
        if not ev:
            continue
        g = []
        for e in ev:
            i = at.get(norm(e.get("url")))
            if i is None:
                bad.append((j, "no article has the url", norm(e.get("url"))[:80]))
            elif norm(arts[i]["title"]) != norm(e.get("title")):
                bad.append((j, "the url's article has another title", norm(e.get("title"))[:80]))
            else:
                g.append(i)
        ids.append(j)
        golds.append(np.unique(np.asarray(g, np.int64)))
        types.append(str(q["question_type"]))
        items.append(len(ev))
    if bad:
        raise SystemExit(f"gate G2: {len(bad)} evidence items name no single article (e.g. {bad[:5]})")
    return ids, golds, types, items


# ── the forms ───────────────────────────────────────────────────────────────


def rule_forms(arts):
    """CN_all (F0) and F0T over the articles (zul1's build). Returns ({form: sorted unique keys}, stats)."""
    n = len(arts)
    passes = lambda: ((a["title"], a["body"]) for a in arts)  # noqa: E731
    src, grp, tsrc, tgrp, nv = UL.build_entries(passes, n)
    (fa, fb, _fw), (la, lb, _lw) = UL.forms_from_entries(src, grp, tsrc, tgrp, n)
    f0 = UL.pair_keys(fa, fb, n)
    lt = UL.pair_keys(la, lb, n)
    if f0.size != fa.size or lt.size != la.size or not ((fa < fb).all() and (la < lb).all()):
        raise SystemExit("gate G3: a co-mention table holds a pair twice or a pair that is not a < b")
    f0t = np.union1d(f0, lt)
    if not UL.member(f0, f0t).all():
        raise SystemExit("gate G3: CN_all is not inside F0T")
    st = {"keys": int(nv), "name_entries": int(src.size), "title_run_entries": int(tsrc.size),
          "articles_with_a_title_run": int(np.unique(tsrc).size), "lt_pairs": int(lt.size)}
    return {"CN_all": f0, "F0T": f0t}, st


def spacy_form(arts, ner=None):
    """N_sp: b1_build.ner_edges on the bodies in node order (the package's rule, unchanged)."""
    if ner is None:
        import b1_build
        ner = b1_build.ner_edges
    s, d, _w, n_ents = ner([a["body"] for a in arts])
    k = UL.pair_keys(s, d, len(arts))
    if k.size != np.asarray(s).size:
        raise SystemExit("gate G3: N_sp holds a pair twice or a self pair")
    return k, int(n_ents)


# ── the measures ────────────────────────────────────────────────────────────


def connected(m, edges):
    """Whether m nodes form one component under the (a, b) edges."""
    par = list(range(m))

    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x
    for a, b in edges:
        par[find(a)] = find(b)
    return len({find(x) for x in range(m)}) == 1


def gs_flags(keys, golds, n):
    """Per question with two or more golds: whether its golds form one component in keys' pairs among them."""
    out = np.zeros(len(golds), bool)
    for j, g in enumerate(golds):
        if g.size < 2:
            continue
        a, b = np.triu_indices(g.size, 1)
        lk = UL.member(g[a] * n + g[b], keys)
        out[j] = connected(g.size, [(int(x), int(y)) for x, y, ok in zip(a, b, lk) if ok])
    return out


def share_call(c, qidx, nq, base):
    """The GP share with its interval (zap1.boot_diff against nothing) and the lift call against a base rate."""
    d, lo, hi = Z.boot_diff(c, np.zeros(c.size), qidx, nq)
    return {"share": Z.pct(d), "lo": Z.pct(lo), "hi": Z.pct(hi), "density": Z.pct(base),
            "lift": round(d / base, 2) if base > 0 else None,
            "call": "POINTS" if lo > base else ("BELOW_BASE" if hi < base else "NO_LIFT")}


def gp_block(F, golds, n, sel=None):
    """GP rows over the questions in sel (all if None): per form linked, share, lift; F0T and CN_all minus N_sp."""
    gl = [g for j, g in enumerate(golds) if sel is None or sel[j]]   # a group's interval resamples the group's questions only
    nq = len(gl)
    qidx, sk = UL.gp_slots(gl, n)
    if sk.size == 0:
        return None
    pairs = n * (n - 1) / 2
    cov = {f: UL.member(sk, F[f]) for f in FORMS}
    rows = {"slots": int(sk.size), "questions": int(np.unique(qidx).size)}
    for f in FORMS:
        rows[f] = {"linked": int(cov[f].sum()), **share_call(cov[f], qidx, nq, F[f].size / pairs)}
    for f in ("F0T", "CN_all"):
        rows[f"{f}_vs_N_sp"] = UL.grade_diff(cov[f], cov["N_sp"], qidx, nq)
    return rows


def gs_block(F, golds, n, sel=None):
    has = np.array([g.size >= 2 and (sel is None or sel[j]) for j, g in enumerate(golds)])
    if not has.any():
        return None
    q = np.flatnonzero(has)
    unit = np.arange(q.size)
    flags = {f: gs_flags(F[f], golds, n)[q] for f in FORMS}
    rows = {"questions": int(q.size)}
    for f in FORMS:
        rows[f] = {"connected": int(flags[f].sum()), "share": Z.pct(flags[f].mean())}
    for f in ("F0T", "CN_all"):
        rows[f"{f}_vs_N_sp"] = UL.grade_diff(flags[f], flags["N_sp"], unit, q.size)
    return rows


def grade(F, golds, types, n):
    """The record's measures for the forms F ({form: sorted unique keys})."""
    pairs = n * (n - 1) / 2
    sizes = {f: int(F[f].size) for f in FORMS}
    res = {"pairs": sizes, "ratio_to_N_sp": {f: round(sizes[f] / sizes["N_sp"], 4) if sizes["N_sp"] else None for f in FORMS},
           "density_pct": {f: Z.pct(sizes[f] / pairs) for f in FORMS}, "GP": gp_block(F, golds, n), "GS": gs_block(F, golds, n)}
    gp = res["GP"]
    res["price_pairs_per_linked_slot"] = {f: (round(sizes[f] / gp[f]["linked"], 1) if gp and gp[f]["linked"] else None) for f in FORMS[1:]}
    ty = np.asarray(types)
    res["by_type"] = {t: {"GP": gp_block(F, golds, n, ty == t), "GS": gs_block(F, golds, n, ty == t)} for t in sorted(set(types))}
    gc = np.array([g.size for g in golds])
    res["by_gold_count"] = {str(k): {"GP": gp_block(F, golds, n, gc == k), "GS": gs_block(F, golds, n, gc == k)} for k in GOLD_COUNTS}
    return res


def decide(res):
    """Section 5: CARRIED F0T OFF WIKIPEDIA if F0T's GP minus N_sp's is not BELOW and its pairs are at most BOUND x N_sp's."""
    fails = []
    c = res["GP"]["F0T_vs_N_sp"] if res["GP"] else None
    if c is None:
        fails.append({"condition": "GP", "why": "no GP slot"})
    elif c["call"] == "BELOW":
        fails.append({"condition": "GP", **c})
    r = res["ratio_to_N_sp"]["F0T"]
    if r is None or r > BOUND:
        fails.append({"condition": "size", "ratio_to_N_sp": r})
    return ("CARRIED F0T OFF WIKIPEDIA" if not fails else "NOT_CARRIED"), fails


# ── commands ────────────────────────────────────────────────────────────────


def run(arts, qs, shas, ner=None):
    t0 = time.time()
    n = len(arts)
    ids, golds, types, items = golds_of(arts, qs)
    log(f"G1, G2 pass: {n} articles, {len(ids)} kept queries of {len(qs)}")
    F, st = rule_forms(arts)
    F["N_sp"], n_ents = spacy_form(arts, ner)
    F["none"] = np.zeros(0, np.int64)
    log(f"forms built ({time.time() - t0:.0f}s)")
    res = grade(F, golds, types, n)
    call, fails = decide(res)
    gc = np.array([g.size for g in golds])
    return {"declared_in": DECLARED, "revision": REVISION, "files_sha256": shas, "articles": n, "queries": len(qs),
            "kept_queries": len(ids), "dropped_null": len(qs) - len(ids), "evidence_items": int(sum(items)),
            "gold_counts": {str(k): int((gc == k).sum()) for k in sorted(set(gc.tolist()))},
            "types": {t: types.count(t) for t in sorted(set(types))}, "rule_build": st, "spacy_entities": n_ents,
            **res, "call": call, "fails": fails, "seconds": round(time.time() - t0, 1), "script_sha256": script_sha256()}


def cmd_grade(_a):
    arts, qs, shas = load()
    rec = run(arts, qs, shas)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "grade.json").write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
    log(f"{rec['call']}; written {OUT / 'grade.json'} ({rec['seconds']}s)")
    return 0


def _gp_line(name, b):
    if not b or not b.get("GP"):
        return f"| {name} | - | - | - | - | - | - |"
    g, s = b["GP"], b.get("GS") or {}
    d = g["F0T_vs_N_sp"]
    gs = f"{s['N_sp']['share']} / {s['F0T']['share']}" if s else "-"
    return (f"| {name} | {g['questions']} | {g['slots']} | {g['N_sp']['share']} | {g['F0T']['share']} | "
            f"{d['diff']:+} [{d['lo']:+}, {d['hi']:+}] {d['call']} | {gs} |")


def render(rec):
    L = [f"# D84 report (declared in {rec['declared_in']})", "",
         f"MultiHop-RAG revision {rec['revision'][:12]}; {rec['articles']} articles; {rec['kept_queries']} kept queries of {rec['queries']} "
         f"({rec['dropped_null']} null dropped); gold counts {rec['gold_counts']}.", "",
         f"**Call: {rec['call']}**" + (f"; fails: {rec['fails']}" if rec["fails"] else ""), "",
         "| form | pairs | ratio to N_sp | density % | GP linked | GP share % [95%] | lift | lift call | GS share % | price |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    gp, gs = rec["GP"], rec["GS"]
    for f in FORMS:
        g = gp[f]
        L.append(f"| {f} | {rec['pairs'][f]:,} | {rec['ratio_to_N_sp'][f]} | {rec['density_pct'][f]} | {g['linked']} | "
                 f"{g['share']} [{g['lo']}, {g['hi']}] | {g['lift']} | {g['call']} | {gs[f]['share']} | "
                 f"{rec['price_pairs_per_linked_slot'].get(f, '-')} |")
    L += ["", f"GP: {gp['slots']} slots over {gp['questions']} questions. GS: {gs['questions']} questions.", "",
          "| minus N_sp | GP diff [95%] call | GS diff [95%] call |", "|---|---|---|"]
    for f in ("F0T", "CN_all"):
        a, b = gp[f"{f}_vs_N_sp"], gs[f"{f}_vs_N_sp"]
        L.append(f"| {f} | {a['diff']:+} [{a['lo']:+}, {a['hi']:+}] {a['call']} | {b['diff']:+} [{b['lo']:+}, {b['hi']:+}] {b['call']} |")
    L += ["", "Beside, by question type and gold count (GP share N_sp, F0T; F0T minus N_sp; GS share N_sp / F0T):", "",
          "| group | questions | slots | N_sp GP % | F0T GP % | F0T - N_sp | GS N_sp / F0T % |", "|---|---|---|---|---|---|---|"]
    for t, b in rec["by_type"].items():
        L.append(_gp_line(t, b))
    for k, b in rec["by_gold_count"].items():
        L.append(_gp_line(f"{k} golds", b))
    st = rec["rule_build"]
    L += ["", f"Rule build: {st['keys']:,} name keys; {st['name_entries']:,} name entries; {st['title_run_entries']:,} title-run entries "
          f"added on {st['articles_with_a_title_run']} articles; LT {st['lt_pairs']:,} pairs. spaCy entities: {rec['spacy_entities']:,}.",
          f"Script sha256 {rec['script_sha256'][:12]}; {rec['seconds']}s."]
    return "\n".join(L) + "\n"


def cmd_report(_a):
    rec = json.loads((OUT / "grade.json").read_text(encoding="utf-8"))
    if rec["script_sha256"] != script_sha256():
        raise SystemExit("the grade is not this code's")
    (OUT / "report.md").write_text(render(rec), encoding="utf-8")
    print(render(rec))
    return 0


# ── self-test ───────────────────────────────────────────────────────────────


def selftest():
    import itertools
    # gate G1 refuses other bytes
    with tempfile.TemporaryDirectory() as td:
        p = Path(td)
        (p / "corpus.json").write_text("[]", encoding="utf-8")
        (p / "MultiHopRAG.json").write_text("[]", encoding="utf-8")
        try:
            load(p)
            raise AssertionError("G1 passed other bytes")
        except SystemExit as e:
            assert "gate G1" in str(e)
        ok = {f: ((p / f).stat().st_size, Z.sha_file(p / f)) for f in ("corpus.json", "MultiHopRAG.json")}
        assert load(p, ok)[0] == []
    # a toy corpus: names shared by 2..3 articles, one article's title run adds a name
    body = ["Alice Moore met Bob Stone in Paris. The talks went well.",
            "Bob Stone spoke to Carol King about the deal.",
            "Carol King and Alice Moore signed it in Paris yesterday.",
            "Nothing here is named at all, said the report.",
            "The weather was mild and nobody came.",
            "Zeta Corp hired Dana Fox last week."]
    titles = ["Talks in Paris", "Deal news", "Signing", "Quiet day", "Zeta Corp news", "Hiring at Zeta Corp"]
    arts = [{"title": t, "body": b, "url": f"u{i}"} for i, (t, b) in enumerate(zip(titles, body))]
    n = len(arts)
    F, st = rule_forms(arts)
    # brute force CN_all: names per article by zul1's own name rule, pairs of articles sharing a name of df in [2, 25]
    import zu2r as U
    cap, low = {}, {}
    for a in arts:
        U.count_stat(U.tokens(a["body"]), cap, low)
    usual = lambda k: not (low.get(k, 0) > cap.get(k, 0))  # noqa: E731
    names = [U.names_of(U.tokens(a["body"]), usual) for a in arts]
    want = set()
    allk = set().union(*names)
    for k in allk:
        has = [i for i in range(n) if k in names[i]]
        if 2 <= len(has) <= 25:
            want |= {a * n + b for a, b in itertools.combinations(has, 2)}
    assert set(F["CN_all"].tolist()) == want, (sorted(F["CN_all"].tolist()), sorted(want))
    assert set(F["CN_all"].tolist()) <= set(F["F0T"].tolist())
    assert 4 * n + 5 in set(F["F0T"].tolist()), "the title run Zeta Corp should link articles 4 and 5"
    assert st["articles_with_a_title_run"] >= 1
    # gate G2: urls, titles
    qs = [{"query": "q0", "question_type": "comparison_query", "evidence_list": [{"url": "u0", "title": "Talks in Paris"}, {"url": "u2", "title": "Signing"}]},
          {"query": "q1", "question_type": "null_query", "evidence_list": []},
          {"query": "q2", "question_type": "inference_query", "evidence_list": [{"url": "u0", "title": "Talks  in Paris"}, {"url": "u1", "title": "Deal news"},
                                                                                {"url": "u3", "title": "Quiet day"}]},
          {"query": "q3", "question_type": "temporal_query", "evidence_list": [{"url": "u4", "title": "Zeta Corp news"}, {"url": "u5", "title": "Hiring at Zeta Corp"}]},
          {"query": "q4", "question_type": "temporal_query", "evidence_list": [{"url": "u1", "title": "Deal news"}, {"url": "u1", "title": "Deal news"}]}]
    ids, golds, types, items = golds_of(arts, qs)
    assert ids == [0, 2, 3, 4] and [g.tolist() for g in golds] == [[0, 2], [0, 1, 3], [4, 5], [1]] and items == [2, 3, 2, 2]
    for bad in ({"url": "u9", "title": "x"}, {"url": "u0", "title": "Deal news"}):
        try:
            golds_of(arts, [{"query": "x", "question_type": "t", "evidence_list": [bad]}])
            raise AssertionError("G2 passed a bad item")
        except SystemExit as e:
            assert "gate G2" in str(e)
    try:
        golds_of(arts + [dict(arts[0])], qs)
        raise AssertionError("G2 passed a duplicate url")
    except SystemExit as e:
        assert "corpus urls" in str(e)
    # the whole run with a stub spaCy store (pairs 0-1 and 3-4), against brute force
    stub = lambda texts: (np.array([0, 3], np.int32), np.array([1, 4], np.int32), np.ones(2, np.float32), 7)  # noqa: E731
    with tempfile.TemporaryDirectory() as td:
        rec = run(arts, qs, {"x": "y"}, ner=stub)
    Fb = {"none": set(), "N_sp": {0 * n + 1, 3 * n + 4}, "CN_all": set(F["CN_all"].tolist()), "F0T": set(F["F0T"].tolist())}
    slots = [(j, a * n + b) for j, g in enumerate(golds) for a, b in itertools.combinations(g.tolist(), 2)]
    assert rec["GP"]["slots"] == len(slots) == 1 + 3 + 1
    for f in FORMS:
        hit = [k in Fb[f] for _j, k in slots]
        assert rec["GP"][f]["linked"] == sum(hit), (f, rec["GP"][f]["linked"], hit)
        assert abs(rec["GP"][f]["share"] - Z.pct(sum(hit) / len(slots))) < 1e-9
        # GS brute force: golds connected by Fb[f]
        con = 0
        for g in golds:
            if g.size < 2:
                continue
            e = [(x, y) for x, y in itertools.combinations(range(g.size), 2) if int(g[x]) * n + int(g[y]) in Fb[f]]
            con += connected(g.size, e)
        assert rec["GS"][f]["connected"] == con, (f, rec["GS"][f]["connected"], con)
    assert rec["GS"]["questions"] == 3 and rec["pairs"]["N_sp"] == 2 and rec["spacy_entities"] == 7
    assert set(rec["by_type"]) == {"comparison_query", "inference_query", "temporal_query"}
    assert rec["by_type"]["temporal_query"]["GP"]["slots"] == 1 and rec["by_gold_count"]["3"]["GP"]["slots"] == 3
    # the lift call
    c = np.ones(50)
    assert share_call(c, np.arange(50), 50, 0.5)["call"] == "POINTS"
    assert share_call(np.zeros(50), np.arange(50), 50, 0.5)["call"] == "BELOW_BASE"
    assert share_call(np.r_[np.ones(25), np.zeros(25)], np.arange(50), 50, 0.5)["call"] == "NO_LIFT"
    # the rule
    base = {"GP": {"F0T_vs_N_sp": {"call": "LEVEL"}}, "ratio_to_N_sp": {"F0T": 1.2}}
    assert decide(base)[0] == "CARRIED F0T OFF WIKIPEDIA"
    assert decide({**base, "ratio_to_N_sp": {"F0T": 1.31}})[1][0]["condition"] == "size"
    assert decide({**base, "GP": {"F0T_vs_N_sp": {"call": "BELOW", "diff": -1}}})[0] == "NOT_CARRIED"
    assert render(rec).startswith("# D84 report")
    print("zd84 self-test: OK")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest").set_defaults(fn=lambda _a: selftest())
    sub.add_parser("grade").set_defaults(fn=cmd_grade)
    sub.add_parser("report").set_defaults(fn=cmd_report)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
