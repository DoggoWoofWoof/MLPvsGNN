"""SWP Part D (docs/SWP_S_TO_R_SWAP.md section 5b): the models of record read on an oracle multi-hop graph, MuSiQue's B1 setting. A diagnostic ceiling, never a method.

B1L's machinery (zb1l.py, which is zb1h rebound; neither is edited) on two new carves whose structural family is the rule's W2_S pairs (b1w's folder) plus the oracle pairs
of oraclepairs.py (the oracle entries are untyped, as b1o's):

    b1c   W2_S pairs + ORC (a question's dependency tree between its supporting paragraphs; 1,090 pairs)
    b1q   W2_S pairs + ORA (every two supporting paragraphs of a question; 1,750 pairs)

The carve is chosen by --variant c|q (or B1G_VARIANT, which the chain's step processes inherit).

    python outputs/mp_unified/zb1g.py --variant c bench --dataset musique        # the laptop: the setting's folder
    python outputs/mp_unified/zb1g.py --variant c settings [--host]              # zb1h's: the CSR, its entry types, the checks
    python outputs/mp_unified/zb1g.py --variant c chain --dataset musique --host # PKO's chain on the carve
    python outputs/mp_unified/zb1g.py --variant c gate --tag musique --host
    python outputs/mp_unified/zb1g.py --variant c fork | lists | cols | read-gnn | read-mlp ...    # zb1h's, on B1G's roots
    python outputs/mp_unified/zb1g.py report                                     # the laptop: b1c / b1q against b1t, b1w (paired by question), B1L's b1o beside
    python outputs/mp_unified/zb1g.py selftest
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


def _variant():
    """--variant c|q from the command line (read before anything else, so the constants below are the variant's) or B1G_VARIANT."""
    a = sys.argv[1:]
    v = a[a.index("--variant") + 1] if "--variant" in a and a.index("--variant") + 1 < len(a) else os.environ.get("B1G_VARIANT", "c")
    if v not in ("c", "q"):
        raise SystemExit("B1G: --variant is c or q")
    os.environ["B1G_VARIANT"] = v                              # the chain's step processes inherit it
    return v


VARIANT = _variant()
if "--variant" in sys.argv:                                    # zb1h's header reads sys.argv[1:2] for its BLAS threads: the command goes first
    _i = sys.argv.index("--variant")
    del sys.argv[_i:_i + 2]

import zb1l as L  # noqa: E402  (imports zb1h, whose header sets BLAS from sys.argv[1:2] before numpy loads)

import argparse  # noqa: E402
import contextlib  # noqa: E402
import copy  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

H = L.H
ROOT = H.ROOT
DECLARED = "docs/SWP_S_TO_R_SWAP.md"
STAGE = "B1G"
KIND = {"c": "ORC", "q": "ORA"}
CARVE = f"b1{VARIANT}"                                        # b1c / b1q
OUT = ROOT / "outputs" / CARVE
LTM_OUT = OUT / "ltm"
B1W = ROOT / "outputs" / "bench" / "hipporag2_w"              # b1w's folder: B1a's files with the rule's W2_S structural family
BENCH_G = OUT / "bench" / f"hipporag2_{VARIANT}"
SETTINGS = ("musique",)
TAGS = {"musique": ("musique",)}
CARVES = ("b1t", "b1w", CARVE)
GRAPH_TEXT = (f"B1G {CARVE}: B1a's NER and KNN families, and the rule's W2_S pairs plus the oracle {KIND[VARIANT]} pairs of the gold chains as the "
              "structural family (unweighted); a ceiling, the pairs use the labels")
NO_STEP1 = {"state": f"not applicable: B1's settings have no step-1 cache (B1G, {DECLARED})"}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] zb1g: {msg}", flush=True)


def read_roots():
    d = f"outputs/{CARVE}"
    return {"pools": f"{d}/pools", "plook": f"{d}/look", "cache": f"{d}/cache", "rmatch": f"{d}/chains/rmatch", "zrc": f"{d}/chains/zrc",
            "links": f"{d}/links", "kt": f"{d}/kt"}


def bind():
    """zb1h's module globals set to B1G's (its functions look them up at call time); safe to call again. zb1l's on_b1o (the variant's lists patch) is carve-free."""
    for k, v in (("DECLARED", DECLARED), ("STAGE", STAGE), ("OUT", OUT), ("LTM_OUT", LTM_OUT), ("B1A", BENCH_G), ("CARVE", CARVE),
                 ("CARVES", CARVES), ("SETTINGS", SETTINGS), ("TAGS", TAGS), ("GRAPH_TEXT", GRAPH_TEXT), ("NO_STEP1", NO_STEP1),
                 ("read_roots", read_roots), ("on_b1h", L.on_b1o)):
        setattr(H, k, v)


# ── the setting's folder (bench) ──────────────────────────────────────────────

def union_pairs(w_src, w_dst, oracle):
    """(a, b) a < b of the rule's directed edges and the oracle pairs, distinct, sorted; and the counts."""
    w = {(int(a), int(b)) for a, b in zip(w_src, w_dst) if a < b}
    o = {(int(a), int(b)) for a, b in oracle}
    assert all(a < b for a, b in o), "oracle pairs are (a < b)"
    both = sorted(w | o)
    return (np.asarray([a for a, _ in both], np.int64), np.asarray([b for _, b in both], np.int64),
            {"w2s_pairs": len(w), "oracle_pairs": len(o), "oracle_in_w2s": len(o & w), "oracle_new": len(o - w), "union_pairs": len(both)})


def variant_build(build, struct_sha, edges, links, script_sha, utc):
    """B1W's build.json for the variant: the structural sha replaced; the u1a block names B1a's structural sha (the one B1a's lists were built for,
    which look_b1u's lists check reads), taken from b1w's own u1a block."""
    b = json.loads(json.dumps(build))
    b["files"]["graph_structural.npz"] = struct_sha
    b["u1a"] = {"declared_in": DECLARED, "variant": VARIANT, "carve": CARVE,
                "structural": f"the rule's W2_S pairs plus the oracle {KIND[VARIANT]} pairs (each pair once each way, unweighted)",
                "b1a_structural_sha256": build["u1a"]["b1a_structural_sha256"], "edges": int(edges), "script_sha256": script_sha, "utc": utc}
    b["b1g"] = links
    return b


def bench(name):
    import u1a
    import oraclepairs as OP
    t0 = time.time()
    if name != "musique":
        raise SystemExit("B1G: the oracle pairs exist for MuSiQue only")
    src_dir, out = B1W / name, BENCH_G / name
    build = H.read_json(src_dir / "build.json")
    bad = [f for f in u1a.FILES if f != u1a.STRUCT and u1a.sha_file(src_dir / f) != build["files"].get(f)]
    if bad or u1a.sha_file(src_dir / u1a.STRUCT) != build["files"][u1a.STRUCT]:
        raise SystemExit(f"B1G {name}: {bad or [u1a.STRUCT]} differ from b1w's build.json")
    raw, corpus, nodes, queries = OP.load(name)
    pairs, _rows, meta = OP.build(raw, corpus, nodes, queries)
    n = len(H.read_json(src_dir / "nodes.json"))
    if n != len(nodes) or n != int(build["passages"]):
        raise SystemExit(f"B1G {name}: b1w's folder holds {n} passages, B1a's {len(nodes)}")
    z = np.load(src_dir / u1a.STRUCT)
    a, b, cnt = union_pairs(z["src"], z["dst"], pairs[KIND[VARIANT]])
    if cnt["oracle_pairs"] != meta["pairs"][KIND[VARIANT]]:
        raise SystemExit(f"B1G {name}: {cnt['oracle_pairs']} oracle pairs, the builder counted {meta['pairs']}")
    src, dst = L.both_directions(a, b, n)
    out.mkdir(parents=True, exist_ok=True)
    for f in u1a.FILES:
        if f != u1a.STRUCT:
            shutil.copyfile(src_dir / f, out / f)
            if u1a.sha_file(out / f) != build["files"][f]:
                raise SystemExit(f"B1G {out / f}: the copy is not b1w's")
    u1a.save_npz(out / u1a.STRUCT, src=src, dst=dst, rel=np.zeros(src.size, np.int16))
    links = {"kind": KIND[VARIANT], "oracle": meta, **cnt, "directed_edges_file": int(src.size), "oraclepairs_sha256": H.sha_src(OP.__file__),
             "weights_entered": False}
    u1a.write_json(out / "build.json", variant_build(build, u1a.sha_file(out / u1a.STRUCT), src.size, links, u1a.sha_src(__file__), u1a.utc()))
    log(f"bench {name} {CARVE}: {cnt}; {src.size} directed edges; {time.time() - t0:.0f} s")
    return 0


# ── zb1h's steps with B1G's names ─────────────────────────────────────────────

def settings(host=False, only=None):
    """zb1h.settings on the variant's folder; the record's label names the variant and the oracle kind."""
    bind()
    orig = H.write_json

    def labelled(p, obj):
        if isinstance(obj, dict) and "desc_b1" in obj:
            obj["desc_b1"]["variant"] = VARIANT
            obj["desc_b1"]["oracle"] = KIND[VARIANT]
            obj["b1g"] = {"script_sha256": H.sha_src(__file__), "zb1h_sha256": H.sha_src(H.__file__)}
        return orig(p, obj)

    H.write_json = labelled
    try:
        return H.settings(host, only)
    finally:
        H.write_json = orig


def chain(name, host=False, root=None):
    """zb1h.chain, whose steps are run as this file's (zb1h spawns its own __file__); B1G_VARIANT is inherited."""
    bind()
    saved = H.__file__
    H.__file__ = str(Path(__file__).resolve())
    try:
        return H.chain(name, host, root)
    finally:
        H.__file__ = saved


def gate(tag, host=False):
    bind()
    ZB = H.zb()
    M = ZB.mods(host)
    R = copy.copy(H.roots_of(M))
    R.gate = H.gate_path(tag)
    carves = [(ds, CARVE) for ds in TAGS[tag]]
    with ZB.checks(M):
        verdict = M.PK.gate(R, carves=carves)
    H.write_json(OUT / f"b1g_gate_{tag}.json", {
        "declared_in": DECLARED, "stage": STAGE, "tag": tag, "pko_gate": H.rel(R.gate), "pko_gate_sha256": H.sha_file(R.gate),
        "verdict": verdict, "carves": carves, "settings": {ds: H.sha_file(H.settings_paths(ds)[1]) for ds in TAGS[tag]},
        "zb1h_sha256": H.sha_src(H.__file__), "zb1g_sha256": H.sha_src(__file__), "utc": H.utc()})
    return 0


# ── the report (the laptop): b1c and b1q against b1t and b1w, paired by question; B1L's b1o beside as a reference ──

GNN, MLP = L.GNN, L.MLP
MODELS = (GNN, MLP)
STEP = ("UP", "DOWN", "FLAT")                                  # by the paired interval: above 0, below 0, neither
CELL = {"t": "b1t", "w": "b1w", "c": "b1c", "q": "b1q"}
PAIRS_G = (("b1c-b1w", "c", "w"), ("b1q-b1w", "q", "w"), ("b1q-b1c", "q", "c"), ("b1c-b1t", "c", "t"), ("b1q-b1t", "q", "t"), ("b1w-b1t", "w", "t"))
HOPS = ("2hop", "3hop", "4hop")
REPORT_OUT = ROOT / "outputs" / "b1g"
GRAPH_KIND = {"c": "R + ORC (the chain's dependency edges)", "q": "R + ORA (every two supporting paragraphs)"}


def consts_of(v):
    out = ROOT / "outputs" / f"b1{v}"
    return {"CARVE": f"b1{v}", "OUT": out, "LTM_OUT": out / "ltm", "B1A": out / "bench" / f"hipporag2_{v}", "CARVES": ("b1t", "b1w", f"b1{v}")}


@contextlib.contextmanager
def as_variant(v):
    """zb1h's module globals set to variant v's carve and folders (read paths, the carve list) and put back."""
    keep = {k: getattr(H, k) for k in consts_of(v)}
    try:
        for k, x in consts_of(v).items():
            setattr(H, k, x)
        yield
    finally:
        for k, x in keep.items():
            setattr(H, k, x)


def hop_of(kind):
    return f"{str(kind)[0]}hop"


def pairs_g(means, which, hop_mask=None):
    """paired['b1c-b1w M'] = {R@5, ci} for every pair of PAIRS_G; calls[M] = {pair: UP | DOWN | FLAT}; over the questions in hop_mask if given."""
    paired, calls = {}, {}
    for arm in MODELS:
        d = {k: means.get((arm, c, which)) for k, c in CELL.items()}
        if any(x is None for x in d.values()):
            continue
        if hop_mask is not None:
            d = {k: x[hop_mask] for k, x in d.items()}
        for key, a_, b_ in PAIRS_G:
            paired[f"{key} {arm}"] = H.ci_of(d[a_] - d[b_])
        calls[arm] = {key: L.word(paired[f"{key} {arm}"]["ci"], *STEP) for key, _a, _b in PAIRS_G}
    return paired, calls


def agree(words):
    """The one word the models share, MIXED where they differ, INCOMPLETE where a model is missing."""
    words = list(words)
    if len(words) != len(MODELS) or not all(words):
        return "INCOMPLETE"
    return words[0] if len(set(words)) == 1 else "MIXED"


def report():
    import b1b as BB
    rec = {"declared_in": DECLARED, "stage": STAGE, "candidate": BB.CAND, "boot": {"resamples": BB.BOOT, "seed": BB.SEED},
           "in_domain": {k: list(x) for k, x in H.IN_DOMAIN.items()}, "missing": [], "kinds": dict(GRAPH_KIND)}
    ds = "musique"
    qs = H.read_json(ROOT / "outputs" / "bench" / "hipporag2" / ds / "queries.json")
    ids, gc = [str(q["id"]) for q in qs], [len(q["golds"]) for q in qs]
    hops = np.asarray([hop_of(q["kind"]) for q in qs])
    means, blocks = {}, {}
    for v in "cq":
        if not (consts_of(v)["B1A"] / ds / "build.json").exists():
            raise SystemExit(f"B1G report: {consts_of(v)['B1A'] / ds} is not built")
        with as_variant(v):
            perq, shas = H.read_set(ds, ids, gc, rec["missing"])
            blk, mean = H.reads_block(ds, perq, shas)
        blocks[v] = blk
        for k, x in mean.items():
            if k[1] in ("b1t", "b1w") and k in means:
                assert np.array_equal(means[k], x), f"B1G: {k} differs between the variants' reads of the shared carves"
            means[k] = x
    s = {"questions": len(ids), "hops": {h: int((hops == h).sum()) for h in HOPS}, "graph": dict(GRAPH_KIND),
         "in_domain": {}, "zero_shot": {}, "fits": {v: blocks[v]["fits"] for v in "cq"}}
    for which, pk, ck in (("in_domain", "paired", "calls"), ("zero_shot", "paired_zero_shot", "calls_zero_shot")):
        for v in "cq":
            for k, x in blocks[v][which].items():
                s[which][k] = x
        s[pk], s[ck] = pairs_g(means, which)
    s["by_hops"] = {}
    for h in HOPS:
        m = hops == h
        pr, cl = pairs_g(means, "in_domain", m)
        s["by_hops"][h] = {"questions": int(m.sum()), "paired": pr, "calls": cl}
    s["call"] = {key: agree([s["calls"].get(a, {}).get(key) for a in MODELS]) for key, _a, _b in PAIRS_G}
    b1l = ROOT / "outputs" / "b1l" / "report.json"
    if b1l.exists():                                            # B1L's b1o rows (H4o's LLM-built edges), filed, as a labelled reference
        ref = H.read_json(b1l)["settings"][ds]
        s["reference_b1o"] = {"source": H.rel(b1l), "sha256": H.sha_file(b1l), "note": "H4o's LLM-built P pairs; a read-only reference, not a cell of this stage",
                              "in_domain": {a: ref["in_domain"].get(f"{a}/b1o") for a in MODELS},
                              "zero_shot": {a: ref["zero_shot"].get(f"{a}/b1o") for a in MODELS}}
    s["links"] = {v: H.read_json(consts_of(v)["B1A"] / ds / "build.json")["b1g"] for v in "cq"}
    rec["settings"] = {ds: s}
    rec["zb1g_sha256"] = H.sha_src(__file__)
    rec["zb1h_sha256"] = H.sha_src(H.__file__)
    rec["utc"] = H.utc()
    H.write_json(REPORT_OUT / "report.json", rec)
    H.write_json(REPORT_OUT / "calls.json", {"declared_in": DECLARED, "call": s["call"], "calls": s["calls"], "calls_zero_shot": s["calls_zero_shot"],
                                             "by_hops": {h: x["calls"] for h, x in s["by_hops"].items()}})
    write_md(rec)
    log(f"report: {H.rel(REPORT_OUT / 'report.md')}; missing {len(rec['missing'])}; call {s['call']}")
    return 0


def write_md(rec):
    def lv(c):
        return f"{100 * c['R@5']:.1f} ({100 * c['R5_ci'][0]:.1f}-{100 * c['R5_ci'][1]:.1f})" if c else "--"

    def df(c):
        return f"{100 * c['R@5']:+.1f} ({100 * c['ci'][0]:+.1f}, {100 * c['ci'][1]:+.1f})" if c else "--"

    s = rec["settings"]["musique"]
    T = [f"# B1G report (declared in {DECLARED}, section 5b)", "",
         "R@5 x100 on HippoRAG 2's MuSiQue setting (1,000 questions), no new fit. The models of record (zg6r, zrckpn; fitted without LLM links, on W2_S pools) are read on "
         "b1t (the text links only), b1w (the rule's W2_S links) and two carves whose structural family is **the rule's pairs plus oracle pairs drawn between the supporting "
         f"paragraphs of each question** (b1c: {GRAPH_KIND['c']}; b1q: {GRAPH_KIND['q']}). The oracle uses the labels: it is a ceiling for what a better multi-hop graph could give, "
         "never a method and never an estimate of a graph anyone can build. In-domain = the mean over the fits that trained on MuSiQue's partner split; "
         "95% bootstrap intervals over the questions (2,000 resamples, seed 0); calls by the interval (UP / DOWN / FLAT), no threshold.", ""]
    for which, title, key in (("in_domain", "In-domain", "paired"), ("zero_shot", "Held-out (fits that did not train on MuSiQue)", "paired_zero_shot")):
        T += [f"## {title}", "", "| model | b1t | b1w | b1c | b1q | b1c - b1w | b1q - b1w | b1q - b1c | b1w - b1t |", "|---|---:|---:|---:|---:|---|---|---|---|"]
        for arm in MODELS:
            ld = {c: s[which].get(f"{arm}/{c}") for c in ("b1t", "b1w", "b1c", "b1q")}
            if not any(ld.values()):
                continue
            pr = {k: s[key].get(f"{k} {arm}") for k, _a, _b in PAIRS_G}
            T.append(f"| {arm} | {lv(ld['b1t'])} | {lv(ld['b1w'])} | {lv(ld['b1c'])} | {lv(ld['b1q'])} | {df(pr['b1c-b1w'])} | {df(pr['b1q-b1w'])} | "
                     f"{df(pr['b1q-b1c'])} | {df(pr['b1w-b1t'])} |")
        T.append("")
    ref = s.get("reference_b1o")
    if ref:
        T += ["## Reference: B1L's b1o (H4o's LLM-built P pairs), the filed rows, not a cell of this stage", "", "| model | in-domain b1o | held-out b1o |", "|---|---:|---:|"]
        for arm in MODELS:
            T.append(f"| {arm} | {lv(ref['in_domain'].get(arm))} | {lv(ref['zero_shot'].get(arm))} |")
        T += ["", f"Source `{ref['source']}` (sha256 {ref['sha256'][:12]}). The comparison with b1c / b1q is across stages: same fits, same questions, different link sets.", ""]
    T += ["## By hop depth (in-domain, paired by question)", "", "| hops | questions | model | b1c - b1w | b1q - b1w | b1q - b1c | calls (c-w / q-w / q-c) |", "|---|---:|---|---|---|---|---|"]
    for h in HOPS:
        x = s["by_hops"][h]
        for arm in MODELS:
            pr, cl = x["paired"], x["calls"].get(arm)
            if not cl:
                continue
            T.append(f"| {h} | {x['questions']} | {arm} | {df(pr[f'b1c-b1w {arm}'])} | {df(pr[f'b1q-b1w {arm}'])} | {df(pr[f'b1q-b1c {arm}'])} | "
                     f"{cl['b1c-b1w']} / {cl['b1q-b1w']} / {cl['b1q-b1c']} |")
    T += ["", "## The call (in-domain, both models)", "", "| pair | call |", "|---|---|"]
    for key, _a, _b in PAIRS_G:
        T.append(f"| {key} | {s['call'][key]} |")
    T += ["", "## The oracle pairs", "", "| carve | oracle pairs | already in W2_S | new | union pairs | directed edges in the file |", "|---|---:|---:|---:|---:|---:|"]
    for v in "cq":
        k = s["links"][v]
        T.append(f"| b1{v} | {k['oracle_pairs']} | {k['oracle_in_w2s']} | {k['oracle_new']} | {k['union_pairs']} | {k['directed_edges_file']} |")
    T += ["", f"Missing reads: {len(rec['missing'])}."]
    REPORT_OUT.mkdir(parents=True, exist_ok=True)
    (REPORT_OUT / "report.md").write_text("\n".join(T) + "\n", encoding="utf-8")


# ── self-test ─────────────────────────────────────────────────────────────────

def selftest():
    bind()
    fit = "scr-ml1-zg6r-s0-hp"
    assert CARVE == f"b1{VARIANT}" and OUT == ROOT / "outputs" / CARVE and BENCH_G == OUT / "bench" / f"hipporag2_{VARIANT}"
    assert H.read_path(L.GNN, fit, CARVE, "musique") == OUT / "fits/h" / fit / f"reads/musique__{CARVE}.npz", "the carve's read path"
    assert H.read_path(L.GNN, fit, "b1t", "musique") == H.B1P / "fits/t" / fit / "reads/musique__b1t.npz", "b1t is B1P's"
    assert H.read_flags("musique") == ["--read", f"musique={CARVE}", "--out-root", f"outputs/{CARVE}/fits/h"], H.read_flags("musique")
    assert H.gate_path("musique") == OUT / "gate_musique.json" and H.settings_paths("musique")[0] == OUT / "settings" / "musique.npz"
    assert H.B1A == BENCH_G and H.CARVES == CARVES and H.CARVE == CARVE and CARVES[-1] == CARVE
    assert H.on_b1h is L.on_b1o and H.read_roots is read_roots and all(v.startswith(f"outputs/{CARVE}/") for v in read_roots().values())
    assert set(read_roots()) == {"pools", "plook", "cache", "rmatch", "zrc", "links", "kt"}
    # the union: the rule's directed edges (both ways) and the oracle pairs, once each way, none dropped, none invented
    ws, wd = np.asarray([0, 1, 1, 2, 4, 3], np.int32), np.asarray([1, 0, 2, 1, 3, 4], np.int32)            # 0-1, 1-2, 3-4
    a, b, cnt = union_pairs(ws, wd, [(0, 1), (2, 5), (3, 5)])
    assert list(zip(a.tolist(), b.tolist())) == [(0, 1), (1, 2), (2, 5), (3, 4), (3, 5)], (a, b)
    assert cnt == {"w2s_pairs": 3, "oracle_pairs": 3, "oracle_in_w2s": 1, "oracle_new": 2, "union_pairs": 5}, cnt
    s, d = L.both_directions(a, b, 6)
    assert s.size == 10 and {(int(i), int(j)) for i, j in zip(s, d)} == {(i, j) for i, j in zip(a.tolist(), b.tolist())} | \
        {(j, i) for i, j in zip(a.tolist(), b.tolist())}, "both ways"
    refused = False
    try:
        union_pairs(ws, wd, [(5, 2)])
    except AssertionError:
        refused = True
    assert refused, "an oracle pair must be (a < b)"
    # the variant's build.json: b1w's, the structural sha replaced, B1a's sha kept from b1w's own u1a block
    build = {"files": {"graph_structural.npz": "ww", "nodes.json": "bb"}, "passages": 5, "u1a": {"b1a_structural_sha256": "aa"}}
    vb = variant_build(build, "cc", 8, {"kind": KIND[VARIANT]}, "dd", "t")
    assert vb["files"] == {"graph_structural.npz": "cc", "nodes.json": "bb"} and build["files"]["graph_structural.npz"] == "ww"
    assert vb["u1a"]["b1a_structural_sha256"] == "aa" and vb["u1a"]["variant"] == VARIANT and vb["u1a"]["carve"] == CARVE
    assert vb["b1g"] == {"kind": KIND[VARIANT]} and vb["passages"] == 5
    # the settings record's label
    seen_ = []
    orig, hold = H.write_json, H.settings
    try:
        H.write_json = lambda p, obj: seen_.append(obj)
        H.settings = lambda host=False, only=None: H.write_json(Path("x.json"), {"dataset": "musique", "desc_b1": {"variant": "h"}}) or 0
        assert settings(False, "musique") == 0
    finally:
        H.settings, H.write_json = hold, orig
    assert seen_[0]["desc_b1"]["variant"] == VARIANT and seen_[0]["desc_b1"]["oracle"] == KIND[VARIANT] and "b1g" in seen_[0], "label"
    assert H.write_json is orig and H.settings is hold, "put back"
    assert os.environ["B1G_VARIANT"] == VARIANT, "the chain's step processes inherit the variant"
    # the report: as_variant sets zb1h's globals to the other variant's and puts them back; the pairs, the calls, the hop split and the markdown on fake reads
    here = (H.CARVE, H.OUT, H.CARVES)
    other = "q" if VARIANT == "c" else "c"
    with as_variant(other):
        assert H.CARVE == f"b1{other}" and H.OUT == ROOT / "outputs" / f"b1{other}" and H.CARVES == ("b1t", "b1w", f"b1{other}")
        assert H.read_path(L.GNN, "scr-ml1-zg6r-s0-hp", f"b1{other}", "musique") == ROOT / "outputs" / f"b1{other}" / "fits/h" / "scr-ml1-zg6r-s0-hp" / f"reads/musique__b1{other}.npz"
    assert (H.CARVE, H.OUT, H.CARVES) == here, "put back"
    rs = np.random.RandomState(7)
    nq = 90
    base = rs.rand(nq) * 0.5
    fake_means = {}
    for arm in MODELS:
        for which in ("in_domain", "zero_shot"):
            fake_means[(arm, "b1t", which)] = base
            fake_means[(arm, "b1w", which)] = base + 0.02 + rs.randn(nq) * 0.03
            fake_means[(arm, "b1c", which)] = fake_means[(arm, "b1w", which)] + 0.10 + rs.randn(nq) * 0.03
            fake_means[(arm, "b1q", which)] = fake_means[(arm, "b1c", which)] + 0.05 + rs.randn(nq) * 0.03
    pr, cl = pairs_g(fake_means, "in_domain")
    assert set(pr) == {f"{k} {a}" for a in MODELS for k, _x, _y in PAIRS_G} and set(cl) == set(MODELS)
    for a in MODELS:
        assert cl[a]["b1c-b1w"] == "UP" and cl[a]["b1q-b1c"] == "UP" and cl[a]["b1c-b1t"] == "UP", cl[a]
        assert abs(pr[f"b1q-b1t {a}"]["R@5"] - pr[f"b1q-b1c {a}"]["R@5"] - pr[f"b1c-b1w {a}"]["R@5"] - pr[f"b1w-b1t {a}"]["R@5"]) < 4e-4, "ladder"
    kinds = ["2hop"] * 50 + ["3hop1"] * 25 + ["4hop2"] * 15
    hops = np.asarray([hop_of(k) for k in kinds])
    assert [int((hops == h).sum()) for h in HOPS] == [50, 25, 15], "hop groups"
    pr2, cl2 = pairs_g(fake_means, "in_domain", hops == "4hop")
    assert set(cl2) == set(MODELS) and pr2[f"b1c-b1w {GNN}"]["R@5"] != pr[f"b1c-b1w {GNN}"]["R@5"], "over the 4-hop questions only"
    assert pairs_g({k: v for k, v in fake_means.items() if k[1] != "b1q"}, "in_domain") == ({}, {}), "no b1q, no pairs"
    assert agree(["UP", "UP"]) == "UP" and agree(["UP", "FLAT"]) == "MIXED" and agree(["UP", None]) == "INCOMPLETE" and agree([]) == "INCOMPLETE"
    cell = {"R@5": 0.5, "R5_ci": [0.4, 0.6]}
    rec = {"missing": [], "settings": {"musique": {
        "in_domain": {f"{a}/{c}": cell for a in MODELS for c in ("b1t", "b1w", "b1c", "b1q")}, "zero_shot": {},
        "paired": pr, "paired_zero_shot": {}, "by_hops": {h: {"questions": 5, "paired": pr, "calls": cl} for h in HOPS},
        "call": {k: "UP" for k, _x, _y in PAIRS_G},
        "reference_b1o": {"source": "outputs/b1l/report.json", "sha256": "0" * 64, "in_domain": {a: cell for a in MODELS}, "zero_shot": {a: None for a in MODELS}},
        "links": {v: {"oracle_pairs": 1, "oracle_in_w2s": 0, "oracle_new": 1, "union_pairs": 2, "directed_edges_file": 4} for v in "cq"}}}}
    import tempfile
    global REPORT_OUT
    keep_out = REPORT_OUT
    try:
        REPORT_OUT = Path(tempfile.mkdtemp())
        write_md(rec)
        md = (REPORT_OUT / "report.md").read_text(encoding="utf-8")
    finally:
        REPORT_OUT = keep_out
    assert "| zg6r | 50.0 (40.0-60.0)" in md and "## By hop depth" in md and "## Reference: B1L's b1o" in md and "ceiling" in md and "no threshold" in md, md[:400]
    print(f"zb1g self-test: OK ({CARVE})")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?")
    ap.add_argument("kind", nargs="?")
    ap.add_argument("--dataset", choices=SETTINGS)
    ap.add_argument("--split", choices=H.SPLITS)
    ap.add_argument("--tag", choices=tuple(TAGS))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--index")
    ap.add_argument("--name")
    ap.add_argument("--root")
    ap.add_argument("--questions", type=int, default=8)
    ap.add_argument("--host", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "selftest":
        return selftest()
    need = {"bench": ("dataset",), "pools": ("dataset",), "chain": ("dataset",), "step": ("dataset",), "lists": ("dataset",),
            "cols": ("split", "dataset"), "gate": ("tag",), "read-gnn": ("name", "tag"), "read-mlp": ("name", "tag"),
            "look-dev": ("dataset", "root")}
    for k in need.get(a.cmd, ()):
        if getattr(a, k) is None:
            raise SystemExit(f"{a.cmd}: --{k} is required")
    bind()
    if a.root and Path(a.root).resolve() == OUT.resolve():
        a.root = None
    if a.cmd == "bench":
        return bench(a.dataset)
    if a.cmd == "settings":
        return settings(a.host, a.dataset)
    if a.cmd == "pools":
        return H.pools(a.dataset, a.limit, a.host, a.root)
    if a.cmd == "chain":
        return chain(a.dataset, a.host, a.root)
    if a.cmd == "step":
        return H.run_step(a.kind, a.dataset, H.parse_index(a.index), a.host, a.root)
    if a.cmd == "gate":
        return gate(a.tag, a.host)
    if a.cmd == "fork":
        return H.fork()
    if a.cmd == "lists":
        return H.lists(a.dataset)
    if a.cmd == "cols":
        return H.cols(a.split, a.dataset, a.host)
    if a.cmd == "read-gnn":
        return H.read_gnn(a.name, a.tag)
    if a.cmd == "read-mlp":
        return H.read_mlp(a.name, a.tag)
    if a.cmd == "report":
        return report()
    if a.cmd == "look-dev":
        return H.look_dev(a.dataset, a.root, a.questions)
    raise SystemExit(__doc__)


if __name__ == "__main__":
    sys.exit(main())
