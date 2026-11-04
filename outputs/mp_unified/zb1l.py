"""B1L (docs/B1L_MODELS_OF_RECORD_ON_THE_LLM_GRAPH.md): the two models of record, zg6r (ML1's GNN) and zrckpn (LTM's MLP), fitted
without LLM links, read on b1o: B1a's text families (NER, KNN) plus, as the structural family, the P pairs of an LLM-built
reference graph matched to the setting's passages in P2c part 1 (musique: HippoRAG 2's OpenIE with gpt-4o-mini, H4o; 2wiki:
SiReRAG's propositions, SR), next to B1P's and B1M's reads of the same twelve fits on b1t (the text links only) and b1w (the
rule's W2_S links). Nothing trains. This module is zb1h's, rebound: every function of zb1h that B1H runs is run here on B1L's
roots, carve and bench folder (zb1h.py is never edited), and only what differs is written here: the folder (bench), the
lists patch (on_b1o), the settings record's label, the gate's file name, the report.

    python outputs/mp_unified/zb1l.py --selftest
    python outputs/mp_unified/zb1l.py bench --dataset D                  # the laptop: the setting's folder, structural = the reference's P pairs
    python outputs/mp_unified/zb1l.py settings [--dataset D] [--host]    # zb1h's: the CSR, its entry types, the three checks
    python outputs/mp_unified/zb1l.py pools|chain|step|gate|fork|lists|cols|read-gnn|read-mlp|look-dev ...    # zb1h's, on B1L's roots
    python outputs/mp_unified/zb1l.py report [--dataset D]                # the laptop: the ladder b1t, b1w, b1o, paired by question

bench     per setting: B1a's six files byte for byte except the structural family, which holds the P pairs of the reference
          (phrases in 2 to 25 passages, each pair once each way, no weights: the setting's structural graph is unweighted by
          contract; the 1/df weights are saved beside), and a build.json that names B1a's structural sha, as B1f's variants do.
          The reference is re-read from the released file (read only) and must equal P2c's stored arrays and filed counts.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import zb1h as H  # noqa: E402   its header sets BLAS from sys.argv[1:2] before numpy loads

import argparse  # noqa: E402
import contextlib  # noqa: E402
import copy  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import time  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import numpy as np  # noqa: E402

ROOT = H.ROOT
DECLARED = "docs/B1L_MODELS_OF_RECORD_ON_THE_LLM_GRAPH.md"
STAGE = "B1L"
OUT = ROOT / "outputs" / "b1l"
LTM_OUT = OUT / "ltm"
B1A_REAL = ROOT / "outputs" / "bench" / "hipporag2"       # B1a's own folder: the source of every file but the structural family
BENCH_O = OUT / "bench" / "hipporag2_o"                   # b1o's folder
P2C = ROOT / "outputs" / "p2c"
SETTINGS = ("musique", "2wiki")                           # block 1, block 2
TAGS = {"musique": ("musique",), "2wiki": ("2wiki",)}
REF = {"musique": "H4o", "2wiki": "SR"}
CARVE = "b1o"
CARVES = ("b1t", "b1w", "b1o")                            # the ladder
GNN, MLP = H.GNN, H.MLP
MODELS = (GNN, MLP)
STEP = ("UP", "DOWN", "FLAT")                             # b1o - b1t and b1w - b1t by their interval: above 0, below 0, neither
RULE_VS_LLM = ("GREATER", "LESS", "SAME")                 # b1w - b1o by its interval
GRAPH_TEXT = ("B1L b1o: B1a's NER and KNN families, and the P pairs of the setting's LLM-built reference graph as the "
              "structural family (unweighted)")
NO_STEP1 = {"state": f"not applicable: B1's settings have no step-1 cache (B1L, {DECLARED})"}
REF_KEYS = ("sha256", "bytes", "items", "passages", "phrases", "pairs", "mentions")   # the filed counts of the reference's read
PAIRS = (("b1o-b1t", "o", "t"), ("b1w-b1t", "w", "t"), ("b1w-b1o", "w", "o"))

_ON_B1H = H.on_b1h


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] zb1l: {msg}", flush=True)


def read_roots():
    return {"pools": "outputs/b1l/pools", "plook": "outputs/b1l/look", "cache": "outputs/b1l/cache",
            "rmatch": "outputs/b1l/chains/rmatch", "zrc": "outputs/b1l/chains/zrc", "links": "outputs/b1l/links",
            "kt": "outputs/b1l/kt"}


@contextlib.contextmanager
def on_b1o(M, name, stats, look=False):
    """zb1h.on_b1h with the variant's lists: b1o's folder holds no lists of its own, so look_b1's load_lists is look_b1u's (B1a's
    lists, checked against this folder's other files and the structural sha it replaced), as zb1p.on_b1 sets it for b1w and b1t."""
    P_ = H.zb().Patches()
    try:
        P_.set(M.LB, load_lists=M.LBU.load_lists)
        with _ON_B1H(M, name, stats, look) as B:
            yield B
    finally:
        P_.restore()


def bind():
    """zb1h's module globals set to B1L's (its functions look them up at call time); safe to call again."""
    for k, v in (("DECLARED", DECLARED), ("STAGE", STAGE), ("OUT", OUT), ("LTM_OUT", LTM_OUT), ("B1A", BENCH_O),
                 ("CARVE", CARVE), ("CARVES", CARVES), ("SETTINGS", SETTINGS), ("TAGS", TAGS), ("GRAPH_TEXT", GRAPH_TEXT),
                 ("NO_STEP1", NO_STEP1), ("read_roots", read_roots), ("on_b1h", on_b1o)):
        setattr(H, k, v)


# ── the setting's folder (bench) ──────────────────────────────────────────────

def both_directions(a, b, n):
    """(src, dst) int32 of the pairs (a, b), each once each way, sorted and distinct (b1f.cut's layout); n positions."""
    s = np.concatenate([np.asarray(a, np.int64), np.asarray(b, np.int64)])
    d = np.concatenate([np.asarray(b, np.int64), np.asarray(a, np.int64)])
    k = np.unique(s * (n + 1) + d)
    return (k // (n + 1)).astype(np.int32), (k % (n + 1)).astype(np.int32)


def quantiles(x):
    return {str(q): round(float(np.quantile(x, q)), 4) for q in (0.05, 0.25, 0.5, 0.75, 0.95, 0.99)} if len(x) else {}


def variant_build(build, struct_sha, edges, links, script_sha, utc):
    """B1a's build.json for the variant folder: the structural file's sha replaced and the u1a block that look_b1u's lists check
    reads (the structural sha this variant replaced); everything else B1a's."""
    b = json.loads(json.dumps(build))
    b["files"]["graph_structural.npz"] = struct_sha
    b["u1a"] = {"declared_in": DECLARED, "variant": "o", "carve": CARVE,
                "structural": "the P pairs of the setting's LLM-built reference (each pair once each way, unweighted)",
                "b1a_structural_sha256": build["files"]["graph_structural.npz"], "edges": int(edges), "script_sha256": script_sha,
                "utc": utc}
    b["b1l"] = links
    return b


def reference_pairs(name):
    """(sets, a, b, w, meta, graph): the reference's phrase sets over B1's passages, re-read from the released file (read only)
    exactly as P2c part 1 did and required to equal its stored ones (phrases, entities and relation pairs as strings per passage:
    the vocabulary's ids follow the set iteration order of each process) and its filed counts, and its P pairs (a < b, weights
    sum 1/df)."""
    import p2c_triples as P2
    ref = REF[name]
    with np.load(P2C / f"ref_{name}_{ref}.npz") as z:
        smeta = json.loads(str(z["meta"]))
        g = P2.graph_from_ref(ref, z)
    saved = P2.REFS[name]
    P2.REFS[name] = {ref: saved[ref]}
    try:
        arrays, meta = P2.refs_stage(name, write=False)[ref]
    finally:
        P2.REFS[name] = saved
    g2 = P2.graph_from_ref(ref, arrays)

    def ents(x):
        return [{x.words[i] for i in e} for e in x.ents]

    if not (g2.phrase_sets() == g.phrase_sets() and g2.pair_sets() == g.pair_sets() and ents(g2) == ents(g)):
        raise SystemExit(f"B1L {name}: the reference re-read from the released file is not P2c's stored one")
    if {k: meta.get(k) for k in REF_KEYS} != {k: smeta.get(k) for k in REF_KEYS}:
        raise SystemExit(f"B1L {name}: the reference's counts are not P2c's stored ones")
    sets = g.phrase_sets()
    a, b, w = P2.p_edges(sets)
    filed = H.read_json(P2C / f"b1_{name}.json")["agreement"][ref]["NER"]["P_edges"]["reference"]
    if int(a.size) != int(filed):
        raise SystemExit(f"B1L {name}: {a.size} P pairs, P2c filed {filed}")
    return sets, a, b, w, {k: smeta[k] for k in smeta if k in REF_KEYS + ("file", "reference", "kind")}, g


def bench(name):
    import u1a
    t0 = time.time()
    src_dir, out = B1A_REAL / name, BENCH_O / name
    build = H.read_json(src_dir / "build.json")
    bad = [f for f in u1a.FILES if u1a.sha_file(src_dir / f) != build["files"].get(f)]
    if bad:
        raise SystemExit(f"B1L {name}: {bad} differ from B1a's build.json")
    sets, a, b, w, meta, g = reference_pairs(name)
    nodes = H.read_json(src_dir / "nodes.json")
    n = len(nodes)
    if len(sets) != n or n != int(build["passages"]):
        raise SystemExit(f"B1L {name}: the reference holds {len(sets)} passages, B1's setting {n}")
    src, dst = both_directions(a, b, n)
    out.mkdir(parents=True, exist_ok=True)
    for f in u1a.FILES:
        if f != u1a.STRUCT:
            shutil.copyfile(src_dir / f, out / f)
            if u1a.sha_file(out / f) != build["files"][f]:
                raise SystemExit(f"B1L {out / f}: the copy is not B1a's")
    u1a.save_npz(out / u1a.STRUCT, src=src, dst=dst, rel=np.zeros(src.size, np.int16))
    # the pair set against the rule's, in package rows (the setting's own entries), and the weights beside
    rows = np.asarray([x["package_row"] for x in nodes], np.int64)
    wj = H.read_json(ROOT / "outputs" / "b1p" / "settings" / "w" / f"{name}.json")
    N = int(wj["n_nodes"])
    ip_h, col_h, pairs = H.hyperlink_csr(N, rows[a], rows[b])
    with np.load(ROOT / "outputs" / "b1p" / "settings" / "w" / f"{name}.npz") as zw:
        ip_w, col_w = zw["ip"].astype(np.int64), zw["col"].astype(np.int32)
    in_w = H.lookup_entries(N, ip_h, col_h, ip_w, col_w) >= 0
    w_in_p = H.lookup_entries(N, ip_w, col_w, ip_h, col_h) >= 0
    deg = np.diff(ip_h)[np.diff(ip_h) > 0]
    df = g.df
    links = {"reference": REF[name], "source": meta, "df_window": [2, 25], "weights_entered": False,
             "pairs_over_positions": int(a.size), "pairs": int(pairs), "entries": int(col_h.size), "directed_edges_file": int(src.size),
             "nodes_with_an_edge": int(deg.size), "degree": {"mean": round(float(deg.mean()), 2), "median": float(np.median(deg)),
                                                              "p95": float(np.quantile(deg, 0.95)), "max": int(deg.max())},
             "phrases": int(len(g.words)), "phrases_df_1": int((df == 1).sum()), "phrases_df_2_25": int(((df >= 2) & (df <= 25)).sum()),
             "phrases_df_over_25": int((df > 25).sum()), "weight_quantiles": quantiles(w), "weight_max": round(float(w.max()), 4),
             "w2s_pairs_in_setting": int(col_w.size // 2), "p_entries_in_w2s": int(in_w.sum()), "p_pairs_in_w2s": int(in_w.sum()) // 2,
             "p_entries_not_in_w2s": int((~in_w).sum()), "p_pairs_not_in_w2s": int((~in_w).sum()) // 2,
             "w2s_entries_in_p": int(w_in_p.sum()), "w2s_entries_not_in_p": int((~w_in_p).sum())}
    np.savez_compressed(OUT / "bench" / f"p_weights_{name}.npz", a=a.astype(np.int32), b=b.astype(np.int32), w=w)
    bj = variant_build(build, u1a.sha_file(out / u1a.STRUCT), src.size, links, u1a.sha_src(__file__), u1a.utc())
    u1a.write_json(out / "build.json", bj)
    log(f"bench {name}: {a.size} P pairs ({pairs} over package rows, {col_h.size} entries; {int(in_w.sum())} entries are W2_S entries, "
        f"{int((~in_w).sum())} are not); {time.time() - t0:.0f} s")
    return 0


# ── zb1h's steps with B1L's names ────────────────────────────────────────────

def settings(host=False, only=None):
    """zb1h.settings on b1o's folder; the record's label says variant o and names the reference."""
    bind()
    orig = H.write_json

    def labelled(p, obj):
        if isinstance(obj, dict) and "desc_b1" in obj:
            obj["desc_b1"]["variant"] = "o"
            obj["desc_b1"]["reference"] = REF[obj["dataset"]]
            obj["b1l"] = {"script_sha256": H.sha_src(__file__), "zb1h_sha256": H.sha_src(H.__file__)}
        return orig(p, obj)

    H.write_json = labelled
    try:
        return H.settings(host, only)
    finally:
        H.write_json = orig


def chain(name, host=False, root=None):
    """zb1h.chain, whose steps are run as this file's (zb1h spawns its own __file__)."""
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
    H.write_json(OUT / f"b1l_gate_{tag}.json", {
        "declared_in": DECLARED, "stage": STAGE, "tag": tag, "pko_gate": H.rel(R.gate), "pko_gate_sha256": H.sha_file(R.gate),
        "verdict": verdict, "carves": carves, "settings": {ds: H.sha_file(H.settings_paths(ds)[1]) for ds in TAGS[tag]},
        "zb1h_sha256": H.sha_src(H.__file__), "zb1l_sha256": H.sha_src(__file__), "utc": H.utc()})
    return 0


# ── the report (the laptop): the ladder b1t, b1w, b1o, paired by question ────

def word(ci, up, down, flat):
    return up if ci[0] > 0 else down if ci[1] < 0 else flat


def resamples(q):
    """b1b.boot_ci's own resamples: BOOT index vectors of q questions from default_rng(SEED), in its order."""
    import b1b as BB
    rng = np.random.default_rng(BB.SEED)
    return np.stack([rng.integers(0, q, q) for _ in range(BB.BOOT)])


def share_of(t, w, o, idx):
    """(b1w - b1t) / (b1o - b1t), defined only where b1o - b1t is above 0 by its interval; its interval over the resamples whose
    denominator is above 0 (their count beside), P2c's share convention."""
    d_o, d_w = o - t, w - t
    ci_o = H.ci_of(d_o)
    if not ci_o["ci"][0] > 0:
        return {"share": None, "reason": "b1o - b1t is not above 0 by its interval", "b1o-b1t": ci_o}
    num, den = d_w[idx].mean(1), d_o[idx].mean(1)
    ok = den > 0
    r = num[ok] / den[ok]
    return {"share": round(float(d_w.mean() / d_o.mean()), 3),
            "ci": [round(float(np.quantile(r, 0.025)), 3), round(float(np.quantile(r, 0.975)), 3)],
            "resamples": int(ok.sum()), "of": int(idx.shape[0]), "b1o-b1t": ci_o}


def pairs_block(means, which, idx):
    """paired['b1o-b1t M'], ['b1w-b1t M'], ['b1w-b1o M'] = {R@5, ci}; calls[M] = the three words; share[M] = share_of."""
    paired, calls, share = {}, {}, {}
    for arm in MODELS:
        t, w, o = (means.get((arm, c, which)) for c in CARVES)
        if t is None or w is None or o is None:
            continue
        d = {"o": o, "w": w, "t": t}
        for key, a_, b_ in PAIRS:
            paired[f"{key} {arm}"] = H.ci_of(d[a_] - d[b_])
        ci = {key: paired[f"{key} {arm}"]["ci"] for key, _a, _b in PAIRS}
        calls[arm] = {"b1o-b1t": word(ci["b1o-b1t"], *STEP), "b1w-b1t": word(ci["b1w-b1t"], *STEP),
                      "b1w-b1o": word(ci["b1w-b1o"], *RULE_VS_LLM)}
        share[arm] = share_of(t, w, o, idx)
    return paired, calls, share


def stage_call(words):
    """The setting's call from the two models' in-domain b1w - b1o words: GREATER, SAME or LESS where both agree; NOT_LESS where
    neither is LESS; MIXED otherwise."""
    words = list(words)
    if len(words) != len(MODELS) or not all(words):
        return "INCOMPLETE"
    if len(set(words)) == 1:
        return words[0]
    return "NOT_LESS" if "LESS" not in words else "MIXED"


def report(only=None):
    bind()
    import b1b as BB
    import b1f as B1F
    rec = {"declared_in": DECLARED, "stage": STAGE, "candidate": BB.CAND, "boot": {"resamples": BB.BOOT, "seed": BB.SEED},
           "in_domain": {k: list(x) for k, x in H.IN_DOMAIN.items()}, "settings": {}, "missing": []}
    b1m = H.read_json(ROOT / "outputs" / "b1m" / "report.json")
    rec["inputs_sha256"] = {x: H.sha_file(ROOT / "outputs" / x / "report.json") for x in ("b1p", "b1m")}
    checks = {"equal_to_b1m": {"checked": 0, "bad": []}}
    for ds in ((only,) if only else SETTINGS):
        if not H.settings_paths(ds)[1].exists():
            continue
        qs = H.read_json(B1A_REAL / ds / "queries.json")
        ids, gc = [str(q["id"]) for q in qs], [len(q["golds"]) for q in qs]
        perq, shas = H.read_set(ds, ids, gc, rec["missing"])
        s = {"questions": len(ids), "graph": GRAPH_TEXT, "reference": REF[ds]}
        reads, means = H.reads_block(ds, perq, shas)
        s.update(reads)
        n, bad = H.equal_to_b1m(ds, s, b1m)
        checks["equal_to_b1m"]["checked"] += n
        checks["equal_to_b1m"]["bad"] += [f"{ds} {k}" for k in bad]
        s["best_published"] = {}
        for which, peer_only in (("any", False), ("peer_reviewed", True)):
            val, row = B1F.best_published(ds, peer_only)
            s["best_published"][which] = {"row": row, "R@5": val}
        s["calls"], s["gap_points"] = H.calls_block(s, s["best_published"])
        idx = resamples(len(ids))
        s["paired"], c_in, s["share_of_llm_step"] = pairs_block(means, "in_domain", idx)
        s["paired_zero_shot"], c_zs, s["share_of_llm_step_zero_shot"] = pairs_block(means, "zero_shot", idx)
        s["calls_vs_llm_graph"] = {"in_domain": c_in, "zero_shot": c_zs}
        s["call"] = stage_call([c_in.get(a, {}).get("b1w-b1o") for a in MODELS])
        sj = H.read_json(H.settings_paths(ds)[1])
        s["types"] = {"edges": sj["edges"], "em": sj["types"]["em"], "score": sj["types"]["score"], "checks": sj["checks"],
                      "content_sha256": sj["content_sha256"]}
        s["links"] = H.read_json(BENCH_O / ds / "build.json")["b1l"]
        rec["settings"][ds] = s
        log(f"{ds}: done (call {s['call']})")
    if checks["equal_to_b1m"]["bad"]:
        raise SystemExit(f"B1L: rows recomputed from B1P's and B1M's reads differ from B1M's filed rows: "
                         f"{checks['equal_to_b1m']['bad']}")
    rec["checks"] = checks
    rec["zb1l_sha256"] = H.sha_src(__file__)
    rec["zb1h_sha256"] = H.sha_src(H.__file__)
    rec["utc"] = H.utc()
    H.write_json(OUT / "report.json", rec)
    H.write_json(OUT / "calls.json", {
        "declared_in": DECLARED,
        "call": {ds: s["call"] for ds, s in rec["settings"].items()},
        "calls_vs_llm_graph": {ds: s["calls_vs_llm_graph"] for ds, s in rec["settings"].items()},
        "share_of_llm_step": {ds: {"in_domain": s["share_of_llm_step"], "zero_shot": s["share_of_llm_step_zero_shot"]}
                              for ds, s in rec["settings"].items()},
        "gnn_minus_mlp": {ds: {k: c["label"] for k, c in s["gnn_minus_mlp"].items()} for ds, s in rec["settings"].items()}})
    write_md(rec)
    log(f"report: {H.rel(OUT / 'report.md')}; missing {len(rec['missing'])}; equal to B1M's rows: {checks['equal_to_b1m']['checked']} checked")
    return 0


def write_md(rec):
    def lv(c):
        return f"{100 * c['R@5']:.1f} ({100 * c['R5_ci'][0]:.1f}-{100 * c['R5_ci'][1]:.1f})" if c else "--"

    def df(c):
        return f"{100 * c['R@5']:+.1f} ({100 * c['ci'][0]:+.1f}, {100 * c['ci'][1]:+.1f})" if c else "--"

    def sh(c):
        if not c:
            return "--"
        if c["share"] is None:
            return "n/a: b1o - b1t not above 0"
        return f"{c['share']:.2f} ({c['ci'][0]:.2f}-{c['ci'][1]:.2f})"

    L = [f"# B1L report (declared in {DECLARED})", "",
         f"R@5 x100 of the {rec['candidate']} candidate on HippoRAG 2's settings (1,000 questions each), no new fit. The two models of "
         "record, zg6r and zrckpn, were fitted without LLM links (on W2_S pools) and are read on three link sets: b1t (the text "
         "links only), b1w (the rule's W2_S links, W2_S-typed) and b1o (the text links plus the P pairs of the setting's LLM-built "
         "reference graph: passages sharing a phrase that occurs in 2 to 25 passages, each pair once each way; the 1/df weights do "
         "not enter, the structural family being unweighted by contract, and their distribution is recorded beside). In-domain = "
         "the mean over the fits that trained on the dataset; 95% bootstrap intervals over the questions (2,000 resamples, seed 0).",
         "", "**What the calls say.** b1o's entries are NULL-heavy (most have no W2_S type) while b1w's are W2_S-typed and the fits "
         "were fitted on W2_S types, so b1o - b1t is a lower bound on what these models can take from an LLM-built graph, and "
         "b1w - b1o mixes the pair set with the typing. The calls compare how the models of record, fitted without LLM links, read "
         "on the rule's links and on the LLM-built pair set; they are not a statement that the rule's links beat the LLM's.", ""]
    for which, title in (("in_domain", "In-domain"), ("zero_shot", "Held-out (fits that did not train on the dataset)")):
        L += [f"## {title}", "",
              "| setting | model | b1t | b1w | b1o | b1o - b1t | b1w - b1t | b1w - b1o | calls (o-t / w-t / w-o) | "
              "share of the LLM step the rule's step recovers |", "|---|---|---:|---:|---:|---|---|---|---|---|"]
        for ds, s in rec["settings"].items():
            key, ckey, skey = ((("paired", "in_domain", "share_of_llm_step")) if which == "in_domain"
                               else ("paired_zero_shot", "zero_shot", "share_of_llm_step_zero_shot"))
            for arm in MODELS:
                ld = {c: s[which].get(f"{arm}/{c}") for c in CARVES}
                if not any(ld.values()):
                    continue
                pr = {k: s[key].get(f"{k} {arm}") for k, _a, _b in PAIRS}
                cl = s["calls_vs_llm_graph"][ckey].get(arm)
                L.append(f"| {ds} | {arm} | {lv(ld['b1t'])} | {lv(ld['b1w'])} | {lv(ld['b1o'])} | {df(pr['b1o-b1t'])} | "
                         f"{df(pr['b1w-b1t'])} | {df(pr['b1w-b1o'])} | "
                         + (f"{cl['b1o-b1t']} / {cl['b1w-b1t']} / {cl['b1w-b1o']}" if cl else "--") + f" | {sh(s[skey].get(arm))} |")
        L.append("")
    L += ["## The call per setting (in-domain b1w - b1o, both models)", "", "| setting | call | zg6r | zrckpn |", "|---|---|---|---|"]
    for ds, s in rec["settings"].items():
        c = s["calls_vs_llm_graph"]["in_domain"]
        L.append(f"| {ds} | {s['call']} | {c.get(GNN, {}).get('b1w-b1o', '--')} | {c.get(MLP, {}).get('b1w-b1o', '--')} |")
    L += ["", "## zg6r - zrckpn on each carve, paired by question", "",
          "| setting | read | carve | zg6r - zrckpn (CI) | label |", "|---|---|---|---|---|"]
    for ds, s in rec["settings"].items():
        for k, c in s["gnn_minus_mlp"].items():
            L.append(f"| {ds} | {k.split('/')[0].replace('_', '-')} | {k.split('/')[1]} | {df(c)} | {c['label']} |")
    L += ["", "## The link sets, by entry type", "",
          "| setting | reference | P pairs | entries | entries that are W2_S entries | entries with the null score type | "
          "W2_S pairs in the setting | W2_S entries that are P entries |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for ds, s in rec["settings"].items():
        t, k = s["types"], s["links"]
        L.append(f"| {ds} | {s['reference']} | {t['edges']['pairs']} | {t['edges']['entries']} | {t['score']['entries_w2s']} "
                 f"({100 * t['score']['entries_w2s_share']:.1f}%) | {t['score']['entries_null']} "
                 f"({100 * t['score']['entries_null_share']:.1f}%) | {t['edges']['w2s_pairs_in_setting']} | {k['w2s_entries_in_p']} |")
    L += ["", "Best published rows: " + "; ".join(
        f"{ds} {s['best_published']['any']['row']} {s['best_published']['any']['R@5']} / "
        f"{s['best_published']['peer_reviewed']['row']} {s['best_published']['peer_reviewed']['R@5']}"
        for ds, s in rec["settings"].items()), ""]
    c = rec["checks"]["equal_to_b1m"]
    L += [f"Check: the b1t and b1w rows recomputed from B1P's and B1M's read files equal B1M's filed rows "
          f"({c['checked']} checked, {len(c['bad'])} differ). Missing reads: {len(rec['missing'])}."]
    (OUT / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")


# ── self-test ─────────────────────────────────────────────────────────────────

def selftest():
    import p2c_triples as P2
    bind()
    rs = np.random.RandomState(5)
    # 1. the rebinding: zb1h's paths, flags, carve and roots are B1L's; b1t and b1w stay B1P's and B1M's files
    fit = "scr-ml1-zg6r-s0-hp"
    assert H.read_path(GNN, fit, "b1o", "musique") == OUT / "fits/h" / fit / "reads/musique__b1o.npz", "b1o's read path"
    assert H.read_path(GNN, fit, "b1t", "musique") == H.B1P / "fits/t" / fit / "reads/musique__b1t.npz", "b1t is B1P's"
    assert H.read_path(MLP, "scr-ltm-zrckpn-s1", "b1w", "2wiki") == H.B1M / "fits/w/scr-ltm-zrckpn-s1/reads/2wiki__b1w.npz"
    assert H.read_flags("musique") == ["--read", "musique=b1o", "--out-root", "outputs/b1l/fits/h"], H.read_flags("musique")
    assert H.read_flags("2wiki")[1] == "2wiki=b1o" and H.gate_path("musique") == OUT / "gate_musique.json"
    assert H.settings_paths("musique")[0] == OUT / "settings" / "musique.npz" and H.B1A == BENCH_O and H.CARVES == CARVES
    assert H.on_b1h is on_b1o and H.read_roots is read_roots and set(read_roots()) == {"pools", "plook", "cache", "rmatch", "zrc",
                                                                                       "links", "kt"}
    assert all(v.startswith("outputs/b1l/") for v in read_roots().values()), "roots"
    assert set(sum(TAGS.values(), ())) == set(SETTINGS) == set(REF) and CARVES[-1] == CARVE, "tags"
    # 2. the P pairs: a phrase in 2 to 25 passages joins every pair of them, weight sum 1/df; df 1 and df 26 join nothing
    sets = [{"x", "y", "solo"}, {"x", "y"}, {"x", "z"}, {"z"}] + [{"big"} for _ in range(26)]
    a, b, w = P2.p_edges(sets)
    got = {(int(i), int(j)): float(x) for i, j, x in zip(a, b, w)}
    assert set(got) == {(0, 1), (0, 2), (1, 2), (2, 3)}, got
    assert abs(got[(0, 1)] - (1 / 3 + 1 / 2)) < 1e-12 and abs(got[(0, 2)] - 1 / 3) < 1e-12 and abs(got[(2, 3)] - 0.5) < 1e-12, got
    s, d = both_directions(a, b, len(sets))
    assert s.dtype == d.dtype == np.int32 and s.size == 8 and (np.diff(s.astype(np.int64) * 100 + d) > 0).all(), "both ways, sorted"
    assert {(int(i), int(j)) for i, j in zip(s, d)} == set(got) | {(j, i) for i, j in got}, "the edges"
    # 3. the variant's build.json: B1a's, with the structural sha replaced and the sha it replaced named (look_b1u's lists check)
    build = {"files": {"graph_structural.npz": "aa", "nodes.json": "bb"}, "passages": 5, "graph": "x"}
    vb = variant_build(build, "cc", 8, {"pairs": 4}, "dd", "t")
    assert vb["files"] == {"graph_structural.npz": "cc", "nodes.json": "bb"} and build["files"]["graph_structural.npz"] == "aa"
    assert vb["u1a"]["b1a_structural_sha256"] == "aa" and vb["u1a"]["variant"] == "o" and vb["u1a"]["carve"] == CARVE
    assert vb["passages"] == 5 and vb["b1l"] == {"pairs": 4}, "the rest is B1a's"
    # 4. the lists patch is on while zb1h's on_b1h runs and off after, also on a failure
    class LB_:
        load_lists = "orig"

    M_ = SimpleNamespace(LB=LB_, LBU=SimpleNamespace(load_lists="variant"))
    seen = []

    @contextlib.contextmanager
    def fake(M, name, stats, look=False):
        seen.append((M.LB.load_lists, name, look))
        yield "B"

    saved_on = globals()["_ON_B1H"]
    globals()["_ON_B1H"] = fake
    try:
        with on_b1o(M_, "musique", {}, True) as B_:
            assert B_ == "B" and LB_.load_lists == "variant", "patched inside"
        assert LB_.load_lists == "orig" and seen == [("variant", "musique", True)], "restored"
        try:
            with on_b1o(M_, "musique", {}):
                raise RuntimeError("x")
        except RuntimeError:
            pass
        assert LB_.load_lists == "orig", "restored after a failure"
    finally:
        globals()["_ON_B1H"] = saved_on
    # 5. the resamples are b1b.boot_ci's own; the calls and the share by their intervals
    import b1b as BB
    q = 300
    idx = resamples(q)
    x = rs.randn(q) * 0.2 + 0.05
    m = x[idx].mean(1)
    assert [round(float(np.quantile(m, 0.025)), 4), round(float(np.quantile(m, 0.975)), 4)] == BB.boot_ci(x), "boot_ci's resamples"
    assert word([0.01, 0.1], *STEP) == "UP" and word([-0.1, -0.01], *STEP) == "DOWN" and word([-0.1, 0.1], *STEP) == "FLAT"
    assert (word([0.01, 0.1], *RULE_VS_LLM), word([-0.1, -0.01], *RULE_VS_LLM), word([0, 0.1], *RULE_VS_LLM)) == \
        ("GREATER", "LESS", "SAME"), "the rule's words; an interval touching 0 is SAME"
    t = rs.rand(q) * 0.5
    o = t + 0.2 + rs.randn(q) * 0.05
    w_ = t + 0.1 + rs.randn(q) * 0.05
    sh_ = share_of(t, w_, o, idx)
    assert sh_["share"] == round(float((w_ - t).mean() / (o - t).mean()), 3) and sh_["resamples"] == sh_["of"] == BB.BOOT
    assert sh_["ci"][0] <= sh_["share"] <= sh_["ci"][1], "the share's interval"
    flat = share_of(t, w_, t.copy(), idx)
    assert flat["share"] is None and "not above 0" in flat["reason"], "no share where b1o - b1t is not above 0"
    assert share_of(t, w_, t - 0.2, idx)["share"] is None, "none where b1o is below b1t"
    # 6. the paired block's layout and the ladder's arithmetic, with zb1h's own reads_block on fake reads
    nq = 80

    def fake_r(level):
        r5 = rs.binomial(2, level, nq) / 2.0
        return {"R@5": r5, "FC@5": (r5 == 1).astype(np.float64), "hit@1": (rs.rand(nq) < level).astype(np.float64)}

    pq, sh = {}, {}
    for carve, lv in zip(CARVES, (0.5, 0.7, 0.6)):
        for arm in MODELS:
            for ft in H.FITS[arm]:
                pq[(arm, carve, ft)], sh[(arm, carve, ft)] = fake_r(lv), f"{arm}{carve}{ft}"
    blk, mean = H.reads_block("musique", pq, sh)
    assert set(blk["in_domain"]) == {f"{a}/{c}" for a in MODELS for c in CARVES} and len(blk["fits"]) == 36, "reads_block on b1o"
    assert all(v["fits"] == 3 for v in blk["in_domain"].values()) and all(v["fits"] == 3 for v in blk["zero_shot"].values())
    paired, calls, share = pairs_block(mean, "in_domain", resamples(nq))
    assert set(paired) == {f"{k} {a}" for a in MODELS for k, _x, _y in PAIRS} and set(calls) == set(share) == set(MODELS), "keys"
    for a in MODELS:
        assert abs(paired[f"b1o-b1t {a}"]["R@5"] - paired[f"b1w-b1t {a}"]["R@5"] + paired[f"b1w-b1o {a}"]["R@5"]) < 3e-4, "ladder"
        assert set(calls[a]) == {"b1o-b1t", "b1w-b1t", "b1w-b1o"} and calls[a]["b1w-b1o"] in RULE_VS_LLM
    assert pairs_block({k: v for k, v in mean.items() if k[1] != "b1o"}, "in_domain", resamples(nq)) == ({}, {}, {}), "no b1o, no pairs"
    assert [stage_call(x) for x in (["GREATER"] * 2, ["SAME"] * 2, ["LESS"] * 2, ["GREATER", "SAME"], ["LESS", "SAME"],
                                    ["GREATER", None], [])] == ["GREATER", "SAME", "LESS", "NOT_LESS", "MIXED", "INCOMPLETE", "INCOMPLETE"]
    # 7. the settings record's label, with zb1h's write_json stood in for
    seen_ = []
    orig, hold = H.write_json, H.settings
    try:
        H.write_json = lambda p, obj: seen_.append(obj)
        H.settings = lambda host=False, only=None: H.write_json(Path("x.json"), {"dataset": "musique", "desc_b1": {"variant": "h"}}) or 0
        assert settings(False, "musique") == 0
    finally:
        H.settings, H.write_json = hold, orig
    assert seen_[0]["desc_b1"]["variant"] == "o" and seen_[0]["desc_b1"]["reference"] == "H4o" and "b1l" in seen_[0], "label"
    assert H.write_json is orig and H.settings is hold, "put back"
    print("zb1l self-test: OK")
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
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest or a.cmd == "selftest":
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
        return report(a.dataset)
    if a.cmd == "look-dev":
        return H.look_dev(a.dataset, a.root, a.questions)
    raise SystemExit(__doc__)


if __name__ == "__main__":
    sys.exit(main())
