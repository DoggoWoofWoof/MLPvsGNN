"""B1L-U (docs/B1L_MODELS_OF_RECORD_ON_THE_LLM_GRAPH.md section 15): the union carve b1u. B1L's reads of the two models of record on b1o
(B1a's text families plus the P pairs of the setting's LLM-built reference) say what an LLM-built graph gives them; this module reads the
same twelve fits on b1u, whose structural family is the UNION of the rule's W2_S pairs in the setting (b1w's) and the reference's P pairs
(b1o's), to say what each adds beyond the other. Nothing trains. zb1l.py (and through it zb1h.py) is reused unchanged, rebound to the
union's roots, carve and bench folder; only what differs is written here: the folder (bench), the label of the settings record, the chain's
step file, the report.

    python outputs/mp_unified/zb1lu.py --selftest
    python outputs/mp_unified/zb1lu.py bench --dataset D                  # the laptop: the union's folder from b1o's and b1w's
    python outputs/mp_unified/zb1lu.py settings [--dataset D] [--host]    # zb1h's: the CSR, its entry types (rule entries W2_S-typed, the rest NULL)
    python outputs/mp_unified/zb1lu.py pools|chain|step|gate|fork|lists|cols|read-gnn|read-mlp|look-dev ...    # zb1l's, on the union's roots
    python outputs/mp_unified/zb1lu.py report [--dataset D]               # the laptop: b1t, b1w, b1o, b1u, paired by question

Outputs only under outputs/b1l/u/. b1o's reads are B1L's (outputs/b1l/fits/h), b1t's and b1w's are B1P's and B1M's; all read only.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import zb1l as L  # noqa: E402   it imports zb1h first: its header sets BLAS from sys.argv[1:2] before numpy loads

import argparse  # noqa: E402
import shutil  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

H = L.H
ROOT = H.ROOT
DECLARED = L.DECLARED
STAGE = "B1L-U"
B1L_OUT = L.OUT                                           # outputs/b1l: b1o's reads and settings
OUT = B1L_OUT / "u"
LTM_OUT = OUT / "ltm"
BENCH_O = L.BENCH_O                                       # b1o's folder (B1L's bench)
BENCH_W = ROOT / "outputs" / "bench" / "hipporag2_w"      # B1f's W2_S cut: b1w's folder, read only
BENCH_U = OUT / "bench" / "hipporag2_u"
CARVE = "b1u"
CARVES = ("b1t", "b1w", "b1u")
LADDER = ("b1t", "b1w", "b1o", "b1u")
GRAPH_TEXT = ("B1L-U b1u: B1a's NER and KNN families and, as the structural family, the union of the rule's W2_S pairs in the "
              "setting and the P pairs of the setting's LLM-built reference (unweighted)")
STEP = L.STEP                                             # UP, DOWN, FLAT by the interval
PAIRS = (("b1u-b1w", "b1u", "b1w"), ("b1u-b1o", "b1u", "b1o"), ("b1u-b1t", "b1u", "b1t"))
CALL = {"UP": "ADDS", "FLAT": "NOTHING", "DOWN": "HURTS"}  # the setting's call from the two models' in-domain words
_L0 = {k: getattr(L, k) for k in ("STAGE", "OUT", "LTM_OUT", "BENCH_O", "CARVE", "CARVES", "GRAPH_TEXT", "read_roots")}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] zb1lu: {msg}", flush=True)


def read_roots():
    return {k: v.replace("outputs/b1l/", "outputs/b1l/u/", 1) for k, v in _L0["read_roots"]().items()}


def bind():
    """zb1l's module globals set to the union's, then zb1l's own bind (which sets zb1h's); safe to call again."""
    for k, v in (("STAGE", STAGE), ("OUT", OUT), ("LTM_OUT", LTM_OUT), ("BENCH_O", BENCH_U), ("CARVE", CARVE), ("CARVES", CARVES),
                 ("GRAPH_TEXT", GRAPH_TEXT), ("read_roots", read_roots)):
        setattr(L, k, v)
    L.bind()


def unbind():
    for k, v in _L0.items():
        setattr(L, k, v)
    L.bind()


def read_path(arm, fit, carve, ds):
    """b1o's reads are B1L's, b1u's are this stage's (zb1h's CARVE), b1t's and b1w's are B1P's and B1M's."""
    if carve == "b1o":
        return B1L_OUT / "fits" / "h" / fit / "reads" / f"{ds}__b1o.npz"
    return H.read_path(arm, fit, carve, ds)


# ── the union's folder (bench) ────────────────────────────────────────────────

def union_directed(p, q, n):
    """(src, dst) int32 of the directed edges of p and q (each a (src, dst) pair of arrays over n positions), distinct and sorted."""
    k = np.unique(np.concatenate([np.asarray(p[0], np.int64) * (n + 1) + p[1], np.asarray(q[0], np.int64) * (n + 1) + q[1]]))
    return (k // (n + 1)).astype(np.int32), (k % (n + 1)).astype(np.int32)


def variant_build(build, struct_sha, edges, links, script_sha, utc):
    bj = L.variant_build(build, struct_sha, edges, links, script_sha, utc)
    bj["u1a"]["variant"] = "u"
    bj["u1a"]["structural"] = "the union of the rule's W2_S pairs in the setting and the P pairs of the setting's LLM-built reference"
    return bj


def bench(name):
    import u1a
    bind()
    t0 = time.time()
    src_dir, out = L.B1A_REAL / name, BENCH_U / name
    build = H.read_json(src_dir / "build.json")
    nodes = H.read_json(src_dir / "nodes.json")
    n = len(nodes)
    parts = {}
    for tag, folder in (("o", BENCH_O), ("w", BENCH_W)):
        fb = H.read_json(folder / name / "build.json")
        if u1a.sha_file(folder / name / u1a.STRUCT) != fb["files"][u1a.STRUCT]:
            raise SystemExit(f"B1L-U {name}: {folder / name / u1a.STRUCT} is not the file its build.json names")
        bad = [f for f in u1a.FILES if f != u1a.STRUCT and u1a.sha_file(folder / name / f) != build["files"].get(f)]
        if bad:
            raise SystemExit(f"B1L-U {name}: {bad} of {folder.name} are not B1a's")
        with np.load(folder / name / u1a.STRUCT) as z:
            parts[tag] = (z["src"].astype(np.int64), z["dst"].astype(np.int64))
    src, dst = union_directed(parts["o"], parts["w"], n)
    out.mkdir(parents=True, exist_ok=True)
    for f in u1a.FILES:
        if f != u1a.STRUCT:
            shutil.copyfile(src_dir / f, out / f)
            if u1a.sha_file(out / f) != build["files"][f]:
                raise SystemExit(f"B1L-U {out / f}: the copy is not B1a's")
    u1a.save_npz(out / u1a.STRUCT, src=src, dst=dst, rel=np.zeros(src.size, np.int16))
    rows = np.asarray([x["package_row"] for x in nodes], np.int64)
    N = int(H.read_json(ROOT / "outputs" / "b1p" / "settings" / "w" / f"{name}.json")["n_nodes"])
    csr = {t: H.hyperlink_csr(N, rows[s], rows[d]) for t, (s, d) in (("o", parts["o"]), ("w", parts["w"]), ("u", (src, dst)))}
    both = int((H.lookup_entries(N, *csr["w"][:2], *csr["o"][:2]) >= 0).sum())       # the rule's entries that are the reference's
    links = {"reference": L.REF[name], "rule": "B1f's W2_S cut to the setting (b1w's folder)",
             "union_pairs": int(csr["u"][2]), "union_entries": int(csr["u"][1].size), "directed_edges_file": int(src.size),
             "b1o_pairs": int(csr["o"][2]), "b1o_entries": int(csr["o"][1].size),
             "b1w_pairs": int(csr["w"][2]), "b1w_entries": int(csr["w"][1].size),
             "rule_entries_that_are_reference_entries": both, "rule_only_entries": int(csr["w"][1].size) - both,
             "reference_only_entries": int(csr["o"][1].size) - both,
             "typed_entries": int(csr["w"][1].size), "null_entries": int(csr["o"][1].size) - both}
    if links["union_entries"] != links["b1o_entries"] + links["rule_only_entries"]:
        raise SystemExit(f"B1L-U {name}: the union's entries are not b1o's plus the rule's own: {links}")
    bj = variant_build(build, u1a.sha_file(out / u1a.STRUCT), src.size, links, u1a.sha_src(__file__), u1a.utc())
    u1a.write_json(out / "build.json", bj)
    log(f"bench {name}: {links['union_pairs']} pairs, {links['union_entries']} entries ({links['typed_entries']} typed by the rule, "
        f"{links['null_entries']} NULL); {time.time() - t0:.0f} s")
    return 0


# ── zb1l's steps with the union's names ──────────────────────────────────────

def settings(host=False, only=None):
    """zb1h.settings on the union's folder; the record's label says variant u."""
    bind()
    orig = H.write_json

    def labelled(p, obj):
        if isinstance(obj, dict) and "desc_b1" in obj:
            obj["desc_b1"]["variant"] = "u"
            obj["desc_b1"]["reference"] = f"{L.REF[obj['dataset']]} + W2_S"
            obj["b1l_u"] = {"script_sha256": H.sha_src(__file__), "zb1l_sha256": H.sha_src(L.__file__),
                            "zb1h_sha256": H.sha_src(H.__file__)}
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


# ── the report (the laptop): b1t, b1w, b1o, b1u, paired by question ──────────

def union_call(words):
    """ADDS where both models' words are UP, NOTHING where both FLAT, HURTS where both DOWN; NOT_DOWN where neither is DOWN; MIXED
    otherwise; INCOMPLETE where a read is missing."""
    words = list(words)
    if len(words) != len(L.MODELS) or not all(words):
        return "INCOMPLETE"
    if len(set(words)) == 1:
        return CALL[words[0]]
    return "NOT_DOWN" if "DOWN" not in words else "MIXED"


def paired_block(lv):
    """lv = {carve: per-question R@5 of one model}: paired['b1u-b1w'] = {R@5, ci} ..., calls = the words by the interval."""
    paired = {key: H.ci_of(lv[a] - lv[b]) for key, a, b in PAIRS}
    return paired, {key: L.word(p["ci"], *STEP) for key, p in paired.items()}


def report(only=None):
    bind()
    import b1b as BB
    rec = {"declared_in": DECLARED, "stage": STAGE, "candidate": BB.CAND, "boot": {"resamples": BB.BOOT, "seed": BB.SEED},
           "in_domain": {k: list(x) for k, x in H.IN_DOMAIN.items()}, "settings": {}, "missing": []}
    for ds in ((only,) if only else L.SETTINGS):
        if not H.settings_paths(ds)[1].exists():
            continue
        qs = H.read_json(L.B1A_REAL / ds / "queries.json")
        ids, gc = [str(q["id"]) for q in qs], [len(q["golds"]) for q in qs]
        perq, shas = {}, {}
        for carve in LADDER:
            for arm in L.MODELS:
                for fit in H.FITS[arm]:
                    p = read_path(arm, fit, carve, ds)
                    if not p.exists():
                        rec["missing"].append(H.rel(p))
                        continue
                    perq[(arm, carve, fit)] = H.load_pq(p, ids, gc)
                    shas[(arm, carve, fit)] = H.sha_file(p)
        s = {"questions": len(ids), "reference": L.REF[ds], "graph": GRAPH_TEXT, "levels": {}, "paired": {}, "calls": {},
             "gnn_minus_mlp": {}, "reads": {f"{a}/{c}/{f}": {"R@5": round(float(m["R@5"].mean()), 4), "file_sha256": shas[(a, c, f)]}
                                            for (a, c, f), m in perq.items() if c in ("b1o", "b1u")}}
        for which, splits in (("in_domain", H.IN_DOMAIN[ds]), ("zero_shot", tuple(x for x in H.SPLITS if x not in H.IN_DOMAIN[ds]))):
            if not splits:
                continue
            means = {}
            for arm in L.MODELS:
                for carve in LADDER:
                    fits = H.fits_of(arm, splits)
                    got = [perq[(arm, carve, f)]["R@5"] for f in fits if (arm, carve, f) in perq]
                    if len(got) == len(fits):
                        means[(arm, carve)] = np.mean(np.stack(got), axis=0)
                if all((arm, c) in means for c in LADDER):
                    lv = {c: means[(arm, c)] for c in LADDER}
                    s["levels"][f"{which}/{arm}"] = {c: {"R@5": round(float(v.mean()), 4), "R5_ci": BB.boot_ci(v)} for c, v in lv.items()}
                    s["paired"][f"{which}/{arm}"], s["calls"][f"{which}/{arm}"] = paired_block(lv)
            if all((a, "b1u") in means for a in L.MODELS):
                d = means[(L.GNN, "b1u")] - means[(L.MLP, "b1u")]
                ci = BB.boot_ci(d)
                s["gnn_minus_mlp"][f"{which}/b1u"] = {"R@5": round(float(d.mean()), 4), "ci": ci, "label": H.label(ci)}
        s["call"] = {key: union_call([s["calls"].get(f"in_domain/{a}", {}).get(key) for a in L.MODELS]) for key, _a, _b in PAIRS[:2]}
        sj = H.read_json(H.settings_paths(ds)[1])
        s["types"] = {"edges": sj["edges"], "em": sj["types"]["em"], "score": sj["types"]["score"], "checks": sj["checks"],
                      "content_sha256": sj["content_sha256"]}
        s["links"] = H.read_json(BENCH_U / ds / "build.json")["b1l"]
        rec["settings"][ds] = s
        log(f"{ds}: done (calls {s['call']})")
    rec["inputs"] = {"b1l_report_sha256": H.sha_file(B1L_OUT / "report.json") if (B1L_OUT / "report.json").exists() else None}
    rec["zb1lu_sha256"] = H.sha_src(__file__)
    rec["zb1l_sha256"] = H.sha_src(L.__file__)
    rec["zb1h_sha256"] = H.sha_src(H.__file__)
    rec["utc"] = H.utc()
    H.write_json(OUT / "report.json", rec)
    H.write_json(OUT / "calls.json", {"declared_in": DECLARED, "call": {ds: s["call"] for ds, s in rec["settings"].items()},
                                      "calls": {ds: s["calls"] for ds, s in rec["settings"].items()},
                                      "gnn_minus_mlp": {ds: {k: c["label"] for k, c in s["gnn_minus_mlp"].items()}
                                                        for ds, s in rec["settings"].items()}})
    write_md(rec)
    log(f"report: {H.rel(OUT / 'report.md')}; missing {len(rec['missing'])}")
    return 0


def write_md(rec):
    def lv(c):
        lo, hi = c["R5_ci"]
        return f"{100 * c['R@5']:.1f} ({100 * lo:.1f}-{100 * hi:.1f})"

    def df(p):
        lo, hi = p["ci"]
        return f"{100 * p['R@5']:+.1f} ({100 * lo:+.1f}, {100 * hi:+.1f})"

    lines = [f"# B1L-U report (declared in {DECLARED}, section 15)", "",
             "R@5 x100 of the p@swa candidate on HippoRAG 2's setting (B1: 1,000 questions), no new fit. The two models of record, "
             "zg6r and zrckpn, fitted without LLM links, read on four link sets: b1t (the text links only), b1w (the rule's W2_S links, "
             "W2_S-typed), b1o (the text links plus the P pairs of the setting's LLM-built reference, mostly NULL-typed) and b1u (the text "
             "links plus the union of the rule's pairs and the reference's: the rule's entries W2_S-typed, the reference's own NULL). "
             "b1u - b1w is what the reference's pair set adds to the rule's graph; b1u - b1o is what the rule's pairs add to the "
             "reference's. Nothing here says more than these twelve fits on these four graphs.", ""]
    for which, title in (("in_domain", "In-domain"), ("zero_shot", "Held-out")):
        rows = [(ds, arm, s) for ds, s in rec["settings"].items() for arm in L.MODELS if f"{which}/{arm}" in s["levels"]]
        if not rows:
            continue
        lines += [f"## {title}", "", "| setting | model | b1t | b1w | b1o | b1u | b1u - b1w | b1u - b1o | b1u - b1t | words (u-w / u-o / u-t) |",
                  "|---|---|---:|---:|---:|---:|---|---|---|---|"]
        for ds, arm, s in rows:
            k = f"{which}/{arm}"
            c, p, w = s["levels"][k], s["paired"][k], s["calls"][k]
            lines.append(f"| {ds} | {arm} | {lv(c['b1t'])} | {lv(c['b1w'])} | {lv(c['b1o'])} | {lv(c['b1u'])} | {df(p['b1u-b1w'])} | "
                         f"{df(p['b1u-b1o'])} | {df(p['b1u-b1t'])} | {w['b1u-b1w']} / {w['b1u-b1o']} / {w['b1u-b1t']} |")
        lines.append("")
    lines += ["## The call per setting (in-domain, both models)", "", "| setting | b1u - b1w | b1u - b1o |", "|---|---|---|"]
    lines += [f"| {ds} | {s['call']['b1u-b1w']} | {s['call']['b1u-b1o']} |" for ds, s in rec["settings"].items()]
    lines += ["", f"Missing reads: {len(rec['missing'])}.", ""]
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")


# ── the self-test ────────────────────────────────────────────────────────────

def selftest():
    bind()
    try:
        fit = "scr-ml1-zg6r-s0-hp"
        assert H.OUT == OUT and H.CARVE == "b1u" and H.B1A == BENCH_U and H.LTM_OUT == LTM_OUT and H.STAGE == STAGE, "binding"
        assert H.on_b1h is L.on_b1o and H.read_roots is read_roots and all(v.startswith("outputs/b1l/u/") for v in read_roots().values())
        assert set(read_roots()) == set(_L0["read_roots"]()), "the roots' names"
        assert read_path(L.GNN, fit, "b1u", "musique") == OUT / "fits/h" / fit / "reads/musique__b1u.npz", "b1u's read path"
        assert read_path(L.GNN, fit, "b1o", "musique") == B1L_OUT / "fits/h" / fit / "reads/musique__b1o.npz", "b1o is B1L's"
        assert read_path(L.MLP, "scr-ltm-zrckpn-s1", "b1w", "2wiki") == H.B1M / "fits/w/scr-ltm-zrckpn-s1/reads/2wiki__b1w.npz"
        assert read_path(L.GNN, fit, "b1t", "musique") == H.B1P / "fits/t" / fit / "reads/musique__b1t.npz", "b1t is B1P's"
        assert H.read_flags("musique") == ["--read", "musique=b1u", "--out-root", "outputs/b1l/u/fits/h"], H.read_flags("musique")
        assert H.gate_path("musique") == OUT / "gate_musique.json" and H.settings_paths("musique")[0] == OUT / "settings" / "musique.npz"
        # the union of two edge sets, each pair once each way, sorted and distinct
        n = 6
        s1, d1 = L.both_directions([0, 1, 2], [1, 2, 3], n)
        s2, d2 = L.both_directions([1, 4], [2, 5], n)
        s, d = union_directed((s1, d1), (s2, d2), n)
        got = set(zip(s.tolist(), d.tolist()))
        assert got == {(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2), (1, 2), (4, 5), (5, 4)} and s.size == len(got) == 8, got
        assert s.dtype == d.dtype == np.int32 and (np.diff(s.astype(np.int64) * (n + 1) + d) > 0).all(), "sorted and distinct"
        # the variant's build: the lists check's structural sha is B1a's, the label says u
        b = {"files": {"graph_structural.npz": "aa", "x.npz": "bb"}, "passages": 3}
        v = variant_build(b, "cc", 8, {"k": 1}, "dd", "t")
        assert v["files"]["graph_structural.npz"] == "cc" and v["u1a"]["b1a_structural_sha256"] == "aa" and v["u1a"]["variant"] == "u"
        assert v["u1a"]["carve"] == "b1u" and v["b1l"] == {"k": 1} and b["files"]["graph_structural.npz"] == "aa", "variant_build"
        # the words and the call
        assert union_call(["UP", "UP"]) == "ADDS" and union_call(["FLAT", "FLAT"]) == "NOTHING" and union_call(["DOWN", "DOWN"]) == "HURTS"
        assert union_call(["UP", "FLAT"]) == "NOT_DOWN" and union_call(["UP", "DOWN"]) == "MIXED" and union_call(["UP", None]) == "INCOMPLETE"
        assert union_call([]) == "INCOMPLETE" and union_call(["UP"]) == "INCOMPLETE"
        rs = np.random.RandomState(3)
        base = rs.rand(200)
        lv = {"b1t": base, "b1w": base + 0.02, "b1o": base + 0.01, "b1u": base + 0.05}
        paired, words = paired_block(lv)
        assert words == {"b1u-b1w": "UP", "b1u-b1o": "UP", "b1u-b1t": "UP"} and abs(paired["b1u-b1w"]["R@5"] - 0.03) < 1e-4, (paired, words)
        lv["b1u"] = lv["b1w"].copy()
        assert paired_block(lv)[1]["b1u-b1w"] == "FLAT" and paired_block(lv)[1]["b1u-b1o"] == "UP"
        lv["b1u"] = base - 0.05
        assert paired_block(lv)[1]["b1u-b1w"] == "DOWN"
    finally:
        unbind()
    assert L.OUT == B1L_OUT and L.CARVE == "b1o" and H.CARVE == "b1o" and H.OUT == B1L_OUT, "unbind restores zb1l's binding"
    print("zb1lu self-test: OK")
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--selftest" in argv or argv[:1] == ["selftest"]:
        return selftest()
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("cmd", nargs="?")
    ap.add_argument("--dataset", choices=L.SETTINGS)
    ap.add_argument("--host", action="store_true")
    ap.add_argument("--root")
    a, _rest = ap.parse_known_args(argv)
    if a.cmd in ("bench", "chain") and a.dataset is None:
        raise SystemExit(f"{a.cmd}: --dataset is required")
    if a.cmd == "bench":
        return bench(a.dataset)
    if a.cmd == "settings":
        return settings(a.host, a.dataset)
    if a.cmd == "chain":
        root = None if a.root and Path(a.root).resolve() == OUT.resolve() else a.root
        return chain(a.dataset, a.host, root)
    if a.cmd == "report":
        return report(a.dataset)
    bind()                                               # everything else is zb1l's, on the union's globals
    return L.main(argv)


if __name__ == "__main__":
    sys.exit(main())
