"""AP2 (docs/AP2_RULE_NER_SLOT_ON_B1.md): the models of record read with the rule-built NER slot in the place of spaCy's.

B1L's machinery (zb1l.py, which is zb1h rebound; neither is edited) on carves whose NER family is a form of AP1b's (outputs/ap1b/<setting>/levels.npz) and whose
other five files are b1w's byte for byte (R in the structural slot, B1a's kNN family):

    b1n   R + F0T + K   musique, 2wiki, hotpotqa   (Part A: CN_all's names plus each passage's own title as a name)
    b1x   R + F3T + K   musique                    (Part B, descriptive: the deepest form, refused by S1's price)

The carve is chosen by --variant n|x (or AP2_VARIANT, which the chain's step processes inherit). The command goes first (zb1h reads sys.argv[1:2]).

    python outputs/mp_unified/zap2.py bench --dataset musique --variant n     # the laptop: the setting's folder
    python outputs/mp_unified/zap2.py settings [--dataset D] [--host] --variant n
    python outputs/mp_unified/zap2.py chain --dataset D --host --variant n    # PKO's chain on the carve
    python outputs/mp_unified/zap2.py gate --tag D --host --variant n
    python outputs/mp_unified/zap2.py fork | lists | cols | read-gnn | read-mlp ... --variant n    # zb1h's, on AP2's roots
    python outputs/mp_unified/zap2.py report                                  # the laptop: b1n (and b1x) against b1w and b1t, paired by question
    python outputs/mp_unified/zap2.py selftest
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


def _variant():
    """--variant n|x from the command line (read before anything else, so the constants below are the variant's) or AP2_VARIANT."""
    a = sys.argv[1:]
    v = a[a.index("--variant") + 1] if "--variant" in a and a.index("--variant") + 1 < len(a) else os.environ.get("AP2_VARIANT", "n")
    if v not in ("n", "x"):
        raise SystemExit("AP2: --variant is n or x")
    os.environ["AP2_VARIANT"] = v                               # the chain's step processes inherit it
    return v


VARIANT = _variant()
if "--variant" in sys.argv:                                     # zb1h's header reads sys.argv[1:2] for its BLAS threads: the command goes first
    _i = sys.argv.index("--variant")
    del sys.argv[_i:_i + 2]

import zb1l as L  # noqa: E402  (imports zb1h, whose header sets BLAS from sys.argv[1:2] before numpy loads)

import argparse  # noqa: E402
import contextlib  # noqa: E402
import copy  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import time  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import numpy as np  # noqa: E402

H = L.H
ROOT = H.ROOT
DECLARED = "docs/AP2_RULE_NER_SLOT_ON_B1.md"
STAGE = "AP2"
FORM = {"n": "F0T", "x": "F3T"}
# AP1b's level tables each form is the union of, shallow to deep (at a level the table without the title first): the deepest table's weight wins
TABLES = {"F0T": ("L0", "L0T"), "F3T": ("L0", "L0T", "L1", "L1T", "L2", "L2T", "L3", "L3T")}
CARVE = f"b1{VARIANT}"
OUT = ROOT / "outputs" / CARVE
LTM_OUT = OUT / "ltm"
B1W = ROOT / "outputs" / "bench" / "hipporag2_w"               # b1w's folder: the source of five files
B1A_REAL = ROOT / "outputs" / "bench" / "hipporag2"            # B1a's own folder: its lists and the NER file they name
BENCH_G = OUT / "bench" / f"hipporag2_{VARIANT}"
AP1B = ROOT / "outputs" / "ap1b"
NER = "graph_ner.npz"
SETTINGS = ("2wiki", "hotpotqa", "musique") if VARIANT == "n" else ("musique",)
TAGS = {ds: (ds,) for ds in SETTINGS}                           # one gate and one set of reads per setting, as B1L
CARVES = ("b1t", "b1w", CARVE)
def graph_text(v):
    return (f"AP2 b1{v}: b1w's structural (R) and kNN families, and AP1b's form {FORM[v]} as the NER family "
            "(co-mention weights, df in [2, 25])")


GRAPH_TEXT = graph_text(VARIANT)
NO_STEP1 = {"state": f"not applicable: B1's settings have no step-1 cache (AP2, {DECLARED})"}
GNN, MLP = L.GNN, L.MLP
MODELS = (GNN, MLP)
STEP = ("UP", "DOWN", "FLAT")                                   # a pair against b1t by its interval: above 0, below 0, neither
VS_W = ("GREATER", "LESS", "SAME")                              # b1n - b1w, b1x - b1w, b1x - b1n by their interval
HOPS = ("2hop", "3hop", "4hop")
REPORT_OUT = ROOT / "outputs" / "ap2"
_ON_B1H = L._ON_B1H


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] zap2: {msg}", flush=True)


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read_roots():
    d = f"outputs/{CARVE}"
    return {"pools": f"{d}/pools", "plook": f"{d}/look", "cache": f"{d}/cache", "rmatch": f"{d}/chains/rmatch", "zrc": f"{d}/chains/zrc",
            "links": f"{d}/links", "kt": f"{d}/kt"}


# ── the lists check: B1a's lists name B1a's files; this carve replaced two of them (structural, NER) ──────────

def check_lists(named, shas, build):
    """B1a's lists record (setting_files_sha256 `named`) against this carve's file shas: every file but the structural and NER ones equal, the
    structural one the sha this variant replaced (u1a.b1a_structural_sha256), the NER one the sha b1w's file has (ap2.b1a_ner_sha256)."""
    swapped = {"graph_structural.npz": build["u1a"]["b1a_structural_sha256"], NER: build["ap2"]["b1a_ner_sha256"]}
    if {f: s for f, s in named.items() if f not in swapped} != {f: s for f, s in shas.items() if f not in swapped}:
        raise SystemExit("AP2: B1a's lists were built on other setting files than this carve's")
    for f, s in swapped.items():
        if named.get(f) != s:
            raise SystemExit(f"AP2: B1a's lists name another {f} than the one this carve replaced")
    return named


def lists_of(M, st):
    """look_b1u.load_lists with the NER file checked as the structural one is: B1a's lists, read from B1a's folder."""
    rec = json.loads((M.LBU.B1A / st.name / "b1_lists.json").read_text(encoding="utf-8"))
    named = check_lists(rec["setting_files_sha256"], st.shas, st.build)
    here = M.LB.BENCH
    M.LB.BENCH = M.LBU.B1A
    try:
        return M.LBU._ORIG_LISTS(SimpleNamespace(**{**vars(st), "shas": named}))
    finally:
        M.LB.BENCH = here


@contextlib.contextmanager
def on_b1n(M, name, stats, look=False):
    """zb1h.on_b1h with this carve's lists: the folder holds no lists of its own (B1a's, checked by check_lists)."""
    P_ = H.zb().Patches()
    try:
        P_.set(M.LB, load_lists=lambda st: lists_of(M, st))
        with _ON_B1H(M, name, stats, look) as B:
            yield B
    finally:
        P_.restore()


def bind():
    """zb1h's module globals set to AP2's (its functions look them up at call time); safe to call again."""
    for k, v in (("DECLARED", DECLARED), ("STAGE", STAGE), ("OUT", OUT), ("LTM_OUT", LTM_OUT), ("B1A", BENCH_G), ("CARVE", CARVE),
                 ("CARVES", CARVES), ("SETTINGS", SETTINGS), ("TAGS", TAGS), ("GRAPH_TEXT", GRAPH_TEXT), ("NO_STEP1", NO_STEP1),
                 ("read_roots", read_roots), ("on_b1h", on_b1n)):
        setattr(H, k, v)


# ── the setting's folder (bench) ──────────────────────────────────────────────

def form_pairs(z, n, tables):
    """(a, b, w) of the union of the level tables of `z` (a levels.npz mapping: L0_src, L0_dst, L0_w, ...), shallow to deep, a < b, ascending
    by (a, b); where several tables hold a pair the last one's weight."""
    ks, ws = [], []
    for t in tables:
        a, b = z[f"{t}_src"].astype(np.int64), z[f"{t}_dst"].astype(np.int64)
        if a.size and (not (a < b).all() or a.min() < 0 or b.max() >= n):
            raise SystemExit(f"AP2: table {t} holds a pair that is not a < b inside the {n} passages")
        ks.append(a * n + b)
        ws.append(z[f"{t}_w"].astype(np.float32))
    k, w = np.concatenate(ks), np.concatenate(ws)
    o = np.argsort(k, kind="stable")                            # equal keys keep the table order: the deepest is the last of its run
    k, w = k[o], w[o]
    last = np.ones(k.size, bool)
    last[:-1] = k[1:] != k[:-1]
    k, w = k[last], w[last]
    return (k // n).astype(np.int32), (k % n).astype(np.int32), w


def variant_build(build, ner_sha, links, script_sha, utc):
    """b1w's build.json for the carve: the NER file's sha replaced, the u1a block relabelled (its structural sha is still B1a's), an ap2 block."""
    b = json.loads(json.dumps(build))
    b["files"][NER] = ner_sha
    b["u1a"] = {**build["u1a"], "declared_in": DECLARED, "variant": VARIANT, "carve": CARVE, "script_sha256": script_sha, "utc": utc,
                "structural": f"{build['u1a'].get('structural', 'the rule W2_S pairs')} (b1w's, unchanged)"}
    b["ap2"] = links
    return b


def bench(name):
    import u1a
    t0 = time.time()
    form = FORM[VARIANT]
    src_dir, out = B1W / name, BENCH_G / name
    build = H.read_json(src_dir / "build.json")
    bad = [f for f in u1a.FILES if u1a.sha_file(src_dir / f) != build["files"].get(f)]
    if bad:
        raise SystemExit(f"AP2 {name}: {bad} differ from b1w's build.json")
    real = H.read_json(B1A_REAL / name / "build.json")
    named = H.read_json(B1A_REAL / name / "b1_lists.json")["setting_files_sha256"]
    if not (build["files"][NER] == real["files"][NER] == named[NER]):
        raise SystemExit(f"AP2 {name}: b1w's NER file is not the one B1a's build.json and lists name (G2)")
    n = len(H.read_json(src_dir / "nodes.json"))
    if n != int(build["passages"]):
        raise SystemExit(f"AP2 {name}: b1w's folder holds {n} passages, build.json {build['passages']}")
    rep = H.read_json(AP1B / "report.json")["settings"][name]
    lv = AP1B / name / "levels.npz"
    if sha_file(lv) != rep["levels_npz_sha256"]:
        raise SystemExit(f"AP2 {name}: levels.npz is not the one AP1b's report names (G1)")
    with np.load(lv) as z:
        a, b, w = form_pairs(z, n, TABLES[form])
    want = int(rep["forms"][form]["form_pairs"])
    if int(a.size) != want:
        raise SystemExit(f"AP2 {name}: {a.size} {form} pairs, AP1b's report {want} (G1)")
    if not (np.isfinite(w).all() and float(w.min()) >= 0.04 - 1e-6):                    # G3: 1/25 is the smallest weight at the cap
        raise SystemExit(f"AP2 {name}: a weight below 1/25 or not finite (G3)")
    out.mkdir(parents=True, exist_ok=True)
    for f in u1a.FILES:
        if f != NER:
            shutil.copyfile(src_dir / f, out / f)
            if u1a.sha_file(out / f) != build["files"][f]:
                raise SystemExit(f"AP2 {out / f}: the copy is not b1w's")
    u1a.save_npz(out / NER, src=a, dst=b, weight=w)
    with np.load(src_dir / NER) as zs:
        spacy = int(zs["src"].size)
    links = {"form": form, "tables": list(TABLES[form]), "pairs": int(a.size), "spacy_pairs": spacy, "directed_edges_file": int(a.size),
             "weight_quantiles": L.quantiles(w), "weight_max": round(float(w.max()), 4), "levels_npz_sha256": rep["levels_npz_sha256"],
             "ap1b_report_sha256": sha_file(AP1B / "report.json"), "b1a_ner_sha256": build["files"][NER], "weights_entered": True,
             "zap1b_script_sha256": H.read_json(AP1B / "report.json")["script_sha256"]}
    vb = variant_build(build, u1a.sha_file(out / NER), links, u1a.sha_src(__file__), u1a.utc())
    u1a.write_json(out / "build.json", vb)
    shas = {f: u1a.sha_file(out / f) for f in u1a.FILES}                                # G2b: the folder as written is the record's, and B1a's lists accept it
    if shas != vb["files"] or any(shas[f] != build["files"][f] for f in u1a.FILES if f != NER):
        raise SystemExit(f"AP2 {name}: the folder as written is not its build.json's, or a copied file is not b1w's (G2b)")
    check_lists(named, shas, vb)
    log(f"bench {name} {CARVE}: {a.size} {form} pairs ({spacy} spaCy pairs in b1w); weights {L.quantiles(w)}; {time.time() - t0:.0f} s")
    return 0


# ── zb1h's steps with AP2's names ─────────────────────────────────────────────

def settings(host=False, only=None):
    """zb1h.settings on the carve's folder; the record's label names the form."""
    bind()
    orig = H.write_json

    def labelled(p, obj):
        if isinstance(obj, dict) and "desc_b1" in obj:
            obj["desc_b1"]["variant"] = VARIANT
            obj["desc_b1"]["form"] = FORM[VARIANT]
            obj["ap2"] = {"script_sha256": H.sha_src(__file__), "zb1h_sha256": H.sha_src(H.__file__)}
        return orig(p, obj)

    H.write_json = labelled
    try:
        return H.settings(host, only)
    finally:
        H.write_json = orig


def chain(name, host=False, root=None):
    """zb1h.chain, whose steps are run as this file's (zb1h spawns its own __file__); AP2_VARIANT is inherited."""
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
    H.write_json(OUT / f"ap2_gate_{tag}.json", {
        "declared_in": DECLARED, "stage": STAGE, "tag": tag, "pko_gate": H.rel(R.gate), "pko_gate_sha256": H.sha_file(R.gate),
        "verdict": verdict, "carves": carves, "settings": {ds: H.sha_file(H.settings_paths(ds)[1]) for ds in TAGS[tag]},
        "zb1h_sha256": H.sha_src(H.__file__), "zap2_sha256": H.sha_src(__file__), "utc": H.utc()})
    return 0


# ── the report (the laptop): b1n (and b1x) against b1w and b1t, paired by question ──

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


def word(ci, up, down, flat):
    return up if ci[0] > 0 else down if ci[1] < 0 else flat


PAIRS = {"n": (("b1n-b1w", "n", "w", VS_W), ("b1n-b1t", "n", "t", STEP), ("b1w-b1t", "w", "t", STEP)),
         "x": (("b1x-b1w", "x", "w", VS_W), ("b1x-b1n", "x", "n", VS_W), ("b1x-b1t", "x", "t", STEP), ("b1w-b1t", "w", "t", STEP))}


def pairs_block(means, which, v, hop_mask=None):
    """paired['b1n-b1w M'] = {R@5, ci}; calls[M] = {pair: word}; over the questions in hop_mask if given. means: {(model, carve, which): R@5 per question}."""
    paired, calls = {}, {}
    letter = {"t": "b1t", "w": "b1w", "n": "b1n", "x": "b1x"}
    for arm in MODELS:
        d = {k: means.get((arm, c, which)) for k, c in letter.items()}
        need = {x for _key, a_, b_, _w in PAIRS[v] for x in (a_, b_)}
        if any(d[k] is None for k in need):
            continue
        if hop_mask is not None:
            d = {k: (x[hop_mask] if x is not None else None) for k, x in d.items()}
        for key, a_, b_, _words in PAIRS[v]:
            paired[f"{key} {arm}"] = H.ci_of(d[a_] - d[b_])
        calls[arm] = {key: word(paired[f"{key} {arm}"]["ci"], *words) for key, _a, _b, words in PAIRS[v]}
    return paired, calls


def stage_call(words):
    return L.stage_call(words)


def verdict(calls_by_ds, datasets=("musique", "2wiki", "hotpotqa")):
    """Part A: REPLACE when no setting's call is LESS or MIXED (all of them GREATER, SAME or NOT_LESS); HOLD when one is LESS or MIXED
    (decided even if another is missing); INCOMPLETE otherwise."""
    calls = [calls_by_ds.get(ds) for ds in datasets]
    if any(c in ("LESS", "MIXED") for c in calls):
        return "HOLD"
    if any(c in (None, "INCOMPLETE") for c in calls):
        return "INCOMPLETE"
    return "REPLACE"


def hop_of(kind):
    return f"{str(kind)[0]}hop"


def read_carve(v, ds, rec):
    """The reads of carve b1<v> (with b1t and b1w) on setting ds: the block of the report and the per-question means."""
    with as_variant(v):
        qs = H.read_json(B1A_REAL / ds / "queries.json")
        ids, gc = [str(q["id"]) for q in qs], [len(q["golds"]) for q in qs]
        perq, shas = H.read_set(ds, ids, gc, rec["missing"])
        s = {"questions": len(ids), "graph": graph_text(v)}
        reads, means = H.reads_block(ds, perq, shas)
        s.update(reads)
    return s, means, qs


def report():
    bind()
    import b1b as BB
    rec = {"declared_in": DECLARED, "stage": STAGE, "candidate": BB.CAND, "boot": {"resamples": BB.BOOT, "seed": BB.SEED},
           "in_domain": {k: list(x) for k, x in H.IN_DOMAIN.items()}, "settings": {}, "partB": {}, "missing": []}
    b1m = H.read_json(ROOT / "outputs" / "b1m" / "report.json")
    rec["inputs_sha256"] = {x: H.sha_file(ROOT / "outputs" / x / "report.json") for x in ("b1p", "b1m", "ap1b")}
    checks = {"equal_to_b1m": {"checked": 0, "bad": []}}
    calls_by_ds, means_n = {}, {}
    for ds in ("2wiki", "hotpotqa", "musique"):
        if not (consts_of("n")["B1A"] / ds / "build.json").exists() or not (ROOT / "outputs" / "b1n" / "settings" / f"{ds}.json").exists():
            continue
        s, means, qs = read_carve("n", ds, rec)
        means_n[ds] = means
        n, bad = H.equal_to_b1m(ds, s, b1m)
        checks["equal_to_b1m"]["checked"] += n
        checks["equal_to_b1m"]["bad"] += [f"{ds} {k}" for k in bad]
        s["paired"], c_in = pairs_block(means, "in_domain", "n")
        s["paired_zero_shot"], c_zs = pairs_block(means, "zero_shot", "n")
        s["calls_vs_b1w"] = {"in_domain": c_in, "zero_shot": c_zs}
        s["call"] = stage_call([c_in.get(a, {}).get("b1n-b1w") for a in MODELS])
        calls_by_ds[ds] = s["call"]
        if ds == "musique":
            hops = np.asarray([hop_of(q["kind"]) for q in qs])
            s["by_hops"] = {}
            for h in HOPS:
                m = hops == h
                pr, cl = pairs_block(means, "in_domain", "n", m)
                s["by_hops"][h] = {"questions": int(m.sum()), "paired": pr, "calls": cl}
        s["links"] = H.read_json(consts_of("n")["B1A"] / ds / "build.json")["ap2"]
        rec["settings"][ds] = s
        log(f"{ds}: done (call {s['call']})")
    if checks["equal_to_b1m"]["bad"]:
        raise SystemExit(f"AP2: rows recomputed from B1P's and B1M's reads differ from B1M's filed rows: {checks['equal_to_b1m']['bad']}")
    rec["checks"] = checks
    rec["verdict"] = verdict(calls_by_ds)
    if (consts_of("x")["B1A"] / "musique" / "build.json").exists() and (ROOT / "outputs" / "b1x" / "settings" / "musique.json").exists() \
            and "musique" in means_n:
        s, means, qs = read_carve("x", "musique", rec)
        means = {**means, **{k: v for k, v in means_n["musique"].items() if k[1] == "b1n"}}
        s["paired"], c_in = pairs_block(means, "in_domain", "x")
        s["paired_zero_shot"], c_zs = pairs_block(means, "zero_shot", "x")
        s["calls_vs_b1w"] = {"in_domain": c_in, "zero_shot": c_zs}
        s["links"] = H.read_json(consts_of("x")["B1A"] / "musique" / "build.json")["ap2"]
        rec["partB"]["musique"] = s
    rec["zap2_sha256"] = H.sha_src(__file__)
    rec["zb1h_sha256"] = H.sha_src(H.__file__)
    rec["utc"] = H.utc()
    H.write_json(REPORT_OUT / "report.json", rec)
    H.write_json(REPORT_OUT / "calls.json", {
        "declared_in": DECLARED, "verdict": rec["verdict"], "call": {ds: s["call"] for ds, s in rec["settings"].items()},
        "calls_vs_b1w": {ds: s["calls_vs_b1w"] for ds, s in rec["settings"].items()},
        "partB": {ds: s["calls_vs_b1w"] for ds, s in rec["partB"].items()}})
    write_md(rec)
    log(f"report: {H.rel(REPORT_OUT / 'report.md')}; missing {len(rec['missing'])}; verdict {rec['verdict']}; "
        f"equal to B1M's rows: {checks['equal_to_b1m']['checked']} checked")
    return 0


def lv(c):
    return f"{100 * c['R@5']:.1f} ({100 * c['R5_ci'][0]:.1f}-{100 * c['R5_ci'][1]:.1f})" if c else "--"


def df(c):
    return f"{100 * c['R@5']:+.1f} ({100 * c['ci'][0]:+.1f}, {100 * c['ci'][1]:+.1f})" if c else "--"


def write_md(rec):
    T = [f"# AP2 report (declared in {DECLARED})", "",
         "R@5 x100 on HippoRAG 2's settings (1,000 questions each), no new fit. The models of record (zg6r, zrckpn) are read on b1t (N + K), b1w (R + spaCy's N + K) and **b1n "
         "(R + F0T + K)**: the NER slot holds the rule's names with each passage's own title (AP1b), the other files are b1w's. A read-time shift: the models were fitted on "
         "spaCy's NER store. In-domain = the mean over the fits that trained on the setting's partner split. 95% bootstrap intervals over the questions (2,000 resamples, "
         "seed 0); words by the interval, no threshold (GREATER / SAME / LESS for b1n - b1w; UP / DOWN / FLAT against b1t).", "",
         f"**Verdict (Part A): {rec['verdict']}** (REPLACE when no setting's call is LESS or MIXED).", ""]
    for which, title, key in (("in_domain", "In-domain", "paired"), ("zero_shot", "Held-out", "paired_zero_shot")):
        T += [f"## {title}", "", "| setting | model | b1t | b1w | b1n | b1n - b1w | b1n - b1t | b1w - b1t | word (b1n - b1w) |", "|---|---|---:|---:|---:|---|---|---|---|"]
        for ds, s in rec["settings"].items():
            for arm in MODELS:
                ld = {c: s[which].get(f"{arm}/{c}") for c in ("b1t", "b1w", "b1n")}
                if not any(ld.values()):
                    continue
                pr = {k: s[key].get(f"{k} {arm}") for k in ("b1n-b1w", "b1n-b1t", "b1w-b1t")}
                w = s["calls_vs_b1w"][which].get(arm, {}).get("b1n-b1w", "--")
                T.append(f"| {ds} | {arm} | {lv(ld['b1t'])} | {lv(ld['b1w'])} | {lv(ld['b1n'])} | {df(pr['b1n-b1w'])} | {df(pr['b1n-b1t'])} | {df(pr['b1w-b1t'])} | {w} |")
        T.append("")
    T += ["## The setting's call (in-domain, both models)", "", "| setting | call |", "|---|---|"]
    for ds, s in rec["settings"].items():
        T.append(f"| {ds} | {s['call']} |")
    T += ["", "## zg6r - zrckpn at b1n, paired by question", "", "| setting | read | zg6r - zrckpn (CI) | label |", "|---|---|---|---|"]
    for ds, s in rec["settings"].items():
        for k, c in s.get("gnn_minus_mlp", {}).items():
            if k.split("/")[1] == "b1n":
                T.append(f"| {ds} | {k.split('/')[0].replace('_', '-')} | {df(c)} | {c['label']} |")
    mq = rec["settings"].get("musique")
    if mq and mq.get("by_hops"):
        T += ["", "## MuSiQue by hop depth (in-domain, paired by question, descriptive)", "", "| hops | questions | model | b1n - b1w |", "|---|---:|---|---|"]
        for h in HOPS:
            x = mq["by_hops"][h]
            for arm in MODELS:
                if x["paired"].get(f"b1n-b1w {arm}"):
                    T.append(f"| {h} | {x['questions']} | {arm} | {df(x['paired'][f'b1n-b1w {arm}'])} |")
    T += ["", "## The NER files", "", "| setting | form | pairs | spaCy pairs in b1w | weight max |", "|---|---|---:|---:|---:|"]
    for ds, s in rec["settings"].items():
        k = s["links"]
        T.append(f"| {ds} | {k['form']} | {k['pairs']} | {k['spacy_pairs']} | {k['weight_max']} |")
    if rec.get("partB"):
        T += ["", "## Part B: b1x (R + F3T + K), MuSiQue, descriptive, no verdict", "",
              "| model | b1w | b1n | b1x | b1x - b1w | b1x - b1n | word (b1x - b1w) | word (b1x - b1n) |", "|---|---:|---:|---:|---|---|---|---|"]
        s = rec["partB"]["musique"]
        for arm in MODELS:
            ld = {c: (s["in_domain"].get(f"{arm}/{c}") or (rec["settings"]["musique"]["in_domain"].get(f"{arm}/{c}") if rec["settings"].get("musique") else None))
                  for c in ("b1w", "b1n", "b1x")}
            pr = {k: s["paired"].get(f"{k} {arm}") for k in ("b1x-b1w", "b1x-b1n")}
            cl = s["calls_vs_b1w"]["in_domain"].get(arm, {})
            T.append(f"| {arm} | {lv(ld['b1w'])} | {lv(ld['b1n'])} | {lv(ld['b1x'])} | {df(pr['b1x-b1w'])} | {df(pr['b1x-b1n'])} | {cl.get('b1x-b1w', '--')} | {cl.get('b1x-b1n', '--')} |")
    T += ["", f"Missing reads: {len(rec['missing'])}."]
    REPORT_OUT.mkdir(parents=True, exist_ok=True)
    (REPORT_OUT / "report.md").write_text("\n".join(T) + "\n", encoding="utf-8")


# ── self-test ─────────────────────────────────────────────────────────────────

def selftest():
    bind()
    fit = "scr-ml1-zg6r-s0-hp"
    assert CARVE == f"b1{VARIANT}" and OUT == ROOT / "outputs" / CARVE and BENCH_G == OUT / "bench" / f"hipporag2_{VARIANT}"
    assert H.read_path(L.GNN, fit, CARVE, SETTINGS[0]) == OUT / "fits/h" / fit / f"reads/{SETTINGS[0]}__{CARVE}.npz", "the carve's read path"
    assert H.read_path(L.GNN, fit, "b1t", "musique") == H.B1P / "fits/t" / fit / "reads/musique__b1t.npz", "b1t is B1P's"
    assert H.read_flags(SETTINGS[0]) == ["--read", f"{SETTINGS[0]}={CARVE}", "--out-root", f"outputs/{CARVE}/fits/h"], H.read_flags(SETTINGS[0])
    assert H.gate_path("musique" if "musique" in TAGS else SETTINGS[0]).parent == OUT
    assert H.B1A == BENCH_G and H.CARVES == CARVES and H.CARVE == CARVE and CARVES[-1] == CARVE
    assert H.on_b1h is on_b1n and H.read_roots is read_roots and all(v.startswith(f"outputs/{CARVE}/") for v in read_roots().values())
    assert set(read_roots()) == {"pools", "plook", "cache", "rmatch", "zrc", "links", "kt"}
    assert os.environ["AP2_VARIANT"] == VARIANT, "the chain's step processes inherit the variant"
    assert set(TAGS) == set(SETTINGS) and all(TAGS[d] == (d,) for d in TAGS)
    assert set(TABLES["F0T"]) == {"L0", "L0T"} and TABLES["F3T"] == tuple(f"L{i}{t}" for i in range(4) for t in ("", "T"))
    # the form's pairs: the union of the tables, a < b, ascending, the deepest table's weight
    n = 10
    z = {"L0_src": np.asarray([0, 1, 4], np.int32), "L0_dst": np.asarray([1, 2, 5], np.int32), "L0_w": np.asarray([0.5, 0.25, 0.04], np.float32),
         "L0T_src": np.asarray([0, 2], np.int32), "L0T_dst": np.asarray([1, 3], np.int32), "L0T_w": np.asarray([0.75, 0.1], np.float32)}
    a, b, w = form_pairs(z, n, ("L0", "L0T"))
    assert list(zip(a.tolist(), b.tolist())) == [(0, 1), (1, 2), (2, 3), (4, 5)] and np.allclose(w, [0.75, 0.25, 0.1, 0.04], rtol=0, atol=1e-6), (a, b, w)
    assert a.dtype == np.int32 and b.dtype == np.int32 and w.dtype == np.float32
    z2 = {"L0_src": np.asarray([3], np.int32), "L0_dst": np.asarray([2], np.int32), "L0_w": np.asarray([0.5], np.float32)}
    refused = False
    try:
        form_pairs(z2, n, ("L0",))
    except SystemExit:
        refused = True
    assert refused, "a pair must be a < b"
    # the lists check: B1a's named shas against the carve's, the structural and NER files replaced
    build = {"u1a": {"b1a_structural_sha256": "S0"}, "ap2": {"b1a_ner_sha256": "N0"}}
    named = {"first_stage.npz": "f", "graph_structural.npz": "S0", NER: "N0", "graph_knn.npz": "k", "nodes.json": "n", "queries.json": "q"}
    shas = {"first_stage.npz": "f", "graph_structural.npz": "S1", NER: "N1", "graph_knn.npz": "k", "nodes.json": "n", "queries.json": "q"}
    assert check_lists(named, shas, build) == named
    for bad_named, bad_shas in (({**named, "graph_knn.npz": "x"}, shas), ({**named, NER: "N9"}, shas), ({**named, "graph_structural.npz": "S9"}, shas)):
        refused = False
        try:
            check_lists(bad_named, bad_shas, build)
        except SystemExit:
            refused = True
        assert refused, "the lists check must refuse a file that differs"
    # the build.json of the carve: b1w's, the NER sha replaced, b1w's structural sha kept, the u1a block relabelled
    bw = {"files": {NER: "N0", "graph_structural.npz": "SW"}, "passages": 5, "u1a": {"b1a_structural_sha256": "S0", "structural": "U2b's W2_S", "variant": "w", "edges": 7}}
    vb = variant_build(bw, "N1", {"form": FORM[VARIANT], "b1a_ner_sha256": "N0"}, "dd", "t")
    assert vb["files"] == {NER: "N1", "graph_structural.npz": "SW"} and bw["files"][NER] == "N0", "b1w's record is not touched"
    assert vb["u1a"]["b1a_structural_sha256"] == "S0" and vb["u1a"]["variant"] == VARIANT and vb["u1a"]["carve"] == CARVE and vb["u1a"]["edges"] == 7
    assert vb["ap2"]["b1a_ner_sha256"] == "N0" and vb["passages"] == 5
    # the settings record's label
    seen_ = []
    orig, hold = H.write_json, H.settings
    try:
        H.write_json = lambda p, obj: seen_.append(obj)
        H.settings = lambda host=False, only=None: H.write_json(Path("x.json"), {"dataset": "musique", "desc_b1": {"variant": "h"}}) or 0
        assert settings(False, "musique") == 0
    finally:
        H.settings, H.write_json = hold, orig
    assert seen_[0]["desc_b1"]["variant"] == VARIANT and seen_[0]["desc_b1"]["form"] == FORM[VARIANT] and "ap2" in seen_[0], "label"
    assert H.write_json is orig and H.settings is hold, "put back"
    # as_variant sets zb1h's globals to the other variant's and puts them back
    here = (H.CARVE, H.OUT, H.CARVES)
    other = "x" if VARIANT == "n" else "n"
    with as_variant(other):
        assert H.CARVE == f"b1{other}" and H.OUT == ROOT / "outputs" / f"b1{other}" and H.CARVES == ("b1t", "b1w", f"b1{other}")
        assert H.read_path(L.GNN, "scr-ml1-zg6r-s0-hp", f"b1{other}", "musique") == ROOT / "outputs" / f"b1{other}" / "fits/h" / "scr-ml1-zg6r-s0-hp" / f"reads/musique__b1{other}.npz"
    assert (H.CARVE, H.OUT, H.CARVES) == here, "put back"
    # the pairs, the words, the setting's call and the verdict on fake reads
    rs = np.random.RandomState(7)
    nq = 90
    base = rs.rand(nq) * 0.5
    means = {}
    for arm in MODELS:
        for which in ("in_domain", "zero_shot"):
            means[(arm, "b1t", which)] = base
            means[(arm, "b1w", which)] = base + 0.05 + rs.randn(nq) * 0.03
            means[(arm, "b1n", which)] = means[(arm, "b1w", which)] + 0.04 + rs.randn(nq) * 0.03          # b1n above b1w
            means[(arm, "b1x", which)] = means[(arm, "b1n", which)] - 0.10 + rs.randn(nq) * 0.03          # b1x below b1n
    pr, cl = pairs_block(means, "in_domain", "n")
    assert set(pr) == {f"{k} {a}" for a in MODELS for k, _x, _y, _w in PAIRS["n"]} and set(cl) == set(MODELS)
    for a in MODELS:
        assert cl[a]["b1n-b1w"] == "GREATER" and cl[a]["b1n-b1t"] == "UP" and cl[a]["b1w-b1t"] == "UP", cl[a]
    pr_x, cl_x = pairs_block(means, "in_domain", "x")
    for a in MODELS:
        assert cl_x[a]["b1x-b1n"] == "LESS" and cl_x[a]["b1x-b1w"] in ("LESS", "SAME"), cl_x[a]
    assert pairs_block({k: v for k, v in means.items() if k[1] != "b1n"}, "in_domain", "n") == ({}, {}), "no b1n, no pairs"
    kinds = ["2hop"] * 50 + ["3hop1"] * 25 + ["4hop2"] * 15
    hops = np.asarray([hop_of(k) for k in kinds])
    assert [int((hops == h).sum()) for h in HOPS] == [50, 25, 15]
    pr2, cl2 = pairs_block(means, "in_domain", "n", hops == "4hop")
    assert set(cl2) == set(MODELS) and pr2[f"b1n-b1w {GNN}"]["R@5"] != pr[f"b1n-b1w {GNN}"]["R@5"], "over the 4-hop questions only"
    assert stage_call(["GREATER", "GREATER"]) == "GREATER" and stage_call(["GREATER", "SAME"]) == "NOT_LESS"
    assert stage_call(["LESS", "SAME"]) == "MIXED" and stage_call(["LESS", "LESS"]) == "LESS" and stage_call(["SAME", None]) == "INCOMPLETE"
    assert verdict({"musique": "GREATER", "2wiki": "SAME", "hotpotqa": "NOT_LESS"}) == "REPLACE"
    assert verdict({"musique": "SAME", "2wiki": "LESS", "hotpotqa": "SAME"}) == "HOLD"
    assert verdict({"musique": "SAME", "2wiki": "MIXED"}) == "HOLD", "a LESS or MIXED decides even with a setting missing"
    assert verdict({"musique": "SAME", "2wiki": "SAME"}) == "INCOMPLETE" and verdict({}) == "INCOMPLETE"
    # the markdown on a fake record
    cell = {"R@5": 0.5, "R5_ci": [0.4, 0.6]}
    srec = {"in_domain": {f"{a}/{c}": cell for a in MODELS for c in ("b1t", "b1w", "b1n")}, "zero_shot": {}, "paired": pr, "paired_zero_shot": {},
            "calls_vs_b1w": {"in_domain": cl, "zero_shot": {}}, "call": "GREATER",
            "by_hops": {h: {"questions": 5, "paired": pr, "calls": cl} for h in HOPS},
            "gnn_minus_mlp": {"in_domain/b1n": {"R@5": 0.01, "ci": [0.0, 0.02], "label": "GNN_ABOVE"}, "in_domain/b1w": {"R@5": 0.0, "ci": [-0.01, 0.01], "label": "LEVEL"}},
            "links": {"form": "F0T", "pairs": 3, "spacy_pairs": 2, "weight_max": 1.0}}
    rec = {"verdict": "REPLACE", "settings": {"musique": srec}, "partB": {"musique": {
        "in_domain": {f"{a}/b1x": cell for a in MODELS}, "paired": pr_x, "calls_vs_b1w": {"in_domain": cl_x}}}, "missing": []}
    import tempfile
    global REPORT_OUT
    keep_out = REPORT_OUT
    try:
        REPORT_OUT = Path(tempfile.mkdtemp())
        write_md(rec)
        md = (REPORT_OUT / "report.md").read_text(encoding="utf-8")
    finally:
        REPORT_OUT = keep_out
    assert "**Verdict (Part A): REPLACE**" in md and "| musique | zg6r | 50.0 (40.0-60.0)" in md and "## Part B" in md and "no threshold" in md and "hop depth" in md, md[:500]
    gap = md.split("## zg6r - zrckpn at b1n")[1].split("\n## ")[0]
    assert "| musique | in-domain | +1.0 (+0.0, +2.0) | GNN_ABOVE |" in gap and "LEVEL" not in gap, gap
    print(f"zap2 self-test: OK ({CARVE})")
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
