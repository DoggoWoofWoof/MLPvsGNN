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
    if a.cmd == "look-dev":
        return H.look_dev(a.dataset, a.root, a.questions)
    raise SystemExit(__doc__)


if __name__ == "__main__":
    sys.exit(main())
