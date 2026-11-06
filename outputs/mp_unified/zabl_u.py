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


def build_graph(titles, texts, dense_get, rule=None, lo=0, progress=None):
    """(indptr, col, record) of W2_S on the given titles and texts: the mention links of variant W2 (the context choice
    by dense_get), plus the sibling links, as an undirected CSR."""
    rule = rule or zu2b.NEW_RULE
    n = len(titles)
    S = zu2b.Surfaces(titles, rule)
    desc = np.asarray([zu2b.is_descriptive(t, x) for t, x in zip(titles, texts)], bool)
    cap, low = zu2b.stat_pass(S, texts, titles, rule)
    common = low > cap
    edges, stats = zu2b_fast.link_pass(S, texts, titles, rule, common, dense_get, VARIANT, lo=lo, progress=progress)
    s_, t_ = (a.astype(np.int64) for a in edges["W2"])
    key = s_ * n + t_
    if np.unique(key).size != key.size or (s_ == t_).any():
        raise SystemExit("a mention pair repeats or loops")
    sib_s, sib_d = zu2b.sibling_pairs(S.group, desc)
    ip, col = zu2b.undirected_csr(n, np.concatenate([s_, sib_s]), np.concatenate([t_, sib_d]))
    gs = np.bincount(S.group[desc]) if desc.any() else np.zeros(1, np.int64)
    rec = {"nodes": n, "descriptive_nodes": int(desc.sum()), "surfaces": len(S.surf), "common_forms": int(common.sum()),
           "forms_seen": int(((cap + low) > 0).sum()), "mention_directed_edges": int(s_.size),
           "sibling_pairs": int(sib_s.size), "groups_with_siblings": int((gs > 1).sum()), "largest_group": int(gs.max()),
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


def build_stage(name, host=False):
    if name not in KB:
        raise SystemExit(f"a KB dataset of {KB}")
    served, freeze = u1b.package_root(host)
    titles, _names, n, nodes_sha = u1b.read_nodes(served, name)
    src, dst, _rel, _fam = u1b.package_structural(served, name)
    kip, kcol = kb_neighbours(n, src, dst)
    log(f"{name}: {n} nodes, {kcol.size} KB neighbour entries")
    texts = texts_of(titles, kip, kcol)
    recipe_sha = hashlib.sha256((SEP + "|" + json.dumps(texts_of(["a", "b", "c"], *kb_neighbours(
        3, [0, 1], [1, 2])))).encode()).hexdigest()
    dense = u1d.Dense(served, name, n)
    t0 = time.time()
    ip, col, rec = build_graph(titles, texts, dense.get, progress=lambda i, e: log(f"  {i}/{n} nodes, edges {e}"))
    rec["overlap"] = overlap(n, ip, col, kip, kcol)
    od = OUT / name
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
    log(f"{name}: W2_S {rec['undirected_neighbours']} entries (mean degree {rec['mean_degree']}), "
        f"{rec['overlap']['share_of_r_in_kb']} of them KB neighbours, {rec['overlap']['share_of_kb_in_r']} of the KB's in it; "
        f"common forms {rec['common_forms']}, largest title group {rec['largest_group']}; {rec['secs']} s")
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
    log(f"zabl_u self-test: the recipe's texts, the mention links (a duplicated name reaches both owners), the sibling "
        f"links, the isolated node and the overlap check agree on a toy KB ({rcol.size} entries)")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", nargs="?", choices=("build",))
    ap.add_argument("dataset", nargs="?")
    ap.add_argument("--host", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return _selftest()
    if a.cmd != "build" or not a.dataset:
        ap.error("build DATASET, or --selftest")
    return build_stage(a.dataset, a.host)


if __name__ == "__main__":
    sys.exit(main())
