"""Design look (untracked; not a result and not filed): multi-resolution walk types, so one model reads a graph with its
relations, with some of them unseen, and with none.

Every structural pool edge appears twice in the walk graph: once with its phrase (on a KB, its relation) token and once
with 'other' (rank R_TOP, which every rank map sends to K). A one-edge walk then has its typed type and its untyped type,
and a two-edge walk all four combinations. walk_types is set-valued, so with every phrase sent to 'other' the doubled
view gives exactly the plain view's walk types: the NR read of a model fitted here is its own untyped component, which
every training row also trained, where a plain model's 'other' token was trained on rare phrases only. Relation dropout
(--rdrop_row, --rdrop_edge) sets how often the typed walks are missing in training. The doubling adds no parameter and
reads no neighbour: it is an edge-label feature, like the phrase itself.

Modes, each the named reader run unchanged on the doubled view (the view builders are wrapped in-process):
    fit       anchor_gen3's fit: ID or MASK, NR, REV with a hold, relation dropout with --rdrop_*
    xread     anchor_gen3's cross-dataset read: XD, XD_NR
    kb        anchor_gen_kb's KB read: XD, XD_1R, XD_SHUF, XD_NR
    nn        anchor_nn's backoff reads: BASE, NR, NN@theta, NNH@theta, REV
A checkpoint does not record its view, so a model fitted here is read here, and the outputs carry 'mres' in their names.

    python outputs/mp_approx_2wiki_anchor/host/anchor_mres.py fit --dataset 2wiki --variants x4+x5+x6:A256-1 --out PATH [--rdrop_row 0.25]
    python outputs/mp_approx_2wiki_anchor/host/anchor_mres.py nn --target hotpotqa --models a.pt --out PATH
"""
import json
import os
import sys

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "4"

from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "outputs" / "mp_approx_kb_anchor" / "host"))
import anchor_gen3 as G3  # noqa: E402
import anchor_gen_kb as KB  # noqa: E402
import anchor_nn as NN  # noqa: E402

AG = G3.AG
AW = AG.AW
R_TOP = AG.R_TOP
PINS = {"anchor_gen3": "ee5f76dc7d391649e04ab2b3575d4fdd30dd62cd2a34b1b7dbd114d56f7b14a7",
        "anchor_gen_kb": "4e09d59de0a60767eb591863352389ff99e164f0e2cd841b858debd309ee7952",
        "anchor_nn": "e3bf4e1f1598ac4d62eadc58cefbcc1056d27e8d812c7e11c33c8b31ccb8678c"}
EDGE_KEYS = ("u", "v", "fam", "fwd", "bwd", "w", "rel")   # one entry per pool edge
STRUCT_KEYS = ("w2_f", "w2_b", "w1_f", "w1_b")            # one entry per structural pool edge, in edge order
STATS = {"rows": 0, "struct_edges": 0}


def double(q):
    """q with each structural edge repeated once more after the last edge, the copy's ranks R_TOP (-1 stays -1)."""
    if q.get("mres"):
        return q
    m = q["fam"] == 0
    s = np.flatnonzero(m)
    E = q["u"].size
    if s.size:
        idx = np.concatenate([np.arange(E), s])
        for k in EDGE_KEYS:
            if k in q:
                if q[k].shape[0] != E:
                    raise SystemExit(f"edge key {k}: {q[k].shape[0]} entries for {E} edges")
                q[k] = q[k][idx]
        for k in STRUCT_KEYS:
            if k in q:
                a = q[k]
                if a.shape[0] != s.size:
                    raise SystemExit(f"structural key {k}: {a.shape[0]} entries for {s.size} structural edges")
                q[k] = np.concatenate([a, np.where(a >= 0, R_TOP, -1).astype(a.dtype)])
    q["mres"] = True
    STATS["rows"] += 1
    STATS["struct_edges"] += int(s.size)
    return q


_anchor_tables = AW.anchor_tables
_kb_view = KB.kb_view


def anchor_tables_mres(Q):
    checks, vocab = _anchor_tables(Q)
    for q in Q:
        double(q)
    return {**checks, "mres_doubled": dict(STATS)}, vocab


def kb_view_mres(Q0, rank_of, one=False):
    out, st = _kb_view(Q0, rank_of, one=one)
    for q in out:
        double(q)
    return out, {**st, "mres_doubled": dict(STATS)}


def install():
    for mod, key in ((G3, "anchor_gen3"), (KB, "anchor_gen_kb"), (NN, "anchor_nn")):
        if AW.sha(Path(mod.__file__)) != PINS[key]:
            raise SystemExit(f"{mod.__file__} is not the pinned file")
    AW.anchor_tables = anchor_tables_mres
    KB.kb_view = kb_view_mres


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in ("fit", "xread", "kb", "nn"):
        raise SystemExit("mode first: fit, xread, kb or nn")
    mode, rest = argv[0], argv[1:]
    out = rest[rest.index("--out") + 1] if "--out" in rest else ""
    if "mres" not in Path(out).name:
        raise SystemExit("--out must carry 'mres' in its file name (a model fitted here is read here)")
    if mode != "fit":
        mp = rest[rest.index("--models") + 1] if "--models" in rest else ""
        if any("mres" not in Path(p).parent.name for p in mp.split(",") if p):
            raise SystemExit("--models must be models fitted here (a '*mres*_models' directory)")
    install()
    if mode == "fit":
        G3.main(rest)
    elif mode == "xread":
        G3.main(["--xread"] + rest)
    elif mode == "kb":
        KB.main(rest)
    else:
        NN.main(rest)
    o = Path(out)
    r = json.loads(o.read_text(encoding="utf-8"))
    r["mres"] = {"script_sha256": AW.sha(Path(__file__)), "mode": mode, "pins": PINS, "doubled": dict(STATS)}
    o.write_text(json.dumps(r, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
