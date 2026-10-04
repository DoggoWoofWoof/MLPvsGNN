"""Design look (untracked; not a result and not filed): does a two-edge relation model use the ORDER of its two relations?

A two-edge walk is coded (b, t1, t2): from a seed of bucket b through an edge of token t1, then one of token t2
(l16_look_analyze.walk_types: code = (b * (nt + 1) + t1 + 1) * (nt + 1) + t2 + 1). A commutative operator scores
(t1, t2) and (t2, t1) alike (gtrans: s0 + h1 + h2; grope: rotations in fixed planes, R(a) R(b) = R(a + b)); an
order-aware one can tell them apart (the free two-edge vectors and glin's position codes linearly; gropet, ghouse,
gaffine, ggate and house through a non-commutative composition; T0 through its direction positions). This look reads each
model on its own dataset's B rows (x1[1::2], as every anchor look) as fitted and with every two-edge walk's two tokens
swapped, the reach set kept:
    BASE / SWAP           the fit's own map (ID; MASK with a hold)
    NR / NR_SWAP          every phrase 'other': only the order of the two edges' directions is left
    REV / REV_SWAP        held text and gen models: the held fold by its own text codes
    SWAP - BASE           how much of the read rests on the order the model learned; for a commutative operator it is 0
                          up to float rounding
The swap is a bijection on the codes ((b, t1, t2) -> (b, t2, t1), one-edge codes fixed), so no two walks merge.
'order rows' are the B rows with a two-edge walk whose tokens differ, the only rows a swap can change. In-domain only.

    python outputs/mp_approx_2wiki_anchor/host/anchor_order.py --target 2wiki --models a.pt,b.pt --out PATH
"""
import os
import sys

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "4"

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import anchor_nn as NN  # noqa: E402

G3, AG, KB = NN.G3, NN.AG, NN.KB
AW6, A16, AW, AW3 = AG.AW6, AG.A16, AG.AW, AG.AW3
R_TOP = AG.R_TOP
ANCHOR_NN_SHA = "e3bf4e1f1598ac4d62eadc58cefbcc1056d27e8d812c7e11c33c8b31ccb8678c"
log = AG.log


def swap_code(c, tb):
    t2p = c % tb
    if not t2p:
        return int(c)
    r = c // tb
    return int(((r // tb) * tb + t2p) * tb + r % tb)


def swap_types(TY, nt, rows):
    tb = nt + 1
    out = list(TY)
    for i in rows:
        out[i] = {swap_code(c, tb): R for c, R in TY[i].items()}
        if len(out[i]) != len(TY[i]):
            raise SystemExit("the swap merged two walk codes")
    return out


def order_rows(TY, nt, rows):
    """The rows with a two-edge walk whose two tokens differ."""
    tb = nt + 1
    keep = []
    for i in rows:
        for c in TY[i]:
            t2p = c % tb
            if t2p and (c // tb) % tb != t2p:
                keep.append(i)
                break
    return keep


def run(a):
    out_path = Path(a.out)
    torch.set_num_threads(2)
    t0 = time.time()
    paths = [Path(p) for p in a.models.split(",")]
    cks = [torch.load(p, map_location="cpu", weights_only=False) for p in paths]
    if any(ck["dataset"] != a.target for ck in cks):
        raise SystemExit("anchor_order reads in-domain: every model fitted on the target")
    if any(ck["spec"]["max_len"] != 2 for ck in cks):
        raise SystemExit("anchor_order reads two-edge models only")
    Q, part, vocab_t, _phi_t, _own, info = NN.load_target(a.target, 2, False, t0)
    B_rows = part["x1"][1::2]
    RB = AG.Reader(Q, B_rows, 20261002)
    z_of = [A16.zscore(q["score"][:, 0]) for q in Q]
    res = {"look": "anchor_order", "target": a.target, "script_sha256": AW.sha(Path(__file__)),
           "pins": {"anchor_nn": ANCHOR_NN_SHA, "anchor_gen3": NN.ANCHOR_GEN3_SHA, "anchor_gen": G3.ANCHOR_GEN_SHA,
                    "anchor_gen2": G3.ANCHOR_GEN2_SHA, "anchor_gen_kb": NN.ANCHOR_GEN_KB_SHA},
           "target_info": info, "B": RB.base(), "models": {}}
    per_row = {}
    pos = {i: j for j, i in enumerate(B_rows)}
    for jm, (p, ck) in enumerate(zip(paths, cks)):
        t1 = time.time()
        sp, K = ck["spec"], ck["spec"]["K"]
        t0m = sp["tok"] == "T0"
        phi_tr, _kind = (None, None) if t0m else NN.train_table(ck)
        model, _ck = NN.load_any(p, phi_tr)
        ent = {"from": ck["dataset"], "name": ck.get("name"), "seed": ck.get("seed"), "hold": ck.get("hold"), "rdrop": ck.get("rdrop"),
               "spec": sp, "margin": ck["margin"], "model_file": str(p), "reads": {}}
        maps = {}
        if t0m:
            maps["BASE"] = None
        else:
            held = NN.held_of(ck)
            maps["BASE"] = NN.exact_map(K, vocab_t, ck["vocab_w2"], held)
            maps["NR"] = np.full(R_TOP + 1, K, dtype=np.int64)
            if sp["family"] != "free" and held.any():
                maps["REV"] = AG.identity_map(K)
        orows = None
        reads = {}
        for nm, rm in maps.items():
            _nt, TY = AG.types_for(Q, sp, rm, B_rows)
            if nm == "BASE":
                orows = order_rows(TY, ck["nt"], B_rows)
            TS = swap_types(TY, ck["nt"], B_rows)
            if nm == "BASE":
                back = swap_types(TS, ck["nt"], B_rows)
                if any(back[i].keys() != TY[i].keys() for i in B_rows[:200]):
                    raise SystemExit("the swap is not an involution")
            for tag, T in ((nm, TY), (nm + "_SWAP", TS)):
                reads[tag] = AG.read(model, Q, T, ck["nt"], B_rows, z_of, ck["margin"])
                per_row[f"{jm}_B_{tag}"] = reads[tag]
        ox = np.asarray([pos[i] for i in orows], dtype=np.int64)
        ent["order_rows"] = int(ox.size)
        for tag, m in reads.items():
            e = RB.record(m, by_type=False)
            if tag.endswith("_SWAP"):
                ref = reads[tag[:-5]]
                e["minus_unswapped"] = AW3.boot_pair(m - ref, RB.W)
                e["identical_to_unswapped"] = bool(np.array_equal(m, ref))
                e["rows_changed"] = int((m != ref).any(1).sum())
                if ox.size:
                    e["order_rows_minus_unswapped"] = AW3.boot_pair((m - ref)[ox], RB.W[:, ox])
            if ox.size:
                e["order_rows"] = RB.subset(m, orows)
            ent["reads"][tag] = e
        ent["seconds"] = round(time.time() - t1, 1)
        res["models"][f"{p.parent.name}/{p.name}"] = ent
        log(f"{p.parent.name}/{p.name} ({sp['model']}): order rows {ox.size}; " + ", ".join(
            f"{k} {v['rho (R@5, FC@5, hit@1)']}" for k, v in ent["reads"].items()))
        out_path.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
        np.savez(out_path.with_suffix(".npz"), **per_row, B_rows=np.asarray(B_rows))
    res["seconds"] = round(time.time() - t0, 1)
    out_path.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    log(f"done in {res['seconds']}s")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, choices=AG.DATASETS)
    ap.add_argument("--models", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if "order" not in Path(a.out).name:
        raise SystemExit("--out must carry 'order' in its file name")
    if AW.sha(Path(NN.__file__)) != ANCHOR_NN_SHA:
        raise SystemExit("anchor_nn.py is not the pinned file")
    if AW.sha(Path(G3.__file__)) != NN.ANCHOR_GEN3_SHA:
        raise SystemExit("anchor_gen3.py is not the pinned file")
    if AW.sha(Path(AG.__file__)) != G3.ANCHOR_GEN_SHA or AW.sha(Path(G3.G2N.__file__)) != G3.ANCHOR_GEN2_SHA:
        raise SystemExit("anchor_gen.py or anchor_gen2.py is not the pinned file")
    if AW.sha(Path(KB.__file__)) != NN.ANCHOR_GEN_KB_SHA:
        raise SystemExit("anchor_gen_kb.py is not the pinned file")
    AG.check_pins()
    run(a)


if __name__ == "__main__":
    main()
