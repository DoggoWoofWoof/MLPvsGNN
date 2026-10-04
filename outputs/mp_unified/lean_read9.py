"""Design look (untracked; not a result and not filed): saved lean MLPs read with blocks switched off. Nothing is fitted.

Why. On squad x1 every lean read with SEMB in it is below rrf alone, while pick without SEMB (pns, a separate fit)
reaches rrf alone; on 2wiki pns reads 0.93 to 0.94 of the GNN's R@5 against pick's 0.99 (lean_mlp6, l5-2w-*-zs). So
SEMB carries the in-domain gain and costs the transfer. lean_mlp5's 'drop' and 'regd' configs train with lean_mlp.fit's
block dropout: each batch row keeps each block (rank always) with a probability drawn per row from U(0.15, 1), and the
model reads a keep indicator per block. A model fitted so has scored with every subset of its blocks, SEMB-free ones
included, so one such model may carry both paths: SEMB on where it helps, off where it does not. This look reads both
paths of each saved model on the same rows. If the off path reaches rrf alone off-domain and the on path keeps the
in-domain gain, what is left is a label-free switch between them.

For each saved member and ensemble (lean_mlp5's file; --only picks names by regex, an ensemble needing only its own name
to match) and each --keeps variant (all: every block on, the keep indicator 1 as every earlier read; -B1+B2: those
blocks off, their raw and z-scored columns 0 and their keep indicator 0, as block dropout presents an absent block in
training), on each --read carve: the ID read against the twin and the GNN, the paired difference against rrf alone
(minus_rrf), and for an off variant the paired difference against the same model with every block on (minus_on).
rank is never switched off (block dropout always keeps it). An off variant naming a block a model does not have is not
read for that model. lean_mlp8's read path (read8) is imported and called unchanged, with its seg_zscore8 installed
(lean_mlp's forward bit for bit), so per-row metrics go to <out>.rows.npz under lean_mlp8's keys (rrf@carve,
twin0@carve, gnn0@carve, NAME@carve) and pair row by row with lean_mlp8's reads of the same carve. read_costs_ms is the
whole set's compile estimate for every variant (an off block's columns would not need computing; not credited here).

Non-MP: the lean MLPs read here pass no message.

    python outputs/mp_unified/lean_host9.py lean_read9 --load-models outputs/mp_unified/lean/l5-2w-pick_models.pt \\
        --keeps "all;-SEMB" --read 2wiki=x1,hotpotqa=x1,squad=x1 --threads 2 --out outputs/mp_unified/lean/r9-2w-pick.json
    python outputs/mp_unified/lean_read9.py --selftest
"""
import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import lean_mlp8 as L8  # noqa: E402

L7, L5, L3, L2, LM = L8.L7, L8.L5, L8.L3, L8.L2, L8.LM
log = L8.log
SEP = "~"


def parse_keeps(spec):
    """'all;-SEMB;-SEMB+WALK' -> [(label, frozenset of blocks off)]; 'all' is the empty set."""
    out = []
    for part in [p.strip() for p in spec.split(";") if p.strip()]:
        if part == "all":
            off = frozenset()
        elif part.startswith("-") and len(part) > 1:
            blocks = [b for b in part[1:].split("+") if b]
            if not blocks or len(set(blocks)) != len(blocks):
                raise SystemExit(f"keeps {part!r}: -BLOCK+BLOCK..., each once")
            if any(b in LM.NEVER for b in blocks):
                raise SystemExit(f"keeps {part!r}: {LM.NEVER} is never switched off")
            off = frozenset(blocks)
        else:
            raise SystemExit(f"keeps {part!r}: 'all' or -BLOCK+BLOCK...")
        out.append((part, off))
    if not out or len({o for _l, o in out}) != len(out):
        raise SystemExit("keeps: at least one variant, no two alike")
    return out


def build_models(blob, only):
    """name -> (model, blocks) for the members and ensembles --only selects (an ensemble with every member it needs)."""
    pat = re.compile(only) if only else None
    members = {name: L5.build(d["blocks"], d["widths"], d["hidden"], d["state"]) for name, d in blob["models"].items()}
    for m in members.values():
        m.ctx_mode = "none"                      # read8 asks every model that is not an ensemble for its context mode
    out = {}
    for name, d in blob["models"].items():
        if pat is None or pat.search(name):
            out[name] = (members[name], list(d["blocks"]))
    for name, mem in blob.get("ensembles", {}).items():
        if pat is None or pat.search(name):
            out[name] = (L5.Ens([members[x] for x in mem]), list(blob["models"][mem[0]]["blocks"]))
    if not out:
        raise SystemExit(f"--only {only!r} selects no saved model")
    return out


def variants_of(models, keeps):
    """(read8's models: name -> (model, blocks, keep map), refs: name -> {'on': name with every block on})."""
    every = {b for _m, bl in models.values() for b in bl}
    for label, off in keeps:
        if off - every:
            raise SystemExit(f"keeps {label!r}: no saved model has {sorted(off - every)}")
    has_all = any(not off for _l, off in keeps)
    out, refs = {}, {}
    for name, (m, bl) in models.items():
        for label, off in keeps:
            if off - set(bl):
                continue
            nm = name if not off else f"{name}{SEP}{label}"
            out[nm] = (m, bl, {b: (0.0 if b in off else 1.0) for b in bl})
            refs[nm] = {"on": name} if off and has_all else {}
    return out, refs


def run_read9(a, blob, carve, head):
    keeps = parse_keeps(a.keeps)
    models, refs = variants_of(build_models(blob, a.only), keeps)
    out = {**head, "mode": "read_only", "keeps": [[lbl, sorted(off)] for lbl, off in keeps], "only": a.only,
           "fit_args": blob.get("args"), "configs": blob.get("configs"), "sets": blob.get("sets"),
           "ensembles": blob.get("ensembles"), "read_names": sorted(models)}
    log(f"{len(models)} reads per carve: {sorted(models)}")
    L8.read8(a, models, out, carve, refs)
    return out


def shas():
    s = {f"{n}_sha256": hashlib.sha256((HERE / f"{n}.py").read_bytes()).hexdigest()
         for n in ("lean_mlp", "lean_mlp2", "lean_mlp3", "lean_mlp5", "lean_mlp7", "lean_mlp7g", "lean_mlp8")}
    s["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--load-models", help="a lean_mlp5 models file (members and ensembles)")
    ap.add_argument("--only", default="", help="a regex on saved names (default: every member and ensemble)")
    ap.add_argument("--keeps", default="all;-SEMB", help="'all' and/or -BLOCK+BLOCK..., ';'-separated")
    ap.add_argument("--read", default="2wiki=x1")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not a.load_models:
        raise SystemExit("--load-models is required")
    torch.set_num_threads(a.threads)
    t0 = time.time()
    L8.install()
    LM.LeanMLP = L3.LeanMLP3
    blob = torch.load(a.load_models, weights_only=False)
    if blob.get("store") != "pca256":
        raise SystemExit(f"store {blob.get('store')!r}: this look reads pca256 files")
    L3.LeanMLP3.dim = L3.STORE_DIM
    names = sorted({ds for ds, _cv in LM.parse_sets(a.read)})
    nodes, freeze = L3.open_nodes(names)
    store = L3.Store(blob["basis"])

    def carve(ds, cv):
        return L3.Carve3(ds, cv, store, nodes.get(ds), a.limit)

    head = {"look": "lean_read9", "store": "pca256", "basis": {k: v for k, v in blob["basis"].items() if k not in ("m", "V", "w")},
            "freeze": freeze, "args": vars(a), "loaded": a.load_models,
            "loaded_sha256": hashlib.sha256(Path(a.load_models).read_bytes()).hexdigest(), **shas()}
    out = run_read9(a, blob, carve, head)
    out["seconds"] = time.time() - t0
    L2.write(a, out)
    log(f"done in {time.time() - t0:.1f}s")
    return 0


# ── selftest ─────────────────────────────────────────────────────────────────


def selftest():
    import tempfile
    from types import SimpleNamespace
    torch.set_num_threads(1)
    old = (LM.seg_zscore, LM.batch_of, LM.LeanMLP, L3.LeanMLP3.dim)
    L3.LeanMLP3.dim = LM.PROJ_DIM
    try:
        L8.install()
        LM.LeanMLP = L3.LeanMLP3
        blocks = ["rank", "WALK", "SEMB"]
        tr, se = [L8.ToyCarve8(60, 1), L8.ToyCarve8(40, 4)], [L8.ToyCarve8(30, 2)]
        # 1. a toy lean_mlp5 file: two block-dropout seeds and one plain seed, their ensembles, saved as lean_mlp5 saves
        cfgs = L5.parse_configs("drop=1e-2:1e-4:0.1:3:1:drop;base=1e-2:1e-4:0.1:3:1")
        saved, ens = {}, {}
        for cname, seeds in (("drop", (0, 1)), ("base", (0,))):
            for s in seeds:
                f = L5.fit5(tr, se, blocks, cfgs[cname], s, 16)
                for kind in ("best", "swa"):
                    saved[f"a/{cname}/s{s}@{kind}"] = {"state": f[kind], "blocks": blocks, "widths": f["widths"], "hidden": 16,
                                                     "keep": {b: 1.0 for b in blocks}}
        for kind in ("best", "swa"):
            ens[f"a/drop/ens@{kind}"] = [f"a/drop/s0@{kind}", f"a/drop/s1@{kind}"]
        blob = {"models": saved, "ensembles": ens, "store": "toy", "args": {"x": 1}, "configs": cfgs, "sets": {"a": blocks}}
        # 2. parsing: rank is never off, unknown or doubled blocks and repeated variants are refused
        assert parse_keeps("all;-SEMB;-SEMB+WALK") == [("all", frozenset()), ("-SEMB", frozenset({"SEMB"})),
                                                       ("-SEMB+WALK", frozenset({"SEMB", "WALK"}))]
        for bad in ("-rank", "-SEMB+SEMB", "all;all", "", "SEMB", "-"):
            try:
                parse_keeps(bad)
                raise AssertionError(bad)
            except SystemExit:
                pass
        try:
            variants_of(build_models(blob, ""), parse_keeps("all;-NOPE"))
            raise AssertionError("unknown block")
        except SystemExit:
            pass
        # 3. --only: a member regex, and an ensemble built from members the regex does not pick
        mo = build_models(blob, r"/drop/ens@swa$")
        assert list(mo) == ["a/drop/ens@swa"] and isinstance(mo["a/drop/ens@swa"][0], L5.Ens)
        assert len(mo["a/drop/ens@swa"][0].members) == 2
        assert set(build_models(blob, "")) == set(saved) | set(ens)
        # 4. the reads: 'all' is the model as saved (keep 1, every earlier read's scores bit for bit), '-SEMB' is the
        #    keep map with SEMB 0 bit for bit, minus_on pairs it with 'all', and switching SEMB off changes the scores
        c9 = L8.ToyCarve8(25, 9, one=True)

        def toy(ds, cv):
            return L8.ToyCarve8(25, 9, one=True)

        with tempfile.TemporaryDirectory() as td:
            a = SimpleNamespace(keeps="all;-SEMB;-WALK", only="", read="toy=t9", out=str(Path(td) / "r.json"))
            o = run_read9(a, blob, toy, {"look": "lean_read9"})
            r = o["reads"]
            names = set(saved) | set(ens)
            want = names | {f"{n}{SEP}-SEMB" for n in names} | {f"{n}{SEP}-WALK" for n in names}
            assert set(o["read_names"]) == want, sorted(set(o["read_names"]) ^ want)
            for n in names:
                assert "minus_on" not in r[f"{n}@toy=t9"] and "minus_rrf" in r[f"{n}@toy=t9"]
                assert "minus_on" in r[f"{n}{SEP}-SEMB@toy=t9"] and "minus_rrf" in r[f"{n}{SEP}-SEMB@toy=t9"]
            z = np.load(Path(td) / "r.rows.npz")
            ok = c9.gold_total > 0
            moved = 0
            for n in ("a/drop/s0@swa", "a/base/s0@best", "a/drop/ens@best"):
                m = build_models(blob, "^" + re.escape(n) + "$")[n][0]
                s_all = LM.scores_of(m, c9, blocks, {b: 1.0 for b in blocks})
                s_off = LM.scores_of(m, c9, blocks, {"rank": 1.0, "WALK": 1.0, "SEMB": 0.0})
                assert np.array_equal(z[f"{n}@toy=t9"], LM.row_metrics(s_all, c9.gold, c9.off, c9.gold_total)[ok].astype(np.float32))
                assert np.array_equal(z[f"{n}{SEP}-SEMB@toy=t9"],
                                      LM.row_metrics(s_off, c9.gold, c9.off, c9.gold_total)[ok].astype(np.float32))
                moved += int(not np.array_equal(s_all, s_off))
            assert moved == 3, "switching SEMB off must change every model's scores"
            assert "rrf@toy=t9" in z.files and z["rrf@toy=t9"].shape[1] == 3
            z.close()
            # 5. the same read again gives the same numbers (the bootstrap is seeded in read8)
            o2 = run_read9(SimpleNamespace(**{**vars(a), "out": str(Path(td) / "r2.json")}), blob, toy, {"look": "lean_read9"})
            assert json.dumps(o["reads"], sort_keys=True) == json.dumps(o2["reads"], sort_keys=True)
            # 6. a keep variant that only some models can take is read for those only; with no 'all' variant an off
            #    variant has nothing to pair with
            two = {"x": (LM.LeanMLP(["rank", "WALK"], {"rank": 3, "WALK": 4}, 16), ["rank", "WALK"]),
                   "y": build_models(blob, "^a/base/s0@best$")["a/base/s0@best"]}
            mods, refs = variants_of(two, parse_keeps("all;-SEMB"))
            assert set(mods) == {"x", "y", f"y{SEP}-SEMB"} and refs[f"y{SEP}-SEMB"] == {"on": "y"} and refs["x"] == {}
            assert mods[f"y{SEP}-SEMB"][2] == {"rank": 1.0, "WALK": 1.0, "SEMB": 0.0}
            mods, refs = variants_of(two, parse_keeps("-SEMB"))
            assert set(mods) == {f"y{SEP}-SEMB"} and refs[f"y{SEP}-SEMB"] == {}
    finally:
        LM.seg_zscore, LM.batch_of, LM.LeanMLP, L3.LeanMLP3.dim = old
    print("selftest: keep variants parse (rank never off, unknown and repeated refused); --only builds ensembles from "
          "unpicked members; 'all' reads the saved model's scores and '-SEMB' the keep map with SEMB 0, bit for bit; "
          "minus_on pairs each off variant with its model; rows file keys are lean_mlp8's; reads repeat exactly. "
          "all checks passed")


if __name__ == "__main__":
    main()
