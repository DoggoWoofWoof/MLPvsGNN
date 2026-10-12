"""Stage F1, step 1 and step 2's sets (docs/F1_ONE_INPUT_SET.md): which input families each trained model uses, read by
within-question permutation on the select carves, and the sets the declared rule names. No training.

A family is shuffled within each question by a fixed generator, seeded by (seed, the batch's first question), so every
model sees the same shuffles on a carve:
  a block      its rows (SEMB: the rows' store codes; the question's embedding stays);
  chains       each chain entry moves with its row (typed graphs);
  links        both ends of each pool edge (zsp, zgn): the graph keeps its shape on the wrong rows.
Every other input and the gold labels stay. A family's drop is R@5 unshuffled minus R@5 shuffled, the mean over the
seeds. It is USED by a model when the drop reaches MIN_DROP on at least one of the model's reads. rank holds rrf, every
model's base score: reported, never cut. links is the message passing itself: reported, never in a set.

    python outputs/mp_unified/f1_usage.py read --model zrc --fits outputs/screen/fits/scr-zrct,outputs/screen/fits/scr-zrct-hp \\
        --out outputs/f1/usage-zrc --device cuda --host
    python outputs/mp_unified/f1_usage.py read --model zgn --fits outputs/screen/fits/scr-zgn --carves metaqa/select,squad/select \\
        --max-q 16 --out outputs/f1/smoke-zgn --device cuda --host
    python outputs/mp_unified/f1_usage.py sets --mlp outputs/f1/usage-zrc.json --gnn outputs/f1/usage-zsp.json \\
        --beside outputs/f1/usage-zgn.json --out outputs/f1/sets
    python outputs/mp_unified/f1_usage.py --selftest
"""
import os
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import contextlib  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import zfeat as ZF  # noqa: E402
import zprop as ZP  # noqa: E402
import zgnn as ZG  # noqa: E402

ZC, ZM, RM, ZL = ZF.ZC, ZF.ZM, ZF.RM, ZP.ZL
S, LG, LC = ZF.S, ZF.LG, ZF.LC
log = S.log

DOC = "docs/F1_ONE_INPUT_SET.md"
CHAINS, LINKS = "chains", "links"
SEEDS = (0, 1)
MIN_DROP = 0.002
NEVER_CUT = ("rank",)
MP_ONLY = (LINKS,)
SEL_CARVES = ZF.SEL_CARVES
MODELS = {"zrc": (ZM.ZRM, ZC.ChainCarveZRC, contextlib.nullcontext),
          "zsp": (ZP.ZProp, ZL.LinkCarveBase, contextlib.nullcontext),
          "zgn": (ZG.ZGNN, ZG.GNNCarve, ZF.with_sets)}
SETTINGS = {"seeds": list(SEEDS), "min_drop": MIN_DROP, "never_cut": list(NEVER_CUT), "mp_only": list(MP_ONLY),
            "select_carves": [list(x) for x in SEL_CARVES],
            "rule": "USED by a model: a drop of at least min_drop on at least one of its reads",
            "sets": "S_both = used by both the MLP and the GNN; S_mlp, S_gnn = used by each; rank always; links never"}


# ── the shuffles ─────────────────────────────────────────────────────────────


def within_perm(cnt, seed, q0):
    """(perm, inv) over a batch's rows: every row stays in its question; new row i holds old row perm[i]."""
    cnt = np.asarray(cnt, np.int64)
    n = int(cnt.sum())
    g = np.random.default_rng([int(seed), int(q0)])
    key = np.repeat(np.arange(cnt.size, dtype=np.float64), cnt) + g.random(n)
    perm = np.argsort(key, kind="stable")
    inv = np.empty_like(perm)
    inv[perm] = np.arange(n)
    return perm, inv


def shuffled(feats, fam, perm, inv, dev):
    """feats with one family moved by the permutation (new row i holds old row perm[i]'s values); the rest shared."""
    out = dict(feats)
    if fam == CHAINS:
        ch = dict(feats[RM.CH_KEY])
        ch["row"] = inv[ch["row"]]
        out[RM.CH_KEY] = ch
    elif fam == LINKS:
        lk = dict(feats[ZL.LK_KEY])
        it = torch.from_numpy(inv).to(dev)
        lk["u"], lk["v"] = it[lk["u"]], it[lk["v"]]
        out[ZL.LK_KEY] = lk
    elif fam == "SEMB":
        qe, pr = feats["SEMB"]
        out["SEMB"] = (qe, pr[torch.from_numpy(perm).to(dev)])
    else:
        out[fam] = feats[fam][torch.from_numpy(perm).to(dev)]
    return out


def families(blocks, c):
    fams = list(blocks)
    if getattr(c, "chains", None) is not None:
        fams.append(CHAINS)
    if getattr(c, "links", None) is not None:
        fams.append(LINKS)
    return fams


@torch.no_grad()
def usage_reads(c, model, blocks, fams, seeds=SEEDS, rows_cap=LG.READ_ROWS, max_q=None):
    """{(family, seed): (R@5, FC@5, hit@1)} on carve c, ('none', -1) unshuffled; the first max_q questions if given."""
    dev = c.device
    n = c.n_np if max_q is None else c.n_np[:max_q]
    Q = n.size
    variants = [("none", -1)] + [(f, s) for f in fams for s in seeds]
    tops = {v: np.zeros(Q, np.int64) for v in variants}
    hits = {v: np.zeros(Q, np.int64) for v in variants}
    for q0, q1 in LG.batches(n, rows_cap):
        qs = np.arange(q0, q1)
        B = q1 - q0
        feats, nq, base_z, gold = c.batch(qs, blocks)
        cnt_np = c.n_np[q0:q1]
        cnt = torch.from_numpy(cnt_np).to(dev)
        seg = torch.cumsum(cnt, 0) - cnt
        keep = torch.ones((B, len(blocks)), dtype=torch.float32, device=dev)
        perms = {s: within_perm(cnt_np, s, q0) for s in seeds}
        for v in variants:
            if v[0] == "none":
                f2, bz = feats, base_z
            else:
                perm, inv = perms[v[1]]
                f2 = shuffled(feats, v[0], perm, inv, dev)
                bz = base_z[torch.from_numpy(perm).to(dev)] if v[0] == "rank" else base_z
            sc = model(f2, keep, nq, B, bz)
            t, h = LG.top_hit(sc, gold, B, seg, cnt)
            tops[v][q0:q1] = t.cpu().numpy()
            hits[v][q0:q1] = h.cpu().numpy()
    gt = c.gt_np[:Q]
    ok = gt > 0
    return {v: [float(x) for x in LG.metrics_of(tops[v], hits[v], gt)[ok].mean(0)] for v in variants}


def read_record(m, fams, seeds=SEEDS):
    base = m[("none", -1)]
    out = {}
    for f in fams:
        per = [m[(f, s)] for s in seeds]
        out[f] = {"seeds": [[round(x, 5) for x in p] for p in per],
                  "drop": [round(base[k] - float(np.mean([p[k] for p in per])), 5) for k in range(3)]}
    return out


def usage_of(reads):
    """Per family over a model's reads: the largest, mean and smallest R@5 drop, the reads it was read on, USED."""
    order = []
    for r in reads:
        order += [f for f in r["families"] if f not in order]
    out = {}
    for f in order:
        d = [r["families"][f]["drop"][0] for r in reads if f in r["families"]]
        out[f] = {"reads": len(d), "max_drop": round(max(d), 5), "mean_drop": round(float(np.mean(d)), 5),
                  "min_drop": round(min(d), 5), "used": bool(max(d) >= MIN_DROP), "never_cut": f in NEVER_CUT,
                  "mp_only": f in MP_ONLY}
    return out


def used_set(usage):
    return {f for f, r in usage.items() if f not in MP_ONLY and (r["used"] or f in NEVER_CUT)}


def sets_of(u_mlp, u_gnn):
    m, g = used_set(u_mlp), used_set(u_gnn)
    every = [f for f in list(u_mlp) + [x for x in u_gnn if x not in u_mlp] if f not in MP_ONLY]
    return {"S_both": [f for f in every if f in m and f in g], "S_mlp": [f for f in every if f in m],
            "S_gnn": [f for f in every if f in g], "mlp_only": [f for f in every if f in m and f not in g],
            "gnn_only": [f for f in every if f in g and f not in m],
            "cut_from_both": [f for f in every if f not in m and f not in g]}


# ── read ─────────────────────────────────────────────────────────────────────


def read(model_name, fits, out, device, host=False, verify=True, carves=None, max_q=None):
    t0 = time.time()
    if model_name not in MODELS:
        raise SystemExit(f"f1 read: --model is one of {sorted(MODELS)}")
    cls, carve_cls, sets = MODELS[model_name]
    if S.ARMS.get(model_name) != (cls, carve_cls):
        raise SystemExit(f"f1 read: the arm {model_name} is {S.ARMS.get(model_name)}, not {(cls, carve_cls)}")
    flags = LG.set_flags(device)
    LG.bind_device_ops()
    if host:
        import lean_host as LH
        LH.substitute()
    carves = carves or SEL_CARVES
    reads = []
    for f in fits:
        fdir = Path(f)
        sj = json.loads((fdir / "screen.json").read_text(encoding="utf-8"))
        if sj.get("arm") != model_name:
            raise SystemExit(f"f1 read: {fdir} is arm {sj.get('arm')}'s, not {model_name}'s")
        blob = torch.load(fdir / "models.pt", weights_only=False)
        with sets():
            models = [x for x in LG.load_models(blob, device, cls=cls) if x[0] == "p@swa"]
        if len(models) != 1:
            raise SystemExit(f"f1 read: {fdir} has no single p@swa")
        _n, model, blocks = models[0]
        train = [d for d, _cv in blob["train"]]
        for ds, cv in carves:
            tc = time.time()
            with sets():
                c = carve_cls(ds, cv, blob["basis"], device, LC.OUT, verify)
                if c.basis_sha256 != blob["basis_sha256"]:
                    raise SystemExit(f"{ds}/{cv}: the cache's basis is not the fit's")
                fams = families(blocks, c)
                m = usage_reads(c, model, blocks, fams, max_q=max_q)
            q = c.rows if max_q is None else min(c.rows, max_q)
            rec = {"model": model_name, "fit": fdir.name, "split": sj.get("split"), "dataset": ds, "carve": cv,
                   "zero_shot": ds not in train, "questions": q, "with_gold": int((c.gt_np[:q] > 0).sum()),
                   "none": [round(x, 5) for x in m[("none", -1)]], "families": read_record(m, fams)}
            reads.append(rec)
            top = sorted(((v["drop"][0], f) for f, v in rec["families"].items()), reverse=True)[:4]
            log(f"  {model_name} {fdir.name} {ds}={cv}{' (zero-shot)' if rec['zero_shot'] else ''}: {q} questions, "
                f"R@5 {rec['none'][0]:.4f}; largest drops {', '.join(f'{f} {d:+.4f}' for d, f in top)} "
                f"({time.time() - tc:.0f}s)")
            del c
            if str(device).startswith("cuda"):
                torch.cuda.empty_cache()
    usage = usage_of(reads)
    rec = {"stage": "F1 step 1", "declared_in": DOC, "model": model_name, "fits": [Path(f).name for f in fits],
           "usage": usage, "reads": reads, "settings": SETTINGS, "max_q": max_q,
           "carves": [list(x) for x in carves], "flags": flags, "where": "host" if host else "laptop",
           "f1_usage_sha256": LC.sha_src(__file__), "seconds": round(time.time() - t0, 1)}
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    LC.write_json(out.with_suffix(".json"), rec)
    out.with_suffix(".md").write_text(render(rec), encoding="utf-8")
    log(f"f1 read {model_name}: used {sorted(f for f, r in usage.items() if r['used'])}; unused "
        f"{sorted(f for f, r in usage.items() if not r['used'])} ({time.time() - t0:.0f}s)")
    return 0


def render(rec):
    reads = rec["reads"]
    head = [f"{r['fit']} {r['dataset']}{' zs' if r['zero_shot'] else ''}" for r in reads]
    lines = [f"# F1 step 1: what {rec['model']} uses (R@5 drop when a family is shuffled within each question)", "",
             f"Fits {', '.join(rec['fits'])}; select carves; seeds {rec['settings']['seeds']}; USED at a drop of at least "
             f"{rec['settings']['min_drop']} on any read. Declared in {rec['declared_in']}.", "",
             "| family | used | largest | mean | smallest | " + " | ".join(head) + " |",
             "| --- | --- | ---: | ---: | ---: | " + " | ".join("---:" for _ in head) + " |"]
    for f, u in rec["usage"].items():
        cells = [f"{r['families'][f]['drop'][0]:+.4f}" if f in r["families"] else "" for r in reads]
        tag = "yes" if u["used"] else "no"
        if u["never_cut"]:
            tag += " (never cut)"
        if u["mp_only"]:
            tag += " (message passing)"
        lines.append(f"| {f} | {tag} | {u['max_drop']:+.4f} | {u['mean_drop']:+.4f} | {u['min_drop']:+.4f} | "
                     + " | ".join(cells) + " |")
    lines += ["", "| read | questions | R@5 | FC@5 | hit@1 |", "| --- | ---: | ---: | ---: | ---: |"]
    for r, h in zip(reads, head):
        lines.append(f"| {h} | {r['questions']} | {r['none'][0]:.4f} | {r['none'][1]:.4f} | {r['none'][2]:.4f} |")
    return "\n".join(lines) + "\n"


def write_sets(mlp, gnn, beside, out):
    um = json.loads(Path(mlp).read_text(encoding="utf-8"))
    ug = json.loads(Path(gnn).read_text(encoding="utf-8"))
    if (um["model"], ug["model"]) != ("zrc", "zsp") or um.get("max_q") or ug.get("max_q"):
        raise SystemExit("f1 sets: the MLP's record is zrc's and the GNN's zsp's, both over whole carves")
    s = sets_of(um["usage"], ug["usage"])
    side = {}
    for p in beside:
        ub = json.loads(Path(p).read_text(encoding="utf-8"))
        side[ub["model"]] = {"used": sorted(used_set(ub["usage"])),
                             "unused": sorted(f for f, r in ub["usage"].items() if not r["used"])}
    rec = {"stage": "F1 step 2's sets", "declared_in": DOC, "mlp": "zrc", "gnn": "zsp", **s, "beside": side,
           "records": {Path(p).name: LC.sha_file(p) for p in [mlp, gnn] + list(beside)}, "settings": SETTINGS,
           "f1_usage_sha256": LC.sha_src(__file__)}
    out = Path(out)
    LC.write_json(out.with_suffix(".json"), rec)
    lines = ["# F1 step 2: the sets", "", f"- **S_both** (the controlled pair): {', '.join(s['S_both'])}",
             f"- **S_mlp** (zrc's ceiling): {', '.join(s['S_mlp'])}", f"- **S_gnn** (zsp's ceiling): {', '.join(s['S_gnn'])}",
             f"- used by the MLP only: {', '.join(s['mlp_only']) or 'none'}",
             f"- used by the GNN only: {', '.join(s['gnn_only']) or 'none'}",
             f"- cut from both: {', '.join(s['cut_from_both']) or 'none'}"]
    for k, v in side.items():
        lines.append(f"- beside, {k}: uses {', '.join(v['used'])}; does not use {', '.join(v['unused']) or 'none'}")
    out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"f1 sets: S_both {s['S_both']}; MLP only {s['mlp_only']}; GNN only {s['gnn_only']}; cut {s['cut_from_both']}")
    return 0


# ── selftest ─────────────────────────────────────────────────────────────────


def selftest():
    t0 = time.time()
    assert ZF.MIN_DROP == MIN_DROP
    for k, (cls, cc, _s) in MODELS.items():
        assert S.ARMS[k] == (cls, cc), k
    # the permutation keeps every row in its question, is a bijection, and is fixed by (seed, q0)
    cnt = np.array([3, 1, 5, 0, 4], np.int64)
    nq = np.repeat(np.arange(cnt.size), cnt)
    perm, inv = within_perm(cnt, 0, 7)
    assert np.array_equal(nq[perm], nq) and np.array_equal(perm[inv], np.arange(cnt.sum()))
    assert np.array_equal(within_perm(cnt, 0, 7)[0], perm) and not np.array_equal(within_perm(cnt, 1, 7)[0], perm)
    N, dev = int(cnt.sum()), torch.device("cpu")
    pt = torch.from_numpy(perm)
    # a block and SEMB: new row i holds old row perm[i]
    x = torch.randn(N, 4)
    qe, pr = torch.randn(cnt.size, 6), torch.randn(N, 7)
    f = {"rank": x, "SEMB": (qe, pr)}
    s1 = shuffled(f, "rank", perm, inv, dev)
    assert torch.equal(s1["rank"], x[pt]) and s1["SEMB"] is f["SEMB"]
    s2 = shuffled(f, "SEMB", perm, inv, dev)
    assert s2["SEMB"][0] is qe and torch.equal(s2["SEMB"][1], pr[pt]) and s2["rank"] is x
    # chains: a row's summed mass after the move is the mass of the row it now stands for
    rows = np.array([0, 0, 2, 4, 5, 8, 8, 9, 12], np.int64)
    mass = np.arange(1, rows.size + 1, dtype=np.float64)
    f[RM.CH_KEY] = {"row": rows, "m": mass}
    s3 = shuffled(f, CHAINS, perm, inv, dev)
    old = np.bincount(rows, mass, N)
    new = np.bincount(s3[RM.CH_KEY]["row"], mass, N)
    assert np.allclose(new, old[perm]) and f[RM.CH_KEY]["row"] is rows
    # links: moving both ends by inv relabels the graph; with the scores moved by perm too, zsp's propagation inputs
    # are the old ones moved by perm (equivariance), so links alone moved means the right shape on the wrong rows
    g = torch.Generator().manual_seed(3)
    nq_t = torch.from_numpy(nq)
    u = torch.from_numpy(np.array([0, 1, 2, 4, 5, 6, 7, 9, 10, 11, 12, 9], np.int64))
    v = torch.from_numpy(np.array([1, 2, 0, 5, 6, 7, 8, 10, 11, 12, 9, 11], np.int64))
    assert np.array_equal(nq[u.numpy()], nq[v.numpy()])                       # edges stay inside a question
    fam = torch.randint(0, ZP.FAMS, (u.numel(),), generator=g)
    sc = torch.randn(N, generator=g)
    lk = {"u": u, "v": v, "f": fam}
    f[ZL.LK_KEY] = lk
    s4 = shuffled(f, LINKS, perm, inv, dev)
    a = ZP.prop_inputs(sc, lk, nq_t, cnt.size)
    b = ZP.prop_inputs(sc[pt], s4[ZL.LK_KEY], nq_t, cnt.size)
    assert torch.allclose(a[pt], b, atol=1e-6), (a[pt] - b).abs().max()
    assert f[ZL.LK_KEY] is lk and torch.equal(s4[ZL.LK_KEY]["f"], fam)
    # usage and the sets: USED at MIN_DROP on any read; rank never cut; links never in a set
    def rd(drops):
        return {"families": {k: {"drop": [d, 0.0, 0.0]} for k, d in drops.items()}}
    um = usage_of([rd({"rank": 0.2, "WALK": 0.0005, "SEED": 0.003, "dense_cos": 0.0}),
                   rd({"rank": 0.3, "WALK": 0.0019, "SEED": -0.001, "dense_cos": 0.0, "chains": 0.004})])
    assert um["SEED"]["used"] and not um["WALK"]["used"] and um["chains"]["reads"] == 1 and um["chains"]["used"]
    ug = usage_of([rd({"rank": 0.1, "WALK": 0.01, "SEED": 0.0, "dense_cos": 0.0, "links": 0.05, "chains": 0.0})])
    st = sets_of(um, ug)
    assert st["S_both"] == ["rank"] and st["S_mlp"] == ["rank", "SEED", "chains"] and st["S_gnn"] == ["rank", "WALK"]
    assert st["mlp_only"] == ["SEED", "chains"] and st["gnn_only"] == ["WALK"] and st["cut_from_both"] == ["dense_cos"]
    assert "links" not in sum((v for v in st.values()), [])
    um2 = usage_of([rd({"rank": 0.0001})])
    assert not um2["rank"]["used"] and "rank" in used_set(um2)
    print(f"selftest: shuffles keep questions and move blocks, store codes, chain rows and edge ends as declared (zsp's "
          f"propagation inputs move with them); USED at {MIN_DROP}; rank always kept, links never in a set "
          f"({time.time() - t0:.1f}s)")
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--selftest" in argv:
        return selftest()
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("read", "sets"))
    ap.add_argument("--model")
    ap.add_argument("--fits")
    ap.add_argument("--carves", default="")
    ap.add_argument("--max-q", type=int)
    ap.add_argument("--mlp")
    ap.add_argument("--gnn")
    ap.add_argument("--beside", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--host", action="store_true")
    ap.add_argument("--no-verify", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "read":
        carves = [tuple(x.split("/")) for x in a.carves.split(",") if x] or None
        return read(a.model, a.fits.split(","), a.out, a.device, a.host, not a.no_verify, carves, a.max_q)
    return write_sets(a.mlp, a.gnn, [x for x in a.beside.split(",") if x], a.out)


if __name__ == "__main__":
    sys.exit(main())
