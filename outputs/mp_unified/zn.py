"""zn: the lean MLP without message passing (docs/ZN_LEAN_NO_MP.md, declared before any number).

The user, 12 October: the MLP has no message passing; the lean pair may read different inputs, each what its model
needs; the MLP is to be at least 2-3x faster, cold, than the fastest GNN. zrc reads five families computed over the
pool graph's edges from the seeds (topo_STRUCT, depth_STRUCT, WALK, DISTS, WALKF) and, on the KB graphs, the chain
entries (typed relation paths from the seeds). zn reads none of them, only what is computed without an edge:

    rank       dense_rr, splade_rr, rrf, agreement, is_seed       the retrieval lists
    dense_cos  cos(e_v, q)                                         the row's own embedding
    SEMB       (q U) * (p_v V)                                     the question's embedding and the row's stored projection
    SEED       p_v against the seed rows' projections              mean, max, max over bucket 0 (seeds named by retrieval)

Its model is zret's (lean_screen3.ZRet: every block's within-pool z-score and rrf's base z-score, against the pool's
retrieved rows) over the base carve: no chain entries and no chain match. Its training is zrc's (lean_screen2's train,
lean_gpu's config, hidden, seed and flags, the same carves and caches). A row reads no other row's input or score but the
question's seed rows and its pool's statistics of its own columns.

    python outputs/mp_unified/zn.py --selftest
    python outputs/mp_unified/zn.py smoke --device cuda --host
    python outputs/mp_unified/zn.py train --split L-musique --name scr-zn --arm zn --device cuda --host
    python outputs/mp_unified/zn.py read --name scr-zn --device cuda --host
    python outputs/mp_unified/zn.py compare --split L-musique --new outputs/screen/fits/scr-zn \\
        --out outputs/zn/compare-L-musique
    python outputs/mp_unified/zn.py summary --out outputs/zn/summary

compare reads zn's fit against zrc's and zsp's fits of the split (lean_screen's compare, zrc's first); summary puts both
splits side by side with the share of zsp's gain over rrf that zn keeps, (zn - rrf) / (zsp - rrf), per read (not where
zsp - rrf < MIN_GAIN). No adoption call: the user's constraints decide the inputs; this measures what they cost.
"""
import os
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import contextlib  # noqa: E402
import json  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import torch  # noqa: E402

import lean_screen3 as S3  # noqa: E402

S2, S = S3.S2, S3.S
LG, LC, LM = S.LG, S.LC, S.LM
log = S.log

ARM = "zn"
NMP = ["rank", "dense_cos", "SEMB", "SEED"]
EDGE = ("topo_STRUCT", "depth_STRUCT", "WALK", "DISTS", "WALKF")       # pick's families computed over edges
MIN_GAIN = 0.01
OUT = Path("outputs") / "zn"                          # relative: the items run at the repo root, and records keep
FITS = Path("outputs") / "screen" / "fits"            # no host path
BASES = {"L-musique": {"zn": "scr-zn", "zrc": "scr-zrct", "zsp": "scr-zspg"},
         "L-hotpotqa": {"zn": "scr-zn-hp", "zrc": "scr-zrct-hp", "zsp": "scr-zspg-hp"}}
DOC = "docs/ZN_LEAN_NO_MP.md"

S.ARMS.update({ARM: (S3.ZRet, S.ARMS["base"][1])})


@contextlib.contextmanager
def nmp_sets():
    """lean_gpu's pick set narrowed to zn's blocks (train_cmd's variant p and read_carve's batch); restored on exit."""
    saved = list(LG.SETS["pick"])
    try:
        LG.SETS["pick"] = [b for b in saved if b in NMP]
        yield
    finally:
        LG.SETS["pick"] = saved


# ── train and read ───────────────────────────────────────────────────────────


def train(argv, split):
    """lean_screen2's train for the arm zn, on zn's blocks; the record stamped with this file and its blocks."""
    from relcols import arm_of
    arm = arm_of(argv)
    if arm != ARM:
        raise SystemExit(f"zn: train takes --arm {ARM}, not {arm}")
    with nmp_sets():
        rc = S2.train(argv, split)
    name, out_root = S2.where(argv)
    sj = out_root / name / "screen.json"
    if sj.exists():
        rec = json.loads(sj.read_text(encoding="utf-8"))
        rec["zn_sha256"] = LC.sha_src(__file__)
        rec["lean_screen3_sha256"] = LC.sha_src(S3.__file__)
        rec["zn"] = {"blocks": NMP, "model": "lean_screen3.ZRet", "carve": "lean_screen's base carve (no chain entries)",
                     "declared_in": DOC}
        LC.write_json(sj, rec)
    return rc


def read(argv):
    name, out_root = S2.where(argv)
    arm = json.loads((out_root / name / "screen.json").read_text(encoding="utf-8"))["arm"]
    if arm != ARM:
        raise SystemExit(f"zn: {name} was trained as {arm}, not {ARM}")
    with nmp_sets():
        return S2.main(["read"] + argv)


# ── compare and summary ──────────────────────────────────────────────────────


def compare(split, new, out, fits=FITS):
    """lean_screen's compare of zn's fit against zrc's and zsp's fits of the split (zrc's first)."""
    b = BASES[split]
    bases = ",".join(str(Path(fits) / b[k]) for k in ("zrc", "zsp"))
    return S.main(["compare", "--new", str(new), "--base", bases, "--out", str(out)])


def share(new, rrf, ref):
    g = ref - rrf
    return None if g < MIN_GAIN else round((new - rrf) / g, 3)


def summary(compares, out):
    """Both splits' reads: rrf, zn, zrc, zsp R@5, zn's differences with their intervals, and the share of each model's
    gain over rrf that zn keeps."""
    rows = []
    for split, path in compares.items():
        rec = json.loads(Path(path).read_text(encoding="utf-8"))
        by = rec["by_base"]
        keys = list(by)
        if len(keys) != 2 or not keys[0].replace("\\", "/").endswith(BASES[split]["zrc"]) \
                or not keys[1].replace("\\", "/").endswith(BASES[split]["zsp"]):
            raise SystemExit(f"{path}: the bases are {keys}, not zrc's and zsp's fits of {split}")
        zr = {r["dataset"]: r for r in by[keys[0]]["rows"]}
        zs = {r["dataset"]: r for r in by[keys[1]]["rows"]}
        for ds in LG.EVAL_ORDER:
            if ds not in zr:
                continue
            a, c = zr[ds], zs[ds]
            if a["new"] != c["new"] or a["rrf"] != c["rrf"] or a["questions"] != c["questions"]:
                raise SystemExit(f"{split} {ds}: the two comparisons read different zn or rrf values")
            rrf, zn = a["rrf"][0], a["new"][0]
            rows.append({"split": split, "dataset": ds, "read": a["read"], "questions": a["questions"], "rrf": rrf,
                         "zn": zn, "zrc": a["base"][0], "zsp": c["base"][0],
                         "zn_minus_zrc": [a["delta"][0], a["lo"][0], a["hi"][0]],
                         "zn_minus_zsp": [c["delta"][0], c["lo"][0], c["hi"][0]],
                         "fc5_minus_zsp": c["delta"][1], "hit1_minus_zsp": c["delta"][2],
                         "share_of_zsp_gain": share(zn, rrf, c["base"][0]),
                         "share_of_zrc_gain": share(zn, rrf, a["base"][0])})
    rec = {"declared_in": DOC, "min_gain": MIN_GAIN, "compares": {k: str(v) for k, v in compares.items()}, "rows": rows,
           "script_sha256": LC.sha_src(__file__), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    md = ["# zn: the lean MLP without message passing, against zrc and zsp", "",
          ("R@5 on the six s1eval carves (p@swa, seed 0). Differences are zn minus the model, with the 95% question "
           f"bootstrap. share = (zn - rrf) / (model - rrf), not computed where the model's gain over rrf is under "
           f"{MIN_GAIN}. ({DOC})"), "",
          ("| split | dataset | read | rrf | zn | zrc | zsp | zn - zrc [95% CI] | zn - zsp [95% CI] | share of zsp's gain | "
           "share of zrc's gain |"), "|---|---|---|---:|---:|---:|---:|---|---|---:|---:|"]

    def f(v):
        return "n/a" if v is None else f"{v:.3f}"

    for r in rows:
        d1, d2 = r["zn_minus_zrc"], r["zn_minus_zsp"]
        md.append(f"| {r['split']} | {r['dataset']} | {r['read']} | {r['rrf']:.4f} | {r['zn']:.4f} | {r['zrc']:.4f} | "
                  f"{r['zsp']:.4f} | {d1[0]:+.4f} [{d1[1]:+.4f}, {d1[2]:+.4f}] | {d2[0]:+.4f} [{d2[1]:+.4f}, "
                  f"{d2[2]:+.4f}] | {f(r['share_of_zsp_gain'])} | {f(r['share_of_zrc_gain'])} |")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    LC.write_json(out.with_suffix(".json"), rec)
    out.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    log(f"zn summary: {len(rows)} reads; {out.with_suffix('.md')}")
    return rec


# ── smoke ────────────────────────────────────────────────────────────────────


def smoke(device, host, out_root=None):
    """zn on 2wiki's select carve (1 epoch, run twice: a repeat must be IDENTICAL), then read. A crash or a differing
    repeat fails it (exit 1)."""
    t0 = time.time()
    root = Path(out_root or S.OUT / "smoke_zn")
    h = ["--host"] if host else []
    rc = main(["train", "--name", "smoke-zn", "--arm", ARM, "--train", "2wiki=select", "--basis", "2wiki", "--variants",
               "p", "--config", "2e-3:1e-4:0.1:1:0", "--repeat", "2", "--device", device, "--out-root", str(root)] + h)
    rr = main(["read", "--name", "smoke-zn", "--read", "2wiki=select", "--device", device, "--out-root", str(root)] + h)
    blob = torch.load(root / "smoke-zn" / "models.pt", map_location="cpu", weights_only=False) \
        if (root / "smoke-zn" / "models.pt").exists() else None
    blocks = blob["variants"]["p"]["blocks"] if blob else None
    ok = rc == 0 and rr == 0 and blocks == NMP
    rec = {"train_rc": rc, "repeat": "IDENTICAL" if rc == 0 else "DIFFERENT", "read_rc": rr, "blocks": blocks, "ok": ok,
           "seconds": time.time() - t0, "script_sha256": LC.sha_src(__file__)}
    LC.write_json(root / "smoke.json", rec)
    log(f"zn smoke: {json.dumps(rec)}; {'ok' if ok else 'FAILED'}")
    return 0 if ok else 1


# ── selftest ─────────────────────────────────────────────────────────────────


def selftest():
    t0 = time.time()
    pick = LG.SETS["pick"]
    # zn's blocks: pick without its edge-reading families, in pick's order
    assert set(NMP) | set(EDGE) == set(pick) and not set(NMP) & set(EDGE), (pick, NMP, EDGE)
    assert [b for b in pick if b in NMP] == NMP
    assert not set(NMP) & {"NBR", "nbr_agg", "gcs", "typed_rel", "typed_v2", "ordered", "seed_r", "topo_NER", "topo_KNN",
                           "topo_FULL", "depth_FULL"}
    # the set narrows inside the context only, and comes back after an exception too
    with nmp_sets():
        assert LG.SETS["pick"] == NMP
    assert LG.SETS["pick"] == pick
    try:
        with nmp_sets():
            raise KeyError("x")
    except KeyError:
        pass
    assert LG.SETS["pick"] == pick and LG.SETS["pns"] == [b for b in pick if b != "SEMB"]
    # the arm: zret's model over the base carve, as lean_screen patches it
    assert S.ARMS[ARM] == (S3.ZRet, S.ARMS["base"][1])
    with S.patched(ARM):
        assert LG.LeanMLP8D is S3.ZRet and LG.CacheCarve is S.ARMS["base"][1]
    assert LG.LeanMLP8D is S.ARMS["base"][0]
    # the model on zn's blocks: a forward and finite gradients; a row's score depends on no other row's inputs except
    # through its pool's statistics (scores of a pool unchanged when another pool's rows change)
    LG.bind_device_ops()
    torch.manual_seed(0)
    n = torch.tensor([7, 5, 4])
    B, N = n.numel(), int(n.sum())
    nq = torch.repeat_interleave(torch.arange(B), n)
    widths = {"rank": len(LM.SPLIT["rank"]), "dense_cos": 1, "SEMB": LM.SEMB_DIM, "SEED": 3}
    pdim = LG.L3.STORE_DIM                                    # the cache's stored projection (pca256 + 1)
    feats = {"rank": torch.rand(N, 5) * 0.03 + 0.001, "dense_cos": torch.rand(N, 1), "SEED": torch.randn(N, 3),
             "SEMB": (torch.randn(B, 1536), torch.randn(N, pdim))}
    m = S3.ZRet(NMP, widths, 16, dropout=0.1, seed=0).eval()
    with torch.no_grad():
        for p in m.out.parameters():
            p.normal_()
        keep = torch.ones(B, len(NMP))
        bz = S3.seg_zscore_ref(feats["rank"][:, S3.I_RRF:S3.I_RRF + 1], nq, B, S3.retrieved(feats)).squeeze(1)
        s = m(feats, keep, nq, B, bz)
        assert s.shape == (N,) and torch.isfinite(s).all()
        f2 = {"rank": feats["rank"].clone(), "dense_cos": feats["dense_cos"].clone(), "SEED": feats["SEED"].clone(),
              "SEMB": (feats["SEMB"][0].clone(), feats["SEMB"][1].clone())}
        f2["dense_cos"][7:12] = torch.rand(5, 1)                         # pool 1 changes
        f2["SEMB"][1][7:12] = torch.randn(5, pdim)
        bz2 = S3.seg_zscore_ref(f2["rank"][:, S3.I_RRF:S3.I_RRF + 1], nq, B, S3.retrieved(f2)).squeeze(1)
        s2 = m(f2, keep, nq, B, bz2)
        assert torch.equal(s[:7], s2[:7]) and torch.equal(s[12:], s2[12:]) and not torch.equal(s[7:12], s2[7:12])
    m.train()
    LG.listwiseD(m(feats, keep, nq, B, bz), (torch.arange(N) % 3 == 0), nq, B).backward()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
    # the summary: shares, the base check, the md
    with tempfile.TemporaryDirectory() as td:
        T = Path(td)
        comp = {}
        for split, b in BASES.items():
            rec = {"by_base": {}}
            for k, zv in (("zrc", 0.70), ("zsp", 0.72)):
                rec["by_base"][f"outputs/screen/fits/{b[k]}"] = {"rows": [
                    {"dataset": ds, "read": "in-domain", "questions": 10, "base": [zv, 0.5, 0.4], "new": [0.66, 0.5, 0.4],
                     "rrf": [0.60, 0.4, 0.3], "delta": [0.66 - zv, 0.0, 0.0], "lo": [-0.1, 0, 0], "hi": [0.0, 0, 0],
                     "call": "WITHIN"} for ds in LG.EVAL_ORDER]}
            LC.write_json(T / f"c-{split}.json", rec)
            comp[split] = T / f"c-{split}.json"
        r = summary(comp, T / "sum")
        assert len(r["rows"]) == 12 and r["rows"][0]["share_of_zsp_gain"] == 0.5 and r["rows"][0]["share_of_zrc_gain"] == 0.6
        assert share(0.61, 0.60, 0.605) is None
        assert "| L-musique | metaqa |" in (T / "sum.md").read_text(encoding="utf-8")
        bad = json.loads((T / "c-L-musique.json").read_text(encoding="utf-8"))
        bad["by_base"] = dict(reversed(list(bad["by_base"].items())))
        LC.write_json(T / "bad.json", bad)
        try:
            summary({"L-musique": T / "bad.json"}, T / "sum2")
            raise AssertionError("summary took the bases in the wrong order")
        except SystemExit:
            pass
    log(f"zn selftest: blocks {NMP} = pick without {list(EDGE)}; the set narrows only inside its context; the arm is "
        f"zret's model over the base carve; a pool's scores ignore other pools' rows; finite gradients; the summary's "
        f"shares and base order ({time.time() - t0:.1f}s): ok")
    return 0


# ── main ─────────────────────────────────────────────────────────────────────


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("cmd", nargs="?")
    ap.add_argument("--split", default="L-musique", choices=S2.SPLITS)
    ap.add_argument("--selftest", action="store_true")
    k, rest = ap.parse_known_args(argv)
    if k.selftest:
        return selftest()
    if k.cmd == "smoke":
        sp = argparse.ArgumentParser()
        sp.add_argument("--device", default="cpu")
        sp.add_argument("--host", action="store_true")
        sp.add_argument("--out-root")
        s = sp.parse_args(rest)
        return smoke(s.device, s.host, s.out_root)
    if k.cmd == "train":
        return train([k.cmd] + rest, k.split)
    if k.cmd == "read":
        return read(rest)
    if k.cmd == "compare":
        cp = argparse.ArgumentParser()
        cp.add_argument("--new", required=True)
        cp.add_argument("--out", required=True)
        cp.add_argument("--fits", default=str(FITS))
        c = cp.parse_args(rest)
        return compare(k.split, c.new, c.out, c.fits)
    if k.cmd == "summary":
        sp = argparse.ArgumentParser()
        sp.add_argument("--compares", default="")
        sp.add_argument("--out", default=str(OUT / "summary"))
        s = sp.parse_args(rest)
        comp = dict(x.split("=", 1) for x in s.compares.split(",") if x) or \
            {sp_: str(OUT / f"compare-{sp_}.json") for sp_ in BASES}
        summary(comp, s.out)
        return 0
    raise SystemExit("zn: train, read, compare, summary, smoke, or --selftest")


if __name__ == "__main__":
    sys.exit(main())
