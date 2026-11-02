"""C1R (docs/C1R_CLAIM1_UNDER_WQ.md): Claim 1 measured again with P3c's intervals, under WQ on PKO's pools. The MLP of record
(zrckp; zrckpn or zrckpc by amendment A1) is read against the GNN of record (zg6r, or zg6r-m if ISO replaces it), seed 0 of
the stage fits, from their reads.

scripts/p3c_claim1.py is imported and not edited; its bootstrap, shares, excess and calls are used as they are. What changes
is the regime's map. There are two WQ splits instead of J5 and five leave-outs, so every in-domain read is its own clause, and
the zero-shot clause pairs a dataset's zero-shot read with its in-domain read under the other split. metaqa, never trained,
is allowed no gap, as P3c's webqsp.

    python outputs/mp_unified/zc1r.py run [--gnn zg6r|zg6rm] [--mlp zrckp|zrckpn|zrckpc] [--limit 50]
    python outputs/mp_unified/zc1r.py --selftest
"""
import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (HERE, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import p3c_claim1 as P3  # noqa: E402

DECLARED = "docs/C1R_CLAIM1_UNDER_WQ.md"
OUT = ROOT / "outputs" / "c1r"
SPLITS = {"WQ-musique": ("webqsp", "squad", "hotpotqa", "2wiki"), "WQ-hotpotqa": ("webqsp", "squad", "musique", "2wiki")}
SUF = {"WQ-musique": "", "WQ-hotpotqa": "-hp"}
EVAL = P3.EVAL_ORDER
ZS = (("musique", "WQ-musique", "WQ-hotpotqa"), ("hotpotqa", "WQ-hotpotqa", "WQ-musique"),
      ("metaqa", "WQ-musique", None), ("metaqa", "WQ-hotpotqa", None))     # (dataset, zero-shot split, in-domain split)
FITS = "outputs/screen/fits"
# per family: (fit name, its compare at seed k, the compare's side holding the fit, the stamp key, the stamp's arm)
GNNS = {"zg6r": ("scr-ml1-zg6r-s{k}{suf}", "scr-ml1-zg6r-vs-zspk-s{k}{suf}", "new", "ml1", "zg6r"),
        "zg6rm": ("scr-iso-zg6rm-s{k}{suf}", "scr-iso-zg6rm-vs-zg6r-s{k}{suf}", "new", "iso", "zg6r")}
# A1: LTM's zrckpn and LTC's zrckpc, each new in its compare against DOB's CPU zrckp of the same seed and split
MLPS = {"zrckp": ("scr-ml1-zrckp-s{k}{suf}", "scr-ml1-zrckpt-vs-zrckp-s{k}{suf}", "base", "ml1", "zrckp"),
        "zrckpn": ("scr-ltm-zrckpn-s{k}{suf}", "scr-ltm-zrckpn-vs-zrckp-s{k}{suf}", "new", "ltm", "zrckpn"),
        "zrckpc": ("scr-ltc-zrckpc-s{k}{suf}", "scr-ltc-zrckpc-vs-zrckp-s{k}{suf}", "new", "ltc", "zrckpc")}
MLP = MLPS["zrckp"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] zc1r: {msg}", flush=True)


def norm(p):
    return str(p or "").replace("\\", "/")


def fam(gnn, mlp="zrckp"):
    return {"mlp": MLPS[mlp], "gnn": GNNS[gnn]}


def rec_name(gnn, mlp="zrckp"):
    """zrckp's record keeps its filed name (claim1_zg6r); another MLP's names both families."""
    return f"claim1_{gnn}" if mlp == "zrckp" else f"claim1_{gnn}_{mlp}"


def stamp_ok(root, fit, key, arm, split, seed=0):
    st = json.loads((Path(root) / FITS / fit / "screen.json").read_text(encoding="utf-8"))
    k = st.get(key) or {}
    if (k.get("fit"), k.get("arm"), k.get("seed"), k.get("split")) != (fit, arm, seed, split):
        raise SystemExit(f"{fit}: its {key} stamp says {k.get('fit')}, {k.get('arm')}, seed {k.get('seed')}, "
                         f"{k.get('split')}; expected {arm}, seed {seed}, {split}")
    if key == "iso" and ((k.get("encoder") or {}).get("start"), (k.get("encoder") or {}).get("copies")) != ("own", 0):
        raise SystemExit(f"{fit}: not an own-start zg6r-m fit")
    nxt = k.get("next") or {}
    if key in ("ltm", "ltc") and not nxt.get("fit_sha256"):
        raise SystemExit(f"{fit}: its {key} stamp holds no NEXT block (the scorer's sha256)")
    if key == "ltc" and not (nxt.get("chain") or {}).get("carves"):
        raise SystemExit(f"{fit}: its ltc stamp holds no CHAIN block")
    return {"fit": fit, "arm": arm, "split": split, "seed": seed, "stamp": key}


def filed_rows(root, cmp_name, side, fit):
    """The compare's rows by dataset; refuses unless the fit is on the named side at p@swa on s1eval."""
    d = json.loads((Path(root) / "outputs" / "screen" / f"{cmp_name}.json").read_text(encoding="utf-8"))
    where = norm(d.get("new")) if side == "new" else norm((d.get("bases") or [None])[0])
    if where != f"{FITS}/{fit}" or d.get("candidate") != P3.CAND or d.get("carve") != P3.CARVE:
        raise SystemExit(f"{cmp_name}: {fit} is not its {side} at {P3.CAND} on {P3.CARVE}")
    return {r["dataset"]: r for r in d["by_base"][d["bases"][0]]["rows"]}


def run(gnn="zg6r", root=ROOT, limit=0, out=None, mlp="zrckp"):
    root = Path(root)
    t0 = time.time()
    F = fam(gnn, mlp)
    stamps, reads = [], {}
    for role in ("mlp", "gnn"):
        fit_t, cmp_t, side, key, arm = F[role]
        for split, suf in SUF.items():
            fit = fit_t.format(k=0, suf=suf)
            stamps.append({**stamp_ok(root, fit, key, arm, split), "role": role})
            filed = None if limit else filed_rows(root, cmp_t.format(k=0, suf=suf), side, fit)
            for ds in EVAL:
                r = P3.load_read(root, f"{FITS}/{fit}", ds, limit)
                if filed is not None:
                    ok = r["gt"] > 0
                    f = filed[ds]
                    if f["questions"] != int(ok.sum()) or P3.r4(r["m"][ok].mean(0)) != f[side] \
                            or P3.r4(r["rrf"][ok].mean(0)) != f["rrf"]:
                        raise SystemExit(f"{fit} {ds}: the read does not reproduce {cmp_t.format(k=0, suf=suf)}")
                reads[(role, split, ds)] = r
    log(f"read {len(reads)} reads; the stamps hold" + ("" if limit else "; every read reproduces its filed compare"))
    rows = []
    for split in SUF:
        for ds in EVAL:
            a, b = reads[("mlp", split, ds)], reads[("gnn", split, ds)]
            P3.same_questions(a, b, f"{split} {ds}")
            if not all(np.array_equal(x, y) for x, y in zip(a["rrf_counts"], b["rrf_counts"])):
                raise SystemExit(f"{split} {ds}: the two reads' RRF columns differ")
            ok = a["gt"] > 0
            st = P3.pair_stats(a["m"][ok], b["m"][ok], a["rrf"][ok], seed=EVAL.index(ds))
            kind = "in-domain" if ds in SPLITS[split] else "zero-shot"
            rows.append({"split": split, "dataset": ds, "read": kind, "questions": int(ok.sum()), "st": st,
                         "files": {x: reads[(x, split, ds)]["path"] for x in ("mlp", "gnn")},
                         "sha256": {x: reads[(x, split, ds)]["sha256"] for x in ("mlp", "gnn")}})
    in_calls = {f"{r['split']} {r['dataset']}": P3.in_domain_call(r["st"]) for r in rows if r["read"] == "in-domain"}
    zs, zs_calls = {}, {}
    for ds, zsp, isp in ZS:
        az, bz = reads[("mlp", zsp, ds)], reads[("gnn", zsp, ds)]
        ok = az["gt"] > 0
        if isp is None:
            ex = P3.excess_stats(None, None, az["m"][ok], bz["m"][ok], seed=EVAL.index(ds))
        else:
            ai, bi = reads[("mlp", isp, ds)], reads[("gnn", isp, ds)]
            P3.same_questions(ai, az, f"{ds}: {isp} and {zsp}")
            ex = P3.excess_stats(ai["m"][ok], bi["m"][ok], az["m"][ok], bz["m"][ok], seed=EVAL.index(ds))
        c = f"{zsp} {ds}"
        zs[c] = {**ex, "in_split": isp}
        zs_calls[c] = P3.zero_shot_call(ex)
    call = P3.claim_call(in_calls, zs_calls)
    log(f"{len(in_calls)} in-domain and {len(zs_calls)} zero-shot clauses ({time.time() - t0:.0f}s)")
    if limit:
        log(f"--limit {limit}: a development run; nothing is written and no number is logged")
        return None
    rec = {"stage": "C1R", "doc": DECLARED, "family": {"mlp": mlp, "gnn": gnn}, "candidate": P3.CAND, "carve": P3.CARVE,
           "boot": P3.BOOT, "bar": P3.BAR, "margin": P3.MARGIN, "fits": stamps, "call": call,
           "in_domain": in_calls, "zero_shot": zs_calls,
           "zero_shot_reads": {c: {"in_split": v["in_split"], "zs_gap": round(v["zs_gap"], 4),
                                   "in_gap": None if v["in_gap"] is None else round(v["in_gap"], 4),
                                   "allowed": round(v["allowed"], 4), "excess": P3.r4(v["excess"]),
                                   "call": zs_calls[c]} for c, v in zs.items()},
           "reads": [{"split": r["split"], "dataset": r["dataset"], "read": r["read"], "questions": r["questions"],
                      "mlp": P3.r4(r["st"]["mlp"]), "gnn": P3.r4(r["st"]["gnn"]), "rrf": P3.r4(r["st"]["rrf"]),
                      "gnn_gain": P3.triple(r["st"]["gnn_gain"]), "mlp_gain": P3.triple(r["st"]["mlp_gain"]),
                      "share": P3.triple(r["st"]["share"]), "gap": P3.triple(r["st"]["gap"]),
                      "files": r["files"], "sha256": r["sha256"]} for r in rows],
           "seeds_beside": seeds_beside(gnn, root, mlp),
           "script_sha256": P3.sha_file(Path(__file__)), "p3c_sha256": P3.sha_file(Path(P3.__file__)),
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    out, nm = (Path(out) if out else OUT), rec_name(gnn, mlp)
    P3.write_json(out / f"{nm}.json", rec)
    (out / f"{nm}.md").write_text(report_md(rec), encoding="utf-8")
    log(f"C1R ({mlp} against {gnn}): {call}; {out / f'{nm}.md'}")
    return rec


def seeds_beside(gnn, root=ROOT, mlp="zrckp"):
    """Section 5: the R@5 share's point value at seeds 1 and 2 from the filed compares' means. Decides nothing."""
    F, out = fam(gnn, mlp), []
    for k in (1, 2):
        for split, suf in SUF.items():
            try:
                m = filed_rows(root, F["mlp"][1].format(k=k, suf=suf), F["mlp"][2], F["mlp"][0].format(k=k, suf=suf))
                g = filed_rows(root, F["gnn"][1].format(k=k, suf=suf), F["gnn"][2], F["gnn"][0].format(k=k, suf=suf))
            except (SystemExit, OSError, KeyError) as e:
                out.append({"seed": k, "split": split, "missing": str(e)[:200]})
                continue
            for ds in EVAL:
                mm, gg, rr = m[ds][F["mlp"][2]][0], g[ds][F["gnn"][2]][0], m[ds]["rrf"][0]
                out.append({"seed": k, "split": split, "dataset": ds, "mlp": mm, "gnn": gg, "rrf": rr,
                            "share": round((mm - rr) / (gg - rr), 4) if gg - rr > P3.MARGIN else None})
    return out


def report_md(rec):
    L = [f"# C1R: Claim 1 under WQ on PKO's pools, {rec['family']['mlp']} against {rec['family']['gnn']}", "",
         f"**{rec['call']}** (bar {rec['bar']}, {rec['boot']} resamples, seed-0 fits)", "",
         "| split | dataset | read | MLP | GNN | RRF | GNN gain [lo, hi] | share [lo, hi] | clause |",
         "|---|---|---|---:|---:|---:|---|---|---|"]
    for r in rec["reads"]:
        c = rec["in_domain"].get(f"{r['split']} {r['dataset']}") or rec["zero_shot"].get(f"{r['split']} {r['dataset']}", "")
        g, s = r["gnn_gain"], r["share"]
        L.append(f"| {r['split']} | {r['dataset']} | {r['read']} | {r['mlp'][0]:.4f} | {r['gnn'][0]:.4f} | {r['rrf'][0]:.4f} | "
                 f"{g['point'][0]:+.4f} [{g['lo'][0]:+.4f}, {g['hi'][0]:+.4f}] | "
                 f"{P3.fmt_share(s)} | {c} |")
    L += ["", "Zero-shot clauses (R@5): excess = zero-shot gap minus the allowed in-domain gap (floored at 0).", "",
          "| read | in-domain split | zs gap | allowed | excess [lo, hi] | call |", "|---|---|---:|---:|---|---|"]
    for c, v in rec["zero_shot_reads"].items():
        e = v["excess"]
        L.append(f"| {c} | {v['in_split'] or 'none (allowed 0)'} | {v['zs_gap']:+.4f} | {v['allowed']:.4f} | "
                 f"{e[0]:+.4f} [{e[1]:+.4f}, {e[2]:+.4f}] | {v['call']} |")
    return "\n".join(L) + "\n"


# ── self-test (made-up reads on a temporary root) ────────────────────────────


def selftest():
    rng = np.random.default_rng(7)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        n = 400
        ids = np.array([f"q{i}" for i in range(n)])
        gts = {ds: rng.integers(1, 4, n) for ds in EVAL}
        rrf_c = {ds: (np.zeros(n, np.int64), np.zeros(n, np.int64)) for ds in EVAL}

        def counts(gt, p):
            top = np.minimum(rng.binomial(gt, p), gt)
            return top, (top > 0).astype(np.int64)

        def write(fit, key, arm, split, side_cmp, seed_k=0):
            fd = root / FITS / fit
            (fd / "reads").mkdir(parents=True, exist_ok=True)
            st = {key: {"fit": fit, "arm": arm, "seed": seed_k, "split": split}}
            if key == "iso":
                st[key]["encoder"] = {"start": "own", "copies": 0}
            if key in ("ltm", "ltc"):
                st[key]["next"] = {"fit_sha256": "f" * 64}
            if key == "ltc":
                st[key]["next"]["chain"] = {"carves": {"musique/s1eval": "c" * 64}}
            (fd / "screen.json").write_text(json.dumps(st), encoding="utf-8")
            means = {}
            for ds in EVAL:
                p = {"mlp": 0.55, "gnn": 0.60}["gnn" if arm == "zg6r" else "mlp"]
                top, hit = counts(gts[ds], p)
                np.savez(fd / "reads" / f"{ds}__s1eval.npz", candidates=np.array(["p@ep0", "p@swa"]),
                         refs=np.array(["rrf", "twin0", "gnn0"]), ids=ids, gold_total=gts[ds],
                         top=np.stack([top, top]), hit=np.stack([hit, hit]),
                         ref_top=np.stack([rrf_c[ds][0]] * 3), ref_hit=np.stack([rrf_c[ds][1]] * 3))
                m = P3.metrics_of(top, hit, gts[ds])
                r = P3.metrics_of(*rrf_c[ds], gts[ds])
                means[ds] = (P3.r4(m.mean(0)), P3.r4(r.mean(0)))
            return means

        def file_cmp(cn, side, fit, means, suf):
            fitp = f"{FITS}/{fit.format(k=0, suf=suf)}".replace("/", "\\")
            other = "outputs\\screen\\fits\\other"
            rec = {"new": fitp if side == "new" else other, "bases": [fitp if side == "base" else other],
                   "candidate": "p@swa", "carve": "s1eval", "by_base": {}}
            rec["by_base"][rec["bases"][0]] = {"rows": [{"dataset": ds, "questions": n, side: means[ds][0],
                                                         "rrf": means[ds][1]} for ds in EVAL]}
            p = root / "outputs" / "screen" / f"{cn.format(k=0, suf=suf)}.json"
            p.write_text(json.dumps(rec), encoding="utf-8")

        for split, suf in SUF.items():
            for fit_t, cmp_t, side, key, arm in (GNNS["zg6r"], *MLPS.values()):
                file_cmp(cmp_t, side, fit_t, write(fit_t.format(k=0, suf=suf), key, arm, split, side), suf)
        recd = run("zg6r", root=root, out=root / "out")
        assert recd["call"] in ("CLAIM1_HOLDS", "CLAIM1_NOT_ALL")
        assert len(recd["in_domain"]) == 8 and len(recd["zero_shot"]) == 4 and len(recd["reads"]) == 12
        assert sorted(recd["zero_shot"]) == sorted(["WQ-musique musique", "WQ-hotpotqa hotpotqa", "WQ-musique metaqa",
                                                    "WQ-hotpotqa metaqa"])
        assert recd["zero_shot_reads"]["WQ-musique metaqa"]["allowed"] == 0.0
        assert all(s.get("missing") for s in recd["seeds_beside"])           # no seed-1/2 compares in the fake root
        assert recd["family"] == {"mlp": "zrckp", "gnn": "zg6r"} and (root / "out" / "claim1_zg6r.json").is_file()
        # A1: zrckpn and zrckpc, each new in its compare, under its own stamp key and record name
        for m in ("zrckpn", "zrckpc"):
            r = run("zg6r", root=root, out=root / "out", mlp=m)
            assert r["family"] == {"mlp": m, "gnn": "zg6r"} and len(r["reads"]) == 12 and len(r["in_domain"]) == 8
            assert {f["stamp"] for f in r["fits"] if f["role"] == "mlp"} == {MLPS[m][3]}
            assert all((root / "out" / f"claim1_zg6r_{m}{x}").is_file() for x in (".json", ".md"))
            assert all(s.get("missing") for s in r["seeds_beside"])
        # an LTM fit stamped without its NEXT block, or an LTC fit without its CHAIN block, is refused
        for m, drop, msg in (("zrckpn", "next", "no NEXT block"), ("zrckpc", "chain", "no CHAIN block")):
            stp = root / FITS / MLPS[m][0].format(k=0, suf="") / "screen.json"
            keep = stp.read_text(encoding="utf-8")
            st = json.loads(keep)
            (st[MLPS[m][3]] if drop == "next" else st[MLPS[m][3]]["next"]).pop(drop)
            stp.write_text(json.dumps(st), encoding="utf-8")
            try:
                run("zg6r", root=root, limit=20, mlp=m)
                raise AssertionError(f"{m} without its {drop} block passed")
            except SystemExit as e:
                assert msg in str(e), e
            stp.write_text(keep, encoding="utf-8")
        # a read that does not reproduce its filed compare is refused
        p = root / FITS / "scr-ml1-zrckp-s0" / "reads" / "squad__s1eval.npz"
        z = dict(np.load(p))
        z["top"] = z["top"] * 0
        np.savez(p, **z)
        try:
            run("zg6r", root=root, out=root / "out2")
            raise AssertionError("a changed read passed")
        except SystemExit as e:
            assert "does not reproduce" in str(e)
        # a wrong stamp is refused
        st = root / FITS / "scr-ml1-zg6r-s0-hp" / "screen.json"
        st.write_text(json.dumps({"ml1": {"fit": "scr-ml1-zg6r-s0-hp", "arm": "zspk", "seed": 0, "split": "WQ-hotpotqa"}}),
                      encoding="utf-8")
        try:
            run("zg6r", root=root, limit=20)
            raise AssertionError("a wrong stamp passed")
        except SystemExit as e:
            assert "expected zg6r" in str(e)
    log("selftest: the map (8 in-domain, 4 zero-shot clauses), metaqa's allowed 0, the filed-compare and stamp refusals, "
        "and A1's zrckpn and zrckpc (their compares, stamp keys, record names, NEXT and CHAIN refusals) pass")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", choices=("run",))
    ap.add_argument("--gnn", choices=sorted(GNNS), default="zg6r")
    ap.add_argument("--mlp", choices=sorted(MLPS), default="zrckp")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.cmd == "run":
        run(a.gnn, limit=a.limit, mlp=a.mlp)
        return 0
    ap.error("a command or --selftest")


if __name__ == "__main__":
    sys.exit(main())
