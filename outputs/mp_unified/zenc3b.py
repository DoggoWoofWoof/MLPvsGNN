"""ENC E3B (docs/ENC_ENCODER_COMPARISON.md section 18): E2B's recipe search (section 16) on E0N's encoders.

E2B's code runs unchanged, on its own module, with five things set for this stage: the output folder (outputs/enc/e3b),
the arms, the encoders that get windows and maps, fusion for every two-encoder arm (E2B fused only its mpnet arm), and
the stage's name in its records. Section 18 has the gates and the call; this file adds only the report.

    python outputs/mp_unified/zenc3b.py selftest [--real]
    python outputs/mp_unified/zenc3b.py winenc --enc ENC --device cuda    (gtemodern, granite or granitesm)
    python outputs/mp_unified/zenc3b.py maps --threads 8
    python outputs/mp_unified/zenc3b.py grid --arm ARM --threads 8
    python outputs/mp_unified/zenc3b.py report
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.dont_write_bytecode = True

import zenc as E  # noqa: E402  (first: sets the BLAS threads before numpy loads)
import zenc3 as Z3  # noqa: E402  (registers E0N's three encoders and their pins)
import zenc2m as M  # noqa: E402
import zenc2s as S  # noqa: E402
import zenc2b as B  # noqa: E402

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

STAGE = "ENC E3B"
DECLARED = "docs/ENC_ENCODER_COMPARISON.md section 18"
OUT = E.OUT / "e3b"
DS = B.DS
NEW = ("gtemodern", "granite", "granitesm")
ARMS = {"gtemodern": ("gtemodern",), "gtemodern+granite": ("gtemodern", "granite"),
        "gtemodern+granitesm": ("gtemodern", "granitesm"), "granitesm": ("granitesm",)}
E2B_READS = E.OUT / "e2b" / "reads"
NEAR_PTS = 0.5

# ── E2B's module, set for this stage ─────────────────────────────────────────
B.OUT, B.STAGE, B.DECLARED = OUT, STAGE, DECLARED
B.ARMS, B.SMALL, B.NEW_WIN, B.NEW_MAPS, B.HOT_ARMS = ARMS, tuple(ARMS), NEW, NEW, ()
_code_shas = B.code_shas


def code_shas():
    return dict(_code_shas(), **{f"outputs/mp_unified/{Path(p).name}": E.sha256(Path(p)) for p in (__file__, Z3.__file__)})


def log(msg):
    print(f"[zenc3b {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def stop(msg):
    raise SystemExit(f"zenc3b: {msg}")


def weights_of(arm, w):
    """[(encoder, weight)] with weight > 0: (1 - w) on the arm's first encoder, w on its second."""
    encs = ARMS[arm]
    if len(encs) == 2:
        return [(e, x) for e, x in ((encs[0], 1.0 - w), (encs[1], w)) if x > 0]
    return [(encs[0], 1.0)]


def step_a(arm, data, maps):
    """B.step_a, with w over WEIGHTS for every two-encoder arm (E2B's tied that to the mpnet arm)."""
    encs = ARMS[arm]
    ws = B.WEIGHTS if len(encs) == 2 else (0.0,)
    bw = [(b, W) for b in B.BETAS for W in ((None,) if b == 0.0 else B.WINS)]
    tops = {}
    for g in B.GAMMAS:
        for ds in DS:
            cs, _, P = data[ds]
            Qm = {e: B.qmap(maps[e], P[e]["Qs"], g) for e in encs}
            nq = len(cs["ids"])
            for q0 in range(0, nq, B.QB):
                q1 = min(nq, q0 + B.QB)
                Sd = {e: Qm[e][q0:q1] @ P[e]["D"].T for e in encs}
                Sw = {(e, W): np.maximum.reduceat(Qm[e][q0:q1] @ P[e]["win"][W][0].T, P[e]["win"][W][1], axis=1)
                      for e in encs for W in B.WINS}
                for b, W in bw:
                    part = {e: Sd[e] if b == 0.0 else (Sw[(e, W)] if b == 1.0 else
                                                       np.float32(1.0 - b) * Sd[e] + np.float32(b) * Sw[(e, W)])
                            for e in encs}
                    for w in ws:
                        key = (g, b, W, w, ds)
                        if key not in tops:
                            tops[key] = np.empty((nq, 5), np.int64)
                        tops[key][q0:q1] = B.top_ids(B.mix(part, weights_of(arm, w)), 5)
        log(f"grid {arm} step A: gamma={g} done")
    curve = []
    for g in B.GAMMAS:
        for b, W in bw:
            for w in ws:
                r = {ds: S.r5_of(tops[(g, b, W, w, ds)], data[ds][0]) for ds in DS}
                curve.append({"setting": B.setting(g, b, W, w), "s1sel_r5": {k: round(v, 6) for k, v in r.items()},
                              "mean": float(np.mean(list(r.values())))})
    return curve


B.code_shas, B.log, B.stop, B.weights_of, B.step_a = code_shas, log, stop, weights_of, step_a


# ── the report (18.6) ────────────────────────────────────────────────────────

def bars():
    e2 = json.loads(B.E2R.read_text(encoding="utf-8"))
    return {"mpnet": B.bars_of(e2, "mpnet"), "qwen3e06": B.bars_of(e2, "qwen3e06")}


def clears(gap, b):
    """Both carves' gaps (points) at or above one deciding model's bars: that model's name, else None."""
    for m in B.MODELS:
        if all(gap[ds] >= b[m][ds] for ds in DS):
            return m
    return None


def gate(gap, bs):
    """18.5 on V: ADVANCE clears qwen3e06's bars for one deciding model; NEAR clears mpnet's; else SHORT."""
    if clears(gap, bs["qwen3e06"]):
        return "ADVANCE"
    if clears(gap, bs["mpnet"]):
        return "NEAR"
    return "SHORT"


def mixed_ok(wb):
    """MIXED: above 0 on one carve, and not below 0 on the other (its interval's top at or above 0)."""
    lo = {ds: wb[ds]["ci"][0] for ds in DS}
    hi = {ds: wb[ds]["ci"][1] for ds in DS}
    return any(lo[a] > 0 and hi[b] >= 0 for a in DS for b in DS if a != b)


def call_14_5(wb):
    if all(wb[ds]["ci"][0] > 0 for ds in DS):
        return "IMPROVES"
    return "MIXED" if mixed_ok(wb) else "NO"


def params_of(arm, models):
    return int(sum(models[e]["params"] for e in ARMS[arm]))


def width_of(arm):
    return int(sum(E.ENCODERS[e]["dim"] for e in ARMS[arm]))


def build(grids, z, bs, models, beside):
    """The report from the grids, the reads z[(name, ds)] and the references (E0's reads: frozen and each arm's first
    encoder; E2B's qwen3e06 and frozen V and F)."""
    rep = {"stage": STAGE, "declared_in": DECLARED, "bars": bs, "arms": {}}
    pq = lambda name, ds: M.per_q(z[(name, ds)], "d5")  # noqa: E731
    for arm, g in grids.items():
        raw = ARMS[arm][0]
        r = {"chosen": g["chosen"], "params": params_of(arm, models), "width": width_of(arm), "recipes": {}}
        for rc in ("V", "F"):
            name = f"{arm}-{rc}"
            wb = {ds: M.gap(pq(name, ds), pq(f"e0:{raw}", ds)) for ds in DS}
            gp = {ds: M.gap(pq(name, ds), pq("e0:frozen", ds)) for ds in DS}
            x = {"setting": g["chosen"][rc], "won_back": wb, "call": call_14_5(wb), "gap_to_frozen": gp,
                 "recall": {ds: {k: round(100 * float(np.mean(M.per_q(z[(name, ds)], k))), 2)
                                 for k in ("d5", "d20", "d1000", "rrf5")} for ds in DS},
                 "loo": {}, "beside": {}}
            for key, ref in beside.items():
                if ref.endswith(f"-{rc}") and (ref, DS[0]) in z:
                    x["beside"][key] = {ds: M.gap(pq(name, ds), pq(ref, ds)) for ds in DS}
            for p in B.parts_on(g["chosen"][rc]):
                ln = f"{arm}-{rc}-no-{p}"
                x["loo"][p] = {ds: M.gap(pq(name, ds), pq(ln, ds)) for ds in DS}
            gpts = {ds: gp[ds]["pts"] for ds in DS}
            x["clears"] = {k: clears(gpts, b) for k, b in bs.items()}
            if rc == "V":
                x["gate"] = gate(gpts, bs)
            x["mean_gap_to_frozen"] = round(float(np.mean(list(gpts.values()))), 2)
            r["recipes"][rc] = x
        rep["arms"][arm] = r
    rank = {"ADVANCE": 0, "NEAR": 1, "SHORT": 2}
    best_call = min((r["recipes"]["V"]["gate"] for r in rep["arms"].values()), key=rank.get)
    cand = [a for a, r in rep["arms"].items() if r["recipes"]["V"]["gate"] == best_call]
    best = max(cand, key=lambda a: (rep["arms"][a]["recipes"]["V"]["mean_gap_to_frozen"], -rep["arms"][a]["params"]))
    bm = rep["arms"][best]["recipes"]["V"]["mean_gap_to_frozen"]
    smaller = [a for a in cand if rep["arms"][a]["params"] < rep["arms"][best]["params"]
               and rep["arms"][a]["recipes"]["V"]["mean_gap_to_frozen"] >= bm - NEAR_PTS]
    small = min(smaller, key=lambda a: rep["arms"][a]["params"]) if smaller else None
    rep["next"] = {"call": best_call, "best": best, "smaller_within_0.5": small}
    return rep


def fmt(g):
    return f"{g['pts']:+.2f} [{g['ci'][0]:+.2f}, {g['ci'][1]:+.2f}]"


def report_md(rep):
    L = [f"# {STAGE} report", "", f"Declared in {DECLARED}. s1eval dense R@5, points, paired 95% bootstrap over "
         "questions (2,000, seed 0); SQuAD / MuSiQue.", "",
         "| arm | params / width | recipe | setting | R@5 | won back (vs raw) | call | gap to frozen | gate (V) |",
         "|---|---|---|---|---|---|---|---|---|"]
    for arm, r in rep["arms"].items():
        for rc, x in r["recipes"].items():
            L.append(f"| {arm} | {r['params'] / 1e6:.0f}M / {r['width']} | {rc} | `{json.dumps(x['setting'])}` | "
                     f"{x['recall']['squad']['d5']:.2f} / {x['recall']['musique']['d5']:.2f} | "
                     f"{fmt(x['won_back']['squad'])} / {fmt(x['won_back']['musique'])} | {x['call']} | "
                     f"{fmt(x['gap_to_frozen']['squad'])} / {fmt(x['gap_to_frozen']['musique'])} | {x.get('gate', '')} |")
    L += ["", "## Beside (no call)", ""]
    for arm, r in rep["arms"].items():
        for rc, x in r["recipes"].items():
            L.append(f"- {arm} {rc}: R@20 {x['recall']['squad']['d20']:.2f} / {x['recall']['musique']['d20']:.2f}, "
                     f"R@1000 {x['recall']['squad']['d1000']:.2f} / {x['recall']['musique']['d1000']:.2f}, "
                     f"RRF@5 {x['recall']['squad']['rrf5']:.2f} / {x['recall']['musique']['rrf5']:.2f}; leave one out "
                     + (", ".join(f"{p} {fmt(v['squad'])} / {fmt(v['musique'])}" for p, v in x["loo"].items()) or "none")
                     + "; against " + (", ".join(f"{k} {fmt(v['squad'])} / {fmt(v['musique'])}"
                                                 for k, v in x["beside"].items()) or "none")
                     + f"; clears {json.dumps(x['clears'])}")
    L += ["", f"Bars (pointers): {json.dumps(rep['bars'])}", "", f"Next (18.7): {json.dumps(rep['next'])}", ""]
    return "\n".join(L)


def load_all_reads(grids):
    z = {}
    for ds in DS:
        for enc in {ARMS[a][0] for a in grids} | {"frozen"}:
            z[(f"e0:{enc}", ds)] = np.load(E.OUT / "e0" / enc / f"{ds}.npz", allow_pickle=False)
        for arm, g in grids.items():
            for name in g["plan"]:
                z[(name, ds)] = np.load(OUT / "reads" / f"{B.read_name(g, name, ds)}__{ds}.npz", allow_pickle=False)
        for ref in ("qwen3e06-V", "qwen3e06-F", "frozen-V", "frozen-F"):
            p = E2B_READS / f"{ref}__{ds}.npz"
            if not p.is_file():
                stop(f"E2B's {ref} read for {ds} is missing")
            z[(ref, ds)] = np.load(p, allow_pickle=False)
        ids = z[("e0:frozen", ds)]["ids"]
        for k, v in z.items():
            if k[1] == ds and not (v["ids"].shape == ids.shape and (v["ids"] == ids).all()):
                stop(f"{k[0]} {ds}: other question ids than E0's frozen read")
    return z


BESIDE = {"qwen3e06 V": "qwen3e06-V", "qwen3e06 F": "qwen3e06-F", "frozen V": "frozen-V", "frozen F": "frozen-F"}


def cmd_report(a):
    grids = {}
    for arm in ARMS:
        g = json.loads((OUT / f"grid-{arm}.json").read_text(encoding="utf-8"))
        bad = [k for k, v in g["g1"].items() if not v["pass"]] + [k for x in g["g2"].values() for k, v in x.items()
                                                                  if not v["pass"]]
        if bad:
            stop(f"{arm}: G1 or G2 failed ({bad})")
        grids[arm] = g
    z = load_all_reads(grids)
    models = json.loads((Z3.OUT / "models.json").read_text(encoding="utf-8"))
    rep = build(grids, z, bars(), models, BESIDE)
    rep.update(maps=json.loads((OUT / "maps.json").read_text(encoding="utf-8"))["maps"],
               code_sha256=code_shas(), utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    for v in rep["maps"].values():
        v.pop("curve", None)
    E.wjson(OUT / "report.json", rep)
    (OUT / "report.md").write_text(report_md(rep), encoding="utf-8")
    log("gates: " + json.dumps({a_: r["recipes"]["V"]["gate"] for a_, r in rep["arms"].items()}) + "; next "
        + json.dumps(rep["next"]))
    return 0


# ── self-test ────────────────────────────────────────────────────────────────

def selftest_toy():
    # the module is set for this stage
    assert B.OUT == OUT and B.ARMS is ARMS and B.NEW_WIN == NEW and B.NEW_MAPS == NEW and B.HOT_ARMS == ()
    assert B.step_a is step_a and B.weights_of is weights_of and B.code_shas is code_shas
    assert B.win_vec_path("granite", "squad", 64) == OUT / "win" / "granite_squad_w64.npy"
    assert B.win_doc_path("squad", 64) == E.OUT / "e2s" / "win" / "squad_w64_doc.npy"
    # fusion: the ends are each encoder alone, and the one-vector form gives the same score
    assert weights_of("gtemodern+granite", 0.0) == [("gtemodern", 1.0)]
    assert weights_of("gtemodern+granite", 1.0) == [("granite", 1.0)]
    assert weights_of("gtemodern+granitesm", 0.25) == [("gtemodern", 0.75), ("granitesm", 0.25)]
    assert weights_of("granitesm", 0.5) == [("granitesm", 1.0)]
    rng = np.random.default_rng(0)
    nq, nd = 40, 300
    P = {e: {"D": M.unit(rng.standard_normal((nd, E.ENCODERS[e]["dim"])).astype(np.float32)), "win": {}}
         for e in ("gtemodern", "granitesm")}
    Qm = {e: M.unit(rng.standard_normal((nq, E.ENCODERS[e]["dim"])).astype(np.float32)) for e in P}
    w = 0.25
    got = B.score_fn("gtemodern+granitesm", P, Qm, B.setting(w=w))(0, nq)
    cq = np.concatenate([np.sqrt(1 - w) * Qm["gtemodern"], np.sqrt(w) * Qm["granitesm"]], 1)
    cd = np.concatenate([np.sqrt(1 - w) * P["gtemodern"]["D"], np.sqrt(w) * P["granitesm"]["D"]], 1)
    assert np.allclose(cq @ cd.T, got, atol=1e-5)
    # step A's grid sizes: 135 for a two-encoder arm, 27 for one, with toy windows
    cs = {"ids": [f"q{i}" for i in range(nq)], "gt": np.ones(nq, np.int64),
          "flat": rng.integers(0, nd, nq).astype(np.int64), "off": np.arange(nq + 1, dtype=np.int64)}
    for e in P:
        P[e]["Qs"] = Qm[e]
        for W in B.WINS:
            doc = np.sort(rng.integers(0, nd, 2 * nd))
            doc[:nd] = np.arange(nd)
            doc = np.sort(doc)
            P[e]["win"][W] = (M.unit(rng.standard_normal((doc.size, E.ENCODERS[e]["dim"])).astype(np.float32)),
                              np.searchsorted(doc, np.arange(nd)))
    eye = {e: (None, "linear") for e in P}
    orig_qmap = B.qmap
    B.qmap = lambda mp, X, gamma, block=65536: X           # noqa: E731  (toy: no map)
    try:
        data = {ds: (cs, cs, P) for ds in DS}
        orig_r5 = S.r5_of
        S.r5_of = lambda top, c: float(np.mean([c["flat"][i] in top[i] for i in range(len(c["ids"]))]))  # noqa: E731
        try:
            assert len(step_a("gtemodern+granitesm", data, eye)) == 135
            ARMS["_one"] = ("granitesm",)
            assert len(step_a("_one", data, eye)) == 27
        finally:
            ARMS.pop("_one", None)
            S.r5_of = orig_r5
    finally:
        B.qmap = orig_qmap
    # the gate and the call words
    bs = {"mpnet": M.DECLARED_BARS, "qwen3e06": {"zrckpn": {"squad": -3.79, "musique": -1.14},
                                                 "gnx": {"squad": -4.25, "musique": -1.25}}}
    assert gate({"squad": -3.0, "musique": -1.0}, bs) == "ADVANCE"
    assert gate({"squad": -4.0, "musique": -1.2}, bs) == "ADVANCE"         # GNX's bars
    assert gate({"squad": -4.6, "musique": -1.45}, bs) == "NEAR"
    assert gate({"squad": -6.0, "musique": -1.0}, bs) == "SHORT"
    g = lambda lo, hi: {"pts": (lo + hi) / 2, "ci": [lo, hi]}  # noqa: E731
    assert call_14_5({"squad": g(0.1, 1), "musique": g(0.2, 2)}) == "IMPROVES"
    assert call_14_5({"squad": g(0.1, 1), "musique": g(-0.5, 0.4)}) == "MIXED"
    assert call_14_5({"squad": g(0.1, 1), "musique": g(-0.9, -0.1)}) == "NO"
    assert call_14_5({"squad": g(-0.1, 1), "musique": g(-0.5, 0.4)}) == "NO"
    print("zenc3b self-test: toy OK", flush=True)


def selftest_real():
    root = E.served_root()
    out = {"bars": bars()}
    assert out["bars"]["mpnet"] == M.DECLARED_BARS
    for e in NEW:
        assert Z3.pinned(e) == Z3.NEW3[e]["rev"]
        for ds in DS:
            for kind in ("queries", "docs"):
                assert len(E.vec_shards(root, e, ds, kind)) == E.manifest(root, ds, kind)["n_shards"]
            z = np.load(E.OUT / "e0" / e / f"{ds}.npz", allow_pickle=False)
            out[f"e0 {e} {ds}"] = round(100 * float(np.mean(M.per_q(z, "d5"))), 2)
    for ds in DS:                                     # the window index is E2S's, on the files E2B used
        for W in B.WINS:
            assert B.win_doc_path(ds, W).is_file()
    print("zenc3b real: " + json.dumps(out), flush=True)


def cmd_selftest(a):
    selftest_toy()
    if a.real:
        selftest_real()
    print("zenc3b self-test: OK", flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["selftest", "winenc", "maps", "grid", "report"])
    ap.add_argument("--enc", choices=NEW)
    ap.add_argument("--arm", choices=tuple(ARMS))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    if a.cmd == "winenc" and not a.enc:
        stop("winenc needs --enc")
    if a.cmd == "grid" and not a.arm:
        stop("grid needs --arm")
    return {"selftest": cmd_selftest, "winenc": B.cmd_winenc, "maps": B.cmd_maps, "grid": B.cmd_grid,
            "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
