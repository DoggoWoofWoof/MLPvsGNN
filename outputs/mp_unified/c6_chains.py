"""C6 (docs/C6_FAST_CHAINS.md): the MLP's chain match on the typed KB graphs in an exact serving form, timed cold
against C5's forms in the same run.

  zrc6, zsp6   C5's zrc5 / zsp5 with
               chains    chains_fast.fast_entries (rmatch.question_entries / zrc.contrib_entries, the same entries bit
                         for bit)
               forward   FastZ6: FastZ5 with chains_fast.FastChain (the chain term over the pool's relations, the
                         weight-only product at index time)
  zrc5, zsp5, gnn5   C5's paths unchanged (references, timed in the same run)

Checks (untimed, every question): C5's checks on its three paths; zrc6's / zsp6's entries equal zrc5's / zsp5's bit for
bit (which equal the filed builds); FastZ6 within TOL of the fit's forward on the same rows and entries, same top 5.

    python outputs/mp_unified/c6_chains.py check --dataset webqsp --queries 30
    python outputs/mp_unified/c6_chains.py run --dataset webqsp
    python outputs/mp_unified/c6_chains.py report
"""
import os
import sys
from pathlib import Path

THREADS = 1
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS",
           "LEAN_TIME_THREADS"):
    os.environ[_v] = str(THREADS)
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import json  # noqa: E402
import time  # noqa: E402

import c5_typed as C5  # noqa: E402
import chains_fast as CF  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402

C1, C2, C3, RM, ZC = C5.C1, C5.C2, C5.C3, C5.RM, C5.ZC
LC = C1.LC
ROOT = HERE.parents[1]
OUT = ROOT / "outputs" / "c6"
PATHS = ("zrc6", "zsp6", "gnn5", "zrc5", "zsp5")
NEW = {"zrc6": "zrc5", "zsp6": "zsp5"}
TOL = C3.TOL
clock = time.perf_counter


class FastZ6(C5.FastZ5):
    """FastZ5 with the chain term by chains_fast.FastChain."""

    def __init__(self, model, blocks, span, store, rel_unit, rel_centered):
        super().__init__(model, blocks, span, store)
        self.fch = CF.FastChain(model, rel_unit, rel_centered)

    @torch.inference_mode()
    def __call__(self, r):
        m = self.m
        X = torch.from_numpy(r["X"]).to(torch.float32)
        R = torch.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).index_select(1, self.fixed)
        qe = torch.from_numpy(r["q_emb16"]).to(torch.float32)
        if self.semb:
            P = torch.from_numpy(r["codes"]).to(torch.float32) * self.s + self.c
            P = torch.cat([P, torch.full((P.shape[0], 1), self.kappa)], 1)
            R = torch.cat([R, (qe @ m.U) * (P @ m.V)], 1)
        ref = R[:, self.i_rrf] > 0
        Z = C3.zcols(R, ref)
        h = Fn.gelu(torch.addmm(self.b1, R, self.Wr).addmm_(Z, self.Wz))
        h = Fn.gelu(m.l2(h))
        s = m.base_w * Z[:, self.i_rrf] + m.out(h).squeeze(-1)
        n = s.shape[0]
        if r.get("chains") is not None:
            fm, fr = self.fch(qe, r["chains"], n)
            s = s + (m.cm_gate[0] * fm + m.cm_gate[1] * fr)
        if not self.zsp:
            return s
        _ne, a, b, f = r["links"]
        zs = C3.zcols(s.unsqueeze(1), torch.ones(n, dtype=torch.bool)).squeeze(1)
        zc = zs.clamp(-C1.ZP.ZCLIP, C1.ZP.ZCLIP).numpy()
        deg, sm, ex = C3.prop_numba(zc, a.astype(np.int64), b.astype(np.int64), f.astype(np.int64), n, C1.ZP.FAMS)
        deg, sm, ex = torch.from_numpy(deg), torch.from_numpy(sm), torch.from_numpy(ex)
        has = deg > 0
        zero = torch.zeros_like(deg)
        x = torch.cat([torch.where(has, sm / deg.clamp_min(1.0), zero),
                       torch.where(has, torch.log(ex.clamp_min(1e-30)), zero), torch.log1p(deg), zs.unsqueeze(1)], 1)
        return s + m.prop_head(x)


class Fast6(C5.Fast5):
    def __init__(self, su):
        super().__init__(su)
        self.z6 = {k: FastZ6(su.models[k][0].eval(), su.models[k][1], su.span, su.store, su.rel_unit, su.rel_centered)
                   for k in ("zrc", "zsp")}


def chains_of6(su, k, r, e_rel, qemb):
    """C5.chains_of with chains_fast.fast_entries."""
    eu, ev, ef, efw, ebw = r["e"]
    c = {"e_fam": ef, "e_u": eu, "e_v": ev, "e_fwd": efw, "e_bwd": ebw, "e_rel": e_rel}
    hl, tl, sl = RM.row_triples(c, 0, int(ef.size))
    sl0, bk = r["sl"], r["sb"]
    ok = sl0 >= 0
    seeds = [np.unique(sl0[ok & (bk == b)]).astype(np.int64) for b in range(RM.BUCKETS)]
    qr = np.unique(sl[sl >= 0])
    step = ZC.step_of(qemb, su.U, qr) if k == "zrc" else None
    row, zz, bb, mm = CF.fast_entries(int(r["n"]), hl, tl, sl, seeds, 2 * su.n_rel, step)
    return {"row": row.astype(np.int16), "z": zz.astype(np.int16), "b": bb.astype(np.int8), "m": mm.astype(np.float16),
            "q_rel": qr.astype(np.int16)}


def run_path6(su, fa, ks, k, i, T):
    if k not in NEW:
        return C5.run_path5(su, fa, ks, k, i, T)
    fit = "zsp" if k == "zsp6" else "zrc"
    sh = C2.shared_mlp(su, i, T)
    typed = ks.take()
    r = C3.lean_rows3(su, sh, T)
    c0 = clock()
    e_rel = C5.e_rel_of(r, typed)
    r["chains"] = C5.one_question_chains(su, chains_of6(su, fit, r, e_rel, sh["qemb"]))
    c1 = clock()
    if k == "zsp6":
        eu, ev, ef = r["e"][:3]
        r["links"] = C3.links_fast(r["n"], eu, ev, ef)
    c2 = clock()
    fz = fa.z6[fit]
    sc = fz(r)
    torch.topk(sc, min(5, r["n"]))
    c3 = clock()
    fz(r)
    c4 = clock()
    T["chains"] = c1 - c0
    if k == "zsp6":
        T["links"] = c2 - c1
    T["forward"], T["warm_forward"] = c3 - c2, c4 - c3
    r["e_rel"] = e_rel
    return sc, sh, r


def check_c6(su, i, out, filed, part0):
    c = C5.check_c5(su, i, out, filed, part0)
    for new, old in NEW.items():
        a, b = out[new][2]["chains"], out[old][2]["chains"]
        c[f"{new}_entries_bits"] = bool(all(
            a[x].dtype == b[x].dtype and a[x].shape == b[x].shape and np.array_equal(C1.bits(a[x]), C1.bits(b[x]))
            for x in ("ent_row", "ent_z", "ent_b", "ent_m", "q_rel")))
        fit = "zsp" if new == "zsp6" else "zrc"
        want = C5.ref_forward(su, fit, out[new][2], out[new][2]["chains"])
        d, top = C3.compare(want, out[new][0])
        c[f"{new}_diff"], c[f"{new}_top5_same"], c[f"{new}_within_tol"] = d, top, d <= TOL
    return c


GOOD = C5.GOOD + ("zrc6_entries_bits", "zsp6_entries_bits", "zrc6_within_tol", "zrc6_top5_same", "zsp6_within_tol",
                  "zsp6_top5_same")


def summarize(checks):
    s = C5.summarize(checks)
    for k in GOOD[len(C5.GOOD):]:
        s[k] = int(sum(bool(c[k]) for c in checks))
    for k in ("zrc6_diff", "zsp6_diff"):
        s[f"{k}_max"] = float(max(c[k] for c in checks)) if checks else 0.0
    return s


def run(a):
    import gc
    t_start = time.time()
    pinned = C1.pin() if not a.no_pin else {"pinned": False, "why": "--no-pin"}
    torch.set_num_threads(THREADS)
    if not a.no_pin and not pinned.get("pinned"):
        raise SystemExit("the process could not pin itself; nothing is timed unpinned")
    su, _fa, ks, nq, filed, frec, part0 = C5.prepare(a)
    fa = Fast6(su)
    t = time.time()
    for k in PATHS:
        run_path6(su, fa, ks, k, nq, {})
    su.index["jit_question_s"] = time.time() - t
    C1.log(f"{a.dataset}: {nq} measured questions, index {su.index}")
    recs, checks = [], []
    gc.collect()
    gc.disable()
    try:
        for i in range(nq):
            order = PATHS[i % len(PATHS):] + PATHS[:i % len(PATHS)]
            T, out = {"order": list(order)}, {}
            for k in order:
                T[k] = {}
                out[k] = run_path6(su, fa, ks, k, i, T[k])
            T["n"] = int(out["gnn5"][1]["pool"].size)
            recs.append(T)
            checks.append(check_c6(su, i, out, filed, part0))
            del out
            if (i + 1) % 25 == 0:
                gc.collect()
                C1.log(f"  {i + 1}/{nq}")
    finally:
        gc.enable()
    rec = {"dataset": a.dataset, "carve": C1.CARVE, "queries": nq, "declared_in": "docs/C6_FAST_CHAINS.md",
           "threads": THREADS, "pin": pinned, "index": su.index, "per_question": recs, "checks": checks,
           "check_summary": summarize(checks), "tol": TOL,
           "fits": {k: {"dir": str(C1.FITS[k].relative_to(ROOT)).replace("\\", "/"),
                        "models_sha256": C1.sha_file(C1.FITS[k] / "models.pt"), "candidate": C1.CAND} for k in C1.FITS},
           "chain_builds": {k: {"root": str(C5.CH_ROOT[k].relative_to(ROOT)).replace("\\", "/"), **frec[k]} for k in frec},
           "step1_part0_record_sha256": part0.record_sha256, "relations_sha256": su.rel_sha256,
           "basis": su.basis, "basis_sha256": su.basis_sha256, "look_records": su.look_head["records"],
           "peak_rss_bytes": LC.peak_rss(), "script_sha256": C1.sha_file(__file__),
           "chains_fast_sha256": C1.sha_file(CF.__file__), "c5_sha256": C1.sha_file(C5.__file__),
           "numpy": np.__version__, "torch": torch.__version__, "seconds": round(time.time() - t_start, 1),
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    out = Path(a.out or OUT / f"{a.dataset}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    LC.write_json(out, rec)
    C1.log(f"{a.dataset}: done in {rec['seconds']}s; checks {rec['check_summary']}; -> {out}")
    return 0


def check(a):
    C1.pin(2)
    torch.set_num_threads(THREADS)
    su, _fa, ks, nq, filed, frec, part0 = C5.prepare(a)
    fa = Fast6(su)
    bad, checks = [], []
    for i in [nq] + list(range(nq)):
        out = {k: run_path6(su, fa, ks, k, i, {}) for k in PATHS}
        if i == nq:
            continue
        c = check_c6(su, i, out, filed, part0)
        checks.append(c)
        if not all(c[k] for k in GOOD):
            bad.append((i, {k: c[k] for k in GOOD if not c[k]}))
    C1.log(f"c6 check {a.dataset}: {nq} questions, {len(bad)} failing {bad[:3]}")
    s = summarize(checks)
    C1.log(f"  {({k: s[k] for k in s if k.startswith(('zrc6', 'zsp6'))})}")
    return 0 if not bad else 1


def cold(T, k):
    return sum(v for s_, v in T[k].items() if s_ != "warm_forward")


def report():
    res = {}
    for f in sorted(OUT.glob("*.json")):
        if f.name.startswith("report"):
            continue
        r = json.loads(f.read_text(encoding="utf-8"))
        P = r["per_question"]
        good = [j for j, c in enumerate(r["checks"]) if all(c[k] for k in GOOD)]
        d = {"questions": len(P), "kept": len(good), "stopped": 1 - len(good) / max(len(P), 1) > 0.02}
        rng = np.random.default_rng(0)
        col = lambda k: np.asarray([cold(P[j], k) for j in good])  # noqa: E731
        for k in PATHS:
            d[f"{k}_cold_ms"] = {str(p): float(np.percentile(col(k) * 1e3, p)) for p in (50, 95, 99)}
            d[f"{k}_stage_p50_ms"] = {s_: float(np.percentile([P[j][k][s_] for j in good], 50)) * 1e3
                                      for s_ in P[good[0]][k] if s_ != "warm_forward"}
        for num, den in (("gnn5", "zrc6"), ("gnn5", "zsp6"), ("gnn5", "zrc5"), ("zrc5", "zrc6"), ("zsp5", "zsp6")):
            d[f"cold_{num}_over_{den}"] = C2.ratio_ci(col(num), col(den), rng)
        wf = lambda k: np.asarray([P[j][k]["warm_forward"] for j in good])  # noqa: E731
        d["warm_forward_gnn5_over_zrc6"] = C2.ratio_ci(wf("gnn5"), wf("zrc6"), rng)
        d["checks"] = r["check_summary"]
        res[r["dataset"]] = d
    LC.write_json(OUT / "report.json", res)
    f = lambda r_: f"{r_['ratio']:.2f} [{r_['ci'][0]:.2f}, {r_['ci'][1]:.2f}]"  # noqa: E731
    lines = ["Cold, batch 1, total per question, p50 ms:", "",
             "| dataset | zrc6 | zsp6 | gnn5 | **gnn5 / zrc6** | gnn5 / zsp6 | gnn5 / zrc5 (C5, same run) | zrc5 / zrc6 | "
             "zsp5 / zsp6 | chains p50 zrc5 -> zrc6 | forward p50 zrc5 -> zrc6 | warm forward gnn5 / zrc6 |",
             "| --- | ---: | ---: | ---: | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for ds, d in res.items():
        m = lambda k: f"{d[f'{k}_cold_ms']['50']:.1f}"  # noqa: E731
        st = lambda k, s_: d[f"{k}_stage_p50_ms"][s_]  # noqa: E731
        lines.append(f"| {ds} | {m('zrc6')} | {m('zsp6')} | {m('gnn5')} | **{f(d['cold_gnn5_over_zrc6'])}** | "
                     f"{f(d['cold_gnn5_over_zsp6'])} | {f(d['cold_gnn5_over_zrc5'])} | {f(d['cold_zrc5_over_zrc6'])} | "
                     f"{f(d['cold_zsp5_over_zsp6'])} | {st('zrc5', 'chains'):.2f} -> {st('zrc6', 'chains'):.2f} | "
                     f"{st('zrc5', 'forward'):.2f} -> {st('zrc6', 'forward'):.2f} | "
                     f"{f(d['warm_forward_gnn5_over_zrc6'])} |")
    lines += ["", "Stage p50 ms:", ""]
    for ds, d in res.items():
        for k in PATHS:
            lines.append(f"- {ds} {k}: " + ", ".join(f"{s_} {v:.2f}" for s_, v in d[f"{k}_stage_p50_ms"].items()))
    lines += ["", "Checks:", ""] + [f"- {ds}: kept {d['kept']}/{d['questions']}; {d['checks']}" for ds, d in res.items()]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("run", "check", "report"))
    ap.add_argument("--dataset")
    ap.add_argument("--queries", type=int, default=200)
    ap.add_argument("--out")
    ap.add_argument("--no-pin", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "report":
        return report()
    if a.dataset not in C1.TYPED:
        raise SystemExit(f"C6 serves {C1.TYPED}")
    return run(a) if a.cmd == "run" else check(a)


if __name__ == "__main__":
    sys.exit(main())
