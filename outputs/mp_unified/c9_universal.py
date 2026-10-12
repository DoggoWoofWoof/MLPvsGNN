"""C9 (docs/C9_UNIVERSAL_PAIR.md): the universal pair (u_mlp_v2_mix against u_gnn_v2_ef) and the lean pair (zrc
against zsp), each model at its fastest exact form, cold, in one run.

FastTwin is u_mlp_v2_mix's serving form at batch 1: c3_fast.FusedGNN's input block (the same code: the numba
per-family segment mean in edge order per node) with the twin's column gates, then the module's body and readout. The
twin reads each family's edges (u, v) straight from the compile. It reads no edge attribute and no relation slot, so it
has no edge pack. No node store, no cache across questions.

The run is C4's (untyped graphs) or C6's (typed graphs) with one more path, umlp9; every check of theirs is kept, and
FastTwin is checked against the twin module's own forward (within C3's TOL, the same top 5) and the look's stored twin0
scores (the same top 5).

    python outputs/mp_unified/c9_universal.py check --dataset 2wiki --queries 30
    python outputs/mp_unified/c9_universal.py run --dataset 2wiki
    python outputs/mp_unified/c9_universal.py report
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

import c4_fast as C4  # noqa: E402
import c6_chains as C6  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402

C5, C3, C2, C1 = C6.C5, C6.C3, C6.C2, C6.C1
LC, LM, LT, S6, GF = C1.LC, C1.LM, C1.LT, C1.S6, C1.GF
UM = GF.UM
FAMILIES, MAX_SEEDS = LT.FAMILIES, LT.MAX_SEEDS
ROOT = HERE.parents[1]
OUT = ROOT / "outputs" / "c9"
TOL = C3.TOL
TWIN_GOOD = ("umlp9_pool", "umlp9_within_tol", "umlp9_top5_same", "umlp9_top5_look")
clock = time.perf_counter


# ── the twin ─────────────────────────────────────────────────────────────────


def load_twin(inputs):
    """twin0 of the six pair, as mp_approx_six_base_score.pair_models loads it (no relation bank: the twin has none)."""
    SB = S6.SB
    key = SB.fit_key(0)
    bad = S6.checkpoint_problems(S6.FITS, key, SB.ARM)
    if bad:
        raise SystemExit(f"{key}: {bad}")
    m = S6.V2.make_model(SB.ARM, inputs, None)
    if SB.parameter_count(m) != SB.PARAMETERS:
        raise SystemExit(f"{key}: {SB.parameter_count(m)} parameters, not {SB.PARAMETERS}")
    pt = S6.FITS / f"{key}.pt"
    m.load_state_dict(torch.load(pt, map_location="cpu"))
    m.eval()
    if not isinstance(m.input, UM.GatedInputBlock):
        raise SystemExit(f"{key}: not the twin's gated input block")
    return m, {"key": key, "arm": SB.ARM, "dir": str(S6.FITS.relative_to(ROOT)).replace("\\", "/"),
               "state_sha256": SB.sha256_file(pt), "parameters": SB.PARAMETERS}


class FastTwin:
    """u_mlp_v2_mix at batch 1: FusedGNN._input's code with the column gates, the module's body and readout."""

    def __init__(self, model):
        self.m = model.eval()
        self.inp = model.input
        self.idx = self.inp.block_index + 1          # each column's position in [1 | gates]; -1 (no block) -> 1

    @torch.inference_mode()
    def __call__(self, b):
        h, base_z = self._input(b)
        return self.m.readout(self.m.body(h), base_z)

    def _input(self, b):
        inp = self.inp
        x = b["x"]
        N = x.shape[0]
        P = inp.semantic.projection_dim
        nq = torch.zeros(N, dtype=torch.long)
        z = GF.segment_zscore(x, nq, 1)
        base_z = z[:, inp.base_index]
        pq = inp.semantic.query_projection(b["qemb"])
        # the column gates, from the normalised projected query (GatedInputBlock.column_gates); one row, broadcast
        gates = torch.sigmoid(Fn.normalize(Fn.gelu(pq), dim=-1) @ inp.gate_weight.t() + inp.gate_bias)
        g = torch.cat([torch.ones(1, 1, dtype=gates.dtype), gates], dim=1)[:, self.idx]
        pre_q = pq.index_select(0, nq)
        raw_q = Fn.gelu(pre_q)
        q_state = Fn.normalize(raw_q, dim=-1)
        emb = b["emb"] if b["emb"].dtype == torch.float32 else b["emb"].to(torch.float32)
        pre_n = inp.semantic.node_projection(emb)
        raw_n = Fn.gelu(pre_n)
        n_state = Fn.normalize(raw_n, dim=-1)
        semantic = torch.cat([
            q_state, n_state, q_state * n_state, (q_state - n_state).abs(),
            (q_state * n_state).sum(-1, keepdim=True), (raw_q * raw_n).sum(-1, keepdim=True) / raw_n.shape[-1] ** 0.5,
        ], dim=-1)
        difference = ((emb - b["qemb"]).abs() @ inp.difference_weight).unsqueeze(1)
        del emb
        parts = [x * g, z * g, semantic, difference]
        pre_np = pre_n.numpy()
        for u, v in b["edges"]:
            if u.size == 0:
                parts.append(torch.zeros(N, P, dtype=x.dtype))
                continue
            order, ptr = C3.csr_by_dst(v, N)
            pm, count = C3.seg_mean_rows(pre_np, u, v, order, ptr)
            p_state = Fn.normalize(Fn.gelu(torch.from_numpy(pm)), dim=-1)
            p_state = torch.where(torch.from_numpy(count > 0).unsqueeze(1), p_state, torch.zeros_like(p_state))
            parts.append(p_state * q_state)
        seed_pre = torch.zeros(1, MAX_SEEDS, P, dtype=pre_n.dtype)
        sn = b["seed_nodes"]
        valid = sn >= 0
        seed_pre[valid] = pre_n[sn[valid]]
        w = b["seedw"]
        reach_pre = torch.einsum("ns,nsp->np", w, seed_pre.index_select(0, nq)) / w.sum(1, keepdim=True).clamp_min(1e-12)
        r_state = Fn.normalize(Fn.gelu(reach_pre), dim=-1)
        r_state = torch.where((w.sum(1) > 0).unsqueeze(1), r_state, torch.zeros_like(r_state))
        parts.append(r_state * n_state)
        h = Fn.gelu(inp.linear(torch.cat(parts, dim=-1)))
        return h, base_z


def twin_pack(comp, E, qemb, columns):
    """The twin's inputs for one question: lean_time.fast_pack's x, embeddings, query, seeds and seed weights, and each
    family's (u, v) as the compile gives them; no edge attribute, no relation slot."""
    row = np.full(MAX_SEEDS, -1, dtype=np.int64)
    sl = np.asarray(comp.seeds_local, dtype=np.int64)
    row[:sl.size] = sl[:MAX_SEEDS]
    return {"x": torch.from_numpy(np.ascontiguousarray(comp.scalars[:, columns])).to(torch.float32),
            "emb": torch.from_numpy(E), "qemb": torch.from_numpy(qemb[None, :]), "seedw": torch.from_numpy(comp.seedw),
            "seed_nodes": torch.from_numpy(row[None, :]), "edges": [comp.edges[f][:2] for f in FAMILIES]}


def look_twin(ds, nq):
    """The look's stored twin0 scores of its first nq questions, in carve order."""
    files, _ids, _head = C1.look_first(ds, nq)
    out = []
    for f in files:
        z = np.load(f)
        qps = z["q_pool_size"]
        no = np.r_[0, np.cumsum(qps)]
        for i in range(qps.size):
            out.append(z["score"][no[i]:no[i + 1]][:, LM.SCORE_COL["twin0"]])
            if len(out) >= nq:
                return out
    return out


def run_twin(su, ft, i, T, gallop=None, take=None):
    """umlp9 for question i, cold: the shared stages (C1.shared, the full v2 compile at C4's galloping form), the
    twin's inputs, FastTwin and the top 5; a warm repeat of the forward beside."""
    if gallop is not None:
        gallop.use(True)
    try:
        sh = C1.shared(su, i, T)
    finally:
        if gallop is not None:
            gallop.use(False)
    if take is not None:
        take()                                       # the compile's typed entries: the twin does not read them
    c0 = clock()
    b = twin_pack(sh["comp"], sh["E"], sh["qemb"], su.columns)
    c1 = clock()
    sc = ft(b)
    torch.topk(sc, min(5, int(sh["pool"].size)))
    c2 = clock()
    ft(b)
    c3 = clock()
    T["pack"], T["forward"], T["warm_forward"] = c1 - c0, c2 - c1, c3 - c2
    return sc, sh, b


def check_twin(su, ft, i, out, tl):
    """Untimed. umlp9's pool against the look's; FastTwin against the twin module's forward on the six GNN's pack of
    the same question (within TOL, the same top 5) and against the look's stored twin0 scores (the same top 5)."""
    sc, sh, _b = out["umlp9"]
    pk = LT.fast_pack(sh["comp"], sh["E"], sh["qemb"], su.columns, True)
    with torch.inference_mode():
        want = ft.m(pk)
    d, top = C3.compare(want, sc)
    s = sc.numpy()
    look = np.asarray(tl[i], np.float32)
    return {"umlp9_pool": bool(np.array_equal(sh["pool"], su.ref[i]["pool"])), "umlp9_diff": d,
            "umlp9_within_tol": d <= TOL, "umlp9_top5_same": top,
            "umlp9_vs_look": float(np.abs(s - look).max()) if s.size else 0.0,
            "umlp9_top5_look": bool(np.array_equal(C1.top5(s), C1.top5(look)))}


# ── the run ──────────────────────────────────────────────────────────────────


class Bench:
    """C4's (untyped) or C6's (typed) setup, paths and checks, with umlp9 added."""

    def __init__(self, a):
        self.typed = a.dataset in C1.TYPED
        if self.typed:
            su, _fa, ks, nq, self.filed, self.frec, self.part0 = C5.prepare(a)
            self.fa = C6.Fast6(su)
            self.paths = ("umlp9",) + C6.PATHS
            self.good = C6.GOOD + TWIN_GOOD
            self.pair = {"universal": ("gnn5", "umlp9"), "lean": ("zsp6", "zrc6")}
        else:
            su = C1.Setup(a.dataset, a.queries)
            self.fa = C3.Fast(su)
            ks = C4.Kernels(su.fc)
            nq = len(su.prep.pools) - 1
            self.paths = ("umlp9",) + C4.PATHS
            self.good = C4.GOOD + TWIN_GOOD
            self.pair = {"universal": ("gnn4", "umlp9"), "lean": ("zsp4", "zrc4")}
        self.su, self.ks, self.nq = su, ks, nq
        t = time.time()
        twin, self.twin_rec = load_twin(su.inputs)
        self.ft = FastTwin(twin)
        su.index["twin_s"] = time.time() - t
        self.tl = look_twin(a.dataset, nq + 1)

    def path(self, k, i, T):
        su, ks = self.su, self.ks
        if k == "umlp9":
            return run_twin(su, self.ft, i, T, take=ks.take) if self.typed else run_twin(su, self.ft, i, T, gallop=ks)
        if self.typed:
            return C6.run_path6(su, self.fa, ks, k, i, T)
        return C4.run_path4(su, self.fa, ks, k, i, T)

    def check(self, i, out):
        c = C6.check_c6(self.su, i, out, self.filed, self.part0) if self.typed else C4.check_c4(self.su, i, out)
        c.update(check_twin(self.su, self.ft, i, out, self.tl))
        return c

    def summarize(self, checks):
        s = {"questions": len(checks), "kept": int(sum(all(c[k] for k in self.good) for c in checks))}
        for k in self.good:
            s[k] = int(sum(bool(c[k]) for c in checks))
        for k in sorted({k for c in checks for k in c if k.endswith(("_diff", "_vs_look"))}):
            v = [c[k] for c in checks if k in c]
            s[f"{k}_max"] = float(max(v)) if v else 0.0
        return s


def run(a):
    import gc
    t_start = time.time()
    pinned = C1.pin() if not a.no_pin else {"pinned": False, "why": "--no-pin"}
    torch.set_num_threads(THREADS)
    if not a.no_pin and not pinned.get("pinned"):
        raise SystemExit("the process could not pin itself; nothing is timed unpinned")
    bn = Bench(a)
    su, nq, paths = bn.su, bn.nq, bn.paths
    t = time.time()
    for k in paths:
        bn.path(k, nq, {})
    su.index["jit_question_s"] = time.time() - t
    C1.log(f"{a.dataset}: {nq} measured questions, index {su.index}")
    recs, checks = [], []
    gc.collect()
    gc.disable()
    try:
        for i in range(nq):
            order = paths[i % len(paths):] + paths[:i % len(paths)]
            T, out = {"order": list(order)}, {}
            for k in order:
                T[k] = {}
                out[k] = bn.path(k, i, T[k])
            T["n"] = int(out["umlp9"][1]["pool"].size)
            recs.append(T)
            checks.append(bn.check(i, out))
            del out
            if (i + 1) % 25 == 0:
                gc.collect()
                C1.log(f"  {i + 1}/{nq}")
    finally:
        gc.enable()
    summ = bn.summarize(checks)
    U6 = S6.U6
    gpt = U6.SIX_FITS / f"{U6.fit_key(0)}.pt"
    rec = {"dataset": a.dataset, "carve": C1.CARVE, "queries": nq, "declared_in": "docs/C9_UNIVERSAL_PAIR.md",
           "threads": THREADS, "pin": pinned, "index": su.index, "paths": list(paths),
           "pairs": {k: list(v) for k, v in bn.pair.items()}, "per_question": recs, "checks": checks,
           "check_summary": summ, "tol": TOL, "twin": bn.twin_rec,
           "gnn": {"key": U6.fit_key(0), "arm": U6.ARM, "state_sha256": S6.SB.sha256_file(gpt)},
           "fits": {k: {"dir": str(C1.FITS[k].relative_to(ROOT)).replace("\\", "/"),
                        "models_sha256": C1.sha_file(C1.FITS[k] / "models.pt"), "candidate": C1.CAND} for k in C1.FITS},
           "look_records": su.look_head["records"], "peak_rss_bytes": LC.peak_rss(),
           "script_sha256": C1.sha_file(__file__), "c3_sha256": C1.sha_file(C3.__file__),
           "c4_sha256": C1.sha_file(C4.__file__), "c6_sha256": C1.sha_file(C6.__file__),
           "numpy": np.__version__, "torch": torch.__version__, "seconds": round(time.time() - t_start, 1),
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if bn.typed:
        rec["chain_builds"] = {k: {"root": str(C5.CH_ROOT[k].relative_to(ROOT)).replace("\\", "/"), **bn.frec[k]}
                               for k in bn.frec}
        rec["step1_part0_record_sha256"] = bn.part0.record_sha256
        rec["relations_sha256"] = su.rel_sha256
    out = Path(a.out or OUT / f"{a.dataset}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    LC.write_json(out, rec)
    C1.log(f"{a.dataset}: done in {rec['seconds']}s; checks {summ}; -> {out}")
    return 0


def check(a):
    """The development check: every path on every question, untimed; prints the failing questions."""
    C1.pin(2)
    torch.set_num_threads(THREADS)
    bn = Bench(a)
    bad, checks = [], []
    for i in [bn.nq] + list(range(bn.nq)):
        out = {k: bn.path(k, i, {}) for k in bn.paths}
        if i == bn.nq:
            continue
        c = bn.check(i, out)
        checks.append(c)
        if not all(c[k] for k in bn.good):
            bad.append((i, {k: c[k] for k in bn.good if not c[k]}))
    s = bn.summarize(checks)
    C1.log(f"c9 check {a.dataset}: {bn.nq} questions, {len(bad)} failing {bad[:3]}")
    C1.log(f"  {({k: v for k, v in s.items() if k.startswith('umlp9') or k in ('questions', 'kept')})}")
    return 0 if not bad else 1


# ── the report ───────────────────────────────────────────────────────────────


def cold(T, k):
    return sum(v for s_, v in T[k].items() if s_ != "warm_forward")


def report():
    res = {}
    for f in sorted(OUT.glob("*.json")):
        if f.name.startswith("report"):
            continue
        r = json.loads(f.read_text(encoding="utf-8"))
        P, paths = r["per_question"], r["paths"]
        good_keys = (C6.GOOD if r["dataset"] in C1.TYPED else C4.GOOD) + TWIN_GOOD
        good = [j for j, c in enumerate(r["checks"]) if all(c[k] for k in good_keys)]
        d = {"questions": len(P), "kept": len(good), "stopped": 1 - len(good) / max(len(P), 1) > 0.02,
             "pairs": r["pairs"]}
        rng = np.random.default_rng(0)
        col = lambda k: np.asarray([cold(P[j], k) for j in good])  # noqa: E731
        wf = lambda k: np.asarray([P[j][k]["warm_forward"] for j in good])  # noqa: E731
        for k in paths:
            d[f"{k}_cold_ms"] = {str(p): float(np.percentile(col(k) * 1e3, p)) for p in (50, 95, 99)}
            d[f"{k}_stage_p50_ms"] = {s_: float(np.percentile([P[j][k][s_] for j in good], 50)) * 1e3
                                      for s_ in P[good[0]][k] if s_ != "warm_forward"}
            d[f"{k}_warm_forward_p50_ms"] = float(np.percentile(wf(k), 50)) * 1e3
        g, um = r["pairs"]["universal"]
        zs, zr = r["pairs"]["lean"]
        for num, den in ((g, um), (zs, zr), (g, zr), (g, zs), (um, zr)):
            d[f"cold_{num}_over_{den}"] = C2.ratio_ci(col(num), col(den), rng)
        for num, den in ((g, um), (zs, zr)):
            d[f"warm_forward_{num}_over_{den}"] = C2.ratio_ci(wf(num), wf(den), rng)
        d["checks"] = r["check_summary"]
        res[r["dataset"]] = d
    LC.write_json(OUT / "report.json", res)
    f = lambda r_: f"{r_['ratio']:.2f} [{r_['ci'][0]:.2f}, {r_['ci'][1]:.2f}]"  # noqa: E731
    lines = ["Cold, batch 1, total per question, p50 ms; ratios GNN / MLP within each family (bootstrap 95% CI):", "",
             "| dataset | universal MLP | universal GNN | **universal GNN / MLP** | lean MLP zrc | lean GNN zsp | "
             "**lean zsp / zrc** | universal GNN / zrc | universal MLP / zrc | warm forward universal GNN / MLP | "
             "warm forward zsp / zrc |",
             "| --- | ---: | ---: | --- | ---: | ---: | --- | --- | --- | --- | --- |"]
    for ds, d in res.items():
        g, um = d["pairs"]["universal"]
        zs, zr = d["pairs"]["lean"]
        m = lambda k: f"{d[f'{k}_cold_ms']['50']:.1f}"  # noqa: E731
        lines.append(f"| {ds} | {m(um)} | {m(g)} | **{f(d[f'cold_{g}_over_{um}'])}** | {m(zr)} | {m(zs)} | "
                     f"**{f(d[f'cold_{zs}_over_{zr}'])}** | {f(d[f'cold_{g}_over_{zr}'])} | {f(d[f'cold_{um}_over_{zr}'])} | "
                     f"{f(d[f'warm_forward_{g}_over_{um}'])} | {f(d[f'warm_forward_{zs}_over_{zr}'])} |")
    lines += ["", "Stage p50 ms (cold), and the warm forward p50 beside:", ""]
    for ds, d in res.items():
        for k in [*d["pairs"]["universal"], *d["pairs"]["lean"]]:
            lines.append(f"- {ds} {k}: " + ", ".join(f"{s_} {v:.2f}" for s_, v in d[f"{k}_stage_p50_ms"].items())
                         + f"; warm forward {d[f'{k}_warm_forward_p50_ms']:.2f}")
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
    return run(a) if a.cmd == "run" else check(a)


if __name__ == "__main__":
    sys.exit(main())
