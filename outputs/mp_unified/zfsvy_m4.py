"""FSV amendment A3, correction 3 (docs/FSV_FAST_VERSIONS.md section 15): M4, the timed pass of Family 2. Per dataset one job: the first 200
questions and the JIT question, three versions of each question in a rotating order (the full MLP zrckpn, the full GNN zg6r, the cut GNN
zg6r without topo_STRUCT, depth_STRUCT, WALK and WALKF), cold at batch 1 and one thread, then the same inputs warm at batches of 1 and of
16, at BLAS 2 (the record) and then BLAS 1 (beside it), after a smoke of 32 questions. The cut path's forms and its fits are
zfsvy_cut.py's (imported, not edited), A4's, A6's and A7's forms are zfsvy.py's and zfsvy_b16.py's; zfsvc.py's reduction is reused.

    python outputs/mp_unified/zfsvy_m4.py selftest
    python outputs/host_ops/pylib_run.py outputs/mp_unified/zfsvy_m4.py run --dataset hotpotqa --nq 20 --blas 2 --lps 2,4 --require-pin
    python outputs/host_ops/pylib_run.py outputs/mp_unified/zfsvy_m4.py run --dataset hotpotqa --blas 2 --lps 2,4 --require-pin
    python outputs/mp_unified/zfsvy_m4.py report

The raw files go to outputs/fsv/timing/m4/raw (raw_smoke for a run with --nq) and are never committed; report.json and report.md in
outputs/fsv/timing/m4 are. The full run refuses to start before --not-before (09:30 IST on 6 November 2026, Correction 4).
"""
import sys

import zfsvy_cut as K                       # first: it imports zfsvy_b16, zfsvy, zfsvx and zfsvc, which set the thread variables before numpy loads

import argparse
import datetime
import gc
import json
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

B = K.B
Y = K.Y
X = K.X
C = K.C
T = K.T
np = K.np
clock = K.clock

VERS = ("mlp", "gnn", "cut")                 # the full MLP, the full GNN, the cut GNN
V16 = (("mlp", "m5"), ("gnn", "m5"), ("gnn", "a7"), ("cut", "m5"), ("cut", "a7"))   # a batch's five forwards: the layer step is the module's (m5) or A7's
OUT_M4 = C.OUT / "m4"
PASSES = (("b2", "BLAS 2 (the record)"), ("b1", "BLAS 1 (beside the record)"))
NOT_BEFORE = "2026-11-06T04:00:00+00:00"     # 09:30 IST on 6 November 2026 (Correction 4; Correction 3 had 00:00 IST on 7 November)
SMOKE_NQ = 32
STAGES = ("compile", "rows", "links", "inputs", "b", "c", "model_only", "end_to_end", "warm_b1")      # the series every ratio is taken over


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] zfsvy_m4: {msg}", flush=True)


def order(j, vs=VERS):
    """The versions' order on question j: the cycle starts at j mod 3, so each leads in turn."""
    k = j % len(vs)
    return vs[k:] + vs[:k]


def pool_check(S, j, A):
    """True when question j's pool, KT columns and I are PKO's rule's own (zfsvc.run's check)."""
    f = C.Z.rule_batch(S.cx, S.ctx, [j])[0]
    return bool(np.array_equal(A.pool, np.union1d(f["B"], f["exp"])) and np.array_equal(A.kt, f["kt"])
                and np.array_equal(A.I, np.asarray(S.cx.inc[j], np.int64)))


def next_flips(S, j, pack):
    """The MLP's float16 NEXT values that differ from the cache's (counted, not failed)."""
    cm = S.boxes["mlp"]["carve"]
    nx = cm.next_X[int(cm.off_np[j]):int(cm.off_np[j + 1])].numpy()
    return int((nx.view(np.uint16) != pack["nxt"].view(np.uint16)).sum())


def score_check(S, v, j, s):
    """zfsvc.check_scores for version v (the cut GNN's inside its view)."""
    if v == "cut":
        with K.cut_view(S):
            return C.check_scores(S, "gnn", j, s)
    return C.check_scores(S, v, j, s)


def build_v(S, v, packs):
    if v == "cut":
        with K.cut_view(S):
            return C.build(S, "gnn", packs)
    return C.build(S, v, packs)


def timed_forward(S, v, f, Bn):
    """One forward at batch Bn on built features, and its time; the cut GNN's inside its view, the view's entry outside the clock."""
    if v == "cut":
        with K.cut_view(S):
            c0 = clock()
            C.forward(S, "gnn", *f, Bn)
            return clock() - c0
    c0 = clock()
    C.forward(S, v, *f, Bn)
    return clock() - c0


def timed_forward16(S, key, step, f):
    """The batch of 16's forward with the layer step 'm5' (the module's) or 'a7' (the batched step), its time and scores."""
    if key == "cut":
        with K.cut_view(S):
            return B.forward_t(S, "gnn", step, *f)
    return B.forward_t(S, key, step, *f)


# ── the passes ───────────────────────────────────────────────────────────────


def cold(S, nq, jit_q):
    """The JIT question, then each question cold at batch 1: stage (a) once (charged to every version), then the three versions in a rotating
    order, each (b) and (c). Returns the rows, the pools, each version's packs and scores."""
    A = C.stage_a(S, jit_q, {})
    C.fill_codes(S, jit_q, A.pool)
    for v in VERS:
        K.run_version(S, v, jit_q, A, {})
    gc.collect()
    rows, As = [], []
    packs = {v: [] for v in VERS}
    scores = {v: [] for v in VERS}
    for j in range(nq):
        Ta = {}
        A = C.stage_a(S, j, Ta)
        C.fill_codes(S, j, A.pool)
        row = {"q": j, "n": int(A.pool.size), "a": Ta}
        for v in order(j):
            Tm = {}
            s, pack, _f = K.run_version(S, v, j, A, Tm)
            row[v] = Tm
            packs[v].append(pack)
            scores[v].append(s)
        rows.append(row)
        As.append(A)
    return rows, As, packs, scores


def warm_b1(S, rows, packs, nq):
    """The forward alone at batch 1 on inputs built beforehand (untimed, one question at a time), the three versions in the question's order."""
    for j in range(nq):
        f = {v: build_v(S, v, packs[v][j:j + 1]) for v in VERS}
        for v in order(j):
            rows[j][v]["warm_b1"] = timed_forward(S, v, f[v], 1)


def warm_b16(S, packs, nb, reps):
    """The warm batches of 16: reps rounds over the nb batches after one that is not counted; each batch's features are built outside the
    clock and its five forwards (V16) timed in an order that rotates. Returns (seconds by variant: reps lists of nb batch times, the first
    round's scores by variant and batch, the number of forwards that took the batched step)."""
    times = {f"{k}:{v}": [] for k, v in V16}
    outs = {name: [None] * nb for name in times}
    n_a7 = 0
    for rep in range(reps + 1):
        row = {name: [] for name in times}
        for g in range(nb):
            sl = slice(g * C.B16, (g + 1) * C.B16)
            fe = {k: build_v(S, k, packs[k][sl]) for k in VERS}
            k0 = (g + rep) % len(V16)
            for key, step in V16[k0:] + V16[:k0]:
                dt, s = timed_forward16(S, key, step, fe[key])
                n_a7 += step == "a7"
                name = f"{key}:{step}"
                row[name].append(dt)
                if outs[name][g] is None:
                    outs[name][g] = s.copy()
                elif not B.same32(outs[name][g], s):
                    raise SystemExit(f"{name} batch {g}: the forward is not the same twice")
        if rep:
            for name in times:
                times[name].append(row[name])
    return times, outs, n_a7


def checks16(S, packs, outs, nb):
    """Untimed: the batched step's scores byte for byte the module's (both GNNs), batch by batch; every variant's scores, question by
    question, against its fit's filed read (within 1e-4, the same top five)."""
    res = {"a7_same_bytes_as_m5": {k: [bool(B.same32(outs[f"{k}:m5"][g], outs[f"{k}:a7"][g])) for g in range(nb)] for k in ("gnn", "cut")}}
    for name, per in outs.items():
        key = name.split(":")[0]
        bad, md = 0, 0.0
        for g in range(nb):
            off = np.r_[0, np.cumsum([int(p["n"]) for p in packs[key][g * C.B16:(g + 1) * C.B16]])]
            for i in range(C.B16):
                c = score_check(S, key, g * C.B16 + i, per[g][off[i]:off[i + 1]])
                bad += not c["ok"]
                md = max(md, c["max_diff"] or 0.0)
        res[name] = {"failing": int(bad), "max_diff": float(md), "questions": nb * C.B16}
    return res


def checks_cold(S, nq, As, packs, scores):
    """Untimed: each version's scores against its fit's filed read, the pool against PKO's rule, the MLP's NEXT flips."""
    sc = {v: [score_check(S, v, j, scores[v][j]) for j in range(nq)] for v in VERS}
    pool_ok = [bool(pool_check(S, j, As[j])) for j in range(nq)]
    flips = [next_flips(S, j, packs["mlp"][j]) for j in range(nq)]
    return sc, pool_ok, flips


def one_pass(S, nq, jit_q, blas, reps):
    """One BLAS count's pass: the cold questions, warm batch 1, warm batches of 16 (as many as nq holds), the checks. Returns what the raw file
    holds (rows, b16, checks, pool_ok, next_flips, kept, stop) and the load around the pass."""
    nb = nq // C.B16
    m = nb * C.B16
    beside = int(blas) != 2
    S.threads = C.set_threads(blas)
    ld = {"cpu_percent_before": C.cpu_load()}
    rows, As, packs, scores = cold(S, nq, jit_q)
    warm_b1(S, rows, packs, nq)
    times, outs, n_a7 = warm_b16(S, {v: packs[v][:m] for v in VERS}, nb, reps) if nb else ({}, {}, 0)
    ld["cpu_percent_after"] = C.cpu_load()
    sc, pool_ok, flips = checks_cold(S, nq, As, packs, scores)
    c16 = checks16(S, {v: packs[v][:m] for v in VERS}, outs, nb) if nb else {}
    bad = [not (pool_ok[j] and all(sc[v][j]["ok"] for v in VERS)) for j in range(nq)]
    a7_same = all(all(x) for x in c16.get("a7_same_bytes_as_m5", {}).values()) if nb else True
    stop = ((sum(bad) / nq > 0.02) or not a7_same) and not beside                  # the beside pass counts, it does not stop
    checks = {"questions": nq, "failing": int(sum(bad)), "share": sum(bad) / nq, "stop": bool(stop),
              "pool_failing": int(sum(not x for x in pool_ok)), "next_flip_values": int(sum(flips)),
              "next_flip_questions": int(sum(x > 0 for x in flips)),
              **{v: {"failing": int(sum(not c["ok"] for c in sc[v])), "max_diff": max((c["max_diff"] or 0.0) for c in sc[v])} for v in VERS},
              "a7_same_bytes": bool(a7_same)}
    return {"rows": rows, "b16": {"times": times, "checks": c16, "batches": nb}, "checks": checks, "pool_ok": pool_ok, "next_flips": flips,
            "kept": [True] * nq if beside else [not b for b in bad], "stop": bool(stop), "load": ld, "n_a7": int(n_a7)}


# ── the job ──────────────────────────────────────────────────────────────────


def too_early(a):
    """True for a full run before --not-before (a smoke run, with --nq, is not bound)."""
    if a.nq:
        return False
    return datetime.datetime.now(datetime.timezone.utc) < datetime.datetime.fromisoformat(a.not_before)


def sha_all():
    return {"zfsvc": T.sha_src(C.__file__), "zfsvt": T.sha_src(T.__file__), "zfsvx": T.sha_src(X.__file__), "zfsvy": T.sha_src(Y.__file__),
            "zfsvy_b16": T.sha_src(B.__file__), "zfsvy_cut": T.sha_src(K.__file__), "zfsvy_m4": T.sha_src(__file__)}


def run(a):
    """M4 for one dataset: setup, the cut fit's model and reference, index-time work, the pin, a smoke pass (first SMOKE_NQ questions, BLAS 2, one
    round of batches of 16) whose failure stops the job before the record, then each BLAS count's pass over the 200 questions. The raw files go
    to outputs/fsv/timing/m4/raw (raw_smoke with --nq)."""
    if too_early(a):
        log(f"{a.dataset}: not before {a.not_before}: nothing started")
        return 3
    C._load()
    K.bind()
    nq = a.nq or T.NQ
    full = nq == T.NQ
    out_dir = OUT_M4 / ("raw" if full else "raw_smoke")
    blases = [int(x) for x in a.blases.split(",")]
    files = {b: out_dir / f"{a.dataset}__b{b}.json" for b in blases}
    for f in files.values():
        if full and f.exists() and not a.overwrite:
            raise SystemExit(f"{f.name} exists: a finished pass is not repeated for its numbers (--overwrite)")
    for k in K.STATS:
        K.STATS[k] = 0
    reps = int(a.reps)
    with Y.installed():
        S = C.setup(a.dataset, nq)
        K.attach_cut(S, a.dataset, nq)
        C.prepass(S, range(nq + 1))
        S.threads = C.set_threads(blases[0])
        load = C.pin_and_load([int(x) for x in a.lps.split(",")])
        if a.require_pin and not load["pin"].get("pinned"):
            raise SystemExit(f"{a.dataset}: not pinned: {load}")
        log(f"pin: {load['pin']}")
        for v in ("mlp", "gnn"):
            log(X.model_line(S, v))
        with K.cut_view(S):
            log("cut " + X.model_line(S, "gnn"))
        zl = X._zg6r().LAYERS
        fits = {"mlp": S.boxes["mlp"]["fit"], "gnn": S.boxes["gnn"]["fit"], "cut": S.cut.fit}
        runs, n_a7, stops = 0, 0, []

        def forms_ran():
            """True when every form ran, in the passes so far (the cut forms once per cold run, A7's step in each batched forward)."""
            ok = bool(B.STATS_B["step_b"] == zl * n_a7 and B.STATS_B["step_orig"] == 0 and Y.forms_ran(nq))
            return ok and bool(K.STATS == {"compile_cut": runs, "walk_zero": runs, "walkf_zero": runs} and K.put_back_ok())

        if full and not a.no_smoke:
            ns = min(int(a.smoke_nq), nq)
            res = one_pass(S, ns, nq, blases[0], 1)
            runs += 1 + ns
            n_a7 += res["n_a7"]
            log(f"{a.dataset} smoke: {ns} questions at BLAS {blases[0]}; checks {res['checks']}; batches of 16 {res['b16']['checks']}")
            if res["stop"] or not forms_ran():
                raise SystemExit(f"{a.dataset}: the smoke pass failed (checks or forms): nothing recorded")
            del res
            gc.collect()
        for blas in blases:
            res = one_pass(S, nq, nq, blas, reps)
            runs += 1 + nq
            n_a7 += res["n_a7"]
            log(f"{a.dataset} BLAS {blas}: {nq} questions timed; checks {res['checks']}")
            log(f"{a.dataset} BLAS {blas}: batches of 16 {res['b16']['checks']}")
            if not forms_ran():
                raise SystemExit(f"the forms did not run: {B.STATS_B} {X.STATS} {Y.STATS} {K.STATS} (cut runs expected {runs})")
            forms = {"fast_index": S.fast_index, "stats": dict(X.STATS), "stats_y": dict(Y.STATS), "stats_b": dict(B.STATS_B),
                     "stats_cut": dict(K.STATS), "cut_runs": runs}
            head = {"dataset": a.dataset, "questions": nq, "batches": res["b16"]["batches"], "blas": int(blas), "reps": reps,
                    "pass": "record: BLAS 2" if blas == 2 else f"beside: BLAS {blas}", "threads": S.threads, "fits": fits,
                    "code_conflicts": S.code_conflicts, "load_and_pin": {**res["load"], "pin": load["pin"]}, "stop": res["stop"],
                    "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "script_sha256": sha_all(), "forms": forms,
                    "smoke": bool(full and not a.no_smoke)}
            T.write_json(files[blas], {"head": head, "rows": res["rows"], "b16": res["b16"], "checks": res["checks"],
                                       "pool_ok": res["pool_ok"], "next_flips": res["next_flips"], "kept": res["kept"]})
            log(f"{a.dataset}: raw written to {out_dir.name}/{files[blas].name}; stop {res['stop']}")
            stops.append(res["stop"])
            del res
            gc.collect()
    return 1 if any(stops) else 0


# ── the report ───────────────────────────────────────────────────────────────


def series3(rows, kept):
    """zfsvc._series (stage (a), the MLP and the full GNN) with the cut GNN's series added in the same form (seconds, over the kept questions)."""
    s = C._series(rows, kept)
    rr = [r for r, k in zip(rows, kept) if k]
    m = {k: np.array([r["cut"][k] for r in rr]) for k in ("compile", "edges", "lean", "links", "next", "assemble", "b_total", "forward", "total")}
    s["cut"] = {"compile": m["compile"], "rows": m["edges"] + m["lean"], "links": m["links"], "next": m["next"], "inputs": m["assemble"],
                "b": m["b_total"], "c": m["forward"], "model_only": m["total"], "end_to_end": s["a"]["total"] + m["total"],
                "warm_b1": np.array([r["cut"].get("warm_b1", np.nan) for r in rr])}
    return s


def led_positions(s):
    """Model-only p50 (ms) of each version when it led, was second and was third in its question's order (beside, deciding nothing)."""
    q = s["q"]
    out = {}
    for i, v in enumerate(VERS):
        pos = (i - q) % len(VERS)                             # 0 = first (the version leads when j mod 3 is its index)
        out[v] = [float(np.median(s[v]["model_only"][pos == p]) * 1000) if (pos == p).any() else None for p in range(len(VERS))]
    return out


def reduce_pass(raw):
    s = series3(raw["rows"], raw["kept"])
    h = raw["head"]
    st = {"a": {k: C._pq(v) for k, v in s["a"].items()}, **{v: {k: C._pq(x) for k, x in s[v].items() if k != "warm_b16"} for v in VERS}}
    t16 = {k: np.asarray(x, np.float64).reshape(-1) / C.B16 for k, x in raw["b16"]["times"].items()}
    w16 = {k: C._pq(x) for k, x in t16.items()}
    r = {"cut_over_mlp": {k: C._ratio(s["cut"][k], s["mlp"][k]) for k in STAGES},
         "gnn_over_mlp": {k: C._ratio(s["gnn"][k], s["mlp"][k]) for k in STAGES},
         "gnn_over_cut": {k: C._ratio(s["gnn"][k], s["cut"][k]) for k in STAGES}}
    if t16:
        r["cut_over_mlp"]["warm_b16"] = C._ratio(t16["cut:a7"], t16["mlp:m5"])
        r["gnn_over_mlp"]["warm_b16"] = C._ratio(t16["gnn:a7"], t16["mlp:m5"])
        r["gnn_over_cut"]["warm_b16"] = C._ratio(t16["gnn:a7"], t16["cut:a7"])
        r["cut_over_mlp"]["warm_b16_module_step"] = C._ratio(t16["cut:m5"], t16["mlp:m5"])
        r["gnn_over_mlp"]["warm_b16_module_step"] = C._ratio(t16["gnn:m5"], t16["mlp:m5"])
        r["gnn_over_cut"]["warm_b16_module_step"] = C._ratio(t16["gnn:m5"], t16["cut:m5"])
    h2 = s["q"].size // 2
    ms = lambda x: float(np.median(x) * 1000)                       # noqa: E731
    noise = {"model_only_first_half_second_half": {v: [ms(s[v]["model_only"][:h2]), ms(s[v]["model_only"][h2:])] for v in VERS},
             "a_first_half_second_half": [ms(s["a"]["total"][:h2]), ms(s["a"]["total"][h2:])],
             "model_only_led_first_second_third": led_positions(s),
             "p95_over_p50_model_only": {v: float(np.percentile(s[v]["model_only"], 95) / np.median(s[v]["model_only"])) for v in VERS},
             "p95_over_p50_a": float(np.percentile(s["a"]["total"], 95) / np.median(s["a"]["total"]))}
    return {"questions": h["questions"], "kept": int(s["q"].size), "batches": h["batches"], "reps": h["reps"], "blas": h.get("blas"),
            "pass": h.get("pass"), "utc": h["utc"], "fits": h["fits"], "threads": h["threads"], "code_conflicts": h["code_conflicts"],
            "script_sha256": h["script_sha256"], "forms": h["forms"], "smoke": h.get("smoke"),
            "pool_rows": {"p50": float(np.median(s["n"])), "max": int(s["n"].max())}, "checks": raw["checks"],
            "b16_checks": raw["b16"]["checks"], "load": C._load_summary(h), "stages_ms": st, "warm_b16_ms": w16, "ratios": r,
            "a_share_of_end_to_end": {v: st["a"]["total"]["p50"] / st[v]["end_to_end"]["p50"] for v in VERS}, "noise": noise}


def rt(d):
    return f"{d['ratio']:.2f} [{d['lo']:.2f}, {d['hi']:.2f}]"


def pq(d):
    return f"{d['p50']:.1f} / {d['p95']:.1f}"


NAMES = {"mlp": "MLP (zrckpn)", "gnn": "GNN, full (zg6r)", "cut": "GNN, cut (zg6r L2)"}


def md_of(rep):
    L = []
    w = L.append
    w("# FSV timing, Family 2 (M4)")
    w("")
    w("Family 2 is the cut zg6r (without topo_STRUCT, depth_STRUCT, WALK and WALKF; level L2) against the full zrckpn; Family 1, the full zg6r "
      "against the full zrckpn, is timed again beside it in the same job. All three versions run at the forms of amendments A2, A4, A6 and A7 "
      "(`docs/FSV_FAST_VERSIONS.md` sections 10 and 12 to 14) and, for the cut GNN, section 15 and its corrections 2 and 3. Times are ms per "
      "question at batch 1 on the host's CPU, cold unless labelled warm; each cell is p50 / p95 over the kept questions. Stage (a), the shared "
      "pool, is charged in full to every version: model-only = (b) + (c), end to end = (a) + (b) + (c). The headline is end to end; "
      "model-only is beside it. The three versions run on each question in an order that rotates, so each leads in turn.")
    w("")
    w(f"Ratios are the first version / the second of the p50s, paired by question (by batch and round for the warm batches of 16), with the "
      f"2.5th and 97.5th percentiles over {C.NBOOT:,} resamples with replacement (`numpy.random.default_rng(0)`). A ratio under 1 means the first "
      "is faster. Nothing here is a call: the bars of section 5 are applied by a later file.")
    for tag, P in rep["passes"].items():
        D = P["datasets"]
        w("")
        w(f"## {P['label']}")
        w("")
        w("### Ratios (first / second, p50s, paired)")
        w("")
        w("| dataset | pair | end to end | model-only | (b) | (c) | warm b1 | warm b16 (A7's step) | warm b16 (module's step) |")
        w("|---|---|---|---|---|---|---|---|---|")
        for ds, d in D.items():
            for key, label in (("cut_over_mlp", "cut GNN / MLP (Family 2)"), ("gnn_over_mlp", "full GNN / MLP (Family 1)"),
                               ("gnn_over_cut", "full GNN / cut GNN (the cut's gain)")):
                r = d["ratios"][key]
                w(f"| {ds} | {label} | {rt(r['end_to_end'])} | {rt(r['model_only'])} | {rt(r['b'])} | {rt(r['c'])} | {rt(r['warm_b1'])} | "
                  f"{rt(r['warm_b16']) if 'warm_b16' in r else 'n/a'} | "
                  f"{rt(r['warm_b16_module_step']) if 'warm_b16_module_step' in r else 'n/a'} |")
        w("")
        w("### Cold, per stage (ms, p50 / p95)")
        w("")
        w("| dataset | kept | pool rows p50 | (a) | version | (b) | (c) | model-only | end to end |")
        w("|---|---|---|---|---|---|---|---|---|")
        for ds, d in D.items():
            st = d["stages_ms"]
            for i, v in enumerate(VERS):
                lead = f"{ds} | {d['kept']} / {d['questions']} | {d['pool_rows']['p50']:.0f} | {pq(st['a']['total'])}" if i == 0 else " | | |"
                w(f"| {lead} | {NAMES[v]} | " + " | ".join(pq(st[v][k]) for k in ("b", "c", "model_only", "end_to_end")) + " |")
        w("")
        w("### Parts of (b) (p50 ms) and the cut's gain per part (full GNN / cut GNN, paired)")
        w("")
        w("| dataset | version | compile | rows | links | NEXT | inputs | gain: compile | rows | links | inputs |")
        w("|---|---|---|---|---|---|---|---|---|---|---|")
        for ds, d in D.items():
            st, g = d["stages_ms"], d["ratios"]["gnn_over_cut"]
            for i, v in enumerate(VERS):
                parts = [f"{st[v][k]['p50']:.2f}" for k in ("compile", "rows", "links", "next", "inputs")]
                gains = [rt(g[k]) for k in ("compile", "rows", "links", "inputs")] if v == "cut" else ["", "", "", ""]
                w(f"| {ds if i == 0 else ''} | {NAMES[v]} | " + " | ".join(parts + gains) + " |")
        w("")
        w("### Stage (a), its share of each end-to-end time")
        w("")
        w("| dataset | (a) p50 ms | share of MLP | share of full GNN | share of cut GNN |")
        w("|---|---|---|---|---|")
        for ds, d in D.items():
            sh = d["a_share_of_end_to_end"]
            w(f"| {ds} | {d['stages_ms']['a']['total']['p50']:.1f} | " + " | ".join(f"{100 * sh[v]:.0f}%" for v in VERS) + " |")
        w("")
        w("### Warm (the forward alone, per question; labelled warm)")
        w("")
        w("| dataset | version | b1 | b16, module's step | b16, A7's step |")
        w("|---|---|---|---|---|")
        for ds, d in D.items():
            st, w16 = d["stages_ms"], d["warm_b16_ms"]
            for i, v in enumerate(VERS):
                m5 = w16.get(f"{v}:m5")
                a7 = w16.get(f"{v}:a7")
                w(f"| {ds if i == 0 else ''} | {NAMES[v]} | {pq(st[v]['warm_b1'])} | {pq(m5) if m5 else 'n/a'} | {pq(a7) if a7 else 'n/a'} |")
        w("")
        w("### Checks, the forms that ran, pin and the host's load (1 s samples before and after the timed pass)")
        w("")
        w("| dataset | questions failing a check | pool failing | score checks failing MLP / GNN / cut | max score diff MLP / GNN / cut | "
          "NEXT values differing | A7's step = module's (batches) | cut forms ran (runs) | pinned | load before: pinned LPs / all mean / max | "
          "load after: pinned LPs / all mean / max |")
        w("|---|---|---|---|---|---|---|---|---|---|---|")
        for ds, d in D.items():
            c, ld, c16 = d["checks"], d["load"], d["b16_checks"]
            f = lambda x: "n/a" if x is None else f"{x['pinned_lps_mean']:.0f}% / {x['all_lps_mean']:.0f}% / {x['all_lps_max']:.0f}%"   # noqa: E731
            same = c16.get("a7_same_bytes_as_m5", {})
            ns = "; ".join(f"{k} {sum(x)} / {len(x)}" for k, x in same.items()) or "n/a"
            w(f"| {ds} | {c['failing']} of {c['questions']} | {c['pool_failing']} | "
              f"{c['mlp']['failing']} / {c['gnn']['failing']} / {c['cut']['failing']} | "
              f"{c['mlp']['max_diff']:.2g} / {c['gnn']['max_diff']:.2g} / {c['cut']['max_diff']:.2g} | {c['next_flip_values']} | {ns} | "
              f"{d['forms']['stats_cut']} ({d['forms']['cut_runs']}) | {ld['pinned']} | {f(ld['before'])} | {f(ld['after'])} |")
        w("")
        w("### Noise, beside (p50 ms; deciding nothing)")
        w("")
        w("| dataset | (a): first / second half | version | model-only: first / second half | led first / second / third | p95 / p50 of model-only |")
        w("|---|---|---|---|---|---|")
        for ds, d in D.items():
            n = d["noise"]
            for i, v in enumerate(VERS):
                h2 = n["model_only_first_half_second_half"][v]
                lp = n["model_only_led_first_second_third"][v]
                fmt = " / ".join("n/a" if x is None else f"{x:.0f}" for x in lp)
                lead = f"{ds} | {n['a_first_half_second_half'][0]:.0f} / {n['a_first_half_second_half'][1]:.0f}" if i == 0 else " |"
                w(f"| {lead} | {NAMES[v]} | {h2[0]:.0f} / {h2[1]:.0f} | {fmt} | {n['p95_over_p50_model_only'][v]:.2f} |")
    w("")
    w("## Caveats")
    w("")
    w("- The host is shared: other jobs run on it while a timed job does, so every time carries that noise. The process is pinned to two logical "
      "processors (its own affinity, EcoQoS opted out, above-normal priority), the three versions rotate in leading each question, and the load of "
      "every logical processor is sampled for 1 s before and after the pass; the other jobs are not pinned and may share those processors.")
    w("- The first-stage retrieval (the question's encoding, the dense and SPLADE lists) is not timed; PKO's prepared lists are read before timing "
      "as index-time data.")
    w("- Stage (a) is PKO's rule's own calls on one question at batch 1 on one thread, with the walks on P4b's records and only the orders the "
      "pool and the KT columns read (amendment A4); the layer step, stage (b)'s links, the MLP's NEXT columns and the encoder's segment z-scores "
      "are A4's and A6's (eager torch, one thread); OpenBLAS runs on 2 threads in the record and on 1 in the beside pass.")
    w("- The cut GNN does not compute topo_STRUCT, depth_STRUCT, WALK or WALKF (section 15): the topology, D and depth-basis laps of the compile "
      "and the two walk calls of the rows are left out, its pick has five blocks and its encoder is the cut fit's own. The forms that ran are "
      "counted in the checks table.")
    w("- Warm times are the forward alone on inputs built beforehand (one question or one batch at a time, outside the clock); b16 is the time of "
      "a batch of the first 192 questions over 16, over the rounds, and its ratios are paired by batch and round. The warm batches of 16 are "
      "timed with the module's layer step and with A7's batched step (byte for byte the module's); the MLP has one step.")
    w("")
    return "\n".join(L)


def report(a):
    """The M4 reduction from outputs/fsv/timing/m4/raw: per dataset and BLAS pass the cold stages of the three versions, the warm times, the
    ratios (cut GNN / MLP; full GNN / MLP; full GNN / cut GNN), the checks, the forms that ran, the load and the noise. Writes report.json and
    report.md (masked before they are committed, as M5's)."""
    raw_dir = Path(a.raw_dir) if a.raw_dir else OUT_M4 / "raw"
    out_dir = Path(a.out_dir) if a.out_dir else OUT_M4
    rep = {"label": "M4 (Family 2 and Family 1)", "amendment": "docs/FSV_FAST_VERSIONS.md section 15 (correction 3)",
           "recipe": {"cells": "p50 and p95 in ms over the kept questions (numpy.percentile, linear)",
                      "end_to_end": "(a) + (b) + (c), stage (a) charged in full to every version",
                      "ratio": f"first / second of the p50s, paired; 2.5th and 97.5th percentiles over {C.NBOOT} resamples (numpy default_rng(0))",
                      "warm": "the forward alone; b16 = the batch's time / 16, over the batches and rounds"},
           "passes": {}}
    for tag, label in PASSES:
        ds = {}
        for name in T.DATASETS:
            f = raw_dir / f"{name}__{tag}.json"
            if f.exists():
                ds[name] = reduce_pass(json.loads(f.read_text(encoding="utf-8")))
        if ds:
            rep["passes"][tag] = {"label": label, "datasets": ds}
    if not rep["passes"]:
        raise SystemExit(f"M4 report: no <dataset>__b2.json or __b1.json in {raw_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    T.write_json(out_dir / "report.json", rep)
    (out_dir / "report.md").write_text(md_of(rep), encoding="utf-8", newline="\n")
    log(f"M4 report: {out_dir / 'report.json'}, report.md; passes {list(rep['passes'])}")
    return 0


# ── the check without data ───────────────────────────────────────────────────


def selftest_order():
    """Each version leads equally often, every order is a permutation, and the cycle is the same every three questions."""
    bad = 0
    lead = {v: 0 for v in VERS}
    for j in range(60):
        o = order(j)
        bad += sorted(o) != sorted(VERS) or o != order(j + 3)
        lead[o[0]] += 1
    bad += len(set(lead.values())) != 1
    log(f"order: leads {lead}; {bad} differ")
    return int(bad)


def fake_world(nq):
    """A world with no data for one_pass: S holds fake models for the cut view, the stages and the forwards are functions of the question and
    the version (the cut faster than the full GNN, the full GNN slower than the MLP) and every check passes but one (the cut GNN, question 5)."""
    ms = {"mlp": 0.010, "gnn": 0.012, "cut": 0.009}
    fb = {"mlp": 0.002, "gnn": 0.005, "cut": 0.0045}
    S = SimpleNamespace(models={"mlp": "m", "gnn": "full"}, blocks={"mlp": [1], "gnn": ["f"]}, boxes={"mlp": 1, "gnn": "fullbox"},
                        cut=SimpleNamespace(model="cut", blocks=["c"], box="cutbox"), threads=None)
    ns = [100 + 3 * j for j in range(nq + 1)]

    def stage_a(S_, j, T_):
        T_.update(a_pool=0.050, a_base=0.010, a_link=0.005, a_rule=0.035, a_walk=0.01, a_order=0.01, a_kt=0.01)
        return SimpleNamespace(pool=np.arange(ns[j]), kt=None, I=None)

    def run_version(S_, v, j, A, T_):
        T_.update(compile=ms[v] * 0.4, edges=ms[v] * 0.02, lean=ms[v] * 0.1, links=ms[v] * 0.01, next=0.001 if v == "mlp" else 0.0,
                  assemble=ms[v] * 0.05, forward=fb[v], b_total=ms[v] - fb[v], total=ms[v])
        if v == "cut":
            with K.cut_view(S_):
                view = S_.boxes["gnn"]
        else:
            view = S_.boxes["gnn"]
        if v != "mlp" and view != ("cutbox" if v == "cut" else "fullbox"):
            raise SystemExit("the fake's view is wrong")
        return np.full(ns[j], float(j), np.float32), {"n": ns[j], "nxt": np.zeros((ns[j], 2), np.float16)}, None

    def fwd1(S_, v, f, Bn):
        return fb[v] * 0.9

    return S, ns, stage_a, run_version, fwd1, ms, fb


def selftest_pass(tmp):
    """one_pass, write and report on a fake world (no clocks in the results): the rows, the kept mask, the ratios and the report's bytes."""
    bad = 0
    nq, reps = 64, 2                                                        # one failing question of 64 is under the 2% that stops a record pass
    S, ns, stage_a, run_version, fwd1, ms, fb = fake_world(nq)
    M = sys.modules[__name__]
    saved = {k: getattr(M, k) for k in ("timed_forward", "timed_forward16", "build_v", "score_check", "pool_check", "next_flips")}
    saved_c = {k: getattr(C, k) for k in ("stage_a", "fill_codes", "set_threads", "cpu_load")}
    saved_k = K.run_version

    def bv(S_, v, packs):                                                   # the fake's "features": the rows in the batch (None for one question)
        if len(packs) == 1:
            return (None, 1, None)
        return (sum(int(p["n"]) for p in packs), len(packs), None)

    def sc(S_, v, j, s):
        ok = not (v == "cut" and j == 5)
        return {"ok": ok, "max_diff": 1e-6, "top5_same": ok}
    try:
        M.timed_forward, M.build_v, M.score_check = fwd1, bv, sc
        M.timed_forward16 = lambda S_, key, step, f: (fb[key] * 16 * (0.8 if step == "a7" else 1.0), np.full(f[0], 0.5, np.float32))
        M.pool_check = lambda S_, j, A: True
        M.next_flips = lambda S_, j, pack: 0
        C.stage_a, C.fill_codes, C.set_threads, C.cpu_load = stage_a, (lambda S_, j, pool: 0), (lambda blas: {"blas_declared": blas}), (lambda: [10.0, 20.0])
        K.run_version = run_version
        for blas in (2, 1):
            res = one_pass(S, nq, nq, blas, reps)
            tag = f"b{blas}"
            if len(res["rows"]) != nq or any(set(r) != {"q", "n", "a", "mlp", "gnn", "cut"} for r in res["rows"]):
                bad += 1
                log("rows: the wrong shape")
            if any("warm_b1" not in r[v] for r in res["rows"] for v in VERS):
                bad += 1
                log("rows: a version has no warm b1")
            want_kept = [True] * nq if blas != 2 else [j != 5 for j in range(nq)]
            if res["kept"] != want_kept or res["checks"]["failing"] != 1 or res["stop"]:
                bad += 1
                log(f"pass {tag}: kept {sum(res['kept'])}, failing {res['checks']['failing']}, stop {res['stop']}")
            if res["n_a7"] != 2 * (reps + 1) * (nq // C.B16):
                bad += 1
                log(f"pass {tag}: {res['n_a7']} batched forwards")
            if not res["b16"]["checks"]["a7_same_bytes_as_m5"]["cut"] or not res["checks"]["a7_same_bytes"]:
                bad += 1
                log(f"pass {tag}: the batched step is not the module's in the fake")
            head = {"dataset": "musique", "questions": nq, "batches": res["b16"]["batches"], "blas": blas, "reps": reps,
                    "pass": "record: BLAS 2" if blas == 2 else f"beside: BLAS {blas}", "threads": {"blas_declared": blas},
                    "fits": {"mlp": "m", "gnn": "g", "cut": "c"}, "code_conflicts": 0,
                    "load_and_pin": {**res["load"], "pin": {"pinned": True, "affinity_lps": [0, 1]}}, "stop": res["stop"],
                    "utc": "2026-11-06T00:00:00Z", "script_sha256": {}, "forms": {"stats_cut": {"compile_cut": 1}, "cut_runs": 1}, "smoke": False}
            T.write_json(Path(tmp) / "raw" / f"musique__{tag}.json", {"head": head, "rows": res["rows"], "b16": res["b16"],
                                                                      "checks": res["checks"], "pool_ok": res["pool_ok"],
                                                                      "next_flips": res["next_flips"], "kept": res["kept"]})
    finally:
        for k, v in saved.items():
            setattr(M, k, v)
        for k, v in saved_c.items():
            setattr(C, k, v)
        K.run_version = saved_k
    outs = []
    for name in ("one", "two"):
        report(SimpleNamespace(raw_dir=str(Path(tmp) / "raw"), out_dir=str(Path(tmp) / name)))
        outs.append(((Path(tmp) / name / "report.json").read_bytes(), (Path(tmp) / name / "report.md").read_bytes()))
    if outs[0] != outs[1]:
        bad += 1
        log("report: two reductions of the same raw files differ")
    rep = json.loads(outs[0][0])
    d = rep["passes"]["b2"]["datasets"]["musique"]
    r = d["ratios"]
    want = {"cut_over_mlp": ms["cut"] / ms["mlp"], "gnn_over_mlp": ms["gnn"] / ms["mlp"], "gnn_over_cut": ms["gnn"] / ms["cut"]}
    for key, wv in want.items():
        got = r[key]["model_only"]["ratio"]
        if abs(got - wv) > 1e-9 or not (r[key]["model_only"]["lo"] <= got <= r[key]["model_only"]["hi"]):
            bad += 1
            log(f"report: {key} model-only ratio {got} against {wv}")
    got16 = r["gnn_over_cut"]["warm_b16"]["ratio"]
    if abs(got16 - fb["gnn"] / fb["cut"]) > 1e-9 or abs(r["cut_over_mlp"]["warm_b16_module_step"]["ratio"] - fb["cut"] / fb["mlp"]) > 1e-9:
        bad += 1
        log("report: a warm b16 ratio")
    if d["kept"] != nq - 1 or rep["passes"]["b1"]["datasets"]["musique"]["kept"] != nq:
        bad += 1
        log(f"report: kept {d['kept']} (record) and {rep['passes']['b1']['datasets']['musique']['kept']} (beside)")
    e2e = (0.050 + 0.010 + 0.005 + 0.035 + ms["cut"]) / (0.050 + 0.010 + 0.005 + 0.035 + ms["mlp"])
    if abs(r["cut_over_mlp"]["end_to_end"]["ratio"] - e2e) > 1e-9:
        bad += 1
        log(f"report: end-to-end ratio {r['cut_over_mlp']['end_to_end']['ratio']} against {e2e}")
    md = outs[0][1].decode("utf-8")
    if ("| dataset | pair | end to end | model-only |" not in md or "\r" in md or "## BLAS 2 (the record)" not in md
            or "## BLAS 1" not in md):
        bad += 1
        log("report.md: the headline's column, line ends or sections")
    gone = False
    try:
        report(SimpleNamespace(raw_dir=str(Path(tmp) / "none"), out_dir=str(Path(tmp) / "three")))
    except SystemExit:
        gone = True
    if not gone:
        bad += 1
        log("report: missing raw files did not stop it")
    log(f"fake passes: rows, kept, stop, the batched forwards, the files, the ratios and the report's bytes; {bad} differ")
    return int(bad)


def selftest_api():
    """The signatures this module calls in the modules it imports (a change in them would show here, not on the host)."""
    import inspect
    want = [(C.stage_a, "S j T_"), (C.fill_codes, "S j pool"), (C.prepass, "S qs"), (C.build, "S key packs"),
            (C.forward, "S key feats nq base_z Bn"), (C.run_model, "S key j A T_"), (C.check_scores, "S key j s"),
            (C.set_threads, "blas"), (C.pin_and_load, "lps"), (C.setup, "ds_name nq full_rule"), (C._series, "rows kept"),
            (C._ratio, "g m"), (C._pq, "x"), (C._load_summary, "h"), (B.forward_t, "S key v feats nqt base_z"), (B.same32, "a b"),
            (K.run_version, "S v j A T_"), (K.attach_cut, "S ds nq"), (K.cut_view, "S"), (K.put_back_ok, ""), (K.bind, ""),
            (Y.forms_ran, "nq b16"), (Y.installed, "out_dir"), (X.model_line, "S key"), (X._zg6r, ""), (T.write_json, "p obj"),
            (T.sha_src, "p")]
    bad = 0
    for fn, names in want:
        got = " ".join(inspect.signature(fn).parameters)
        if got != names:
            bad += 1
            log(f"api: {fn.__module__}.{fn.__name__} takes ({got}), not ({names})")
    bad += (C.B16, C.NBOOT, T.NQ) != (16, 2000, 200)
    bad += tuple(B.V16)[:3] != (("mlp", "m5"), ("gnn", "m5"), ("gnn", "a7")) or V16[:3] != tuple(B.V16)
    log(f"api: {len(want)} signatures, the batch size, the resamples and A7's variants; {bad} differ")
    return int(bad)


def selftest_guard():
    """The full run is refused before --not-before and a smoke run (--nq) is not; both parse."""
    bad = 0
    early = SimpleNamespace(nq=0, not_before=(datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=1)).isoformat())
    late = SimpleNamespace(nq=0, not_before=(datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=1)).isoformat())
    smoke = SimpleNamespace(nq=20, not_before=early.not_before)
    bad += (not too_early(early)) + too_early(late) + too_early(smoke)
    bad += datetime.datetime.fromisoformat(NOT_BEFORE).astimezone(datetime.timezone(datetime.timedelta(hours=5, minutes=30))).isoformat()[:19] != "2026-11-06T09:30:00"
    log(f"guard: before {NOT_BEFORE} (09:30 IST on 6 November) a full run starts nothing; {bad} differ")
    return int(bad)


def selftest(a):
    C._load()
    K.bind()
    bad = {"order": selftest_order(), "guard": selftest_guard(), "api": selftest_api()}
    with tempfile.TemporaryDirectory() as tmp:
        bad["fake"] = selftest_pass(tmp)
    log("SELFTEST " + ("PASS" if not any(bad.values()) else f"FAIL {bad}"))
    return 0 if not any(bad.values()) else 1


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("selftest", "run", "report"))
    ap.add_argument("--dataset", choices=T.DATASETS, default="musique")
    ap.add_argument("--blas", default=C.BLAS, help="OpenBLAS threads at import (zfsvc reads it from argv too); the passes set their own")
    ap.add_argument("--blases", default="2,1", help="run: the BLAS counts, in this order (the record first)")
    ap.add_argument("--nq", type=int, default=0, help="run: a smoke run of this many questions (raw_smoke), 0 = the 200")
    ap.add_argument("--lps", default="2,4", help="run: the logical processors the process is pinned to")
    ap.add_argument("--reps", type=int, default=5, help="run: rounds over the batches of 16 (one more is not counted)")
    ap.add_argument("--smoke-nq", dest="smoke_nq", type=int, default=SMOKE_NQ, help="run: the questions of the smoke pass before the record")
    ap.add_argument("--no-smoke", dest="no_smoke", action="store_true")
    ap.add_argument("--require-pin", dest="require_pin", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--not-before", dest="not_before", default=NOT_BEFORE, help="run: a full run starts nothing before this time")
    ap.add_argument("--raw-dir", dest="raw_dir", default="", help="report: the raw files (default outputs/fsv/timing/m4/raw)")
    ap.add_argument("--out-dir", dest="out_dir", default="", help="report: where report.json and report.md go (m4)")
    a = ap.parse_args(argv)
    return {"selftest": selftest, "run": run, "report": report}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
