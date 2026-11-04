"""FSV amendment A7 (docs/FSV_FAST_VERSIONS.md section 14): a batched layer step for the GNN's warm batches of 16. `share` is the
development job (what share of each model's forward at batch 16 is the GNN's layer step, zg6r.g6r_step, three calls, and what
the MLP's forward is made of; no timed number), `selftest` the candidate step against the module's on random layers, `run` and
`report` the timed re-time of the warm batches of 16 with A4's, A6's and A7's forms. zfsvc.py, zfsvx.py and zfsvy.py are
imported, not edited: the batched step is installed only inside this script's own timed forwards, and its outputs are byte for
byte the module's.

    python outputs/mp_unified/zfsvy_b16.py selftest
    python outputs/host_ops/pylib_run.py outputs/mp_unified/zfsvy_b16.py share --dataset hotpotqa --questions 32 --blas 2 --lp 8
    python outputs/host_ops/pylib_run.py outputs/mp_unified/zfsvy_b16.py run --dataset hotpotqa --blas 2 --lps 2,4 --require-pin
    python outputs/mp_unified/zfsvy_b16.py report
"""
import sys

import zfsvy as Y                       # first: it imports zfsvx and zfsvc, which set the thread variables before numpy loads

import argparse
import gc
import json
import math
import os
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

X = Y.X
C = X.C
np = C.np
clock = C.clock

STATS_B = {"step_b": 0, "step_orig": 0}
FAM_GS = ("aten::index_add_", "aten::index", "aten::index_select", "aten::gather", "aten::scatter_add_", "aten::scatter_",
          "aten::index_put_", "aten::index_copy")                          # the gathers and scatters
FAM_MM = ("aten::addmm", "aten::mm", "aten::bmm")                             # the matrix products
VARS = {"gnn": ("m3", "m5", "form"), "mlp": ("m3", "m5")}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] zfsvy_b16: {msg}", flush=True)


def ms(v):
    return float(np.median(np.asarray(v, np.float64))) * 1000.0


# ── the candidate form: A4's layer step for a batch ──────────────────────────


def g6r_step_b(m, k, h, xb, w, u, v, f, at, deg, qv, nq, B, te, codes, cte):
    """zfsvx.g6r_step_x for B questions (a batch of 16 in the warm passes): the rows of the questions are runs of nq, every edge
    lies inside its question, so the attention logit's per-node dot product takes its question's context from c[nq] (the
    original's per-edge c[qv] gathered at the source row: the same float32 values), and the two scatter-adds are zfsvx's one
    numba pass over the edges in their order (the edges of different questions touch different rows, so each row's sum is the
    original's). Every other op is the original's, in its order. B == 1 is A4's step; a typed edge, another dtype, a gradient or
    an edge that leaves its question takes the original's (counted)."""
    R, torch = X._zg6r(), C.torch
    if B == 1:
        return X.g6r_step_x(m, k, h, xb, w, u, v, f, at, deg, qv, nq, B, te, codes, cte)
    if (te.numel() or h.dtype != torch.float32 or torch.is_grad_enabled() or nq.numel() != h.shape[0]
            or not bool((nq[u] == qv).all())):
        STATS_B["step_orig"] += 1
        return X._ORIG["g6r_step"](m, k, h, xb, w, u, v, f, at, deg, qv, nq, B, te, codes, cte)
    Fn, D, FAMS, LCLIP = torch.nn.functional, R.D, R.FAMS, R.LCLIP
    wm, wa, fb, ws, wmean, wagg, wx, b = (getattr(m, n) for n in R.G6.G6_LAYER[k])
    N = h.shape[0]
    c = torch.zeros(B, D, dtype=h.dtype).index_add_(0, nq, w * h)
    hm = Fn.linear(h, wm)
    ln = (Fn.linear(h, wa) * c[nq]).sum(1) / math.sqrt(D)
    lg = (ln[u] + fb[f]).clamp(-LCLIP, LCLIP)
    s, ex = torch.sigmoid(lg), torch.exp(lg)
    mean_a = torch.zeros(N * FAMS, D, dtype=h.dtype)
    num = torch.zeros(N, D, dtype=h.dtype)
    X.gkernels()["edge_acc"](hm.numpy(), u.numpy(), f.numpy(), v.numpy(), at.numpy(), s.numpy(), ex.numpy(), mean_a.numpy(),
                             num.numpy())
    den = torch.zeros(N, dtype=h.dtype).index_add_(0, v, ex)
    mean = mean_a.view(N, FAMS, D) / deg
    att = num / den.clamp_min(1e-30).unsqueeze(1)
    pre = Fn.linear(h, ws) + Fn.linear(mean.reshape(N, FAMS * D), wmean) + Fn.linear(att, wagg) + Fn.linear(xb, wx) + b
    STATS_B["step_b"] += 1
    return h + Fn.gelu(Fn.layer_norm(pre, (D,)))


def same32(a, b):
    """Two float32 arrays equal bit for bit (NaN and signed zero included)."""
    a, b = np.ascontiguousarray(a), np.ascontiguousarray(b)
    return a.shape == b.shape and bool(np.array_equal(a.view(np.uint32), b.view(np.uint32)))


# ── the check without data ───────────────────────────────────────────────────


def fake_layers(rng, R):
    """An object with the module's tensors of every layer (zg6.G6_LAYER's names) at random values, and zg6r's four (unused with no
    typed edge)."""
    torch = C.torch
    D, FAMS = R.D, R.FAMS
    nb = int(R.G6.NB)
    m = SimpleNamespace()
    for k in range(R.LAYERS):
        shapes = [(FAMS * D, D), (D, D), (FAMS,), (D, D), (D, FAMS * D), (D, D), (D, nb), (D,)]
        for n, sh in zip(R.G6.G6_LAYER[k], shapes):
            setattr(m, n, torch.from_numpy((rng.standard_normal(sh) * 0.3).astype(np.float32)))
        for n in R.G6R_LAYER[k]:
            setattr(m, n, torch.zeros(1))
    return m


def fake_batch(rng, R, B, nodes, edges, leave=0):
    """B questions' rows (runs of nq), the edges inside their questions in a random order (leave of them from one question to
    another), as zfsvc.build lays a batch out, and the layer loop's own arguments for them."""
    torch = C.torch
    D, FAMS = R.D, R.FAMS
    ns = rng.integers(nodes // 2, nodes, B)
    ne = rng.integers(edges // 2, edges, B)
    off = np.r_[0, np.cumsum(ns)]
    N = int(off[-1])
    nqa = np.repeat(np.arange(B), ns)
    eq = np.repeat(np.arange(B), ne)
    u = off[eq] + rng.integers(0, ns[eq])
    v = off[eq] + rng.integers(0, ns[eq])
    if leave:
        v[:leave] = rng.integers(0, N, leave)
    f = rng.integers(0, FAMS, u.size)
    nq = torch.from_numpy(nqa.astype(np.int64))
    u, v, f = (torch.from_numpy(x.astype(np.int64)) for x in (u, v, f))
    h = torch.from_numpy(rng.standard_normal((N, D)).astype(np.float32))
    xb = torch.from_numpy(rng.standard_normal((N, int(R.G6.NB))).astype(np.float32))
    w = torch.from_numpy((rng.random((N, 1)) + 0.01).astype(np.float32))
    at = v * FAMS + f
    deg = torch.zeros(N * FAMS, dtype=h.dtype).index_add_(0, at, torch.ones(u.numel(), dtype=h.dtype)).view(N, FAMS, 1).clamp_min(1.0)
    return dict(h=h, xb=xb, w=w, u=u, v=v, f=f, at=at, deg=deg, qv=nq[v], nq=nq, B=B,
                te=torch.zeros(0, dtype=torch.long), codes=None, cte=None)


def selftest(a):
    """g6r_step_b against zg6r.g6r_step on random layers and graphs: B in 2, 5 and 16, threads 1 and 2, small and the size of a
    warm batch (about 2000 rows and 20000 edges a question), every layer; the outputs byte for byte; the fallback for an edge
    that leaves its question; and the two steps' times (a development number on this machine)."""
    C._load()
    torch = C.torch
    R = X._zg6r()
    rng = np.random.default_rng(int(a.seed))
    m = fake_layers(rng, R)
    X.gkernels()
    X._ORIG.setdefault("g6r_step", R.g6r_step)
    orig = X._ORIG["g6r_step"]
    bad, trials = 0, 0
    sizes = [(2, 200, 1500), (5, 300, 3000), (16, 400, 4000)] * int(a.trials) + [(16, 2000, 20000)]
    for threads in (1, 2):
        torch.set_num_threads(threads)
        for B, nodes, edges in sizes:
            d = fake_batch(rng, R, B, nodes, edges)
            for k in range(R.LAYERS):
                args = (m, k, d["h"], d["xb"], d["w"], d["u"], d["v"], d["f"], d["at"], d["deg"], d["qv"], d["nq"], d["B"],
                        d["te"], d["codes"], d["cte"])
                with torch.no_grad():
                    t0 = clock()
                    want = orig(*args)
                    t1 = clock()
                    got = g6r_step_b(*args)
                    t2 = clock()
                trials += 1
                if not same32(want.numpy(), got.numpy()):
                    bad += 1
                    log(f"DIFFERS: threads {threads} B {B} nodes {int(d['h'].shape[0])} edges {int(d['u'].numel())} layer {k}")
            if edges == 20000:
                log(f"threads {threads} B {B}, {int(d['h'].shape[0])} rows, {int(d['u'].numel())} edges, last layer: module's step "
                    f"{(t1 - t0) * 1000:.0f} ms, batched form {(t2 - t1) * 1000:.0f} ms")
    d = fake_batch(rng, R, 4, 300, 3000, leave=5)                           # an edge leaves its question: the original's step
    args = (m, 0, d["h"], d["xb"], d["w"], d["u"], d["v"], d["f"], d["at"], d["deg"], d["qv"], d["nq"], d["B"], d["te"], d["codes"],
            d["cte"])
    before = STATS_B["step_orig"]
    with torch.no_grad():
        ok_fb = same32(orig(*args).numpy(), g6r_step_b(*args).numpy()) and STATS_B["step_orig"] == before + 1
    log(f"{trials} steps compared, {bad} differ; the fallback for an edge that leaves its question: {'ok' if ok_fb else 'WRONG'}; "
        f"stats {STATS_B}")
    log("SELFTEST " + ("PASS" if bad == 0 and ok_fb else "FAIL"))
    return 0 if bad == 0 and ok_fb else 1


# ── the profile on the data ──────────────────────────────────────────────────


def pin_two(lp):
    """zfsvx.pin_dev's rule for two processors, as the timed passes have: two outside 2 to 5 on two cores (a processor and its
    next number share a core), EcoQoS opted out, priority unchanged. Returns what was done."""
    if os.name != "nt" or lp < 0:
        return {"pinned": False, "why": "not asked or not Windows"}
    import ctypes
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    HANDLE = ctypes.c_void_p
    k32.GetCurrentProcess.restype = HANDLE
    k32.GetCurrentProcess.argtypes = []
    k32.GetProcessAffinityMask.argtypes = [HANDLE, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
    k32.GetProcessAffinityMask.restype = ctypes.c_int
    k32.SetProcessAffinityMask.argtypes = [HANDLE, ctypes.c_size_t]
    k32.SetProcessAffinityMask.restype = ctypes.c_int
    k32.SetProcessInformation.argtypes = [HANDLE, ctypes.c_int, ctypes.c_void_p, ctypes.c_ulong]
    k32.SetProcessInformation.restype = ctypes.c_int
    h = k32.GetCurrentProcess()
    pm, sm = ctypes.c_size_t(), ctypes.c_size_t()
    if not k32.GetProcessAffinityMask(h, ctypes.byref(pm), ctypes.byref(sm)):
        return {"pinned": False, "why": "GetProcessAffinityMask failed"}
    allowed = [i for i in range(64) if (pm.value >> i) & 1 and i not in (2, 3, 4, 5)]
    first = lp if lp in allowed else (allowed[len(allowed) // 2] if allowed else -1)
    rest = [i for i in allowed if i // 2 != first // 2]
    if first < 0 or not rest:
        return {"pinned": False, "why": "no two cores outside 2 to 5", "mask": int(pm.value)}
    second = min([i for i in rest if i > first] or rest)
    out = {"allowed": allowed, "lps": [first, second], "affinity_ok": bool(k32.SetProcessAffinityMask(h, (1 << first) | (1 << second)))}
    st = C.C1._PTS(1, 0x1, 0)
    out["ecoqos_off_ok"] = bool(k32.SetProcessInformation(h, 4, ctypes.byref(st), ctypes.sizeof(st)))
    out["pinned"] = out["affinity_ok"]
    return out


def timed(fn, acc, name):
    """fn with its wall time and calls added to acc[name]."""
    def run(*args, **kw):
        c0 = clock()
        try:
            return fn(*args, **kw)
        finally:
            acc[name] = acc.get(name, 0.0) + clock() - c0
            acc[name + "_calls"] = acc.get(name + "_calls", 0) + 1
    return run


def forward_as(S, key, v, feats, nqt, base_z, acc, orig_layers):
    """One forward at batch 16 as the variant has it, and its time: m3 = the modules' own functions (no form of A6, the module's
    layer step: M3's warm batches); m5 = A6's forms and the module's layer step (what M5's warm batches run); form = A6's forms and
    the batched layer step. The GNN's layer step and layer loop are timed inside; the entry into the variant is not."""
    R = X._zg6r()
    step = g6r_step_b if v == "form" else X._ORIG["g6r_step"]
    saved = (R.g6r_step, R.g6r_layers)
    R.g6r_step, R.g6r_layers = timed(step, acc, "step"), timed(orig_layers, acc, "layers")
    try:
        with Y.variant("orig" if v == "m3" else "all"):
            c0 = clock()
            s = C.forward(S, key, feats, nqt, base_z, C.B16)
            dt = clock() - c0
    finally:
        R.g6r_step, R.g6r_layers = saved
    return dt, s.numpy().copy()


def family(pr):
    """The shares of torch's self CPU time by op family: the gathers and scatters, the matrix products, the rest."""
    ev = list(pr.key_averages())
    tot = sum(e.self_cpu_time_total for e in ev) or 1.0
    gs = sum(e.self_cpu_time_total for e in ev if e.key in FAM_GS)
    mm = sum(e.self_cpu_time_total for e in ev if e.key in FAM_MM)
    return {"self_ms": tot / 1000.0, "gather_scatter": gs / tot, "matmul": mm / tot, "other": 1.0 - (gs + mm) / tot,
            "calls": int(sum(e.count for e in ev))}


def measure(S, packs, blas, a):
    """Every variant of both models on each batch of 16 of the first questions, in a rotating order, a.repeats times after one
    that is not counted: the forward's time, the layer step's and the layer loop's inside it, and the outputs compared bit for bit."""
    KEYS = C.KEYS
    R = X._zg6r()
    orig_layers = R.g6r_layers
    S.threads = C.set_threads(blas)
    res = {k: {v: {"total": [], "step": [], "layers": [], "step_calls": []} for v in VARS[k]} for k in KEYS}
    same = {"gnn:m3_vs_m5": [], "gnn:m5_vs_form": [], "mlp:m3_vs_m5": []}
    nb = len(packs["gnn"]) // C.B16
    for key in KEYS:
        for g in range(nb):
            feats, nqt, base_z = C.build(S, key, packs[key][g * C.B16:(g + 1) * C.B16])
            vs = VARS[key]
            first = {}
            for r in range(int(a.repeats) + 1):
                for v in vs[r % len(vs):] + vs[:r % len(vs)]:
                    acc = {}
                    dt, s = forward_as(S, key, v, feats, nqt, base_z, acc, orig_layers)
                    if v not in first:
                        first[v] = s
                    elif not same32(first[v], s):
                        log(f"{key} batch {g} variant {v}: the forward is not the same twice")
                        same.setdefault("not_repeatable", []).append([key, g, v])
                    if r:
                        d = res[key][v]
                        d["total"].append(dt)
                        d["step"].append(acc.get("step", 0.0))
                        d["layers"].append(acc.get("layers", 0.0))
                        d["step_calls"].append(acc.get("step_calls", 0))
            same[f"{key}:m3_vs_m5"].append(same32(first["m3"], first["m5"]))
            if key == "gnn":
                same["gnn:m5_vs_form"].append(same32(first["m5"], first["form"]))
        gc.collect()
    out = {"blas": int(blas), "threads": S.threads, "batches": nb, "repeats": int(a.repeats), "same": same, "variants": {}}
    for key in KEYS:
        for v in VARS[key]:
            d = res[key][v]
            tot, st, ly = ms(d["total"]), ms(d["step"]), ms(d["layers"])
            share = float(np.median(np.asarray(d["step"]) / np.asarray(d["total"])))
            out["variants"][f"{key}:{v}"] = {"batch_ms": tot, "per_question_ms": tot / C.B16, "step_ms": st, "layers_ms": ly,
                                              "step_share": share, "layers_share": float(np.median(
                                                  np.asarray(d["layers"]) / np.asarray(d["total"]))),
                                              "step_calls": int(max(d["step_calls"] or [0])), "forwards": len(d["total"])}
    return out


def summary(ds, o):
    v = o["variants"]
    g3, g5, gf = (v[f"gnn:{x}"] for x in ("m3", "m5", "form"))
    m3, m5 = v["mlp:m3"], v["mlp:m5"]
    r3, r5, rf = g3["batch_ms"] / m3["batch_ms"], g5["batch_ms"] / m5["batch_ms"], gf["batch_ms"] / m5["batch_ms"]
    s = g5["step_share"]
    log(f"{ds} BLAS {o['blas']}: warm b16 per question, ms: gnn {g3['per_question_ms']:.1f} (M3 forms) -> {g5['per_question_ms']:.1f} "
        f"(A6 forms, module's step) -> {gf['per_question_ms']:.1f} (batched step); mlp {m3['per_question_ms']:.1f} -> "
        f"{m5['per_question_ms']:.1f}")
    log(f"{ds} BLAS {o['blas']}: the module's layer step is {100 * s:.1f}% of the GNN's b16 forward ({g5['step_ms']:.0f} of "
        f"{g5['batch_ms']:.0f} ms a batch), the layer loop {100 * g5['layers_share']:.1f}%; ratio gnn/mlp {r3:.2f} (M3 forms) -> "
        f"{r5:.2f} (A6 forms) -> {rf:.2f} (batched step; the step {g5['step_ms'] / max(gf['step_ms'], 1e-9):.2f}x faster); "
        f"limit with the step free {r5 * (1 - s):.2f}; same bytes: gnn m3/m5 {all(o['same']['gnn:m3_vs_m5'])}, "
        f"gnn m5/form {all(o['same']['gnn:m5_vs_form'])}, mlp m3/m5 {all(o['same']['mlp:m3_vs_m5'])}")


def share(a):
    C._load()
    torch, KEYS = C.torch, C.KEYS
    pinned = pin_two(a.lp)
    log(f"pin: {pinned}")
    nq = int(a.questions)
    if nq < C.B16 or nq % C.B16:
        raise SystemExit("--questions is a multiple of 16")
    with Y.installed():
        S = C.setup(a.dataset, nq)
        C.prepass(S, range(nq + 1))
        S.threads = C.set_threads(C.BLAS)
        log(f"threads {S.threads}")
        for key in KEYS:                                                    # the JIT question, both models
            A = C.stage_a(S, nq, {})
            C.fill_codes(S, nq, A.pool)
            C.run_model(S, key, nq, A, {})
        packs = {k: [] for k in KEYS}
        for j in range(nq):
            A = C.stage_a(S, j, {})
            C.fill_codes(S, j, A.pool)
            for key in (KEYS if j % 2 == 0 else KEYS[::-1]):
                _s, pack, _f = C.run_model(S, key, j, A, {})
                packs[key].append(pack)
        gc.collect()
        sz = {"rows": [int(p["n"]) for p in packs["gnn"]], "edges": [int(p["ne"]) for p in packs["gnn"]]}
        log(f"{a.dataset}: {nq} questions, rows a question p50 {np.median(sz['rows']):.0f}, edges p50 {np.median(sz['edges']):.0f}")
        out = {"dataset": a.dataset, "questions": nq, "sizes": sz, "pin": pinned, "passes": {}}
        for blas in [int(x) for x in a.blases.split(",")]:
            o = measure(S, packs, blas, a)
            out["passes"][str(blas)] = o
            summary(a.dataset, o)
        S.threads = C.set_threads(2)
        from torch.profiler import ProfilerActivity, profile                  # the ops, once a model, with A6's forms and the module's step
        out["family"] = {}
        for key in KEYS:
            feats, nqt, base_z = C.build(S, key, packs[key][:C.B16])
            C.forward(S, key, feats, nqt, base_z, C.B16)
            try:
                with Y.variant("all"), torch.no_grad(), profile(activities=[ProfilerActivity.CPU]) as pr:
                    C.forward(S, key, feats, nqt, base_z, C.B16)
                X.log_prof(pr, f"{key} forward at batch 16 with A6's forms (module's layer step)", n=int(a.top))
                out["family"][key] = family(pr)
                log(f"{key}: of the profiler's self time {100 * out['family'][key]['gather_scatter']:.1f}% gathers and scatters, "
                    f"{100 * out['family'][key]['matmul']:.1f}% matrix products, {100 * out['family'][key]['other']:.1f}% the rest")
            except Exception as e:  # noqa: BLE001
                log(f"{key} profile failed (the numbers stand): {type(e).__name__}: {str(e)[:300]}")
        out["step_stats"] = dict(STATS_B)
        out["forms_stats"] = {"x": dict(X.STATS), "y": dict(Y.STATS)}
        path = C.OUT / "a6" / "dev" / f"b16share__{a.dataset}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        C.T.write_json(path, out)
        log(f"written {path}")
        log("B16SHARE_JSON " + json.dumps({"dataset": a.dataset, "passes": {b: {"variants": p["variants"], "same": p["same"]}
                                                                           for b, p in out["passes"].items()},
                                           "family": out["family"], "step_stats": out["step_stats"]}))
    log("B16SHARE DONE")
    return 0


# ── the timed warm batches of 16 (amendment A7's re-time) ────────────────────

OUT7 = C.OUT / "a7"
PASSES7 = (("b2", "BLAS 2 (the record)"), ("b1", "BLAS 1 (beside the record)"))
V16 = (("mlp", "m5"), ("gnn", "m5"), ("gnn", "a7"))     # a batch's three forwards: the MLP's, the GNN's with the module's layer step
                                                        # (what M5's warm batches run) and the GNN's with the batched step (A7)


def forward_t(S, key, v, feats, nqt, base_z):
    """One timed forward at batch 16 and its scores: v 'a7' takes the batched layer step, 'm5' the module's (A4's and A6's forms
    are in force either way: the caller installed them). Only the forward is inside the clock."""
    R = X._zg6r()
    saved = R.g6r_step
    R.g6r_step = g6r_step_b if v == "a7" else X._ORIG["g6r_step"]
    try:
        c0 = clock()
        s = C.forward(S, key, feats, nqt, base_z, C.B16)
        dt = clock() - c0
    finally:
        R.g6r_step = saved
    return dt, s.numpy()


def pass16(S, packs, blas, nb, reps):
    """One BLAS count's warm batches of 16: reps rounds over the nb batches after one round that is not counted; each batch's
    features are built (outside the clock) and its three forwards (V16) timed in an order that rotates, so each leads in turn.
    Returns (times in seconds by variant: reps lists of nb batch times, the first round's scores by variant and batch, the
    number of forwards that took the batched step)."""
    S.threads = C.set_threads(blas)
    times = {f"{k}:{v}": [] for k, v in V16}
    outs = {name: [None] * nb for name in times}
    n_a7 = 0
    for rep in range(reps + 1):
        row = {name: [] for name in times}
        for g in range(nb):
            sl = slice(g * C.B16, (g + 1) * C.B16)
            fm, fg = C.build(S, "mlp", packs["mlp"][sl]), C.build(S, "gnn", packs["gnn"][sl])
            k0 = (g + rep) % len(V16)
            for key, v in V16[k0:] + V16[:k0]:
                dt, s = forward_t(S, key, v, *(fm if key == "mlp" else fg))
                n_a7 += v == "a7"
                name = f"{key}:{v}"
                row[name].append(dt)
                if outs[name][g] is None:
                    outs[name][g] = s.copy()
                elif not same32(outs[name][g], s):
                    raise SystemExit(f"{name} batch {g}: the forward is not the same twice")
        if rep:
            for name in times:
                times[name].append(row[name])
    return times, outs, n_a7


def checks16(S, packs, outs, nb):
    """Untimed: the batched step's scores byte for byte the module's, batch by batch; every variant's scores, question by question,
    against the fit's filed read (zfsvc.check_scores: within 1e-4, the same top five)."""
    res = {"gnn_a7_same_bytes_as_m5": [bool(same32(outs["gnn:m5"][g], outs["gnn:a7"][g])) for g in range(nb)]}
    for name, per in outs.items():
        key = name.split(":")[0]
        bad, md = 0, 0.0
        for g in range(nb):
            off = np.r_[0, np.cumsum([int(p["n"]) for p in packs[key][g * C.B16:(g + 1) * C.B16]])]
            for i in range(C.B16):
                c = C.check_scores(S, key, g * C.B16 + i, per[g][off[i]:off[i + 1]])
                bad += not c["ok"]
                md = max(md, c["max_diff"] or 0.0)
        res[name] = {"failing": int(bad), "max_diff": float(md), "questions": nb * C.B16}
    return res


def run(a):
    """The warm batches of 16 of zfsvc.run, re-timed with A4's, A6's and A7's forms: setup, the first 16 * (questions // 16) questions'
    inputs built as in M5 (untimed), then for each BLAS count the three forwards of V16 timed per batch, and the checks. The raw
    files go to outputs/fsv/timing/a7 (M5's are not touched)."""
    C._load()
    nq = a.nq or C.T.NQ
    full = nq == C.T.NQ
    out_dir = OUT7 / ("raw" if full else "raw_smoke")
    blases = [int(x) for x in a.blases.split(",")]
    files = {b: out_dir / f"{a.dataset}__b{b}.json" for b in blases}
    for f in files.values():
        if full and f.exists() and not a.overwrite:
            raise SystemExit(f"{f.name} exists: a finished pass is not repeated for its numbers (--overwrite)")
    nb = nq // C.B16
    m = nb * C.B16
    KEYS = C.KEYS
    with Y.installed():
        S = C.setup(a.dataset, nq)
        C.prepass(S, range(nq + 1))
        S.threads = C.set_threads(C.BLAS)
        load = C.pin_and_load([int(x) for x in a.lps.split(",")])
        if a.require_pin and not load["pin"].get("pinned"):
            raise SystemExit(f"{a.dataset}: not pinned: {load}")
        for key in KEYS:                                                    # the JIT question
            A = C.stage_a(S, nq, {})
            C.fill_codes(S, nq, A.pool)
            C.run_model(S, key, nq, A, {})
        packs = {k: [] for k in KEYS}
        for j in range(m):                                                  # the inputs, untimed, as the cold passes build them
            A = C.stage_a(S, j, {})
            C.fill_codes(S, j, A.pool)
            for key in (KEYS if j % 2 == 0 else KEYS[::-1]):
                _s, pack, _f = C.run_model(S, key, j, A, {})
                packs[key].append(pack)
        gc.collect()
        done, n_a7 = {}, 0
        for blas in blases:
            ld = {"cpu_percent_before": C.cpu_load(), "pin": load["pin"]}
            times, outs, n = pass16(S, packs, blas, nb, int(a.reps))
            ld["cpu_percent_after"] = C.cpu_load()
            n_a7 += n
            done[blas] = (times, checks16(S, packs, outs, nb), ld, dict(S.threads))
            log(f"{a.dataset} BLAS {blas}: {nb} batches x {a.reps} rounds timed; checks {done[blas][1]}")
        if not (STATS_B["step_b"] == X._zg6r().LAYERS * n_a7 and STATS_B["step_orig"] == 0 and Y.forms_ran(m, 0)):
            raise SystemExit(f"the forms did not run: {STATS_B} {X.STATS} {Y.STATS}")
        for blas, (times, chk, ld, threads) in done.items():
            stop = (not all(chk["gnn_a7_same_bytes_as_m5"])) or chk["gnn:a7"]["failing"] / chk["gnn:a7"]["questions"] > 0.02
            head = {"dataset": a.dataset, "questions": nq, "batches": nb, "blas": int(blas), "reps": int(a.reps), "stop": bool(stop),
                    "pass": "record: BLAS 2" if blas == 2 else f"beside: BLAS {blas}", "threads": threads,
                    "fits": {k: S.boxes[k]["fit"] for k in KEYS}, "load_and_pin": ld,
                    "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "script_sha256": {"zfsvc": C.T.sha_src(C.__file__), "zfsvt": C.T.sha_src(C.T.__file__),
                                      "zfsvx": C.T.sha_src(X.__file__), "zfsvy": C.T.sha_src(Y.__file__),
                                      "zfsvy_b16": C.T.sha_src(__file__)},
                    "forms": {"fast_index": S.fast_index, "stats": dict(X.STATS), "stats_y": dict(Y.STATS), "stats_b": dict(STATS_B)}}
            out_dir.mkdir(parents=True, exist_ok=True)
            C.T.write_json(files[blas], {"head": head, "times": times, "checks": chk})
            log(f"{a.dataset}: raw written to {out_dir.name}/{files[blas].name}; stop {stop}")
    return 1 if any(((not all(c["gnn_a7_same_bytes_as_m5"])) for _t, c, _l, _h in done.values())) else 0


def _cell(x):
    return f"{x['p50']:.2f} / {x['p95']:.2f}"


def _ratio_cell(x):
    return f"{x['ratio']:.2f} [{x['lo']:.2f}, {x['hi']:.2f}]"


def report(a):
    """The A7 reduction: per dataset and BLAS pass, the warm forward at batch 16 per question (p50 / p95 ms over the batches and rounds),
    zfsvc's ratio (GNN / MLP of the p50s, with its bootstrap over the same samples) for the GNN with the module's step and with A7's,
    the GNN's speed-up, and the checks. Writes outputs/fsv/timing/a7/report.json and report.md (masked as M5's)."""
    raw_dir = Path(a.raw_dir) if a.raw_dir else OUT7 / "raw"
    out_dir = Path(a.out_dir) if a.out_dir else OUT7
    rep = {"label": "A7 forms (warm batches of 16)", "amendment": "docs/FSV_FAST_VERSIONS.md section 14 (A7)",
           "recipe": {"cells": "ms per question = a batch's forward time / 16, p50 / p95 over the batches and rounds",
                      "ratio": f"GNN / MLP of the p50s, paired by batch and round, 2.5th and 97.5th percentiles over {C.NBOOT} resamples "
                               "of the batch times with replacement (numpy default_rng(0))"},
           "passes": {}}
    for tag, label in PASSES7:
        ds = {}
        for name in C.T.DATASETS:
            f = raw_dir / f"{name}__{tag}.json"
            if not f.exists():
                continue
            d = json.loads(f.read_text(encoding="utf-8"))
            t = {k: np.asarray(v, np.float64).reshape(-1) / C.B16 for k, v in d["times"].items()}
            h = d["head"]
            ds[name] = {"batches": h["batches"], "reps": h["reps"], "blas": h["blas"], "utc": h["utc"], "fits": h["fits"],
                        "threads": h["threads"], "script_sha256": h["script_sha256"], "forms": h["forms"],
                        "load": C._load_summary(h), "checks": d["checks"], "stop": h["stop"],
                        "ms_per_question": {k: C._pq(x) for k, x in t.items()},
                        "ratios_gnn_over_mlp": {"module_step": C._ratio(t["gnn:m5"], t["mlp:m5"]),
                                                "batched_step": C._ratio(t["gnn:a7"], t["mlp:m5"])},
                        "gnn_speedup_p50": float(np.median(t["gnn:m5"]) / np.median(t["gnn:a7"]))}
        if ds:
            rep["passes"][tag] = {"label": label, "datasets": ds}
    if not rep["passes"]:
        raise SystemExit(f"A7 report: no <dataset>__b2.json or __b1.json in {raw_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    C.T.write_json(out_dir / "report.json", rep)
    L = ["# FSV timing, Family 1, the warm batches of 16 (A7)", "",
         "Family 1 is zrckpn (the MLP) against zg6r (the GNN), every input kept, at amendment A4's, A6's and A7's forms "
         "(`docs/FSV_FAST_VERSIONS.md` sections 12 to 14). Times are ms per question: a batch of 16's forward time over 16, on the "
         "host's CPU, p50 / p95 over the batches and rounds. The GNN is timed with the module's layer step (what M5's warm "
         "batches run) and with A7's batched step, the MLP once, in the same job. Ratios are GNN / MLP of the p50s, paired by "
         f"batch and round, with the 2.5th and 97.5th percentiles over {C.NBOOT:,} resamples of the batch times with replacement "
         "(`numpy.random.default_rng(0)`)."]
    for tag, P in rep["passes"].items():
        L += ["", f"## {P['label']}", "",
              "| dataset | MLP | GNN, module's step | GNN, A7's step | GNN speed-up | ratio, module's step | ratio, A7's step | "
              "same bytes (batches) | failing vs reads (MLP / GNN module / GNN A7) |", "|---|---|---|---|---|---|---|---|---|"]
        for ds, d in P["datasets"].items():
            q, r, c = d["ms_per_question"], d["ratios_gnn_over_mlp"], d["checks"]
            L.append(f"| {ds} | {_cell(q['mlp:m5'])} | {_cell(q['gnn:m5'])} | {_cell(q['gnn:a7'])} | {d['gnn_speedup_p50']:.2f}x | "
                     f"{_ratio_cell(r['module_step'])} | {_ratio_cell(r['batched_step'])} | {sum(c['gnn_a7_same_bytes_as_m5'])} / {d['batches']} | "
                     f"{c['mlp:m5']['failing']} / {c['gnn:m5']['failing']} / {c['gnn:a7']['failing']} of {c['gnn:a7']['questions']} |")
    (out_dir / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")
    log(f"A7 report: {out_dir / 'report.json'}, report.md")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("selftest", "share", "run", "report"))
    ap.add_argument("--dataset", default="musique")
    ap.add_argument("--questions", type=int, default=32)
    ap.add_argument("--blas", default=C.BLAS, help="OpenBLAS threads; zfsvc reads it from argv too")
    ap.add_argument("--blases", default="2,1", help="share: the BLAS thread counts measured, in this order")
    ap.add_argument("--lp", type=int, default=8, help="share: the first of the two logical processors (outside 2 to 5)")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--top", type=int, default=14)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--trials", type=int, default=2)
    ap.add_argument("--nq", type=int, default=0, help="run: a smoke run of this many questions, 0 = the 200")
    ap.add_argument("--lps", default="2,4", help="run: the logical processors the process is pinned to")
    ap.add_argument("--reps", type=int, default=5, help="run: rounds over the batches (one more is not counted)")
    ap.add_argument("--require-pin", dest="require_pin", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--raw-dir", dest="raw_dir", default="", help="report: the raw files (default outputs/fsv/timing/a7/raw)")
    ap.add_argument("--out-dir", dest="out_dir", default="", help="report: where report.json and report.md go (a7)")
    a = ap.parse_args(argv)
    if a.cmd == "share":                           # a development job: the verdict is a line of the log, so a chain of them goes on
        try:
            return share(a)
        except (Exception, SystemExit) as e:  # noqa: BLE001
            log(f"B16SHARE STOPPED: {type(e).__name__}: {e}")
            traceback.print_exc()
            return 1 if "[WinError 5]" in str(e) or "[WinError 32]" in str(e) else 0
    return {"selftest": selftest, "run": run, "report": report}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
