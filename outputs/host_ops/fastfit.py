"""Run a fit script with lean_gpu's training and reads made faster and every number unchanged (systems code; no science).

    python outputs/host_ops/fastfit.py [--no-host-x] [--no-prefetch] -- SCRIPT.py ARGS ...
    python outputs/host_ops/fastfit.py --selftest

Why (3 November 2026). Two GNN screen fits kept the card 12% busy (nvidia-smi) at about 0.24 s a step, and memory, not
compute, set how many could share it. A WQ-hotpotqa fit peaks at 18.4 GB, 10.5 GB of it the training carves' fixed blocks
(ml1-zg6r's train.json), and a WQ-musique fit at 7.9 GB, 6.5 GB of it the carves. So one WQ-hotpotqa fit ran at a time.

Two changes, each on unless turned off:

1. Host X (--no-host-x). lean_gpu.CacheCarve keeps a carve's fixed blocks as one float16 matrix X on the device. For a
   CUDA device the carve is built by CacheCarve's own __init__ on the CPU, and then everything but X goes to the device
   (gold, row, the code table, q_emb, s, c, kappa). X stays in host memory. batch() copies the batch's rows of X to the
   device (each question's rows are one slice of X); the float32 conversion, nan_to_num, the blocks and rrf's z-score
   then run there as before. A carve class whose own code names self.X keeps X on the device, as before.
2. Prefetch (--no-prefetch). fit_variant's loop builds each step's batch (numpy indexing, host arrays to the device) while
   the card waits. Here one thread builds the next steps' batches, in the loop's own order, while the main thread runs the
   step. The steps, their order, their random draws and the step's code are fit_variant's, and no carve's batch draws a
   random number.

Neither changes a number. The device gets the same bits, and the operations of each step run as before, on the same stream.
Each change is made only when lean_gpu's code is the code it was written against (CODE_SHA256); otherwise the run goes on
as before and says so. fastfit_check.py compares X and every trained state bit for bit with and without the changes, on
the host's card.

The script runs in this process (runpy, as __main__), as cuda_alloc.py runs it, so the two runners nest either way. The
changes are made when lean_gpu has been imported, through an import finder, so the script's own imports and the order of
its environment settings are as before. At exit it prints one "[fastfit]" line: the carves whose X stayed in host
memory and the steps prefetched.
"""
import argparse
import atexit
import hashlib
import importlib.abc
import importlib.util
import inspect
import os
import queue
import re
import runpy
import sys
import threading
import time

CODE_SHA256 = {                      # inspect.getsource of lean_gpu's functions as of 7d10987
    "CacheCarve.__init__": "38ea6b3ee691949b4199d9a66f63c248695c00eb7cc767695c3fdfe59ceb32d3",
    "CacheCarve.batch": "e6b72a3be3f846b183a27e058d87d7c4a5f4bd8ca389a4d25dcb426ea6716b94",
    "fit_variant": "3085c4bbc6b8a0a3c409d495aef04c5afac3012d2f8b6cbfe712aa08ab44808a",
}
MOVED = ("gold", "row", "tab", "q_emb", "s", "c", "kappa")      # CacheCarve's device tensors other than X
X_RE = re.compile(r"(?<![A-Za-z0-9_])self\.X\b")
STATE = {"host_x": True, "prefetch": True, "cpu_as_device": False, "made": None, "refused": [], "carves": [],
         "kept": [], "fits": 0, "steps": 0, "wait_s": 0.0}
ORIG = {}
LG = None


def log(msg):
    print(f"[fastfit] {msg}", flush=True)


def src_sha(f):
    return hashlib.sha256(inspect.getsource(f).encode("utf-8")).hexdigest()


# ── 1. host X ────────────────────────────────────────────────────────────────


def reads_x(cls):
    """The first method above CacheCarve in cls's order whose own code names self.X (or has no source), or None."""
    for k in cls.__mro__:
        if k is LG.CacheCarve:
            return None
        for name, f in vars(k).items():
            f = getattr(f, "__func__", f)
            if not hasattr(f, "__code__"):
                continue
            try:
                src = inspect.getsource(f)
            except (OSError, TypeError):
                return f"{k.__name__}.{name} (no source)"
            if X_RE.search(src):
                return f"{k.__name__}.{name}"
    return None


def carve_init(self, ds, carve, basis, device, *args, **kw):
    """CacheCarve.__init__; for a CUDA device, built on the CPU by the original and then all but X moved to the device."""
    torch = LG.torch
    dev = torch.device(device)
    why = None if STATE["host_x"] else "off"
    if why is None and dev.type != "cuda" and not STATE["cpu_as_device"]:
        why = "not CUDA"
    if why is None:
        why = reads_x(type(self))
    if why is not None:
        self.fastfit_host_x = False
        ORIG["init"](self, ds, carve, basis, device, *args, **kw)
        if why not in ("off", "not CUDA"):
            STATE["kept"].append(f"{ds}/{carve}: {why}")
            log(f"{ds}/{carve}: X on the device ({why} names self.X)")
        return
    ORIG["init"](self, ds, carve, basis, "cpu", *args, **kw)
    for k in MOVED:
        setattr(self, k, getattr(self, k).to(dev))
    self.device = dev
    self.fastfit_host_x = True
    gb = self.X.numel() * self.X.element_size() / 1e9
    STATE["carves"].append((f"{ds}/{carve}", round(gb, 3)))
    log(f"{ds}/{carve}: X in host memory ({gb:.2f} GB), the rest on {dev}")


def host_rows(c, qs):
    """X's rows of questions qs, in qs's order (host memory, float16): each question's rows are one slice of X."""
    off = c.off_np
    if qs.size == 0:
        return c.X[:0]
    if qs.size == 1 or bool((qs[1:] == qs[:-1] + 1).all()):
        return c.X[int(off[qs[0]]):int(off[qs[-1] + 1])]
    return LG.torch.cat([c.X[int(off[q]):int(off[q + 1])] for q in qs.tolist()])


def carve_batch(self, qs, blocks):
    """CacheCarve.batch; with X in host memory, X's rows come from there (the one changed line)."""
    if not getattr(self, "fastfit_host_x", False):
        return ORIG["batch"](self, qs, blocks)
    np, torch = LG.np, LG.torch
    qs = np.asarray(qs, np.int64)
    cnt = self.n_np[qs]
    B = qs.size
    N = int(cnt.sum())
    seg = np.cumsum(cnt) - cnt
    idx = np.repeat(self.off_np[qs] - seg, cnt) + np.arange(N)
    nq_np = np.repeat(np.arange(B), cnt)
    dev = self.device
    idx_t = torch.from_numpy(idx).to(dev)
    nq = torch.from_numpy(nq_np).to(dev)
    Xf = host_rows(self, qs).to(dev).to(torch.float32)          # was: self.X[idx_t].to(torch.float32)
    Xc = torch.nan_to_num(Xf, nan=0.0, posinf=0.0, neginf=0.0)
    feats = {}
    for b in blocks:
        if b == "SEMB":
            feats[b] = (self.q_emb[torch.from_numpy(qs).to(dev)].to(torch.float32), self.decode(self.row[idx_t]))
        else:
            a, e = self.span[b]
            feats[b] = Xc[:, a:e].contiguous()
    rrf = Xf[:, self.c_rrf].contiguous()
    base_z = LG.seg_zscore8D(rrf.unsqueeze(1), nq, B).squeeze(1)
    return feats, nq, base_z, self.gold[idx_t]


# ── 2. prefetch ──────────────────────────────────────────────────────────────


class Prefetch:
    """One thread building plan's batches in the plan's order, a step or two ahead of the step that takes them."""

    def __init__(self, tr, plan, blocks):
        self.tr, self.plan, self.blocks = tr, plan, blocks
        self.q = queue.Queue(maxsize=1)
        self.stop = threading.Event()
        self.t = threading.Thread(target=self.run, name="fastfit-prefetch", daemon=True)

    def put(self, item):
        while not self.stop.is_set():
            try:
                self.q.put(item, timeout=0.2)
                return True
            except queue.Full:
                pass
        return False

    def run(self):
        try:
            for ci, qs in self.plan:
                if self.stop.is_set() or not self.put(("ok", self.tr[ci].batch(qs, self.blocks))):
                    return
        except BaseException as e:  # noqa: BLE001  (raised again by the step that takes this batch)
            self.put(("err", e))

    def take(self):
        t = time.perf_counter()
        kind, v = self.q.get()
        STATE["wait_s"] += time.perf_counter() - t
        if kind == "err":
            raise v
        return v

    def __enter__(self):
        self.t.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        while self.t.is_alive():
            try:
                self.q.get(timeout=0.2)
            except queue.Empty:
                pass
        self.t.join()
        return False


def fit_variant(tr, blocks, cfg, seed, hidden, ctx, device, tag=""):
    """lean_gpu.fit_variant (CODE_SHA256's), line for line, except that each epoch's steps are listed first and their
    batches built by Prefetch."""
    if not STATE["prefetch"]:
        return ORIG["fit_variant"](tr, blocks, cfg, seed, hidden, ctx, device, tag)
    np, torch, json, L8 = LG.np, LG.torch, LG.json, LG.L8
    if cfg["cos"] or cfg["adamw"] or cfg["drop"]:
        raise SystemExit("fit8 takes lr:wd:dropout:epochs:swa_from only")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    widths = {b: tr[0].widths[b] for b in blocks}
    model = LG.LeanMLP8D(blocks, widths, hidden, dropout=cfg["dropout"], seed=seed, arm="ctl", ctx=ctx).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    t_ctx = time.time()
    stats = LG.set_ctx_statsD(model, tr, blocks)
    if stats:
        stats["seconds"] = time.time() - t_ctx
    units_ci = np.concatenate([np.full(c.rows, ci, np.int64) for ci, c in enumerate(tr)])
    units_q = np.concatenate([np.arange(c.rows, dtype=np.int64) for c in tr])
    keep_all = None
    states, curve = [], []
    for ep in range(cfg["epochs"]):
        model.train()
        order = rng.permutation(units_ci.size)
        t0 = time.time()
        tot = torch.zeros((), dtype=torch.float64, device=device)
        nb, steps = 0, 0
        guard = {"skipped_loss": 0, "skipped_grad": 0, "first": None}
        plan = []
        for k in range(0, order.size, 32):
            sel = order[k:k + 32]
            cis, qq = units_ci[sel], units_q[sel]
            for ci in sorted(set(cis.tolist())):
                plan.append((ci, qq[cis == ci]))
        with Prefetch(tr, plan, blocks) as pf:
            for ci, qs in plan:
                feats, nq, base_z, gold = pf.take()
                B = qs.size
                steps += 1
                if keep_all is None or keep_all.shape[0] != B:
                    keep_all = torch.ones((B, len(blocks)), dtype=torch.float32, device=device)
                s = model(feats, keep_all, nq, B, base_z)
                loss = LG.listwiseD(s, gold, nq, B)
                if not bool(torch.isfinite(loss).all()):
                    guard["skipped_loss"] += 1
                    if guard["first"] is None:
                        guard["first"] = {"epoch": ep, "at": "loss", "graph": tr[ci].ds, "rows": [int(v) for v in qs[:32]]}
                    continue
                opt.zero_grad()
                loss.backward()
                grads = [p.grad for p in model.parameters() if p.grad is not None]
                if not bool(torch.stack([torch.isfinite(g).all() for g in grads]).all()):
                    guard["skipped_grad"] += 1
                    if guard["first"] is None:
                        guard["first"] = {"epoch": ep, "at": "grad", "graph": tr[ci].ds, "rows": [int(v) for v in qs[:32]],
                                          "nonfinite_grads": [n_ for n_, p in model.named_parameters()
                                                              if p.grad is not None and not bool(torch.isfinite(p.grad).all())][:20]}
                    opt.zero_grad()
                    continue
                opt.step()
                tot += loss.detach().to(torch.float64)
                nb += 1
        STATE["steps"] += steps
        states.append(LG.cpu_state(model))
        rec = {"epoch": ep, "loss": float(tot) / max(nb, 1), "steps": steps, "seconds": time.time() - t0}
        if ctx != "none":
            rec["gate_w_norm"] = float(model.gate_w.detach().norm())
            rec["gate_b"] = float(model.gate_b.detach())
        if guard["skipped_loss"] or guard["skipped_grad"]:
            rec["guard"] = guard
            LG.log(f"  {tag} ep {ep}: GUARD skipped {guard['skipped_loss']} at a non-finite loss, {guard['skipped_grad']} at a "
                   f"non-finite gradient; first {json.dumps(guard['first'])}")
        curve.append(rec)
        LG.log(f"  {tag} ep {ep}: loss {rec['loss']:.4f}, {steps} steps ({rec['seconds']:.0f}s)")
    acc, n_acc = None, 0
    for ep, st in enumerate(states):
        if ep < cfg["swa_from"]:
            continue
        if acc is None:
            acc = {k: v.double().clone() for k, v in st.items() if k not in L8.CTX_BUFFERS}
        else:
            for k in acc:
                acc[k] += st[k].double()
        n_acc += 1
    swa = {k: ((acc[k] / n_acc).to(torch.float32) if k in acc else states[-1][k].clone()) for k in states[-1]}
    STATE["fits"] += 1
    return {"states": states, "swa": swa, "curve": curve, "widths": widths, "ctx_stats": stats,
            "swa_epochs": list(range(cfg["swa_from"], cfg["epochs"]))}


# ── installing ───────────────────────────────────────────────────────────────


def install(lg):
    """Make the changes on lean_gpu, once; a change whose original is not CODE_SHA256's is not made."""
    global LG
    if STATE["made"] is not None:
        return STATE["made"]
    LG = lg
    have = {"CacheCarve.__init__": lg.CacheCarve.__init__, "CacheCarve.batch": lg.CacheCarve.batch,
            "fit_variant": lg.fit_variant}
    sha = {k: src_sha(f) for k, f in have.items()}
    made = []
    if all(sha[k] == CODE_SHA256[k] for k in ("CacheCarve.__init__", "CacheCarve.batch")):
        ORIG["init"], ORIG["batch"] = lg.CacheCarve.__init__, lg.CacheCarve.batch
        lg.CacheCarve.__init__, lg.CacheCarve.batch = carve_init, carve_batch
        made.append("host_x")
    else:
        STATE["refused"].append("host_x")
    if sha["fit_variant"] == CODE_SHA256["fit_variant"]:
        ORIG["fit_variant"] = lg.fit_variant
        lg.fit_variant = fit_variant
        made.append("prefetch")
    else:
        STATE["refused"].append("prefetch")
    STATE["made"] = made
    on = [m for m in made if STATE[m]]
    log(f"lean_gpu patched: {', '.join(on) or 'nothing'} on"
        + (f"; {', '.join(STATE['refused'])} refused (lean_gpu's code is not CODE_SHA256's)" if STATE["refused"] else ""))
    return made


def uninstall():
    """lean_gpu as it was (the self-test and the check use this)."""
    if LG is None or STATE["made"] is None:
        return
    if "host_x" in STATE["made"]:
        LG.CacheCarve.__init__, LG.CacheCarve.batch = ORIG["init"], ORIG["batch"]
    if "prefetch" in STATE["made"]:
        LG.fit_variant = ORIG["fit_variant"]
    STATE["made"] = None
    STATE["refused"] = []


class Finder(importlib.abc.MetaPathFinder):
    """Runs install() right after lean_gpu's module has executed, whoever imports it first."""

    def find_spec(self, name, path=None, target=None):
        if name != "lean_gpu":
            return None
        sys.meta_path.remove(self)
        spec = importlib.util.find_spec(name)
        if spec is None or spec.loader is None:
            sys.meta_path.insert(0, self)
            return spec
        loader, run0 = spec.loader, spec.loader.exec_module

        def exec_module(module):
            run0(module)
            install(module)

        loader.exec_module = exec_module
        return spec


def report():
    if STATE["made"] is None and not STATE["refused"]:
        log("lean_gpu was not imported; nothing changed")
        return
    gb = sum(g for _n, g in STATE["carves"])
    log(f"{len(STATE['carves'])} carve(s) with X in host memory ({gb:.2f} GB), {len(STATE['kept'])} kept on the device; "
        f"{STATE['fits']} fit(s), {STATE['steps']} steps prefetched, {STATE['wait_s']:.0f}s waiting on a batch")


# ── self-test (CPU) ──────────────────────────────────────────────────────────


def toy_carve(lg, seed, Q=40):
    """A CacheCarve with made arrays (no cache): every attribute batch(), fit_variant and the model read."""
    np, torch, LM = lg.np, lg.torch, lg.LM
    g = np.random.default_rng(seed)
    c = object.__new__(lg.CacheCarve)
    c.ds, c.carve, c.basis, c.device = f"toy{seed}", "fit", "toy", torch.device("cpu")
    c.n_np = g.integers(3, 30, Q).astype(np.int64)
    c.off_np = np.concatenate([[0], np.cumsum(c.n_np)])
    R = int(c.off_np[-1])
    c.span = {"rank": (0, 5), "WALK": (5, 21)}
    c.W = 21
    c.widths = {"rank": 5, "WALK": 16, "SEMB": LM.SEMB_DIM}
    c.c_rrf = LM.SPLIT["rank"].index("rrf")
    X = g.standard_normal((R, c.W)).astype(np.float16)
    X[g.random((R, c.W)) < 0.02] = np.nan
    c.X = torch.from_numpy(X)
    c.gold = torch.from_numpy(g.random(R) < 0.2)
    T = 50
    c.row = torch.from_numpy(g.integers(0, T, R).astype(np.int64))
    D = lg.L3.STORE_DIM - 1
    c.tab = torch.from_numpy(g.integers(-127, 128, (T, D)).astype(np.int8))
    c.q_emb = torch.from_numpy(g.standard_normal((Q, 1536)).astype(np.float16))   # the cache's q_emb: float16, 1536 wide
    c.s = torch.from_numpy((g.random(D) * 0.01 + 0.001).astype(np.float32))
    c.c = torch.from_numpy((g.standard_normal(D) * 0.01).astype(np.float32))
    c.kappa = torch.tensor([[0.5]], dtype=torch.float32)
    c.rows = Q
    c.gt_np = np.zeros(Q, np.int64)
    c.fastfit_host_x = False
    return c


def bits_equal(a, b):
    torch = LG.torch
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    if a.dtype.is_floating_point:
        iv = {2: torch.int16, 4: torch.int32, 8: torch.int64}[a.element_size()]
        return torch.equal(a.contiguous().view(iv), b.contiguous().view(iv))
    return torch.equal(a, b)


def selftest():
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(os.path.dirname(here), "mp_unified"))
    import lean_gpu as lg
    torch, np = lg.torch, lg.np
    lg.set_flags("cpu")
    lg.bind_device_ops()
    assert install(lg) == ["host_x", "prefetch"], STATE
    blocks = ["rank", "WALK", "SEMB"]
    cfg = {"lr": 0.002, "wd": 0.0001, "dropout": 0.1, "epochs": 3, "swa_from": 1, "cos": False, "adamw": False,
           "drop": False}

    def run(host, pre):
        STATE["host_x"], STATE["prefetch"] = host, pre
        tr = [toy_carve(lg, 1), toy_carve(lg, 2, Q=25)]
        for c in tr:
            c.fastfit_host_x = host
        return tr, fit_variant(tr, blocks, cfg, 0, 16, "none", "cpu", tag="selftest")

    tr0, f0 = run(False, False)
    out = {}
    for host, pre in ((True, False), (False, True), (True, True)):
        _tr, f = run(host, pre)
        same = all(bits_equal(f0["states"][e][k], f["states"][e][k]) for e in range(cfg["epochs"]) for k in f0["states"][e])
        same &= all(bits_equal(f0["swa"][k], f["swa"][k]) for k in f0["swa"])
        same &= [r["loss"] for r in f0["curve"]] == [r["loss"] for r in f["curve"]]
        same &= [r["steps"] for r in f0["curve"]] == [r["steps"] for r in f["curve"]]
        out[f"host_x={host} prefetch={pre}"] = same
    # batch() on host X against the original, on shuffled, consecutive, one and no questions
    c = tr0[0]
    for qs in (np.array([7, 3, 11, 0, 39]), np.arange(5, 17), np.array([22]), np.zeros(0, np.int64)):
        c.fastfit_host_x = False
        a = ORIG["batch"](c, qs, blocks)
        c.fastfit_host_x = True
        b = carve_batch(c, qs, blocks)
        ok = all(bits_equal(a[0][k], b[0][k]) for k in ("rank", "WALK")) and bits_equal(a[0]["SEMB"][0], b[0]["SEMB"][0])
        ok &= bits_equal(a[0]["SEMB"][1], b[0]["SEMB"][1]) and torch.equal(a[1], b[1]) and bits_equal(a[2], b[2])
        ok &= torch.equal(a[3], b[3])
        out[f"batch {qs.tolist()[:5]}"] = ok
    # a worker's failure reaches the step that takes its batch
    class Boom(Exception):
        pass
    bad = toy_carve(lg, 3)
    bad.batch = lambda qs, bl: (_ for _ in ()).throw(Boom("made"))
    try:
        with Prefetch([bad], [(0, np.arange(3))], blocks) as pf:
            pf.take()
        out["worker failure raised"] = False
    except Boom:
        out["worker failure raised"] = True
    # the refusal: a changed fit_variant is not patched
    uninstall()
    saved = CODE_SHA256["fit_variant"]
    CODE_SHA256["fit_variant"] = "0" * 64
    made = install(lg)
    out["changed code refused"] = made == ["host_x"] and STATE["refused"] == ["prefetch"] and lg.fit_variant is ORIG["fit_variant"]
    uninstall()
    CODE_SHA256["fit_variant"] = saved
    out["restored"] = lg.fit_variant is ORIG["fit_variant"] and lg.CacheCarve.batch is ORIG["batch"]
    for k, v in out.items():
        log(f"selftest {k}: {'ok' if v else 'FAILED'}")
    return 0 if all(out.values()) else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description="lean_gpu's training and reads, faster and unchanged")
    ap.add_argument("--no-host-x", action="store_true", help="keep every carve's X on the device")
    ap.add_argument("--no-prefetch", action="store_true", help="build each batch in the step, as fit_variant does")
    ap.add_argument("--selftest", action="store_true")
    argv = list(sys.argv[1:] if argv is None else argv)      # the script's own command comes after "--"
    cut = argv.index("--") if "--" in argv else len(argv)
    a, cmd = ap.parse_args(argv[:cut]), argv[cut + 1:]
    if a.selftest:
        return selftest()
    if not cmd or not cmd[0].endswith(".py"):
        ap.error("need SCRIPT.py [ARGS ...] after --")
    STATE["host_x"], STATE["prefetch"] = not a.no_host_x, not a.no_prefetch
    if "lean_gpu" in sys.modules:
        install(sys.modules["lean_gpu"])
    else:
        sys.meta_path.insert(0, Finder())
    atexit.register(report)
    script = cmd[0]
    sys.argv = list(cmd)
    sys.path[0] = os.path.dirname(os.path.abspath(script))     # as `python SCRIPT.py` would have it
    runpy.run_path(script, run_name="__main__")
    return 0


if __name__ == "__main__":
    sys.exit(main())
