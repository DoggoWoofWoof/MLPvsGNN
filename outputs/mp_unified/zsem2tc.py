"""zsem2tc.py: zsem2t.py with a resumable traversal (25 Nov 2026). A wrapper; no pinned module is edited.

Why: the R2a reads run for hours on the host (2Wiki's read ran 198 min; two earlier reads failed after 16.7 and 26.9 min and restarted from
zero), and zsem2t.traverse keeps its lists, the alpha sums and the oracle picks only in memory until the very end. User 25 Nov: "make sure
losing anything means a few mins of progress, checkpoint all that can be and clean up once done".

What it does: the same arguments as zsem2t.py (read | sel, --ds, --arm, --host, --threads, --out) and, for the WebQSP row, the same
--qb N that zsem2tb.py takes. It replaces zsem2t.traverse by a copy with the same signature and defaults whose loop body is the original's,
line for line. Every EVERY_S seconds (300), after the batch that is under way, and at the end of the loop, it writes the running state (the
lists of the questions done, the alpha sum and count, the oracle picks, the next question) to <OUT>/ckpt/<cmd>-<arm>-<ds>-<key>/state.npz
atomically (a temporary file, then os.replace), and at the start loads it and continues from the next batch: a kill or a cancel costs at most
five minutes plus the batch under way. A failed save (disk full) is logged and skipped, it never stops the row.

The checkpoint key is the command, dataset, arm, question batch, node block, k, the specs, the BLAS thread count, zsem2t.py's sha256 and this
file's sha256, the shapes, the question vectors and seeds in full, a strided sample of the node vectors and of 1/|P v|, every parameter of the
saved rotation model, the graph's keys (sizes and a sample), the operator map and the gold lists. A checkpoint of another key, of another shape,
or one that does not read is ignored (the traversal starts at question 0). Because the loop body is the original's and batches do not depend
on each other, a resumed run gives the arrays an uninterrupted run gives, bit for bit (wselftest).

Clean-up: when zsem2t's own main returns 0 (the record is written), this file deletes its checkpoint folder (state files only) and the folders
of earlier attempts of the same row under other keys; a failed or killed row keeps its folder, so the feeder's retry resumes from it. At the
start it removes the row's own strays from a kill: <ds>.npz with no <ds>.json beside it, and <ds>.json.tmp. The record gets one field,
"traversal_checkpoint" (wrapper, resumed_from_question, saves); the numbers are the original's.

Stacking with zsem2tb.py (the WebQSP row, --qb 32): this file replaces T.traverse first and keeps its six positional defaults (qb is the
fifth from the end), then, when --qb is on the line, calls zsem2tb.main(argv) unchanged; zsem2tb sets the default qb on the function it
finds in T.traverse, which is this one. The feed row changes only the script name: zsem2tb.py -> zsem2tc.py.

Usage (host):  python outputs/mp_unified/zsem2tc.py read --ds hotpotqa --arm e3v --host --threads 8
               python outputs/mp_unified/zsem2tc.py read --ds webqsp --arm e3v --host --threads 8 --qb 32
Self-test (laptop, CPU): python outputs/mp_unified/zsem2tc.py wselftest --threads 2
"""
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
sys.dont_write_bytecode = True

import zsem2t as T  # noqa: E402  (zenc is the first pinned module it imports: --threads sets the BLAS threads before numpy loads)

import numpy as np  # noqa: E402

WRAPPER = "outputs/mp_unified/zsem2tc.py"
EVERY_S = 300.0
CALLS = {"n": 0, "resumed_from": None, "saves": 0, "dirs": [], "ctx": {}}
ORIG = T.traverse
OWN = {"state.npz", "state.tmp.npz"}


def log(msg):
    print(f"[zsem2tc {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _h(h, a):
    a = np.ascontiguousarray(a)
    h.update(repr((a.shape, str(a.dtype))).encode())
    h.update(a.tobytes())


def _sample(h, a, m=1025):
    a = np.asarray(a)
    h.update(repr((a.shape, str(a.dtype))).encode())
    if a.size:
        _h(h, a[::max(1, a.shape[0] // m)])
        _h(h, a[-1:])


def key_of(m1, dirs, X, inv_m, Qn, seeds, specs, k, G, op_of, flat, off, qb, nb):
    ctx = CALLS["ctx"]
    h = hashlib.sha256()
    h.update(repr((ctx.get("cmd"), ctx.get("ds"), ctx.get("arm"), int(qb), int(nb), int(k), [(kd, float(lam)) for kd, lam in specs],
                   [bool(d) for d in dirs], int(T.E.THREADS), T.script_sha(), T.Z.sha_file(Path(__file__)), Qn.shape,
                   str(Qn.dtype), seeds.shape, int(X.n), int(X.D))).encode())
    _h(h, Qn)
    _h(h, seeds)
    _h(h, X.rows(np.unique(np.linspace(0, X.n - 1, 257).astype(np.int64))))
    _sample(h, inv_m, 65537)
    for name, t in sorted(m1.state_dict().items()):
        h.update(name.encode())
        _h(h, t.detach().cpu().numpy())
    if G is not None:
        h.update(repr((int(G.n), int(G.nR), int(G.RS), [len(x) for x in G.r_keys], len(G.s_key))).encode())
        for x in list(G.r_keys) + [G.s_key]:
            _sample(h, x)
    for x in (op_of, flat, off):
        if x is not None:
            _h(h, np.asarray(x))
    return h.hexdigest()[:20]


def folder_of(key):
    ctx = CALLS["ctx"]
    if ctx.get("ds"):
        return T.OUT / "ckpt" / f"{ctx['cmd']}-{ctx['arm']}-{ctx['ds']}-{key}"
    return T.OUT / "ckpt" / key


def rm_own(d):
    """Delete a checkpoint folder only if it holds nothing but this wrapper's files."""
    d = Path(d)
    if not d.is_dir():
        return False
    names = {p.name for p in d.iterdir()}
    if not names <= OWN:
        log(f"  not removing {d}: it holds other files {sorted(names - OWN)}")
        return False
    for p in d.iterdir():
        p.unlink()
    d.rmdir()
    return True


def traverse_resumable(torch, m1, dirs, X, inv_m, Qn, seeds, specs, k, G=None, op_of=None, flat=None, off=None, qb=T.QB, nb=T.NB, *,
                       ckpt=None, every_s=None, stop_after_batches=None):
    """zsem2t.traverse with a checkpoint every `every_s` seconds and at the end. The loop body is the original's."""
    oracle_quads, plan_batch, lists_batch, log_ = T.oracle_quads, T.plan_batch, T.lists_batch, T.log
    CALLS["n"] += 1
    every_s = EVERY_S if every_s is None else every_s
    key = key_of(m1, dirs, X, inv_m, Qn, seeds, specs, k, G, op_of, flat, off, qb, nb)
    folder = Path(ckpt) if ckpt else folder_of(key)
    path = folder / "state.npz"
    CALLS["dirs"].append(folder)
    nq = Qn.shape[0]
    out = np.empty((len(specs), nq, k), np.int64)
    a_sum = np.zeros(len(dirs))
    a_cnt = 0
    picked = np.full(seeds.shape, -1, np.int64)
    start = 0
    if not ckpt and CALLS["ctx"].get("ds"):          # earlier attempts of this row under other keys (other code or inputs)
        for d in sorted((T.OUT / "ckpt").glob(folder.name.rsplit("-", 1)[0] + "-*")):
            if d != folder and rm_own(d):
                log(f"  removed the stale checkpoint {d.name} (another key)")
    if path.exists():
        try:
            with np.load(path) as z:
                q_next = int(z["q_next"])
                ok = (str(z["key"]) == key and 0 <= q_next <= nq and (q_next % qb == 0 or q_next == nq)
                      and z["out"].shape == (len(specs), q_next, k) and z["picked"].shape == (q_next,) + seeds.shape[1:]
                      and z["a_sum"].shape == a_sum.shape)
                if ok:
                    out[:, :q_next] = z["out"]
                    picked[:q_next] = z["picked"]
                    a_sum = z["a_sum"].astype(np.float64)
                    a_cnt = int(z["a_cnt"])
                    start = q_next
                    CALLS["resumed_from"] = q_next
                    log(f"  resuming at question {q_next:,} of {nq:,} (checkpoint {folder.name})")
                else:
                    log(f"  checkpoint {folder.name} does not match (key or shape): starting at question 0")
        except Exception as e:  # a torn or unreadable file is ignored, never trusted
            log(f"  checkpoint {folder.name} does not read ({type(e).__name__}: {e}): starting at question 0")
            start, a_cnt = 0, 0
            a_sum = np.zeros(len(dirs))
            picked = np.full(seeds.shape, -1, np.int64)

    def save(q1):
        try:
            folder.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name("state.tmp.npz")
            np.savez(tmp, key=np.array(key), q_next=np.array(q1), a_cnt=np.array(a_cnt), a_sum=a_sum, out=out[:, :q1], picked=picked[:q1])
            os.replace(tmp, path)
            CALLS["saves"] += 1
        except OSError as e:
            log(f"  checkpoint not saved ({e}); the traversal goes on")

    orc = any(kd == "oracle" for kd, _ in specs)
    t0 = time.time()
    last = time.time()
    done_b = 0
    for q0 in range(start, nq, qb):
        q1 = min(nq, q0 + qb)
        sd = seeds[q0:q1]
        quads = oracle_quads(G, op_of, sd, flat, off, q0) if orc else None
        pb = plan_batch(torch, m1, dirs, X, Qn[q0:q1], sd, quads)
        out[:, q0:q1] = lists_batch(torch, X, inv_m, specs, pb, sd, k, nb)
        v = sd >= 0
        a_sum += pb["alpha"][v].sum(0)
        a_cnt += int(v.sum())
        if orc:
            picked[q0:q1] = pb["picked"]
        if q0 == 0 or (q0 // qb) % 10 == 0:
            log_(f"  traversal {q1:,}/{nq:,} questions ({time.time() - t0:.0f} s)")
        done_b += 1
        if q1 < nq and time.time() - last >= every_s:
            t1 = time.time()
            save(q1)
            log(f"  checkpoint: {q1:,}/{nq:,} questions ({time.time() - t1:.1f} s to save)")
            last = time.time()
        if stop_after_batches is not None and done_b >= stop_after_batches:
            if q1 < nq:
                save(q1)
            raise KeyboardInterrupt("selftest stop")
    if start < nq:
        save(nq)                                      # the end of the loop: a kill after it resumes with nothing left to do
    return out, a_sum / max(a_cnt, 1), picked


def clean_orphans(ctx):
    """The row's own strays from a kill: <ds>.npz with no <ds>.json beside it, and <ds>.json.tmp."""
    if ctx.get("cmd") != "read" or not ctx.get("ds") or not ctx.get("arm"):
        return
    d = Path(ctx["out"]) / "read" / ctx["arm"] if ctx.get("out") else T.OUT / "read" / ctx["arm"]
    npz, js, tmp = d / f"{ctx['ds']}.npz", d / f"{ctx['ds']}.json", d / f"{ctx['ds']}.json.tmp"
    if npz.is_file() and not js.is_file():
        npz.unlink()
        log(f"removed the stray {npz.name} of a killed read (no record beside it)")
    if tmp.is_file():
        tmp.unlink()
        log(f"removed the stray {tmp.name}")


def cleanup():
    """After a successful row: delete this run's checkpoint folders (state files only) and the empty ckpt folder."""
    n = sum(rm_own(d) for d in dict.fromkeys(CALLS["dirs"]))
    try:
        (T.OUT / "ckpt").rmdir()
    except OSError:
        pass
    return n


def argval(argv, flag):
    return argv[argv.index(flag) + 1] if flag in argv and argv.index(flag) + 1 < len(argv) else None


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "wselftest":
        return selftest()
    if argv and argv[0] == "wchild":
        return child(argv[1:])
    if not argv or argv[0] not in ("read", "sel"):
        raise SystemExit("zsem2tc: STOP: only the read and sel commands are wrapped (report and selftest have no long loop)")
    T.traverse = traverse_resumable
    assert T.traverse is traverse_resumable and T.cmd_read.__globals__["traverse"] is traverse_resumable \
        and T.cmd_sel.__globals__["traverse"] is traverse_resumable, "the patch did not take"
    d = T.traverse.__defaults__
    assert len(d) == 6 and d[-2] == T.QB and d[-1] == T.NB, d     # zsem2tb's own assertion holds on this function
    CALLS["ctx"] = {"cmd": argv[0], "ds": argval(argv, "--ds"), "arm": argval(argv, "--arm"), "out": argval(argv, "--out")}
    log(f"traversal is resumable (zsem2t.traverse replaced; a checkpoint every {EVERY_S:.0f} s)")
    clean_orphans(CALLS["ctx"])
    if "--qb" in argv:
        import zsem2tb as B
        rc = B.main(argv)
    else:
        rc = T.main(argv)
    assert T.traverse is traverse_resumable
    if rc == 0:
        ctx = CALLS["ctx"]
        p = T.OUT / ctx["cmd"] / ctx["arm"] / f"{ctx['ds']}.json"
        if p.is_file():
            rec = json.loads(p.read_text(encoding="utf-8"))
            rec["traversal_checkpoint"] = {"wrapper": WRAPPER, "resumed_from_question": CALLS["resumed_from"], "saves": CALLS["saves"]}
            T.Z.wjson(p, rec)
        log(f"checkpoint folders removed: {cleanup()}")
    return rc


# ── self-test (laptop CPU) ───────────────────────────────────────────────────


def build_toy():
    """zsem2t's own toy (selftest), with the oracle spec, so lists, alpha means and picks are all exercised."""
    torch = T.Z.torch_()
    rng = np.random.default_rng(0)
    n, D, Dp, nq, k = 2000, 32, 40, 120, 50
    dirs = [False, False, True, True]
    m1 = T.Z.make_model(torch, D, Dp, 4, "rotation")
    with torch.no_grad():
        th = rng.uniform(2.0, 3.0, (4, Dp // 2)) * rng.choice([-1.0, 1.0], (4, Dp // 2))
        m1.theta.copy_(torch.from_numpy(th.astype(np.float32)))
        m1.w.copy_(torch.from_numpy(rng.uniform(0.5, 1.5, (4, Dp // 2)).astype(np.float32)))
    m1.eval()
    V = rng.standard_normal((n, D)).astype(np.float32)
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    src = rng.choice(n // 2, nq, replace=False)
    dst = n // 2 + rng.choice(n // 2, nq, replace=False)
    rel = rng.integers(2, 4, nq)
    Q = np.empty((nq, D), np.float32)
    with torch.no_grad():
        xs = m1.embed(torch.from_numpy(V[src]))
        for i in range(nq):
            y = m1.query(xs[i:i + 1], torch.tensor([int(rel[i])]), torch.tensor([True]))[0]
            y = y / y.norm()
            t = y[:D].numpy() + 0.05 * rng.standard_normal(D).astype(np.float32)
            V[dst[i]] = t / np.linalg.norm(t)
            res = (y - (y @ xs[i]) * xs[i])[:D].numpy()
            qv = V[src[i]] + 0.3 * res / max(np.linalg.norm(res), 1e-9)
            Q[i] = qv / np.linalg.norm(qv)
    X = T.Z.Vecs(V.astype(np.float16))
    inv_m = T.mapped_inv(torch, m1, X)
    flat, off = T.S1.flatten([[int(s), int(d)] for s, d in zip(src, dst)])
    v_top, _ = T.S1.search_mem(Q, X.rows(slice(0, n)), k)
    spl = np.stack([rng.permutation(n)[:k] for _ in range(nq)]).astype(np.int64)
    seeds = T.seeds_of(v_top, spl)
    keysA = np.unique(np.sort(rng.integers(0, n, (300, 2)), 1) @ np.array([n, 1]))
    keysA = keysA[keysA // n != keysA % n]
    skey = np.unique((src * n + dst) * 2 + (rel - 2))
    G = T.Z.Graph("toy", n, [("R.a", keysA), ("R.b", np.zeros(0, np.int64))], skey, ("x", "y"))
    return dict(torch=torch, m1=m1, dirs=dirs, X=X, inv_m=inv_m, Q=Q, seeds=seeds, k=k, G=G, flat=flat, off=off,
                op_of=np.asarray([0, 1, 2, 3], np.int64),
                specs=[("trav", 0.0), ("trav", 0.5), ("trav0", 0.0), ("trav0", 0.5), ("oracle", 0.0)])


def run(fn, t, Q=None, qb=32, nb=300, specs=None, **kw):
    return fn(t["torch"], t["m1"], t["dirs"], t["X"], t["inv_m"], t["Q"] if Q is None else Q, t["seeds"], specs or t["specs"], t["k"],
              t["G"], t["op_of"], t["flat"], t["off"], qb=qb, nb=nb, **kw)


def same(a, b, what):
    for x, y, nm in zip(a, b, ("lists", "alpha mean", "picked")):
        assert np.array_equal(x, y), f"{what}: {nm} differ"


def npz_arrays(path, res, seeds):
    """The arrays zsem2t's read writes from the traversal (with_seeds lists, top 5, picked as int16) and the alpha mean, to an npz."""
    lists = T.with_seeds(res[0], seeds)
    np.savez(path, lists=lists.astype(np.int32), top5=lists[:, :, :5].astype(np.int32), picked=res[2].astype(np.int16), amean=res[1])
    with np.load(path) as z:
        return {n: z[n].copy() for n in z.files}


def child(args):
    """Selftest child: a traversal that is killed (os._exit, no cleanup) at its die_at-th batch."""
    folder, die_at = args[0], int(args[1])
    t = build_toy()
    n = {"c": 0}
    real = T.lists_batch

    def hook(*a, **kw):
        n["c"] += 1
        if n["c"] == die_at:
            os._exit(137)
        return real(*a, **kw)

    T.lists_batch = hook
    run(traverse_resumable, t, ckpt=folder, every_s=0.0)
    return 3


def selftest():
    t = build_toy()
    nbat = 4                                            # 120 questions, batches of 32: 32 + 32 + 32 + 24
    ref = run(ORIG, t)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        ck = td / "ck"
        st = lambda: ck / "state.npz"  # noqa: E731

        def killed(after, Q=None, specs=None):
            try:
                run(traverse_resumable, t, Q=Q, specs=specs, ckpt=ck, every_s=0.0, stop_after_batches=after)
                raise AssertionError("the stop did not fire")
            except KeyboardInterrupt:
                pass

        # 1. uninterrupted, default interval (no mid-loop save), equal to the original; the final state is on disk until the row ends
        b = run(traverse_resumable, t, ckpt=ck)
        same(ref, b, "uninterrupted")
        assert CALLS["resumed_from"] is None and st().exists()
        with np.load(st()) as z:
            assert int(z["q_next"]) == t["Q"].shape[0], "the end-of-loop state is the full one"
        assert rm_own(ck) and not ck.exists(), "the checkpoint folder is removed"
        # 2. killed mid-loop (after 2 batches), resumed
        killed(2)
        with np.load(st()) as z:
            assert int(z["q_next"]) == 64 and z["out"].shape[1] == 64
        c = run(traverse_resumable, t, ckpt=ck, every_s=0.0)
        assert CALLS["resumed_from"] == 64, CALLS
        same(ref, c, "resumed after 2 batches")
        rm_own(ck)
        # 3. killed in the last batch's turn (after 3 batches: the last, short batch is still to do), resumed
        CALLS["resumed_from"] = None
        killed(3)
        d = run(traverse_resumable, t, ckpt=ck, every_s=0.0)
        assert CALLS["resumed_from"] == 96, CALLS
        same(ref, d, "resumed before the last batch")
        # 4. resumed from the end-of-loop state: no batch is computed again
        CALLS["resumed_from"] = None
        real, cnt = T.lists_batch, {"n": 0}
        T.lists_batch = lambda *a, **kw: (cnt.__setitem__("n", cnt["n"] + 1), real(*a, **kw))[1]
        try:
            e = run(traverse_resumable, t, ckpt=ck, every_s=0.0)
        finally:
            T.lists_batch = real
        assert CALLS["resumed_from"] == t["Q"].shape[0] and cnt["n"] == 0, (CALLS, cnt)
        same(ref, e, "resumed from the end state")
        rm_own(ck)
        # 5. a real kill (os._exit mid-batch, nothing saved after the 2nd batch), resumed in this process
        CALLS["resumed_from"] = None
        r = subprocess.run([sys.executable, str(Path(__file__)), "wchild", str(ck), "3", "--threads", str(T.E.THREADS)],
                           capture_output=True, text=True)
        assert r.returncode == 137, (r.returncode, r.stdout[-400:], r.stderr[-400:])
        with np.load(st()) as z:
            assert int(z["q_next"]) == 64
        (ck / "state.tmp.npz").write_bytes(b"torn write")  # a kill during a save leaves a torn temporary file: it is never read
        f = run(traverse_resumable, t, ckpt=ck, every_s=0.0)
        assert CALLS["resumed_from"] == 64, CALLS
        same(ref, f, "resumed after a hard kill")
        rm_own(ck)
        # 6. a checkpoint of another input is ignored: other questions, other specs, other batch
        Q2 = t["Q"].copy()
        Q2[7] = -Q2[7]
        killed(2)
        CALLS["resumed_from"] = None
        g = run(traverse_resumable, t, Q=Q2, ckpt=ck, every_s=0.0)
        assert CALLS["resumed_from"] is None, "a checkpoint of other questions was used"
        same(run(ORIG, t, Q=Q2), g, "foreign questions")
        rm_own(ck)
        killed(2)
        sp2 = t["specs"][:3]
        h = run(traverse_resumable, t, specs=sp2, ckpt=ck, every_s=0.0)
        assert CALLS["resumed_from"] is None, "a checkpoint of other specs was used"
        same(run(ORIG, t, specs=sp2), h, "foreign specs")
        rm_own(ck)
        killed(2)
        i = run(traverse_resumable, t, qb=24, ckpt=ck, every_s=0.0)
        assert CALLS["resumed_from"] is None, "a checkpoint of another batch size was used"
        same(run(ORIG, t, qb=24), i, "foreign batch")
        rm_own(ck)
        # 7. an unreadable state file is ignored
        killed(2)
        st().write_bytes(st().read_bytes()[:200])
        j = run(traverse_resumable, t, ckpt=ck, every_s=0.0)
        assert CALLS["resumed_from"] is None
        same(ref, j, "torn state")
        rm_own(ck)
        # 8. the arrays zsem2t writes: the npz contents of original, uninterrupted and killed-and-resumed runs are equal
        killed(2)
        k_ = run(traverse_resumable, t, ckpt=ck, every_s=0.0)
        rm_own(ck)
        A = npz_arrays(td / "a.npz", ref, t["seeds"])
        for nm, res in (("uninterrupted", b), ("resumed", k_)):
            Bz = npz_arrays(td / "b.npz", res, t["seeds"])
            assert A.keys() == Bz.keys() and all(np.array_equal(A[x], Bz[x]) for x in A), f"npz contents differ ({nm})"
        # 9. rm_own refuses a folder with a file that is not this wrapper's
        (td / "other").mkdir()
        (td / "other" / "keep.txt").write_text("x")
        assert not rm_own(td / "other") and (td / "other" / "keep.txt").exists()
        # 10. the strays of a killed read: an npz without its record, a json.tmp; a complete pair stays
        old_out = T.OUT
        T.OUT = td / "r2"
        try:
            rd = T.OUT / "read" / "e3v"
            rd.mkdir(parents=True)
            (rd / "hotpotqa.npz").write_bytes(b"partial")
            (rd / "hotpotqa.json.tmp").write_text("{")
            (rd / "webqsp.npz").write_bytes(b"whole")
            (rd / "webqsp.json").write_text("{}")
            clean_orphans({"cmd": "read", "ds": "hotpotqa", "arm": "e3v", "out": None})
            clean_orphans({"cmd": "read", "ds": "webqsp", "arm": "e3v", "out": None})
            assert not (rd / "hotpotqa.npz").exists() and not (rd / "hotpotqa.json.tmp").exists()
            assert (rd / "webqsp.npz").exists() and (rd / "webqsp.json").exists()
            # 11. through main(): zsem2t.main is faked by a function that calls T.traverse (this wrapper's) and writes the record;
            #     once with zsem2tb stacked (--qb 32), once killed and rerun. Stale and foreign folders under ckpt/.
            stack_test(t, ref)
        finally:
            T.OUT = old_out
    print("zsem2tc wselftest: ok (original == uninterrupted == resumed mid-loop == resumed before the last batch == resumed from the end state "
          "== resumed after a hard kill; npz contents equal; foreign questions/specs/batch and a torn state ignored; checkpoint folder removed; "
          "stacked with zsem2tb)")
    return 0


def stack_test(t, ref):
    import zsem2tb as B
    real_main, real_defaults = T.main, T.traverse.__defaults__
    ctx_args = ["read", "--ds", "hotpotqa", "--arm", "e3v", "--host", "--threads", str(T.E.THREADS)]
    seen = {}

    def fake_main(av, kill_after=None):
        res = T.traverse(t["torch"], t["m1"], t["dirs"], t["X"], t["inv_m"], t["Q"], t["seeds"], t["specs"], t["k"], t["G"], t["op_of"],
                         t["flat"], t["off"], nb=300, **({"stop_after_batches": kill_after} if kill_after else {}))
        seen["res"] = res
        seen["argv"] = list(av)
        rec = {"recall": {"x": 1.0}, "script_sha256": T.script_sha()}
        T.Z.wjson(T.OUT / av[0] / av[av.index("--arm") + 1] / f"{av[av.index('--ds') + 1]}.json", rec)
        return 0

    T.main = fake_main
    global EVERY_S
    old_every = EVERY_S
    try:
        ck_root = T.OUT / "ckpt"
        (ck_root / "read-e3v-hotpotqa-deadbeef").mkdir(parents=True)                      # an earlier attempt, another key: removed
        (ck_root / "read-e3v-hotpotqa-deadbeef" / "state.npz").write_bytes(b"old")
        (ck_root / "read-e3v-webqsp-0123").mkdir()                                       # another row's folder: kept
        (ck_root / "read-e3v-webqsp-0123" / "state.npz").write_bytes(b"theirs")
        EVERY_S = 0.0
        # the first run is killed after 2 batches (an exception out of main: no clean-up), the second resumes and cleans up
        T.main = lambda av: fake_main(av, kill_after=2)
        CALLS["resumed_from"], CALLS["dirs"], CALLS["saves"] = None, [], 0
        try:
            main(ctx_args + ["--qb", "32"])
            raise AssertionError("the kill did not fire")
        except KeyboardInterrupt:
            pass
        mine = [d for d in ck_root.iterdir() if d.name.startswith("read-e3v-hotpotqa-")]
        assert len(mine) == 1 and (mine[0] / "state.npz").exists() and mine[0].name != "read-e3v-hotpotqa-deadbeef", mine
        assert (ck_root / "read-e3v-webqsp-0123" / "state.npz").read_bytes() == b"theirs"
        T.main = fake_main
        CALLS["dirs"] = []
        T.traverse.__defaults__ = real_defaults           # one run per process on the host: the first main() above set qb 32 already
        rc = main(ctx_args + ["--qb", "32"])
        assert rc == 0 and CALLS["resumed_from"] == 64, CALLS
        same(run(ORIG, t, qb=32), seen["res"], "through main, stacked with zsem2tb, killed and resumed")
        assert "--qb" not in seen["argv"], "zsem2tb passes the line on without --qb"
        rec = json.loads((T.OUT / "read" / "e3v" / "hotpotqa.json").read_text(encoding="utf-8"))
        assert rec["traversal_batch"] == 32 and rec["wrapper"] == "outputs/mp_unified/zsem2tb.py", rec
        assert rec["traversal_checkpoint"]["resumed_from_question"] == 64 and rec["traversal_checkpoint"]["wrapper"] == WRAPPER, rec
        assert not any(d.name.startswith("read-e3v-hotpotqa-") for d in ck_root.iterdir()), "the row's folders are removed"
        assert (ck_root / "read-e3v-webqsp-0123").exists(), "another row's folder is kept"
        rm_own(ck_root / "read-e3v-webqsp-0123")
        ck_root.rmdir()
        # without --qb: the same, the original batch default (zsem2tb's patch of the default is undone first)
        T.traverse.__defaults__ = real_defaults
        EVERY_S = old_every
        CALLS["resumed_from"], CALLS["dirs"] = None, []
        rc = main(["sel"] + ctx_args[1:])
        assert rc == 0 and CALLS["resumed_from"] is None
        same(run(ORIG, t, qb=T.QB), seen["res"], "through main, no --qb")
        assert json.loads((T.OUT / "sel" / "e3v" / "hotpotqa.json").read_text(encoding="utf-8"))["traversal_checkpoint"]["saves"] >= 1
        assert not (T.OUT / "ckpt").exists(), "no checkpoint folder is left behind when the row ends"
    finally:
        T.main = real_main
        EVERY_S = old_every
        traverse_resumable.__defaults__ = real_defaults
        T.traverse = ORIG
    _ = B


if __name__ == "__main__":
    sys.exit(main())
