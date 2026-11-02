"""A GNN stage's read keeps p@swa only (the stages' amendments of 3 November 2026: LTG A2, DEP A3, RTR A2, IRA A2, ISO A3, and
ISB from its declaration).

lean_gpu's read scores every candidate a fit saved: every epoch's state and the SWA state, nine at 8 epochs and seventeen at 16.
Every rule of the GNN stages reads p@swa alone (zml1's twelve reads, DEP's slices, the compares, C1R's intervals and the knee,
each looking the candidate up by its name), and the read took as long as the train (ML1's zg6r at seed 0: 4,618 s against
4,320 s). Each candidate's forward is its own (lean_gpu.read_carve calls each model on the same batch), and in eval zg6r draws
nothing (its g6_gen draws only while training), so p@swa's rows are the rows a full read gives, up to the GPU's own read noise
(ML1 A1). With on_read, lean_gpu.load_models cuts the blob's candidates to p@swa before it builds the models; lean_gpu's read
writes read.json and each carve's npz from that list, so both list p@swa alone. models.pt keeps every state, and the train is
unchanged.

    import readswa as RS
    with RS.on_read():          # lean_gpu.load_models keeps p@swa; restored after, also on a failure
        ...
    python outputs/mp_unified/readswa.py --selftest
    python outputs/mp_unified/readswa.py --check-host    # the self-test, the stages' self-tests, ML1's real zg6r fits
"""
import argparse
import contextlib
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import zwq as WQ  # noqa: E402  (stdlib at import)

KEEP = ("p@swa",)
STAGES = ("zltg", "zdep", "zrtr", "zira", "zisoz")    # the stages whose fit contexts enter on_read (3 November)
OUT = "outputs/readswa/check.json"                     # --check-host's record (repository paths only)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] readswa: {msg}", flush=True)


def keep_swa(blob):
    """The blob's candidates cut to KEEP, in place; the names it held before."""
    names = [str(c) for c in blob["candidates"]]
    kept = [c for c in names if c in KEEP]
    if kept != list(KEEP) or any(c not in blob["states"] for c in kept):
        raise SystemExit(f"readswa: the fit's candidates {names} do not hold {list(KEEP)} with their states")
    blob["candidates"] = kept
    return names


def swa_only(load):
    """lean_gpu.load_models building KEEP's models only."""
    def load_models(blob, device, *a, **k):
        keep_swa(blob)
        return load(blob, device, *a, **k)
    load_models.readswa = load
    return load_models


def installed(LG=None):
    if LG is None:
        import lean_gpu as LG
    return getattr(LG.load_models, "readswa", None) is not None


@contextlib.contextmanager
def on_read(LG=None):
    """lean_gpu.load_models with KEEP's candidates only; restored after, also on a failure. Entered twice, it wraps once."""
    if LG is None:
        import lean_gpu as LG
    if installed(LG):
        yield LG
        return
    with WQ.patch(LG, load_models=swa_only(LG.load_models)):
        yield LG


def code():
    return {"readswa_sha256": WQ.sha_src(__file__)}


# ── self-test (laptop or host, CPU) ──────────────────────────────────────────


def selftest():
    import numpy as np
    import torch
    import lean_gpu as LG
    import lean_screen as S

    def blob(cands):
        return {"candidates": list(cands), "states": {c: {"w": torch.zeros(1)} for c in cands}}

    nine = [f"p@ep{e}" for e in range(8)] + ["p@swa"]
    # 1. the cut: nine to p@swa, in place; a fit without p@swa or without its state stops
    b = blob(nine)
    assert keep_swa(b) == nine and b["candidates"] == ["p@swa"] and sorted(b["states"]) == sorted(nine)
    for bad in (blob(nine[:8]), {"candidates": nine, "states": {c: 0 for c in nine[:8]}}):
        try:
            keep_swa(bad)
            raise AssertionError("keep_swa passed a fit without p@swa")
        except SystemExit:
            pass
    # 2. on_read: installed and restored (also on a failure), wraps once when entered twice; lean_screen's per-arm
    #    load_models taken inside it passes its class through to lean_gpu's own
    got = []

    def rec(blob_, device, cls=None):
        got.append(([str(c) for c in blob_["candidates"]], device, cls))
        return []

    with WQ.patch(LG, load_models=rec):
        with on_read(LG):
            w = LG.load_models
            assert installed(LG) and w.readswa is rec
            with on_read(LG):
                assert LG.load_models is w
            with S.patched("base"):
                LG.load_models(blob(nine), "cpu")
            assert LG.load_models is w
        assert LG.load_models is rec and not installed(LG)
        try:
            with on_read(LG):
                raise RuntimeError("inside")
        except RuntimeError:
            pass
        assert LG.load_models is rec
    assert got == [(["p@swa"], "cpu", LG.LeanMLP8D)], got
    assert not installed(LG)
    # 3. lean_gpu.read_carve on a toy carve: p@swa's rows (top, hit, the first questions' scores) are the same read with
    #    the eight other candidates before it and alone, over several batches; the references too

    class Carve:
        def __init__(self, q=9, seed=0):
            g = np.random.default_rng(seed)
            self.device, self.rows = "cpu", q
            self.n_np = g.integers(2, 9, q).astype(np.int64)
            self.off_np = np.concatenate([[0], np.cumsum(self.n_np)]).astype(np.int64)
            r = int(self.off_np[-1])
            self.x = torch.from_numpy(g.standard_normal((r, 4)).astype(np.float32))
            self.gold = torch.from_numpy(g.random(r) < 0.25)
            self.score2 = g.standard_normal((r, 2)).astype(np.float32)

        def batch(self, qs, blocks):
            r0, r1 = int(self.off_np[qs[0]]), int(self.off_np[qs[-1] + 1])
            nq = torch.repeat_interleave(torch.arange(len(qs)), torch.from_numpy(self.n_np[qs]))
            return self.x[r0:r1], nq, self.x[r0:r1, 0].clone(), self.gold[r0:r1]

    class Toy(torch.nn.Module):
        def __init__(self, s):
            super().__init__()
            self.w = torch.nn.Parameter(torch.randn(4, generator=torch.Generator().manual_seed(s)))

        def forward(self, feats, keep, nq, B, base_z):
            return feats @ self.w + 0.1 * base_z

    c = Carve()
    models = [(n, Toy(k).eval(), ["toy"]) for k, n in enumerate(nine)]
    full = LG.read_carve(c, models, rows_cap=12, first_q=4)
    alone = LG.read_carve(c, models[-1:], rows_cap=12, first_q=4)
    assert len(LG.batches(c.n_np, 12)) > 2
    assert np.array_equal(full["top"][-1], alone["top"][0]) and np.array_equal(full["hit"][-1], alone["hit"][0])
    assert np.array_equal(full["scores64"][-1], alone["scores64"][0])
    assert all(np.array_equal(full[k], alone[k]) for k in ("ref_top", "ref_hit"))
    assert alone["top"].shape == (1, c.rows) and full["top"].shape == (9, c.rows)
    log("selftest: the cut, on_read's install and restore (also on a failure, and nested), lean_screen's per-arm "
        "load_models inside it, and read_carve's p@swa rows alone and among nine pass")
    return 0


# ── the host check (CPU): the stages' self-tests and ML1's real zg6r fits ────


def stage_selftests():
    """Each stage's own --selftest in its own process, from the repository's root."""
    out = []
    for m in STAGES:
        t0 = time.time()
        r = subprocess.run([sys.executable, str(HERE / f"{m}.py"), "--selftest"], cwd=str(ROOT), capture_output=True,
                           text=True, encoding="utf-8", errors="replace", env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        s = round(time.time() - t0, 1)
        log(f"{m} --selftest: exit {r.returncode} in {s} s")
        out.append({"module": m, "exit": r.returncode, "seconds": s,
                    **({"tail": WQ.clean((r.stdout + r.stderr)[-800:])} if r.returncode else {})})
    return out


def real_fits():
    """ML1's seed-0 zg6r fits of both WQ splits, loaded as lean_gpu's read loads them: under on_read() and lean_screen's zg6r
    arm, lean_gpu.load_models builds one ZG6R model, p@swa, holding p@swa's state, and lean_gpu's own load_models is back
    after. on_read() patches the lean_gpu that zg6r's read calls. Returns (rows, problems)."""
    import importlib
    import torch
    import zml1s as M1
    G, _U2N = M1.import_arm("zg6r")
    LG, S = G.LG, G.S
    rows, problems = [], []
    if not (LG is importlib.import_module("lean_gpu") and S.LG is LG):
        problems.append("zg6r's lean_gpu is not the module on_read() patches")
    for sp in M1.SPLITS:
        name = M1.fit_name("zg6r", 0, sp)
        p = WQ.fit_dir(name) / "models.pt"
        if not p.is_file():
            problems.append(f"{name}: no models.pt")
            continue
        blob = torch.load(p, weights_only=False)
        before = [str(c) for c in blob["candidates"]]
        orig, inside = LG.load_models, False
        try:
            with on_read():
                inside = installed(LG)
                with S.patched("zg6r"):
                    models = LG.load_models(blob, "cpu")
        except SystemExit as e:
            problems.append(f"{name}: {WQ.clean(e)}")
            continue
        finally:
            back = LG.load_models is orig and not installed(LG)
        got = [(n, type(m).__name__) for n, m, _bl in models]
        ref = blob["states"]["p@swa"]
        sd = models[0][1].state_dict() if models else {}
        same = sorted(sd) == sorted(ref) and all(torch.equal(sd[k].cpu(), ref[k].cpu()) for k in ref)
        rows.append({"split": sp, "fit": name, "models_sha256": WQ.sha_file(p), "candidates_before": before,
                     "candidates_after": [str(c) for c in blob["candidates"]], "models": got, "state_same": same,
                     "installed_inside": inside, "restored": back})
        if got != [("p@swa", G.ZG6R.__name__)] or not same or not inside or not back:
            problems.append(f"{name}: models {got}, p@swa's state {same}, installed {inside}, restored {back}")
    return rows, problems


def check_host():
    """The self-test, the stages' self-tests and ML1's real zg6r fits; OUT written; exit 0 when all pass."""
    problems = []
    try:
        selftest()
        own = "passed"
    except (AssertionError, SystemExit) as e:
        own = WQ.clean(repr(e))
        problems.append(f"readswa's self-test: {own}")
    tests = stage_selftests()
    problems += [f"{t['module']}'s self-test exited {t['exit']}" for t in tests if t["exit"] != 0]
    rows, p = real_fits()
    problems += p
    rec = {"stage": "readswa", "what": "host check", "selftest": own, "stage_selftests": tests, "fits": rows,
           "problems": problems, "passed": not problems, "code": code(),
           "modules": {f"{m}_sha256": WQ.sha_src(HERE / f"{m}.py") for m in STAGES},
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    WQ.write_json(ROOT / OUT, rec)
    for r in rows:
        log(f"{r['fit']}: {len(r['candidates_before'])} candidates to {r['candidates_after']}, models {r['models']}, "
            f"p@swa's state {r['state_same']}, restored {r['restored']}")
    log(f"{'passed' if not problems else 'FAILED: ' + '; '.join(problems)} ({OUT})")
    return 1 if problems else 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--check-host", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.check_host:
        return check_host()
    ap.error("--selftest or --check-host")


if __name__ == "__main__":
    import readswa
    sys.exit(readswa.main())
