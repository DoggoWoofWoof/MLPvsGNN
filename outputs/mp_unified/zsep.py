"""Screens, sixteenth round (docs/SCREENS.md): an objective on zret's base. zsep is zret's model (lean_screen3.ZRet,
unchanged) trained with each gold row set against its question's non-gold rows only.

lean_gpu's loss (listwiseD) scores a question with golds G as the mean over G of -log softmax over every row of its
pool, so each gold's denominator holds the question's other golds. Once one gold leads, its own term pushes it down
(its gradient is p_g - 1/|G|), and the non-gold rows between it and a trailing gold get almost no push (their softmax
share is small next to the leader's). zsep's loss is the mean over G of

    softplus(LSE_N(s) - s_g) = -log(e^{s_g} / (e^{s_g} + sum_{n in N} e^{s_n})),   N the pool's non-gold rows,

and the mean over the questions with a gold, as before. No gold is pushed down, and each gold's term pushes the
non-gold rows down, the highest first, hardest for the golds that trail them. A question with one gold has the same
loss and gradient as before (up to rounding), and a pool whose rows are all gold scores 0. zret's model, carves,
batches, config and seed are unchanged, and the objective acts in training only, so reads and serving are zret's. No
new column, block or hyperparameter. Each training run records, per training carve, how many questions have two or
more golds in the pool (the questions where the two losses differ): screen.json's 'zsep' 'census'.

Two fits (L-musique and L-hotpotqa), each compared with zret's fit of its split (that comparison decides) and with step
1's (reported); the pair, its re-call under the seed null and the full run's grade are relz.py's, run under zsep's name
with zret's fits in place of rel's (zgs.py's mapping of zret's fits).

    python outputs/mp_unified/zsep.py smoke --device cuda --host
    python outputs/mp_unified/zsep.py train --split L-musique --name scr-zsep --arm zsep --device cuda --host
    python outputs/mp_unified/zsep.py read --name scr-zsep --device cuda --host
    python outputs/mp_unified/zsep.py compare --new outputs/screen/fits/scr-zsep \\
        --base outputs/screen/fits/scr-zret,outputs/step1/fits/L-musique --out outputs/screen/scr-zsep
    python outputs/mp_unified/zsep.py pair --screens outputs/screen/scr-zsep.json,outputs/screen/scr-zsep-hp.json \\
        --out outputs/screen/scr-zsep-pair
    python outputs/mp_unified/zsep.py recall --null N1,N2,N3,N4 --pair outputs/screen/scr-zsep-pair.json \\
        --out outputs/screen/scr-zsep-pair-recall
    python outputs/mp_unified/zsep.py grade --full-root outputs/full_zsep \\
        --reuse L-musique=outputs/screen/scr-zsep.json,L-hotpotqa=outputs/screen/scr-zsep-hp.json
    python outputs/mp_unified/zsep.py --selftest
N1..N4 are the seed null's comparisons (outputs/screen/scr-null-s1, -s2, -s1-hp, -s2-hp .json). The full run's
re-grade is nullx.py's regrade with zret's six comparisons as --base-compares.

Speed: the objective changes training only. zsep reads and serves as zret's model does, so any latency figure for it
is zret's, and cold (8 October): each question timed from scratch, with no warm-up pass and nothing kept from an
earlier question.
"""
import os
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import contextlib  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402

import lean_screen3 as S3  # noqa: E402
import lean_screen5 as S5  # noqa: E402
import zgs as G  # noqa: E402

Z = G.Z
S2, S = Z.S2, Z.S
LG, LC = S.LG, S.LC
log = S.log
ARM, BASE_ARM = "zsep", "zret"
S.ARMS.update({ARM: S.ARMS[BASE_ARM]})
STATS = {"census": None}


# ── the objective ────────────────────────────────────────────────────────────


def listwise_sep(scores, gold, nq, B):
    """Each gold row against its question's non-gold rows only: a question with golds G and non-gold rows N scores
    (1/|G|) sum_{g in G} softplus(LSE_N(s) - s_g); the mean over the questions with a gold, as listwiseD's. A pool whose
    rows are all gold scores 0. lean_gpu.listwiseD's signature, on the scores' device and dtype."""
    dev, dt = scores.device, scores.dtype
    g = gold.to(torch.bool)
    sn = torch.where(g, torch.full_like(scores, -float("inf")), scores)        # the non-gold rows; -inf on the golds
    mx = torch.full((B,), -float("inf"), dtype=dt, device=dev).scatter_reduce(0, nq, sn, reduce="amax",
                                                                              include_self=True)
    has_n = torch.isfinite(mx)
    mx = torch.where(has_n, mx, torch.zeros_like(mx))
    den = torch.zeros(B, dtype=dt, device=dev).index_add_(0, nq, (sn - mx[nq]).exp())
    lse = torch.where(has_n, mx + torch.where(has_n, den, torch.ones_like(den)).log(),
                      torch.full_like(den, -float("inf")))
    gf = g.to(dt)
    t = Fn.softplus(lse[nq] - scores) * gf
    ng = torch.zeros(B, dtype=dt, device=dev).index_add_(0, nq, gf)
    pq = torch.zeros(B, dtype=dt, device=dev).index_add_(0, nq, t)
    has = ng > 0
    return (pq[has] / ng[has]).mean() if bool(has.any()) else scores.sum() * 0.0


def sep_numpy(s, gold, nq, B):
    """listwise_sep in float64 numpy, question by question: (loss, d loss / d s)."""
    s, gold, nq = np.asarray(s, np.float64), np.asarray(gold, bool), np.asarray(nq, np.int64)
    grad, terms = np.zeros_like(s), []
    for q in range(B):
        i = np.flatnonzero(nq == q)
        gi, ni = i[gold[i]], i[~gold[i]]
        if gi.size == 0:
            continue
        if ni.size == 0:
            terms.append(0.0)
            continue
        m = s[ni].max()
        lse = m + np.log(np.exp(s[ni] - m).sum())
        x = lse - s[gi]
        sig = 1.0 / (1.0 + np.exp(-x))
        terms.append(float(np.logaddexp(0.0, x).mean()))
        grad[gi] -= sig / gi.size
        grad[ni] += sig.sum() / gi.size * np.exp(s[ni] - lse)
    if not terms:
        return 0.0, grad
    return float(np.mean(terms)), grad / len(terms)


def census(tr):
    """Per training carve (dataset/carve): its questions, those with a gold in the pool, with two or more (where
    listwise_sep and listwiseD differ), with every row gold, and the golds in all."""
    out = {}
    for c in tr:
        key = f"{c.ds}/{c.carve}"
        if key in out:
            continue
        n = np.asarray(c.n_np, np.int64)
        gold = c.gold.detach().to("cpu").numpy().astype(np.float64)
        k = np.rint(np.bincount(np.repeat(np.arange(n.size), n), weights=gold, minlength=n.size)).astype(np.int64)
        out[key] = {"questions": int(n.size), "with_gold": int((k > 0).sum()), "two_or_more": int((k > 1).sum()),
                    "all_rows_gold": int(((k == n) & (k > 0)).sum()), "golds": int(k.sum())}
    return out


@contextlib.contextmanager
def sep_loss():
    """lean_gpu's loop with listwise_sep in place of listwiseD (its one loss), each fit's training carves counted
    (census, the first fit's); both restored after, an error too."""
    saved_loss, saved_fit = LG.listwiseD, LG.fit_variant

    def fit_variant(tr, *args, **kw):
        if STATS["census"] is None:
            STATS["census"] = census(tr)
        return saved_fit(tr, *args, **kw)

    LG.listwiseD, LG.fit_variant = listwise_sep, fit_variant
    try:
        yield
    finally:
        LG.listwiseD, LG.fit_variant = saved_loss, saved_fit


# ── train and read ───────────────────────────────────────────────────────────


def train(argv, split):
    """lean_screen3's train (lean_screen2's, with zret's model) under listwise_sep."""
    arm = S5.arm_of(argv)
    if arm != ARM:
        raise SystemExit(f"zsep: train takes --arm {ARM}, not {arm}")
    STATS["census"] = None
    with sep_loss():
        rc = S3.train(argv, split)
    name, out_root = S2.where(argv)
    sj = out_root / name / "screen.json"
    if sj.exists():
        rec = json.loads(sj.read_text(encoding="utf-8"))
        rec["zsep_sha256"] = LC.sha_src(__file__)
        rec["lean_screen3_sha256"] = LC.sha_src(S3.__file__)
        rec["zsep"] = {"model": "lean_screen3.ZRet", "loss": "zsep.listwise_sep: each gold against the non-gold rows",
                       "census": STATS["census"]}
        LC.write_json(sj, rec)
    return rc


def read(argv):
    name, out_root = S2.where(argv)
    arm = json.loads((out_root / name / "screen.json").read_text(encoding="utf-8"))["arm"]
    if arm != ARM:
        raise SystemExit(f"zsep: {name} was trained as {arm}, not {ARM}")
    return S2.main(["read"] + argv)


# ── relz.py's pair, re-call and grade, decided against zret's fits ───────────


@contextlib.contextmanager
def on_zret():
    """relz.py's records under zsep's name, each read decided against zret's fit of its split (zgs.py's mapping);
    relz restored after."""
    with G.on_zret():
        Z.ARM = ARM
        yield


FIX = (("(rel's screen fits scr-rel and scr-rel-hp)", "(zret's fits: its screen fit scr-zret and its full run's "
                                                      "L-hotpotqa fit)"),
       ("(rel's screen fits)", "(zret's fits)"),
       ("| rel R@5 | relz R@5 |", "| zret R@5 | zsep R@5 |"),
       ("section 2 and the tenth round", "section 2 and the sixteenth round"),
       ("docs/FULL_ROUND10.md", "docs/FULL_ROUND16.md"))


def restamp(out):
    """A record relz.py wrote under zsep's name: zret named as the base in its md, this file's sha added."""
    out = Path(out)
    md = out.with_suffix(".md")
    if md.exists():
        t = md.read_text(encoding="utf-8")
        for a, b in FIX:
            t = t.replace(a, b)
        md.write_text(t, encoding="utf-8")
    js = out.with_suffix(".json")
    if js.exists():
        rec = json.loads(js.read_text(encoding="utf-8"))
        rec["decided_against"] = "zret's fit of each split"
        rec["zsep_sha256"] = LC.sha_src(__file__)
        LC.write_json(js, rec)


def pair(screens, out):
    with on_zret():
        rec = Z.pair(screens, out)
    restamp(out)
    return rec


def recall(null_files, pair_file, out, zret_screens=None):
    with on_zret():
        rec = Z.recall(null_files, pair_file, out, zret_screens)
    restamp(out)
    return rec


def grade(full_root, reuse, out, check_fits=True):
    with on_zret():
        rec = Z.grade(full_root, reuse, out, check_fits)
    restamp(out)
    return rec


# ── smoke ────────────────────────────────────────────────────────────────────


SMOKE_SETS = "2wiki=select,hotpotqa=select"


def check_loss(s, gold, nq, B, tol=1e-4):
    """listwise_sep on scores s against sep_numpy in float64 (the loss, and its gradient relative to the largest), and
    on the questions with exactly one gold against listwiseD (the same, up to rounding)."""
    s1 = s.detach().clone().requires_grad_(True)
    loss = listwise_sep(s1, gold, nq, B)
    loss.backward()
    nq_np, gold_np = nq.cpu().numpy(), gold.cpu().numpy().astype(bool)
    ref, rgrad = sep_numpy(s.detach().cpu().double().numpy(), gold_np, nq_np, B)
    g = s1.grad.detach().cpu().double().numpy()
    rec = {"questions": int(B), "loss": float(loss), "loss_ref": ref,
           "loss_err": abs(float(loss) - ref) / max(1.0, abs(ref)),
           "grad_err": float(np.abs(g - rgrad).max()) / max(1e-12, float(np.abs(rgrad).max())),
           "grads_finite": bool(np.isfinite(g).all())}
    k = np.rint(np.bincount(nq_np, weights=gold_np.astype(np.float64), minlength=B)).astype(np.int64)
    one = np.flatnonzero(k == 1)
    rec["one_gold"], rec["two_or_more"] = int(one.size), int((k > 1).sum())
    ok = rec["grads_finite"] and rec["loss_err"] < tol and rec["grad_err"] < tol
    if one.size:
        sel_np = np.isin(nq_np, one)
        remap = np.full(B, -1, np.int64)
        remap[one] = np.arange(one.size)
        sel = torch.from_numpy(sel_np).to(s.device)
        nq1 = torch.from_numpy(remap[nq_np[sel_np]]).to(s.device)
        out = {}
        for nm, fn in (("sep", listwise_sep), ("listwiseD", LG.listwiseD)):
            x = s.detach()[sel].clone().to(torch.float32).requires_grad_(True)
            v = fn(x, gold[sel], nq1, one.size)
            v.backward()
            out[nm] = (float(v), x.grad.detach().cpu().double().numpy())
        a, b = out["sep"], out["listwiseD"]
        rec["one_gold_loss_err"] = abs(a[0] - b[0]) / max(1.0, abs(b[0]))
        rec["one_gold_grad_err"] = float(np.abs(a[1] - b[1]).max()) / max(1e-12, float(np.abs(b[1]).max()))
        ok = ok and rec["one_gold_loss_err"] < tol and rec["one_gold_grad_err"] < tol
    rec["ok"] = bool(ok)
    return bool(ok), rec


def loss_check(ds, carve, device, q=64):
    """A carve's census, and check_loss on its first q questions under zret's model at its start (seed 0), under
    train's flags."""
    LG.set_flags(device)
    LG.bind_device_ops()
    c = LG.CacheCarve(ds, carve, "2wiki", device)
    rec = {"census": census([c])[f"{ds}/{carve}"]}
    blocks = [x for x in LG.SETS["pick"] if x in c.widths]
    widths = {x: c.widths[x] for x in blocks}
    qs = np.arange(min(q, c.rows))
    feats, nq, base_z, gold = c.batch(qs, blocks)
    keep = torch.ones((qs.size, len(blocks)), dtype=torch.float32, device=c.device)
    torch.manual_seed(0)
    m = S3.ZRet(blocks, widths, seed=0).to(c.device).eval()
    with torch.no_grad():
        s = m(feats, keep, nq, qs.size, base_z)
    ok, rec["first_questions"] = check_loss(s, gold, nq, qs.size)
    del c, m, feats
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return ok, rec


def smoke(device, host, out_root=None):
    """The loss on real carves (2wiki's and hotpotqa's select): its census, and on the first questions under zret's
    model at its start the float64 reference and, where a question has one gold, listwiseD's loss and gradient. Then
    zret once and zsep twice (the repeat must be IDENTICAL), one epoch each on both carves, each read on 2wiki's. zsep
    must hold zret's model (its state's keys), count both carves (the same census as the check, with questions of two
    or more golds), and its scores must differ from zret's (the same initialisation and batches, so the difference is
    the objective). A crash fails it (exit 1)."""
    t0 = time.time()
    root = Path(out_root or S.OUT / "smoke16")
    h = ["--host"] if host else []
    rec, ok = {"loss_checks": {}}, True
    for ds in ("2wiki", "hotpotqa"):
        good, chk = loss_check(ds, "select", device)
        rec["loss_checks"][f"{ds}/select"] = chk
        ok = ok and good
    common = ["--train", SMOKE_SETS, "--basis", "2wiki", "--variants", "p", "--config", "2e-3:1e-4:0.1:1:0",
              "--device", device, "--out-root", str(root)] + h
    runs = {BASE_ARM: (S3.train, "1"), ARM: (train, "2")}
    for arm, (fn, rep) in runs.items():
        nm = f"smoke-{arm}"
        rc = fn(["train", "--name", nm, "--arm", arm, "--repeat", rep] + common, "L-musique")
        rr = S2.main(["read", "--name", nm, "--read", "2wiki=select", "--device", device, "--out-root", str(root)] + h)
        sj = json.loads((root / nm / "screen.json").read_text(encoding="utf-8"))
        tj = json.loads((root / nm / "train.json").read_text(encoding="utf-8"))
        blob = torch.load(root / nm / "models.pt", map_location="cpu", weights_only=False)
        rec[arm] = {"train_rc": rc, "repeats": int(rep), "read_rc": rr, "screen_arm": sj.get("arm"),
                    "curve": tj["variants"]["p"]["curve"], "census": (sj.get("zsep") or {}).get("census"),
                    "state_keys": sorted(blob["states"][sorted(blob["states"])[0]])}
        ok = ok and rc == 0 and rr == 0 and sj.get("arm") == arm
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
    rec[ARM]["repeat"] = "IDENTICAL" if rec[ARM]["train_rc"] == 0 else "DIFFERENT"
    rec["zsep_is_zrets_model"] = rec[ARM]["state_keys"] == rec[BASE_ARM]["state_keys"]
    cen = rec[ARM]["census"] or {}
    rec["census_matches_check"] = (sorted(cen) == sorted(rec["loss_checks"])
                                   and all(cen[k] == v["census"] for k, v in rec["loss_checks"].items()))
    rec["two_or_more_golds"] = {k: v["two_or_more"] for k, v in cen.items()}
    ok = ok and rec["zsep_is_zrets_model"] and rec["census_matches_check"] and sum(rec["two_or_more_golds"].values()) > 0
    npz = {a: root / f"smoke-{a}" / "reads" / "2wiki__select.npz" for a in runs}
    if all(p.exists() for p in npz.values()):
        r = {a: np.load(p) for a, p in npz.items()}
        ids = {a: [str(x) for x in v["ids"]] for a, v in r.items()}
        rec["same_questions"] = ids[ARM] == ids[BASE_ARM]
        rec["scores_differ_from_zret"] = bool(not np.array_equal(r[ARM]["scores64"], r[BASE_ARM]["scores64"]))
        k = [str(x) for x in r[ARM]["candidates"]].index("p@ep0")
        rec["p@ep0_hit@1"] = {a: float(v["hit"][k].mean()) for a, v in r.items()}
        ok = ok and rec["same_questions"] and rec["scores_differ_from_zret"]
    else:
        rec["same_questions"] = None
        ok = False
    for a in runs:
        rec[a].pop("state_keys")
    rec.update({"ok": bool(ok), "seconds": time.time() - t0, "script_sha256": LC.sha_src(__file__)})
    LC.write_json(root / "smoke.json", rec)
    log(f"smoke16: {json.dumps(rec)}; {'ok' if ok else 'FAILED'}")
    return 0 if ok else 1


# ── selftest ─────────────────────────────────────────────────────────────────


def toy_pools(sizes, golds, seed, scale=3.0, dtype=torch.float64):
    """Pools of the given sizes, the first golds[q] rows of pool q gold, rows shuffled: (scores, gold, nq, B)."""
    rng = np.random.default_rng(seed)
    nq = np.repeat(np.arange(len(sizes)), sizes)
    gold = np.concatenate([np.arange(n) < k for n, k in zip(sizes, golds)])
    perm = rng.permutation(nq.size)
    s = torch.from_numpy(rng.standard_normal(nq.size) * scale).to(dtype)
    return s[perm], torch.from_numpy(gold[perm]), torch.from_numpy(nq[perm]), len(sizes)


def grad_of(fn, s, gold, nq, B):
    x = s.detach().clone().requires_grad_(True)
    v = fn(x, gold, nq, B)
    v.backward()
    return v.detach(), x.grad.detach()


def selftest():
    t0 = time.time()
    LG.bind_device_ops()
    # 1. the arm is zret's model on step 1's carve; zret's own arm is unchanged
    assert S.ARMS[ARM] == (S3.ZRet, S.ARMS["base"][1]) and S.ARMS[BASE_ARM] == S.ARMS[ARM]
    # 2. the loss against the float64 reference: pools with no gold, one, two, five, every row gold, a pool of one gold
    #    row, rows in any order; float64 to 1e-8 (softplus is linear past 20 in torch, exact in numpy), float32 to 1e-5
    sizes, golds = [6, 7, 9, 12, 4, 1, 3, 15], [0, 1, 2, 5, 4, 1, 1, 0]
    for seed in range(5):
        s, gold, nq, B = toy_pools(sizes, golds, seed)
        ref, rgrad = sep_numpy(s.numpy(), gold.numpy(), nq.numpy(), B)
        v, gr = grad_of(listwise_sep, s, gold, nq, B)
        assert abs(float(v) - ref) < 1e-8 and np.abs(gr.numpy() - rgrad).max() < 1e-8, (float(v), ref)
        v32, gr32 = grad_of(listwise_sep, s.float(), gold, nq, B)
        assert abs(float(v32) - ref) < 1e-5 and np.abs(gr32.double().numpy() - rgrad).max() < 1e-5
        allg = np.isin(nq.numpy(), [4, 5])                       # pools whose rows are all gold: no term, no gradient
        assert np.all(gr.numpy()[allg] == 0.0) and torch.isfinite(gr).all()
        ok, chk = check_loss(s.float(), gold, nq, B)
        assert ok and chk["one_gold"] == 3 and chk["two_or_more"] == 3, chk
    # 3. a question with one gold: listwiseD's loss and gradient, up to rounding
    s, gold, nq, B = toy_pools([7, 3, 1, 20], [1, 1, 1, 1], 7, dtype=torch.float32)
    a, ga = grad_of(listwise_sep, s, gold, nq, B)
    b, gb = grad_of(LG.listwiseD, s, gold, nq, B)
    assert abs(float(a) - float(b)) < 1e-6 and float((ga - gb).abs().max()) < 1e-6
    # 4. the gradients: no gold is pushed down, no non-gold row up; on a pool with a leading and a trailing gold,
    #    listwiseD pushes the leader down and sep up, and sep pushes the rows between them down harder
    for seed in range(5):
        s, gold, nq, B = toy_pools([9, 12, 30], [2, 5, 3], seed)
        _v, gr = grad_of(listwise_sep, s, gold, nq, B)
        assert bool((gr[gold] <= 0).all()) and bool((gr[~gold] >= 0).all())
    s = torch.tensor([5.0, 2.0, 1.5, -1.0, -3.0])
    gold, nq = torch.tensor([True, False, False, True, False]), torch.zeros(5, dtype=torch.int64)
    _a, gs = grad_of(listwise_sep, s, gold, nq, 1)
    _b, gd = grad_of(LG.listwiseD, s, gold, nq, 1)
    assert gd[0] > 0 and gs[0] < 0
    assert gs[1] > 5 * gd[1] and gs[2] > 5 * gd[2] and gs[3] < 0 and gd[3] < 0
    # 5. edge cases: every row gold (0, no gradient), no gold anywhere (0), extreme scores (finite)
    s, gold, nq, B = toy_pools([3, 2], [3, 2], 1, dtype=torch.float32)
    v, gr = grad_of(listwise_sep, s, gold, nq, B)
    assert float(v) == 0.0 and bool((gr == 0).all())
    s, gold, nq, B = toy_pools([3, 4], [0, 0], 1, dtype=torch.float32)
    v, gr = grad_of(listwise_sep, s, gold, nq, B)
    assert float(v) == 0.0 and bool((gr == 0).all())
    s = torch.tensor([-80.0, 80.0, 79.0, 80.0, -80.0, -79.0])
    gold, nq = torch.tensor([True, False, False, True, False, False]), torch.tensor([0, 0, 0, 1, 1, 1])
    v, gr = grad_of(listwise_sep, s, gold, nq, 2)
    assert bool(torch.isfinite(v)) and bool(torch.isfinite(gr).all()) and 79.0 < float(v) < 82.0
    # 6. the swap: inside, lean_gpu's loss is listwise_sep and its loop counts the carves; restored after, an error too
    saved = LG.listwiseD, LG.fit_variant
    with sep_loss():
        assert LG.listwiseD is listwise_sep and LG.fit_variant is not saved[1]
    assert (LG.listwiseD, LG.fit_variant) == saved
    try:
        with sep_loss():
            raise KeyError("x")
    except KeyError:
        pass
    assert (LG.listwiseD, LG.fit_variant) == saved
    # 7. on two toy carves: under the swap zret's model trains through lean_gpu's loop with the new loss, so the fit
    #    differs from zret's own and repeats bit for bit; the census is the hand count; zret's loop is untouched after
    cfg = {"lr": 2e-3, "wd": 1e-4, "dropout": 0.1, "epochs": 2, "swa_from": 1, "cos": 0, "adamw": 0, "drop": 0}
    wd = {"rank": 5, "WALK": 4}
    a = S5.ToyCarve("dsa", np.random.default_rng(1).integers(3, 20, 60), 11, wd)
    b = S5.ToyCarve("dsb", np.random.default_rng(2).integers(3, 20, 40), 12, wd)
    for c in (a, b):
        c.F["rank"][:, S3.I_RRF] = c.F["rank"][:, 0]
    blocks = ["rank", "WALK"]
    with S.patched(ARM):
        assert isinstance(LG.LeanMLP8D(blocks, wd, 16), S3.ZRet)
        ref = LG.fit_variant([a, b], blocks, cfg, 0, 16, "none", "cpu")
        STATS["census"] = None
        with sep_loss():
            f1 = LG.fit_variant([a, b], blocks, cfg, 0, 16, "none", "cpu")
            f2 = LG.fit_variant([a, b], blocks, cfg, 0, 16, "none", "cpu")
        ref2 = LG.fit_variant([a, b], blocks, cfg, 0, 16, "none", "cpu")
    assert LG.same_state(ref["swa"], f1["swa"]) and not LG.same_state(f1["swa"], f2["swa"])
    assert not LG.same_state(ref["swa"], ref2["swa"])
    assert all(np.isfinite(r["loss"]) for r in f1["curve"])
    for c in (a, b):
        k = [int(c.gold[int(c.off_np[q]):int(c.off_np[q + 1])].sum()) for q in range(c.rows)]
        want = {"questions": c.rows, "with_gold": sum(x > 0 for x in k), "two_or_more": sum(x > 1 for x in k),
                "all_rows_gold": sum(x == n and x > 0 for x, n in zip(k, c.n_np)), "golds": sum(k)}
        assert STATS["census"][f"{c.ds}/toy"] == want, (STATS["census"], want)
        assert want["two_or_more"] > 0
    # 8. refusals: another arm's training, another arm's fit at read
    try:
        train(["train", "--name", "x", "--arm", BASE_ARM], "L-musique")
        raise AssertionError("zsep trained another arm")
    except SystemExit as e:
        assert "takes --arm zsep" in str(e)
    tmp = Path(tempfile.mkdtemp())
    try:
        (tmp / "x").mkdir()
        LC.write_json(tmp / "x" / "screen.json", {"arm": BASE_ARM})
        try:
            read(["--name", "x", "--out-root", str(tmp)])
            raise AssertionError("zsep read zret's fit")
        except SystemExit as e:
            assert "trained as zret" in str(e)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    # 9. relz's pair, re-call and grade under zsep's name, decided against zret's fits; relz restored after; restamp
    #    names zret and this round
    saved = Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit
    with on_zret():
        assert Z.tag({})["arm"] == ARM and Z.BASE_ARM == BASE_ARM and Z.rel_fit("J5") == ("full_zret", "fits", "J5")
    assert (Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit) == saved
    try:
        with on_zret():
            raise KeyError("x")
    except KeyError:
        pass
    assert (Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit) == saved
    SR = Z.SR
    with tempfile.TemporaryDirectory() as td:
        T = Path(td)
        null1 = SR.fake_null(T / "null", "n1", {("L-hotpotqa", "hotpotqa"): (-0.010, 0.012)})   # floor 0.0221
        zs = {sp: str(SR.fake_compare(T / G.zret_fit(sp)[0], G.zret_fit(sp)[2], sp, {ds: 0.1 for ds in LG.EVAL_ORDER},
                                      arm=BASE_ARM))
              for sp in G.ZRET_FITS}
        zb = {sp: "/".join(("outputs",) + G.zret_fit(sp)) for sp in G.ZRET_FITS}
        za = SR.fake_compare(T / "z", "scr-zsep", "L-musique", {"musique": 0.0842}, arm=ARM, base_r5=0.6,
                             base=zb["L-musique"])
        zh = SR.fake_compare(T / "z", "scr-zsep-hp", "L-hotpotqa", {"hotpotqa": -0.0147}, arm=ARM, base_r5=0.6,
                             base=zb["L-hotpotqa"])
        pr = pair([za, zh], T / "pz")
        assert pr["verdict"] == "MIXED" and pr["arm"] == ARM and pr["base_arm"] == BASE_ARM
        assert "scr-zret" in (T / "pz.md").read_text(encoding="utf-8")
        assert json.loads((T / "pz.json").read_text(encoding="utf-8"))["decided_against"] == "zret's fit of each split"
        rc = recall(null1, T / "pz.json", T / "rz", zs)
        assert rc["verdict"] == "PROMISING" and rc["changed"] == ["L-hotpotqa hotpotqa LOSS -> WITHIN"], rc["changed"]
        assert all(r["base"] == 0.6 for r in rc["rows"])
        t = (T / "rz.md").read_text(encoding="utf-8")
        assert "| zret R@5 | zsep R@5 |" in t and "sixteenth round" in t and "rel's" not in t
        st1 = SR.fake_compare(T / "z2", "scr-zsep", "L-musique", {}, arm=ARM)       # decided against step 1's fit
        SR.must_stop(pair, [st1, zh], T / "bad")
        full = T / "full"
        full.mkdir()
        for sp in S2.SPLITS:
            if sp not in G.ZRET_FITS:
                d = {"musique": 0.02} if sp == "J5" else {}
                SR.fake_compare(full / "src", f"fit-{sp}", sp, d, arm=ARM,
                                base="/".join(("outputs",) + G.zret_fit(sp))).replace(full / f"compare-{sp}.json")
        g = grade(full, {"L-musique": str(za), "L-hotpotqa": str(zh)}, full / "grade", check_fits=False)
        assert g["verdict"] == "NOT_ADOPTED" and g["losses"] == 1 and g["arm"] == ARM
        assert "docs/FULL_ROUND16.md" in (full / "grade.md").read_text(encoding="utf-8")
    assert (Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit) == saved
    log(f"zsep selftest: the arm is zret's model; the loss matches its float64 reference (pools with no gold, one, "
        f"several, all rows gold, rows in any order), equals listwiseD where a question has one gold, never pushes a "
        f"gold down or a non-gold row up, pushes the rows between a leading and a trailing gold down harder than "
        f"listwiseD, and stays finite at the edges; the swap restores; a toy fit under it differs from zret's, repeats "
        f"bit for bit and counts its carves by hand; refusals; relz's pair, re-call and grade under zsep's name against "
        f"zret's fits, naming zret and this round, relz restored ({time.time() - t0:.1f}s): ok")
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
        return S2.main([k.cmd] + rest)
    if k.cmd in ("pair", "recall", "grade"):
        gp = argparse.ArgumentParser()
        gp.add_argument("--screens", default="")
        gp.add_argument("--null", default="")
        gp.add_argument("--pair")
        gp.add_argument("--full-root")
        gp.add_argument("--reuse", default="")
        gp.add_argument("--out")
        g = gp.parse_args(rest)
        null = [x for x in g.null.split(",") if x]
        need = {"pair": g.screens and g.out, "recall": g.pair and len(null) == 4 and g.out, "grade": g.full_root}[k.cmd]
        if not need:
            gp.error(f"{k.cmd}: pair needs --screens A,B and --out; recall --pair, the four --null files and --out; "
                     "grade --full-root")
        try:
            if k.cmd == "pair":
                pair([x for x in g.screens.split(",") if x], g.out)
            elif k.cmd == "recall":
                recall(null, g.pair, g.out)
            else:
                reuse = dict(x.split("=", 1) for x in g.reuse.split(",") if x)
                v = grade(g.full_root, reuse, g.out or str(Path(g.full_root) / "grade"))["verdict"]
                return 1 if v == "INCOMPLETE" else 0
        except SystemExit as e:
            log(f"zsep {k.cmd}: {e}")
            return 2
        return 0
    raise SystemExit("zsep: train, read, compare, pair, recall, grade, smoke, or --selftest")


if __name__ == "__main__":
    sys.exit(main())
