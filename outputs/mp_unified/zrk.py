"""Screens, twentieth round (docs/SCREENS.md; full run docs/FULL_ROUND20.md): an objective for recall at five, on
zrm, the base.

zrk is zrm (zrm.ZRM over rmatch.ChainCarveBase, unchanged; the base since docs/BASE_ZRM_ZRS.md re-graded ADOPT) trained
with a smooth recall at five added to lean_gpu's loss (listwiseD, the loss of every fit since step 1):

    loss = listwiseD + LAM * mean over the questions with a gold of (1 - (1/|G|) sum_{g in G} h_g)
    h_g  = sigmoid((K + 0.5 - r_g) / TAU_R)
    r_g  = 1 + sum_{n in N} sigmoid((s_n - s_g) / TAU) + sum_{g' in G, g' != g} [sigmoid((s_g' - s_g) / TAU)]

G the question's gold rows in the pool, N its other rows, [.] counted without a gradient. r_g is g's rank in its pool
made smooth (ApproxNDCG's form, Qin, Liu and Li 2010), and h_g whether that rank is at most K made smooth (the Recall@k
surrogate of Patel, Tolias and Matas, CVPR 2022). Two golds that swap never change recall at five, so the terms between
golds count toward the rank and carry no gradient: no gold is pushed down by another, and no row but a gold is pushed
up. K = 5 (R@5, the screens' metric), TAU = 0.1 score units (a row one unit below a gold adds 4.5e-5 to its rank, so the
far rows of a 2,000-row pool do not inflate it), TAU_R = 1 rank, LAM = 1.

Why: R@5 is what every screen and grade decides on, and listwiseD trains for it only through the softmax over the whole
pool. Where zrm loses R@5 it mostly finds a question's golds in part: 0.737 and 0.765 of musique's lost R@5 in zrm's
screen fits, 0.875 and 0.887 of 2wiki's (outputs/diag/goldsplit-zrm). The surrogate's gradient sits on the golds just
below the fifth row and the rows just above them, which is where a partly found question's next gold is.

The objective acts in training only, so zrk reads and serves as zrm does. No new column, block, carve, graph or model.
Every training question, on every dataset, takes the same loss. Each training run records, per training carve, its
questions with a gold, with two golds or more, its golds, its largest pool, and the largest gold-by-row count of one
question (the surrogate's pairs): screen.json's 'zrk' 'census'. Before the first epoch, the loss and its gradient on the
first questions of every training carve, at the fit's start, are checked against a float64 reference (`check`); a
mismatch stops the fit (exit 1).

    python outputs/mp_unified/zrk.py train --split L-musique --name scr-zrk --arm zrk --device cuda --host
    python outputs/mp_unified/zrk.py read --name scr-zrk --device cuda --host
    python outputs/mp_unified/zrk.py compare --new outputs/screen/fits/scr-zrk \\
        --base outputs/screen/fits/scr-zrm,outputs/screen/fits/scr-zret,outputs/step1/fits/L-musique \\
        --out outputs/screen/scr-zrk
    python outputs/mp_unified/zrk.py pair --screens outputs/screen/scr-zrk.json,outputs/screen/scr-zrk-hp.json \\
        --out outputs/screen/scr-zrk-pair
    python outputs/mp_unified/zrk.py recall --null N1,N2,N3,N4 --pair outputs/screen/scr-zrk-pair.json \\
        --out outputs/screen/scr-zrk-pair-recall
    python outputs/mp_unified/zrk.py gate --recall outputs/screen/scr-zrk-pair-recall.json \\
        --base outputs/zbase/grade-nullx.json
    python outputs/mp_unified/zrk.py grade --full-root outputs/full_zrk \\
        --reuse L-musique=outputs/screen/scr-zrk.json,L-hotpotqa=outputs/screen/scr-zrk-hp.json
    python outputs/mp_unified/zrk.py --selftest
N1..N4 are the seed null's comparisons (outputs/screen/scr-null-s1, -s2, -s1-hp, -s2-hp .json).

Each read is decided against zrm's fit of its split: relz.py's pair, re-call and grade under zrk's name with zrm's fits
in place of rel's (zrc.py's mapping: zrm's screen fits scr-zrm and scr-zrm-hp, its full run's fits on the other splits;
the re-call's and the re-grade's base R@5 from outputs/zrc/base-zrm-<split>.json). zret's and step 1's fits are reported
beside. The full run starts only when the screen's re-call is PROMISING and zrm is the base: `gate`.

Speed: the objective changes training only. zrk reads and serves as zrm does, so any latency figure for it is zrm's,
and cold (8216ffe): each question timed from scratch, the walk, the move of its entries to the device and the forward,
with no warm-up pass and nothing kept from an earlier question.
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

import zrc as ZC  # noqa: E402

ZM, RM = ZC.ZM, ZC.RM
Z, R = RM.Z, RM.R
S2, S = RM.S2, RM.S
LG, LC = S.LG, S.LC
SR = Z.SR
S3 = ZM.S3
log = S.log

ARM, BASE_ARM = "zrk", "zrm"
ROUND, DOC = "twentieth", "docs/FULL_ROUND20.md"
K, TAU, TAU_R, LAM = 5, 0.1, 1.0, 1.0
CHECK_Q = 8
LISTWISE = LG.listwiseD                 # lean_gpu's loss, held before any swap
S.ARMS.update({ARM: S.ARMS[BASE_ARM]})
STATS = {"census": None, "check": None}


# ── the objective ────────────────────────────────────────────────────────────


def recall_k(scores, gold, nq, B, k=K, tau=TAU, tau_r=TAU_R):
    """The smooth recall at k of the module's docstring, as a loss: the mean over the questions with a gold in the pool
    of 1 - (1/|G|) sum_g h_g. lean_gpu.listwiseD's signature, on the scores' device and dtype."""
    dev, dt = scores.device, scores.dtype
    g = gold.to(torch.bool)
    gi = torch.nonzero(g).squeeze(1)
    if gi.numel() == 0:
        return scores.sum() * 0.0
    n = nq.numel()
    cnt = torch.zeros(B, dtype=torch.int64, device=dev).index_add_(0, nq, torch.ones_like(nq))
    order = torch.argsort(nq, stable=True)
    start = torch.cumsum(cnt, 0) - cnt
    pos = torch.empty_like(nq)
    pos[order] = torch.arange(n, device=dev) - start[nq[order]]          # each row's place in its pool
    width = int(cnt.max())
    sp = torch.zeros((B, width), dtype=dt, device=dev).index_put((nq, pos), scores)
    gp = torch.zeros((B, width), dtype=torch.bool, device=dev).index_put((nq, pos), g)
    vp = torch.zeros((B, width), dtype=torch.bool, device=dev).index_put((nq, pos), torch.ones_like(g))
    qg = nq[gi]
    me = torch.arange(gi.numel(), device=dev)
    sig = torch.sigmoid((sp[qg] - scores[gi].unsqueeze(1)) / tau)        # (golds, width): row j above gold g
    valid = vp[qg]
    valid[me, pos[gi]] = False                                            # not g itself
    other = gp[qg] & valid                                                # the question's other golds
    rest = valid & ~other                                                 # its other rows
    zero = torch.zeros((), dtype=dt, device=dev)
    r = 1.0 + torch.where(rest, sig, zero).sum(1) + torch.where(other, sig.detach(), zero).sum(1)
    h = torch.sigmoid((k + 0.5 - r) / tau_r)
    ng = torch.zeros(B, dtype=dt, device=dev).index_add_(0, qg, torch.ones_like(h))
    sh = torch.zeros(B, dtype=dt, device=dev).index_add_(0, qg, h)
    has = ng > 0
    return (1.0 - sh[has] / ng[has]).mean()


def objective(scores, gold, nq, B):
    """zrk's loss: listwiseD plus LAM times the smooth recall at K."""
    return LISTWISE(scores, gold, nq, B) + LAM * recall_k(scores, gold, nq, B)


def rk_numpy(s, gold, nq, B, k=K, tau=TAU, tau_r=TAU_R):
    """recall_k in float64 numpy, question by question and gold by gold: (loss, d loss / d s)."""
    s, gold, nq = np.asarray(s, np.float64), np.asarray(gold, bool), np.asarray(nq, np.int64)
    sigm = lambda x: 0.5 * (1.0 + np.tanh(0.5 * x))  # noqa: E731
    grad, terms = np.zeros_like(s), []
    for q in range(B):
        i = np.flatnonzero(nq == q)
        gi, ni = i[gold[i]], i[~gold[i]]
        if gi.size == 0:
            continue
        gq = np.zeros_like(s)
        hs = []
        for g in gi:
            sn = sigm((s[ni] - s[g]) / tau)
            og = gi[gi != g]
            r = 1.0 + sn.sum() + sigm((s[og] - s[g]) / tau).sum()
            h = sigm((k + 0.5 - r) / tau_r)
            hs.append(h)
            c = h * (1.0 - h) / tau_r / gi.size                       # -d(term)/dr_g ... the term is 1 - mean h
            dn = c * sn * (1.0 - sn) / tau                              # d(term)/ds_n = c dr/ds_n, r rising in s_n
            gq[ni] += dn
            gq[g] -= dn.sum()
        terms.append(1.0 - float(np.mean(hs)))
        grad += gq
    if not terms:
        return 0.0, grad
    return float(np.mean(terms)), grad / len(terms)


def hard_recall(s, gold, nq, B, k=K):
    """Per question with a gold: the share of its golds among its k highest rows (scores distinct)."""
    s, gold, nq = np.asarray(s, np.float64), np.asarray(gold, bool), np.asarray(nq, np.int64)
    out = []
    for q in range(B):
        i = np.flatnonzero(nq == q)
        if not gold[i].any():
            continue
        top = i[np.argsort(-s[i], kind="stable")[:k]]
        out.append(gold[top].sum() / gold[i].sum())
    return np.asarray(out)


def census(tr):
    """Per training carve (dataset/carve): its questions, those with a gold in the pool, with two or more, its golds,
    its largest pool, and the largest golds-by-rows count of one question (recall_k's pairs)."""
    out = {}
    for c in tr:
        key = f"{c.ds}/{c.carve}"
        if key in out:
            continue
        n = np.asarray(c.n_np, np.int64)
        gold = c.gold.detach().to("cpu").numpy().astype(np.float64)
        kq = np.rint(np.bincount(np.repeat(np.arange(n.size), n), weights=gold, minlength=n.size)).astype(np.int64)
        out[key] = {"questions": int(n.size), "with_gold": int((kq > 0).sum()), "two_or_more": int((kq > 1).sum()),
                    "golds": int(kq.sum()), "largest_pool": int(n.max()) if n.size else 0,
                    "largest_pairs": int((kq * n).max()) if n.size else 0}
    return out


def check_loss(s, gold, nq, B, tol=1e-4):
    """recall_k on scores s against rk_numpy in float64: the loss, and its gradient relative to the largest."""
    s1 = s.detach().clone().requires_grad_(True)
    v = recall_k(s1, gold, nq, B)
    v.backward()
    ref, rgrad = rk_numpy(s.detach().cpu().double().numpy(), gold.cpu().numpy().astype(bool), nq.cpu().numpy(), B)
    gr = s1.grad.detach().cpu().double().numpy()
    rec = {"questions": int(B), "loss": float(v), "loss_ref": ref, "loss_err": abs(float(v) - ref) / max(1.0, abs(ref)),
           "grad_err": float(np.abs(gr - rgrad).max()) / max(1e-12, float(np.abs(rgrad).max())),
           "grads_finite": bool(np.isfinite(gr).all())}
    rec["ok"] = bool(rec["grads_finite"] and rec["loss_err"] < tol and rec["grad_err"] < tol)
    return rec["ok"], rec


def carve_check(tr, blocks, hidden, seed, device):
    """check_loss on the first CHECK_Q questions of every training carve, scored by the arm's model at the fit's start
    (seed, every block kept, eval). batch() only gathers rows, and the fit seeds itself after, so the fit is unchanged."""
    out, ok = {}, True
    widths = {b: tr[0].widths[b] for b in blocks}
    torch.manual_seed(seed)
    m = LG.LeanMLP8D(blocks, widths, hidden, dropout=0.1, seed=seed, arm="ctl", ctx="none").to(device).eval()
    for c in tr:
        key = f"{c.ds}/{c.carve}"
        if key in out:
            continue
        qs = np.arange(min(CHECK_Q, c.rows), dtype=np.int64)
        feats, nq, bz, gold = c.batch(qs, blocks)
        keep = torch.ones((qs.size, len(blocks)), dtype=torch.float32, device=device)
        with torch.no_grad():
            s = m(feats, keep, nq, qs.size, bz)
        good, out[key] = check_loss(s.to(torch.float32), gold, nq, qs.size)
        ok = ok and good
        del feats
    del m
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return ok, out


@contextlib.contextmanager
def rk_loss():
    """lean_gpu's loop with objective in place of listwiseD (its one loss); the first fit's training carves counted and
    checked before it trains (a failed check stops it); both restored after, an error too."""
    saved_loss, saved_fit = LG.listwiseD, LG.fit_variant

    def fit_variant(tr, blocks, cfg, seed, hidden, ctx, device, *args, **kw):
        if STATS["census"] is None:
            STATS["census"] = census(tr)
            ok, STATS["check"] = carve_check(tr, blocks, hidden, seed, device)
            log(f"zrk check: {json.dumps(STATS['check'])}")
            if not ok:
                raise SystemExit("zrk: the loss on a training carve does not match its float64 reference")
        return saved_fit(tr, blocks, cfg, seed, hidden, ctx, device, *args, **kw)

    LG.listwiseD, LG.fit_variant = objective, fit_variant
    try:
        yield
    finally:
        LG.listwiseD, LG.fit_variant = saved_loss, saved_fit


# ── train, read and compare ──────────────────────────────────────────────────


def train(argv, split):
    """rmatch.py's train (zrm's settings and model) for the arm zrk, under objective; this file's records stamped."""
    arm = R.arm_of(argv)
    if arm != ARM:
        raise SystemExit(f"zrk: train takes --arm {ARM}, not {arm}")
    if S.ARMS.get(ARM) != (ZM.ZRM, RM.ChainCarveBase) or S.ARMS.get(BASE_ARM) != (ZM.ZRM, RM.ChainCarveBase):
        raise SystemExit(f"zrk: the arm {ARM} is {S.ARMS.get(ARM)}, not zrm's model on rmatch's chain carve")
    STATS["census"], STATS["check"] = None, None
    with rk_loss():
        rc = RM.train(argv, split)
    name, out_root = S2.where(argv)
    sj = out_root / name / "screen.json"
    if sj.exists():
        rec = json.loads(sj.read_text(encoding="utf-8"))
        rec.update({"zrk_sha256": LC.sha_src(__file__), "zrm_sha256": LC.sha_src(ZM.__file__),
                    "lean_screen3_sha256": LC.sha_src(S3.__file__),
                    "zrk": {"model": "zrm.ZRM (unchanged)", "carve": "rmatch.ChainCarveBase",
                            "loss": "listwiseD + LAM * (1 - smooth recall at K), zrk.objective",
                            "K": K, "TAU": TAU, "TAU_R": TAU_R, "LAM": LAM,
                            "census": STATS["census"], "check": STATS["check"]}})
        LC.write_json(sj, rec)
    return rc


def read(argv):
    name, out_root = S2.where(argv)
    arm = json.loads((out_root / name / "screen.json").read_text(encoding="utf-8"))["arm"]
    if arm != ARM:
        raise SystemExit(f"zrk: {name} was trained as {arm}, not {ARM}")
    return RM.read(argv)


# ── relz.py's pair, re-call and grade, decided against zrm's fits ────────────


@contextlib.contextmanager
def on_zrm():
    """relz.py's records under zrk's name, each read decided against zrm's fit of its split (zrc.py's mapping); relz
    restored after."""
    with ZC.on_zrm():
        Z.ARM = ARM
        yield


FIX = (("(rel's screen fits scr-rel and scr-rel-hp)", "(zrm's screen fits scr-zrm and scr-zrm-hp)"),
       ("(rel's screen fits)", "(zrm's screen fits)"),
       ("| rel R@5 | relz R@5 |", "| zrm R@5 | zrk R@5 |"),
       ("section 2 and the tenth round", f"section 2 and the {ROUND} round"),
       ("docs/FULL_ROUND10.md", DOC))


def restamp(out):
    """A record relz.py wrote under zrk's name: zrm named as the base in its md, this round and this file's sha added."""
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
        rec.update({"decided_against": "zrm's fit of each split", "round": ROUND, "zrk_sha256": LC.sha_src(__file__)})
        LC.write_json(js, rec)


def pair(screens, out):
    with on_zrm():
        rec = Z.pair(screens, out)
    restamp(out)
    return rec


def recall(null_files, pair_file, out, zrm_screens=None):
    with on_zrm():
        rec = Z.recall(null_files, pair_file, out, zrm_screens)
    restamp(out)
    return rec


def grade(full_root, reuse, out, check_fits=True):
    with on_zrm():
        rec = Z.grade(full_root, reuse, out, check_fits)
    restamp(out)
    return rec


# ── the full run's gate ──────────────────────────────────────────────────────


def gate(recall_file, base_file):
    """0 when the screen's re-call is PROMISING and zrm is the base (docs/BASE_ZRM_ZRS.md's re-grade ADOPT), 1
    otherwise; a missing file, or one that is not the declared record, stops (2)."""
    got = []
    for fn, arm, want in ((recall_file, ARM, "zrm's fit of each split"), (base_file, "zrm", "zrs's fit of each split")):
        p = Path(fn)
        if not p.exists():
            raise SystemExit(f"zrk gate: {p} missing")
        rec = json.loads(p.read_text(encoding="utf-8"))
        if rec.get("arm") != arm or rec.get("decided_against") != want:
            raise SystemExit(f"zrk gate: {p} is arm {rec.get('arm')}'s, decided against "
                             f"{rec.get('decided_against')!r}; not the declared record")
        got.append(rec.get("verdict"))
    if "null_splits" not in json.loads(Path(base_file).read_text(encoding="utf-8")):
        raise SystemExit(f"zrk gate: {base_file} is not a re-grade under the null")
    ok = got[0] == "PROMISING" and got[1] == "ADOPT"
    log(f"zrk gate: the re-call {got[0]}, zrm against zrs {got[1]} -> {'the full run starts' if ok else 'no full run'}")
    return 0 if ok else 1


# ── selftest ─────────────────────────────────────────────────────────────────


def toy_pools(sizes, golds, seed, scale=1.0, dtype=torch.float64):
    """Pools of the given sizes, the first golds[q] rows of pool q gold, rows shuffled: (scores, gold, nq, B)."""
    rng = np.random.default_rng(seed)
    nq = np.repeat(np.arange(len(sizes)), sizes)
    gold = np.concatenate([np.arange(n) < kk for n, kk in zip(sizes, golds)])
    perm = rng.permutation(nq.size)
    s = torch.from_numpy(rng.standard_normal(nq.size) * scale).to(dtype)
    return s[perm], torch.from_numpy(gold[perm]), torch.from_numpy(nq[perm]), len(sizes)


def grad_of(fn, s, gold, nq, B):
    x = s.detach().clone().requires_grad_(True)
    v = fn(x, gold, nq, B)
    v.backward()
    return v.detach(), x.grad.detach()


class Untyped:
    """A toy carve without its chains: an untyped graph's batches (the same rows, blocks and golds)."""

    def __init__(self, c, ds):
        self.c, self.ds, self.carve, self.chains = c, ds, c.carve, None
        self.rows, self.widths, self.n_np, self.off_np, self.gold = c.rows, c.widths, c.n_np, c.off_np, c.gold

    def nbytes(self):
        return 0

    def batch(self, qs, blocks):
        feats, nq, bz, gold = self.c.batch(qs, blocks)
        return {k: v for k, v in feats.items() if k != RM.CH_KEY}, nq, bz, gold


def selftest():
    t0 = time.time()
    LG.bind_device_ops()
    # 1. the arm is zrm's model on rmatch's chain carve, zrm's own arm unchanged; train takes no other arm
    assert S.ARMS[ARM] == (ZM.ZRM, RM.ChainCarveBase) and S.ARMS[BASE_ARM] == S.ARMS[ARM]
    assert LISTWISE is LG.listwiseD
    SR.must_stop(train, ["train", "--arm", BASE_ARM, "--name", "x"], "L-musique")
    SR.must_stop(train, ["train", "--name", "x"], "L-musique")
    # 2. the loss against its float64 reference: pools with no gold, one, two, five, every row gold, a pool of one gold
    #    row, a pool of 40 rows, rows in any order; float64 to 1e-9, float32 to 1e-5 (relative to the largest gradient)
    sizes, golds = [6, 7, 9, 12, 4, 1, 3, 15, 40], [0, 1, 2, 5, 4, 1, 1, 0, 3]
    for seed in range(6):
        for scale in (0.05, 0.3, 2.0):
            s, gold, nq, B = toy_pools(sizes, golds, seed, scale)
            ref, rgrad = rk_numpy(s.numpy(), gold.numpy(), nq.numpy(), B)
            v, gr = grad_of(recall_k, s, gold, nq, B)
            assert abs(float(v) - ref) < 1e-9 and np.abs(gr.numpy() - rgrad).max() < 1e-9, (float(v), ref)
            ok, chk = check_loss(s.float(), gold, nq, B)
            assert ok, chk
            assert torch.isfinite(gr).all()
            # 3. no gold is pushed down and no other row up; a pool whose rows are all gold, or one of a single row,
            #    takes no gradient
            assert bool((gr[gold] <= 0).all()) and bool((gr[~gold] >= 0).all())
            flat = np.isin(nq.numpy(), [4, 5])
            assert np.all(gr.numpy()[flat] == 0.0)
    # 4. two golds that swap their scores leave the loss unchanged (recall at five does not change either)
    s, gold, nq, B = toy_pools([30, 25], [4, 3], 11, 0.3)
    v0 = float(recall_k(s, gold, nq, B))
    gi = torch.nonzero(gold & (nq == 0)).squeeze(1)
    s2 = s.clone()
    s2[gi[0]], s2[gi[1]] = s[gi[1]], s[gi[0]]
    assert abs(float(recall_k(s2, gold, nq, B)) - v0) < 1e-12
    # 5. sharp temperatures give one minus the hard recall at five, from lean_gpu's own ranking of the rows
    for seed in range(4):
        s, gold, nq, B = toy_pools([12, 30, 7, 50, 3], [2, 4, 1, 9, 1], 20 + seed, 1.0)
        hard = hard_recall(s.numpy(), gold.numpy(), nq.numpy(), B)
        v = float(recall_k(s, gold, nq, B, tau=1e-6, tau_r=1e-3))
        assert abs(v - (1.0 - hard.mean())) < 1e-6, (v, 1.0 - hard.mean())
        o = torch.argsort(nq, stable=True)                                # question-major, as lean_gpu's reads
        cnt = torch.bincount(nq, minlength=B)
        top, _hit = LG.top_hit(s[o].float(), gold[o], B, torch.cumsum(cnt, 0) - cnt, cnt)
        gt = np.bincount(nq.numpy(), weights=gold.numpy().astype(np.float64), minlength=B)
        m = LG.metrics_of(top.cpu().numpy(), np.zeros(B), gt)
        assert np.allclose(m[gt > 0, 0], hard)
    # 6. the gradient sits at the boundary: in one pool spaced 0.2 apart, a gold sixth gets a far larger push than one
    #    first or fortieth
    s = torch.linspace(4.0, -4.0, 41, dtype=torch.float64)
    nq = torch.zeros(41, dtype=torch.int64)
    gold = torch.zeros(41, dtype=torch.bool)
    gold[[0, 5, 39]] = True
    _v, gr = grad_of(recall_k, s, gold, nq, 1)
    assert -gr[5] > 10 * -gr[0] and -gr[5] > 1000 * -gr[39] and float(-gr[5]) > 0.05, (gr[0], gr[5], gr[39])
    # 7. edge cases: no gold anywhere (0, no gradient), extreme scores (finite), a single-row batch
    s, gold, nq, B = toy_pools([3, 4], [0, 0], 1, dtype=torch.float32)
    v, gr = grad_of(recall_k, s, gold, nq, B)
    assert float(v) == 0.0 and bool((gr == 0).all())
    s = torch.tensor([-80.0, 80.0, 79.0, 80.0, -80.0, -79.0])
    gold, nq = torch.tensor([True, False, False, True, False, False]), torch.tensor([0, 0, 0, 1, 1, 1])
    v, gr = grad_of(recall_k, s, gold, nq, 2)
    assert bool(torch.isfinite(v)) and bool(torch.isfinite(gr).all())
    v, gr = grad_of(recall_k, torch.tensor([0.3]), torch.tensor([True]), torch.tensor([0]), 1)
    assert bool(torch.isfinite(v)) and float(gr.abs().sum()) == 0.0
    # 8. the objective is listwiseD plus LAM times recall_k
    s, gold, nq, B = toy_pools(sizes, golds, 3, 1.0, torch.float32)
    assert abs(float(objective(s, gold, nq, B)) - float(LG.listwiseD(s, gold, nq, B))
               - LAM * float(recall_k(s, gold, nq, B))) < 1e-5
    # 9. the swap: inside, lean_gpu's loss is the objective and its loop counts and checks the carves; restored after,
    #    an error too
    saved = LG.listwiseD, LG.fit_variant
    with rk_loss():
        assert LG.listwiseD is objective and LG.fit_variant is not saved[1]
    assert (LG.listwiseD, LG.fit_variant) == saved
    try:
        with rk_loss():
            raise KeyError("x")
    except KeyError:
        pass
    assert (LG.listwiseD, LG.fit_variant) == saved
    tmp = Path(tempfile.mkdtemp(prefix="zrk_"))
    try:
        # 10. toy fits through lean_gpu's loop with zrm's model on a typed and an untyped toy carve: under the swap the
        #     fit is checked and counted before it trains, differs from zrm's own, and repeats bit for bit; zrm's loop
        #     is untouched after
        rng = np.random.default_rng(21)
        RM.toy_roots(tmp, rng, chunks=(12, 12))
        RM.build("metaqa", "toy", out_root=tmp / "ch", cache_root=tmp / "cache", look_root=tmp / "look",
                 rel_dir=tmp / "rel")
        c = RM.chain_carve(ZM.ToyZ, tmp / "ch", tmp / "rel")("metaqa", "toy", "2wiki", "cpu", tmp / "cache")
        u = Untyped(c, "2wiki")
        blocks = ["rank", "SEMB"]
        cfg = {"lr": 2e-2, "wd": 1e-4, "dropout": 0.1, "epochs": 3, "swa_from": 1, "cos": False, "adamw": False,
               "drop": False}
        with S.patched(ARM):
            assert isinstance(LG.LeanMLP8D(blocks, c.widths, 16), ZM.ZRM)
            ref = LG.fit_variant([c, u], blocks, cfg, 0, 16, "none", "cpu", tag="toy-zrm")
            STATS["census"], STATS["check"] = None, None
            with rk_loss():
                f1 = LG.fit_variant([c, u], blocks, cfg, 0, 16, "none", "cpu", tag="toy-zrk")
                cen, chk = STATS["census"], STATS["check"]
                f2 = LG.fit_variant([c, u], blocks, cfg, 0, 16, "none", "cpu", tag="toy-zrk repeat")
            ref2 = LG.fit_variant([c, u], blocks, cfg, 0, 16, "none", "cpu", tag="toy-zrm again")
        every = lambda f: f["states"] + [f["swa"]]  # noqa: E731
        assert all(not LG.same_state(a, b) for a, b in zip(every(f1), every(f2)))     # same_state: the keys that differ
        assert all(not LG.same_state(a, b) for a, b in zip(every(ref), every(ref2)))
        assert LG.same_state(ref["swa"], f1["swa"])
        assert all(np.isfinite(r["loss"]) for r in f1["curve"])
        assert sorted(chk) == ["2wiki/toy", "metaqa/toy"] and all(v["ok"] for v in chk.values()), chk
        n = np.asarray(c.n_np, np.int64)
        kq = [int(c.gold[int(c.off_np[q]):int(c.off_np[q + 1])].sum()) for q in range(c.rows)]
        want = {"questions": c.rows, "with_gold": sum(x > 0 for x in kq), "two_or_more": sum(x > 1 for x in kq),
                "golds": sum(kq), "largest_pool": int(n.max()), "largest_pairs": int(max(x * m for x, m in zip(kq, n)))}
        assert cen["metaqa/toy"] == want and cen["2wiki/toy"] == want, (cen, want)
        # 11. a failed check stops the fit before it trains
        saved_rk = globals()["recall_k"]
        globals()["recall_k"] = lambda s_, g_, n_, b_, **kw: saved_rk(s_, g_, n_, b_, **kw) * 1.01
        try:
            STATS["census"], STATS["check"] = None, None
            with S.patched(ARM), rk_loss():
                LG.fit_variant([c, u], blocks, cfg, 0, 16, "none", "cpu", tag="toy-zrk broken")
            raise AssertionError("zrk trained past a failed check")
        except SystemExit as e:
            assert "float64 reference" in str(e)
        finally:
            globals()["recall_k"] = saved_rk
        assert (LG.listwiseD, LG.fit_variant) == saved
        # 12. read refuses another arm's fit
        (tmp / "fits" / "x").mkdir(parents=True)
        LC.write_json(tmp / "fits" / "x" / "screen.json", {"arm": BASE_ARM})
        try:
            read(["--name", "x", "--out-root", str(tmp / "fits")])
            raise AssertionError("zrk read zrm's fit")
        except SystemExit as e:
            assert "trained as zrm" in str(e)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    # 13. relz's pair, re-call and grade under zrk's name, decided against zrm's fits; relz restored after
    saved = Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit
    with on_zrm():
        assert Z.tag({})["arm"] == ARM and Z.BASE_ARM == BASE_ARM
        assert Z.rel_fit("J5") == ("full_zrm", "fits", "J5") and Z.rel_fit("L-musique") == ("screen", "fits", "scr-zrm")
    assert (Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit) == saved
    with tempfile.TemporaryDirectory() as td:
        T = Path(td)

        # 14. the gate: PROMISING and ADOPT -> 0; anything else -> 1; a wrong or missing record stops
        def rec(name, **kw):
            p = T / f"{name}.json"
            p.write_text(json.dumps(kw), encoding="utf-8")
            return str(p)

        rc_p = rec("rp", arm=ARM, decided_against="zrm's fit of each split", verdict="PROMISING")
        rc_m = rec("rm", arm=ARM, decided_against="zrm's fit of each split", verdict="MIXED")
        rc_c = rec("rc", arm="zkind", decided_against="zrm's fit of each split", verdict="PROMISING")
        b_a = rec("ba", arm="zrm", decided_against="zrs's fit of each split", verdict="ADOPT", null_splits=["J5"])
        b_n = rec("bn", arm="zrm", decided_against="zrs's fit of each split", verdict="NOT_ADOPTED", null_splits=["J5"])
        b_f = rec("bf", arm="zrm", decided_against="zrs's fit of each split", verdict="ADOPT")         # filed, not nullx
        assert gate(rc_p, b_a) == 0 and gate(rc_m, b_a) == 1 and gate(rc_p, b_n) == 1
        SR.must_stop(gate, rc_c, b_a)
        SR.must_stop(gate, rc_p, b_f)
        SR.must_stop(gate, b_a, rc_p)
        SR.must_stop(gate, rc_p, str(T / "none.json"))
        # 15. the pair, the re-call and the grade against zrm's fits, named for this round
        null1 = SR.fake_null(T / "null", "n1", {("L-hotpotqa", "hotpotqa"): (-0.010, 0.012)})   # floor 0.0221
        zs = {sp: str(SR.fake_compare(T / ZC.zrm_fit(sp)[0], ZC.zrm_fit(sp)[2], sp, {ds: 0.1 for ds in LG.EVAL_ORDER},
                                      arm=BASE_ARM))
              for sp in ZC.ZRM_FITS}                                            # zrm's fits against step 1's (0.6)
        zb = {sp: "/".join(("outputs",) + ZC.zrm_fit(sp)) for sp in ZC.ZRM_FITS}
        za = SR.fake_compare(T / "z", "scr-zrk", "L-musique", {"musique": 0.0215}, arm=ARM, base_r5=0.6,
                             base=zb["L-musique"])
        zh = SR.fake_compare(T / "z", "scr-zrk-hp", "L-hotpotqa", {"hotpotqa": -0.0147}, arm=ARM, base_r5=0.6,
                             base=zb["L-hotpotqa"])
        st1 = SR.fake_compare(T / "z2", "scr-zrk", "L-musique", {}, arm=ARM)       # decided against step 1's fit
        SR.must_stop(pair, [st1, zh], T / "bad")
        pr = pair([za, zh], T / "pz")
        assert pr["verdict"] == "MIXED" and pr["arm"] == ARM
        pj = json.loads((T / "pz.json").read_text(encoding="utf-8"))
        assert pj["round"] == ROUND and pj["decided_against"] == "zrm's fit of each split"
        r = recall(null1, T / "pz.json", T / "rz", zs)
        assert r["verdict"] == "PROMISING" and r["changed"] == ["L-hotpotqa hotpotqa LOSS -> WITHIN"], r["changed"]
        t = (T / "rz.md").read_text(encoding="utf-8")
        assert f"{ROUND} round" in t and "tenth round" not in t and "| zrm R@5 | zrk R@5 |" in t
        assert gate(str(T / "rz.json"), b_a) == 0
        full = T / "full"
        full.mkdir()
        for sp in S2.SPLITS:
            if sp not in ZC.ZRM_FITS:
                SR.fake_compare(full / "src", f"fit-{sp}", sp, {"musique": 0.03} if sp == "J5" else {}, arm=ARM,
                                base="/".join(("outputs",) + ZC.zrm_fit(sp))).replace(full / f"compare-{sp}.json")
        g = grade(full, {"L-musique": str(za), "L-hotpotqa": str(zh)}, full / "grade", check_fits=False)
        assert g["verdict"] == "NOT_ADOPTED" and g["losses"] == 1 and g["arm"] == ARM, (g["verdict"], g["losses"])
        t = (full / "grade.md").read_text(encoding="utf-8")
        assert DOC in t and "FULL_ROUND10" not in t
    assert (Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit) == saved
    log(f"zrk selftest: the arm is zrm's model and carve and train takes no other; the smooth recall at five matches its "
        f"float64 reference (pools with no gold, one, several, all rows gold, rows in any order, three score scales), "
        f"never pushes a gold down or another row up, is unchanged when two golds swap, is one minus the hard recall "
        f"at sharp temperatures, pushes hardest at the fifth row, and stays finite at the edges; the objective is "
        f"listwiseD plus it; the swap restores; a toy fit under it is checked and counted by hand, differs from zrm's "
        f"and repeats bit for bit, and a failed check stops it; the gate; relz's pair, re-call and grade against zrm's "
        f"fits, named for the {ROUND} round ({time.time() - t0:.1f}s): ok")
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
    if k.cmd == "train":
        return train([k.cmd] + rest, k.split)          # as zrm.py's main
    if k.cmd == "read":
        return read(rest)
    if k.cmd == "compare":
        return S2.main([k.cmd] + rest)
    if k.cmd == "gate":
        gp = argparse.ArgumentParser()
        gp.add_argument("--recall", required=True)
        gp.add_argument("--base", required=True)
        g = gp.parse_args(rest)
        try:
            return gate(g.recall, g.base)
        except SystemExit as e:
            log(f"zrk gate: {e}")
            return 2
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
            log(f"zrk {k.cmd}: {e}")
            return 2
        return 0
    raise SystemExit("zrk: train, read, compare, pair, recall, gate, grade, or --selftest")


if __name__ == "__main__":
    sys.exit(main())
