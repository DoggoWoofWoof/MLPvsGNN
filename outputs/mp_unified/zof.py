"""Screens, thirty-second round (docs/SCREENS.md; stage S1 of docs/PROGRAM_2026_10_09.md): a query-conditioned offset
from the first stage's leading rows, on zrc, the MLP's base.

zof is zrc's model (zrm.ZRM over zrc.ChainCarveZRC) with a small head added to its score. Why: D1 found the golds zrc
misses linked to rows it found (2wiki 0.77, musique 0.38/0.77 on 2-hop), D2 found zrc's per-row inputs at their limit
(FEATURE_LIMIT), and rounds 29-31 found that counting the leading rows' edges (zbr) or propagating scores over them (zdp,
zg1) adds nothing. None of those asks what the next hop's text should be, given the question and the row already found.
S1's offset space does: a relation, set by the question, that maps a found row onto the row it leads to. zof learns it
end to end, in DistMult's form (a diagonal relation per question), from the store vectors zrc already reads for SEMB:

- the anchors are the first stage's leading rows: each question's top K rows by rrf (the batch's base_z, rrf's z-score
  in its pool, ranked in lean_gpu.top_hit's order by zlink.pool_rank). No row's score enters, so the head is not
  message passing (docs/SCREENS.md, "rounds twenty-two and twenty-three are message passing"); it reads fixed rows as
  SEED and DISTS do, and needs no edge;
- per channel c (C channels, each Dd wide): a = norm(P Wa_c) for the anchor, r = norm(q Wq_c) for the question,
  t = norm(P Wt_c) for the row (P: the row's decoded store vector with kappa, q: the question's embedding, as SEMB
  reads them; norm: unit length), and the score Dd * sum(a * r * t);
- the head's inputs per row (C * (K + 1) + 1): the score against each anchor in rrf's order (0 for the row itself and
  for a missing anchor), each channel's maximum over its anchors (0 when none), and whether the row is an anchor;
- head: Linear(inputs, HID), GELU, Linear(HID, 1). The last layer starts at zero and every other parameter is drawn from
  the module's own generator (seed + SEED_OFF), so at the start zof's forward is zrc's bit for bit and no draw of zrc's
  moves. Settings and training are zrm's (rmatch.py's train).

The same rule on all six datasets. No new graph, column, encoder, text or edge; the encoder's vectors are read, never
changed. Speed: per question K x C x Dd products per row, query-local, before nothing else (cold figures for zof belong
to their own declared stage).

    python outputs/mp_unified/zof.py train --split L-musique --name scr-zof --arm zof --device cuda --host
    python outputs/mp_unified/zof.py read --name scr-zof --device cuda --host
    python outputs/mp_unified/zof.py compare --new outputs/screen/fits/scr-zof \\
        --base outputs/screen/fits/scr-zrct,outputs/step1/fits/L-musique --out outputs/screen/scr-zof
    python outputs/mp_unified/zof.py pair --screens outputs/screen/scr-zof.json,outputs/screen/scr-zof-hp.json \\
        --out outputs/screen/scr-zof-pair
    python outputs/mp_unified/zof.py recall --null N1,N2,N3,N4 --pair outputs/screen/scr-zof-pair.json \\
        --out outputs/screen/scr-zof-pair-recall
    python outputs/mp_unified/zof.py --selftest
N1..N4 are the seed null's comparisons (outputs/screen/scr-null-s1, -s2, -s1-hp, -s2-hp .json).
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
import torch.nn as nn  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402

import zlink as ZL  # noqa: E402

ZC, ZM, RM = ZL.ZC, ZL.ZM, ZL.RM
Z, R = RM.Z, RM.R
S2, S = RM.S2, RM.S
LG, LC = S.LG, S.LC
SR = Z.SR
log = S.log

ARM, BASE_ARM = "zof", "zrc"
DECIDED = "zrc's fit of each split"
ZRC_FITS = {"L-musique": ("screen", "fits", "scr-zrct"), "L-hotpotqa": ("screen", "fits", "scr-zrct-hp")}
ZRC_SCREENS = {sp: f"outputs/zbase2/base-zrc-{sp}.json" for sp in ZRC_FITS}
K, C, DD, HID, SEED_OFF = 5, 4, 16, 32, 3201
N_IN = C * (K + 1) + 1
OF = ("of_wa", "of_wq", "of_wt", "of_w1", "of_b1", "of_w2", "of_b2")
SETTINGS = {"anchors": K, "channels": C, "width": DD, "hidden": HID, "inputs": N_IN, "seed_offset": SEED_OFF,
            "form": "DistMult: Dd * sum(norm(P_anchor Wa) * norm(q Wq) * norm(P_row Wt)) per channel",
            "anchors_by": "rrf (base_z) in lean_gpu.top_hit's order (zlink.pool_rank)"}


def zrc_fit(split):
    """zrc's fit of a split: its screen fits on L-musique and L-hotpotqa, its full run's (outputs/full_zrct) elsewhere."""
    return ZRC_FITS.get(split, ("full_zrct", "fits", split))


# ── the model ────────────────────────────────────────────────────────────────


@torch.no_grad()
def anchors(bz, nq, B):
    """(rank, A): each row's rrf rank in its pool, and A[b, k] the row holding rank k in question b (-1 if none)."""
    rank = ZL.pool_rank(bz.to(torch.float32), nq, B)
    A = torch.full((B, K), -1, dtype=torch.long, device=bz.device)
    m = rank < K
    A[nq[m], rank[m]] = torch.arange(nq.numel(), device=bz.device)[m]
    return rank, A


def offset_inputs(m, qe, P, rank, A, nq):
    """(N, N_IN): each row's channel scores against its question's anchors, their maxima and the anchor flag."""
    N = nq.numel()
    a = Fn.normalize(Fn.linear(P, m.of_wa).view(N, C, DD), dim=-1)
    t = Fn.normalize(Fn.linear(P, m.of_wt).view(N, C, DD), dim=-1)
    r = Fn.normalize(Fn.linear(qe, m.of_wq).view(qe.shape[0], C, DD), dim=-1)
    Aj = A[nq]                                                         # (N, K)
    valid = (Aj >= 0) & (Aj != torch.arange(N, device=nq.device).unsqueeze(1))
    aq = a.index_select(0, A.clamp_min(0).reshape(-1)).view(A.shape[0], K, C, DD)   # (B, K, C, DD), once per question
    sc = (aq.index_select(0, nq) * (r.index_select(0, nq) * t).unsqueeze(1)).sum(-1) * DD  # (N, K, C)
    v3 = valid.unsqueeze(-1)
    sc = torch.where(v3, sc, torch.zeros_like(sc))
    mx = torch.where(v3, sc, torch.full_like(sc, -float("inf"))).amax(1)
    mx = torch.where(torch.isfinite(mx), mx, torch.zeros_like(mx))
    return torch.cat([sc.reshape(N, K * C), mx, (rank < K).to(sc.dtype).unsqueeze(1)], 1)


class ZOff(ZM.ZRM):
    """zof: zrc's score plus a head (zero at the start) over each row's query-conditioned offset scores against the
    first stage's leading rows."""

    def __init__(self, blocks, widths, hidden=128, dropout=0.1, seed=0, arm="ctl", ctx="none"):
        super().__init__(blocks, widths, hidden, dropout, seed, arm, ctx)
        if "SEMB" not in self.blocks:
            raise SystemExit("zof reads SEMB's store vectors and question embedding: its block set must hold SEMB")
        g = torch.Generator().manual_seed(int(seed) + SEED_OFF)
        uni = lambda *shape, fan: (torch.rand(*shape, generator=g) * 2 - 1) / fan ** 0.5  # noqa: E731
        p_dim, q_dim = int(self.V.shape[0]), int(self.U.shape[0])
        self.of_wa = nn.Parameter(uni(C * DD, p_dim, fan=p_dim))
        self.of_wq = nn.Parameter(uni(C * DD, q_dim, fan=q_dim))
        self.of_wt = nn.Parameter(uni(C * DD, p_dim, fan=p_dim))
        self.of_w1 = nn.Parameter(uni(HID, N_IN, fan=N_IN))
        self.of_b1 = nn.Parameter(torch.zeros(HID))
        self.of_w2 = nn.Parameter(torch.zeros(1, HID))
        self.of_b2 = nn.Parameter(torch.zeros(1))

    def offset_head(self, x):
        return Fn.linear(Fn.gelu(Fn.linear(x, self.of_w1, self.of_b1)), self.of_w2, self.of_b2).squeeze(-1)

    def forward(self, feats, keep, nq, B, base_z):
        s = super().forward(feats, keep, nq, B, base_z)
        qe, P = feats["SEMB"]
        rank, A = anchors(base_z, nq, B)
        return s + self.offset_head(offset_inputs(self, qe, P, rank, A, nq))


S.ARMS.update({ARM: (ZOff, ZC.ChainCarveZRC)})


# ── train, read and compare ──────────────────────────────────────────────────


def train(argv, split):
    """rmatch.py's train (zrm's settings) for the arm zof: zrc's model plus the offset head over zrc's builds; records
    stamped."""
    arm = R.arm_of(argv)
    if arm != ARM:
        raise SystemExit(f"zof: train takes --arm {ARM}, not {arm}")
    if S.ARMS.get(ARM) != (ZOff, ZC.ChainCarveZRC) or S.ARMS.get(BASE_ARM) != (ZM.ZRM, ZC.ChainCarveZRC):
        raise SystemExit(f"zof: the arm {ARM} is {S.ARMS.get(ARM)}, not zof's model on zrc's carve")
    rc = RM.train(argv, split)
    name, out_root = S2.where(argv)
    sj = out_root / name / "screen.json"
    if sj.exists():
        rec = json.loads(sj.read_text(encoding="utf-8"))
        rec.update({"zof_sha256": LC.sha_src(__file__), "zrc_sha256": LC.sha_src(ZC.__file__),
                    "zrm_sha256": LC.sha_src(ZM.__file__),
                    "zof": {"model": "zof.ZOff (zrm.ZRM plus a head over query-conditioned offsets from rrf's top rows)",
                            "carve": "zrc.ChainCarveZRC", "params": list(OF), "settings": SETTINGS},
                    "zrc_records": RM.built_records(ZC.CH_OUT)})
        LC.write_json(sj, rec)
    return rc


def read(argv):
    name, out_root = S2.where(argv)
    arm = json.loads((out_root / name / "screen.json").read_text(encoding="utf-8"))["arm"]
    if arm != ARM:
        raise SystemExit(f"zof: {name} was trained as {arm}, not {ARM}")
    return RM.read(argv)


def compare(rest):
    return S2.main(["compare"] + rest)


# ── relz.py's pair and re-call, decided against zrc's fits ───────────────────


@contextlib.contextmanager
def on_zrc():
    """relz.py's records under zof's name, each read decided against zrc's fit of its split; relz restored after."""
    saved = Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit
    Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit = ARM, BASE_ARM, ZRC_FITS, ZRC_SCREENS, zrc_fit
    try:
        yield
    finally:
        Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit = saved


FIX = (("(rel's screen fits scr-rel and scr-rel-hp)", "(zrc's screen fits scr-zrct and scr-zrct-hp)"),
       ("(rel's screen fits)", "(zrc's screen fits)"),
       ("| rel R@5 | relz R@5 |", "| zrc R@5 | zof R@5 |"),
       ("section 2 and the tenth round", "section 2 and the thirty-second round"))


def restamp(out):
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
        rec.update({"decided_against": DECIDED, "round": "thirty-second", "zof_sha256": LC.sha_src(__file__)})
        LC.write_json(js, rec)


def pair(screens, out):
    with on_zrc():
        rec = Z.pair(screens, out)
    restamp(out)
    return rec


def recall(null_files, pair_file, out, zrc_screens=None):
    with on_zrc():
        rec = Z.recall(null_files, pair_file, out, zrc_screens)
    restamp(out)
    return rec


# ── selftest ─────────────────────────────────────────────────────────────────


def hand_inputs(m, qe, P, bz, n_np, qs):
    """offset_inputs by hand: per question, python ranks its rows and loops over anchors, channels and widths."""
    Wa, Wq, Wt = (getattr(m, n).detach().double().numpy() for n in ("of_wa", "of_wq", "of_wt"))
    Pn, qn, b = P.double().numpy(), qe.double().numpy(), bz.double().numpy()
    unit = lambda x: x / max(np.linalg.norm(x), 1e-12)  # noqa: E731
    out, at = [], 0
    for j, q in enumerate(qs):
        n = int(n_np[q])
        order = sorted(range(n), key=lambda i: (-b[at + i], i))
        anc = order[:K]
        rk = {i: r_ for r_, i in enumerate(order)}
        for i in range(n):
            sc = np.zeros((K, C))
            ok = np.zeros(K, bool)
            for k_, w in enumerate(anc):
                if w == i:
                    continue
                ok[k_] = True
                for c in range(C):
                    sl = slice(c * DD, (c + 1) * DD)
                    a = unit(Wa[sl] @ Pn[at + w])
                    r = unit(Wq[sl] @ qn[j])
                    t = unit(Wt[sl] @ Pn[at + i])
                    sc[k_, c] = DD * float((a * r * t).sum())
            mx = sc[ok].max(0) if ok.any() else np.zeros(C)
            out.append(list(sc.reshape(-1)) + list(mx) + [float(rk[i] < K)])
        at += n
    return np.asarray(out)


def selftest():
    t0 = time.time()
    LG.bind_device_ops()
    # 1. the arm: zrc's model plus the offset head over zrc's builds; zrc's arm unchanged; train takes no other arm
    assert S.ARMS[ARM] == (ZOff, ZC.ChainCarveZRC) and S.ARMS[BASE_ARM] == (ZM.ZRM, ZC.ChainCarveZRC)
    assert ZOff.__mro__[1] is ZM.ZRM
    SR.must_stop(train, ["train", "--arm", BASE_ARM, "--name", "x"], "L-musique")
    SR.must_stop(train, ["train", "--arm", "zbr", "--name", "x"], "L-musique")
    assert zrc_fit("J5") == ("full_zrct", "fits", "J5") and zrc_fit("L-hotpotqa") == ("screen", "fits", "scr-zrct-hp")
    tmp = Path(tempfile.mkdtemp(prefix="zof_"))
    try:
        rng = np.random.default_rng(32)
        RM.toy_roots(tmp, rng, chunks=(12, 12))
        RM.build("metaqa", "toy", out_root=tmp / "ch", cache_root=tmp / "cache", look_root=tmp / "look",
                 rel_dir=tmp / "rel")
        cls = RM.chain_carve(ZM.ToyZ, tmp / "ch", tmp / "rel")
        c = cls("metaqa", "toy", "2wiki", "cpu", tmp / "cache")
        blocks = ["rank", "SEMB"]
        SR.must_stop(ZOff, ["rank"], c.widths, 16)
        qs = np.r_[np.arange(c.rows)[::-1][:5], np.arange(c.rows)[:4]].astype(np.int64)
        B = qs.size
        feats, nq, bz, _g = c.batch(qs, blocks)
        cnt = c.n_np[qs]
        assert int(cnt.max()) > K and int(cnt.min()) >= 1
        # 2. from the same seed zof is zrc's state plus the head, the last layer zero, and its forward is zrc's bit for
        #    bit, in eval and with the same dropout draws in training; the global generator is where zrc leaves it
        keep = torch.ones(B, len(blocks))
        nets, after = {}, {}
        for k, cl in ((BASE_ARM, ZM.ZRM), (ARM, ZOff)):
            torch.manual_seed(0)
            nets[k] = cl(blocks, c.widths, 16, dropout=0.1, seed=0)
            after[k] = torch.rand(4)
        assert torch.equal(after[ARM], after[BASE_ARM])
        sz, sk = nets[BASE_ARM].state_dict(), nets[ARM].state_dict()
        assert sorted(set(sk) - set(sz)) == sorted(OF) and all(torch.equal(sz[k], sk[k]) for k in sz)
        assert float(sk["of_w2"].abs().sum()) == 0.0 and float(sk["of_w1"].abs().sum()) > 0
        with torch.no_grad():
            for mode in (False, True):
                out = {}
                for k, m in nets.items():
                    m.train(mode)
                    torch.manual_seed(5)
                    out[k] = m(feats, keep, nq, B, bz)
                assert torch.equal(out[ARM], out[BASE_ARM]), mode
            # 3. the inputs are offset_inputs by hand; they read no score (only rrf's base_z ranks the anchors); set,
            #    the head adds to zrc's score
            nz, nk = nets[BASE_ARM].eval(), nets[ARM].eval()
            s = nz(feats, keep, nq, B, bz)
            qe, P = feats["SEMB"]
            rank, A = anchors(bz, nq, B)
            x = offset_inputs(nk, qe, P, rank, A, nq)
            xh = hand_inputs(nk, qe, P, bz, c.n_np, qs)
            assert x.shape == (int(cnt.sum()), N_IN) and np.allclose(x.numpy(), xh, atol=1e-4), \
                np.abs(x.numpy() - xh).max()
            assert float(x[:, :K * C].abs().sum()) > 0 and int(x[:, -1].sum()) == int(np.minimum(cnt, K).sum())
            gen = torch.Generator().manual_seed(3)
            for k in ("of_w2", "of_b2"):
                getattr(nk, k).copy_(torch.randn(getattr(nk, k).shape, generator=gen))
            hd = Fn.gelu(x @ nk.of_w1.T + nk.of_b1) @ nk.of_w2.T + nk.of_b2
            assert torch.allclose(nk(feats, keep, nq, B, bz), s + hd.squeeze(-1), atol=1e-6)
        # 4. toy fits through lean_gpu's loop: the head and the offsets move, the fit leaves zrc's, a repeat is identical
        cfg = {"lr": 2e-2, "wd": 1e-4, "dropout": 0.1, "epochs": 3, "swa_from": 1, "cos": False, "adamw": False,
               "drop": False}
        every = lambda f: f["states"] + [f["swa"]]  # noqa: E731
        saved_arms = dict(S.ARMS)
        S.ARMS.update({ARM: (ZOff, cls), BASE_ARM: (ZM.ZRM, cls)})
        try:
            fits = {}
            for arm in (BASE_ARM, ARM):
                with S.patched(arm):
                    fits[arm] = LG.fit_variant([c], blocks, cfg, 0, 16, "none", "cpu", tag=f"toy-{arm}")
            with S.patched(ARM):
                rep = LG.fit_variant([c], blocks, cfg, 0, 16, "none", "cpu", tag="toy-zof repeat")
        finally:
            S.ARMS.clear()
            S.ARMS.update(saved_arms)
        sw = fits[ARM]["swa"]
        assert float(sw["of_w2"].abs().sum()) > 0
        assert not torch.equal(sw["of_wa"], sk["of_wa"]) and not torch.equal(sw["of_wq"], sk["of_wq"])
        assert any(not torch.equal(sw[k], fits[BASE_ARM]["swa"][k]) for k in fits[BASE_ARM]["swa"])
        assert all(not LG.same_state(a_, b_) for a_, b_ in zip(every(fits[ARM]), every(rep)))
        # 5. read refuses another arm's fit
        (tmp / "fits" / "x").mkdir(parents=True)
        LC.write_json(tmp / "fits" / "x" / "screen.json", {"arm": BASE_ARM})
        SR.must_stop(read, ["--name", "x", "--out-root", str(tmp / "fits")])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    # 6. relz's pair and re-call under zof's name, decided against zrc's fits, named for this round; relz restored
    saved = Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit
    with on_zrc():
        assert Z.tag({})["arm"] == ARM and Z.BASE_ARM == BASE_ARM
    assert (Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit) == saved
    with tempfile.TemporaryDirectory() as td:
        T = Path(td)
        null1 = SR.fake_null(T / "null", "n1", {("L-hotpotqa", "hotpotqa"): (-0.010, 0.012)})
        zs = {sp: str(SR.fake_compare(T / zrc_fit(sp)[0], zrc_fit(sp)[2], sp, {ds: 0.1 for ds in LG.EVAL_ORDER},
                                      arm=BASE_ARM))
              for sp in ZRC_FITS}
        zb = {sp: "/".join(("outputs",) + zrc_fit(sp)) for sp in ZRC_FITS}
        za = SR.fake_compare(T / "z", "scr-zof", "L-musique", {"webqsp": 0.0215}, arm=ARM, base_r5=0.6,
                             base=zb["L-musique"])
        zh = SR.fake_compare(T / "z", "scr-zof-hp", "L-hotpotqa", {"hotpotqa": -0.0147}, arm=ARM, base_r5=0.6,
                             base=zb["L-hotpotqa"])
        zm = SR.fake_compare(T / "z2", "scr-zof", "L-musique", {}, arm=ARM, base_r5=0.6,
                             base="outputs/screen/fits/scr-zrm")
        SR.must_stop(pair, [zm, zh], T / "bad")
        pr = pair([za, zh], T / "pz")
        assert pr["verdict"] == "MIXED" and pr["arm"] == ARM
        pj = json.loads((T / "pz.json").read_text(encoding="utf-8"))
        assert pj["round"] == "thirty-second" and pj["decided_against"] == DECIDED
        r = recall(null1, T / "pz.json", T / "rz", zs)
        assert r["verdict"] == "PROMISING" and r["changed"] == ["L-hotpotqa hotpotqa LOSS -> WITHIN"], r["changed"]
        t = (T / "rz.md").read_text(encoding="utf-8")
        assert "thirty-second round" in t and "| zrc R@5 | zof R@5 |" in t
    assert (Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit) == saved
    n_par = sum(int(sk[n].numel()) for n in OF)
    log(f"zof selftest: the arm is zrc's model plus the offset head ({n_par} parameters on the toy) over zrc's builds, "
        f"and train takes no other; from the same seed zof is zrc's state plus the head and zrc's forward bit for bit, "
        f"the global generator untouched; the inputs match a hand computation and read no score; set, the head adds "
        f"to zrc's score; a toy fit moves it, its repeat identical; relz's pair and re-call run against zrc's fits and "
        f"name this round ({time.time() - t0:.1f}s): ok")
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
        return train([k.cmd] + rest, k.split)
    if k.cmd == "read":
        return read(rest)
    if k.cmd == "compare":
        return compare(rest)
    if k.cmd in ("pair", "recall"):
        gp = argparse.ArgumentParser()
        gp.add_argument("--screens", default="")
        gp.add_argument("--null", default="")
        gp.add_argument("--pair")
        gp.add_argument("--out")
        g = gp.parse_args(rest)
        null = [x for x in g.null.split(",") if x]
        need = {"pair": g.screens and g.out, "recall": g.pair and len(null) == 4 and g.out}[k.cmd]
        if not need:
            gp.error(f"{k.cmd}: pair needs --screens A,B and --out; recall --pair, the four --null files and --out")
        try:
            if k.cmd == "pair":
                pair([x for x in g.screens.split(",") if x], g.out)
            else:
                recall(null, g.pair, g.out)
        except SystemExit as e:
            log(f"zof {k.cmd}: {e}")
            return 2
        return 0
    raise SystemExit("zof: train, read, compare, pair, recall, or --selftest")


if __name__ == "__main__":
    sys.exit(main())
