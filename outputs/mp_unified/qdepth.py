"""Screens, eleventh round (docs/SCREENS.md): one idea, a learned depth prior per question, screened on one base.
The base is rel's if rel's full run is ADOPTED under the seed null over every split (outputs/full_rel/grade-nullx),
step 1's if it is NOT_ADOPTED; `base` is the gate that says which (exit 0 for the base that runs).

    python outputs/mp_unified/qdepth.py smoke --device cuda --host
    python outputs/mp_unified/qdepth.py base --rel-grade outputs/full_rel/grade-nullx.json --want step1
    python outputs/mp_unified/qdepth.py train --name scr-qdepth --arm qdepth --device cuda --host
    python outputs/mp_unified/qdepth.py train --split L-hotpotqa --name scr-relqd-hp --arm relqd --device cuda --host
    python outputs/mp_unified/qdepth.py read --name scr-qdepth --device cuda --host
    python outputs/mp_unified/qdepth.py compare --new outputs/screen/fits/scr-qdepth \\
        --base outputs/step1/fits/L-musique --out outputs/screen/scr-qdepth
    python outputs/mp_unified/qdepth.py pair-rel --screens outputs/screen/scr-relqd.json,outputs/screen/scr-relqd-hp.json \\
        --out outputs/screen/scr-relqd-pair
    python outputs/mp_unified/qdepth.py recall-rel --null N1,N2,N3,N4 --pair outputs/screen/scr-relqd-pair.json \\
        --out outputs/screen/scr-relqd-pair-recall
    python outputs/mp_unified/qdepth.py grade-rel --full-root outputs/full_relqd --reuse L-musique=A,L-hotpotqa=B
    python outputs/mp_unified/qdepth.py --selftest
On step 1's base the pair and its re-call are screen_pair.py's and screen_recall.py's, unchanged. On rel's they are
relz.py's (each read decided against rel's screen fit of its split), run under relqd's name.

The arms (each changes every training dataset and every read alike):
  qdepth   lean_gpu's model plus a learned depth prior for each question, added to every row's score:
               s(v) = lean_gpu's s(v) + l_q[depth(v)],   l_q = q U + P_q W + b   (5 values per question)
           depth(v) is hopdiag.py's class of the row from WALK's raw columns: 0 a seed, 1 to 3 the first structural
           hop from a seed (the nearest, should a row carry two), 4 unreached. q is the question's embedding (SEMB's
           input), P_q the question's pool profile: the share of its rows in each depth class and log(1 + its size).
           U (1536 x 5), W (6 x 5) and b (5) start at zero, so the arm starts as its base model, bit for bit, from the
           same initialisation and batches; the listwise loss alone trains them, with the model's Adam, learning rate
           and weight decay. No new column, block or hyperparameter, and nothing reads a label at read time. A
           question whose WALK or SEMB block is dropped takes no prior.
  relqd    the same prior on rel's carve and blocks (relcols.py: step 1's nine, then typed_rel, typed_v2, ordered).
  Why (docs/SCREENS.md, 'Hop depth of the top-1 errors'): a perfect question-to-depth attention lifts the MLP's
  metaqa 3-hop hit@1 by 0.07 to 0.08, and doubles its webqsp zero-shot hit@1, where it ranks a seed first in 0.60 to
  0.95 of the questions (0.07 of the golds are seeds). The question says how many hops it asks for (the KB systems'
  hop attention); the pool's profile says what kind of graph it is asked on, which the question cannot.
"""
import os
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import contextlib  # noqa: E402
import json  # noqa: E402
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

import relcols as R  # noqa: E402
import relz as Z  # noqa: E402

S2, S = R.S2, R.S
LG, LC, LM = S.LG, S.LC, S.LM
log = S.log
ARM, REL_ARM, REL = "qdepth", "relqd", "rel"
DEPTHS = 5
WALK_W, FIRST_HOP, IS_SEED = 16, (12, 13, 14), 15        # lean_mlp's WALK: first-hop one-hot (hops 1-3), is-seed
Q_DIM = 1536
PROF_W = DEPTHS + 1


# ── the depth prior ──────────────────────────────────────────────────────────


def depth_of(walk):
    """(N,) int64 depth class from WALK's raw columns: 0 is-seed, 1-3 the first hop (the nearest), 4 unreached."""
    if walk.dim() != 2 or walk.shape[1] != WALK_W:
        raise SystemExit(f"WALK is {tuple(walk.shape)}, lean_mlp's WALK is {WALK_W} wide")
    d = torch.full((walk.shape[0],), DEPTHS - 1, dtype=torch.long, device=walk.device)
    for k in (2, 1, 0):
        d = torch.where(walk[:, FIRST_HOP[k]] > 0.5, torch.full_like(d, k + 1), d)
    return torch.where(walk[:, IS_SEED] > 0.5, torch.zeros_like(d), d)


def profile(d, nq, B):
    """(B, 6) float32 per pool: the share of its rows in each depth class, and log(1 + its size)."""
    dev = d.device
    ones = torch.ones(d.numel(), dtype=torch.float32, device=dev)
    cnt = torch.zeros(B, dtype=torch.float32, device=dev).index_add_(0, nq, ones)
    oh = Fn.one_hot(d, DEPTHS).to(torch.float32)
    share = torch.zeros(B, DEPTHS, dtype=torch.float32, device=dev).index_add_(0, nq, oh)
    return torch.cat([share / cnt.clamp_min(1.0).unsqueeze(1), torch.log1p(cnt).unsqueeze(1)], 1)


class QDepth(LG.LeanMLP8D):
    """qdepth and relqd: lean_gpu's forward plus the question's learned depth prior at each row's depth class."""

    def __init__(self, blocks, widths, hidden=128, dropout=0.1, seed=0, arm="ctl", ctx="none"):
        if ctx != "none":
            raise SystemExit("qdepth takes ctx none only")
        if "WALK" not in blocks or "SEMB" not in blocks:
            raise SystemExit("qdepth needs the WALK and SEMB blocks")
        if int(widths["WALK"]) != WALK_W:
            raise SystemExit(f"qdepth: WALK has {widths['WALK']} columns, lean_mlp's WALK {WALK_W}")
        super().__init__(blocks, widths, hidden, dropout, seed, arm, ctx)
        self.j_walk, self.j_semb = self.blocks.index("WALK"), self.blocks.index("SEMB")
        self.qd_u = nn.Parameter(torch.zeros(Q_DIM, DEPTHS))
        self.qd_p = nn.Parameter(torch.zeros(PROF_W, DEPTHS))
        self.qd_b = nn.Parameter(torch.zeros(DEPTHS))

    def depth_logits(self, feats, nq, B):
        """(depth class (N,), the questions' priors (B, 5))."""
        d = depth_of(feats["WALK"])
        qe = feats["SEMB"][0]
        return d, qe @ self.qd_u + profile(d, nq, B) @ self.qd_p + self.qd_b

    def forward(self, feats, keep, nq, B, base_z):
        s = super().forward(feats, keep, nq, B, base_z)
        d, lq = self.depth_logits(feats, nq, B)
        m = keep[nq, self.j_walk] * keep[nq, self.j_semb]
        return s + (lq[nq] * Fn.one_hot(d, DEPTHS).to(lq.dtype)).sum(1) * m


S.ARMS.update({ARM: (QDepth, S.ARMS["base"][1]), REL_ARM: (QDepth, R.RelCarveRel)})


# ── train and read (lean_screen2's; relqd under rel's block sets) ────────────


def sets_of(arm):
    """rel's block sets for relqd (relcols.with_sets), lean_gpu's own otherwise."""
    return R.with_sets(REL) if arm == REL_ARM else contextlib.nullcontext()


def train(argv, split):
    arm = R.arm_of(argv)
    with sets_of(arm):
        rc = S2.train(argv, split)
    name, out_root = S2.where(argv)
    sj = out_root / name / "screen.json"
    if sj.exists():
        rec = json.loads(sj.read_text(encoding="utf-8"))
        rec["qdepth_sha256"] = LC.sha_src(__file__)
        rec["qdepth"] = {"depths": DEPTHS, "walk_columns": {"first_hop": list(FIRST_HOP), "is_seed": IS_SEED},
                         "profile": "depth-class shares and log(1 + pool size)", "init": "zero"}
        if arm == REL_ARM:
            rec["relcols_sha256"] = LC.sha_src(R.__file__)
            rec["relcols_blocks"] = list(R.ARM_BLOCKS[REL])
            rec["relcols_records"] = R.built_records()
        LC.write_json(sj, rec)
    return rc


def read(argv):
    name, out_root = S2.where(argv)
    arm = json.loads((out_root / name / "screen.json").read_text(encoding="utf-8"))["arm"]
    with sets_of(arm):
        return S2.main(["read"] + argv)


# ── the base gate, and relz.py's rel-based records under relqd's name ────────


def base_gate(rel_grade, want):
    """Exit 0 when rel's re-grade names this base: want 'rel' passes on ADOPT, want 'step1' on NOT_ADOPTED. A missing
    or INCOMPLETE re-grade passes neither (exit 1), and neither screen runs."""
    if want not in ("rel", "step1"):
        raise SystemExit(f"qdepth base: --want rel or step1, not {want}")
    p = Path(rel_grade)
    v = json.loads(p.read_text(encoding="utf-8")).get("verdict") if p.exists() else None
    ok = (want == "rel" and v == "ADOPT") or (want == "step1" and v == "NOT_ADOPTED")
    log(f"qdepth base gate: {p} {v}; the {want} base {'runs' if ok else 'does not run'}")
    return 0 if ok else 1


@contextlib.contextmanager
def as_relqd():
    """relz.py's pair, re-call and grade against rel's fits, under relqd's name; restored on the way out."""
    saved = Z.ARM
    Z.ARM = REL_ARM
    try:
        yield
    finally:
        Z.ARM = saved


FIX = (("the tenth round", "the eleventh round"), ("docs/FULL_ROUND10.md", "docs/FULL_ROUND11.md"))


def restamp(out):
    """A record relz.py wrote under relqd's name: this round's documents named in its md, this file's sha added."""
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
        rec["qdepth_sha256"] = LC.sha_src(__file__)
        LC.write_json(js, rec)


# ── smoke ────────────────────────────────────────────────────────────────────


def head_norms(models_pt):
    """{candidate: (|U|, |W|, |b|)} of a fit's saved states (None where a state holds no depth prior)."""
    blob = torch.load(models_pt, map_location="cpu", weights_only=False)
    out = {}
    for name, st in blob["states"].items():
        out[name] = [float(st[k].norm()) for k in ("qd_u", "qd_p", "qd_b")] if "qd_u" in st else None
    return out


def moved(norms):
    return bool(norms) and all(v is not None and all(np.isfinite(v)) and v[0] > 0 and v[2] > 0 for v in norms.values())


def smoke(device, host, out_root=None):
    """Step 1's base arm once, qdepth twice (the repeat must be IDENTICAL) and relqd once, one epoch each on metaqa's
    select carve, each then read on it. qdepth's and relqd's states must hold a prior that moved from zero and is
    finite; qdepth's scores must differ from the base arm's (the same init and batches, so a difference is the prior);
    relqd's fit must hold rel's blocks live. A crash fails it (exit 1)."""
    t0 = time.time()
    root = Path(out_root or S.OUT / "smoke11")
    h = ["--host"] if host else []
    common = ["--train", "metaqa=select", "--basis", "2wiki", "--variants", "p", "--config", "2e-3:1e-4:0.1:1:0",
              "--device", device, "--out-root", str(root)] + h
    rec, ok = {}, True
    for arm, rep in (("base", "1"), (ARM, "2"), (REL_ARM, "1")):
        nm = f"smoke-{arm}"
        rc = main(["train", "--name", nm, "--arm", arm, "--repeat", rep] + common)
        rr = main(["read", "--name", nm, "--read", "metaqa=select", "--device", device, "--out-root", str(root)] + h)
        sj = json.loads((root / nm / "screen.json").read_text(encoding="utf-8"))
        tj = json.loads((root / nm / "train.json").read_text(encoding="utf-8"))
        blocks = tj["variants"]["p"]["blocks"]
        rec[arm] = {"train_rc": rc, "repeats": int(rep), "read_rc": rr, "screen_arm": sj.get("arm"), "blocks": blocks,
                    "head_norms": head_norms(root / nm / "models.pt")}
        ok = ok and rc == 0 and rr == 0 and sj.get("arm") == arm
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
    rec[ARM]["repeat"] = "IDENTICAL" if rec[ARM]["train_rc"] == 0 else "DIFFERENT"
    rec["prior_moved"] = {a: moved(rec[a]["head_norms"]) for a in (ARM, REL_ARM)}
    rec["base_has_no_prior"] = all(v is None for v in rec["base"]["head_norms"].values())
    rec["relqd_rel_blocks_live"] = all(b in rec[REL_ARM]["blocks"] for b in R.ARM_BLOCKS[REL])
    rec["qdepth_blocks_are_base"] = rec[ARM]["blocks"] == rec["base"]["blocks"]
    ok = (ok and all(rec["prior_moved"].values()) and rec["base_has_no_prior"] and rec["relqd_rel_blocks_live"]
          and rec["qdepth_blocks_are_base"])
    fb = root / "smoke-base" / "reads" / "metaqa__select.npz"
    fq = root / f"smoke-{ARM}" / "reads" / "metaqa__select.npz"
    fr = root / f"smoke-{REL_ARM}" / "reads" / "metaqa__select.npz"
    if fb.exists() and fq.exists() and fr.exists():
        a, b, c = np.load(fb), np.load(fq), np.load(fr)
        rec["same_questions"] = [str(x) for x in a["ids"]] == [str(x) for x in b["ids"]] == [str(x) for x in c["ids"]]
        rec["scores_differ_from_base"] = bool(not np.array_equal(a["scores64"], b["scores64"]))
        k = [str(x) for x in a["candidates"]].index("p@ep0")
        rec["p@ep0_hit@1"] = {"base": float(a["hit"][k].mean()), ARM: float(b["hit"][k].mean()),
                              REL_ARM: float(c["hit"][k].mean())}
        ok = ok and rec["same_questions"] and rec["scores_differ_from_base"]
    else:
        rec["scores_differ_from_base"] = None
        ok = False
    rec.update({"ok": ok, "seconds": time.time() - t0, "script_sha256": LC.sha_src(__file__)})
    LC.write_json(root / "smoke.json", rec)
    log(f"smoke11: {json.dumps(rec)}; {'ok' if ok else 'FAILED'}")
    return 0 if ok else 1


# ── selftest ─────────────────────────────────────────────────────────────────


def toy(seed=0):
    """Three pools (5, 4 and 3 rows) with every depth class, a seed that is also a first hop and a row with two."""
    g = torch.Generator().manual_seed(seed)
    n = torch.tensor([5, 4, 3])
    B, N = n.numel(), int(n.sum())
    nq = torch.repeat_interleave(torch.arange(B), n)
    walk = torch.zeros(N, WALK_W)
    walk[:, :12] = torch.rand(N, 12, generator=g) * 3
    hop = [0, 1, 2, 3, 4, 1, 2, 0, 4, 3, 1, 4]        # 0 seed, 1-3 first hop, 4 unreached
    for i, k in enumerate(hop):
        if k == 0:
            walk[i, IS_SEED] = 1.0
        elif k < 4:
            walk[i, FIRST_HOP[k - 1]] = 1.0
    walk[7, FIRST_HOP[1]] = 1.0                       # a seed with a first hop: still a seed
    walk[9, FIRST_HOP[0]] = 1.0                       # first hops 1 and 3: the nearest
    hop[9] = 1
    widths = {"rank": 5, "SEMB": LM.SEMB_DIM, "WALK": WALK_W}
    feats = {"rank": torch.rand(N, 5, generator=g) * 0.03 + 0.001,
             "SEMB": (Fn.normalize(torch.randn(B, Q_DIM, generator=g), dim=1),
                      torch.randn(N, LG.L3.STORE_DIM, generator=g)),
             "WALK": walk}
    return feats, widths, nq, B, N, hop


def selftest():
    LG.bind_device_ops()
    feats, widths, nq, B, N, hop = toy()
    blocks = ["rank", "SEMB", "WALK"]
    # 1. the depth classes and the profile against hand computations
    d = depth_of(feats["WALK"])
    assert d.tolist() == hop, (d.tolist(), hop)
    P = profile(d, nq, B).numpy()
    for q in range(B):
        rows = np.flatnonzero(nq.numpy() == q)
        want = [np.mean([hop[i] == k for i in rows]) for k in range(DEPTHS)] + [np.log1p(rows.size)]
        assert np.allclose(P[q], want, atol=1e-6), (q, P[q], want)
    # 2. at zero initialisation the arm is the base model bit for bit (same seed, same draws)
    torch.manual_seed(0)
    base = LG.LeanMLP8D(blocks, widths, 16, dropout=0.1, seed=0).eval()
    torch.manual_seed(0)
    qd = QDepth(blocks, widths, 16, dropout=0.1, seed=0).eval()
    shared = {k: v for k, v in qd.state_dict().items() if not k.startswith("qd_")}
    assert all(torch.equal(v, base.state_dict()[k]) for k, v in shared.items()) and len(shared) == len(base.state_dict())
    keep = torch.ones(B, len(blocks))
    bz = LG.seg_zscore8D(feats["rank"][:, 1:2], nq, B).squeeze(1)
    with torch.no_grad():
        for p in qd.out.parameters():
            p.normal_()
        base.load_state_dict({k: v for k, v in qd.state_dict().items() if not k.startswith("qd_")})
        assert torch.equal(qd(feats, keep, nq, B, bz), base(feats, keep, nq, B, bz))
        # 3. with a prior: the base score plus l_q at each row's depth, by hand; a dropped WALK or SEMB takes none
        for p in (qd.qd_u, qd.qd_p, qd.qd_b):
            p.normal_()
        lq = (feats["SEMB"][0] @ qd.qd_u + profile(d, nq, B) @ qd.qd_p + qd.qd_b).numpy()
        prior = np.array([lq[int(nq[i]), hop[i]] for i in range(N)])
        assert np.allclose(qd(feats, keep, nq, B, bz).numpy(), base(feats, keep, nq, B, bz).numpy() + prior, atol=1e-5)
        q1 = nq.numpy() == 1
        for j in (blocks.index("WALK"), blocks.index("SEMB")):
            k2 = keep.clone()
            k2[1, j] = 0.0
            got, ref = qd(feats, k2, nq, B, bz).numpy(), base(feats, k2, nq, B, bz).numpy()
            assert np.array_equal(got[q1], ref[q1]) and np.allclose(got[~q1], ref[~q1] + prior[~q1], atol=1e-5)
    # 4. from zero, the listwise loss moves the prior toward the golds' depth (and nothing is non-finite)
    torch.manual_seed(0)
    qd0 = QDepth(blocks, widths, 16, dropout=0.1, seed=0)
    gold = torch.tensor([h == 2 for h in hop])
    LG.listwiseD(qd0(feats, keep, nq, B, bz), gold, nq, B).backward()
    assert all(torch.isfinite(p.grad).all() for p in qd0.parameters() if p.grad is not None)
    assert float(qd0.qd_u.grad.abs().sum()) > 0 and float(qd0.qd_b.grad[2]) < 0
    assert abs(float(qd0.qd_b.grad.sum())) < 1e-6
    # 5. refusals
    for kw, bl, wd in (({"ctx": "film"}, blocks, widths), ({}, ["rank", "WALK"], widths), ({}, ["rank", "SEMB"], widths),
                       ({}, blocks, {**widths, "WALK": 15})):
        try:
            QDepth(bl, wd, 16, **kw)
            raise AssertionError(f"qdepth took {kw} {bl} {wd.get('WALK')}")
        except SystemExit:
            pass
    try:
        depth_of(torch.zeros(3, 15))
        raise AssertionError("depth_of took a 15-wide WALK")
    except SystemExit:
        pass
    # 6. load_models builds the arm's class under patching, its scores unchanged; the arms and patching restore
    blob = {"candidates": ["p@ep0"], "variants": {"p": {"blocks": blocks, "widths": widths, "ctx": "none"}},
            "hidden": 16, "states": {"p@ep0": qd.state_dict()}}
    assert S.ARMS[ARM] == (QDepth, S.ARMS["base"][1]) and S.ARMS[REL_ARM] == (QDepth, R.RelCarveRel)
    assert S.ARMS[REL] == (S.ARMS["base"][0], R.RelCarveRel) and S.ARMS["relz"][1] is R.RelCarveRel
    for arm in (ARM, REL_ARM):
        with S.patched(arm):
            assert LG.LeanMLP8D is QDepth and LG.CacheCarve is S.ARMS[arm][1]
            (name, m, bl), = LG.load_models(blob, "cpu")
            assert isinstance(m, QDepth) and bl == blocks
            with torch.no_grad():
                assert torch.equal(m(feats, keep, nq, B, bz), qd(feats, keep, nq, B, bz))
    assert LG.LeanMLP8D is S.ARMS["base"][0] and LG.CacheCarve is S.ARMS["base"][1]
    # 7. relqd trains and reads under rel's block sets, qdepth under lean_gpu's own; both restored after
    saved = {k: list(v) for k, v in LG.SETS.items()}
    with sets_of(REL_ARM):
        assert all(LG.SETS[k] == saved[k] + [b for b in R.ARM_BLOCKS[REL] if b not in saved[k]] for k in saved)
        assert all(b in LG.SETS["pick"] for b in ("WALK", "SEMB"))
    assert {k: list(v) for k, v in LG.SETS.items()} == saved
    with sets_of(ARM):
        assert {k: list(v) for k, v in LG.SETS.items()} == saved
    # 8. the base gate: ADOPT runs rel's base, NOT_ADOPTED step 1's; INCOMPLETE or a missing re-grade runs neither
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "g.json"
        for v, want in (("ADOPT", {"rel": 0, "step1": 1}), ("NOT_ADOPTED", {"rel": 1, "step1": 0}),
                        ("INCOMPLETE", {"rel": 1, "step1": 1}), (None, {"rel": 1, "step1": 1})):
            if v is None:
                f.unlink()
            else:
                f.write_text(json.dumps({"verdict": v}), encoding="utf-8")
            for w, rc in want.items():
                assert base_gate(f, w) == rc, (v, w)
        try:
            base_gate(f, "rel2")
            raise AssertionError("the base gate took want rel2")
        except SystemExit:
            pass
        # 9. relz.py's records under relqd's name, restored after (an error too); restamp names this round
        assert Z.ARM == "relz"
        with as_relqd():
            assert Z.ARM == REL_ARM and Z.tag({})["arm"] == REL_ARM
        assert Z.ARM == "relz"
        try:
            with as_relqd():
                raise KeyError("x")
        except KeyError:
            pass
        assert Z.ARM == "relz"
        o = Path(td) / "r"
        o.with_suffix(".md").write_text("the tenth round; docs/FULL_ROUND10.md\n", encoding="utf-8")
        LC.write_json(o.with_suffix(".json"), {"arm": REL_ARM})
        restamp(o)
        assert o.with_suffix(".md").read_text(encoding="utf-8") == "the eleventh round; docs/FULL_ROUND11.md\n"
        assert json.loads(o.with_suffix(".json").read_text(encoding="utf-8"))["qdepth_sha256"] == LC.sha_src(__file__)
    print("selftest: depth classes (a seed with a first hop is a seed; two first hops take the nearest) and pool "
          "profiles match hand computations; at zero the arm is the base model bit for bit; with a prior it adds l_q "
          "at each row's depth, none for a question whose WALK or SEMB is dropped; from zero the loss moves the prior "
          "toward the golds' depth with finite gradients; refusals; load_models under patching for both arms; relqd "
          "under rel's block sets; the base gate; relz's records under relqd's name, restored; restamp. "
          "all checks passed")
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
    if k.cmd == "base":
        gp = argparse.ArgumentParser()
        gp.add_argument("--rel-grade", required=True)
        gp.add_argument("--want", required=True, choices=("rel", "step1"))
        g = gp.parse_args(rest)
        return base_gate(g.rel_grade, g.want)
    if k.cmd == "train":
        return train([k.cmd] + rest, k.split)
    if k.cmd == "read":
        return read(rest)
    if k.cmd == "compare":
        return S2.main([k.cmd] + rest)
    if k.cmd in ("pair-rel", "recall-rel", "grade-rel"):
        gp = argparse.ArgumentParser()
        gp.add_argument("--screens", default="")
        gp.add_argument("--null", default="")
        gp.add_argument("--pair")
        gp.add_argument("--full-root")
        gp.add_argument("--reuse", default="")
        gp.add_argument("--out")
        g = gp.parse_args(rest)
        null = [x for x in g.null.split(",") if x]
        need = {"pair-rel": g.screens and g.out, "recall-rel": g.pair and len(null) == 4 and g.out,
                "grade-rel": g.full_root}[k.cmd]
        if not need:
            gp.error(f"{k.cmd}: pair-rel needs --screens A,B and --out; recall-rel --pair, the four --null files and "
                     "--out; grade-rel --full-root")
        out = g.out or str(Path(g.full_root) / "grade")
        try:
            with as_relqd():
                if k.cmd == "pair-rel":
                    Z.pair([x for x in g.screens.split(",") if x], out)
                elif k.cmd == "recall-rel":
                    Z.recall(null, g.pair, out)
                else:
                    reuse = dict(x.split("=", 1) for x in g.reuse.split(",") if x)
                    v = Z.grade(g.full_root, reuse, out)["verdict"]
                    restamp(out)
                    return 1 if v == "INCOMPLETE" else 0
        except SystemExit as e:
            log(f"qdepth {k.cmd}: {e}")
            return 2
        restamp(out)
        return 0
    raise SystemExit("qdepth: train, read, compare, base, pair-rel, recall-rel, grade-rel, smoke, or --selftest")


if __name__ == "__main__":
    sys.exit(main())
