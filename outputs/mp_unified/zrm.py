"""Screens, fourteenth round (docs/SCREENS.md): rmatch's question-relation match on zret's base. zrm is zret's model
(lean_screen3.ZRet: every block's within-pool z-score, and rrf's base z-score, taken against the pool's retrieved rows)
with rmatch's learned match of the question to each hop of the typed relation chains that reach a row from the
question's seeds added to every row's score (rmatch.ChainMatch, on rmatch's chain carve and its chains as built for
the twelfth round). Both parts are used unchanged: the class is ChainMatch over ZRet, so ChainMatch's forward adds its
gated match to ZRet's scores. No new column, block or hyperparameter. The match's gates start at zero, so at its start
the arm is zret's model bit for bit, from the same initialisation and batches; on a graph without typed relations
(squad, musique, hotpotqa, 2wiki) it is zret's model. Two fits (L-musique and L-hotpotqa), each compared with zret's fit
of its split (that comparison decides) and with rmatch's and step 1's (reported); the pair, its re-call under the seed
null and the full run's grade are relz.py's, run under zrm's name with zret's fits in place of rel's (zgs.py's
mapping of zret's fits).

    python outputs/mp_unified/zrm.py smoke --device cuda --host
    python outputs/mp_unified/zrm.py train --split L-musique --name scr-zrm --arm zrm --device cuda --host
    python outputs/mp_unified/zrm.py read --name scr-zrm --device cuda --host
    python outputs/mp_unified/zrm.py compare --new outputs/screen/fits/scr-zrm \\
        --base outputs/screen/fits/scr-zret,outputs/screen/fits/scr-rmatch,outputs/step1/fits/L-musique \\
        --out outputs/screen/scr-zrm
    python outputs/mp_unified/zrm.py pair --screens outputs/screen/scr-zrm.json,outputs/screen/scr-zrm-hp.json \\
        --out outputs/screen/scr-zrm-pair
    python outputs/mp_unified/zrm.py recall --null N1,N2,N3,N4 --pair outputs/screen/scr-zrm-pair.json \\
        --out outputs/screen/scr-zrm-pair-recall
    python outputs/mp_unified/zrm.py grade --full-root outputs/full_zrm \\
        --reuse L-musique=outputs/screen/scr-zrm.json,L-hotpotqa=outputs/screen/scr-zrm-hp.json
    python outputs/mp_unified/zrm.py --selftest
N1..N4 are the seed null's comparisons (outputs/screen/scr-null-s1, -s2, -s1-hp, -s2-hp .json). A read's floor is the
null's; its base R@5 is zret's fit's (outputs/screen/scr-zret.json and outputs/full_zret/compare-L-hotpotqa.json, both
compared with the null's seed 0). The full run's re-grade is nullx.py's regrade with zret's six comparisons as
--base-compares.

Speed (8216ffe, docs/FULL_ROUND12.md section 6): the chains are a query-local compile, so any latency figure for this
arm is cold: each question timed from scratch, the typed walk from its pool's arrays, the move of its entries to the
device and the forward, with no warm-up pass and nothing kept from an earlier question.
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

import lean_screen3 as S3  # noqa: E402
import rmatch as RM  # noqa: E402
import zgs as G  # noqa: E402

Z, R = RM.Z, RM.R
S2, S = RM.S2, RM.S
LG, LC = S.LG, S.LC
log = S.log
ARM, BASE_ARM = "zrm", "zret"


class ZRM(RM.ChainMatch, S3.ZRet):
    """zrm: rmatch's gated chain match (zero gates at the start) added to zret's forward."""


S.ARMS.update({ARM: (ZRM, RM.ChainCarveBase)})


# ── train and read (rmatch.py's, under zrm's arm) ────────────────────────────


def train(argv, split):
    """rmatch.py's train (lean_screen2's, with rmatch's settings and chain records) for zrm only."""
    arm = R.arm_of(argv)
    if arm != ARM:
        raise SystemExit(f"zrm: train takes --arm {ARM}, not {arm}")
    rc = RM.train(argv, split)
    name, out_root = S2.where(argv)
    sj = out_root / name / "screen.json"
    if sj.exists():
        rec = json.loads(sj.read_text(encoding="utf-8"))
        rec["zrm_sha256"] = LC.sha_src(__file__)
        rec["lean_screen3_sha256"] = LC.sha_src(S3.__file__)
        rec["zrm"] = {"model": "rmatch.ChainMatch over lean_screen3.ZRet", "carve": "rmatch.ChainCarveBase"}
        LC.write_json(sj, rec)
    return rc


def read(argv):
    name, out_root = S2.where(argv)
    arm = json.loads((out_root / name / "screen.json").read_text(encoding="utf-8"))["arm"]
    if arm != ARM:
        raise SystemExit(f"zrm: {name} was trained as {arm}, not {ARM}")
    return RM.read(argv)


# ── relz.py's pair, re-call and grade, decided against zret's fits ───────────


@contextlib.contextmanager
def on_zret():
    """relz.py's records under zrm's name, each read decided against zret's fit of its split (zgs.py's mapping);
    relz restored after."""
    with G.on_zret():
        Z.ARM = ARM
        yield


FIX = (("(rel's screen fits scr-rel and scr-rel-hp)", "(zret's fits: its screen fit scr-zret and its full run's "
                                                      "L-hotpotqa fit)"),
       ("(rel's screen fits)", "(zret's fits)"),
       ("| rel R@5 | relz R@5 |", "| zret R@5 | zrm R@5 |"),
       ("section 2 and the tenth round", "section 2 and the fourteenth round"),
       ("docs/FULL_ROUND10.md", "docs/FULL_ROUND14.md"))


def restamp(out):
    """A record relz.py wrote under zrm's name: zret named as the base in its md, this file's sha added."""
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
        rec["zrm_sha256"] = LC.sha_src(__file__)
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


def carve_check(ds, carve, device):
    """One carve through rmatch's chain carve: on its first questions zrm at its start scores as zret's model bit for
    bit (the same seed), with finite chain features on a typed graph. Whether zret's forward differs from lean_gpu's
    there is recorded (equal only when every row of those pools is retrieved)."""
    # 8 Oct fix (07:20): the forwards run under train's and read's flags (lean_gpu.set_flags: deterministic algorithms,
    # TF32 off). zret's z-scores against the retrieved rows sum with index_add_, which on the card is not deterministic
    # without them: two forwards of one model differed, and the first smoke (07:06) failed its identity on every
    # carve, 2wiki's untyped one included. zret's own repeat is now recorded and required.
    LG.set_flags(device)
    LG.bind_device_ops()
    b = RM.ChainCarveBase(ds, carve, "2wiki", device)
    rec = {"typed": b.chains is not None}
    ok = rec["typed"] == (ds in RM.TYPED)
    blocks = [x for x in LG.SETS["pick"] if x in b.widths]
    widths = {x: b.widths[x] for x in blocks}
    qs = np.arange(min(16, b.rows))
    feats, nq, base_z, _gold = b.batch(qs, blocks)
    keep = torch.ones((qs.size, len(blocks)), dtype=torch.float32, device=b.device)
    nets = {}
    for k, cls in (("base", S.ARMS["base"][0]), (BASE_ARM, S3.ZRet), (ARM, ZRM)):
        torch.manual_seed(0)
        nets[k] = cls(blocks, widths, 32, seed=0).to(b.device).eval()
    with torch.no_grad():
        s = {k: m(feats, keep, nq, qs.size, base_z) for k, m in nets.items()}
        rec["zret_repeat_equal"] = bool(torch.equal(nets[BASE_ARM](feats, keep, nq, qs.size, base_z), s[BASE_ARM]))
        rec["identity_at_start"] = bool(torch.equal(s[ARM], s[BASE_ARM]))
        rec["zret_differs_from_lean_gpu"] = bool(not torch.equal(s[BASE_ARM], s["base"]))
        rec["retrieved_share_first_questions"] = round(float(S3.retrieved(feats).mean()), 4)
        if b.chains is not None:
            fm, fr = nets[ARM].chain_feats(feats, nq.numel())
            rec["entries_first_questions"] = int(feats[RM.CH_KEY]["eq"].size)
            rec["features_finite"] = bool(torch.isfinite(fm).all() and torch.isfinite(fr).all())
            ok = ok and rec["features_finite"] and rec["entries_first_questions"] > 0
    ok = ok and rec["zret_repeat_equal"] and rec["identity_at_start"]
    del b, nets
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return ok, rec


def smoke(device, host, out_root=None):
    """The carve check on metaqa select, webqsp s1eval and 2wiki select; then zret once, rmatch once and zrm twice (the
    repeat must be IDENTICAL), one epoch each on metaqa's and 2wiki's select carves (metaqa is typed, so the match
    trains; 2wiki's pools are partly unretrieved, so zret's forward is not lean_gpu's there), each read on both. zrm's
    and rmatch's matches must move from zero and stay finite, zret's fit holds none, and zrm's scores must differ from
    zret's on metaqa (the same init and batches: the difference is the match) and from rmatch's on 2wiki (the same init,
    batches and match: the difference is zret's forward). A crash fails it (exit 1)."""
    t0 = time.time()
    root = Path(out_root or S.OUT / "smoke14")
    h = ["--host"] if host else []
    rec, ok = {"carves": {}}, True
    for ds, cv in RM.SMOKE_CARVES:
        good, chk = carve_check(ds, cv, device)
        rec["carves"][f"{ds}={cv}"] = chk
        ok = ok and good
    sets = "metaqa=select,2wiki=select"
    common = ["--train", sets, "--basis", "2wiki", "--variants", "p", "--config", "2e-3:1e-4:0.1:1:0", "--device",
              device, "--out-root", str(root)] + h
    runs = {BASE_ARM: (S3.train, "1"), RM.ARM: (RM.train, "1"), ARM: (train, "2")}
    for arm, (fn, rep) in runs.items():
        nm = f"smoke-{arm}"
        rc = fn(["train", "--name", nm, "--arm", arm, "--repeat", rep] + common, "L-musique")
        rr = RM.read(["--name", nm, "--read", sets, "--device", device, "--out-root", str(root)] + h)
        sj = json.loads((root / nm / "screen.json").read_text(encoding="utf-8"))
        tj = json.loads((root / nm / "train.json").read_text(encoding="utf-8"))
        rec[arm] = {"train_rc": rc, "repeats": int(rep), "read_rc": rr, "screen_arm": sj.get("arm"),
                    "blocks": tj["variants"]["p"]["blocks"], "head_norms": RM.head_norms(root / nm / "models.pt"),
                    "peak_gpu_gb": tj.get("peak_gpu_gb")}
        ok = ok and rc == 0 and rr == 0 and sj.get("arm") == arm
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
    rec[ARM]["repeat"] = "IDENTICAL" if rec[ARM]["train_rc"] == 0 else "DIFFERENT"
    rec["match_moved"] = {a: RM.moved(rec[a]["head_norms"]) for a in (RM.ARM, ARM)}
    rec["zret_has_no_match"] = all(v is None for v in rec[BASE_ARM]["head_norms"].values())
    rec["same_blocks"] = rec[ARM]["blocks"] == rec[BASE_ARM]["blocks"] == rec[RM.ARM]["blocks"]
    ok = ok and all(rec["match_moved"].values()) and rec["zret_has_no_match"] and rec["same_blocks"]
    rec["reads"] = {}
    for ds in ("metaqa", "2wiki"):
        npz = {a: root / f"smoke-{a}" / "reads" / f"{ds}__select.npz" for a in runs}
        if not all(p.exists() for p in npz.values()):
            rec["reads"][ds] = None
            continue
        r = {a: np.load(p) for a, p in npz.items()}
        ids = {a: [str(x) for x in v["ids"]] for a, v in r.items()}
        k = [str(x) for x in r[ARM]["candidates"]].index("p@ep0")
        rec["reads"][ds] = {"same_questions": ids[ARM] == ids[BASE_ARM] == ids[RM.ARM],
                            "differ_from_zret": bool(not np.array_equal(r[ARM]["scores64"], r[BASE_ARM]["scores64"])),
                            "differ_from_rmatch": bool(not np.array_equal(r[ARM]["scores64"], r[RM.ARM]["scores64"])),
                            "p@ep0_hit@1": {a: float(v["hit"][k].mean()) for a, v in r.items()}}
    rd = rec["reads"]
    ok = ok and all(v is not None and v["same_questions"] for v in rd.values())
    ok = ok and rd["metaqa"]["differ_from_zret"] and rd["2wiki"]["differ_from_rmatch"]
    rec.update({"ok": ok, "seconds": time.time() - t0, "script_sha256": LC.sha_src(__file__)})
    LC.write_json(root / "smoke.json", rec)
    log(f"smoke14: {json.dumps(rec)}; {'ok' if ok else 'FAILED'}")
    return 0 if ok else 1


# ── selftest ─────────────────────────────────────────────────────────────────


class ToyZ(RM.ToyBase):
    """rmatch's toy carve with about 40% of its rows unretrieved (rrf 0), so zret's z-scores are not lean_gpu's."""

    def __init__(self, ds, carve, basis, device, cache_root=LC.OUT, verify=True, score2=False):
        super().__init__(ds, carve, basis, device, cache_root, verify, score2)
        g = torch.Generator().manual_seed(7)
        self.X[torch.rand(self.X.shape[0], generator=g) < 0.4, S3.I_RRF] = 0.0


def selftest():
    t0 = time.time()
    LG.bind_device_ops()
    base_cls, base_carve = S.ARMS["base"]
    # 1. the arm is rmatch's match over zret's model on rmatch's chain carve; zret's and rmatch's own arms unchanged
    assert S.ARMS[ARM] == (ZRM, RM.ChainCarveBase) and ZRM.__mro__[1:4] == (RM.ChainMatch, S3.ZRet, base_cls)
    assert S.ARMS[BASE_ARM] == (S3.ZRet, base_carve) and S.ARMS[RM.ARM] == (RM.ChainMatch, RM.ChainCarveBase)
    tmp = Path(tempfile.mkdtemp(prefix="zrm_"))
    try:
        # 2. on rmatch's toy carve, some rows unretrieved: at the start zrm is zret's model bit for bit (and zret is not
        #    lean_gpu's); with every parameter set it adds the gated match as rmatch computes it by hand, the same
        #    match rmatch adds to lean_gpu's model; a question whose SEMB is dropped takes none; a batch without
        #    entries (an untyped graph) is zret's model
        rng = np.random.default_rng(14)
        RM.toy_roots(tmp, rng)
        RM.build("metaqa", "toy", out_root=tmp / "ch", cache_root=tmp / "cache", look_root=tmp / "look",
                 rel_dir=tmp / "rel")
        c = RM.chain_carve(ToyZ, tmp / "ch", tmp / "rel")("metaqa", "toy", "2wiki", "cpu", tmp / "cache")
        ch = c.chains
        has = np.flatnonzero(np.asarray(ch["q_ent"]) > 0)
        assert has.size >= 2, has
        qs = np.r_[has[::-1][:3], np.setdiff1d(np.arange(c.rows), has)[:1]].astype(np.int64)
        blocks = ["rank", "SEMB"]
        feats, nq, bz, gold = c.batch(qs, blocks)
        B = qs.size
        keep = torch.ones(B, 2)
        ref = S3.retrieved(feats)
        assert 0 < float(ref.sum()) < ref.numel()
        nets = {}
        for k, cls in (("base", base_cls), (BASE_ARM, S3.ZRet), (RM.ARM, RM.ChainMatch), (ARM, ZRM)):
            torch.manual_seed(0)
            nets[k] = cls(blocks, c.widths, 16, dropout=0.1, seed=0).eval()
        sd = nets[ARM].state_dict()
        assert set(sd) == set(nets[RM.ARM].state_dict())
        assert all(torch.equal(v, sd[k]) for k, v in nets[RM.ARM].state_dict().items())
        assert all(torch.equal(v, sd[k]) for k, v in nets[BASE_ARM].state_dict().items())
        with torch.no_grad():
            s = {k: m(feats, keep, nq, B, bz) for k, m in nets.items()}
            assert torch.equal(s[ARM], s[BASE_ARM]) and not torch.equal(s[BASE_ARM], s["base"])
            for p in nets[ARM].out.parameters():
                p.normal_()
            gen = torch.Generator().manual_seed(3)
            for k, p in nets[ARM].named_parameters():
                if k.startswith("cm_"):
                    p.copy_(torch.randn(p.shape, generator=gen) * (0.02 if k in ("cm_b", "cm_pi_w") else 0.5))
            sd = nets[ARM].state_dict()
            nets[RM.ARM].load_state_dict(sd)
            shared = {k: v for k, v in sd.items() if not k.startswith("cm_")}
            nets[BASE_ARM].load_state_dict(shared)
            nets["base"].load_state_dict(shared)
            s = {k: m(feats, keep, nq, B, bz) for k, m in nets.items()}
            got = (s[ARM] - s[BASE_ARM]).double().numpy()
            want = RM.hand_feats(nets[ARM], ch, c, qs, np.ones(B))
            assert np.allclose(got, want, atol=2e-4, rtol=1e-4), np.abs(got - want).max()
            assert np.abs(want[nq.numpy() == 0]).max() > 0 and (want[nq.numpy() == B - 1] == 0).all()
            assert torch.allclose(s[ARM] - s[BASE_ARM], s[RM.ARM] - s["base"], atol=1e-5)
            assert not torch.equal(s[ARM], s[RM.ARM])
            k2 = keep.clone()
            k2[0, blocks.index("SEMB")] = 0.0
            d2 = (nets[ARM](feats, k2, nq, B, bz) - nets[BASE_ARM](feats, k2, nq, B, bz)).double().numpy()
            assert (d2[nq.numpy() == 0] == 0).all() and np.allclose(d2[nq.numpy() == 1], got[nq.numpy() == 1])
            plain = {k: v for k, v in feats.items() if k != RM.CH_KEY}
            assert torch.equal(nets[ARM](plain, keep, nq, B, bz), nets[BASE_ARM](plain, keep, nq, B, bz))
        # 3. from the start, with the same dropout draws, zrm's loss and every shared gradient are zret's bit for bit;
        #    of the match only the gates take a gradient
        fresh = {}
        for k, cls in ((BASE_ARM, S3.ZRet), (ARM, ZRM)):
            torch.manual_seed(0)
            fresh[k] = cls(blocks, c.widths, 16, dropout=0.1, seed=0)
        loss = {}
        for k, m in fresh.items():
            torch.manual_seed(5)
            loss[k] = LG.listwiseD(m(feats, keep, nq, B, bz), gold, nq, B)
            loss[k].backward()
        assert torch.equal(loss[ARM], loss[BASE_ARM])
        gz = dict(fresh[BASE_ARM].named_parameters())
        for k, p in fresh[ARM].named_parameters():
            if k == "cm_gate":
                assert float(p.grad.abs().sum()) > 0
            elif k.startswith("cm_"):
                assert p.grad is None or float(p.grad.abs().sum()) == 0, k
            else:
                assert (p.grad is None and gz[k].grad is None) or torch.equal(p.grad, gz[k].grad), k
        # 4. load_models builds zrm under patching, its scores unchanged; the patching restores
        blob = {"candidates": ["p@ep0"], "variants": {"p": {"blocks": blocks, "widths": c.widths, "ctx": "none"}},
                "hidden": 16, "states": {"p@ep0": nets[ARM].state_dict()}}
        with S.patched(ARM):
            assert LG.LeanMLP8D is ZRM and LG.CacheCarve is RM.ChainCarveBase
            (_name, m, bl), = LG.load_models(blob, "cpu")
            assert type(m) is ZRM and bl == blocks
            with torch.no_grad():
                assert torch.equal(m(feats, keep, nq, B, bz), s[ARM])
        assert LG.LeanMLP8D is base_cls and LG.CacheCarve is base_carve
        # 5. refusals: no SEMB, a narrow rank block, another context, training another arm, reading another arm's fit
        for bad in (lambda: ZRM(["rank"], {"rank": 5}, 16), lambda: ZRM(blocks, dict(c.widths, rank=4), 16),
                    lambda: ZRM(blocks, c.widths, 16, ctx="film"),
                    lambda: train(["train", "--name", "x", "--arm", RM.ARM], "L-musique")):
            try:
                bad()
                raise AssertionError("a refusal was accepted")
            except SystemExit:
                pass
        (tmp / "fits" / "x").mkdir(parents=True)
        LC.write_json(tmp / "fits" / "x" / "screen.json", {"arm": RM.ARM})
        try:
            read(["--name", "x", "--out-root", str(tmp / "fits")])
            raise AssertionError("zrm read rmatch's fit")
        except SystemExit as e:
            assert "trained as rmatch" in str(e)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    # 6. relz's pair, re-call and grade under zrm's name, decided against zret's fits; relz restored after (an error
    #    too); restamp names zret and this round
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
              for sp in G.ZRET_FITS}                                         # zret's fits against step 1's (0.5)
        zb = {sp: "/".join(("outputs",) + G.zret_fit(sp)) for sp in G.ZRET_FITS}
        za = SR.fake_compare(T / "z", "scr-zrm", "L-musique", {"metaqa": 0.1268}, arm=ARM, base_r5=0.6,
                             base=zb["L-musique"])
        zh = SR.fake_compare(T / "z", "scr-zrm-hp", "L-hotpotqa", {"hotpotqa": -0.0147}, arm=ARM, base_r5=0.6,
                             base=zb["L-hotpotqa"])
        pr = pair([za, zh], T / "pz")
        assert pr["verdict"] == "MIXED" and pr["arm"] == ARM and pr["base_arm"] == BASE_ARM
        assert "scr-zret" in (T / "pz.md").read_text(encoding="utf-8")
        assert json.loads((T / "pz.json").read_text(encoding="utf-8"))["decided_against"] == "zret's fit of each split"
        rc = recall(null1, T / "pz.json", T / "rz", zs)
        assert rc["verdict"] == "PROMISING" and rc["changed"] == ["L-hotpotqa hotpotqa LOSS -> WITHIN"], rc["changed"]
        assert all(r["base"] == 0.6 for r in rc["rows"])
        t = (T / "rz.md").read_text(encoding="utf-8")
        assert "| zret R@5 | zrm R@5 |" in t and "fourteenth round" in t and "rel's" not in t
        st1 = SR.fake_compare(T / "z2", "scr-zrm", "L-musique", {}, arm=ARM)       # decided against step 1's fit
        SR.must_stop(pair, [st1, zh], T / "bad")
        full = T / "full"
        full.mkdir()
        for sp in S2.SPLITS:
            if sp not in G.ZRET_FITS:
                d = {"webqsp": -0.02} if sp == "J5" else {}
                SR.fake_compare(full / "src", f"fit-{sp}", sp, d, arm=ARM,
                                base="/".join(("outputs",) + G.zret_fit(sp))).replace(full / f"compare-{sp}.json")
        g = grade(full, {"L-musique": str(za), "L-hotpotqa": str(zh)}, full / "grade", check_fits=False)
        assert g["verdict"] == "NOT_ADOPTED" and g["losses"] == 2 and g["arm"] == ARM, (g["verdict"], g["losses"])
        assert "docs/FULL_ROUND14.md" in (full / "grade.md").read_text(encoding="utf-8")
    assert (Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit) == saved
    log(f"zrm selftest: the arm is rmatch's match over zret's model on rmatch's chain carve; at the start it is zret's "
        f"model bit for bit (zret not lean_gpu's on the toy), with every parameter set it adds rmatch's hand-computed "
        f"gated match, the same match rmatch adds to lean_gpu's model, none for a dropped SEMB, none on an untyped "
        f"graph; from the start its loss and shared gradients are zret's bit for bit and only the gates move; "
        f"load_models under patching; refusals; relz's pair, re-call and grade run under zrm's name against zret's fits "
        f"(a pair against step 1's is refused), name zret and this round, and relz is restored "
        f"({time.time() - t0:.1f}s): ok")
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
            log(f"zrm {k.cmd}: {e}")
            return 2
        return 0
    raise SystemExit("zrm: train, read, compare, pair, recall, grade, smoke, or --selftest")


if __name__ == "__main__":
    sys.exit(main())
