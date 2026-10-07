"""Screens, second round (docs/SCREENS.md), and the full run of a PROMISING screen (docs/FULL_BDROP20.md).

    python outputs/mp_unified/lean_screen2.py smoke --device cuda --host
    python outputs/mp_unified/lean_screen2.py train --name scr-pad --arm pad --device cuda --host
    python outputs/mp_unified/lean_screen2.py read --name scr-pad --device cuda --host
    python outputs/mp_unified/lean_screen2.py compare --new outputs/screen/fits/scr-pad \\
        --base outputs/step1/fits/L-musique --out outputs/screen/scr-pad
    python outputs/mp_unified/lean_screen2.py train --split J5 --name J5 --arm bdrop20 \\
        --out-root outputs/full_bdrop20/fits --device cuda --host
    python outputs/mp_unified/lean_screen2.py grade --full-root outputs/full_bdrop20 \\
        --reuse L-musique=outputs/screen/scr-bdrop20.json
    python outputs/mp_unified/lean_screen2.py --selftest

lean_screen.py's arms, commands and rule, unchanged. This file adds two arms, a --split option (lean_screen's train on
another of step 1's splits: that split's fit carves and basis) and the full run's grade.

Arms (each changes every training dataset alike):
  pad      pool padding in training. With probability 0.5 a training question's pool gets copies of its own unranked
           non-gold rows (rrf 0) until retrieval's ranked rows are a share t of it, t uniform in [0.08, 0.20] (the
           big pools' measured ranked shares: metaqa 0.10, webqsp 0.15, musique 0.18), at most 4,096 rows. A pool
           with no unranked non-gold row (squad's) or no ranked row is never padded. Each training carve draws from
           its own generator, reset at the start of each fit, so the loop's order and a repeat are the base arm's.
           The copies enter the pool's z-scores, rrf's base z-score and the listwise softmax as rows of their own.
           Reads are never padded.
  padbd20  pad and lean_screen's bdrop20 together.

The full run of an arm: its six splits' fits (variant p, seed 0, the SWA state), each read on the six s1eval carves
and compared with step 1's fit of the same split by lean_screen's rule; grade turns the six comparisons into the
verdict docs/FULL_BDROP20.md declares.
"""
import os
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
import zlib  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import lean_screen as S  # noqa: E402

LG, LC, LM = S.LG, S.LC, S.LM
ROOT = S.ROOT
FULL = ROOT / "outputs" / "full_bdrop20"
SPLITS = tuple(LG.FITS)
log = S.log


# ── the pad arm ──────────────────────────────────────────────────────────────


class PadCarve(LG.CacheCarve):
    """pad: lean_gpu's carve. A carve made while PadCarve.TRAIN is set (lean_screen2's train) pads its pools in
    batch(); any other carve (every read) is lean_gpu's."""

    TRAIN = False
    P_APPLY, SHARE, CAP, SEED = 0.5, (0.08, 0.20), 4096, 20261007
    made = []

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.padding = PadCarve.TRAIN
        if self.padding:
            self.pad_index()
            PadCarve.made.append(self)

    def pad_index(self):
        """Each question's unranked non-gold rows (CSR over the carve's rows) and its count of ranked rows."""
        t = time.time()
        rrf = self.X[:, self.c_rrf].to(torch.float32)
        ranked = (torch.isfinite(rrf) & (rrf > 0)).cpu().numpy()
        cand = ~ranked & ~self.gold.cpu().numpy()
        self.pad_rows = np.flatnonzero(cand).astype(np.int64)
        self.pad_off = np.searchsorted(self.pad_rows, self.off_np).astype(np.int64)
        cs = np.concatenate([[0], np.cumsum(ranked, dtype=np.int64)])
        self.ranked_n = cs[self.off_np[1:]] - cs[self.off_np[:-1]]
        self.pad_reset()
        log(f"  {self.ds}/{self.carve}: pad index, {self.pad_rows.size} unranked non-gold rows; "
            f"{int(((np.diff(self.pad_off) > 0) & (self.ranked_n > 0)).sum())} of {self.n_np.size} questions can pad "
            f"({time.time() - t:.0f}s)")

    def pad_reset(self):
        """The carve's generator from its seed, and its counts from 0 (at the start of each fit)."""
        self.pad_rng = np.random.default_rng([self.SEED, zlib.crc32(f"{self.ds}/{self.carve}".encode())])
        self.pad_stats = {"batches": 0, "questions": 0, "padded": 0, "rows_in": 0, "rows_added": 0}

    def pad_plan(self, qs):
        """A padded batch's row indices and question positions; each question's own rows first, in carve order."""
        parts, nqs = [], []
        st = self.pad_stats
        for j, q in enumerate(qs.tolist()):
            a, e = int(self.off_np[q]), int(self.off_np[q + 1])
            rows = np.arange(a, e, dtype=np.int64)
            c0, c1, r = int(self.pad_off[q]), int(self.pad_off[q + 1]), int(self.ranked_n[q])
            if self.pad_rng.random() < self.P_APPLY and c1 > c0 and r > 0:
                t = self.pad_rng.uniform(self.SHARE[0], self.SHARE[1])
                total = min(self.CAP, int(math.ceil(r / t)))
                if total > e - a:
                    add = self.pad_rows[c0 + self.pad_rng.integers(0, c1 - c0, total - (e - a))]
                    rows = np.concatenate([rows, add])
                    st["padded"] += 1
                    st["rows_added"] += int(add.size)
            st["questions"] += 1
            st["rows_in"] += e - a
            parts.append(rows)
            nqs.append(np.full(rows.size, j, np.int64))
        st["batches"] += 1
        return np.concatenate(parts), np.concatenate(nqs)

    def batch(self, qs, blocks):
        """lean_gpu.CacheCarve.batch, on pad_plan's rows when the carve pads."""
        if not self.padding:
            return super().batch(qs, blocks)
        qs = np.asarray(qs, np.int64)
        B = qs.size
        idx, nq_np = self.pad_plan(qs)
        dev = self.device
        idx_t = torch.from_numpy(idx).to(dev)
        nq = torch.from_numpy(nq_np).to(dev)
        Xf = self.X[idx_t].to(torch.float32)
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


S.ARMS.update({"pad": (S.ARMS["base"][0], PadCarve), "padbd20": (S.BDrop, PadCarve)})


class training:
    """PadCarve.TRAIN set, and every padding carve's generator reset at the start of each fit (LG.fit_variant)."""

    def __enter__(self):
        self.saved = LG.fit_variant
        orig = self.saved

        def fit_variant(tr, *args, **kw):
            for c in tr:
                if getattr(c, "padding", False):
                    c.pad_reset()
            return orig(tr, *args, **kw)

        LG.fit_variant = fit_variant
        PadCarve.TRAIN = True
        PadCarve.made = []
        return self

    def __exit__(self, *exc):
        LG.fit_variant = self.saved
        PadCarve.TRAIN = False
        return False


# ── train, read, compare ─────────────────────────────────────────────────────


def where(argv):
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--name")
    ap.add_argument("--out-root")
    k, _ = ap.parse_known_args(argv)
    return k.name, Path(k.out_root) if k.out_root else S.OUT / "fits"


def train(argv, split):
    """lean_screen's train on split's fit carves and basis; a padding arm's counts added to its screen.json."""
    saved = S.SPLIT
    S.SPLIT = split
    try:
        with training():
            rc = S.main(argv)
    finally:
        S.SPLIT = saved
    name, out_root = where(argv)
    sj = out_root / name / "screen.json"
    if sj.exists():
        rec = json.loads(sj.read_text(encoding="utf-8"))
        rec["lean_screen2_sha256"] = LC.sha_src(__file__)
        if PadCarve.made:
            rec["pad"] = {"p_apply": PadCarve.P_APPLY, "share": list(PadCarve.SHARE), "cap": PadCarve.CAP,
                          "seed": PadCarve.SEED, "last_fit": [{"dataset": c.ds, "carve": c.carve, **c.pad_stats}
                                                              for c in PadCarve.made]}
            for c in PadCarve.made:
                st = c.pad_stats
                log(f"  pad {c.ds}/{c.carve}: {st['padded']} of {st['questions']} questions padded, rows "
                    f"{st['rows_in']} + {st['rows_added']} (last fit)")
        LC.write_json(sj, rec)
    PadCarve.made = []
    return rc


# ── the full run's grade ─────────────────────────────────────────────────────


def parts_of(p):
    return tuple(x for x in str(p).replace("\\", "/").split("/") if x)


def primary(split, ds):
    """Step 1's eleven primary reads: J5 on every dataset, each leave-out fit on its held-out dataset."""
    return split == "J5" or split == f"L-{ds}"


def grade(full_root, reuse, arm, out=None, check_fits=True):
    """The six comparisons into one verdict (docs/FULL_BDROP20.md): ADOPT when a primary read GAINs and no read of
    the thirty-six LOSEs, NOT_ADOPTED otherwise; INCOMPLETE (exit 1) when a comparison is missing or not the
    declared one."""
    full_root = Path(full_root)
    rows, problems, sources = [], [], {}
    for split in SPLITS:
        fn = Path(reuse.get(split) or full_root / f"compare-{split}.json")
        sources[split] = str(fn)
        if not fn.exists():
            problems.append(f"{split}: {fn} missing")
            continue
        rec = json.loads(fn.read_text(encoding="utf-8"))
        if parts_of(rec["bases"][0])[-3:] != ("step1", "fits", split):
            problems.append(f"{split}: decided against {rec['bases'][0]}, not step 1's {split}")
        if rec["candidate"] != "p@swa" or rec["carve"] != "s1eval":
            problems.append(f"{split}: {rec['candidate']} on {rec['carve']}, not p@swa on s1eval")
        if sorted(rec["trained_on"]) != sorted(LG.FITS[split]):
            problems.append(f"{split}: trained on {rec['trained_on']}, not {sorted(LG.FITS[split])}")
        if check_fits:
            sj = ROOT / Path(*parts_of(rec["new"])) / "screen.json"
            if not sj.exists():
                problems.append(f"{split}: {sj} missing")
            else:
                s = json.loads(sj.read_text(encoding="utf-8"))
                if s.get("arm") != arm or s.get("split") != split:
                    problems.append(f"{split}: the fit is arm {s.get('arm')} on {s.get('split')}, not {arm} on {split}")
        for r in rec["by_base"][rec["bases"][0]]["rows"]:
            rows.append({"split": split, "dataset": r["dataset"], "read": r["read"],
                         "primary": primary(split, r["dataset"]), "base": r["base"][0], "new": r["new"][0],
                         "rrf": r["rrf"][0], "delta": r["delta"][0], "lo": r["lo"][0], "hi": r["hi"][0],
                         "delta_fc5": r["delta"][1], "delta_hit1": r["delta"][2], "call": r["call"]})
    n_prim = sum(r["primary"] for r in rows)
    if len(rows) != len(SPLITS) * len(LG.EVAL_ORDER) or n_prim != 11:
        problems.append(f"{len(rows)} reads ({n_prim} primary), not 36 (11)")
    gains = [r for r in rows if r["primary"] and r["call"] == "GAIN"]
    losses = [r for r in rows if r["call"] == "LOSS"]
    v = "INCOMPLETE" if problems else "ADOPT" if gains and not losses else "NOT_ADOPTED"
    rec = {"arm": arm, "verdict": v, "problems": problems, "sources": sources, "primary_gains": len(gains),
           "losses": len(losses), "rows": rows, "floor": S.FLOOR, "script_sha256": LC.sha_src(__file__),
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    md = [f"# Full run of {arm}: **{v}**", "",
          f"R@5 of {arm}'s p@swa minus step 1's p@swa of the same split, on the six s1eval carves (95% question-bootstrap "
          f"interval; call by docs/SCREENS.md's rule). Bold: step 1's eleven primary reads. zs: read zero-shot.", "",
          "| split | " + " | ".join(LG.EVAL_ORDER) + " |", "|---|" + "---|" * len(LG.EVAL_ORDER)]
    for split in SPLITS:
        cells = []
        for ds in LG.EVAL_ORDER:
            r = next((x for x in rows if x["split"] == split and x["dataset"] == ds), None)
            if r is None:
                cells.append("—")
                continue
            c = f"{r['delta']:+.4f} {r['call']}{' zs' if r['read'] == 'zero-shot' else ''}"
            cells.append(f"**{c}**" if r["primary"] else c)
        md.append(f"| {split} | " + " | ".join(cells) + " |")
    md += ["", f"Primary reads with a GAIN: {len(gains)} of {n_prim}. Reads with a LOSS: {len(losses)} of {len(rows)}.",
           "ADOPT when at least one primary read GAINs and none of the thirty-six LOSEs (docs/FULL_BDROP20.md)."]
    if problems:
        md += ["", "Problems:"] + [f"- {p}" for p in problems]
    if out:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        LC.write_json(out.with_suffix(".json"), rec)
        out.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    log(f"grade {arm}: {v}; {len(gains)} primary GAIN, {len(losses)} LOSS" + (f"; problems {problems}" if problems else ""))
    return rec


# ── smoke ────────────────────────────────────────────────────────────────────


def plan_of(split):
    return [[d, LG.FIT_CARVE[d]] for d in LG.FITS[split]], LG.basis_of(LG.FITS[split])


def smoke(device, host, out_root=None):
    """pad and padbd20 on 2wiki's select carve (1 epoch, run twice: a padding repeat must be IDENTICAL), each read;
    and every split's fit carves and basis by lean_screen's train rule against step 1's train.json. A crash, a
    padding arm that pads nothing, a differing repeat or a differing plan fails it (exit 1)."""
    t0 = time.time()
    root = Path(out_root or S.OUT / "smoke2")
    h = ["--host"] if host else []
    rec, ok = {"arms": {}, "plan": {}}, True
    for arm in ("pad", "padbd20"):
        rc = main(["train", "--name", f"smoke-{arm}", "--arm", arm, "--train", "2wiki=select", "--basis", "2wiki",
                   "--variants", "p", "--config", "2e-3:1e-4:0.1:1:0", "--repeat", "2", "--device", device,
                   "--out-root", str(root)] + h)
        sj = json.loads((root / f"smoke-{arm}" / "screen.json").read_text(encoding="utf-8"))
        padded = sum(c["padded"] for c in sj.get("pad", {}).get("last_fit", []))
        rr = main(["read", "--name", f"smoke-{arm}", "--read", "2wiki=select", "--device", device, "--out-root",
                   str(root)] + h)
        rec["arms"][arm] = {"train_rc": rc, "repeat": "IDENTICAL" if rc == 0 else "DIFFERENT", "read_rc": rr,
                            "padded_questions": padded, "pad": sj.get("pad")}
        ok = ok and rc == 0 and rr == 0 and padded > 0
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
    for split in SPLITS:
        mine = plan_of(split)
        tj = ROOT / "outputs" / "step1" / "fits" / split / "train.json"
        r = json.loads(tj.read_text(encoding="utf-8"))
        theirs = [[t["dataset"], t["carve"]] for t in r["train"]], r["basis"]
        same = [list(x) for x in mine[0]] == theirs[0] and mine[1] == theirs[1]
        rec["plan"][split] = {"carves": mine[0], "basis": mine[1], "step1": {"carves": theirs[0], "basis": theirs[1]},
                              "same": same}
        ok = ok and same
    rec["ok"], rec["seconds"], rec["script_sha256"] = ok, time.time() - t0, LC.sha_src(__file__)
    LC.write_json(root / "smoke.json", rec)
    log(f"smoke2: {json.dumps({a: {k: v for k, v in r.items() if k != 'pad'} for a, r in rec['arms'].items()})}; "
        f"plans {'same as step 1' if all(p['same'] for p in rec['plan'].values()) else 'DIFFER'}; "
        f"{'ok' if ok else 'FAILED'} ({rec['seconds']:.0f}s)")
    return 0 if ok else 1


# ── selftest ─────────────────────────────────────────────────────────────────


def fake_carve(ds="dsx", carve="cv"):
    """A four-question PadCarve on the CPU: q0 3 ranked + 3 unranked rows; q1 all ranked; q2 1 ranked + 4 unranked
    (one gold); q3 none ranked. Column 0 holds the row id."""
    c = object.__new__(PadCarve)
    c.ds, c.carve, c.device = ds, carve, torch.device("cpu")
    c.n_np = np.array([6, 4, 5, 3], np.int64)
    c.off_np = np.concatenate([[0], np.cumsum(c.n_np)])
    X = torch.zeros(18, 6)
    X[:, 0] = torch.arange(18, dtype=torch.float32)
    X[:, 2] = torch.tensor([.03, .02, .01, 0, 0, 0, .03, .02, .02, .01, .02, 0, 0, 0, 0, 0, 0, 0])
    X[:, 5] = torch.linspace(-1.0, 1.0, 18)
    c.X = X.half()
    c.gold = torch.tensor([0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0], dtype=torch.bool)
    c.span, c.c_rrf, c.padding = {"rank": (0, 5), "WALK": (5, 6)}, 2, True
    c.pad_index()
    return c


def selftest():
    LG.bind_device_ops()
    blocks = ["rank", "WALK"]
    c = fake_carve()
    assert c.pad_rows.tolist() == [3, 4, 5, 11, 12, 14, 15, 17] and c.ranked_n.tolist() == [3, 4, 1, 0]
    assert c.pad_off.tolist() == [0, 3, 3, 6, 8]
    qs = np.array([0, 1, 2, 3])
    c.padding = False
    ref = LG.CacheCarve.batch(c, qs, blocks)
    got = c.batch(qs, blocks)
    assert all(torch.equal(ref[0][b], got[0][b]) for b in blocks) and all(torch.equal(x, y) for x, y in zip(ref[1:], got[1:]))
    c.padding, c.P_APPLY = True, 1.0
    for _ in range(50):
        feats, nq, base_z, gold = c.batch(qs, blocks)
        rid = feats["rank"][:, 0].round().long()
        for j, q in enumerate(qs.tolist()):
            r = rid[nq == j].tolist()
            a, e = int(c.off_np[q]), int(c.off_np[q + 1])
            assert r[:e - a] == list(range(a, e))                                     # its own rows first
            assert set(r[e - a:]) <= set(c.pad_rows[c.pad_off[q]:c.pad_off[q + 1]].tolist())   # copies: unranked non-gold
            assert int(gold[nq == j].sum()) == int(c.gold[a:e].sum())                 # the golds stay
            if q in (1, 3):
                assert len(r) == e - a                                                # nothing to pad with / no ranked row
            if q == 0:
                assert 15 <= len(r) <= 38 and 0.08 - 1e-9 <= 3 / len(r) <= 0.2 + 1e-9  # ranked share in [0.08, 0.20]
        assert torch.equal(base_z, LG.seg_zscore8D(feats["rank"][:, 2:3], nq, 4).squeeze(1))
    c.CAP = 10
    feats, nq, _, _ = c.batch(np.array([0]), blocks)
    assert int((nq == 0).sum()) == 10
    c.CAP = PadCarve.CAP
    a1, a2 = fake_carve(), fake_carve()
    for _ in range(5):
        x1, x2 = a1.batch(qs, blocks), a2.batch(qs, blocks)
        assert torch.equal(x1[0]["rank"], x2[0]["rank"]) and torch.equal(x1[1], x2[1])     # seeded per carve
    fresh = fake_carve()
    start = [fresh.batch(qs, blocks)[0]["rank"] for _ in range(3)]
    a1.pad_reset()
    again = [a1.batch(qs, blocks)[0]["rank"] for _ in range(3)]
    assert all(torch.equal(x, y) for x, y in zip(again, start))                     # a reset restarts the stream
    d = fake_carve()
    n = 4000
    for _ in range(n):
        d.batch(np.array([0]), blocks)
    assert abs(d.pad_stats["padded"] / n - 0.5) < 0.03 and d.pad_stats["questions"] == n   # P_APPLY 0.5
    u, w = fake_carve(), fake_carve(carve="cv2")
    assert any(not torch.equal(u.batch(qs, blocks)[1], w.batch(qs, blocks)[1]) for _ in range(10))  # its own stream
    assert S.ARMS["pad"] == (S.ARMS["base"][0], PadCarve) and S.ARMS["padbd20"] == (S.BDrop, PadCarve)
    with S.patched("padbd20"):
        assert LG.CacheCarve is PadCarve and LG.LeanMLP8D is S.BDrop
    assert LG.CacheCarve is S.ARMS["base"][1]
    orig = LG.fit_variant
    try:
        with training():
            assert PadCarve.TRAIN and LG.fit_variant is not orig
            raise KeyError("x")
    except KeyError:
        pass
    assert not PadCarve.TRAIN and LG.fit_variant is orig
    calls = []

    def fake_fit(tr, *args, **kw):
        calls.append([c_.pad_stats["questions"] for c_ in tr])
        return "fit"

    LG.fit_variant = fake_fit
    try:
        with training():
            e = fake_carve()
            e.batch(qs, blocks)
            assert e.pad_stats["questions"] == 4
            assert LG.fit_variant([e], None) == "fit" and calls == [[0]]              # reset before the fit it wraps
    finally:
        LG.fit_variant = orig
    for split in SPLITS:
        carves, basis = plan_of(split)
        assert [d for d, _ in carves] == list(LG.FITS[split]) and basis == LG.basis_of(LG.FITS[split])
    assert primary("J5", "webqsp") and primary("L-squad", "squad") and not primary("L-squad", "metaqa")
    assert sum(primary(s, d) for s in SPLITS for d in LG.EVAL_ORDER) == 11
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)

        def write(calls_):
            for split in SPLITS:
                rws = []
                for ds in LG.EVAL_ORDER:
                    cl = calls_.get((split, ds), "WITHIN")
                    dv = {"GAIN": 0.02, "LOSS": -0.02, "WITHIN": 0.0}[cl]
                    rws.append({"dataset": ds, "read": "in-domain" if ds in LG.FITS[split] else "zero-shot",
                                "base": [0.5, 0.4, 0.3], "new": [0.5 + dv, 0.4, 0.3], "rrf": [0.4, 0.3, 0.2],
                                "delta": [dv, 0.0, 0.0], "lo": [dv - 0.01] * 3, "hi": [dv + 0.01] * 3, "call": cl})
                base = f"outputs\\step1\\fits\\{split}"
                LC.write_json(td / f"compare-{split}.json", {
                    "new": f"outputs/full_bdrop20/fits/{split}", "bases": [base], "candidate": "p@swa",
                    "carve": "s1eval", "trained_on": sorted(LG.FITS[split]),
                    "by_base": {base: {"rows": rws, "verdict": "x", "decides": True}}})

        write({("L-musique", "musique"): "GAIN"})
        assert grade(td, {}, "bdrop20", check_fits=False)["verdict"] == "ADOPT"
        write({("L-musique", "musique"): "GAIN", ("L-squad", "hotpotqa"): "LOSS"})
        assert grade(td, {}, "bdrop20", check_fits=False)["verdict"] == "NOT_ADOPTED"     # a secondary LOSS blocks
        write({("L-squad", "hotpotqa"): "GAIN"})
        assert grade(td, {}, "bdrop20", check_fits=False)["verdict"] == "NOT_ADOPTED"     # only a secondary GAIN
        write({})
        (td / "compare-J5.json").unlink()
        assert grade(td, {}, "bdrop20", check_fits=False)["verdict"] == "INCOMPLETE"
        write({("J5", "webqsp"): "GAIN"})
        r = json.loads((td / "compare-L-2wiki.json").read_text(encoding="utf-8"))
        r["candidate"] = "p@ep7"
        LC.write_json(td / "compare-L-2wiki.json", r)
        assert grade(td, {}, "bdrop20", check_fits=False)["verdict"] == "INCOMPLETE"
        write({("J5", "webqsp"): "GAIN"})
        LC.write_json(td / "elsewhere.json", json.loads((td / "compare-L-musique.json").read_text(encoding="utf-8")))
        (td / "compare-L-musique.json").unlink()
        g = grade(td, {"L-musique": str(td / "elsewhere.json")}, "bdrop20", out=td / "grade", check_fits=False)
        assert g["verdict"] == "ADOPT" and (td / "grade.md").exists()
    print("selftest: pad copies only a question's own unranked non-gold rows after its own rows, keeps its golds, "
          "lands the ranked share in [0.08, 0.20] and the cap, never pads a pool without ranked rows or without "
          "unranked non-gold rows, recomputes rrf's base z-score on the padded pool, pads about half the questions, "
          "is seeded per carve and reset per fit, and a carve made outside training is lean_gpu's; the arms and "
          "patching; every split's plan; the full run's grade (ADOPT, a secondary LOSS, a secondary-only GAIN, a "
          "missing or undeclared comparison, a reused comparison). all checks passed")
    return 0


# ── main ─────────────────────────────────────────────────────────────────────


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("cmd", nargs="?")
    ap.add_argument("--split", default="L-musique", choices=SPLITS)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--full-root", default=str(FULL))
    ap.add_argument("--reuse", default="", help="grade: SPLIT=COMPARE_JSON,... (a comparison filed elsewhere)")
    ap.add_argument("--grade-arm", default="bdrop20")
    k, rest = ap.parse_known_args(argv)
    if k.selftest:
        return selftest()
    if k.cmd == "grade":
        reuse = dict(x.split("=", 1) for x in k.reuse.split(",") if x)
        g = grade(k.full_root, reuse, k.grade_arm, out=Path(k.full_root) / "grade")
        return 1 if g["verdict"] == "INCOMPLETE" else 0
    if k.cmd == "smoke":
        sp = argparse.ArgumentParser()
        sp.add_argument("--device", default="cpu")
        sp.add_argument("--host", action="store_true")
        sp.add_argument("--out-root")
        s = sp.parse_args(rest)
        return smoke(s.device, s.host, s.out_root)
    if k.cmd == "train":
        return train([k.cmd] + rest, k.split)
    if k.cmd in ("read", "compare"):
        if PadCarve.TRAIN:
            raise SystemExit("a read must not pad")
        return S.main([k.cmd] + rest)
    raise SystemExit("lean_screen2: train, read, compare, grade, smoke, or --selftest")


if __name__ == "__main__":
    sys.exit(main())
