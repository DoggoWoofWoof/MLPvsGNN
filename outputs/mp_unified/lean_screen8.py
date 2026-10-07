"""Screens, eighth round (docs/SCREENS.md): one arm on lean_screen2's commands and lean_screen's rule, unchanged; two
fits per screen (L-musique and L-hotpotqa), their verdict over both by screen_pair.py.

    python outputs/mp_unified/lean_screen8.py smoke --device cuda --host
    python outputs/mp_unified/lean_screen8.py train --split L-musique --name scr-pret --arm pret --device cuda --host
    python outputs/mp_unified/lean_screen8.py read --name scr-pret --device cuda --host
    python outputs/mp_unified/lean_screen8.py compare --new outputs/screen/fits/scr-pret \\
        --base outputs/step1/fits/L-musique --out outputs/screen/scr-pret
    python outputs/mp_unified/lean_screen8.py --selftest

The arm (it changes every training dataset and every read alike):
  pret     rank inputs for the retrieval blocks only (a preprocessing step; prank, narrowed): the two retrieval blocks,
           rank (dense_rr, splade_rr, rrf, agreement, is_seed) and dense_cos, take prank's form [raw m, hi m, lo m, m],
           hi and lo being each column's competition rank in the pool from the top and from the bottom as
           60 / (60 + min(rank, 50)) (lean_screen5.rank_inputs, unchanged). Every other block keeps step 1's form
           [raw m, z m, m]: the structure blocks (topo_STRUCT, depth_STRUCT, WALK, WALKF, SEED, DISTS) keep their
           within-pool z-scores, and SEMB (learned) keeps its z-score. rrf's base z-score is step 1's, unchanged.
           A retrieval column's z-score moves with the share of the pool that retrieval ranked (a thinly ranked pool
           puts its ranked rows far out in the column's tail); its rank among the top 50 does not. Label-free, the
           same in training and at read time; no new hyperparameter. With no block ranked it is step 1's model, with
           every fixed block ranked it is prank (the selftest checks both, weight for weight).
"""
import os
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402

import lean_screen5 as S5  # noqa: E402

S2 = S5.S2
S = S2.S
LG, LC, LM = S.LG, S.LC, S.LM
log = S.log
RANKED = ("rank", "dense_cos")   # the retrieval blocks: lean_mlp.SPLIT's rank columns and dense_cos


# ── the pret arm ─────────────────────────────────────────────────────────────


class PRet(LG.LeanMLP8D):
    """pret: lean_mlp's model with [raw m, hi m, lo m, m] for the blocks in `ranked` and [raw m, z m, m] for the
    rest."""

    ranked = RANKED

    def __init__(self, blocks, widths, hidden=128, dropout=0.1, seed=0, arm="ctl", ctx="none"):
        if ctx != "none":
            raise SystemExit("pret takes ctx none only")
        super().__init__(blocks, widths, hidden, dropout, seed, arm, ctx)
        if "SEMB" in self.ranked:
            raise SystemExit("pret never ranks SEMB")
        self.in_w = sum(3 * w + 1 if b in self.ranked else 2 * w + 1 for b, w in self.widths.items())
        self.l1 = nn.Linear(self.in_w, hidden)

    def forward(self, feats, keep, nq, B, base_z):
        parts = []
        for j, b in enumerate(self.blocks):
            m = keep[nq, j].unsqueeze(1)
            raw = S.raw_of(self, feats, b, nq)
            if b in self.ranked:
                hi, lo = S5.rank_inputs(raw, nq, B)
                parts += [raw * m, hi * m, lo * m, m]
            else:
                parts += [raw * m, LM.seg_zscore(raw, nq, B) * m, m]
        h = self.drop(Fn.gelu(self.l1(torch.cat(parts, 1))))
        h = self.drop(Fn.gelu(self.l2(h)))
        return self.base_w * base_z + self.out(h).squeeze(-1)


S.ARMS.update({"pret": (PRet, S.ARMS["base"][1])})
ROUND8 = ("pret",)


# ── train (lean_screen2's, with this file's sha) ─────────────────────────────


def train(argv, split):
    rc = S2.train(argv, split)
    name, out_root = S2.where(argv)
    sj = out_root / name / "screen.json"
    if sj.exists():
        rec = json.loads(sj.read_text(encoding="utf-8"))
        rec["lean_screen8_sha256"] = LC.sha_src(__file__)
        rec["lean_screen5_sha256"] = LC.sha_src(S5.__file__)
        LC.write_json(sj, rec)
    return rc


# ── smoke ────────────────────────────────────────────────────────────────────


SMOKE_TRAIN = {"pret": "2wiki=select"}


def smoke(device, host, out_root=None):
    """The arm on 2wiki select (1 epoch, run twice: a repeat must be IDENTICAL), then read. A crash or a differing
    repeat fails it (exit 1)."""
    t0 = time.time()
    root = Path(out_root or S.OUT / "smoke8")
    h = ["--host"] if host else []
    rec, ok = {"arms": {}}, True
    for arm in ROUND8:
        rc = main(["train", "--name", f"smoke-{arm}", "--arm", arm, "--train", SMOKE_TRAIN[arm], "--basis", "2wiki",
                   "--variants", "p", "--config", "2e-3:1e-4:0.1:1:0", "--repeat", "2", "--device", device,
                   "--out-root", str(root)] + h)
        rr = main(["read", "--name", f"smoke-{arm}", "--read", "2wiki=select", "--device", device, "--out-root",
                   str(root)] + h)
        rec["arms"][arm] = {"train_rc": rc, "repeat": "IDENTICAL" if rc == 0 else "DIFFERENT", "read_rc": rr}
        ok = ok and rc == 0 and rr == 0
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
    rec["ok"], rec["seconds"], rec["script_sha256"] = ok, time.time() - t0, LC.sha_src(__file__)
    LC.write_json(root / "smoke.json", rec)
    log(f"smoke8: {json.dumps(rec['arms'])}; {'ok' if ok else 'FAILED'} ({rec['seconds']:.0f}s)")
    return 0 if ok else 1


# ── selftest ─────────────────────────────────────────────────────────────────


def copy_into(dst, src):
    """dst's parameters and buffers set to src's (same names and shapes)."""
    sd = src.state_dict()
    assert set(sd) == set(dst.state_dict()), (sorted(sd), sorted(dst.state_dict()))
    dst.load_state_dict(sd)


def selftest():
    LG.bind_device_ops()
    torch.manual_seed(0)
    widths = {"rank": 5, "dense_cos": 1, "WALK": 4, "SEED": 3, "SEMB": 6}
    blocks = list(widths)
    n = torch.tensor([30, 12, 4, 61])
    B, N = n.numel(), int(n.sum())
    nq = torch.repeat_interleave(torch.arange(B), n)

    def with_semb(cls):
        class Stub(cls):
            def __init__(self, *a, **k):
                super().__init__(*a, **k)
                self.U = nn.Parameter(torch.randn(8, 6) * 0.1)
                self.V = nn.Parameter(torch.randn(5, 6) * 0.1)
        return Stub

    rank = torch.rand(N, 5) * (torch.rand(N, 1) < 0.3)
    feats = {"rank": rank, "dense_cos": torch.rand(N, 1), "WALK": torch.randn(N, 4), "SEED": torch.randn(N, 3),
             "SEMB": (torch.randn(B, 8), torch.randn(N, 5))}
    keep = (torch.rand(B, len(blocks)) < 0.8).float()
    bz = LG.seg_zscore8D(rank[:, 2:3], nq, B).squeeze(1)

    # pret's width: the retrieval blocks 3w+1, every other block 2w+1
    m = with_semb(PRet)(blocks, widths, 16, dropout=0.1, seed=0)
    assert m.l1.in_features == (3 * 5 + 1) + (3 * 1 + 1) + (2 * 4 + 1) + (2 * 3 + 1) + (2 * 6 + 1)
    with torch.no_grad():
        for p in m.out.parameters():
            p.normal_()
    # with no block ranked, pret is step 1's model weight for weight; with every fixed block ranked, prank
    none_ = type("PRetNone", (with_semb(PRet),), {"ranked": ()})(blocks, widths, 16, dropout=0.1, seed=0)
    base = with_semb(LG.LeanMLP8D)(blocks, widths, 16, dropout=0.1, seed=0)
    copy_into(none_, base)
    allf = type("PRetAll", (with_semb(PRet),), {"ranked": tuple(b for b in blocks if b != "SEMB")})(
        blocks, widths, 16, dropout=0.1, seed=0)
    prank = with_semb(S5.PRank)(blocks, widths, 16, dropout=0.1, seed=0)
    copy_into(allf, prank)
    for a, b in ((none_, base), (allf, prank)):
        a.eval()
        b.eval()
        with torch.no_grad():
            assert torch.equal(a(feats, keep, nq, B, bz), b(feats, keep, nq, B, bz)), type(a).__name__
    # the ranked part reads the same under any increasing map of a retrieval column; the structure part does not
    m.eval()
    with torch.no_grad():
        f2 = dict(feats)
        f2["rank"] = torch.exp(3 * rank) - 7
        hi1, lo1 = S5.rank_inputs(rank, nq, B)
        hi2, lo2 = S5.rank_inputs(f2["rank"], nq, B)
        assert torch.equal(hi1, hi2) and torch.equal(lo1, lo2)
        s1 = m(feats, keep, nq, B, bz)
        s2 = m(f2, keep, nq, B, bz)
        assert not torch.equal(s1, s2)                        # the raw columns still enter
    # a pool's thinly or richly ranked retrieval column: a ranked row's z-score moves with the ranked share, its rank
    # inputs do not
    col_thin = torch.zeros(2000, 1)
    col_rich = torch.zeros(2000, 1)
    vals = torch.linspace(1.0, 0.5, 200).unsqueeze(1)
    col_thin[:200] = vals
    col_rich[:200] = vals
    col_rich[200:1000] = torch.linspace(0.49, 0.01, 800).unsqueeze(1)
    z0 = torch.zeros(2000, dtype=torch.long)
    zt = LM.seg_zscore(col_thin, z0, 1)
    zr = LM.seg_zscore(col_rich, z0, 1)
    assert float((zt[:5] - zr[:5]).abs().min()) > 0.5
    ht, _ = S5.rank_inputs(col_thin, z0, 1)
    hr, _ = S5.rank_inputs(col_rich, z0, 1)
    assert torch.equal(ht[:50], hr[:50])
    # finite gradients in training, eval reproducible, keep masks honoured
    m.train()
    s = m(feats, keep, nq, B, bz)
    LG.listwiseD(s, torch.arange(N) % 4 == 0, nq, B).backward()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
    m.eval()
    with torch.no_grad():
        assert torch.equal(m(feats, keep, nq, B, bz), m(feats, keep, nq, B, bz))
    try:
        PRet(["rank"], {"rank": 5}, 16, ctx="film")
        raise AssertionError("pret took a context")
    except SystemExit:
        pass
    # the arm, patching
    assert S.ARMS["pret"] == (PRet, S.ARMS["base"][1]) and S.ARMS["prank"][0] is S5.PRank
    with S.patched("pret"):
        assert LG.LeanMLP8D is PRet
    assert LG.LeanMLP8D is S.ARMS["base"][0]
    print("selftest: pret's width (retrieval blocks 3w+1, others 2w+1); with no block ranked it is step 1's model and "
          "with every fixed block ranked it is prank, weight for weight; retrieval ranks unchanged by an increasing "
          "map while the raw columns still enter; a ranked row's z-score moves with the ranked share, its rank inputs "
          "do not; finite gradients, eval reproducible; arm and patching. all checks passed")
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
    if k.cmd in ("read", "compare"):
        return S2.main([k.cmd] + rest)
    raise SystemExit("lean_screen8: train, read, compare, smoke, or --selftest")


if __name__ == "__main__":
    sys.exit(main())
