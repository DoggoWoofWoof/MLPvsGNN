"""Screens, fourth round (docs/SCREENS.md): one arm on lean_screen2's commands and lean_screen's rule, unchanged.

    python outputs/mp_unified/lean_screen4.py smoke --device cuda --host
    python outputs/mp_unified/lean_screen4.py train --name scr-ztop50 --arm ztop50 --device cuda --host
    python outputs/mp_unified/lean_screen4.py read --name scr-ztop50 --device cuda --host
    python outputs/mp_unified/lean_screen4.py compare --new outputs/screen/fits/scr-ztop50 \\
        --base outputs/step1/fits/L-musique,outputs/screen/fits/scr-zret --out outputs/screen/scr-ztop50
    python outputs/mp_unified/lean_screen4.py --selftest

The arm (it changes every training dataset and every read alike):
  ztop50   zret (lean_screen3.py) with a fixed-size reference: every pool z-score, rrf's base z-score included, is
           taken against the pool's top 50 retrieved rows by rrf (rrf > 0; ties by row order) in place of all of its
           retrieved rows. zret took the share of retrieved rows out of the inputs; their number stayed in (205 to 379
           in the big pools, 50 to 63 in the small ones), and with it how far a top row sits in its pool's tail. A
           pool with fewer than two reference rows, or a column constant over them, keeps its whole-pool z-score
           there, as in zret. Label-free, the same in training and at read; no new parameters. Where no pool has more
           than 50 retrieved rows it is zret bit for bit, and where every row is retrieved (squad's 50-row pools)
           lean_gpu's model bit for bit.
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
import torch.nn.functional as Fn  # noqa: E402

import lean_screen3 as S3  # noqa: E402

S2, S = S3.S2, S3.S
LG, LC, LM = S.LG, S.LC, S.LM
I_RRF = S3.I_RRF
log = S.log
K = 50


# ── the ztop50 arm ───────────────────────────────────────────────────────────


def top_ref(rrf, nq, B, k=K):
    """1 for each of a pool's k best retrieved rows by rrf (rrf > 0; ties by row order), else 0."""
    dev, N = rrf.device, rrf.numel()
    o1 = torch.sort(-rrf, stable=True).indices                     # rrf descending, ties by row order
    o2 = torch.sort(nq[o1], stable=True).indices                   # then by pool, keeping that order
    order = o1[o2]
    cnt = torch.zeros(B, dtype=torch.float32, device=dev).index_add_(
        0, nq, torch.ones(N, dtype=torch.float32, device=dev)).long()
    start = torch.cumsum(cnt, 0) - cnt
    pos = torch.empty(N, dtype=torch.long, device=dev)
    pos[order] = torch.arange(N, device=dev)
    return ((pos - start[nq] < k) & (rrf > 0)).to(rrf.dtype)


class ZTop(S3.ZRet):
    """ztop50: zret's forward with the reference rows cut to the pool's top K retrieved rows by rrf."""

    K = K

    def forward(self, feats, keep, nq, B, base_z):
        rrf = feats["rank"][:, I_RRF]
        ref = top_ref(rrf, nq, B, self.K)
        parts = []
        for j, b in enumerate(self.blocks):
            m = keep[nq, j].unsqueeze(1)
            raw = S.raw_of(self, feats, b, nq)
            parts += [raw * m, S3.seg_zscore_ref(raw, nq, B, ref) * m, m]
        h = self.drop(Fn.gelu(self.l1(torch.cat(parts, 1))))
        h = self.drop(Fn.gelu(self.l2(h)))
        bz = S3.seg_zscore_ref(rrf.unsqueeze(1), nq, B, ref).squeeze(1)
        return self.base_w * bz + self.out(h).squeeze(-1)


S.ARMS.update({"ztop50": (ZTop, S.ARMS["base"][1])})
ROUND4 = ("ztop50",)


# ── train (lean_screen2's, with this file's sha) ─────────────────────────────


def train(argv, split):
    rc = S2.train(argv, split)
    name, out_root = S2.where(argv)
    sj = out_root / name / "screen.json"
    if sj.exists():
        rec = json.loads(sj.read_text(encoding="utf-8"))
        rec["lean_screen4_sha256"] = LC.sha_src(__file__)
        LC.write_json(sj, rec)
    return rc


# ── smoke ────────────────────────────────────────────────────────────────────


def smoke(device, host, out_root=None):
    """ztop50 on 2wiki's select carve (1 epoch, run twice: a repeat must be IDENTICAL), then read. A crash or a
    differing repeat fails it (exit 1)."""
    t0 = time.time()
    root = Path(out_root or S.OUT / "smoke4")
    h = ["--host"] if host else []
    rec, ok = {"arms": {}}, True
    for arm in ROUND4:
        rc = main(["train", "--name", f"smoke-{arm}", "--arm", arm, "--train", "2wiki=select", "--basis", "2wiki",
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
    log(f"smoke4: {json.dumps(rec['arms'])}; {'ok' if ok else 'FAILED'} ({rec['seconds']:.0f}s)")
    return 0 if ok else 1


# ── selftest ─────────────────────────────────────────────────────────────────


def top_ref_numpy(rrf, nq, B, k):
    """top_ref by hand, for the selftest."""
    rrf, nq = rrf.numpy(), nq.numpy()
    out = np.zeros(rrf.size)
    for q in range(B):
        rows = np.flatnonzero(nq == q)
        best = sorted(rows, key=lambda i: (-rrf[i], i))[:k]
        out[[i for i in best if rrf[i] > 0]] = 1.0
    return out


def selftest():
    LG.bind_device_ops()
    torch.manual_seed(0)
    # top_ref against a hand computation: ties, unretrieved rows, pools in any row order, k = 3
    n = torch.tensor([7, 5, 2, 4, 1])
    B, N = n.numel(), int(n.sum())
    nq = torch.repeat_interleave(torch.arange(B), n)
    rrf = torch.tensor([0.03, 0.01, 0.03, 0.0, 0.02, 0.03, 0.005,  0.0, 0.0, 0.01, 0.0, 0.0,  0.02, 0.02,
                        0.01, 0.0, 0.01, 0.01,  0.0])
    for k in (1, 2, 3, 10):
        assert np.array_equal(top_ref(rrf, nq, B, k).numpy(), top_ref_numpy(rrf, nq, B, k)), k
    perm = torch.randperm(N)                                      # rows of a pool need not be contiguous
    assert np.array_equal(top_ref(rrf[perm], nq[perm], B, 3).numpy(), top_ref_numpy(rrf[perm], nq[perm], B, 3))
    assert top_ref(rrf, nq, B, 3)[:7].tolist() == [1, 0, 1, 0, 0, 1, 0]    # the three 0.03s; the 0.02 is fourth
    big = torch.rand(400)
    nqb = torch.zeros(400, dtype=torch.long)
    assert int(top_ref(big, nqb, 1).sum()) == K
    # the model: zret bit for bit where no pool has more than K retrieved rows, lean_gpu's where every row is
    # retrieved, different from zret where a pool has more than K retrieved rows
    n = torch.tensor([60, 40, 3])
    B, N = n.numel(), int(n.sum())
    nq = torch.repeat_interleave(torch.arange(B), n)
    widths = {"rank": 5, "WALK": 3}
    rank = torch.rand(N, 5) * 0.03 + 0.001
    feats = {"rank": rank, "WALK": torch.randn(N, 3)}
    base = LG.LeanMLP8D(["rank", "WALK"], widths, 16, dropout=0.1, seed=0).eval()
    zt = ZTop(["rank", "WALK"], widths, 16, dropout=0.1, seed=0).eval()
    zr = S3.ZRet(["rank", "WALK"], widths, 16, dropout=0.1, seed=0).eval()
    with torch.no_grad():
        for p in zt.out.parameters():
            p.normal_()
        base.load_state_dict(zt.state_dict())
        zr.load_state_dict(zt.state_dict())
        keep = torch.ones(B, 2)
        bz = LG.seg_zscore8D(rank[:, I_RRF:I_RRF + 1], nq, B).squeeze(1)
        few = {"rank": rank.clone(), "WALK": feats["WALK"]}
        few["rank"][10:60, I_RRF] = 0.0                            # pool 0: 10 retrieved rows
        assert torch.equal(zt(few, keep, nq, B, bz), zr(few, keep, nq, B, bz))
        sq = {"rank": rank[60:], "WALK": feats["WALK"][60:]}       # every row retrieved, pools of 40 and 3
        nq2 = nq[60:] - 1
        bz2 = LG.seg_zscore8D(sq["rank"][:, I_RRF:I_RRF + 1], nq2, 2).squeeze(1)
        assert torch.equal(zt(sq, keep[1:], nq2, 2, bz2), base(sq, keep[1:], nq2, 2, bz2))
        s_t, s_r = zt(feats, keep, nq, B, bz), zr(feats, keep, nq, B, bz)
        assert not torch.allclose(s_t[:60], s_r[:60]) and torch.equal(s_t[60:], s_r[60:])
    zt.train()
    s = zt(feats, keep, nq, B, bz)
    LG.listwiseD(s, (torch.arange(N) % 3 == 0), nq, B).backward()
    assert all(torch.isfinite(p.grad).all() for p in zt.parameters() if p.grad is not None)
    try:
        ZTop(["rank", "WALK"], widths, 16, ctx="film")
        raise AssertionError("ztop50 took a context")
    except SystemExit:
        pass
    # the arms and patching
    assert S.ARMS["ztop50"] == (ZTop, S.ARMS["base"][1]) and S.ARMS["zret"] == (S3.ZRet, S.ARMS["base"][1])
    with S.patched("ztop50"):
        assert LG.LeanMLP8D is ZTop and LG.CacheCarve is S.ARMS["base"][1]
    assert LG.LeanMLP8D is S.ARMS["base"][0]
    print("selftest: the top-K reference matches a hand computation (ties by row order, unretrieved rows out, pools "
          "in any row order); ztop50 equals zret bit for bit where no pool has more than K retrieved rows, lean_gpu's "
          "model where every row is retrieved, and differs from zret where a pool has more; finite gradients; the arm "
          "and patching. all checks passed")
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
    raise SystemExit("lean_screen4: train, read, compare, smoke, or --selftest")


if __name__ == "__main__":
    sys.exit(main())
