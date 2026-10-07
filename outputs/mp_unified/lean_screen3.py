"""Screens, third round (docs/SCREENS.md): one arm on lean_screen2's commands and lean_screen's rule, unchanged.

    python outputs/mp_unified/lean_screen3.py smoke --device cuda --host
    python outputs/mp_unified/lean_screen3.py train --name scr-zret --arm zret --device cuda --host
    python outputs/mp_unified/lean_screen3.py read --name scr-zret --device cuda --host
    python outputs/mp_unified/lean_screen3.py compare --new outputs/screen/fits/scr-zret \\
        --base outputs/step1/fits/L-musique --out outputs/screen/scr-zret
    python outputs/mp_unified/lean_screen3.py --selftest

The arm (it changes every training dataset and every read alike):
  zret     every pool z-score taken against the pool's retrieved rows (rrf > 0: a node in the dense or the splade
           top-1000 list) in place of all of its rows: each block's z-scores in the model, and rrf's base z-score. A
           pool with fewer than two retrieved rows, or a column constant over them, keeps its whole-pool z-score
           there. The share of a pool that retrieval ranked (10 to 18% in the big pools, 56 to 100% in the small ones)
           then no longer moves a row's input. Label-free, the same in training and at read; no new parameters. Where
           every row is retrieved (squad's pools) it is lean_gpu's model bit for bit.
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

import lean_screen2 as S2  # noqa: E402

S = S2.S
LG, LC, LM = S.LG, S.LC, S.LM
I_RRF = LM.SPLIT["rank"].index("rrf")
log = S.log


# ── the zret arm ─────────────────────────────────────────────────────────────


def seg_zscore_ref(x, nq, B, ref, eps=1e-6):
    """Each column's z-score against its segment's reference rows (ref: 1 or 0 per row). A segment with fewer than
    two reference rows, or a column whose sd over them is under eps, takes lean_gpu's whole-segment z-score."""
    dev = x.device
    full = LG.seg_zscore8D(x, nq, B, eps)
    r = ref.to(x.dtype)
    cnt = torch.zeros(B, dtype=x.dtype, device=dev).index_add_(0, nq, r)
    den = cnt.clamp_min(1.0).unsqueeze(1)
    w = r.unsqueeze(1)
    mean = torch.zeros(B, x.shape[1], dtype=x.dtype, device=dev).index_add_(0, nq, x * w) / den
    c = x - mean[nq]
    var = torch.zeros(B, x.shape[1], dtype=x.dtype, device=dev).index_add_(0, nq, c * c * w) / den
    zero = var == 0
    sd = torch.where(zero, torch.zeros_like(var), torch.where(zero, torch.ones_like(var), var).sqrt())
    ok = (cnt.unsqueeze(1) >= 2) & (sd >= eps)
    z = c / sd.clamp_min(eps)[nq]
    return torch.where(ok[nq], z, full)


def retrieved(feats):
    """1 for a row retrieval ranked (rrf > 0, the rank block's rrf column), else 0."""
    rrf = feats["rank"][:, I_RRF]
    return (rrf > 0).to(rrf.dtype)


class ZRet(LG.LeanMLP8D):
    """zret: lean_mlp's forward with every z-score, rrf's base z-score included, against the pool's retrieved rows."""

    def __init__(self, blocks, widths, hidden=128, dropout=0.1, seed=0, arm="ctl", ctx="none"):
        if ctx != "none":
            raise SystemExit("zret takes ctx none only")
        if "rank" not in blocks or widths["rank"] != len(LM.SPLIT["rank"]):
            raise SystemExit("zret needs the rank block: its rrf column marks the retrieved rows")
        super().__init__(blocks, widths, hidden, dropout, seed, arm, ctx)

    def forward(self, feats, keep, nq, B, base_z):
        ref = retrieved(feats)
        parts = []
        for j, b in enumerate(self.blocks):
            m = keep[nq, j].unsqueeze(1)
            raw = S.raw_of(self, feats, b, nq)
            parts += [raw * m, seg_zscore_ref(raw, nq, B, ref) * m, m]
        h = self.drop(Fn.gelu(self.l1(torch.cat(parts, 1))))
        h = self.drop(Fn.gelu(self.l2(h)))
        bz = seg_zscore_ref(feats["rank"][:, I_RRF:I_RRF + 1], nq, B, ref).squeeze(1)
        return self.base_w * bz + self.out(h).squeeze(-1)


S.ARMS.update({"zret": (ZRet, S.ARMS["base"][1])})
ROUND3 = ("zret",)


# ── train (lean_screen2's, with this file's sha) ─────────────────────────────


def train(argv, split):
    rc = S2.train(argv, split)
    name, out_root = S2.where(argv)
    sj = out_root / name / "screen.json"
    if sj.exists():
        rec = json.loads(sj.read_text(encoding="utf-8"))
        rec["lean_screen3_sha256"] = LC.sha_src(__file__)
        LC.write_json(sj, rec)
    return rc


# ── smoke ────────────────────────────────────────────────────────────────────


def smoke(device, host, out_root=None):
    """zret on 2wiki's select carve (1 epoch, run twice: a repeat must be IDENTICAL), then read. A crash or a
    differing repeat fails it (exit 1)."""
    t0 = time.time()
    root = Path(out_root or S.OUT / "smoke3")
    h = ["--host"] if host else []
    rec, ok = {"arms": {}}, True
    for arm in ROUND3:
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
    log(f"smoke3: {json.dumps(rec['arms'])}; {'ok' if ok else 'FAILED'} ({rec['seconds']:.0f}s)")
    return 0 if ok else 1


# ── selftest ─────────────────────────────────────────────────────────────────


def ref_numpy(x, nq, B, ref, eps=1e-6):
    """seg_zscore_ref by hand, in float64, for the selftest."""
    x, nq, ref = x.double().numpy(), nq.numpy(), ref.numpy().astype(bool)
    out = np.zeros_like(x)
    for q in range(B):
        rows = nq == q
        xa, ra = x[rows], ref[rows]
        for k in range(x.shape[1]):
            col = xa[:, k]
            m_all, s_all = col.mean(), col.std()
            whole = np.zeros_like(col) if s_all < eps else (col - m_all) / s_all
            if ra.sum() >= 2 and col[ra].std() >= eps:
                out[rows, k] = (col - col[ra].mean()) / col[ra].std()
            else:
                out[rows, k] = whole
    return out


def selftest():
    LG.bind_device_ops()
    torch.manual_seed(0)
    n = torch.tensor([6, 5, 4, 3, 2])
    B, N = n.numel(), int(n.sum())
    nq = torch.repeat_interleave(torch.arange(B), n)
    x = torch.randn(N, 4) * torch.tensor([1.0, 10.0, 0.01, 3.0])
    ref = torch.tensor([1, 1, 1, 0, 0, 0,  1, 1, 1, 1, 1,  1, 0, 0, 0,  0, 0, 0,  1, 1], dtype=torch.float32)
    x[6:11, 3] = 2.5                                                    # constant over q1's retrieved rows
    # every row retrieved: lean_gpu's z-score bit for bit
    assert torch.equal(seg_zscore_ref(x, nq, B, torch.ones(N)), LG.seg_zscore8D(x, nq, B))
    # against a hand computation: q0 and q4 by their retrieved rows, q2 (one) and q3 (none) by the whole pool,
    # q1's constant column by the whole pool (0)
    got = seg_zscore_ref(x, nq, B, ref).double().numpy()
    want = ref_numpy(x, nq, B, ref)
    assert np.allclose(got, want, atol=1e-4, rtol=1e-4), np.abs(got - want).max()
    assert np.allclose(got[11:18], LG.seg_zscore8D(x, nq, B)[11:18].double().numpy())
    assert np.all(got[6:11, 3] == 0)
    # finite gradients, through segments with zero and one retrieved row and a constant column
    xg = x.clone().requires_grad_(True)
    seg_zscore_ref(xg, nq, B, ref).square().sum().backward()
    assert torch.isfinite(xg.grad).all()
    # the model: equal to lean_gpu's where every row is retrieved, different where not
    widths = {"rank": 5, "WALK": 3}
    feats = {"rank": torch.rand(N, 5) * 0.03 + 0.001, "WALK": torch.randn(N, 3)}
    base = LG.LeanMLP8D(["rank", "WALK"], widths, 16, dropout=0.1, seed=0).eval()
    zr = ZRet(["rank", "WALK"], widths, 16, dropout=0.1, seed=0).eval()
    zr.load_state_dict(base.state_dict())
    with torch.no_grad():
        for p in zr.out.parameters():
            p.normal_()
        base.load_state_dict(zr.state_dict())
        keep = torch.ones(B, 2)
        bz = LG.seg_zscore8D(feats["rank"][:, I_RRF:I_RRF + 1], nq, B).squeeze(1)
        assert torch.equal(zr(feats, keep, nq, B, bz), base(feats, keep, nq, B, bz))
        f2 = {"rank": feats["rank"] * ref.unsqueeze(1), "WALK": feats["WALK"]}
        bz2 = LG.seg_zscore8D(f2["rank"][:, I_RRF:I_RRF + 1], nq, B).squeeze(1)
        s_z, s_b = zr(f2, keep, nq, B, bz2), base(f2, keep, nq, B, bz2)
        assert not torch.allclose(s_z[:6], s_b[:6]) and torch.equal(s_z[11:18], s_b[11:18])  # q2, q3: whole pool
    zr.train()
    s = zr(f2, keep, nq, B, bz2)
    LG.listwiseD(s, (torch.arange(N) % 3 == 0), nq, B).backward()
    assert all(torch.isfinite(p.grad).all() for p in zr.parameters() if p.grad is not None)
    for bad in (dict(ctx="film"),):
        try:
            ZRet(["rank", "WALK"], widths, 16, **bad)
            raise AssertionError("zret took a context")
        except SystemExit:
            pass
    try:
        ZRet(["WALK"], {"WALK": 3}, 16)
        raise AssertionError("zret took no rank block")
    except SystemExit:
        pass
    # the arms and patching
    assert S.ARMS["zret"] == (ZRet, S.ARMS["base"][1])
    with S.patched("zret"):
        assert LG.LeanMLP8D is ZRet and LG.CacheCarve is S.ARMS["base"][1]
    assert LG.LeanMLP8D is S.ARMS["base"][0]
    print("selftest: zret's z-score equals lean_gpu's bit for bit where every row is retrieved, matches a hand "
          "computation against the retrieved rows, keeps the whole-pool value for pools with fewer than two retrieved "
          "rows and for columns constant over them, and gives finite gradients there; the zret model equals lean_gpu's "
          "where every row is retrieved and differs where not; the arm and patching. all checks passed")
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
    raise SystemExit("lean_screen3: train, read, compare, smoke, or --selftest")


if __name__ == "__main__":
    sys.exit(main())
