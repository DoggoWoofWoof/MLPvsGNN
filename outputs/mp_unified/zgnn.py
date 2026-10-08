"""Screens, twenty-fifth round (docs/SCREENS.md): feature selection for the GNN, with the same features as the MLP.
The GNN track. **Message passing:** zgn passes each row's hidden state over its question's pool graph, so it is a GNN,
never the MLP and never non-MP.

The user (8 October about 20:55): "also declare the gnn feature selection round, because want both models to have the
same features". Round twenty-four (zfeat.py) selects among seven look blocks the pick leaves out for the MLP (zfs);
this round trains the GNN track's model on the same blocks the same way, and round twenty-four's selection, amended
before any number of it existed, becomes one subset for both: the largest smaller of the two models' mean gains,
admissible for both (zfeat.joint). Both models then read with that subset.

zgn is zfs's model (zfeat.ZFS: zrm over the pick and the seven candidates, the candidates dropped at 0.5 per question
and block in training, masked at read outside the chosen subset) with LAYERS residual message-passing layers between
its hidden state and its output, GraphSAGE's mean form over the pool graph zlink.py built for round twenty-two (the
look's pool edges, undirected, once per family, no self-loops; structural, NER and kNN families on the passage graphs,
the KB's relation edges on metaqa and webqsp):

    a_f(v) = the mean of h(u) over v's neighbours u in family f (0 with none)
    h <- h + GELU(W_t [a_0, a_1, a_2] + b_t),    t = 1 .. LAYERS

W_t and b_t start at zero and draw nothing, so at the start zgn's forward is zfs's and every draw of zfs's (its
initial weights, its dropout, its candidate masks) is zgn's. No dropout on the messages: the draws stay zfs's. Then
zfs's output layer, rrf's base term and rmatch's gated chain match, unchanged. The extra parameters: LAYERS x (3 H x H
+ H), 98,560 at H = 128. Settings and training are zrm's (rmatch.py's train).

Left out of the candidates as for zfs: nbr_agg, gcs and typed_v2 (frozen features or retrieval scores moved over the
pool graph). zgn moves its own hidden state, so the GNN needs none of them, and the two models' inputs stay the same.

The round, per split (L-musique and L-hotpotqa, the screen's two fits):
  train    zgn, as zfs, on the card (seed 0, p@swa).
  select   zfeat.py select reads both models' fits on the five select carves, 128 subsets each (round twenty-four's
           step 3, amended): the common subset.
  read     each zgn fit on the six s1eval carves twice: with the common subset kept (scr-zgn, scr-zgn-hp) and with
           every candidate masked (a work copy of the same fit, scr-zgn-none, scr-zgn-none-hp; the fit's own folder is
           not written by the second read).
  compare  the subset read against the same fit's every-candidate-masked read (decides): does the common subset help
           the GNN? zfs's, zrm's and zlk's fits beside (zgn against zfs, same features: what the message passing adds).
           Pair and re-call under the seed null, relz.py's, each read's base R@5 the masked read's (base compares of
           scr-zgn-none against step 1's fit, as zrc's base-zrm files).
  The verdict is the re-call's: PROMISING is at least one GAIN and no LOSS among the twelve reads. A full run is
  declared in its own file before any of its numbers, after the verdict.

    python outputs/mp_unified/zgnn.py train --split L-musique --name scr-zgn --arm zgn --device cuda --host
    python outputs/mp_unified/zgnn.py read --name scr-zgn --select outputs/zfeat/select.json --device cuda --host
    python outputs/mp_unified/zgnn.py read --name scr-zgn-none --source outputs/screen/fits/scr-zgn --keep none \\
        --device cuda --host
    python outputs/mp_unified/zgnn.py compare --new outputs/screen/fits/scr-zgn-none \\
        --base outputs/step1/fits/L-musique --out outputs/zgnn/base-none-L-musique
    python outputs/mp_unified/zgnn.py compare --new outputs/screen/fits/scr-zgn \\
        --base outputs/screen/fits/scr-zgn-none,outputs/screen/fits/scr-zfs,outputs/screen/fits/scr-zrm \\
        --out outputs/screen/scr-zgn
    python outputs/mp_unified/zgnn.py pair --screens outputs/screen/scr-zgn.json,outputs/screen/scr-zgn-hp.json \\
        --out outputs/screen/scr-zgn-pair
    python outputs/mp_unified/zgnn.py recall --null N1,N2,N3,N4 --pair outputs/screen/scr-zgn-pair.json \\
        --out outputs/screen/scr-zgn-pair-recall
    python outputs/mp_unified/zgnn.py --selftest

Speed: per question, LAYERS passes over its pool's edges (about 4 to 10 a row) after zfs's hidden layers. Any latency
figure for zgn is cold (8216ffe).
"""
import os
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import contextlib  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402

import zfeat as ZF  # noqa: E402
import zlink as ZL  # noqa: E402

ZM, RM, Z, R, S2, S = ZF.ZM, ZF.RM, ZF.Z, ZF.R, ZF.S2, ZF.S
LG, LC, SR = ZF.LG, ZF.LC, Z.SR
S3 = ZM.S3
log = S.log

ARM = "zgn"
assert ARM == ZF.GNN_ARM
LAYERS = 2
FAMS = ZL.FAMS
MP = ("mp_w", "mp_b")
SETTINGS = {"layers": LAYERS, "families": FAMS, "aggregation": "mean per family over the pool graph's neighbours",
            "update": "h + GELU(W [a_0, a_1, a_2] + b), W and b zero at the start", "message_dropout": None,
            "edges": ZL.SETTINGS["edges"], "features": "zfeat.py's (the pick and the seven candidates)"}
OUT = ZF.ROOT / "outputs" / "zgnn"


# ── the model ────────────────────────────────────────────────────────────────


def family_means(h, lk, N):
    """(N, FAMS * H): each row's mean neighbour state per family (0 with no neighbour in a family)."""
    u, v, f = lk["u"], lk["v"], lk["f"]
    H = h.shape[1]
    at = v * FAMS + f
    s = torch.zeros(N * FAMS, H, dtype=h.dtype, device=h.device).index_add_(0, at, h[u])
    d = torch.zeros(N * FAMS, dtype=h.dtype, device=h.device).index_add_(0, at, torch.ones_like(at, dtype=h.dtype))
    return (s / d.clamp_min(1.0).unsqueeze(1)).view(N, FAMS * H)


class ZGNN(ZF.ZFS):
    """zgn: zfs's forward with LAYERS residual mean-aggregation layers over the pool graph after its hidden layers."""

    def __init__(self, blocks, widths, hidden=128, dropout=0.1, seed=0, arm="ctl", ctx="none"):
        super().__init__(blocks, widths, hidden, dropout, seed, arm, ctx)
        H = self.out.in_features
        self.mp_w = nn.Parameter(torch.zeros(LAYERS, H, FAMS * H))
        self.mp_b = nn.Parameter(torch.zeros(LAYERS, H))

    def propagate(self, h, lk):
        N = h.shape[0]
        for t in range(LAYERS):
            h = h + Fn.gelu(Fn.linear(family_means(h, lk, N), self.mp_w[t], self.mp_b[t]))
        return h

    def forward(self, feats, keep, nq, B, base_z):
        if ZL.LK_KEY not in feats:
            raise SystemExit("zgn reads its pool edges (zgnn.GNNCarve)")
        keep = self.keep_of(keep)
        # lean_screen3.ZRet's forward, the message passing between its hidden layers and its output, then
        # rmatch.ChainMatch's gated chain match (zkind.py's form)
        ref = S3.retrieved(feats)
        parts = []
        for j, b in enumerate(self.blocks):
            m = keep[nq, j].unsqueeze(1)
            raw = S.raw_of(self, feats, b, nq)
            parts += [raw * m, S3.seg_zscore_ref(raw, nq, B, ref) * m, m]
        h = self.drop(Fn.gelu(self.l1(torch.cat(parts, 1))))
        h = self.drop(Fn.gelu(self.l2(h)))
        h = self.propagate(h, feats[ZL.LK_KEY])
        bz = S3.seg_zscore_ref(feats["rank"][:, S3.I_RRF:S3.I_RRF + 1], nq, B, ref).squeeze(1)
        s = self.base_w * bz + self.out(h).squeeze(-1)
        if RM.CH_KEY not in feats:
            # a graph without typed relations holds no chains: no chain match, as in rmatch.ChainMatch's forward
            # (bug fix, 9 October about 01:00: the guard above asked for chains on every carve, and the first fit
            # stopped at squad's)
            return s
        fm, fr = self.chain_feats(feats, nq.numel())
        return s + (self.cm_gate[0] * fm + self.cm_gate[1] * fr) * keep[nq, self.j_semb]


GNNCarve = ZL.link_carve(ZF.FeatChainCarve)
S.ARMS.update({ARM: (ZGNN, GNNCarve)})


# ── train, read and compare ──────────────────────────────────────────────────


def train(argv, split):
    """rmatch.py's train (zrm's settings) for the arm zgn, with zfeat's candidates in the sets."""
    if R.arm_of(argv) != ARM:
        raise SystemExit(f"zgnn: train takes --arm {ARM}")
    if S.ARMS.get(ARM) != (ZGNN, GNNCarve) or S.ARMS.get(ZF.ARM) != (ZF.ZFS, ZF.FeatChainCarve):
        raise SystemExit("zgnn: the arms are not zgn's model on its carve and zfs's")
    with ZF.with_sets():
        rc = RM.train(argv, split)
    name, out_root = S2.where(argv)
    ZF.stamp(out_root / name / "screen.json",
             {"zgnn_sha256": LC.sha_src(__file__), "zfeat_sha256": LC.sha_src(ZF.__file__),
              "zlink_sha256": LC.sha_src(ZL.__file__), "zgnn": SETTINGS, "zfeat": ZF.SETTINGS,
              "zfeat_records": ZF.built_records(), "zlink_records": ZL.built_records(), "message_passing": True})
    return rc


def work_copy(source, dest):
    """dest as a copy of fit source's files (no reads), for the every-candidate-masked read; refused when dest holds
    another fit."""
    source, dest = Path(source), Path(dest)
    if not (source / "models.pt").exists():
        raise SystemExit(f"zgnn: {source} holds no fit")
    files = [f for f in source.iterdir() if f.is_file()]
    if (dest / "models.pt").exists():
        if LC.sha_file(dest / "models.pt") != LC.sha_file(source / "models.pt"):
            raise SystemExit(f"zgnn: {dest} holds another fit than {source}")
    dest.mkdir(parents=True, exist_ok=True)
    for f in files:
        tmp = dest / (f.name + ".tmp")
        shutil.copyfile(f, tmp)
        os.replace(tmp, dest / f.name)
    LC.write_json(dest / "work_copy.json", {"source": source.name, "files": sorted(f.name for f in files),
                                            "models_sha256": LC.sha_file(dest / "models.pt"),
                                            "read": "every candidate masked", "zgnn_sha256": LC.sha_src(__file__)})


def read(argv):
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--select")
    ap.add_argument("--keep", choices=("none",))
    ap.add_argument("--source")
    k, rest = ap.parse_known_args(argv)
    name, out_root = S2.where(rest)
    if (k.select is None) == (k.keep is None):
        raise SystemExit("zgnn read: --select FILE (the common subset) or --keep none (every candidate masked)")
    if k.keep == "none":
        if not k.source:
            raise SystemExit("zgnn read: --keep none reads a work copy of --source")
        work_copy(ZF.ROOT / k.source if not Path(k.source).is_absolute() else k.source, out_root / name)
        kept, sha = (), None
    else:
        kept, sha = ZF.load_selection(k.select)
    arm = json.loads((out_root / name / "screen.json").read_text(encoding="utf-8"))["arm"]
    if arm != ARM:
        raise SystemExit(f"zgnn: {name} was trained as {arm}, not {ARM}")
    with ZF.with_sets(), ZF.read_keep(kept):
        rc = RM.read(rest)
    LC.write_json(out_root / name / "reads" / "zgnn_read.json",
                  {"kept": list(kept), "masked": [b for b in ZF.CAND if b not in kept], "selection_sha256": sha,
                   "zgnn_sha256": LC.sha_src(__file__), "zfeat_sha256": LC.sha_src(ZF.__file__)})
    return rc


def compare(rest):
    return S2.main(["compare"] + rest)


# ── relz.py's pair and re-call, decided against the same fit with every candidate masked ──


NONE_FITS = {"L-musique": ("screen", "fits", "scr-zgn-none"), "L-hotpotqa": ("screen", "fits", "scr-zgn-none-hp")}
NONE_SCREENS = {sp: f"outputs/zgnn/base-none-{sp}.json" for sp in NONE_FITS}
DECIDED = "zgn's own fit of each split, every candidate masked"


@contextlib.contextmanager
def on_none():
    saved = Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit
    Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS = ARM, "zgn-none", NONE_FITS, NONE_SCREENS
    Z.rel_fit = lambda sp: NONE_FITS[sp]
    try:
        yield
    finally:
        Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit = saved


FIX = (("(rel's screen fits scr-rel and scr-rel-hp)", "(zgn's fits with every candidate masked, scr-zgn-none and "
                                                     "scr-zgn-none-hp)"),
       ("(rel's screen fits)", "(zgn's fits with every candidate masked)"),
       ("| rel R@5 | relz R@5 |", "| zgn-none R@5 | zgn R@5 |"),
       ("section 2 and the tenth round", "section 2 and the twenty-fifth round"))


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
        rec.update({"decided_against": DECIDED, "round": "twenty-fifth", "zgnn_sha256": LC.sha_src(__file__),
                    "message_passing": True})
        LC.write_json(js, rec)


def pair(screens, out):
    with on_none():
        rec = Z.pair(screens, out)
    restamp(out)
    return rec


def recall(null_files, pair_file, out):
    with on_none():
        rec = Z.recall(null_files, pair_file, out)
    restamp(out)
    return rec


# ── selftest ─────────────────────────────────────────────────────────────────


def selftest():
    t0 = time.time()
    torch.manual_seed(0)
    # the aggregation: per family means over in-neighbours, 0 without any
    h = torch.tensor([[1.0, 0.0], [3.0, 2.0], [5.0, 4.0], [0.0, 8.0]])
    lk = {"u": torch.tensor([1, 2, 0, 3]), "v": torch.tensor([0, 0, 1, 0]), "f": torch.tensor([0, 0, 2, 1])}
    a = family_means(h, lk, 4)
    assert torch.equal(a[0], torch.tensor([4.0, 3.0, 0.0, 8.0, 0.0, 0.0]))       # rows 1, 2 in family 0; row 3 in 1
    assert torch.equal(a[1], torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 0.0])) and float(a[2:].abs().sum()) == 0
    assert S.ARMS[ARM] == (ZGNN, GNNCarve) and issubclass(GNNCarve, ZF.FeatChainCarve)
    assert sys.modules["zfeat"] is ZF
    # a toy: rmatch's toy chain carve with two candidates and a random pool graph beside it
    import tempfile
    LG.bind_device_ops()
    tmp = Path(tempfile.mkdtemp(prefix="zgnn_"))
    try:
        rng = np.random.default_rng(25)
        RM.toy_roots(tmp, rng, chunks=(12, 12))
        RM.build("metaqa", "toy", out_root=tmp / "ch", cache_root=tmp / "cache", look_root=tmp / "look",
                 rel_dir=tmp / "rel")
        base = RM.chain_carve(ZM.ToyZ, tmp / "ch", tmp / "rel")("metaqa", "toy", "2wiki", "cpu", tmp / "cache")

        class Toy:
            def __init__(self, c):
                self.c, self.ds, self.carve, self.chains, self.device = c, c.ds, c.carve, c.chains, c.device
                self.rows, self.n_np, self.off_np = c.rows, c.n_np, c.off_np
                gb = c.gold.cpu().numpy().astype(np.int64)
                self.gt_np = np.array([gb[a_:e].sum() for a_, e in zip(c.off_np[:-1], c.off_np[1:])], np.int64)
                self.widths = dict(c.widths, topo_NER=11, seedcond=6)
                g = torch.Generator().manual_seed(5)
                gold = c.gold.to(torch.float32).unsqueeze(1)
                self.X2 = torch.cat([torch.randn(gold.shape[0], 11, generator=g) + 1.0 * gold,
                                     torch.randn(gold.shape[0], 6, generator=g)], 1)
                # each pool's edges: golds linked to golds (family 0), random pairs (families 1 and 2)
                e = []
                r = np.random.default_rng(7)
                for q in range(self.rows):
                    a_, n = int(c.off_np[q]), int(c.n_np[q])
                    gl = np.flatnonzero(gb[a_:a_ + n])
                    for i in gl:
                        for j in gl:
                            if i != j:
                                e.append((q, i, j, 0))
                    for _k in range(2 * n):
                        i, j = r.integers(0, n, 2)
                        if i != j:
                            ff = int(r.integers(1, FAMS))
                            e += [(q, i, j, ff), (q, j, i, ff)]
                self.E = np.asarray(sorted(set(e)), np.int64).reshape(-1, 4)

            def nbytes(self):
                return 0

            def batch(self, qs, blocks):
                feats, nq, bz, gold = self.c.batch(qs, [b for b in blocks if b not in ZF.CAND])
                qs = np.asarray(qs, np.int64)
                cnt = self.n_np[qs]
                seg = np.cumsum(cnt) - cnt
                idx = torch.from_numpy(np.repeat(self.off_np[qs] - seg, cnt) + np.arange(int(cnt.sum())))
                if "topo_NER" in blocks:
                    feats["topo_NER"] = self.X2[idx, :11].contiguous()
                if "seedcond" in blocks:
                    feats["seedcond"] = self.X2[idx, 11:].contiguous()
                pos = {int(q): k for k, q in enumerate(qs)}
                sel = np.isin(self.E[:, 0], qs)
                E = self.E[sel]
                at = np.asarray([seg[pos[int(q)]] for q in E[:, 0]], np.int64)
                feats[ZL.LK_KEY] = {"u": torch.from_numpy(at + E[:, 1]), "v": torch.from_numpy(at + E[:, 2]),
                                    "f": torch.from_numpy(E[:, 3])}
                return feats, nq, bz, gold

        tc = Toy(base)
        tb = ["rank", "SEMB", "topo_NER", "seedcond"]
        # at the start zgn is zfs: the same weights drawn, a zero update
        torch.manual_seed(3)
        zf = ZF.ZFS(tb, tc.widths, 16, dropout=0.1, seed=0)
        torch.manual_seed(3)
        zg = ZGNN(tb, tc.widths, 16, dropout=0.1, seed=0)
        sf, sg = zf.state_dict(), zg.state_dict()
        assert set(sg) - set(sf) == set(MP) and all(torch.equal(sf[k], sg[k]) for k in sf)
        zf.eval()
        zg.eval()
        qs = np.arange(min(tc.rows, 8))
        feats, nq, bz, _g = tc.batch(qs, tb)
        one = torch.ones(qs.size, len(tb))
        assert torch.equal(zf(feats, one, nq, qs.size, bz), zg(feats, one, nq, qs.size, bz))
        # a carve without typed relations (squad's, musique's, hotpotqa's, 2wiki's) holds no chains: zgn still runs,
        # and at the start it is zfs there too
        plain = {k: v for k, v in feats.items() if k != RM.CH_KEY}
        assert torch.equal(zf(plain, one, nq, qs.size, bz), zg(plain, one, nq, qs.size, bz))
        try:
            zg({k: v for k, v in feats.items() if k != ZL.LK_KEY}, one, nq, qs.size, bz)
            raise AssertionError("zgn without its pool edges")
        except SystemExit:
            pass
        cfg = {"lr": 2e-2, "wd": 1e-4, "dropout": 0.1, "epochs": 3, "swa_from": 1, "cos": False, "adamw": False,
               "drop": False}
        with S.patched(ARM):
            f1 = LG.fit_variant([tc], tb, cfg, 0, 16, "none", "cpu", tag="toy-zgn")
        assert all(bool(torch.isfinite(v).all()) for v in f1["swa"].values() if v.is_floating_point())
        assert float(f1["swa"]["mp_w"].abs().sum()) > 0                       # the message passing trained
        net = ZGNN(tb, tc.widths, 16, dropout=0.1, seed=0)
        net.load_state_dict(f1["swa"])
        net.eval()
        got = ZF.subset_reads(tc, net, tb, [(), ("topo_NER",), ("topo_NER", "seedcond")], rows_cap=4096)
        assert all(len(v) == 3 and all(np.isfinite(v)) for v in got.values()), got
        with ZF.read_keep(()):
            assert torch.equal(net(feats, one, nq, qs.size, bz), net(feats, net.masked(one, set()), nq, qs.size, bz))
        # the joint selection's record reads with zfeat's loader; the work copy refuses another fit
        src, dst = tmp / "fit", tmp / "copy"
        src.mkdir()
        (src / "models.pt").write_bytes(b"a")
        (src / "screen.json").write_text('{"arm": "zgn"}', encoding="utf-8")
        work_copy(src, dst)
        assert (dst / "models.pt").read_bytes() == b"a" and (dst / "work_copy.json").exists()
        (src / "models.pt").write_bytes(b"b")
        try:
            work_copy(src, dst)
            raise AssertionError("a work copy over another fit")
        except SystemExit:
            pass
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    n_mp = LAYERS * (3 * 128 * 128 + 128)
    print(f"selftest: per-family means, zgn = zfs at the start (same draws, zero update), a toy fit trains the message "
          f"passing, subset reads and the masked read work, the work copy guards its fit; {n_mp} message-passing "
          f"weights at H = 128 ({time.time() - t0:.1f}s)")
    return 0


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
        gp.add_argument("--out", required=True)
        g = gp.parse_args(rest)
        try:
            if k.cmd == "pair":
                pair([x for x in g.screens.split(",") if x], g.out)
            else:
                recall([x for x in g.null.split(",") if x], g.pair, g.out)
        except SystemExit as e:
            log(f"zgnn {k.cmd}: {e}")
            return 2
        return 0
    ap.error("zgnn: train, read, compare, pair or recall")


if __name__ == "__main__":
    sys.exit(main())
