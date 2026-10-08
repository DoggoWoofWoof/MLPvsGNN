"""Screens, twenty-third round (docs/SCREENS.md; full run docs/FULL_ROUND23.md): each row's neighbours' scores, on zrm,
the base, trained and read on the host's CPU against zrm trained and read on the CPU.

zsp is zrm (zrm.ZRM over rmatch.ChainCarveBase) with a small head added to its score. zrm scores the pool; each row then
reads its neighbours' scores in the question's own pool graph, per edge family: the mean of their z-scores, a soft
maximum of them (the log of the sum of their exponentials, each z-score clipped to [-8, 8]) and the row's degree; those
nine numbers and the row's own z-score go through a two-layer head whose output is added to zrm's score. Why
(docs/DIAG_BRIDGE.md): the golds zrm misses in partly-found questions sit next to the rows it ranks highly (D1), and a
stronger scorer on the per-row inputs gains nothing (D2, FEATURE_LIMIT). Round twenty-two (zlink.py) gives each row its
edges to zrm's top five and top row; this round gives each row every neighbour's score, so a row next to a gold zrm
ranks sixth or tenth is lifted too, and how far is learned from the score, not fixed by a cut at five.

The edges are zlink's (outputs/zlink/cache, built for round twenty-two): each question's own pool graph, undirected,
once per family, no self-loops; the KB's relation edges on metaqa and webqsp. The same rule on all six datasets; no new
graph, column, encoder or text, and no new build. The inputs come from zrm's own scores under no_grad, so they are
query-local and label-free, the same at training and at read.

The head's inputs (10 per row): per family, the mean of the neighbours' z-scores (3), their soft maximum (3) and
log1p of the degree (3), 0 for a row with no neighbour in the family; and the row's own z-score (1). Head: Linear(10, 32),
GELU, Linear(32, 1); the last layer starts at zero and the first is drawn from its own generator (seed + 2301), so at the
start zsp's forward is zrm's bit for bit and no draw of zrm's moves. Settings and training are zrm's (rmatch.py's
train); the head's 385 parameters train with the rest.

The CPU. Round twenty-three runs on the host's CPU, beside the card's queue: zsp's two screen fits and zrm's two
refits on the same splits (`base`), each trained and read on the CPU. Each read is decided against zrm's CPU fit of its
split, so the device is the same on both sides; zrm's card fits are reported beside, and zrm's CPU refits compared with
its card fits are the device's own spread, reported only.

    python outputs/mp_unified/zprop.py check --dataset hotpotqa --carve s1eval --threads 2 --host
    python outputs/mp_unified/zprop.py base --split L-musique --name scr-zrm-cpu --threads 6 --host
    python outputs/mp_unified/zrm.py read --name scr-zrm-cpu --device cpu --threads 6 --host
    python outputs/mp_unified/zprop.py compare --new outputs/screen/fits/scr-zrm-cpu \\
        --base outputs/step1/fits/L-musique,outputs/screen/fits/scr-zrm --out outputs/zprop/base-zrm-cpu-L-musique
    python outputs/mp_unified/zprop.py train --split L-musique --name scr-zsp --arm zsp --device cpu --threads 6 --host
    python outputs/mp_unified/zprop.py read --name scr-zsp --device cpu --threads 6 --host
    python outputs/mp_unified/zprop.py compare --new outputs/screen/fits/scr-zsp \\
        --base outputs/screen/fits/scr-zrm-cpu,outputs/screen/fits/scr-zrm,outputs/step1/fits/L-musique \\
        --out outputs/screen/scr-zsp
    python outputs/mp_unified/zprop.py pair --screens outputs/screen/scr-zsp.json,outputs/screen/scr-zsp-hp.json \\
        --out outputs/screen/scr-zsp-pair
    python outputs/mp_unified/zprop.py recall --null N1,N2,N3,N4 --pair outputs/screen/scr-zsp-pair.json \\
        --out outputs/screen/scr-zsp-pair-recall
    python outputs/mp_unified/zprop.py gate --recall outputs/screen/scr-zsp-pair-recall.json \\
        --base outputs/zbase/grade-nullx.json
    python outputs/mp_unified/zprop.py grade --full-root outputs/full_zsp \\
        --reuse L-musique=outputs/screen/scr-zsp.json,L-hotpotqa=outputs/screen/scr-zsp-hp.json
    python outputs/mp_unified/zprop.py --selftest
N1..N4 are the seed null's comparisons (outputs/screen/scr-null-s1, -s2, -s1-hp, -s2-hp .json).

Speed: per question, one pass over its pool's edges after zrm's forward (about 4 to 10 a row). Any latency figure for
zsp is cold (8216ffe): each question timed from scratch, the walk, the move of its entries and edges to the device and
the forward, with no warm-up pass and nothing kept from an earlier question.
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
Z, R = ZL.Z, ZL.R
S2, S = ZL.S2, ZL.S
LG, LC = ZL.LG, ZL.LC
SR = ZL.SR
log = S.log

ARM, BASE_ARM = "zsp", "zrm"
LK_KEY, FAMS = ZL.LK_KEY, ZL.FAMS
N_IN, HID, SEED_OFF, ZCLIP = 3 * FAMS + 1, 32, 2301, 8.0
SP = ("sp_w1", "sp_b1", "sp_w2", "sp_b2")
DEVICE = "cpu"
SETTINGS = {"families": FAMS, "inputs": N_IN, "hidden": HID, "seed_offset": SEED_OFF, "z_clip": ZCLIP,
            "edges": "zlink's pool edges (outputs/zlink/cache), undirected, once per family, no self-loops",
            "device": DEVICE}
# zrm's CPU fits: its screen refits on L-musique and L-hotpotqa, the full run's own on every other split
CPU_FITS = {"L-musique": ("screen", "fits", "scr-zrm-cpu"), "L-hotpotqa": ("screen", "fits", "scr-zrm-cpu-hp")}
CPU_SCREENS = {sp: f"outputs/zprop/base-zrm-cpu-{sp}.json" for sp in CPU_FITS}


def zrm_cpu_fit(split):
    return CPU_FITS.get(split, ("full_zsp", "zrm", split))


# ── the model ────────────────────────────────────────────────────────────────


@torch.no_grad()
def prop_inputs(s, lk, nq, B):
    """(N, 10): per family the mean and soft maximum of each row's neighbours' z-scores and log1p of its degree, and the
    row's own z-score (see the module's docstring)."""
    N = nq.numel()
    dt = s.dtype
    z = LG.seg_zscore8D(s.detach().unsqueeze(1), nq, B).squeeze(1)
    zc = z.clamp(-ZCLIP, ZCLIP)
    u, v, f = lk["u"], lk["v"], lk["f"]
    at = v * FAMS + f
    deg = torch.zeros(N * FAMS, dtype=dt, device=s.device).index_add_(0, at, torch.ones_like(zc[u])).view(N, FAMS)
    sm = torch.zeros(N * FAMS, dtype=dt, device=s.device).index_add_(0, at, zc[u]).view(N, FAMS)
    ex = torch.zeros(N * FAMS, dtype=dt, device=s.device).index_add_(0, at, torch.exp(zc[u])).view(N, FAMS)
    has = deg > 0
    zero = torch.zeros_like(deg)
    mean = torch.where(has, sm / deg.clamp_min(1.0), zero)
    soft = torch.where(has, torch.log(ex.clamp_min(1e-30)), zero)
    return torch.cat([mean, soft, torch.log1p(deg), z.unsqueeze(1)], 1)


class ZProp(ZM.ZRM):
    """zsp: zrm's score plus a head (zero at the start) over each row's neighbours' scores."""

    def __init__(self, blocks, widths, hidden=128, dropout=0.1, seed=0, arm="ctl", ctx="none"):
        super().__init__(blocks, widths, hidden, dropout, seed, arm, ctx)
        g = torch.Generator().manual_seed(int(seed) + SEED_OFF)
        self.sp_w1 = nn.Parameter((torch.rand(HID, N_IN, generator=g) * 2 - 1) / N_IN ** 0.5)
        self.sp_b1 = nn.Parameter(torch.zeros(HID))
        self.sp_w2 = nn.Parameter(torch.zeros(1, HID))
        self.sp_b2 = nn.Parameter(torch.zeros(1))

    def prop_head(self, x):
        return Fn.linear(Fn.gelu(Fn.linear(x, self.sp_w1, self.sp_b1)), self.sp_w2, self.sp_b2).squeeze(-1)

    def forward(self, feats, keep, nq, B, base_z):
        s = super().forward(feats, keep, nq, B, base_z)
        if LK_KEY not in feats:
            return s
        return s + self.prop_head(prop_inputs(s, feats[LK_KEY], nq, B))


S.ARMS.update({ARM: (ZProp, ZL.LinkCarveBase)})


# ── train, read and compare, on the CPU ──────────────────────────────────────


def device_of(argv):
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=2)
    k, _ = ap.parse_known_args(argv)
    return k.device, k.threads


def stamp(argv, key, body):
    name, out_root = S2.where(argv)
    sj = out_root / name / "screen.json"
    if sj.exists():
        rec = json.loads(sj.read_text(encoding="utf-8"))
        rec.update({"zprop_sha256": LC.sha_src(__file__), key: body})
        LC.write_json(sj, rec)


def train(argv, split):
    """rmatch.py's train (zrm's settings) for the arm zsp, on the CPU; this file's records stamped."""
    arm = R.arm_of(argv)
    if arm != ARM:
        raise SystemExit(f"zprop: train takes --arm {ARM}, not {arm}")
    if S.ARMS.get(ARM) != (ZProp, ZL.LinkCarveBase) or S.ARMS.get(BASE_ARM) != (ZM.ZRM, RM.ChainCarveBase):
        raise SystemExit(f"zprop: the arm {ARM} is {S.ARMS.get(ARM)}, not zsp's model on zlink's carve")
    dev, threads = device_of(argv)
    if dev != DEVICE:
        raise SystemExit(f"zprop: round twenty-three trains on the {DEVICE}, not {dev}")
    rc = RM.train(argv, split)
    stamp(argv, "zprop", {"model": "zprop.ZProp (zrm.ZRM plus a head over each row's neighbours' scores)",
                          "carve": "zlink.LinkCarveBase over rmatch.ChainCarveBase", "head": list(SP),
                          "settings": SETTINGS, "threads": threads, "zlink_records": ZL.built_records()})
    return rc


def base(argv, split):
    """zrm's train (zrm.py's), on the CPU: zrm's fit of split for this round's comparisons."""
    argv = list(argv)
    if "--arm" not in argv:
        argv += ["--arm", BASE_ARM]
    if R.arm_of(argv) != BASE_ARM:
        raise SystemExit(f"zprop: base trains {BASE_ARM} only, not {R.arm_of(argv)}")
    dev, threads = device_of(argv)
    if dev != DEVICE:
        raise SystemExit(f"zprop: zrm's base fits for round twenty-three train on the {DEVICE}, not {dev}")
    if "--device" not in argv:
        argv += ["--device", DEVICE]
    rc = ZM.train(argv, split)
    stamp(argv, "zprop_base", {"arm": BASE_ARM, "device": DEVICE, "threads": threads,
                               "for": "round twenty-three's comparisons (docs/FULL_ROUND23.md)"})
    return rc


def read(argv):
    name, out_root = S2.where(argv)
    arm = json.loads((out_root / name / "screen.json").read_text(encoding="utf-8"))["arm"]
    if arm != ARM:
        raise SystemExit(f"zprop: {name} was trained as {arm}, not {ARM}")
    dev, _t = device_of(argv)
    if dev != DEVICE:
        raise SystemExit(f"zprop: round twenty-three reads on the {DEVICE}, not {dev}")
    return RM.read(argv)


def compare(rest):
    return S2.main(["compare"] + rest)


def on_cpu(compare_files, root=None):
    """Each comparison's new fit is zsp's trained on the CPU and its deciding base zrm's CPU fit (zprop's stamps), or
    stop."""
    root = Path(root) if root else S2.ROOT
    for fn in compare_files:
        rec = json.loads(Path(fn).read_text(encoding="utf-8"))
        for side, path, key, arm in (("new", rec["new"], "zprop", ARM), ("base", rec["bases"][0], "zprop_base",
                                                                         BASE_ARM)):
            parts = S2.parts_of(path)
            if Path(path).is_absolute() and Path(path).exists():
                p = Path(path)
            else:
                p = root / Path(*(parts[parts.index("outputs"):] if "outputs" in parts else parts))
            sj = p / "screen.json"
            if not sj.exists():
                raise SystemExit(f"zprop: {fn}'s {side} fit {sj} missing")
            s = json.loads(sj.read_text(encoding="utf-8"))
            dev = (s.get(key) or {}).get("device") if key == "zprop_base" else (s.get(key) or {}).get(
                "settings", {}).get("device")
            if s.get("arm") != arm or dev != DEVICE:
                raise SystemExit(f"zprop: {fn}'s {side} fit is arm {s.get('arm')} on {dev}, not {arm} on the {DEVICE}")
    return True


# ── a real carve: at the start, zsp is zrm ───────────────────────────────────


def check(ds, carve, device="cpu", n_q=256, out=None):
    """On a built carve: zlink's edges load under every check, and from the same seed zsp's forward equals zrm's bit for
    bit on its first n_q questions (eval, and train with the same dropout draws). Reported beside, not decided: the mean
    soft maximum of the neighbours' z-scores of gold and non-gold rows, under the untrained score."""
    t0 = time.time()
    LG.bind_device_ops()
    c = ZL.LinkCarveBase(ds, carve, "2wiki", device)
    blocks = list(LG.SETS["pick"])
    qs = np.arange(min(n_q, c.rows), dtype=np.int64)
    feats, nq, bz, gold = c.batch(qs, blocks)
    B = qs.size
    keep = torch.ones(B, len(blocks), device=c.device)
    nets = {}
    for k, cls in ((BASE_ARM, ZM.ZRM), (ARM, ZProp)):
        torch.manual_seed(0)
        nets[k] = cls(blocks, c.widths, LG.HIDDEN, dropout=0.1, seed=0).to(c.device)
    rows = {}
    with torch.no_grad():
        for mode in (False, True):
            got = {}
            for k, m in nets.items():
                m.train(mode)
                torch.manual_seed(5)
                got[k] = m(feats, keep, nq, B, bz)
            rows["train" if mode else "eval"] = "IDENTICAL" if torch.equal(got[ARM], got[BASE_ARM]) else "DIFFERENT"
        nets[BASE_ARM].eval()
        s = nets[BASE_ARM](feats, keep, nq, B, bz)
        x = prop_inputs(s, feats[LK_KEY], nq, B)
    g = gold.bool()
    soft = x[:, FAMS:2 * FAMS].max(1).values
    share = {"gold": float(soft[g].mean()) if bool(g.any()) else None,
             "non_gold": float(soft[~g].mean()) if bool((~g).any()) else None}
    rec = {"dataset": ds, "carve": carve, "questions": int(B), "rows": int(nq.numel()),
           "edges": int(feats[LK_KEY]["u"].numel()), "identity": rows, "mean_soft_max": share,
           "finite": bool(torch.isfinite(x).all()), "script_sha256": LC.sha_src(__file__),
           "seconds": round(time.time() - t0, 1)}
    ok = all(v == "IDENTICAL" for v in rows.values()) and rec["finite"]
    log(f"zprop check {ds}/{carve}: {B} questions, {rec['edges']} edges; zsp at the start against zrm {rows}; inputs "
        f"finite {rec['finite']}; the neighbours' soft maximum, gold {share['gold']}, non-gold {share['non_gold']} "
        f"({rec['seconds']}s) -> {'ok' if ok else 'FAIL'}")
    if out:
        R.write_json(Path(out), rec)
    return 0 if ok else 1


# ── relz.py's pair, re-call and grade, decided against zrm's CPU fits ────────


@contextlib.contextmanager
def on_zrm_cpu():
    """relz.py's records under zsp's name, each read decided against zrm's CPU fit of its split; relz restored after."""
    saved = Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit
    Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit = ARM, BASE_ARM, CPU_FITS, CPU_SCREENS, zrm_cpu_fit
    try:
        yield
    finally:
        Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit = saved


FIX = (("(rel's screen fits scr-rel and scr-rel-hp)", "(zrm's CPU fits scr-zrm-cpu and scr-zrm-cpu-hp)"),
       ("(rel's screen fits)", "(zrm's CPU fits)"),
       ("| rel R@5 | relz R@5 |", "| zrm (CPU) R@5 | zsp R@5 |"),
       ("section 2 and the tenth round", "section 2 and the twenty-third round"),
       ("docs/FULL_ROUND10.md", "docs/FULL_ROUND23.md"))
AGAINST = "zrm's CPU fit of each split"


def restamp(out):
    """A record relz.py wrote under zsp's name: zrm's CPU fits named as the base in its md, this round and this file's
    sha added."""
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
        rec.update({"decided_against": AGAINST, "round": "twenty-third", "device": DEVICE,
                    "zprop_sha256": LC.sha_src(__file__)})
        LC.write_json(js, rec)


def pair(screens, out, check_device=True):
    if check_device:
        on_cpu(screens)
    with on_zrm_cpu():
        rec = Z.pair(screens, out)
    restamp(out)
    return rec


def recall(null_files, pair_file, out, cpu_screens=None):
    with on_zrm_cpu():
        rec = Z.recall(null_files, pair_file, out, cpu_screens)
    restamp(out)
    return rec


def grade(full_root, reuse, out, check_fits=True):
    if check_fits:
        on_cpu([reuse.get(sp) or Path(full_root) / f"compare-{sp}.json" for sp in S2.SPLITS])
    with on_zrm_cpu():
        rec = Z.grade(full_root, reuse, out, check_fits)
    restamp(out)
    return rec


def gate(recall_file, base_file):
    """0 when the screen's re-call is PROMISING and zrm is the base (docs/BASE_ZRM_ZRS.md's re-grade ADOPT), 1
    otherwise; a missing file, or one that is not the declared record, stops (2)."""
    got = []
    for fn, arm, want in ((recall_file, ARM, AGAINST), (base_file, "zrm", "zrs's fit of each split")):
        p = Path(fn)
        if not p.exists():
            raise SystemExit(f"zprop gate: {p} missing")
        rec = json.loads(p.read_text(encoding="utf-8"))
        if rec.get("arm") != arm or rec.get("decided_against") != want:
            raise SystemExit(f"zprop gate: {p} is arm {rec.get('arm')}'s, decided against "
                             f"{rec.get('decided_against')!r}; not the declared record")
        got.append(rec.get("verdict"))
    if "null_splits" not in json.loads(Path(base_file).read_text(encoding="utf-8")):
        raise SystemExit(f"zprop gate: {base_file} is not a re-grade under the null")
    ok = got[0] == "PROMISING" and got[1] == "ADOPT"
    log(f"zprop gate: the re-call {got[0]}, zrm against zrs {got[1]} -> {'the full run starts' if ok else 'no full run'}")
    return 0 if ok else 1


# ── selftest ─────────────────────────────────────────────────────────────────


def hand_inputs(s, n_np, qs, links_of):
    """prop_inputs by hand: per question, python loops over its own edge list."""
    out = []
    at = 0
    for q in qs:
        n = int(n_np[q])
        sc = [float(x) for x in s[at:at + n]]
        mu = sum(sc) / n
        sd = (sum((x - mu) ** 2 for x in sc) / n) ** 0.5
        z = [(x - mu) / sd if sd >= 1e-6 else 0.0 for x in sc]
        zc = [min(max(x, -ZCLIP), ZCLIP) for x in z]
        nb = [[[] for _ in range(FAMS)] for _ in range(n)]
        for a, b, f in links_of(q):
            nb[b][f].append(zc[a])
        for i in range(n):
            mean = [sum(t) / len(t) if t else 0.0 for t in nb[i]]
            soft = [float(np.log(sum(np.exp(x) for x in t))) if t else 0.0 for t in nb[i]]
            out.append(mean + soft + [float(np.log1p(len(t))) for t in nb[i]] + [z[i]])
        at += n
    return np.asarray(out, np.float64)


def selftest():
    t0 = time.time()
    LG.bind_device_ops()
    # 1. the arm: zrm's model plus the head, on zlink's carve; zrm's and zlk's arms unchanged; train takes no other arm
    #    and no other device; base trains zrm only, on the CPU
    assert S.ARMS[ARM] == (ZProp, ZL.LinkCarveBase) and S.ARMS[BASE_ARM] == (ZM.ZRM, RM.ChainCarveBase)
    assert S.ARMS[ZL.ARM] == (ZL.ZLink, ZL.LinkCarveBase) and ZProp.__mro__[1] is ZM.ZRM
    SR.must_stop(train, ["train", "--arm", BASE_ARM, "--name", "x"], "L-musique")
    SR.must_stop(train, ["train", "--name", "x"], "L-musique")
    SR.must_stop(train, ["train", "--arm", ARM, "--name", "x", "--device", "cuda"], "L-musique")
    SR.must_stop(base, ["train", "--arm", ARM, "--name", "x"], "L-musique")
    SR.must_stop(base, ["train", "--name", "x", "--device", "cuda"], "L-musique")
    assert zrm_cpu_fit("J5") == ("full_zsp", "zrm", "J5") and zrm_cpu_fit("L-hotpotqa") == CPU_FITS["L-hotpotqa"]
    tmp = Path(tempfile.mkdtemp(prefix="zprop_"))
    try:
        rng = np.random.default_rng(23)
        qq = RM.toy_roots(tmp, rng, chunks=(12, 12))
        RM.build("metaqa", "toy", out_root=tmp / "ch", cache_root=tmp / "cache", look_root=tmp / "look",
                 rel_dir=tmp / "rel")
        ZL.build("metaqa", "toy", out_root=tmp / "lk", cache_root=tmp / "cache", look_root=tmp / "look")
        cls = ZL.link_carve(RM.chain_carve(ZM.ToyZ, tmp / "ch", tmp / "rel"), tmp / "lk")
        c = cls("metaqa", "toy", "2wiki", "cpu", tmp / "cache")

        def links_of(q):
            t = qq[q]
            e = {(int(x), int(y), int(f)) for x, y, f in zip(t["e_u"], t["e_v"], t["e_fam"]) if x != y}
            return sorted(e | {(y, x, f) for x, y, f in e})

        blocks = ["rank", "SEMB"]
        qs = np.r_[np.arange(c.rows)[::-1][:5], np.arange(c.rows)[:4]].astype(np.int64)
        B = qs.size
        feats, nq, bz, _g = c.batch(qs, blocks)
        cnt = c.n_np[qs]
        lk = feats[LK_KEY]
        # 2. from the same seed zsp is zrm's state plus the head, the last layer zero, and its forward is zrm's bit for
        #    bit, in eval and with the same dropout draws in training; the global generator is where zrm leaves it
        keep = torch.ones(B, len(blocks))
        nets, after = {}, {}
        for k, cl in ((BASE_ARM, ZM.ZRM), (ARM, ZProp)):
            torch.manual_seed(0)
            nets[k] = cl(blocks, c.widths, 16, dropout=0.1, seed=0)
            after[k] = torch.rand(4)
        assert torch.equal(after[ARM], after[BASE_ARM])
        sz, sk = nets[BASE_ARM].state_dict(), nets[ARM].state_dict()
        assert sorted(set(sk) - set(sz)) == sorted(SP) and all(torch.equal(sz[k], sk[k]) for k in sz)
        assert sum(int(sk[k].numel()) for k in SP) == HID * N_IN + HID + HID + 1 == 385
        assert float(sk["sp_w2"].abs().sum()) == 0.0 and float(sk["sp_b2"].abs().sum()) == 0.0
        with torch.no_grad():
            for mode in (False, True):
                out = {}
                for k, m in nets.items():
                    m.train(mode)
                    torch.manual_seed(5)
                    out[k] = m(feats, keep, nq, B, bz)
                assert torch.equal(out[ARM], out[BASE_ARM]), mode
            # 3. the inputs are prop_inputs by hand (mean, soft maximum and degree per family, own z-score); a row with
            #    no neighbour in a family reads 0 there; with the head set, zsp is zrm plus the head on them; a batch
            #    without edges is zrm's own
            nz, nk = nets[BASE_ARM].eval(), nets[ARM].eval()
            s = nz(feats, keep, nq, B, bz)
            x = prop_inputs(s, lk, nq, B)
            xh = hand_inputs(s.numpy(), c.n_np, qs, links_of)
            assert x.shape == (int(cnt.sum()), N_IN) and np.allclose(x.numpy(), xh, atol=1e-5), \
                np.abs(x.numpy() - xh).max()
            assert bool(torch.isfinite(x).all()) and float(x[:, 2 * FAMS:3 * FAMS].sum()) > 0
            assert bool((x[:, 2 * FAMS:3 * FAMS] == 0).any())
            big = s * 1e4
            xb = prop_inputs(big, lk, nq, B)
            assert bool(torch.isfinite(xb).all()) and float(xb[:, FAMS:2 * FAMS].max()) <= ZCLIP + np.log(64) + 1e-4
            gen = torch.Generator().manual_seed(3)
            for k in SP:
                getattr(nk, k).copy_(torch.randn(getattr(nk, k).shape, generator=gen))
            hd = Fn.gelu(x @ nk.sp_w1.T + nk.sp_b1) @ nk.sp_w2.T + nk.sp_b2
            assert torch.equal(nk(feats, keep, nq, B, bz), s + hd.squeeze(-1))
            f2 = {k: v for k, v in feats.items() if k != LK_KEY}
            assert torch.equal(nk(f2, keep, nq, B, bz), nz(f2, keep, nq, B, bz))
        # 4. toy fits through lean_gpu's loop: the head moves, the fit leaves zrm's, and a repeat is identical
        cfg = {"lr": 2e-2, "wd": 1e-4, "dropout": 0.1, "epochs": 3, "swa_from": 1, "cos": False, "adamw": False,
               "drop": False}
        every = lambda f: f["states"] + [f["swa"]]  # noqa: E731
        fits = {}
        for arm in (BASE_ARM, ARM):
            with S.patched(arm):
                fits[arm] = LG.fit_variant([c], blocks, cfg, 0, 16, "none", "cpu", tag=f"toy-{arm}")
        with S.patched(ARM):
            rep = LG.fit_variant([c], blocks, cfg, 0, 16, "none", "cpu", tag="toy-zsp repeat")
        sw = fits[ARM]["swa"]
        assert float(sw["sp_w2"].abs().sum()) > 0
        assert any(not torch.equal(sw[k], fits[BASE_ARM]["swa"][k]) for k in fits[BASE_ARM]["swa"])
        assert all(not LG.same_state(a_, b_) for a_, b_ in zip(every(fits[ARM]), every(rep)))
        # 5. read refuses another arm's fit and another device
        (tmp / "fits" / "x").mkdir(parents=True)
        LC.write_json(tmp / "fits" / "x" / "screen.json", {"arm": BASE_ARM})
        try:
            read(["--name", "x", "--out-root", str(tmp / "fits")])
            raise AssertionError("zprop read zrm's fit")
        except SystemExit as e:
            assert "trained as zrm" in str(e)
        LC.write_json(tmp / "fits" / "x" / "screen.json", {"arm": ARM})
        SR.must_stop(read, ["--name", "x", "--out-root", str(tmp / "fits"), "--device", "cuda"])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    # 6. relz's pair, re-call and grade under zsp's name, decided against zrm's CPU fits; relz restored after
    saved = Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit
    with on_zrm_cpu():
        assert Z.tag({})["arm"] == ARM and Z.BASE_ARM == BASE_ARM
        assert Z.rel_fit("J5") == ("full_zsp", "zrm", "J5") and Z.rel_fit("L-musique") == CPU_FITS["L-musique"]
    assert (Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit) == saved
    with tempfile.TemporaryDirectory() as td:
        T = Path(td)

        # 7. the gate: PROMISING and ADOPT -> 0; anything else -> 1; a wrong or missing record stops
        def rec(name, **kw):
            p = T / f"{name}.json"
            p.write_text(json.dumps(kw), encoding="utf-8")
            return str(p)

        rc_p = rec("rp", arm=ARM, decided_against=AGAINST, verdict="PROMISING")
        rc_m = rec("rm", arm=ARM, decided_against=AGAINST, verdict="MIXED")
        rc_g = rec("rg", arm=ARM, decided_against="zrm's fit of each split", verdict="PROMISING")
        b_a = rec("ba", arm="zrm", decided_against="zrs's fit of each split", verdict="ADOPT", null_splits=["J5"])
        b_n = rec("bn", arm="zrm", decided_against="zrs's fit of each split", verdict="NOT_ADOPTED", null_splits=["J5"])
        b_f = rec("bf", arm="zrm", decided_against="zrs's fit of each split", verdict="ADOPT")
        assert gate(rc_p, b_a) == 0 and gate(rc_m, b_a) == 1 and gate(rc_p, b_n) == 1
        SR.must_stop(gate, rc_g, b_a)
        SR.must_stop(gate, rc_p, b_f)
        SR.must_stop(gate, rc_p, str(T / "none.json"))
        # 8. the pair, the re-call and the grade against zrm's CPU fits, named for this round; on_cpu needs zsp's CPU
        #    stamp on the new fit and zrm's on the base
        null1 = SR.fake_null(T / "null", "n1", {("L-hotpotqa", "hotpotqa"): (-0.010, 0.012)})
        zs = {sp: str(SR.fake_compare(T / zrm_cpu_fit(sp)[0], zrm_cpu_fit(sp)[2], sp, {ds: 0.1 for ds in LG.EVAL_ORDER},
                                      arm=BASE_ARM))
              for sp in CPU_FITS}
        zb = {sp: "/".join(("outputs",) + zrm_cpu_fit(sp)) for sp in CPU_FITS}
        za = SR.fake_compare(T / "z", "scr-zsp", "L-musique", {"webqsp": 0.0215}, arm=ARM, base_r5=0.6,
                             base=zb["L-musique"])
        zh = SR.fake_compare(T / "z", "scr-zsp-hp", "L-hotpotqa", {"hotpotqa": -0.0147}, arm=ARM, base_r5=0.6,
                             base=zb["L-hotpotqa"])
        gpu = SR.fake_compare(T / "z2", "scr-zsp", "L-musique", {}, arm=ARM, base="outputs/screen/fits/scr-zrm")
        SR.must_stop(pair, [gpu, zh], T / "bad", False)
        pr = pair([za, zh], T / "pz", check_device=False)
        assert pr["verdict"] == "MIXED" and pr["arm"] == ARM
        pj = json.loads((T / "pz.json").read_text(encoding="utf-8"))
        assert pj["round"] == "twenty-third" and pj["decided_against"] == AGAINST and pj["device"] == DEVICE
        r = recall(null1, T / "pz.json", T / "rz", zs)
        assert r["verdict"] == "PROMISING" and r["changed"] == ["L-hotpotqa hotpotqa LOSS -> WITHIN"], r["changed"]
        t = (T / "rz.md").read_text(encoding="utf-8")
        assert "twenty-third round" in t and "tenth round" not in t and "| zrm (CPU) R@5 | zsp R@5 |" in t
        assert gate(str(T / "rz.json"), b_a) == 0
        full = T / "full"
        full.mkdir()
        for sp in S2.SPLITS:
            if sp not in CPU_FITS:
                SR.fake_compare(full / "src", f"fit-{sp}", sp, {"webqsp": 0.03} if sp == "J5" else {}, arm=ARM,
                                base="/".join(("outputs",) + zrm_cpu_fit(sp))).replace(full / f"compare-{sp}.json")
        g = grade(full, {"L-musique": str(za), "L-hotpotqa": str(zh)}, full / "grade", check_fits=False)
        assert g["verdict"] == "NOT_ADOPTED" and g["losses"] == 1 and g["arm"] == ARM, (g["verdict"], g["losses"])
        t = (full / "grade.md").read_text(encoding="utf-8")
        assert "docs/FULL_ROUND23.md" in t and "FULL_ROUND10" not in t
        # on_cpu: the stamps decide; a card fit on either side stops
        root = T / "root"
        for path, body in (("outputs/screen/fits/scr-zsp", {"arm": ARM, "zprop": {"settings": {"device": "cpu"}}}),
                           ("outputs/screen/fits/scr-zrm-cpu", {"arm": BASE_ARM, "zprop_base": {"device": "cpu"}}),
                           ("outputs/screen/fits/scr-zrm", {"arm": BASE_ARM})):
            (root / path).mkdir(parents=True)
            (root / path / "screen.json").write_text(json.dumps(body), encoding="utf-8")
        ok = rec("ok", new="outputs\\screen\\fits\\scr-zsp", bases=["outputs\\screen\\fits\\scr-zrm-cpu"])
        bad = rec("bad", new="outputs\\screen\\fits\\scr-zsp", bases=["outputs\\screen\\fits\\scr-zrm"])
        assert on_cpu([ok], root)
        SR.must_stop(on_cpu, [bad], root)
    log(f"zprop selftest: the arm is zrm's model plus the neighbour head on zlink's carve, train takes no other arm and "
        f"no other device, base trains zrm on the CPU only; from the same seed zsp is zrm's state plus the head and "
        f"zrm's forward bit for bit, the global generator untouched; the inputs match a hand count and stay finite at "
        f"any scale; set, the head adds to zrm's score; a toy fit moves it, its repeat identical; the gate needs a "
        f"PROMISING re-call against zrm's CPU fits and zrm ADOPTED under the null; relz's pair, re-call and grade run "
        f"against zrm's CPU fits and name this round; on_cpu refuses a card fit ({time.time() - t0:.1f}s): ok")
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
    if k.cmd == "check":
        cp = argparse.ArgumentParser()
        cp.add_argument("--dataset", required=True)
        cp.add_argument("--carve", required=True)
        cp.add_argument("--device", default="cpu")
        cp.add_argument("--threads", type=int, default=0)
        cp.add_argument("--out")
        cp.add_argument("--host", action="store_true")
        c = cp.parse_args(rest)
        if c.host:
            import lean_host as LH
            LH.substitute()
        if c.threads:
            torch.set_num_threads(c.threads)
        return check(c.dataset, c.carve, c.device, out=c.out)
    if k.cmd == "train":
        return train([k.cmd] + rest, k.split)
    if k.cmd == "base":
        return base(["train"] + rest, k.split)
    if k.cmd == "read":
        return read(rest)
    if k.cmd == "compare":
        return compare(rest)
    if k.cmd == "gate":
        gp = argparse.ArgumentParser()
        gp.add_argument("--recall", required=True)
        gp.add_argument("--base", required=True)
        g = gp.parse_args(rest)
        try:
            return gate(g.recall, g.base)
        except SystemExit as e:
            log(f"zprop gate: {e}")
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
            log(f"zprop {k.cmd}: {e}")
            return 2
        return 0
    raise SystemExit("zprop: check, base, train, read, compare, pair, recall, gate, grade, or --selftest")


if __name__ == "__main__":
    sys.exit(main())
