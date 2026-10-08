"""Screens, twenty-second round (docs/SCREENS.md; full run docs/FULL_ROUND22.md): a row's link to the pool's leading
rows, on zrm, the base.

zlk is zrm (zrm.ZRM over rmatch.ChainCarveBase, the base since docs/BASE_ZRM_ZRS.md re-graded ADOPT) with a small head
added to its score. zrm scores the pool; its own top five rows (and its top row) become the question's leaders; each row
counts its edges to them in the question's own pool graph, per edge family, and its two-step paths from them; those
counts, the row's leader flag and its score's z-score in the pool go through a two-layer head whose output is added to
zrm's score. Why (docs/DIAG_BRIDGE.md): D1 found the golds zrm misses in partly-found questions sit next to a found gold
(2wiki 0.77, hotpotqa 0.92-0.93, musique 0.38 and 0.77 within two hops; lift 6 to 14), while no fixed re-rank uses it;
D2 found a stronger scorer on the same per-row inputs gains nothing (FEATURE_LIMIT). Every per-row input is computed
before any row is scored, and the only graph inputs (WALK, WALKF, SEED, DISTS) tie a row to the retrieval's fixed seeds;
none ties it to the rows the scorer ranks highly. The link inputs do, learned as a soft term inside the scorer.

The edges are the pool graph each question already has (the look's chunks: structural, NER and kNN families on the
passage graphs; the KB's relation edges, family 0 and 2, on metaqa and webqsp), read as undirected: each edge is kept in
both directions, once per family, with no self-loops. The same rule on all six datasets; no new graph, column, encoder or
text. The leaders are zrm's own scores, ranked under no_grad in lean_gpu.top_hit's order (descending, ties by pool
position), so the inputs are query-local and label-free, the same at training and at read.

The head's inputs (9 per row): log1p of the edges from the top five, per family (3); log1p of the edges from the top
row, per family (3); log1p of the two-step paths from the top five over any family (1); the row's top-five flag (1); and
the z-score of zrm's score in its pool (1, detached). Head: Linear(9, 32), GELU, Linear(32, 1); the last layer starts at
zero and the first is drawn from its own generator (seed + 2201), so at the start zlk's forward is zrm's bit for bit and
no draw of zrm's moves. Settings and training are zrm's (rmatch.py's train, lean_screen2's); the head's 353 parameters
train with the rest.

    python outputs/mp_unified/zlink.py build --dataset 2wiki --carve s1eval --host
    python outputs/mp_unified/zlink.py check --dataset hotpotqa --carve s1eval --host
    python outputs/mp_unified/zlink.py train --split L-musique --name scr-zlk --arm zlk --device cuda --host
    python outputs/mp_unified/zlink.py read --name scr-zlk --device cuda --host
    python outputs/mp_unified/zlink.py compare --new outputs/screen/fits/scr-zlk \\
        --base outputs/screen/fits/scr-zrm,outputs/screen/fits/scr-zret,outputs/step1/fits/L-musique \\
        --out outputs/screen/scr-zlk
    python outputs/mp_unified/zlink.py pair --screens outputs/screen/scr-zlk.json,outputs/screen/scr-zlk-hp.json \\
        --out outputs/screen/scr-zlk-pair
    python outputs/mp_unified/zlink.py recall --null N1,N2,N3,N4 --pair outputs/screen/scr-zlk-pair.json \\
        --out outputs/screen/scr-zlk-pair-recall
    python outputs/mp_unified/zlink.py gate --recall outputs/screen/scr-zlk-pair-recall.json \\
        --base outputs/zbase/grade-nullx.json
    python outputs/mp_unified/zlink.py grade --full-root outputs/full_zlk \\
        --reuse L-musique=outputs/screen/scr-zlk.json,L-hotpotqa=outputs/screen/scr-zlk-hp.json
    python outputs/mp_unified/zlink.py --selftest
N1..N4 are the seed null's comparisons (outputs/screen/scr-null-s1, -s2, -s1-hp, -s2-hp .json).

Each read is decided against zrm's fit of its split (zkind.py's mapping, zrc.py's: scr-zrm and scr-zrm-hp, its full
run's fits on the other splits). zret's and step 1's fits are reported beside. The full run starts only when the
screen's re-call is PROMISING and zrm is the base: `gate`.

Speed: per question, a sort of its pool's scores and one pass over its pool's edges (about 4 to 10 a row), after zrm's
forward. Any latency figure for zlk is cold (8216ffe): each question timed from scratch, the walk, the move of its
entries and edges to the device and the forward, with no warm-up pass and nothing kept from an earlier question.
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

import zrc as ZC  # noqa: E402

ZM, RM = ZC.ZM, ZC.RM
Z, R = RM.Z, RM.R
S2, S = RM.S2, RM.S
LG, LC, LM = S.LG, S.LC, RM.LM
SR = Z.SR
log = S.log

ARM, BASE_ARM = "zlk", "zrm"
ROOT = HERE.parents[1]
LK_OUT = ROOT / "outputs" / "zlink" / "cache"
LK_KEY = "_link"
FAMS, TOP, N_IN, HID, SEED_OFF = 3, 5, 9, 32, 2201
NMAX = 1 << 15                                            # pool-local rows are stored as int16
LK = ("lk_w1", "lk_b1", "lk_w2", "lk_b2")
SETTINGS = {"families": FAMS, "top": TOP, "inputs": N_IN, "hidden": HID, "seed_offset": SEED_OFF,
            "edges": "the look's pool edges, undirected (both directions), once per family, no self-loops"}


# ── the edge cache ───────────────────────────────────────────────────────────


def chunk_edges(c, where):
    """One look chunk's pool edges as zlk reads them: per question, (u, v, fam) undirected, unique, no self-loops,
    sorted by (u, v, fam). Returns (edges per question, u, v, fam, counts)."""
    qps = c["q_pool_size"].astype(np.int64)
    eo = np.r_[0, np.cumsum(c["q_edges"].astype(np.int64))]
    u, v, f = c["e_u"].astype(np.int64), c["e_v"].astype(np.int64), c["e_fam"].astype(np.int64)
    if int(eo[-1]) != u.size or v.size != u.size or f.size != u.size:
        raise SystemExit(f"{where}: {u.size} edges against q_edges {int(eo[-1])}")
    if qps.size and int(qps.max()) >= NMAX:
        raise SystemExit(f"{where}: a pool of {int(qps.max())} rows, above int16")
    qi = np.repeat(np.arange(qps.size), np.diff(eo))
    n_e = qps[qi]
    if bool(((u < 0) | (v < 0) | (u >= n_e) | (v >= n_e)).any()) or bool(((f < 0) | (f >= FAMS)).any()):
        raise SystemExit(f"{where}: an edge is outside its pool or its family is not in 0..{FAMS - 1}")
    loop = u == v
    q2, a, b, ff = (np.r_[qi[~loop], qi[~loop]], np.r_[u[~loop], v[~loop]], np.r_[v[~loop], u[~loop]],
                    np.r_[f[~loop], f[~loop]])
    key = np.unique(((q2 * NMAX + a) * NMAX + b) * FAMS + ff)
    ff, key = key % FAMS, key // FAMS
    b, key = key % NMAX, key // NMAX
    a, q2 = key % NMAX, key // NMAX
    ne = np.bincount(q2, minlength=qps.size).astype(np.int64)
    return ne, a.astype(np.int16), b.astype(np.int16), ff.astype(np.int8), {
        "raw": int(u.size), "self_loops": int(loop.sum()), "kept": int(key.size)}


def build(ds, carve, out_root=LK_OUT, cache_root=LC.OUT, look_root=LM.LOOK, placement=None):
    """Each step-1 cache part's pool edges (chunk_edges) from the look's chunks it was built from, tied to its record."""
    t0 = time.time()
    d, files, _ids, head = LC.look_chunks(ds, carve, look_root)
    parts, recs = LC.part_dirs(ds, carve, cache_root)
    summary = {}
    for p, r in zip(parts, recs):
        if r["look"]["carve_ids_sha256"] != head["carve_ids_sha256"] or r["look"]["records"] != head["records"]:
            raise SystemExit(f"{p}: built from another look than {d}")
        tp = time.time()
        lo, hi = r["chunks"]
        acc = {"q_ne": [], "e_u": [], "e_v": [], "e_fam": []}
        n_all, tot = [], {"raw": 0, "self_loops": 0, "kept": 0}
        for ch in range(lo, hi):
            with np.load(files[ch]) as zf:
                c = {k: zf[k] for k in ("q_pool_size", "q_edges", "e_u", "e_v", "e_fam")}
            ne, a, b, ff, st = chunk_edges(c, files[ch])
            for k, x in zip(("q_ne", "e_u", "e_v", "e_fam"), (ne, a, b, ff)):
                acc[k].append(x)
            for k in tot:
                tot[k] += st[k]
            n_all.append(c["q_pool_size"].astype(np.int64))
        arrays = {k: np.concatenate(v) for k, v in acc.items()}
        n_all = np.concatenate(n_all)
        if n_all.size != r["queries"] or int(n_all.sum()) != r["rows"]:
            raise SystemExit(f"{p}: {n_all.size} queries and {int(n_all.sum())} rows against {r['queries']} and {r['rows']}")
        if (p / "n.npy").exists():
            if not np.array_equal(np.asarray(LC.load_array(p, "n", r, verify=True)).astype(np.int64), n_all):
                raise SystemExit(f"{p}: the chunks' pool sizes are not step 1's cached n")
            against = "IDENTICAL"
        else:
            against = "step 1's arrays are not on disk here"
        out = Path(out_root) / ds / carve / p.name
        out.mkdir(parents=True, exist_ok=True)
        shas = {k: R.save_npy(out, k, v) for k, v in arrays.items()}
        fam = np.bincount(arrays["e_fam"].astype(np.int64), minlength=FAMS)
        rec = {"dataset": ds, "carve": carve, "part": p.name, "step1_record_sha256": LC.sha_file(p / "record.json"),
               "chunks": [lo, hi], "queries": int(n_all.size), "rows": r["rows"], "edges": int(arrays["e_u"].size),
               "edges_per_row": round(arrays["e_u"].size / max(1, r["rows"]), 3), "families": [int(x) for x in fam],
               "look_edges": tot, "look": head["records"], "carve_ids_sha256": head["carve_ids_sha256"],
               "n_against_step1": against, "settings": SETTINGS,
               "arrays": {k: {"dtype": v.dtype.str, "shape": list(v.shape), "sha256": shas[k]} for k, v in arrays.items()},
               "placement": placement or {"where": "laptop"}, "script_sha256": LC.sha_src(__file__),
               "seconds": round(time.time() - tp, 1), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        R.write_json(out / "record.json", rec)
        summary[p.name] = {"queries": int(n_all.size), "edges": rec["edges"], "against": against}
        log(f"  {ds}/{carve} {p.name}: {n_all.size} queries, {rec['edges']} edges ({rec['edges_per_row']} a row; "
            f"families {rec['families']}; {tot['self_loops']} self-loops dropped); n {against} ({time.time() - t0:.0f}s)")
    log(f"zlink build {ds}/{carve}: {len(parts)} part(s) ({time.time() - t0:.0f}s)")
    return summary


def load_links(c, lk_root=LK_OUT, cache_root=LC.OUT, verify=True):
    """A carve's edges in host memory, its parts in order, each tied to its step-1 part's record, every edge checked
    inside its question's pool."""
    t0 = time.time()
    parts, recs = LC.part_dirs(c.ds, c.carve, cache_root)
    Q = c.n_np.size
    hdrs = []
    for p, r in zip(parts, recs):
        cp = Path(lk_root) / c.ds / c.carve / p.name
        if not (cp / "record.json").exists():
            raise SystemExit(f"{cp}: no zlink edges built for {p}")
        h = json.loads((cp / "record.json").read_text(encoding="utf-8"))
        if h["step1_record_sha256"] != c.part_shas[p.name] or h["queries"] != r["queries"] or h["rows"] != r["rows"]:
            raise SystemExit(f"{cp}: not built from {p}")
        if h["settings"] != SETTINGS:
            raise SystemExit(f"{cp}: built with other settings ({h['settings']})")
        hdrs.append((cp, h))
    ne_all = sum(h["edges"] for _cp, h in hdrs)
    lk = {"q_ne": np.empty(Q, np.int64), "e_u": np.empty(ne_all, np.int16), "e_v": np.empty(ne_all, np.int16),
          "e_fam": np.empty(ne_all, np.int8)}
    qa = ea = 0
    for (cp, h), r in zip(hdrs, recs):
        if r["query_range"][0] != qa:
            raise SystemExit(f"{cp}: its questions do not follow {qa}")
        arr = {}
        for k in lk:
            fn = cp / f"{k}.npy"
            if verify and LC.sha_file(fn) != h["arrays"][k]["sha256"]:
                raise SystemExit(f"{fn}: sha256 is not its record's")
            arr[k] = np.load(fn)
            if arr[k].dtype.str != h["arrays"][k]["dtype"] or list(arr[k].shape) != h["arrays"][k]["shape"]:
                raise SystemExit(f"{fn}: {arr[k].dtype} {arr[k].shape}, its record says otherwise")
        q, e_ = arr["q_ne"].size, int(arr["q_ne"].sum())
        if q != r["queries"] or e_ != arr["e_u"].size or e_ != h["edges"]:
            raise SystemExit(f"{cp}: {q} questions and {e_} edges do not match its arrays")
        n_e = np.repeat(c.n_np[qa:qa + q], arr["q_ne"])
        bad = [bool((arr["e_u"] < 0).any()), bool((arr["e_v"] < 0).any()),
               bool((arr["e_u"].astype(np.int64) >= n_e).any()), bool((arr["e_v"].astype(np.int64) >= n_e).any()),
               bool(((arr["e_fam"] < 0) | (arr["e_fam"] >= FAMS)).any()), bool((arr["e_u"] == arr["e_v"]).any())]
        if any(bad):
            raise SystemExit(f"{cp}: an edge is out of range (checks {bad})")
        lk["q_ne"][qa:qa + q] = arr["q_ne"]
        for k in ("e_u", "e_v", "e_fam"):
            lk[k][ea:ea + e_] = arr[k]
        qa, ea = qa + q, ea + e_
    if qa != Q:
        raise SystemExit(f"{c.ds}/{c.carve}: the built parts hold {qa} of {Q} questions")
    lk["e_off"] = np.r_[0, np.cumsum(lk["q_ne"])]
    lk["records"] = {cp.name: LC.sha_file(cp / "record.json") for cp, _h in hdrs}
    log(f"  {c.ds}/{c.carve}: {ne_all} link edges ({sum(lk[k].nbytes for k in ('e_u', 'e_v', 'e_fam')) / 1e9:.2f} GB "
        f"in host memory) from {Path(lk_root) / c.ds / c.carve} ({len(parts)} part(s), {time.time() - t0:.0f}s)")
    return lk


def link_batch(c, qs):
    """The edges of questions qs, in qs's order, as batch rows (a question's rows lie where lean_gpu.CacheCarve.batch
    puts them), on the carve's device."""
    lk = c.links
    qs = np.asarray(qs, np.int64)
    cnt = c.n_np[qs]
    rseg = np.cumsum(cnt) - cnt
    ne = lk["q_ne"][qs]
    idx = np.repeat(lk["e_off"][qs] - (np.cumsum(ne) - ne), ne) + np.arange(int(ne.sum()))
    eq = np.repeat(np.arange(qs.size), ne)
    dev = c.device
    return {"u": torch.from_numpy(rseg[eq] + lk["e_u"][idx].astype(np.int64)).to(dev),
            "v": torch.from_numpy(rseg[eq] + lk["e_v"][idx].astype(np.int64)).to(dev),
            "f": torch.from_numpy(lk["e_fam"][idx].astype(np.int64)).to(dev)}


def link_carve(base, lk_root=LK_OUT):
    """base's carve class with the pool edges beside it: batch() adds them under LK_KEY; every block, row, chain and
    gold is base's, unchanged."""

    class LinkCarve(base):
        LK_ROOT = lk_root

        def __init__(self, ds, carve, basis, device, cache_root=LC.OUT, verify=True, score2=False):
            super().__init__(ds, carve, basis, device, cache_root, verify, score2)
            self.links = load_links(self, type(self).LK_ROOT, cache_root, verify)

        def batch(self, qs, blocks):
            feats, nq, base_z, gold = super().batch(qs, blocks)
            feats[LK_KEY] = link_batch(self, qs)
            return feats, nq, base_z, gold

    LinkCarve.__name__ = LinkCarve.__qualname__ = f"Link{base.__name__}"
    return LinkCarve


LinkCarveBase = link_carve(RM.ChainCarveBase)


# ── the model ────────────────────────────────────────────────────────────────


@torch.no_grad()
def pool_rank(s, nq, B):
    """Each row's rank in its question's pool, lean_gpu.top_hit's order (descending score, NaN last, ties by pool
    position; -0.0 ties +0.0). Rows are question-major, in pool order."""
    s0 = s + 0.0
    nan = torch.isnan(s0)
    key = torch.where(nan, torch.zeros_like(s0), -s0) + 0.0
    o = torch.sort(key, stable=True).indices
    o = o[torch.sort(nan[o].to(torch.int8), stable=True).indices]
    o = o[torch.sort(nq[o], stable=True).indices]
    cnt = torch.bincount(nq, minlength=B)
    seg = torch.cumsum(cnt, 0) - cnt
    rank = torch.empty_like(o)
    rank[o] = torch.arange(o.numel(), device=s.device) - seg[nq[o]]
    return rank


@torch.no_grad()
def link_inputs(s, lk, nq, B):
    """(N, 9): the link counts of each row given the pool's leaders under score s (see the module's docstring)."""
    N = nq.numel()
    rank = pool_rank(s, nq, B)
    dt = s.dtype
    l5, l1 = (rank < TOP).to(dt), (rank == 0).to(dt)
    u, v, f = lk["u"], lk["v"], lk["f"]
    at = v * FAMS + f
    c5 = torch.zeros(N * FAMS, dtype=dt, device=s.device).index_add_(0, at, l5[u]).view(N, FAMS)
    c1 = torch.zeros(N * FAMS, dtype=dt, device=s.device).index_add_(0, at, l1[u]).view(N, FAMS)
    two = torch.zeros(N, dtype=dt, device=s.device).index_add_(0, v, c5.sum(1)[u])
    zs = LG.seg_zscore8D(s.detach().unsqueeze(1), nq, B).squeeze(1)
    return torch.cat([torch.log1p(c5), torch.log1p(c1), torch.log1p(two).unsqueeze(1), l5.unsqueeze(1),
                      zs.unsqueeze(1)], 1)


class ZLink(ZM.ZRM):
    """zlk: zrm's score plus a head (zero at the start) over each row's links to zrm's leading rows."""

    def __init__(self, blocks, widths, hidden=128, dropout=0.1, seed=0, arm="ctl", ctx="none"):
        super().__init__(blocks, widths, hidden, dropout, seed, arm, ctx)
        g = torch.Generator().manual_seed(int(seed) + SEED_OFF)
        self.lk_w1 = nn.Parameter((torch.rand(HID, N_IN, generator=g) * 2 - 1) / N_IN ** 0.5)
        self.lk_b1 = nn.Parameter(torch.zeros(HID))
        self.lk_w2 = nn.Parameter(torch.zeros(1, HID))
        self.lk_b2 = nn.Parameter(torch.zeros(1))

    def link_head(self, x):
        return Fn.linear(Fn.gelu(Fn.linear(x, self.lk_w1, self.lk_b1)), self.lk_w2, self.lk_b2).squeeze(-1)

    def forward(self, feats, keep, nq, B, base_z):
        s = super().forward(feats, keep, nq, B, base_z)
        if LK_KEY not in feats:
            return s
        return s + self.link_head(link_inputs(s, feats[LK_KEY], nq, B))


S.ARMS.update({ARM: (ZLink, LinkCarveBase)})


# ── train, read and compare ──────────────────────────────────────────────────


def built_records(root=LK_OUT):
    return {str(f.parent.relative_to(root)).replace("\\", "/"): LC.sha_file(f)
            for f in sorted(Path(root).glob("*/*/part_*/record.json"))}


def train(argv, split):
    """rmatch.py's train (zrm's settings) for the arm zlk; this file's records stamped."""
    arm = R.arm_of(argv)
    if arm != ARM:
        raise SystemExit(f"zlink: train takes --arm {ARM}, not {arm}")
    if S.ARMS.get(ARM) != (ZLink, LinkCarveBase) or S.ARMS.get(BASE_ARM) != (ZM.ZRM, RM.ChainCarveBase):
        raise SystemExit(f"zlink: the arm {ARM} is {S.ARMS.get(ARM)}, not zlk's model on its link carve")
    rc = RM.train(argv, split)
    name, out_root = S2.where(argv)
    sj = out_root / name / "screen.json"
    if sj.exists():
        rec = json.loads(sj.read_text(encoding="utf-8"))
        rec.update({"zlink_sha256": LC.sha_src(__file__), "zrm_sha256": LC.sha_src(ZM.__file__),
                    "zlink": {"model": "zlink.ZLink (zrm.ZRM plus a head over each row's links to zrm's leading rows)",
                              "carve": "zlink.LinkCarveBase over rmatch.ChainCarveBase", "head": list(LK),
                              "settings": SETTINGS},
                    "zlink_records": built_records()})
        LC.write_json(sj, rec)
    return rc


def read(argv):
    name, out_root = S2.where(argv)
    arm = json.loads((out_root / name / "screen.json").read_text(encoding="utf-8"))["arm"]
    if arm != ARM:
        raise SystemExit(f"zlink: {name} was trained as {arm}, not {ARM}")
    return RM.read(argv)


def compare(rest):
    return S2.main(["compare"] + rest)


# ── a real carve: the edges load and, at the start, zlk is zrm ───────────────


def check(ds, carve, device="cpu", n_q=256, out=None):
    """On a built carve: the edges load under every check, and from the same seed zlk's forward equals zrm's bit for bit
    on its first n_q questions (eval, and train with the same dropout draws). Reported beside, not decided: the share
    of gold and non-gold rows outside the top five with an edge to a top-five row, under the untrained score."""
    t0 = time.time()
    LG.bind_device_ops()
    basis = "2wiki"
    c = LinkCarveBase(ds, carve, basis, device)
    blocks = list(LG.SETS["pick"])
    qs = np.arange(min(n_q, c.rows), dtype=np.int64)
    feats, nq, bz, gold = c.batch(qs, blocks)
    B = qs.size
    keep = torch.ones(B, len(blocks), device=c.device)
    nets = {}
    for k, cls in ((BASE_ARM, ZM.ZRM), (ARM, ZLink)):
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
        x = link_inputs(s, feats[LK_KEY], nq, B)
    out5 = x[:, 7] == 0
    near = (x[:, 0:3].sum(1) > 0)
    g = gold.bool()
    share = {"gold": float(near[g & out5].float().mean()) if bool((g & out5).any()) else None,
             "non_gold": float(near[~g & out5].float().mean()) if bool((~g & out5).any()) else None}
    rec = {"dataset": ds, "carve": carve, "questions": int(B), "rows": int(nq.numel()),
           "edges": int(feats[LK_KEY]["u"].numel()), "identity": rows, "near_top5_outside": share,
           "script_sha256": LC.sha_src(__file__), "seconds": round(time.time() - t0, 1)}
    ok = all(v == "IDENTICAL" for v in rows.values())
    log(f"zlink check {ds}/{carve}: {B} questions, {rec['edges']} edges; zlk at the start against zrm {rows}; rows "
        f"outside the untrained top five next to one: gold {share['gold']}, non-gold {share['non_gold']} "
        f"({rec['seconds']}s) -> {'ok' if ok else 'FAIL'}")
    if out:
        R.write_json(Path(out), rec)
    return 0 if ok else 1


# ── relz.py's pair, re-call and grade, decided against zrm's fits ────────────


@contextlib.contextmanager
def on_zrm():
    """relz.py's records under zlk's name, each read decided against zrm's fit of its split (zrc.py's mapping); relz
    restored after."""
    with ZC.on_zrm():
        Z.ARM = ARM
        yield


FIX = (("(rel's screen fits scr-rel and scr-rel-hp)", "(zrm's screen fits scr-zrm and scr-zrm-hp)"),
       ("(rel's screen fits)", "(zrm's screen fits)"),
       ("| rel R@5 | relz R@5 |", "| zrm R@5 | zlk R@5 |"),
       ("section 2 and the tenth round", "section 2 and the twenty-second round"),
       ("docs/FULL_ROUND10.md", "docs/FULL_ROUND22.md"))


def restamp(out):
    """A record relz.py wrote under zlk's name: zrm named as the base in its md, this round and this file's sha added."""
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
        rec.update({"decided_against": "zrm's fit of each split", "round": "twenty-second",
                    "zlink_sha256": LC.sha_src(__file__)})
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


def gate(recall_file, base_file):
    """0 when the screen's re-call is PROMISING and zrm is the base (docs/BASE_ZRM_ZRS.md's re-grade ADOPT), 1
    otherwise; a missing file, or one that is not the declared record, stops (2)."""
    got = []
    for fn, arm, want in ((recall_file, ARM, "zrm's fit of each split"), (base_file, "zrm", "zrs's fit of each split")):
        p = Path(fn)
        if not p.exists():
            raise SystemExit(f"zlink gate: {p} missing")
        rec = json.loads(p.read_text(encoding="utf-8"))
        if rec.get("arm") != arm or rec.get("decided_against") != want:
            raise SystemExit(f"zlink gate: {p} is arm {rec.get('arm')}'s, decided against "
                             f"{rec.get('decided_against')!r}; not the declared record")
        got.append(rec.get("verdict"))
    if "null_splits" not in json.loads(Path(base_file).read_text(encoding="utf-8")):
        raise SystemExit(f"zlink gate: {base_file} is not a re-grade under the null")
    ok = got[0] == "PROMISING" and got[1] == "ADOPT"
    log(f"zlink gate: the re-call {got[0]}, zrm against zrs {got[1]} -> {'the full run starts' if ok else 'no full run'}")
    return 0 if ok else 1


# ── selftest ─────────────────────────────────────────────────────────────────


def hand_inputs(s, n_np, qs, links_of):
    """link_inputs by hand: per question, python sorts and loops over its own edge list."""
    out, at = [], 0
    for qi, q in enumerate(qs):
        n = int(n_np[q])
        sc = [float(x) for x in s[at:at + n]]
        order = sorted(range(n), key=lambda i: (-sc[i], i))
        rank = [0] * n
        for r_, i in enumerate(order):
            rank[i] = r_
        E = links_of(q)
        c5 = [[0] * FAMS for _ in range(n)]
        c1 = [[0] * FAMS for _ in range(n)]
        for a, b, f in E:
            c5[b][f] += rank[a] < TOP
            c1[b][f] += rank[a] == 0
        two = [0] * n
        for a, b, _f in E:
            two[b] += sum(c5[a])
        mu = sum(sc) / n
        sd = (sum((x - mu) ** 2 for x in sc) / n) ** 0.5
        for i in range(n):
            z = (sc[i] - mu) / sd if sd >= 1e-6 else 0.0
            out.append([np.log1p(x) for x in c5[i]] + [np.log1p(x) for x in c1[i]] + [np.log1p(two[i]),
                                                                                       float(rank[i] < TOP), z])
        at += n
    return np.asarray(out, np.float64)


def selftest():
    t0 = time.time()
    LG.bind_device_ops()
    # 1. the arm: zrm's model plus the link head, on zrm's chain carve with the edges beside; zrm's arm unchanged; train
    #    takes no other arm
    assert S.ARMS[ARM] == (ZLink, LinkCarveBase) and S.ARMS[BASE_ARM] == (ZM.ZRM, RM.ChainCarveBase)
    assert ZLink.__mro__[1] is ZM.ZRM and LinkCarveBase.__mro__[1] is RM.ChainCarveBase
    SR.must_stop(train, ["train", "--arm", BASE_ARM, "--name", "x"], "L-musique")
    SR.must_stop(train, ["train", "--name", "x"], "L-musique")
    # 2. chunk_edges: undirected, once per family, no self-loops, each question's own; a row outside its pool stops
    ck = {"q_pool_size": np.array([3, 4]), "q_edges": np.array([4, 3]),
          "e_u": np.array([0, 1, 2, 2, 0, 3, 1]), "e_v": np.array([1, 0, 2, 0, 3, 0, 2]),
          "e_fam": np.array([0, 0, 1, 2, 1, 1, 0])}
    ne, a, b, ff, st = chunk_edges(ck, "toy")
    assert ne.tolist() == [4, 4] and st == {"raw": 7, "self_loops": 1, "kept": 8}
    assert list(zip(a.tolist(), b.tolist(), ff.tolist())) == [(0, 1, 0), (0, 2, 2), (1, 0, 0), (2, 0, 2),
                                                             (0, 3, 1), (1, 2, 0), (2, 1, 0), (3, 0, 1)]
    SR.must_stop(chunk_edges, dict(ck, e_v=np.array([1, 0, 2, 3, 0, 3, 1])), "toy")
    SR.must_stop(chunk_edges, dict(ck, e_fam=np.array([0, 0, 1, 3, 1, 1, 0])), "toy")
    tmp = Path(tempfile.mkdtemp(prefix="zlink_"))
    try:
        rng = np.random.default_rng(22)
        qq = RM.toy_roots(tmp, rng, chunks=(12, 12))
        RM.build("metaqa", "toy", out_root=tmp / "ch", cache_root=tmp / "cache", look_root=tmp / "look",
                 rel_dir=tmp / "rel")
        build("metaqa", "toy", out_root=tmp / "lk", cache_root=tmp / "cache", look_root=tmp / "look")
        cls = link_carve(RM.chain_carve(ZM.ToyZ, tmp / "ch", tmp / "rel"), tmp / "lk")
        c = cls("metaqa", "toy", "2wiki", "cpu", tmp / "cache")

        def links_of(q):
            """Question q's toy edges, undirected, once per family, no self-loops, from the toy's own arrays."""
            t = qq[q]
            e = {(int(x), int(y), int(f)) for x, y, f in zip(t["e_u"], t["e_v"], t["e_fam"]) if x != y}
            return sorted(e | {(y, x, f) for x, y, f in e})

        # 3. the built edges are each question's own, by hand; the batch puts them on its questions' rows
        for q in range(c.rows):
            lo, hi = c.links["e_off"][q], c.links["e_off"][q + 1]
            got = list(zip(c.links["e_u"][lo:hi].tolist(), c.links["e_v"][lo:hi].tolist(),
                           c.links["e_fam"][lo:hi].tolist()))
            assert got == links_of(q), q
        blocks = ["rank", "SEMB"]
        qs = np.r_[np.arange(c.rows)[::-1][:5], np.arange(c.rows)[:4]].astype(np.int64)
        B = qs.size
        bt = c.batch(qs, blocks)
        feats, nq, bz, _g = bt
        assert LK_KEY in feats and RM.CH_KEY in feats
        cnt = c.n_np[qs]
        rseg = np.cumsum(cnt) - cnt
        lk = feats[LK_KEY]
        want = sorted((int(rseg[j] + x), int(rseg[j] + y), f) for j, q in enumerate(qs) for x, y, f in links_of(q))
        assert sorted(zip(lk["u"].tolist(), lk["v"].tolist(), lk["f"].tolist())) == want
        # 4. a part whose step-1 record is not the one its edges were built on stops the carve
        rp = next((tmp / "lk" / "metaqa" / "toy").glob("part_*/record.json"))
        saved_rec = rp.read_text(encoding="utf-8")
        h = json.loads(saved_rec)
        h["step1_record_sha256"] = "0" * 64
        rp.write_text(json.dumps(h), encoding="utf-8")
        SR.must_stop(cls, "metaqa", "toy", "2wiki", "cpu", tmp / "cache")
        rp.write_text(saved_rec, encoding="utf-8")
        # 5. from the same seed zlk is zrm's state plus the head, the last layer zero, and its forward is zrm's bit for
        #    bit, in eval and with the same dropout draws in training; the global generator is where zrm leaves it
        keep = torch.ones(B, len(blocks))
        nets, after = {}, {}
        for k, cl in ((BASE_ARM, ZM.ZRM), (ARM, ZLink)):
            torch.manual_seed(0)
            nets[k] = cl(blocks, c.widths, 16, dropout=0.1, seed=0)
            after[k] = torch.rand(4)
        assert torch.equal(after[ARM], after[BASE_ARM])
        sz, sk = nets[BASE_ARM].state_dict(), nets[ARM].state_dict()
        assert sorted(set(sk) - set(sz)) == sorted(LK) and all(torch.equal(sz[k], sk[k]) for k in sz)
        assert float(sk["lk_w2"].abs().sum()) == 0.0 and float(sk["lk_b2"].abs().sum()) == 0.0
        assert float(sk["lk_w1"].abs().sum()) > 0
        with torch.no_grad():
            for mode in (False, True):
                out = {}
                for k, m in nets.items():
                    m.train(mode)
                    torch.manual_seed(5)
                    out[k] = m(feats, keep, nq, B, bz)
                assert torch.equal(out[ARM], out[BASE_ARM]), mode
            # 6. the inputs are link_inputs by hand (ranks in top_hit's order, counts per family, two-step paths,
            #    z-score); with the head set, zlk is zrm plus the head on them; a batch without edges is zrm's own
            nz, nk = nets[BASE_ARM].eval(), nets[ARM].eval()
            s = nz(feats, keep, nq, B, bz)
            x = link_inputs(s, lk, nq, B)
            xh = hand_inputs(s.numpy(), c.n_np, qs, links_of)
            assert x.shape == (int(cnt.sum()), N_IN) and np.allclose(x.numpy(), xh, atol=1e-5), np.abs(x.numpy() - xh).max()
            assert float(x[:, 0:6].sum()) > 0 and float(x[:, 6].sum()) > 0
            top, _hit = LG.top_hit(s, torch.ones_like(s, dtype=torch.bool), B, torch.from_numpy(rseg),
                                   torch.from_numpy(cnt))
            assert torch.equal(top, torch.from_numpy(np.minimum(cnt, TOP))) and int(x[:, 7].sum()) == int(top.sum())
            gen = torch.Generator().manual_seed(3)
            for k in LK:
                getattr(nk, k).copy_(torch.randn(getattr(nk, k).shape, generator=gen))
            hd = Fn.gelu(x @ nk.lk_w1.T + nk.lk_b1) @ nk.lk_w2.T + nk.lk_b2
            assert torch.equal(nk(feats, keep, nq, B, bz), s + hd.squeeze(-1))
            f2 = {k: v for k, v in feats.items() if k != LK_KEY}
            assert torch.equal(nk(f2, keep, nq, B, bz), nz(f2, keep, nq, B, bz))
        # 7. toy fits through lean_gpu's loop: the head moves, the fit leaves zrm's, and a repeat is identical
        cfg = {"lr": 2e-2, "wd": 1e-4, "dropout": 0.1, "epochs": 3, "swa_from": 1, "cos": False, "adamw": False,
               "drop": False}
        every = lambda f: f["states"] + [f["swa"]]  # noqa: E731
        fits = {}
        for arm in (BASE_ARM, ARM):
            with S.patched(arm):
                fits[arm] = LG.fit_variant([c], blocks, cfg, 0, 16, "none", "cpu", tag=f"toy-{arm}")
        with S.patched(ARM):
            rep = LG.fit_variant([c], blocks, cfg, 0, 16, "none", "cpu", tag="toy-zlk repeat")
        sw = fits[ARM]["swa"]
        assert float(sw["lk_w2"].abs().sum()) > 0
        assert any(not torch.equal(sw[k], fits[BASE_ARM]["swa"][k]) for k in fits[BASE_ARM]["swa"])
        assert all(not LG.same_state(a_, b_) for a_, b_ in zip(every(fits[ARM]), every(rep)))
        # 8. read refuses another arm's fit
        (tmp / "fits" / "x").mkdir(parents=True)
        LC.write_json(tmp / "fits" / "x" / "screen.json", {"arm": BASE_ARM})
        try:
            read(["--name", "x", "--out-root", str(tmp / "fits")])
            raise AssertionError("zlink read zrm's fit")
        except SystemExit as e:
            assert "trained as zrm" in str(e)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    # 9. relz's pair, re-call and grade under zlk's name, decided against zrm's fits; relz restored after
    saved = Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit
    with on_zrm():
        assert Z.tag({})["arm"] == ARM and Z.BASE_ARM == BASE_ARM
        assert Z.rel_fit("J5") == ("full_zrm", "fits", "J5") and Z.rel_fit("L-musique") == ("screen", "fits", "scr-zrm")
    assert (Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit) == saved
    with tempfile.TemporaryDirectory() as td:
        T = Path(td)

        # 10. the gate: PROMISING and ADOPT -> 0; anything else -> 1; a wrong or missing record stops
        def rec(name, **kw):
            p = T / f"{name}.json"
            p.write_text(json.dumps(kw), encoding="utf-8")
            return str(p)

        rc_p = rec("rp", arm=ARM, decided_against="zrm's fit of each split", verdict="PROMISING")
        rc_m = rec("rm", arm=ARM, decided_against="zrm's fit of each split", verdict="MIXED")
        rc_c = rec("rc", arm="zkind", decided_against="zrm's fit of each split", verdict="PROMISING")
        b_a = rec("ba", arm="zrm", decided_against="zrs's fit of each split", verdict="ADOPT", null_splits=["J5"])
        b_n = rec("bn", arm="zrm", decided_against="zrs's fit of each split", verdict="NOT_ADOPTED", null_splits=["J5"])
        b_f = rec("bf", arm="zrm", decided_against="zrs's fit of each split", verdict="ADOPT")
        assert gate(rc_p, b_a) == 0 and gate(rc_m, b_a) == 1 and gate(rc_p, b_n) == 1
        SR.must_stop(gate, rc_c, b_a)
        SR.must_stop(gate, rc_p, b_f)
        SR.must_stop(gate, b_a, rc_p)
        SR.must_stop(gate, rc_p, str(T / "none.json"))
        # 11. the pair, the re-call and the grade against zrm's fits, named for this round
        null1 = SR.fake_null(T / "null", "n1", {("L-hotpotqa", "hotpotqa"): (-0.010, 0.012)})
        zs = {sp: str(SR.fake_compare(T / ZC.zrm_fit(sp)[0], ZC.zrm_fit(sp)[2], sp, {ds: 0.1 for ds in LG.EVAL_ORDER},
                                      arm=BASE_ARM))
              for sp in ZC.ZRM_FITS}
        zb = {sp: "/".join(("outputs",) + ZC.zrm_fit(sp)) for sp in ZC.ZRM_FITS}
        za = SR.fake_compare(T / "z", "scr-zlk", "L-musique", {"webqsp": 0.0215}, arm=ARM, base_r5=0.6,
                             base=zb["L-musique"])
        zh = SR.fake_compare(T / "z", "scr-zlk-hp", "L-hotpotqa", {"hotpotqa": -0.0147}, arm=ARM, base_r5=0.6,
                             base=zb["L-hotpotqa"])
        st1 = SR.fake_compare(T / "z2", "scr-zlk", "L-musique", {}, arm=ARM)
        SR.must_stop(pair, [st1, zh], T / "bad")
        pr = pair([za, zh], T / "pz")
        assert pr["verdict"] == "MIXED" and pr["arm"] == ARM
        pj = json.loads((T / "pz.json").read_text(encoding="utf-8"))
        assert pj["round"] == "twenty-second" and pj["decided_against"] == "zrm's fit of each split"
        r = recall(null1, T / "pz.json", T / "rz", zs)
        assert r["verdict"] == "PROMISING" and r["changed"] == ["L-hotpotqa hotpotqa LOSS -> WITHIN"], r["changed"]
        t = (T / "rz.md").read_text(encoding="utf-8")
        assert "twenty-second round" in t and "tenth round" not in t and "| zrm R@5 | zlk R@5 |" in t
        assert gate(str(T / "rz.json"), b_a) == 0
        full = T / "full"
        full.mkdir()
        for sp in S2.SPLITS:
            if sp not in ZC.ZRM_FITS:
                SR.fake_compare(full / "src", f"fit-{sp}", sp, {"webqsp": 0.03} if sp == "J5" else {}, arm=ARM,
                                base="/".join(("outputs",) + ZC.zrm_fit(sp))).replace(full / f"compare-{sp}.json")
        g = grade(full, {"L-musique": str(za), "L-hotpotqa": str(zh)}, full / "grade", check_fits=False)
        assert g["verdict"] == "NOT_ADOPTED" and g["losses"] == 1 and g["arm"] == ARM, (g["verdict"], g["losses"])
        t = (full / "grade.md").read_text(encoding="utf-8")
        assert "docs/FULL_ROUND22.md" in t and "FULL_ROUND10" not in t
    log(f"zlink selftest: the arm is zrm's model plus the link head on zrm's chain carve with the pool edges beside, and "
        f"train takes no other; the edges are each question's own, undirected, once per family, no self-loops, and a "
        f"part built on another record stops; from the same seed zlk is zrm's state plus the head and zrm's forward bit "
        f"for bit, the global generator untouched; the inputs match a hand count; set, the head adds to zrm's score; a "
        f"toy fit moves it, its repeat identical; the gate needs a PROMISING re-call and zrm ADOPTED under the null; "
        f"relz's pair, re-call and grade run against zrm's fits and name this round ({time.time() - t0:.1f}s): ok")
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
    if k.cmd == "build":
        bp = argparse.ArgumentParser()
        bp.add_argument("--dataset", required=True)
        bp.add_argument("--carve", required=True)
        bp.add_argument("--host", action="store_true")
        b = bp.parse_args(rest)
        build(b.dataset, b.carve, placement={"where": "host" if b.host else "laptop"})
        return 0
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
            log(f"zlink gate: {e}")
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
            log(f"zlink {k.cmd}: {e}")
            return 2
        return 0
    raise SystemExit("zlink: build, check, train, read, compare, pair, recall, gate, grade, or --selftest")


if __name__ == "__main__":
    sys.exit(main())
