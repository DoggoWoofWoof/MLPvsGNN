"""Screens, twenty-fourth round (docs/SCREENS.md; full run docs/FULL_ROUND24.md): feature selection for the MLP over all
six datasets, on zrm, the base. No message passing: every candidate column is computed in preprocessing from the
question, the row, the frozen embeddings and the graph's fixed structure around the fixed seeds, before any row is
scored; none reads another row's score or a pass over neighbours at read time.

Why: zrm's inputs are step 1's pick (lean_mlp8.PICK), and the pick came from one greedy on 2wiki's select carve
(l3-2w). That greedy dropped the NER, kNN and all-family views of the structure, the all-family depth rings and the seed
similarities, while D1 (docs/DIAG_BRIDGE.md) finds the golds zrm misses linked to the ones it finds, on musique mostly
over NER edges. No selection of zrm's inputs has looked at all six datasets or at a dataset read zero-shot.

The candidates are blocks of the look's 129 compiled columns that the pick leaves out (CAND, NAMES below):
  topo_NER (11), topo_KNN (11), topo_FULL (11)   distances to the seeds, seeds within one and two hops, pool and global
                                                  degree, walks, the seeds' component: the NER, kNN and all-family views
                                                  of what topo_STRUCT holds for the structural edges
  depth_FULL (17)                                 depth_STRUCT's rings and seed counts on every edge family
  seedcond (6)                                    the row's cosine with the seeds' prototype and nearest seed, and with
                                                  the reached seeds' (one and two hops)
  typed_rel (10), ordered (10)                    the question's cosine with the relation text on the row's typed edges
                                                  and on the best typed walk from a seed (0 on graphs without types)
Left out as message passing (the GNN track's candidates): nbr_agg (neighbours' frozen features pooled per family), gcs
and typed_v2 (retrieval scores moved over the pool graph).

The round, per split (L-musique and L-hotpotqa, the screen's two fits):
  train   zfs: zrm's model, settings and training with the seven blocks added after the pick's, each entering as every
          block does ([raw, within-pool z, keep flag]); in training each (question, candidate block) is kept with
          probability 1 - P_DROP from the model's own generator (lean_screen's bdrop20, on the candidates only), so a
          masked block reads as the model learned to read its absence. The pick's blocks are never dropped.
  select  both fits' p@swa read on the five training datasets' select carves (in-domain on four, the held-out one
          zero-shot), every one of the 2^7 subsets of the candidates kept and the rest masked: R@5 of each subset minus
          R@5 with every candidate masked, ten reads. The chosen subset has the largest mean of the ten with no read
          below -MIN_DROP; ties go to fewer blocks. Below MIN_GAIN the choice is the empty set and the round stops
          (NO_SELECTION, exit 1). webqsp has no carve outside its evaluation carve and is not in the selection.
  read    each fit on the six s1eval carves with the chosen subset kept and the other candidates masked.
  compare against zrm's screen fit of the split (scr-zrm, scr-zrm-hp), zret's and step 1's beside: lean_screen's
          comparison; pair and re-call under the seed null as every round (relz.py under zfs's name, zrc.py's mapping).

Amended before any number of the round existed (8 October about 21:00; docs/SCREENS.md, twenty-fifth round): one
subset for both models. The GNN track's model zgn (zgnn.py: zfs's model with two message-passing layers over the pool
graph, trained the same way on the same blocks) is read over the same 128 subsets and ten reads, and the chosen subset
is the one with the largest smaller of the two models' mean gains, admissible for both (no read of either below
-MIN_DROP); ties go to fewer blocks; below MIN_GAIN the choice is empty for both (NO_SELECTION). Each model's own best
subset is reported beside. Both models read with the common subset.

    python outputs/mp_unified/zfeat.py build --dataset metaqa --carve fit --host
    python outputs/mp_unified/zfeat.py train --split L-musique --name scr-zfs --arm zfs --device cuda --host
    python outputs/mp_unified/zfeat.py select --fits outputs/screen/fits/scr-zfs,outputs/screen/fits/scr-zfs-hp \\
        --gnn-fits outputs/screen/fits/scr-zgn,outputs/screen/fits/scr-zgn-hp --out outputs/zfeat/select --device cuda --host
    python outputs/mp_unified/zfeat.py read --name scr-zfs --select outputs/zfeat/select.json --device cuda --host
    python outputs/mp_unified/zfeat.py compare --new outputs/screen/fits/scr-zfs \\
        --base outputs/screen/fits/scr-zrm,outputs/screen/fits/scr-zret,outputs/step1/fits/L-musique --out outputs/screen/scr-zfs
    python outputs/mp_unified/zfeat.py pair --screens outputs/screen/scr-zfs.json,outputs/screen/scr-zfs-hp.json \\
        --out outputs/screen/scr-zfs-pair
    python outputs/mp_unified/zfeat.py recall --null N1,N2,N3,N4 --pair outputs/screen/scr-zfs-pair.json \\
        --out outputs/screen/scr-zfs-pair-recall
    python outputs/mp_unified/zfeat.py gate --recall outputs/screen/scr-zfs-pair-recall.json --base outputs/zbase/grade-nullx.json
    python outputs/mp_unified/zfeat.py --selftest

Speed: the candidates are look columns, compiled with the pick's per question. Any latency figure for zfs is cold
(8216ffe): each question compiled and scored from scratch, no warm-up pass, nothing kept from an earlier question.
"""
import os
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import contextlib  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import zrc as ZC  # noqa: E402

sys.modules.setdefault("zfeat", sys.modules[__name__])      # zgnn.py imports this module, never a second copy of it

ZM, RM = ZC.ZM, ZC.RM
Z, R = RM.Z, RM.R
S2, S = RM.S2, RM.S
LG, LC, LM = S.LG, S.LC, S.LM
log = S.log

ARM, BASE_ARM = "zfs", "zrm"
FZ_OUT = ROOT / "outputs" / "zfeat" / "cache"
NAMES = {
    "topo_NER": ("dist1_NER", "dist2_NER", "dist3_NER", "distunreached_NER", "seeds_1hop_NER", "seeds_2hop_frac_NER",
                 "deg_pool_NER", "deg_global_NER", "avail_NER", "seed_component_NER", "component_size_NER"),
    "topo_KNN": ("dist1_KNN", "dist2_KNN", "dist3_KNN", "distunreached_KNN", "seeds_1hop_KNN", "seeds_2hop_frac_KNN",
                 "deg_pool_KNN", "deg_global_KNN", "walks2_KNN", "seed_component_KNN", "component_size_KNN"),
    "topo_FULL": ("dist1_FULL", "dist2_FULL", "dist3_FULL", "distunreached_FULL", "seeds_1hop_FULL",
                  "seeds_2hop_frac_FULL", "deg_pool_FULL", "deg_global_FULL", "walks2_FULL", "seed_component_FULL",
                  "component_size_FULL"),
    "depth_FULL": ("ring_qmean_h1_FULL", "ring_qmax_h1_FULL", "seeds_at_h2_FULL", "support_h2_FULL", "ring_qmean_h2_FULL",
                   "ring_qmax_h2_FULL", "seedproto_h2_FULL", "seeds_at_h3_FULL", "seedmass_h3_FULL", "paths_h3_FULL",
                   "branch_h3_FULL", "support_h3_FULL", "has_h3_FULL", "ring_qmean_h3_FULL", "ring_qmax_h3_FULL",
                   "ring_n_h3_FULL", "seedproto_h3_FULL"),
    "seedcond": ("cos_v_seedproto", "max_cos_v_seed", "cos_v_reachproto", "has_reach_seed", "cos_v_seedproto_h2",
                 "has_seed_h2"),
    "typed_rel": ("relmax_seed", "relmax_in", "relmean_in", "has_typed_edge", "rel_ief", "rel_div", "dir_in_frac",
                  "seed_edges_out", "seed_edges_in", "relchain2_max"),
    "ordered": ("opath_h2_dir1", "opath_h2_adj12", "opath_h3_q1", "opath_h3_q2", "opath_h3_q3", "opath_h3_dir1",
                "opath_h3_dir2", "opath_h3_dir3", "opath_h3_adj12", "opath_h3_adj23"),
}
CAND = tuple(NAMES)
MP_LEFT_OUT = ("nbr_agg", "gcs", "typed_v2")
P_DROP = 0.5
MASK_SEED_OFF = 2401
SEL_CARVES = tuple((d, "select") for d in LG.TRAIN_ORDER)
MIN_DROP = 0.002
MIN_GAIN = 0.001
SETTINGS = {"candidates": list(CAND), "p_drop": P_DROP, "mask_seed_offset": MASK_SEED_OFF,
            "select_carves": [list(x) for x in SEL_CARVES], "min_drop": MIN_DROP, "min_gain": MIN_GAIN,
            "left_out_as_message_passing": list(MP_LEFT_OUT),
            "joint_with": "zgn (zgnn.py): the largest smaller of the two models' mean gains, admissible for both"}
GNN_ARM = "zgn"


# ── build: the candidates' look columns, part by part as step 1's cache holds them ──


def look_cols(head):
    """Each candidate's column indices in the look, in NAMES order; refused unless the look's own block lists them."""
    ci = {c: i for i, c in enumerate(head["columns"])}
    out = {}
    for b, names in NAMES.items():
        missing = [n for n in names if n not in ci]
        if missing:
            raise SystemExit(f"the look has no column {missing}")
        idx = [ci[n] for n in names]
        if [int(i) for i in head["column_blocks"][b]] != idx:
            raise SystemExit(f"the look's {b} block is {head['column_blocks'][b]}, not {names} ({idx})")
        out[b] = idx
    return out


def build(ds, carve, out_root=FZ_OUT, cache_root=LC.OUT, look_root=LM.LOOK, placement=None):
    """relcols.build for the candidates: the same chunks in the same order, step 1's xc and n checked bit for bit."""
    t0 = time.time()
    d, files, _ids, head = LC.look_chunks(ds, carve, look_root)
    cols = look_cols(head)
    xcols, _spans, _ci = LC.xc_columns(head)
    parts, recs = LC.part_dirs(ds, carve, cache_root)
    for p, r in zip(parts, recs):
        if r["look"]["carve_ids_sha256"] != head["carve_ids_sha256"] or r["look"]["records"] != head["records"]:
            raise SystemExit(f"{p}: built from another look than {d}")
        lo, hi = r["chunks"]
        acc = {b: [] for b in CAND}
        xc, n = [], []
        for ch in range(lo, hi):
            z = np.load(files[ch])
            x, qps = z["x"], z["q_pool_size"]
            if x.dtype != np.float16 or x.shape[0] != int(qps.sum()):
                raise SystemExit(f"{files[ch]}: x {x.dtype} {x.shape}, its pools hold {int(qps.sum())} rows")
            for b in CAND:
                acc[b].append(np.ascontiguousarray(x[:, cols[b]]))
            xc.append(x[:, xcols])
            n.append(qps.astype(np.int64))
        arrays = {b: np.concatenate(v) for b, v in acc.items()}
        xc, n = np.concatenate(xc), np.concatenate(n)
        if n.size != r["queries"] or xc.shape[0] != r["rows"] or int(n.sum()) != r["rows"]:
            raise SystemExit(f"{p}: {n.size} queries and {xc.shape[0]} rows against {r['queries']} and {r['rows']}")
        if all((p / f"{k}.npy").exists() for k in ("xc", "n")):
            ref = np.asarray(LC.load_array(p, "xc", r, verify=True))
            if ref.shape != xc.shape or not np.array_equal(ref.view(np.uint16), xc.view(np.uint16)):
                raise SystemExit(f"{p}: the chunks' step-1 columns are not step 1's cached xc bit for bit")
            if not np.array_equal(np.asarray(LC.load_array(p, "n", r, verify=True)).astype(np.int64), n):
                raise SystemExit(f"{p}: the chunks' pool sizes are not step 1's cached n")
            against = "IDENTICAL"
        else:
            against = "step 1's arrays are not on disk here"
        out = Path(out_root) / ds / carve / p.name
        out.mkdir(parents=True, exist_ok=True)
        shas = {b: R.save_npy(out, b, v) for b, v in arrays.items()}
        rec = {"dataset": ds, "carve": carve, "part": p.name, "step1_record_sha256": LC.sha_file(p / "record.json"),
               "chunks": [lo, hi], "queries": int(n.size), "rows": r["rows"], "look": head["records"],
               "carve_ids_sha256": head["carve_ids_sha256"], "columns": {b: list(v) for b, v in NAMES.items()},
               "look_columns": cols, "xc_and_n_against_step1": against,
               "nonzero": {b: [round(float(c), 4) for c in (v != 0).mean(0)] for b, v in arrays.items()},
               "nonfinite": {b: int((~np.isfinite(v)).sum()) for b, v in arrays.items()},
               "arrays": {b: {"dtype": str(v.dtype), "shape": list(v.shape), "sha256": shas[b]} for b, v in arrays.items()},
               "placement": placement or {"where": "laptop"}, "script_sha256": LC.sha_src(__file__),
               "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        R.write_json(out / "record.json", rec)
        nz = {b: round(float(np.mean(v)), 3) for b, v in rec["nonzero"].items()}
        log(f"  {ds}/{carve} {p.name}: {n.size} queries, {r['rows']} rows; xc and n {against}; mean nonzero share "
            f"{nz} ({time.time() - t0:.0f}s)")
    log(f"zfeat build {ds}/{carve}: {len(parts)} part(s) ({time.time() - t0:.0f}s)")
    return 0


def built_records(root=FZ_OUT):
    return {str(f.parent.relative_to(root)).replace("\\", "/"): LC.sha_file(f)
            for f in sorted(Path(root).glob("*/*/part_*/record.json"))}


# ── the carve: step 1's, the candidates beside it, then rmatch's chains ─────


class FeatCarve(S.ARMS["base"][1]):
    """lean_gpu.CacheCarve with a second float16 matrix X2 holding the candidates (this file's build), part by part,
    each built part tied to its step-1 part's record; batch() reads the candidates from X2 (non-finite reads 0) and
    every other block as lean_gpu does. relcols.RelCarve's form."""

    FZ_ROOT = FZ_OUT

    def __init__(self, ds, carve, basis, device, cache_root=LC.OUT, verify=True, score2=False):
        super().__init__(ds, carve, basis, device, cache_root, verify, score2)
        t0 = time.time()
        self.span2, at = {}, 0
        for b in CAND:
            self.span2[b] = (at, at + len(NAMES[b]))
            at += len(NAMES[b])
        self.X2 = torch.empty((int(self.off_np[-1]), at), dtype=torch.float16, device=self.device)
        parts, recs = LC.part_dirs(ds, carve, cache_root)
        self.fz_parts, r_at = {}, 0
        root = Path(type(self).FZ_ROOT)
        for p, r in zip(parts, recs):
            rp = root / ds / carve / p.name
            if not (rp / "record.json").exists():
                raise SystemExit(f"{rp}: no zfeat columns built for {p}")
            rr = json.loads((rp / "record.json").read_text(encoding="utf-8"))
            if rr["step1_record_sha256"] != self.part_shas[p.name] or rr["rows"] != r["rows"] \
                    or r["row_range"][0] != r_at:
                raise SystemExit(f"{rp}: not built from {p}")
            if rr["columns"] != {b: list(v) for b, v in NAMES.items()}:
                raise SystemExit(f"{rp}: built with other columns")
            k = r["rows"]
            for b in CAND:
                f = rp / f"{b}.npy"
                if verify and LC.sha_file(f) != rr["arrays"][b]["sha256"]:
                    raise SystemExit(f"{f}: sha256 is not its record's")
                arr = np.load(f)
                a, e = self.span2[b]
                if arr.shape != (k, e - a) or arr.dtype != np.float16:
                    raise SystemExit(f"{f}: {arr.dtype} {arr.shape}, the carve's {b} is ({k}, {e - a}) float16")
                self.X2[r_at:r_at + k, a:e] = torch.from_numpy(arr).to(self.device)
            self.fz_parts[p.name] = LC.sha_file(rp / "record.json")
            r_at += k
        if r_at != self.X2.shape[0]:
            raise SystemExit(f"{ds}/{carve}: the built parts hold {r_at} rows, the carve {self.X2.shape[0]}")
        for b in CAND:
            self.widths[b] = len(NAMES[b])
        log(f"  {ds}/{carve}: the candidates from {root / ds / carve} ({len(parts)} part(s), {time.time() - t0:.0f}s)")

    def nbytes(self):
        return super().nbytes() + self.X2.numel() * self.X2.element_size()

    def block_values(self, b):
        if b in self.span2:
            a, e = self.span2[b]
            return self.X2[:, a:e]
        return super().block_values(b)

    def batch(self, qs, blocks):
        mine = [b for b in blocks if b in self.span2]
        feats, nq, base_z, gold = super().batch(qs, [b for b in blocks if b not in self.span2])
        if mine:
            qs = np.asarray(qs, np.int64)
            cnt = self.n_np[qs]
            seg = np.cumsum(cnt) - cnt
            idx = np.repeat(self.off_np[qs] - seg, cnt) + np.arange(int(cnt.sum()))
            X2 = torch.nan_to_num(self.X2[torch.from_numpy(idx).to(self.device)].to(torch.float32), nan=0.0,
                                  posinf=0.0, neginf=0.0)
            for b in mine:
                a, e = self.span2[b]
                feats[b] = X2[:, a:e].contiguous()
        return feats, nq, base_z, gold


FeatChainCarve = RM.chain_carve(FeatCarve)


# ── the model ────────────────────────────────────────────────────────────────


class ZFS(ZM.ZRM):
    """zfs: zrm over the pick and the candidates. Training: each (question, candidate) kept with probability
    1 - P_DROP from the model's own generator; the pick's blocks always. Read: the candidates outside READ_KEEP masked
    (READ_KEEP None keeps every block)."""

    READ_KEEP = None

    def __init__(self, blocks, widths, hidden=128, dropout=0.1, seed=0, arm="ctl", ctx="none"):
        super().__init__(blocks, widths, hidden, dropout, seed, arm, ctx)
        self.j_cand = [j for j, b in enumerate(self.blocks) if b in CAND]
        self.mask_seed = int(seed) + MASK_SEED_OFF
        self.gen = None

    def masked(self, keep, kept):
        """keep with every candidate outside kept set to 0."""
        js = [j for j in self.j_cand if self.blocks[j] not in kept]
        if not js:
            return keep
        keep = keep.clone()
        keep[:, js] = 0.0
        return keep

    def keep_of(self, keep):
        """keep as this model reads it: the training draw on the candidates, or READ_KEEP's mask at read."""
        if self.training and self.j_cand:
            if self.gen is None:
                self.gen = torch.Generator(device=keep.device)
                self.gen.manual_seed(self.mask_seed)
            m = (torch.rand((keep.shape[0], len(self.j_cand)), generator=self.gen, device=keep.device)
                 >= P_DROP).to(keep.dtype)
            keep = keep.clone()
            keep[:, self.j_cand] = keep[:, self.j_cand] * m
        elif not self.training and type(self).READ_KEEP is not None:
            keep = self.masked(keep, set(type(self).READ_KEEP))
        return keep

    def forward(self, feats, keep, nq, B, base_z):
        return super().forward(feats, self.keep_of(keep), nq, B, base_z)


S.ARMS.update({ARM: (ZFS, FeatChainCarve)})


@contextlib.contextmanager
def with_sets():
    """lean_gpu's block sets with the candidates after the pick's; restored on exit."""
    saved = {k: list(v) for k, v in LG.SETS.items()}
    try:
        for k in saved:
            LG.SETS[k] = saved[k] + [b for b in CAND if b not in saved[k]]
        yield
    finally:
        for k in saved:
            LG.SETS[k] = saved[k]


@contextlib.contextmanager
def read_keep(kept):
    saved = ZFS.READ_KEEP
    ZFS.READ_KEEP = None if kept is None else tuple(kept)
    try:
        yield
    finally:
        ZFS.READ_KEEP = saved


# ── train, select, read and compare ──────────────────────────────────────────


def stamp(sj, extra):
    if sj.exists():
        rec = json.loads(sj.read_text(encoding="utf-8"))
        rec.update(extra)
        LC.write_json(sj, rec)


def train(argv, split):
    """rmatch.py's train (zrm's settings) for the arm zfs, with the candidates in the sets."""
    if R.arm_of(argv) != ARM:
        raise SystemExit(f"zfeat: train takes --arm {ARM}")
    if S.ARMS.get(ARM) != (ZFS, FeatChainCarve) or S.ARMS.get(BASE_ARM) != (ZM.ZRM, RM.ChainCarveBase):
        raise SystemExit("zfeat: the arms are not zfs's model on its carve and zrm's")
    with with_sets():
        rc = RM.train(argv, split)
    name, out_root = S2.where(argv)
    stamp(out_root / name / "screen.json",
          {"zfeat_sha256": LC.sha_src(__file__), "zrm_sha256": LC.sha_src(ZM.__file__), "zfeat": SETTINGS,
           "zfeat_records": built_records(), "message_passing": False})
    return rc


def objective(table, n_reads):
    """table: {subset (tuple of blocks): [R@5 per read]}, the empty subset included. Returns (chosen subset, its row,
    the rows of every subset): gain = R@5 minus the empty subset's per read; admissible when no read falls below
    -MIN_DROP; the largest mean gain wins, ties to fewer blocks; below MIN_GAIN the empty subset."""
    rows = gain_rows(table, n_reads)
    ok = [r for r in rows if r["admissible"]]
    best = max(ok, key=lambda r: (r["mean_gain"], -len(r["subset"]))) if ok else None
    if best is None or best["mean_gain"] < MIN_GAIN:
        best = next(r for r in rows if not r["subset"])
    return tuple(best["subset"]), best, sorted(rows, key=lambda r: -r["mean_gain"])


def joint(table_m, table_g, n_reads):
    """The common subset of the two models (zfs's table, zgn's): admissible for both, the largest smaller of the two
    mean gains, ties to fewer blocks; below MIN_GAIN the empty subset. Returns (chosen, its joint row, joint rows)."""
    rm = {tuple(r["subset"]): r for r in gain_rows(table_m, n_reads)}
    rg = {tuple(r["subset"]): r for r in gain_rows(table_g, n_reads)}
    if sorted(rm) != sorted(rg):
        raise SystemExit("zfeat select: the two models' tables hold different subsets")
    rows = [{"subset": list(s), "joint_gain": min(rm[s]["mean_gain"], rg[s]["mean_gain"]),
             "mlp_mean_gain": rm[s]["mean_gain"], "gnn_mean_gain": rg[s]["mean_gain"],
             "mlp_min_gain": rm[s]["min_gain"], "gnn_min_gain": rg[s]["min_gain"],
             "admissible": rm[s]["admissible"] and rg[s]["admissible"]} for s in rm]
    ok = [r for r in rows if r["admissible"]]
    best = max(ok, key=lambda r: (r["joint_gain"], -len(r["subset"]))) if ok else None
    if best is None or best["joint_gain"] < MIN_GAIN:
        best = next(r for r in rows if not r["subset"])
    return tuple(best["subset"]), best, sorted(rows, key=lambda r: -r["joint_gain"])


def gain_rows(table, n_reads):
    """Each subset's gains over the empty subset, read by read, and whether it is admissible."""
    base = np.asarray(table[()], np.float64)
    rows = []
    for sub, r5 in table.items():
        g = np.asarray(r5, np.float64) - base
        if g.size != n_reads:
            raise SystemExit(f"zfeat select: subset {sub} has {g.size} reads, not {n_reads}")
        rows.append({"subset": list(sub), "mean_gain": round(float(g.mean()), 6), "min_gain": round(float(g.min()), 6),
                     "gains": [round(float(x), 6) for x in g], "admissible": bool(g.min() >= -MIN_DROP)})
    return rows


@torch.no_grad()
def subset_reads(c, model, blocks, subsets, rows_cap=LG.READ_ROWS):
    """R@5, FC@5 and hit@1 of model on carve c for each subset of the candidates kept (the rest masked)."""
    dev = c.device
    tops = {s: np.zeros(c.rows, np.int64) for s in subsets}
    hits = {s: np.zeros(c.rows, np.int64) for s in subsets}
    for q0, q1 in LG.batches(c.n_np, rows_cap):
        qs = np.arange(q0, q1)
        B = q1 - q0
        feats, nq, base_z, gold = c.batch(qs, blocks)
        cnt = torch.from_numpy(c.n_np[q0:q1]).to(dev)
        seg = torch.cumsum(cnt, 0) - cnt
        ones = torch.ones((B, len(blocks)), dtype=torch.float32, device=dev)
        for s in subsets:
            sc = model(feats, model.masked(ones, set(s)), nq, B, base_z)
            t, h = LG.top_hit(sc, gold, B, seg, cnt)
            tops[s][q0:q1] = t.cpu().numpy()
            hits[s][q0:q1] = h.cpu().numpy()
    ok = c.gt_np > 0
    return {s: [float(x) for x in LG.metrics_of(tops[s], hits[s], c.gt_np)[ok].mean(0)] for s in subsets}


def model_reads(fits, arm, cls, carve_cls, device, verify, subsets):
    """Every subset's (R@5, FC@5, hit@1) on the select carves, fit by fit, for one model; and the reads' records."""
    per_read, reads = {s: [] for s in subsets}, []
    for f in fits:
        fdir = Path(f)
        sj = json.loads((fdir / "screen.json").read_text(encoding="utf-8"))
        if sj.get("arm") != arm:
            raise SystemExit(f"zfeat select: {fdir} is arm {sj.get('arm')}'s, not {arm}'s")
        blob = torch.load(fdir / "models.pt", weights_only=False)
        with with_sets():
            models = [x for x in LG.load_models(blob, device, cls=cls) if x[0] == "p@swa"]
        if len(models) != 1:
            raise SystemExit(f"zfeat select: {fdir} has no single p@swa")
        _n, model, blocks = models[0]
        missing = [b for b in CAND if b not in blocks]
        if missing:
            raise SystemExit(f"zfeat select: {fdir}'s p@swa does not read {missing} (dead in training?)")
        train = [d for d, _cv in blob["train"]]
        for ds, cv in SEL_CARVES:
            tc = time.time()
            c = carve_cls(ds, cv, blob["basis"], device, LC.OUT, verify)
            if c.basis_sha256 != blob["basis_sha256"]:
                raise SystemExit(f"{ds}/{cv}: the cache's basis is not the fit's")
            m = subset_reads(c, model, blocks, subsets)
            for s in subsets:
                per_read[s].append(m[s])
            reads.append({"model": arm, "fit": fdir.name, "split": sj.get("split"), "dataset": ds, "carve": cv,
                          "zero_shot": ds not in train, "questions": c.rows, "with_gold": int((c.gt_np > 0).sum()),
                          "none": [round(x, 4) for x in m[()]], "all": [round(x, 4) for x in m[tuple(CAND)]]})
            log(f"  {arm} {fdir.name} {ds}={cv}{' (zero-shot)' if ds not in train else ''}: {c.rows} questions; R@5 "
                f"none {m[()][0]:.4f}, all {m[tuple(CAND)][0]:.4f} ({time.time() - tc:.0f}s)")
            del c
            if str(device).startswith("cuda"):
                torch.cuda.empty_cache()
    return per_read, reads


def select(fits, out, device, host=False, verify=True, gnn_fits=()):
    t0 = time.time()
    flags = LG.set_flags(device)
    LG.bind_device_ops()
    placement = None
    if host:
        import lean_host as LH
        placement = LH.substitute()
    if len(gnn_fits) != len(fits):
        raise SystemExit("zfeat select: one zgn fit per zfs fit (the joint selection, twenty-fifth round)")
    import zgnn as ZG
    subsets = [()] + [tuple(b for b in CAND if b in comb) for k in range(1, len(CAND) + 1)
                      for comb in itertools.combinations(CAND, k)]
    per_m, reads_m = model_reads(fits, ARM, ZFS, FeatChainCarve, device, verify, subsets)
    per_g, reads_g = model_reads(gnn_fits, GNN_ARM, ZG.ZGNN, ZG.GNNCarve, device, verify, subsets)
    if [(r["split"], r["dataset"]) for r in reads_m] != [(r["split"], r["dataset"]) for r in reads_g]:
        raise SystemExit("zfeat select: the two models' reads are not the same splits and carves")
    n = len(reads_m)
    tm = {s: [r[0] for r in per_m[s]] for s in subsets}
    tg = {s: [r[0] for r in per_g[s]] for s in subsets}
    chosen, best, rows = joint(tm, tg, n)
    own = {}
    for arm, tab, per in ((ARM, tm, per_m), (GNN_ARM, tg, per_g)):
        ch, bst, rws = objective(tab, n)
        own[arm] = {"chosen_alone": list(ch), "best_alone": bst,
                    "subsets": [dict(r, fc5=[round(x[1], 4) for x in per[tuple(r["subset"])]],
                                     hit1=[round(x[2], 4) for x in per[tuple(r["subset"])]]) for r in rws],
                    "single_blocks": {b: next(r for r in rws if r["subset"] == [b]) for b in CAND}}
    verdict = "SELECTED" if chosen else "NO_SELECTION"
    rec = {"round": "twenty-fourth (joint with the twenty-fifth)", "arm": ARM, "gnn_arm": GNN_ARM, "verdict": verdict,
           "chosen": list(chosen), "best": best, "joint": rows, "models": own, "reads": reads_m + reads_g,
           "settings": SETTINGS, "fits": [Path(f).name for f in fits], "gnn_fits": [Path(f).name for f in gnn_fits],
           "flags": flags, "placement": placement, "zfeat_sha256": LC.sha_src(__file__),
           "zgnn_sha256": LC.sha_src(ZG.__file__), "seconds": round(time.time() - t0, 1)}
    out = Path(out)
    LC.write_json(out.with_suffix(".json"), rec)
    lines = [f"# Rounds twenty-four and twenty-five: the common selection, {verdict}", "",
             f"Chosen for both models: {', '.join(chosen) if chosen else 'none'}. Mean R@5 gain over every candidate "
             f"masked: zfs {best['mlp_mean_gain']:+.4f} (lowest read {best['mlp_min_gain']:+.4f}), zgn "
             f"{best['gnn_mean_gain']:+.4f} (lowest {best['gnn_min_gain']:+.4f}); ten select reads per model, two fits.",
             "", f"Each model alone would choose: zfs {'+'.join(own[ARM]['chosen_alone']) or 'none'} "
             f"({own[ARM]['best_alone']['mean_gain']:+.4f}), zgn {'+'.join(own[GNN_ARM]['chosen_alone']) or 'none'} "
             f"({own[GNN_ARM]['best_alone']['mean_gain']:+.4f}).", "",
             "| subset | smaller mean gain | zfs mean | zfs lowest | zgn mean | zgn lowest | admissible |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for r in rows[:15]:
        lines.append(f"| {'+'.join(r['subset']) or 'none'} | {r['joint_gain']:+.4f} | {r['mlp_mean_gain']:+.4f} | "
                     f"{r['mlp_min_gain']:+.4f} | {r['gnn_mean_gain']:+.4f} | {r['gnn_min_gain']:+.4f} | "
                     f"{'yes' if r['admissible'] else 'no'} |")
    lines += ["", "Each block alone:", "", "| block | zfs mean | zfs lowest | zgn mean | zgn lowest |",
              "| --- | --- | --- | --- | --- |"]
    for b in CAND:
        m_, g_ = own[ARM]["single_blocks"][b], own[GNN_ARM]["single_blocks"][b]
        lines.append(f"| {b} | {m_['mean_gain']:+.4f} | {m_['min_gain']:+.4f} | {g_['mean_gain']:+.4f} | "
                     f"{g_['min_gain']:+.4f} |")
    lines += ["", "| model | fit | read | zero-shot | R@5 none | R@5 all |", "| --- | --- | --- | --- | --- | --- |"]
    for r in reads_m + reads_g:
        lines.append(f"| {r['model']} | {r['fit']} | {r['dataset']}={r['carve']} | {'yes' if r['zero_shot'] else ''} | "
                     f"{r['none'][0]:.4f} | {r['all'][0]:.4f} |")
    out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"zfeat select: {verdict}; chosen {list(chosen)}; mean gains zfs {best['mlp_mean_gain']:+.4f}, zgn "
        f"{best['gnn_mean_gain']:+.4f} ({time.time() - t0:.0f}s)")
    return 0 if chosen else 1


def load_selection(path):
    rec = json.loads(Path(path).read_text(encoding="utf-8"))
    if rec.get("arm") != ARM or rec.get("settings") != SETTINGS or rec.get("verdict") != "SELECTED":
        raise SystemExit(f"zfeat: {path} is not this round's selection ({rec.get('verdict')})")
    return tuple(rec["chosen"]), LC.sha_file(path)


def read(argv):
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--select", required=True)
    k, rest = ap.parse_known_args(argv)
    kept, sha = load_selection(k.select)
    name, out_root = S2.where(rest)
    arm = json.loads((out_root / name / "screen.json").read_text(encoding="utf-8"))["arm"]
    if arm != ARM:
        raise SystemExit(f"zfeat: {name} was trained as {arm}, not {ARM}")
    with with_sets(), read_keep(kept):
        rc = RM.read(rest)
    LC.write_json(out_root / name / "reads" / "zfeat_read.json",
                  {"kept": list(kept), "masked": [b for b in CAND if b not in kept], "selection_sha256": sha,
                   "zfeat_sha256": LC.sha_src(__file__)})
    return rc


def compare(rest):
    return S2.main(["compare"] + rest)


# ── relz.py's pair, re-call and grade, decided against zrm's fits ────────────


@contextlib.contextmanager
def on_zrm():
    with ZC.on_zrm():
        Z.ARM = ARM
        yield


FIX = (("(rel's screen fits scr-rel and scr-rel-hp)", "(zrm's screen fits scr-zrm and scr-zrm-hp)"),
       ("(rel's screen fits)", "(zrm's screen fits)"),
       ("| rel R@5 | relz R@5 |", "| zrm R@5 | zfs R@5 |"),
       ("section 2 and the tenth round", "section 2 and the twenty-fourth round"),
       ("docs/FULL_ROUND10.md", "docs/FULL_ROUND24.md"))


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
        rec.update({"decided_against": "zrm's fit of each split", "round": "twenty-fourth",
                    "zfeat_sha256": LC.sha_src(__file__), "message_passing": False})
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
    """0 when the screen's re-call is PROMISING and zrm is the base, 1 otherwise; 2 on a missing or foreign file."""
    got = []
    for fn, arm, want in ((recall_file, ARM, "zrm's fit of each split"), (base_file, "zrm", "zrs's fit of each split")):
        p = Path(fn)
        if not p.exists():
            raise SystemExit(f"zfeat gate: {p} missing")
        rec = json.loads(p.read_text(encoding="utf-8"))
        if rec.get("arm") != arm or rec.get("decided_against") != want:
            raise SystemExit(f"zfeat gate: {p} is arm {rec.get('arm')}'s, decided against "
                             f"{rec.get('decided_against')!r}; not the declared record")
        got.append(rec.get("verdict"))
    ok = got[0] == "PROMISING" and got[1] == "ADOPT"
    log(f"zfeat gate: the re-call {got[0]}, zrm against zrs {got[1]} -> {'the full run starts' if ok else 'no full run'}")
    return 0 if ok else 1


# ── selftest ─────────────────────────────────────────────────────────────────


def selftest():
    t0 = time.time()
    torch.manual_seed(0)
    # the model: masks touch the candidates only; READ_KEEP masks at read; training draws from its own generator
    blocks = ["rank", "SEMB", "topo_NER", "seedcond"]
    assert all(b in CAND for b in blocks[2:]) and not any(b in CAND for b in LG.SETS["pick"])
    assert not set(CAND) & set(MP_LEFT_OUT)
    m = ZFS.__new__(ZFS)
    torch.nn.Module.__init__(m)
    m.blocks, m.j_cand, m.mask_seed, m.gen = blocks, [2, 3], 7, None
    ones = torch.ones(5, 4)
    k = m.masked(ones, {"seedcond"})
    assert torch.equal(k[:, :2], ones[:, :2]) and float(k[:, 2].abs().sum()) == 0 and torch.equal(k[:, 3], ones[:, 3])
    assert m.masked(ones, set(CAND)) is ones
    k0 = m.masked(ones, set())
    assert float(k0[:, 2:].sum()) == 0 and float(k0[:, :2].sum()) == 10
    # the objective: largest admissible mean gain, ties to fewer blocks, an inadmissible subset never chosen
    table = {(): [0.5, 0.5, 0.5], ("a",): [0.51, 0.51, 0.499], ("b",): [0.52, 0.52, 0.49], ("a", "b"): [0.51, 0.51, 0.499]}
    ch, best, rows = objective(table, 3)
    assert ch == ("a",) and best["admissible"] and abs(best["mean_gain"] - 0.019 / 3) < 1e-6
    assert not next(r for r in rows if r["subset"] == ["b"])["admissible"]
    ch2, _b, _r = objective({(): [0.5, 0.5], ("a",): [0.5004, 0.5004]}, 2)
    assert ch2 == ()                                                         # below MIN_GAIN: nothing chosen
    # the joint choice: admissible for both, the largest smaller mean gain ('b' is best for the MLP alone, but the
    # GNN loses on it; 'a' helps both)
    tg = {(): [0.6, 0.6, 0.6], ("a",): [0.605, 0.606, 0.604], ("b",): [0.59, 0.6, 0.6], ("a", "b"): [0.6, 0.6, 0.6]}
    tm = {(): [0.5, 0.5, 0.5], ("a",): [0.505, 0.505, 0.505], ("b",): [0.52, 0.52, 0.52], ("a", "b"): [0.52, 0.52, 0.52]}
    cj, bj, rj = joint(tm, tg, 3)
    assert cj == ("a",) and abs(bj["joint_gain"] - 0.005) < 1e-6 and objective(tm, 3)[0] == ("b",)
    assert not next(r for r in rj if r["subset"] == ["b"])["admissible"]
    assert joint({(): [0.5], ("a",): [0.51]}, {(): [0.5], ("a",): [0.5005]}, 1)[0] == ()
    # the sets: the candidates after the pick, restored after
    before = {k: list(v) for k, v in LG.SETS.items()}
    with with_sets():
        assert LG.SETS["pick"] == before["pick"] + list(CAND)
    assert LG.SETS == before
    with read_keep(("topo_NER",)):
        assert ZFS.READ_KEEP == ("topo_NER",)
    assert ZFS.READ_KEEP is None
    assert sum(len(v) for v in NAMES.values()) == 76 and len(CAND) == 7
    assert S.ARMS[ARM] == (ZFS, FeatChainCarve) and issubclass(FeatChainCarve, FeatCarve)
    # a toy fit through lean_gpu's loop on rmatch's toy carve with two candidates beside it, then the subset reads
    import shutil
    import tempfile
    LG.bind_device_ops()
    tmp = Path(tempfile.mkdtemp(prefix="zfeat_"))
    try:
        rng = np.random.default_rng(24)
        RM.toy_roots(tmp, rng, chunks=(12, 12))
        RM.build("metaqa", "toy", out_root=tmp / "ch", cache_root=tmp / "cache", look_root=tmp / "look",
                 rel_dir=tmp / "rel")
        base = RM.chain_carve(ZM.ToyZ, tmp / "ch", tmp / "rel")("metaqa", "toy", "2wiki", "cpu", tmp / "cache")

        class WithCand:
            def __init__(self, c):
                self.c, self.ds, self.carve, self.chains, self.device = c, c.ds, c.carve, c.chains, c.device
                self.rows, self.n_np, self.off_np = c.rows, c.n_np, c.off_np
                gb = c.gold.cpu().numpy().astype(np.int64)
                self.gt_np = np.array([gb[a:e].sum() for a, e in zip(c.off_np[:-1], c.off_np[1:])], np.int64)
                self.widths = dict(c.widths, topo_NER=11, seedcond=6)
                g = torch.Generator().manual_seed(5)
                gold = c.gold.to(torch.float32).unsqueeze(1)
                self.X2 = torch.cat([torch.randn(gold.shape[0], 11, generator=g) + 2.0 * gold,
                                     torch.randn(gold.shape[0], 6, generator=g)], 1)

            def nbytes(self):
                return 0

            def batch(self, qs, blocks):
                feats, nq, bz, gold = self.c.batch(qs, [b for b in blocks if b not in CAND])
                cnt = self.n_np[qs]
                seg = np.cumsum(cnt) - cnt
                idx = torch.from_numpy(np.repeat(self.off_np[qs] - seg, cnt) + np.arange(int(cnt.sum())))
                if "topo_NER" in blocks:
                    feats["topo_NER"] = self.X2[idx, :11].contiguous()
                if "seedcond" in blocks:
                    feats["seedcond"] = self.X2[idx, 11:].contiguous()
                return feats, nq, bz, gold

        wc = WithCand(base)
        tb = ["rank", "SEMB", "topo_NER", "seedcond"]
        cfg = {"lr": 2e-2, "wd": 1e-4, "dropout": 0.1, "epochs": 3, "swa_from": 1, "cos": False, "adamw": False,
               "drop": False}
        with S.patched(ARM):
            f1 = LG.fit_variant([wc], tb, cfg, 0, 16, "none", "cpu", tag="toy-zfs")
        assert all(bool(torch.isfinite(v).all()) for v in f1["swa"].values() if v.is_floating_point())
        net = ZFS(tb, wc.widths, 16, dropout=0.1, seed=0)
        net.load_state_dict(f1["swa"])
        net.eval()
        subs = [(), ("topo_NER",), ("seedcond",), ("topo_NER", "seedcond")]
        got = subset_reads(wc, net, tb, subs, rows_cap=4096)
        assert all(len(v) == 3 and all(np.isfinite(v)) for v in got.values())
        assert got[("topo_NER",)][2] > got[()][2], got                      # the planted column helps hit@1 when kept
        with read_keep(("topo_NER", "seedcond")):
            qs = np.arange(min(wc.rows, 6))
            feats, nq, bz, _g = wc.batch(qs, tb)
            one = torch.ones(qs.size, len(tb))
            assert torch.equal(net(feats, one, nq, qs.size, bz), net(feats, net.masked(one, set(CAND)), nq, qs.size, bz))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"selftest: masks touch the candidates only, the objective picks the best admissible subset (ties to fewer "
          f"blocks, below {MIN_GAIN} none), the sets restore ({time.time() - t0:.1f}s)")
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
    if k.cmd == "build":
        bp = argparse.ArgumentParser()
        bp.add_argument("--dataset", required=True)
        bp.add_argument("--carve", required=True)
        bp.add_argument("--host", action="store_true")
        b = bp.parse_args(rest)
        placement = None
        if b.host:
            import lean_host as LH
            placement = LH.substitute()
        return build(b.dataset, b.carve, placement=placement)
    if k.cmd == "train":
        return train([k.cmd] + rest, k.split)
    if k.cmd == "select":
        sp = argparse.ArgumentParser()
        sp.add_argument("--fits", required=True)
        sp.add_argument("--gnn-fits", required=True)
        sp.add_argument("--out", required=True)
        sp.add_argument("--device", default="cuda")
        sp.add_argument("--host", action="store_true")
        sp.add_argument("--no-verify", action="store_true")
        s = sp.parse_args(rest)
        return select(s.fits.split(","), s.out, s.device, s.host, not s.no_verify, s.gnn_fits.split(","))
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
            log(f"zfeat gate: {e}")
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
            log(f"zfeat {k.cmd}: {e}")
            return 2
        return 0
    ap.error("zfeat: build, train, select, read, compare, pair, recall, grade or gate")


if __name__ == "__main__":
    sys.exit(main())
