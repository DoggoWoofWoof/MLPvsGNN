"""Screens, ninth round (docs/SCREENS.md): compiled columns that step 1's pick leaves out and the published systems use,
on lean_screen2's commands and lean_screen's rule, unchanged; two fits per screen (L-musique and L-hotpotqa), their
verdict over both by screen_pair.py, re-called under the seed null by screen_recall.py.

    python outputs/mp_unified/relcols.py build --dataset metaqa --carve fit --host
    python outputs/mp_unified/relcols.py smoke --device cuda --host
    python outputs/mp_unified/relcols.py train --split L-musique --name scr-rel --arm rel --device cuda --host
    python outputs/mp_unified/relcols.py read --name scr-rel --device cuda --host
    python outputs/mp_unified/relcols.py compare --new outputs/screen/fits/scr-rel \\
        --base outputs/step1/fits/L-musique --out outputs/screen/scr-rel
    python outputs/mp_unified/relcols.py --selftest

The arms add blocks of the look's 129 compiled columns (the twin's and the GNN's input contract) that step 1's pick does
not read. Each enters the model as every block does, [raw, within-pool z, keep flag], after step 1's nine blocks; the
loop, the batches, the loss and every other block are step 1's. The values are the look's float16 columns, row for row,
as step 1's cached columns are (lean_cache.build_part copies its xc from the same chunks the same way).
  rel   the question-relation columns, 26 in three blocks, all from the frozen encoder's embeddings of the question and
        of each relation's text. On a graph without typed relations every one of them is 0.
        typed_rel (10)  relmax_seed, relmax_in, relmean_in: cos(q, e_r) over the typed edges to a seed / incident to v
                        (max, max, mean); has_typed_edge; rel_ief (max over incident edges of log((1 + n) / (1 + the
                        relation's corpus count))); rel_div; dir_in_frac; seed_edges_out and seed_edges_in;
                        relchain2_max (max over chains seed - x - v of the mean cos(q, e_r) of the two steps)
        typed_v2 (6)    qsupport_h2, qsupport_h3: (W_q^t s)_v, retrieval mass s = rrf / max rrf moved t steps over the
                        typed structural in-edges, each weighted exp(cos(q, e_r) / 0.1) and normalised; typed_walks_h3;
                        relpath_max/mean/min_h3: over typed walks of length 3 from a seed, the mean cos(q, e_r) (max,
                        mean) and the min along the walk (max)
        ordered (10)    the best typed walk to v (the one relpath_max scores), position by position: cos(q, e_r) at
                        each step of the 3-step walk, the direction each step traverses its stored relation (+1 / -1),
                        and the cosine of consecutive relations' embeddings (opath_h2_dir1, opath_h2_adj12,
                        opath_h3_q1-3, opath_h3_dir1-3, opath_h3_adj12, opath_h3_adj23)
  gcs   parameter-free propagation of retrieval over the pool graph (2): gcs_full and gcs_struct, max(p_2, s) with
        p_{t+1} = 0.5 s + 0.5 W p_t, s = rrf / max rrf in the pool, W the row-normalised symmetric pool graph over every
        edge family (FULL) or the structural edges (STRUCT): a two-step personalised PageRank from retrieval's scores.

build      the four blocks for one carve, part by part as step 1's cache holds it (the same look chunks in the same
           order), as float16 under outputs/relcols/cache/<ds>/<carve>/<part>/<block>.npy with a record tying each part
           to its step-1 part (that part's record sha256). The same chunks' step-1 columns must be step 1's cached xc
           bit for bit, and their pool sizes its n, wherever those arrays are on disk.
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
ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import lean_screen2 as S2  # noqa: E402

S = S2.S
LG, LC, LM = S.LG, S.LC, S.LM
log = S.log
RC_OUT = ROOT / "outputs" / "relcols" / "cache"
NAMES = {
    "gcs": ("gcs_full", "gcs_struct"),
    "typed_rel": ("relmax_seed", "relmax_in", "relmean_in", "has_typed_edge", "rel_ief", "rel_div", "dir_in_frac",
                  "seed_edges_out", "seed_edges_in", "relchain2_max"),
    "typed_v2": ("qsupport_h2", "qsupport_h3", "typed_walks_h3", "relpath_max_h3", "relpath_mean_h3", "relpath_min_h3"),
    "ordered": ("opath_h2_dir1", "opath_h2_adj12", "opath_h3_q1", "opath_h3_q2", "opath_h3_q3", "opath_h3_dir1",
                "opath_h3_dir2", "opath_h3_dir3", "opath_h3_adj12", "opath_h3_adj23"),
}
BLOCKS = tuple(NAMES)
ARM_BLOCKS = {"rel": ("typed_rel", "typed_v2", "ordered"), "gcs": ("gcs",)}
SMOKE_CARVES = (("metaqa", "select"), ("webqsp", "s1eval"), ("2wiki", "select"))


# ── build ────────────────────────────────────────────────────────────────────


def look_cols(head):
    """Each block's column indices in the look, in NAMES order; refused unless the look's own block lists them."""
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


def replace_retried(src, dst, tries=60):
    """os.replace, retried for about two minutes (WinError 32 on the host: hubwalk.py, 7 October 21:23)."""
    for k in range(tries):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if k == tries - 1:
                raise
            time.sleep(2)


def save_npy(d, name, a):
    tmp = d / f"{name}.tmp.npy"
    np.save(tmp, np.ascontiguousarray(a))
    replace_retried(tmp, d / f"{name}.npy")
    return LC.sha_file(d / f"{name}.npy")


def write_json(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1), encoding="utf-8")
    replace_retried(tmp, p)


def build(ds, carve, out_root=RC_OUT, cache_root=LC.OUT, look_root=LM.LOOK, placement=None):
    t0 = time.time()
    d, files, _ids, head = LC.look_chunks(ds, carve, look_root)
    cols = look_cols(head)
    xcols, _spans, _ci = LC.xc_columns(head)
    parts, recs = LC.part_dirs(ds, carve, cache_root)
    summary = {}
    for p, r in zip(parts, recs):
        if r["look"]["carve_ids_sha256"] != head["carve_ids_sha256"] or r["look"]["records"] != head["records"]:
            raise SystemExit(f"{p}: built from another look than {d}")
        lo, hi = r["chunks"]
        acc = {b: [] for b in BLOCKS}
        xc, n = [], []
        for ch in range(lo, hi):
            z = np.load(files[ch])
            x, qps = z["x"], z["q_pool_size"]
            if x.dtype != np.float16 or x.shape[0] != int(qps.sum()):
                raise SystemExit(f"{files[ch]}: x {x.dtype} {x.shape}, its pools hold {int(qps.sum())} rows")
            for b in BLOCKS:
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
        shas = {b: save_npy(out, b, v) for b, v in arrays.items()}
        rec = {"dataset": ds, "carve": carve, "part": p.name, "step1_record_sha256": LC.sha_file(p / "record.json"),
               "chunks": [lo, hi], "queries": int(n.size), "rows": r["rows"], "look": head["records"],
               "carve_ids_sha256": head["carve_ids_sha256"], "columns": {b: list(v) for b, v in NAMES.items()},
               "look_columns": cols, "xc_and_n_against_step1": against,
               "nonzero": {b: [round(float(c), 4) for c in (v != 0).mean(0)] for b, v in arrays.items()},
               "nonfinite": {b: int((~np.isfinite(v)).sum()) for b, v in arrays.items()},
               "arrays": {b: {"dtype": str(v.dtype), "shape": list(v.shape), "sha256": shas[b]} for b, v in arrays.items()},
               "placement": placement or {"where": "laptop"}, "script_sha256": LC.sha_src(__file__),
               "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        write_json(out / "record.json", rec)
        summary[p.name] = {"queries": int(n.size), "rows": r["rows"], "against": against}
        typed = float(np.mean([c != 0 for c in rec["nonzero"]["typed_rel"]]))
        log(f"  {ds}/{carve} {p.name}: {n.size} queries, {r['rows']} rows; xc and n {against}; typed_rel columns with "
            f"a nonzero row {typed:.2f}; gcs nonzero {rec['nonzero']['gcs']} ({time.time() - t0:.0f}s)")
    log(f"relcols build {ds}/{carve}: {len(parts)} part(s) ({time.time() - t0:.0f}s)")
    return summary


# ── the carves ───────────────────────────────────────────────────────────────


class RelCarve(LG.CacheCarve):
    """lean_gpu.CacheCarve, and beside its matrix a second float16 matrix X2 with the arm's blocks (EXTRA) from this
    file's build, part by part; each built part must name its step-1 part's record (sha256) and rows. batch() and
    block_values() read the arm's blocks from X2 and every other block as lean_gpu does."""

    EXTRA = ()

    def __init__(self, ds, carve, basis, device, cache_root=LC.OUT, verify=True, score2=False):
        super().__init__(ds, carve, basis, device, cache_root, verify, score2)
        t0 = time.time()
        self.span2, at = {}, 0
        for b in self.EXTRA:
            self.span2[b] = (at, at + len(NAMES[b]))
            at += len(NAMES[b])
        self.X2 = torch.empty((int(self.off_np[-1]), at), dtype=torch.float16, device=self.device)
        parts, recs = LC.part_dirs(ds, carve, cache_root)
        self.rc_parts, r_at = {}, 0
        for p, r in zip(parts, recs):
            rp = RC_OUT / ds / carve / p.name
            rr = json.loads((rp / "record.json").read_text(encoding="utf-8"))
            if rr["step1_record_sha256"] != self.part_shas[p.name] or rr["rows"] != r["rows"] \
                    or r["row_range"][0] != r_at:
                raise SystemExit(f"{rp}: not built from {p}")
            k = r["rows"]
            for b in self.EXTRA:
                f = rp / f"{b}.npy"
                if verify and LC.sha_file(f) != rr["arrays"][b]["sha256"]:
                    raise SystemExit(f"{f}: sha256 is not its record's")
                arr = np.load(f)
                a, e = self.span2[b]
                if arr.shape != (k, e - a) or arr.dtype != np.float16:
                    raise SystemExit(f"{f}: {arr.dtype} {arr.shape}, the carve's {b} is ({k}, {e - a}) float16")
                self.X2[r_at:r_at + k, a:e] = torch.from_numpy(arr).to(self.device)
            self.rc_parts[p.name] = LC.sha_file(rp / "record.json")
            r_at += k
        if r_at != self.X2.shape[0]:
            raise SystemExit(f"{ds}/{carve}: the built parts hold {r_at} rows, the carve {self.X2.shape[0]}")
        for b in self.EXTRA:
            self.widths[b] = len(NAMES[b])
        log(f"  {ds}/{carve}: {list(self.EXTRA)} from {RC_OUT / ds / carve} ({len(parts)} part(s), "
            f"{time.time() - t0:.0f}s)")

    def nbytes(self):
        return super().nbytes() + self.X2.numel() * self.X2.element_size()

    def block_values(self, b):
        if b in self.span2:
            a, e = self.span2[b]
            return self.X2[:, a:e]
        return super().block_values(b)

    def batch(self, qs, blocks):
        """lean_gpu.CacheCarve.batch for the other blocks, unchanged; the arm's blocks from X2's same rows, cleaned as
        lean_gpu cleans X (a non-finite value reads 0)."""
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


class RelCarveRel(RelCarve):
    EXTRA = ARM_BLOCKS["rel"]


class RelCarveGcs(RelCarve):
    EXTRA = ARM_BLOCKS["gcs"]


CARVES = {"rel": RelCarveRel, "gcs": RelCarveGcs}
S.ARMS.update({arm: (S.ARMS["base"][0], c) for arm, c in CARVES.items()})


@contextlib.contextmanager
def with_sets(arm):
    """lean_gpu's block sets (train_cmd's and read_carve's) with the arm's blocks after step 1's; restored on exit."""
    saved = {k: list(v) for k, v in LG.SETS.items()}
    try:
        for k in saved:
            LG.SETS[k] = saved[k] + [b for b in ARM_BLOCKS.get(arm, ()) if b not in saved[k]]
        yield
    finally:
        for k in saved:
            LG.SETS[k] = saved[k]


# ── train and read (lean_screen2's, under the arm's sets) ────────────────────


def built_records():
    return {str(f.parent.relative_to(RC_OUT)).replace("\\", "/"): LC.sha_file(f)
            for f in sorted(RC_OUT.glob("*/*/part_*/record.json"))}


def arm_of(argv):
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--arm", default="base")
    k, _ = ap.parse_known_args(argv)
    return k.arm


def train(argv, split):
    with with_sets(arm_of(argv)):
        rc = S2.train(argv, split)
    name, out_root = S2.where(argv)
    sj = out_root / name / "screen.json"
    if sj.exists():
        rec = json.loads(sj.read_text(encoding="utf-8"))
        rec["relcols_sha256"] = LC.sha_src(__file__)
        rec["relcols_blocks"] = list(ARM_BLOCKS.get(rec.get("arm"), ()))
        rec["relcols_records"] = built_records()
        LC.write_json(sj, rec)
    return rc


def read(argv):
    name, out_root = S2.where(argv)
    arm = json.loads((out_root / name / "screen.json").read_text(encoding="utf-8"))["arm"]
    with with_sets(arm):
        return S2.main(["read"] + argv)


# ── smoke ────────────────────────────────────────────────────────────────────


def carve_check(ds, carve, device):
    """One carve through each arm's carve against lean_gpu.CacheCarve: the base matrix, golds and rows unchanged, the
    arm's columns its build's, row for row."""
    a = LG.CacheCarve(ds, carve, "2wiki", device)
    parts, _recs = LC.part_dirs(ds, carve)
    out = {}
    for arm, cls in CARVES.items():
        b = cls(ds, carve, "2wiki", device)
        ref = np.concatenate([np.concatenate([np.load(RC_OUT / ds / carve / p.name / f"{blk}.npy") for blk in cls.EXTRA], 1)
                              for p in parts])
        out[arm] = {"base_equal": bool(torch.equal(a.X, b.X) and torch.equal(a.gold, b.gold) and torch.equal(a.row, b.row)),
                    "columns_equal": bool(torch.equal(b.X2, torch.from_numpy(ref).to(b.device))),
                    "nonzero": [round(float(v), 4) for v in (b.X2 != 0).to(torch.float32).mean(0).tolist()]}
        del b
    del a
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return all(v["base_equal"] and v["columns_equal"] for v in out.values()), out


def smoke(device, host, out_root=None):
    """The carve check on metaqa select, webqsp s1eval and 2wiki select; then each arm trained for one epoch twice on
    metaqa's select carve (the repeat must be IDENTICAL, the arm's blocks live in the fit) and read on it."""
    t0 = time.time()
    root = Path(out_root or S.OUT / "smoke9")
    h = ["--host"] if host else []
    rec, ok = {"carves": {}, "arms": {}}, True
    for ds, cv in SMOKE_CARVES:
        good, chk = carve_check(ds, cv, device)
        rec["carves"][f"{ds}={cv}"] = chk
        ok = ok and good
    for arm in CARVES:
        rc = main(["train", "--name", f"smoke-{arm}", "--arm", arm, "--train", "metaqa=select", "--basis", "2wiki",
                   "--variants", "p", "--config", "2e-3:1e-4:0.1:1:0", "--repeat", "2", "--device", device,
                   "--out-root", str(root)] + h)
        tj = json.loads((root / f"smoke-{arm}" / "train.json").read_text(encoding="utf-8"))
        blocks = tj["variants"]["p"]["blocks"]
        live = all(b in blocks for b in ARM_BLOCKS[arm])
        rr = main(["read", "--name", f"smoke-{arm}", "--read", "metaqa=select", "--device", device, "--out-root",
                   str(root)] + h)
        rec["arms"][arm] = {"train_rc": rc, "repeat": "IDENTICAL" if rc == 0 else "DIFFERENT", "blocks": blocks,
                            "arm_blocks_live": live, "read_rc": rr}
        ok = ok and rc == 0 and rr == 0 and live
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
    rec.update({"ok": ok, "seconds": time.time() - t0, "script_sha256": LC.sha_src(__file__)})
    LC.write_json(root / "smoke.json", rec)
    log(f"smoke9: {json.dumps(rec)}; {'ok' if ok else 'FAILED'}")
    return 0 if ok else 1


# ── selftest ─────────────────────────────────────────────────────────────────


def fake_carve(cls):
    """A three-question carve of cls on the CPU: base columns rank (5) and WALK (2), the arm's blocks after; column 0
    of each matrix holds the row id; one non-finite value in X2."""
    c = object.__new__(cls)
    c.ds, c.carve, c.device = "dsx", "cv", torch.device("cpu")
    c.n_np = np.array([4, 2, 3], np.int64)
    c.off_np = np.concatenate([[0], np.cumsum(c.n_np)])
    R = int(c.off_np[-1])
    X = torch.zeros(R, 7)
    X[:, 0] = torch.arange(R, dtype=torch.float32)
    X[:, 2] = torch.linspace(0.0, 1.0, R)
    X[:, 5:7] = torch.randn(R, 2, generator=torch.Generator().manual_seed(0))
    c.X = X.half()
    c.gold = torch.tensor([0, 1, 0, 0, 1, 0, 0, 0, 1], dtype=torch.bool)
    c.span, c.c_rrf = {"rank": (0, 5), "WALK": (5, 7)}, 2
    c.span2, at = {}, 0
    for b in cls.EXTRA:
        c.span2[b] = (at, at + len(NAMES[b]))
        at += len(NAMES[b])
    X2 = torch.randn(R, at, generator=torch.Generator().manual_seed(1))
    X2[:, 0] = torch.arange(R, dtype=torch.float32)
    X2[3, 1] = float("nan")
    c.X2 = X2.half()
    c.widths = {"rank": 5, "WALK": 2, **{b: len(NAMES[b]) for b in cls.EXTRA}}
    return c


def selftest():
    t0 = time.time()
    LG.bind_device_ops()
    # 1. the look's blocks are the pinned names, on a KB carve and a passage carve; their values on one chunk each
    seen = {}
    for ds, cv in (("metaqa", "fit"), ("2wiki", "select"), ("2wiki", "fit")):
        d = LM.LOOK / ds / cv
        recs = sorted(d.glob("record*.json"))
        head = json.loads(recs[0].read_text(encoding="utf-8"))
        cols = look_cols(head)
        f = sorted((d / "chunks").glob("c*.npz"))[0]
        x = np.load(f)["x"]
        seen[ds] = {b: (x[:, cols[b]] != 0).mean(0) for b in BLOCKS}
        assert all(np.isfinite(x[:, cols[b]].astype(np.float32)).all() for b in BLOCKS), (ds, cv)
    assert (seen["metaqa"]["typed_rel"] > 0).sum() >= 8 and (seen["metaqa"]["typed_v2"] > 0).all()
    assert (seen["metaqa"]["ordered"] > 0).all() and (seen["metaqa"]["gcs"] > 0.5).all()
    assert all((seen["2wiki"][b] == 0).all() for b in ARM_BLOCKS["rel"]) and (seen["2wiki"]["gcs"] > 0.5).all()
    bad = json.loads(sorted((LM.LOOK / "metaqa" / "fit").glob("record*.json"))[0].read_text(encoding="utf-8"))
    bad["column_blocks"]["gcs"] = list(reversed(bad["column_blocks"]["gcs"]))
    try:
        look_cols(bad)
        raise AssertionError("a reordered block was accepted")
    except SystemExit:
        pass
    # 2. build: a whole local carve, its rows as step 1's records say, its blocks the chunks' columns row for row
    tmp = Path(tempfile.mkdtemp(prefix="relcols_"))
    try:
        built = None
        for ds, cv in (("2wiki", "select"), ("2wiki", "fit")):
            try:
                _d, files, _ids, head = LC.look_chunks(ds, cv)
            except SystemExit:
                continue
            built = build(ds, cv, out_root=tmp)
            cols = look_cols(head)
            parts, recs = LC.part_dirs(ds, cv)
            for p, r in zip(parts, recs):
                hr = json.loads((tmp / ds / cv / p.name / "record.json").read_text(encoding="utf-8"))
                assert hr["rows"] == r["rows"] and hr["step1_record_sha256"] == LC.sha_file(p / "record.json")
                lo, hi = r["chunks"]
                for b in BLOCKS:
                    got = np.load(tmp / ds / cv / p.name / f"{b}.npy")
                    want = np.concatenate([np.load(files[ch])["x"][:, cols[b]] for ch in range(lo, hi)])
                    assert got.dtype == np.float16 and got.shape == (r["rows"], len(NAMES[b]))
                    assert np.array_equal(got.view(np.uint16), want.view(np.uint16)), (ds, cv, b)
                assert hr["nonzero"]["typed_rel"] == [0.0] * 10 and min(hr["nonzero"]["gcs"]) > 0.5
            break
        assert built, "no whole carve's look on disk here"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    # 3. the carve: base blocks lean_gpu's bit for bit, the arm's from X2's same rows (non-finite read 0)
    blocks = ["rank", "WALK"]
    qs = np.array([2, 0])
    for arm, cls in CARVES.items():
        c = fake_carve(cls)
        ref = LG.CacheCarve.batch(c, qs, blocks)
        got = c.batch(qs, blocks + list(cls.EXTRA))
        assert all(torch.equal(ref[0][b], got[0][b]) for b in blocks)
        assert all(torch.equal(x, y) for x, y in zip(ref[1:], got[1:]))
        rows = got[0]["rank"][:, 0].round().long()
        assert rows.tolist() == [6, 7, 8, 0, 1, 2, 3]
        for b in cls.EXTRA:
            a, e = c.span2[b]
            want = torch.nan_to_num(c.X2[rows][:, a:e].float(), nan=0.0, posinf=0.0, neginf=0.0)
            assert torch.equal(got[0][b], want) and torch.isfinite(got[0][b]).all()
            assert torch.equal(c.block_values(b).view(torch.int16), c.X2[:, a:e].view(torch.int16))   # NaN kept, bit for bit
        assert torch.equal(c.block_values("WALK"), c.X[:, 5:7])
        # the model reads the arm's blocks after step 1's
        m = LG.LeanMLP8D(blocks + list(cls.EXTRA), c.widths, 8, ctx="none")
        keep = torch.ones((qs.size, len(m.blocks)))
        s = m(got[0], keep, got[1], qs.size, got[2])
        assert s.shape == (7,) and torch.isfinite(s).all()
        assert m.in_w == sum(2 * w + 1 for w in c.widths.values())
    # 4. a block that is 0 on every training row is dead (lean_gpu's rule), so a split with no KB trains step 1's pick
    c = fake_carve(RelCarveRel)
    c.X2[:, c.span2["typed_rel"][0]:c.span2["typed_rel"][1]] = 0
    assert LG.dead_blocks([c], ["typed_rel", "typed_v2", "WALK"]) == ["typed_rel"]
    # 5. the sets, the arms and the patch
    pick, pns = list(LG.SETS["pick"]), list(LG.SETS["pns"])
    with with_sets("rel"):
        assert LG.SETS["pick"] == pick + list(ARM_BLOCKS["rel"]) and LG.SETS["pns"] == pns + list(ARM_BLOCKS["rel"])
    try:
        with with_sets("gcs"):
            assert LG.SETS["pick"][-1] == "gcs"
            raise KeyError("x")
    except KeyError:
        pass
    with with_sets("base"):
        assert LG.SETS["pick"] == pick
    assert LG.SETS["pick"] == pick and LG.SETS["pns"] == pns
    for arm, cls in CARVES.items():
        assert S.ARMS[arm] == (S.ARMS["base"][0], cls)
        with S.patched(arm):
            assert LG.CacheCarve is cls
    assert LG.CacheCarve is S.ARMS["base"][1]
    assert arm_of(["train", "--arm", "rel", "--split", "L-musique"]) == "rel" and arm_of(["read"]) == "base"
    log(f"selftest: the look's four blocks are the pinned columns (a reordered block is refused); metaqa's relation "
        "columns are live and 2wiki's are 0, gcs live on both; a whole local carve builds with step 1's rows, each "
        "block the chunks' columns bit for bit; the carve keeps lean_gpu's batch for step 1's blocks and reads the "
        "arm's from the same rows (a non-finite value reads 0); the model takes the arm's blocks after step 1's; a block "
        f"0 on every row is dead; the sets restore; the arms patch. all checks passed ({time.time() - t0:.0f}s)")
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
        bp.add_argument("--out-root")
        bp.add_argument("--host", action="store_true")
        b = bp.parse_args(rest)
        build(b.dataset, b.carve, Path(b.out_root) if b.out_root else RC_OUT,
              placement={"where": "host" if b.host else "laptop"})
        return 0
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
    raise SystemExit("relcols: build, train, read, compare, smoke, or --selftest")


if __name__ == "__main__":
    sys.exit(main())
