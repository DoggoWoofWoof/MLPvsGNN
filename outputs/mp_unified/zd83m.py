"""D83's model changes (docs/D83_UNIFIED_R_REFIT.md section 3): the label axis, LTM's label bits and LAB, for D83's fits,
LTM columns and reads. Nothing here edits a record file; every change is a patch in force in a D83 process only.

- **Labels (zg6r, GNX, and every reader of D83's links):** zlink.FAMS = 5 and zlink.SETTINGS = D83's (the settings D83's
  links and typed sets were built under, zd83.d83_settings), set before zprop's first import, so zprop, zg2, zg6, zg6b,
  zg6b2, zg6r and zu2n take five labels where the record took three families: per-label message maps, attention biases,
  label means and the label dropout (zg2.family_keep unchanged: a question keeps its lowest label when every label it has
  would go, which is the slot's label, S on a KB and R.mention on passages, wherever it has one, as the record kept the slot).
- **LTM (the NEXT columns):** ltwdiag's candidate features hold the five label bits (lab_S ... lab_K.knn) in place of the
  three family bits (feature_rows, low_fam and the per-family summary over five labels); LTM's lists, fold fits and columns
  run through `ltm` below, in the D83 world, on an arm's cache and links (D83F's fit carves, D83E's s1eval carves).
- **LAB (zrckpn and GNX):** ten columns per pool row, built by `lab` below from an arm's labelled links and its cache's rrf
  column: for each label, log1p of the row's links with that label to any pool row, then to rrf's five leading rows (CHN's
  order: descending, NaN last, ties by pool position). zrckpn takes it as a block of its encoder after NEXT (pick = DOB's
  nine blocks, NEXT, LAB); GNX takes it beside NEXT in its l1 hook (the hook's input [raw, z, 1] over 7 + 10 columns: 35).
  zg6r reads neither, as in the record.

    python outputs/mp_unified/zd83m.py lab --arm D83F --dataset musique --carve s1fit
    python outputs/host_ops/d83world.py run -- python outputs/mp_unified/zd83m.py ltm lists --arm D83F --dataset musique --carve s1fit
    python outputs/host_ops/d83world.py run -- python outputs/mp_unified/zd83m.py ltm fit --split WQ-musique --host
    python outputs/host_ops/d83world.py run -- python outputs/mp_unified/zd83m.py ltm cols --arm D83F --split WQ-musique --dataset 2wiki --carve fit --host
    python outputs/mp_unified/zd83m.py selftest

No numpy or torch at import: an `ltm` step imports ltmcols first (its 2-thread BLAS environment), and zd83t imports zswpt
first (zltm's 6 threads for a zrckpn fit). This file imports neither zabl nor zd83 outside its self-test.
"""
import contextlib
import hashlib
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

DECLARED = "docs/D83_UNIFIED_R_REFIT.md"
STAGE = "D83"
WORLD = "d83world"
ARMS = ("D83F", "D83E")                                 # the fit carves' arm, the s1eval carves' arm
LABELS = ("S", "R.mention", "R.name_comention", "R.title_comention", "K.knn")   # zd83.LABELS, ids 0 to 4
NL = len(LABELS)
EDGES = "the look's pool edges, undirected (both directions), once per label the pair carries (D83), no self-loops"
LAB = "LAB"                                             # the block's batch key
TOP = 5                                                 # rrf's leading rows (LTM's anchors, ltwdiag.TOP)
LAB_COLS = tuple(f"{lab}|all" for lab in LABELS) + tuple(f"{lab}|top{TOP}" for lab in LABELS)
LAB_WIDTH = len(LAB_COLS)
LTM_BITS = tuple(f"lab_{lab}" for lab in LABELS)       # LTM's five label bits, features 11 to 15
BASE_BITS = ("fam_struct", "fam_ner", "fam_knn")
GNX_IN = 2 * (7 + LAB_WIDTH) + 1                        # GNX's hook input: [raw, z, 1] over NEXT's 7 and LAB's 10 columns
LOADED = []                                             # per carve a fit or read loaded LAB for: its record


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] D83M: {msg}", flush=True)


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sha_src(p):
    """A source file's sha256 with line ends normalised (lean_cache.sha_src's rule)."""
    return hashlib.sha256(Path(p).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def write_json(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1), encoding="utf-8")
    os.replace(tmp, p)


def read_json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def rel(p):
    try:
        return Path(p).resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(p)


# ── the label axis ───────────────────────────────────────────────────────────

AXIS_MODULES = ("zprop", "zg2", "zg6", "zg6b", "zg6b2", "zg6r", "zu2n", "zml1s", "zgnx", "zltm", "zp1c3p")


def d83_settings(base):
    """zd83.d83_settings on zlink's own settings (the self-test checks the two agree)."""
    return dict(base, families=NL, labels=list(LABELS), edges=EDGES)


def labels_on():
    """zlink.FAMS = 5 and D83's settings, before zprop's first import (zprop takes FAMS from zlink at import, and zg2, zg6,
    zg6r from zprop). Refuses when a module that took the record's three families is already imported. Returns zlink."""
    import zlink as ZL
    if getattr(ZL, "D83_LABELS", False):
        return ZL
    early = [m for m in AXIS_MODULES if m in sys.modules]
    if early:
        raise SystemExit(f"D83M: {early} imported before the label axis was set; they hold the record's {ZL.FAMS} families")
    if ZL.FAMS != 3 or ZL.SETTINGS.get("families") != 3:
        raise SystemExit(f"D83M: zlink holds {ZL.FAMS} families, not the record's 3")
    ZL.SETTINGS = d83_settings(ZL.SETTINGS)
    ZL.FAMS = NL
    ZL.D83_LABELS = True
    return ZL


def axis_record():
    """The label axis each imported module holds (taken after a fit: every one must say 5)."""
    got = {m: int(getattr(sys.modules[m], "FAMS")) for m in ("zlink", "zprop", "zg2", "zg6", "zg6b", "zg6r")
           if m in sys.modules and hasattr(sys.modules[m], "FAMS")}
    bad = {m: f for m, f in got.items() if f != NL}
    if bad:
        raise SystemExit(f"D83M: modules with another label axis than {NL}: {bad}")
    import zlink as ZL
    return {"labels": list(LABELS), "fams": got, "zlink_settings": ZL.SETTINGS,
            "family_dropout": "zg2.family_keep, unchanged: per label; when every label a question has would go it keeps its "
                              "lowest, the slot's label (S on a KB, R.mention on passages) wherever it has one"}


# ── LTM: five label bits ─────────────────────────────────────────────────────


def ltm_on(LTW):
    """ltwdiag with the five label bits in place of the three family bits (features 11 to 15), its low_fam and its
    per-family summary over the five labels. Idempotent."""
    if getattr(LTW, "D83_BITS", False):
        return LTW
    if tuple(LTW.FEATS[11:]) != BASE_BITS or len(LTW.FEATS) != 14 or LTW.FEATS[10] != "log1p_rank":
        raise SystemExit(f"D83M: ltwdiag's features are {LTW.FEATS}, not the record's fourteen")
    np = LTW.np

    def feature_rows(cv, s, ca, res, pair, cxc, crank, cfam):
        F = np.empty((cv.size, len(LTW.FEATS)), np.float64)
        F[:, 0], F[:, 1], F[:, 2], F[:, 3], F[:, 4] = cv, s, res, pair, ca
        F[:, 5:10] = cxc[:, :LTW.J_DC]
        F[:, 10] = np.log1p(crank)
        for k in range(NL):
            F[:, 11 + k] = (cfam.astype(np.int64) >> k) & 1
        return F

    def low_fam(bits):
        b = int(bits)
        return next(k for k in range(NL) if b >> k & 1)

    def summarise_a(ranks, fams, hops):
        res = {}
        for s, rk in ranks.items():
            a = np.asarray(rk, np.float64)
            res[s] = {"T1": float(np.mean(a <= 1)) if a.size else None, "T2": float(np.mean(a <= 2)) if a.size else None,
                      "median": float(np.median(a)) if a.size else None, "MRR": float(np.mean(1.0 / a)) if a.size else None}
            f, h = np.asarray(fams), np.asarray(hops)
            res[s]["T2_by_family"] = {str(k): (float(np.mean(a[f == k] <= 2)) if (f == k).any() else None) for k in range(NL)}
            if (h > 0).any():
                res[s]["T2_by_hop"] = {f"{k}hop": (float(np.mean(a[h == k] <= 2)) if (h == k).any() else None)
                                       for k in (2, 3, 4)}
        return res

    LTW.FEATS = tuple(LTW.FEATS[:11]) + LTM_BITS
    LTW.feature_rows, LTW.low_fam, LTW.summarise_a = feature_rows, low_fam, summarise_a
    LTW.D83_BITS = True
    return LTW


def world_ok(where="D83M"):
    """A step that writes the D83 world's own folders runs there only: D83WORLD=1, a tree named d83world."""
    if os.environ.get("D83WORLD") != "1" or ROOT.name != WORLD:
        raise SystemExit(f"{where}: runs only in the D83 world (python outputs/host_ops/d83world.py run -- ...)")
    return True


def arm_root(arm):
    if arm not in ARMS:
        raise SystemExit(f"D83M: {arm} is not one of {ARMS}")
    return ROOT / "outputs" / "abl" / arm


def links_ok(lp, arm):
    """A links part D83's links step wrote for the arm (its settings name the five labels, its d83 block the arm)."""
    h = read_json(Path(lp) / "record.json")
    st, d = h.get("settings") or {}, h.get("d83") or {}
    if st.get("families") != NL or st.get("labels") != list(LABELS) or d.get("arm") != arm:
        raise SystemExit(f"D83M: {rel(lp)} is not a labelled links part of {arm} (settings {st.get('families')}, "
                         f"labels {st.get('labels')}, arm {d.get('arm')})")
    return h


def ltm(argv):
    """LTM's lists, fold fits and columns (ltmcols, not edited) with the five label bits, in the D83 world; lists and cols
    on the named arm's cache and links (every links part checked as the arm's labelled links first)."""
    import argparse
    import ltmcols as LTMC                                  # first: its BLAS thread environment before numpy
    ap = argparse.ArgumentParser(prog="zd83m.py ltm")
    ap.add_argument("cmd", choices=("lists", "fit", "cols"))
    ap.add_argument("--arm", choices=ARMS)
    ap.add_argument("--dataset")
    ap.add_argument("--carve")
    ap.add_argument("--split", choices=tuple(LTMC.WQ.SPLITS))
    ap.add_argument("--host", action="store_true")
    a = ap.parse_args(argv)
    world_ok("D83M ltm")
    labels_on()
    ltm_on(LTMC.LTW)
    root = arm_root(a.arm) if a.cmd in ("lists", "cols") else None
    if a.cmd in ("lists", "cols"):
        if (a.dataset, a.carve) not in LTMC.ALL_CARVES:
            raise SystemExit(f"D83M: {a.dataset}/{a.carve} is not one of LTM's carves")
        import lean_cache as LC
        parts, _recs = LC.part_dirs(a.dataset, a.carve, root / "cache")
        for p in parts:
            links_ok(root / "links" / a.dataset / a.carve / p.name, a.arm)
    if a.cmd == "lists":
        LTMC.lists(a.dataset, a.carve, root=root)
        return 0
    if a.host:
        import lean_host as LH
        LH.substitute()
    if a.cmd == "fit":
        LTMC.fit(a.split, host=a.host)
        return 0
    LTMC.cols(a.split, a.dataset, a.carve, root=root, host=a.host)
    return 0


# ── LAB ──────────────────────────────────────────────────────────────────────


def lab_rows(n, rrf, q_ne, e_u, e_v, e_lab):
    """(rows, 10) float16: per row, log1p of its links per label to any pool row, then to rrf's five leading rows of its
    question (CHN's order). n: pool sizes; rrf: the rows' rrf scores; q_ne, e_u, e_v, e_lab: the part's links (both
    directions stored, pool-local rows)."""
    import numpy as np
    n = np.asarray(n, np.int64)
    rows = int(n.sum())
    off = np.r_[0, np.cumsum(n)].astype(np.int64)
    qrow = np.repeat(np.arange(n.size, dtype=np.int64), n)
    pos = np.arange(rows, dtype=np.int64) - off[qrow]
    s = np.asarray(rrf, np.float64)
    key = np.where(np.isnan(s), np.inf, -s) + 0.0
    o = np.lexsort((pos, key, qrow))                       # by question, then descending rrf (NaN last), then position
    rank = np.empty(rows, np.int64)
    rank[o] = np.arange(rows, dtype=np.int64) - off[qrow[o]]
    in_top = rank < TOP
    qe = np.repeat(np.arange(n.size, dtype=np.int64), np.asarray(q_ne, np.int64))
    gu = off[qe] + np.asarray(e_u, np.int64)
    gv = off[qe] + np.asarray(e_v, np.int64)
    lab = np.asarray(e_lab, np.int64)
    if lab.size and (int(lab.min()) < 0 or int(lab.max()) >= NL):
        raise SystemExit(f"D83M: a link's label is outside 0..{NL - 1}")
    allc = np.bincount(gu * NL + lab, minlength=rows * NL).reshape(rows, NL)
    t = in_top[gv]
    topc = np.bincount(gu[t] * NL + lab[t], minlength=rows * NL).reshape(rows, NL)
    return np.log1p(np.concatenate([allc, topc], 1).astype(np.float64)).astype(np.float16), allc.sum(0), topc.sum(0)


def lab_dir(root, ds, carve):
    return Path(root) / "lab" / ds / carve


def rrf_column(head):
    """The rrf column of a cache's xc (ltwdiag's RANK_COLS where lean_cache.xc_columns puts them)."""
    import lean_cache as LC
    import ltwdiag as LTW
    cols, spans, _ci = LC.xc_columns(head)
    names = [head["columns"][int(c)] for c in cols]
    r0, r1 = spans["rank"]
    if tuple(names[r0:r1]) != LTW.RANK_COLS:
        raise SystemExit(f"D83M: xc holds {names[r0:r1]} where xc_columns puts {LTW.RANK_COLS}")
    return r0 + LTW.RANK_COLS.index("rrf")


def lab(arm, ds, carve, root=None):
    """One carve's LAB columns per cache part, from the arm's cache (rrf) and labelled links; records per part and carve."""
    import numpy as np
    import lean_cache as LC
    import ltwdiag as LTW
    t0 = time.time()
    root = arm_root(arm) if root is None else Path(root)
    parts, recs = LC.part_dirs(ds, carve, root / "cache")
    j = rrf_column(recs[0]["look"])
    out = lab_dir(root, ds, carve)
    plist, tot_all, tot_top, rows_all, q_all = [], np.zeros(NL, np.int64), np.zeros(NL, np.int64), 0, 0
    for p, r in zip(parts, recs):
        lp = root / "links" / ds / carve / p.name
        links_ok(lp, arm)
        A, E, lsha = LTW.part_arrays(p, r, root / "links")
        X, ca, ct = lab_rows(np.asarray(A["n"], np.int64), np.asarray(A["xc"][:, j], np.float32), E["q_ne"], E["e_u"],
                             E["e_v"], E["e_fam"])
        if X.shape != (int(r["rows"]), LAB_WIDTH):
            raise SystemExit(f"D83M: {rel(p)}: LAB holds {X.shape}, the part {r['rows']} rows")
        pd = out / p.name
        pd.mkdir(parents=True, exist_ok=True)
        tmp = pd / "lab.tmp.npy"
        np.save(tmp, X)
        os.replace(tmp, pd / "lab.npy")
        prec = {"stage": STAGE, "declared_in": DECLARED, "what": LAB, "arm": arm, "dataset": ds, "carve": carve,
                "part": p.name, "cache_record_sha256": LC.sha_file(p / "record.json"), "links_record_sha256": lsha,
                "rows": int(r["rows"]), "queries": int(r["queries"]), "columns": list(LAB_COLS),
                "lab_sha256": sha_file(pd / "lab.npy"), "links_by_label": dict(zip(LABELS, (int(x) for x in ca))),
                "links_to_top_by_label": dict(zip(LABELS, (int(x) for x in ct)))}
        write_json(pd / "record.json", prec)
        plist.append({k: prec[k] for k in ("part", "rows", "queries", "cache_record_sha256", "links_record_sha256",
                                            "lab_sha256")} | {"record_sha256": sha_file(pd / "record.json")})
        tot_all += ca
        tot_top += ct
        rows_all += int(r["rows"])
        q_all += int(r["queries"])
        del A, E, X
        log(f"lab {arm} {ds}/{carve} {p.name}: {r['rows']} rows ({time.time() - t0:.0f}s)")
    rec = {"stage": STAGE, "declared_in": DECLARED, "what": LAB, "arm": arm, "dataset": ds, "carve": carve,
           "columns": list(LAB_COLS), "top": TOP, "rows": rows_all, "queries": q_all, "parts": plist,
           "links_by_label": dict(zip(LABELS, (int(x) for x in tot_all))),
           "links_to_top_by_label": dict(zip(LABELS, (int(x) for x in tot_top))),
           "rule": "per label: log1p of the row's links with it to any pool row, then to rrf's five leading rows (CHN's order)",
           "script_sha256": sha_src(__file__), "seconds": round(time.time() - t0, 1),
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    write_json(out / "record.json", rec)
    log(f"lab {arm} {ds}/{carve}: {rows_all} rows, {q_all} questions, links by label {rec['links_by_label']}")
    return rec


def lab_root_of(cache_root):
    """The LAB root beside a carve's cache root (outputs/pko/cache in the world: the mode's arm)."""
    p = Path(cache_root)
    return Path(os.path.realpath(p if p.is_absolute() else ROOT / p)).parent


def load_lab(ds, carve, part_shas, root, verify=True):
    """(rows x 10 float16, the carve's LAB record, its sha256), checked against the carve's cache parts (names, record
    sha256, rows) and, with verify, each part's file sha256."""
    import numpy as np
    d = lab_dir(root, ds, carve)
    p = d / "record.json"
    if not p.is_file():
        raise SystemExit(f"D83M: no LAB of {ds}/{carve} under {rel(root)} ({p})")
    rec = read_json(p)
    got = tuple(rec.get(k) for k in ("stage", "declared_in", "what", "dataset", "carve", "columns"))
    if got != (STAGE, DECLARED, LAB, ds, carve, list(LAB_COLS)):
        raise SystemExit(f"D83M: {rel(p)} is {got}, not D83's LAB of {ds}/{carve}")
    names = [q["part"] for q in rec["parts"]]
    if names != list(part_shas):
        raise SystemExit(f"D83M: {ds}/{carve}'s LAB holds the parts {names}, its cache {list(part_shas)}")
    xs = []
    for q in rec["parts"]:
        if q["cache_record_sha256"] != part_shas[q["part"]]:
            raise SystemExit(f"D83M: {ds}/{carve} {q['part']}: LAB was built from another cache part")
        pd = d / q["part"]
        if sha_file(pd / "record.json") != q["record_sha256"]:
            raise SystemExit(f"D83M: {rel(pd / 'record.json')} is not the record its carve names")
        f = pd / "lab.npy"
        if verify and sha_file(f) != q["lab_sha256"]:
            raise SystemExit(f"D83M: {rel(f)}: sha256 is not its record's")
        x = np.load(f)
        if x.dtype != np.float16 or x.shape != (q["rows"], LAB_WIDTH):
            raise SystemExit(f"D83M: {rel(f)} is {x.dtype} {x.shape}, not float16 ({q['rows']}, {LAB_WIDTH})")
        xs.append(x)
    X = np.concatenate(xs) if xs else np.zeros((0, LAB_WIDTH), np.float16)
    if X.shape[0] != rec["rows"]:
        raise SystemExit(f"D83M: {ds}/{carve}'s LAB parts hold {X.shape[0]} rows, its record {rec['rows']}")
    return X, rec, sha_file(p)


def lab_loaded(ds, carve, rec, sha, root):
    r = {"dataset": ds, "carve": carve, "arm": rec["arm"], "root": rel(root), "record_sha256": sha, "rows": rec["rows"]}
    LOADED.append(r)
    return r


# ── zrckpn: LAB as a block after NEXT ────────────────────────────────────────


def lab_carve(base):
    """base (zltm's NextCarve) with LAB beside it: batch() adds LAB for the batch's rows, block_values() hands it to
    lean_gpu's dead-block check, widths names its ten columns; every other block is base's, unchanged."""
    import numpy as np
    import torch
    import lean_cache as LC

    class LabCarve(base):
        def __init__(self, ds, carve, basis, device, cache_root=LC.OUT, verify=True, score2=False):
            super().__init__(ds, carve, basis, device, cache_root, verify, score2)
            root = lab_root_of(cache_root)
            X, rec, sha = load_lab(ds, carve, self.part_shas, root, verify)
            if X.shape[0] != int(self.off_np[-1]):
                raise SystemExit(f"D83M: {ds}/{carve}'s LAB holds {X.shape[0]} rows, its carve {int(self.off_np[-1])}")
            self.lab_X = torch.from_numpy(X).to(self.device)
            self.lab_record = lab_loaded(ds, carve, rec, sha, root)
            self.widths = {**self.widths, LAB: LAB_WIDTH}

        def nbytes(self):
            return super().nbytes() + self.lab_X.numel() * self.lab_X.element_size()

        def block_values(self, b):
            if b == LAB:
                return self.lab_X
            return super().block_values(b)

        def batch(self, qs, blocks):
            feats, nq, base_z, gold = super().batch(qs, [b for b in blocks if b != LAB])
            if LAB in blocks:
                qs = np.asarray(qs, np.int64)
                cnt = self.n_np[qs]
                idx = np.repeat(self.off_np[qs] - (np.cumsum(cnt) - cnt), cnt) + np.arange(int(cnt.sum()))
                x = self.lab_X[torch.from_numpy(idx).to(self.lab_X.device)].to(torch.float32)
                feats[LAB] = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            return feats, nq, base_z, gold

    LabCarve.__name__ = LabCarve.__qualname__ = f"Lab{base.__name__}"
    return LabCarve


def zrckpn_on(LTM):
    """zltm with LAB: its carve class is NextCarve with LAB (registered by import_arm, which must not have run), its pick
    DOB's nine blocks, NEXT and LAB, and a fit that did not take LAB into every set (or found it dead) stops unstamped."""
    if getattr(LTM, "D83_LAB", False):
        return LTM
    if LTM._ARM:
        raise SystemExit("D83M: zltm registered zrckpn before LAB was put on its carve")
    orig_carve, orig_used = LTM.next_carve, LTM.used_ok

    def next_carve(base):
        return lab_carve(orig_carve(base))

    def used_ok(tj, name):
        orig_used(tj, name)
        sets, dead = tj.get("sets") or {}, tj.get("dead") or []
        if any(LAB not in bl for bl in sets.values()) or LAB in dead:
            raise SystemExit(f"D83M: {name}'s trainer fitted the sets {sets} with the dead blocks {dead}; unstamped")

    LTM.next_carve, LTM.used_ok = next_carve, used_ok
    LTM.PICK = tuple(LTM.DOB.PICK) + (LTM.NEXT, LAB)
    LTM.D83_LAB = True
    return LTM


# ── GNX: LAB beside NEXT in the hook ─────────────────────────────────────────


def gnx_on(GX):
    """zgnx with LAB beside NEXT: each carve's GNX columns are NEXT's seven then LAB's ten (gnx_X), and the hook's input
    is [raw, z, 1] over the seventeen (IN = 35)."""
    if getattr(GX, "D83_LAB", False):
        return GX
    if GX.IN != 2 * GX.WIDTH + 1 or GX.WIDTH != 7:
        raise SystemExit(f"D83M: zgnx's input is {GX.IN} over {GX.WIDTH} columns, not the record's 15 over 7")
    orig = GX.on_carve

    @contextlib.contextmanager
    def on_carve(G, C=None):
        import torch
        with orig(G, C):
            K = G.G6rLinkCarve if C is None else C
            init_next = K.__init__

            def init(self, ds, carve, basis, device, cache_root=G.LC.OUT, verify=True, score2=False):
                init_next(self, ds, carve, basis, device, cache_root, verify, score2)
                root = lab_root_of(cache_root)
                X, rec, sha = load_lab(ds, carve, self.part_shas, root, verify)
                if X.shape[0] != self.gnx_X.shape[0] or self.gnx_X.shape[1] != 7:
                    raise SystemExit(f"D83M: {ds}/{carve}: LAB {X.shape}, NEXT {tuple(self.gnx_X.shape)}")
                self.gnx_X = torch.cat([self.gnx_X, torch.from_numpy(X)], 1)
                self.gnx_record["lab"] = lab_loaded(ds, carve, rec, sha, root)

            with GX.M1.WQ.patch(K, __init__=init):
                yield

    GX.on_carve, GX.IN = on_carve, GNX_IN
    GX.D83_LAB = True
    return GX


@contextlib.contextmanager
def model_changes(kind):
    """D83's changes for a fit of this kind, in force for the fit: the label axis for all (set before this is entered),
    LAB on zrckpn's carve and pick, LAB beside NEXT in GNX's hook; zspk and zg6r take the label axis only."""
    labels_on()
    if kind == "zrckpn":
        import zltm
        zrckpn_on(zltm)
    elif kind == "gnx":
        import zgnx
        gnx_on(zgnx)
    elif kind not in ("zspk", "zg6r"):
        raise SystemExit(f"D83M: no model change for {kind}")
    LOADED.clear()
    yield


def fit_record(kind):
    """The D83 block of a fit's record: the label axis every imported module holds, LAB's columns and carves loaded."""
    r = {"declared_in": DECLARED, "kind": kind, "axis": axis_record(), "lab": None}
    if kind in ("zrckpn", "gnx"):
        r["lab"] = {"columns": list(LAB_COLS), "block": LAB if kind == "zrckpn" else "beside NEXT in the l1 hook",
                    "inputs": GNX_IN if kind == "gnx" else 2 * LAB_WIDTH + 1, "carves": list(LOADED)}
    return r


# ── self-test ────────────────────────────────────────────────────────────────


def gnn_check():
    """In a fresh process with the label axis set: zg6r's stack over links of all five labels (a passage question without
    S) equals its loop reference (zg6r.by_hand), with and without the label dropout; the keep rule keeps a question's
    lowest label when every label would go; the gradient reaches every label's message map."""
    labels_on()
    import torch
    import zg2 as G2
    import zg6 as G6
    import zg6r as GR
    assert G6.FAMS == GR.FAMS == G2.FAMS == NL
    D, NB, F, Hd = G6.D, G6.NB, NL, 16
    g = torch.Generator().manual_seed(5)
    m = torch.nn.Module()

    def P(*s):
        return torch.nn.Parameter(torch.randn(*s, generator=g) * 0.3)
    m.g6_in_w, m.g6_in_b, m.g6_ctx_w = P(D, Hd + NB), P(D), P(1, D)
    for nm, na, nf, ns, nmean, nagg, nx, nb in G6.G6_LAYER:
        for nme, s in ((nm, (F * D, D)), (na, (D, D)), (nf, (F,)), (ns, (D, D)), (nmean, (D, F * D)), (nagg, (D, D)),
                       (nx, (D, NB)), (nb, (D,))):
            setattr(m, nme, P(*s))
    for na, nb, nd, nc in GR.G6R_LAYER:
        for nme, s in ((na, (D, GR.K_R)), (nb, (GR.K_R, D)), (nd, (GR.K_R,)), (nc, (GR.K_R, D))):
            setattr(m, nme, P(*s))
    m.g6_out_w, m.g6_skip_w, m.g6_out_b = P(1, D), P(1, D), P(1)
    n, B = [6, 5, 7], 3
    off = [0, 6, 11, 18]
    nq = torch.repeat_interleave(torch.arange(B), torch.tensor(n))
    us, vs, fs = [], [], []
    rng = torch.Generator().manual_seed(9)
    for q in range(B):
        for i in range(14):
            a, b = torch.randint(off[q], off[q + 1], (2,), generator=rng).tolist()
            lab = (i % (F - 1)) + 1 if q == 2 else i % F               # question 2: passages, no S label
            if a != b:
                us += [a, b]
                vs += [b, a]
                fs += [lab, lab]
    lk = {"u": torch.tensor(us), "v": torch.tensor(vs), "f": torch.tensor(fs)}
    gd = {"ci": torch.full((len(us),), -1, dtype=torch.long), "rel": None, "crel": None, "cdir": None}
    H, xb = torch.randn(sum(n), Hd, generator=g), torch.randn(sum(n), NB, generator=g)
    with torch.no_grad():
        got, want = GR.g6r_stack(m, H, xb, lk, gd, nq, B), GR.by_hand(m, H, xb, lk, gd, nq, B)
        assert torch.allclose(got, want, atol=1e-5, rtol=1e-5), float((got - want).abs().max())
        u01 = torch.rand(B, F, generator=g)
        ke = G2.family_keep(lk["f"], nq[lk["v"]], B, u01)
        got, want = GR.g6r_stack(m, H, xb, lk, gd, nq, B, u01), GR.by_hand(m, H, xb, lk, gd, nq, B, ke)
        assert torch.allclose(got, want, atol=1e-5, rtol=1e-5) and 0 < int(ke.sum()) < ke.numel()
        k0 = G2.family_keep(lk["f"], nq[lk["v"]], B, torch.zeros(B, F))   # every label would go
        qe = nq[lk["v"]]
        for q in range(B):
            fq, kq = lk["f"][qe == q], k0[qe == q]
            assert set(fq[kq].tolist()) == {int(fq.min())} and int(fq.min()) == (1 if q == 2 else 0)
    GR.g6r_stack(m, H, xb, lk, gd, nq, B).sum().backward()
    gm = getattr(m, G6.G6_LAYER[0][0]).grad.view(F, D, D)
    assert all(float(gm[k].abs().sum()) > 0 for k in range(F)) and set(fs) == set(range(F))
    print("gnn_check: OK")
    return 0


def selftest():
    import subprocess
    import numpy as np
    assert (NL, LAB_WIDTH, GNX_IN, len(LTM_BITS)) == (5, 10, 35, 5)
    # the label axis must be set before zprop: in a fresh process, every axis module sees five labels
    code = ("import sys; sys.path.insert(0, r'%s'); import zd83m as M; M.labels_on(); import zg6r, zprop, zg2, zg6, zu2n;"
            "print(zprop.FAMS, zg2.FAMS, zg6.FAMS, zg6r.FAMS, zprop.N_IN, zg6r.G6.FAMS)" % HERE)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=str(HERE))
    assert out.returncode == 0 and out.stdout.split() == ["5", "5", "5", "5", "16", "5"], (out.stdout, out.stderr[-800:])
    out = subprocess.run([sys.executable, str(Path(__file__)), "gnn_check"], capture_output=True, text=True, cwd=str(HERE))
    assert out.returncode == 0 and "gnn_check: OK" in out.stdout, (out.stdout[-400:], out.stderr[-1200:])
    late = "import sys; sys.path.insert(0, r'%s'); import zprop, zd83m as M; M.labels_on()" % HERE
    out = subprocess.run([sys.executable, "-c", late], capture_output=True, text=True, cwd=str(HERE))
    assert out.returncode != 0 and "imported before the label axis" in out.stderr, out.stderr[-400:]
    # D83's settings here are zd83's (the ones its links and typed sets were built under)
    code = ("import sys, json; sys.path.insert(0, r'%s'); import zlink, zd83, zd83m;"
            "print(json.dumps([zd83.d83_settings(zlink) == zd83m.d83_settings(zlink.SETTINGS), list(zd83.LABELS)]))" % HERE)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=str(HERE))
    assert out.returncode == 0 and json.loads(out.stdout.strip().splitlines()[-1]) == [True, list(LABELS)], out.stderr[-600:]
    # LAB against a loop over questions (CHN's order_of, both directions stored)
    rng = np.random.default_rng(0)
    n = np.array([7, 1, 12, 5, 9], np.int64)
    rrf = rng.random(int(n.sum())).astype(np.float32)
    rrf[[2, 3]] = rrf[4]                                     # ties: pool position decides
    rrf[10] = np.nan
    qne, eu, ev, el = [], [], [], []
    for q, m in enumerate(n):
        pairs = {(int(a), int(b), int(c)) for a, b, c in zip(rng.integers(0, m, 3 * m), rng.integers(0, m, 3 * m),
                                                             rng.integers(0, NL, 3 * m)) if a != b}
        e = sorted(pairs | {(b, a, c) for a, b, c in pairs})
        qne.append(len(e))
        eu += [x[0] for x in e]
        ev += [x[1] for x in e]
        el += [x[2] for x in e]
    X, ca, ct = lab_rows(n, rrf, np.array(qne), np.array(eu, np.int16), np.array(ev, np.int16), np.array(el, np.int8))
    import chndiag as CH
    off, eo = np.r_[0, np.cumsum(n)], np.r_[0, np.cumsum(qne)]
    ref = np.zeros((int(n.sum()), LAB_WIDTH))
    for q, m in enumerate(n):
        top = set(int(t) for t in CH.order_of(rrf[off[q]:off[q + 1]])[:TOP])
        for k in range(eo[q], eo[q + 1]):
            ref[off[q] + eu[k], el[k]] += 1
            if ev[k] in top:
                ref[off[q] + eu[k], NL + el[k]] += 1
    assert X.dtype == np.float16 and np.array_equal(X, np.log1p(ref).astype(np.float16))
    assert int(ca.sum()) == len(eu) and int(ref[:, NL:].sum()) == int(ct.sum())
    # LTM's five bits: the record's columns 0-10 unchanged, bits 11-15 from the label bit set
    import ltwdiag as LTW
    base_rows = LTW.feature_rows
    cv, s = np.array([0.1, 0.2, 0.3]), np.array([0.5, 0.4, 0.3])
    rs, pr, cxc = cv - 0.2 * s, cv + s, np.arange(18, dtype=np.float32).reshape(3, 6)
    f0 = base_rows(cv, s, 0.2, rs, pr, cxc, np.array([4, 7, 2]), np.array([1, 6, 5], np.int8))
    base_feats = tuple(LTW.FEATS)
    ltm_on(LTW)
    assert len(LTW.FEATS) == 16 and LTW.FEATS[11:] == LTM_BITS and LTW.FEATS[:11] == base_feats[:11]
    f1 = LTW.feature_rows(cv, s, 0.2, rs, pr, cxc, np.array([4, 7, 2]), np.array([1, 6, 24], np.int8))
    assert np.array_equal(f1[:, :11], f0[:, :11]) and f1[:, 11:].tolist() == [[1, 0, 0, 0, 0], [0, 1, 1, 0, 0], [0, 0, 0, 1, 1]]
    assert [LTW.low_fam(b) for b in (1, 6, 24, 16)] == [0, 1, 3, 4] and ltm_on(LTW) is LTW
    assert LTW.summarise_a({"x": [1, 3]}, [3, 4], [0, 0])["x"]["T2_by_family"] == {"0": None, "1": None, "2": None,
                                                                                     "3": 1.0, "4": 0.0}
    # LAB's loader: names, cache shas, file shas and shapes are checked
    import tempfile
    import shutil
    tmp = Path(tempfile.mkdtemp(prefix="d83m_t_"))
    try:
        d = lab_dir(tmp, "toy", "fit")
        plist = []
        for i, rows in enumerate((3, 2)):
            pd = d / f"part_{i}of2"
            pd.mkdir(parents=True)
            np.save(pd / "lab.npy", np.full((rows, LAB_WIDTH), i, np.float16))
            write_json(pd / "record.json", {"part": pd.name})
            plist.append({"part": pd.name, "rows": rows, "queries": 1, "cache_record_sha256": f"c{i}",
                          "links_record_sha256": "l", "lab_sha256": sha_file(pd / "lab.npy"),
                          "record_sha256": sha_file(pd / "record.json")})
        write_json(d / "record.json", {"stage": STAGE, "declared_in": DECLARED, "what": LAB, "arm": "D83F", "dataset": "toy",
                                       "carve": "fit", "columns": list(LAB_COLS), "rows": 5, "parts": plist})
        X, rec, _sha = load_lab("toy", "fit", {"part_0of2": "c0", "part_1of2": "c1"}, tmp)
        assert X.shape == (5, LAB_WIDTH) and float(X[4, 0]) == 1.0
        for bad in ({"part_0of2": "c0", "part_1of2": "zz"}, {"part_0of2": "c0"}):
            try:
                load_lab("toy", "fit", bad, tmp)
                raise AssertionError(bad)
            except SystemExit:
                pass
        np.save(d / "part_1of2" / "lab.npy", np.zeros((2, LAB_WIDTH), np.float16))
        try:
            load_lab("toy", "fit", {"part_0of2": "c0", "part_1of2": "c1"}, tmp)
            raise AssertionError("a changed LAB file must be refused")
        except SystemExit:
            pass
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    # the zrckpn and GNX patches, in a fresh process with the label axis set first
    code = ("import sys; sys.path.insert(0, r'%s'); import zd83m as M; M.labels_on(); import zltm, zgnx;"
            "M.zrckpn_on(zltm); M.gnx_on(zgnx); C = zltm.next_carve(type('B', (), {}));"
            "print(zltm.PICK[-2:], zgnx.IN, C.__name__, zltm.next_carve.__name__)" % HERE)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=str(HERE))
    assert out.returncode == 0 and out.stdout.split() == ["('NEXT',", "'LAB')", "35", "LabNextB", "next_carve"], \
        (out.stdout, out.stderr[-800:])
    try:
        world_ok()
        assert ROOT.name == WORLD, "world_ok must refuse outside the D83 world"
    except SystemExit:
        pass
    print("zd83m self-test: OK")
    return 0


def main(argv=None):
    a = list(sys.argv[1:] if argv is None else argv)
    if a[:1] == ["selftest"]:
        return selftest()
    if a[:1] == ["gnn_check"]:
        return gnn_check()
    if a[:1] == ["ltm"]:
        return ltm(a[1:])
    if a[:1] == ["lab"]:
        import argparse
        ap = argparse.ArgumentParser(prog="zd83m.py lab")
        ap.add_argument("--arm", choices=ARMS, required=True)
        ap.add_argument("--dataset", required=True)
        ap.add_argument("--carve", required=True)
        k = ap.parse_args(a[1:])
        lab(k.arm, k.dataset, k.carve)
        return 0
    raise SystemExit(__doc__)


if __name__ == "__main__":
    raise SystemExit(main())
