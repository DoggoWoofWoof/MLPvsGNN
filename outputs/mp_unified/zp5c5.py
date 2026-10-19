"""P5c5 (docs/P5C5_FIVE_FOLDS.md): P5c's offset input with g cross-fitted over five folds instead of two, and for the
GNN P5b's normalised log p as a fifth column. P5c (docs/P5C_OFFSET_INPUT.md, outputs/mp_unified/zp5c.py) had no GAIN on
any of its 24 reads, and its section 5 fixed this form for that outcome. Under P5c's two folds each g fitted on half of
the training questions (the first suspect). Here g_k fits on every fold but k, four fifths of them, and each question
takes its columns from the g of its own fold, the one g that never saw it:

    fold(q) = sha256('dataset:question id') read as an integer, mod 5;   question q's columns from g_fold(q)

At two folds this is P5c's cross-fit exactly (g_k fits on the other fold: P5c's g_(1-k)); the selftest builds both on a
toy and gets P5c's g and columns bit for bit. The columns are P5c's four (pt, mx, pt0, mx0) and lp, P5b's log p of the
row: the mixture over (seed, head) of g's softmax over the question's candidates. Each is clipped to [-40, 40], divided
by 10 and stored float16. lp is normalised over the question's rows, so only the GNN reads it:

    zrckpo5  zrckpo's model unchanged (zp5c.py: zrckp plus P5c's head, Linear(4, 16), GELU, Linear(16, 1)), on zrckp's
             carve with the four five-fold columns beside; its carve never opens lp
    zspko5   zspk's model plus the head over five columns, Linear(5, 16), GELU, Linear(16, 1) (113 parameters), added
             with the KT head before zsp's propagation reads the score, on zspk's carve with the five columns beside

The builds (units, fitg, cols) are scripts/p5c_offset_cols.py's stages at five folds, its functions imported unchanged.
They write only under outputs/p5c5 (units, g, cols/<split>/...): nothing here reads or writes P5c's builds.

    python outputs/mp_unified/zp5c5.py units DATASET CARVE --host --threads 2
    python outputs/mp_unified/zp5c5.py fitg SPLIT K --host --device cuda --threads 2
    python outputs/mp_unified/zp5c5.py cols DATASET CARVE --host --threads 2
    python outputs/mp_unified/zp5c5.py train --split L-musique --name scr-p5c5-zrckpo5 --arm zrckpo5 --device cuda \\
        --host --cache-root outputs/p1c2/cache
    python outputs/mp_unified/zp5c5.py read --name scr-p5c5-zrckpo5 --device cuda --host --cache-root outputs/p1c2/cache
    python outputs/mp_unified/zp5c5.py compare --new outputs/screen/fits/scr-p5c5-zrckpo5 \\
        --base outputs/screen/fits/scr-p1c3p-zrckp,outputs/step1/fits/L-musique --out outputs/screen/scr-p5c5-zrckpo5
    python outputs/mp_unified/zp5c5.py pair --arm zrckpo5 --screens S1.json,S2.json --out ...
    python outputs/mp_unified/zp5c5.py recall --arm zrckpo5 --null N1,N2,N3,N4 --pair P.json --out ...
    python outputs/mp_unified/zp5c5.py verdict
    python outputs/mp_unified/zp5c5.py --selftest
"""
import os
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import contextlib  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import zp5c as P5  # noqa: E402  (P5c: its head, zrckpo's model, the batch's row order; through it zp1c3, zp1c3p, zp1c2)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

Z3, Z3P, Z2 = P5.Z3, P5.Z3P, P5.Z2
ZP, ZL, ZM, RM = P5.ZP, P5.ZL, P5.ZM, P5.RM
Z, R, S2, S, LG, LC, SR = P5.Z, P5.R, P5.S2, P5.S, P5.LG, P5.LC, P5.SR
log = S.log

DECLARED = "docs/P5C5_FIVE_FOLDS.md"
OUT = ROOT / "outputs" / "p5c5"
UNITS, GDIR, COLS_DIR = OUT / "units", OUT / "g", OUT / "cols"
SCREEN = ROOT / "outputs" / "screen"
P5C_KEY = P5.P5C_KEY
COLNAMES = P5.COLNAMES                     # pt, mx, pt0, mx0: P5c's four
LPNAME = "lp"                              # P5b's log p: normalised over the question's rows, the GNN's fifth column
NCOL, NCOL_G = len(COLNAMES), len(COLNAMES) + 1
CLIP, SCALE, FOLDS = P5.CLIP, P5.SCALE, 5
P5C_HID, P5C_SEED_OFF = P5.P5C_HID, P5.P5C_SEED_OFF
P5C = P5.P5C


def build_settings(folds):
    """The settings every fit and column record carries: FOLDS (5) in every build, 2 only in the selftest's
    reproduction of P5c's columns."""
    return {"columns": list(COLNAMES), "lp": "P5b's log p: the mixture over (seed, head) of g's softmax over the "
                                             "question's candidates (the pool minus the seeds), the GNN's alone",
            "clip": [-CLIP, CLIP], "scale": SCALE, "space": "V", "model": "G", "folds": folds,
            "fold": "sha256('dataset:question id') read as an integer, mod the folds",
            "cross_fit": "g_k fits on every fold but k; each question's columns from the g of its own fold",
            "fit": "P5b's g, refit per split without its held-out dataset on the four training datasets' fit "
                   "questions with a counted gold outside one fold, P5b2's held-out stop inside them",
            "pools": "P1c2's KIND_T files; rows in P1c2's cache order", "dtype": "float16"}


BUILD_SETTINGS = build_settings(FOLDS)
SPLITS = Z3.SPLITS
BASE_OF = {"zrckpo5": "zrckp", "zspko5": "zspk"}
ARM_NAMES = tuple(BASE_OF)
OTHER = {"zrckpo5": "zspko5", "zspko5": "zrckpo5"}
READS = {"zrckpo5": COLNAMES, "zspko5": COLNAMES + (LPNAME,)}
N_HEAD = {a: P5C_HID * len(READS[a]) + P5C_HID + P5C_HID + 1 for a in ARM_NAMES}
SETTINGS = {a: {"columns": list(READS[a]), "hidden": P5C_HID, "seed_offset": P5C_SEED_OFF, "parameters": N_HEAD[a],
                "score_input": False, "build": BUILD_SETTINGS} for a in ARM_NAMES}
FIT = {a: {"L-musique": f"scr-p5c5-{a}", "L-hotpotqa": f"scr-p5c5-{a}-hp"} for a in ARM_NAMES}
BASE_FITS = {"zrckpo5": {"L-musique": ("screen", "fits", "scr-p1c3p-zrckp"),
                         "L-hotpotqa": ("screen", "fits", "scr-p1c3p-zrckp-hp")},
             "zspko5": {"L-musique": ("screen", "fits", "scr-p1c3-zspk"),
                        "L-hotpotqa": ("screen", "fits", "scr-p1c3-zspk-hp")}}
BASE_SCREENS = {a: {sp: f"outputs/l1/base-{BASE_OF[a]}-{sp}.json" for sp in SPLITS} for a in ARM_NAMES}
AGAINST = {a: f"{BASE_OF[a]}'s card fit of each split ({BASE_FITS[a]['L-musique'][2]}, "
              f"{BASE_FITS[a]['L-hotpotqa'][2]})" for a in ARM_NAMES}
WHAT = {"zrckpo5": "zrckp plus P5c's head over the four columns, g cross-fitted over five folds",
        "zspko5": "zspk plus the head over the four five-fold columns and P5b's log p, before its propagation"}
PAIR = {a: SCREEN / f"scr-p5c5-{a}-pair" for a in ARM_NAMES}
RECALL = {a: SCREEN / f"scr-p5c5-{a}-pair-recall" for a in ARM_NAMES}
VERDICT = OUT / "screen-verdict"
BESIDE = ("webqsp", "metaqa")
CLAIM1_READ = ("L-hotpotqa", "musique")
RULE = ("per arm: ADOPT_FOR_FULL_RUN when its re-call under the seed null has no LOSS on any of its twelve reads and at "
        "least one GAIN; NOT_ADOPTED otherwise; INCOMPLETE when its re-call is missing, not P5c5's re-call of the arm, "
        "or not twelve reads (docs/P5C5_FIVE_FOLDS.md, section 4)")


# ── the builder: P5c's three stages at five folds ────────────────────────────

BLD = None


def bld():
    """scripts/p5c_offset_cols.py (P5c's builder), imported unchanged on first use by the build commands and the
    selftest. Its units, distances, columns and checks are called; its own stages and folders (outputs/p5c) are not,
    except in the selftest on temporary roots."""
    global BLD
    if BLD is None:
        sp = str(ROOT / "scripts")
        if sp not in sys.path:
            sys.path.insert(0, sp)            # kept: P5a's modules() imports step 4e's scripts from there later
        import p5c_offset_cols
        BLD = p5c_offset_cols
    return BLD


def script_sha256():
    return LC.sha_src(__file__)


def fold_of(ds, qid, folds=FOLDS):
    """sha256('dataset:question id') read as an integer, mod folds (at two folds, P5c's parity)."""
    return int(hashlib.sha256(f"{ds}:{qid}".encode("utf-8")).hexdigest(), 16) % folds


def units_paths(name, carve, root=UNITS):
    return Path(root) / name / f"{carve}.npz", Path(root) / name / f"{carve}.json"


def g_paths(split, k, root=GDIR):
    return Path(root) / f"{split}_k{k}.pt", Path(root) / f"{split}_k{k}.json"


def units_stage(name, carve, threads, host, limit=None, out=UNITS):
    """P5c's units of one carve, built again under outputs/p5c5 with P5c's functions unchanged: every question on its
    final pool (P5b's unit), and on a training dataset's fit carve the check against P5b's filed base. Folds play no
    part here."""
    b = bld()
    A, B = b.A, b.B
    if (name, carve) not in b.CARVES:
        raise SystemExit(f"zp5c5: {name}/{carve}: no screen split reads this carve")
    t0 = time.time()
    E, _F, S4 = A.modules()
    op = A.opened(host)
    ds, emb = A.dense_docs(op, name)
    frame, tables, tshas, made = B.tables_for(name, emb, limit)
    _stores, adj = A.family_stores(S4, ds, name, A.csr_dir(op))
    if any(ip.size - 1 != emb.n_rows for ip, _c in adj):
        raise SystemExit(f"zp5c5: {name}: a family's store is not the corpus's size")
    get, pinfo = B.finals_for(name, limit)
    arr, crec = b.collect_all(op, E, name, carve, threads, limit, frame, tables, adj, get)
    same = b.same_as_p5b(name, arr) if (not limit and name in b.TRAIN5 and carve == A.fit_carve(name)) else None
    if limit:
        log(f"zp5c5 units {name}/{carve} development run: {crec['kept']} questions, {int(arr['cand'].size)} "
            f"candidates; tables: {made}; pools: {B.pools_said(pinfo[carve])}; {time.time() - t0:.0f}s, peak "
            f"{A.peak_rss() / 2 ** 30:.1f} GB; nothing written")
        return 0
    p, j = units_paths(name, carve, out)
    b.save_npz(p, **arr)
    rec = {"declared_in": DECLARED, "stage": "units", "dataset": name, **crec, "pools": pinfo[carve],
           "tables": tshas, "npz_sha256": b.sha_file(p), "candidates": int(arr["cand"].size),
           "seeds": int(arr["seeds"].size), "same_as_p5b": same, "index_time": B.INDEX_TIME,
           "freeze_RECORD_SHA256": op.freeze["RECORD_SHA256"], "script_sha256": script_sha256(),
           "p5c_script_sha256": b.script_sha256(), "p5b_script_sha256": B.script_sha256(),
           "p5a_script_sha256": A.script_sha256(), "peak_rss_bytes": A.peak_rss(),
           "seconds": round(time.time() - t0, 1), "utc": A.utc()}
    b.write_json(j, rec)
    log(f"zp5c5 units {name}/{carve}: {crec['kept']} questions ({crec['with_a_counted_gold']} with a counted gold), "
        f"{rec['candidates']} candidates" + (f"; P5b's {same['questions_checked']} questions reproduce" if same else "")
        + f" ({rec['seconds']:.0f}s)")
    return 0


def load_units(name, carve, root=UNITS):
    b = bld()
    p, j = units_paths(name, carve, root)
    if not j.exists():
        raise SystemExit(f"zp5c5: {j} missing (zp5c5.py units {name} {carve})")
    rec = json.loads(j.read_text(encoding="utf-8"))
    if rec.get("declared_in") != DECLARED or rec.get("stage") != "units" or b.sha_file(p) != rec["npz_sha256"]:
        raise SystemExit(f"zp5c5: {p}: not P5c5's units file, or not its record's")
    with np.load(p) as z:
        arr = {k: z[k] for k in z.files}
    if set(arr) != b.B2.KEYS:
        raise SystemExit(f"zp5c5: {p}: keys {sorted(arr)}")
    return arr, rec, b.sha_file(j)


def counted(arr):
    """Per question (file order): does it have a counted gold, a gold among its candidates."""
    cp = np.asarray(arr["cand_ptr"], np.int64)
    gold = np.asarray(arr["gold"], bool)
    return np.array([gold[cp[i]:cp[i + 1]].any() for i in range(cp.size - 1)], bool)


def fit_index(name, arr, k, folds=FOLDS):
    """The questions g_k fits on, in file order: those with a counted gold, in every fold but k (at two folds, P5c's
    fold_index of the other fold)."""
    f = np.array([fold_of(name, q, folds) for q in arr["ids"].tolist()], np.int64)
    return np.flatnonzero((f != k) & counted(arr))


def fitg_stage(split, k, device, threads, out=GDIR, units_root=UNITS, epochs=None, folds=FOLDS):
    """g_k of one split: P5c's fit (P5b's g, P5b2's held-out stop inside the fit questions, unchanged) on the split's
    four training datasets' fit questions with a counted gold outside fold k. The record lists every question it
    fitted on (train and held out), so the columns can check that no question takes its columns from a g that saw it."""
    b = bld()
    A, B, B2 = b.A, b.B, b.B2
    epochs = b.EPOCHS if epochs is None else epochs
    if split not in b.SPLITS or k not in range(folds):
        raise SystemExit(f"zp5c5: fitg takes a split of {sorted(b.SPLITS)} and a fold of 0..{folds - 1}")
    torch.set_num_threads(threads)
    A.full_fp32(torch)
    dev = torch.device(device)
    t0 = time.time()
    names = list(b.SPLITS[split])
    if b.HELD_OUT[split] in names:
        raise SystemExit(f"zp5c5: {split}: its held-out dataset would train g")
    packs, hos, news, counts, urecs, kap, tshas, fit_ids = [], [], [], {}, {}, {}, {}, {}
    for name in names:
        arr, urec, urecs[name] = load_units(name, A.fit_carve(name), units_root)
        _frame, tables, tshas[name] = B.filed_tables(name)
        if tshas[name] != urec["tables"]:
            raise SystemExit(f"zp5c5: {name}: the units ran on other tables")
        idx = fit_index(name, arr, k, folds)
        sub = B2.subset(arr, idx)
        fit_ids[name] = [str(q) for q in sub["ids"].tolist()]
        if any(fold_of(name, q, folds) == k for q in fit_ids[name]):
            raise SystemExit(f"zp5c5: {split}: g_{k} would fit on a question of fold {k}")
        a_tr, a_ho, new, counts[name] = B2.split_base(sub)
        counts[name]["fit_questions_with_a_counted_gold"] = int(idx.size)
        kap[name] = B.kappa_of(tables["V"])
        packs.append(B.Pack(a_tr, tables["V"], kap[name], "V"))
        hos.append(B.Pack(a_ho, tables["V"], kap[name], "V"))
        news.append(new)
    log(f"zp5c5 {split} g_{k} (folds {', '.join(str(j) for j in range(folds) if j != k)} of {folds}): G on V, "
        f"{sum(p.n for p in packs)} train and {sum(h.n for h in hos)} held-out questions of {', '.join(names)}")
    g, hist, kept = B2.fit_stop(torch, dev, packs, hos, news, names, epochs)
    pt, js = g_paths(split, k, out)
    config = {"d_in": B.D_IN, "width": B.WIDTH, "heads": B.HEADS, "dim": B.DIM, "rank_c": B.RANK_C,
              "weights": B.N_WEIGHTS}
    B.save_g(pt, {"G": g}, config)
    rec = {"declared_in": DECLARED, "stage": "fitg", "split": split, "fold": k, "folds": folds,
           "fits_on_folds": [j for j in range(folds) if j != k], "datasets": names,
           "held_out_dataset": b.HELD_OUT[split], "units_records": urecs, "tables": tshas, "split_counts": counts,
           "fit_question_ids": fit_ids, "kappa": kap, "kept_epoch": kept, "beta": g.beta.detach().item(),
           "hist": hist,
           "settings": {"hold": B2.HOLD, "split_seed": B2.SPLIT_SEED, "max_epochs": epochs, "stop": B2.STOP_RULE,
                        "batch": B.BATCH, "lr": B.LR, "optimizer": "Adam", "seed": 0,
                        "precision": "float32, TF32 off", "kappa_pairs": B.KAPPA_PAIRS, **config,
                        "build": build_settings(folds)},
           "g_sha256": b.sha_file(pt), "script_sha256": script_sha256(), "p5c_script_sha256": b.script_sha256(),
           "p5b_script_sha256": B.script_sha256(), "p5b2_script_sha256": B2.script_sha256(), "device": str(dev),
           "gpu_peak_bytes": A.gpu_peak(torch, dev), "peak_rss_bytes": A.peak_rss(),
           "seconds": round(time.time() - t0, 1), "utc": A.utc()}
    b.write_json(js, rec)
    log(f"zp5c5 {split} g_{k}: kept epoch {kept}, beta {rec['beta']:.3f} ({rec['seconds']:.0f}s)")
    return 0


def load_fit(split, k, root=GDIR, folds=FOLDS):
    b = bld()
    pt, js = g_paths(split, k, root)
    if not js.exists():
        raise SystemExit(f"zp5c5: {js} missing (zp5c5.py fitg {split} {k})")
    rec = json.loads(js.read_text(encoding="utf-8"))
    if rec.get("declared_in") != DECLARED or rec.get("stage") != "fitg" or b.sha_file(pt) != rec["g_sha256"]:
        raise SystemExit(f"zp5c5: {pt}: not P5c5's g, or not its record's")
    if (rec["settings"]["build"] != build_settings(folds) or rec.get("folds") != folds or rec.get("fold") != k
            or rec.get("split") != split):
        raise SystemExit(f"zp5c5: {js}: fitted under other settings, folds, fold or split")
    blob = torch.load(pt, map_location="cpu", weights_only=True)
    g = b.B.new_g(torch, torch.device("cpu"))
    g.load_state_dict(blob["models"]["G"])
    g.eval()
    return g, rec, b.sha_file(js)


def cols_stage(name, carve, threads, host=False, out=COLS_DIR, units_root=UNITS, g_root=GDIR, cache_root=None,
               look_root=None, placement=None, folds=FOLDS):
    """P5c's columns of one carve (its loop, its functions) for every split that reads it, each question's from g of
    its own fold, with lp beside them (cols.npy: rows x 4, lp.npy: rows x 1, float16). Stops if a question would take
    its columns from a g whose record lists it among its fit questions."""
    b = bld()
    A, B = b.A, b.B
    cache_root = b.CACHE if cache_root is None else cache_root      # P1c2's cache and look, as P5c's
    look_root = b.LOOK if look_root is None else look_root
    torch.set_num_threads(threads)
    t0 = time.time()
    splits = [sp for sp in b.SPLITS if (name, carve) in b.CARVES_OF[sp]]
    if not splits:
        raise SystemExit(f"zp5c5: {name}/{carve}: no screen split reads this carve")
    arr, urec, usha = load_units(name, carve, units_root)
    _frame, tables, tshas = B.filed_tables(name)
    if tshas != urec["tables"]:
        raise SystemExit(f"zp5c5: {name}: the units ran on other tables")
    U = tables["V"]
    kappa = B.kappa_of(U)
    fits, saw = {}, {}
    for sp in splits:
        fits[sp] = [load_fit(sp, k, g_root, folds) for k in range(folds)]
        for _g, grec, _s in fits[sp]:
            if name in grec["datasets"] and grec["kappa"][name] != kappa:
                raise SystemExit(f"zp5c5: {name}: kappa is not the one {sp}'s g was fitted with")
            if name in grec["datasets"] and grec["units_records"][name] != usha and carve == A.fit_carve(name):
                raise SystemExit(f"zp5c5: {name}/{carve}: these units are not the ones {sp}'s g was fitted on")
        saw[sp] = [frozenset(grec["fit_question_ids"].get(name, ())) for _g, grec, _s in fits[sp]]
    at = {str(q): i for i, q in enumerate(arr["ids"].tolist())}
    d, files, ids, head = LC.look_chunks(name, carve, look_root)
    parts, recs = LC.part_dirs(name, carve, cache_root)
    done = {}
    for p, r in zip(parts, recs):
        if r["look"]["carve_ids_sha256"] != head["carve_ids_sha256"] or r["look"]["records"] != head["records"]:
            raise SystemExit(f"zp5c5: {p}: built from another look than {d}")
        tp = time.time()
        step1 = LC.sha_file(p / "record.json")
        if LC.sha_file(p / "n.npy") != r["arrays"]["n"]["sha256"]:
            raise SystemExit(f"zp5c5: {p / 'n.npy'}: sha256 is not its record's")
        n = np.load(p / "n.npy").astype(np.int64)
        q0, q1 = r["query_range"]
        lo, hi = r["chunks"]
        if (n.size != q1 - q0 or n.size != r["queries"] or int(n.sum()) != r["rows"]
                or r.get("ids", ids[q0:q1]) != ids[q0:q1]):
            raise SystemExit(f"zp5c5: {p}: its questions, ids and rows do not agree")
        xs = {sp: [] for sp in splits}
        lps = {sp: [] for sp in splits}
        fold_rows = np.zeros(folds, np.int64)
        checked, worst_lp, gq, no_seed, guarded = 0, 0.0, q0, 0, 0
        for ch in range(lo, hi):
            with np.load(files[ch]) as zf:
                qps = zf["q_pool_size"].astype(np.int64)
                pools = zf["pool"].astype(np.int64)
            ro = np.r_[0, np.cumsum(qps)]
            if int(ro[-1]) != pools.size:
                raise SystemExit(f"zp5c5: {files[ch]}: its pool and question sizes do not agree")
            for j in range(qps.size):
                if qps[j] != n[gq - q0]:
                    raise SystemExit(f"zp5c5: {p}: question {gq} has {qps[j]} look rows and {n[gq - q0]} cache rows")
                look_pool = pools[ro[j]:ro[j + 1]]
                gq += 1
                if not look_pool.size:
                    continue
                qid = str(ids[gq - 1])
                i = at.get(qid)
                if i is None:
                    raise SystemExit(f"zp5c5: {name}/{carve}: question {qid} is in the look, not in the units")
                F, US, seeds, pool = b.question_inputs(arr, i, U)
                if not np.array_equal(np.sort(look_pool), pool):
                    raise SystemExit(f"zp5c5: {name}/{carve}: question {qid}: the cache's rows are not its final pool")
                f = fold_of(name, qid, folds)
                fold_rows[f] += look_pool.size
                for sp in splits:       # the cross-fit's guarantee: the g that scores the question never fitted on it
                    if qid in saw[sp][f]:
                        raise SystemExit(f"zp5c5: {name}/{carve}: question {qid} is in fold {f}, and {sp}'s g_{f} "
                                         f"fitted on it")
                guarded += 1
                if not seeds.size:
                    for sp in splits:
                        xs[sp].append(np.full((look_pool.size, NCOL), -CLIP))
                        lps[sp].append(np.full(look_pool.size, -CLIP))
                    no_seed += 1
                    continue
                cm = ~np.isin(look_pool, seeds)
                UP = B.rows_of(U, look_pool, np.float32)
                for sp in splits:
                    g = fits[sp][f][0]
                    x, lp = b.columns(torch, g, kappa, F, US, UP, cm)
                    if checked < 2 and cm.any():
                        ref = b.p5b_lp(torch, g, kappa, F, US, UP[cm])
                        dev = float(np.abs(ref - lp[cm]).max())
                        if dev > 2e-3:
                            raise SystemExit(f"zp5c5: {name}/{carve} question {qid}: lp is off P5b's log p by "
                                             f"{dev:.2e}")
                        worst_lp = max(worst_lp, dev)
                    xs[sp].append(x)
                    lps[sp].append(lp)
                checked += 1
        if gq != q1:
            raise SystemExit(f"zp5c5: {p}: its chunks hold {gq - q0} questions, its record {q1 - q0}")
        for sp in splits:
            x = np.concatenate(xs[sp]) if xs[sp] else np.zeros((0, NCOL))
            lp = np.concatenate(lps[sp]) if lps[sp] else np.zeros(0)
            if x.shape[0] != int(n.sum()) or lp.shape[0] != x.shape[0] or not (
                    np.isfinite(x).all() and np.isfinite(lp).all()):
                raise SystemExit(f"zp5c5: {p}: {x.shape[0]} rows of columns, {lp.shape[0]} of lp, against "
                                 f"{int(n.sum())}, or one not finite")
            x16, y = b.stored(x)
            l16, ly = b.stored(lp[:, None])
            od = Path(out) / sp / name / carve / p.name
            od.mkdir(parents=True, exist_ok=True)
            sha, sha_lp = LC.save_npy(od, "cols", x16), LC.save_npy(od, "lp", l16)
            xx, yy = np.concatenate([x, lp[:, None]], 1), np.concatenate([y, ly], 1)
            qs_ = lambda v: [round(float(t), 4) for t in np.quantile(v, [0.05, 0.5, 0.95])] if v.size else []
            err = (max(float(np.abs(x16.astype(np.float64) - y).max()), float(np.abs(l16.astype(np.float64) - ly).max()))
                   if y.size else 0.0)
            rec = {"declared_in": DECLARED, "stage": "P5c5", "split": sp, "dataset": name, "carve": carve,
                   "part": p.name, "step1_record_sha256": step1, "units_record_sha256": usha,
                   "g_records_sha256": [fits[sp][k][2] for k in range(folds)],
                   "kept_epochs": [fits[sp][k][1]["kept_epoch"] for k in range(folds)],
                   "kappa": kappa, "queries": int(n.size), "rows": int(n.sum()),
                   "rows_by_fold_of_question": fold_rows.tolist(), "questions_without_a_seed": no_seed,
                   "questions_checked_against_their_g": guarded, "settings": build_settings(folds),
                   "quantiles_5_50_95": {c: qs_(yy[:, i]) for i, c in enumerate(READS["zspko5"])},
                   "clipped_share": {c: round(float((np.abs(xx[:, i]) >= CLIP).mean()), 4) if xx.size else 0.0
                                     for i, c in enumerate(READS["zspko5"])},
                   "lp_against_p5b_log_p_max_abs": worst_lp, "max_abs_float16_error": err,
                   "arrays": {"cols": {"dtype": x16.dtype.str, "shape": list(x16.shape), "sha256": sha},
                              "lp": {"dtype": l16.dtype.str, "shape": list(l16.shape), "sha256": sha_lp}},
                   "cache_root": Path(cache_root).resolve().relative_to(ROOT).as_posix()
                   if Path(cache_root).resolve().is_relative_to(ROOT) else "toy",
                   "placement": placement or {"where": "host" if host else "laptop"},
                   "script_sha256": script_sha256(), "p5c_script_sha256": b.script_sha256(),
                   "seconds": round(time.time() - tp, 1), "utc": A.utc()}
            b.write_json(od / "record.json", rec)
        done[p.name] = {"queries": int(n.size), "rows": int(n.sum())}
        log(f"  zp5c5 cols {name}/{carve} {p.name}: {int(n.size)} questions, {int(n.sum())} rows, splits "
            f"{', '.join(splits)}; lp against P5b's log p {worst_lp:.1e} ({time.time() - t0:.0f}s)")
    log(f"zp5c5 cols {name}/{carve}: {len(done)} part(s) ({time.time() - t0:.0f}s)")
    return done


# ── the carve ────────────────────────────────────────────────────────────────


def load_p5c5(c, root, split, cache_root, lp, verify=True):
    """A carve's columns of one split on its device (float16), each part tied to its step-1 record: the four (pt, mx,
    pt0, mx0), and lp beside them when lp (the GNN's carve). The MLP's carve never opens lp.npy."""
    parts, _recs = LC.part_dirs(c.ds, c.carve, cache_root)
    keys = (("cols", NCOL), ("lp", 1)) if lp else (("cols", NCOL),)
    xs, shas = [], {}
    for p in parts:
        d = Path(root) / split / c.ds / c.carve / p.name
        rp = d / "record.json"
        if not rp.exists():
            raise SystemExit(f"zp5c5: {rp} missing (zp5c5.py cols {c.ds} {c.carve})")
        h = json.loads(rp.read_text(encoding="utf-8"))
        if h.get("declared_in") != DECLARED or h.get("stage") != "P5c5" or h.get("settings") != BUILD_SETTINGS:
            raise SystemExit(f"zp5c5: {rp} is not P5c5's build")
        if h.get("split") != split or h.get("dataset") != c.ds or h.get("carve") != c.carve:
            raise SystemExit(f"zp5c5: {rp} holds {h.get('split')}'s columns of {h.get('dataset')}/{h.get('carve')}, "
                             f"not {split}'s of {c.ds}/{c.carve}")
        if h["step1_record_sha256"] != c.part_shas[p.name]:
            raise SystemExit(f"zp5c5: {rp} was built on another step-1 part")
        got = []
        for key, width in keys:
            fn = d / f"{key}.npy"
            if not fn.exists() or key not in h.get("arrays", {}):
                raise SystemExit(f"zp5c5: {fn} missing, or not in its record (zp5c5.py cols {c.ds} {c.carve})")
            if verify and LC.sha_file(fn) != h["arrays"][key]["sha256"]:
                raise SystemExit(f"zp5c5: {fn}: sha256 is not its record's")
            x = np.load(fn)
            if x.dtype != np.float16 or list(x.shape) != h["arrays"][key]["shape"] or x.shape[1:] != (width,):
                raise SystemExit(f"zp5c5: {fn}: {x.dtype} {x.shape}, not its record's float16 (rows, {width})")
            got.append(x)
        if len({x.shape[0] for x in got}) != 1:
            raise SystemExit(f"zp5c5: {d}: cols and lp hold different rows")
        xs.append(np.concatenate(got, 1) if len(got) > 1 else got[0])
        shas[p.name] = LC.sha_file(rp)
    x = np.concatenate(xs)
    if x.shape[0] != int(c.off_np[-1]):
        raise SystemExit(f"zp5c5: {c.ds}/{c.carve}: {x.shape[0]} rows of columns against the cache's "
                         f"{int(c.off_np[-1])}")
    return torch.from_numpy(x).to(c.device), shas


def p5c5_carve(base, lp, root=COLS_DIR):
    """base's carve class with P5c5's columns of the split set by on_split beside it: batch() adds them under P5C_KEY
    (four columns, or five with lp); every block, row, side key and gold is base's, unchanged."""

    class P5c5Carve(base):
        P5C5_ROOT = root
        P5C5_LP = lp
        P5C5_SPLIT = None

        def __init__(self, ds, carve, basis, device, cache_root=LC.OUT, verify=True, score2=False):
            sp = type(self).P5C5_SPLIT
            if sp not in SPLITS:
                raise SystemExit(f"zp5c5: no screen split is set for P5c5's columns ({sp}); run under on_split")
            super().__init__(ds, carve, basis, device, cache_root, verify, score2)
            self.p5c_split = sp
            self.p5c, self.p5c_records = load_p5c5(self, type(self).P5C5_ROOT, sp, cache_root, type(self).P5C5_LP,
                                                   verify)

        def nbytes(self):
            return super().nbytes() + self.p5c.numel() * self.p5c.element_size()

        def batch(self, qs, blocks):
            feats, nq, base_z, gold = super().batch(qs, blocks)
            feats[P5C_KEY] = P5.p5c_batch(self, qs)
            return feats, nq, base_z, gold

    P5c5Carve.__name__ = P5c5Carve.__qualname__ = f"P5c5{base.__name__}"
    return P5c5Carve


P5c5KTChainCarveZRC = p5c5_carve(Z3.KTChainCarveZRC, lp=False)
P5c5KTLinkCarveBase = p5c5_carve(Z3.KTLinkCarveBase, lp=True)
CARVES = (P5c5KTChainCarveZRC, P5c5KTLinkCarveBase)


@contextlib.contextmanager
def on_split(split, carves=CARVES):
    """Every P5c5 carve class reads the split's columns inside, and none outside."""
    if split not in SPLITS:
        raise SystemExit(f"zp5c5: P5c5's columns are built for {', '.join(SPLITS)}, not {split}")
    saved = [c.P5C5_SPLIT for c in carves]
    for c in carves:
        c.P5C5_SPLIT = split
    try:
        yield
    finally:
        for c, s in zip(carves, saved):
            c.P5C5_SPLIT = s


# ── the head ─────────────────────────────────────────────────────────────────


def p5c5_init(m, seed, ncol):
    """P5c's head over ncol columns (at four, P5c's p5c_init exactly): the first layer from its own generator
    (seed + 5701), the last at zero."""
    g = torch.Generator().manual_seed(int(seed) + P5C_SEED_OFF)
    m.p5c_w1 = nn.Parameter((torch.rand(P5C_HID, ncol, generator=g) * 2 - 1) / ncol ** 0.5)
    m.p5c_b1 = nn.Parameter(torch.zeros(P5C_HID))
    m.p5c_w2 = nn.Parameter(torch.zeros(1, P5C_HID))
    m.p5c_b2 = nn.Parameter(torch.zeros(1))


class ZRCKPO5(P5.ZRCKPO):
    """zrckpo5: zrckpo's model unchanged (zrckp plus P5c's head over a row's own four columns); only its columns differ,
    g cross-fitted over five folds."""


class P5c5Mix(Z3.KTMix):
    """zrm's score, the KT head, then the head over five columns (the four and lp): in zspko5, the score zsp's
    propagation reads."""

    def __init__(self, blocks, widths, hidden=128, dropout=0.1, seed=0, arm="ctl", ctx="none"):
        super().__init__(blocks, widths, hidden, dropout, seed, arm, ctx)
        p5c5_init(self, seed, NCOL_G)

    p5c_head = P5.p5c_head

    def forward(self, feats, keep, nq, B, base_z):
        s = super().forward(feats, keep, nq, B, base_z)
        if P5C_KEY not in feats:
            return s
        return s + self.p5c_head(feats[P5C_KEY])


class ZSPKO5(ZP.ZProp, P5c5Mix):
    """zspko5: zspk's model with the five-column head added before zsp's propagation reads the score (MRO: ZProp,
    P5c5Mix, KTMix, ZRM)."""


ARMS = {"zrckpo5": (ZRCKPO5, P5c5KTChainCarveZRC), "zspko5": (ZSPKO5, P5c5KTLinkCarveBase)}
S.ARMS.update(ARMS)


# ── train and read ───────────────────────────────────────────────────────────


def p5c5_records(carves, split, root=COLS_DIR, cache_root=None):
    """Each carve part's P5c5 record sha256 for the split; stops when one is missing or not P5c5's of that split."""
    out = {}
    for ds, cv in carves:
        parts, _ = LC.part_dirs(ds, cv, cache_root or Z2.CACHE)
        for p in parts:
            rp = Path(root) / split / ds / cv / p.name / "record.json"
            if not rp.exists():
                raise SystemExit(f"zp5c5: {rp} missing (zp5c5.py cols {ds} {cv})")
            h = json.loads(rp.read_text(encoding="utf-8"))
            if h.get("declared_in") != DECLARED or h.get("split") != split or h.get("stage") != "P5c5":
                raise SystemExit(f"zp5c5: {rp} is not P5c5's of {split}")
            out[f"{ds}/{cv}/{p.name}"] = LC.sha_file(rp)
    return out


def split_of_fit(name):
    for a in ARM_NAMES:
        for sp, n in FIT[a].items():
            if n == name:
                return a, sp
    return None, None


def stamp(argv, arm, split):
    name, out_root = S2.where(argv)
    sj = out_root / name / "screen.json"
    if sj.exists():
        rec = json.loads(sj.read_text(encoding="utf-8"))
        rec.update({"zp5c5_sha256": LC.sha_src(__file__), "zp5c_sha256": LC.sha_src(P5.__file__),
                    "zp1c3p_sha256": LC.sha_src(Z3P.__file__), "zp1c3_sha256": LC.sha_src(Z3.__file__),
                    "zp1c2_sha256": LC.sha_src(Z2.__file__), "p5c5": {
                        "declared_in": DECLARED, "arm": arm, "base_arm": BASE_OF[arm], "split": split,
                        "device": Z2.DEVICE, "head": list(P5C), "reads": list(READS[arm]), "settings": SETTINGS[arm],
                        "pools": "P1c2's (outputs/p1c2/pools)", "cache_root": Z2.CACHE.relative_to(ROOT).as_posix(),
                        "gate": LC.sha_file(Z2.GATE),
                        "p5c5_records": p5c5_records(Z3.train_carves(split), split),
                        "p5c5_eval_records": p5c5_records(Z3.eval_carves(), split),
                        "kt_records": Z3.kt_records(Z3.train_carves(split)), "rmatch_records": RM.built_records(),
                        "zrc_records": Z2._BUILT(Z2.CH_ZRC), "zlink_records": ZL.built_records()}})
        LC.write_json(sj, rec)


def train(argv, split):
    arm = R.arm_of(argv)
    if arm not in ARMS or S.ARMS.get(arm) != ARMS[arm]:
        raise SystemExit(f"zp5c5: train takes --arm zrckpo5 or zspko5, not {arm} ({S.ARMS.get(arm)})")
    if split not in SPLITS:
        raise SystemExit(f"zp5c5: P5c5's screen fits L-musique and L-hotpotqa, not {split}")
    name, _ = S2.where(argv)
    if name != FIT[arm][split]:
        raise SystemExit(f"zp5c5: {arm}'s fit of {split} is {FIT[arm][split]}, not {name}")
    Z2.cache_root_of(argv)
    argv = Z2.on_card(argv)
    Z2.gate_passed()
    Z3.kt_records(Z3.train_carves(split))
    p5c5_records(Z3.train_carves(split), split)
    p5c5_records(Z3.eval_carves(), split)
    with on_split(split), Z2.on_p1c2():
        rc = RM.train(argv, split)
        stamp(argv, arm, split)
    return rc


def read(argv):
    name, out_root = S2.where(argv)
    sj = out_root / name / "screen.json"
    rec = json.loads(sj.read_text(encoding="utf-8")) if sj.exists() else {}
    arm, split = split_of_fit(name)
    if rec.get("arm") not in ARMS or "p5c5" not in rec or rec.get("arm") != arm:
        raise SystemExit(f"zp5c5: {name} is not a P5c5 fit (arm {rec.get('arm')})")
    if rec["p5c5"].get("split") != split:
        raise SystemExit(f"zp5c5: {name} is {split}'s fit, its record says {rec['p5c5'].get('split')}")
    Z2.cache_root_of(argv)
    Z3.kt_records(Z3.eval_carves())
    p5c5_records(Z3.eval_carves(), split)
    with on_split(split), Z2.on_p1c2():
        return RM.read(Z2.on_card(argv))


def compare(rest):
    return S2.main(["compare"] + rest)


# ── relz's pair and re-call, each arm decided against its base arm's card fits ───────────────────────────────


@contextlib.contextmanager
def on_base(arm):
    if arm not in ARMS:
        raise SystemExit(f"zp5c5: --arm takes {' or '.join(ARM_NAMES)}, not {arm}")
    fits = BASE_FITS[arm]

    def base_fit(split):
        if split not in fits:
            raise SystemExit(f"zp5c5: no base fit of {BASE_OF[arm]} on {split}")
        return fits[split]

    saved = Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit
    Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit = arm, BASE_OF[arm], fits, BASE_SCREENS[arm], base_fit
    try:
        yield
    finally:
        Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit = saved


def fix(arm):
    b, f = BASE_OF[arm], BASE_FITS[arm]["L-musique"][2]
    return (("(rel's screen fits scr-rel and scr-rel-hp)", f"({b}'s card fits {f} and {f}-hp)"),
            ("(rel's screen fits)", f"({b}'s card fits)"),
            ("| rel R@5 | relz R@5 |", f"| {b} R@5 | {arm} R@5 (P5c5's five-fold offset input) |"),
            ("section 2 and the tenth round", f"section 2, and {DECLARED}"))


def restamp(out, arm):
    out = Path(out)
    md = out.with_suffix(".md")
    if md.exists():
        t = md.read_text(encoding="utf-8")
        for a, b in fix(arm):
            t = t.replace(a, b)
        md.write_text(f"P5c5 ({DECLARED}): {arm} ({WHAT[arm]}) against {BASE_OF[arm]}.\n\n" + t, encoding="utf-8")
    js = out.with_suffix(".json")
    if js.exists():
        rec = json.loads(js.read_text(encoding="utf-8"))
        rec.update({"stage": "P5c5", "declared_in": DECLARED, "decided_against": AGAINST[arm], "device": Z2.DEVICE,
                    "zp5c5_sha256": LC.sha_src(__file__)})
        rec.pop("round", None)
        LC.write_json(js, rec)


def pair(arm, screens, out):
    with on_base(arm):
        rec = Z.pair(screens, out)
    restamp(out, arm)
    return rec


def recall(arm, null_files, pair_file, out, base_screens=None):
    with on_base(arm):
        rec = Z.recall(null_files, pair_file, out, base_screens)
    restamp(out, arm)
    return rec


# ── the verdict ──────────────────────────────────────────────────────────────


def tag(r):
    return f"{r['split']} {r['dataset']}"


NEXT = {"ADOPT_FOR_FULL_RUN": "the arm fitted again on P1g-b's pools, then its full run, its own declared file",
        "INCOMPLETE": "the missing re-call is run, nothing else changes"}
BACKOFF = {"zrckpo5": "the back-off form, its own declared file: P5b3's gate, refit per split and cross-fitted over the "
                      "same five folds, as a fifth column beside the four, from the six of its inputs the MLP may read "
                      "(log n, ring1, ring2, log seeds, pi_max, delta_rms)",
           "zspko5": "the back-off form, its own declared file: P5b3's gate, refit per split and cross-fitted over the "
                     "same five folds, as a sixth column beside the five, from all ten of its inputs"}
JOIN = {a: "the other arm's back-off reaches this model too (docs/SCREENS.md, 3b): " + BACKOFF[a] for a in ARM_NAMES}
REACH = {"zrckpo5": "none on the MLP: it read the four columns at two folds (P5c) and at five (P5c5) with no GAIN, and "
                    "may not read lp; zspko5's full-run file decides under F1 whether the GNN keeps an input the MLP "
                    "does not read",
         "zspko5": "zrckpo5's input on the GNN, its own declared file: zspk with the four five-fold columns and no lp, "
                   "one input set for both models (F1)"}
WAIT = "decided with the other arm's verdict, once its missing re-call is run"
CLOSE = ("none: the learned offsets close as a fixed input to the lean pair, this file named as the close "
         "(docs/VERDICT_AUDIT_2026_10_17.md, section 6): they reached both models at two folds and at five, the GNN "
         "also with P5b's normalised log p, with no GAIN on any read")


def next_form(arm, per):
    """Section 5's next form for the arm, from its own outcome and the other arm's."""
    me, other = per[arm], per[OTHER[arm]]
    if me["verdict"] in NEXT:
        return NEXT[me["verdict"]]
    if me["gains"]:
        return BACKOFF[arm]
    if other["verdict"] == "INCOMPLETE":
        return WAIT
    if other["verdict"] == "ADOPT_FOR_FULL_RUN":
        return REACH[arm]
    if other["gains"]:
        return JOIN[arm]
    return CLOSE


def verdict(recalls=None, out=None):
    """P5c5's screen verdict, per arm, from its re-call (section 4); webqsp and metaqa beside, musique's in-domain
    read of both arms side by side, and the next form each outcome names (section 5)."""
    recalls = recalls or {a: RECALL[a].with_suffix(".json") for a in ARM_NAMES}
    per = {}
    for a in ARM_NAMES:
        rp, problems, rows = Path(recalls[a]), [], []
        if not rp.exists():
            problems.append(f"{rp.name} missing")
        else:
            rec = json.loads(rp.read_text(encoding="utf-8"))
            if rec.get("stage") != "P5c5" or rec.get("declared_in") != DECLARED or rec.get("arm") != a:
                problems.append(f"{rp.name} is not P5c5's re-call of {a}")
            rows = rec.get("rows", [])
            if sorted((r.get("split"), r.get("dataset")) for r in rows) != sorted(
                    (sp, ds) for sp in SPLITS for ds in LG.EVAL_ORDER):
                problems.append(f"{rp.name}: {len(rows)} reads, not the twelve of {', '.join(SPLITS)} on the six")
        gains = [tag(r) for r in rows if r.get("call") == "GAIN"]
        losses = [tag(r) for r in rows if r.get("call") == "LOSS"]
        v = "INCOMPLETE" if problems else ("NOT_ADOPTED" if losses or not gains else "ADOPT_FOR_FULL_RUN")
        per[a] = {"verdict": v, "gains": gains, "losses": losses, "problems": problems,
                  "calls": {tag(r): r.get("call") for r in rows}, "delta_r5": {tag(r): r.get("delta") for r in rows},
                  "beside": {tag(r): {"call": r.get("call"), "delta": r.get("delta")} for r in rows
                             if r.get("dataset") in BESIDE},
                  "recall_sha256": LC.sha_file(rp) if rp.exists() else None}
    for a in ARM_NAMES:
        per[a]["next_form"] = next_form(a, per)
    ck = f"{CLAIM1_READ[0]} {CLAIM1_READ[1]}"
    side = {a: {"call": per[a]["calls"].get(ck), "delta": per[a]["delta_r5"].get(ck)} for a in ARM_NAMES}
    rec = {"stage": "P5c5", "declared_in": DECLARED, "rule": RULE, "arms": per, "musique_in_domain_beside": side,
           "zp5c5_sha256": LC.sha_src(__file__), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    md = [f"# P5c5's screen verdict: zrckpo5 **{per['zrckpo5']['verdict']}**, zspko5 **{per['zspko5']['verdict']}**",
          "", f"Rule: {RULE}.", ""]
    num = lambda x: f"{x:+.4f}" if isinstance(x, (int, float)) else "-"
    for a in ARM_NAMES:
        p = per[a]
        md.append(f"- **{a}** ({WHAT[a]}) against {AGAINST[a]}, after the seed-null re-call: {len(p['gains'])} GAIN "
                  f"({', '.join(p['gains']) or 'none'}), {len(p['losses'])} LOSS ({', '.join(p['losses']) or 'none'})"
                  + (f"; problems: {'; '.join(p['problems'])}" if p["problems"] else "") + ". Beside: "
                  + (", ".join(f"{k} {b['call']} ({num(b['delta'])})" for k, b in p["beside"].items()) or "none")
                  + f". Next: {p['next_form']}.")
    md += ["", f"Beside, deciding nothing: musique in-domain ({ck}): "
           + ", ".join(f"{a} {side[a]['call']} ({num(side[a]['delta'])})" for a in ARM_NAMES) + ".",
           "", "This verdict closes only this form (the five-fold columns through a pointwise head, and lp on the "
           "GNN); docs/SCREENS.md, 3b."]
    out = Path(out or VERDICT)
    out.parent.mkdir(parents=True, exist_ok=True)
    LC.write_json(out.with_suffix(".json"), rec)
    out.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    log("zp5c5 verdict: " + ", ".join(f"{a} {per[a]['verdict']}" for a in ARM_NAMES))
    return rec


# ── selftest ─────────────────────────────────────────────────────────────────


def tree(p):
    """Every file under p with its size and mtime (None when p does not exist)."""
    p = Path(p)
    if not p.exists():
        return None
    return sorted((x.relative_to(p).as_posix(), x.stat().st_size, x.stat().st_mtime_ns) for x in p.rglob("*")
                  if x.is_file())


def same_g(a, b):
    sa, sb = a.state_dict(), b.state_dict()
    return sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)


def builder_selftest(rng):
    """The builder on a toy (P5c's toy units and look): at two folds it gives P5c's builder's g and columns bit for
    bit; at five, every question's columns come from the one g whose fit never held it."""
    import inspect
    import tempfile
    b = bld()
    A, B = b.A, b.B
    # 1. the roots: every build of this file is under outputs/p5c5, none under P5c's outputs/p5c; P5c's builder's
    #    splits, carves, columns and transform are the ones used
    for fn, keys in ((units_stage, ("out",)), (load_units, ("root",)), (fitg_stage, ("out", "units_root")),
                     (load_fit, ("root",)), (cols_stage, ("out", "units_root", "g_root")), (units_paths, ("root",)),
                     (g_paths, ("root",)), (p5c5_carve, ("root",)), (p5c5_records, ("root",))):
        for kk in keys:
            dflt = Path(inspect.signature(fn).parameters[kk].default).resolve()
            assert dflt.is_relative_to(OUT.resolve()) and not dflt.is_relative_to(b.OUT.resolve()), (fn, kk)
    assert all(Path(c.P5C5_ROOT) == COLS_DIR for c in CARVES) and OUT != b.OUT
    assert b.CACHE.parent.name == "p1c2" and b.LOOK.parent.name == "p1c2"
    assert (b.COLNAMES, b.NCOL, b.CLIP, b.SCALE, b.FOLDS) == (COLNAMES, NCOL, CLIP, SCALE, 2)
    assert tuple(b.SPLITS) == tuple(SPLITS) and P5.BUILD_SETTINGS == b.BUILD_SETTINGS
    # 2. folds: deterministic, all five used, about a fifth each; at two folds P5c's parity; g_k's questions are every
    #    fold but k (at two, P5c's fold_index of the other fold)
    f5 = np.array([fold_of("musique", f"q{k}") for k in range(4000)])
    assert set(f5.tolist()) == set(range(FOLDS)) and max(abs((f5 == k).mean() - 0.2) for k in range(FOLDS)) < 0.03
    assert all(fold_of("squad", f"q{k}", 2) == b.fold_of("squad", f"q{k}") for k in range(4000))
    fa = {"ids": np.array(list("abcdefgh")), "cand_ptr": np.array([0, 2, 3, 5, 6, 8, 9, 11, 12]),
          "gold": np.array([0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0], bool)}
    for k in range(2):
        assert fit_index("squad", fa, k, 2).tolist() == b.fold_index("squad", fa, 1 - k).tolist()
    for k in range(FOLDS):
        assert fit_index("squad", fa, k).tolist() == [i for i, q in enumerate("abcdefgh") if fold_of("squad", q) != k
                                                      and i in (0, 2, 4, 6)]
    # 3. end to end on P5c's toy
    N = 400
    U = rng.normal(0, 0.3, (N, B.DIM)).astype(np.float32)
    shift = np.full(B.DIM, 0.12, np.float32)
    tshas = {"frame": "toy-frame", "V": "toy-V", "V0": "toy-V0", "space_record": "toy-space"}
    saved = B.filed_tables
    B.filed_tables = lambda name: ((None, None), {"V": U, "V0": U}, tshas)
    try:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            arrs = {}
            for name in b.TRAIN5:
                arrs[name] = b.toy_units(rng, name, 60, N, U, shift)
                for (pz, pj), decl in ((b.units_paths(name, A.fit_carve(name), tmp / "p5c_units"), b.DECLARED),
                                       (units_paths(name, A.fit_carve(name), tmp / "units"), DECLARED)):
                    b.save_npz(pz, **arrs[name])
                    b.write_json(pj, {"declared_in": decl, "stage": "units", "npz_sha256": b.sha_file(pz),
                                      "tables": tshas})
            # 3a. two folds: P5c's builder and this one on the same units and look give the same g's (this g_k is
            #     P5c's g_(1-k)) and the same columns, bit for bit; the two folds' g's differ, so the match is the
            #     cross-fit's mapping and not g's start
            for sp in b.SPLITS:
                for f in range(2):
                    b.fitg_stage(sp, f, "cpu", 1, out=tmp / "p5c_g", units_root=tmp / "p5c_units", epochs=4)
                    fitg_stage(sp, f, "cpu", 1, out=tmp / "g2", units_root=tmp / "units", epochs=4, folds=2)
            order = b.toy_look(tmp, "metaqa", "fit", arrs["metaqa"], rng)
            ref = b.cols_stage("metaqa", "fit", 1, out=tmp / "p5c_cols", units_root=tmp / "p5c_units",
                               g_root=tmp / "p5c_g", cache_root=tmp / "cache", look_root=tmp / "look")
            two = cols_stage("metaqa", "fit", 1, out=tmp / "cols2", units_root=tmp / "units", g_root=tmp / "g2",
                             cache_root=tmp / "cache", look_root=tmp / "look", folds=2)
            assert sorted(ref) == sorted(two) == ["part_0of2", "part_1of2"] and ref == two
            for sp in b.SPLITS:
                mine = [load_fit(sp, k, tmp / "g2", folds=2)[0] for k in range(2)]
                theirs = [b.load_fit(torch, sp, k, tmp / "p5c_g")[0] for k in range(2)]
                assert same_g(mine[0], theirs[1]) and same_g(mine[1], theirs[0]) and not same_g(mine[0], mine[1])
                for pn in ref:
                    a_, b_ = (r / sp / "metaqa" / "fit" / pn / "cols.npy" for r in (tmp / "p5c_cols", tmp / "cols2"))
                    assert LC.sha_file(a_) == LC.sha_file(b_) and np.array_equal(np.load(a_), np.load(b_))
            # 3b. five folds: g_k fitted on exactly the counted questions of the other four folds of the split's four
            #     datasets, none of fold k, so every counted question sits in four of the five fits
            for sp in b.SPLITS:
                for k in range(FOLDS):
                    fitg_stage(sp, k, "cpu", 1, out=tmp / "g5", units_root=tmp / "units", epochs=4)
            done = cols_stage("metaqa", "fit", 1, out=tmp / "cols5", units_root=tmp / "units", g_root=tmp / "g5",
                              cache_root=tmp / "cache", look_root=tmp / "look")
            assert done == ref
            cq = {nm: {str(q) for q, c in zip(a["ids"].tolist(), counted(a)) if c} for nm, a in arrs.items()}
            kap = B.kappa_of(U)
            arr = arrs["metaqa"]
            at = {str(q): i for i, q in enumerate(arr["ids"].tolist())}
            ch = tmp / "look" / "metaqa" / "fit" / "chunks"
            with np.load(ch / "c00000.npz") as z0, np.load(ch / "c00001.npz") as z1:
                sizes = np.concatenate([z0["q_pool_size"], z1["q_pool_size"]])
                pools = np.concatenate([z0["pool"], z1["pool"]])
            scored = 0
            for sp in b.SPLITS:
                fits = [load_fit(sp, k, tmp / "g5") for k in range(FOLDS)]
                sets = []
                for k, (_g, rec, _s) in enumerate(fits):
                    assert sorted(rec["fit_question_ids"]) == sorted(b.SPLITS[sp]) and rec["fits_on_folds"] == [
                        j for j in range(FOLDS) if j != k]
                    sets.append({nm: set(v) for nm, v in rec["fit_question_ids"].items()})
                    for nm, s_ in sets[-1].items():
                        assert s_ == {q for q in cq[nm] if fold_of(nm, q) != k}
                        assert len(s_) == len(rec["fit_question_ids"][nm])
                for nm in b.SPLITS[sp]:
                    assert all(sum(q in s_[nm] for s_ in sets) == FOLDS - 1 for q in cq[nm])
                assert sum(not same_g(fits[i][0], fits[j][0]) for i in range(FOLDS) for j in range(i)) >= FOLDS - 1
                # 3c. every row's columns and lp are the g of its question's fold's, bit for bit; that g never fitted
                #     on the question, which other g's did; another fold's g gives other columns; lp is P5b's log p
                got = {key: np.concatenate([np.load(tmp / "cols5" / sp / "metaqa" / "fit" / pn / f"{key}.npy")
                                            for pn in sorted(done)]) for key in ("cols", "lp")}
                assert got["cols"].dtype == got["lp"].dtype == np.float16 and got["lp"].shape[1:] == (1,)
                want, wlp, r, seen_elsewhere, differs = [], [], 0, 0, 0
                for q, s in zip(order, sizes.tolist()):
                    if not s:
                        continue
                    pl = pools[r:r + s]
                    r += s
                    F, US, seeds, _pool = b.question_inputs(arr, at[q], U)
                    f = fold_of("metaqa", q)
                    cm = ~np.isin(pl, seeds)
                    UP = B.rows_of(U, pl, np.float32)
                    x, lp = b.columns(torch, fits[f][0], kap, F, US, UP, cm)
                    want.append(b.stored(x)[0])
                    wlp.append(b.stored(lp[:, None])[0])
                    assert q not in sets[f]["metaqa"]
                    seen_elsewhere += q in cq["metaqa"] and all(q in sets[j]["metaqa"] for j in range(FOLDS) if j != f)
                    xh, lph = b.columns_by_hand(fits[f][0], kap, F, US, UP, cm)
                    assert np.abs(b.stored(x)[1] - b.stored(xh)[1]).max() < 1e-4
                    assert np.abs(b.stored(lp)[1] - b.stored(lph)[1]).max() < 1e-4
                    if cm.any():
                        assert np.abs(b.p5b_lp(torch, fits[f][0], kap, F, US, UP[cm]) - lp[cm]).max() < 2e-3
                    j = next((j for j in range(FOLDS) if j != f and not same_g(fits[j][0], fits[f][0])), None)
                    if j is not None:
                        differs += not np.array_equal(b.stored(b.columns(torch, fits[j][0], kap, F, US, UP, cm)[0])[0],
                                                      want[-1])
                    scored += 1
                assert np.array_equal(got["cols"], np.concatenate(want)) and np.array_equal(got["lp"],
                                                                                            np.concatenate(wlp))
                assert seen_elsewhere > 0 and differs > 0, (seen_elsewhere, differs)
                assert float(np.abs(got["lp"].astype(np.float64) - got["cols"][:, :1]).max()) > 1e-3  # lp is not pt
                rec = json.loads((tmp / "cols5" / sp / "metaqa" / "fit" / "part_0of2" / "record.json").read_text(
                    encoding="utf-8"))
                assert rec["settings"] == BUILD_SETTINGS and rec["stage"] == "P5c5" and rec["cache_root"] == "toy"
                assert sum(rec["rows_by_fold_of_question"]) == rec["rows"] and len(rec["rows_by_fold_of_question"]) == 5
                assert rec["questions_checked_against_their_g"] > 0 and set(rec["arrays"]) == {"cols", "lp"}
            assert scored > 0
            # 3d. the guard: a g whose record lists a question of its own fold stops the columns; so do two-fold fits,
            #     a sixth fold, and carves no split reads
            q0 = next(q for q in order if q in cq["metaqa"])
            js = g_paths("L-musique", fold_of("metaqa", q0), tmp / "g5")[1]
            keep = js.read_bytes()
            h = json.loads(keep)
            h["fit_question_ids"]["metaqa"].append(q0)
            b.write_json(js, h)
            b.refused(lambda: cols_stage("metaqa", "fit", 1, out=tmp / "cols_bad", units_root=tmp / "units",
                                         g_root=tmp / "g5", cache_root=tmp / "cache", look_root=tmp / "look"),
                      "a question took its columns from a g that fitted on it")
            js.write_bytes(keep)
            b.refused(lambda: cols_stage("metaqa", "fit", 1, out=tmp / "cols_bad", units_root=tmp / "units",
                                         g_root=tmp / "g2", cache_root=tmp / "cache", look_root=tmp / "look"),
                      "five-fold columns built from two-fold fits")
            b.refused(lambda: fitg_stage("L-musique", FOLDS, "cpu", 1, out=tmp / "g5", units_root=tmp / "units"),
                      "a sixth fold fitted")
            b.refused(lambda: cols_stage("webqsp", "fit", 1, out=tmp / "cols_bad", units_root=tmp / "units",
                                         g_root=tmp / "g5"), "an unread carve built")
            b.refused(lambda: units_stage("webqsp", "fit", 1, False, out=tmp / "units"), "an unread carve's units")
            b.refused(lambda: load_units("metaqa", "fit", tmp / "p5c_units"), "P5c's units read as P5c5's")
    finally:
        B.filed_tables = saved


def fake_recall(path, arm, calls, **over):
    rows = [{"split": sp, "dataset": ds, "read": "x", "delta": 0.0, "call": calls.get((sp, ds), "WITHIN")}
            for sp in SPLITS for ds in LG.EVAL_ORDER]
    rec = {"stage": "P5c5", "declared_in": DECLARED, "arm": arm, "verdict": "x", "rows": rows}
    rec.update(over)
    LC.write_json(path, rec)
    return path


def toy_cols(root, cache_root, ds, carve, split, rng, **over):
    """Columns and lp of one split for every step-1 part of a toy carve, with the records the builder writes."""
    parts, recs = LC.part_dirs(ds, carve, cache_root)
    for p, r in zip(parts, recs):
        d = Path(root) / split / ds / carve / p.name
        d.mkdir(parents=True, exist_ok=True)
        x = np.clip(rng.normal(-1.0, 1.2, (int(r["rows"]), NCOL)), -4, 4).astype(np.float16)
        lp = np.clip(rng.normal(-0.6, 0.4, (int(r["rows"]), 1)), -4, 4).astype(np.float16)
        sha, sha_lp = LC.save_npy(d, "cols", x), LC.save_npy(d, "lp", lp)
        rec = {"declared_in": DECLARED, "stage": "P5c5", "split": split, "dataset": ds, "carve": carve,
               "part": p.name, "step1_record_sha256": LC.sha_file(p / "record.json"), "settings": BUILD_SETTINGS,
               "arrays": {"cols": {"dtype": x.dtype.str, "shape": list(x.shape), "sha256": sha},
                          "lp": {"dtype": lp.dtype.str, "shape": list(lp.shape), "sha256": sha_lp}}}
        rec.update(over)
        LC.write_json(d / "record.json", rec)


def selftest():
    import shutil
    import tempfile
    t0 = time.time()
    b = bld()
    watched = {"P5c's folder": b.OUT, "this stage's folder": OUT}
    before = {k: tree(v) for k, v in watched.items()}
    nt = torch.get_num_threads()
    builder_selftest(np.random.default_rng(71))
    torch.set_num_threads(nt)
    tb = time.time() - t0
    LG.bind_device_ops()
    torch.use_deterministic_algorithms(True)
    # 4. the arms: zrckpo's model and zspk's model plus the five-column head, on their carves with the columns beside;
    #    the bases and P5c's arms unchanged
    assert S.ARMS["zrckpo5"] == (ZRCKPO5, P5c5KTChainCarveZRC) and S.ARMS["zspko5"] == (ZSPKO5, P5c5KTLinkCarveBase)
    assert S.ARMS["zrckp"] == (Z3P.ZRCKP, Z3.KTChainCarveZRC) and S.ARMS["zspk"] == (Z3.ZSPK, Z3.KTLinkCarveBase)
    assert S.ARMS["zrckpo"] == P5.ARMS["zrckpo"] and S.ARMS["zspko"] == P5.ARMS["zspko"]
    assert ZRCKPO5.__mro__[1] is P5.ZRCKPO and vars(ZRCKPO5).keys() <= {"__module__", "__doc__", "__qualname__",
                                                                           "__firstlineno__", "__static_attributes__"}
    assert ZSPKO5.__mro__[1:5] == (ZP.ZProp, P5c5Mix, Z3.KTMix, ZM.ZRM)
    assert P5c5KTChainCarveZRC.__mro__[1] is Z3.KTChainCarveZRC and P5c5KTLinkCarveBase.__mro__[1] is Z3.KTLinkCarveBase
    assert P5c5KTChainCarveZRC.P5C5_LP is False and P5c5KTLinkCarveBase.P5C5_LP is True
    assert all(c.P5C5_SPLIT is None for c in CARVES) and N_HEAD == {"zrckpo5": 97, "zspko5": 113}
    rng = np.random.default_rng(57)
    tmp = Path(tempfile.mkdtemp(prefix="zp5c5_"))
    try:
        RM.toy_roots(tmp, rng, chunks=(12, 12))
        RM.build("metaqa", "toy", out_root=tmp / "ch", cache_root=tmp / "cache", look_root=tmp / "look",
                 rel_dir=tmp / "rel")
        ZL.build("metaqa", "toy", out_root=tmp / "lk", cache_root=tmp / "cache", look_root=tmp / "look")
        for sp in SPLITS:
            toy_cols(tmp / "p5c5", tmp / "cache", "metaqa", "toy", sp, rng)
        cls_ch = RM.chain_carve(ZM.ToyZ, tmp / "ch", tmp / "rel")
        cls_lk = ZL.link_carve(cls_ch, tmp / "lk")
        C_m, C_g = p5c5_carve(cls_ch, False, tmp / "p5c5"), p5c5_carve(cls_lk, True, tmp / "p5c5")
        # 5. a carve reads only under on_split, only its split's columns, only columns built on its parts under
        #    BUILD_SETTINGS (five folds: P5c's two-fold build and its stage are refused); the MLP's carve never opens lp
        SR.must_stop(C_m, "metaqa", "toy", "2wiki", "cpu", tmp / "cache")
        SR.must_stop(lambda: on_split("L-2wiki", (C_m,)).__enter__())
        with on_split("L-hotpotqa", (C_m, C_g)):
            c_mh = C_m("metaqa", "toy", "2wiki", "cpu", tmp / "cache")
        assert C_m.P5C5_SPLIT is None and c_mh.p5c_split == "L-hotpotqa"
        with on_split("L-musique", (C_m, C_g)):
            c_m = C_m("metaqa", "toy", "2wiki", "cpu", tmp / "cache")
            c_g = C_g("metaqa", "toy", "2wiki", "cpu", tmp / "cache")
        assert not torch.equal(c_m.p5c, c_mh.p5c)
        assert c_m.p5c.shape[1] == NCOL and c_g.p5c.shape[1] == NCOL_G and torch.equal(c_g.p5c[:, :NCOL], c_m.p5c)
        part1 = tmp / "p5c5" / "L-musique" / "metaqa" / "toy" / "part_1of2"
        bad = part1 / "record.json"
        keep = bad.read_bytes()
        for key, val in (("split", "L-hotpotqa"), ("step1_record_sha256", "x"), ("stage", "P5c"),
                         ("declared_in", P5.DECLARED), ("settings", build_settings(2)),
                         ("settings", P5.BUILD_SETTINGS)):
            h = json.loads(keep)
            h[key] = val
            LC.write_json(bad, h)
            with on_split("L-musique", (C_m, C_g)):
                SR.must_stop(C_m, "metaqa", "toy", "2wiki", "cpu", tmp / "cache")
                SR.must_stop(C_g, "metaqa", "toy", "2wiki", "cpu", tmp / "cache")
        bad.write_bytes(keep)
        os.replace(part1 / "lp.npy", part1 / "lp.away")
        with on_split("L-musique", (C_m, C_g)):
            assert torch.equal(C_m("metaqa", "toy", "2wiki", "cpu", tmp / "cache").p5c, c_m.p5c)
            SR.must_stop(C_g, "metaqa", "toy", "2wiki", "cpu", tmp / "cache")
        os.replace(part1 / "lp.away", part1 / "lp.npy")
        # 6. a batch's columns are its rows' own, in the cache's row order: the MLP's the four, the GNN's the four and
        #    lp; the MLP's carve holds no edge
        blocks = ["rank", "SEMB"]
        qs = np.r_[np.arange(c_m.rows)[::-1][:7], np.arange(c_m.rows)[:5]].astype(np.int64)
        B = qs.size
        fm, nq, bz, _gold = c_m.batch(qs, blocks)
        fg, nq_g, bz_g, _ = c_g.batch(qs, blocks)
        assert ZL.LK_KEY not in fm and not hasattr(c_m, "links") and ZL.LK_KEY in fg
        assert fm[P5C_KEY].shape[1] == NCOL and fg[P5C_KEY].shape[1] == NCOL_G and torch.equal(nq, nq_g)
        assert torch.equal(fm[P5C_KEY], fg[P5C_KEY][:, :NCOL]) and fm[P5C_KEY].dtype == torch.float32
        for key, sl in (("cols", fm[P5C_KEY]), ("lp", fg[P5C_KEY][:, NCOL:])):
            allx = np.concatenate([np.load(tmp / "p5c5" / "L-musique" / "metaqa" / "toy" / f"part_{i}of2" /
                                           f"{key}.npy") for i in (0, 1)])
            want = np.concatenate([allx[c_m.off_np[q]:c_m.off_np[q] + c_m.n_np[q]] for q in qs]).astype(np.float32)
            assert np.array_equal(sl.numpy(), want) and want.shape[0] == int(c_m.n_np[qs].sum())
        # 7. the models: zrckpo5 is P5c's zrckpo bit for bit (its head at four columns is P5c's); each arm starts as
        #    its base bit for bit; with the head moved, zrckpo5 adds it over the four columns, and zspko5's propagation
        #    reads the score with the head over all five added, lp moving it; without the columns each is its base
        keep_ = torch.ones(B, len(blocks))
        torch.manual_seed(0)
        mp5 = P5.ZRCKPO(blocks, c_m.widths, 16, dropout=0.1, seed=0)
        torch.manual_seed(0)
        assert not LG.same_state(mp5.state_dict(), ZRCKPO5(blocks, c_m.widths, 16, dropout=0.1, seed=0).state_dict())
        h4, h5 = nn.Module(), nn.Module()
        P5.p5c_init(h4, 3)
        p5c5_init(h5, 3, NCOL)
        assert not LG.same_state(h4.state_dict(), h5.state_dict())     # LG.same_state lists the keys that differ
        for arm, cls, base, cv in (("zrckpo5", ZRCKPO5, Z3P.ZRCKP, c_m), ("zspko5", ZSPKO5, Z3.ZSPK, c_g)):
            f_, nq_, bz_, _ = cv.batch(qs, blocks)
            torch.manual_seed(0)
            m0 = base(blocks, cv.widths, 16, dropout=0.1, seed=0)
            torch.manual_seed(0)
            m1 = cls(blocks, cv.widths, 16, dropout=0.1, seed=0)
            s0, s1 = m0.state_dict(), m1.state_dict()
            assert sorted(set(s1) - set(s0)) == sorted(P5C) and set(s0) <= set(s1)
            assert not LG.same_state(s0, {k: s1[k] for k in s0})
            assert sum(s1[k].numel() for k in P5C) == N_HEAD[arm] and s1["p5c_w1"].shape[1] == len(READS[arm])
            m0.eval()
            m1.eval()
            with torch.no_grad():
                if arm == "zspko5":
                    w = torch.randn(1, ZP.HID, generator=torch.Generator().manual_seed(8))
                    m0.sp_w2.copy_(w)
                    m1.sp_w2.copy_(w)
                y0 = m0(f_, keep_, nq_, B, bz_)
                assert torch.equal(y0, m1(f_, keep_, nq_, B, bz_))
                m1.p5c_w2.copy_(torch.randn(1, P5C_HID, generator=torch.Generator().manual_seed(9)))
                y1 = m1(f_, keep_, nq_, B, bz_)
                head = m1.p5c_head(f_[P5C_KEY])
                assert float(head.abs().max()) > 1e-3
                # the head is added to the base's score: exactly (the base's float32 score reaches ~500 on the toy, so
                # y1 - y0 rounds; y0 + head is the sum the model forms); zsp's propagation within float32's rounding
                close = lambda a, b: torch.allclose(a, b, rtol=0, atol=1e-5 + 8 * torch.finfo(a.dtype).eps * float(
                    torch.maximum(a.abs().max(), b.abs().max())))
                if arm == "zrckpo5":
                    assert torch.equal(y1, y0 + head)
                else:
                    pre = super(ZP.ZProp, m1).forward(f_, keep_, nq_, B, bz_)
                    kt_only = Z3.KTMix.forward(m1, f_, keep_, nq_, B, bz_)
                    assert torch.equal(pre, kt_only + head)
                    assert close(y1, pre + m1.prop_head(ZP.prop_inputs(pre, f_[ZL.LK_KEY], nq_, B)))
                    assert float((y1 - y0 - head).abs().max()) > 1e-4
                    moved = dict(f_)
                    moved[P5C_KEY] = f_[P5C_KEY].clone()
                    moved[P5C_KEY][:, NCOL] -= 1.0
                    assert float((m1(moved, keep_, nq_, B, bz_) - y1).abs().max()) > 1e-4
                bare = {k: x for k, x in f_.items() if k != P5C_KEY}
                assert torch.equal(m1(bare, keep_, nq_, B, bz_), m0(bare, keep_, nq_, B, bz_))
        # 8. toy fits through lean_gpu's loop: a repeat is bit-identical and the head learns; on the carve without the
        #    columns each arm's fit is its base's bit for bit on every shared key, its head at the start
        cfg = {"lr": 2e-2, "wd": 1e-4, "dropout": 0.1, "epochs": 3, "swa_from": 1, "cos": False, "adamw": False,
               "drop": False}
        for arm, cv, bare_c in (("zrckpo5", c_m, cls_ch("metaqa", "toy", "2wiki", "cpu", tmp / "cache")),
                                ("zspko5", c_g, cls_lk("metaqa", "toy", "2wiki", "cpu", tmp / "cache"))):
            with S.patched(arm):
                fit = LG.fit_variant([cv], blocks, cfg, 0, 16, "none", "cpu", tag=f"toy {arm}")
                rep = LG.fit_variant([cv], blocks, cfg, 0, 16, "none", "cpu", tag=f"toy {arm} repeat")
                bare = LG.fit_variant([bare_c], blocks, cfg, 0, 16, "none", "cpu", tag=f"toy {arm}, no columns")
            with S.patched(BASE_OF[arm]):
                base = LG.fit_variant([bare_c], blocks, cfg, 0, 16, "none", "cpu", tag=f"toy {BASE_OF[arm]}")
            assert all(not LG.same_state(a, b_) for a, b_ in zip(fit["states"] + [fit["swa"]],
                                                                 rep["states"] + [rep["swa"]]))
            assert float(fit["swa"]["p5c_w2"].abs().max()) > 0
            for a, b_ in zip(bare["states"] + [bare["swa"]], base["states"] + [base["swa"]]):
                assert not LG.same_state({k: a[k] for k in b_}, b_), arm
                assert float(a["p5c_w2"].abs().max()) == 0.0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    # 9. the refusals; the fit names give the split
    assert split_of_fit("scr-p5c5-zspko5-hp") == ("zspko5", "L-hotpotqa") and split_of_fit("scr-p5c-zspko-hp") == (
        None, None)
    good = ["train", "--arm", "zrckpo5", "--name", FIT["zrckpo5"]["L-musique"], "--cache-root", "outputs/p1c2/cache"]
    for arm_ in ("zrckp", "zrckpo"):
        SR.must_stop(train, ["train", "--arm", arm_, "--name", FIT["zrckpo5"]["L-musique"], "--cache-root",
                             "outputs/p1c2/cache"], "L-musique")
    SR.must_stop(train, good, "L-2wiki")
    SR.must_stop(train, good, "L-hotpotqa")
    SR.must_stop(train, ["train", "--arm", "zspko5", "--name", FIT["zrckpo5"]["L-musique"], "--cache-root",
                         "outputs/p1c2/cache"], "L-musique")
    SR.must_stop(train, ["train", "--arm", "zspko5", "--name", "scr-p5c-zspko", "--cache-root",
                         "outputs/p1c2/cache"], "L-musique")
    SR.must_stop(train, ["train", "--arm", "zspko5", "--name", FIT["zspko5"]["L-musique"], "--cache-root",
                         "outputs/p1gb/cache"], "L-musique")
    SR.must_stop(train, good + ["--device", "cpu"], "L-musique")
    with tempfile.TemporaryDirectory() as td:
        T_ = Path(td)
        for nm, rec in (("y", {"arm": "zrckp", "p1c3p": {}}), ("scr-p5c5-zspko5", {"arm": "zspko", "p5c": {}}),
                        ("scr-p5c5-zspko5-hp", {"arm": "zspko5", "p5c5": {"split": "L-musique"}}),
                        ("scr-p5c5-zrckpo5", {"arm": "zrckpo5", "p5c": {"split": "L-musique"}})):
            (T_ / "fits" / nm).mkdir(parents=True)
            LC.write_json(T_ / "fits" / nm / "screen.json", rec)
            SR.must_stop(read, ["--name", nm, "--out-root", str(T_ / "fits"), "--cache-root", "outputs/p1c2/cache"])
        # 10. relz's pair and re-call for each arm against its base arm's card fits, stamped P5c5's; relz restored
        saved_z = Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit
        for arm in ARM_NAMES:
            fits = BASE_FITS[arm]
            null1 = SR.fake_null(T_ / arm / "null", "n1", {("L-hotpotqa", "hotpotqa"): (-0.010, 0.012)})
            bs = {sp: str(SR.fake_compare(T_ / arm / fits[sp][0], fits[sp][2], sp, {ds: 0.1 for ds in LG.EVAL_ORDER},
                                          arm=BASE_OF[arm])) for sp in fits}
            zb = {sp: "/".join(("outputs",) + fits[sp]) for sp in fits}
            za = SR.fake_compare(T_ / arm / "z", FIT[arm]["L-musique"], "L-musique", {"webqsp": 0.0215}, arm=arm,
                                 base_r5=0.6, base=zb["L-musique"])
            zh = SR.fake_compare(T_ / arm / "z", FIT[arm]["L-hotpotqa"], "L-hotpotqa", {"hotpotqa": -0.0147}, arm=arm,
                                 base_r5=0.6, base=zb["L-hotpotqa"])
            pr = pair(arm, [za, zh], T_ / arm / "pz")
            assert pr["verdict"] == "MIXED" and pr["arm"] == arm and pr["base_arm"] == BASE_OF[arm], pr["verdict"]
            pj = json.loads((T_ / arm / "pz.json").read_text(encoding="utf-8"))
            assert pj["stage"] == "P5c5" and pj["decided_against"] == AGAINST[arm] and pj["declared_in"] == DECLARED
            r = recall(arm, null1, T_ / arm / "pz.json", T_ / arm / "rz", bs)
            assert r["verdict"] == "PROMISING" and r["changed"] == ["L-hotpotqa hotpotqa LOSS -> WITHIN"], r["changed"]
            t = (T_ / arm / "rz.md").read_text(encoding="utf-8")
            assert t.startswith(f"P5c5 ({DECLARED})")
            assert f"| {BASE_OF[arm]} R@5 | {arm} R@5 (P5c5's five-fold offset input) |" in t
        assert (Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit) == saved_z
        # 11. the verdict, per arm, with section 5's next form from both arms' outcomes; webqsp and metaqa beside, and
        #     musique's in-domain read
        gm, mi, hw = ("L-musique", "musique"), CLAIM1_READ, ("L-hotpotqa", "webqsp")
        V = lambda a_calls, g_calls, **over: verdict({
            "zrckpo5": fake_recall(T_ / "a.json", "zrckpo5", a_calls, **over.get("a", {})),
            "zspko5": fake_recall(T_ / "b.json", "zspko5", g_calls, **over.get("g", {}))}, T_ / "v")["arms"]
        rec = V({mi: "GAIN"}, {gm: "GAIN", hw: "LOSS"})
        assert rec["zrckpo5"]["verdict"] == "ADOPT_FOR_FULL_RUN" and rec["zrckpo5"]["next_form"] == NEXT[
            "ADOPT_FOR_FULL_RUN"]
        assert rec["zspko5"]["verdict"] == "NOT_ADOPTED" and rec["zspko5"]["next_form"] == BACKOFF["zspko5"]
        assert sorted(rec["zspko5"]["beside"]) == sorted(f"{sp} {d}" for sp in SPLITS for d in BESIDE)
        full = verdict({"zrckpo5": T_ / "a.json", "zspko5": T_ / "b.json"}, T_ / "v")
        assert full["musique_in_domain_beside"] == {"zrckpo5": {"call": "GAIN", "delta": 0.0},
                                                    "zspko5": {"call": "WITHIN", "delta": 0.0}}
        assert "closes only this form" in (T_ / "v.md").read_text(encoding="utf-8")
        rec = V({}, {gm: "GAIN"})
        assert rec["zspko5"]["verdict"] == "ADOPT_FOR_FULL_RUN" and rec["zrckpo5"]["next_form"] == REACH["zrckpo5"]
        rec = V({gm: "GAIN"}, {hw: "LOSS"})
        assert rec["zspko5"]["verdict"] == "NOT_ADOPTED" and rec["zspko5"]["next_form"] == REACH["zspko5"]
        rec = V({gm: "GAIN", hw: "LOSS"}, {})
        assert rec["zrckpo5"]["next_form"] == BACKOFF["zrckpo5"] and rec["zspko5"]["next_form"] == JOIN["zspko5"]
        rec = V({hw: "LOSS"}, {})
        assert [rec[a]["verdict"] for a in ARM_NAMES] == ["NOT_ADOPTED"] * 2
        assert rec["zrckpo5"]["next_form"] == rec["zspko5"]["next_form"] == CLOSE
        rec = V({}, {gm: "GAIN"}, g={"stage": "P5c"})
        assert rec["zspko5"]["verdict"] == "INCOMPLETE" and rec["zspko5"]["next_form"] == NEXT["INCOMPLETE"]
        assert rec["zrckpo5"]["verdict"] == "NOT_ADOPTED" and rec["zrckpo5"]["next_form"] == WAIT
        rec = verdict({"zrckpo5": T_ / "none.json", "zspko5": fake_recall(T_ / "b.json", "zrckpo5", {gm: "GAIN"})},
                      T_ / "v")["arms"]
        assert rec["zrckpo5"]["verdict"] == "INCOMPLETE" and rec["zspko5"]["verdict"] == "INCOMPLETE"
        rec = V({}, {}, a={"rows": []})
        assert rec["zrckpo5"]["verdict"] == "INCOMPLETE" and rec["zspko5"]["next_form"] == WAIT
    after = {k: tree(v) for k, v in watched.items()}
    assert after == before, "a folder outside the temporary roots changed"
    log(f"zp5c5 selftest: the builds' roots are outputs/p5c5's, never P5c's; five folds deterministic and balanced, at "
        f"two P5c's parity; on P5c's toy at two folds this builder gives P5c's builder's g (g_k = P5c's g_(1-k)) and "
        f"columns bit for bit; at five, g_k fitted on exactly the counted questions outside fold k, every question's "
        f"columns and lp are its own fold's g's bit for bit, a g that fitted on it never scores it (the guard stops a "
        f"record that says otherwise), another fold's g gives other columns, lp is P5b's log p; two-fold fits, a sixth "
        f"fold and unread carves refused ({tb:.1f}s). The arms: zrckpo5 is P5c's zrckpo on the four five-fold columns, "
        f"its carve never opening lp; zspko5 is zspk plus the head over five, lp moving its score before the "
        f"propagation; a carve reads only under on_split, only its split's five-fold build; each arm starts as its "
        f"base bit for bit, and without the columns each is its base; toy fits repeat bit for bit, the head learns; "
        f"train and read refusals; relz's pair and re-call stamped P5c5's; the verdict per arm with section 5's next "
        f"form for every outcome; P5c's folder and this stage's untouched ({time.time() - t0:.1f}s): ok")
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
    if k.cmd in ("units", "fitg", "cols"):
        bp = argparse.ArgumentParser(prog=f"zp5c5.py {k.cmd}")
        bp.add_argument("a")
        bp.add_argument("b")
        bp.add_argument("--host", action="store_true")
        bp.add_argument("--threads", type=int, default=2)
        bp.add_argument("--limit", type=int)
        bp.add_argument("--device", default="cpu")
        g = bp.parse_args(rest)
        if k.cmd == "units":
            return units_stage(g.a, g.b, g.threads, g.host, g.limit)
        if k.cmd == "fitg":
            return fitg_stage(g.a, int(g.b), g.device, g.threads)
        return 0 if cols_stage(g.a, g.b, g.threads, g.host, placement={"where": "host" if g.host else "laptop"}) else 1
    if k.cmd == "train":
        return train([k.cmd] + rest, k.split)
    if k.cmd == "read":
        return read(rest)
    if k.cmd == "compare":
        return compare(rest)
    if k.cmd == "verdict":
        return 0 if all(v["verdict"] != "INCOMPLETE" for v in verdict()["arms"].values()) else 1
    if k.cmd in ("pair", "recall"):
        gp = argparse.ArgumentParser()
        gp.add_argument("--arm", required=True, choices=ARM_NAMES)
        gp.add_argument("--screens", default="")
        gp.add_argument("--null", default="")
        gp.add_argument("--pair")
        gp.add_argument("--out", required=True)
        g = gp.parse_args(rest)
        null = [x for x in g.null.split(",") if x]
        try:
            if k.cmd == "pair":
                pair(g.arm, [x for x in g.screens.split(",") if x], g.out)
            else:
                if len(null) != 4 or not g.pair:
                    gp.error("recall: --pair and the four --null files")
                recall(g.arm, null, g.pair, g.out)
        except SystemExit as e:
            log(f"zp5c5 {k.cmd}: {e}")
            return 2
        return 0
    raise SystemExit("zp5c5: units, fitg, cols, train, read, compare, pair, recall, verdict, or --selftest")


if __name__ == "__main__":
    sys.exit(main())
