"""P5c (docs/P5C_OFFSET_INPUT.md): P5b's learned offsets g(question, seed, graph features) as a fixed input to both of
the lean pair's models. This script builds the columns; outputs/mp_unified/zp5c.py trains and reads the models on them.

    python scripts/p5c_offset_cols.py units DATASET CARVE [--host] [--threads 2] [--limit Q]
    python scripts/p5c_offset_cols.py fitg SPLIT FOLD [--host] [--device cuda] [--threads 2]
    python scripts/p5c_offset_cols.py cols DATASET CARVE [--host] [--threads 2]
    python scripts/p5c_offset_cols.py --selftest

units: every question of one carve with its final pool (P1c2's KIND_T file), in P5b's base layout (B.unit, unchanged):
its frame vector; its seeds (the dense top 5, then the SPLADE top 5) with their first-stage ranks and graph features;
its candidates (the pool minus the seeds) with the golds among them. Unlike P5b's base, every question is kept, not a
sample of 2,000, and questions without a counted gold are kept too (their columns are needed; g's fit skips them). On
a training dataset's fit carve, every question P5b's filed base holds must come out as P5b's.
fitg: g on P5a's space V, refit for one screen split without its held-out dataset, on one fold of the four training
datasets' fit questions (fold = sha256 of 'dataset:question id', parity), with P5b2's held-out stop inside the fold.
cols: four columns per cache row of every part of P1c2's cache, for each split that reads the carve, each question's
from the g of the other fold (the one that never saw it). With l = -kappa e^beta |u_s + Delta_h - u_v|^2 (P5b's
score of row v from seed s and head h) and pi g's weights over the question's (seed, head) pairs:
    pt    logsumexp over (s, h) of log pi + l: P5b's mixture, pointwise
    mx    the largest l over (s, h): the nearest target
    pt0   pt with g at its start (no offset, equal weights, beta 0): logsumexp over s of -log S - kappa |u_s - u_v|^2
    mx0   mx with g at its start: -kappa times the squared distance to the nearest seed
each clipped to [-CLIP, CLIP] and divided by SCALE, stored float16, one file per step-1 cache part. A seed row gets
the same formulas. A question without a seed gets -CLIP / SCALE in every column.
No column is normalised over the question's rows (the user's rule of 14 October: no softmax, rank or z-score of a
score over the question), and none aggregates over graph neighbours at read time or reads either model's score: g runs
before either model, on the question, its seeds' first-stage ranks and index-time graph features (P5b's inputs). P5b's
own log p (its softmax over the candidates) is computed only to check, on each part's first questions, that g, its
inputs and the distances are P5b's.
--limit: a development run of units: it writes nothing.
"""
from __future__ import annotations

import os
import sys

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "scripts", ROOT / "outputs" / "mp_unified"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import p5b_offsets as B  # noqa: E402  (P5b's filed script, imported unchanged)
import p5b2_stop as B2  # noqa: E402  (P5b2's filed script: its filed bases, split and stopped fit, unchanged)

A = B.A
DECLARED = "docs/P5C_OFFSET_INPUT.md"
OUT = ROOT / "outputs" / "p5c"
UNITS, GDIR, COLS = OUT / "units", OUT / "g", OUT / "cols"
CACHE = ROOT / "outputs" / "p1c2" / "cache"
LOOK = ROOT / "outputs" / "p1c2" / "look"
SIX, TRAIN5 = A.SIX, A.TRAIN5
SPLITS = {"L-musique": ("metaqa", "squad", "hotpotqa", "2wiki"), "L-hotpotqa": ("metaqa", "squad", "musique", "2wiki")}
HELD_OUT = {"L-musique": "musique", "L-hotpotqa": "hotpotqa"}
CARVES_OF = {sp: tuple((d, A.fit_carve(d)) for d in SPLITS[sp]) + tuple((d, "s1eval") for d in SIX) for sp in SPLITS}
CARVES = tuple(sorted({c for sp in SPLITS for c in CARVES_OF[sp]}))
FOLDS = 2
COLNAMES = ("pt", "mx", "pt0", "mx0")
NCOL = len(COLNAMES)
CLIP, SCALE = 40.0, 10.0
EPOCHS = B2.EPOCHS
BUILD_SETTINGS = {"columns": list(COLNAMES), "clip": [-CLIP, CLIP], "scale": SCALE, "space": "V", "model": "G",
                  "folds": FOLDS, "fold": "parity of sha256('dataset:question id') read as an integer",
                  "cross_fit": "each question's columns from the g of the other fold",
                  "fit": "P5b's g, refit per split without its held-out dataset on one fold of the four training "
                         "datasets' fit questions with a counted gold, P5b2's held-out stop inside the fold",
                  "pools": "P1c2's KIND_T files; rows in P1c2's cache order", "dtype": "float16"}
log, utc, sha_file, write_json, save_npz = A.log, A.utc, A.sha_file, A.write_json, A.save_npz


def script_sha256():
    return hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def fold_of(ds, qid):
    return int(hashlib.sha256(f"{ds}:{qid}".encode("utf-8")).hexdigest(), 16) % FOLDS


def units_paths(name, carve, root=UNITS):
    return root / name / f"{carve}.npz", root / name / f"{carve}.json"


def g_paths(split, fold, root=GDIR):
    return root / f"{split}_f{fold}.pt", root / f"{split}_f{fold}.json"


# ── units ────────────────────────────────────────────────────────────────────


def collect_all(op, E, name, carve, threads, limit, frame, tables, adj, get_finals):
    """P5b's collect over every question of the carve (the first `limit` in a development run): each question's unit
    on its final pool, with or without a counted gold."""
    holder = {}

    def sink(name_, carve_, pop, prep, U, BQ, bs, inc, linked, union, constant, families, construction):
        Q = len(pop.ids)
        js = np.arange(Q if not limit else min(limit, Q))
        finals = get_finals(carve_)
        cache = {}
        units, golds, ids, pos = [], [], [], []
        t = time.time()
        for c, j in enumerate(js.tolist()):
            pool = finals.pool(pop.ids[j], bs[j], inc[j]) if finals else np.unique(np.asarray(inc[j], np.int64))
            u = B.unit(j, pop, prep, pool, tables, adj, frame, cache)
            units.append(u)
            golds.append(np.intersect1d(np.unique(np.asarray(pop.golds[j], np.int64)), u["cand"]))
            ids.append(pop.ids[j])
            pos.append(j)
            if (c + 1) % 1000 == 0:
                log(f"{name}/{carve_}: {c + 1}/{js.size} questions ({time.time() - t:.0f}s)")
        holder["arr"] = B.pack_units(units, golds, ids, pos)
        holder["rec"] = {"carve": carve_, "carve_questions": Q, "kept": len(units),
                         "with_a_counted_gold": int(sum(g.size > 0 for g in golds)), "carve_sha256": pop.digest,
                         "seconds": round(time.time() - t, 1)}

    E.run_carves(op, name, {carve}, (), threads, sink)
    return holder["arr"], holder["rec"]


def same_as_p5b(name, arr):
    """On a training dataset's fit carve: every question of P5b's filed base comes out as P5b's (its seeds, ranks,
    degrees, candidates and golds exactly; its frame vector and m_s within 1e-5)."""
    base, _rec, bsha = B2.filed_base(name)
    at = {str(q): i for i, q in enumerate(arr["ids"].tolist())}
    worst = 0.0
    for i, q in enumerate(base["ids"].tolist()):
        k = at.get(str(q))
        if k is None:
            raise SystemExit(f"{name}: P5b's question {q} is not in the units")
        s0, s1 = base["seed_ptr"][i], base["seed_ptr"][i + 1]
        t0, t1 = arr["seed_ptr"][k], arr["seed_ptr"][k + 1]
        c0, c1 = base["cand_ptr"][i], base["cand_ptr"][i + 1]
        d0, d1 = arr["cand_ptr"][k], arr["cand_ptr"][k + 1]
        for key in ("seeds", "rank_d", "rank_s", "deg"):
            if not np.array_equal(base[key][s0:s1], arr[key][t0:t1]):
                raise SystemExit(f"{name}: question {q}: {key} is not P5b's")
        for key in ("cand", "gold"):
            if not np.array_equal(base[key][c0:c1], arr[key][d0:d1]):
                raise SystemExit(f"{name}: question {q}: {key} is not P5b's")
        dev = max(float(np.abs(base["x"][i] - arr["x"][k]).max()),
                  *(float(np.abs(base[m][s0:s1] - arr[m][t0:t1]).max(initial=0.0)) for m in ("m_V", "m_V0")))
        if dev > 1e-5:
            raise SystemExit(f"{name}: question {q}: the frame vector or m_s is off P5b's by {dev:.2e}")
        worst = max(worst, dev)
    return {"p5b_base_record": bsha, "questions_checked": int(base["ids"].size), "max_abs_deviation": worst}


def units_stage(name, carve, threads, host, limit=None, out=UNITS):
    if (name, carve) not in CARVES:
        raise SystemExit(f"{name}/{carve}: no screen split reads this carve")
    t0 = time.time()
    E, _F, S4 = A.modules()
    op = A.opened(host)
    ds, emb = A.dense_docs(op, name)
    frame, tables, tshas, made = B.tables_for(name, emb, limit)
    _stores, adj = A.family_stores(S4, ds, name, A.csr_dir(op))
    if any(ip.size - 1 != emb.n_rows for ip, _c in adj):
        raise SystemExit(f"{name}: a family's store is not the corpus's size")
    get, pinfo = B.finals_for(name, limit)
    arr, crec = collect_all(op, E, name, carve, threads, limit, frame, tables, adj, get)
    same = same_as_p5b(name, arr) if (not limit and name in TRAIN5 and carve == A.fit_carve(name)) else None
    if limit:
        log(f"{name}/{carve} development run: {crec['kept']} questions, {int(arr['cand'].size)} candidates; tables: "
            f"{made}; pools: {B.pools_said(pinfo[carve])}; {time.time() - t0:.0f}s, peak "
            f"{A.peak_rss() / 2 ** 30:.1f} GB; nothing written")
        return 0
    p, j = units_paths(name, carve, out)
    save_npz(p, **arr)
    rec = {"declared_in": DECLARED, "stage": "units", "dataset": name, **crec, "pools": pinfo[carve],
           "tables": tshas, "npz_sha256": sha_file(p), "candidates": int(arr["cand"].size),
           "seeds": int(arr["seeds"].size), "same_as_p5b": same, "index_time": B.INDEX_TIME,
           "freeze_RECORD_SHA256": op.freeze["RECORD_SHA256"], "script_sha256": script_sha256(),
           "p5b_script_sha256": B.script_sha256(), "p5a_script_sha256": A.script_sha256(),
           "peak_rss_bytes": A.peak_rss(), "seconds": round(time.time() - t0, 1), "utc": utc()}
    write_json(j, rec)
    log(f"{name}/{carve}: {crec['kept']} questions ({crec['with_a_counted_gold']} with a counted gold), "
        f"{rec['candidates']} candidates" + (f"; P5b's {same['questions_checked']} questions reproduce" if same else "")
        + f" ({rec['seconds']:.0f}s)")
    return 0


def load_units(name, carve, root=UNITS):
    p, j = units_paths(name, carve, root)
    if not j.exists():
        raise SystemExit(f"{j} missing (p5c_offset_cols.py units {name} {carve})")
    rec = json.loads(j.read_text(encoding="utf-8"))
    if rec.get("declared_in") != DECLARED or rec.get("stage") != "units" or sha_file(p) != rec["npz_sha256"]:
        raise SystemExit(f"{p}: not P5c's units file, or not its record's")
    with np.load(p) as z:
        arr = {k: z[k] for k in z.files}
    if set(arr) != B2.KEYS:
        raise SystemExit(f"{p}: keys {sorted(arr)}")
    return arr, rec, sha_file(j)


# ── g, refit per split and fold ──────────────────────────────────────────────


def fold_index(name, arr, fold, counted=True):
    """The questions of one fold (in file order), only those with a counted gold when counted."""
    cp = np.asarray(arr["cand_ptr"], np.int64)
    gold = np.asarray(arr["gold"], bool)
    has = np.array([gold[cp[i]:cp[i + 1]].any() for i in range(cp.size - 1)], bool)
    f = np.array([fold_of(name, q) for q in arr["ids"].tolist()], np.int64)
    return np.flatnonzero((f == fold) & (has | (not counted)))


def fitg_stage(split, fold, device, threads, out=GDIR, units_root=UNITS, epochs=EPOCHS):
    import torch
    if split not in SPLITS or fold not in range(FOLDS):
        raise SystemExit(f"fitg takes a split of {sorted(SPLITS)} and a fold of 0..{FOLDS - 1}")
    torch.set_num_threads(threads)
    A.full_fp32(torch)
    dev = torch.device(device)
    t0 = time.time()
    names = list(SPLITS[split])
    if HELD_OUT[split] in names:
        raise SystemExit(f"{split}: its held-out dataset would train g")
    packs, hos, news, counts, urecs, kap, tshas = [], [], [], {}, {}, {}, {}
    for name in names:
        arr, urec, urecs[name] = load_units(name, A.fit_carve(name), units_root)
        _frame, tables, tshas[name] = B.filed_tables(name)
        if tshas[name] != urec["tables"]:
            raise SystemExit(f"{name}: the units ran on other tables")
        idx = fold_index(name, arr, fold)
        a_tr, a_ho, new, counts[name] = B2.split_base(B2.subset(arr, idx))
        counts[name]["fold_questions_with_a_counted_gold"] = int(idx.size)
        kap[name] = B.kappa_of(tables["V"])
        packs.append(B.Pack(a_tr, tables["V"], kap[name], "V"))
        hos.append(B.Pack(a_ho, tables["V"], kap[name], "V"))
        news.append(new)
    log(f"{split} fold {fold}: G on V, {sum(p.n for p in packs)} train and {sum(h.n for h in hos)} held-out questions "
        f"of {', '.join(names)}")
    g, hist, kept = B2.fit_stop(torch, dev, packs, hos, news, names, epochs)
    pt, js = g_paths(split, fold, out)
    config = {"d_in": B.D_IN, "width": B.WIDTH, "heads": B.HEADS, "dim": B.DIM, "rank_c": B.RANK_C,
              "weights": B.N_WEIGHTS}
    B.save_g(pt, {"G": g}, config)
    rec = {"declared_in": DECLARED, "stage": "fitg", "split": split, "fold": fold, "datasets": names,
           "held_out_dataset": HELD_OUT[split], "units_records": urecs, "tables": tshas, "split_counts": counts,
           "kappa": kap, "kept_epoch": kept, "beta": g.beta.detach().item(), "hist": hist,
           "settings": {"hold": B2.HOLD, "split_seed": B2.SPLIT_SEED, "max_epochs": epochs, "stop": B2.STOP_RULE,
                        "batch": B.BATCH, "lr": B.LR, "optimizer": "Adam", "seed": 0,
                        "precision": "float32, TF32 off", "kappa_pairs": B.KAPPA_PAIRS, **config,
                        "build": BUILD_SETTINGS},
           "g_sha256": sha_file(pt), "script_sha256": script_sha256(), "p5b_script_sha256": B.script_sha256(),
           "p5b2_script_sha256": B2.script_sha256(), "device": str(dev), "gpu_peak_bytes": A.gpu_peak(torch, dev),
           "peak_rss_bytes": A.peak_rss(), "seconds": round(time.time() - t0, 1), "utc": utc()}
    write_json(js, rec)
    log(f"{split} fold {fold}: kept epoch {kept}, beta {rec['beta']:.3f} ({rec['seconds']:.0f}s)")
    return 0


def load_fit(torch, split, fold, root=GDIR):
    pt, js = g_paths(split, fold, root)
    if not js.exists():
        raise SystemExit(f"{js} missing (p5c_offset_cols.py fitg {split} {fold})")
    rec = json.loads(js.read_text(encoding="utf-8"))
    if rec.get("declared_in") != DECLARED or rec.get("stage") != "fitg" or sha_file(pt) != rec["g_sha256"]:
        raise SystemExit(f"{pt}: not P5c's g, or not its record's")
    if rec["settings"]["build"] != BUILD_SETTINGS:
        raise SystemExit(f"{js}: fitted under other settings")
    blob = torch.load(pt, map_location="cpu", weights_only=True)
    g = B.new_g(torch, torch.device("cpu"))
    g.load_state_dict(blob["models"]["G"])
    g.eval()
    return g, rec, sha_file(js)


# ── the columns ──────────────────────────────────────────────────────────────


def columns(torch, g, kappa, F, US, UP, cm):
    """One question's rows UP: the four raw columns (float64, before clip and scale), and P5b's log p of each row (its
    softmax over the candidates, cm True; the identity check only, never stored). F and US are the seeds' g inputs and
    vectors."""
    with torch.no_grad():
        delta, a = g(torch.from_numpy(np.asarray(F, np.float32)))
        T = (torch.from_numpy(np.asarray(US, np.float32))[:, None, :] + delta).double()
        Ud = torch.from_numpy(np.asarray(UP, np.float64))
        Sd = torch.from_numpy(np.asarray(US, np.float64))
        d2 = ((T * T).sum(-1)[..., None] + (Ud * Ud).sum(-1)[None, None, :]
              - 2.0 * torch.einsum("shd,cd->shc", T, Ud)).clamp_min(0.0)
        lg = -(float(kappa) * float(torch.exp(g.beta.double()))) * d2
        S, H = a.shape
        lpi = torch.log_softmax(a.double().reshape(S * H), 0).reshape(S, H)
        pt = torch.logsumexp(lpi[..., None] + lg, dim=(0, 1))
        mx = lg.amax(dim=(0, 1))
        d0 = ((Sd * Sd).sum(-1)[:, None] + (Ud * Ud).sum(-1)[None, :] - 2.0 * Sd @ Ud.T).clamp_min(0.0)
        l0 = -float(kappa) * d0
        pt0 = torch.logsumexp(l0 - np.log(S), dim=0)
        mx0 = l0.amax(dim=0)
        c = torch.from_numpy(np.asarray(cm, bool))
        lz = (torch.logsumexp(lg[..., c], dim=-1, keepdim=True) if bool(c.any())
              else torch.zeros(S, H, 1, dtype=torch.float64))
        lp = torch.logsumexp(lpi[..., None] + lg - lz, dim=(0, 1))
        return torch.stack([pt, mx, pt0, mx0], 1).numpy(), lp.numpy()


def stored(x):
    """Clip to [-CLIP, CLIP], divide by SCALE: the stored float16 columns, and the float64 values they stand for."""
    y = np.clip(np.asarray(x, np.float64), -CLIP, CLIP) / SCALE
    return y.astype(np.float16), y


def question_inputs(arr, i, U):
    """One units question's g inputs (P5b's features, unchanged) and its pool, sorted."""
    s0, s1 = int(arr["seed_ptr"][i]), int(arr["seed_ptr"][i + 1])
    c0, c1 = int(arr["cand_ptr"][i]), int(arr["cand_ptr"][i + 1])
    seeds = np.asarray(arr["seeds"][s0:s1], np.int64)
    US = B.rows_of(U, seeds, np.float32)
    F = B.features(np.repeat(arr["x"][i][None], seeds.size, 0), US, arr["m_V"][s0:s1], arr["deg"][s0:s1],
                   arr["rank_d"][s0:s1], arr["rank_s"][s0:s1])
    pool = np.union1d(seeds, np.asarray(arr["cand"][c0:c1], np.int64))
    return F, US, seeds, pool


def p5b_lp(torch, g, kappa, F, US, UC):
    """P5b's own log p (B.log_p, float32) over candidates UC: the identity check of lp."""
    t = lambda a: torch.from_numpy(np.asarray(a))[None]
    with torch.no_grad():
        lp = B.log_p(torch, g, t(np.asarray(F, np.float32)), t(np.asarray(US, np.float32)),
                     t(np.ones(len(US), bool)), t(np.asarray(UC, np.float32)), t(np.ones(len(UC), bool)),
                     torch.tensor([float(kappa)], dtype=torch.float32))
    return lp[0].double().numpy()


def cols_stage(name, carve, threads, host=False, out=COLS, units_root=UNITS, g_root=GDIR, cache_root=CACHE,
               look_root=LOOK, placement=None):
    import torch
    import lean_cache as LC
    torch.set_num_threads(threads)
    t0 = time.time()
    splits = [sp for sp in SPLITS if (name, carve) in CARVES_OF[sp]]
    if not splits:
        raise SystemExit(f"{name}/{carve}: no screen split reads this carve")
    arr, urec, usha = load_units(name, carve, units_root)
    _frame, tables, tshas = B.filed_tables(name)
    if tshas != urec["tables"]:
        raise SystemExit(f"{name}: the units ran on other tables")
    U = tables["V"]
    kappa = B.kappa_of(U)
    fits = {}
    for sp in splits:
        fits[sp] = [load_fit(torch, sp, f, g_root) for f in range(FOLDS)]
        for _g, grec, _s in fits[sp]:
            if name in grec["datasets"] and grec["kappa"][name] != kappa:
                raise SystemExit(f"{name}: kappa is not the one {sp}'s g was fitted with")
            if name in grec["datasets"] and grec["units_records"][name] != usha and carve == A.fit_carve(name):
                raise SystemExit(f"{name}/{carve}: these units are not the ones {sp}'s g was fitted on")
    at = {str(q): i for i, q in enumerate(arr["ids"].tolist())}
    d, files, ids, head = LC.look_chunks(name, carve, look_root)
    parts, recs = LC.part_dirs(name, carve, cache_root)
    done = {}
    for p, r in zip(parts, recs):
        if r["look"]["carve_ids_sha256"] != head["carve_ids_sha256"] or r["look"]["records"] != head["records"]:
            raise SystemExit(f"{p}: built from another look than {d}")
        tp = time.time()
        step1 = LC.sha_file(p / "record.json")
        if LC.sha_file(p / "n.npy") != r["arrays"]["n"]["sha256"]:
            raise SystemExit(f"{p / 'n.npy'}: sha256 is not its record's")
        n = np.load(p / "n.npy").astype(np.int64)
        q0, q1 = r["query_range"]
        lo, hi = r["chunks"]
        if n.size != q1 - q0 or n.size != r["queries"] or int(n.sum()) != r["rows"] or r.get("ids", ids[q0:q1]) != ids[q0:q1]:
            raise SystemExit(f"{p}: its questions, ids and rows do not agree")
        xs = {sp: [] for sp in splits}
        folds = np.zeros(FOLDS, np.int64)
        checked, worst_lp, gq, no_seed = 0, 0.0, q0, 0
        for ch in range(lo, hi):
            with np.load(files[ch]) as zf:
                qps = zf["q_pool_size"].astype(np.int64)
                pools = zf["pool"].astype(np.int64)
            ro = np.r_[0, np.cumsum(qps)]
            if int(ro[-1]) != pools.size:
                raise SystemExit(f"{files[ch]}: its pool and question sizes do not agree")
            for j in range(qps.size):
                if qps[j] != n[gq - q0]:
                    raise SystemExit(f"{p}: question {gq} has {qps[j]} look rows and {n[gq - q0]} cache rows")
                look_pool = pools[ro[j]:ro[j + 1]]
                gq += 1
                if not look_pool.size:
                    continue
                qid = str(ids[gq - 1])
                i = at.get(qid)
                if i is None:
                    raise SystemExit(f"{name}/{carve}: question {qid} is in the look, not in the units")
                F, US, seeds, pool = question_inputs(arr, i, U)
                if not np.array_equal(np.sort(look_pool), pool):
                    raise SystemExit(f"{name}/{carve}: question {qid}: the cache's rows are not its final pool")
                f = fold_of(name, qid)
                folds[f] += look_pool.size
                if not seeds.size:
                    for sp in splits:
                        xs[sp].append(np.full((look_pool.size, NCOL), -CLIP))
                    no_seed += 1
                    continue
                cm = ~np.isin(look_pool, seeds)
                UP = B.rows_of(U, look_pool, np.float32)
                for sp in splits:
                    g = fits[sp][1 - f][0]
                    x, lp = columns(torch, g, kappa, F, US, UP, cm)
                    if checked < 2 and cm.any():
                        ref = p5b_lp(torch, g, kappa, F, US, UP[cm])
                        dev = float(np.abs(ref - lp[cm]).max())
                        if dev > 2e-3:
                            raise SystemExit(f"{name}/{carve} question {qid}: lp is off P5b's log p by {dev:.2e}")
                        worst_lp = max(worst_lp, dev)
                    xs[sp].append(x)
                checked += 1
        if gq != q1:
            raise SystemExit(f"{p}: its chunks hold {gq - q0} questions, its record {q1 - q0}")
        for sp in splits:
            x = np.concatenate(xs[sp]) if xs[sp] else np.zeros((0, NCOL))
            if x.shape[0] != int(n.sum()) or not np.isfinite(x).all():
                raise SystemExit(f"{p}: {x.shape[0]} rows of columns against {int(n.sum())}, or one not finite")
            x16, y = stored(x)
            od = Path(out) / sp / name / carve / p.name
            od.mkdir(parents=True, exist_ok=True)
            sha = LC.save_npy(od, "cols", x16)
            qs_ = lambda v: [round(float(t), 4) for t in np.quantile(v, [0.05, 0.5, 0.95])] if v.size else []
            rec = {"declared_in": DECLARED, "stage": "P5c", "split": sp, "dataset": name, "carve": carve,
                   "part": p.name, "step1_record_sha256": step1, "units_record_sha256": usha,
                   "g_records_sha256": [fits[sp][f][2] for f in range(FOLDS)],
                   "kept_epochs": [fits[sp][f][1]["kept_epoch"] for f in range(FOLDS)],
                   "kappa": kappa, "queries": int(n.size), "rows": int(n.sum()),
                   "rows_by_fold_of_question": folds.tolist(), "questions_without_a_seed": no_seed,
                   "settings": BUILD_SETTINGS,
                   "quantiles_5_50_95": {k: qs_(y[:, c]) for c, k in enumerate(COLNAMES)},
                   "clipped_share": {k: round(float((np.abs(x[:, c]) >= CLIP).mean()), 4) if x.size else 0.0
                                     for c, k in enumerate(COLNAMES)},
                   "lp_against_p5b_log_p_max_abs": worst_lp,
                   "max_abs_float16_error": float(np.abs(x16.astype(np.float64) - y).max()) if y.size else 0.0,
                   "arrays": {"cols": {"dtype": x16.dtype.str, "shape": list(x16.shape), "sha256": sha}},
                   "cache_root": Path(cache_root).resolve().relative_to(ROOT).as_posix()
                   if Path(cache_root).resolve().is_relative_to(ROOT) else "toy",
                   "placement": placement or {"where": "host" if host else "laptop"},
                   "script_sha256": script_sha256(), "seconds": round(time.time() - tp, 1), "utc": utc()}
            write_json(od / "record.json", rec)
        done[p.name] = {"queries": int(n.size), "rows": int(n.sum())}
        log(f"  {name}/{carve} {p.name}: {int(n.size)} questions, {int(n.sum())} rows, splits {', '.join(splits)}; "
            f"lp against P5b's log p {worst_lp:.1e} ({time.time() - t0:.0f}s)")
    log(f"p5c cols {name}/{carve}: {len(done)} part(s) ({time.time() - t0:.0f}s)")
    return done


# ── selftest ─────────────────────────────────────────────────────────────────


def columns_by_hand(g, kappa, F, US, UP, cm):
    """The four columns and P5b's log p in float64, seed by seed, head by head, row by row."""
    import torch
    with torch.no_grad():
        delta, a = g(torch.from_numpy(np.asarray(F, np.float32)))
    delta, a = delta.double().numpy(), a.double().numpy()
    beta = float(g.beta.detach())
    S, H = a.shape
    w = np.exp(a.reshape(-1) - a.max())
    lpi = np.log(w / w.sum()).reshape(S, H)
    C = UP.shape[0]
    US64, UP64 = np.asarray(US, np.float64), np.asarray(UP, np.float64)
    lg = np.zeros((S, H, C))
    for s in range(S):
        for h in range(H):
            for v in range(C):
                lg[s, h, v] = -kappa * np.exp(beta) * float(((US64[s] + delta[s, h] - UP64[v]) ** 2).sum())
    l0 = np.array([[-kappa * float(((US64[s] - UP64[v]) ** 2).sum()) for v in range(C)] for s in range(S)])

    def lse(z):
        z = np.asarray(z, np.float64)
        m = z.max()
        return m + np.log(np.exp(z - m).sum())

    out, lp = np.zeros((C, NCOL)), np.zeros(C)
    for v in range(C):
        out[v] = [lse([lpi[s, h] + lg[s, h, v] for s in range(S) for h in range(H)]), lg[:, :, v].max(),
                  lse([l0[s, v] - np.log(S) for s in range(S)]), l0[:, v].max()]
        lp[v] = lse([lpi[s, h] + lg[s, h, v] - (lse(lg[s, h, cm]) if cm.any() else 0.0)
                     for s in range(S) for h in range(H)])
    return out, lp


def toy_units(rng, name, nq, N, U=None, shift=None):
    units, golds, ids, pos = [], [], [], []
    for j in range(nq):
        S = int(rng.integers(2, 5))
        pool = rng.choice(N, int(rng.integers(S + 3, S + 12)), replace=False)
        seeds = np.sort(pool[:S])
        cand = np.setdiff1d(pool, seeds)
        units.append({"x": rng.normal(0, 1, B.DIM + 1).astype(np.float32), "seeds": seeds,
                      "rd": rng.integers(1, 1002, S), "rs": rng.integers(1, 1002, S),
                      "deg": rng.uniform(0, 9, (S, 3)).astype(np.float32),
                      "m": {sp: rng.normal(0, 0.1, (S, B.DIM)).astype(np.float32) for sp in B.SPACES}, "cand": cand})
        if U is None or rng.random() < 0.15:
            golds.append(cand[rng.random(cand.size) < 0.3] if U is None else np.zeros(0, np.int64))
        else:                       # learnable: the two candidates nearest the first seed moved by a fixed offset
            d = ((U[cand] - (U[seeds[0]] + shift)) ** 2).sum(1)
            golds.append(cand[np.argsort(d)[:2]])
        ids.append(f"{name}-q{j}")
        pos.append(j)
    return B.pack_units(units, golds, ids, pos)


def toy_look(tmp, name, carve, arr, rng, chunks=(7, 6)):
    """A look (two chunks: each question's pool, shuffled, and one question the units lack, with no row) and step 1's
    cache records for it (a part per chunk), as lean_cache reads them."""
    import lean_cache as LC
    look = tmp / "look" / name / carve
    (look / "chunks").mkdir(parents=True)
    order = [str(q) for q in arr["ids"].tolist()][:sum(chunks) - 1] + ["no-pool"]
    at = {str(q): i for i, q in enumerate(arr["ids"].tolist())}
    q0, per = 0, []
    for ci, nqc in enumerate(chunks):
        sizes, pools = [], []
        for q in order[q0:q0 + nqc]:
            if q == "no-pool":
                sizes.append(0)
                continue
            i = at[q]
            s0, s1, c0, c1 = arr["seed_ptr"][i], arr["seed_ptr"][i + 1], arr["cand_ptr"][i], arr["cand_ptr"][i + 1]
            pl = rng.permutation(np.concatenate([arr["seeds"][s0:s1], arr["cand"][c0:c1]]))
            sizes.append(pl.size)
            pools.append(pl)
        np.savez(look / "chunks" / f"c{ci:05d}.npz", q_pool_size=np.asarray(sizes, np.int64),
                 pool=np.concatenate(pools).astype(np.int64))
        per.append(np.asarray(sizes, np.int64))
        q0 += nqc
    rec = {"chunks": list(range(len(chunks))), "n_chunks": len(chunks), "carve_ids_sha256": "toy", "full": True,
           "limit": None, "carve_queries": len(order), "chunk_queries": max(chunks), "columns": ["rrf"],
           "column_blocks": {}}
    write_json(look / "record_0of1.json", rec)
    write_json(look / "ids.json", order)
    head_records = {"record_0of1.json": LC.sha_file(look / "record_0of1.json")}
    q0 = r0 = 0
    for pi, n in enumerate(per):
        pd = tmp / "cache" / name / carve / f"part_{pi}of{len(per)}"
        pd.mkdir(parents=True)
        sha = LC.save_npy(pd, "n", n)
        write_json(pd / "record.json", {"query_range": [q0, q0 + n.size], "row_range": [r0, r0 + int(n.sum())],
                                        "ids": order[q0:q0 + n.size],
                                        "look": {"carve_queries": len(order), "carve_ids_sha256": "toy",
                                                 "records": head_records},
                                        "chunks": [pi, pi + 1], "queries": int(n.size), "rows": int(n.sum()),
                                        "arrays": {"n": {"sha256": sha}}})
        q0, r0 = q0 + n.size, r0 + int(n.sum())
    return order


def refused(call, what):
    try:
        call()
    except SystemExit:
        return
    raise AssertionError(what)


def toy_end_to_end(torch, rng):
    import tempfile
    N = 400
    U = rng.normal(0, 0.3, (N, B.DIM)).astype(np.float32)
    shift = np.full(B.DIM, 0.12, np.float32)
    tshas = {"frame": "toy-frame", "V": "toy-V", "V0": "toy-V0", "space_record": "toy-space"}
    saved = (B.filed_tables, B2.filed_base)
    B.filed_tables = lambda name: ((None, None), {"V": U, "V0": U}, tshas)
    try:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            ur, gr, cr = tmp / "units", tmp / "g", tmp / "cols"
            arrs = {}
            for name in TRAIN5:
                arrs[name] = toy_units(rng, name, 60, N, U, shift)
                pz, pj = units_paths(name, A.fit_carve(name), ur)
                save_npz(pz, **arrs[name])
                write_json(pj, {"declared_in": DECLARED, "stage": "units", "npz_sha256": sha_file(pz),
                                "tables": tshas})
            # the P5b check: P5b's questions (a subset) reproduce; one moved candidate is refused
            sub = B2.subset(arrs["squad"], np.array([1, 4, 9]))
            B2.filed_base = lambda name, root=None: (sub, {}, "toy-base")
            assert same_as_p5b("squad", arrs["squad"])["questions_checked"] == 3
            bad = {k: v.copy() for k, v in sub.items()}
            bad["cand"][0] += 1
            B2.filed_base = lambda name, root=None: (bad, {}, "toy-base")
            refused(lambda: same_as_p5b("squad", arrs["squad"]), "a moved candidate passed the P5b check")
            for sp in SPLITS:
                for f in range(FOLDS):
                    fitg_stage(sp, f, "cpu", 1, out=gr, units_root=ur, epochs=4)
            order = toy_look(tmp, "metaqa", "fit", arrs["metaqa"], rng)
            done = cols_stage("metaqa", "fit", 1, out=cr, units_root=ur, g_root=gr, cache_root=tmp / "cache",
                              look_root=tmp / "look")
            assert sorted(done) == ["part_0of2", "part_1of2"]
            arr = arrs["metaqa"]
            at = {str(q): i for i, q in enumerate(arr["ids"].tolist())}
            kap = B.kappa_of(U)
            ch = tmp / "look" / "metaqa" / "fit" / "chunks"
            with np.load(ch / "c00000.npz") as z0, np.load(ch / "c00001.npz") as z1:
                sizes = np.concatenate([z0["q_pool_size"], z1["q_pool_size"]])
                pools = np.concatenate([z0["pool"], z1["pool"]])
            for sp in SPLITS:
                g = {f: load_fit(torch, sp, f, gr)[0] for f in range(FOLDS)}
                got = np.concatenate([np.load(cr / sp / "metaqa" / "fit" / pn / "cols.npy") for pn in sorted(done)])
                assert got.dtype == np.float16 and got.shape == (sum(done[pn]["rows"] for pn in done), NCOL)
                want, r = [], 0
                for q, s in zip(order, sizes.tolist()):
                    if not s:
                        continue
                    lp_ = pools[r:r + s]
                    r += s
                    F, US, seeds, _pool = question_inputs(arr, at[q], U)
                    x, _lp = columns_by_hand(g[1 - fold_of("metaqa", q)], kap, F, US, U[lp_], ~np.isin(lp_, seeds))
                    want.append(stored(x)[1])
                want = np.concatenate(want)
                dev = np.abs(got.astype(np.float64) - want).max()
                assert dev < 3e-3, dev
                rec = json.loads((cr / sp / "metaqa" / "fit" / "part_0of2" / "record.json").read_text("utf-8"))
                assert rec["settings"] == BUILD_SETTINGS and rec["cache_root"] == "toy" and rec["rows"] > 0
            # a fold's own g gives other columns: the cross-fit is not a no-op
            F, US, seeds, pool = question_inputs(arr, 0, U)
            cm = ~np.isin(pool, seeds)
            (ga, ra, _), (gb, rb, _) = (load_fit(torch, "L-musique", f, gr) for f in range(FOLDS))
            assert ra["kept_epoch"] > 0 and rb["kept_epoch"] > 0, (ra["kept_epoch"], rb["kept_epoch"])
            assert np.abs(columns(torch, ga, kap, F, US, U[pool], cm)[0]
                          - columns(torch, gb, kap, F, US, U[pool], cm)[0]).max() > 1e-6
            # refusals: a carve no split reads; a cache row outside the final pool
            refused(lambda: cols_stage("webqsp", "fit", 1, out=cr, units_root=ur, g_root=gr), "an unread carve built")
            refused(lambda: units_stage("webqsp", "fit", 1, False, out=ur), "an unread carve's units built")
            with np.load(ch / "c00000.npz") as z:
                c0 = {k: z[k] for k in z.files}
            c0["pool"][0] = int(np.setdiff1d(np.arange(N), c0["pool"])[0])
            np.savez(ch / "c00000.npz", **c0)
            refused(lambda: cols_stage("metaqa", "fit", 1, out=tmp / "cols2", units_root=ur, g_root=gr,
                                       cache_root=tmp / "cache", look_root=tmp / "look"),
                    "a row outside the final pool passed")
    finally:
        B.filed_tables, B2.filed_base = saved


def selftest():
    import torch
    t0 = time.time()
    A.full_fp32(torch)
    rng = np.random.default_rng(71)
    # 1. the splits: each split's carves are its four training datasets' fit carves and the six s1eval carves; its
    #    held-out dataset trains no g
    for sp, names in SPLITS.items():
        assert HELD_OUT[sp] not in names and len(names) == 4 and set(names) <= set(TRAIN5)
        assert CARVES_OF[sp][:4] == tuple((d, A.fit_carve(d)) for d in names)
        assert {c for c in CARVES_OF[sp][4:]} == {(d, "s1eval") for d in SIX}
    assert len(CARVES) == 11 and ("musique", "s1fit") in CARVES and ("hotpotqa", "fit") in CARVES
    # 2. folds: deterministic, both used, about half each; the cross-fit takes the other fold's g
    f = np.array([fold_of("musique", f"q{k}") for k in range(4000)])
    assert set(f.tolist()) == {0, 1} and abs(f.mean() - 0.5) < 0.03
    assert fold_of("musique", "q1") == fold_of("musique", "q1") and any(
        fold_of("musique", f"q{k}") != fold_of("squad", f"q{k}") for k in range(20))
    # 3. the columns against the float64 reference, for a moved g; P5b's log p (the check) is B.log_p's; at g's start,
    #    pt and mx are pt0 and mx0; pt <= mx <= 0, and pt is not P5b's log p (nothing is normalised over the rows)
    g = B.new_g(torch, torch.device("cpu"))
    with torch.no_grad():
        for prm in g.parameters():
            prm.add_(0.05 * torch.randn(prm.shape, generator=torch.Generator().manual_seed(prm.numel())))
        g.beta.fill_(0.3)
    S, C = 5, 23
    US = rng.normal(0, 0.3, (S, B.DIM)).astype(np.float32)
    UP = rng.normal(0, 0.3, (C, B.DIM)).astype(np.float32)
    UP[:2] = US[:2]
    cm = np.ones(C, bool)
    cm[:2] = False
    F = np.concatenate([rng.normal(0, 1, (S, B.D_IN - 2)), rng.uniform(0, 0.02, (S, 2))], 1).astype(np.float32)
    kappa = 1.2
    got, lp = columns(torch, g, kappa, F, US, UP, cm)
    ref, lp_ref = columns_by_hand(g, kappa, F, US, UP, cm)
    assert got.shape == (C, NCOL) and np.abs(got - ref).max() < 1e-4, np.abs(got - ref).max()
    assert np.abs(lp - lp_ref).max() < 1e-4
    assert np.abs(p5b_lp(torch, g, kappa, F, US, UP[cm]) - lp[cm]).max() < 2e-3
    g0 = B.new_g(torch, torch.device("cpu"))
    at0, _ = columns(torch, g0, kappa, F, US, UP, cm)
    assert np.abs(at0[:, 0] - at0[:, 2]).max() < 1e-9 and np.abs(at0[:, 1] - at0[:, 3]).max() < 1e-9
    assert np.abs(at0[:, 2:] - got[:, 2:]).max() < 1e-9          # the start's columns do not depend on g
    assert np.all(got[:, 1] <= 0) and np.all(got[:, 0] <= got[:, 1] + 1e-9) and np.all(got[:, 3] <= 0)
    assert float(np.abs(got[cm, 0] - lp[cm]).max()) > 1e-3
    assert np.abs(got[:2, 3]).max() < 1e-9                       # a seed row is at distance 0 from itself
    # 4. a pool of seeds only, and a row's columns do not depend on the other rows (pointwise)
    only, _ = columns(torch, g, kappa, F, US, US[:3], np.zeros(3, bool))
    ref_only, _ = columns_by_hand(g, kappa, F, US, US[:3], np.zeros(3, bool))
    assert np.isfinite(only).all() and np.abs(only - ref_only).max() < 1e-4
    alone = np.vstack([columns(torch, g, kappa, F, US, UP[v:v + 1], cm[v:v + 1])[0] for v in range(C)])
    assert np.abs(alone - got).max() < 1e-9
    # 5. stored: clipped below at -CLIP, divided by SCALE, float16 within its rounding
    x16, y = stored(np.array([[0.0, -5.0, -45.0, -39.9], [52.0, 7.0, 0.0, 40.0]]))
    assert np.allclose(y, [[0.0, -0.5, -4.0, -3.99], [4.0, 0.7, 0.0, 4.0]])
    assert np.abs(x16.astype(np.float64) - y).max() < 2e-3
    # 6. the features are P5b's: question_inputs builds B.features' rows from a units question
    arr = {"x": rng.normal(0, 1, (2, B.DIM + 1)).astype(np.float32), "seed_ptr": np.array([0, 3, 5]),
           "seeds": np.array([4, 9, 2, 7, 1]), "rank_d": np.array([1, 2, 1001, 3, 4], np.int32),
           "rank_s": np.array([5, 1001, 1, 2, 1], np.int32), "deg": rng.uniform(0, 9, (5, 3)).astype(np.float32),
           "m_V": rng.normal(0, 0.1, (5, B.DIM)).astype(np.float32), "cand_ptr": np.array([0, 4, 6]),
           "cand": np.array([0, 3, 5, 8, 2, 6])}
    Utab = rng.normal(0, 0.3, (10, B.DIM)).astype(np.float32)
    F1, US1, s1, pool1 = question_inputs(arr, 1, Utab)
    assert np.array_equal(s1, [7, 1]) and np.array_equal(pool1, [1, 2, 6, 7])
    assert np.array_equal(US1, Utab[[7, 1]]) and F1.shape == (2, B.D_IN)
    assert np.allclose(F1[:, :B.DIM + 1], arr["x"][1]) and np.allclose(F1[:, -2], 1.0 / (B.RANK_C + np.array([3, 4])))
    # 7. fold_index: the fold's questions, only those with a counted gold
    fa = {"ids": np.array(["a", "b", "c", "d"]), "cand_ptr": np.array([0, 2, 3, 5, 6]),
          "gold": np.array([0, 1, 0, 1, 1, 0], bool)}
    want = [i for i, q in enumerate("abcd") if fold_of("squad", q) == 1 and i in (0, 2)]
    assert fold_index("squad", fa, 1).tolist() == want
    assert fold_index("squad", fa, 1, counted=False).tolist() == [i for i, q in enumerate("abcd")
                                                                   if fold_of("squad", q) == 1]
    toy_end_to_end(torch, rng)
    log(f"p5c selftest: the splits' carves and held-out datasets; folds deterministic and balanced; the four columns "
        f"match the float64 reference and are pointwise, P5b's log p (the check) is B.log_p's, at g's start pt and mx "
        f"are pt0 and mx0; a pool of seeds only; the stored transform; P5b's features from a units question; the fold's questions; end to end on a "
        f"toy: the P5b check of units and its refusal, four g fits, the columns of both splits from the other fold's g, "
        f"the build's refusals "
        f"({time.time() - t0:.1f}s): ok")
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["--selftest"]:
        return selftest()
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("units", "fitg", "cols"))
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--host", action="store_true")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--device", default="cpu")
    k = ap.parse_args(argv)
    if k.cmd == "units":
        return units_stage(k.a, k.b, k.threads, k.host, k.limit)
    if k.cmd == "fitg":
        return fitg_stage(k.a, int(k.b), k.device, k.threads)
    return 0 if cols_stage(k.a, k.b, k.threads, k.host, placement={"where": "host" if k.host else "laptop"}) else 1


if __name__ == "__main__":
    sys.exit(main())
