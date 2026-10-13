"""P5b2: P5b's learned offset g(question, seed, graph features), its fit stopped at the epoch that best predicts new
golds on held-out fit questions (docs/P5B2_HELD_OUT_STOP.md). A diagnostic: no ranking model trains or reads, and
nothing is adopted.

    python scripts/p5b2_stop.py train [DATASET --limit Q] [--host] [--device cuda] [--threads 2]
    python scripts/p5b2_stop.py read DATASET [--host] [--device cuda] [--threads 2] [--limit Q]
    python scripts/p5b2_stop.py verdict
    python scripts/p5b2_stop.py --selftest

train: P5b's five filed base files, each split by a seeded permutation into 80% train and 20% held out. G (on V) and
G0 (on V0) fit on the train questions as P5b's did, and each keeps the epoch (0 to 30; 0 is g's start) with the lowest
dataset-balanced held-out loss over new golds: held-out golds that no train question of the dataset has.
read: on the s1eval sample, every P5b order with P5b2's g, beside P5b's filed G (GB) and its fusion with H2 (FB); every
count split into golds seen in the fit sample and new golds.
verdict: section 5, on the six read records; GB and FB are checked against P5b's filed reads.
--limit: a development run. It writes nothing and logs no measure: only that every step ran, the checks, and that the
fit's loss is finite. Where P5b's files are not on disk, this dataset's base is built in memory (P5b's development
path), and g at its start stands in for P5b's g.
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
import tempfile  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import p5b_offsets as B  # noqa: E402  (P5b's filed script, imported unchanged)

A = B.A                                       # P5a's filed script, through P5b
DECLARED = "docs/P5B2_HELD_OUT_STOP.md"
OUT = ROOT / "outputs" / "p5b2"
P5B_OUT = B.OUT
SIX, TRAIN5, SPACES, MODEL_OF, READS, GRID, RINGS = B.SIX, B.TRAIN5, B.SPACES, B.MODEL_OF, B.READS, B.GRID, B.RINGS
ORDERS = B.ORDERS + ("GB", "FB")              # GB: P5b's filed G; FB: GB fused with H2
CLASSES = ("seen", "new")
HOLD = 0.2
SPLIT_SEED = 0
EPOCHS = B.EPOCHS                             # 30, the most the stop can keep
DEV_EPOCHS = B.DEV_EPOCHS
KEYS = {"ids", "pos", "x", "seed_ptr", "seeds", "rank_d", "rank_s", "deg", "m_V", "m_V0", "cand_ptr", "cand", "gold"}
# P5b's files as filed (a13b484): the base records, the train record and the script that wrote them
P5B_SCRIPT = "d1fe54765dbe31c8950f1b0f6a253a440c00eb1e66adf55fc1ee708a0cdeb134"
P5B_TRAIN = "6e8cb10cab96f80ef39be78c9c81dbfbe65126fd2e4317a8c9684dab5c004bcb"
P5B_BASE = {"metaqa": "5f82fa3244e0b5cd37afbe771f242d05d49c43e6a3d9df0b58bac4be38be26b4",
            "squad": "404ec4dc4f0909010fb15af24dbdb2d30767cbb83ab9250cea5d618ce5586855",
            "musique": "b842c7689ee318bcaf05ab08f9417374b45e132376d31d0d61ea2bfe40ed244a",
            "hotpotqa": "5a09c7dd3afbcee025537ee1c6a7c30160e39e3680c80cbfcd55d1aefefe3000",
            "2wiki": "6572e767e23570fb8ef5a5f7139cf195877f948ac3f6720d0ddc387a7cbf154a"}
STOP_RULE = ("each fit keeps the epoch (0 to 30; 0 is g's start) whose dataset-balanced mean, over the held-out "
             "questions with a new gold, of -mean log p over their new golds is lowest; ties go to the earlier epoch")
log, utc, sha_file, write_json, save_npz = A.log, A.utc, A.sha_file, A.write_json, A.save_npz


def script_sha256():
    return hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def out_paths(out=OUT):
    return B.out_paths(out)


def p5b_is_filed():
    if B.script_sha256() != P5B_SCRIPT:
        raise SystemExit("scripts/p5b_offsets.py is not P5b's filed script")


# ── P5b's filed bases ────────────────────────────────────────────────────────


def filed_base(name, root=P5B_OUT):
    """P5b's base arrays of a training dataset, its record and the record's sha256, each checked against the filed
    record."""
    base = B.out_paths(root)["base"]
    bj = base / f"fit_{name}.json"
    if sha_file(bj) != P5B_BASE[name]:
        raise SystemExit(f"{bj}: not P5b's filed base record")
    rec = json.loads(bj.read_text(encoding="utf-8"))
    bp = base / f"fit_{name}.npz"
    if sha_file(bp) != rec["npz_sha256"] or rec["script_sha256"] != P5B_SCRIPT:
        raise SystemExit(f"{bp}: not its record's file, or written by another script")
    with np.load(bp) as z:
        arr = {k: z[k] for k in z.files}
    if set(arr) != KEYS:
        raise SystemExit(f"{bp}: keys {sorted(arr)}")
    return arr, rec, P5B_BASE[name]


def has_filed_base(name, root=P5B_OUT):
    base = B.out_paths(root)["base"]
    return (base / f"fit_{name}.json").exists() and (base / f"fit_{name}.npz").exists()


def golds_of(arr):
    """The counted golds of a base's questions, as sorted unique node ids."""
    return np.unique(np.asarray(arr["cand"], np.int64)[np.asarray(arr["gold"], bool)])


# ── the held-out split ───────────────────────────────────────────────────────


def split_of(n, frac=HOLD, seed=SPLIT_SEED):
    """(train, held-out) question indices, each sorted: a seeded permutation's first round(frac n) are held out."""
    perm = np.random.default_rng(seed).permutation(n)
    k = int(round(frac * n))
    return np.sort(perm[k:]), np.sort(perm[:k])


def subset(arr, idx):
    """A base's questions at idx (in that order), with their seeds' and candidates' rows and fresh pointers."""
    idx = np.asarray(idx, np.int64)

    def rows(ptr):
        ptr = np.asarray(ptr, np.int64)
        lens = ptr[idx + 1] - ptr[idx]
        sel = (np.concatenate([np.arange(ptr[i], ptr[i + 1]) for i in idx.tolist()]) if idx.size
               else np.zeros(0, np.int64))
        return sel.astype(np.int64), np.r_[0, np.cumsum(lens)].astype(np.int64)

    ss, sp = rows(arr["seed_ptr"])
    cs, cp = rows(arr["cand_ptr"])
    out = {"ids": arr["ids"][idx], "pos": arr["pos"][idx], "x": arr["x"][idx], "seed_ptr": sp, "cand_ptr": cp}
    for k in ("seeds", "rank_d", "rank_s", "deg", "m_V", "m_V0"):
        out[k] = arr[k][ss]
    for k in ("cand", "gold"):
        out[k] = arr[k][cs]
    return out


def split_base(arr):
    """(train arrays, held-out arrays, the held-out candidates' new-gold flags, counts)."""
    tr, ho = split_of(int(arr["ids"].size))
    a_tr, a_ho = subset(arr, tr), subset(arr, ho)
    new = np.asarray(a_ho["gold"], bool) & ~np.isin(a_ho["cand"], golds_of(a_tr))
    with_new = int(sum(new[a_ho["cand_ptr"][i]:a_ho["cand_ptr"][i + 1]].any() for i in range(ho.size)))
    counts = {"train": int(tr.size), "held_out": int(ho.size), "held_out_golds": int(np.asarray(a_ho["gold"]).sum()),
              "held_out_new_golds": int(new.sum()), "held_out_with_a_new_gold": with_new}
    return a_tr, a_ho, new, counts


# ── the fit, stopped on new golds ────────────────────────────────────────────


def held_out(torch, dev, g, pack, new):
    """Per held-out question: -mean log p over its golds, and over its new golds (nan where it has none)."""
    la, ln = np.zeros(pack.n), np.full(pack.n, np.nan)
    with torch.no_grad():
        for lo in range(0, pack.n, B.BATCH):
            items = [(0, i) for i in range(lo, min(pack.n, lo + B.BATCH))]
            F, US, sm, UC, cm, gm, kap = B.batch(torch, dev, [pack], items)
            lp = B.log_p(torch, g, F, US, sm, UC, cm, kap)
            nm = np.zeros(tuple(gm.shape), bool)
            for b, (_d, i) in enumerate(items):
                c0, c1 = int(pack.cp[i]), int(pack.cp[i + 1])
                nm[b, :c1 - c0] = new[c0:c1]
            la[lo:lo + len(items)] = B.question_loss(torch, lp, gm).double().cpu().numpy()
            l_new = B.question_loss(torch, lp, torch.from_numpy(nm).to(dev)).double().cpu().numpy()
            ln[lo:lo + len(items)] = np.where(nm.any(1), l_new, np.nan)
    return la, ln


def criterion(per):
    """{dataset: (la, ln)} -> (the stop's criterion, the held-out loss over all golds and over new golds, by dataset)."""
    by_all = {n: float(la.mean()) for n, (la, _ln) in per.items() if la.size}
    by_new = {n: float(np.nanmean(ln)) for n, (_la, ln) in per.items() if np.isfinite(ln).any()}
    crit = float(np.mean(list(by_new.values()))) if by_new else float("nan")
    return crit, by_all, by_new


def fit_stop(torch, dev, packs, hos, news, names, epochs, logf=log):
    """P5b's fit (B.fit_g's loop, unchanged) on the train questions, with the held-out check after every epoch and at
    g's start; returns g at the kept epoch, the history and the kept epoch."""
    g = B.new_g(torch, dev)
    opt = torch.optim.Adam(g.parameters(), lr=B.LR)
    items = [(d, i) for d, p in enumerate(packs) for i in range(p.n)]
    dsi = np.asarray([d for d, _i in items], np.int64)
    nd = np.asarray([packs[d].n for d in dsi], np.float64)
    omega = (len(items) / (len(packs) * nd)).astype(np.float32)     # each dataset weighs the same; mean weight 1
    rng = np.random.default_rng(0)

    def check():
        return criterion({names[d]: held_out(torch, dev, g, hos[d], news[d]) for d in range(len(names)) if hos[d].n})

    crit, ba, bn = check()
    if not np.isfinite(crit):
        raise SystemExit("no held-out question has a new gold")
    hist = [{"epoch": 0, "J": None, "by_dataset": None, "beta": g.beta.detach().item(), "held_out_all": ba,
             "held_out_new": bn, "criterion": crit, "seconds": 0.0}]
    best = (crit, 0, {k: v.detach().clone() for k, v in g.state_dict().items()})
    t = time.time()
    for ep in range(epochs):
        perm = rng.permutation(len(items))
        per_q = np.zeros(len(items))
        J = 0.0
        for lo in range(0, len(items), B.BATCH):
            idx = perm[lo:lo + B.BATCH]
            F, US, sm, UC, cm, gm, kap = B.batch(torch, dev, packs, [items[k] for k in idx])
            ql = B.question_loss(torch, B.log_p(torch, g, F, US, sm, UC, cm, kap), gm)
            loss = (torch.from_numpy(omega[idx]).to(dev) * ql).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            per_q[idx] = ql.detach().double().cpu().numpy()
            J += loss.detach().item() * idx.size
        by = {names[d]: float(per_q[dsi == d].mean()) for d in range(len(packs))}
        crit, ba, bn = check()
        hist.append({"epoch": ep + 1, "J": J / len(items), "by_dataset": by, "beta": g.beta.detach().item(),
                     "held_out_all": ba, "held_out_new": bn, "criterion": crit,
                     "seconds": round(time.time() - t, 1)})
        vals = [J, crit, *by.values(), *ba.values(), *bn.values()]
        if not all(np.isfinite(v) for v in vals):
            raise SystemExit(f"a loss is not finite at epoch {ep + 1}")
        if crit < best[0]:
            best = (crit, ep + 1, {k: v.detach().clone() for k, v in g.state_dict().items()})
        logf(f"  epoch {ep + 1}/{epochs} ({time.time() - t:.0f}s)")
    g.load_state_dict(best[2])
    g.eval()
    return g, hist, best[1]


def fit_both(torch, dev, names, splits, tables, epochs, logf=log):
    """G on V and G0 on V0, each stopped; returns {space: (g, hist, kept epoch)} and kappa per space and dataset."""
    fits, kap = {}, {}
    for sp in SPACES:
        kap[sp] = {n: B.kappa_of(tables[n][sp]) for n in names}
        packs = [B.Pack(splits[n][0], tables[n][sp], kap[sp][n], sp) for n in names]
        hos = [B.Pack(splits[n][1], tables[n][sp], kap[sp][n], sp) for n in names]
        news = [splits[n][2] for n in names]
        logf(f"{MODEL_OF[sp]} on {sp}: {sum(p.n for p in packs)} train and {sum(h.n for h in hos)} held-out questions")
        fits[sp] = fit_stop(torch, dev, packs, hos, news, list(names), epochs, logf)
        del packs, hos
    return fits, kap


def train_stage(device, threads, host, limit=None, dataset=None, out=OUT):
    import torch
    torch.set_num_threads(threads)
    A.full_fp32(torch)
    p5b_is_filed()
    dev = torch.device(device)
    t0 = time.time()
    paths = out_paths(out)
    names, splits, tables, brecs, counts = [], {}, {}, {}, {}
    p5b_kappa = None
    if limit:
        if dataset not in TRAIN5:
            raise SystemExit("a development train needs DATASET, one of the five training datasets")
        arr, crec, tab = B.base_data(dataset, threads, host, limit)
        names, tables[dataset] = [dataset], tab
        a_tr, a_ho, new, counts[dataset] = split_base(arr)
        splits[dataset] = (a_tr, a_ho, new)
        made = crec["tables_made"]
    else:
        if dataset:
            raise SystemExit("train reads P5b's five base files; DATASET is for development runs")
        tj = B.out_paths(P5B_OUT)["train"]
        if sha_file(tj) != P5B_TRAIN:
            raise SystemExit(f"{tj}: not P5b's filed train record")
        p5b_kappa = json.loads(tj.read_text(encoding="utf-8"))["kappa"]
        for name in TRAIN5:
            arr, rec, brecs[name] = filed_base(name)
            _frame, tables[name], shas = B.filed_tables(name)
            if shas != rec["tables"]:
                raise SystemExit(f"{name}: P5b's base ran on other tables")
            a_tr, a_ho, new, counts[name] = split_base(arr)
            if a_tr["ids"].size:
                names.append(name)
                splits[name] = (a_tr, a_ho, new)
        made = "filed"
    if not names:
        raise SystemExit("no training question has a counted gold")
    epochs = DEV_EPOCHS if limit else EPOCHS
    fits, kap = fit_both(torch, dev, names, splits, tables, epochs)
    if p5b_kappa is not None and any(kap[sp][n] != p5b_kappa[sp][n] for sp in SPACES for n in names):
        raise SystemExit("kappa is not P5b's")
    if limit:
        log(f"train development run on {dataset}: both fits ran ({epochs} epoch, the held-out check at every epoch), "
            f"every loss finite; tables: {made}; {time.time() - t0:.0f}s, gpu peak "
            f"{A.gpu_peak(torch, dev) / 2 ** 30:.2f} GB; nothing written")
        return 0
    config = {"d_in": B.D_IN, "width": B.WIDTH, "heads": B.HEADS, "dim": B.DIM, "rank_c": B.RANK_C,
              "weights": B.N_WEIGHTS}
    B.save_g(paths["g"], {MODEL_OF[sp]: fits[sp][0] for sp in SPACES}, config)
    rec = {"declared_in": DECLARED, "stage": "train", "datasets": names, "split": counts,
           "p5b_base_records": brecs, "p5b_train_record": P5B_TRAIN, "p5b_script_sha256": P5B_SCRIPT,
           "kappa": kap, "settings": {"hold": HOLD, "split_seed": SPLIT_SEED, "max_epochs": EPOCHS,
                                      "stop": STOP_RULE, "batch": B.BATCH, "lr": B.LR, "optimizer": "Adam",
                                      "seed": 0, "precision": "float32, TF32 off", "kappa_pairs": B.KAPPA_PAIRS,
                                      **config},
           "fits": {MODEL_OF[sp]: {"space": sp, "hist": fits[sp][1], "kept_epoch": fits[sp][2],
                                   "beta": fits[sp][0].beta.detach().item()} for sp in SPACES},
           "index_time": B.INDEX_TIME, "g_sha256": sha_file(paths["g"]), "script_sha256": script_sha256(),
           "p5a_script_sha256": A.script_sha256(), "device": str(dev), "gpu_peak_bytes": A.gpu_peak(torch, dev),
           "peak_rss_bytes": A.peak_rss(), "seconds": round(time.time() - t0, 1), "utc": utc()}
    write_json(paths["train"], rec)
    log(f"outputs/p5b2/train.json ({rec['seconds']:.0f}s)")
    return 0


# ── read ─────────────────────────────────────────────────────────────────────


def read_stage(name, device, threads, host, limit=None, out=OUT):
    import torch
    torch.set_num_threads(threads)
    A.full_fp32(torch)
    p5b_is_filed()
    E, _F, S4 = A.modules()
    op = A.opened(host)
    dev = torch.device(device)
    t0 = time.time()
    paths = out_paths(out)
    ds, emb = A.dense_docs(op, name)
    n = emb.n_rows
    frame, tables, tshas, made = B.tables_for(name, emb, limit)
    _stores, adj = A.family_stores(S4, ds, name, A.csr_dir(op))
    if any(ip.size - 1 != n for ip, _c in adj):
        raise SystemExit(f"{name}: a family's store is not the corpus's size")
    dev_arr = None
    if name not in TRAIN5:
        seen, seen_made = np.zeros(0, np.int64), "never fits: every gold is new"
    elif has_filed_base(name):
        arr, _rec, _sha = filed_base(name)
        seen, seen_made = golds_of(arr), "P5b's filed base"
        del arr
    elif limit:
        dev_arr = B.base_data(name, threads, host, limit)[0]
        seen, seen_made = golds_of(dev_arr), "development: this dataset's base built in memory"
    else:
        raise SystemExit(f"{name}: P5b's base is not on disk")
    if paths["g"].exists() and paths["train"].exists():
        models, trec = B.load_g(torch, dev, paths)
        g_made = "filed"
    elif limit:
        trec = None
        if name in TRAIN5 and dev_arr is not None and dev_arr["ids"].size:
            a_tr, a_ho, new, _c = split_base(dev_arr)
            tab = {name: tables}
            fits, _kap = fit_both(torch, dev, [name], {name: (a_tr, a_ho, new)}, tab, DEV_EPOCHS,
                                  logf=lambda *_: None)
            models, g_made = ({MODEL_OF[sp]: fits[sp][0] for sp in SPACES},
                              "development: a one-epoch stopped g on this dataset's base")
        else:
            models, g_made = {"G": B.new_g(torch, dev), "G0": B.new_g(torch, dev)}, "development: g at its start"
    else:
        raise SystemExit("g is not on disk: run train first")
    pb = B.out_paths(P5B_OUT)
    if pb["g"].exists() and pb["train"].exists():
        if sha_file(pb["train"]) != P5B_TRAIN:
            raise SystemExit(f"{pb['train']}: not P5b's filed train record")
        gb, trb = B.load_g(torch, dev, pb)
        models["GB"], gb_made, gb_sha = gb["G"], "P5b's filed G", trb["g_sha256"]
    elif limit:
        models["GB"], gb_made, gb_sha = B.new_g(torch, dev).eval(), "development: g at its start stands in", None
    else:
        raise SystemExit("P5b's g is not on disk")
    models["NEAR"] = B.new_g(torch, dev).eval()
    r1 = B.read1_or_none(name, limit)
    get, pinfo = B.finals_for(name, limit)
    kap = {sp: B.kappa_of(tables[sp]) for sp in SPACES}
    if trec is not None and any(name in trec["kappa"][sp] and trec["kappa"][sp][name] != kap[sp] for sp in SPACES):
        raise SystemExit(f"{name}: kappa is not the fit's")
    mark = np.zeros(n, np.int8)
    buf = np.empty(n, np.int64)
    log(f"{name}: {n} nodes (tables {made}; g {g_made}; GB {gb_made}; seen golds {seen_made}; {device}); setup "
        f"{time.time() - t0:.0f}s")
    holder = {}

    def sink(name_, carve, pop, prep, U, BQ, bs, inc, linked, union, constant, families, construction):
        Q = len(pop.ids)
        samp = A.sample_of(Q)
        sample_sha = A.ids_sha([pop.ids[j] for j in samp])
        if limit:
            samp = samp[:limit]
        ids = [pop.ids[j] for j in samp]
        if r1 is not None and (r1["sample_sha256"] != sample_sha or r1["ids"][:len(ids)] != ids
                               or r1["pos"][:len(ids)] != samp.tolist()):
            raise SystemExit(f"{name}: the sample is not read1's")
        finals = get(carve)
        cache = {}
        units, golds, checked = [], [], 0
        t = time.time()
        for qi, j in enumerate(samp.tolist()):
            pool = finals.pool(pop.ids[j], bs[j], inc[j]) if finals else np.unique(np.asarray(inc[j], np.int64))
            if r1 is not None:
                if not (np.array_equal(np.unique(np.asarray(inc[j], np.int64)), np.unique(r1["pools"][qi]))
                        and np.array_equal(np.unique(np.asarray(prep.seeds[j], np.int64)),
                                           np.unique(r1["seeds"][qi]))):
                    raise SystemExit(f"{name}: {pop.ids[j]}: today's pool or seeds are not read1's")
                checked += 1
            u = B.unit(j, pop, prep, pool, tables, adj, frame, cache)
            units.append(u)
            golds.append(np.intersect1d(np.unique(np.asarray(pop.golds[j], np.int64)), u["cand"]))
        log(f"{name}: {len(units)} questions' inputs ({time.time() - t:.0f}s)")
        cos = A.cosines(emb, np.stack([prep.qemb[j] for j in samp]), [u["cand"] for u in units], dev)
        arr = B.pack_units(units, golds, ids, samp)
        lp, pimax, drms = {}, None, None
        for k, sp in (("G", "V"), ("G0", "V0"), ("NEAR", "V"), ("GB", "V")):
            pk = B.Pack(arr, tables[sp], kap[sp], sp)
            lp[k], pm, dr = B.score_pack(torch, dev, models[k], pk)
            if k == "G":
                pimax, drms = pm, dr
            del pk
        nQ, nO, nN = len(units), len(ORDERS), len(GRID)
        res = {rd: {"T": np.zeros(nQ, np.int64), "T_ring": np.zeros((nQ, RINGS), np.int64),
                    "T_seen": np.zeros((nQ, 2), np.int64),
                    "found": np.zeros((nQ, nO, nN), np.int32), "found_ring": np.zeros((nQ, nO, nN, RINGS), np.int32),
                    "found_seen": np.zeros((nQ, nO, nN, 2), np.int32),
                    "f_not_h2": np.zeros((nQ, nN), np.int32), "g_not_h2": np.zeros((nQ, nN), np.int32)}
               for rd in READS}
        sizes = np.zeros((nQ, 3), np.int64)
        for qi, u in enumerate(units):
            cand = u["cand"]
            r_1, r_2 = A.rings_of(adj, u["seeds"], mark, buf)
            o = B.all_orders(cand, cos[qi], r_1, r_2, {k: lp[k][qi] for k in ("G", "G0", "NEAR")})
            o["GB"] = B.order_by(lp["GB"][qi], cand)
            o["FB"] = B.fused(o["GB"], o["H2"], cand)
            top = np.asarray(r1["top5"][qi], np.int64) if r1 is not None else np.empty(0, np.int64)
            gl = golds[qi]
            T = {"low": gl[~np.isin(gl, top)], "pool": gl}
            sizes[qi] = (r_1.size, r_2.size, cand.size)
            for rd in READS:
                Tr = T[rd]
                rg = B.ring_class(Tr, r_1, r_2)
                sn = np.isin(Tr, seen)
                rr = res[rd]
                rr["T"][qi] = Tr.size
                for c in range(RINGS):
                    rr["T_ring"][qi, c] = int((rg == c + 1).sum())
                rr["T_seen"][qi] = (int(sn.sum()), int((~sn).sum()))
                for oi, k in enumerate(ORDERS):
                    rr["found"][qi, oi] = A.found_counts(o[k], Tr, GRID)
                    for c in range(RINGS):
                        rr["found_ring"][qi, oi, :, c] = A.found_counts(o[k], Tr[rg == c + 1], GRID)
                    for c, m in enumerate((sn, ~sn)):
                        rr["found_seen"][qi, oi, :, c] = A.found_counts(o[k], Tr[m], GRID)
                for key, k in (("f_not_h2", "F"), ("g_not_h2", "G")):
                    rr[key][qi] = [int(np.isin(np.setdiff1d(o[k][:N], o["H2"][:N]), Tr).sum()) for N in GRID]
            if (qi + 1) % 500 == 0:
                log(f"{name}: {qi + 1}/{nQ} questions ordered ({time.time() - t:.0f}s)")
        holder.update(res=res, sizes=sizes, ids=ids, samp=samp, Q=Q, checked=checked, sample_sha=sample_sha,
                      carve_sha=pop.digest, pimax=pimax, drms=drms, seconds=time.time() - t)

    E.run_carves(op, name, {"s1eval"}, (), threads, sink)
    h = holder
    if limit:
        log(f"{name} development run: {len(h['ids'])} questions, every order, both counts and the seen/new split ran; "
            f"today's pools and seeds checked against read1 on {h['checked']}"
            + ("" if r1 is not None else " (read1 is not on disk)")
            + f"; final pools: {B.pools_said(pinfo['s1eval'])}; {h['seconds']:.0f}s, gpu peak "
            f"{A.gpu_peak(torch, dev) / 2 ** 30:.2f} GB, peak {A.peak_rss() / 2 ** 30:.1f} GB; nothing written")
        return 0
    arrs = {f"{rd}|{k}": v for rd, d in h["res"].items() for k, v in d.items()}
    arrs.update(sizes=h["sizes"], pos=np.asarray(h["samp"], np.int64), ids=np.asarray([str(q) for q in h["ids"]]),
                pi_max=h["pimax"], delta_rms=h["drms"])
    p = out / f"read_{name}.npz"
    save_npz(p, **arrs)
    rec = {"declared_in": DECLARED, "stage": "read", "dataset": name, "carve": "s1eval", "carve_questions": h["Q"],
           "questions": len(h["ids"]), "carve_sha256": h["carve_sha"], "sample_sha256": h["sample_sha"],
           "checks": {"pools_and_seeds_equal_read1": h["checked"], "final_pools": "every check of P5b's section 5 held"},
           "pools": pinfo["s1eval"], "tables": tshas, "tables_made": made, "kappa": kap,
           "g_sha256": trec["g_sha256"], "train_record_sha256": sha_file(paths["train"]),
           "gb": gb_made, "gb_g_sha256": gb_sha, "p5b_train_record": P5B_TRAIN,
           "seen": {"from": seen_made, "golds": int(seen.size),
                    "p5b_base_record": P5B_BASE.get(name)},
           "read1_records": r1["records"], "orders": list(ORDERS), "reads": list(READS), "grid": list(GRID),
           "index_time": B.INDEX_TIME, "npz_sha256": sha_file(p), "script_sha256": script_sha256(),
           "p5b_script_sha256": P5B_SCRIPT, "p5a_script_sha256": A.script_sha256(),
           "freeze_RECORD_SHA256": op.freeze["RECORD_SHA256"], "device": str(dev),
           "gpu_peak_bytes": A.gpu_peak(torch, dev), "peak_rss_bytes": A.peak_rss(),
           "read_s": round(h["seconds"], 1), "seconds": round(time.time() - t0, 1), "utc": utc()}
    write_json(out / f"read_{name}.json", rec)
    log(f"{name}: outputs/p5b2/read_{name}.json ({rec['seconds']:.0f}s)")
    return 0


# ── the verdict (section 5 of the declared file) ─────────────────────────────


def labelled(fa, fb, t):
    """P5a's labels over the grid: (label, points), fa and fb counts (questions x N), t golds per question."""
    pts, signs = [], []
    for k, N in enumerate(GRID):
        d, lo, hi = A.ratio_boot(fa[:, k], fb[:, k], t)
        pts.append({"N": N, "delta": round(d, 4), "lo": round(lo, 4), "hi": round(hi, 4),
                    "counted": A.counted(d, lo, hi)})
        signs.append(pts[-1]["counted"])
    return A.label(signs), pts


def reproduces(D, name, p5b_out):
    """Questions whose GB and FB counts differ from P5b's filed G and F, per read (P5b's read checked by sha256)."""
    pr = json.loads((p5b_out / f"read_{name}.json").read_text(encoding="utf-8"))
    p = p5b_out / f"read_{name}.npz"
    if sha_file(p) != pr["npz_sha256"]:
        raise SystemExit(f"{p}: not P5b's read")
    ob = {o: k for k, o in enumerate(B.ORDERS)}
    o2 = {o: k for k, o in enumerate(ORDERS)}
    out = {}
    with np.load(p) as z:
        if not np.array_equal(z["ids"], D["ids"]):
            raise SystemExit(f"{name}: the questions are not P5b's")
        for rd in READS:
            a, b = D[f"{rd}|found"], z[f"{rd}|found"]
            out[rd] = {"GB_vs_G": int((a[:, o2["GB"]] != b[:, ob["G"]]).any(1).sum()),
                       "FB_vs_F": int((a[:, o2["FB"]] != b[:, ob["F"]]).any(1).sum()),
                       "H2": int((a[:, o2["H2"]] != b[:, ob["H2"]]).any(1).sum())}
    return out


def verdict_stage(out=OUT, p5b_out=P5B_OUT):
    paths = out_paths(out)
    trec = json.loads(paths["train"].read_text(encoding="utf-8"))
    for k, f in trec["fits"].items():
        for e in f["hist"]:
            vals = [e["criterion"], *e["held_out_all"].values(), *e["held_out_new"].values()]
            if e["epoch"] > 0:
                vals += [e["J"], *e["by_dataset"].values()]
            if not all(np.isfinite(v) for v in vals):
                raise SystemExit(f"{k}'s loss is not finite")
    recs, data = {}, {}
    for name in SIX:
        rp = out / f"read_{name}.json"
        recs[name] = json.loads(rp.read_text(encoding="utf-8"))
        p = out / f"read_{name}.npz"
        if sha_file(p) != recs[name]["npz_sha256"]:
            raise SystemExit(f"{p}: not its record's file")
        with np.load(p) as z:
            data[name] = {k: z[k] for k in z.files}
    for key in ("script_sha256", "g_sha256", "gb_g_sha256"):
        if len({r[key] for r in recs.values()}) != 1:
            raise SystemExit(f"the reads ran on different {key}")
    if next(iter(recs.values()))["g_sha256"] != trec["g_sha256"]:
        raise SystemExit("the reads' g is not the train record's")
    checks = {}
    for name, r in recs.items():
        pools = r["pools"]
        ok = (r["checks"]["pools_and_seeds_equal_read1"] == r["questions"] and r["tables_made"] == "filed"
              and isinstance(pools, dict) and pools.get("construction") == B.CONSTRUCTION
              and r["gb"] == "P5b's filed G")
        checks[name] = {"questions": r["questions"],
                        "pools_and_seeds_equal_read1": r["checks"]["pools_and_seeds_equal_read1"],
                        "tables": r["tables_made"], "gb": r["gb"],
                        "pools": pools.get("construction") if isinstance(pools, dict) else pools,
                        "differ_from_p5b": reproduces(data[name], name, p5b_out)}
        if not ok:
            raise SystemExit(f"{name}: the checks fail ({checks[name]})")
    oi = {o: k for k, o in enumerate(ORDERS)}
    calls, found, beside = {}, {}, {}
    for rd in READS:
        calls[rd], found[rd], beside[rd] = {}, {}, {}
        for name in SIX:
            D = data[name]
            T = D[f"{rd}|T"]
            keep = T > 0
            Fd = D[f"{rd}|found"][keep].astype(np.float64)
            t = T[keep]
            found[rd][name] = {"questions": int(keep.sum()), "golds": int(t.sum()),
                               **{o: [round(float(Fd[:, oi[o], k].sum() / max(1, t.sum())), 4)
                                      for k in range(len(GRID))] for o in ORDERS}}
            if not keep.any():
                calls[rd][name] = {"label": "NO_GOLDS"}
                continue
            c = {}
            for o in ORDERS:
                if o != "H2":
                    lab, pts = labelled(Fd[:, oi[o]], Fd[:, oi["H2"]], t)
                    c[o] = {"label": lab, "points": pts}
            calls[rd][name] = {"label": c["F"]["label"], **{f"{o}_against_H2": v for o, v in c.items()}}
            Tr = D[f"{rd}|T_ring"][keep]
            FR = D[f"{rd}|found_ring"][keep]
            TS = D[f"{rd}|T_seen"][keep]
            FS = D[f"{rd}|found_seen"][keep].astype(np.float64)
            split = {}
            for ci, cl in enumerate(CLASSES):
                kc = TS[:, ci] > 0
                tc = TS[kc, ci]
                s = {"golds": int(tc.sum()), "questions": int(kc.sum()),
                     **{o: [round(float(FS[kc, oi[o], k, ci].sum() / max(1, tc.sum())), 4)
                            for k in range(len(GRID))] for o in ORDERS}}
                if kc.any():
                    for o in ("F", "G", "GB", "FB"):
                        s[f"{o}_against_H2"] = labelled(FS[kc, oi[o], :, ci], FS[kc, oi["H2"], :, ci], tc)[0]
                split[cl] = s
            beside[rd][name] = {
                "F_not_H2": [round(float(D[f"{rd}|f_not_h2"][keep][:, k].sum() / t.sum()), 4) for k in range(len(GRID))],
                "G_not_H2": [round(float(D[f"{rd}|g_not_h2"][keep][:, k].sum() / t.sum()), 4) for k in range(len(GRID))],
                "by_ring": {f"ring_{c_ + 1 if c_ < 2 else '3+'}": {
                    "golds": int(Tr[:, c_].sum()),
                    **{o: [round(float(FR[:, oi[o], k, c_].sum() / max(1, Tr[:, c_].sum())), 4)
                           for k in range(len(GRID))] for o in ("F", "G", "H2", "GB", "FB")}} for c_ in range(RINGS)},
                "seen_and_new": split}
    labs = [calls["low"][n]["label"] for n in SIX if calls["low"][n]["label"] != "NO_GOLDS"]
    finds = "FINDS_MORE" if ("MORE" in labs and "FEWER" not in labs) else "DOES_NOT"
    call = "GOES_FURTHER" if finds == "FINDS_MORE" else "STOPS"
    kept = {k: f["kept_epoch"] for k, f in trec["fits"].items()}
    where = None
    if call == "STOPS":
        where = {}
        for n in SIX:
            c = calls["low"][n]
            if c["label"] in ("NO_GOLDS", "MORE"):
                where[n] = "-"
                continue
            parts = [w for o, w in (("F0", "V"), ("FN", "the fit")) if c[f"{o}_against_H2"]["label"] == "MORE"]
            if parts:
                where[n] = " and ".join(parts)
            elif kept["G"] == 0:
                where[n] = "g at its start: no epoch predicted new golds better"
            else:
                where[n] = "what g reaches in this space"
    fits = {}
    for k, f in trec["fits"].items():
        h = f["hist"]
        e = h[f["kept_epoch"]]
        fits[k] = {"space": f["space"], "kept_epoch": f["kept_epoch"], "beta": round(f["beta"], 4),
                   "criterion_by_epoch": [round(x["criterion"], 4) for x in h],
                   "held_out_new_kept": e["held_out_new"], "held_out_all_kept": e["held_out_all"],
                   "train_loss_kept": e["by_dataset"], "J_first_last": [h[1]["J"], h[-1]["J"]] if len(h) > 1 else None}
    g_use = {n: {"pi_max_mean": round(float(data[n]["pi_max"].mean()), 4),
                 "delta_rms_mean": round(float(data[n]["delta_rms"].mean()), 4)} for n in SIX}
    rec = {"declared_in": DECLARED, "call": call, "finds": finds, "calls": calls, "found": found, "beside": beside,
           "where_it_fell_short": where, "fits": fits, "split": trec["split"], "g_use": g_use, "checks": checks,
           "grid": list(GRID), "boot": A.BOOT, "tie": A.TIE, "reads": {n: sha_file(out / f"read_{n}.json") for n in SIX},
           "train_record": sha_file(paths["train"]), "script_sha256": script_sha256(), "utc": utc()}
    write_json(paths["verdict"], rec)
    paths["verdict_md"].write_text(report_md(rec), encoding="utf-8", newline="\n")
    log(f"P5b2: {call}; low {finds} (" + ", ".join(f"{n} {calls['low'][n]['label']}" for n in SIX)
        + f"); kept epochs G {kept['G']}, G0 {kept['G0']}")
    return 0


def report_md(rec):
    grid = rec["grid"]
    others = [o for o in ORDERS if o != "H2"]
    L = [f"# P5b2 verdict: {rec['call']}", "",
         "Kept epochs: " + ", ".join(f"{k} {f['kept_epoch']}" for k, f in rec["fits"].items()), "",
         f"## The low read (the call: F against H2): {rec['finds']}", "",
         "| dataset | " + " | ".join(others) + " | " + " | ".join(f"F N={N}" for N in grid) + " |",
         "| --- | " + " | ".join("---" for _ in others) + " | " + " | ".join("---:" for _ in grid) + " |"]
    for n, c in rec["calls"]["low"].items():
        if c["label"] == "NO_GOLDS":
            L.append(f"| {n} | NO_GOLDS |" + " |" * (len(others) - 1 + len(grid)))
            continue
        labs = " | ".join(c[f"{o}_against_H2"]["label"] for o in others)
        L.append(f"| {n} | {labs} | " + " | ".join(f"{p['delta']:+.4f} [{p['lo']:+.4f}, {p['hi']:+.4f}]"
                                                   for p in c["F_against_H2"]["points"]) + " |")
    L += ["", "Seen and new golds, the low read (found(N) at N = 10 and 25; labels against H2 over the grid):", "",
          "| dataset | class | golds | F | H2 | G | GB | FB | F label | G label | GB label |",
          "| --- | --- | ---: | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for n, b in rec["beside"]["low"].items():
        for cl, s in b["seen_and_new"].items():
            if not s["golds"]:
                L.append(f"| {n} | {cl} | 0 | | | | | | | | |")
                continue
            v = " | ".join(f"{s[o][1]:.4f}, {s[o][2]:.4f}" for o in ("F", "H2", "G", "GB", "FB"))
            L.append(f"| {n} | {cl} | {s['golds']} | {v} | {s['F_against_H2']} | {s['G_against_H2']} | "
                     f"{s['GB_against_H2']} |")
    for rd in READS:
        L += ["", f"found(N), the {rd} read:", "", "| dataset | order | " + " | ".join(f"N={N}" for N in grid) + " |",
              "| --- | --- | " + " | ".join("---:" for _ in grid) + " |"]
        for n, f in rec["found"][rd].items():
            for o in ORDERS:
                L.append(f"| {n} | {o} | " + " | ".join(f"{v:.4f}" for v in f[o]) + " |")
    return "\n".join(L) + "\n"


# ── self-test ────────────────────────────────────────────────────────────────


def selftest():
    import torch
    A.full_fp32(torch)
    dev = torch.device("cpu")
    rng = np.random.default_rng(0)
    p5b_is_filed()
    # the split: seeded, disjoint, covering, round(0.2 n) held out
    for n in (1, 5, 104, 1974):
        tr, ho = split_of(n)
        assert ho.size == int(round(HOLD * n)) and np.array_equal(np.union1d(tr, ho), np.arange(n))
        assert not np.intersect1d(tr, ho).size
        tr2, ho2 = split_of(n)
        assert np.array_equal(tr, tr2) and np.array_equal(ho, ho2)
    # subset: equals packing the chosen questions directly
    N_ = 60
    spec = [(rng.normal(size=B.DIM + 1), np.sort(rng.choice(N_, rng.integers(1, 4), replace=False)),
             np.arange(10 + 3 * q, 14 + 3 * q), np.asarray([11 + 3 * q])) for q in range(12)]
    full = B.toy_arr(rng, spec, N_)
    full["deg"] = rng.integers(0, 6, full["deg"].shape).astype(np.float32)
    for sp in SPACES:
        full[f"m_{sp}"] = rng.normal(size=full[f"m_{sp}"].shape).astype(np.float32)
    full["rank_d"] = rng.integers(1, 50, full["rank_d"].shape).astype(np.int32)
    pick = np.asarray([7, 2, 9])
    sub = subset(full, pick)
    for i, q in enumerate(pick.tolist()):
        s0, s1 = full["seed_ptr"][q], full["seed_ptr"][q + 1]
        c0, c1 = full["cand_ptr"][q], full["cand_ptr"][q + 1]
        t0_, t1_ = sub["seed_ptr"][i], sub["seed_ptr"][i + 1]
        d0, d1 = sub["cand_ptr"][i], sub["cand_ptr"][i + 1]
        for k in ("seeds", "rank_d", "rank_s", "deg", "m_V", "m_V0"):
            assert np.array_equal(sub[k][t0_:t1_], full[k][s0:s1]), k
        for k in ("cand", "gold"):
            assert np.array_equal(sub[k][d0:d1], full[k][c0:c1]), k
        assert sub["ids"][i] == full["ids"][q] and np.array_equal(sub["x"][i], full["x"][q])
    assert set(subset(full, np.zeros(0, np.int64))) == KEYS and not subset(full, np.zeros(0, np.int64))["cand"].size
    # new golds: held-out golds no train question has
    toy = {"ids": np.asarray(["a", "b", "c"]), "pos": np.arange(3), "x": np.zeros((3, B.DIM + 1), np.float32),
           "seed_ptr": np.asarray([0, 1, 2, 3]), "seeds": np.asarray([0, 1, 2]),
           "rank_d": np.ones(3, np.int32), "rank_s": np.ones(3, np.int32), "deg": np.zeros((3, 3), np.float32),
           "m_V": np.zeros((3, B.DIM), np.float32), "m_V0": np.zeros((3, B.DIM), np.float32),
           "cand_ptr": np.asarray([0, 3, 6, 9]), "cand": np.asarray([10, 11, 12, 11, 13, 14, 10, 11, 15]),
           "gold": np.asarray([0, 1, 0, 1, 1, 0, 0, 0, 1], bool)}
    a_tr, a_ho = subset(toy, [0, 2]), subset(toy, [1])
    new = np.asarray(a_ho["gold"], bool) & ~np.isin(a_ho["cand"], golds_of(a_tr))
    assert golds_of(a_tr).tolist() == [11, 15] and new.tolist() == [False, True, False]
    # the held-out losses equal a question-by-question brute force, over all golds and over new golds
    Ut = rng.normal(size=(N_ + 60, B.DIM)).astype(np.float32)
    pk = B.Pack(full, Ut, 0.05, "V")
    g = B.new_g(torch, dev)
    with torch.no_grad():
        g.l3.weight.normal_(0, 0.05)
        g.l3.bias.normal_(0, 0.05)
    newf = np.zeros(full["cand"].size, bool)
    newf[full["cand_ptr"][3] + 1] = True                              # question 3's gold is new
    la, ln = held_out(torch, dev, g, pk, newf)
    for i in range(pk.n):
        c0, c1 = int(pk.cp[i]), int(pk.cp[i + 1])
        bl = B.brute_log_p(torch, g, pk, i)
        assert np.isclose(la[i], -bl[pk.gold[c0:c1]].mean(), atol=1e-5)
        assert (np.isclose(ln[i], -bl[newf[c0:c1]].mean(), atol=1e-5) if i == 3 else np.isnan(ln[i]))
    crit, by_all, by_new = criterion({"a": (np.asarray([1.0, 3.0]), np.asarray([np.nan, 2.0])),
                                      "b": (np.asarray([5.0]), np.asarray([np.nan])),
                                      "c": (np.asarray([1.0]), np.asarray([4.0]))})
    assert crit == 3.0 and by_all == {"a": 2.0, "b": 5.0, "c": 1.0} and by_new == {"a": 2.0, "c": 4.0}
    # the stop: on golds at a question-chosen offset (P5b's toy), the kept epoch beats g's start on new golds and
    # g comes back at that epoch
    K, Qn = 3, 240
    R = rng.normal(size=(K, B.DIM))
    rows, spec2 = [], []
    for q in range(Qn):
        k = q % K
        s = rng.normal(size=(2, B.DIM))
        near_ = s[0] + rng.normal(0, 0.1, (5, B.DIM))
        gold = s[(q // K) % 2] + R[k] + rng.normal(0, 0.1, B.DIM)
        other = rng.normal(size=(10, B.DIM))
        b0 = len(rows)
        rows += list(s) + list(near_) + [gold] + list(other)
        xq = np.zeros(B.DIM + 1)
        xq[k] = 3.0
        xq[B.DIM] = 1.0
        spec2.append((xq + rng.normal(0, 0.05, B.DIM + 1) * (np.arange(B.DIM + 1) < B.DIM), np.arange(b0, b0 + 2),
                      np.arange(b0 + 2, b0 + 18), np.asarray([b0 + 7])))
    Ug = np.asarray(rows, np.float32)
    kap = B.kappa_of(Ug, 5000)
    a_tr, a_ho, new2, cnt = split_base(B.toy_arr(rng, spec2, Ug.shape[0]))
    assert cnt["held_out"] == 48 and cnt["held_out_new_golds"] == 48 == cnt["held_out_with_a_new_gold"]
    ptr_, pho = B.Pack(a_tr, Ug, kap, "V"), B.Pack(a_ho, Ug, kap, "V")
    gk, hist, kept = fit_stop(torch, dev, [ptr_], [pho], [new2], ["toy"], 40, logf=lambda *_: None)
    assert len(hist) == 41 and hist[0]["epoch"] == 0 and kept > 0
    assert hist[kept]["criterion"] == min(h["criterion"] for h in hist) < hist[0]["criterion"] * 0.5
    again = criterion({"toy": held_out(torch, dev, gk, pho, new2)})[0]
    assert np.isclose(again, hist[kept]["criterion"], rtol=1e-6), (again, hist[kept]["criterion"])
    # golds no question can predict (a fresh random gold per question): the stop keeps an early epoch, never a worse one
    spec3 = [(rng.normal(size=B.DIM + 1), np.asarray([3 * q]), np.arange(3 * q + 1, 3 * q + 3) % 300,
              np.asarray([(3 * q + 1 + (q % 2)) % 300])) for q in range(100)]
    U3 = rng.normal(size=(300, B.DIM)).astype(np.float32)
    t3, h3, n3, _c3 = split_base(B.toy_arr(rng, spec3, 300))
    _g3, hist3, kept3 = fit_stop(torch, dev, [B.Pack(t3, U3, 0.05, "V")], [B.Pack(h3, U3, 0.05, "V")], [n3], ["r"],
                                 6, logf=lambda *_: None)
    assert hist3[kept3]["criterion"] == min(h["criterion"] for h in hist3)
    verdict_selftest()
    log("p5b2 self-test passed")
    return 0


def verdict_selftest():
    rng = np.random.default_rng(1)
    with tempfile.TemporaryDirectory() as td:
        out, pb = Path(td) / "p5b2", Path(td) / "p5b"
        out.mkdir()
        pb.mkdir()
        paths = out_paths(out)

        def hist_(kept):
            h = [{"epoch": 0, "J": None, "by_dataset": None, "beta": 0.0, "held_out_all": {"metaqa": 3.0},
                  "held_out_new": {"metaqa": 3.0}, "criterion": 3.0, "seconds": 0.0},
                 {"epoch": 1, "J": 2.0, "by_dataset": {"metaqa": 2.0}, "beta": 0.1, "held_out_all": {"metaqa": 2.5},
                  "held_out_new": {"metaqa": 2.9 if kept else 3.1}, "criterion": 2.9 if kept else 3.1,
                  "seconds": 1.0}]
            return h

        def train_(kept):
            write_json(paths["train"], {"g_sha256": "g", "split": {"metaqa": {"train": 8, "held_out": 2}},
                                        "fits": {"G": {"space": "V", "hist": hist_(kept), "kept_epoch": kept,
                                                       "beta": 0.1},
                                                 "G0": {"space": "V0", "hist": hist_(kept), "kept_epoch": kept,
                                                        "beta": 0.1}}})

        def fake(name, up, down):
            nQ, nO, nN = 300, len(ORDERS), len(GRID)
            arr, arrb = {}, {}
            ob = {o: k for k, o in enumerate(B.ORDERS)}
            for rd in READS:
                T = rng.integers(1, 4, nQ)
                Fd = np.zeros((nQ, nO, nN), np.int32)
                for k in range(nN):
                    b_ = np.minimum(T, rng.integers(0, 2, nQ))
                    Fd[:, :, k] = b_[:, None]
                    if up:
                        Fd[:, 0, k] = T
                    if down:
                        Fd[:, 0, k] = 0
                TS = np.stack([T // 2, T - T // 2], 1)
                FS = np.zeros((nQ, nO, nN, 2), np.int32)
                FS[..., 0] = np.minimum(Fd, TS[:, None, None, 0])
                FS[..., 1] = Fd - FS[..., 0]
                arr.update({f"{rd}|T": T, f"{rd}|T_ring": np.stack([T, 0 * T, 0 * T], 1), f"{rd}|T_seen": TS,
                            f"{rd}|found": Fd, f"{rd}|found_ring": np.repeat(Fd[..., None], RINGS, 3) * 0,
                            f"{rd}|found_seen": FS,
                            f"{rd}|f_not_h2": np.zeros((nQ, nN), np.int32), f"{rd}|g_not_h2": np.zeros((nQ, nN), np.int32)})
                Fb = np.zeros((nQ, len(B.ORDERS), nN), np.int32)
                Fb[:, ob["G"]] = Fd[:, ORDERS.index("GB")]
                Fb[:, ob["F"]] = Fd[:, ORDERS.index("FB")]
                Fb[:, ob["H2"]] = Fd[:, ORDERS.index("H2")]
                arrb[f"{rd}|found"] = Fb
            ids = np.asarray([f"q{i}" for i in range(nQ)])
            arr.update(pi_max=np.full(nQ, 0.5), delta_rms=np.full(nQ, 1.0), ids=ids)
            p = out / f"read_{name}.npz"
            save_npz(p, **arr)
            write_json(out / f"read_{name}.json", {
                "questions": nQ, "checks": {"pools_and_seeds_equal_read1": nQ}, "tables_made": "filed",
                "pools": {"construction": B.CONSTRUCTION}, "npz_sha256": sha_file(p), "script_sha256": "s",
                "g_sha256": "g", "gb_g_sha256": "gb", "gb": "P5b's filed G"})
            q = pb / f"read_{name}.npz"
            save_npz(q, ids=ids, **arrb)
            write_json(pb / f"read_{name}.json", {"npz_sha256": sha_file(q)})

        train_(1)
        for nm in SIX:
            fake(nm, nm == "musique", False)
        verdict_stage(out, pb)
        v = json.loads(paths["verdict"].read_text(encoding="utf-8"))
        assert v["call"] == "GOES_FURTHER" and v["calls"]["low"]["musique"]["label"] == "MORE", v["finds"]
        assert v["calls"]["low"]["metaqa"]["label"] == "TIE"
        assert all(x == 0 for c in v["checks"].values() for r in c["differ_from_p5b"].values() for x in r.values())
        assert v["beside"]["low"]["musique"]["seen_and_new"]["new"]["F_against_H2"] in ("MORE", "TIE")
        fake("webqsp", False, True)
        verdict_stage(out, pb)
        v = json.loads(paths["verdict"].read_text(encoding="utf-8"))
        assert v["call"] == "STOPS" and v["calls"]["low"]["webqsp"]["label"] == "FEWER"
        assert v["where_it_fell_short"]["webqsp"] == "what g reaches in this space"
        assert v["where_it_fell_short"]["musique"] == "-"
        train_(0)
        verdict_stage(out, pb)
        v = json.loads(paths["verdict"].read_text(encoding="utf-8"))
        assert v["where_it_fell_short"]["webqsp"] == "g at its start: no epoch predicted new golds better"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", nargs="?", choices=("train", "read", "verdict"))
    ap.add_argument("dataset", nargs="?")
    ap.add_argument("--host", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.stage == "read" and a.dataset not in SIX:
        raise SystemExit(f"dataset must be one of {SIX}")
    device = a.device
    if a.stage in ("train", "read") and device is None:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if a.stage == "train":
        return train_stage(device, a.threads, a.host, a.limit, a.dataset)
    if a.stage == "read":
        return read_stage(a.dataset, device, a.threads, a.host, a.limit)
    if a.stage == "verdict":
        return verdict_stage()
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
