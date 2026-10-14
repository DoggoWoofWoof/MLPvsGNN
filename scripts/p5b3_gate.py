"""P5b3: a learned back-off gate between H2 and P5b2's g (docs/P5B3_BACKOFF_GATE.md). A diagnostic: no ranking model
trains or reads, and nothing is adopted.

    python scripts/p5b3_gate.py read DATASET [--host] [--device cuda] [--threads 2] [--limit Q]
    python scripts/p5b3_gate.py gate
    python scripts/p5b3_gate.py verdict
    python scripts/p5b3_gate.py --selftest

read (on the host): for a training dataset, P5b2's held-out fit questions (P5b's filed base, P5b2's split): each
candidate's rank in H2 and in G (P5b2's g), the gate's inputs, and which candidates are golds and new golds. Then, on
every dataset, the same on P5b2's s1eval sample, with its low and pool golds and which golds are seen.
gate (local): the gate's weights, fit on the five fit-side files' new golds (section 3).
verdict (local): FG, the gated order, against H2 on s1eval (section 5), after checking that the gate at 1/2, 1 and 0
gives P5b2's F, H2 and G on every question.
--limit: a development run of read. It writes nothing and logs no measure: only that every step ran, and the checks.
Where P5b2's g or P5a's node tables are not on disk, g at its start stands in; where P5b's base is not, this dataset's
base is built in memory (P5b's development path).
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

import p5b2_stop as C  # noqa: E402  (P5b2's filed script, imported unchanged)

B = C.B                                       # P5b's filed script, through P5b2
A = C.A                                       # P5a's filed script, through P5b
DECLARED = "docs/P5B3_BACKOFF_GATE.md"
OUT = ROOT / "outputs" / "p5b3"
P5B2_OUT = C.OUT
SIX, TRAIN5, READS, GRID, RINGS = B.SIX, B.TRAIN5, B.READS, B.GRID, B.RINGS
RANK_C = B.RANK_C                             # 60: P5b's fusion constant
ORDERS = ("FG", "F", "H2", "G", "FC")         # FG: the gate's order (the call); FC: one weight for every question
FEATURES = ("log_n", "ring1", "ring2", "log_seeds", "g_entropy", "g_ring1", "g_ring2", "pi_max", "delta_rms",
            "overlap25")
TOP = 25                                      # overlap25: the share of G's first TOP candidates in H2's first TOP
RIDGE = 1e-3                                  # on the inputs' weights, not on the bias
SD_FLOOR = 1e-6                               # an input whose spread is below this is not scaled
LOSS_TOL = 1e-5                               # the held-out loss against P5b2's train record at its kept epoch
X_TOL = 1e-4                                  # a held-out question's frame vector against P5b's base
G_TOL = 1e-5                                  # pi_max and delta_rms on s1eval against P5b2's read
# P5b2's files as filed (004c31c): its script, train record, g and six read records
P5B2_SCRIPT = "abbf05c4f54901c674a571bd8ce2a3b3f50294d156c647b37c242dc4c5ba53d8"
P5B2_TRAIN = "a9eb05a7f36098b1405d338f723c2d87c8759018918957ff92ffaf1433f45e07"
P5B2_G = "559323e5572cd8b2d62a26ae93a165247cd9401995c0822072d1802d25fa8d85"
P5B2_READ = {"metaqa": "5bcc0b0e4f117d61078cf6eee742b970bbb4c46cc0f26f11950020f93b1526fe",
             "squad": "2c42f0c5718701a8639add108b79bdebbfe60e9d599278477589524ee2dc77a0",
             "musique": "c52c296201da23339ff2728cb2d6d8b3dabdf55985b714107186092c02126b26",
             "hotpotqa": "90ae6b54cbebda6e17d5355e2ea652655f586fc38d245de2438af6417b5e9b89",
             "2wiki": "ccf53d5271a0f9c8e9e613aacedbfea15d994e0197a6eb10ef24d2005b5392f7",
             "webqsp": "bf036a085e902803f0783ecdf5595216865a587e93061aab645e1603c4d2577e"}
FIT_KEYS = {"ids", "pos", "ptr", "r_h2", "r_g", "ring", "gold", "new", "X", "la", "ln"}
EVAL_KEYS = {"ids", "pos", "ptr", "r_h2", "r_g", "ring", "pool_gold", "low_gold", "seen", "X"}
log, utc, sha_file, write_json, save_npz = A.log, A.utc, A.sha_file, A.write_json, A.save_npz


def script_sha256():
    return hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def p5b2_is_filed():
    C.p5b_is_filed()
    if C.script_sha256() != P5B2_SCRIPT:
        raise SystemExit("scripts/p5b2_stop.py is not P5b2's filed script")


def out_paths(out=OUT):
    return {"gate": out / "gate.json", "verdict": out / "verdict.json", "verdict_md": out / "verdict.md"}


def read_paths(name, out=OUT):
    return {"rec": out / f"read_{name}.json", "fit": out / f"fit_{name}.npz", "eval": out / f"eval_{name}.npz"}


# ── orders ───────────────────────────────────────────────────────────────────


def ranks_of(order, cand):
    """Each candidate's rank (1 is first) in order; cand sorted, order holding every candidate once."""
    if order.size != cand.size:
        raise SystemExit("an order must hold every candidate once")
    r = np.empty(cand.size, np.int64)
    r[np.searchsorted(cand, order)] = np.arange(1, order.size + 1)
    return r


def by_rank(r, cand):
    """The candidates in the order that gives them ranks r."""
    o = np.empty(cand.size, cand.dtype)
    o[np.asarray(r, np.int64) - 1] = cand
    return o


def gated(lam, ra, rb, cand, c=RANK_C):
    """The candidates (sorted) by lam / (c + ra) + (1 - lam) / (c + rb), highest first, ties by node id. At lam = 1/2
    this is P5b's fusion of the two orders, at 1 the order of ra and at 0 the order of rb."""
    ra = np.asarray(ra, np.int64)
    rb = np.asarray(rb, np.int64)
    lam = float(lam)
    s = lam / (c + ra) + (1.0 - lam) / (c + rb)
    return cand[np.lexsort((cand, -s))]


def gate_inputs(cand, ring, lp, pimax, drms, n_seeds, h2, go):
    """The gate's ten inputs for one question (section 3), from its candidates, their rings, g's log p and the two
    orders; nothing from golds."""
    n = int(cand.size)
    if n == 0:
        return np.zeros(len(FEATURES), np.float64)
    lp = np.asarray(lp, np.float64)
    p = np.exp(lp - lp.max())
    p /= p.sum()
    nz = p > 0
    ent = float(-(p[nz] * np.log(p[nz])).sum() / np.log(n)) if n > 1 else 0.0
    k = min(TOP, n)
    return np.asarray([np.log(n), float((ring == 1).mean()), float((ring == 2).mean()), float(np.log(max(1, n_seeds))),
                       ent, float(p[ring == 1].sum()), float(p[ring == 2].sum()), float(pimax), float(drms),
                       float(np.intersect1d(h2[:k], go[:k]).size / k)], np.float64)


def flat(rows, key, dtype):
    return np.concatenate([np.asarray(r[key]) for r in rows]).astype(dtype) if rows else np.zeros(0, dtype)


def small_ranks(r):
    if r.size and int(r.max()) > np.iinfo(np.int16).max:
        raise SystemExit("a pool is too large for 16-bit ranks")
    return r.astype(np.int16)


def ordered(cand, cc, r_1, r_2, lp, pimax, drms, n_seeds):
    """One question's ranks in H2 and G, its candidates' rings and the gate's inputs."""
    h2 = B.h2_low(cand, cc, r_1, r_2)
    go = B.order_by(lp, cand)
    ring = B.ring_class(cand, r_1, r_2)
    return {"r_h2": ranks_of(h2, cand), "r_g": ranks_of(go, cand), "ring": ring,
            "x": gate_inputs(cand, ring, lp, pimax, drms, n_seeds, h2, go)}


def packed(rows, extra):
    """A side's rows as flat arrays (ranks 16-bit, rings 8-bit), with its question ids and positions in extra."""
    return {**extra, "ptr": np.r_[0, np.cumsum([r["r_h2"].size for r in rows])].astype(np.int64),
            "r_h2": small_ranks(flat(rows, "r_h2", np.int64)), "r_g": small_ranks(flat(rows, "r_g", np.int64)),
            "ring": flat(rows, "ring", np.int8),
            "X": np.stack([r["x"] for r in rows]) if rows else np.zeros((0, len(FEATURES)))}


# ── read (on the host) ───────────────────────────────────────────────────────


def the_g(torch, dev, name, tables, made, limit):
    """P5b2's filed G, its train record and kappa on V, each checked; in a development run where P5b2's g or P5a's
    node tables are not on disk, g at its start."""
    kap = B.kappa_of(tables["V"])
    paths = C.out_paths(P5B2_OUT)
    rp = P5B2_OUT / f"read_{name}.json"
    if made == "filed" and paths["g"].exists() and paths["train"].exists() and rp.exists():
        if sha_file(paths["train"]) != P5B2_TRAIN or sha_file(rp) != P5B2_READ[name]:
            raise SystemExit("P5b2's train or read record is not the filed one")
        models, trec = B.load_g(torch, dev, paths)
        if trec["g_sha256"] != P5B2_G:
            raise SystemExit("g is not P5b2's")
        k_read = json.loads(rp.read_text(encoding="utf-8"))["kappa"]["V"]
        if k_read != kap or (name in TRAIN5 and trec["kappa"]["V"][name] != kap):
            raise SystemExit(f"{name}: kappa is not P5b2's")
        return models["G"], "P5b2's filed G", trec, kap
    if not limit:
        raise SystemExit("P5b2's g, its records or P5a's node tables are not on disk")
    return B.new_g(torch, dev).eval(), "development: g at its start stands in", None, kap


def fit_prep(name, arr, trec, g, tables, kap, torch, dev):
    """P5b2's held-out questions of a training dataset (its split of P5b's base) and g on them: the held-out loss,
    checked against P5b2's train record at its kept epoch where filed; log p, pi_max and delta_rms."""
    _a_tr, a_ho, new, counts = C.split_base(arr)
    if trec is not None and counts != trec["split"][name]:
        raise SystemExit(f"{name}: the split is not P5b2's")
    pack = B.Pack(a_ho, tables["V"], kap, "V")
    la, ln = C.held_out(torch, dev, g, pack, new)
    held = {"all": float(la.mean()) if la.size else None,
            "new": float(np.nanmean(ln)) if np.isfinite(ln).any() else None}
    filed = None
    if trec is not None:
        f = trec["fits"]["G"]
        e = f["hist"][f["kept_epoch"]]
        if e["epoch"] != f["kept_epoch"]:
            raise SystemExit("P5b2's train record's history is not by epoch")
        filed = {"all": e["held_out_all"].get(name), "new": e["held_out_new"].get(name)}
        if any(held[k] is None or filed[k] is None or abs(held[k] - filed[k]) > LOSS_TOL for k in held):
            raise SystemExit(f"{name}: the held-out loss {held} is not P5b2's at its kept epoch {filed}")
    lp, pimax, drms = B.score_pack(torch, dev, g, pack)
    return {"a_ho": a_ho, "new": new, "counts": counts, "la": la, "ln": ln, "held": held, "filed": filed, "lp": lp,
            "pimax": pimax, "drms": drms}


def fit_units(name, fp, pop, prep, bs, inc, finals, frame):
    """The held-out questions on today's carve: each one's seeds and candidates (its final pool less its seeds) must be
    the base's, and its frame vector within X_TOL of the base's. Returns the question vectors."""
    a_ho = fp["a_ho"]
    qv, worst = [], 0.0
    for i in range(int(a_ho["ids"].size)):
        j = int(a_ho["pos"][i])
        if str(pop.ids[j]) != str(a_ho["ids"][i]):
            raise SystemExit(f"{name}: held-out question {i} is not the base's")
        pool = finals.pool(pop.ids[j], bs[j], inc[j]) if finals else np.unique(np.asarray(inc[j], np.int64))
        seeds = np.unique(np.asarray(prep.seeds[j], np.int64))
        s0, s1 = int(a_ho["seed_ptr"][i]), int(a_ho["seed_ptr"][i + 1])
        c0, c1 = int(a_ho["cand_ptr"][i]), int(a_ho["cand_ptr"][i + 1])
        if not (np.array_equal(seeds, a_ho["seeds"][s0:s1]) and np.array_equal(np.setdiff1d(pool, seeds),
                                                                                 a_ho["cand"][c0:c1])):
            raise SystemExit(f"{name}: {pop.ids[j]}: the seeds or candidates are not the base's")
        x = A.qframe(prep.qemb[j], *frame)
        worst = max(worst, float(np.abs(x.astype(np.float64) - a_ho["x"][i]).max()))
        qv.append(np.array(prep.qemb[j]))
    if worst > X_TOL:
        raise SystemExit(f"{name}: a held-out question's frame vector is {worst:.2e} from the base's")
    return {"qv": qv, "worst": worst, "Q": len(pop.ids), "carve_sha": pop.digest}


def eval_units(name, pop, prep, bs, inc, finals, r1, limit, tables, adj, frame):
    """P5b2's s1eval sample on its final pools, as P5b2's read builds it (checked against read1 where on disk)."""
    Q = len(pop.ids)
    samp = A.sample_of(Q)
    sample_sha = A.ids_sha([pop.ids[j] for j in samp])
    if limit:
        samp = samp[:limit]
    ids = [pop.ids[j] for j in samp]
    if r1 is not None and (r1["sample_sha256"] != sample_sha or r1["ids"][:len(ids)] != ids
                           or r1["pos"][:len(ids)] != samp.tolist()):
        raise SystemExit(f"{name}: the sample is not read1's")
    cache = {}
    units, golds, checked = [], [], 0
    t = time.time()
    for qi, j in enumerate(samp.tolist()):
        pool = finals.pool(pop.ids[j], bs[j], inc[j]) if finals else np.unique(np.asarray(inc[j], np.int64))
        if r1 is not None:
            if not (np.array_equal(np.unique(np.asarray(inc[j], np.int64)), np.unique(r1["pools"][qi]))
                    and np.array_equal(np.unique(np.asarray(prep.seeds[j], np.int64)), np.unique(r1["seeds"][qi]))):
                raise SystemExit(f"{name}: {pop.ids[j]}: today's pool or seeds are not read1's")
            checked += 1
        u = B.unit(j, pop, prep, pool, tables, adj, frame, cache)
        units.append(u)
        golds.append(np.intersect1d(np.unique(np.asarray(pop.golds[j], np.int64)), u["cand"]))
    log(f"{name}: {len(units)} s1eval questions' inputs ({time.time() - t:.0f}s)")
    return {"units": units, "golds": golds, "ids": ids, "samp": samp, "Q": Q, "checked": checked,
            "sample_sha": sample_sha, "carve_sha": pop.digest, "qv": np.stack([prep.qemb[j] for j in samp])}


def fit_rows(fp, fu, emb, adj, mark, buf, dev):
    a_ho, cp, sp_ = fp["a_ho"], fp["a_ho"]["cand_ptr"], fp["a_ho"]["seed_ptr"]
    nq = int(a_ho["ids"].size)
    cands = [a_ho["cand"][cp[i]:cp[i + 1]] for i in range(nq)]
    cos = A.cosines(emb, np.stack(fu["qv"]), cands, dev) if nq else []
    rows = []
    for i in range(nq):
        seeds = a_ho["seeds"][sp_[i]:sp_[i + 1]]
        r_1, r_2 = A.rings_of(adj, seeds, mark, buf)
        rows.append(ordered(cands[i], cos[i], r_1, r_2, fp["lp"][i], fp["pimax"][i], fp["drms"][i], seeds.size))
    return packed(rows, {"ids": a_ho["ids"], "pos": a_ho["pos"], "gold": np.asarray(a_ho["gold"], bool),
                         "new": np.asarray(fp["new"], bool), "la": fp["la"], "ln": fp["ln"]})


def eval_rows(name, eu, emb, adj, mark, buf, tables, kap, g, torch, dev, r1, seen):
    units, golds = eu["units"], eu["golds"]
    t = time.time()
    cos = A.cosines(emb, eu["qv"], [u["cand"] for u in units], dev)
    pk = B.Pack(B.pack_units(units, golds, eu["ids"], eu["samp"]), tables["V"], kap, "V")
    lp, pimax, drms = B.score_pack(torch, dev, g, pk)
    del pk
    rows, low, pool_gold, sn = [], [], [], []
    for qi, u in enumerate(units):
        cand = u["cand"]
        r_1, r_2 = A.rings_of(adj, u["seeds"], mark, buf)
        rows.append(ordered(cand, cos[qi], r_1, r_2, lp[qi], pimax[qi], drms[qi], u["seeds"].size))
        top = np.asarray(r1["top5"][qi], np.int64) if r1 is not None else np.empty(0, np.int64)
        gl = golds[qi]
        pool_gold.append(np.isin(cand, gl))
        low.append(np.isin(cand, gl[~np.isin(gl, top)]))
        sn.append(np.isin(cand, seen))
        if (qi + 1) % 500 == 0:
            log(f"{name}: {qi + 1}/{len(units)} s1eval questions ordered ({time.time() - t:.0f}s)")
    cat = (lambda xs: np.concatenate(xs).astype(bool) if xs else np.zeros(0, bool))
    return packed(rows, {"ids": np.asarray([str(q) for q in eu["ids"]]), "pos": np.asarray(eu["samp"], np.int64),
                         "pool_gold": cat(pool_gold), "low_gold": cat(low), "seen": cat(sn)})


def read_stage(name, device, threads, host, limit=None, out=OUT):
    import torch
    torch.set_num_threads(threads)
    A.full_fp32(torch)
    p5b2_is_filed()
    E, _F, S4 = A.modules()
    op = A.opened(host)
    dev = torch.device(device)
    t0 = time.time()
    ds, emb = A.dense_docs(op, name)
    n = emb.n_rows
    frame, tables, tshas, made = B.tables_for(name, emb, limit)
    _stores, adj = A.family_stores(S4, ds, name, A.csr_dir(op))
    if any(ip.size - 1 != n for ip, _c in adj):
        raise SystemExit(f"{name}: a family's store is not the corpus's size")
    g, g_made, trec, kap = the_g(torch, dev, name, tables, made, limit)
    arr, brec, base_made, base_sha = None, None, None, None
    if name in TRAIN5:
        if C.has_filed_base(name):
            arr, brec, base_sha = C.filed_base(name)
            if made == "filed" and brec["tables"] != tshas:
                raise SystemExit(f"{name}: P5b's base ran on other tables")
            base_made = "P5b's filed base"
        elif limit:
            arr = B.base_data(name, threads, host, limit)[0]
            base_made = "development: this dataset's base built in memory"
        else:
            raise SystemExit(f"{name}: P5b's base is not on disk")
    fp = fit_prep(name, arr, trec, g, tables, kap, torch, dev) if arr is not None else None
    seen = C.golds_of(arr) if arr is not None else np.zeros(0, np.int64)
    del arr
    fit_carve = A.fit_carve(name) if fp is not None else None
    r1 = B.read1_or_none(name, limit)
    get, pinfo = B.finals_for(name, limit)
    mark = np.zeros(n, np.int8)
    buf = np.empty(n, np.int64)
    log(f"{name}: {n} nodes (tables {made}; g {g_made}; base {base_made}; {device}); setup {time.time() - t0:.0f}s")
    holder = {}

    def sink(name_, carve, pop, prep, U, BQ, bs, inc, linked, union, constant, families, construction):
        finals = get(carve)
        if carve == "s1eval":
            holder["eval"] = eval_units(name, pop, prep, bs, inc, finals, r1, limit, tables, adj, frame)
        else:
            holder["fit"] = fit_units(name, fp, pop, prep, bs, inc, finals, frame)

    E.run_carves(op, name, {"s1eval"} | ({fit_carve} if fit_carve else set()), (), threads, sink)
    if "eval" not in holder or (fp is not None and "fit" not in holder):
        raise SystemExit(f"{name}: a carve was not read")
    fit_rec = fit_arrs = None
    if fp is not None:
        today = pinfo[fit_carve] if isinstance(pinfo[fit_carve], dict) else {}
        if brec is not None and any(brec["pools"].get(k) != today.get(k) for k in ("file", "sha256")):
            raise SystemExit(f"{name}/{fit_carve}: today's pools file is not the one P5b's base ran on")
        t = time.time()
        fit_arrs = fit_rows(fp, holder["fit"], emb, adj, mark, buf, dev)
        fu = holder.pop("fit")
        fit_rec = {"carve": fit_carve, "carve_questions": fu["Q"], "carve_sha256": fu["carve_sha"],
                   "split": fp["counts"], "held_out_loss": fp["held"], "held_out_loss_p5b2": fp["filed"],
                   "frame_vector_max_abs_diff": fu["worst"], "pools": pinfo[fit_carve],
                   "seconds": round(time.time() - t, 1)}
        log(f"{name}: the fit side, {fp['counts']['held_out']} held-out questions ({fit_rec['seconds']:.0f}s)")
    t = time.time()
    eu = holder.pop("eval")
    ev_arrs = eval_rows(name, eu, emb, adj, mark, buf, tables, kap, g, torch, dev, r1, seen)
    ev_rec = {"carve": "s1eval", "carve_questions": eu["Q"], "questions": len(eu["ids"]),
              "carve_sha256": eu["carve_sha"], "sample_sha256": eu["sample_sha"],
              "checks": {"pools_and_seeds_equal_read1": eu["checked"]}, "pools": pinfo["s1eval"],
              "read1_records": r1["records"] if r1 is not None else None, "seconds": round(time.time() - t, 1)}
    del eu
    if limit:
        fit_said = "no fit side (never fits); "
        if fit_rec:
            fin = all(np.isfinite(v) for v in fit_rec["held_out_loss"].values() if v is not None)
            fit_said = (f"the fit side ran on {fit_rec['split']['held_out']} held-out questions (seeds and candidates "
                        f"the base's, frame vectors within {X_TOL:g}; held-out loss finite: {fin}); ")
        log(f"{name} development run: " + fit_said
            + f"the s1eval side ran on {ev_rec['questions']} questions, pools and seeds checked against read1 on "
              f"{ev_rec['checks']['pools_and_seeds_equal_read1']}; final pools: {B.pools_said(ev_rec['pools'])}; "
              f"{time.time() - t0:.0f}s, gpu peak {A.gpu_peak(torch, dev) / 2 ** 30:.2f} GB, "
              f"peak {A.peak_rss() / 2 ** 30:.1f} GB; nothing written")
        return 0
    paths = read_paths(name, out)
    if fit_arrs is not None:
        save_npz(paths["fit"], **fit_arrs)
        fit_rec["npz_sha256"] = sha_file(paths["fit"])
    save_npz(paths["eval"], **ev_arrs)
    ev_rec["npz_sha256"] = sha_file(paths["eval"])
    rec = {"declared_in": DECLARED, "stage": "read", "dataset": name, "g": g_made, "g_sha256": P5B2_G,
           "p5b2_train_record": P5B2_TRAIN, "p5b2_read_record": P5B2_READ[name], "kappa_V": kap,
           "tables": tshas, "tables_made": made, "base": base_made, "p5b_base_record": base_sha,
           "seen": {"golds": int(seen.size), "from": base_made or "never fits: every gold is new"},
           "fit_side": fit_rec, "eval_side": ev_rec, "features": list(FEATURES), "rank_c": RANK_C, "top": TOP,
           "index_time": B.INDEX_TIME, "script_sha256": script_sha256(), "p5b2_script_sha256": P5B2_SCRIPT,
           "p5b_script_sha256": C.P5B_SCRIPT, "p5a_script_sha256": A.script_sha256(),
           "freeze_RECORD_SHA256": op.freeze["RECORD_SHA256"], "device": str(dev),
           "gpu_peak_bytes": A.gpu_peak(torch, dev), "peak_rss_bytes": A.peak_rss(),
           "seconds": round(time.time() - t0, 1), "utc": utc()}
    write_json(paths["rec"], rec)
    log(f"{name}: outputs/p5b3/read_{name}.json ({rec['seconds']:.0f}s)")
    return 0


# ── the gate (section 3) ─────────────────────────────────────────────────────


def sig(x):
    return 0.5 * (1.0 + np.tanh(0.5 * np.asarray(x, np.float64)))


def per_question(lam, ia, ib, ptr):
    """Each question's mean, over its new golds, of log(lam ia + (1 - lam) ib)."""
    cnt = np.diff(ptr)
    reps = np.repeat(np.arange(cnt.size), cnt)
    lam = np.broadcast_to(np.asarray(lam, np.float64), cnt.shape)
    s = lam[reps] * ia + (1.0 - lam[reps]) * ib
    return np.bincount(reps, weights=np.log(s), minlength=cnt.size) / cnt, s, reps, cnt


def objective(theta, Z, qw, ia, ib, ptr, ridge=RIDGE):
    """-J and its gradient: J = the qw-weighted sum over questions of the mean, over each question's new golds, of
    log(lam ia + (1 - lam) ib), less ridge |w|^2; lam = sig(b + z . w); ia and ib a gold's 1 / (c + rank) in H2 and G."""
    b, w = float(theta[0]), np.asarray(theta[1:], np.float64)
    lam = sig(b + Z @ w)
    per, s, reps, cnt = per_question(lam, ia, ib, ptr)
    J = float(qw @ per - ridge * (w @ w))
    dl = np.bincount(reps, weights=(ia - ib) / s, minlength=cnt.size) / cnt
    de = qw * dl * lam * (1.0 - lam)
    return -J, -np.r_[de.sum(), Z.T @ de - 2.0 * ridge * w]


def fit_gate(X, qw, ia, ib, ptr, inputs=True):
    """The gate's bias and weights by L-BFGS from zero (lam = 1/2: P5b's F), with the inputs scaled by their
    qw-weighted mean and spread; inputs=False fits the bias alone (one weight for every question)."""
    from scipy.optimize import minimize
    X = np.asarray(X, np.float64)
    tot = qw.sum()
    mu = qw @ X / tot
    sd = np.sqrt(qw @ (X - mu) ** 2 / tot)
    sd = np.where(sd > SD_FLOOR, sd, 1.0)
    Z = (X - mu) / sd if inputs else np.zeros((X.shape[0], 0))
    res = minimize(objective, np.zeros(Z.shape[1] + 1), args=(Z, qw, ia, ib, ptr), jac=True, method="L-BFGS-B",
                   options={"maxiter": 5000, "maxfun": 10000, "gtol": 1e-8, "ftol": 1e-12})
    if not np.all(np.isfinite(res.x)):
        raise SystemExit("the gate's fit is not finite")
    return res.x, mu, sd, res


def lam_of(gate, X):
    """A gate's weight on H2 for each question (rows of X)."""
    th = np.asarray(gate["theta"], np.float64)
    if th.size == 1:
        return np.full(np.asarray(X).shape[0], float(sig(th[0])))
    Z = (np.asarray(X, np.float64) - np.asarray(gate["mu"])) / np.asarray(gate["sd"])
    return sig(th[0] + Z @ th[1:])


def fit_data(D):
    """A fit-side file's questions with a new gold: their inputs, each new gold's 1 / (c + rank) in H2 and in G, and
    pointers."""
    ptr = D["ptr"]
    keep, ia, ib, cnt = [], [], [], []
    for q in range(ptr.size - 1):
        c0, c1 = int(ptr[q]), int(ptr[q + 1])
        m = D["new"][c0:c1]
        if not m.any():
            continue
        keep.append(q)
        ia.append(1.0 / (RANK_C + D["r_h2"][c0:c1][m].astype(np.float64)))
        ib.append(1.0 / (RANK_C + D["r_g"][c0:c1][m].astype(np.float64)))
        cnt.append(int(m.sum()))
    keep = np.asarray(keep, np.int64)
    return (D["X"][keep] if keep.size else np.zeros((0, len(FEATURES))), np.concatenate(ia) if ia else np.zeros(0),
            np.concatenate(ib) if ib else np.zeros(0), np.r_[0, np.cumsum(cnt)].astype(np.int64))


def load_reads(out, names, side):
    recs, data = {}, {}
    for n in names:
        p = read_paths(n, out)
        recs[n] = json.loads(p["rec"].read_text(encoding="utf-8"))
        r = recs[n][f"{side}_side"]
        if sha_file(p[side]) != r["npz_sha256"]:
            raise SystemExit(f"{p[side]}: not its record's file")
        with np.load(p[side]) as z:
            data[n] = {k: z[k] for k in z.files}
        if set(data[n]) != (FIT_KEYS if side == "fit" else EVAL_KEYS):
            raise SystemExit(f"{p[side]}: keys {sorted(data[n])}")
    return recs, data


def reads_agree(recs, pins=True):
    for key in ("script_sha256", "g_sha256", "g", "tables_made"):
        if len({str(r[key]) for r in recs.values()}) != 1:
            raise SystemExit(f"the reads ran on different {key}")
    r = next(iter(recs.values()))
    if pins and (r["g"] != "P5b2's filed G" or r["tables_made"] != "filed" or r["g_sha256"] != P5B2_G):
        raise SystemExit("the reads did not run on P5b2's filed g and P5a's filed tables")


def gate_stage(out=OUT, pins=True):
    recs, data = load_reads(out, TRAIN5, "fit")
    reads_agree(recs, pins)
    for n, r in recs.items():
        if pins and (r["base"] != "P5b's filed base" or r["fit_side"]["held_out_loss_p5b2"] is None):
            raise SystemExit(f"{n}: the fit side did not run on P5b's filed base and P5b2's train record")
    parts = {n: fit_data(data[n]) for n in TRAIN5}
    names = [n for n in TRAIN5 if parts[n][0].shape[0]]
    if not names:
        raise SystemExit("no held-out question has a new gold")
    X = np.concatenate([parts[n][0] for n in names])
    ia = np.concatenate([parts[n][1] for n in names])
    ib = np.concatenate([parts[n][2] for n in names])
    ptr = np.r_[0, np.cumsum(np.concatenate([np.diff(parts[n][3]) for n in names]))].astype(np.int64)
    dsi = np.concatenate([np.full(parts[n][0].shape[0], k) for k, n in enumerate(names)])
    qw = 1.0 / (len(names) * np.bincount(dsi)[dsi].astype(np.float64))
    theta, mu, sd, res = fit_gate(X, qw, ia, ib, ptr)
    theta_c, _mu, _sd, res_c = fit_gate(X, qw, ia, ib, ptr, inputs=False)
    lam_fit = sig(theta[0] + ((X - mu) / sd) @ theta[1:])

    def J_at(lam):
        per = per_question(lam, ia, ib, ptr)[0]
        return {"all": round(float(qw @ per), 6),
                **{n: round(float(per[dsi == k].mean()), 6) for k, n in enumerate(names)}}

    rec = {"declared_in": DECLARED, "stage": "gate", "features": list(FEATURES), "rank_c": RANK_C, "ridge": RIDGE,
           "datasets": names, "questions": {n: int((dsi == k).sum()) for k, n in enumerate(names)},
           "new_golds": {n: int(np.diff(parts[n][3]).sum()) for n in names},
           "gate": {"theta": theta.tolist(), "mu": mu.tolist(), "sd": sd.tolist()},
           "weights": dict(zip(("bias",) + FEATURES, [round(float(v), 6) for v in theta])),
           "constant": {"theta": theta_c.tolist(), "lam": round(float(sig(theta_c[0])), 6)},
           "fit": {"converged": bool(res.success), "message": str(res.message), "iterations": int(res.nit),
                   "constant_converged": bool(res_c.success)},
           "J": {"gate": J_at(lam_fit), "constant": J_at(float(sig(theta_c[0]))), "F (1/2)": J_at(0.5),
                 "H2 (1)": J_at(1.0), "G (0)": J_at(0.0)},
           "lam_on_fit": {n: {"mean": round(float(lam_fit[dsi == k].mean()), 4),
                              "quantiles_10_50_90": [round(float(v), 4)
                                                     for v in np.quantile(lam_fit[dsi == k], (0.1, 0.5, 0.9))]}
                          for k, n in enumerate(names)},
           "reads": {n: sha_file(read_paths(n, out)["rec"]) for n in TRAIN5}, "script_sha256": script_sha256(),
           "utc": utc()}
    write_json(out_paths(out)["gate"], rec)
    log(f"P5b3 gate: fit on {X.shape[0]} questions ({', '.join(f'{n} {v}' for n, v in rec['questions'].items())}); "
        f"converged {res.success}")
    return 0


# ── the verdict (section 5) ──────────────────────────────────────────────────


def eval_counts(D, lam_q, lam_c):
    """P5b2's counts on every question for each order of ORDERS, from the stored ranks (candidate positions stand in
    for node ids: the candidates are sorted, so ties break the same way)."""
    ptr = D["ptr"]
    nQ, nO, nN = ptr.size - 1, len(ORDERS), len(GRID)
    res = {rd: {"T": np.zeros(nQ, np.int64), "T_ring": np.zeros((nQ, RINGS), np.int64),
                "T_seen": np.zeros((nQ, 2), np.int64), "found": np.zeros((nQ, nO, nN), np.int32),
                "found_ring": np.zeros((nQ, nO, nN, RINGS), np.int32),
                "found_seen": np.zeros((nQ, nO, nN, 2), np.int32)} for rd in READS}
    for q in range(nQ):
        c0, c1 = int(ptr[q]), int(ptr[q + 1])
        idx = np.arange(c1 - c0, dtype=np.int64)
        ra, rb = D["r_h2"][c0:c1].astype(np.int64), D["r_g"][c0:c1].astype(np.int64)
        o = {"FG": gated(lam_q[q], ra, rb, idx), "F": gated(0.5, ra, rb, idx), "H2": by_rank(ra, idx),
             "G": by_rank(rb, idx), "FC": gated(lam_c, ra, rb, idx)}
        ring, seen = D["ring"][c0:c1], D["seen"][c0:c1]
        for rd, flag in (("low", D["low_gold"][c0:c1]), ("pool", D["pool_gold"][c0:c1])):
            Tr, rg, sn = idx[flag], ring[flag], seen[flag]
            rr = res[rd]
            rr["T"][q] = Tr.size
            for c in range(RINGS):
                rr["T_ring"][q, c] = int((rg == c + 1).sum())
            rr["T_seen"][q] = (int(sn.sum()), int((~sn).sum()))
            for oi, k in enumerate(ORDERS):
                rr["found"][q, oi] = A.found_counts(o[k], Tr, GRID)
                for c in range(RINGS):
                    rr["found_ring"][q, oi, :, c] = A.found_counts(o[k], Tr[rg == c + 1], GRID)
                for c, m in enumerate((sn, ~sn)):
                    rr["found_seen"][q, oi, :, c] = A.found_counts(o[k], Tr[m], GRID)
    return res


def against_p5b2(res, D, name, p5b2_out, pins=True):
    """Questions whose golds or F, H2 or G counts differ from P5b2's filed read, per read, and the largest difference
    of the stored g inputs pi_max and delta_rms from P5b2's."""
    rp = p5b2_out / f"read_{name}.json"
    if pins and sha_file(rp) != P5B2_READ[name]:
        raise SystemExit(f"{rp}: not P5b2's filed read record")
    pr = json.loads(rp.read_text(encoding="utf-8"))
    p = p5b2_out / f"read_{name}.npz"
    if sha_file(p) != pr["npz_sha256"]:
        raise SystemExit(f"{p}: not P5b2's read")
    o2 = {o: k for k, o in enumerate(C.ORDERS)}
    o3 = {o: k for k, o in enumerate(ORDERS)}
    out = {}
    with np.load(p) as z:
        if not np.array_equal(z["ids"], D["ids"]) or not np.array_equal(z["pos"], D["pos"]):
            raise SystemExit(f"{name}: the questions are not P5b2's")
        if not np.array_equal(z["sizes"][:, 2], np.diff(D["ptr"])):
            raise SystemExit(f"{name}: the candidates are not P5b2's")
        for rd in READS:
            a, b = res[rd], {k: z[f"{rd}|{k}"] for k in ("T", "T_ring", "T_seen", "found", "found_ring", "found_seen")}
            d = {"golds": int(((a["T"] != b["T"]) | (a["T_ring"] != b["T_ring"]).any(1)
                               | (a["T_seen"] != b["T_seen"]).any(1)).sum())}
            for o in ("F", "H2", "G"):
                d[o] = int(((a["found"][:, o3[o]] != b["found"][:, o2[o]]).any(1)
                            | (a["found_ring"][:, o3[o]] != b["found_ring"][:, o2[o]]).any((1, 2))
                            | (a["found_seen"][:, o3[o]] != b["found_seen"][:, o2[o]]).any((1, 2))).sum())
            out[rd] = d
        xi = {f: k for k, f in enumerate(FEATURES)}
        out["g_inputs_max_abs_diff"] = float(max(np.abs(D["X"][:, xi["pi_max"]] - z["pi_max"]).max(initial=0.0),
                                                 np.abs(D["X"][:, xi["delta_rms"]] - z["delta_rms"]).max(initial=0.0)))
    return out


def verdict_stage(out=OUT, p5b2_out=P5B2_OUT, pins=True):
    paths = out_paths(out)
    grec = json.loads(paths["gate"].read_text(encoding="utf-8"))
    recs, data = load_reads(out, SIX, "eval")
    reads_agree(recs, pins)
    for n in TRAIN5:
        if grec["reads"][n] != sha_file(read_paths(n, out)["rec"]):
            raise SystemExit(f"{n}: the gate was fit on another read")
    checks, res, lam = {}, {}, {}
    lam_c = float(sig(grec["constant"]["theta"][0]))
    for n in SIX:
        r = recs[n]["eval_side"]
        if pins and (r["checks"]["pools_and_seeds_equal_read1"] != r["questions"] or not isinstance(r["pools"], dict)
                     or r["pools"].get("construction") != B.CONSTRUCTION):
            raise SystemExit(f"{n}: the s1eval side's checks fail")
        lam[n] = lam_of(grec["gate"], data[n]["X"])
        res[n] = eval_counts(data[n], lam[n], lam_c)
        diff = against_p5b2(res[n], data[n], n, p5b2_out, pins)
        checks[n] = {"questions": r["questions"],
                     "pools_and_seeds_equal_read1": r["checks"]["pools_and_seeds_equal_read1"],
                     "differ_from_p5b2": diff}
        if any(v for rd in READS for v in diff[rd].values()) or diff["g_inputs_max_abs_diff"] > G_TOL:
            raise SystemExit(f"{n}: the gate at 1/2, 1 and 0 does not give P5b2's F, H2 and G ({diff})")
    oi = {o: k for k, o in enumerate(ORDERS)}
    calls, found, beside = {}, {}, {}
    for rd in READS:
        calls[rd], found[rd], beside[rd] = {}, {}, {}
        for n in SIX:
            R = res[n][rd]
            keep = R["T"] > 0
            Fd = R["found"][keep].astype(np.float64)
            t = R["T"][keep]
            found[rd][n] = {"questions": int(keep.sum()), "golds": int(t.sum()),
                            **{o: [round(float(Fd[:, oi[o], k].sum() / max(1, t.sum())), 4) for k in range(len(GRID))]
                               for o in ORDERS}}
            if not keep.any():
                calls[rd][n] = {"label": "NO_GOLDS"}
                continue
            c = {}
            for o, base in (("FG", "H2"), ("FG", "F"), ("F", "H2"), ("G", "H2"), ("FC", "H2")):
                lab, pts = C.labelled(Fd[:, oi[o]], Fd[:, oi[base]], t)
                c[f"{o}_against_{base}"] = {"label": lab, "points": pts}
            calls[rd][n] = {"label": c["FG_against_H2"]["label"], **c}
            TS, FS = R["T_seen"][keep], R["found_seen"][keep].astype(np.float64)
            TR, FR = R["T_ring"][keep], R["found_ring"][keep].astype(np.float64)
            split = {}
            for ci, cl in enumerate(C.CLASSES):
                kc = TS[:, ci] > 0
                tc = TS[kc, ci]
                s = {"golds": int(tc.sum()), "questions": int(kc.sum()),
                     **{o: [round(float(FS[kc, oi[o], k, ci].sum() / max(1, tc.sum())), 4) for k in range(len(GRID))]
                        for o in ORDERS}}
                if kc.any():
                    s["FG_against_H2"] = C.labelled(FS[kc, oi["FG"], :, ci], FS[kc, oi["H2"], :, ci], tc)[0]
                split[cl] = s
            lq = lam[n][keep]
            beside[rd][n] = {
                "lam": {"mean": round(float(lq.mean()), 4),
                        "quantiles_10_50_90": [round(float(v), 4) for v in np.quantile(lq, (0.1, 0.5, 0.9))]},
                "by_ring": {("ring_1", "ring_2", "ring_3+")[c_]: {
                    "golds": int(TR[:, c_].sum()),
                    **{o: [round(float(FR[:, oi[o], k, c_].sum() / max(1, TR[:, c_].sum())), 4)
                           for k in range(len(GRID))] for o in ("FG", "F", "H2", "G")}} for c_ in range(RINGS)},
                "seen_and_new": split}
    labs = [calls["low"][n]["label"] for n in SIX if calls["low"][n]["label"] != "NO_GOLDS"]
    finds = "FINDS_MORE" if ("MORE" in labs and "FEWER" not in labs) else "DOES_NOT"
    call = "GOES_FURTHER" if finds == "FINDS_MORE" else "STOPS"
    where = None
    if call == "STOPS":
        where = {}
        for n in SIX:
            c = calls["low"][n]
            if c["label"] in ("NO_GOLDS", "MORE", "TIE"):
                where[n] = "-"
                continue
            where[n] = (f"{c['label']} at a mean weight on H2 of {beside['low'][n]['lam']['mean']:.2f}; G alone "
                        f"{c['G_against_H2']['label']}, one weight for all {c['FC_against_H2']['label']}")
        if "MORE" not in labs:
            where["all"] = "MORE nowhere: the gate kept none of g's gains"
    rec = {"declared_in": DECLARED, "call": call, "finds": finds, "calls": calls, "found": found, "beside": beside,
           "where_it_fell_short": where, "gate": {k: grec[k] for k in ("weights", "constant", "J", "lam_on_fit", "fit")},
           "checks": checks, "grid": list(GRID), "boot": A.BOOT, "tie": A.TIE,
           "reads": {n: sha_file(read_paths(n, out)["rec"]) for n in SIX}, "gate_record": sha_file(paths["gate"]),
           "script_sha256": script_sha256(), "utc": utc()}
    write_json(paths["verdict"], rec)
    paths["verdict_md"].write_text(report_md(rec), encoding="utf-8", newline="\n")
    log(f"P5b3: {call}; low {finds} (" + ", ".join(f"{n} {calls['low'][n]['label']}" for n in SIX) + ")")
    return 0


def report_md(rec):
    grid = rec["grid"]
    others = ("FG_against_F", "F_against_H2", "G_against_H2", "FC_against_H2")
    L = [f"# P5b3 verdict: {rec['call']}", "", f"## The low read (the call: FG against H2): {rec['finds']}", "",
         "| dataset | label | mean weight on H2 | " + " | ".join(o.replace("_against_", " vs ") for o in others)
         + " | " + " | ".join(f"FG - H2, N={N}" for N in grid) + " |",
         "| --- | --- | ---: | " + " | ".join("---" for _ in others) + " | " + " | ".join("---:" for _ in grid) + " |"]
    for n, c in rec["calls"]["low"].items():
        if c["label"] == "NO_GOLDS":
            L.append(f"| {n} | NO_GOLDS |" + " |" * (1 + len(others) + len(grid)))
            continue
        m = rec["beside"]["low"][n]["lam"]["mean"]
        L.append(f"| {n} | {c['label']} | {m:.2f} | " + " | ".join(c[o]["label"] for o in others) + " | "
                 + " | ".join(f"{p['delta']:+.4f} [{p['lo']:+.4f}, {p['hi']:+.4f}]"
                              for p in c["FG_against_H2"]["points"]) + " |")
    L += ["", "The gate's weights (inputs scaled on the fit questions): "
          + ", ".join(f"{k} {v:+.3f}" for k, v in rec["gate"]["weights"].items()), "",
          f"One weight for every question: {rec['gate']['constant']['lam']:.3f} on H2.", ""]
    for rd in READS:
        L += [f"found(N), the {rd} read:", "", "| dataset | order | " + " | ".join(f"N={N}" for N in grid) + " |",
              "| --- | --- | " + " | ".join("---:" for _ in grid) + " |"]
        for n, f in rec["found"][rd].items():
            for o in ORDERS:
                L.append(f"| {n} | {o} | " + " | ".join(f"{v:.4f}" for v in f[o]) + " |")
        L.append("")
    return "\n".join(L) + "\n"


# ── self-test ────────────────────────────────────────────────────────────────


def made_up_question(rng, n, n_gold, h2_good):
    """A made-up question's ranks in H2 and in G: its golds first in one order (H2 where h2_good) and last in the
    other."""
    gold = np.zeros(n, bool)
    gold[rng.choice(n, n_gold, replace=False)] = True
    k = int(gold.sum())
    good, bad = np.empty(n, np.int64), np.empty(n, np.int64)
    good[gold], good[~gold] = np.arange(1, k + 1), rng.permutation(np.arange(k + 1, n + 1))
    bad[gold], bad[~gold] = np.arange(n - k + 1, n + 1), rng.permutation(np.arange(1, n - k + 1))
    return (good, bad, gold) if h2_good else (bad, good, gold)


def made_up_eval(rng, Q):
    """A made-up s1eval file and, per question, its candidates, two orders, golds, rings and seen flags."""
    rows, expect = [], []
    for _q in range(Q):
        n = int(rng.integers(8, 120))
        cand = np.sort(rng.choice(1_000_000, n, replace=False)).astype(np.int64)
        a, b = rng.permutation(cand), rng.permutation(cand)
        if _q % 4 == 0:
            b = a[::-1].copy()                                       # mirrored ranks: every pair ties under F
        poolg = np.sort(rng.choice(cand, int(rng.integers(0, 4)), replace=False))
        low = poolg[rng.random(poolg.size) < 0.7]
        ring = rng.integers(1, 4, n).astype(np.int8)
        seen = rng.random(n) < 0.3
        rows.append({"r_h2": ranks_of(a, cand), "r_g": ranks_of(b, cand), "ring": ring, "x": rng.normal(size=10),
                     "pool_gold": np.isin(cand, poolg), "low_gold": np.isin(cand, low), "seen": seen})
        expect.append((cand, a, b, low, poolg, ring, seen))
    D = packed(rows, {"ids": np.asarray([f"q{i}" for i in range(Q)]), "pos": np.arange(Q, dtype=np.int64)})
    for k in ("pool_gold", "low_gold", "seen"):
        D[k] = flat(rows, k, bool)
    return D, expect


def selftest():
    rng = np.random.default_rng(0)
    # 1. the gated order: 1/2 is P5b's fusion, 1 is H2's order and 0 G's, ties included; ranks round-trip; positions
    #    stand in for node ids
    for trial in range(300):
        n = int(rng.integers(1, 60))
        cand = np.sort(rng.choice(10_000, n, replace=False)).astype(np.int64)
        a, b = rng.permutation(cand), rng.permutation(cand)
        if trial % 3 == 0:
            b = a[::-1].copy()
        ra, rb = ranks_of(a, cand), ranks_of(b, cand)
        assert np.array_equal(by_rank(ra, cand), a) and np.array_equal(by_rank(rb, cand), b)
        assert np.array_equal(gated(0.5, ra, rb, cand), B.fused(a, b, cand))
        assert np.array_equal(gated(0.5, ra, rb, cand), B.fused(b, a, cand))
        assert np.array_equal(gated(1.0, ra, rb, cand), a) and np.array_equal(gated(0.0, ra, rb, cand), b)
        idx = np.arange(n)
        for lam in (0.5, 1.0, 0.0, float(rng.random())):
            assert np.array_equal(cand[gated(lam, ra, rb, idx)], gated(lam, ra, rb, cand))
    # 2. the gate's inputs, by hand
    cand = np.asarray([3, 5, 9, 12], np.int64)
    ring = np.asarray([1, 2, 3, 1])
    lp = np.log(np.asarray([0.5, 0.25, 0.125, 0.125]))
    h2, go = np.asarray([12, 3, 5, 9]), np.asarray([3, 5, 9, 12])
    x = gate_inputs(cand, ring, lp + 7.0, 0.8, 2.5, 2, h2, go)
    ent = -(0.5 * np.log(0.5) + 0.25 * np.log(0.25) + 2 * 0.125 * np.log(0.125)) / np.log(4)
    assert np.allclose(x, [np.log(4), 0.5, 0.25, np.log(2), ent, 0.625, 0.25, 0.8, 2.5, 1.0]), x
    assert np.array_equal(gate_inputs(cand[:0], ring[:0], lp[:0], 0, 0, 1, h2[:0], go[:0]), np.zeros(len(FEATURES)))
    assert gate_inputs(cand[:1], ring[:1], lp[:1], 1.0, 0.0, 1, h2[2:3] * 0 + 3, go[:1])[4] == 0.0
    # 3. the objective's gradient against finite differences; at lam = 1 J is the mean log 1 / (c + H2's rank)
    Q = 40
    cnt = rng.integers(1, 4, Q)
    ptr = np.r_[0, np.cumsum(cnt)]
    ia = 1.0 / (RANK_C + rng.integers(1, 500, ptr[-1]))
    ib = 1.0 / (RANK_C + rng.integers(1, 500, ptr[-1]))
    Z = rng.normal(size=(Q, 3))
    qw = rng.random(Q)
    qw /= qw.sum()
    th = rng.normal(size=4)
    g0 = objective(th, Z, qw, ia, ib, ptr)[1]
    for k in range(4):
        e = np.zeros(4)
        e[k] = 1e-6
        fd = (objective(th + e, Z, qw, ia, ib, ptr)[0] - objective(th - e, Z, qw, ia, ib, ptr)[0]) / 2e-6
        assert abs(fd - g0[k]) < 1e-6 * max(1.0, abs(fd)), (k, fd, g0[k])
    gc = objective(th[:1], Z[:, :0], qw, ia, ib, ptr)[1]
    fd = (objective(th[:1] + 1e-6, Z[:, :0], qw, ia, ib, ptr)[0]
          - objective(th[:1] - 1e-6, Z[:, :0], qw, ia, ib, ptr)[0]) / 2e-6
    assert gc.size == 1 and abs(fd - gc[0]) < 1e-6 * max(1.0, abs(fd))
    reps = np.repeat(np.arange(Q), cnt)
    want = qw @ (np.bincount(reps, weights=np.log(ia), minlength=Q) / cnt)
    assert abs(-objective(np.r_[60.0, 0, 0, 0], Z, qw, ia, ib, ptr, ridge=0.0)[0] - want) < 1e-9
    # 4. the fit finds a planted gate: golds lead in H2 where the first input is positive, in G otherwise
    rows, X, gw = [], [], []
    for d in range(3):
        for _q in range(150 + 50 * d):
            f = rng.normal(size=len(FEATURES))
            rows.append(made_up_question(rng, int(rng.integers(30, 200)), int(rng.integers(1, 4)), f[0] > 0))
            X.append(f)
            gw.append(d)
    X, gw = np.asarray(X), np.asarray(gw)
    ia = np.concatenate([1.0 / (RANK_C + r[0][r[2]]) for r in rows])
    ib = np.concatenate([1.0 / (RANK_C + r[1][r[2]]) for r in rows])
    ptr = np.r_[0, np.cumsum([int(r[2].sum()) for r in rows])]
    qw = 1.0 / (3 * np.bincount(gw)[gw].astype(np.float64))
    theta, mu, sd, _res = fit_gate(X, qw, ia, ib, ptr)
    lam = lam_of({"theta": theta, "mu": mu, "sd": sd}, X)
    assert lam[X[:, 0] > 1].min() > 0.9 and lam[X[:, 0] < -1].max() < 0.1, (lam[X[:, 0] > 1].min(),
                                                                            lam[X[:, 0] < -1].max())
    Zs = (X - mu) / sd
    Jfit = -objective(theta, Zs, qw, ia, ib, ptr)[0]
    for b0 in (0.0, 60.0, -60.0):
        assert Jfit > -objective(np.r_[b0, np.zeros(len(FEATURES))], Zs, qw, ia, ib, ptr)[0]
    tc = fit_gate(X, qw, ia, ib, ptr, inputs=False)[0]
    assert tc.size == 1 and np.allclose(lam_of({"theta": tc}, X), sig(tc[0]))
    # 5. counts from stored ranks equal counts along the explicit orders (node ids), by ring and seen and new
    D, expect = made_up_eval(rng, 25)
    lam_q = rng.random(25)
    lam_q[:3] = (0.5, 1.0, 0.0)
    res_ = eval_counts(D, lam_q, 0.3)
    for q, (cand, a, b, low, poolg, ring, seen) in enumerate(expect):
        ra, rb = ranks_of(a, cand), ranks_of(b, cand)
        o = {"FG": gated(lam_q[q], ra, rb, cand), "F": B.fused(b, a, cand), "H2": a, "G": b,
             "FC": gated(0.3, ra, rb, cand)}
        for rd, T in (("low", low), ("pool", poolg)):
            rg, sn = ring[np.searchsorted(cand, T)], seen[np.searchsorted(cand, T)]
            assert res_[rd]["T"][q] == T.size and res_[rd]["T_seen"][q].tolist() == [int(sn.sum()), int((~sn).sum())]
            for oi, k in enumerate(ORDERS):
                assert res_[rd]["found"][q, oi].tolist() == A.found_counts(o[k], T, GRID)
                for c in range(RINGS):
                    assert res_[rd]["found_ring"][q, oi, :, c].tolist() == A.found_counts(o[k], T[rg == c + 1], GRID)
                for c, m in enumerate((sn, ~sn)):
                    assert res_[rd]["found_seen"][q, oi, :, c].tolist() == A.found_counts(o[k], T[m], GRID)
    verdict_selftest(rng)
    log("p5b3 self-test passed")
    return 0


def verdict_selftest(rng):
    """gate and verdict on made-up reads. On 2wiki and metaqa half the questions have their golds first in G and last
    in H2, marked by a positive first input; every other question has its golds first in H2, G at chance, and a
    negative first input. The gate must learn the input: FG finds MORE than H2 on 2wiki and metaqa and ties elsewhere,
    so the call GOES_FURTHER. Then a changed P5b2 count is refused."""
    with tempfile.TemporaryDirectory() as td:
        out, p2 = Path(td) / "p5b3", Path(td) / "p5b2"
        out.mkdir()
        p2.mkdir()
        for n in SIX:
            rec = {"declared_in": DECLARED, "stage": "read", "dataset": n, "g": "made up", "g_sha256": "x",
                   "tables_made": "made up", "base": "made up", "script_sha256": "x"}
            for side, Q in (("fit", 120), ("eval", 300)):
                if side == "fit" and n not in TRAIN5:
                    continue
                rows = []
                for q in range(Q):
                    f = rng.normal(size=len(FEATURES))
                    g_good = n in ("2wiki", "metaqa") and q % 2 == 0
                    f[0] = (1.0 if g_good else -1.0) * (0.5 + abs(f[0]))
                    size = int(rng.integers(40, 150))
                    r_h2, r_g, gold = made_up_question(rng, size, int(rng.integers(1, 3)), not g_good)
                    if not g_good:
                        r_g = rng.permutation(size) + 1
                    row = {"r_h2": r_h2, "r_g": r_g, "ring": rng.integers(1, 4, size).astype(np.int8), "x": f}
                    if side == "fit":
                        row.update(gold=gold, new=gold & (rng.random(size) < 0.8))
                    else:
                        row.update(pool_gold=gold, low_gold=gold, seen=np.zeros(size, bool))
                    rows.append(row)
                keys = ("gold", "new") if side == "fit" else ("pool_gold", "low_gold", "seen")
                arrs = packed(rows, {"ids": np.asarray([f"{n}{i}" for i in range(Q)]),
                                     "pos": np.arange(Q, dtype=np.int64), **{k: flat(rows, k, bool) for k in keys}})
                if side == "fit":
                    arrs.update(la=np.zeros(Q), ln=np.zeros(Q))
                p = read_paths(n, out)[side]
                save_npz(p, **arrs)
                rec[f"{side}_side"] = {"npz_sha256": sha_file(p), "questions": Q,
                                       "checks": {"pools_and_seeds_equal_read1": Q}}
                D = arrs
            write_json(read_paths(n, out)["rec"], rec)
            # P5b2's read of the same questions: its F, H2 and G as P5b2 counts them
            res = eval_counts(D, np.full(D["ptr"].size - 1, 0.5), 0.5)
            z = {"sizes": np.stack([np.zeros(D["ptr"].size - 1, np.int64)] * 2 + [np.diff(D["ptr"])], 1),
                 "ids": D["ids"], "pos": D["pos"], "pi_max": D["X"][:, FEATURES.index("pi_max")],
                 "delta_rms": D["X"][:, FEATURES.index("delta_rms")]}
            for rd in READS:
                for k in ("T", "T_ring", "T_seen"):
                    z[f"{rd}|{k}"] = res[rd][k]
                for k in ("found", "found_ring", "found_seen"):
                    a = np.zeros((res[rd][k].shape[0], len(C.ORDERS)) + res[rd][k].shape[2:], np.int32)
                    for o in ("F", "H2", "G"):
                        a[:, C.ORDERS.index(o)] = res[rd][k][:, ORDERS.index(o)]
                    z[f"{rd}|{k}"] = a
            save_npz(p2 / f"read_{n}.npz", **z)
            write_json(p2 / f"read_{n}.json", {"npz_sha256": sha_file(p2 / f"read_{n}.npz")})
        assert gate_stage(out, pins=False) == 0
        assert verdict_stage(out, p2, pins=False) == 0
        v = json.loads(out_paths(out)["verdict"].read_text(encoding="utf-8"))
        g = json.loads(out_paths(out)["gate"].read_text(encoding="utf-8"))
        labs = {n: v["calls"]["low"][n]["label"] for n in SIX}
        assert labs["2wiki"] == "MORE" and labs["metaqa"] == "MORE", labs
        assert all(labs[n] == "TIE" for n in ("squad", "musique", "hotpotqa", "webqsp")), labs
        assert v["call"] == "GOES_FURTHER" and g["weights"]["log_n"] < 0.0, (v["call"], g["weights"])
        assert g["J"]["gate"]["all"] > max(g["J"][k]["all"] for k in ("F (1/2)", "H2 (1)", "G (0)", "constant"))
        assert v["calls"]["low"]["musique"]["F_against_H2"]["label"] in ("FEWER", "MIXED"), v["calls"]["low"]["musique"]
        assert (out_paths(out)["verdict_md"]).read_text(encoding="utf-8").startswith("# P5b3 verdict: GOES_FURTHER")
        # a changed P5b2 count is refused
        with np.load(p2 / "read_squad.npz") as zz:
            zd = {k: zz[k] for k in zz.files}
        zd["low|found"] = zd["low|found"].copy()
        zd["low|found"][0, C.ORDERS.index("H2"), -1] += 1
        save_npz(p2 / "read_squad.npz", **zd)
        write_json(p2 / "read_squad.json", {"npz_sha256": sha_file(p2 / "read_squad.npz")})
        try:
            verdict_stage(out, p2, pins=False)
        except SystemExit as e:
            assert "does not give P5b2's" in str(e), e
        else:
            raise AssertionError("a changed P5b2 count was not refused")


# ── main ─────────────────────────────────────────────────────────────────────


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", nargs="?", choices=("read", "gate", "verdict"))
    ap.add_argument("dataset", nargs="?")
    ap.add_argument("--host", action="store_true")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.stage == "read":
        if a.dataset not in SIX:
            raise SystemExit(f"read needs DATASET, one of {SIX}")
        return read_stage(a.dataset, a.device, a.threads, a.host, a.limit)
    if a.stage == "gate":
        return gate_stage()
    if a.stage == "verdict":
        return verdict_stage()
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
