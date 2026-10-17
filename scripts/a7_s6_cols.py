"""A7 (docs/A7_S6_SCORER_INPUT.md): S6's scorer, refit per screen split and fold, as a fixed input to both of the lean
pair's models. This script builds the columns; outputs/mp_unified/za7s6.py trains and reads the models on them.

    python scripts/a7_s6_cols.py sbuild DATASET CARVE [--transform T] [--host]
    python scripts/a7_s6_cols.py sfit SPLIT FOLD [--host]
    python scripts/a7_s6_cols.py cols DATASET CARVE [--host]
    python scripts/a7_s6_cols.py --selftest

The scorer is S6's fg-gr with chainscore21's length offset (lb): chainscore19's joint model at T = 0 (no EM round),

    s(v) = f(v) + gamma log(eps + sum_c p0(c) m_c(v)),    p0 = softmax over the row's chains of g(c) - log C_level,length

f an MLP over v's 13 node features (its question cosine, first-stage rank, degree, seed distances, walk masses), g an
MLP over the chain's pair features, m_c(v) chain c's walk mass at v. s(v) is v's own: nothing is normalised over the
question's rows and no row reads another row's score (p0 is a softmax over chains, from chain features). It is S6's
best MLP-f zero-shot form (chainscore20's K5: em-gr, the EM-coupled form, is below fg-gr on webqsp zero-shot).

sbuild  chainscore19's build of one carve of P1c2's look (outputs/p1c2/look), S6's settings:
            metaqa      hops 3, levels 2,4; on its fit carve also the four transforms (al4, sp4: rows even / odd,
                        hops 2, levels 2,4,16, regimes 1,0.05,0; mg3: hops 3, level 2, six regimes; rf: rows even,
                        hops 3, levels 2,4, regimes 1,0.05,0)
            webqsp      hops 2, levels 4,16,64,256
            passages    chainscore21's typing (256 pseudo-relations: spherical k-means on the endpoint projections of
                        the carve's structural triples), hops 2, levels 4,16,64
        regimes 1,0.25,0.05,0.01,0.002,0 on a fit carve and 1 on s1eval (the read is regime 1, the whole carve).
sfit    the scorer of one split and fold, on the split's four training datasets' fit builds (metaqa's five): the fold's
        questions (parity of sha256('dataset:question id')) less 20% held out (P5b2's split_of, seed 0); each example's
        dataset uniform among the four (chainscore21's draw), then chainscore19's draw within it; Adam 1e-3, weight
        decay 1e-4, 12 epochs of 5,960 examples in batches of 64, one thread, deterministic; inputs standardised by
        metaqa's id fit build; the epoch (1 to 12) whose dataset-balanced held-out objective (node loss + chain loss,
        regime 1) is lowest, ties to the earlier.
cols    four columns per cache row of every part of P1c2's cache, for each split that reads the carve, each question's
        from the scorer of the fold that never saw it (the other fold's):
            f       f(v)
            lm      log(eps + sum_c p0(c) m_c(v))
            has     1 when sum_c p0(c) m_c(v) > 0
            seed    1 on the question's bucket-0 seeds
        stored float16. An s1eval carve is built inside the item (under outputs/a7/cache/tmp) and its build removed.
"""
import os
import sys

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"                 # one CPU per item; the k-means and the builds' levels depend on the count
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MPU = ROOT / "outputs" / "mp_unified"
if str(MPU) not in sys.path:
    sys.path.insert(0, str(MPU))

import chainpop17 as CP  # noqa: E402
import chainscore18 as CS  # noqa: E402
import chainscore19 as C19  # noqa: E402
import chainscore21 as C21  # noqa: E402
import reltype11 as RT  # noqa: E402
import lean_cache as LC  # noqa: E402

DECLARED = "docs/A7_S6_SCORER_INPUT.md"
OUT = ROOT / "outputs" / "a7"
BUILDS = OUT / "cache"
FITS = OUT / "fits"
COLS = OUT / "cols"
CACHE = ROOT / "outputs" / "p1c2" / "cache"
LOOK = ROOT / "outputs" / "p1c2" / "look"
REL = ROOT / "outputs" / "m3b" / "relations"
SIX = ("metaqa", "webqsp", "squad", "musique", "hotpotqa", "2wiki")
KB = ("metaqa", "webqsp")
SPLITS = {"L-musique": ("metaqa", "squad", "hotpotqa", "2wiki"), "L-hotpotqa": ("metaqa", "squad", "musique", "2wiki")}
HELD_OUT = {"L-musique": "musique", "L-hotpotqa": "hotpotqa"}
FIT_CARVE = {"musique": "s1fit"}
FOLDS = 2
HOLD, SPLIT_SEED = 0.2, 0                           # P5b2's held-out split
ARM = "fg-gr"
SPEC = C19.arm_spec(ARM)
TRANSFORMS = C19.TRANSFORMS                         # id, al4, sp4, mg3, rf
REG6 = "1,0.25,0.05,0.01,0.002,0"
KB_ARGS = {"metaqa": {"hops": 3, "ks": "2,4"}, "webqsp": {"hops": 2, "ks": "4,16,64,256"}}
TF_ARGS = {"al4": {"rows": "even", "hops": 2, "ks": "2,4,16", "regimes": "1,0.05,0"},
           "sp4": {"rows": "odd", "hops": 2, "ks": "2,4,16", "regimes": "1,0.05,0"},
           "mg3": {"rows": "all", "hops": 3, "ks": "2", "regimes": REG6},
           "rf": {"rows": "even", "hops": 3, "ks": "2,4", "regimes": "1,0.05,0"}}
P_ARGS = {"hops": 2, "ks": "4,16,64", "k_fine": C21.K_FINE}
EPOCHS, PER_EPOCH, BATCH, LR, WD, SEED = C19.EPOCHS, C19.PER_EPOCH, C19.BATCH, C19.LR, C19.WD, C19.SEED
COLNAMES = ("f", "lm", "has", "seed")
NCOL = len(COLNAMES)
STOP_RULE = ("the epoch (1 to 12) whose dataset-balanced mean, over the split's four datasets, of the held-out "
             "objective (chainscore19's node loss plus chain loss on the fold's held-out questions of the id build, "
             "regime 1) is lowest; ties go to the earlier")
BUILD_SETTINGS = {"columns": list(COLNAMES),
                  "scorer": "chainscore19's joint model fg-gr (f and g MLPs, T = 0) with chainscore21's lb offset",
                  "read": "regime 1 (the whole carve)", "folds": FOLDS,
                  "fold": "parity of sha256('dataset:question id') read as an integer",
                  "cross_fit": "each question's columns from the scorer of the other fold",
                  "fit": "per split on its four training datasets' fit builds (metaqa's five), one fold's questions "
                         "less 20% held out (P5b2's split_of, seed 0), datasets uniform; " + STOP_RULE,
                  "builds": {"kb": KB_ARGS, "transforms": TF_ARGS, "passages": P_ARGS, "fit_regimes": REG6,
                             "read_regimes": "1", "max_chains": CP.MAX_CHAINS},
                  "pools": "P1c2's looks (outputs/p1c2/look); rows in P1c2's cache order", "dtype": "float16"}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def write_json(p, rec):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_bytes(json.dumps(rec, indent=1).encode("utf-8"))
    os.replace(tmp, p)


def script_sha256():
    return LC.sha_src(__file__)


def rel(p):
    """A path relative to the repository when it is inside it (records and the builds' metas carry no host path)."""
    p = Path(p)
    try:
        return p.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return p.as_posix()


def fold_of(ds, qid):
    return int(hashlib.sha256(f"{ds}:{qid}".encode("utf-8")).hexdigest(), 16) % FOLDS


def split_of(n, frac=HOLD, seed=SPLIT_SEED):
    """P5b2's split: (train, held-out) indices, each sorted; a seeded permutation's first round(frac n) are held out."""
    perm = np.random.default_rng(seed).permutation(n)
    k = int(round(frac * n))
    return np.sort(perm[k:]), np.sort(perm[:k])


def fit_carve(ds):
    return FIT_CARVE.get(ds, "fit")


def carves_of(splits=SPLITS, evals=SIX):
    return {sp: tuple((d, fit_carve(d)) for d in splits[sp]) + tuple((d, "s1eval") for d in evals) for sp in splits}


def peak_rss():
    return LC.peak_rss()


# ── builds ───────────────────────────────────────────────────────────────────


def build_paths(ds, carve, tf="id", root=BUILDS):
    d = Path(root) / ds
    return d / f"{carve}-{tf}.npz", d / f"{carve}-{tf}.json"


def build_args(ds, carve, tf="id"):
    if tf not in TRANSFORMS:
        raise SystemExit(f"a7: transform {tf} is not one of {TRANSFORMS}")
    fit = carve == fit_carve(ds) and ds != "webqsp"
    if tf != "id" and not (ds == "metaqa" and fit):
        raise SystemExit(f"a7: the transforms are metaqa's fit carve's only, not {ds}/{carve}")
    if not fit and carve != "s1eval":
        raise SystemExit(f"a7: {ds}/{carve} is neither {ds}'s fit carve nor s1eval")
    regimes = REG6 if fit else "1"
    if ds in KB:
        a = dict(KB_ARGS[ds], regimes=regimes, transform="id", rows="all")
        if tf != "id":
            a.update(TF_ARGS[tf])
            a["transform"] = tf
    else:
        a = dict(P_ARGS, regimes=regimes, transform="id", rows="all")
    a["max_chains"] = CP.MAX_CHAINS
    return a


def look_pool_sizes(files):
    return np.concatenate([np.load(f)["q_pool_size"].astype(np.int64) for f in files])


def check_build(d, ds, carve, tf, files, ids):
    """The build's rows are the look's questions in order (all rows) or its even / odd ones, each with its whole
    pool."""
    a = build_args(ds, carve, tf)
    ri = np.asarray(d["row_ix"], np.int64)
    want = np.arange(len(ids))
    if a["rows"] == "even":
        want = want[want % 2 == 0]
    elif a["rows"] == "odd":
        want = want[want % 2 == 1]
    if not np.array_equal(ri, want) or np.asarray(d["carve_ix"]).any():
        raise SystemExit(f"a7: {ds}/{carve}-{tf}: the build's rows are not the look's {a['rows']} questions in order")
    if tf != "rf" and not np.array_equal(np.asarray(d["n"], np.int64), look_pool_sizes(files)[want]):
        raise SystemExit(f"a7: {ds}/{carve}-{tf}: the build's rows are not the look's pools")


def sbuild(ds, carve, tf="id", root=BUILDS, look_root=LOOK, rel_dir=REL, placement=None):
    t0 = time.time()
    a = build_args(ds, carve, tf)
    npz, js = build_paths(ds, carve, tf, root)
    npz.parent.mkdir(parents=True, exist_ok=True)
    _d, files, ids, head = LC.look_chunks(ds, carve, look_root)
    ns = SimpleNamespace(ds=ds, carve=carve, hops=a["hops"], ks=a["ks"], regimes=a["regimes"],
                         transform=a["transform"], rows=a["rows"], max_chains=a["max_chains"], limit=None,
                         out=rel(npz))
    side = None
    if ds in KB:
        C19.build19(ns, root=Path(look_root), rel_dir=Path(rel_dir))
    else:
        ns.k_fine, ns.src_root, ns.edges_out = a["k_fine"], rel(look_root), None
        side = C21.pbuild_cmd(ns)
    with np.load(npz) as z:
        d = {k: z[k] for k in ("n", "row_ix", "carve_ix", "level_names", "meta")}
    meta = json.loads(str(d["meta"]))
    check_build(d, ds, carve, tf, files, ids)
    rec = {"declared_in": DECLARED, "stage": "A7-build", "dataset": ds, "carve": carve, "transform": tf, "args": a,
           "look": head["records"], "carve_ids_sha256": head["carve_ids_sha256"], "queries": len(ids),
           "rows": int(d["n"].size), "nodes": meta["nodes"], "pairs": meta["pairs"],
           "levels": [str(x) for x in d["level_names"]], "regimes": meta["regimes"],
           "relations_present": meta.get("relations_present"), "name_embeddings_sha256": meta.get("name_embeddings_sha256"),
           "passage_types": None if side is None else {k: side["types"][k] for k in ("K", "triples", "sample", "seed",
                                                                                         "empty", "centroids_sha256")},
           "npz_sha256": LC.sha_file(npz), "npz_bytes": npz.stat().st_size,
           "chainscore19_sha256": LC.sha_src(C19.__file__), "chainscore21_sha256": LC.sha_src(C21.__file__),
           "chainscore18_sha256": LC.sha_src(CS.__file__), "chainpop17_sha256": LC.sha_src(CP.__file__),
           "reltype11_sha256": LC.sha_src(RT.__file__), "script_sha256": script_sha256(),
           "placement": placement or {"where": "laptop"}, "peak_rss_bytes": peak_rss(),
           "seconds": round(time.time() - t0, 1), "utc": utc()}
    write_json(js, rec)
    log(f"a7 sbuild {ds}/{carve}-{tf}: {rec['rows']} rows, {rec['nodes']} nodes, {rec['pairs']} pairs, levels "
        f"{rec['levels']}, {rec['npz_bytes'] / 1e6:.0f} MB ({rec['seconds']:.0f}s)")
    return rec


def load_build(ds, carve, tf, root, head):
    """A stored build and its record: A7's, of this look, its arrays the record's."""
    npz, js = build_paths(ds, carve, tf, root)
    if not js.exists():
        raise SystemExit(f"a7: {js} missing (a7_s6_cols.py sbuild {ds} {carve} --transform {tf})")
    rec = json.loads(js.read_text(encoding="utf-8"))
    if rec.get("declared_in") != DECLARED or rec.get("stage") != "A7-build" or rec.get("args") != build_args(ds, carve, tf):
        raise SystemExit(f"a7: {js} is not A7's build of {ds}/{carve}-{tf}")
    if rec["look"] != head["records"] or rec["carve_ids_sha256"] != head["carve_ids_sha256"]:
        raise SystemExit(f"a7: {js} was built from another look of {ds}/{carve}")
    if LC.sha_file(npz) != rec["npz_sha256"]:
        raise SystemExit(f"a7: {npz}: sha256 is not its record's")
    return CS.load(npz), rec, LC.sha_file(js)


# ── the scorer ───────────────────────────────────────────────────────────────


def make_batch(items, builds, LB, st, train=True):
    """chainscore19's batch with chainscore21's lb offset (log C_level,length) in place of log C_level."""
    import torch
    bt = C19.make_batch(items, builds, st, pairs=True, train=train)
    lc = bt["lc"].numpy().copy()
    for j, (bi, i, _ri) in enumerate(np.asarray(items, np.int64).reshape(-1, 3)):
        d = builds[bi]
        p0, p1 = int(d["rowoff"][i]), int(d["rowoff"][i + 1])
        lc[j, :p1 - p0] = LB[bi][p0:p1]
    bt["lc"] = torch.from_numpy(lc)
    return bt


def parts(model, bt):
    """chainscore19's forward at T = 0, in its parts: f, log(eps + sum_c p0(c) m_c), the mass flag, and s."""
    import torch
    F_ = torch.nn.functional
    B, Ln, Lk = bt["B"], bt["Ln"], bt["Lk"]
    f = torch.zeros(B * Ln).index_copy(0, bt["npos"], model.f(bt["Xn"]).squeeze(-1)).reshape(B, Ln)
    gamma, eps = F_.softplus(model.gamma_r), torch.exp(model.leps)
    if bt["Xp"] is None:                 # chainscore19's forward without pairs: f + gamma leps
        agg = torch.zeros(B, Ln)
        return f, model.leps.expand(B, Ln), agg > 0, f + gamma * model.leps
    else:
        z = torch.full((B * Lk,), C19.NEG).index_copy(0, bt["ppos"], model.g(bt["Xp"]).squeeze(-1))
        lp0 = C19.masked_logsoftmax(z.reshape(B, Lk) - bt["lc"], bt["pm"])
        agg = torch.zeros(B * Ln).index_add(0, bt["in"], torch.exp(lp0).reshape(-1)[bt["ip"]] * bt["im"]).reshape(B, Ln)
    lm = torch.log(eps + agg)
    return f, lm, agg > 0, f + gamma * lm


def held_objective(model, d, lb, st, rows, batch=BATCH):
    """chainscore19's objective over the rows at regime 1: the node loss's mean over rows with a gold plus the chain
    loss's mean over rows with a labelled chain."""
    import torch
    sn = sc = 0.0
    ng = no = 0
    with torch.no_grad():
        for b0 in range(0, rows.size, batch):
            r = rows[b0:b0 + batch]
            bt = make_batch(np.c_[np.zeros(r.size, np.int64), r, np.zeros(r.size, np.int64)], [d], [lb], st)
            s, lp = C19.forward(model, bt)
            _l, ln, lc = C19.objective(s, lp, bt)
            hg, ho = int(bt["gm"].any(1).sum()), int(bt["on"].any(1).sum())
            sn, sc, ng, no = sn + ln * hg, sc + lc * ho, ng + hg, no + ho
    return sn / max(ng, 1) + sc / max(no, 1), ng, no


def fit_paths(split, fold, root=FITS):
    return Path(root) / f"{split}-f{fold}.pt", Path(root) / f"{split}-f{fold}.json"


def stats_json(st):
    return {k: [float(x) for x in np.asarray(v, np.float32)] for k, v in st.items()}


def stats_np(js):
    return {k: np.asarray(v, np.float32) for k, v in js.items()}


def sfit(split, fold, splits=SPLITS, builds_root=BUILDS, look_root=LOOK, out=FITS, epochs=EPOCHS, per_epoch=PER_EPOCH,
         batch=BATCH, placement=None):
    import torch
    if split not in splits or fold not in range(FOLDS):
        raise SystemExit(f"a7: sfit takes a split of {sorted(splits)} and a fold of 0..{FOLDS - 1}")
    names = list(splits[split])
    if HELD_OUT.get(split) in names or names[0] != "metaqa":
        raise SystemExit(f"a7: {split}: its held-out dataset would train the scorer, or metaqa is not first")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    t0 = time.time()
    builds, LB, groups, held, counts, brecs = [], [], [], {}, {}, {}
    for ds in names:
        cv = fit_carve(ds)
        _d, _files, ids, head = LC.look_chunks(ds, cv, look_root)
        f_of = np.asarray([fold_of(ds, q) for q in ids], np.int64)
        mine = np.flatnonzero(f_of == fold)
        tr_i, ho_i = split_of(mine.size)
        tr_q = np.zeros(len(ids), bool)
        ho_q = np.zeros(len(ids), bool)
        tr_q[mine[tr_i]] = True
        ho_q[mine[ho_i]] = True
        grp = []
        for tf in (TRANSFORMS if ds == "metaqa" else ("id",)):
            d, rec, brecs[f"{ds}/{cv}-{tf}"] = load_build(ds, cv, tf, builds_root, head)
            d = C19.prep(d)
            q = np.asarray(d["row_ix"], np.int64)
            lab = d["train_rows"]
            d["train_rows"] = lab[tr_q[q[lab]]]
            if not d["train_rows"].size:
                raise SystemExit(f"a7: {ds}/{cv}-{tf}: no training row in fold {fold}")
            if tf == "id":
                held[ds] = (len(builds), lab[ho_q[q[lab]]])
            counts[f"{ds}/{tf}"] = {"rows": int(q.size), "labelled": int(lab.size),
                                    "train": int(d["train_rows"].size),
                                    "held_out": int(held[ds][1].size) if tf == "id" else 0}
            builds.append(d)
            LB.append(C21.lb_offsets(d))
            grp.append(len(builds) - 1)
        counts[ds] = {"questions": len(ids), "fold_questions": int(mine.size), "train_questions": int(tr_i.size),
                      "held_out_questions": int(ho_i.size)}
        groups.append(grp)
    st = C19.input_stats(builds[0])
    model = C19.make_joint(SPEC, SEED)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    rng = np.random.default_rng(SEED)
    hist, best = [], None
    log(f"a7 sfit {split} fold {fold}: {', '.join(names)}; " + ", ".join(f"{k} {v['train']}" for k, v in counts.items()
                                                                          if "/" in k) + " training rows")
    for ep in range(1, epochs + 1):
        t1 = time.time()
        model.train()
        items = C21.draw21(rng, builds, groups, per_epoch)
        tot, nb = 0.0, 0
        for b0 in range(0, len(items), batch):
            bt = make_batch(items[b0:b0 + batch], builds, LB, st)
            s, lp = C19.forward(model, bt)
            loss, _ln, _lc = C19.objective(s, lp, bt)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss)
            nb += 1
        model.eval()
        ho = {ds: held_objective(model, builds[bi], LB[bi], st, rows)[0] for ds, (bi, rows) in held.items()}
        score = float(np.mean([ho[ds] for ds in names]))
        hist.append({"epoch": ep, "loss": round(tot / max(nb, 1), 6), "held_out": {k: round(v, 6) for k, v in ho.items()},
                     "held_out_mean": round(score, 6), "by_build": np.bincount(items[:, 0], minlength=len(builds)).tolist(),
                     **C19.coupling(model), "seconds": round(time.time() - t1, 1)})
        log(f"  ep {ep}: loss {tot / max(nb, 1):.4f}, held-out {score:.4f} "
            + ", ".join(f"{k} {v:.4f}" for k, v in ho.items()) + f" ({time.time() - t1:.0f}s)")
        if best is None or score < best[1]:
            best = (ep, score, {k: v.detach().clone() for k, v in model.state_dict().items()})
    model.load_state_dict(best[2])
    pt, js = fit_paths(split, fold, out)
    pt.parent.mkdir(parents=True, exist_ok=True)
    tmp = pt.with_name(pt.name + ".tmp")
    torch.save(model.state_dict(), tmp)
    os.replace(tmp, pt)
    rec = {"declared_in": DECLARED, "stage": "A7-fit", "split": split, "fold": fold, "datasets": names,
           "held_out_dataset": HELD_OUT.get(split), "arm": ARM, "spec": SPEC, "builds": brecs, "counts": counts,
           "kept_epoch": best[0], "kept_held_out": round(best[1], 6), "coupling": C19.coupling(model), "hist": hist,
           "stats": stats_json(st),
           "settings": {"epochs": epochs, "per_epoch": per_epoch, "batch": batch, "lr": LR, "wd": WD, "seed": SEED,
                        "hold": HOLD, "split_seed": SPLIT_SEED, "stop": STOP_RULE, "draw": "chainscore21's draw21: "
                        "the dataset uniform, then chainscore19's draw (the build, the row, the regime)",
                        "threads": 1, "build": BUILD_SETTINGS},
           "state_sha256": LC.sha_file(pt), "script_sha256": script_sha256(),
           "chainscore19_sha256": LC.sha_src(C19.__file__), "chainscore21_sha256": LC.sha_src(C21.__file__),
           "placement": placement or {"where": "laptop"}, "peak_rss_bytes": peak_rss(),
           "seconds": round(time.time() - t0, 1), "utc": utc()}
    write_json(js, rec)
    log(f"a7 sfit {split} fold {fold}: kept epoch {best[0]} (held-out {best[1]:.4f}), {C19.coupling(model)} "
        f"({rec['seconds']:.0f}s)")
    return rec


def load_fit(split, fold, root=FITS):
    import torch
    pt, js = fit_paths(split, fold, root)
    if not js.exists():
        raise SystemExit(f"a7: {js} missing (a7_s6_cols.py sfit {split} {fold})")
    rec = json.loads(js.read_text(encoding="utf-8"))
    if rec.get("declared_in") != DECLARED or rec.get("stage") != "A7-fit" or rec.get("split") != split \
            or rec.get("fold") != fold or rec.get("settings", {}).get("build") != BUILD_SETTINGS:
        raise SystemExit(f"a7: {js} is not A7's fit of {split} fold {fold}")
    if LC.sha_file(pt) != rec["state_sha256"]:
        raise SystemExit(f"a7: {pt}: sha256 is not its record's")
    model = C19.make_joint(SPEC, SEED)
    model.load_state_dict(torch.load(pt, weights_only=True))
    model.eval()
    return model, stats_np(rec["stats"]), rec, LC.sha_file(js)


# ── the columns ──────────────────────────────────────────────────────────────


def scorer_cols(model, d, lb, st, rows, X, batch=BATCH):
    """The four columns of the build's rows, written into X at their nodes."""
    import torch
    with torch.no_grad():
        for b0 in range(0, rows.size, batch):
            r = rows[b0:b0 + batch]
            bt = make_batch(np.c_[np.zeros(r.size, np.int64), r, np.zeros(r.size, np.int64)], [d], [lb], st,
                            train=False)
            f, lm, has, _s = parts(model, bt)
            f, lm, has = f.double().numpy(), lm.double().numpy(), has.numpy()
            for j, i in enumerate(r):
                a, n = int(d["xn_off"][i]), int(d["n"][i])
                X[a:a + n, 0] = f[j, :n]
                X[a:a + n, 1] = lm[j, :n]
                X[a:a + n, 2] = has[j, :n]
                X[a:a + n, 3] = d["seed0"][a:a + n]


def cols(ds, carve, splits=SPLITS, evals=SIX, builds_root=BUILDS, fits_root=FITS, look_root=LOOK, cache_root=CACHE,
         out=COLS, rel_dir=REL, placement=None):
    import torch
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    t0 = time.time()
    co = carves_of(splits, evals)
    sps = [sp for sp in splits if (ds, carve) in co[sp]]
    if not sps:
        raise SystemExit(f"a7: {ds}/{carve}: no screen split reads this carve")
    fits = {sp: [load_fit(sp, f, fits_root) for f in range(FOLDS)] for sp in sps}
    _d, files, ids, head = LC.look_chunks(ds, carve, look_root)
    parts_, recs = LC.part_dirs(ds, carve, cache_root)
    for p, r in zip(parts_, recs):
        if r["look"]["carve_ids_sha256"] != head["carve_ids_sha256"] or r["look"]["records"] != head["records"]:
            raise SystemExit(f"a7: {p}: built from another look than {ds}/{carve}'s")
        if r["ids"] != ids[r["query_range"][0]:r["query_range"][1]]:
            raise SystemExit(f"a7: {p}: its question ids are not the look's")
        if LC.sha_file(p / "n.npy") != r["arrays"]["n"]["sha256"]:
            raise SystemExit(f"a7: {p / 'n.npy'}: sha256 is not its record's")
    stored = carve == fit_carve(ds) and ds != "webqsp"
    tmp = None
    try:
        if stored:
            d, brec, bsha = load_build(ds, carve, "id", builds_root, head)
        else:
            (Path(builds_root) / "tmp").mkdir(parents=True, exist_ok=True)
            tmp = Path(tempfile.mkdtemp(prefix=f"{ds}-{carve}-", dir=Path(builds_root) / "tmp"))
            brec = sbuild(ds, carve, "id", root=tmp, look_root=look_root, rel_dir=rel_dir, placement=placement)
            d, brec, bsha = load_build(ds, carve, "id", tmp, head)
        check_build(d, ds, carve, "id", files, ids)
        lb = C21.lb_offsets(d)
        N = int(d["xn_off"][-1])
        f_of = np.asarray([fold_of(ds, q) for q in ids], np.int64)
        rows_of = {f: np.flatnonzero(f_of == f) for f in range(FOLDS)}
        X = {}
        for sp in sps:
            X[sp] = np.zeros((N, NCOL))
            for m in range(FOLDS):           # model m (fitted on fold m) gives the other fold's questions their columns
                model, st, _rec, _sha = fits[sp][m]
                scorer_cols(model, d, lb, st, rows_of[1 - m], X[sp])
        # the first question of each part: its first-stage ranks are the look's
        n_all = np.asarray(d["n"], np.int64)
        done = {}
        for p, r in zip(parts_, recs):
            qa, qb = r["query_range"]
            ra, rb = int(d["xn_off"][qa]), int(d["xn_off"][qb]) if qb < n_all.size else N
            n = np.load(p / "n.npy").astype(np.int64)
            if n.size != qb - qa or not np.array_equal(n, n_all[qa:qb]) or rb - ra != r["rows"] \
                    or [ra, rb] != list(r["row_range"]):
                raise SystemExit(f"a7: {p}: its questions and rows are not the build's {qa}..{qb}")
            lo = r["chunks"][0]
            with np.load(files[lo]) as zf:
                x0 = zf["x"][:int(zf["q_pool_size"][0]), head["columns"].index("rrf")].astype(np.float64)
            _o, rank = CP.rr_order(x0)
            if not np.array_equal(rank, d["rank"][d["rank_off"][qa]:d["rank_off"][qa + 1]]):
                raise SystemExit(f"a7: {p}: question {qa}'s first-stage ranks in the build are not the look's")
            folds = np.bincount(f_of[qa:qb], weights=n, minlength=FOLDS).astype(np.int64)
            for sp in sps:
                x = X[sp][ra:rb]
                if not np.isfinite(x).all():
                    raise SystemExit(f"a7: {p}: a column is not finite")
                x16 = x.astype(np.float16)
                od = Path(out) / sp / ds / carve / p.name
                od.mkdir(parents=True, exist_ok=True)
                sha = LC.save_npy(od, "cols", x16)
                qs_ = lambda v: [round(float(t), 4) for t in np.quantile(v, [0.05, 0.5, 0.95])] if v.size else []
                rec = {"declared_in": DECLARED, "stage": "A7", "split": sp, "dataset": ds, "carve": carve,
                       "part": p.name, "step1_record_sha256": LC.sha_file(p / "record.json"),
                       "build_record_sha256": bsha, "build_kept": stored,
                       "fit_records_sha256": [fits[sp][f][3] for f in range(FOLDS)],
                       "kept_epochs": [fits[sp][f][2]["kept_epoch"] for f in range(FOLDS)],
                       "queries": int(n.size), "rows": int(n.sum()), "rows_by_fold_of_question": folds.tolist(),
                       "settings": BUILD_SETTINGS,
                       "quantiles_5_50_95": {k: qs_(x[:, c]) for c, k in enumerate(COLNAMES[:2])},
                       "shares": {k: round(float(x[:, c].mean()), 4) if x.size else 0.0
                                  for c, k in ((2, "has"), (3, "seed"))},
                       "max_abs_float16_error": float(np.abs(x16.astype(np.float64) - x).max()) if x.size else 0.0,
                       "arrays": {"cols": {"dtype": x16.dtype.str, "shape": list(x16.shape), "sha256": sha}},
                       "cache_root": rel(cache_root), "placement": placement or {"where": "laptop"},
                       "script_sha256": script_sha256(), "utc": utc()}
                write_json(od / "record.json", rec)
            done[p.name] = {"queries": int(n.size), "rows": int(n.sum())}
            log(f"  {ds}/{carve} {p.name}: {int(n.size)} questions, {int(n.sum())} rows, splits {', '.join(sps)}")
    finally:
        if tmp is not None:
            shutil.rmtree(tmp, ignore_errors=True)
    log(f"a7 cols {ds}/{carve}: {len(done)} part(s), splits {', '.join(sps)} ({time.time() - t0:.0f}s, peak rss "
        f"{(peak_rss() or 0) / 1e9:.1f} GB)")
    return done


# ── selftest ─────────────────────────────────────────────────────────────────


def toy_look(tmp, ds, carve, rng, rows, nparts=2):
    """A toy look of ds/carve under tmp/look (chainpop17's typed toy KB for a KB name, chainscore21's untyped toy for
    a passage name), with ids.json, lean_cache's record fields and step-1 cache parts under tmp/cache."""
    src = tmp / "_src" / f"{ds}-{carve}"
    if ds in KB:
        CP.toy_world(src, rng, RT.proj_matrix(), rows=rows, chunk=8)
        s = src / "toy" / "read"
    else:
        s = C21.untyped_toy(src, rng, "read", rows=rows)
    d = tmp / "look" / ds / carve
    d.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(s), str(d))
    files = sorted((d / "chunks").glob("c*.npz"))
    n = look_pool_sizes(files)
    nq = [int(np.load(f)["q_pool_size"].size) for f in files]
    Q = int(n.size)
    ids = [f"{ds}-{carve}-q{i}" for i in range(Q)]
    write_json(d / "ids.json", ids)
    for r_ in sorted(d.glob("record*.json")):
        rr = json.loads(r_.read_text(encoding="utf-8"))
        rr.update({"carve_ids_sha256": f"toy-{ds}-{carve}", "carve_queries": Q, "chunk_queries": max(nq),
                   "column_blocks": {}, "limit": None})
        write_json(r_, rr)
    head = LC.look_chunks(ds, carve, tmp / "look")[3]
    cuts = np.linspace(0, len(files), nparts + 1).round().astype(int)
    q0 = r0 = 0
    for pi in range(nparts):
        lo, hi = int(cuts[pi]), int(cuts[pi + 1])
        k = sum(nq[lo:hi])
        p = tmp / "cache" / ds / carve / f"part_{pi}of{nparts}"
        p.mkdir(parents=True)
        nn = n[q0:q0 + k]
        sha = LC.save_npy(p, "n", nn)
        write_json(p / "record.json", {"query_range": [q0, q0 + k], "row_range": [r0, r0 + int(nn.sum())],
                                       "look": head, "ids": ids[q0:q0 + k], "chunks": [lo, hi], "queries": k,
                                       "rows": int(nn.sum()), "arrays": {"n": {"sha256": sha}}})
        q0, r0 = q0 + k, r0 + int(nn.sum())
    return ids


def np_mlp(seq, X):
    W = [(m.weight.detach().double().numpy(), m.bias.detach().double().numpy()) for m in seq if hasattr(m, "weight")]
    h = X.astype(np.float64)
    for k, (w, b) in enumerate(W):
        h = h @ w.T + b
        if k < len(W) - 1:
            h = np.maximum(h, 0.0)
    return h[:, 0]


def cols_by_hand(model, d, lb, st, i):
    """Row i's four columns from the model's weights in float64 numpy: f, log(eps + sum_c p0(c) m_c), has, seed."""
    import torch
    F_ = torch.nn.functional
    a, n = int(d["xn_off"][i]), int(d["n"][i])
    Xn = (d["XN"][a:a + n].astype(np.float32) - st["mn"]) / st["sn"]
    f = np_mlp(model.f, Xn)
    eps = float(torch.exp(model.leps))
    p0, p1 = int(d["rowoff"][i]), int(d["rowoff"][i + 1])
    agg = np.zeros(n)
    if p1 > p0:
        Xp = (C19.pair_raw(d, 0, p0, p1) - st["mp"]) / st["sp"]
        z = np_mlp(model.g, Xp) - lb[p0:p1].astype(np.float64)
        lp = z - (z.max() + np.log(np.exp(z - z.max()).sum()))
        no = d["node_off"][p0:p1 + 1]
        for c in range(p1 - p0):
            nodes = d["node"][no[c]:no[c + 1]].astype(np.int64)
            np.add.at(agg, nodes, np.exp(lp[c]) * d["mass"][no[c]:no[c + 1]].astype(np.float64))
    del F_
    return np.c_[f, np.log(eps + agg), agg > 0, d["seed0"][a:a + n]]


def refused(call, what):
    try:
        call()
    except SystemExit as e:
        log(f"  refused ({what}): {e}")
        return
    raise AssertionError(f"not refused: {what}")


def selftest():
    import torch
    t0 = time.time()
    assert NCOL == 4 and SPEC == {"f": True, "g": True, "T": 0, "gr": True} and TRANSFORMS[0] == "id"
    assert fold_of("2wiki", "q7") == int(hashlib.sha256(b"2wiki:q7").hexdigest(), 16) % 2
    tr, ho = split_of(10)
    assert ho.size == 2 and np.array_equal(np.sort(np.r_[tr, ho]), np.arange(10))
    co = carves_of()
    assert co["L-musique"][:4] == (("metaqa", "fit"), ("squad", "fit"), ("hotpotqa", "fit"), ("2wiki", "fit"))
    assert ("musique", "s1fit") in co["L-hotpotqa"] and ("musique", "s1fit") not in co["L-musique"]
    assert all((d, "s1eval") in co[sp] for sp in SPLITS for d in SIX)
    assert build_args("metaqa", "fit", "al4")["rows"] == "even" and build_args("webqsp", "s1eval")["regimes"] == "1"
    assert build_args("musique", "s1fit")["regimes"] == REG6 and "k_fine" in build_args("2wiki", "s1eval")
    refused(lambda: build_args("2wiki", "fit", "al4"), "a transform off metaqa")
    refused(lambda: build_args("metaqa", "s1eval", "rf"), "a transform on s1eval")
    refused(lambda: build_args("2wiki", "select"), "a carve neither fit nor s1eval")
    rng = np.random.default_rng(71)
    tmp = Path(tempfile.mkdtemp(prefix="a7_"))
    TOY = {"T": ("metaqa", "2wiki")}
    EV = ("metaqa", "2wiki")
    try:
        ids = {}
        for ds in EV:
            for cv, nrow in (("fit", 96), ("s1eval", 40)):
                ids[(ds, cv)] = toy_look(tmp, ds, cv, rng, nrow)
        kw = dict(look_root=tmp / "look", rel_dir=tmp / "norel")
        # 1. the builds: metaqa's five on its fit carve, 2wiki's typed fit carve; rows are the look's, records A7's
        for tf in TRANSFORMS:
            rec = sbuild("metaqa", "fit", tf, root=tmp / "b", **kw)
            assert rec["args"] == build_args("metaqa", "fit", tf) and rec["rows"] == (48 if tf in ("al4", "sp4", "rf")
                                                                                      else 96)
        rec = sbuild("2wiki", "fit", root=tmp / "b", **kw)
        assert rec["passage_types"]["K"] > 0 and rec["regimes"] == [float(x) for x in REG6.split(",")]
        assert not any((tmp / "b" / "2wiki").glob("cs21tmp_*"))
        # 2. the fits: fold f trains on its questions less the held-out fifth; a repeat is bit for bit; the kept
        #    epoch is the lowest held-out objective
        recs = {}
        for f in range(FOLDS):
            recs[f] = sfit("T", f, splits=TOY, builds_root=tmp / "b", look_root=tmp / "look", out=tmp / "fits",
                           epochs=3, per_epoch=96, batch=16)
            h = recs[f]["hist"]
            assert recs[f]["kept_epoch"] == min(h, key=lambda e: (e["held_out_mean"], e["epoch"]))["epoch"]
            for ds in EV:
                c = recs[f]["counts"][ds]
                assert c["fold_questions"] == sum(fold_of(ds, q) == f for q in ids[(ds, "fit")])
                assert c["held_out_questions"] == round(0.2 * c["fold_questions"])
        rep = sfit("T", 0, splits=TOY, builds_root=tmp / "b", look_root=tmp / "look", out=tmp / "fits2", epochs=3,
                   per_epoch=96, batch=16)
        drop = lambda h: [{k: v for k, v in e.items() if k != "seconds"} for e in h]
        assert drop(rep["hist"]) == drop(recs[0]["hist"]) and rep["kept_epoch"] == recs[0]["kept_epoch"]
        s0 = torch.load(fit_paths("T", 0, tmp / "fits")[0], weights_only=True)
        s1 = torch.load(fit_paths("T", 0, tmp / "fits2")[0], weights_only=True)
        assert all(torch.equal(s0[k], s1[k]) for k in s0)
        # the training rows: fold 0's questions less its held-out ones, nothing of fold 1
        _d, _f, idm, headm = LC.look_chunks("metaqa", "fit", tmp / "look")
        dm, _r, _s = load_build("metaqa", "fit", "id", tmp / "b", headm)
        dm = C19.prep(dm)
        fm = np.asarray([fold_of("metaqa", q) for q in idm])
        mine = np.flatnonzero(fm == 0)
        _tr, hi_ = split_of(mine.size)
        assert recs[0]["counts"]["metaqa/id"]["train"] == int(np.isin(dm["train_rows"], np.setdiff1d(
            mine, mine[hi_])).sum())
        refused(lambda: sfit("T2", 0, splits={"T2": ("2wiki", "metaqa")}, builds_root=tmp / "b",
                             look_root=tmp / "look", out=tmp / "x"), "metaqa not first")
        refused(lambda: sfit("L-musique", 0, splits={"L-musique": ("metaqa", "musique")}, builds_root=tmp / "b",
                             look_root=tmp / "look", out=tmp / "x"), "the held-out dataset trains")
        # 3. the columns: each question's from the other fold's scorer, against a float64 brute force from the
        #    weights; s is chainscore19's forward; the seed column is the look's bucket-0 seeds; s1eval's build is
        #    made in the item and removed
        for ds in EV:
            for cv in ("fit", "s1eval"):
                cols(ds, cv, splits=TOY, evals=EV, builds_root=tmp / "b", fits_root=tmp / "fits",
                     look_root=tmp / "look", cache_root=tmp / "cache", out=tmp / "cols", rel_dir=tmp / "norel")
        assert not any((tmp / "b" / "tmp").iterdir()) and not (tmp / "b" / "metaqa" / "s1eval-id.json").exists()
        fits = [load_fit("T", f, tmp / "fits") for f in range(FOLDS)]
        for ds in EV:
            for cv in ("fit", "s1eval"):
                _d, files, idx, head = LC.look_chunks(ds, cv, tmp / "look")
                if cv == "fit":
                    d, _r, _s = load_build(ds, cv, "id", tmp / "b", head)
                else:
                    sbuild(ds, cv, root=tmp / "rb", **kw)
                    d, _r, _s = load_build(ds, cv, "id", tmp / "rb", head)
                lb = C21.lb_offsets(d)
                got = np.concatenate([np.load(tmp / "cols" / "T" / ds / cv / f"part_{k}of2" / "cols.npy")
                                      for k in range(2)]).astype(np.float64)
                assert got.shape == (int(d["xn_off"][-1]), NCOL)
                own_diff = 0
                for i in range(d["n"].size):
                    f = fold_of(ds, idx[i])
                    m, st = fits[1 - f][0], fits[1 - f][1]
                    ref = cols_by_hand(m, d, lb, st, i)
                    a, n = int(d["xn_off"][i]), int(d["n"][i])
                    g_ = got[a:a + n]
                    tol = 2e-3 * np.abs(ref) + 2e-3
                    assert (np.abs(g_ - ref) <= tol).all(), (ds, cv, i, float(np.abs(g_ - ref).max()))
                    own = cols_by_hand(fits[f][0], d, lb, fits[f][1], i)
                    own_diff += int((np.abs(g_[:, :2] - own[:, :2]) > tol[:, :2]).any())
                assert own_diff > d["n"].size // 2, (ds, cv, own_diff)
                # the seed column is the look's bucket-0 seeds
                seeds = []
                for fl in files:
                    with np.load(fl) as zf:
                        for qq in range(zf["q_pool_size"].size):
                            v = np.zeros(int(zf["q_pool_size"][qq]))
                            sl, bk = zf["q_seed_local"][qq], zf["q_seed_bucket"][qq]
                            v[np.unique(sl[(sl >= 0) & (bk == 0)])] = 1.0
                            seeds.append(v)
                assert np.array_equal(got[:, 3], np.concatenate(seeds))
                # s from the parts is chainscore19's forward bit for bit
                r = np.arange(min(16, d["n"].size))
                bt = make_batch(np.c_[np.zeros(r.size, np.int64), r, np.zeros(r.size, np.int64)], [d], [lb],
                                fits[0][1], train=False)
                with torch.no_grad():
                    assert torch.equal(parts(fits[0][0], bt)[3], C19.forward(fits[0][0], bt)[0])
                assert got[:, 2].any() and (got[:, 2] == 0).any()
        rc = json.loads((tmp / "cols" / "T" / "2wiki" / "s1eval" / "part_1of2" / "record.json").read_text())
        assert rc["stage"] == "A7" and rc["settings"] == BUILD_SETTINGS and not rc["build_kept"]
        assert rc["max_abs_float16_error"] < 0.05 and sum(rc["rows_by_fold_of_question"]) == rc["rows"]
        # 4. the refusals: a part from another look, a part whose n is not its record's, a build of another look,
        #    a fit that is not A7's, a carve no split reads
        ck = dict(splits=TOY, evals=EV, builds_root=tmp / "b", fits_root=tmp / "fits", look_root=tmp / "look",
                  cache_root=tmp / "cache", out=tmp / "x", rel_dir=tmp / "norel")
        pr = tmp / "cache" / "2wiki" / "fit" / "part_1of2" / "record.json"
        keep = pr.read_bytes()
        h = json.loads(keep)
        h["look"]["carve_ids_sha256"] = "other"
        write_json(pr, h)
        refused(lambda: cols("2wiki", "fit", **ck), "a cache part of another look")
        pr.write_bytes(keep)
        pn = tmp / "cache" / "2wiki" / "fit" / "part_0of2" / "n.npy"
        keep_n = pn.read_bytes()
        np.save(pn, np.load(pn) + 1)
        refused(lambda: cols("2wiki", "fit", **ck), "a part's n changed")
        pn.write_bytes(keep_n)
        bj = tmp / "b" / "2wiki" / "fit-id.json"
        keep_b = bj.read_bytes()
        h = json.loads(keep_b)
        h["carve_ids_sha256"] = "other"
        write_json(bj, h)
        refused(lambda: cols("2wiki", "fit", **ck), "a build of another look")
        bj.write_bytes(keep_b)
        fj = fit_paths("T", 1, tmp / "fits")[1]
        keep_f = fj.read_bytes()
        h = json.loads(keep_f)
        h["settings"]["build"] = {**BUILD_SETTINGS, "read": "regime 0.05"}
        write_json(fj, h)
        refused(lambda: cols("2wiki", "fit", **ck), "a fit under other settings")
        fj.write_bytes(keep_f)
        refused(lambda: cols("2wiki", "select", **ck), "a carve no split reads")
        cols("2wiki", "fit", **ck)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    log(f"a7 selftest: the builds (metaqa's five, a typed passage carve) hold the look's rows; each fold's scorer "
        f"trains on its questions less the held-out fifth, keeps the lowest held-out epoch and repeats bit for bit; "
        f"each question's columns come from the other fold's scorer and match a float64 brute force from its weights, "
        f"s is chainscore19's forward bit for bit, the seed column is the look's bucket-0 seeds; s1eval's build is "
        f"made in the item and removed; the refusals ({time.time() - t0:.1f}s): ok")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", choices=("sbuild", "sfit", "cols"))
    ap.add_argument("a", nargs="?")
    ap.add_argument("b", nargs="?")
    ap.add_argument("--transform", default="id", choices=TRANSFORMS)
    ap.add_argument("--host", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    g = ap.parse_args(argv)
    if g.selftest:
        return selftest()
    where = {"where": "host" if g.host else "laptop"}
    if g.cmd == "sbuild":
        sbuild(g.a, g.b, g.transform, placement=where)
    elif g.cmd == "sfit":
        sfit(g.a, int(g.b), placement=where)
    elif g.cmd == "cols":
        cols(g.a, g.b, placement=where)
    else:
        ap.error("sbuild, sfit, cols or --selftest")
    return 0


if __name__ == "__main__":
    sys.exit(main())
