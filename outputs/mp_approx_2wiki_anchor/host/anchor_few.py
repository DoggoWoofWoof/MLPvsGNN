"""Design look (untracked; not a result and not filed): how many labelled questions does a new graph need, and does a
relation model fitted on another graph help with them?

Zero-shot, a relation model fitted on one graph helps another only between 2wiki and hotpotqa (kcal.py with the
label-free cap rule: +0.57 points of R@5 over the fitted kappa, mean of 41 cross reads); on musique and metaqa the best
kappa for every passage model is the bonus switched off (ksweep / kcal: oracle d about -4). A new graph usually has some
labelled questions. This look gives the target graph N of its own training questions (N in --n). Draw d takes the first
N rows of a fixed permutation of the target's pool carve (seed 20261100 + d), so a draw's N's are nested. The first
ceil(3N/4) rows train and the rest choose the epoch. Four arms run on each draw:
    zs       the source model as saved: no training (N = 0), read once
    warm     the source model, every parameter trained from its saved values
    kappa    the source model with only log_kappa and beta trained (a two-number calibration with labels)
    scratch  the same family, initialised as a fit on the target would be (the target's phrase or relation table and,
             for a PCA code, the PCA basis of the target's top-K vectors)
warm and kappa keep the saved values when the selection rows score them at least as high as the best epoch, so the
untrained model is a candidate; scratch has no such candidate. Every arm is read on the target's read carve: x1's half
B on the passage graphs and metaqa, as every read; on webqsp, whose fit stride is 1 so it has no x1, its whole select
carve (305 rows; nothing here is tuned on an A half, and half of it would leave 152). The fit
loop is anchor_walk8.fit_read_kd, imported unchanged: gold cross-entropy, rule p, its batches, optimiser, per-epoch
selection and reads. All arms of a draw share its torch / numpy seed. Epochs are set so that every fit takes at least
--steps optimiser steps. --n 0 reads zero-shot only (and loads no pool).
Targets:
    2wiki, hotpotqa   read x1, pool x4    anchor_gen's loaders; anchor_gen models read the 1024-phrase table they were
    musique           read x1, pool fit   fitted with, anchor_gen3 and anchor_cos models the 4096-phrase table
    metaqa            read x1, pool fit   the KB's own relation names (anchor_gen_kb.kb_view), ranked by their count
    webqsp            read select, pool fit   over the read carve (label-free), as anchor_gen_kb ranks them
With --alt, the zero-shot model is also read with every relation 'other' (NR: the untyped graph) and, on a KB, with
every relation shown with the next relation's text (SHUF: what the relation's meaning adds), as anchor_gen_kb reads.
Families: T0, text, gen (with or without a PCA code; not on a KB) and cos (anchor_cos bound). Free per-phrase models
carry no vocabulary to a new graph and are refused, as are --mres and cluster-typed models and a model fitted on the
target itself. Nothing reads a neighbour's score or state.

    python outputs/mp_approx_2wiki_anchor/host/anchor_few.py --target hotpotqa --models a.pt,b.pt --n 64,256,1024 --draws 3 --out PATH
"""
import os
import sys

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "4"

import argparse  # noqa: E402
import copy  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "outputs" / "mp_approx_kb_anchor" / "host"))
import anchor_cos as AC  # noqa: E402

G3 = AC.G3
AG, G2N = G3.AG, G3.G2N
AW11, AW8, AW6, A16, AW, AW3 = AG.AW11, AG.AW8, AG.AW6, AG.A16, AG.AW, AG.AW3
G2 = AW8.G2
R_TOP = AG.R_TOP
PINS = {"anchor_cos": "1039cf2f043bc6e3c5d342d0ea8f8f89ee3a1873c6e78a94f4078e99b7f667bf",
        "anchor_gen3": "ee5f76dc7d391649e04ab2b3575d4fdd30dd62cd2a34b1b7dbd114d56f7b14a7",
        "anchor_gen": "bdc938f546cff1192dda82701e92cf7003fce8aded07831c2c7930119ce4f108",
        "anchor_gen2": "cbb976408648e2da0d5398df7310b2695a471e246f7def05413f74cec84711c6"}
KB_PINS = {"anchor_kbfit": "c0f7eedc6d4563a4b43febe42874d26760f5563d93ac4d0544c966634bfc09f7",
           "anchor_gen_kb": "4e09d59de0a60767eb591863352389ff99e164f0e2cd841b858debd309ee7952"}
READ = {"2wiki": "x1", "hotpotqa": "x1", "musique": "x1", "metaqa": "x1", "webqsp": "select"}
POOL = {"2wiki": "x4", "hotpotqa": "x4", "musique": "fit", "metaqa": "fit", "webqsp": "fit"}
KBS = ("metaqa", "webqsp")
WHOLE = ("webqsp",)   # read carves read whole rather than as half B
ARMS = ("warm", "kappa", "scratch")
DRAW_SEED = 20261100
log = AG.log


def check_pins():
    for mod, key in ((AC, "anchor_cos"), (G3, "anchor_gen3"), (AG, "anchor_gen"), (G2N, "anchor_gen2")):
        if AW.sha(Path(mod.__file__)) != PINS[key]:
            raise SystemExit(f"{mod.__file__} is not the pinned file")
    AG.check_pins()


def load_passage(target, max_len, t0, need_pool):
    AW6.rebind(target)
    Q, part, rec_sha, load_state = AG.load_looks([READ[target]] + ([POOL[target]] if need_pool else []), max_len, t0)
    checks, _vocab = AW.anchor_tables(Q)
    log(f"anchors attached to {len(Q)} rows: {checks}, {time.time() - t0:.0f}s")
    info = {"look_score_records": rec_sha, "flag_checks": checks, "pruned_loader": load_state}
    return Q, part[READ[target]], part.get(POOL[target], []), info, None


def load_kb(target, max_len, t0, need_pool):
    import anchor_kbfit as KF
    KB = KF.KB
    for mod, key in ((KF, "anchor_kbfit"), (KB, "anchor_gen_kb")):
        if AW.sha(Path(mod.__file__)) != KB_PINS[key]:
            raise SystemExit(f"{mod.__file__} is not the pinned file")
    AW11.STATE.update({"max_len": max_len, "proj": False, "rows": 0, "edges_in": 0, "edges_kept": 0, "checked_rows": 0})
    Qr, sr, rel = KF.load_carve(target, READ[target])
    Qp, sp_, relp = KF.load_carve(target, POOL[target]) if need_pool else ([], {}, rel)
    if relp != rel:
        raise SystemExit("the carves disagree on the relation bank")
    Q0 = Qr + Qp
    read, pool = list(range(len(Qr))), list(range(len(Qr), len(Q0)))
    log(f"{target}: {READ[target]} {len(Qr)} rows, {POOL[target]} {len(Qp)} rows, {time.time() - t0:.0f}s")
    rank_of, vocab, phi, counts = KB.kb_relations(target, Q0, read, int(rel["n_relations"]))
    Q, st = KB.kb_view(Q0, rank_of)
    del Q0
    info = {"rel_vocab": KB.RELS[target][0], "rel_embeddings": KB.RELS[target][1], "look_score_kb": KB.LOOK_KB_SHA,
            "look_score_kb_records": {**sr, **sp_}, "relations": {**rel, "ranked_by_read_rows": counts[:64], "n_ranked": len(counts)},
            "kb_view": st, "pruned_loader": dict(AW11.STATE)}
    return Q, read, pool, info, (phi, len(vocab))


class Tables:
    """The phrase or relation table each source reads: on a KB its relation table; on a passage graph the 1024-phrase
    table for anchor_gen models (no 'pca' in their spec) and the 4096-phrase table for anchor_gen3 and anchor_cos ones."""

    def __init__(self, target, kb):
        self.target, self.kb, self.got, self.sha = target, kb, {}, {}

    def of(self, sp):
        if sp["family"] == "free":
            return None
        if self.kb is not None:
            return self.kb[0]
        key = 4096 if "pca" in sp else 1024
        if key not in self.got:
            self.got[key], self.sha[key] = (G3.phrase_table if key == 4096 else AG.phrase_table)(self.target)
        return self.got[key]


def check_source(p, ck, target):
    sp = ck["spec"]
    if "mres" in Path(p).parent.name or ck.get("mres"):
        raise SystemExit(f"{p}: an --mres model is read through anchor_mres only")
    if "clusters" in ck or "clu" in Path(p).parent.name.split("_"):
        raise SystemExit(f"{p}: a cluster-typed model is read through anchor_clu only")
    if sp["family"] == "free" and sp["tok"] != "T0":
        raise SystemExit(f"{p}: a free per-phrase model carries no vocabulary to a new graph")
    if sp["rule"] != "p":
        raise SystemExit(f"{p}: rule {sp['rule']}; this look selects as rule p does")
    if ck["dataset"] == target:
        raise SystemExit(f"{p}: fitted on {target} itself")
    if target in KBS and sp.get("pca"):
        raise SystemExit(f"{p}: a PCA code has no target basis on a KB's few relations (anchor_kbfit refuses it too)")


def load_source(p, sp, phi):
    model, ck = (G3.load_model if "pca" in sp else AG.load_model)(p, phi)
    if sp["family"] != "free":
        G3.set_phi(model, phi, sp["K"])
    return model, ck


def select_score(model, Q, TY, nt, rows, z_of):
    """fit_read_kd's per-epoch selection score under rule p, for the untrained model, rounded as its curve is."""
    with torch.no_grad():
        return round(float(G2.read_rows(model, Q, TY, nt, list(rows), z_of)[:, :2].mean()), 4)


def maker(arm, model0, sp, nt, phi):
    if arm == "scratch":
        if "pca" not in sp:
            return AG.make_for(sp, nt, phi)
        basis = G3.pca_basis(phi[:sp["K"]], sp["pca"])[:3] if sp["pca"] else None
        return G3.make_for(sp, nt, phi, basis)

    def make():
        m = copy.deepcopy(model0)
        for nm, p in m.named_parameters():
            p.requires_grad_(arm == "warm" or nm in ("log_kappa", "beta_raw"))
        return m
    return make


def brief(rec):
    return {"rho (R@5, FC@5, hit@1)": rec["rho (R@5, FC@5, hit@1)"], "minus_twin0": rec["minus_twin0"]}


def kappa_beta(m):
    return float(torch.exp(m.log_kappa).item()), float(torch.nn.functional.softplus(m.beta_raw).item())


def run(a):
    out_path = Path(a.out)
    torch.set_num_threads(2)
    t0 = time.time()
    ns = sorted({int(x) for x in a.n.split(",")} - {0})
    if ns and ns[0] < 4:
        raise SystemExit("--n: sizes of at least 4 (or 0 for zero-shot only)")
    paths = [Path(p) for p in a.models.split(",")]
    cks = [torch.load(p, map_location="cpu", weights_only=False) for p in paths]
    for p, ck in zip(paths, cks):
        check_source(p, ck, a.target)
    max_len = max(ck["spec"]["max_len"] for ck in cks)
    if a.target in KBS:
        Q, read, pool, info, kb = load_kb(a.target, max_len, t0, bool(ns))
    else:
        Q, read, pool, info, kb = load_passage(a.target, max_len, t0, bool(ns))
    if ns and ns[-1] > len(pool):
        raise SystemExit(f"--n {ns[-1]} is more than the pool's {len(pool)} rows")
    tables = Tables(a.target, kb)
    B_rows = list(read) if a.target in WHOLE else read[1::2]
    RB = AG.Reader(Q, B_rows, 20261002)
    z_of = [A16.zscore(q["score"][:, 0]) for q in Q]
    zT_of = [None] * len(Q)
    perm = {d: np.random.default_rng(DRAW_SEED + d).permutation(len(pool)) for d in range(a.draws)}
    draws = {d: [pool[i] for i in perm[d][:ns[-1]]] for d in range(a.draws)} if ns else {}
    res = {"look": "anchor_few", "target": a.target, "script_sha256": AW.sha(Path(__file__)), "pins": PINS,
           "kb_pins": KB_PINS if a.target in KBS else None, "target_info": info, "read_carve": READ[a.target],
           "read_rows": "whole" if a.target in WHOLE else "half B",
           "pool": {"carve": POOL[a.target], "rows": len(pool)}, "n": ns, "draws": a.draws if ns else 0, "draw_seed": DRAW_SEED,
           "lr": a.lr, "steps": a.steps, "min_epochs": a.min_epochs, "max_epochs": a.max_epochs, "B": RB.base(), "models": {}}
    per_row = {}

    def save():
        res["phrase_tables"] = {str(k): v for k, v in tables.sha.items()}
        out_path.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
        np.savez(out_path.with_suffix(".npz"), **per_row, B_rows=np.asarray(B_rows))

    for jm, (p, ck) in enumerate(zip(paths, cks)):
        t1 = time.time()
        sp = ck["spec"]
        phi = tables.of(sp)
        model0, _ck = load_source(p, sp, phi)
        K = sp["K"]
        nt, TY = AG.types_for(Q, sp, AG.identity_map(K) if sp["tok"] == "A" else None)
        if nt != ck["nt"]:
            raise SystemExit(f"{p}: {nt} tokens on the target, {ck['nt']} in the checkpoint")
        mZ = AG.read(model0, Q, TY, nt, B_rows, z_of, ck["margin"])
        per_row[f"{jm}_zs"] = mZ
        k0, b0 = kappa_beta(model0)
        ent = {"from": ck["dataset"], "name": ck.get("name"), "seed": ck.get("seed"), "hold": ck.get("hold") or ck.get("hold_rel"),
               "spec": sp, "model_file": str(p), "params": int(sum(q.numel() for q in model0.parameters())), "kappa0": k0, "beta0": b0,
               "zs": RB.record(mZ), "arms": {}}
        if a.alt and sp["tok"] == "A":
            _nt, TYn = AG.types_for(Q, sp, np.full(R_TOP + 1, K, dtype=np.int64), B_rows)
            mN = AG.read(model0, Q, TYn, nt, B_rows, z_of, ck["margin"])
            del TYn
            per_row[f"{jm}_zsNR"] = mN
            ent["zs_NR"], ent["zs_NR - zs"] = RB.record(mN, by_type=False), AW3.boot_pair(mN - mZ, RB.W)
            if kb is not None:
                n_rel = min(kb[1], R_TOP)
                phi_s = phi.copy()
                phi_s[:n_rel] = phi[(np.arange(n_rel) + 1) % n_rel]
                G3.set_phi(model0, phi_s, K)
                mS = AG.read(model0, Q, TY, nt, B_rows, z_of, ck["margin"])
                G3.set_phi(model0, phi, K)
                per_row[f"{jm}_zsSHUF"] = mS
                ent["zs_SHUF"], ent["zs_SHUF - zs"] = RB.record(mS, by_type=False), AW3.boot_pair(mS - mZ, RB.W)
        log(f"{p.parent.name}/{p.name} ({ck['dataset']} -> {a.target}, {sp['model']}): zs {ent['zs']['rho (R@5, FC@5, hit@1)']}"
            + (f", NR {ent['zs_NR']['rho (R@5, FC@5, hit@1)']}" if "zs_NR" in ent else "")
            + (f", SHUF {ent['zs_SHUF']['rho (R@5, FC@5, hit@1)']}" if "zs_SHUF" in ent else ""))
        key = p.name if p.name not in res["models"] else f"{p.parent.name}/{p.name}"
        res["models"][key] = ent
        for N in ns:
            n_tr = math.ceil(3 * N / 4)
            epochs = min(a.max_epochs, max(a.min_epochs, math.ceil(a.steps / math.ceil(n_tr / 128))))
            got = {arm: [] for arm in ARMS}
            for d in range(a.draws):
                lab = draws[d][:N]
                tr, sel = lab[:n_tr], lab[n_tr:]
                s_init = select_score(model0, Q, TY, nt, sel, z_of)
                for arm in ARMS:
                    t2 = time.time()
                    model, reads, best_ep, curve, _margin, _bm = AW8.fit_read_kd(Q, TY, nt, tr, sel, {"B": B_rows}, z_of, zT_of,
                                                                                 maker(arm, model0, sp, nt, phi), epochs, DRAW_SEED + d,
                                                                                 "p", a.lr, 0.0, 1.0, 0.0, 1.0)
                    kept = arm != "scratch" and s_init >= curve[best_ep]
                    m = mZ if kept else reads["B"]
                    per_row[f"{jm}_{arm}_{N}_{d}"] = m
                    got[arm].append(m)
                    kp, bt = kappa_beta(model0 if kept else model)
                    e = ent["arms"].setdefault(arm, {}).setdefault(str(N), {"epochs": epochs, "train_rows": len(tr), "select_rows": len(sel),
                                                                             "draws": []})
                    e["draws"].append({"draw": d, "best_epoch": -1 if kept else best_ep, "kept_init": bool(kept), "select_init": s_init,
                                       "select_best": curve[best_ep], "curve": curve, "kappa": kp, "beta": bt,
                                       "B": brief(RB.record(m, by_type=False)), "seconds": round(time.time() - t2, 1)})
                log(f"  N {N} draw {d}: " + ", ".join(
                    f"{arm} {ent['arms'][arm][str(N)]['draws'][-1]['B']['rho (R@5, FC@5, hit@1)'][0]}"
                    + ("=zs" if ent["arms"][arm][str(N)]["draws"][-1]["kept_init"] else "") for arm in ARMS))
            mean = {arm: np.mean(np.stack(got[arm]), 0) for arm in ARMS}
            for arm in ARMS:
                e = ent["arms"][arm][str(N)]
                e["mean_of_draws"] = brief(RB.record(mean[arm], by_type=False))
                e["draw_rho_R@5"] = [x["B"]["rho (R@5, FC@5, hit@1)"][0] for x in e["draws"]]
                e[f"{arm} - zs"] = AW3.boot_pair(mean[arm] - mZ, RB.W)
            ent["arms"]["warm"][str(N)]["warm - scratch"] = AW3.boot_pair(mean["warm"] - mean["scratch"], RB.W)
            ent["arms"]["warm"][str(N)]["warm - kappa"] = AW3.boot_pair(mean["warm"] - mean["kappa"], RB.W)
            ent["seconds"] = round(time.time() - t1, 1)
            save()
        del TY
        ent["seconds"] = round(time.time() - t1, 1)
        save()
    res["seconds"] = round(time.time() - t0, 1)
    save()
    log(f"done in {res['seconds']}s")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, choices=tuple(READ))
    ap.add_argument("--models", required=True)
    ap.add_argument("--n", default="64,256,1024")
    ap.add_argument("--draws", type=int, default=3)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--steps", type=int, default=96)
    ap.add_argument("--min_epochs", type=int, default=12)
    ap.add_argument("--max_epochs", type=int, default=200)
    ap.add_argument("--alt", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.draws < 1 or a.steps < 1 or a.min_epochs < 1 or a.max_epochs < a.min_epochs:
        raise SystemExit("--draws, --steps and --min_epochs at least 1, --max_epochs at least --min_epochs")
    if "few" not in Path(a.out).name:
        raise SystemExit("--out must carry 'few' in its file name")
    check_pins()
    AC.bind()
    run(a)


if __name__ == "__main__":
    main()
