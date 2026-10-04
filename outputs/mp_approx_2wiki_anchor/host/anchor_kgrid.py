"""Design look (untracked; not a result and not filed): with N labelled questions on a new graph, is choosing the
bonus's scale from them enough to make a relation model safe there?

anchor_few's kappa arm trains log_kappa and beta by gradient (lr 3e-3, about 96 steps): log_kappa moves by about 0.1 at
most, so it cannot switch off a bonus that hurts, and it kept its initial value in most draws. Here the scale is chosen
from a grid instead: the model is read at log_kappa0 + d for every d in --deltas (-inf is the bonus off, the twin's
ranking, rho 0). Each row is packed and scored once at every d by kcal.sweep. On the N labelled rows the d with the
highest select score is kept (anchor_few's: the mean of R@5 and FC@5 under rule p with every row's twin rank 1 placed
first, as G2.read_rows does, rounded as fit_read_kd's curve is), ties going to the d nearest 0, then the smaller. The
codes, beta and every other parameter stay as saved. The draws are anchor_few's (the first N rows of the pool carve's
permutation with seed 20261100 + draw); all N rows select, since nothing trains. Every read is on anchor_few's read
carve (x1's half B; webqsp's select, whole) with the checkpoint's margin, as anchor_few reads. At d = 0 the read is
anchor_few's zero-shot read, checked row for row against AG.read, and on the first draw of each N the select score at
d = 0 is checked against anchor_few.select_score. Also recorded: the read at every d, and the oracle d on the read rows
(the best the grid reaches; no rule may use it). Models with a gate or a spread exponent are refused, as kcal refuses
them. Nothing reads a neighbour's score or state.

    python outputs/mp_approx_2wiki_anchor/host/anchor_kgrid.py --target musique --models a.pt,b.pt --out PATH_with_kgrid
"""
import os
import sys

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "4"

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "mp_approx_kb_anchor" / "host"))
import anchor_few as AF  # noqa: E402
import kcal as KC  # noqa: E402

AC, G3, AG, G2 = AF.AC, AF.G3, AF.AG, AF.G2
AW, AW3, A16 = AF.AW, AF.AW3, AF.A16
PINS = {"anchor_few": "1bcb67a9fffba58923e8589a730ff549f59b0757aee1c599edddfb759587d2bd",
        "kcal": "96cc375a493beb5c4f208722ce9fb3e25ebf0bd8d6d55058f6843efc4c142b38"}
DELTAS = "-inf,-6,-5,-4,-3,-2.5,-2,-1.5,-1,-0.5,0,0.5,1"
OFF = float("-inf")
log = AG.log


def dkey(d):
    return "off" if d == OFF else f"{d:g}"


def pick(scores, deltas):
    """The d with the highest score; ties to the d nearest 0, then the smaller d."""
    best = max(scores[d] for d in deltas)
    return min((d for d in deltas if scores[d] == best), key=lambda d: (abs(d), d))


def run(a):
    out_path = Path(a.out)
    torch.set_num_threads(2)
    t0 = time.time()
    deltas = [float(x) for x in a.deltas.split(",")]
    if len(set(deltas)) != len(deltas) or 0.0 not in deltas:
        raise SystemExit("--deltas: distinct values, 0 among them")
    ns = sorted({int(x) for x in a.n.split(",")})
    if not ns or ns[0] < 4:
        raise SystemExit("--n: sizes of at least 4")
    paths = [Path(p) for p in a.models.split(",") if p]
    if len(set(paths)) != len(paths):
        raise SystemExit("--models: repeated model")
    cks = [torch.load(p, map_location="cpu", weights_only=False) for p in paths]
    for p, ck in zip(paths, cks):
        AF.check_source(p, ck, a.target)
    max_len = max(ck["spec"]["max_len"] for ck in cks)
    loader = AF.load_kb if a.target in AF.KBS else AF.load_passage
    Q, read, pool, info, kb = loader(a.target, max_len, t0, True)
    if ns[-1] > len(pool):
        raise SystemExit(f"--n {ns[-1]} is more than the pool's {len(pool)} rows")
    tables = AF.Tables(a.target, kb)
    B_rows = list(read) if a.target in AF.WHOLE else read[1::2]
    RB = AG.Reader(Q, B_rows, 20261002)
    z_of = [A16.zscore(q["score"][:, 0]) for q in Q]
    perm = {d: np.random.default_rng(AF.DRAW_SEED + d).permutation(len(pool)) for d in range(a.draws)}
    draws = {d: [pool[i] for i in perm[d][:ns[-1]]] for d in range(a.draws)}
    L_rows = sorted({r for v in draws.values() for r in v})
    pos = {r: i for i, r in enumerate(L_rows)}
    res = {"look": "anchor_kgrid", "target": a.target, "script_sha256": AW.sha(Path(__file__)), "pins": PINS,
           "few_pins": AF.PINS, "kb_pins": AF.KB_PINS if a.target in AF.KBS else None, "target_info": info,
           "read_carve": AF.READ[a.target], "read_rows": "whole" if a.target in AF.WHOLE else "half B",
           "pool": {"carve": AF.POOL[a.target], "rows": len(pool)}, "labelled_rows": len(L_rows), "n": ns, "draws": a.draws,
           "draw_seed": AF.DRAW_SEED, "deltas": [dkey(d) for d in deltas], "B": RB.base(), "models": {}}
    per_row = {}

    def save():
        res["phrase_tables"] = {str(k): v for k, v in tables.sha.items()}
        out_path.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
        np.savez(out_path.with_suffix(".npz"), **per_row, B_rows=np.asarray(B_rows), L_rows=np.asarray(L_rows))

    for jm, (p, ck) in enumerate(zip(paths, cks)):
        t1 = time.time()
        sp = ck["spec"]
        phi = tables.of(sp)
        model0, _ck = AF.load_source(p, sp, phi)
        if isinstance(model0, G2.LG.Gate) or hasattr(model0, "gamma"):
            raise SystemExit(f"{p}: a gated or spread-scaled model has no single scale")
        nt, TY = AG.types_for(Q, sp, AG.identity_map(sp["K"]) if sp["tok"] == "A" else None)
        if nt != ck["nt"]:
            raise SystemExit(f"{p}: {nt} tokens on the target, {ck['nt']} in the checkpoint")
        margin = ck["margin"]
        mZ = AG.read(model0, Q, TY, nt, B_rows, z_of, margin)
        _sb, mb = KC.sweep(model0, Q, TY, nt, B_rows, z_of, margin, deltas, True)
        if not np.array_equal(mb[0.0], mZ):
            raise SystemExit(f"{p}: kcal.sweep at d = 0 is not anchor_few's zero-shot read")
        _sl, ml = KC.sweep(model0, Q, TY, nt, L_rows, z_of, 0.0, deltas, True)
        per_row[f"{jm}_zs"] = mZ
        for d in deltas:
            per_row[f"{jm}_d{KC.dtag(d)}_B"] = mb[d]
        b_score = {d: round(float(mb[d][:, :2].mean()), 4) for d in deltas}
        d_or = pick(b_score, deltas)
        ent = {"from": ck["dataset"], "name": ck.get("name"), "seed": ck.get("seed"), "spec": sp, "model_file": str(p),
               "params": int(sum(q.numel() for q in model0.parameters())), "kappa0": float(torch.exp(model0.log_kappa).item()),
               "beta0": float(torch.nn.functional.softplus(model0.beta_raw).item()), "margin": None if margin is None else float(margin),
               "zs": RB.record(mZ), "by_d": {dkey(d): AF.brief(RB.record(mb[d], by_type=False)) for d in deltas},
               "oracle": {"d": dkey(d_or), "B_score": b_score[d_or], "B": AF.brief(RB.record(mb[d_or], by_type=False))}, "kgrid": {}}
        for N in ns:
            got, e = [], {"draws": []}
            for d in range(a.draws):
                lab = draws[d][:N]
                idx = [pos[r] for r in lab]
                sc = {x: round(float(ml[x][idx, :2].mean()), 4) for x in deltas}
                if d == 0:
                    ref = AF.select_score(model0, Q, TY, nt, lab, z_of)
                    if sc[0.0] != ref:
                        raise SystemExit(f"{p}: the select score at d = 0 is {sc[0.0]}, anchor_few's {ref}")
                ds = pick(sc, deltas)
                got.append(mb[ds])
                per_row[f"{jm}_kgrid_{N}_{d}"] = mb[ds]
                e["draws"].append({"draw": d, "d": dkey(ds), "select": {dkey(x): sc[x] for x in deltas},
                                   "B": AF.brief(RB.record(mb[ds], by_type=False))})
            mean = np.mean(np.stack(got), 0)
            e["mean_of_draws"] = AF.brief(RB.record(mean, by_type=False))
            e["draw_rho_R@5"] = [x["B"]["rho (R@5, FC@5, hit@1)"][0] for x in e["draws"]]
            e["d_chosen"] = [x["d"] for x in e["draws"]]
            e["kgrid - zs"] = AW3.boot_pair(mean - mZ, RB.W)
            if OFF in deltas:
                e["kgrid - off"] = AW3.boot_pair(mean - mb[OFF], RB.W)
            ent["kgrid"][str(N)] = e
        del TY
        ent["seconds"] = round(time.time() - t1, 1)
        key = p.name if p.name not in res["models"] else f"{p.parent.name}/{p.name}"
        res["models"][key] = ent
        log(f"{p.parent.name}/{p.name} ({ck['dataset']} -> {a.target}, {sp['model']}): zs {ent['zs']['rho (R@5, FC@5, hit@1)']}, "
            f"oracle d {ent['oracle']['d']} {ent['oracle']['B']['rho (R@5, FC@5, hit@1)']}; "
            + "; ".join(f"N {N}: d {ent['kgrid'][str(N)]['d_chosen']} rho {ent['kgrid'][str(N)]['mean_of_draws']['rho (R@5, FC@5, hit@1)']}"
                        for N in ns))
        save()
    res["seconds"] = round(time.time() - t0, 1)
    save()
    log(f"done in {res['seconds']}s")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, choices=tuple(AF.READ))
    ap.add_argument("--models", required=True)
    ap.add_argument("--n", default="16,64,256,1024")
    ap.add_argument("--draws", type=int, default=3)
    ap.add_argument("--deltas", default=DELTAS)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.draws < 1:
        raise SystemExit("--draws: at least 1")
    if "kgrid" not in Path(a.out).name:
        raise SystemExit("--out must carry 'kgrid' in its file name")
    for mod, key in ((AF, "anchor_few"), (KC, "kcal")):
        if AW.sha(Path(mod.__file__)) != PINS[key]:
            raise SystemExit(f"{mod.__file__} is not the pinned file")
    AF.check_pins()
    AC.bind()
    run(a)


if __name__ == "__main__":
    main()
