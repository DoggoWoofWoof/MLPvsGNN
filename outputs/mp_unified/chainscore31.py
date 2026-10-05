"""S6 part 15 (chainscore31.py): the feature groups, each model retrained without them.

The MLP and the GNN use the same features. Part 12's feature-group Shapley (chainscore28) removes a group from a
trained model's inputs at read time: it measures what that model leans on, not what a model trained without the group
would do on a new graph. Here each model is retrained with a group's standardized columns set to 0 in every batch
(train, select and read alike; 0 is the group's training mean, so the model sees no variation in it), through
chainscore30's runner: everything else is the family's own run. A group whose presence costs a model on webqsp is a
cause of that model's zero-shot gap; a group that helps one model and costs the other is where the static and the
message-passing regimes generalise differently.

Groups: node (chainscore28's NODE_GROUPS: text, dist, struct, walk) and pair (PAIR_GROUPS: text, shape, evidence, pop).
The prior's offset (log C from the build, lb) is not a feature column and stays.
Arms (MASK: the groups kept, bit b for the b-th group in that order; 15 keeps all four):
    kn-MASK  the node groups in MASK kept, every pair group kept
    kp-MASK  the pair groups in MASK kept, every node group kept
Classes, seeds 0, 1, 2:
    mlp (chainscore24 k5, CPU) and ena (chainscore25 e-b-lb-pq, GPU): every MASK in 0..14 (exact Shapley);
    rgu (chainscore29 rgu, CPU) and rga (chainscore29 rga, GPU): the four MASKs with one bit off (leave one out).
    Each arm's control: for the CPU classes, chainscore30's base runs (the family's run unchanged, on the CPU, one
    thread); for the GPU classes, the parts 9 / 13 runs.

Grade (R@5 per question; seeds averaged per row over the seeds every arm of the set has; paired row bootstrap):
    S1 exact Shapley (mlp, ena), per kind and read: phi_g = sum over S without g of |S|! (3 - |S|)! / 4! [v(S + g) -
       v(S)], v(S) the arm keeping S (v(all) the control), per row; efficiency (sum phi = v(all) - v(none)) checked.
       A group HELPS a class on a read if phi's CI is above 0, COSTS it if below.
    S2 leave one out (every class): v(all) - v(all - g); DROP carries for g if the arm without g, against the control,
       meets chainscore30's CARRY rule (ABOVE on webqsp selectf + fit and not BELOW on metaqa, or AT and ABOVE).
    S3 the regimes against each other: phi_g(ena) - phi_g(mlp) (S1), and [rga - rga without g] - [rgu - rgu without
       g] (S2), per read, bootstrapped by rows.
    2wiki and hotpotqa are reported beside the KB reads, without a rule.
No webqsp label is used in training or selection; webqsp train_holdout and test are not touched. Smokes write under
smoke30/ and are never graded.

Rules fixed at 11:20 on 5 Oct 2026, with part 14's, before any number of parts 12 to 15 was looked at.
"""
import os
import sys

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")      # chainscore19's count
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import chainscore28 as C28  # noqa: E402
import chainscore30 as C30  # noqa: E402

SEED = C30.SEED
SEEDS = C30.SEEDS
KB_READS, P_SECOND, PRIMARY, GUARD = C30.KB_READS, C30.P_SECOND, C30.PRIMARY, C30.GUARD
NODE, PAIR = tuple(C28.NODE_GROUPS), tuple(C28.PAIR_GROUPS)
FULL = 15
CLASSES = {"mlp": "shapley", "ena": "shapley", "rgu": "loo", "rga": "loo"}
PAIRS = (("ena", "mlp", "S1"), ("rga", "rgu", "S2"))
OUT_PREFIX = "outputs/mp_unified/lean/cs31-"
log, sha = C30.log, C30.sha


def arms_of(cls):
    masks = range(FULL) if CLASSES[cls] == "shapley" else [FULL ^ (1 << b) for b in range(4)]
    return [f"kn-{m}" for m in masks] + [f"kp-{m}" for m in masks]


def dropped(arm):
    kind, m = arm.split("-")
    m = int(m)
    if kind not in ("kn", "kp") or not 0 <= m < FULL:
        raise SystemExit(f"unknown arm {arm}")
    groups = NODE if kind == "kn" else PAIR
    drop = [g for b, g in enumerate(groups) if not (m >> b) & 1]
    return (drop, []) if kind == "kn" else ([], drop)


def patch_drop(nd, pd):
    """Zero the dropped groups' standardized columns in every batch the family makes (chainscore21's make_batch21)."""
    ncols = sorted(c for g in nd for c in C28.NCOLS[g])
    pcols = sorted(c for g in pd for c in C28.PCOLS[g])
    seen = {"batches": 0, "pair_batches": 0, "node_cols": ncols, "pair_cols": pcols}

    def post(bt, _args):
        if ncols:
            bt["Xn"][:, ncols] = 0.0
        if pcols and bt.get("Xp") is not None:
            bt["Xp"][:, pcols] = 0.0
            seen["pair_batches"] += 1
        seen["batches"] += 1

    C30.wrap_batch(post)
    return seen


def train_cmd(a, rest):
    if a.arm not in arms_of(a.cls):
        raise SystemExit(f"class {a.cls} has arms {arms_of(a.cls)}")
    look, arm, _f, device = C30.CLASSES[a.cls][:4]
    nd, pd = dropped(a.arm)
    seen = patch_drop(nd, pd)
    js = C30.run_family(look, arm, device, rest, a.threads, "part15",
                        {"cls": a.cls, "arm15": a.arm, "drop_node": nd, "drop_pair": pd, "zeroed": seen}, a.smoke,
                        prefix=OUT_PREFIX)
    if not seen["batches"] or (pd and not seen["pair_batches"]):
        raise SystemExit("no batch had its columns zeroed")
    log(f"  part 15 {a.cls} {a.arm} (drops {nd + pd}) s{js['seed']}: " + ", ".join(
        f"{n} {js['reads'][n]['mean']}" for n in js["reads"]))
    return 0


# ── grade ────────────────────────────────────────────────────────────────────────────────────────────────────────


def shap_weights(n=4):
    return [math.factorial(s) * math.factorial(n - s - 1) / math.factorial(n) for s in range(n)]


def shapley(v):
    """v: {mask: per-row values} over all 16 masks of four players -> per-row phi (4 x rows)."""
    w = shap_weights()
    phi = []
    for g in range(4):
        tot = 0.0
        for m in range(16):
            if (m >> g) & 1:
                continue
            tot = tot + w[bin(m).count("1")] * (v[m | (1 << g)] - v[m])
        phi.append(tot)
    return np.asarray(phi)


def load_rows(p):
    js = json.loads(Path(p).read_text(encoding="utf-8"))
    return js, np.load(js["rows_out"])


def grade_cmd(a):
    rng = np.random.default_rng(SEED + 31)
    res = {"look": "chainscore31-grade", "script_sha256": sha(__file__), "primary": PRIMARY, "guard": GUARD,
           "groups": {"node": list(NODE), "pair": list(PAIR)}, "S1": {}, "S2": {}, "S3": {}, "decision": {}}
    arms = {}
    for p in a.run:
        js, rows = load_rows(p)
        part = js.get("part15")
        if not part or js.get("smoke"):
            raise SystemExit(f"{p}: not a part 15 run")
        if js["device"] != C30.CLASSES[part["cls"]][3]:
            raise SystemExit(f"{p}: class {part['cls']} trains on {C30.CLASSES[part['cls']][3]}")
        arms.setdefault(part["cls"], {}).setdefault(part["arm15"], {})[int(js["seed"])] = rows
    for p in a.control:
        js, rows = load_rows(p)
        if js.get("smoke") or "part15" in js:
            raise SystemExit(f"{p}: not a control")
        p14 = js.get("part14")
        if p14:
            cls = p14["cls"]
            if p14["axis"] != "base" or cls not in ("mlp", "rgu") or js["device"] != "cpu":
                raise SystemExit(f"{p}: a part 14 control is a CPU base run of mlp or rgu")
        else:
            cls = C30.cls_of(js)
            if cls not in ("ena", "rga") or js["device"] != "cuda":
                raise SystemExit(f"{p}: a family control is a GPU run of ena or rga")
        arms.setdefault(cls, {}).setdefault(f"kn-{FULL}", {})[int(js["seed"])] = rows
        arms[cls].setdefault(f"kp-{FULL}", {})[int(js["seed"])] = rows
    vals = {}
    for cls, byA in sorted(arms.items()):
        need = [f"kn-{FULL}"] + arms_of(cls)
        missing = [x for x in need if x not in byA]
        if missing:
            res["S2"][cls] = {"missing": missing}
            continue
        seeds = sorted(set.intersection(*[set(byA[x]) for x in need]))
        if not seeds:
            res["S2"][cls] = {"error": "no seed in every arm"}
            continue
        for x in need + [f"kp-{FULL}"]:
            for n in KB_READS + P_SECOND:
                if all(n in byA[x][s].files for s in seeds):
                    vals[(cls, x, n)] = C30.seed_rows({s: byA[x][s] for s in seeds}, n)
        loo = {"seeds": seeds}
        for kind, groups in (("kn", NODE), ("kp", PAIR)):
            for b, g in enumerate(groups):
                arm = f"{kind}-{FULL ^ (1 << b)}"
                e = {}
                for n in KB_READS + P_SECOND:
                    if (cls, arm, n) in vals and (cls, f"{kind}-{FULL}", n) in vals:
                        e[n] = C30.delta(vals[(cls, f"{kind}-{FULL}", n)], vals[(cls, arm, n)], rng, n in KB_READS)
                if PRIMARY in e and GUARD in e:
                    # the arm without g against the control: the negative of the deltas above
                    lp = {"ABOVE": "BELOW", "BELOW": "ABOVE", "AT": "AT"}
                    e["DROP_CARRIES"] = C30.carry(lp[e[PRIMARY][3]], lp[e[GUARD][3]])
                loo[f"{kind}:{g}"] = e
        res["S2"][cls] = loo
        if CLASSES[cls] == "shapley":
            ent = {"seeds": seeds}
            for kind, groups in (("kn", NODE), ("kp", PAIR)):
                for n in KB_READS + P_SECOND:
                    if not all((cls, f"{kind}-{m}", n) in vals for m in range(16)):
                        continue
                    v = {m: vals[(cls, f"{kind}-{m}", n)] for m in range(16)}
                    phi = shapley(v)
                    gap = float(np.abs(phi.sum(0) - (v[FULL] - v[0])).max())
                    if gap > 1e-9:
                        raise SystemExit(f"efficiency fails by {gap}")
                    vals[(cls, f"phi-{kind}", n)] = phi
                    ent[f"{kind}:{n}"] = {g: C30.delta(phi[b], np.zeros_like(phi[b]), rng, n in KB_READS)
                                          for b, g in enumerate(groups)}
                    ent[f"{kind}:{n}"]["v_all"] = round(float(v[FULL].mean()), 5)
                    ent[f"{kind}:{n}"]["v_none"] = round(float(v[0].mean()), 5)
            res["S1"][cls] = ent
    for g_, s_, how in PAIRS:
        e = {}
        for kind, groups in (("kn", NODE), ("kp", PAIR)):
            for n in KB_READS + P_SECOND:
                for b, g in enumerate(groups):
                    if how == "S1":
                        ks = [(g_, f"phi-{kind}", n), (s_, f"phi-{kind}", n)]
                        if all(k in vals for k in ks):
                            e[f"{kind}:{g}:{n}"] = C30.delta(vals[ks[0]][b], vals[ks[1]][b], rng)
                    else:
                        arm = f"{kind}-{FULL ^ (1 << b)}"
                        ks = [(c, x, n) for c in (g_, s_) for x in (f"{kind}-{FULL}", arm)]
                        if all(k in vals for k in ks):
                            e[f"{kind}:{g}:{n}"] = C30.delta(vals[ks[0]] - vals[ks[1]], vals[ks[2]] - vals[ks[3]], rng)
        res["S3"][f"{g_}-{s_}"] = e or "missing"
    for cls in CLASSES:
        s2 = res["S2"].get(cls, {})
        res["decision"][cls] = {"drop_carries": sorted(k for k, e in s2.items() if isinstance(e, dict)
                                                       and e.get("DROP_CARRIES"))}
    return res


def selftest():
    import torch
    assert len(arms_of("mlp")) == 30 and len(arms_of("rgu")) == 8
    assert dropped("kn-0") == (list(NODE), []) and dropped("kp-14") == ([], [PAIR[0]])
    assert dropped("kn-13") == ([NODE[1]], [])
    rng = np.random.default_rng(0)
    # Shapley of an additive game is each player's own value, and it is efficient
    base = rng.normal(size=(4, 50))
    v = {m: sum(base[g] for g in range(4) if (m >> g) & 1) + 0.0 * base[0] for m in range(16)}
    phi = shapley(v)
    assert np.allclose(phi, base) and np.allclose(phi.sum(0), v[15] - v[0])
    # with an interaction, the pair splits it
    v2 = {m: v[m] + (((m & 3) == 3) * 1.0) for m in range(16)}
    phi2 = shapley(v2)
    assert np.allclose(phi2[0] - phi[0], 0.5) and np.allclose(phi2[2], phi[2])
    assert abs(sum(math.comb(3, s) * w for s, w in enumerate(shap_weights())) - 1.0) < 1e-12
    # the batch patch zeroes exactly the dropped columns
    import chainscore21 as C21
    orig = C21.make_batch21
    C21.make_batch21 = lambda *a_, **k_: {"Xn": torch.ones(5, len(C30.C19.NODEF)),
                                          "Xp": torch.ones(7, len(C30.C19.PAIRF))}
    seen = patch_drop(["dist"], ["pop", "text"])
    bt = C21.make_batch21(None)
    zn = sorted(np.flatnonzero((bt["Xn"] == 0).all(0).numpy()).tolist())
    zp = sorted(np.flatnonzero((bt["Xp"] == 0).all(0).numpy()).tolist())
    assert zn == sorted(C28.NCOLS["dist"]) and zp == sorted(C28.PCOLS["pop"] + C28.PCOLS["text"])
    assert seen["batches"] == 1 and seen["pair_batches"] == 1
    C21.make_batch21 = orig
    print("selftest ok")
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    rest = []
    if "--" in argv:
        i = argv.index("--")
        argv, rest = argv[:i], argv[i + 1:]
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd")
    t = sub.add_parser("train", help="a family's train command with feature groups zeroed (after --)")
    t.add_argument("--cls", required=True, choices=list(CLASSES))
    t.add_argument("--arm", required=True)
    t.add_argument("--threads", type=int, default=1)
    t.add_argument("--smoke", action="store_true")
    g = sub.add_parser("grade")
    g.add_argument("--run", action="append", default=[])
    g.add_argument("--control", action="append", default=[])
    g.add_argument("--out", required=True)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.cmd == "train":
        return train_cmd(a, rest)
    if rest:
        raise SystemExit("only train takes arguments after --")
    if a.cmd != "grade":
        ap.print_help()
        return 2
    res = grade_cmd(a)
    p = Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(res, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)),
                   encoding="utf-8")
    os.replace(tmp, p)
    log(f"wrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
