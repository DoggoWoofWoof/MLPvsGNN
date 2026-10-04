"""Design look (untracked; not a result and not filed): ensembles of the anchor relation models (seeds, or families).

In-domain the operator families sit inside one another's bootstrap (paired intervals about +-0.09 rho on 2wiki B), and
one fit moves by about 0.2 across seeds and near-identical settings, so a single fit is a noisy draw. This look reads an
ensemble: the members' bonuses averaged on the twin's score,
    s = z + (1/M) sum_m exp(log_kappa_m) * boost_m       (each member with its own types, kappa, beta and bonus)
with the members' common protect margin applied after the average, as anchor_walk6.read_rule applies it to one model.
Each member is also read alone (its read equals read_rule's). Reads per group, with anchor_nn's semantics:
    BASE   in-domain the fit's own map (ID; MASK with a hold); on another dataset free models by string, text and gen
           models by their own text codes on the target's table
    NR     every phrase 'other' (a T0 member reads its BASE: it has no phrase)
    REV    in-domain, when a member is a held text or gen model: those members read the held fold by their own text
           codes, the rest their BASE
A group is name=a.pt+b.pt+...; groups are separated by ';'. B rows are x1[1::2], as every anchor look.

    python outputs/mp_approx_2wiki_anchor/host/anchor_ens.py --target 2wiki --groups "a1=x.pt+y.pt+z.pt;..." --out PATH
"""
import os
import sys

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "4"

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import anchor_nn as NN  # noqa: E402

G3, AG, KB = NN.G3, NN.AG, NN.KB
AW6, A16, AW, AW3, G2 = AG.AW6, AG.A16, AG.AW, AG.AW3, AG.G2
R_TOP = AG.R_TOP
ANCHOR_NN_SHA = "e3bf4e1f1598ac4d62eadc58cefbcc1056d27e8d812c7e11c33c8b31ccb8678c"
log = AG.log


def parse_groups(s):
    out = []
    for part in s.split(";"):
        name, sep, members = part.partition("=")
        if not sep or not name or not members:
            raise SystemExit(f"--groups: '{part}' is not name=a.pt+b.pt")
        out.append((name, [Path(p) for p in members.split("+")]))
    if len({n for n, _ in out}) != len(out):
        raise SystemExit("--groups: repeated group name")
    return out


def member_maps(ck, in_domain, vocab_t):
    """{read: (rank map or None, read with the target's own text table)} for one member."""
    sp, K = ck["spec"], ck["spec"]["K"]
    if sp["tok"] == "T0":
        return {"BASE": (None, False), "NR": (None, False)}
    held = NN.held_of(ck)
    m = {}
    if in_domain:
        m["BASE"] = (NN.exact_map(K, vocab_t, ck["vocab_w2"], held), False)
        if sp["family"] != "free" and held.any():
            m["REV"] = (AG.identity_map(K), False)
    elif sp["family"] == "free":
        m["BASE"] = (NN.exact_map(K, vocab_t, ck["vocab_w2"], held), False)
    else:
        m["BASE"] = (AG.identity_map(K), True)
    m["NR"] = (np.full(R_TOP + 1, K, dtype=np.int64), False)
    return m


def read_members(members, Q, rows, z_of, margin):
    """members: [(model, nt, TY)]. The ensemble's metrics and each member's, on rows."""
    ens, solo = [], [[] for _ in members]
    with torch.no_grad():
        for s0 in range(0, len(rows), 256):
            rr = rows[s0:s0 + 256]
            S, z = [], None
            for model, nt, TY in members:
                PX = G2.LG.pack_gate(rr, Q, TY, z_of, nt)
                s, _gold = G2.scores(model, PX)
                if z is None:
                    z = PX[0][9]
                elif not torch.equal(z, PX[0][9]):
                    raise SystemExit("members packed different twin scores")
                S.append(s)
            fin = torch.isfinite(z)
            z0 = torch.where(fin, z, torch.zeros_like(z))
            bonus = torch.stack([torch.where(fin, s - z0, torch.zeros_like(s)) for s in S]).mean(0)
            s_ens = torch.where(fin, z + bonus, z)
            ar = torch.arange(len(rr))
            if margin is not None:
                first = G2.top1(z)
                zz = z.clone()
                zz[ar, first] = float("-inf")
                prot = (z[ar, first] - zz.max(1).values) >= margin
            for k, s in enumerate([s_ens] + S):
                s = s.clone()
                if margin is not None:
                    s[ar[prot], first[prot]] = float("inf")
                dst = ens if k == 0 else solo[k - 1]
                for bi, i in enumerate(rr):
                    dst.append(A16.metrics_of(s[bi, :Q[i]["n"]].numpy(), Q[i]["gold"], Q[i]["gt"]))
    return np.asarray(ens), [np.asarray(x) for x in solo]


def run(a):
    out_path = Path(a.out)
    torch.set_num_threads(2)
    t0 = time.time()
    groups = parse_groups(a.groups)
    cks = {p: torch.load(p, map_location="cpu", weights_only=False) for _n, ps in groups for p in ps}
    max_len = max(ck["spec"]["max_len"] for ck in cks.values())
    Q, part, vocab_t, _phi_t, own_table, info = NN.load_target(a.target, max_len, False, t0)
    B_rows = part["x1"][1::2]
    RB = AG.Reader(Q, B_rows, 20261002)
    z_of = [A16.zscore(q["score"][:, 0]) for q in Q]
    res = {"look": "anchor_ens", "target": a.target, "script_sha256": AW.sha(Path(__file__)),
           "pins": {"anchor_nn": ANCHOR_NN_SHA, "anchor_gen3": NN.ANCHOR_GEN3_SHA, "anchor_gen": G3.ANCHOR_GEN_SHA,
                    "anchor_gen2": G3.ANCHOR_GEN2_SHA, "anchor_gen_kb": NN.ANCHOR_GEN_KB_SHA},
           "target_info": info, "B": RB.base(), "groups": {}}
    per_row = {}
    for gname, ps in groups:
        t1 = time.time()
        margins = {cks[p]["margin"] for p in ps}
        if len(margins) != 1:
            raise SystemExit(f"{gname}: the members' protect margins differ ({margins})")
        margin = margins.pop()
        mem = []
        for p in ps:
            ck = cks[p]
            in_domain = ck["dataset"] == a.target
            phi_tr, kind = (None, None) if ck["spec"]["tok"] == "T0" else NN.train_table(ck)
            model, _ck = NN.load_any(p, phi_tr)
            model.eval()
            mem.append({"p": p, "ck": ck, "model": model, "phi_tr": phi_tr, "kind": kind, "maps": member_maps(ck, in_domain, vocab_t),
                        "in_domain": in_domain})
        names = ["BASE", "NR"] + (["REV"] if any("REV" in m["maps"] for m in mem) else [])
        g = {"members": [{"file": str(m["p"]), "from": m["ck"]["dataset"], "name": m["ck"].get("name"), "seed": m["ck"].get("seed"),
                          "hold": m["ck"].get("hold"), "model": m["ck"]["spec"]["model"], "K": m["ck"]["spec"]["K"],
                          "reads_REV": "REV" in m["maps"]} for m in mem],
             "margin": margin, "reads": {}}
        cache = {}
        for nm in names:
            members = []
            for m in mem:
                rm, own = m["maps"].get(nm, m["maps"]["BASE"])
                sp = m["ck"]["spec"]
                if sp["family"] != "free" and sp["tok"] != "T0":
                    G3.set_phi(m["model"], own_table(m["kind"]) if own else m["phi_tr"], sp["K"])
                key = (sp["tok"], sp["K"], sp["max_len"], None if rm is None else hashlib.sha256(rm.tobytes()).hexdigest())
                if key not in cache:
                    cache[key] = AG.types_for(Q, sp, rm, B_rows)[1]
                members.append((m["model"], m["ck"]["nt"], cache[key]))
            ens, solo = read_members(members, Q, B_rows, z_of, margin)
            e = RB.record(ens, by_type=False)
            mean_solo = np.mean(solo, 0)
            e["members_rho"] = [RB.rho(s) for s in solo]
            e["mean_member_rho"] = RB.rho(mean_solo)
            e["minus_mean_member"] = AW3.boot_pair(ens - mean_solo, RB.W)
            g["reads"][nm] = e
            per_row[f"{gname}_B_{nm}"] = ens
            for k, s in enumerate(solo):
                per_row[f"{gname}_B_{nm}_m{k}"] = s
        g["seconds"] = round(time.time() - t1, 1)
        res["groups"][gname] = g
        log(f"{gname} ({len(ps)} members): " + ", ".join(f"{k} {v['rho (R@5, FC@5, hit@1)']} (members {v['mean_member_rho']})" for k, v in g["reads"].items()))
        out_path.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
        np.savez(out_path.with_suffix(".npz"), **per_row, B_rows=np.asarray(B_rows))
    res["seconds"] = round(time.time() - t0, 1)
    out_path.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    log(f"done in {res['seconds']}s")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, choices=AG.DATASETS)
    ap.add_argument("--groups", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if "ens" not in Path(a.out).name:
        raise SystemExit("--out must carry 'ens' in its file name")
    if AW.sha(Path(NN.__file__)) != ANCHOR_NN_SHA:
        raise SystemExit("anchor_nn.py is not the pinned file")
    if AW.sha(Path(G3.__file__)) != NN.ANCHOR_GEN3_SHA:
        raise SystemExit("anchor_gen3.py is not the pinned file")
    if AW.sha(Path(AG.__file__)) != G3.ANCHOR_GEN_SHA or AW.sha(Path(G3.G2N.__file__)) != G3.ANCHOR_GEN2_SHA:
        raise SystemExit("anchor_gen.py or anchor_gen2.py is not the pinned file")
    if AW.sha(Path(KB.__file__)) != NN.ANCHOR_GEN_KB_SHA:
        raise SystemExit("anchor_gen_kb.py is not the pinned file")
    AG.check_pins()
    run(a)


if __name__ == "__main__":
    main()
