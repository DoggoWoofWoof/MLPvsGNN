"""NMD, the names at the next hop (docs/NMD_NAME_LINK_DIAG.md): do the passages' own names, read without the encoder,
link the next-hop golds that PKO's links leave unlinked to their anchor, and do they single them out from the rows of
the same rrf rank?

The names are U2r's (docs/U2R_NAME_RULE.md; outputs/u2r/<dataset>/names.npz, read only, its sha256 checked against its
names.json and its nodes against the served package's). CN_u joins two passages that share an untitled name of df 2 to
25, with weight the sum of 1/df over the names they share; CN_all (any name) is read beside. U2r graded these links as
pool links (E1 was chosen; N1 = IS + CN_u lost 2wiki's pools). NMD reads them at the next hop instead. The A cases are
L2D's (LTA's and LTH's: lthcols.question and a_case, imported, not edited), and a target is unlinked when PKO's links do
not join it to its anchor (L2D's depth: either direction, every family). Fit carves only (LTW's three); no vectors and
no fits. NMD writes only under outputs/nmd.

    python outputs/mp_unified/nmddiag.py census --dataset musique --carve s1fit --host [--limit N]   # N: outputs/nmd/dev
    python outputs/mp_unified/nmddiag.py report
    python outputs/mp_unified/nmddiag.py --selftest
"""
import os
import sys

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (ROOT / "src", ROOT / "scripts", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402

import l2ddiag as L2D  # noqa: E402  (L2D's depths, AUC, strata, orders and toy links; not edited)

LTH, LTW, WQ = L2D.LTH, L2D.LTW, L2D.WQ
DECLARED = "docs/NMD_NAME_LINK_DIAG.md"
STAGE = "NMD"
OUT = ROOT / "outputs" / "nmd"
NAMES = ROOT / "outputs" / "u2r"                 # U2r's names.json and names.npz: read only, checked
PKO = L2D.PKO                                    # PKO's cache and links parts: read only
LTA_REPORT = L2D.LTA_REPORT                      # LTA's census of the same A cases: read only, checked
CARVES = L2D.CARVES                              # musique/s1fit, 2wiki/fit, hotpotqa/fit
DECIDE = L2D.DECIDE
FAR, HOPS, STRATA = L2D.FAR, L2D.HOPS, L2D.STRATA
KINDS = {"u": "cn_u", "all": "cn_all"}           # CN_u (the rule's), CN_all (beside)
VARS = ("u_a", "u_s", "all_a", "all_s")          # from the anchor (_a, the rule's), or summed over the top five (_s)
MATCHED = ("u_a", "u_s", "all_a")
W_EDGES = (0.1, 0.5)                             # CN_u weight from the anchor: below 0.1, 0.1 to 0.5, 0.5 and more
W_BINS = ("<0.1", "0.1-0.5", ">=0.5")
ORDERS = ("rrf", "names_first", "names_w")       # section 3's fixed orders of a case's unlinked candidates
SHARE_MIN, LIFT_MIN, MIN_CASES = 0.10, 2.0, 200  # section 4's rule
assert DECIDE in CARVES and len(W_BINS) == len(W_EDGES) + 1 and LTH.TOP == 5


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] nmddiag: {msg}", flush=True)


def census_path(out, ds, carve):
    return Path(out) / f"{ds}__{carve}.json"


# ── the names ───────────────────────────────────────────────────────────────


def name_csr(n, a, b, w):
    """U2r's pairs (a < b, each once, weight w) as one symmetric table: (row starts, neighbours ascending, weights)."""
    a, b = np.asarray(a, np.int64), np.asarray(b, np.int64)
    if a.size and not (np.all(a < b) and int(a.min()) >= 0 and int(b.max()) < n):
        raise SystemExit("nmd: a name pair is not (a < b) within the nodes")
    key = np.r_[a * n + b, b * n + a]
    o = np.argsort(key, kind="stable")
    key = key[o]
    if key.size and not np.all(np.diff(key) > 0):
        raise SystemExit("nmd: a name pair repeats")
    ww = np.r_[np.asarray(w, np.float32), np.asarray(w, np.float32)][o]
    s = key // n
    ip = np.r_[0, np.cumsum(np.bincount(s, minlength=n))].astype(np.int64)
    return ip, (key % n).astype(np.int64), ww


def name_w(C, x, ys):
    """The name weight between node x and each node of ys, 0 where they share no name (and between a node and itself)."""
    ip, nb, w = C
    lo, hi = int(ip[x]), int(ip[x + 1])
    out = np.zeros(ys.size, np.float64)
    if hi > lo:
        r = nb[lo:hi]
        p = np.minimum(np.searchsorted(r, ys), hi - lo - 1)
        hit = r[p] == ys
        out[hit] = w[lo:hi][p[hit]]
    return out


def package_of(ds, host):
    """The served package's pin and the dataset's nodes, as U2r read them (u1b)."""
    import u1b
    served, freeze = u1b.package_root(host)
    man = json.loads((Path(served) / ds / "DATASET.json").read_text(encoding="utf-8"))
    return {"freeze": freeze, "nodes_sha256": man["nodes"]["sha256"], "nodes": int(man["nodes"]["n"])}


def load_names(ds, root=NAMES, package=None):
    """U2r's CN_u and CN_all of a dataset as symmetric tables, checked: the npz is its names.json's, and the names are
    of the package's nodes (package: package_of's dict)."""
    d = Path(root) / ds
    r = json.loads((d / "names.json").read_text(encoding="utf-8"))
    sha = WQ.sha_file(d / "names.npz")
    if sha != r["names_npz_sha256"]:
        raise SystemExit(f"nmd: {d / 'names.npz'} is not its names.json's")
    if (r["freeze_RECORD_SHA256"], r["nodes_sha256"], int(r["nodes"])) != \
            (package["freeze"], package["nodes_sha256"], int(package["nodes"])):
        raise SystemExit(f"nmd: {ds}: U2r's names are not of the served package's nodes")
    n = int(r["nodes"])
    NM = {}
    with np.load(d / "names.npz") as z:
        for k, key in KINDS.items():
            if z[f"{key}_src"].size != int(r[f"{key}_pairs"]):
                raise SystemExit(f"nmd: {ds}: {key} has {z[f'{key}_src'].size} pairs, names.json {r[f'{key}_pairs']}")
            NM[k] = name_csr(n, z[f"{key}_src"], z[f"{key}_dst"], z[f"{key}_w"])
    return NM, {"names_json_sha256": WQ.sha_file(d / "names.json"), "names_npz_sha256": sha, "nodes": n,
                "freeze_RECORD_SHA256": r["freeze_RECORD_SHA256"], "cn_u_pairs": int(r["cn_u_pairs"]),
                "cn_all_pairs": int(r["cn_all_pairs"]), "package_check": "equal"}


def pko_pins(recs):
    """Every freeze_RECORD_SHA256 the carve's cache records name, anywhere in them."""
    out = set()

    def walk(x):
        if isinstance(x, dict):
            for k, v in x.items():
                if k == "freeze_RECORD_SHA256" and isinstance(v, str):
                    out.add(v)
                else:
                    walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
    walk(list(recs))
    return out


def pin_check(pins, freeze):
    """"equal" if PKO's records name only the names' package, "absent" if they name none; otherwise the census stops."""
    if not pins:
        return "absent"
    if pins != {freeze}:
        raise SystemExit(f"nmd: PKO's records name the package {sorted(pins)}, U2r's names {freeze}")
    return "equal"


def case_weights(NM, node, a, top):
    """Per row of a question, its name weight from the anchor's node and its weight summed over the top five's."""
    W = {}
    for k in KINDS:
        W[f"{k}_a"] = name_w(NM[k], int(node[a]), node)
        s = np.zeros(node.size, np.float64)
        for t in top:
            s += name_w(NM[k], int(node[t]), node)
        W[f"{k}_s"] = s
    return W


# ── a carve's counts ─────────────────────────────────────────────────────────


class Census:
    """A carve's counts, pooled over its A cases (section 3)."""

    def __init__(self):
        w = FAR + 1
        self.c = {k: np.zeros(w, np.int64) for k in ("cand_a", "targ_a")}
        self.c.update({f"{r}_{k}": np.zeros(w, np.int64) for r in ("cand", "targ") for k in VARS})
        self.c.update({k: np.zeros((len(HOPS), w), np.int64)
                       for k in ("cand_hop", "targ_hop", "cand_hop_u", "targ_hop_u")})
        self.c.update({k: np.zeros(len(W_BINS), np.int64) for k in ("cand_wbin", "targ_wbin")})
        self.n = {"cases": 0, "hop_cases": [0] * len(HOPS)}
        self.s = {f"{x}_{k}": 0.0 for k in MATCHED for x in ("obs", "exp")}
        self.s.update({"auc_w": 0.0, "auc_rrf": 0.0, "pairs": 0.0, "auc_wm": 0.0, "pairs_m": 0.0})
        self.per = {k: [] for k in ("n_cand", "n_unl", "n_u_a", "n_u_s")}
        self.ranks = {k: [] for k in ORDERS}

    def add(self, qd, case, hop, W):
        n, g, rk, u, v, f = qd["n"], qd["g"], qd["rk"], qd["u"], qd["v"], qd["f"]
        a, _ka, cand, ct = case
        c, w = self.c, FAR + 1
        src, dst = L2D.sym_edges(n, u, v)
        da = L2D.depths(n, src, dst, [a])
        if not np.array_equal(LTW.fam_bits(n, a, u, v, f)[cand] > 0, da[cand] == 1):
            raise SystemExit("nmd: a row's depth 1 from the anchor is not LTW's link to it")
        tg = cand[ct]
        c["cand_a"] += np.bincount(da[cand], minlength=w)
        c["targ_a"] += np.bincount(da[tg], minlength=w)
        on = {k: x > 0 for k, x in W.items()}
        for k in VARS:
            c[f"cand_{k}"] += np.bincount(da[cand[on[k][cand]]], minlength=w)
            c[f"targ_{k}"] += np.bincount(da[tg[on[k][tg]]], minlength=w)
        if hop in HOPS:
            h = HOPS.index(hop)
            c["cand_hop"][h] += np.bincount(da[cand], minlength=w)
            c["targ_hop"][h] += np.bincount(da[tg], minlength=w)
            c["cand_hop_u"][h] += np.bincount(da[cand[on["u_a"][cand]]], minlength=w)
            c["targ_hop_u"][h] += np.bincount(da[tg[on["u_a"][tg]]], minlength=w)
            self.n["hop_cases"][h] += 1
        unl = cand[da[cand] >= 2]
        ut, r, wu = g[unl], rk[unl], W["u_a"][unl]
        lk = {k: on[k][unl] for k in MATCHED}
        m = wu > 0
        b = np.searchsorted(np.asarray(W_EDGES), wu[m], side="right")
        c["cand_wbin"] += np.bincount(b, minlength=len(W_BINS))
        c["targ_wbin"] += np.bincount(b[ut[m]], minlength=len(W_BINS))
        for k, x in (("n_cand", cand.size), ("n_unl", unl.size), ("n_u_a", int(lk["u_a"].sum())),
                     ("n_u_s", int(lk["u_s"].sum()))):
            self.per[k].append(x)
        self.n["cases"] += 1
        if not ut.any():
            return
        for name, keys in (("rrf", ()), ("names_first", (~lk["u_a"],)), ("names_w", (-wu,))):
            self.ranks[name].extend(L2D.order_ranks(r, keys)[ut].tolist())
        s = self.s
        P, N = int(ut.sum()), int((~ut).sum())
        if N:
            s["pairs"] += P * N
            s["auc_w"] += L2D.auc(wu, ut) * P * N
            s["auc_rrf"] += L2D.auc(-r, ut) * P * N
        bs = L2D.strata_of(r)
        ns = np.bincount(bs, minlength=STRATA).astype(np.float64)
        ts = np.bincount(bs, weights=ut, minlength=STRATA)
        ok = ns > 0
        for k in MATCHED:
            cl = np.bincount(bs, weights=lk[k], minlength=STRATA)
            s[f"obs_{k}"] += float((ut & lk[k]).sum())
            s[f"exp_{k}"] += float((ts[ok] * cl[ok] / ns[ok]).sum())
        for j in np.flatnonzero(ts > 0):
            mj = bs == j
            pk, nk = int(ut[mj].sum()), int((~ut[mj]).sum())
            if nk:
                s["auc_wm"] += L2D.auc(wu[mj], ut[mj]) * pk * nk
                s["pairs_m"] += pk * nk

    def record(self):
        def dist(x):
            x = np.asarray(x, np.float64)
            return {"median": float(np.median(x)), "p90": float(np.percentile(x, 90)), "max": float(x.max())} \
                if x.size else None

        def summ(rk):
            a = np.asarray(rk, np.float64)
            return {"ranked": int(a.size), "T2": float(np.mean(a <= 2)), "T10": float(np.mean(a <= 10)),
                    "median": float(np.median(a)), "MRR": float(np.mean(1.0 / a))} if a.size else None
        return {"counts": {k: x.tolist() for k, x in self.c.items()}, "n": dict(self.n), "sums": dict(self.s),
                "per_case": {k: dist(x) for k, x in self.per.items()},
                "orders": {k: summ(x) for k, x in self.ranks.items()}}


def census(ds, carve, root=PKO, out=OUT, lta=LTA_REPORT, names=NAMES, package=None, host=False, limit=None):
    """One fit carve's census (section 3), checked against LTA's census of the same A cases, written to out (a run
    with a limit to out/dev). package: the self-test's package dict; None reads the served package (host: the
    host's mirror)."""
    import lean_cache as LC
    t0 = time.time()
    if (ds, carve) not in CARVES and root == PKO:
        raise SystemExit(f"nmd: {ds}/{carve} is not one of LTW's fit carves")
    NM, nrec = load_names(ds, names, package_of(ds, host) if package is None else package)
    parts, recs = LC.part_dirs(ds, carve, Path(root) / "cache")
    pin = pin_check(pko_pins(recs), nrec["freeze_RECORD_SHA256"])
    take = LTH.xc_take(recs[0]["look"], ds, carve)
    ids = [x for r in recs for x in r["ids"]]
    C = Census()
    cache_parts, links_parts = {}, {}
    qi = rows = 0
    for p, r in zip(parts, recs):
        if limit and qi >= limit:
            break
        Ar, E, lk_sha = LTW.part_arrays(p, r, Path(root) / "links")
        cache_parts[p.name], links_parts[p.name] = LC.sha_file(p / "record.json"), lk_sha
        rows += int(r["rows"])
        xc = np.asarray(Ar["xc"][:, take], np.float32)
        gold = np.asarray(Ar["gold"], bool)
        node = np.asarray(Ar["node"], np.int64)
        if node.size and (int(node.min()) < 0 or int(node.max()) >= nrec["nodes"]):
            raise SystemExit(f"nmd: {p}: a row's node is beyond U2r's names' {nrec['nodes']} nodes")
        nq, gtot = np.asarray(Ar["n"], np.int64), np.asarray(Ar["gold_total"], np.int64)
        eu, ev, ef = np.asarray(E["e_u"]), np.asarray(E["e_v"]), np.asarray(E["e_fam"])
        no = np.r_[0, np.cumsum(nq)]
        eo = np.r_[0, np.cumsum(np.asarray(E["q_ne"], np.int64))]
        for i in range(nq.size):
            if limit and qi >= limit:
                break
            qd = LTH.question(xc, gold, node, nq, eu, ev, ef, no, eo, i)
            case = LTH.a_case(qd, int(gtot[i]))
            if case is not None:
                hop = int(str(ids[qi])[0]) if ds == "musique" and str(ids[qi])[:1] in "234" else 0
                C.add(qd, case, hop, case_weights(NM, qd["node"], case[0], qd["top"]))
            qi += 1
            if qi % 500 == 0:
                log(f"  {ds}/{carve}: {qi} questions, {C.n['cases']} A cases ({time.time() - t0:.0f}s)")
        del Ar, E, xc, gold, node, eu, ev, ef
    if not limit and qi != len(ids):
        raise SystemExit(f"nmd: {ds}/{carve}: read {qi} questions, the records name {len(ids)}")
    rec = C.record()
    ta = rec["counts"]["targ_a"]
    mine = {"questions": qi, "rows": rows, "lists": C.n["cases"], "targets": int(sum(ta)), "linked_targets": int(ta[1])}
    lt = None
    if Path(lta).is_file():
        lt = json.loads(Path(lta).read_text(encoding="utf-8")).get("lists", {}).get(f"{ds}/{carve}")
    if limit:
        check = "limit"
    elif lt is None:
        check = "absent"
    else:
        theirs = {"questions": lt["questions"], "rows": lt["rows"], "lists": lt["A"]["lists"],
                  "targets": lt["A"]["targets"], "linked_targets": lt["A"]["linked_targets"]}
        if mine != theirs:
            raise SystemExit(f"nmd: {ds}/{carve}: the A cases are not LTA's: {mine} against LTA's {theirs}")
        check = "equal"
    rec = {"stage": STAGE, "declared_in": DECLARED, "dataset": ds, "carve": carve, **mine, "limit": limit,
           "lta_check": check, "names": nrec, "pko_pin": pin, **rec, "cache_parts": cache_parts,
           "links_parts": links_parts, "script_sha256": WQ.sha_src(__file__), "l2ddiag_sha256": WQ.sha_src(L2D.__file__),
           "lthcols_sha256": WQ.sha_src(LTH.__file__), "seconds": round(time.time() - t0, 1)}
    rec["derived"] = derive(rec)
    LTW.write_json(census_path(Path(out) / "dev" if limit else out, ds, carve), rec)
    d = rec["derived"]
    log(f"{ds}/{carve}: {C.n['cases']} A cases; unlinked targets {d['unlinked_targets']}; named from the anchor "
        f"{d['share']}; matched lift {d['lift_matched']} ({rec['seconds']}s)")
    return rec


# ── reads and the rule ───────────────────────────────────────────────────────


def derive(z):
    """Section 3's reads from a census record's counts and sums (recomputed, never carried)."""
    c, n, s = z["counts"], z["n"], z["sums"]

    def A(k):
        return np.asarray(c[k], np.int64)
    ca, ta = A("cand_a"), A("targ_a")
    if ca[0] or ta[0]:
        raise SystemExit("nmd: a candidate at depth 0 from its anchor")
    T, Cn, Tu, Cu, Tb = int(ta.sum()), int(ca.sum()), int(ta[2:].sum()), int(ca[2:].sum()), int(ta[3:].sum())

    def reads(k):
        cv, tv = A(f"cand_{k}"), A(f"targ_{k}")
        x, y = L2D.share(tv[2:].sum(), Tu), L2D.share(cv[2:].sum(), Cu)
        r = {"share": x, "candidates_share": y, "lift": L2D.ratio(x, y), "share_at_two": L2D.share(tv[2], ta[2]),
             "share_beyond_two": L2D.share(tv[3:].sum(), Tb), "linked_targets_too": L2D.share(tv[1], ta[1]),
             "unlinked_targets_at": L2D.at(tv, Tu)}
        if f"obs_{k}" in s:
            r["lift_matched"] = L2D.ratio(s[f"obs_{k}"], s[f"exp_{k}"])
        return r
    nm = {k: reads(k) for k in VARS}
    tua = A("targ_u_a")
    cw, tw = A("cand_wbin"), A("targ_wbin")
    ch, th, chu, thu = (np.asarray(c[k], np.int64) for k in ("cand_hop", "targ_hop", "cand_hop_u", "targ_hop_u"))

    def hop_reads(j):
        tuh, cuh = int(th[j][2:].sum()), int(ch[j][2:].sum())
        x = L2D.share(thu[j][2:].sum(), tuh)
        return {"A_cases": int(n["hop_cases"][j]), "linked_share": L2D.share(th[j][1], th[j].sum()),
                "unlinked_targets": tuh, "share": x, "lift": L2D.ratio(x, L2D.share(chu[j][2:].sum(), cuh))}
    return {
        "A_cases": int(n["cases"]), "targets": T, "candidates": Cn, "linked_targets": int(ta[1]),
        "linked_share": L2D.share(ta[1], T), "unlinked_targets": Tu, "unlinked_candidates": Cu,
        "beyond_two_targets": Tb, "unlinked_targets_at": L2D.at(ta, Tu),
        "share": nm["u_a"]["share"], "lift_matched": nm["u_a"]["lift_matched"],
        "reach_with_two_links": L2D.share(ta[2] + tua[3:].sum(), Tu),
        "auc_w": L2D.share(s["auc_w"], s["pairs"]), "auc_w_matched": L2D.share(s["auc_wm"], s["pairs_m"]),
        "auc_rrf": L2D.share(s["auc_rrf"], s["pairs"]),
        "names": nm,
        "weight_bins": {b: {"unlinked_targets": L2D.share(tw[j], Tu), "unlinked_candidates": L2D.share(cw[j], Cu),
                            "lift": L2D.ratio(L2D.share(tw[j], Tu), L2D.share(cw[j], Cu))}
                        for j, b in enumerate(W_BINS)},
        "hops": {f"{h}hop": hop_reads(j) for j, h in enumerate(HOPS) if n["hop_cases"][j]},
        "per_case": z["per_case"], "orders": z["orders"]}


def decide(d):
    """Section 4's rule on the deciding carve's reads: (call, reading)."""
    if d["A_cases"] < MIN_CASES or d["share"] is None:
        return "INCOMPLETE", f"{d['A_cases']} A cases (at least {MIN_CASES}), share {d['share']}"
    if d["share"] < SHARE_MIN:
        return "NO_SIGNAL", "RARE"
    if d["lift_matched"] is None or d["lift_matched"] < LIFT_MIN:
        return "NO_SIGNAL", "NOT_SELECTIVE"
    return "SIGNAL", "MATCHED_LIFT"


FOLLOWS = {
    "SIGNAL": "The names term is declared in its own file now: the CN_u name weight between each rrf top-five row and "
              "each row outside the top five (from the passages' text, no encoder, fixed before any row is scored) as "
              "an input of the next-hop term of both models, the MLP's and the GNN's, the same rule on every dataset. "
              "Its columns are built on the host's CPU so its screens can start before the freeze.",
    "NO_SIGNAL/RARE": "No names term is built. U2r's untitled names do not reach musique's unlinked next hop: the names "
                      "line moves to its open forms (prefix-merged titled names, names as entity nodes), each read by "
                      "this census before anything is built from it.",
    "NO_SIGNAL/NOT_SELECTIVE": "No names term is built. The names reach the unlinked targets, but no more often than "
                               "they reach the other rows of the same rrf rank: a name shared with the anchor does not "
                               "single out the next hop. The next-hop line keeps PKO's links and LT2's two-link rows.",
    "INCOMPLETE": "What is missing is fixed in an amendment, before any number is read.",
}


def fmt(x, nd=4):
    return "-" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def report(out=OUT):
    """The three carves' reads and section 4's call on musique/s1fit, to out/report.json and out/report.md."""
    me, rows, problems = WQ.sha_src(__file__), {}, []
    for ds, cv in CARVES:
        p, key = census_path(out, ds, cv), f"{ds}/{cv}"
        if not p.is_file():
            problems.append(f"{key}: no census")
            continue
        z = json.loads(p.read_text(encoding="utf-8"))
        if z.get("stage") != STAGE or z.get("script_sha256") != me:
            problems.append(f"{key}: the census is not this script's")
            continue
        if z.get("limit"):
            problems.append(f"{key}: a run limited to {z['limit']} questions")
        elif z.get("lta_check") != "equal":
            problems.append(f"{key}: LTA's census is {z.get('lta_check')}, not equal")
        if (z.get("names") or {}).get("package_check") != "equal":
            problems.append(f"{key}: the names are not checked against the package")
        rows[key] = derive(z)
    dk = f"{DECIDE[0]}/{DECIDE[1]}"
    bad = [x for x in problems if x.startswith(dk + ":")]
    call, reading = ("INCOMPLETE", "; ".join(bad)) if bad or dk not in rows else decide(rows[dk])
    follows = FOLLOWS["SIGNAL" if call == "SIGNAL" else call if call == "INCOMPLETE" else f"{call}/{reading}"]
    rep = {"stage": STAGE, "declared_in": DECLARED, "decides_on": dk, "call": call, "reading": reading,
           "rule": {"SHARE_MIN": SHARE_MIN, "LIFT_MIN": LIFT_MIN, "MIN_CASES": MIN_CASES},
           "follows": follows, "problems": problems, "carves": rows, "script_sha256": me}
    LTW.write_json(Path(out) / "report.json", rep)
    L = [f"# NMD: the passages' names at the next hop (declared in {DECLARED})", "",
         f"## Call on {dk}: {call} ({reading})", "", follows, "",
         f"Rule: SIGNAL iff the share of unlinked targets that share a CN_u name with their anchor >= {SHARE_MIN} and "
         f"its rrf-decile-matched lift >= {LIFT_MIN}, with at least {MIN_CASES} A cases.", ""]
    if problems:
        L += ["Problems: " + "; ".join(problems), ""]
    L += ["## CN_u from the anchor (the rule's read)", "",
          "| carve | A cases | unlinked targets | named | candidates named | lift | matched lift | named at two links | "
          "named beyond two | linked targets also named | reach with two links | AUC weight | matched AUC | AUC rrf |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for k, d in rows.items():
        x = d["names"]["u_a"]
        L.append(f"| {k} | {d['A_cases']} | {d['unlinked_targets']} | {fmt(x['share'])} | {fmt(x['candidates_share'])} | "
                 f"{fmt(x['lift'], 3)} | {fmt(x['lift_matched'], 3)} | {fmt(x['share_at_two'])} | "
                 f"{fmt(x['share_beyond_two'])} | {fmt(x['linked_targets_too'])} | {fmt(d['reach_with_two_links'])} | "
                 f"{fmt(d['auc_w'])} | {fmt(d['auc_w_matched'])} | {fmt(d['auc_rrf'])} |")
    L += ["", "## Every form (shares of the unlinked targets and candidates)", "",
          "| carve | names | from | named | candidates named | lift | matched lift | named beyond two links |",
          "|---|---|---|---:|---:|---:|---:|---:|"]
    for k, d in rows.items():
        for v, x in d["names"].items():
            kind, frm = v.split("_")
            L.append(f"| {k} | CN_{kind} | {'anchor' if frm == 'a' else 'top five'} | {fmt(x['share'])} | "
                     f"{fmt(x['candidates_share'])} | {fmt(x['lift'], 3)} | {fmt(x.get('lift_matched'), 3)} | "
                     f"{fmt(x['share_beyond_two'])} |")
    L += ["", "## CN_u weight from the anchor (shares of the unlinked rows; lift = targets over candidates)", "",
          "| carve | " + " | ".join(f"{b}: targets, candidates, lift" for b in W_BINS) + " |",
          "|---|" + "---|" * len(W_BINS)]
    for k, d in rows.items():
        L.append(f"| {k} | " + " | ".join(f"{fmt(x['unlinked_targets'])}, {fmt(x['unlinked_candidates'])}, "
                                          f"{fmt(x['lift'], 3)}" for x in d["weight_bins"].values()) + " |")
    L += ["", "## Per case", "", "| carve | candidates (median) | unlinked (median) | named from the anchor (median, p90) "
          "| named from the top five (median, p90) |", "|---|---:|---:|---|---|"]
    for k, d in rows.items():
        pc = d["per_case"]
        L.append(f"| {k} | {fmt((pc['n_cand'] or {}).get('median'), 1)} | {fmt((pc['n_unl'] or {}).get('median'), 1)} | "
                 f"{fmt((pc['n_u_a'] or {}).get('median'), 1)}, {fmt((pc['n_u_a'] or {}).get('p90'), 1)} | "
                 f"{fmt((pc['n_u_s'] or {}).get('median'), 1)}, {fmt((pc['n_u_s'] or {}).get('p90'), 1)} |")
    L += ["", "## Fixed orders of the unlinked candidates (the unlinked targets' ranks)", "",
          "| carve | order | ranked | T2 | T10 | median | MRR |", "|---|---|---:|---:|---:|---:|---:|"]
    for k, d in rows.items():
        for o, x in d["orders"].items():
            if x:
                L.append(f"| {k} | {o} | {x['ranked']} | {fmt(x['T2'])} | {fmt(x['T10'])} | {fmt(x['median'], 1)} | "
                         f"{fmt(x['MRR'])} |")
    hp = [(k, h, x) for k, d in rows.items() for h, x in d["hops"].items()]
    if hp:
        L += ["", "## By musique hop (CN_u from the anchor)", "",
              "| carve | hop | A cases | linked | unlinked | named | lift |", "|---|---|---:|---:|---:|---:|---:|"]
        for k, h, x in hp:
            L.append(f"| {k} | {h} | {x['A_cases']} | {fmt(x['linked_share'])} | {x['unlinked_targets']} | "
                     f"{fmt(x['share'])} | {fmt(x['lift'], 3)} |")
    (Path(out) / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    log(f"report: {call} ({reading}) on {dk}")
    return rep


# ── self-test ────────────────────────────────────────────────────────────────


def brute(root, ds, carve, names):
    """A carve's counts and sums, by hand: names as a dict of pairs, rrf's order by a stable argsort, BFS with a
    queue (L2D's), deciles by position."""
    import lean_cache as LC
    parts, recs = LC.part_dirs(ds, carve, Path(root) / "cache")
    take = LTH.xc_take(recs[0]["look"], ds, carve)
    w = FAR + 1
    c = {k: np.zeros(w, np.int64) for k in ("cand_a", "targ_a")}
    c.update({f"{r}_{k}": np.zeros(w, np.int64) for r in ("cand", "targ") for k in VARS})
    c.update({k: np.zeros((len(HOPS), w), np.int64) for k in ("cand_hop", "targ_hop", "cand_hop_u", "targ_hop_u")})
    c.update({k: np.zeros(len(W_BINS), np.int64) for k in ("cand_wbin", "targ_wbin")})
    n_ = {"cases": 0, "hop_cases": [0] * len(HOPS)}
    s = {f"{x}_{k}": 0.0 for k in MATCHED for x in ("obs", "exp")}
    s.update({"auc_w": 0.0, "auc_rrf": 0.0, "pairs": 0.0, "auc_wm": 0.0, "pairs_m": 0.0})
    per = {k: [] for k in ("n_cand", "n_unl", "n_u_a", "n_u_s")}
    ranks = {k: [] for k in ORDERS}

    def pair_auc(sc, pos):
        P = [x for x, p in zip(sc, pos) if p]
        N = [x for x, p in zip(sc, pos) if not p]
        return sum((p > q) + 0.5 * (p == q) for p in P for q in N), len(P) * len(N)

    def nw(kind, x, y):
        return 0.0 if x == y else names[kind].get((min(x, y), max(x, y)), 0.0)
    for p, r in zip(parts, recs):
        lp = Path(root) / "links" / ds / carve / p.name
        nq = np.load(p / "n.npy").astype(np.int64)
        gt_all = np.load(p / "gold_total.npy").astype(np.int64)
        gold, node = np.load(p / "gold.npy"), np.load(p / "node.npy").astype(np.int64)
        xc = np.load(p / "xc.npy")[:, take].astype(np.float64)
        qn = np.load(lp / "q_ne.npy").astype(np.int64)
        eu, ev = (np.load(lp / f"{k}.npy").astype(np.int64) for k in ("e_u", "e_v"))
        no, eo = np.r_[0, np.cumsum(nq)], np.r_[0, np.cumsum(qn)]
        for i in range(nq.size):
            n = int(nq[i])
            g = gold[no[i]:no[i + 1]].astype(bool)
            nd = node[no[i]:no[i + 1]].tolist()
            o = np.argsort(-xc[no[i]:no[i + 1], LTW.J_RRF], kind="stable")
            rank = np.empty(n, np.int64)
            rank[o] = np.arange(1, n + 1)
            top = [int(x) for x in o[:5]]
            found = [x for x in top if g[x]]
            cand = [x for x in range(n) if x not in top]
            if gt_all[i] <= 0 or not found or not any(g[x] for x in cand):
                continue
            a = found[0]
            da = L2D.bfs(L2D.adj_of(n, eu[eo[i]:eo[i + 1]], ev[eo[i]:eo[i + 1]]), [a], n)
            W = {}
            for k in KINDS:
                W[f"{k}_a"] = [nw(k, nd[a], nd[x]) for x in range(n)]
                W[f"{k}_s"] = [sum(nw(k, nd[t], nd[x]) for t in top) for x in range(n)]
            hop = int(r["ids"][i][0]) if ds == "musique" and r["ids"][i][:1] in "234" else 0
            n_["cases"] += 1
            for x in cand:
                c["cand_a"][da[x]] += 1
                c["targ_a"][da[x]] += int(g[x])
                for k in VARS:
                    if W[k][x] > 0:
                        c[f"cand_{k}"][da[x]] += 1
                        c[f"targ_{k}"][da[x]] += int(g[x])
                if hop in HOPS:
                    h = HOPS.index(hop)
                    c["cand_hop"][h][da[x]] += 1
                    c["targ_hop"][h][da[x]] += int(g[x])
                    if W["u_a"][x] > 0:
                        c["cand_hop_u"][h][da[x]] += 1
                        c["targ_hop_u"][h][da[x]] += int(g[x])
            if hop in HOPS:
                n_["hop_cases"][HOPS.index(hop)] += 1
            unl = sorted((x for x in cand if da[x] >= 2), key=lambda x: rank[x])
            for x in unl:
                wx = W["u_a"][x]
                if wx > 0:
                    j = 0 if wx < W_EDGES[0] else 1 if wx < W_EDGES[1] else 2
                    c["cand_wbin"][j] += 1
                    c["targ_wbin"][j] += int(g[x])
            per["n_cand"].append(len(cand))
            per["n_unl"].append(len(unl))
            per["n_u_a"].append(sum(W["u_a"][x] > 0 for x in unl))
            per["n_u_s"].append(sum(W["u_s"][x] > 0 for x in unl))
            ut = [bool(g[x]) for x in unl]
            if not any(ut):
                continue
            for name, key in (("rrf", lambda x: (rank[x],)), ("names_first", lambda x: (W["u_a"][x] <= 0, rank[x])),
                              ("names_w", lambda x: (-W["u_a"][x], rank[x]))):
                pos = {x: j + 1 for j, x in enumerate(sorted(unl, key=key))}
                ranks[name] += [pos[x] for x in unl if g[x]]
            num, den = pair_auc([W["u_a"][x] for x in unl], ut)
            if den:
                s["auc_w"] += num
                s["auc_rrf"] += pair_auc([-rank[x] for x in unl], ut)[0]
                s["pairs"] += den
            N = len(unl)
            for k in range(STRATA):
                b = [x for j, x in enumerate(unl) if j * STRATA // N == k]
                if not b:
                    continue
                tb = [bool(g[x]) for x in b]
                for v in MATCHED:
                    s[f"obs_{v}"] += sum(1 for x in b if g[x] and W[v][x] > 0)
                    s[f"exp_{v}"] += sum(tb) * sum(1 for x in b if W[v][x] > 0) / len(b)
                if any(tb):
                    num, den = pair_auc([W["u_a"][x] for x in b], tb)
                    if den:
                        s["auc_wm"] += num
                        s["pairs_m"] += den
    return c, n_, s, per, ranks


def toy_names(root, ds, n_nodes, gold_nodes, rng, nodes_sha="toy-nodes", freeze="toy-freeze"):
    """U2r-shaped names.json and names.npz of n_nodes toy nodes: CN_u random pairs plus pairs among gold nodes, CN_all
    CN_u's pairs (heavier) and more; returns the pairs as dicts."""
    def pairs(m, pool):
        a, b = rng.choice(pool, m), rng.choice(pool, m)
        k = np.unique(np.minimum(a, b) * n_nodes + np.maximum(a, b))
        k = k[k // n_nodes != k % n_nodes]
        return k
    ku = np.unique(np.r_[pairs(3 * n_nodes, np.arange(n_nodes)), pairs(4 * len(gold_nodes), np.asarray(gold_nodes))])
    ka = np.unique(np.r_[ku, pairs(2 * n_nodes, np.arange(n_nodes))])
    wu = (1.0 / rng.integers(2, 26, ku.size)).astype(np.float32)
    wa = (1.0 / rng.integers(2, 26, ka.size)).astype(np.float32)
    wa[np.isin(ka, ku)] += wu
    d = Path(root) / ds
    d.mkdir(parents=True, exist_ok=True)
    np.savez(d / "names.npz", cn_all_src=(ka // n_nodes).astype(np.int32), cn_all_dst=(ka % n_nodes).astype(np.int32),
             cn_all_w=wa, cn_u_src=(ku // n_nodes).astype(np.int32), cn_u_dst=(ku % n_nodes).astype(np.int32),
             cn_u_w=wu)
    LTW.write_json(d / "names.json", {"dataset": ds, "nodes": n_nodes, "nodes_sha256": nodes_sha,
                                      "freeze_RECORD_SHA256": freeze, "cn_u_pairs": int(ku.size),
                                      "cn_all_pairs": int(ka.size), "names_npz_sha256": WQ.sha_file(d / "names.npz")})
    return {"u": {(int(k // n_nodes), int(k % n_nodes)): float(x) for k, x in zip(ku, wu)},
            "all": {(int(k // n_nodes), int(k % n_nodes)): float(x) for k, x in zip(ka, wa)}}


def selftest():
    rng = np.random.default_rng(29)
    # 1. the symmetric table against a dict on random pairs; bad pairs stop
    for t in range(80):
        n = int(rng.integers(2, 50))
        a, b = rng.integers(0, n, 3 * n), rng.integers(0, n, 3 * n)
        k = np.unique(np.minimum(a, b) * n + np.maximum(a, b))
        k = k[k // n != k % n]
        w = rng.random(k.size).astype(np.float32)
        C = name_csr(n, k // n, k % n, w)
        dct = {(int(x // n), int(x % n)): float(y) for x, y in zip(k, w)}
        for x in range(n):
            ys = rng.integers(0, n, 30)
            want = [0.0 if x == y else dct.get((min(x, int(y)), max(x, int(y))), 0.0) for y in ys]
            assert name_w(C, x, ys).tolist() == want, (n, x)
        if k.size:
            for bad in ((k % n, k // n), (np.r_[k // n, k[:1] // n], np.r_[k % n, k[:1] % n])):
                try:
                    name_csr(n, bad[0], bad[1], np.ones(bad[0].size, np.float32))
                    raise AssertionError("a bad pair list was read")
                except SystemExit:
                    pass
    assert pko_pins([{"bases": {"g": {"freeze_RECORD_SHA256": "x"}}}, {"a": [{"freeze_RECORD_SHA256": "y"}]}]) == \
        {"x", "y"}
    assert pin_check(set(), "x") == "absent" and pin_check({"x"}, "x") == "equal"
    try:
        pin_check({"x", "y"}, "x")
        raise AssertionError("PKO on another package was read")
    except SystemExit as e:
        assert "package" in str(e)
    tmp = Path(tempfile.mkdtemp(prefix="nmd_"))
    saved = MIN_CASES
    try:
        # 2. PKO-shaped toy carves (LTH's toy_carve, links thinned as L2D's), U2r-shaped toy names, by-hand recount
        NN = 500
        M = LTW.unit(rng.normal(size=(NN, 16))).astype(np.float32)
        sizes = {"musique": [[40, 300, 23, 61, 40, 17, 33, 40, 270, 29], [52, 40, 23, 300, 36, 40, 18, 44]],
                 "2wiki": [[40, 23, 61, 33, 40, 29, 52, 40, 36, 44, 27, 31, 40, 38]],
                 "hotpotqa": [[40, 23, 61, 33, 120, 80, 150], [29, 300, 40, 150, 90, 200, 61]]}
        names, pk = tmp / "u2r", {"freeze": "toy-freeze", "nodes_sha256": "toy-nodes", "nodes": NN}
        dicts = {}
        for ds, cv in CARVES:
            LTH.toy_carve(tmp, ds, cv, M, rng, sizes[ds])
            L2D.thin_links(tmp, ds, cv, {"musique": 0.05, "2wiki": 0.30, "hotpotqa": 0.60}[ds], rng)
            gn = sorted({int(x) for p in (tmp / "cache" / ds / cv).glob("part_*of*")
                         for x in np.load(p / "node.npy")[np.load(p / "gold.npy").astype(bool)]})
            dicts[ds] = toy_names(names, ds, NN, gn, rng)
        out = tmp / "nmd"
        recs = {}
        for ds, cv in CARVES:
            rec = census(ds, cv, root=tmp, out=out, lta=tmp / "no_lta.json", names=names, package=pk)
            assert rec["lta_check"] == "absent" and rec["pko_pin"] == "absent" and census_path(out, ds, cv).is_file()
            c, n_, s, per, ranks = brute(tmp, ds, cv, dicts[ds])
            for k, x in c.items():
                assert np.array_equal(np.asarray(rec["counts"][k]), x), (ds, k, rec["counts"][k], x.tolist())
            for k, x in n_.items():
                assert rec["n"][k] == x, (ds, k, rec["n"][k], x)
            for k, x in s.items():
                assert abs(rec["sums"][k] - x) < 1e-9, (ds, k, rec["sums"][k], x)
            for k, x in per.items():
                y = rec["per_case"][k]
                assert y["median"] == float(np.median(x)) and y["max"] == float(max(x)), (ds, k)
            for k, x in ranks.items():
                y = rec["orders"][k]
                assert (y is None) == (not x) and (not x or (y["ranked"] == len(x) and y["T2"] == np.mean(
                    np.asarray(x) <= 2) and y["median"] == float(np.median(x)))), (ds, k)
            recs[(ds, cv)] = rec
        tot = {k: sum(np.asarray(r["counts"][k]) for r in recs.values()) for k in ("targ_u_a", "targ_u_s", "targ_all_a")}
        assert tot["targ_u_a"][2] > 0 and tot["targ_u_a"][3:].sum() > 0 and tot["targ_u_a"][1] > 0, tot["targ_u_a"]
        assert all(tot[k][2:].sum() >= tot["targ_u_a"][2:].sum() for k in ("targ_u_s", "targ_all_a"))
        mq = recs[DECIDE]
        assert sum(mq["n"]["hop_cases"]) == mq["n"]["cases"] > 0 and mq["counts"]["cand_a"][FAR] > 0
        assert sum(sum(r["counts"]["cand_wbin"]) for r in recs.values()) > 0
        # 3. the names' checks: another npz, another package, nodes beyond the names', all stop the census
        bad = tmp / "u2r_bad"
        shutil.copytree(names / "musique", bad / "musique")
        z = json.loads((bad / "musique" / "names.json").read_text(encoding="utf-8"))
        LTW.write_json(bad / "musique" / "names.json", {**z, "names_npz_sha256": "0" * 64})
        for kw, msg in (({"names": bad}, "not its names.json's"),
                        ({"package": {**pk, "freeze": "another"}}, "not of the served package's nodes")):
            try:
                census("musique", "s1fit", root=tmp, out=tmp / "nmd_bad", lta=tmp / "no_lta.json",
                       **{"names": names, "package": pk, **kw})
                raise AssertionError(f"a census ran on bad names ({msg})")
            except SystemExit as e:
                assert msg in str(e), str(e)
        small = tmp / "u2r_small"
        toy_names(small, "musique", 12, [0, 1, 2], rng)
        try:
            census("musique", "s1fit", root=tmp, out=tmp / "nmd_bad", lta=tmp / "no_lta.json", names=small,
                   package={**pk, "nodes": 12})
            raise AssertionError("a census ran past its names' nodes")
        except SystemExit as e:
            assert "beyond U2r's names" in str(e), str(e)
        assert not (tmp / "nmd_bad").exists()
        # 4. LTA's census: equal passes, any difference stops the census; a limited run writes under dev only
        lta = tmp / "lta_report.json"
        LTW.write_json(lta, {"lists": {f"{ds}/{cv}": {
            "questions": x["questions"], "rows": x["rows"],
            "A": {"lists": x["lists"], "targets": x["targets"], "linked_targets": x["linked_targets"]}}
            for (ds, cv), x in recs.items()}})
        for ds, cv in CARVES:
            assert census(ds, cv, root=tmp, out=out, lta=lta, names=names, package=pk)["lta_check"] == "equal"
        z = json.loads(lta.read_text(encoding="utf-8"))
        z["lists"]["musique/s1fit"]["A"]["targets"] += 1
        LTW.write_json(tmp / "lta_bad.json", z)
        try:
            census("musique", "s1fit", root=tmp, out=tmp / "nmd_bad2", lta=tmp / "lta_bad.json", names=names,
                   package=pk)
            raise AssertionError("a census that is not LTA's was written")
        except SystemExit as e:
            assert "not LTA's" in str(e)
        before = census_path(out, *DECIDE).read_bytes()
        lim = census("musique", "s1fit", root=tmp, out=out, lta=lta, names=names, package=pk, limit=3)
        assert lim["lta_check"] == "limit" and census_path(out / "dev", *DECIDE).is_file()
        assert census_path(out, *DECIDE).read_bytes() == before
        # 5. the report: the toy has fewer than MIN_CASES cases (INCOMPLETE); with MIN_CASES 1 it decides
        rep = report(out)
        assert rep["call"] == "INCOMPLETE" and set(rep["carves"]) == {f"{d}/{c}" for d, c in CARVES}
        globals()["MIN_CASES"] = 1
        rep = report(out)
        assert (rep["call"], rep["reading"]) == decide(derive(recs[DECIDE]))
        assert rep["call"] in ("SIGNAL", "NO_SIGNAL") and (out / "report.md").is_file() and not rep["problems"]
        p = census_path(out, *DECIDE)
        z = json.loads(p.read_text(encoding="utf-8"))
        for stale in ({"script_sha256": "0" * 64}, {"lta_check": "absent"}, {"limit": 3},
                      {"names": {**z["names"], "package_check": "skipped"}}):
            LTW.write_json(p, {**z, **stale})
            assert report(out)["call"] == "INCOMPLETE", stale
        LTW.write_json(p, z)
        assert report(out)["call"] == rep["call"]
        # 6. the rule on fixed reads
        base = {"A_cases": 500, "share": 0.12, "lift_matched": 2.1}
        assert decide(base) == ("SIGNAL", "MATCHED_LIFT")
        assert decide({**base, "share": 0.09}) == ("NO_SIGNAL", "RARE")
        assert decide({**base, "lift_matched": 1.9}) == ("NO_SIGNAL", "NOT_SELECTIVE")
        assert decide({**base, "lift_matched": None}) == ("NO_SIGNAL", "NOT_SELECTIVE")
        assert decide({**base, "share": None})[0] == "INCOMPLETE"
        globals()["MIN_CASES"] = saved
        assert decide({**base, "A_cases": saved - 1})[0] == "INCOMPLETE"
    finally:
        globals()["MIN_CASES"] = saved
        shutil.rmtree(tmp, ignore_errors=True)
    assert OUT == ROOT / "outputs" / "nmd" and NAMES == ROOT / "outputs" / "u2r" and MIN_CASES == 200
    assert (SHARE_MIN, LIFT_MIN) == (0.10, 2.0)
    log("selftest OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", choices=("census", "report"))
    ap.add_argument("--dataset")
    ap.add_argument("--carve")
    ap.add_argument("--host", action="store_true", help="read the served package from the host's mirror")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return 0
    if a.cmd == "census":
        census(a.dataset, a.carve, host=a.host, limit=a.limit)
        return 0
    if a.cmd == "report":
        report()
        return 0
    ap.error("a command or --selftest")


if __name__ == "__main__":
    sys.exit(main())
