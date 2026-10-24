"""U2l's diagnostic. It decides nothing, and it is read before U2l's declaration (docs/U2L_ENTITY_MATCH.md). It asks why
U2e's pick (outputs/u2e/verdict.json) misses golds, cause by cause, and how the passage graphs' links and NER edges
overlap.

    python outputs/mp_unified/zu2l.py diag --dataset webqsp

It reads the golds of every question in which the pick misses a gold. The question's held golds are read beside, as
the contrast. For each gold:
- reach: whether it is one link from B in each family, and its hop distance from B in the union of the families
  (1, 2, 3, beyond);
- on a typed KB:
  - the (B node, relation, direction) groups that hold it, with each group's fan-out;
  - the rank of the group's relation among the B node's relations, by cos(q, e_r);
  - whether a held gold of the question shares the group (a list the pick's cut split);
  - at two links, whether the middle node is a CVT mediator, and the same reads for the second group;
- entity match: whether the gold's name and the question's text, or a B node's name, contain one another.
On a passage dataset, it also reads the share of W2_S's pairs that are NER pairs, and the golds that each family
alone reaches one link from B.

It writes outputs/u2l/diag/<dataset>.json and .md. Nothing here chooses or changes anything. The pick's ledger must
reproduce U2e's filed ledger first, and B, the golds and the in-B flags must equal U2e's arrays.
"""
from __future__ import annotations

import os
import sys

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (ROOT / "scripts", ROOT / "src", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import zu2e as U2E  # noqa: E402  (U2e's arrays, pick, budgets, evaluate and ledger)

U2C = U2E.U2C
log, utc, write_json, sha_file = U2C.log, U2C.utc, U2C.write_json, U2C.sha_file
OUT = ROOT / "outputs" / "u2l" / "diag"
CAUSES = ("HELD", "TAU_CUT", "BUDGET", "UNREACHED")
HOPS_CAP = 2_000_000     # a gold's two-link neighbourhood larger than this is not expanded (recorded as capped)
MIDS_CAP = 5_000         # middle nodes read per gold at two links
NAME_MIN = 4             # the shortest normalized name an entity match reads
LEDGER_KEYS = ("golds", "missed", "questions", "questions_missing_a_gold", "by_cause", "questions_by_cause")


def script_sha256():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


# ── the pick and its ledger, reproduced ──────────────────────────────────────


def pick_causes(d):
    """(verdict, pick, U2e's arrays, held, cause per gold, ledger, U2e's stats record), with the ledger checked
    against U2e's filed one."""
    v = json.loads((U2E.OUT / "verdict.json").read_text(encoding="utf-8"))
    if v.get("script_sha256") != U2E.script_sha256():
        raise SystemExit("outputs/u2e/verdict.json is not the committed zu2e.py's")
    pick = v["pick"]
    by = {c[0]: c for c in U2E.u2e_candidates()}
    if pick not in by:
        raise SystemExit(f"U2e's pick {pick} is not a U2e candidate")
    x = U2E.load_u2e(d)
    if x is None:
        raise SystemExit(f"{d}: no U2e stats")
    a, srec = x
    a2c = U2E.load_u2c(d)[0]
    _nm, ci, _bn, kind, prm, tau = by[pick]
    kq = U2E.budget_k(kind, prm, a["fill"])
    _r, held = U2E.evaluate(a, ci, kq, tau)
    _r1, held1 = U2E.evaluate(a, ci, kq, 1.0)
    reach = (a["pos"] >= 0).any(axis=1) | (a2c["pos"] >= 0).any(axis=1)
    led = U2E.ledger_of(held, held1 | held, reach | held1 | held, a["ptr"])
    filed = v["ledger"][d]
    if any(led[k] != filed[k] for k in LEDGER_KEYS):
        raise SystemExit(f"{d}: the pick's ledger {led} is not U2e's filed {filed}")
    cause = np.where(held, "HELD", np.where(held1, "TAU_CUT", np.where(reach, "BUDGET", "UNREACHED")))
    return v, pick, a, held, cause, led, srec


# ── graph helpers ────────────────────────────────────────────────────────────


def gather(ip, nodes):
    """Positions into col of the rows of nodes, and the row each position belongs to."""
    nodes = np.asarray(nodes, np.int64)
    s, e = ip[nodes], ip[nodes + 1]
    ln = (e - s).astype(np.int64)
    tot = int(ln.sum())
    if tot == 0:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    rep = np.repeat(np.arange(nodes.size), ln)
    off = np.arange(tot, dtype=np.int64) - np.repeat(np.cumsum(ln) - ln, ln)
    return np.repeat(s, ln) + off, rep


def neighbours(csrs, nodes):
    out = [col[gather(ip, nodes)[0]] for ip, col in csrs]
    return np.unique(np.concatenate(out).astype(np.int64)) if out else np.empty(0, np.int64)


def row_size(csrs, nodes):
    nodes = np.asarray(nodes, np.int64)
    return int(sum(int((ip[nodes + 1] - ip[nodes]).sum()) for ip, _c in csrs))


def distance(g, in_b, in_n1, csrs):
    """0 in B; 1 one link from B; 2; 3; 4 beyond three; -1 when the two-link neighbourhood is over HOPS_CAP."""
    if in_b[g]:
        return 0
    if in_n1[g]:
        return 1
    nb1 = neighbours(csrs, [g])
    if nb1.size and in_n1[nb1].any():
        return 2
    if row_size(csrs, nb1) > HOPS_CAP:
        return -1
    nb2 = neighbours(csrs, nb1) if nb1.size else nb1
    return 3 if (nb2.size and in_n1[nb2].any()) else 4


class Typed:
    """A typed KB's rows: (neighbour, relation, direction) per stored edge end, and the relation vectors."""

    def __init__(self, st, emb):
        self.ip = np.asarray(st.typed_indptr, np.int64)
        self.col = np.asarray(st.typed_col)
        self.rel = np.asarray(st.typed_rel)
        self.dir = np.asarray(st.typed_dir)
        self.emb = np.asarray(emb, np.float32)

    def row(self, u):
        s, e = int(self.ip[u]), int(self.ip[u + 1])
        return self.col[s:e], self.rel[s:e], self.dir[s:e]

    def group(self, u, r, dr, q, held_set):
        """(fan-out, relation rank at u by cos(q, e_r), a held gold in the group, the group's members)."""
        nb, rl, dd = self.row(u)
        m = (rl == r) & (dd == dr)
        members = nb[m]
        rels = np.unique(rl)
        sc = self.emb[rels] @ q
        rank = 1 + int((sc > float(self.emb[r] @ q) + 1e-7).sum())
        return int(m.sum()), rank, bool(held_set.intersection(members.tolist())), members


def typed_reads(g, T, in_b, q, held_set, kinds, cvt_code):
    """The typed reads of one gold (module docstring)."""
    out = {"hop1": None, "hop2": None}
    nb, rl, dd = T.row(g)
    at_b = in_b[nb]
    if at_b.any():
        best = None
        split = False
        for u, r, dr in zip(nb[at_b].tolist(), rl[at_b].tolist(), dd[at_b].tolist()):
            fan, rank, sib, _m = T.group(u, r, 3 - dr, q, held_set - {g})
            split |= sib
            if best is None or rank < best[1]:
                best = (fan, rank)
        out["hop1"] = {"fan": best[0], "rank": best[1], "split": split, "groups": int(at_b.sum())}
        return out
    mids = np.unique(nb)
    if mids.size > MIDS_CAP:
        mids = mids[:MIDS_CAP]
    best, any_cvt, split = None, False, False
    for x in mids.tolist():
        xb, xr, xd = T.row(x)
        e1 = in_b[xb]
        if not e1.any():
            continue
        any_cvt |= bool(kinds is not None and kinds[x] == cvt_code)
        for r2, d2 in set(zip(rl[nb == x].tolist(), dd[nb == x].tolist())):
            fan2, rank2, sib2, _m = T.group(x, r2, 3 - d2, q, held_set - {g})
            split |= sib2
            for u, r1, d1 in zip(xb[e1].tolist(), xr[e1].tolist(), xd[e1].tolist()):
                s = 0.5 * (float(T.emb[r1] @ q) + float(T.emb[r2] @ q))
                if best is None or s > best[0]:
                    fan1, rank1, _s1, _m1 = T.group(u, r1, 3 - d1, q, set())
                    best = (s, x, rank1, fan1, rank2, fan2)
    if best is not None:
        s, x, rank1, fan1, rank2, fan2 = best
        out["hop2"] = {"chain": round(s, 4), "mid_cvt": bool(kinds is not None and kinds[x] == cvt_code),
                       "any_mid_cvt": any_cvt, "rank1": rank1, "fan1": fan1, "rank2": rank2, "fan2": fan2,
                       "split": split}
    return out


# ── entity match ─────────────────────────────────────────────────────────────


def contains(a, b):
    """a and b normalized; True when one holds the other as whole words, the shorter at least NAME_MIN long."""
    if not a or not b:
        return False
    s, t = (a, b) if len(a) <= len(b) else (b, a)
    return len(s) >= NAME_MIN and f" {s} " in f" {t} "


def node_names_and_kinds(ds, wanted, norm):
    """{node: normalized name} for the wanted nodes, and every node's kind code with the code table."""
    n = int(ds.n_nodes)
    kinds = np.zeros(n, np.int8)
    table = {}
    names = {}
    for k, rec in enumerate(ds.nodes()):
        kd = str(rec.get("node_kind") or "")
        kinds[k] = table.setdefault(kd, len(table))
        if k in wanted:
            nm = rec.get("title") or rec.get("display_name") or rec.get("text") or ""
            names[k] = norm(str(nm))
    return names, kinds, {v: k for k, v in table.items()}


# ── summaries ────────────────────────────────────────────────────────────────


def dist_of(vals):
    vals = np.asarray(vals, np.float64)
    if vals.size == 0:
        return None
    return {"n": int(vals.size), "median": float(np.median(vals)), "p90": float(np.percentile(vals, 90)),
            "max": float(vals.max())}


def share_at_most(vals, ks):
    vals = np.asarray(vals)
    return {str(k): (round(float((vals <= k).mean()), 4) if vals.size else None) for k in ks}


def summarize(rows, fams):
    out = {}
    for c in CAUSES:
        rs = [r for r in rows if r["cause"] == c]
        if not rs:
            continue
        dd = [r["dist"] for r in rs]
        s = {"golds": len(rs),
             "distance": {str(k): int(sum(1 for x in dd if x == k)) for k in (0, 1, 2, 3, 4, -1)},
             "one_link_by_family": {f: int(sum(1 for r in rs if r["fam1"][f])) for f in fams},
             "name_in_question": int(sum(1 for r in rs if r.get("q_name"))),
             "name_match_B": int(sum(1 for r in rs if r.get("b_name"))),
             "kinds": {}}
        for r in rs:
            s["kinds"][r["kind"]] = s["kinds"].get(r["kind"], 0) + 1
        h1 = [r["typed"]["hop1"] for r in rs if r.get("typed") and r["typed"]["hop1"]]
        h2 = [r["typed"]["hop2"] for r in rs if r.get("typed") and r["typed"]["hop2"]]
        if h1 or h2:
            s["typed_hop1"] = {"golds": len(h1), "split_list": int(sum(x["split"] for x in h1)),
                               "relation_rank": dist_of([x["rank"] for x in h1]),
                               "rank_at_most": share_at_most([x["rank"] for x in h1], (1, 3, 10)),
                               "fan_out": dist_of([x["fan"] for x in h1])}
            s["typed_hop2"] = {"golds": len(h2), "mid_cvt": int(sum(x["mid_cvt"] for x in h2)),
                               "any_mid_cvt": int(sum(x["any_mid_cvt"] for x in h2)),
                               "split_list": int(sum(x["split"] for x in h2)),
                               "rank1": dist_of([x["rank1"] for x in h2]),
                               "rank2": dist_of([x["rank2"] for x in h2]),
                               "rank_both_at_most": share_at_most([max(x["rank1"], x["rank2"]) for x in h2],
                                                                  (1, 3, 10)),
                               "fan1": dist_of([x["fan1"] for x in h2]), "fan2": dist_of([x["fan2"] for x in h2])}
        out[c] = s
    return out


def question_reads(rows, nq_all):
    """Per question with a missed gold: its causes, whether every missed gold shares a typed group with a held gold
    (a split list), and the ALL that fixing exactly those questions would give."""
    byq = {}
    for r in rows:
        byq.setdefault(r["q"], []).append(r)
    n_split, n_unreached = 0, 0
    causes = {}
    for _q, rs in byq.items():
        miss = [r for r in rs if r["cause"] != "HELD"]
        cs = tuple(sorted({r["cause"] for r in miss}))
        causes["+".join(cs)] = causes.get("+".join(cs), 0) + 1
        n_unreached += "UNREACHED" in cs

        def split(r):
            t = r.get("typed") or {}
            return bool((t.get("hop1") or {}).get("split") or (t.get("hop2") or {}).get("split"))

        n_split += bool(miss) and all(split(r) for r in miss)
    nmiss = len(byq)
    return {"questions_missing_a_gold": nmiss, "by_causes": dict(sorted(causes.items())),
            "every_missed_gold_in_a_split_list": n_split,
            "ALL_now": round((nq_all - nmiss) / nq_all, 4),
            "ALL_if_the_split_lists_were_whole": round((nq_all - nmiss + n_split) / nq_all, 4)}


# ── the diagnostic ───────────────────────────────────────────────────────────


def cmd_diag(args):
    import step4e_paper_pools as E
    S4, C4, D4 = E.S4, E.C4, E.D4
    t0 = time.time()
    d = args.dataset
    _v, pick, a, held, cause, led, srec = pick_causes(d)
    log(f"{d}: U2e's ledger reproduced for {pick}: {led['by_cause']} ({time.time() - t0:.0f}s)")
    op = S4.Opened()
    if op.freeze["RECORD_SHA256"] != srec["freeze_RECORD_SHA256"]:
        raise SystemExit("the served freeze is not the one U2e's stats read")
    ds, _construction, families, stores, _union = E.open_dataset(op, d)
    n = int(ds.n_nodes)
    carve = U2C.carve_of(d)
    (_c, pop), = [(c, p) for c, p in C4.populations(op, S4, ds, d) if c == carve]
    path = D4.POOLS / f"{d}__{carve}.npz"
    ent = json.loads(D4.MANIFEST.read_text(encoding="utf-8"))["files"][f"{d}__{carve}"]
    if sha_file(path) != ent["sha256"]:
        raise SystemExit(f"{path}: not step 4d's manifest's file")
    with np.load(path) as z:
        if z["ids"].tolist() != list(pop.ids):
            raise SystemExit(f"{d}/{carve}: step 4d's ids are not the carve's")
        bsn, bsp = z["bs_nodes"].astype(np.int64), z["bs_ptr"].astype(np.int64)
    golds = [np.unique(np.asarray(x, np.int64)) for x in pop.golds]
    ptr = np.concatenate([[0], np.cumsum([x.size for x in golds])]).astype(np.int64)
    if not np.array_equal(ptr, a["ptr"]):
        raise SystemExit(f"{d}/{carve}: the carve's golds are not U2e's")
    desc = {"graph": "cached (the KB)"}
    fam_csr = {f: (np.asarray(stores[f].indptr, np.int64), np.asarray(stores[f].col)) for f in families}
    w2 = U2C.graph_csr(d, "chosen", n)
    if w2 is not None:
        ip, col, desc = w2
        fam_csr["structural"] = (np.asarray(ip, np.int64), np.asarray(col))
    csrs = list(fam_csr.values())
    T = None
    st = stores.get("structural")
    if w2 is None and st is not None and getattr(st, "typed_indptr", None) is not None:
        rt = op.m3b_compile.relation_table_for(ds, d, stores)
        if rt is not None:
            T = Typed(st, rt.embeddings)
    qv = np.asarray(ds.embeddings("dense", "queries").read(pop.idx), np.float32)
    qv /= np.maximum(np.linalg.norm(qv, axis=1, keepdims=True), 1e-12)
    texts = E.query_texts(ds)
    log(f"{d}/{carve}: opened ({n:,} nodes; families {list(fam_csr)}; typed {T is not None}) ({time.time() - t0:.0f}s)")

    qi = np.repeat(np.arange(len(golds)), np.diff(ptr))
    qs = np.unique(qi[cause != "HELD"])
    wanted = set()
    for j in qs.tolist():
        if bsp[j + 1] - bsp[j] != int(a["bs"][j]):
            raise SystemExit(f"{d}/{carve} question {j}: B is not U2e's")
        wanted.update(bsn[bsp[j]:bsp[j + 1]].tolist())
        wanted.update(golds[j].tolist())
    names, kinds, table = node_names_and_kinds(ds, wanted, E.norm)
    cvt = {v_: k for k, v_ in table.items()}.get("CVT_MEDIATOR")
    log(f"{d}/{carve}: names of {len(names):,} nodes and every node's kind read ({time.time() - t0:.0f}s)")

    in_b = np.zeros(n, bool)
    in_n1 = np.zeros(n, bool)
    rows = []
    for j in qs.tolist():
        B = bsn[bsp[j]:bsp[j + 1]]
        g0, g1 = int(ptr[j]), int(ptr[j + 1])
        gj = golds[j]
        if not np.array_equal(np.isin(gj, B), a["pos"][g0:g1, 0] == 0):
            raise SystemExit(f"{d}/{carve} question {j}: the in-B flags are not U2e's")
        in_b[B] = True
        n1 = neighbours(csrs, B)
        in_n1[n1] = True
        held_set = set(gj[held[g0:g1]].tolist())
        for k, g in enumerate(gj.tolist()):
            r = {"q": j, "g": g, "cause": str(cause[g0 + k]), "dist": distance(g, in_b, in_n1, csrs),
                 "fam1": {f: bool(in_b[col[ip[g]:ip[g + 1]]].any()) for f, (ip, col) in fam_csr.items()}}
            if T is not None:
                r["typed"] = typed_reads(g, T, in_b, qv[j], held_set, kinds, cvt)
            rows.append(r)
        in_b[B] = False
        in_n1[n1] = False
    log(f"{d}/{carve}: {len(rows):,} golds of {qs.size} questions read ({time.time() - t0:.0f}s)")

    for r in rows:
        j = r["q"]
        gn = names.get(r["g"], "")
        qn = E.norm(texts[int(pop.idx[j])])
        r["kind"] = table[int(kinds[r["g"]])] or "(none)"
        r["q_name"] = contains(gn, qn) and len(gn) <= len(qn)
        B = bsn[bsp[j]:bsp[j + 1]]
        r["b_name"] = any(contains(gn, names.get(int(b), "")) for b in B.tolist() if int(b) != r["g"])
    log(f"{d}/{carve}: names and kinds read ({time.time() - t0:.0f}s)")

    rec = {"stage": "U2l diagnostic (decides nothing)", "dataset": d, "carve": carve, "pick": pick,
           "graph_detail": desc, "script_sha256": script_sha256(), "zu2e_sha256": U2E.script_sha256(),
           "freeze_RECORD_SHA256": op.freeze["RECORD_SHA256"], "ledger": led,
           "node_kinds": {table[k]: int((kinds == k).sum()) for k in table},
           "golds_per_question": dist_of(np.diff(ptr)),
           "by_cause": summarize(rows, list(fam_csr)), "questions": question_reads(rows, len(golds)),
           "caps": {"hops": HOPS_CAP, "mids": MIDS_CAP, "name_min": NAME_MIN}}
    if w2 is not None and "ner" in fam_csr:
        rec["links_vs_ner"] = links_vs_ner(fam_csr["structural"], fam_csr["ner"], n, golds, bsn, bsp, fam_csr)
    rec["seconds"] = round(time.time() - t0, 1)
    rec["utc"] = utc()
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / f"{d}.json", rec)
    (OUT / f"{d}.md").write_text(report_md(rec), encoding="utf-8", newline="\n")
    log(f"{d}: written to {OUT / (d + '.json')} ({rec['seconds']}s)")
    return 0


def pair_keys(ip, col, n):
    u = np.repeat(np.arange(ip.size - 1, dtype=np.int64), np.diff(ip))
    m = u < col
    return np.unique(u[m] * np.int64(n) + col[m])


def links_vs_ner(w2, ner, n, golds, bsn, bsp, fam_csr):
    """The share of W2_S's pairs that are NER pairs and back, and the golds one link from B by each family alone."""
    kw, kn = pair_keys(*w2, n), pair_keys(*ner, n)
    both = int(np.isin(kw, kn, assume_unique=True).sum())
    out = {"w2s_pairs": int(kw.size), "ner_pairs": int(kn.size), "both": both,
           "share_of_w2s_in_ner": round(both / max(1, kw.size), 4),
           "share_of_ner_in_w2s": round(both / max(1, kn.size), 4)}
    del kw, kn
    in_b = np.zeros(n, bool)
    cnt = {}
    for j, gj in enumerate(golds):
        B = bsn[bsp[j]:bsp[j + 1]]
        in_b[B] = True
        for g in gj.tolist():
            if in_b[g]:
                key = "in B"
            else:
                hit = [f for f, (ip, col) in fam_csr.items() if in_b[col[ip[g]:ip[g + 1]]].any()]
                key = "+".join(hit) if hit else "none"
            cnt[key] = cnt.get(key, 0) + 1
        in_b[B] = False
    out["golds_one_link_from_B"] = dict(sorted(cnt.items()))
    return out


def report_md(rec):
    L = [f"# U2l diagnostic: {rec['dataset']} / {rec['carve']} (decides nothing)", "",
         f"Pick `{rec['pick']}`. Ledger (U2e's, reproduced): `{json.dumps(rec['ledger']['by_cause'])}`, "
         f"{rec['ledger']['questions_missing_a_gold']} questions missing a gold.", "",
         f"Questions: `{json.dumps(rec['questions'])}`", "", "## By cause", ""]
    for c, s in rec["by_cause"].items():
        L += [f"### {c}", "", "```json", json.dumps(s, indent=1), "```", ""]
    if "links_vs_ner" in rec:
        L += ["## Links against NER", "", "```json", json.dumps(rec["links_vs_ner"], indent=1), "```", ""]
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("diag")
    p.add_argument("--dataset", required=True, choices=U2C.SIX)
    p.set_defaults(fn=cmd_diag)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
