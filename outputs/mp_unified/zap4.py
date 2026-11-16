"""AP4 (AP's D80): the label precision ledger (docs/AP4_LABEL_PRECISION_LEDGER.md). A scouting stage at the graph level: it
reads the cache and links parts of D83's labelled arms (D83E: the six s1eval carves; B1V: b1v on B1's three settings) and of
the record's arms whose pools they reuse (REC: PKO's s1eval; B1W: B1P's b1w). It reads no model and trains nothing.

For every question with a gold, the pool's own rrf column orders the pool and gives the top five. There are two anchors:
- each gold in the top five ("found": the diagnostic anchor);
- rrf's first row ("top1": the anchor a model has without knowing the golds).

An anchor's candidates are its linked rows outside the top five, each with its label bits (either direction). D83's labels
are S, R.mention, R.name_comention, R.title_comention and K.knn; the record's families are slot, NER.spacy and K.knn. A
target is a gold among the candidates.

    python outputs/mp_unified/zap4.py ledger --arm D83E|REC|B1V|B1W --dataset D [--limit N --out DIR]
                                                     # host: outputs/ap4/ledger/<arm>/<ds>.npz and .json
    python outputs/mp_unified/zap4.py report         # outputs/ap4/report.json and report.md, from the ledgers present
    python outputs/mp_unified/zap4.py selftest

The measures, the calls and the expectations are fixed in section 3 of the page, before any ledger exists.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

import zd83rep as REP  # noqa: E402  (the arms, labels, json and sha helpers; read only)

DECLARED = "docs/AP4_LABEL_PRECISION_LEDGER.md"
STAGE = "AP4"
OUT = ROOT / "outputs" / "ap4"
LEDGER = OUT / "ledger"
TOP = 5
BOOT, SEED, CHUNK = 2000, 0, 100
KB = ("metaqa", "webqsp")
ARMS = REP.LS_ARMS                    # arm -> (root, carve, names, datasets)
PAIR = REP.LS_PAIR                    # D83 arm -> the record arm whose pools it reuses
KINDS = ("found", "top1")
POP = np.array([bin(i).count("1") for i in range(256)], np.int64)   # label-set sizes
QK = ("n", "gt", "hop", "found", "missed", "out", "inpool", "top1gold")
PK = ("q", "kind", "a", "arank", "c", "bits", "tgt", "rk", "rkc")
PDT = {"q": np.int32, "kind": np.int8, "a": np.int32, "arank": np.int16, "c": np.int32, "bits": np.int8, "tgt": bool,
       "rk": np.int32, "rkc": np.int32}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] zap4: {msg}", flush=True)


# ── one question ─────────────────────────────────────────────────────────────


def question_pairs(n, g, gt, rrf, cos, u, v, f):
    """Return one question's counts and its anchors' candidate lists.

    rrf's order and top five come from CH.order_of (descending, NaN last, ties by pool position). The anchors are the golds
    in the top five (kind 0, in rrf's order) and rrf's first row (kind 1). Each anchor's candidates are its linked rows
    outside the top five, with:
    - their label bits (LTW.fam_bits, either direction);
    - their target flags;
    - their 1-based ranks by rrf and by dense_cos.
    """
    import chndiag as CH
    import ltwdiag as LTW
    g = np.asarray(g, bool)
    o = CH.order_of(rrf)
    top = o[:TOP]
    in_top = np.zeros(n, bool)
    in_top[top] = True
    rk = np.empty(n, np.int64)
    rk[o] = np.arange(1, n + 1)
    rkc = np.empty(n, np.int64)
    rkc[CH.order_of(cos)] = np.arange(1, n + 1)
    q = {"found": int(g[top].sum()), "missed": int((g & ~in_top).sum()), "out": int(n - top.size), "inpool": int(g.sum()),
         "top1gold": int(bool(n) and bool(g[o[0]]))}
    rows = []
    if gt > 0 and n:
        anchors = [(0, int(a)) for a in top if g[a]] + [(1, int(o[0]))]
        for kind, a in anchors:
            b = LTW.fam_bits(n, a, u, v, f)
            c = np.flatnonzero((b > 0) & ~in_top)
            rows.append((kind, a, int(rk[a]), c, b[c], g[c], rk[c], rkc[c]))
    return q, rows, top


class Acc:
    """The ledger's arrays, question by question (the ledger step and the self-test build them the same way)."""

    def __init__(self):
        self.Q = {k: [] for k in QK}
        self.P = {k: [] for k in PK}
        self.qi = 0

    def add(self, n, gt, hop, q, rows):
        for k, x in (("n", n), ("gt", gt), ("hop", hop), *q.items()):
            self.Q[k].append(int(x))
        for kind, a, ar, c, b, t, r, rc in rows:
            m = c.size
            for k, x in (("q", np.full(m, self.qi)), ("kind", np.full(m, kind)), ("a", np.full(m, a)),
                         ("arank", np.full(m, ar)), ("c", c), ("bits", b), ("tgt", t), ("rk", r), ("rkc", rc)):
                self.P[k].append(np.asarray(x).astype(PDT[k]))
        self.qi += 1

    def arrays(self):
        z = {f"Q_{k}": np.asarray(v, np.int64) for k, v in self.Q.items()}
        for k in PK:
            z[f"P_{k}"] = np.concatenate(self.P[k]) if self.P[k] else np.zeros(0, PDT[k])
        return z


def hop_of(ds, qid):
    s = str(qid)
    return int(s[0]) if ds == "musique" and s[:1] in ("2", "3", "4") else 0


# ── the ledger (host) ────────────────────────────────────────────────────────


def ledger(arm, ds, limit=None, out=None):
    import lean_cache as LC
    import ltwdiag as LTW
    if arm not in ARMS or ds not in ARMS[arm][3]:
        raise SystemExit(f"AP4: ledger takes an arm of {', '.join(ARMS)} and its datasets")
    root_rel, carve, names, _dss = ARMS[arm]
    root = ROOT / root_rel
    L = len(names)
    t0 = time.time()
    parts, recs = LC.part_dirs(ds, carve, root / "cache")
    head = recs[0]["look"]
    cols, spans, _ci = LC.xc_columns(head)
    cn = [head["columns"][int(c)] for c in cols]
    r0, r1 = spans["rank"]
    jd = spans["dense_cos"][0]
    if tuple(cn[r0:r1]) != LTW.RANK_COLS or cn[jd] != "dense_cos":
        raise SystemExit(f"AP4: {arm} {ds}/{carve}: xc does not hold {LTW.RANK_COLS} and dense_cos where xc_columns puts them")
    jr = r0 + LTW.RANK_COLS.index("rrf")
    ids = [str(x) for r in recs for x in r["ids"]]
    acc = Acc()
    hp, ht = hashlib.sha256(), hashlib.sha256()
    plist = []
    for p, r in zip(parts, recs):
        if limit and acc.qi >= limit:
            break
        A, E, lsha = LTW.part_arrays(p, r, root / "links")
        x2 = np.asarray(A["xc"][:, [jr, jd]], np.float64)
        gold, node = np.asarray(A["gold"], bool), np.asarray(A["node"], np.int64)
        nq, gtot = np.asarray(A["n"], np.int64), np.asarray(A["gold_total"], np.int64)
        eu, ev, ef = (np.asarray(E[k], np.int64) for k in ("e_u", "e_v", "e_fam"))
        if ef.size and (ef.min() < 0 or ef.max() >= L):
            raise SystemExit(f"AP4: {arm} {ds} {p.name}: a link's label is outside the arm's {L} names")
        no = np.r_[0, np.cumsum(nq)]
        eo = np.r_[0, np.cumsum(np.asarray(E["q_ne"], np.int64))]
        for i in range(nq.size):
            if limit and acc.qi >= limit:
                break
            lo, hi = int(no[i]), int(no[i + 1])
            n, g = hi - lo, gold[lo:hi]
            s = slice(int(eo[i]), int(eo[i + 1]))
            q, rows, top = question_pairs(n, g, int(gtot[i]), x2[lo:hi, 0], x2[lo:hi, 1], eu[s], ev[s], ef[s])
            hp.update(np.int64(n).tobytes())
            hp.update(node[lo:hi].tobytes())
            hp.update(g.astype(np.uint8).tobytes())
            ht.update(np.asarray(top, np.int64).tobytes())
            acc.add(n, int(gtot[i]), hop_of(ds, ids[acc.qi]), q, rows)
        plist.append({"part": p.name, "rows": int(r["rows"]), "queries": int(r["queries"]),
                      "cache_record_sha256": LC.sha_file(p / "record.json"), "links_record_sha256": lsha})
        del A, E, x2, gold, node, eu, ev, ef
        log(f"ledger {arm} {ds}/{carve} {p.name}: {acc.qi} questions ({time.time() - t0:.0f}s)")
    if not limit and acc.qi != len(ids):
        raise SystemExit(f"AP4: {arm} {ds}: read {acc.qi} questions, the records name {len(ids)}")
    z = acc.arrays()
    dst = Path(out) if out else LEDGER / arm
    dst.mkdir(parents=True, exist_ok=True)
    npz = dst / f"{ds}.npz"
    tmp = dst / f"{ds}.tmp.npz"
    np.savez_compressed(tmp, **z)
    tmp.replace(npz)
    kinds = z["P_kind"]
    meta = {"declared_in": DECLARED, "stage": STAGE, "arm": arm, "dataset": ds, "carve": carve, "root": root_rel,
            "names": list(names), "limit": limit, "questions": acc.qi, "with_gold": int((z["Q_gt"] > 0).sum()),
            "pairs": {k: int((kinds == j).sum()) for j, k in enumerate(KINDS)}, "targets": int(z["P_tgt"].sum()),
            "ids_sha256": hashlib.sha256("\n".join(ids[:acc.qi]).encode()).hexdigest(),
            "pool_sha256": hp.hexdigest(), "top5_sha256": ht.hexdigest(), "npz_sha256": REP.sha_file(npz),
            "parts": plist, "zap4_sha256": REP.sha_src(__file__), "seconds": round(time.time() - t0, 1), "utc": REP.utc()}
    REP.write_json(dst / f"{ds}.json", meta)
    log(f"ledger {arm} {ds}: {acc.qi} questions, pairs {meta['pairs']}, targets {meta['targets']}")
    return meta


# ── the measures ─────────────────────────────────────────────────────────────


def units(arm, ds):
    """The named bit masks: each label (or family) alone, the S/R/K groups, and any link."""
    names = ARMS[arm][2]
    u = {f"label:{nm}": 1 << i for i, nm in enumerate(names)}
    if tuple(names) == tuple(REP.LABELS):
        grp = {"S": 1, "R": 2 | 4 | 8, "K": 16} if ds in KB else {"R": 2 | 4 | 8, "K": 16}
        if ds not in KB:
            del u["label:S"]
    else:
        grp = {"S": 1, "R": 2, "K": 4} if ds in KB else {"R": 1 | 2, "K": 4}
    u.update({f"group:{k}": m for k, m in grp.items()})
    u["any"] = (1 << len(names)) - 1
    return u


def ratio(num, den):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.asarray(num, float) / np.asarray(den, float)


def boot(fn, cols, boot_n=BOOT, seed=SEED):
    """fn of the summed columns (each an array over resamples), over questions resampled with replacement (seeded): the
    point value and the 95% interval, or None when the point value is not finite."""
    cols = {k: np.asarray(v, float) for k, v in cols.items()}
    n = len(next(iter(cols.values())))
    pt = float(fn({k: np.array([v.sum()]) for k, v in cols.items()})[0])
    if n == 0 or not np.isfinite(pt):
        return None
    rng = np.random.default_rng(seed)
    vals, done = [], 0
    while done < boot_n:
        b = min(CHUNK, boot_n - done)
        idx = rng.integers(0, n, size=(b, n))
        vals.append(fn({k: v[idx].sum(1) for k, v in cols.items()}))
        done += b
    d = np.concatenate(vals)
    d = d[np.isfinite(d)]
    return {"pt": round(pt, 5), "ci": [round(float(np.percentile(d, 2.5)), 5), round(float(np.percentile(d, 97.5)), 5)]}


def call(ci):
    return None if ci is None else ("ABOVE" if ci["ci"][0] > 0 else ("BELOW" if ci["ci"][1] < 0 else "LEVEL"))


def lift_call(ci):
    return None if ci is None else ("POINTS" if ci["ci"][0] > 1 else ("BELOW_BASE" if ci["ci"][1] < 1 else "NO_LIFT"))


def per_question(z, m, kind, exact=None):
    """Per-question sums over one kind's pairs carrying any bit of mask m (or, with exact, exactly that bit set; or, with
    exact = 'multi' / 'single', two or more / one label): pairs, targets, targets reached (unique rows), anchors with a
    target among the pairs, and those whose best target is first (by rrf, by dense_cos) or in the first two (by rrf);
    the best positions (by rrf) of the anchors with a target."""
    nq = z["Q_n"].size
    bits = z["P_bits"].astype(np.int64)
    sel = z["P_kind"] == kind
    if exact is None:
        sel &= (bits & m) != 0
    elif exact == "multi":
        sel &= POP[bits & 255] >= 2
    elif exact == "single":
        sel &= POP[bits & 255] == 1
    else:
        sel &= bits == exact
    q = z["P_q"][sel].astype(np.int64)
    t = z["P_tgt"][sel].astype(bool)
    c = z["P_c"][sel].astype(np.int64)
    a = z["P_a"][sel].astype(np.int64)
    big = int(z["Q_n"].max()) + 1 if nq else 1
    out = {"den": np.bincount(q, minlength=nq).astype(float), "num": np.bincount(q, weights=t.astype(float), minlength=nq)}
    key = np.unique(q[t] * big + c[t])
    out["reached"] = np.bincount(key // big, minlength=nq).astype(float)
    best = np.zeros(0, np.int64)
    for rname, k1, k2 in (("rk", "r1", "t2"), ("rkc", "r1c", None)):
        r = z[f"P_{rname}"][sel].astype(np.int64)
        gk = q * big + a
        o = np.lexsort((r, gk))
        gs, ts = gk[o], t[o]
        out[k1] = np.zeros(nq)
        if k2:
            out[k2] = np.zeros(nq)
            out["wt"] = np.zeros(nq)
        if not gs.size:
            continue
        st = np.r_[0, np.flatnonzero(np.diff(gs)) + 1]
        pos = np.arange(gs.size) - np.repeat(st, np.diff(np.r_[st, gs.size])) + 1
        bp = np.minimum.reduceat(np.where(ts, pos, np.iinfo(np.int64).max), st)
        gq = gs[st] // big
        has = bp < np.iinfo(np.int64).max
        out[k1] = np.bincount(gq[has & (bp == 1)], minlength=nq).astype(float)
        if k2:
            out[k2] = np.bincount(gq[has & (bp <= 2)], minlength=nq).astype(float)
            out["wt"] = np.bincount(gq[has], minlength=nq).astype(float)
            best = bp[has]
    return out, best


def qsel_of(z, kind, hop=None):
    s = (z["Q_gt"] > 0) & (z["Q_n"] > 0)
    if kind == 0:
        s &= z["Q_found"] > 0
    if hop is not None:
        s &= z["Q_hop"] == hop
    return s


def anchors_of(z, kind):
    return z["Q_found"].astype(float) if kind == 0 else np.ones(z["Q_n"].size)


def unit_block(z, m, kind, qs, anys=None, exact=None):
    """One unit's measures on one kind's anchors over the questions qs (section 3)."""
    pq, best = per_question(z, m, kind, exact)
    S = {k: v[qs] for k, v in pq.items()}
    mis, out_, anc = z["Q_missed"][qs].astype(float), z["Q_out"][qs].astype(float), anchors_of(z, kind)[qs]
    base = float(ratio(mis.sum(), out_.sum()))
    rec = {"questions": int(qs.sum()), "anchors": int(anc.sum()), "pairs": int(S["den"].sum()), "targets": int(S["num"].sum()),
           "price": round(float(ratio(S["den"].sum(), anc.sum())), 4), "base": round(base, 6),
           "precision": boot(lambda s: ratio(s["num"], s["den"]), {"num": S["num"], "den": S["den"]}),
           "lift": boot(lambda s: ratio(s["num"], s["den"]) / ratio(s["mis"], s["out"]),
                        {"num": S["num"], "den": S["den"], "mis": mis, "out": out_}),
           "rank1": boot(lambda s: ratio(s["r1"], s["wt"]), {"r1": S["r1"], "wt": S["wt"]}),
           "top2": round(float(ratio(S["t2"].sum(), S["wt"].sum())), 5) if S["wt"].sum() else None,
           "rank1_cos": round(float(ratio(S["r1c"].sum(), S["wt"].sum())), 5) if S["wt"].sum() else None,
           "best_median": float(np.median(best)) if best.size else None}
    if kind == 0:
        rec["reach"] = boot(lambda s: ratio(s["reached"], s["mis"]), {"reached": S["reached"], "mis": mis})
    rec["lift_call"] = lift_call(rec["lift"])
    if anys is not None:
        d = boot(lambda s: ratio(s["num"], s["den"]) - ratio(s["na"], s["da"]),
                 {"num": S["num"], "den": S["den"], "na": anys["num"][qs], "da": anys["den"][qs]})
        rec["vs_any"] = {"diff": d, "call": call(d)}
    return rec


def combos(z, names, kind, qs, keep=0.99, most=12):
    """The exact label sets of one kind's pairs, most frequent first, until `keep` of the pairs are listed (at most `most`)."""
    sel = (z["P_kind"] == kind) & qs[z["P_q"].astype(np.int64)]
    b = z["P_bits"][sel].astype(np.int64)
    if not b.size:
        return []
    vals, cnt = np.unique(b, return_counts=True)
    o = np.argsort(-cnt, kind="stable")
    out, seen = [], 0
    for j in o[:most]:
        bs = int(vals[j])
        r = unit_block(z, 0, kind, qs, exact=bs)
        out.append({"labels": [nm for i, nm in enumerate(names) if bs >> i & 1], "bits": bs, "share": round(cnt[j] / b.size, 5),
                    "pairs": r["pairs"], "precision": r["precision"], "lift": r["lift"]})
        seen += cnt[j]
        if seen >= keep * b.size:
            break
    return out


def arm_block(z, arm, ds):
    names = ARMS[arm][2]
    U = units(arm, ds)
    rec = {"questions": int(z["Q_n"].size), "with_gold": int((z["Q_gt"] > 0).sum()), "units": {}, "combos": {}, "multi": {}}
    for kind, kn in enumerate(KINDS):
        qs = qsel_of(z, kind)
        anys, _ = per_question(z, U["any"], kind)
        for un, m in U.items():
            rec["units"].setdefault(un, {})[kn] = unit_block(z, m, kind, qs, None if un == "any" else anys)
        rec["combos"][kn] = combos(z, names, kind, qs)
        pm, _ = per_question(z, 0, kind, "multi")
        ps, _ = per_question(z, 0, kind, "single")
        d = boot(lambda s: ratio(s["nm"], s["dm"]) - ratio(s["ns"], s["dsg"]),
                 {"nm": pm["num"][qs], "dm": pm["den"][qs], "ns": ps["num"][qs], "dsg": ps["den"][qs]})
        rec["multi"][kn] = {"multi_pairs": int(pm["den"][qs].sum()), "single_pairs": int(ps["den"][qs].sum()),
                            "diff": d, "call": call(d)}
    if ds == "musique":
        rec["hops"] = {}
        for h in (2, 3, 4):
            for kind, kn in enumerate(KINDS):
                qs = qsel_of(z, kind, h)
                anys, _ = per_question(z, U["any"], kind)
                rec["hops"].setdefault(str(h), {})[kn] = {un: unit_block(z, m, kind, qs, None if un == "any" else anys)
                                                          for un, m in U.items()}
    return rec


def pair_block(za, zb, ma, mb, ds):
    """D83's arm (a) against the record's arm (b) on the same questions, pools and top fives (G6 first): the groups' and any
    link's precision (both kinds), reach and rank1 (found), and price, each a minus b with its paired interval; and the
    K pairs at the anchors (they should be the same pairs: D83's K.knn label is read before the dedup)."""
    for k in ("ids_sha256", "pool_sha256", "top5_sha256"):
        if ma[k] != mb[k]:
            raise SystemExit(f"AP4: G6: {ds}: {ma['arm']} and {mb['arm']} differ in {k}")
    if not (np.array_equal(za["Q_n"], zb["Q_n"]) and np.array_equal(za["Q_gt"], zb["Q_gt"])):
        raise SystemExit(f"AP4: G6: {ds}: the ledgers' questions differ")
    Ua, Ub = units(ma["arm"], ds), units(mb["arm"], ds)
    out = {}
    for gname in [x for x in ("group:S", "group:R", "group:K", "any") if x in Ua and x in Ub]:
        rec = {}
        for kind, kn in enumerate(KINDS):
            qs = qsel_of(za, kind)
            pa, _ = per_question(za, Ua[gname], kind)
            pb, _ = per_question(zb, Ub[gname], kind)
            anc = anchors_of(za, kind)[qs]
            r = {"precision": boot(lambda s: ratio(s["na"], s["da"]) - ratio(s["nb"], s["db"]),
                                   {"na": pa["num"][qs], "da": pa["den"][qs], "nb": pb["num"][qs], "db": pb["den"][qs]}),
                 "price": boot(lambda s: ratio(s["da"], s["an"]) - ratio(s["db"], s["an"]),
                               {"da": pa["den"][qs], "db": pb["den"][qs], "an": anc}),
                 "rank1": boot(lambda s: ratio(s["ra"], s["wa"]) - ratio(s["rb"], s["wb"]),
                               {"ra": pa["r1"][qs], "wa": pa["wt"][qs], "rb": pb["r1"][qs], "wb": pb["wt"][qs]})}
            if kind == 0:
                mis = za["Q_missed"][qs].astype(float)
                r["reach"] = boot(lambda s: ratio(s["ra"], s["m"]) - ratio(s["rb"], s["m"]),
                                  {"ra": pa["reached"][qs], "rb": pb["reached"][qs], "m": mis})
            for k in list(r):
                r[k] = {"diff": r[k], "call": call(r[k])}
            rec[kn] = r
        out[gname] = rec

    def kset(z, m):
        s = (z["P_bits"].astype(np.int64) & m) != 0
        big = int(z["Q_n"].max()) + 1
        k = ((z["P_q"][s].astype(np.int64) * 2 + z["P_kind"][s]) * big + z["P_a"][s]) * big + z["P_c"][s]
        return set(np.unique(k).tolist())
    ka, kb = kset(za, Ua["group:K"]), kset(zb, Ub["group:K"])
    out["K_pairs_at_anchors"] = {"d83": len(ka), "record": len(kb), "both": len(ka & kb), "same": ka == kb}
    return out


# ── the report ───────────────────────────────────────────────────────────────


def load_ledger(arm, ds, root=LEDGER):
    p, j = Path(root) / arm / f"{ds}.npz", Path(root) / arm / f"{ds}.json"
    if not (p.exists() and j.exists()):
        return None
    meta = REP.read_json(j)
    if meta.get("limit"):
        raise SystemExit(f"AP4: {p} is a limited dry run, not a ledger")
    if REP.sha_file(p) != meta["npz_sha256"]:
        raise SystemExit(f"AP4: {p}: its sha256 is not its record's")
    with np.load(p) as f:
        z = {k: f[k] for k in f.files}
    return z, meta


def pt(x):
    return None if x is None else x["pt"]


def expectations(arms):
    """Section 3's six expectations, read from the report's numbers (HELD, NOT_HELD or NOT_READ); they decide nothing."""
    d83 = [a for a in ("D83E", "B1V") if a in arms]
    E = {}

    def top1(arm, ds, un, what="precision"):
        try:
            return pt(arms[arm][ds]["units"][un]["top1"][what])
        except KeyError:
            return None

    def labels_with_pairs(arm, ds):
        return [un for un in arms[arm][ds]["units"] if un.startswith("label:") and arms[arm][ds]["units"][un]["top1"]["pairs"] > 0]

    res = []
    for arm in d83:
        for ds in ("2wiki", "hotpotqa"):
            if ds in arms[arm]:
                L = labels_with_pairs(arm, ds)
                res.append(max(L, key=lambda u: top1(arm, ds, u) or -1) == "label:R.mention")
    E["E1"] = {"text": "On 2Wiki and HotpotQA, R.mention has the highest top1 precision of D83's labels.", "held": res}
    res = []
    for arm in d83:
        for ds in [x for x in arms[arm] if x not in KB]:
            L = labels_with_pairs(arm, ds)
            res.append(min(L, key=lambda u: top1(arm, ds, u, "lift") or 1e9) == "label:K.knn")
    E["E2"] = {"text": "K.knn has the lowest top1 lift of D83's labels on every passage dataset.", "held": res}
    res = [arms[arm][ds]["multi"]["top1"]["call"] == "ABOVE" for arm in d83 for ds in arms[arm]]
    E["E3"] = {"text": "Pairs with two or more labels are ABOVE single-label pairs in top1 precision, on every dataset.",
               "held": res}
    res = []
    if "D83E" in arms:
        for ds in [x for x in KB if x in arms["D83E"]]:
            L = labels_with_pairs("D83E", ds)
            res.append(max(L, key=lambda u: top1("D83E", ds, u) or -1) == "label:S")
    E["E4"] = {"text": "On MetaQA and WebQSP, S has the highest top1 precision of D83's labels.", "held": res}
    res = []
    for arm in d83:
        if "musique" in arms[arm]:
            r = [pt(arms[arm]["musique"]["hops"][str(h)]["found"]["any"].get("reach")) for h in (2, 3, 4)]
            res.append(None not in r and r[0] > r[1] > r[2])
    E["E5"] = {"text": "MuSiQue's found-anchored reach of any link falls with the hops (2 > 3 > 4).", "held": res}
    E["E6"] = {"text": "D83's R is not BELOW the record's R in top1 precision on any paired dataset.", "held": None}
    for e in E.values():
        h = e["held"]
        e["call"] = "NOT_READ" if not h else ("HELD" if all(h) else "NOT_HELD")
    return E


def build(root=LEDGER):
    arms, metas = {}, {}
    for arm, (_r, _c, _n, dss) in ARMS.items():
        for ds in dss:
            got = load_ledger(arm, ds, root)
            if got is None:
                continue
            z, meta = got
            log(f"report: {arm} {ds} ({meta['questions']} questions, {sum(meta['pairs'].values())} pairs)")
            arms.setdefault(arm, {})[ds] = arm_block(z, arm, ds)
            metas[(arm, ds)] = (z, meta)
    pairs = {}
    for a, b in PAIR.items():
        for ds in ARMS[a][3]:
            if (a, ds) in metas and (b, ds) in metas:
                (za, ma), (zb, mb) = metas[(a, ds)], metas[(b, ds)]
                pairs.setdefault(f"{a}-{b}", {})[ds] = pair_block(za, zb, ma, mb, ds)
    E = expectations(arms)
    e6 = [pairs[p][ds]["group:R"]["top1"]["precision"]["call"] != "BELOW" for p in pairs for ds in pairs[p]]
    E["E6"]["held"] = e6
    E["E6"]["call"] = "NOT_READ" if not e6 else ("HELD" if all(e6) else "NOT_HELD")
    weights = {}
    for arm in [x for x in ("D83E", "B1V") if x in arms]:
        for ds, r in arms[arm].items():
            for un, rec in r["units"].items():
                lf = pt(rec["top1"]["lift"])
                if un.startswith("label:") and lf and lf > 0:
                    weights.setdefault(arm, {}).setdefault(ds, {})[un[6:]] = round(float(np.log(lf)), 4)
    return {"declared_in": DECLARED, "stage": STAGE, "utc": REP.utc(), "zap4_sha256": REP.sha_src(__file__),
            "ledgers": {f"{a}/{d}": {k: m[k] for k in ("questions", "with_gold", "pairs", "targets", "ids_sha256",
                                                        "pool_sha256", "top5_sha256", "npz_sha256", "zap4_sha256")}
                        for (a, d), (_z, m) in metas.items()},
            "arms": arms, "pairs": pairs, "expectations": E, "weights_named_not_adopted": weights}


def ci_txt(x, pct=True):
    if x is None:
        return "-"
    f = (lambda v: f"{100 * v:.1f}") if pct else (lambda v: f"{v:.2f}")
    return f"{f(x['pt'])} [{f(x['ci'][0])}, {f(x['ci'][1])}]"


def write_md(rec, path):
    L = [f"# AP4: the label precision ledger ({rec['utc']})", "",
         f"Declared in `{DECLARED}`. Graph level; no model is read. Precision, reach and rank1 are in %; lift is precision over "
         "the base rate (golds outside rrf's top five per row outside it). Intervals: 95%, 2,000 resamples of questions, seed 0.", "",
         "## Expectations (section 3; they decide nothing)", "", "| id | expectation | call | per read |", "|---|---|---|---|"]
    for k, e in rec["expectations"].items():
        L.append(f"| {k} | {e['text']} | {e['call']} | {e['held']} |")
    for p, byds in rec["pairs"].items():
        L += ["", f"## {p}: D83 minus the record, same questions and pools", "",
              "| dataset | group | top1 precision | top1 price | found reach | found rank1 |", "|---|---|---|---|---|---|"]
        for ds, r in byds.items():
            for g, x in r.items():
                if g == "K_pairs_at_anchors":
                    continue
                L.append(f"| {ds} | {g} | {ci_txt(x['top1']['precision']['diff'])} {x['top1']['precision']['call']} | "
                         f"{ci_txt(x['top1']['price']['diff'], False)} | {ci_txt(x['found']['reach']['diff'])} "
                         f"{x['found']['reach']['call']} | {ci_txt(x['found']['rank1']['diff'])} {x['found']['rank1']['call']} |")
            L.append(f"| {ds} | K pairs at the anchors | {r['K_pairs_at_anchors']} | | | |")
    for arm, byds in rec["arms"].items():
        for ds, r in byds.items():
            L += ["", f"## {arm} {ds} ({r['with_gold']} questions with a gold)", "",
                  "| unit | kind | pairs per anchor | precision | lift | call | vs any | reach | rank1 | top2 | rank1 by cos |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
            for un, byk in r["units"].items():
                for kn in KINDS:
                    x = byk[kn]
                    va = x.get("vs_any")
                    L.append(f"| {un} | {kn} | {x['price']} | {ci_txt(x['precision'])} | {ci_txt(x['lift'], False)} | "
                             f"{x['lift_call']} | {'-' if va is None else (ci_txt(va['diff']) + ' ' + str(va['call']))} | "
                             f"{ci_txt(x.get('reach'))} | {ci_txt(x['rank1'])} | {x['top2']} | {x['rank1_cos']} |")
            for kn in KINDS:
                mu = r["multi"][kn]
                L.append(f"\nMulti-label minus single-label precision ({kn}): {ci_txt(mu['diff'])} {mu['call']} "
                         f"({mu['multi_pairs']} / {mu['single_pairs']} pairs).")
            L += ["", "| kind | labels | share of pairs | precision | lift |", "|---|---|---|---|---|"]
            for kn in KINDS:
                for c in r["combos"][kn]:
                    L.append(f"| {kn} | {' + '.join(c['labels'])} | {100 * c['share']:.1f} | {ci_txt(c['precision'])} | "
                             f"{ci_txt(c['lift'], False)} |")
            if "hops" in r:
                L += ["", "| hops | kind | unit | precision | lift | reach | rank1 |", "|---|---|---|---|---|---|---|"]
                for h, byk in r["hops"].items():
                    for kn, byu in byk.items():
                        for un in ("any", "group:R", "group:K"):
                            if un in byu:
                                x = byu[un]
                                L.append(f"| {h} | {kn} | {un} | {ci_txt(x['precision'])} | {ci_txt(x['lift'], False)} | "
                                         f"{ci_txt(x.get('reach'))} | {ci_txt(x['rank1'])} |")
    L += ["", "## Label weights named for a later refit (ln of top1 lift; not adopted)", "",
          "```", json.dumps(rec["weights_named_not_adopted"], indent=1), "```", ""]
    Path(path).write_text("\n".join(L), encoding="utf-8")


def report():
    rec = build()
    REP.write_json(OUT / "report.json", rec)
    write_md(rec, OUT / "report.md")
    log(f"report: {len(rec['ledgers'])} ledgers; expectations " + ", ".join(f"{k} {e['call']}" for k, e in rec["expectations"].items()))
    return 0


# ── self-test ────────────────────────────────────────────────────────────────


def brute_bits(n, a, u, v, f):
    b = [0] * n
    for x, y, l in zip(u.tolist(), v.tolist(), f.tolist()):
        if x == a and y != a:
            b[y] |= 1 << l
        if y == a and x != a:
            b[x] |= 1 << l
    return b


def toy(rng, nq, L, both_dirs=True):
    """Random questions in the ledger's shapes: pools of 1-40 rows, golds, rrf with ties, dense_cos, labelled links (duplicates,
    both directions and self-loops-free), as the cache and links parts give them."""
    Qs = []
    for i in range(nq):
        n = int(rng.integers(1, 41))
        g = rng.random(n) < 0.15
        gt = int(g.sum() + rng.integers(0, 2)) if i % 7 else 0
        rrf = np.round(rng.random(n), 1)
        cos = rng.random(n)
        m = int(rng.integers(0, 4 * n))
        u = rng.integers(0, n, m)
        v = rng.integers(0, n, m)
        keep = u != v
        u, v = u[keep], v[keep]
        f = rng.integers(0, L, u.size)
        if both_dirs:
            u, v, f = np.r_[u, v], np.r_[v, u], np.r_[f, f]
        Qs.append((n, g, gt, rrf, cos, u, v, f))
    return Qs


def selftest():
    import chndiag as CH
    rng = np.random.default_rng(7)
    # 1. question_pairs against a brute force
    for L in (5, 3):
        for n, g, gt, rrf, cos, u, v, f in toy(rng, 60, L):
            q, rows, top = question_pairs(n, g, gt, rrf, cos, u, v, f)
            o = CH.order_of(rrf)
            assert list(top) == list(o[:5])
            assert q["found"] == sum(bool(g[r]) for r in o[:5]) and q["out"] == n - min(n, 5)
            assert q["missed"] == sum(bool(g[r]) for r in range(n) if r not in set(o[:5].tolist()))
            if gt <= 0:
                assert rows == []
                continue
            want = [(0, int(a)) for a in o[:5] if g[a]] + [(1, int(o[0]))]
            assert [(k, a) for k, a, *_ in rows] == want
            pos = {int(r): i + 1 for i, r in enumerate(o)}
            posc = {int(r): i + 1 for i, r in enumerate(CH.order_of(cos))}
            for kind, a, ar, c, b, t, rk, rkc in rows:
                bb = brute_bits(n, a, u, v, f)
                exp = [r for r in range(n) if bb[r] and r not in set(o[:5].tolist())]
                assert c.tolist() == exp and b.tolist() == [bb[r] for r in exp] and ar == pos[a]
                assert t.tolist() == [bool(g[r]) for r in exp] and rk.tolist() == [pos[r] for r in exp]
                assert rkc.tolist() == [posc[r] for r in exp]
    # 2. the measures against a brute force on a toy ledger
    Qs = toy(rng, 300, 5)
    acc = Acc()
    for i, (n, g, gt, rrf, cos, u, v, f) in enumerate(Qs):
        q, rows, _t = question_pairs(n, g, gt, rrf, cos, u, v, f)
        acc.add(n, gt, (2, 3, 4)[i % 3], q, rows)
    z = acc.arrays()
    for kind in (0, 1):
        qs = qsel_of(z, kind)
        for m in (1, 2 | 4 | 8, 16, 31):
            r = unit_block(z, m, kind, qs)
            num = den = r1 = wt = reached = mis = 0
            for i, (n, g, gt, rrf, cos, u, v, f) in enumerate(Qs):
                if not qs[i]:
                    continue
                q, rows, _t = question_pairs(n, g, gt, rrf, cos, u, v, f)
                mis += q["missed"]
                hit = set()
                for k_, a, ar, c, b, t, rk, rkc in rows:
                    if k_ != kind:
                        continue
                    s = [(int(rk_), bool(tt), int(cc)) for bb, tt, rk_, cc in zip(b, t, rk, c) if bb & m]
                    den += len(s)
                    num += sum(tt for _r, tt, _c in s)
                    hit |= {cc for _r, tt, cc in s if tt}
                    s.sort()
                    if any(tt for _r, tt, _c in s):
                        wt += 1
                        r1 += s[0][1]
                reached += len(hit)
            assert r["pairs"] == den and r["targets"] == num, (kind, m)
            if den:
                assert abs(r["precision"]["pt"] - round(num / den, 5)) < 1e-9
            if wt:
                assert abs(r["rank1"]["pt"] - round(r1 / wt, 5)) < 1e-9
            if kind == 0 and mis:
                assert abs(r["reach"]["pt"] - round(reached / mis, 5)) < 1e-9
            if r["precision"]:
                lo, hi = r["precision"]["ci"]
                assert lo <= r["precision"]["pt"] + 1e-9 and r["precision"]["pt"] <= hi + 1e-9
    # 3. the exact-set and multi/single splits partition the pairs
    for kind in (0, 1):
        qs = qsel_of(z, kind)
        tot = unit_block(z, 31, kind, qs)["pairs"]
        sets = sum(c["pairs"] for c in combos(z, REP.LABELS, kind, qs, keep=1.01, most=64))
        pm, _ = per_question(z, 0, kind, "multi")
        ps, _ = per_question(z, 0, kind, "single")
        assert sets == tot == int(pm["den"][qs].sum() + ps["den"][qs].sum()), (sets, tot)
    # 4. the bootstrap is seeded; the calls
    b1 = boot(lambda s: ratio(s["a"], s["b"]), {"a": rng.random(50), "b": np.ones(50)})
    b2 = boot(lambda s: ratio(s["a"], s["b"]), {"a": np.asarray([0.0] * 50), "b": np.zeros(50)})
    assert b2 is None and b1["ci"][0] <= b1["pt"] <= b1["ci"][1]
    assert lift_call({"pt": 2, "ci": [1.1, 3]}) == "POINTS" and lift_call({"pt": 1, "ci": [0.9, 1.2]}) == "NO_LIFT"
    assert lift_call({"pt": 0.5, "ci": [0.4, 0.8]}) == "BELOW_BASE" and call({"pt": 0, "ci": [-1, -0.1]}) == "BELOW"
    # 5. units: the groups cover every bit
    for arm in ARMS:
        for ds in ARMS[arm][3]:
            U = units(arm, ds)
            gm = 0
            for k, mm in U.items():
                if k.startswith("group:"):
                    assert not gm & mm, (arm, ds, k)
                    gm |= mm
            want = U["any"] if ds in KB or arm in ("REC", "B1W") else U["any"] & ~1
            assert gm == want, (arm, ds, gm, want)
    # 6. the pair block: the same ledger against itself is LEVEL everywhere with equal K pairs; G6 stops a mismatch
    meta = {"ids_sha256": "x", "pool_sha256": "p", "top5_sha256": "t"}
    pb = pair_block(z, z, {**meta, "arm": "D83E"}, {**meta, "arm": "D83E"}, "musique")
    for gname, rec in pb.items():
        if gname == "K_pairs_at_anchors":
            assert rec["same"]
            continue
        for kn in KINDS:
            for k, x in rec[kn].items():
                assert x["call"] in ("LEVEL", None) and (x["diff"] is None or abs(x["diff"]["pt"]) < 1e-12), (gname, kn, k)
    try:
        pair_block(z, z, {**meta, "arm": "D83E"}, {**meta, "arm": "REC", "top5_sha256": "u"}, "musique")
        raise AssertionError("G6 did not stop")
    except SystemExit as e:
        assert "G6" in str(e)
    # 7. the arm block and the report's text run on the toy ledger
    blk = arm_block(z, "D83E", "musique")
    rec = {"utc": "t", "expectations": expectations({"D83E": {"musique": blk}}), "pairs": {"D83E-REC": {"musique": pb}},
           "arms": {"D83E": {"musique": blk}}, "weights_named_not_adopted": {}}
    tmp = OUT / "selftest.md"
    OUT.mkdir(parents=True, exist_ok=True)
    write_md(rec, tmp)
    assert tmp.read_text(encoding="utf-8").startswith("# AP4")
    tmp.unlink()
    log("selftest: question_pairs = brute force (5 and 3 labels); precision, reach, rank1 = brute force; exact sets and "
        "multi/single partition the pairs; bootstrap and calls; units; pair block LEVEL on itself, G6 stops; report text: PASS")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("ledger")
    s.add_argument("--arm", required=True)
    s.add_argument("--dataset", required=True)
    s.add_argument("--limit", type=int, default=None)
    s.add_argument("--out", default=None)
    sub.add_parser("report")
    sub.add_parser("selftest")
    a = ap.parse_args(argv)
    if a.cmd == "ledger":
        if a.limit and not a.out:
            raise SystemExit("AP4: a limited dry run writes only to --out, never to the ledger folder")
        ledger(a.arm, a.dataset, a.limit, a.out)
        return 0
    return report() if a.cmd == "report" else selftest()


if __name__ == "__main__":
    sys.exit(main())
