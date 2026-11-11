"""AP3 (docs/AP3_STRUCTURAL_SLOT_TITLE_FORMS.md): rare base-title links in the structural slot of the rule-built carve.

Part A (the laptop, graph level): U2b's own link rule on each B1 setting's passages gives, besides the exact-title links that R (b1w's structural family) holds, the
links of a title's BASE form (parenthetical or comma tail removed). A ladder of caps on the surface's document frequency (df <= c, c = 1, 2, 3, 4, 6, 8, 12, 16, 25,
no cap) adds those pairs to R; S1 (declared in the doc) carries the deepest rung whose added pairs are priced at most R's own pairs per gold slot linked and that adds
at least FLOOR points of the setting's slots. Part B (the host): the carve b1y = R plus the carried rung's pairs in the structural slot, b1n's other files (F0T as the
NER family, B1a's kNN family) byte for byte, read by the models of record, no new fit.

    python outputs/mp_unified/zap3.py selftest                  # host-safe: no corpus, no zu2b
    python outputs/mp_unified/zap3.py selftest-ladder           # the laptop: the link rule on a toy corpus
    python outputs/mp_unified/zap3.py ladder                    # Part A, the laptop: outputs/ap3/ladder.json, ladder.md and <setting>/k1.npz
    python outputs/mp_unified/zap3.py bench --dataset D         # the laptop: the setting's folder (refuses a setting with no carried rung)
    python outputs/mp_unified/zap3.py settings [--dataset D] [--host]
    python outputs/mp_unified/zap3.py chain --dataset D --host  # PKO's chain on the carve
    python outputs/mp_unified/zap3.py gate --tag D --host
    python outputs/mp_unified/zap3.py fork | lists | cols | read-gnn | read-mlp ...      # zb1h's, on AP3's roots
    python outputs/mp_unified/zap3.py report                    # the laptop: b1y against b1n, b1w and b1t, paired by question

The host steps import nothing of the link rule (zap1, zu2b): those imports are inside `ladder` and `selftest-ladder`.
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import zb1l as L  # noqa: E402  (imports zb1h, whose header sets BLAS from sys.argv[1:2] before numpy loads: the command goes first)

import argparse  # noqa: E402
import contextlib  # noqa: E402
import copy  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import numpy as np  # noqa: E402

H = L.H
ROOT = H.ROOT
DECLARED = "docs/AP3_STRUCTURAL_SLOT_TITLE_FORMS.md"
STAGE = "AP3"
VARIANT = "y"
CARVE = "b1y"
FORM = "R plus the rare base-title pairs"
OUT = ROOT / "outputs" / CARVE
LTM_OUT = OUT / "ltm"
B1W = ROOT / "outputs" / "bench" / "hipporag2_w"               # b1w's folder: the base of the structural family
B1N = ROOT / "outputs" / "b1n" / "bench" / "hipporag2_n"       # b1n's folder: the source of five files
B1A_REAL = ROOT / "outputs" / "bench" / "hipporag2"            # B1a's own folder: its lists and the NER file they name
BENCH_G = OUT / "bench" / "hipporag2_y"
AP1B = ROOT / "outputs" / "ap1b"
AP3_OUT = ROOT / "outputs" / "ap3"
LADDER = AP3_OUT / "ladder.json"
STRUCT = "graph_structural.npz"
NER = "graph_ner.npz"
SETTINGS = ("2wiki", "hotpotqa", "musique")                    # the ladder runs on all three; a carve exists only where S1 carried a rung
TAGS = {ds: (ds,) for ds in SETTINGS}                          # one gate and one set of reads per setting, as B1L
CARVES = ("b1t", "b1w", CARVE)
GRAPH_TEXT = ("AP3 b1y: b1n's NER family (F0T) and B1a's kNN family, and a structural family holding b1w's R plus the pairs of AP3's carried rung "
              "(U2b's base-title links at the surface df cap S1 carried; outputs/ap3/ladder.json)")
NO_STEP1 = {"state": f"not applicable: B1's settings have no step-1 cache (AP3, {DECLARED})"}
GNN, MLP = L.GNN, L.MLP
MODELS = (GNN, MLP)
STEP = ("UP", "DOWN", "FLAT")                                   # a pair against b1t by its interval: above 0, below 0, neither
VS_W = ("GREATER", "LESS", "SAME")                              # b1y - b1n, b1y - b1w by their interval
HOPS = ("2hop", "3hop", "4hop")
REPORT_OUT = AP3_OUT
_ON_B1H = L._ON_B1H

# the ladder and S1
INF = 10 ** 9
CAPS = (1, 2, 3, 4, 6, 8, 12, 16, 25, INF)
FLOOR = 2.0                                                     # points of the setting's slots a carried rung must add
BASIS = {"2wiki": "GP", "hotpotqa": "GP", "musique": "CH"}      # the slots S1 reads: gold pairs, MuSiQue's chain slots
JACCARD_MIN = 0.95                                              # G1: the setting-level rebuild of R's rule against the filed R
BOOT, SEED = 2000, 0                                            # zap1's, b1b's
VARS2 = {"K0": ((0,), False), "K1": ((0, 1), True)}             # U2b's W2 (title links) and W5e (title and base-title links, exact first)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] zap3: {msg}", flush=True)


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read_roots():
    d = f"outputs/{CARVE}"
    return {"pools": f"{d}/pools", "plook": f"{d}/look", "cache": f"{d}/cache", "rmatch": f"{d}/chains/rmatch", "zrc": f"{d}/chains/zrc",
            "links": f"{d}/links", "kt": f"{d}/kt"}


# -- Part A: the ladder (graph level) ----------------------------------------------------------------------------------------------

def pair(a, b):
    return (a, b) if a < b else (b, a)


def und_pairs(path):
    """The undirected pairs (a < b) of a graph file's directed edges."""
    with np.load(path) as z:
        s, t = z["src"].astype(np.int64), z["dst"].astype(np.int64)
    return {pair(int(a), int(b)) for a, b in zip(s.tolist(), t.tolist()) if a != b}


def boot_ratio(cx, cr, qidx, nq, boot=BOOT, seed=SEED):
    """zap1.boot_diff's body (the ratio estimator over questions): the share of slots linked by x minus by r, and its 95% interval over resamples of the
    questions. The ladder run checks this copy against zap1.boot_diff."""
    qidx = np.asarray(qidx, np.int64)
    sx = np.bincount(qidx, weights=np.asarray(cx, float), minlength=nq)
    sr = np.bincount(qidx, weights=np.asarray(cr, float), minlength=nq)
    nn = np.bincount(qidx, minlength=nq).astype(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, nq, size=(boot, nq))
    num = sx[idx].sum(1) - sr[idx].sum(1)
    den = nn[idx].sum(1)
    d = num / den
    return float((sx.sum() - sr.sum()) / nn.sum()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def call3(lo, hi):
    return "ABOVE" if lo > 0 else ("BELOW" if hi < 0 else "LEVEL")


def reach_profile(base, k1, slots, caps=CAPS):
    """Over a base pair set, per cap c: the pairs of k1 ({pair: df}) with df <= c that the base lacks, and the slots they newly link.
    slots: [(question, a, b, ...)] with a < b. Returns the base's linked flags, each slot's smallest cap that links it (0: the base does; INF + 1: none) and the rows."""
    add_df = np.asarray([d for p, d in k1.items() if p not in base], np.int64)
    linked = np.zeros(len(slots), bool)
    mincap = np.empty(len(slots), np.int64)
    for j, s in enumerate(slots):
        p = pair(int(s[1]), int(s[2]))
        if p in base:
            linked[j], mincap[j] = True, 0
        else:
            mincap[j] = k1.get(p, INF + 1)
    rows = [{"cap": c, "pairs_added": int((add_df <= c).sum()), "slots_added": int(((mincap > 0) & (mincap <= c)).sum())} for c in caps]
    return {"linked": linked, "mincap": mincap, "base_linked": int(linked.sum()), "rows": rows}


def rung_rows(R, k1, slots, nq, S=None, caps=CAPS, ci=boot_ratio):
    """The ladder over the structural family R on `slots`: per cap, the pairs added over R, the slots linked and added, the gain in points of the setting's
    slots with its 95% interval over questions, the price (pairs added per slot added, None when pairs were added and no slot) and, where S is given, the
    share of the added pairs that are S's. The decisions read the counts, never the rounded fields."""
    prof = reach_profile(R, k1, slots, caps)
    n = len(slots)
    qidx = [s[0] for s in slots]
    rows = []
    for r in prof["rows"]:
        c, g, pa = r["cap"], r["slots_added"], r["pairs_added"]
        cx = prof["linked"] | ((prof["mincap"] > 0) & (prof["mincap"] <= c))
        pt, lo, hi = ci(cx, prof["linked"], qidx, nq)
        row = {"cap": c if c < INF else "inf", "pairs_added": pa, "pairs_total": len(R) + pa, "slots_added": g,
               "slots_linked": prof["base_linked"] + g, "share": round(100.0 * (prof["base_linked"] + g) / n, 2),
               "gain_points": round(100.0 * g / n, 3), "gain_ci": [round(100 * lo, 2), round(100 * hi, 2)], "gain_call": call3(lo, hi),
               "price": round(pa / g, 2) if g else (0.0 if pa == 0 else None)}
        if S is not None:
            row["s_added"] = sum(1 for p, d in k1.items() if d <= c and p not in R and p in S)
        rows.append(row)
    return {"base_linked": prof["base_linked"], "slots": n, "rows": rows}


def carry(rows, base_pairs, base_linked, n_slots, floor=FLOOR):
    """S1. p0 = R's own pairs per slot it links. A rung passes the price when its pairs added over R (all rungs up to it together) per slot added are at most p0
    (pairs added with no slot added never pass; nothing added passes). The carried rung is the deepest one up to which every rung passes, and only if it
    adds at least `floor` points of the setting's slots; otherwise none. Returns (cap or None, why, p0). Reads the counts, not the rounded fields."""
    p0 = base_pairs / base_linked if base_linked else float("inf")
    last = None
    for r in rows:
        pa, g = r["pairs_added"], r["slots_added"]
        ok = (pa == 0) if g == 0 else (pa / g <= p0)
        if not ok:
            break
        last = r
    if last is None:
        return None, "the first rung already costs more than R's own price per slot", p0
    if 100.0 * last["slots_added"] / n_slots < floor:
        return None, f"the deepest rung that passes the price adds {last['slots_added']} slots, below {floor} points of the setting's {n_slots}", p0
    return (INF if last["cap"] == "inf" else last["cap"]), "carried", p0


def form_pairs(z, n, tables):
    """zap2.form_pairs: (a, b, w) of the union of the level tables of z (AP1b's levels.npz), shallow to deep, a < b; where several hold a pair the last one's weight."""
    ks, ws = [], []
    for t in tables:
        a, b = z[f"{t}_src"].astype(np.int64), z[f"{t}_dst"].astype(np.int64)
        if a.size and (not (a < b).all() or a.min() < 0 or b.max() >= n):
            raise SystemExit(f"AP3: table {t} holds a pair that is not a < b inside the {n} passages")
        ks.append(a * n + b)
        ws.append(z[f"{t}_w"].astype(np.float32))
    k, w = np.concatenate(ks), np.concatenate(ws)
    o = np.argsort(k, kind="stable")
    k, w = k[o], w[o]
    last = np.ones(k.size, bool)
    last[:-1] = k[1:] != k[:-1]
    k, w = k[last], w[last]
    return (k // n).astype(np.int32), (k % n).astype(np.int32), w


def k1_links(corpus):
    """U2b's rule (NEW_RULE: look boundary, underscores as spaces, prefixes, descriptive sources) on the setting's own passages, through zu2b.choose_targets one
    surface hit at a time with the two variants K0 (title links, W2) and K1 (title and base-title links, exact first, W5e): an ambiguous surface (several title
    groups) links nothing, as no vectors are read. Returns the K0 pairs with the sibling pairs (the setting-level rebuild of R's rule), the K1-only pairs
    {(a, b): the smallest df of the surfaces that give them}, and counts. A passage's single-hit choices must equal zu2b.choose_targets on all its hits."""
    import zu2b
    rule = zu2b.NEW_RULE
    n = len(corpus)
    titles = [c["title"] for c in corpus]
    texts = [c["text"] for c in corpus]
    S = zu2b.Surfaces(titles, rule)
    tl = [zu2b.norm_title(t, rule) for t in titles]
    cap, low = zu2b.stat_pass(S, texts, tl, rule)
    common = low > cap
    desc = np.asarray([zu2b.is_descriptive(t, x) for t, x in zip(tl, texts)])
    hits = [None] * n
    df = np.zeros(len(S.surf), np.int64)                       # passages whose text holds the surface (case-checked), the owner's own among them
    for i in range(n):
        if desc[i]:
            h, _st, cm = zu2b.scan(S, texts[i], rule)
            hits[i] = (h, cm)
            for k in {k for k, ok in h if ok}:
                df[k] += 1
    k0, k1 = set(), {}
    amb = {"K0": 0, "K1_only": 0}
    checked = 0
    for i in range(n):
        if hits[i] is None:
            continue
        h, cm = hits[i]
        got = {"K0": set(), "K1": set()}
        for k, ok in h:
            if not ok:
                continue
            need = []
            r = zu2b.choose_targets(S, i, [(k, ok)], common, need, VARS2, cm)
            a0 = sum(1 for x in need if x[1] == "K0")
            amb["K0"] += a0
            amb["K1_only"] += sum(1 for x in need if x[1] == "K1") - a0
            for v in ("K0", "K1"):
                got[v].update(r[v])
            if r["K0"]:
                for g in r["K0"]:
                    for j in S.group_nodes(g).tolist():
                        if j != i:
                            k0.add(pair(i, j))
            elif r["K1"]:
                for g in r["K1"]:
                    for j in S.group_nodes(g).tolist():
                        if j != i:
                            p = pair(i, j)
                            k1[p] = min(k1.get(p, INF), int(df[k]))
        whole = zu2b.choose_targets(S, i, h, common, [], VARS2, cm)
        if set(whole["K0"]) != got["K0"] or set(whole["K1"]) != got["K1"]:
            raise SystemExit(f"AP3: passage {i}: the single-hit choices differ from zu2b.choose_targets on the passage's hits (G3)")
        checked += 1
    ss, sd = zu2b.sibling_pairs(S.group, desc)
    k0 |= {pair(int(a), int(b)) for a, b in zip(ss.tolist(), sd.tolist())}
    return {"k0": k0, "k1": k1, "ambiguous_dropped": amb, "surfaces": int(len(S.surf)), "passages_checked": checked, "descriptive": int(desc.sum())}


def ladder_setting(d, Z):
    """One setting's block of the ladder record, and its K1 arrays (a, b, df)."""
    corpus, queries, fam = Z.load(d)
    n, nq = len(corpus), len(queries)
    R, S, K = fam["R"], fam["S"], fam["K"]
    t0 = time.time()
    links = k1_links(corpus)
    k0, k1 = links["k0"], links["k1"]
    log(f"{d}: {n} passages, K0 + siblings {len(k0)} pairs (R {len(R)}), K1-only {len(k1)} pairs, ambiguous dropped {links['ambiguous_dropped']} ({time.time() - t0:.0f} s)")
    jac = len(k0 & R) / len(k0 | R)
    if jac < JACCARD_MIN:
        raise SystemExit(f"AP3 {d}: the setting-level rebuild of R's rule and the filed R have Jaccard {jac:.4f} < {JACCARD_MIN} (G1)")
    if any(not (0 <= a < b < n) for a, b in k1):
        raise SystemExit(f"AP3 {d}: a K1 pair that is not a < b inside the {n} passages (G2)")
    build_w = H.read_json(B1W / d / "build.json")
    if int(build_w["passages"]) != n:
        raise SystemExit(f"AP3 {d}: b1w's build.json holds {build_w['passages']} passages, the corpus {n} (G4)")
    if und_pairs(B1W / d / STRUCT) != R or sha_file(B1W / d / STRUCT) != build_w["files"][STRUCT]:
        raise SystemExit(f"AP3 {d}: R is not b1w's structural file as its build.json records it (G4)")
    gp = Z.gold_slots(queries)
    ch = Z.chain_slots(d, corpus, queries)[0] if d == "musique" else None
    rep = H.read_json(AP1B / "report.json")["settings"][d]["info"]["slots"]
    if rep.get("GP") != len(gp) or (ch is not None and rep.get("CH") != len(ch)):
        raise SystemExit(f"AP3 {d}: the slot counts ({len(gp)} GP{'' if ch is None else f', {len(ch)} CH'}) are not AP1b's {rep} (G4)")
    slots = ch if BASIS[d] == "CH" else gp
    main = rung_rows(R, k1, slots, nq, S)
    z = Z.boot_diff(np.ones(len(slots), bool), np.zeros(len(slots), bool), [s[0] for s in slots], nq)          # copy vs zap1 on a fixed pair of vectors
    if z != boot_ratio(np.ones(len(slots), bool), np.zeros(len(slots), bool), [s[0] for s in slots], nq):
        raise SystemExit("AP3: boot_ratio is not zap1.boot_diff (G6)")
    out = {"passages": n, "questions": nq, "basis": BASIS[d], "slots": len(slots),
           "r": {"pairs": len(R), "slots_linked": main["base_linked"], "share": round(100.0 * main["base_linked"] / len(slots), 2),
                 "s_precision": round(len(R & S) / len(R), 4), "s_recall": round(len(R & S) / len(S), 4)},
           "a0_vs_r": {"a0_pairs": len(k0), "jaccard": round(jac, 4), "intersection": len(k0 & R)},
           "k1": {"pairs": len(k1), "pairs_not_in_r": sum(1 for p in k1 if p not in R), "ambiguous_dropped": links["ambiguous_dropped"],
                  "surfaces": links["surfaces"], "descriptive_passages": links["descriptive"], "passages_checked_against_zu2b": links["passages_checked"]},
           "ladder": main["rows"]}
    if ch is not None:
        out["ladder_gp"] = rung_rows(R, k1, gp, nq, S)["rows"]
    s1_cap, why, p0 = carry(main["rows"], len(R), main["base_linked"], len(slots))
    out["s1"] = {"carried_cap": None if s1_cap is None else ("inf" if s1_cap == INF else s1_cap), "why": why, "p0_pairs_per_slot": round(p0, 3), "floor_points": FLOOR}
    # the union view (reported, not deciding): b1n's union R + F0T + K against the same ladder
    lv = AP1B / d / "levels.npz"
    ap1b = H.read_json(AP1B / "report.json")["settings"][d]
    if sha_file(lv) != ap1b["levels_npz_sha256"]:
        raise SystemExit(f"AP3 {d}: levels.npz is not the one AP1b's report names (G5)")
    with np.load(lv) as zz:
        fa, fb, _w = form_pairs(zz, n, ("L0", "L0T"))
    if int(fa.size) != int(ap1b["forms"]["F0T"]["form_pairs"]):
        raise SystemExit(f"AP3 {d}: {fa.size} F0T pairs, AP1b's report {ap1b['forms']['F0T']['form_pairs']} (G5)")
    U0 = set(R) | set(K) | {(int(a), int(b)) for a, b in zip(fa.tolist(), fb.tolist())}
    uni = rung_rows(U0, k1, slots, nq, None)
    out["union"] = {"pairs": len(U0), "slots_linked": uni["base_linked"], "share": round(100.0 * uni["base_linked"] / len(slots), 2),
                    "rows": [{"cap": r["cap"], "pairs_added": r["pairs_added"], "slots_added": r["slots_added"], "gain_points": r["gain_points"]} for r in uni["rows"]]}
    arr = sorted(k1.items())
    return out, {"a": np.asarray([p[0] for p, _d in arr], np.int32), "b": np.asarray([p[1] for p, _d in arr], np.int32),
                 "df": np.asarray([x for _p, x in arr], np.int32)}


def ladder():
    import u1a
    import zap1 as Z
    import zu2b
    rec = {"declared_in": DECLARED, "stage": STAGE, "part": "A", "caps": [c if c < INF else "inf" for c in CAPS], "floor_points": FLOOR, "basis": BASIS,
           "boot": {"resamples": BOOT, "seed": SEED}, "jaccard_min": JACCARD_MIN, "settings": {}, "carried": {}}
    for d in SETTINGS:
        blk, arrs = ladder_setting(d, Z)
        (AP3_OUT / d).mkdir(parents=True, exist_ok=True)
        p = AP3_OUT / d / "k1.npz"
        np.savez(p, **arrs)
        blk["k1"]["npz_sha256"] = sha_file(p)
        rec["settings"][d] = blk
        rec["carried"][d] = blk["s1"]["carried_cap"]
        log(f"{d}: S1 {blk['s1']}")
    rec["zap1_sha256"] = u1a.sha_src(Z.__file__)
    rec["zu2b_sha256"] = u1a.sha_src(zu2b.__file__)
    rec["script_sha256"] = u1a.sha_src(__file__)
    rec["ap1b_report_sha256"] = sha_file(AP1B / "report.json")
    rec["utc"] = u1a.utc()
    u1a.write_json(LADDER, rec)
    write_ladder_md(rec)
    log(f"ladder: {LADDER}; carried {rec['carried']}")
    return 0


def write_ladder_md(rec, out=None):
    out = Path(out) if out is not None else AP3_OUT
    T = [f"# AP3 Part A: the ladder (declared in {DECLARED})", "",
         "Graph level, no model. R = b1w's structural family (the filed W2_S pairs). A rung adds to R the base-title pairs (U2b's rule, ambiguous surfaces dropped) whose "
         "surface occurs in at most `cap` passages of the setting. Gain = slots newly linked by the structural family, in points of the setting's slots (GP: gold pairs; "
         "MuSiQue: chain slots); price = pairs added per slot added. S1 carries the deepest rung up to which every rung's price is at most R's own pairs per slot linked "
         f"(p0) and whose gain is at least {rec['floor_points']} points.", ""]
    for d, s in rec["settings"].items():
        T += [f"## {d} ({s['basis']}, {s['slots']} slots, {s['passages']} passages)", "",
              f"R: {s['r']['pairs']} pairs link {s['r']['slots_linked']} slots ({s['r']['share']}%); p0 = {s['s1']['p0_pairs_per_slot']} pairs per slot; "
              f"S precision of R {s['r']['s_precision']}, S recall {s['r']['s_recall']}. **S1: {s['s1']['carried_cap']}** ({s['s1']['why']}).", "",
              "| cap | pairs added | slots added | gain (points) [95%] | share linked | price | S share of added | union: pairs added | union: slots added |",
              "|---|---:|---:|---|---:|---:|---:|---:|---:|"]
        for r, u in zip(s["ladder"], s["union"]["rows"]):
            sh = f"{r['s_added']}/{r['pairs_added']}" if "s_added" in r else "--"
            T.append(f"| {r['cap']} | {r['pairs_added']} | {r['slots_added']} | {r['gain_points']:+.2f} [{r['gain_ci'][0]:+.2f}, {r['gain_ci'][1]:+.2f}] {r['gain_call']} "
                     f"| {r['share']} | {r['price'] if r['price'] is not None else 'inf'} | {sh} | {u['pairs_added']} | {u['slots_added']} |")
        T.append("")
    T += ["Carried: " + ", ".join(f"{d} {c}" for d, c in rec["carried"].items()) + "."]
    out.mkdir(parents=True, exist_ok=True)
    (out / "ladder.md").write_text("\n".join(T) + "\n", encoding="utf-8")


# -- the lists check: B1a's lists name B1a's files; this carve replaced two of them (structural, NER) ---------------------------------

def check_lists(named, shas, build):
    """B1a's lists record (setting_files_sha256 `named`) against this carve's file shas: every file but the structural and NER ones equal, the structural one the
    sha this variant replaced (u1a.b1a_structural_sha256), the NER one the sha b1w's file has (ap2.b1a_ner_sha256)."""
    swapped = {STRUCT: build["u1a"]["b1a_structural_sha256"], NER: build["ap2"]["b1a_ner_sha256"]}
    if {f: s for f, s in named.items() if f not in swapped} != {f: s for f, s in shas.items() if f not in swapped}:
        raise SystemExit("AP3: B1a's lists were built on other setting files than this carve's")
    for f, s in swapped.items():
        if named.get(f) != s:
            raise SystemExit(f"AP3: B1a's lists name another {f} than the one this carve replaced")
    return named


def lists_of(M, st):
    """look_b1u.load_lists with the NER file checked as the structural one is: B1a's lists, read from B1a's folder."""
    rec = json.loads((M.LBU.B1A / st.name / "b1_lists.json").read_text(encoding="utf-8"))
    named = check_lists(rec["setting_files_sha256"], st.shas, st.build)
    here = M.LB.BENCH
    M.LB.BENCH = M.LBU.B1A
    try:
        return M.LBU._ORIG_LISTS(SimpleNamespace(**{**vars(st), "shas": named}))
    finally:
        M.LB.BENCH = here


@contextlib.contextmanager
def on_b1y(M, name, stats, look=False):
    """zb1h.on_b1h with this carve's lists: the folder holds no lists of its own (B1a's, checked by check_lists)."""
    P_ = H.zb().Patches()
    try:
        P_.set(M.LB, load_lists=lambda st: lists_of(M, st))
        with _ON_B1H(M, name, stats, look) as B:
            yield B
    finally:
        P_.restore()


def bind():
    """zb1h's module globals set to AP3's (its functions look them up at call time); safe to call again."""
    for k, v in (("DECLARED", DECLARED), ("STAGE", STAGE), ("OUT", OUT), ("LTM_OUT", LTM_OUT), ("B1A", BENCH_G), ("CARVE", CARVE),
                 ("CARVES", CARVES), ("SETTINGS", SETTINGS), ("TAGS", TAGS), ("GRAPH_TEXT", GRAPH_TEXT), ("NO_STEP1", NO_STEP1),
                 ("read_roots", read_roots), ("on_b1h", on_b1y)):
        setattr(H, k, v)


# -- the setting's folder (bench) ----------------------------------------------------------------------------------------------------

def variant_build(build, struct_sha, links, script_sha, utc):
    """b1n's build.json for the carve: the structural file's sha replaced, the u1a block relabelled (its structural sha is still B1a's), the b1f block noted, an ap3 block."""
    b = json.loads(json.dumps(build))
    b["files"][STRUCT] = struct_sha
    b["u1a"] = {**build["u1a"], "declared_in": DECLARED, "variant": VARIANT, "carve": CARVE, "script_sha256": script_sha, "utc": utc,
                "structural": f"U2b's W2_S (b1w's R) plus the base-title pairs of AP3's rung at df cap {links['cap']} (each pair once each way)",
                "edges": int(links["directed_edges_file"])}
    b["b1f"] = {**build.get("b1f", {}), "note": "b1w's structural file (R), the base of this carve's structural file (ap3.structural_sha256 is this carve's)"}
    b["ap3"] = links
    return b


def bench(name):
    import u1a
    t0 = time.time()
    lad = H.read_json(LADDER)
    cap = lad["carried"].get(name)
    if cap is None:
        raise SystemExit(f"AP3 {name}: S1 carried no rung for this setting (outputs/ap3/ladder.json); no carve")
    cap_i = INF if cap == "inf" else int(cap)
    st = lad["settings"][name]
    src_dir, out = B1N / name, BENCH_G / name
    build = H.read_json(src_dir / "build.json")
    bad = [f for f in u1a.FILES if u1a.sha_file(src_dir / f) != build["files"].get(f)]
    if bad:
        raise SystemExit(f"AP3 {name}: {bad} differ from b1n's build.json")
    w = H.read_json(B1W / name / "build.json")
    if build["files"][STRUCT] != w["files"][STRUCT]:
        raise SystemExit(f"AP3 {name}: b1n's structural file is not b1w's (G2)")
    n = len(H.read_json(src_dir / "nodes.json"))
    if n != int(build["passages"]) or n != int(st["passages"]):
        raise SystemExit(f"AP3 {name}: b1n's folder holds {n} passages, build.json {build['passages']}, the ladder {st['passages']} (G4)")
    kp = AP3_OUT / name / "k1.npz"
    if sha_file(kp) != st["k1"]["npz_sha256"]:
        raise SystemExit(f"AP3 {name}: k1.npz is not the one the ladder record names (G1)")
    R = und_pairs(src_dir / STRUCT)
    if len(R) != int(st["r"]["pairs"]):
        raise SystemExit(f"AP3 {name}: R holds {len(R)} pairs, the ladder {st['r']['pairs']} (G1)")
    with np.load(kp) as z:
        ka, kb, kdf = z["a"].astype(np.int64), z["b"].astype(np.int64), z["df"].astype(np.int64)
    if ka.size and (not (ka < kb).all() or ka.min() < 0 or kb.max() >= n):
        raise SystemExit(f"AP3 {name}: a K1 pair that is not a < b inside the {n} passages (G3)")
    keep = kdf <= cap_i
    E = R | {(int(a), int(b)) for a, b in zip(ka[keep].tolist(), kb[keep].tolist())}
    row = next(r for r in st["ladder"] if (INF if r["cap"] == "inf" else int(r["cap"])) == cap_i)
    if len(E) != int(row["pairs_total"]):
        raise SystemExit(f"AP3 {name}: the carve's structural family holds {len(E)} pairs, the ladder's rung {row['pairs_total']} (G1)")
    def written(pairs, path):                                                              # the structural file's writer: both directions, sorted, rel zeros int16
        srt = sorted(pairs)
        s_, d_ = L.both_directions(np.asarray([p[0] for p in srt], np.int64), np.asarray([p[1] for p in srt], np.int64), n)
        u1a.save_npz(path, src=s_, dst=d_, rel=np.zeros(s_.size, np.int16))
        return s_, d_

    with tempfile.TemporaryDirectory() as td:                                              # G3b: the writer reproduces b1w's file from R's pairs, byte for byte
        written(R, Path(td) / STRUCT)
        if u1a.sha_file(Path(td) / STRUCT) != w["files"][STRUCT]:
            raise SystemExit(f"AP3 {name}: the structural writer does not reproduce b1w's file from R's pairs (G3b)")
    out.mkdir(parents=True, exist_ok=True)
    for f in u1a.FILES:
        if f != STRUCT:
            shutil.copyfile(src_dir / f, out / f)
            if u1a.sha_file(out / f) != build["files"][f]:
                raise SystemExit(f"AP3 {out / f}: the copy is not b1n's")
    s, d = written(E, out / STRUCT)
    links = {"cap": cap, "rung": row, "r_pairs": len(R), "pairs_added_over_r": len(E) - len(R), "pairs": len(E), "directed_edges_file": int(s.size),
             "ladder_json_sha256": sha_file(LADDER), "k1_npz_sha256": st["k1"]["npz_sha256"], "r_structural_sha256": w["files"][STRUCT],
             "b1n_build_sha256": sha_file(src_dir / "build.json"), "zap3_ladder_script_sha256": lad["script_sha256"]}
    vb = variant_build(build, u1a.sha_file(out / STRUCT), links, u1a.sha_src(__file__), u1a.utc())
    u1a.write_json(out / "build.json", vb)
    shas = {f: u1a.sha_file(out / f) for f in u1a.FILES}                                  # G2b: the folder as written is the record's, and B1a's lists accept it
    if shas != vb["files"] or any(shas[f] != build["files"][f] for f in u1a.FILES if f != STRUCT):
        raise SystemExit(f"AP3 {name}: the folder as written is not its build.json's, or a copied file is not b1n's (G2b)")
    named = H.read_json(B1A_REAL / name / "b1_lists.json")["setting_files_sha256"]
    check_lists(named, shas, vb)
    got = und_pairs(out / STRUCT)
    if got != E or not np.array_equal(np.sort(s.astype(np.int64) * (n + 1) + d), s.astype(np.int64) * (n + 1) + d):
        raise SystemExit(f"AP3 {name}: the written structural file is not the carve's pairs, once each way, sorted (G3)")
    log(f"bench {name} {CARVE}: R {len(R)} + {len(E) - len(R)} pairs at df cap {cap} = {len(E)} pairs ({s.size} directed edges); {time.time() - t0:.0f} s")
    return 0


# -- zb1h's steps with AP3's names -------------------------------------------------------------------------------------------------

def settings(host=False, only=None):
    """zb1h.settings on the carve's folder; the record's label names the form."""
    bind()
    orig = H.write_json

    def labelled(p, obj):
        if isinstance(obj, dict) and "desc_b1" in obj:
            obj["desc_b1"]["variant"] = VARIANT
            obj["desc_b1"]["form"] = FORM
            obj["ap3"] = {"script_sha256": H.sha_src(__file__), "zb1h_sha256": H.sha_src(H.__file__)}
        return orig(p, obj)

    H.write_json = labelled
    try:
        return H.settings(host, only)
    finally:
        H.write_json = orig


def chain(name, host=False, root=None):
    """zb1h.chain, whose steps are run as this file's (zb1h spawns its own __file__)."""
    bind()
    saved = H.__file__
    H.__file__ = str(Path(__file__).resolve())
    try:
        return H.chain(name, host, root)
    finally:
        H.__file__ = saved


def gate(tag, host=False):
    bind()
    ZB = H.zb()
    M = ZB.mods(host)
    R = copy.copy(H.roots_of(M))
    R.gate = H.gate_path(tag)
    carves = [(ds, CARVE) for ds in TAGS[tag]]
    with ZB.checks(M):
        verdict_ = M.PK.gate(R, carves=carves)
    H.write_json(OUT / f"ap3_gate_{tag}.json", {
        "declared_in": DECLARED, "stage": STAGE, "tag": tag, "pko_gate": H.rel(R.gate), "pko_gate_sha256": H.sha_file(R.gate),
        "verdict": verdict_, "carves": carves, "settings": {ds: H.sha_file(H.settings_paths(ds)[1]) for ds in TAGS[tag]},
        "zb1h_sha256": H.sha_src(H.__file__), "zap3_sha256": H.sha_src(__file__), "utc": H.utc()})
    return 0


# -- the report (the laptop): b1y against b1n, b1w and b1t, paired by question -----------------------------------------------------------

def consts_of(v):
    out = ROOT / "outputs" / f"b1{v}"
    return {"CARVE": f"b1{v}", "OUT": out, "LTM_OUT": out / "ltm", "B1A": out / "bench" / f"hipporag2_{v}", "CARVES": ("b1t", "b1w", f"b1{v}")}


@contextlib.contextmanager
def as_variant(v):
    """zb1h's module globals set to carve b1<v>'s and folders (read paths, the carve list) and put back."""
    keep = {k: getattr(H, k) for k in consts_of(v)}
    try:
        for k, x in consts_of(v).items():
            setattr(H, k, x)
        yield
    finally:
        for k, x in keep.items():
            setattr(H, k, x)


def word(ci, up, down, flat):
    return up if ci[0] > 0 else down if ci[1] < 0 else flat


PAIRS = (("b1y-b1n", "y", "n", VS_W), ("b1y-b1w", "y", "w", VS_W), ("b1y-b1t", "y", "t", STEP), ("b1n-b1t", "n", "t", STEP))


def pairs_block(means, which, hop_mask=None):
    """paired['b1y-b1n M'] = {R@5, ci}; calls[M] = {pair: word}; over the questions in hop_mask if given. means: {(model, carve, which): R@5 per question}."""
    paired, calls = {}, {}
    letter = {"t": "b1t", "w": "b1w", "n": "b1n", "y": "b1y"}
    for arm in MODELS:
        d = {k: means.get((arm, c, which)) for k, c in letter.items()}
        if any(d[k] is None for k in letter):
            continue
        if hop_mask is not None:
            d = {k: x[hop_mask] for k, x in d.items()}
        for key, a_, b_, _words in PAIRS:
            paired[f"{key} {arm}"] = H.ci_of(d[a_] - d[b_])
        calls[arm] = {key: word(paired[f"{key} {arm}"]["ci"], *words) for key, _a, _b, words in PAIRS}
    return paired, calls


def stage_call(words):
    return L.stage_call(words)


def verdict(calls_by_ds):
    """Part B over the carved settings (b1y - b1n, in-domain, both models): HOLD when a setting's call is LESS or MIXED (decided even while another is missing);
    INCOMPLETE while a read is missing; ADOPT when no call is LESS or MIXED and a setting's call is GREATER or NOT_LESS (a model above, none below); NO_GAIN when
    every call is SAME."""
    calls = list(calls_by_ds.values())
    if not calls:
        return "INCOMPLETE"
    if any(c in ("LESS", "MIXED") for c in calls):
        return "HOLD"
    if any(c in (None, "INCOMPLETE") for c in calls):
        return "INCOMPLETE"
    if any(c in ("GREATER", "NOT_LESS") for c in calls):
        return "ADOPT"
    return "NO_GAIN"


def hop_of(kind):
    return f"{str(kind)[0]}hop"


def read_carve(v, ds, rec):
    """The reads of carve b1<v> (with b1t and b1w) on setting ds: the block of the report and the per-question means."""
    with as_variant(v):
        qs = H.read_json(B1A_REAL / ds / "queries.json")
        ids, gc = [str(q["id"]) for q in qs], [len(q["golds"]) for q in qs]
        perq, shas = H.read_set(ds, ids, gc, rec["missing"])
        s = {"questions": len(ids), "graph": GRAPH_TEXT if v == VARIANT else f"AP2 b1{v}"}
        reads, means = H.reads_block(ds, perq, shas)
        s.update(reads)
    return s, means, qs


def report():
    bind()
    import b1b as BB
    lad = H.read_json(LADDER)
    rec = {"declared_in": DECLARED, "stage": STAGE, "candidate": BB.CAND, "boot": {"resamples": BB.BOOT, "seed": BB.SEED},
           "in_domain": {k: list(x) for k, x in H.IN_DOMAIN.items()}, "carried": lad["carried"], "settings": {}, "missing": []}
    b1m = H.read_json(ROOT / "outputs" / "b1m" / "report.json")
    rec["inputs_sha256"] = {x: H.sha_file(ROOT / "outputs" / x / "report.json") for x in ("b1p", "b1m", "ap2")}
    rec["inputs_sha256"]["ap3_ladder"] = sha_file(LADDER)
    checks = {"equal_to_b1m": {"checked": 0, "bad": []}}
    calls_by_ds = {}
    for ds in SETTINGS:
        if lad["carried"].get(ds) is None:
            continue
        have = (consts_of("y")["B1A"] / ds / "build.json").exists() and (ROOT / "outputs" / CARVE / "settings" / f"{ds}.json").exists()
        if not have:
            calls_by_ds[ds] = None
            continue
        s, means, qs = read_carve("y", ds, rec)
        _sn, means_n, _q = read_carve("n", ds, rec)
        means = {**means, **{k: v for k, v in means_n.items() if k[1] == "b1n"}}
        n, bad = H.equal_to_b1m(ds, s, b1m)
        checks["equal_to_b1m"]["checked"] += n
        checks["equal_to_b1m"]["bad"] += [f"{ds} {k}" for k in bad]
        s["paired"], c_in = pairs_block(means, "in_domain")
        s["paired_zero_shot"], c_zs = pairs_block(means, "zero_shot")
        s["calls_vs_b1n"] = {"in_domain": c_in, "zero_shot": c_zs}
        s["call"] = stage_call([c_in.get(a, {}).get("b1y-b1n") for a in MODELS])
        calls_by_ds[ds] = s["call"]
        if ds == "musique":
            hops = np.asarray([hop_of(q["kind"]) for q in qs])
            s["by_hops"] = {}
            for h in HOPS:
                m = hops == h
                pr, cl = pairs_block(means, "in_domain", m)
                s["by_hops"][h] = {"questions": int(m.sum()), "paired": pr, "calls": cl}
        s["links"] = H.read_json(consts_of("y")["B1A"] / ds / "build.json")["ap3"]
        rec["settings"][ds] = s
        log(f"{ds}: done (call {s['call']})")
    if checks["equal_to_b1m"]["bad"]:
        raise SystemExit(f"AP3: rows recomputed from B1P's and B1M's reads differ from B1M's filed rows: {checks['equal_to_b1m']['bad']}")
    rec["checks"] = checks
    rec["verdict"] = verdict(calls_by_ds)
    rec["zap3_sha256"] = H.sha_src(__file__)
    rec["zb1h_sha256"] = H.sha_src(H.__file__)
    rec["utc"] = H.utc()
    H.write_json(REPORT_OUT / "report.json", rec)
    H.write_json(REPORT_OUT / "calls.json", {
        "declared_in": DECLARED, "verdict": rec["verdict"], "carried": lad["carried"], "call": {ds: s["call"] for ds, s in rec["settings"].items()},
        "calls_vs_b1n": {ds: s["calls_vs_b1n"] for ds, s in rec["settings"].items()}})
    write_md(rec)
    log(f"report: {H.rel(REPORT_OUT / 'report.md')}; missing {len(rec['missing'])}; verdict {rec['verdict']}; "
        f"equal to B1M's rows: {checks['equal_to_b1m']['checked']} checked")
    return 0


def lv(c):
    return f"{100 * c['R@5']:.1f} ({100 * c['R5_ci'][0]:.1f}-{100 * c['R5_ci'][1]:.1f})" if c else "--"


def df(c):
    return f"{100 * c['R@5']:+.1f} ({100 * c['ci'][0]:+.1f}, {100 * c['ci'][1]:+.1f})" if c else "--"


def write_md(rec):
    T = [f"# AP3 report, Part B (declared in {DECLARED})", "",
         "R@5 x100 on HippoRAG 2's settings (1,000 questions each), no new fit. The models of record (zg6r, zrckpn) are read on b1t (N + K), b1w (R + spaCy's N + K), b1n "
         "(R + F0T + K, AP2) and **b1y (R + the carried rung's base-title pairs in the structural slot, F0T, K)**: the other files are b1n's. A read-time shift: the models "
         "were fitted with R in the structural slot. In-domain = the mean over the fits that trained on the setting's partner split. 95% bootstrap intervals over the "
         "questions (2,000 resamples, seed 0); words by the interval, no threshold (GREATER / SAME / LESS for b1y - b1n and b1y - b1w; UP / DOWN / FLAT against b1t).", "",
         f"**Verdict (Part B): {rec['verdict']}** (carried rungs: {', '.join(f'{d} {c}' for d, c in rec['carried'].items())}).", ""]
    for which, title, key in (("in_domain", "In-domain", "paired"), ("zero_shot", "Held-out", "paired_zero_shot")):
        T += [f"## {title}", "", "| setting | model | b1t | b1w | b1n | b1y | b1y - b1n | b1y - b1w | b1y - b1t | word (b1y - b1n) |", "|---|---|---:|---:|---:|---:|---|---|---|---|"]
        for ds, s in rec["settings"].items():
            for arm in MODELS:
                ld = {c: s[which].get(f"{arm}/{c}") for c in ("b1t", "b1w", "b1n", "b1y")}
                if not any(ld.values()):
                    continue
                pr = {k: s[key].get(f"{k} {arm}") for k in ("b1y-b1n", "b1y-b1w", "b1y-b1t")}
                w = s["calls_vs_b1n"][which].get(arm, {}).get("b1y-b1n", "--")
                T.append(f"| {ds} | {arm} | {lv(ld['b1t'])} | {lv(ld['b1w'])} | {lv(ld['b1n'])} | {lv(ld['b1y'])} | {df(pr['b1y-b1n'])} | {df(pr['b1y-b1w'])} | {df(pr['b1y-b1t'])} | {w} |")
        T.append("")
    T += ["## The setting's call (in-domain, both models)", "", "| setting | call |", "|---|---|"]
    for ds, s in rec["settings"].items():
        T.append(f"| {ds} | {s['call']} |")
    T += ["", "## zg6r - zrckpn at b1y, paired by question", "", "| setting | read | zg6r - zrckpn (CI) | label |", "|---|---|---|---|"]
    for ds, s in rec["settings"].items():
        for k, c in s.get("gnn_minus_mlp", {}).items():
            if k.split("/")[1] == "b1y":
                T.append(f"| {ds} | {k.split('/')[0].replace('_', '-')} | {df(c)} | {c['label']} |")
    mq = rec["settings"].get("musique")
    if mq and mq.get("by_hops"):
        T += ["", "## MuSiQue by hop depth (in-domain, paired by question, descriptive)", "", "| hops | questions | model | b1y - b1n |", "|---|---:|---|---|"]
        for h in HOPS:
            x = mq["by_hops"][h]
            for arm in MODELS:
                if x["paired"].get(f"b1y-b1n {arm}"):
                    T.append(f"| {h} | {x['questions']} | {arm} | {df(x['paired'][f'b1y-b1n {arm}'])} |")
    T += ["", "## The structural files", "", "| setting | df cap | R pairs | pairs added | pairs in the carve |", "|---|---|---:|---:|---:|"]
    for ds, s in rec["settings"].items():
        k = s["links"]
        T.append(f"| {ds} | {k['cap']} | {k['r_pairs']} | {k['pairs_added_over_r']} | {k['pairs']} |")
    T += ["", f"Missing reads: {len(rec['missing'])}."]
    REPORT_OUT.mkdir(parents=True, exist_ok=True)
    (REPORT_OUT / "report.md").write_text("\n".join(T) + "\n", encoding="utf-8")


# -- self-tests ----------------------------------------------------------------------------------------------------------------------

def selftest():
    bind()
    fit = "scr-ml1-zg6r-s0-hp"
    assert CARVE == f"b1{VARIANT}" and OUT == ROOT / "outputs" / CARVE and BENCH_G == OUT / "bench" / f"hipporag2_{VARIANT}"
    assert H.read_path(L.GNN, fit, CARVE, "hotpotqa") == OUT / "fits/h" / fit / f"reads/hotpotqa__{CARVE}.npz", "the carve's read path"
    assert H.read_path(L.GNN, fit, "b1t", "musique") == H.B1P / "fits/t" / fit / "reads/musique__b1t.npz", "b1t is B1P's"
    assert H.read_flags("hotpotqa") == ["--read", f"hotpotqa={CARVE}", "--out-root", f"outputs/{CARVE}/fits/h"], H.read_flags("hotpotqa")
    assert H.gate_path("musique").parent == OUT
    assert H.B1A == BENCH_G and H.CARVES == CARVES and H.CARVE == CARVE and CARVES[-1] == CARVE
    assert H.on_b1h is on_b1y and H.read_roots is read_roots and all(v.startswith(f"outputs/{CARVE}/") for v in read_roots().values())
    assert set(read_roots()) == {"pools", "plook", "cache", "rmatch", "zrc", "links", "kt"}
    assert set(TAGS) == set(SETTINGS) and all(TAGS[d] == (d,) for d in TAGS) and SETTINGS[-1] == "musique"
    assert CAPS[-1] == INF and list(CAPS) == sorted(set(CAPS)) and CAPS[0] == 1
    # the structural file's layout: each pair once each way, sorted, int32 (zb1l.both_directions), as b1w's file is
    s, d = L.both_directions([1, 0, 3], [2, 2, 1], 4)
    assert list(zip(s.tolist(), d.tolist())) == [(0, 2), (1, 2), (1, 3), (2, 0), (2, 1), (3, 1)] and s.dtype == np.int32 and d.dtype == np.int32
    # the ladder on a toy: R holds (0,1) and (2,3); K1 holds (0,2) df 1, (4,5) df 2, (1,3) df 2, (6,7) df 9, (2,3) df 1 (already in R)
    R = {(0, 1), (2, 3)}
    k1 = {(0, 2): 1, (4, 5): 2, (1, 3): 2, (6, 7): 9, (2, 3): 1}
    slots = [(0, 0, 1), (0, 0, 2), (1, 4, 5), (1, 1, 3), (2, 6, 7), (2, 2, 3), (3, 8, 9)]            # (question, a, b)
    prof = reach_profile(R, k1, slots, (1, 2, 9, INF))
    assert prof["base_linked"] == 2 and prof["rows"] == [{"cap": 1, "pairs_added": 1, "slots_added": 1}, {"cap": 2, "pairs_added": 3, "slots_added": 3},
                                                       {"cap": 9, "pairs_added": 4, "slots_added": 4}, {"cap": INF, "pairs_added": 4, "slots_added": 4}], prof["rows"]
    assert list(prof["mincap"]) == [0, 1, 2, 2, 9, 0, INF + 1]
    out = rung_rows(R, k1, slots, 4, {(0, 2), (4, 5)}, (1, 2, 9, INF), ci=lambda cx, cr, q, nq: boot_ratio(cx, cr, q, nq, 50, 0))
    assert out["base_linked"] == 2 and out["slots"] == 7
    r2 = out["rows"][1]
    assert r2["cap"] == 2 and r2["pairs_added"] == 3 and r2["pairs_total"] == 5 and r2["slots_added"] == 3 and r2["slots_linked"] == 5 and r2["share"] == 71.43
    assert r2["gain_points"] == round(100 * 3 / 7, 3) and r2["price"] == 1.0 and r2["s_added"] == 2 and r2["gain_call"] in ("ABOVE", "LEVEL", "BELOW")
    assert out["rows"][-1]["cap"] == "inf" and all(r["gain_ci"][0] <= r["gain_points"] <= r["gain_ci"][1] + 1e-9 for r in out["rows"][:0] or [])
    # the bootstrap copy: a fixed answer on fixed vectors, the same on a repeat (seeded)
    q = [0, 0, 1, 1, 2]
    cx, cr = [1, 1, 1, 0, 1], [1, 0, 0, 0, 1]
    b1 = boot_ratio(cx, cr, q, 3, 100, 0)
    assert b1 == boot_ratio(cx, cr, q, 3, 100, 0) and abs(b1[0] - 0.4) < 1e-12 and b1[1] <= b1[0] <= b1[2], b1
    # S1 on synthetic ladders (the counts decide, never the rounded fields)
    def rows_of(*pg):
        return [{"cap": c, "pairs_added": p, "slots_added": g} for c, (p, g) in zip((1, 2, 3, 4), pg)]
    base_pairs, base_linked, n_slots = 4000, 600, 1000                                             # p0 = 6.667
    cp, why, p0 = carry(rows_of((15, 5), (200, 40), (350, 55), (500, 60)), base_pairs, base_linked, n_slots)
    assert cp == 3 and why == "carried" and abs(p0 - 4000 / 600) < 1e-12, (cp, why)                  # 350/55 = 6.36 passes, 500/60 = 8.33 stops
    assert carry(rows_of((15, 5), (200, 40), (350, 55), (360, 55)), base_pairs, base_linked, n_slots)[0] == 4, "prices that pass to the end carry the last rung"
    assert carry(rows_of((60, 5), (200, 40)), base_pairs, base_linked, n_slots)[0] is None, "the first rung over p0 stops the ladder"
    assert carry(rows_of((15, 5), (200, 40), (350, 55)), base_pairs, base_linked, 5000)[0] is None, "below the floor: 55 of 5000 slots is 1.1 points"
    assert carry(rows_of((15, 5), (200, 0), (210, 9)), base_pairs, base_linked, 200)[0] == 1, "pairs added with no slot added never pass: the ladder stops at the rung before"
    assert carry(rows_of((15, 5), (200, 0), (210, 9)), base_pairs, base_linked, n_slots)[0] is None, "and that rung adds 5 of 1000 slots, below the floor"
    assert carry(rows_of((0, 0), (0, 0)), base_pairs, base_linked, n_slots)[0] is None, "nothing added: no rung to carry (below the floor)"
    assert carry([{"cap": "inf", "pairs_added": 100, "slots_added": 30}], base_pairs, base_linked, n_slots)[0] == INF, "the uncapped rung carries as INF"
    assert carry(rows_of((50, 20)), base_pairs, 0, n_slots)[2] == float("inf")
    # the lists check: B1a's named shas against the carve's, the structural and NER files replaced
    build = {"u1a": {"b1a_structural_sha256": "S0"}, "ap2": {"b1a_ner_sha256": "N0"}}
    named = {"first_stage.npz": "f", STRUCT: "S0", NER: "N0", "graph_knn.npz": "k", "nodes.json": "n", "queries.json": "q"}
    shas = {"first_stage.npz": "f", STRUCT: "S1", NER: "N1", "graph_knn.npz": "k", "nodes.json": "n", "queries.json": "q"}
    assert check_lists(named, shas, build) == named
    for bad_named, bad_shas in (({**named, "graph_knn.npz": "x"}, shas), ({**named, NER: "N9"}, shas), ({**named, STRUCT: "S9"}, shas)):
        refused = False
        try:
            check_lists(bad_named, bad_shas, build)
        except SystemExit:
            refused = True
        assert refused, "the lists check must refuse a file that differs"
    # the carve's build.json: b1n's, the structural sha replaced, B1a's structural and b1n's NER records kept, the u1a block relabelled, an ap3 block
    bn = {"files": {STRUCT: "SW", NER: "NF"}, "passages": 5, "b1f": {"graph_file_sha256": "SW", "edges": 7},
          "u1a": {"b1a_structural_sha256": "S0", "structural": "b1w's", "variant": "n", "edges": 7}, "ap2": {"b1a_ner_sha256": "N0", "form": "F0T"}}
    vb = variant_build(bn, "SY", {"cap": 3, "directed_edges_file": 11, "pairs": 6}, "dd", "t")
    assert vb["files"] == {STRUCT: "SY", NER: "NF"} and bn["files"][STRUCT] == "SW", "b1n's record is not touched"
    assert vb["u1a"]["b1a_structural_sha256"] == "S0" and vb["u1a"]["variant"] == VARIANT and vb["u1a"]["carve"] == CARVE and vb["u1a"]["edges"] == 11
    assert vb["ap2"] == bn["ap2"] and vb["ap3"]["cap"] == 3 and vb["b1f"]["graph_file_sha256"] == "SW" and "note" in vb["b1f"] and vb["passages"] == 5
    assert check_lists({"x": "1", STRUCT: "S0", NER: "N0"}, {"x": "1", STRUCT: "SY", NER: "NF"}, vb) is not None
    # bench refuses a setting with no carried rung
    keep_ladder = LADDER
    import tempfile
    try:
        tmp = Path(tempfile.mkdtemp())
        globals()["LADDER"] = tmp / "ladder.json"
        H.write_json(globals()["LADDER"], {"carried": {"2wiki": None}, "settings": {}})
        refused = False
        try:
            bench("2wiki")
        except SystemExit as e:
            refused = "no rung" in str(e)
        assert refused, "bench must refuse a setting S1 carried nothing for"
    finally:
        globals()["LADDER"] = keep_ladder
    # the settings record's label
    seen_ = []
    orig, hold = H.write_json, H.settings
    try:
        H.write_json = lambda p, obj: seen_.append(obj)
        H.settings = lambda host=False, only=None: H.write_json(Path("x.json"), {"dataset": "musique", "desc_b1": {"variant": "h"}}) or 0
        assert settings(False, "musique") == 0
    finally:
        H.settings, H.write_json = hold, orig
    assert seen_[0]["desc_b1"]["variant"] == VARIANT and seen_[0]["desc_b1"]["form"] == FORM and "ap3" in seen_[0], "label"
    assert H.write_json is orig and H.settings is hold, "put back"
    # as_variant sets zb1h's globals to the other carve's and puts them back
    here = (H.CARVE, H.OUT, H.CARVES)
    with as_variant("n"):
        assert H.CARVE == "b1n" and H.OUT == ROOT / "outputs" / "b1n" and H.CARVES == ("b1t", "b1w", "b1n")
        assert H.read_path(L.GNN, fit, "b1n", "musique") == ROOT / "outputs" / "b1n" / "fits/h" / fit / "reads/musique__b1n.npz"
    assert (H.CARVE, H.OUT, H.CARVES) == here, "put back"
    # the pairs, the words, the setting calls and the verdict on fake reads
    rs = np.random.RandomState(7)
    nq = 90
    base = rs.rand(nq) * 0.5
    means = {}
    for arm in MODELS:
        for which in ("in_domain", "zero_shot"):
            means[(arm, "b1t", which)] = base
            means[(arm, "b1w", which)] = base + 0.05 + rs.randn(nq) * 0.03
            means[(arm, "b1n", which)] = means[(arm, "b1w", which)] + rs.randn(nq) * 0.001
            means[(arm, "b1y", which)] = means[(arm, "b1n", which)] + 0.04 + rs.randn(nq) * 0.03          # b1y above b1n
    pr, cl = pairs_block(means, "in_domain")
    assert set(pr) == {f"{k} {a}" for a in MODELS for k, _x, _y, _w in PAIRS} and set(cl) == set(MODELS)
    for a in MODELS:
        assert cl[a]["b1y-b1n"] == "GREATER" and cl[a]["b1y-b1t"] == "UP" and cl[a]["b1n-b1t"] == "UP", cl[a]
    assert pairs_block({k: v for k, v in means.items() if k[1] != "b1y"}, "in_domain") == ({}, {}), "no b1y, no pairs"
    kinds = ["2hop"] * 50 + ["3hop1"] * 25 + ["4hop2"] * 15
    hops = np.asarray([hop_of(k) for k in kinds])
    assert [int((hops == h).sum()) for h in HOPS] == [50, 25, 15]
    pr2, cl2 = pairs_block(means, "in_domain", hops == "4hop")
    assert set(cl2) == set(MODELS) and pr2[f"b1y-b1n {GNN}"]["R@5"] != pr[f"b1y-b1n {GNN}"]["R@5"], "over the 4-hop questions only"
    assert stage_call(["GREATER", "GREATER"]) == "GREATER" and stage_call(["GREATER", "SAME"]) == "NOT_LESS"
    assert stage_call(["LESS", "SAME"]) == "MIXED" and stage_call(["LESS", "LESS"]) == "LESS" and stage_call(["SAME", None]) == "INCOMPLETE"
    assert verdict({"musique": "GREATER", "hotpotqa": "SAME"}) == "ADOPT" and verdict({"musique": "SAME", "hotpotqa": "NOT_LESS"}) == "ADOPT"
    assert verdict({"musique": "SAME", "hotpotqa": "SAME"}) == "NO_GAIN" and verdict({"musique": "SAME"}) == "NO_GAIN"
    assert verdict({"musique": "SAME", "hotpotqa": "LESS"}) == "HOLD" and verdict({"musique": None, "hotpotqa": "MIXED"}) == "HOLD", "a LESS or MIXED decides"
    assert verdict({"musique": "GREATER", "hotpotqa": None}) == "INCOMPLETE" and verdict({}) == "INCOMPLETE"
    # the markdown on a fake record
    cell = {"R@5": 0.5, "R5_ci": [0.4, 0.6]}
    srec = {"in_domain": {f"{a}/{c}": cell for a in MODELS for c in ("b1t", "b1w", "b1n", "b1y")}, "zero_shot": {}, "paired": pr, "paired_zero_shot": {},
            "calls_vs_b1n": {"in_domain": cl, "zero_shot": {}}, "call": "GREATER",
            "by_hops": {h: {"questions": 5, "paired": pr, "calls": cl} for h in HOPS},
            "gnn_minus_mlp": {"in_domain/b1y": {"R@5": 0.01, "ci": [0.0, 0.02], "label": "GNN_ABOVE"}, "in_domain/b1n": {"R@5": 0.0, "ci": [-0.01, 0.01], "label": "LEVEL"}},
            "links": {"cap": 3, "r_pairs": 4, "pairs_added_over_r": 2, "pairs": 6}}
    rec = {"verdict": "ADOPT", "carried": {"musique": 3, "2wiki": None}, "settings": {"musique": srec}, "missing": []}
    global REPORT_OUT
    keep_out = REPORT_OUT
    try:
        REPORT_OUT = Path(tempfile.mkdtemp())
        write_md(rec)
        md = (REPORT_OUT / "report.md").read_text(encoding="utf-8")
        write_ladder_md({"floor_points": FLOOR, "carried": {"musique": 3}, "settings": {"musique": {
            "basis": "CH", "slots": 7, "passages": 9, "r": {"pairs": 2, "slots_linked": 2, "share": 28.57, "s_precision": 1.0, "s_recall": 1.0},
            "s1": {"carried_cap": 3, "why": "carried", "p0_pairs_per_slot": 1.0}, "ladder": out["rows"],
            "union": {"rows": [{"pairs_added": 1, "slots_added": 1}] * len(out["rows"])}}}}, REPORT_OUT)
        lmd = (REPORT_OUT / "ladder.md").read_text(encoding="utf-8")
    finally:
        REPORT_OUT = keep_out
    assert "**Verdict (Part B): ADOPT**" in md and "| musique | zg6r | 50.0 (40.0-60.0)" in md and "hop depth" in md and "no threshold" in md, md[:500]
    gap = md.split("## zg6r - zrckpn at b1y")[1].split("\n## ")[0]
    assert "| musique | in-domain | +1.0 (+0.0, +2.0) | GNN_ABOVE |" in gap and "LEVEL" not in gap, gap
    assert "| inf |" in lmd and "**S1: 3**" in lmd and "Carried: musique 3." in lmd, lmd[:400]
    print(f"zap3 self-test: OK ({CARVE})")
    return 0


def selftest_ladder():
    """The link rule on a toy corpus (the laptop: zu2b): base-title links by df, the exact-title links apart, an ambiguous surface dropped, the pairs once each."""
    import zu2b
    corpus = [
        {"title": "Zorblax Quendar (band)", "text": "Zorblax Quendar is a band formed in 1990 by two brothers."},
        {"title": "Mirko Valenti", "text": "Mirko Valenti toured with Zorblax Quendar in 1995 and later left."},
        {"title": "Plenik Sodoro (writer)", "text": "Plenik Sodoro wrote about Zorblax Quendar and about Mirko Valenti in a long essay."},
        {"title": "Harlan Moor (singer)", "text": "Harlan Moor is a singer who works in the north of the country."},
        {"title": "Harlan Moor (painter)", "text": "Harlan Moor is a painter who works in the south of the country."},
        {"title": "Quillon Raste", "text": "Quillon Raste admired Harlan Moor and Mirko Valenti, and met Pomona Vice once."},
        {"title": "Pomona Vice (film)", "text": "Pomona Vice is a film released in 2001 with a long cast list."},
        {"title": "Tarn Belloc", "text": "Tarn Belloc saw Pomona Vice and Zorblax Quendar on the same night in spring."},
    ]
    r = k1_links(corpus)
    k0, k1 = r["k0"], r["k1"]
    t = {c["title"]: i for i, c in enumerate(corpus)}
    zq, mv, ps, hs, hp, qr, pv, tb = (t["Zorblax Quendar (band)"], t["Mirko Valenti"], t["Plenik Sodoro (writer)"], t["Harlan Moor (singer)"], t["Harlan Moor (painter)"],
                                      t["Quillon Raste"], t["Pomona Vice (film)"], t["Tarn Belloc"])
    assert pair(mv, ps) in k0 and pair(mv, qr) in k0, "exact-title links (K0)"
    assert pair(zq, mv) in k1 and pair(zq, ps) in k1 and pair(zq, tb) in k1 and pair(pv, qr) in k1 and pair(pv, tb) in k1, sorted(k1)
    assert k1[pair(zq, tb)] == k1[pair(zq, mv)] == k1[pair(zq, ps)] == 4, "df of 'Zorblax Quendar': the owner and three more passages"
    assert k1[pair(pv, qr)] == k1[pair(pv, tb)] == 3, "df of 'Pomona Vice': the owner and two more"
    assert pair(qr, hs) not in k1 and pair(qr, hp) not in k1 and r["ambiguous_dropped"] == {"K0": 0, "K1_only": 1}, \
        "'Harlan Moor' names two title groups for Quillon Raste's mention: dropped, counted once"
    assert k1[pair(hs, hp)] == 3, "each homonym page's own mention names the other as its one remaining owner (U2b's drop-self rule): a K1 pair at df 3"
    assert all(a < b for a, b in k1) and all(a < b for a, b in k0) and r["passages_checked"] == len(corpus)
    # the rung rows over the toy: the K1 pairs enter at their df
    slots = [(0, zq, mv), (0, mv, ps), (1, pv, qr), (1, pv, tb), (2, zq, tb)]
    slots = [(q, *pair(a, b)) for q, a, b in slots]
    out = rung_rows(k0, k1, slots, 3, None, (1, 3, 4, INF), ci=lambda cx, cr, q, nq: boot_ratio(cx, cr, q, nq, 50, 0))
    assert [x["slots_added"] for x in out["rows"]] == [0, 2, 5 - out["base_linked"], 5 - out["base_linked"]], out["rows"]
    print("zap3 self-test (ladder): OK")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?")
    ap.add_argument("kind", nargs="?")
    ap.add_argument("--dataset", choices=SETTINGS)
    ap.add_argument("--split", choices=H.SPLITS)
    ap.add_argument("--tag", choices=tuple(TAGS))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--index")
    ap.add_argument("--name")
    ap.add_argument("--root")
    ap.add_argument("--questions", type=int, default=8)
    ap.add_argument("--host", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "selftest":
        return selftest()
    if a.cmd == "selftest-ladder":
        return selftest_ladder()
    if a.cmd == "ladder":
        return ladder()
    need = {"bench": ("dataset",), "pools": ("dataset",), "chain": ("dataset",), "step": ("dataset",), "lists": ("dataset",),
            "cols": ("split", "dataset"), "gate": ("tag",), "read-gnn": ("name", "tag"), "read-mlp": ("name", "tag"),
            "look-dev": ("dataset", "root")}
    for k in need.get(a.cmd, ()):
        if getattr(a, k) is None:
            raise SystemExit(f"{a.cmd}: --{k} is required")
    bind()
    if a.root and Path(a.root).resolve() == OUT.resolve():
        a.root = None
    if a.cmd == "bench":
        return bench(a.dataset)
    if a.cmd == "settings":
        return settings(a.host, a.dataset)
    if a.cmd == "pools":
        return H.pools(a.dataset, a.limit, a.host, a.root)
    if a.cmd == "chain":
        return chain(a.dataset, a.host, a.root)
    if a.cmd == "step":
        return H.run_step(a.kind, a.dataset, H.parse_index(a.index), a.host, a.root)
    if a.cmd == "gate":
        return gate(a.tag, a.host)
    if a.cmd == "fork":
        return H.fork()
    if a.cmd == "lists":
        return H.lists(a.dataset)
    if a.cmd == "cols":
        return H.cols(a.split, a.dataset, a.host)
    if a.cmd == "read-gnn":
        return H.read_gnn(a.name, a.tag)
    if a.cmd == "read-mlp":
        return H.read_mlp(a.name, a.tag)
    if a.cmd == "report":
        return report()
    if a.cmd == "look-dev":
        return H.look_dev(a.dataset, a.root, a.questions)
    raise SystemExit(__doc__)


if __name__ == "__main__":
    sys.exit(main())
