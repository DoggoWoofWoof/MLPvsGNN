"""AP1: rule-built names in place of spaCy's NER, and the reach of the name rules on MuSiQue's chains (docs/AP1_RULE_NAMES_AND_MUSIQUE_REACH.md).

A graph-level read on B1's three passage settings (musique, 2wiki, hotpotqa): which pairs of a question's gold passages each edge family links, with
spaCy's co-mention family N_sp (B1a's graph_ner.npz) beside U2r's name edges and one new rule, all rebuilt from the setting's passage texts.
No model reads a vector, nothing trains, no filed graph or record changes. Labels (the gold pairs, MuSiQue's decomposition answers) are the grading
and Part B's diagnostic only; no rule reads one.

    python outputs/mp_unified/zap1.py run          # Parts A, B, C on the three settings -> outputs/ap1/report.json, report.md, <setting>/rules.npz
    python outputs/mp_unified/zap1.py selftest
"""
import os
import sys

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
sys.dont_write_bytecode = True

import hashlib  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from collections import Counter  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (HERE, ROOT / "scripts", ROOT / "src", ROOT / "src" / "mp_retrieval"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import oraclepairs as O  # noqa: E402
import zu2b  # noqa: E402
import zu2l_el as EL  # noqa: E402
import zu2r as U  # noqa: E402

DECLARED = "docs/AP1_RULE_NAMES_AND_MUSIQUE_REACH.md"
BENCH = ROOT / "outputs" / "bench" / "hipporag2"
BENCH_W = ROOT / "outputs" / "bench" / "hipporag2_w"
OUT = ROOT / "outputs" / "ap1"
SETTINGS = ("musique", "2wiki", "hotpotqa")
RAW = {"musique": "musique", "2wiki": "2wikimultihopqa", "hotpotqa": "hotpotqa"}
CAP = EL.DF[1]                                   # NER's df cap, 25
LO = EL.DF[0]                                    # 2
UNIT = 1.0 / CAP                                 # CN_w: the weight one name at the cap carries
TOL = 1e-9
BOOT, SEED = 2000, 0
NER_FORMS = ("none", "N_sp", "CN_u", "CN_all", "CN_wu", "CN_w")
WORLDS = {"R": "R", "S": "S"}                    # the structural family of the union: R (W2_S rule links) or S (B1a's)
REF = "N_sp"
CLAIMED = {"musique": (1648, 1090)}              # the scratch read's chain slots and the distinct ORC pairs, checked by gate G1


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def script_sha256():
    return hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


# ── the setting ─────────────────────────────────────────────────────────────


def load(d):
    """The setting's passages (node i = corpus entry i), B1a's queries and its graph files, checked against their build records."""
    corpus = json.loads((BENCH / "raw" / f"{RAW[d]}_corpus.json").read_text(encoding="utf-8"))
    nodes = json.loads((BENCH / d / "nodes.json").read_text(encoding="utf-8"))
    queries = json.loads((BENCH / d / "queries.json").read_text(encoding="utf-8"))
    if len(corpus) != len(nodes) or any(corpus[i]["title"] != nodes[i]["title"] for i in range(len(nodes))):
        raise SystemExit(f"{d}: the corpus file is not B1a's node order")
    rec_b = json.loads((BENCH / d / "build.json").read_text(encoding="utf-8"))["files"]
    rec_w = json.loads((BENCH_W / d / "build.json").read_text(encoding="utf-8"))["files"]
    for f in ("graph_ner.npz", "graph_knn.npz", "graph_structural.npz"):
        if sha_file(BENCH / d / f) != rec_b[f]:
            raise SystemExit(f"{d}: {f} is not its build record's")
    for f in ("graph_ner.npz", "graph_knn.npz", "graph_structural.npz"):
        if sha_file(BENCH_W / d / f) != rec_w[f]:
            raise SystemExit(f"{d}: hipporag2_w/{f} is not its build record's")
    for f in ("graph_ner.npz", "graph_knn.npz"):               # G5: the R-world differs from B1a's in the structural family only
        if rec_w[f] != rec_b[f]:
            raise SystemExit(f"{d}: b1w's {f} is not b1's")
    fam = {"S": O.undirected(BENCH / d / "graph_structural.npz"), "R": O.undirected(BENCH_W / d / "graph_structural.npz"),
           "N_sp": O.undirected(BENCH / d / "graph_ner.npz"), "K": O.undirected(BENCH / d / "graph_knn.npz")}
    return corpus, queries, fam


def gold_slots(queries):
    """GP: every unordered pair of distinct gold passages of a question, one slot each: [(question index, a, b)] with a < b."""
    out = []
    for qi, q in enumerate(queries):
        g = sorted({int(x) for x in q["golds"]})
        out += [(qi, a, b) for i, a in enumerate(g) for b in g[i + 1:]]
    return out


def chain_slots(d, corpus, queries):
    """CH (MuSiQue): the slots of oraclepairs' ORC, with each slot's bridge answer (the answer of the earlier sub-question its text refers to).
    Stops unless the slot count equals oraclepairs' own build."""
    raw = json.loads((BENCH / "raw" / f"{RAW[d]}.json").read_text(encoding="utf-8"))
    nodes = json.loads((BENCH / d / "nodes.json").read_text(encoding="utf-8"))
    pairs, rows, _meta = O.build(raw, corpus, nodes, queries)
    by_text = {}
    for i, c in enumerate(corpus):
        by_text.setdefault((c["title"], c["text"]), []).append(i)
    out = []
    for qi, q in enumerate(raw):
        pos, deps, _ = O.hop_nodes(q, by_text)
        seen = {}
        for j, r in deps:
            if pos[j] == pos[r]:
                continue
            p = (pos[j], pos[r]) if pos[j] < pos[r] else (pos[r], pos[j])
            seen.setdefault(p, str(q["question_decomposition"][r]["answer"]))
        out += [(qi, a, b, ans) for (a, b), ans in sorted(seen.items())]
    if len(out) != sum(r["orc"] for r in rows) or {(a, b) for _q, a, b, _x in out} != set(map(tuple, pairs["ORC"])):
        raise SystemExit(f"{d}: the chain slots are not oraclepairs' ORC")
    return out, {"slots": len(out), "distinct_pairs": len(pairs["ORC"]), "ora_pairs": len(pairs["ORA"]), "coverage": O.coverage(d)}


# ── the names (U2r's rule, with each node's names kept) ──────────────────────


def names_table(passes, n):
    """U2r's build_names with the per-node name sets and the name vocabulary kept: (per-node sets, src, grp, titled, vocab keys)."""
    cap, low, toks, titles = {}, {}, [], []
    for t, x in passes():
        titles.append(t)
        tk = U.tokens(x) if x is not None else None
        toks.append(tk)
        if tk is not None:
            U.count_stat(tk, cap, low)

    def usual(k):
        return not (low.get(k, 0) > cap.get(k, 0))

    vocab, src, grp, per = {}, [], [], []
    for i, tk in enumerate(toks):
        s = set()
        if tk is not None:
            s = U.names_of(tk, usual)
            for k in s:
                src.append(i)
                grp.append(vocab.setdefault(k, len(vocab)))
        per.append(s)
    if len(titles) != n:
        raise SystemExit(f"{len(titles)} passages, expected {n}")
    tk_set = U.title_key_set(titles)
    titled = np.zeros(len(vocab), bool)
    for k, v in vocab.items():
        titled[v] = k in tk_set
    return per, np.asarray(src, np.int64), np.asarray(grp, np.int64), titled, vocab


def pair_set(a, b):
    return set(zip(a.tolist(), b.tolist()))


def rule_families(src, grp, titled, n):
    """CN_all and CN_u at the cap (U2r), and CN_w and CN_wu: pairs whose summed weight over shared names of any df >= 2 reaches the weight one name at the
    cap carries (1/25). Each as {pair: weight}-free pair arrays (a < b) plus the weights of the cap-free product, for the report."""
    out = {}
    df = np.bincount(grp, minlength=titled.size) if grp.size else np.zeros(0, int)
    hi = int(df.max()) if df.size else CAP
    for tag, keep in (("", np.ones(grp.size, bool)), ("u", ~titled[grp])):
        a, b, w = EL.co_mentions(src[keep], grp[keep], n, lo=LO, hi=CAP)
        out["CN_u" if tag else "CN_all"] = (a, b, w)
        a2, b2, w2 = EL.co_mentions(src[keep], grp[keep], n, lo=LO, hi=max(hi, CAP))
        ok = w2 >= UNIT - TOL
        out["CN_wu" if tag else "CN_w"] = (a2[ok], b2[ok], w2[ok])
    return out


# ── reach and the paired interval ───────────────────────────────────────────


def covered(slots, E):
    return np.fromiter(((s[1], s[2]) in E for s in slots), bool, count=len(slots))


def boot_diff(cx, cr, qidx, nq, boot=BOOT, seed=SEED):
    """The share of slots a family links minus the reference's, by the ratio estimator over questions: the point value and the 95% interval of
    `boot` resamples of the questions (seeded)."""
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


def call(lo, hi):
    return "ABOVE" if lo > 0 else ("BELOW" if hi < 0 else "LEVEL")


def pct(x):
    return round(100 * float(x), 2)


# ── Part A: the unions ──────────────────────────────────────────────────────


def part_a(slots, nq, fam, rules, kinds):
    """For each world (structural family R or S) and NER form: the union with K, its pairs, mean degree, the share of each slot kind it links, and the
    paired difference to the same world's union with N_sp."""
    n_nodes = kinds["n"]
    forms = {"none": set(), "N_sp": fam["N_sp"], **{k: pair_set(v[0], v[1]) for k, v in rules.items()}}
    out = {}
    for world, sk in WORLDS.items():
        base = fam[sk] | fam["K"]
        unions = {f: base | forms[f] for f in NER_FORMS}
        cov = {}
        for kind, sl in slots.items():
            qidx = [s[0] for s in sl]
            cov[kind] = {f: covered(sl, unions[f]) for f in NER_FORMS}
            cov[kind]["_q"] = qidx
        rows = {}
        for f in NER_FORMS:
            row = {"edges": len(unions[f]), "mean_degree": round(2 * len(unions[f]) / n_nodes, 3), "ner_edges": len(forms[f]),
                   "ner_new_edges": len(forms[f] - base)}
            for kind, sl in slots.items():
                c = cov[kind]
                share = pct(c[f].mean())
                dlt, lo, hi = boot_diff(c[f], c[REF], c["_q"], nq)
                row[kind] = {"slots": len(sl), "linked": int(c[f].sum()), "share": share,
                             "vs_N_sp": {"diff": pct(dlt), "lo": pct(lo), "hi": pct(hi), "call": call(lo, hi)}}
                d2, l2, h2 = boot_diff(c[f], c["CN_all"], c["_q"], nq)
                row[kind]["vs_CN_all"] = {"diff": pct(d2), "lo": pct(l2), "hi": pct(h2), "call": call(l2, h2)}
                if f not in ("none", REF):
                    extra = int(c[f].sum() - c["none"].sum())          # slots the NER slot adds over the world's union without it
                    row[kind]["added_over_none"] = extra
            row["edge_ratio_to_ref"] = round(len(unions[f]) / len(unions[REF]), 4)
            rows[f] = row
        out[world] = rows
    return out, cov


def per_edge_price(rows, kind):
    """Edges the NER slot adds over its world's union without it, per slot it adds (None when it adds none)."""
    out = {}
    for f, r in rows.items():
        if f == "none":
            continue
        add = r[kind]["added_over_none"] if "added_over_none" in r[kind] else r[kind]["linked"] - rows["none"][kind]["linked"]
        out[f] = round(r["ner_new_edges"] / add, 1) if add > 0 else None
    return out


def marginal_price(rows, kind, f, g):
    """New pairs form f has over form g's (both in one world), per slot f links that g does not count (None when it links no more)."""
    add = rows[f][kind]["linked"] - rows[g][kind]["linked"]
    return round((rows[f]["ner_new_edges"] - rows[g]["ner_new_edges"]) / add, 1) if add > 0 else None


# ── Part B: MuSiQue's chains, the bridge name ───────────────────────────────


def part_b(ch, per, vocab_df, sets):
    """Per chain slot: does the rule find the bridge name (the earlier sub-question's answer, read only here, as a diagnostic) in both passages, and at
    what df; and which unions link the slot."""
    rows = []
    for _qi, a, b, ans in ch:
        key = U.norm_key(ans.split())
        ina, inb = key in per[a], key in per[b]
        rows.append({"a": a, "b": b, "key": key, "both": ina and inb, "one": ina != inb, "df": vocab_df.get(key, 0),
                     **{f: ((a, b) in sets[f]) for f in sets}})
    return rows


def band(df):
    return "df 2-25" if df <= 25 else ("df 26-50" if df <= 50 else ("df 51-100" if df <= 100 else "df 101+"))


def part_b_tables(rows):
    n = len(rows)
    t = {"slots": n, "bridge_in_both": sum(r["both"] for r in rows), "bridge_in_one": sum(r["one"] for r in rows),
         "bridge_in_neither": sum((not r["both"]) and (not r["one"]) for r in rows)}
    miss = [r for r in rows if not r["ref"]]
    t["unreached_by_ref"] = len(miss)
    t["unreached_ref_bridge_in_both"] = sum(r["both"] for r in miss)
    t["unreached_ref_by_band"] = dict(Counter(band(r["df"]) for r in miss if r["both"]))
    t["unreached_ref_and_not_both"] = sum(not r["both"] for r in miss)
    t["both_by_band_all"] = dict(Counter(band(r["df"]) for r in rows if r["both"]))
    return t


# ── Part C: what spaCy's family links and the rules do not ──────────────────


def part_c(slots_kind, sets, per, titles, limit=12):
    """For one slot kind: the slots the reference union links and the candidate union does not (loss), the reverse (gain), and a few loss examples with the
    names the two passages share by the rule."""
    ref, cand = sets["ref"], sets["cand"]
    loss = [s for s in slots_kind if (s[1], s[2]) in ref and (s[1], s[2]) not in cand]
    gain = [s for s in slots_kind if (s[1], s[2]) in cand and (s[1], s[2]) not in ref]
    ex = []
    for s in loss[:limit]:
        a, b = s[1], s[2]
        ex.append({"a": titles[a], "b": titles[b], "shared_rule_names": sorted(per[a] & per[b])[:6]})
    return {"loss": len(loss), "gain": len(gain), "loss_with_a_shared_rule_name": sum(bool(per[s[1]] & per[s[2]]) for s in loss), "examples": ex}


# ── the run ─────────────────────────────────────────────────────────────────


def run_setting(d):
    t0 = time.time()
    corpus, queries, fam = load(d)
    n = len(corpus)
    nq = len(queries)
    gp = gold_slots(queries)
    slots = {"GP": gp}
    info = {"nodes": n, "questions": nq, "gp_slots": len(gp)}
    ch, chinfo = (None, None)
    if d == "musique":
        ch, chinfo = chain_slots(d, corpus, queries)
        slots["CH"] = [(q, a, b) for q, a, b, _x in ch]
        if (chinfo["slots"], chinfo["distinct_pairs"]) != CLAIMED[d]:
            raise SystemExit(f"gate G1: {d} has {chinfo['slots']} chain slots and {chinfo['distinct_pairs']} distinct ORC pairs, not {CLAIMED[d]}")
        cov = chinfo["coverage"]["graphs"]
        mine = {"S (b1, title mentions)": fam["S"], "R (b1w, W2_S rule)": fam["R"], "N (NER)": fam["N_sp"], "K (kNN)": fam["K"]}
        orc = {(a, b) for _q, a, b, _x in ch}
        for k, e in mine.items():                       # G2: oraclepairs' filed coverage counts
            if len(orc & e) != cov[k]["ORC_linked"]:
                raise SystemExit(f"gate G2: {d} {k}: {len(orc & e)} ORC pairs linked, oraclepairs says {cov[k]['ORC_linked']}")
        info["chain"] = {k: v for k, v in chinfo.items() if k != "coverage"}
    passes = lambda: ((c["title"], c["text"] if zu2b.is_descriptive(c["title"], c["text"]) else None) for c in corpus)  # noqa: E731
    per, src, grp, titled, vocab = names_table(passes, n)
    ga, gu = U.edges(*U.build_names(passes, n)[:3], n)                      # G4: U2r's own build gives the same CN_all / CN_u pairs
    rules = rule_families(src, grp, titled, n)
    if pair_set(ga[0], ga[1]) != pair_set(*rules["CN_all"][:2]) or pair_set(gu[0], gu[1]) != pair_set(*rules["CN_u"][:2]):
        raise SystemExit(f"gate G4: {d}: the names here are not U2r's")
    if not (pair_set(*rules["CN_all"][:2]) <= pair_set(*rules["CN_w"][:2]) and pair_set(*rules["CN_u"][:2]) <= pair_set(*rules["CN_wu"][:2])):
        raise SystemExit(f"gate G3: {d}: CN_w does not contain CN_all")
    info["names"] = {"distinct": int(titled.size), "mentions": int(src.size), "titled": int(titled.sum()),
                     "df_2_25": int(((np.bincount(grp) >= 2) & (np.bincount(grp) <= CAP)).sum()),
                     "df_over_25": int((np.bincount(grp) > CAP).sum())}
    info["spacy_graph"] = {"N_sp_pairs": len(fam["N_sp"]), "S_pairs": len(fam["S"]), "R_pairs": len(fam["R"]), "K_pairs": len(fam["K"])}
    a_tab, cov = part_a(slots, nq, fam, rules, {"n": n})
    res = {"info": info, "partA": a_tab}
    res["price"] = {w: {kind: per_edge_price(a_tab[w], kind) for kind in slots} for w in a_tab}
    res["marginal_price"] = {w: {kind: {"CN_all_over_none": marginal_price(a_tab[w], kind, "CN_all", "none"),
                                        "CN_w_over_CN_all": marginal_price(a_tab[w], kind, "CN_w", "CN_all"),
                                        "CN_wu_over_CN_u": marginal_price(a_tab[w], kind, "CN_wu", "CN_u"),
                                        "N_sp_over_none": marginal_price(a_tab[w], kind, "N_sp", "none")} for kind in slots} for w in a_tab}
    # Part B (MuSiQue) and Part C (every setting): the R-world union with spaCy's family is the reference, with CN_w the candidate
    base_r = fam["R"] | fam["K"]
    ref_set = base_r | fam["N_sp"]
    cand_set = base_r | pair_set(*rules["CN_w"][:2])
    cn_all_set = base_r | pair_set(*rules["CN_all"][:2])
    if ch is not None:
        dfv = np.bincount(grp, minlength=len(vocab))
        vdf = {k: int(dfv[v]) for k, v in vocab.items()}
        rows = part_b(ch, per, vdf, {"ref": ref_set, "cn_all": cn_all_set, "cn_w": cand_set})
        res["partB"] = part_b_tables(rows)
        res["partB"]["linked"] = {f: sum(r[f] for r in rows) for f in ("ref", "cn_all", "cn_w")}
        res["partB"]["bridge_in_both_by_band"] = res["partB"]["both_by_band_all"]
        res["partB"]["ceiling_any_df"] = sum(r["both"] for r in rows)
        res["partB"]["ceiling_df_le_100"] = sum(r["both"] and r["df"] <= 100 for r in rows)
    titles = [c["title"] for c in corpus]
    res["partC"] = {kind: part_c(sl, {"ref": ref_set, "cand": cand_set}, per, titles) for kind, sl in slots.items()}
    od = OUT / d
    od.mkdir(parents=True, exist_ok=True)
    arr = {}
    for k, (a, b, w) in rules.items():
        arr[f"{k}_src"], arr[f"{k}_dst"], arr[f"{k}_w"] = a.astype(np.int32), b.astype(np.int32), w.astype(np.float32)
    np.savez_compressed(od / "rules.npz", **arr)
    res["rules_npz_sha256"] = sha_file(od / "rules.npz")
    res["seconds"] = round(time.time() - t0, 1)
    log(f"{d}: done in {res['seconds']}s")
    return res


def render(res):
    L = [f"# AP1 report ({DECLARED}) - graph level, no model", "",
         "Share of slots linked by the union of (structural family, K, NER form). Difference = form minus N_sp, paired over questions, 95% interval of "
         f"{BOOT} question resamples (seed {SEED}). Calls: ABOVE / LEVEL / BELOW by whether the interval clears 0.", ""]
    for d, r in res["settings"].items():
        i = r["info"]
        L += [f"## {d}", "", f"nodes {i['nodes']}, questions {i['questions']}; names {i['names']['distinct']}, mentions {i['names']['mentions']}, "
              f"df 2-25 {i['names']['df_2_25']}, df > 25 {i['names']['df_over_25']}; pairs: " + ", ".join(f"{k} {v}" for k, v in i["spacy_graph"].items()), ""]
        for world, rows in r["partA"].items():
            kinds = [k for k in ("CH", "GP") if k in rows["none"]]
            L += [f"### {d}, world {world} (union of {world} + K + the NER form)", "",
                  "| NER form | pairs | NER pairs | new pairs | " + " | ".join(f"{k} linked | {k} share | {k} diff vs N_sp [95%] call" for k in kinds) + " | edges / ref |",
                  "|---|---:|---:|---:|" + "---:|---:|:--|" * len(kinds) + "---:|"]
            for f, x in rows.items():
                cells = []
                for k in kinds:
                    v = x[k]
                    cells += [f"{v['linked']}/{v['slots']}", f"{v['share']:.2f}%",
                              f"{v['vs_N_sp']['diff']:+.2f} [{v['vs_N_sp']['lo']:+.2f}, {v['vs_N_sp']['hi']:+.2f}] {v['vs_N_sp']['call']}"]
                L.append(f"| {f} | {x['edges']} | {x['ner_edges']} | {x['ner_new_edges']} | " + " | ".join(cells) + f" | {x['edge_ratio_to_ref']:.3f} |")
            L.append("")
            for k in kinds:
                L.append(f"Price, {world}-world, {k}: new pairs per slot the NER slot adds over the union without it: " +
                         ", ".join(f"{f} {v}" for f, v in r["price"][world][k].items()))
                L.append(f"Marginal price, {world}-world, {k}: " + ", ".join(f"{f} {v}" for f, v in r["marginal_price"][world][k].items()))
                for f in ("CN_u", "CN_wu", "CN_w"):
                    v = rows[f][k]["vs_CN_all"]
                    L.append(f"- {f} minus CN_all, {k}: {v['diff']:+.2f} [{v['lo']:+.2f}, {v['hi']:+.2f}] {v['call']}")
            L.append("")
        if "partB" in r:
            b = r["partB"]
            L += [f"### {d}, Part B: the bridge name (labels read as a diagnostic only)", "",
                  f"{b['slots']} chain slots; the rule finds the bridge name in both passages in {b['bridge_in_both']}, in one in {b['bridge_in_one']}, in neither in "
                  f"{b['bridge_in_neither']}. Linked by R+N_sp+K: {b['linked']['ref']}; by R+CN_all+K: {b['linked']['cn_all']}; by R+CN_w+K: {b['linked']['cn_w']}. "
                  f"Ceiling of any shared-name rule (any df): {b['ceiling_any_df']}; at df <= 100: {b['ceiling_df_le_100']}.",
                  f"Slots R+N_sp+K misses: {b['unreached_by_ref']}; of those the bridge name is in both passages in {b['unreached_ref_bridge_in_both']} "
                  f"(by df band {b['unreached_ref_by_band']}), and not in both in {b['unreached_ref_and_not_both']}.", ""]
        for kind, c in r["partC"].items():
            L.append(f"### {d}, Part C, {kind}: slots linked by R+N_sp+K and not by R+CN_w+K (loss) {c['loss']}, the reverse (gain) {c['gain']}; "
                     f"loss slots whose passages share a rule name at any df: {c['loss_with_a_shared_rule_name']}")
            for e in c["examples"]:
                L.append(f"- {e['a']} | {e['b']} | shared: {e['shared_rule_names']}")
            L.append("")
    return "\n".join(L) + "\n"


def cmd_run():
    res = {"declared_in": DECLARED, "script_sha256": script_sha256(), "zu2r_script_sha256": U.script_sha256(),
           "zu2l_el_sha256": EL.script_sha256(), "boot": BOOT, "seed": SEED, "cap": CAP, "unit_weight": UNIT, "settings": {}}
    OUT.mkdir(parents=True, exist_ok=True)
    for d in SETTINGS:
        res["settings"][d] = run_setting(d)
    (OUT / "report.json").write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    (OUT / "report.md").write_text(render(res), encoding="utf-8", newline="\n")
    log(f"written {OUT / 'report.json'} and report.md")
    return 0


# ── the self-test ───────────────────────────────────────────────────────────


def selftest():
    # the rule edges on a toy corpus: 40 passages hold the two names "Alpha Land" and "Beta Land" (df 40 each: one name alone carries 1/40 < 1/25, both carry
    # 1/20 >= 1/25), 40 more hold only "Gamma Land"; 6 passages hold "Delta Land" (df 6, inside the cap)
    def text(i):
        parts = ["see also here."]
        if i < 40:
            parts.append("It met Alpha Land and Beta Land today.")
        elif i < 80:
            parts.append("It met Gamma Land today.")
        elif i < 86:
            parts.append("It met Delta Land today.")
        return " ".join(parts)

    n = 90
    toy = [{"title": f"T{i}", "text": text(i)} for i in range(n)]
    passes = lambda: ((c["title"], c["text"]) for c in toy)  # noqa: E731
    per, src, grp, titled, vocab = names_table(passes, n)
    assert {"alpha land", "beta land", "gamma land", "delta land"} <= set(vocab), sorted(vocab)
    fam = rule_families(src, grp, titled, n)
    cn_all = pair_set(*fam["CN_all"][:2])
    cn_w = pair_set(*fam["CN_w"][:2])
    assert len(cn_all) == 15, len(cn_all)                                        # Delta Land: C(6,2); nothing from the df-40 names
    assert len(cn_w) == 15 + 780, len(cn_w)                                      # plus the 780 pairs of the 40 holding both df-40 names
    assert cn_all <= cn_w and all(b < 40 for _a, b in cn_w - cn_all)
    assert not any(a >= 40 and a < 80 for a, _b in cn_w), "Gamma Land alone (df 40) must not link"
    assert min(fam["CN_w"][2]) >= UNIT - TOL, "every CN_w pair carries at least the weight of one name at the cap"
    ga, _gu = U.edges(*U.build_names(passes, n)[:3], n)
    assert pair_set(ga[0], ga[1]) == cn_all, "names_table must give U2r's CN_all"
    # reach, the paired interval and the calls
    slots = [(q, 2 * q, 2 * q + 1) for q in range(50)]
    E_all, E_none = {(a, b) for _q, a, b in slots}, set()
    c1, c0 = covered(slots, E_all), covered(slots, E_none)
    qi = [s[0] for s in slots]
    d, lo, hi = boot_diff(c1, c0, qi, 50)
    assert d == 1.0 and lo == 1.0 and hi == 1.0 and call(lo, hi) == "ABOVE"
    d, lo, hi = boot_diff(c1, c1, qi, 50)
    assert d == 0.0 and call(lo, hi) == "LEVEL"
    d, lo, hi = boot_diff(c0, c1, qi, 50)
    assert d == -1.0 and call(lo, hi) == "BELOW"
    half = np.arange(50) % 2 == 0
    d, lo, hi = boot_diff(half, np.zeros(50, bool), qi, 50)
    assert abs(d - 0.5) < 1e-12 and lo < 0.5 < hi and call(lo, hi) == "ABOVE"
    assert boot_diff(half, ~half, qi, 50) == boot_diff(half, ~half, qi, 50), "the bootstrap is seeded"
    # gold slots
    gs = gold_slots([{"golds": [3, 1, 2]}, {"golds": [5]}, {"golds": [7, 7, 9]}])
    assert gs == [(0, 1, 2), (0, 1, 3), (0, 2, 3), (2, 7, 9)], gs
    # the class bands
    assert [band(x) for x in (2, 25, 26, 50, 51, 100, 101)] == ["df 2-25", "df 2-25", "df 26-50", "df 26-50", "df 51-100", "df 51-100", "df 101+"]
    print("zap1 self-test: OK")
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "selftest":
        raise SystemExit(selftest())
    if cmd == "run":
        raise SystemExit(cmd_run())
    raise SystemExit(__doc__)
