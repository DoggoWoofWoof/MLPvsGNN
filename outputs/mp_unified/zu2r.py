"""U2r: names without a model, one entity rule in place of spaCy's NER (docs/U2R_NAME_RULE.md).

    python outputs/mp_unified/zu2r.py names --dataset D [--limit N]   # N > 0: a development run, writes nothing
    python outputs/mp_unified/zu2r.py grade --dataset D [--carve C]
    python outputs/mp_unified/zu2r.py choose
    python outputs/mp_unified/zu2r.py report
    python outputs/mp_unified/zu2r.py --selftest

A name is a run of capitalized tokens (zu2i's connectors inside, without the coordinators), found with the corpus's own
capitalization statistic (u1d's rule per lowercased token, sentence starts excluded). A name equal to a title's key is
titled: it has a node, and U2l's IS links to it. CN_u links the nodes that share an untitled name, CN_all the nodes that
share any name, both at NER's df cap [2, 25] with weight the sum of 1/df. The forms E1 (IS), N1 (IS + CN_u) and
N2 (IS + CN_all) are graded as U2l graded EL, with E3 (IS + spaCy's CO_s) beside.

Outputs only under outputs/u2r/. Arrays (npz) stay uncommitted. Nothing trains; no filed graph or record changes.
"""
import os
import sys

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402
from array import array  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (HERE, ROOT / "scripts", ROOT / "src", ROOT / "src" / "mp_retrieval"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import u1b  # noqa: E402
import u1d  # noqa: E402
import zu2b  # noqa: E402
import zu2i  # noqa: E402
import zu2l_el as EL  # noqa: E402

DECLARED = "docs/U2R_NAME_RULE.md"
OUT = ROOT / "outputs" / "u2r"
SIX = EL.SIX
FIVE = EL.FIVE
CARVE = EL.CARVE
PASSAGE = zu2b.PASSAGE
DF = EL.DF                                         # NER's cap [2, 25]
TIE = 0.002
WORD = re.compile(r"\w+(?:['’\-.]\w+)*")
CONN = frozenset(zu2i.CONNECT - {"and", "&", "y", "e"})
POSS = re.compile(r"['’]s$")
FORMS = {"E1": ("IS",), "N1": ("IS", "CN_u"), "N2": ("IS", "CN_all"), "E3": ("IS", "CO_s")}
ORDER = ("E1", "N1", "N2")                         # the choice's order; E3 is the reference
UNIONS = EL.UNIONS
GRAPHS = {**{f"{f}+{u}": FORMS[f] + UNIONS[u] for f in FORMS for u in UNIONS},
          "W2_S_NER_kNN": ("W2_S", "CO_s", "kNN"), "PKG_NER_kNN": ("PKG", "CO_s", "kNN")}
FILED = [g for g in GRAPHS if not g.startswith(("N1", "N2"))]   # must equal U2l's filed grade record
PICK = "R_kNN"
# gate 3: the scratch diagnostic's musique counts, read before the declaration
MUSIQUE = {"names": 392380, "mentions": 1027176, "cn_all_pairs": 931069, "cn_u_pairs": 766887}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def script_sha256():
    return hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def codes():
    el = {k: v for k, v in EL.codes().items() if k != "script_sha256"}
    return {"script_sha256": script_sha256(), "zu2l_el_sha256": EL.script_sha256(), **el}


# ── the names ───────────────────────────────────────────────────────────────


def tokens(text):
    """[(token, capitalized, starts a sentence, adjacent to the token before)]. A token ends in a word character, so
    the last non-space character before it is in its gap or ends the token before: u1d.sent_start, read from the gap."""
    out, prev = [], 0
    for m in WORD.finditer(text):
        s, w = m.start(), m.group()
        gap = text[prev:s]
        g = gap.rstrip()
        start = (g[-1] in u1d.SENT_END) if g else not out
        out.append((w, w[0].isupper(), start, bool(out) and gap != "" and gap.isspace()))
        prev = m.end()
    return out


def count_stat(toks, cap, low):
    for w, c, st, _a in toks:
        if st:
            continue
        k = w.lower()
        if c:
            cap[k] = cap.get(k, 0) + 1
        elif w == k:
            low[k] = low.get(k, 0) + 1


def norm_key(ws):
    ws = [w.lower() for w in ws]
    if ws and ws[0] == "the":
        ws = ws[1:]
    if ws:
        ws[-1] = POSS.sub("", ws[-1])
    return " ".join(ws)


def names_of(toks, usual):
    out = set()
    i, T = 0, len(toks)
    while i < T:
        w, c, st, _a = toks[i]
        if not c or (st and not usual(w.lower())):
            i += 1
            continue
        run = [w]
        j = i + 1
        while j < T:
            wj, cj, _s, aj = toks[j]
            if not aj:
                break
            if cj:
                run.append(wj)
                j += 1
            elif wj.lower() in CONN and j + 1 < T and toks[j + 1][3] and toks[j + 1][1]:
                run += [wj, toks[j + 1][0]]
                j += 2
            else:
                break
        k = norm_key(run)
        if len(k) > 2 and (len(run) > 1 or usual(k)):
            out.add(k)
        i = j
    return out


def title_key_set(titles):
    ks = set()
    for t in titles:
        a = " ".join(str(t or "").replace("_", " ").split())
        for x in (a, u1d.PAREN.sub("", a)):
            k = norm_key(x.split())
            if k:
                ks.add(k)
    return ks


def stream_passages(served, d, limit=0):
    """(title, text or None for a node the scan does not read), streamed; the file's sha256 and line count checked
    against DATASET.json at the end of a full pass."""
    man = json.loads((served / d / "DATASET.json").read_text(encoding="utf-8"))
    want, n = man["nodes"]["sha256"], int(man["nodes"]["n"])
    h = hashlib.sha256()
    i = 0
    with open(served / d / "nodes.jsonl", "rb") as f:
        for raw in f:
            h.update(raw)
            rec = json.loads(raw)
            t = u1b.title_of(rec)
            x = rec["text"]
            yield t, (x if zu2b.is_descriptive(t, x) else None)
            i += 1
            if limit and i >= limit:
                return
    if i != n or h.hexdigest() != want:
        raise SystemExit(f"{d}: nodes.jsonl is not DATASET.json's ({i} lines, sha256 {h.hexdigest()[:12]})")


def corpus(served, d, limit=0):
    """A callable giving a fresh pass of (title, text or None), the node count and the nodes' sha256."""
    if d in PASSAGE:
        man = json.loads((served / d / "DATASET.json").read_text(encoding="utf-8"))
        n = int(man["nodes"]["n"])
        return (lambda: stream_passages(served, d, limit)), (min(n, limit) if limit else n), man["nodes"]["sha256"], 0
    titles, texts, n, nsha, n_cvt = EL.kb_texts(served, d)   # a KB: names as texts, CVTs blanked (U2l, section 2)
    if limit:
        titles, texts, n = titles[:limit], texts[:limit], min(n, limit)
    return (lambda: ((t, x if x else None) for t, x in zip(titles, texts))), n, nsha, n_cvt


def build_names(passes, n):
    cap, low = {}, {}
    for _t, x in passes():
        if x is not None:
            count_stat(tokens(x), cap, low)

    def usual(k):
        return not (low.get(k, 0) > cap.get(k, 0))

    vocab, src, grp, titles = {}, array("i"), array("i"), []
    for i, (t, x) in enumerate(passes()):
        titles.append(t)
        if x is None:
            continue
        for k in names_of(tokens(x), usual):
            src.append(i)
            grp.append(vocab.setdefault(k, len(vocab)))
    if len(titles) != n:
        raise SystemExit(f"the second pass read {len(titles)} nodes, the first {n}")
    tk = title_key_set(titles)
    del titles
    titled = np.zeros(len(vocab), bool)
    for k, v in vocab.items():
        if k in tk:
            titled[v] = True
    del vocab, tk
    return np.asarray(src).astype(np.int64), np.asarray(grp).astype(np.int64), titled, len(cap) + len(low)


def edges(src, grp, titled, n):
    a = EL.co_mentions(src, grp, n)
    keep = ~titled[grp]
    u = EL.co_mentions(src[keep], grp[keep], n)
    return a, u


def cmd_names(a):
    d, lim = a.dataset, a.limit
    t0 = time.time()
    served, freeze = u1b.package_root(False)
    passes, n, nsha, n_cvt = corpus(served, d, lim)
    src, grp, titled, nstat = build_names(passes, n)
    df = np.bincount(grp, minlength=titled.size)
    t1 = time.time()
    (as_, ad, aw), (us, ud, uw) = edges(src, grp, titled, n)
    rec = {"dataset": d, "declared_in": DECLARED, "freeze_RECORD_SHA256": freeze, "nodes": int(n),
           "nodes_sha256": nsha, "cvt_mediators": int(n_cvt), "statistic_tokens": int(nstat),
           "names": int(titled.size), "mentions": int(src.size), "names_df_2_25": int(((df >= 2) & (df <= 25)).sum()),
           "titled_names": int(titled.sum()), "titled_mentions": int(titled[grp].sum()),
           "cn_all_pairs": int(as_.size), "cn_u_pairs": int(us.size),
           "secs": {"names": round(t1 - t0, 1), "edges": round(time.time() - t1, 1)}, **codes()}
    log(f"{d}: {rec['names']:,} names, {rec['mentions']:,} mentions, {rec['titled_names']:,} titled; CN_all "
        f"{rec['cn_all_pairs']:,} pairs, CN_u {rec['cn_u_pairs']:,} ({time.time() - t0:.0f}s)")
    if d == "musique" and not lim:
        got = {k: rec[k] for k in MUSIQUE}
        rec["gate3"] = "PASS" if got == MUSIQUE else "FAIL"
        if got != MUSIQUE:
            raise SystemExit(f"gate 3 FAIL: musique's counts {got} are not the diagnostic's {MUSIQUE}")
    if lim:
        log(f"{d}: development run on the first {lim} nodes, nothing written")
        return 0
    od = OUT / d
    od.mkdir(parents=True, exist_ok=True)
    rec["names_npz_sha256"] = zu2b.save_npz(
        od / "names.npz", cn_all_src=as_.astype(np.int32), cn_all_dst=ad.astype(np.int32),
        cn_all_w=aw.astype(np.float32), cn_u_src=us.astype(np.int32), cn_u_dst=ud.astype(np.int32),
        cn_u_w=uw.astype(np.float32))
    rec["seconds"] = round(time.time() - t0, 1)
    EL.write_json(od / "names.json", rec)
    log(f"{d}: written to {od / 'names.json'} ({rec['seconds']}s)")
    return 0


# ── the grading: U2l's, with the name edges beside its families ──────────────


def name_families(d, n, F):
    r = json.loads((OUT / d / "names.json").read_text(encoding="utf-8"))
    if EL.sha_file(OUT / d / "names.npz") != r["names_npz_sha256"]:
        raise SystemExit(f"{d}: names.npz is not its record's")
    if r["script_sha256"] != script_sha256() or r["nodes"] != n:
        raise SystemExit(f"{d}: names.json is not this code's or this dataset's")
    with np.load(OUT / d / "names.npz") as z:
        F["CN_all"] = zu2b.undirected_csr(n, z["cn_all_src"], z["cn_all_dst"])
        F["CN_u"] = zu2b.undirected_csr(n, z["cn_u_src"], z["cn_u_dst"])
    return r["names_npz_sha256"]


def cmd_grade(a):
    d = a.dataset
    carve = a.carve or CARVE[d]
    t0 = time.time()
    served, freeze = u1b.package_root(False)
    qrec, q, _base, _seeds, golds, B = EL.questions(d, carve)
    n = int(json.loads((EL.OUT / d / "build.json").read_text(encoding="utf-8"))["nodes"])
    F, shas = EL.families(served, d, n)
    shas["names_npz_sha256"] = name_families(d, n, F)
    D = u1d.Dense(served, d, n)
    t1 = time.time()
    res = EL.reach_unions(F, GRAPHS, B, golds, q["qemb"], D.get)
    for g in res:
        res[g]["choice_score"] = round(zu2b.choice_score(res[g]["curve"]), 4)
        res[g]["families"] = list(GRAPHS[g])
    filed = json.loads((EL.OUT / d / f"grade_{carve}.json").read_text(encoding="utf-8"))
    if filed["questions_npz_sha256"] != qrec["carves"][carve]["npz_sha256"]:
        raise SystemExit(f"{d}/{carve}: U2l's grade read other questions")
    mism = [g for g in FILED if any(res[g][k] != filed["graphs"][g][k]
                                    for k in ("curve", "all_neighbours", "choice_score", "B"))]
    gate2 = {"status": "PASS" if not mism else "FAIL", "reproduced": [g for g in FILED if g not in mism],
             "differs": mism, "u2l_grade_sha256": EL.sha_file(EL.OUT / d / f"grade_{carve}.json")}
    out = {"dataset": d, "carve": carve, "declared_in": DECLARED, "freeze_RECORD_SHA256": freeze, "gate2": gate2,
           "questions": len(golds), "questions_npz_sha256": qrec["carves"][carve]["npz_sha256"],
           "dense_manifest_sha256": D.man_sha, **shas, "family_neighbours": {k: int(v[1].size) for k, v in F.items()},
           "graphs": res, "secs": {"load": round(t1 - t0, 1), "reach": round(time.time() - t1, 1)}, **codes()}
    if mism:
        EL.write_json(OUT / d / f"refused_{carve}.json", out)
        raise SystemExit(f"{d}/{carve}: gate 2 FAIL, U2l's filed scores not reproduced for {mism}")
    EL.write_json(OUT / d / f"grade_{carve}.json", out)
    log(f"{d}/{carve}: graded {len(res)} graphs, gate 2 PASS; " + ", ".join(
        f"{f} {res[f + '+' + PICK]['choice_score']}" for f in FORMS) + f" ({time.time() - t0:.0f}s)")
    return 0


# ── the choice ───────────────────────────────────────────────────────────────


def load_grades():
    out = {}
    for d in SIX:
        p = OUT / d / f"grade_{CARVE[d]}.json"
        if p.exists():
            r = json.loads(p.read_text(encoding="utf-8"))
            if r["script_sha256"] == script_sha256() and r["gate2"]["status"] == "PASS":
                out[d] = r
    return out


def choose(scores):
    """scores: {form: {dataset: three-family score}} over FIVE (+ webqsp beside). Returns the pick and its reads."""
    mean = {f: round(float(np.mean([scores[f][d] for d in FIVE])), 4) for f in FORMS}
    best = max(mean[f] for f in ORDER)
    pick = next(f for f in ORDER if mean[f] >= best - TIE - 1e-12)
    below = [d for d in FIVE if scores[pick][d] < scores["E3"][d] - TIE - 1e-12]
    claim = "NER_MATCHED_ALL" if not below else "NER_MATCHED_NOT_ALL"
    return pick, mean, claim, below


def cmd_choose(_a):
    g = load_grades()
    missing = [d for d in SIX if d not in g]
    if missing:
        rec = {"declared_in": DECLARED, "outcome": "REFUSED", "missing": missing, **codes()}
        EL.write_json(OUT / "choice.json", rec)
        log(f"REFUSED: no passing grade for {missing}")
        return 1
    for d in SIX:
        nr = json.loads((OUT / d / "names.json").read_text(encoding="utf-8"))
        if g[d]["names_npz_sha256"] != nr["names_npz_sha256"]:
            raise SystemExit(f"{d}: the grade did not read the filed names")
    scores = {f: {d: g[d]["graphs"][f"{f}+{PICK}"]["choice_score"] for d in SIX} for f in FORMS}
    pick, mean, claim, below = choose(scores)
    rec = {"declared_in": DECLARED, "outcome": f"CHOSEN {pick}", "pick": pick, "mean_five": mean,
           "scores": scores, "claim": claim, "below_e3": below,
           "versus_e3": {d: round(scores[pick][d] - scores["E3"][d], 4) for d in SIX},
           "versus_e1": {d: round(scores[pick][d] - scores["E1"][d], 4) for d in SIX},
           "beside": {d: {k: g[d]["graphs"][k]["choice_score"] for k in GRAPHS} for d in SIX},
           "all_neighbours": {d: {f: g[d]["graphs"][f"{f}+{PICK}"]["all_neighbours"] for f in FORMS} for d in SIX},
           "names": {d: {k: json.loads((OUT / d / "names.json").read_text(encoding="utf-8"))[k]
                         for k in ("names", "mentions", "titled_names", "cn_u_pairs", "cn_all_pairs")} for d in SIX},
           "grades": {d: EL.sha_file(OUT / d / f"grade_{CARVE[d]}.json") for d in SIX}, **codes()}
    EL.write_json(OUT / "choice.json", rec)
    log(f"CHOSEN {pick}: means {mean}; {claim} {below}")
    return 0


def cmd_report(_a):
    c = json.loads((OUT / "choice.json").read_text(encoding="utf-8"))
    L = [f"# U2r: {c['outcome']}", "", f"Declared in {c['declared_in']}. The claim read: {c['claim']}"
         + (f" (below E3: {', '.join(c['below_e3'])})" if c["below_e3"] else "") + ".", "",
         "## The three-family scores (s1sel; webqsp is s1eval, beside)", "",
         "| form | " + " | ".join(SIX) + " | mean of five |", "|---|" + "---:|" * (len(SIX) + 1)]
    for f in FORMS:
        L.append(f"| {f} | " + " | ".join(f"{c['scores'][f][d]:.4f}" for d in SIX) + f" | {c['mean_five'][f]:.4f} |")
    L += ["", "## The names", "", "| dataset | names | mentions | titled | CN_u pairs | CN_all pairs |",
          "|---|---:|---:|---:|---:|---:|"]
    for d, x in c["names"].items():
        L.append(f"| {d} | {x['names']:,} | {x['mentions']:,} | {x['titled_names']:,} | {x['cn_u_pairs']:,} | "
                 f"{x['cn_all_pairs']:,} |")
    (OUT / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")
    log(f"written {OUT / 'report.md'}")
    return 0


# ── selftest ─────────────────────────────────────────────────────────────────


def _selftest(_a=None):
    # the sentence start read from the gap equals u1d.sent_start at every token
    texts = ['"The Beatles" were an English rock band; In 1990, Leeds United won. ', "  He met Paris, Texas officials!",
             "Apple sued apple growers. The apple fell.", "x", "", "...Then: Rome? yes 'Milan' (Turin) said"]
    for x in texts:
        for m, t in zip(WORD.finditer(x), tokens(x)):
            assert t[2] == u1d.sent_start(x, m.start()), (x, m.group())
    # the statistic, by hand
    cap, low = {}, {}
    count_stat(tokens("Apple sued apple growers. The apple fell. Mid Apple and iPhone"), cap, low)
    assert cap == {"apple": 1}, cap                     # "Mid" opens a sentence; "iPhone" is neither
    assert low == {"sued": 1, "apple": 2, "growers": 1, "fell": 1, "and": 1}, low
    # names, by hand
    low_ = {"the": 50, "in": 40, "however": 9, "he": 30, "we": 20, "church": 3, "she": 7}
    cap_ = {"church": 1}

    def usual(k):
        return not (low_.get(k, 0) > cap_.get(k, 0))

    cases = [
        ("The Beatles were an English rock band formed in Liverpool in 1960.", {"beatles", "english", "liverpool"}),
        ("In 1990, Leeds United won against Manchester United at Elland Road.",
         {"leeds united", "manchester united", "elland road"}),
        ("He studied at the University of Oxford and joined the Bank of England, where Obama's visit was noted.",
         {"university of oxford", "bank of england", "obama"}),
        ("France and Germany signed it. However, John Smith said the Treaty of Rome mattered. The band toured.",
         {"france", "germany", "john smith", "treaty of rome"}),
        ("We met Paris, Texas officials.", {"paris", "texas"}),
        ("She said the Church of England and the Church agreed with The Who.", {"church of england", "who"}),
        ("Ludwig van Beethoven met Al Capone at Le Mans of.", {"ludwig van beethoven", "al capone", "le mans"}),
        ("Mr. X met Jo", set()),
        ("A B. Co", {"a b"}),
    ]
    for x, want in cases:
        got = names_of(tokens(x), usual)
        assert got == want, (x, got, want)
    assert title_key_set(["The_Beatles", "Paris (France)", "Leeds United F.C.", None]) == \
        {"beatles", "paris (france)", "paris", "leeds united f.c."}
    # the edges against a brute force
    rng = np.random.RandomState(0)
    for trial in range(20):
        n, ng = int(rng.randint(2, 40)), int(rng.randint(1, 30))
        m = int(rng.randint(0, 200))
        src, grp = rng.randint(0, n, m).astype(np.int64), rng.randint(0, ng, m).astype(np.int64)
        titled = rng.rand(ng) < 0.4
        (as_, ad, aw), (us, ud, uw) = edges(src, grp, titled, n)
        for (s_, d_, w_), keep in (((as_, ad, aw), np.ones(ng, bool)), ((us, ud, uw), ~titled)):
            holders = {}
            for u, e in zip(src.tolist(), grp.tolist()):
                if keep[e]:
                    holders.setdefault(e, set()).add(u)
            want = {}
            for e, hs in holders.items():
                if DF[0] <= len(hs) <= DF[1]:
                    hs = sorted(hs)
                    for i in range(len(hs)):
                        for j in range(i + 1, len(hs)):
                            want[(hs[i], hs[j])] = want.get((hs[i], hs[j]), 0.0) + 1.0 / len(hs)
            got = {(int(x), int(y)): float(z) for x, y, z in zip(s_, d_, w_)}
            assert set(got) == set(want), trial
            assert all(abs(got[k] - want[k]) < 1e-12 for k in want), trial
    # the choice
    sc = {f: {d: 0.7 for d in SIX} for f in FORMS}
    assert choose(sc)[0] == "E1" and choose(sc)[2] == "NER_MATCHED_ALL"
    sc["N1"] = {d: 0.71 for d in SIX}
    sc["N2"] = {d: 0.7115 for d in SIX}
    assert choose(sc)[0] == "N1"
    sc["N1"]["musique"] = 0.69
    sc["E3"]["musique"] = 0.7
    p, _m, cl, below = choose(sc)
    assert p == "N2" and cl == "NER_MATCHED_ALL", (p, cl, below)
    sc["N2"]["squad"] = 0.697
    p, _m, cl, below = choose(sc)
    assert cl == "NER_MATCHED_NOT_ALL" and below == ["squad"], (p, cl, below)
    print("selftest OK")
    return 0


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    if argv and argv[0] == "--selftest":
        return _selftest()
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("names")
    p.add_argument("--dataset", required=True, choices=SIX)
    p.add_argument("--limit", type=int, default=0)
    p.set_defaults(fn=cmd_names)
    p = sub.add_parser("grade")
    p.add_argument("--dataset", required=True, choices=SIX)
    p.add_argument("--carve", default=None)
    p.set_defaults(fn=cmd_grade)
    sub.add_parser("choose").set_defaults(fn=cmd_choose)
    sub.add_parser("report").set_defaults(fn=cmd_report)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
