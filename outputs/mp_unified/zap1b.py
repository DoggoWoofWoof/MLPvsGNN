"""AP1b: sub-span names and the passage's own title (docs/AP1B_SUBSPAN_NAMES.md), a graph-level read on B1's three passage settings.

U2r's capitalized runs are kept; each run's proper sub-spans (end-anchored, infix, one-word) and the passage's own title are read as further names of the
passage, in four levels and with or without the title (eight forms F0..F3, F0T..F3T); each form's co-mention pairs (U2r's df cap [2, 25], weight sum 1/df) are
unioned with R (b1w's W2_S pairs) and K (b1's 3-NN) and graded by the share of a question's gold pairs (GP; MuSiQue's chain slots CH) the union links, with a
paired 95% interval over questions against CN_all (AP1's carried form) and N_sp (spaCy's family). No model reads a vector, nothing trains, no filed graph or
record changes. Labels (gold pairs, MuSiQue's decomposition answers) are the grading and Part B's diagnostic only; no rule reads one.

    python outputs/mp_unified/zap1b.py run          # three settings -> outputs/ap1b/report.json, report.md, <setting>/levels.npz
    python outputs/mp_unified/zap1b.py selftest
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

import zap1 as Z  # noqa: E402
import zu2b  # noqa: E402
import zu2l_el as EL  # noqa: E402
import zu2r as U  # noqa: E402

DECLARED = "docs/AP1B_SUBSPAN_NAMES.md"
AP1 = ROOT / "outputs" / "ap1"
OUT = ROOT / "outputs" / "ap1b"
SETTINGS = Z.SETTINGS
CAP, LO = Z.CAP, Z.LO
FORMS = ("F0", "F1", "F2", "F3", "F0T", "F1T", "F2T", "F3T")
ORDER = ("F1", "F2", "F3")                       # S1's order of tries; the T counterpart of the carried form is tried last
ALL_FORMS = ("none", "N_sp") + FORMS
BOOT, SEED = Z.BOOT, Z.SEED


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def script_sha256():
    return hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


# ── the names ───────────────────────────────────────────────────────────────


def title_stripped(t):
    """The title as zu2b's kind-1 form reads it: underscores as spaces, a trailing parenthetical removed, the comma tail removed."""
    t = " ".join(str(t or "").replace("_", " ").split())
    t = zu2b.PAREN.sub("", t).strip()
    return t.split(",")[0].strip()


def runs_of(tk, usual):
    """Every run U2r's names_of accepts, as its normalized key (the same loop, the keys listed instead of collected)."""
    out, i, T = [], 0, len(tk)
    while i < T:
        w, c, st, _a = tk[i]
        if not c or (st and not usual(w.lower())):
            i += 1
            continue
        run, j = [w], i + 1
        while j < T:
            wj, cj, _s, aj = tk[j]
            if not aj:
                break
            if cj:
                run.append(wj)
                j += 1
            elif wj.lower() in U.CONN and j + 1 < T and tk[j + 1][3] and tk[j + 1][1]:
                run += [wj, tk[j + 1][0]]
                j += 2
            else:
                break
        k = U.norm_key(run)
        if len(k) > 2 and (len(run) > 1 or usual(k)):
            out.append(k)
        i = j
    return out


def subs_of(key, usual):
    """The proper sub-spans of a run key with two or more words: {level: {sub-key}}. A sub-span's first and last word are not connectors; level 1 =
    end-anchored with two or more words, level 2 = infix with two or more words, level 3 = one word (more than two characters and `usual`)."""
    w = key.split(" ")
    L = len(w)
    out = {1: set(), 2: set(), 3: set()}
    if L < 2:
        return out
    for i in range(L):
        if w[i] in U.CONN:
            continue
        for j in range(i + 1, L + 1):
            if (i == 0 and j == L) or w[j - 1] in U.CONN:
                continue
            s = " ".join(w[i:j])
            m = j - i
            if len(s) <= 2 or (m == 1 and not usual(s)):
                continue
            out[3 if m == 1 else (1 if (i == 0 or j == L) else 2)].add(s)
    return out


def merge_subs(runs, usual):
    """All levels of every run of a passage, kept apart per level."""
    out = {1: set(), 2: set(), 3: set()}
    for k in runs:
        s = subs_of(k, usual)
        for lv in out:
            out[lv] |= s[lv]
    return out


def name_tables(passes, n):
    """Per passage: its runs, their sub-spans by level, its title's runs and their sub-spans by level (empty for a passage whose text is not read)."""
    cap, low, toks, titles = {}, {}, [], []
    for t, x in passes():
        titles.append(t)
        tk = U.tokens(x) if x is not None else None
        toks.append(tk)
        if tk is not None:
            U.count_stat(tk, cap, low)

    def usual(k):
        return not (low.get(k, 0) > cap.get(k, 0))

    if len(titles) != n:
        raise SystemExit(f"{len(titles)} passages, expected {n}")
    runs, subs, truns, tsubs = [], [], [], []
    for i, tk in enumerate(toks):
        r = runs_of(tk, usual) if tk is not None else []
        runs.append(r)
        subs.append(merge_subs(r, usual))
        tr = runs_of(U.tokens(title_stripped(titles[i])), usual) if tk is not None else []
        truns.append(tr)
        tsubs.append(merge_subs(tr, usual))
    return runs, subs, truns, tsubs


def entries(runs, subs, truns, tsubs, level, with_title, vocab):
    """The (passage, key) entries of one level, with or without the title's names."""
    src, grp = [], []
    for i in range(len(runs)):
        s = set(runs[i])
        for lv in range(1, level + 1):
            s |= subs[i][lv]
        if with_title:
            s |= set(truns[i])
            for lv in range(1, level + 1):
                s |= tsubs[i][lv]
        for k in s:
            src.append(i)
            grp.append(vocab.setdefault(k, len(vocab)))
    return np.asarray(src, np.int64), np.asarray(grp, np.int64)


def level_pairs(runs, subs, truns, tsubs, n):
    """{(level, with_title): (pair keys a*n+b ascending, weights)} for the eight levels."""
    vocab, out = {}, {}
    for t in (False, True):
        for lv in range(4):
            src, grp = entries(runs, subs, truns, tsubs, lv, t, vocab)
            a, b, w = EL.co_mentions(src, grp, n, lo=LO, hi=CAP)
            out[(lv, t)] = (a * n + b, w)
    return out


def build_forms(lp):
    """F0..F3 and F0T..F3T as sorted pair-key arrays: each form contains the one before it (section 3)."""
    F, prev = {}, np.zeros(0, np.int64)
    for lv in range(4):
        prev = np.union1d(prev, lp[(lv, False)][0])
        F[f"F{lv}"] = prev
    prev_t = np.zeros(0, np.int64)
    for lv in range(4):
        prev_t = np.union1d(np.union1d(F[f"F{lv}"], prev_t), lp[(lv, True)][0])
        F[f"F{lv}T"] = prev_t
    return F


# ── slots, coverage, grading ────────────────────────────────────────────────


def keys_of_set(pairs, n):
    arr = np.fromiter((a * n + b for a, b in pairs), np.int64, count=len(pairs))
    if not all(a < b for a, b in pairs):
        raise SystemExit("a filed pair is not ordered a < b")
    return np.unique(arr)


def slot_keys(slots, n):
    return np.fromiter((s[1] * n + s[2] for s in slots), np.int64, count=len(slots))


def covered_np(sk, union):
    return np.isin(sk, union)


def pct(x):
    return Z.pct(x)


def grade(c, ref, qidx, nq):
    d, lo, hi = Z.boot_diff(c, ref, qidx, nq)
    return {"diff": pct(d), "lo": pct(lo), "hi": pct(hi), "call": Z.call(lo, hi)}


def marginal(union_f, union_g, linked_f, linked_g):
    add = linked_f - linked_g
    return round((union_f - union_g) / add, 1) if add > 0 else None


def decide(cov, size, nq, price_ref):
    """S1: walk F1, F2, F3 and then the T counterpart of the carried form; each step against the carried form in the R-world: (a) CH call ABOVE, (b) GP call
    not BELOW on any setting, (c) the price over the carried form on CH at most N_sp's. cov[d][kind][form] = bool array, cov[d][kind]['_q'] = question index,
    size[d][form] = union pairs; returns (carried form, the log of tries)."""
    carried, tries = "F0", []

    def step(f, g):
        det = {"form": f, "against": g}
        ch = cov["musique"]["CH"]
        d, lo, hi = Z.boot_diff(ch[f], ch[g], ch["_q"], nq["musique"])
        det["a_CH_diff"], det["a_CH_lo"], det["a_CH_hi"], det["a"] = pct(d), pct(lo), pct(hi), Z.call(lo, hi) == "ABOVE"
        gp = {}
        for s in cov:
            g_ = cov[s]["GP"]
            dd, ll, hh = Z.boot_diff(g_[f], g_[g], g_["_q"], nq[s])
            gp[s] = {"diff": pct(dd), "lo": pct(ll), "hi": pct(hh), "call": Z.call(ll, hh)}
        det["b_GP"], det["b"] = gp, all(v["call"] != "BELOW" for v in gp.values())
        add = int(ch[f].sum() - ch[g].sum())
        det["added_CH_slots"] = add
        det["c_price"] = marginal(size["musique"][f], size["musique"][g], int(ch[f].sum()), int(ch[g].sum()))
        det["c"] = det["c_price"] is not None and det["c_price"] <= price_ref
        det["ok"] = det["a"] and det["b"] and det["c"]
        return det

    for f in ORDER:
        det = step(f, carried)
        tries.append(det)
        if det["ok"]:
            carried = f
    det = step(carried + "T", carried)
    tries.append(det)
    if det["ok"]:
        carried = carried + "T"
    return carried, tries


# ── Part B: what the carried form leaves ────────────────────────────────────


def remaining(ch, unlinked, per_sets, subdf):
    """For MuSiQue's chain slots the carried form's union leaves unlinked: the bridge name (the answer, read as a label) as a name, a sub-span of a name, or
    neither, in each passage, and the df of the key over names and sub-spans; classes as AP1 section 11's scratch read."""
    c = Counter()
    for (qi, a, b, ans), un in zip(ch, unlinked):
        if not un:
            continue
        key = U.norm_key(ans.split())
        st = []
        for node in (a, b):
            st.append("name" if key in per_sets[node][0] else ("sub" if key in per_sets[node][1] else "none"))
        if "none" in st:
            c["a side has neither"] += 1
        else:
            d = subdf.get(key, 0)
            c["both name-or-sub, df > 25"] += d > Z.CAP
            c["both name-or-sub, df in [2, 25]"] += 2 <= d <= Z.CAP
            c["both name-or-sub, df < 2"] += d < 2
    return dict(c)


# ── the run ─────────────────────────────────────────────────────────────────


def read_ap1():
    return json.loads((AP1 / "report.json").read_text(encoding="utf-8"))


def run_setting(d, ap1):
    t0 = time.time()
    corpus, queries, fam = Z.load(d)
    n, nq = len(corpus), len(queries)
    slots = {"GP": Z.gold_slots(queries)}
    ch = None
    if d == "musique":
        ch, _chinfo = Z.chain_slots(d, corpus, queries)
        slots["CH"] = [(q, a, b) for q, a, b, _x in ch]
    passes = lambda: ((c["title"], c["text"] if zu2b.is_descriptive(c["title"], c["text"]) else None) for c in corpus)  # noqa: E731
    per, src, grp, titled, vocab = Z.names_table(passes, n)
    runs, subs, truns, tsubs = name_tables(passes, n)
    if any(set(runs[i]) != per[i] for i in range(n)):                                   # G5
        raise SystemExit(f"gate G5: {d}: the runs here are not U2r's names")
    lp = level_pairs(runs, subs, truns, tsubs, n)
    F = build_forms(lp)
    ap1_rules = np.load(AP1 / d / "rules.npz")
    if Z.sha_file(AP1 / d / "rules.npz") != ap1["settings"][d]["rules_npz_sha256"]:
        raise SystemExit(f"gate G1: {d}: AP1's rules.npz is not its report's")
    cn = np.unique(ap1_rules["CN_all_src"].astype(np.int64) * n + ap1_rules["CN_all_dst"].astype(np.int64))
    if not np.array_equal(cn, F["F0"]):                                                  # G1
        raise SystemExit(f"gate G1: {d}: F0 is not AP1's CN_all")
    order = ["F0", "F1", "F2", "F3"]
    for x, y in zip(order, order[1:]):                                                   # G2
        if not np.isin(F[x], F[y]).all():
            raise SystemExit(f"gate G2: {d}: {x} is not contained in {y}")
    for k in range(4):
        if not np.isin(F[f"F{k}"], F[f"F{k}T"]).all() or (k and not np.isin(F[f"F{k - 1}T"], F[f"F{k}T"]).all()):
            raise SystemExit(f"gate G2: {d}: F{k}T does not contain its predecessors")
    base = np.union1d(keys_of_set(fam["R"], n), keys_of_set(fam["K"], n))
    forms = {"none": np.zeros(0, np.int64), "N_sp": keys_of_set(fam["N_sp"], n), **F}
    unions = {f: np.union1d(base, forms[f]) for f in ALL_FORMS}
    cov = {}
    for kind, sl in slots.items():
        sk = slot_keys(sl, n)
        cov[kind] = {f: covered_np(sk, unions[f]) for f in ALL_FORMS}
        cov[kind]["_q"] = [s[0] for s in sl]
    for kind in slots:                                                                    # G4
        want = ap1["settings"][d]["partA"]["R"]["CN_all"][kind]["linked"]
        if int(cov[kind]["F0"].sum()) != want:
            raise SystemExit(f"gate G4: {d} {kind}: F0's union links {int(cov[kind]['F0'].sum())}, AP1 filed {want}")
    rows = {}
    for f in ALL_FORMS:
        row = {"form_pairs": int(forms[f].size), "union_pairs": int(unions[f].size), "new_over_none": int(unions[f].size - base.size)}
        for kind, sl in slots.items():
            c = cov[kind]
            row[kind] = {"slots": len(sl), "linked": int(c[f].sum()), "share": pct(c[f].mean()),
                         "vs_CN_all": grade(c[f], c["F0"], c["_q"], nq), "vs_N_sp": grade(c[f], c["N_sp"], c["_q"], nq)}
        rows[f] = row
    price = {}
    if d == "musique":
        c = cov["CH"]
        price["N_sp_over_none"] = marginal(unions["N_sp"].size, unions["none"].size, int(c["N_sp"].sum()), int(c["none"].sum()))
        if price["N_sp_over_none"] != ap1["settings"][d]["marginal_price"]["R"]["CH"]["N_sp_over_none"]:   # G3
            raise SystemExit(f"gate G3: {d}: N_sp's price is {price['N_sp_over_none']}, AP1 filed {ap1['settings'][d]['marginal_price']['R']['CH']['N_sp_over_none']}")
        for kind in slots:
            price[kind] = {f: {"over_F0": marginal(unions[f].size, unions["F0"].size, int(cov[kind][f].sum()), int(cov[kind]["F0"].sum())),
                               "over_none": marginal(unions[f].size, unions["none"].size, int(cov[kind][f].sum()), int(cov[kind]["none"].sum()))}
                           for f in FORMS}
    od = OUT / d
    od.mkdir(parents=True, exist_ok=True)
    arr = {}
    for (lv, t), (k, w) in lp.items():
        arr[f"L{lv}{'T' if t else ''}_src"], arr[f"L{lv}{'T' if t else ''}_dst"], arr[f"L{lv}{'T' if t else ''}_w"] = (k // n).astype(np.int32), (k % n).astype(np.int32), w.astype(np.float32)
    np.savez_compressed(od / "levels.npz", **arr)
    res = {"info": {"nodes": n, "questions": nq, "slots": {k: len(v) for k, v in slots.items()}, "base_pairs": int(base.size)},
           "forms": rows, "price": price, "levels_npz_sha256": Z.sha_file(od / "levels.npz"), "seconds": round(time.time() - t0, 1)}
    ctx = {"cov": cov, "unions": unions, "n": n, "nq": nq}
    if ch is not None:                                                                    # Part B tables need the level-3 sets
        per3 = [(set(runs[i]), set().union(subs[i][1], subs[i][2], subs[i][3])) for i in range(n)]
        subdf = Counter()
        for a_, b_ in per3:
            subdf.update(a_ | b_)
        ctx.update(ch=ch, per3=per3, subdf=subdf)
    log(f"{d}: done in {res['seconds']}s")
    return res, ctx


def cmd_run():
    ap1 = read_ap1()
    res, ctxs = {}, {}
    for d in SETTINGS:
        res[d], ctxs[d] = run_setting(d, ap1)
    price_ref = res["musique"]["price"]["N_sp_over_none"]
    cov = {d: ctxs[d]["cov"] for d in SETTINGS}
    size = {d: {f: int(ctxs[d]["unions"][f].size) for f in ALL_FORMS} for d in SETTINGS}
    nq = {d: ctxs[d]["nq"] for d in SETTINGS}
    carried, tries = decide(cov, size, nq, price_ref)
    m = ctxs["musique"]
    unlinked = ~m["cov"]["CH"][carried]
    part_b = {"carried": carried, "unlinked_by_carried": int(unlinked.sum()), "classes": remaining(m["ch"], unlinked, m["per3"], m["subdf"])}
    out = {"declared_in": DECLARED, "script_sha256": script_sha256(), "zap1_sha256": Z.script_sha256(), "ap1_report_script_sha256": ap1["script_sha256"],
           "boot": BOOT, "seed": SEED, "cap": CAP, "price_ref": price_ref, "carried": carried, "tries": tries, "partB": part_b, "settings": res}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(out, indent=1, sort_keys=True), encoding="utf-8", newline="\n")
    (OUT / "report.md").write_text(render(out), encoding="utf-8", newline="\n")
    log(f"carried form: {carried}; written {OUT / 'report.json'} and report.md")


def render(out):
    L = [f"# AP1b report ({DECLARED}) - graph level, no model", "",
         "Share of slots linked by R + K + the form (R-world). Differences are paired over questions, 95% interval of "
         f"{out['boot']} question resamples (seed {out['seed']}); calls ABOVE / LEVEL / BELOW by whether the interval clears 0.", ""]
    for d, r in out["settings"].items():
        kinds = [k for k in ("CH", "GP") if k in r["info"]["slots"]]
        L += [f"## {d}", "", f"nodes {r['info']['nodes']}, questions {r['info']['questions']}, R + K pairs {r['info']['base_pairs']}", "",
              "| form | form pairs | union pairs | new over none | " + " | ".join(f"{k} linked | {k} share | {k} vs CN_all | {k} vs N_sp" for k in kinds) + " |",
              "|---|---:|---:|---:|" + "---:|---:|:--|:--|" * len(kinds)]
        for f in ALL_FORMS:
            x = r["forms"][f]
            cells = []
            for k in kinds:
                y = x[k]
                a, b = y["vs_CN_all"], y["vs_N_sp"]
                cells += [f"{y['linked']}/{y['slots']}", f"{y['share']:.2f}%", f"{a['diff']:+.2f} [{a['lo']:+.2f}, {a['hi']:+.2f}] {a['call']}",
                          f"{b['diff']:+.2f} [{b['lo']:+.2f}, {b['hi']:+.2f}] {b['call']}"]
            L.append(f"| {f} | {x['form_pairs']} | {x['union_pairs']} | {x['new_over_none']} | " + " | ".join(cells) + " |")
        L.append("")
        if r["price"]:
            L.append(f"N_sp's price (new union pairs per CH slot it adds over the union without a NER slot): {r['price']['N_sp_over_none']}")
            for k in kinds:
                L.append(f"- price over F0, {k}: " + ", ".join(f"{f} {r['price'][k][f]['over_F0']}" for f in FORMS))
            L.append("")
    L += ["## S1: the walk", "", f"N_sp's price: {out['price_ref']} pairs per slot. Each step against the carried form.", "",
          "| form | against | (a) CH diff [95%] | (a) | (b) GP calls (musique / 2wiki / hotpotqa) | (b) | added CH slots | (c) price | (c) | ok |", "|---|---|:--|:-:|:--|:-:|---:|---:|:-:|:-:|"]
    for t in out["tries"]:
        g = t["b_GP"]
        L.append(f"| {t['form']} | {t['against']} | {t['a_CH_diff']:+.2f} [{t['a_CH_lo']:+.2f}, {t['a_CH_hi']:+.2f}] | {'yes' if t['a'] else 'no'} | "
                 f"{g['musique']['call']} {g['musique']['diff']:+.2f} / {g['2wiki']['call']} {g['2wiki']['diff']:+.2f} / {g['hotpotqa']['call']} {g['hotpotqa']['diff']:+.2f} | "
                 f"{'yes' if t['b'] else 'no'} | {t['added_CH_slots']} | {t['c_price']} | {'yes' if t['c'] else 'no'} | {'yes' if t['ok'] else 'no'} |")
    pb = out["partB"]
    L += ["", f"**Carried form: {out['carried']}.**", "", f"## Part B: what {pb['carried']} leaves on MuSiQue's chain slots (the answer read as a label, diagnostic only)", "",
          f"{pb['unlinked_by_carried']} chain slots unlinked; by class: " + "; ".join(f"{k}: {v}" for k, v in sorted(pb["classes"].items())), ""]
    return "\n".join(L) + "\n"


# ── self-test ───────────────────────────────────────────────────────────────


def selftest():
    # the title strip
    assert title_stripped("Robert Abel (animator)") == "Robert Abel"
    assert title_stripped("Charleston, South Carolina") == "Charleston"
    assert title_stripped("Real_Time with Bill Maher") == "Real Time with Bill Maher"
    # the sub-spans of a run
    usual = lambda k: True  # noqa: E731
    s = subs_of("chief justice of delta state", usual)
    assert s[1] == {"justice of delta state", "delta state", "chief justice of delta", "chief justice"}, s[1]
    assert s[2] == {"justice of delta"}, s[2]
    assert s[3] == {"chief", "justice", "delta", "state"}, s[3]
    s = subs_of("king of aragon", usual)
    assert s[1] == set() and s[2] == set() and s[3] == {"king", "aragon"}, s
    s = subs_of("birmingham city", usual)
    assert s[1] == set() and s[2] == set() and s[3] == {"birmingham", "city"}, s
    assert all(not v for v in subs_of("nigeria", usual).values())
    s = subs_of("steven universe", lambda k: k != "universe")                          # a one-word sub-span must be usual
    assert s[3] == {"steven"}, s
    # a toy corpus: the runs, the levels, the nesting
    texts = ["she was named Chief Justice of Delta State in 2003.", "an organization for women of Delta State, Nigeria, was founded.",
             "the Chief Justice of Delta State retired.", "unrelated words here about Nigeria."]
    titles = ["Tox0", "Tox1 (toy)", "Tox2", "Prince of Delta State"]
    passes = lambda: ((t, x if zu2b.is_descriptive(t, x) else None) for t, x in zip(titles, texts))  # noqa: E731
    n = len(texts)
    runs, subs, truns, tsubs = name_tables(passes, n)
    assert runs[0] == ["chief justice of delta state"] and runs[1] == ["delta state", "nigeria"], runs
    assert runs[2] == ["chief justice of delta state"] and runs[3] == ["nigeria"], runs
    assert truns[3] == ["prince of delta state"] and truns[1] == ["tox1"], truns
    lp = level_pairs(runs, subs, truns, tsubs, n)
    F = build_forms(lp)

    def pairs(f):
        return {(int(k // n), int(k % n)) for k in F[f]}

    assert pairs("F0") == {(0, 2), (1, 3)}, pairs("F0")
    assert pairs("F1") == {(0, 2), (1, 3), (0, 1), (1, 2)}, pairs("F1")                  # 'delta state' joins the run's sub-span and the run in p1
    assert (0, 3) not in pairs("F1") and (2, 3) not in pairs("F1")
    for x, y in zip(("F0", "F1", "F2", "F3"), ("F1", "F2", "F3", "F3")):
        assert pairs(x) <= pairs(y)
    assert pairs("F0") <= pairs("F0T") and pairs("F1T") >= pairs("F1") | pairs("F0T")
    assert (0, 3) in pairs("F1T") and (1, 3) in pairs("F1T"), pairs("F1T")              # p3's title run 'prince of delta state' holds 'delta state' at level 1
    assert (0, 3) not in pairs("F0T")                                                      # level 0 has no sub-span of the title
    # coverage, grading and the price
    sk = np.asarray([0 * 4 + 1, 0 * 4 + 3], np.int64)
    assert covered_np(sk, F["F1"]).tolist() == [True, False]
    assert marginal(10, 6, 5, 3) == 2.0 and marginal(10, 6, 3, 3) is None
    # the walk on synthetic coverage: the first step is ABOVE on CH, not BELOW on GP, and cheap; the next steps add nothing
    rng = np.random.default_rng(0)
    nqs = {"musique": 200, "2wiki": 100, "hotpotqa": 100}
    cov, size = {}, {}
    for d, nq in nqs.items():
        q = np.repeat(np.arange(nq), 5)
        base = rng.random(q.size) < 0.5
        better = base | (rng.random(q.size) < 0.10)
        added = int(better.sum() - base.sum())
        cov[d] = {"GP": {f: (base if f in ("none", "F0") else better) for f in ALL_FORMS}}
        cov[d]["GP"]["_q"] = q
        size[d] = {f: 1000 + (10 * added if f not in ("none", "F0") else 0) for f in ALL_FORMS}
        if d == "musique":
            cov[d]["CH"] = {f: (base if f in ("none", "F0") else better) for f in ALL_FORMS}
            cov[d]["CH"]["_q"] = q
            size[d]["F1"] = 1000 + 5 * added
    carried, tries = decide(cov, size, nqs, price_ref=128.6)
    assert carried == "F1", (carried, tries)
    assert tries[0]["form"] == "F1" and tries[0]["a"] and tries[0]["b"] and tries[0]["c"] and tries[0]["c_price"] == 5.0, tries[0]
    assert tries[1]["against"] == "F1" and not tries[1]["a"] and not tries[1]["ok"], tries[1]          # F2 equals F1 on CH: no step
    assert tries[-1]["form"] == "F1T" and not tries[-1]["ok"], tries[-1]
    # a step whose price is above N_sp's is refused even when ABOVE
    size["musique"]["F1"] = 1000 + 500 * added
    carried2, tries2 = decide(cov, size, nqs, price_ref=128.6)
    assert tries2[0]["a"] and tries2[0]["b"] and not tries2[0]["c"] and not tries2[0]["ok"], tries2[0]
    print("selftest ok")


def main(argv=None):
    a = (argv if argv is not None else sys.argv[1:]) or ["run"]
    if a[0] == "selftest":
        selftest()
    elif a[0] == "run":
        selftest()
        cmd_run()
    else:
        raise SystemExit("usage: zap1b.py run | selftest")


if __name__ == "__main__":
    main()
