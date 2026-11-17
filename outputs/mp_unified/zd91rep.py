"""D91's report (docs/D91_D83_WITHOUT_LABELS.md section 6, fixed before any read): Call L (D83 minus D91 on the 15 B1 cells: do
the rule names on the links help?), Call A (D91 minus the record on the same cells, D83's rule) and what carries; beside them,
deciding nothing, the six s1eval carves (D91E against D83E and against the record), FC@5 and hit@1, MuSiQue by hops and
2Wiki/HotpotQA by kind, the gaps GNX - zrckpn and zg6r - zrckpn at D91 beside D83's, and the per-family link counts beside
D83's per-label counts. Nothing here trains or reads a model.

    python outputs/mp_unified/zd91rep.py labelstats --arm D91B|D91E --dataset D   # the host's record tree: link counts
    python outputs/mp_unified/zd91rep.py files        # the rx fetch commands for every file the report reads
    python outputs/mp_unified/zd91rep.py report       # the laptop: outputs/d91/report/report.json, report.md, calls.json
    python outputs/mp_unified/zd91rep.py selftest

Every cell, interval and check is zd83rep's code (imported, not copied): a cell is the per-question mean over the cell's fits
(three seeds of one split; six fits, both splits, for 2Wiki in-domain), over the questions with a gold, R@5 at p@swa; the
ratio estimator's paired bootstrap over questions (zap1.boot_diff; 2,000 resamples, seed 0), 95%: ABOVE, BELOW or LEVEL.
D91's side reads D91B (B1's settings, carve b1v) and D91E (s1eval) through D91's markers. D83's side reads D83's B1V and D83E
through D83's markers. The record's side reads b1w and s1eval at their filed sha256. G6 before any number, as D83's: each read's
sha256 is its marker's (D91, D83) or the filed one (the record), and the paired reads hold the same questions in the same
order with the same gold totals.

Call L (section 6): HELP when at least one cell is ABOVE and none BELOW; HURT when at least one is BELOW and none ABOVE; MIXED
when both; LEVEL when neither. While a cell is unread, L is INCOMPLETE, except MIXED, which no further cell can undo (the same
handling as Call A's INCOMPLETE; this report's reading of unread cells, written before any D91 read was looked at).
Call A: ADOPT when the 15 cells are read and none is BELOW; NOT_ADOPTED when one is BELOW; INCOMPLETE otherwise.
What carries: only D83 ADOPT: D83; only D91 ADOPT: D91; both: D83 when L is HELP or LEVEL, D91 when HURT, the user's call
(D95) when MIXED; neither: the record stays, and the report names the next form. D83's call is computed here from the same
reads with zd83rep's rule, and checked equal to D83's filed calls.json when that exists.
"""
import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

import zd83rep as R  # noqa: E402

DECLARED = "docs/D91_D83_WITHOUT_LABELS.md"
STAGE = "D91-report"
D91 = ROOT / "outputs" / "d91"
OUT = D91 / "report"
LSTATS = D91 / "labelstats"
CARVE = {"D91B": "b1v", "D91E": "s1eval"}                  # zd91r.ARMS
ARM83 = {"D91B": "B1V", "D91E": "D83E"}                    # D83's arm each D91 arm is paired with
FAMS = ("F0 slot (S on a KB, R.mention on passages)", "F1 NER (F0T; minus S on a KB)", "F2 K.knn")
LS_ARMS = {"D91B": ("outputs/abl/D91B", "b1v", R.B1), "D91E": ("outputs/abl/D91E", "s1eval", R.SIX)}
L_CALLS = ("HELP", "HURT", "MIXED", "LEVEL", "INCOMPLETE")
CARRY = {"HELP": "D83", "LEVEL": "D83", "HURT": "D91", "MIXED": "D95 (the user's call)", "INCOMPLETE": "INCOMPLETE"}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] zd91rep: {msg}", flush=True)


# ── paths and cells ──────────────────────────────────────────────────────────


def d91_path(arm, fit, ds):
    """zd91r.read_path, where `rx fetch` puts it on the laptop (d91_collect.py copies the world's reads into the record's tree)."""
    return D91 / "reads" / "fits" / arm / fit / "reads" / f"{ds}__{CARVE[arm]}.npz"


def d91_marker(arm, fit, ds):
    return D91 / "reads" / "fits" / arm / fit / f"d91_read_{ds}.json"


def d91_cell(arm, kind, ds, regime, problems, seeds=R.SEEDS):
    """D91's reads of one cell (zd83rep.real_cell's checks with D91's paths and markers), or None (the reason in problems)."""
    fits = R.fits_of(kind, ds, regime, seeds)
    got = []
    for f in fits:
        p = d91_path(arm, f, ds)
        if not p.exists():
            problems["missing"].append(R.rel(p))
            return None
        r = R.load(p)
        mk = d91_marker(arm, f, ds)
        if not mk.exists() or R.read_json(mk).get("read_sha256") != r["sha256"]:
            problems["g6"].append(f"{R.rel(p)}: sha256 is not its D91 marker's ({R.rel(mk)})")
            return None
        got.append(r)
    if not got:
        return None
    for g in got[1:]:
        if not (np.array_equal(g["ids"], got[0]["ids"]) and np.array_equal(g["gt"], got[0]["gt"])):
            raise SystemExit(f"D91REP: d91 {arm} {kind} {ds}: the fits' reads hold different questions or gold totals")
    return {"fits": fits, "ids": got[0]["ids"], "gt": got[0]["gt"], "m": [g["m"] for g in got], "sha256": [g["sha256"] for g in got]}


def relabel(c, lo, hi):
    """zd83rep.compare names its sides 'record' (a, subtracted) and 'd83' (b); here they are lo and hi."""
    if not isinstance(c, dict):
        return c
    out = {}
    for k, v in c.items():
        k2 = {"record": lo, "d83": hi, "per_fit_record": f"per_fit_{lo}", "per_fit_d83": f"per_fit_{hi}",
              "fits_d83_below": f"fits_{hi}_below"}.get(k, k)
        out[k2] = relabel(v, lo, hi) if isinstance(v, dict) else v
    return out


def cmp(a, b, lo, hi, mask=None, what=""):
    return relabel(R.compare(a, b, mask, what), lo, hi)


# ── the calls ────────────────────────────────────────────────────────────────


def call_l(calls):
    """Call L over the 15 cells (D83 minus D91): HELP / HURT / MIXED / LEVEL; INCOMPLETE while a cell is unread, except MIXED."""
    v = [calls.get(f"{k} {ds} {rg}") for k, ds, rg in R.DECIDING]
    above, below = "ABOVE" in v, "BELOW" in v
    if above and below:
        return "MIXED"
    if any(c is None for c in v):
        return "INCOMPLETE"
    return "HELP" if above else ("HURT" if below else "LEVEL")


def carries(v83, v91, lcall):
    """Section 6's table: what carries into the after-paper record's input set."""
    if "INCOMPLETE" in (v83, v91):
        return "INCOMPLETE"
    a83, a91 = v83 == "ADOPT", v91 == "ADOPT"
    if a83 and a91:
        return CARRY[lcall]
    if a83:
        return "D83"
    if a91:
        return "D91"
    return "the record (the report names the next form)"


def build(get):
    """get(side, arm, kind, ds, regime, seeds) -> cell or None; side 'd91', 'd83' or 'rec'; arm D91B or D91E (D83's and the
    record's side read the paired D83 arm). Returns the report record (no file written)."""
    rec = {"declared_in": DECLARED, "stage": STAGE, "candidate": R.CAND,
           "boot": {"resamples": R.BOOT, "seed": R.SEED, "estimator": "zap1.boot_diff (ratio over questions)"},
           "L": {}, "A": {}, "D83": {}, "calls_L": {}, "calls_A": {}, "calls_D83": {}, "by_group": {}, "gaps": {}, "s1eval": {}}
    for kind, ds, rg in R.DECIDING:
        key = f"{kind} {ds} {rg}"
        x, d, r = (get(s, "D91B", kind, ds, rg, R.SEEDS) for s in ("d91", "d83", "rec"))
        x0, d0, r0 = (get(s, "D91B", kind, ds, rg, (0,)) for s in ("d91", "d83", "rec"))
        rec["calls_L"][key] = rec["calls_A"][key] = rec["calls_D83"][key] = None
        if x is not None and d is not None:
            c = cmp(x, d, "d91", "d83", what=f"L {key}")
            c["seed0"] = cmp(x0, d0, "d91", "d83", what=f"L {key} seed 0") if x0 is not None and d0 is not None else None
            c["fits_d91"], c["fits_d83"], c["sha256_d91"], c["sha256_d83"] = x["fits"], d["fits"], x["sha256"], d["sha256"]
            rec["L"][key], rec["calls_L"][key] = c, c["call"]
            rec["by_group"][f"L {key}"] = {g: cmp(x, d, "d91", "d83", m, f"L {key} {g}")
                                           for g, m in R.groups_of(ds, d["ids"]).items()}
        if x is not None and r is not None:
            c = cmp(r, x, "record", "d91", what=f"A {key}")
            c["seed0"] = cmp(r0, x0, "record", "d91", what=f"A {key} seed 0") if x0 is not None and r0 is not None else None
            c["sha256_record"] = r["sha256"]
            rec["A"][key], rec["calls_A"][key] = c, c["call"]
            rec["by_group"][f"A {key}"] = {g: cmp(r, x, "record", "d91", m, f"A {key} {g}")
                                           for g, m in R.groups_of(ds, x["ids"]).items()}
        if d is not None and r is not None:
            c = cmp(r, d, "record", "d83", what=f"D83 {key}")
            rec["D83"][key], rec["calls_D83"][key] = c, c["call"]
    rec["call_L"] = call_l(rec["calls_L"])
    rec["call_A"] = R.verdict(rec["calls_A"])
    rec["call_D83"] = R.verdict(rec["calls_D83"])
    rec["carries"] = carries(rec["call_D83"], rec["call_A"], rec["call_L"])
    for arm, cells in (("D91B", R.B1_CELLS), ("D91E", tuple((ds, rg) for ds in R.SIX for rg in R.regimes(ds)))):
        for ds, rg in cells:
            for a, b in R.GAPS:
                key = f"{arm} {ds} {rg} {a}-{b}"
                cs = {(s, k): get(s, arm, k, ds, rg, R.SEEDS) for s in ("d91", "d83") for k in (a, b)}
                if any(v is None for v in cs.values()):
                    continue
                g91, g83 = R.gap(cs[("d91", a)], cs[("d91", b)]), R.gap(cs[("d83", a)], cs[("d83", b)])
                R.same_questions(cs[("d91", a)], cs[("d83", a)], key)
                z = np.zeros_like(g91)
                rec["gaps"][key] = {"d91": R.interval(g91, z), "d83": R.interval(g83, z), "change": R.interval(g91, g83),
                                    "questions": int(g91.size)}
    for kind in R.KINDS:
        for ds in R.SIX:
            for rg in R.regimes(ds):
                key = f"{kind} {ds} {rg}"
                x, d, r = (get(s, "D91E", kind, ds, rg, R.SEEDS) for s in ("d91", "d83", "rec"))
                if x is None:
                    continue
                rec["s1eval"][key] = {"fits": x["fits"],
                                      "d83_minus_d91": cmp(x, d, "d91", "d83", what="s1eval L " + key) if d is not None else None,
                                      "d91_minus_record": cmp(r, x, "record", "d91", what="s1eval A " + key) if r is not None else None}
    return rec


def label_block():
    out = {}
    for arm, (_root, _carve, dss) in LS_ARMS.items():
        for ds in dss:
            a, b = LSTATS / arm / f"{ds}.json", R.LSTATS / ARM83[arm] / f"{ds}.json"
            out[f"{arm} {ds}"] = {"d91": R.read_json(a) if a.exists() else None, "d83": R.read_json(b) if b.exists() else None}
    return out


# ── the report ───────────────────────────────────────────────────────────────


def dtxt(c):
    return R.dtxt(c) if c else "--"


def write_md(rec, path):
    T = ["# D91 report: D83 without the labels (the record's three family links), the models of record fitted and read again", "",
         f"Declared in {DECLARED} (section 6, fixed before any read). R@5 x100, candidate {R.CAND}. A cell is the per-question mean "
         f"over the cell's fits (three seeds; six fits for 2Wiki in-domain), over the questions with a gold, with the 95% "
         f"interval of the ratio estimator's paired bootstrap over questions ({R.BOOT} resamples, seed {R.SEED}): ABOVE, BELOW "
         f"or LEVEL. In-domain: MuSiQue <- WQ-hotpotqa, HotpotQA <- WQ-musique, 2Wiki <- both; held-out the others.", "",
         f"**Call L (D83 minus D91: the rule names): {rec['call_L']}.** **Call A (D91 minus the record): {rec['call_A']}.** "
         f"D83's own call on the same reads: {rec['call_D83']}"
         + (f" (filed: {rec['d83_filed']})" if rec.get("d83_filed") else "") + f". **Carries: {rec['carries']}.**", "",
         "## Call L: D83 minus D91 on B1's settings (b1v)", "",
         "| model | setting | regime | D91 | D83 | D83 - D91 (95%) | call | seed 0 | D83 fits below |",
         "|---|---|---|---:|---:|---|---|---|---:|"]
    for kind, ds, rg in R.DECIDING:
        c = rec["L"].get(f"{kind} {ds} {rg}")
        if c is None:
            T.append(f"| {R.ROLE[kind]} | {ds} | {rg} | -- | -- | -- | not read | -- | -- |")
            continue
        T.append(f"| {R.ROLE[kind]} | {ds} | {rg} | {R.pct(c['d91'])} | {R.pct(c['d83'])} | {dtxt(c)} | {c['call']} | "
                 f"{dtxt(c['seed0'])} | {c['fits_d83_below']}/{len(c['per_fit_d83'])} |")
    T += ["", "## Call A: D91 minus the record on B1's settings (b1v against b1w)", "",
          "| model | setting | regime | record | D91 | D91 - record (95%) | call | seed 0 | D91 fits below |",
          "|---|---|---|---:|---:|---|---|---|---:|"]
    for kind, ds, rg in R.DECIDING:
        c = rec["A"].get(f"{kind} {ds} {rg}")
        if c is None:
            T.append(f"| {R.ROLE[kind]} | {ds} | {rg} | -- | -- | -- | not read | -- | -- |")
            continue
        T.append(f"| {R.ROLE[kind]} | {ds} | {rg} | {R.pct(c['record'])} | {R.pct(c['d91'])} | {dtxt(c)} | {c['call']} | "
                 f"{dtxt(c['seed0'])} | {c['fits_d91_below']}/{len(c['per_fit_d91'])} |")
    T += ["", "## FC@5 and hit@1 on Call L's cells (beside)", "",
          "| model | setting | regime | FC@5 D91 | FC@5 D83 | FC@5 D83 - D91 (95%) | hit@1 D91 | hit@1 D83 | hit@1 D83 - D91 (95%) |",
          "|---|---|---|---:|---:|---|---:|---:|---|"]
    for key, c in rec["L"].items():
        kind, ds, rg = key.split(" ")
        f, h = c["FC@5"], c["hit@1"]
        T.append(f"| {R.ROLE[kind]} | {ds} | {rg} | {R.pct(f['d91'])} | {R.pct(f['d83'])} | {dtxt(f)} | {R.pct(h['d91'])} | "
                 f"{R.pct(h['d83'])} | {dtxt(h)} |")
    T += ["", "## By hops (MuSiQue) and by kind (2Wiki, HotpotQA), R@5 (beside, descriptive)", "",
          "| call | model | setting | regime | group | questions | lower side | higher side | difference (95%) |",
          "|---|---|---|---|---|---:|---:|---:|---|"]
    for key, gs in rec["by_group"].items():
        which, kind, ds, rg = key.split(" ")
        lo, hi = ("d91", "d83") if which == "L" else ("record", "d91")
        for g, c in gs.items():
            T.append(f"| {which} | {R.ROLE[kind]} | {ds} | {rg} | {g} | {c['questions']} | {lo} {R.pct(c[lo])} | "
                     f"{hi} {R.pct(c[hi])} | {dtxt(c)} |")
    T += ["", "## The gaps between the models, at D91 beside D83's (R@5, paired by question)", "",
          "| carve | dataset | regime | gap | D83 | D91 | change (D91 - D83) |", "|---|---|---|---|---|---|---|"]
    for key, c in rec["gaps"].items():
        arm, ds, rg, xy = key.split(" ")
        T.append(f"| {CARVE[arm]} | {ds} | {rg} | {xy} | {dtxt(c['d83'])} | {dtxt(c['d91'])} | {dtxt(c['change'])} |")
    T += ["", "## The six s1eval carves (beside, deciding nothing)", "",
          "| model | dataset | regime | questions | D91 | D83 - D91 (95%) | D91 - record (95%) |", "|---|---|---|---:|---:|---|---|"]
    for key, c in rec["s1eval"].items():
        kind, ds, rg = key.split(" ")
        l_, a_ = c["d83_minus_d91"], c["d91_minus_record"]
        any_ = l_ or a_
        T.append(f"| {R.ROLE[kind]} | {ds} | {rg} | {any_['questions'] if any_ else '--'} | "
                 f"{R.pct(any_['d91']) if any_ else '--'} | {dtxt(l_)} | {dtxt(a_)} |")
    T += ["", "## Links per family (D91) beside links per label (D83): unique pairs per question, gold-touching, gold-gold", ""]
    for key, x in rec.get("labels", {}).items():
        arm, ds = key.split(" ")
        if not (x["d91"] or x["d83"]):
            continue
        T += [f"**{CARVE[arm]} {ds}**", "", "| store | family or label | links | gold-touching | gold-gold | questions with a gold-gold link |",
              "|---|---|---:|---:|---:|---:|"]
        for side in ("d91", "d83"):
            s = x[side]
            if not s:
                continue
            for i, lab in enumerate(s["labels"]):
                T.append(f"| {side.upper()} | {lab} | {s['links'][i]} | {s['gold_touching'][i]} | {s['gold_gold'][i]} | "
                         f"{s['questions_gold_gold'][i]} |")
        T.append("")
    T += ["", f"Missing reads: {len(rec['problems']['missing'])}; G6 problems: {len(rec['problems']['g6'])}."]
    for p in rec["problems"]["g6"][:20]:
        T.append(f"- {p}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(T) + "\n", encoding="utf-8")


def report():
    pin = R.pins()
    problems = {"missing": [], "g6": []}

    def get(side, arm, kind, ds, rg, seeds):
        if side == "d91":
            return d91_cell(arm, kind, ds, rg, problems, seeds)
        return R.real_cell(side, ARM83[arm], kind, ds, rg, problems, pin, seeds)

    rec = build(get)
    filed = R.OUT / "calls.json"
    if filed.exists():
        rec["d83_filed"] = R.read_json(filed).get("verdict")
        if rec["call_D83"] != "INCOMPLETE" and rec["d83_filed"] != rec["call_D83"]:
            raise SystemExit(f"D91REP: D83's call on these reads is {rec['call_D83']}, its filed calls.json says {rec['d83_filed']}")
    rec["labels"] = label_block()
    rec["problems"] = {k: sorted(set(v)) for k, v in problems.items()}
    rec["pins"] = len(pin)
    rec["zd91rep_sha256"] = R.sha_src(__file__)
    rec["zd83rep_sha256"] = R.sha_src(R.__file__)
    rec["utc"] = R.utc()
    R.write_json(OUT / "report.json", rec)
    R.write_json(OUT / "calls.json", {"declared_in": DECLARED, "call_L": rec["call_L"], "call_A": rec["call_A"],
                                      "call_D83": rec["call_D83"], "carries": rec["carries"], "calls_L": rec["calls_L"],
                                      "calls_A": rec["calls_A"]})
    write_md(rec, OUT / "report.md")
    log(f"report: {R.rel(OUT / 'report.md')}; L {rec['call_L']}, A {rec['call_A']}, D83 {rec['call_D83']}, carries {rec['carries']}; "
        f"L cells read {sum(c is not None for c in rec['calls_L'].values())}/15; missing {len(rec['problems']['missing'])}; "
        f"G6 problems {len(rec['problems']['g6'])}")
    return 0


# ── the family counts (the host's record tree) ───────────────────────────────


def labelstats(arm, ds):
    """zd83rep.labelstats's loop on a D91 arm's own family links (count_links is zd83rep's)."""
    import lean_cache as LC
    import ltwdiag as LTW
    if arm not in LS_ARMS or ds not in LS_ARMS[arm][2]:
        raise SystemExit(f"D91REP: labelstats takes an arm of {', '.join(LS_ARMS)} and its datasets")
    root_rel, carve, _dss = LS_ARMS[arm]
    root = ROOT / root_rel
    L = len(FAMS)
    t0 = time.time()
    parts, recs = LC.part_dirs(ds, carve, root / "cache")
    tot = {k: np.zeros(L, np.int64) for k in ("directed", "links", "gold_touching", "gold_gold", "questions_gold_gold")}
    rows = queries = with_gold = 0
    plist = []
    for p, r in zip(parts, recs):
        A, E, lsha = LTW.part_arrays(p, r, root / "links")
        c = R.count_links(A["n"], A["gold"], E["q_ne"], E["e_u"], E["e_v"], E["e_fam"], L)
        for k in tot:
            tot[k] += c[k]
        rows += int(r["rows"])
        queries += int(r["queries"])
        with_gold += int((np.asarray(A["gold_total"]) > 0).sum())
        plist.append({"part": p.name, "rows": int(r["rows"]), "queries": int(r["queries"]),
                      "cache_record_sha256": LC.sha_file(p / "record.json"), "links_record_sha256": lsha})
        del A, E
        log(f"labelstats {arm} {ds}/{carve} {p.name} ({time.time() - t0:.0f}s)")
    out = {"declared_in": DECLARED, "stage": STAGE, "arm": arm, "dataset": ds, "carve": carve, "root": root_rel,
           "labels": list(FAMS), "rows": rows, "questions": queries, "questions_with_gold": with_gold,
           **{k: [int(x) for x in v] for k, v in tot.items()}, "parts": plist,
           "rule": "links = unique unordered pairs per question and family; gold-touching = one end a gold row of its question; "
                   "gold-gold = both ends; directed = the stored entries (zd83rep.count_links)",
           "zd91rep_sha256": R.sha_src(__file__), "seconds": round(time.time() - t0, 1), "utc": R.utc()}
    R.write_json(LSTATS / arm / f"{ds}.json", out)
    log(f"labelstats {arm} {ds}: links {out['links']}, gold-touching {out['gold_touching']}, gold-gold {out['gold_gold']}")
    return 0


# ── the files ────────────────────────────────────────────────────────────────


def files():
    """The rx fetch commands (run on the laptop after d91_collect.py and the labelstats rows ran on the host), then D83's."""
    rx = "python tools/rx/rx.py fetch --glob"
    for g in ("outputs/d91/reads/fits/*/*/reads/*.npz", "outputs/d91/reads/fits/*/*/*.json", "outputs/d91/fits/*.json",
              "outputs/d91/labelstats/*/*.json", "outputs/d91/collect_manifest.txt"):
        print(f'{rx} "{g}"')
    print("# and D83's side (the same files D83's report reads):")
    return R.files()


# ── self-test ────────────────────────────────────────────────────────────────


def selftest():
    import zd91r as Z
    assert len(R.DECIDING) == 15 and set(CARVE) == set(Z.ARMS) and set(ARM83) == set(Z.ARMS)
    assert R.BASES == Z.BASES and set(Z.FITS) == {f for k in R.KINDS for ds in R.SIX for rg in R.regimes(ds)
                                                  for f in R.fits_of(k, ds, rg)}
    for arm, (carve, dss) in Z.ARMS.items():
        assert CARVE[arm] == carve and set(dss) == set(LS_ARMS[arm][2])
        for f in Z.FITS:
            for ds in dss:
                assert d91_path(arm, f, ds) == Z.read_path(arm, f, ds) and d91_marker(arm, f, ds) == Z.marker_path(arm, f, ds)
    keys = [f"{k} {ds} {rg}" for k, ds, rg in R.DECIDING]
    # Call L and what carries, on the calls alone
    allc = lambda c: {k: c for k in keys}  # noqa: E731
    assert call_l(allc("LEVEL")) == "LEVEL"
    assert call_l(dict(allc("LEVEL"), **{keys[3]: "ABOVE"})) == "HELP"
    assert call_l(dict(allc("LEVEL"), **{keys[3]: "BELOW"})) == "HURT"
    assert call_l(dict(allc("LEVEL"), **{keys[3]: "BELOW", keys[9]: "ABOVE"})) == "MIXED"
    assert call_l(dict(allc("LEVEL"), **{keys[3]: None, keys[9]: "ABOVE"})) == "INCOMPLETE"
    assert call_l(dict(allc(None), **{keys[3]: "BELOW", keys[9]: "ABOVE"})) == "MIXED"
    assert carries("ADOPT", "ADOPT", "HELP") == "D83" and carries("ADOPT", "ADOPT", "LEVEL") == "D83"
    assert carries("ADOPT", "ADOPT", "HURT") == "D91" and carries("ADOPT", "ADOPT", "MIXED").startswith("D95")
    assert carries("ADOPT", "NOT_ADOPTED", "HURT") == "D83" and carries("NOT_ADOPTED", "ADOPT", "HELP") == "D91"
    assert carries("NOT_ADOPTED", "NOT_ADOPTED", "LEVEL").startswith("the record") and carries("INCOMPLETE", "ADOPT", "LEVEL") == "INCOMPLETE"
    # on real reads: the record's b1w reads stand in for all three sides (real files, real shapes)
    pin = R.pins()
    problems = {"missing": [], "g6": []}

    def rec_cell(arm, k, ds, rg, seeds):
        return R.real_cell("rec", "B1V", k, ds, rg, problems, pin, seeds) if arm == "D91B" else None

    def bump(c, d):
        return dict(c, m=[np.c_[np.clip(m[:, :1] + d * (m[:, :1] > 0), 0, None), m[:, 1:]] for m in c["m"]])

    def make(changes):
        def get(s, arm, k, ds, rg, seeds):
            c = rec_cell(arm, k, ds, rg, seeds)
            d = changes.get((s, k, ds, rg))
            return None if (c is None or d == "gone") else (bump(c, d) if d else c)
        return get

    r = build(make({}))
    assert not problems["missing"] and not problems["g6"], problems
    assert (r["call_L"], r["call_A"], r["call_D83"], r["carries"]) == ("LEVEL", "ADOPT", "ADOPT", "D83"), r["carries"]
    assert all(c["diff"] == 0 and c["call"] == "LEVEL" for c in r["L"].values()) and len(r["L"]) == 15 and len(r["gaps"]) == 10
    assert {g: c["questions"] for g, c in r["by_group"]["L gnx musique in-domain"].items()} == {"2 hops": 518, "3 hops": 316, "4 hops": 166}
    assert r["L"]["zg6r musique held-out"]["d91"] == r["L"]["zg6r musique held-out"]["d83"]
    # D91 lower on one cell: L ABOVE there (HELP), A BELOW (NOT_ADOPTED), D83 ADOPT: D83 carries
    r = build(make({("d91", "zrckpn", "hotpotqa", "held-out"): -0.05}))
    assert (r["call_L"], r["call_A"], r["call_D83"], r["carries"]) == ("HELP", "NOT_ADOPTED", "ADOPT", "D83")
    assert r["L"]["zrckpn hotpotqa held-out"]["call"] == "ABOVE" and r["A"]["zrckpn hotpotqa held-out"]["call"] == "BELOW"
    assert r["L"]["zrckpn hotpotqa held-out"]["fits_d83_below"] == 0 and r["A"]["zrckpn hotpotqa held-out"]["fits_d91_below"] == 3
    # D83 lower on one cell: L BELOW (HURT), D83 NOT_ADOPTED, D91 ADOPT: D91 carries
    r = build(make({("d83", "gnx", "musique", "in-domain"): -0.05}))
    assert (r["call_L"], r["call_A"], r["call_D83"], r["carries"]) == ("HURT", "ADOPT", "NOT_ADOPTED", "D91")
    # both above the record on different cells: MIXED, both ADOPT: the user's call
    r = build(make({("d91", "zg6r", "2wiki", "in-domain"): 0.05, ("d83", "gnx", "hotpotqa", "in-domain"): 0.05}))
    assert (r["call_L"], r["call_A"], r["call_D83"]) == ("MIXED", "ADOPT", "ADOPT") and r["carries"].startswith("D95")
    # D91's read missing on a cell: L and A INCOMPLETE
    r = build(make({("d91", "gnx", "2wiki", "in-domain"): "gone"}))
    assert (r["call_L"], r["call_A"], r["carries"]) == ("INCOMPLETE", "INCOMPLETE", "INCOMPLETE")
    assert r["calls_L"]["gnx 2wiki in-domain"] is None and "gnx 2wiki in-domain" not in r["L"]

    # a D91 read out of order: G6 stops
    def shuffled(s, arm, k, ds, rg, seeds):
        c = rec_cell(arm, k, ds, rg, seeds)
        if s == "d91" and c is not None and (k, ds) == ("zg6r", "musique"):
            o = np.r_[1, 0, 2:c["ids"].size]
            c = dict(c, ids=c["ids"][o], gt=c["gt"][o], m=[m[o] for m in c["m"]])
        return c

    try:
        build(shuffled)
        raise AssertionError("a read out of order was paired")
    except SystemExit as e:
        assert "G6" in str(e)
    # a D91 read whose marker does not hold its sha256: G6 problem, the cell unread
    probs = {"missing": [], "g6": []}
    assert d91_cell("D91B", "zg6r", "musique", "held-out", probs) is None
    assert probs["missing"] or probs["g6"]
    log("selftest passed: paths equal zd91r's, Call L and the carry table, LEVEL / HELP / HURT / MIXED / INCOMPLETE on the "
        "record's real b1w reads, G6 on a read out of order")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=("labelstats", "files", "report", "selftest"))
    ap.add_argument("--arm")
    ap.add_argument("--dataset")
    a = ap.parse_args(argv)
    if a.cmd == "labelstats":
        return labelstats(a.arm, a.dataset)
    return {"files": files, "report": report, "selftest": selftest}[a.cmd]()


if __name__ == "__main__":
    sys.exit(main())
