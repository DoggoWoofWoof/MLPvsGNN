"""D83's report (docs/D83_UNIFIED_R_REFIT.md sections 7 and 12): the 15 deciding cells (b1v against b1w on B1's three settings,
zg6r, GNX and zrckpn, three seeds) and the call; beside them, deciding nothing, the six s1eval carves against every record seed
filed, FC@5 and hit@1, MuSiQue by hops and 2Wiki/HotpotQA by kind (PH1's groups), the model gaps at D83 beside the record's,
and the per-label counts of links and of gold-touching links. Nothing here trains or reads a model.

    python outputs/mp_unified/zd83rep.py labelstats --arm B1V|B1W|D83E|REC --dataset D   # the host's record tree: link counts
    python outputs/mp_unified/zd83rep.py files        # the rx fetch commands for every file the report reads
    python outputs/mp_unified/zd83rep.py report       # the laptop: outputs/d83/report/report.json, report.md, calls.json
    python outputs/mp_unified/zd83rep.py selftest

The rule (section 7, fixed before any read): a cell is D83 minus record on the per-question mean over the cell's fits (three
seeds of one split; six fits, both splits, for 2Wiki in-domain), over the questions with a gold, with the ratio estimator's
paired bootstrap over questions (zap1.boot_diff, copied here and checked equal in the self-test; 2,000 resamples, seed 0),
95% interval: ABOVE, BELOW or LEVEL. ADOPT when the 15 cells are read and none is BELOW; NOT_ADOPTED when one is BELOW (decided
even with cells missing); INCOMPLETE otherwise. G6 before any number: a D83 read's sha256 is its marker's, a record read's is
the filed one (B1M's and IDPA's records for b1w, the fit's read.json for s1eval), and the paired reads hold the same questions
in the same order with the same gold totals.
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

DECLARED = "docs/D83_UNIFIED_R_REFIT.md"
STAGE = "D83-report"
D83 = ROOT / "outputs" / "d83"
OUT = D83 / "report"
LSTATS = D83 / "labelstats"
REC_MIRROR = D83 / "rec"                       # rx fetch --into outputs/d83/rec: the record fits' s1eval reads and read.json
BENCH = ROOT / "outputs" / "bench" / "hipporag2"
CAND = "p@swa"
BOOT, SEED = 2000, 0
KINDS = ("zg6r", "gnx", "zrckpn")
ROLE = {"zg6r": "zg6r (GNN)", "gnx": "GNX (GNN, same inputs)", "zrckpn": "zrckpn (MLP)"}
BASES = {"zg6r": "scr-ml1-zg6r", "gnx": "scr-gnx-nx", "zrckpn": "scr-ltm-zrckpn"}          # zd83r.BASES
SEEDS = (0, 1, 2)
SPLITS = ("WQ-musique", "WQ-hotpotqa")
SUF = {"WQ-musique": "", "WQ-hotpotqa": "-hp"}
B1 = ("musique", "hotpotqa", "2wiki")
SIX = ("squad", "hotpotqa", "2wiki", "musique", "metaqa", "webqsp")
IN_DOMAIN = {"musique": ("WQ-hotpotqa",), "hotpotqa": ("WQ-musique",), "2wiki": SPLITS, "squad": SPLITS, "webqsp": SPLITS,
             "metaqa": ()}                                                                   # zswpcr.in_domain_splits
B1_CELLS = (("musique", "in-domain"), ("hotpotqa", "in-domain"), ("2wiki", "in-domain"), ("musique", "held-out"),
            ("hotpotqa", "held-out"))
DECIDING = tuple((k, ds, rg) for k in KINDS for ds, rg in B1_CELLS)
GAPS = (("gnx", "zrckpn"), ("zg6r", "zrckpn"))
CARVE = {"B1V": "b1v", "D83E": "s1eval"}
REC_CARVE = {"B1V": "b1w", "D83E": "s1eval"}
REC_B1W = {"zg6r": "outputs/b1p/fits/w", "zrckpn": "outputs/b1m/fits/w", "gnx": "outputs/idpa/fits/w"}   # B1P, B1M, IDPA
LABELS = ("S", "R.mention", "R.name_comention", "R.title_comention", "K.knn")                      # zd83.LABELS
REC_FAMS = ("slot", "NER.spacy", "K.knn")          # the record's families: 0 the slot (S on a KB, W2_S on passages), 1, 2
LS_ARMS = {"B1V": ("outputs/b1v", "b1v", LABELS, B1), "B1W": ("outputs/b1p", "b1w", REC_FAMS, B1),
           "D83E": ("outputs/abl/D83E", "s1eval", LABELS, SIX), "REC": ("outputs/pko", "s1eval", REC_FAMS, SIX)}
LS_PAIR = {"B1V": "B1W", "D83E": "REC"}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] zd83rep: {msg}", flush=True)


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sha_src(p):
    return hashlib.sha256(Path(p).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def rel(p):
    try:
        return Path(p).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return Path(p).as_posix()


def read_json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def write_json(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str) + "\n", encoding="utf-8")
    tmp.replace(p)


# ── fits, regimes, paths ─────────────────────────────────────────────────────


def splits_of(ds, regime):
    ind = IN_DOMAIN[ds]
    return tuple(sp for sp in SPLITS if (sp in ind) == (regime == "in-domain"))


def regimes(ds):
    return [rg for rg in ("in-domain", "held-out") if splits_of(ds, rg)]


def fits_of(kind, ds, regime, seeds=SEEDS):
    return [f"{BASES[kind]}-s{s}{SUF[sp]}" for sp in splits_of(ds, regime) for s in seeds]


def kind_of(fit):
    for k, b in BASES.items():
        if fit.startswith(b + "-s"):
            return k
    raise SystemExit(f"D83REP: {fit} is not a D83 read fit")


def d83_path(arm, fit, ds):
    """zd83r.read_path, where `rx fetch` puts it on the laptop (d83_collect.py copies the world's reads into the record's tree)."""
    return D83 / "reads" / "fits" / arm / fit / "reads" / f"{ds}__{CARVE[arm]}.npz"


def d83_marker(arm, fit, ds):
    return D83 / "reads" / "fits" / arm / fit / f"d83_read_{ds}.json"


def rec_path(arm, fit, ds):
    if arm == "B1V":
        return ROOT / REC_B1W[kind_of(fit)] / fit / "reads" / f"{ds}__b1w.npz"
    return REC_MIRROR / "outputs" / "screen" / "fits" / fit / "reads" / f"{ds}__s1eval.npz"


def pins():
    """The record reads' filed sha256: b1w from B1M's record (zg6r = B1P's reads, zrckpn) and IDPA's (GNX); s1eval from each
    record fit's read.json (fetched beside its reads). A key two records hold must agree."""
    pin = {}

    def put(k, h):
        if pin.setdefault(k, h) != h:
            raise SystemExit(f"D83REP: the filed records disagree on {k}")

    for name, kinds in (("b1m", ("zg6r", "zrckpn")), ("idpa", ("gnx",))):
        rec = read_json(ROOT / "outputs" / name / "report.json")
        for ds in B1:
            for key, v in rec["settings"][ds]["fits"].items():
                kind, carve, fit = key.split("/")
                if kind in kinds and carve == "b1w" and "file_sha256" in v:
                    put(("B1V", fit, ds), v["file_sha256"])
    for kind in KINDS:
        for f in fits_of(kind, "metaqa", "held-out"):                 # MetaQA is held-out for both splits: all six fits
            rj = REC_MIRROR / "outputs" / "screen" / "fits" / f / "read.json"
            if not rj.exists():
                continue
            carves = read_json(rj).get("carves", {})
            for ds in SIX:
                c = carves.get(f"{ds}=s1eval")
                if c and c.get("sha256"):
                    put(("D83E", f, ds), c["sha256"])
    return pin


# ── reads and cells ──────────────────────────────────────────────────────────


def metrics_of(top, hit, gt):
    """(R@5, FC@5, hit@1) per question, 0 where gold_total is 0 (p3c_claim1.metrics_of)."""
    top, hit, gt = np.asarray(top, np.float64), np.asarray(hit, np.float64), np.asarray(gt, np.int64)
    out = np.zeros((gt.size, 3))
    ok = gt > 0
    out[ok, 0] = top[ok] / gt[ok]
    out[ok, 1] = (top[ok] == gt[ok]).astype(np.float64)
    out[ok, 2] = hit[ok]
    return out


_LOADED = {}


def load(p):
    p = Path(p)
    if p not in _LOADED:
        with np.load(p) as z:
            names = [str(x) for x in z["candidates"]]
            if CAND not in names:
                raise SystemExit(f"D83REP: {rel(p)}: {CAND} is not among {names}")
            k = names.index(CAND)
            ids = np.array([str(x) for x in z["ids"]])
            gt = z["gold_total"].astype(np.int64)
            top, hit = z["top"][k], z["hit"][k]
        _LOADED[p] = {"ids": ids, "gt": gt, "m": metrics_of(top, hit, gt), "sha256": sha_file(p)}
    return _LOADED[p]


def real_cell(side, arm, kind, ds, regime, problems, pin, seeds=SEEDS):
    """The fits' reads of one cell, or None (the reason in problems): side 'd83' or 'rec'; arm B1V (b1v / b1w) or D83E
    (s1eval / s1eval)."""
    fits = fits_of(kind, ds, regime, seeds)
    got = []
    for f in fits:
        p = d83_path(arm, f, ds) if side == "d83" else rec_path(arm, f, ds)
        if not p.exists():
            problems["missing"].append(rel(p))
            return None
        r = load(p)
        if side == "d83":
            mk = d83_marker(arm, f, ds)
            if not mk.exists() or read_json(mk).get("read_sha256") != r["sha256"]:
                problems["g6"].append(f"{rel(p)}: sha256 is not its D83 marker's ({rel(mk)})")
                return None
        elif pin.get((arm, f, ds)) != r["sha256"]:
            problems["g6"].append(f"{rel(p)}: sha256 is not the filed one ({pin.get((arm, f, ds))})")
            return None
        got.append(r)
    if not got:
        return None
    for g in got[1:]:
        if not (np.array_equal(g["ids"], got[0]["ids"]) and np.array_equal(g["gt"], got[0]["gt"])):
            raise SystemExit(f"D83REP: {side} {arm} {kind} {ds}: the fits' reads hold different questions or gold totals")
    return {"fits": fits, "ids": got[0]["ids"], "gt": got[0]["gt"], "m": [g["m"] for g in got], "sha256": [g["sha256"] for g in got]}


# ── the interval ─────────────────────────────────────────────────────────────


def boot_diff(cx, cr, qidx, nq, boot=BOOT, seed=SEED):
    """zap1.boot_diff, copied (the self-test checks it equal): cx minus cr by the ratio estimator over questions, the point value
    and the 95% interval of `boot` resamples of the questions (seeded)."""
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


def same_questions(a, b, what):
    if not (np.array_equal(a["ids"], b["ids"]) and np.array_equal(a["gt"], b["gt"])):
        raise SystemExit(f"D83REP: G6: {what}: the reads hold other questions, order or gold totals")


def mean_m(c):
    return np.mean(np.stack(c["m"]), 0)


def interval(x, r):
    n = int(np.asarray(x).size)
    pt, lo, hi = boot_diff(x, r, np.arange(n), n)
    return {"diff": round(pt, 5), "ci": [round(lo, 5), round(hi, 5)], "call": call(lo, hi)}


def compare(a, b, mask=None, what=""):
    """b minus a over the questions with a gold (and in mask): R@5 (the deciding measure), FC@5 and hit@1, each with its interval;
    the fits' own means beside."""
    same_questions(a, b, what)
    ok = a["gt"] > 0
    if mask is not None:
        ok = ok & mask
    ma, mb = mean_m(a)[ok], mean_m(b)[ok]
    out = {"questions": int(ok.sum()), "record": round(float(ma[:, 0].mean()), 5), "d83": round(float(mb[:, 0].mean()), 5)}
    out.update(interval(mb[:, 0], ma[:, 0]))
    for j, name in ((1, "FC@5"), (2, "hit@1")):
        out[name] = {"record": round(float(ma[:, j].mean()), 5), "d83": round(float(mb[:, j].mean()), 5), **interval(mb[:, j], ma[:, j])}
    out["per_fit_record"] = [round(float(m[ok, 0].mean()), 5) for m in a["m"]]
    out["per_fit_d83"] = [round(float(m[ok, 0].mean()), 5) for m in b["m"]]
    out["fits_d83_below"] = sum(int(float(mb_[ok, 0].mean()) < float(ma_[ok, 0].mean())) for ma_, mb_ in zip(a["m"], b["m"]))
    return out


def gap(x, y):
    """x minus y (two models' cells on the same reads' questions), R@5 over the questions with a gold."""
    same_questions(x, y, "gap")
    ok = x["gt"] > 0
    return mean_m(x)[ok, 0] - mean_m(y)[ok, 0]


# ── the report ───────────────────────────────────────────────────────────────


def verdict(calls):
    if any(c == "BELOW" for c in calls.values()):
        return "NOT_ADOPTED"
    if len(calls) != len(DECIDING) or any(c is None for c in calls.values()):
        return "INCOMPLETE"
    return "ADOPT"


def groups_of(ds, ids):
    """PH1's groups on B1a's questions (MuSiQue by hops, 2Wiki and HotpotQA by kind), as masks over the reads' order."""
    import zph1 as PH
    q = read_json(BENCH / ds / "queries.json")
    if [str(x["id"]) for x in q] != [str(x) for x in ids]:
        raise SystemExit(f"D83REP: {ds}: the reads are not B1a's questions in B1a's order")
    g = PH.groups(ds, [str(x["kind"]) for x in q])
    out = {}
    for k in PH.SIZES[ds]:
        m = np.zeros(len(q), bool)
        m[g[k]] = True
        out[k] = m
    return out


def build(get):
    """get(side, arm, kind, ds, regime, seeds) -> cell or None. Returns the record (no file written)."""
    rec = {"declared_in": DECLARED, "stage": STAGE, "candidate": CAND, "boot": {"resamples": BOOT, "seed": SEED,
                                                                               "estimator": "zap1.boot_diff (ratio over questions)"},
           "deciding": {}, "calls": {}, "by_group": {}, "gaps": {}, "s1eval": {}}
    for kind, ds, rg in DECIDING:
        key = f"{kind} {ds} {rg}"
        r, d = get("rec", "B1V", kind, ds, rg, SEEDS), get("d83", "B1V", kind, ds, rg, SEEDS)
        if r is None or d is None:
            rec["calls"][key] = None
            continue
        c = compare(r, d, what=key)
        r0, d0 = get("rec", "B1V", kind, ds, rg, (0,)), get("d83", "B1V", kind, ds, rg, (0,))
        c["seed0"] = compare(r0, d0, what=key + " seed 0") if r0 is not None and d0 is not None else None
        c["fits"] = d["fits"]
        c["sha256_d83"], c["sha256_record"] = d["sha256"], r["sha256"]
        rec["deciding"][key] = c
        rec["calls"][key] = c["call"]
        rec["by_group"][key] = {g: compare(r, d, m, f"{key} {g}") for g, m in groups_of(ds, d["ids"]).items()}
    rec["verdict"] = verdict(rec["calls"])
    for arm, cells in (("B1V", B1_CELLS), ("D83E", tuple((ds, rg) for ds in SIX for rg in regimes(ds)))):
        for ds, rg in cells:
            for x, y in GAPS:
                key = f"{arm} {ds} {rg} {x}-{y}"
                cs = {(s, k): get(s, arm, k, ds, rg, SEEDS) for s in ("d83", "rec") for k in (x, y)}
                if any(v is None for v in cs.values()):
                    continue
                gd, gr = gap(cs[("d83", x)], cs[("d83", y)]), gap(cs[("rec", x)], cs[("rec", y)])
                same_questions(cs[("d83", x)], cs[("rec", x)], key)
                z = np.zeros_like(gd)
                rec["gaps"][key] = {"d83": interval(gd, z), "record": interval(gr, z), "change": interval(gd, gr),
                                    "questions": int(gd.size)}
    for kind in KINDS:
        for ds in SIX:
            for rg in regimes(ds):
                key = f"{kind} {ds} {rg}"
                r, d = get("rec", "D83E", kind, ds, rg, SEEDS), get("d83", "D83E", kind, ds, rg, SEEDS)
                if r is not None and d is not None:
                    rec["s1eval"][key] = {**compare(r, d, what="s1eval " + key), "fits": d["fits"]}
    return rec


def label_block():
    out = {}
    for arm, ref in LS_PAIR.items():
        for ds in LS_ARMS[arm][3]:
            a, b = LSTATS / arm / f"{ds}.json", LSTATS / ref / f"{ds}.json"
            out[f"{arm} {ds}"] = {"d83": read_json(a) if a.exists() else None, "record": read_json(b) if b.exists() else None}
    return out


def pct(x):
    return f"{100 * x:.1f}"


def dtxt(c):
    return f"{100 * c['diff']:+.1f} ({100 * c['ci'][0]:+.1f}, {100 * c['ci'][1]:+.1f})" if c else "--"


def write_md(rec, path):
    T = [f"# D83 report: the unified R with labelled links, the models of record fitted and read again", "",
         f"Declared in {DECLARED} (section 7, fixed before any read). R@5 x100, candidate {CAND}. A cell is D83 minus the record "
         f"on the per-question mean over the cell's fits (three seeds; six fits for 2Wiki in-domain), over the questions with a "
         f"gold, with the 95% interval of the ratio estimator's paired bootstrap over questions ({BOOT} resamples, seed {SEED}): "
         f"ABOVE, BELOW or LEVEL. The record: zg6r on b1w (B1P), zrckpn on b1w (B1M), GNX on b1w (IDPA); on s1eval, every "
         f"record fit's read. In-domain: MuSiQue <- WQ-hotpotqa, HotpotQA <- WQ-musique, 2Wiki <- both; held-out the others.", "",
         f"**Call: {rec['verdict']}** (ADOPT when the 15 deciding cells are read and none is BELOW).", "",
         "## The deciding cells (B1's settings: b1v against b1w)", "",
         "| model | setting | regime | record | D83 | D83 - record (95%) | call | seed 0 | fits below |", "|---|---|---|---:|---:|---|---|---|---:|"]
    for kind, ds, rg in DECIDING:
        c = rec["deciding"].get(f"{kind} {ds} {rg}")
        if c is None:
            T.append(f"| {ROLE[kind]} | {ds} | {rg} | -- | -- | -- | not read | -- | -- |")
            continue
        T.append(f"| {ROLE[kind]} | {ds} | {rg} | {pct(c['record'])} | {pct(c['d83'])} | {dtxt(c)} | {c['call']} | "
                 f"{dtxt(c['seed0'])} | {c['fits_d83_below']}/{len(c['per_fit_d83'])} |")
    T += ["", "## FC@5 and hit@1 on the deciding cells (beside)", "",
          "| model | setting | regime | FC@5 record | FC@5 D83 | FC@5 diff (95%) | hit@1 record | hit@1 D83 | hit@1 diff (95%) |",
          "|---|---|---|---:|---:|---|---:|---:|---|"]
    for key, c in rec["deciding"].items():
        kind, ds, rg = key.split(" ")
        f, h = c["FC@5"], c["hit@1"]
        T.append(f"| {ROLE[kind]} | {ds} | {rg} | {pct(f['record'])} | {pct(f['d83'])} | {dtxt(f)} | {pct(h['record'])} | "
                 f"{pct(h['d83'])} | {dtxt(h)} |")
    T += ["", "## By hops (MuSiQue) and by kind (2Wiki, HotpotQA): D83 - record, R@5 (beside, descriptive)", "",
          "| model | setting | regime | group | questions | record | D83 | D83 - record (95%) |", "|---|---|---|---|---:|---:|---:|---|"]
    for key, gs in rec["by_group"].items():
        kind, ds, rg = key.split(" ")
        for g, c in gs.items():
            T.append(f"| {ROLE[kind]} | {ds} | {rg} | {g} | {c['questions']} | {pct(c['record'])} | {pct(c['d83'])} | {dtxt(c)} |")
    T += ["", "## The gaps between the models, at D83 beside the record's (R@5, paired by question)", "",
          "| carve | dataset | regime | gap | record | D83 | change (D83 - record) |", "|---|---|---|---|---|---|---|"]
    for key, c in rec["gaps"].items():
        arm, ds, rg, xy = key.split(" ")
        T.append(f"| {CARVE[arm]} | {ds} | {rg} | {xy} | {dtxt(c['record'])} | {dtxt(c['d83'])} | {dtxt(c['change'])} |")
    T += ["", "## The six s1eval carves: D83 against every record seed filed (beside, deciding nothing)", "",
          "| model | dataset | regime | questions | record | D83 | D83 - record (95%) | FC@5 diff | hit@1 diff |", "|---|---|---|---:|---:|---:|---|---|---|"]
    for key, c in rec["s1eval"].items():
        kind, ds, rg = key.split(" ")
        T.append(f"| {ROLE[kind]} | {ds} | {rg} | {c['questions']} | {pct(c['record'])} | {pct(c['d83'])} | {dtxt(c)} | "
                 f"{dtxt(c['FC@5'])} | {dtxt(c['hit@1'])} |")
    T += ["", "## Links per label (unique pairs per question), gold-touching (one end gold) and gold-gold (both ends gold)", ""]
    for key, x in rec.get("labels", {}).items():
        arm, ds = key.split(" ")
        if not x["d83"]:
            continue
        T += [f"**{CARVE[arm]} {ds}** ({x['d83']['questions']} questions)", "", "| store | label | links | gold-touching | gold-gold | questions with a gold-gold link |",
              "|---|---|---:|---:|---:|---:|"]
        for side in ("d83", "record"):
            s = x[side]
            if not s:
                continue
            for i, lab in enumerate(s["labels"]):
                T.append(f"| {'D83' if side == 'd83' else 'record'} | {lab} | {s['links'][i]} | {s['gold_touching'][i]} | "
                         f"{s['gold_gold'][i]} | {s['questions_gold_gold'][i]} |")
        T.append("")
    T += ["", f"Missing reads: {len(rec['problems']['missing'])}; G6 problems: {len(rec['problems']['g6'])}."]
    for p in rec["problems"]["g6"][:20]:
        T.append(f"- {p}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(T) + "\n", encoding="utf-8")


def report():
    pin = pins()
    problems = {"missing": [], "g6": []}

    def get(side, arm, kind, ds, rg, seeds):
        return real_cell(side, arm, kind, ds, rg, problems, pin, seeds)

    rec = build(get)
    rec["labels"] = label_block()
    rec["problems"] = {k: sorted(set(v)) for k, v in problems.items()}
    rec["pins"] = len(pin)
    rec["zd83rep_sha256"] = sha_src(__file__)
    rec["utc"] = utc()
    write_json(OUT / "report.json", rec)
    write_json(OUT / "calls.json", {"declared_in": DECLARED, "verdict": rec["verdict"], "calls": rec["calls"]})
    write_md(rec, OUT / "report.md")
    log(f"report: {rel(OUT / 'report.md')}; verdict {rec['verdict']}; {sum(c is not None for c in rec['calls'].values())}/15 cells read; "
        f"missing {len(rec['problems']['missing'])}; G6 problems {len(rec['problems']['g6'])}")
    return 0


# ── the label counts (the host's record tree) ────────────────────────────────


def count_links(n, gold, q_ne, e_u, e_v, e_lab, L):
    """One part's links per label: directed entries; unique unordered pairs per question; those with one end gold; with both ends
    gold; questions with at least one gold-gold pair. n: pool sizes; gold: per row; links pool-local per question."""
    n = np.asarray(n, np.int64)
    rows = int(n.sum())
    off = np.r_[0, np.cumsum(n)].astype(np.int64)
    qe = np.repeat(np.arange(n.size, dtype=np.int64), np.asarray(q_ne, np.int64))
    gu = off[qe] + np.asarray(e_u, np.int64)
    gv = off[qe] + np.asarray(e_v, np.int64)
    lab = np.asarray(e_lab, np.int64)
    if lab.size and (int(lab.min()) < 0 or int(lab.max()) >= L):
        raise SystemExit(f"D83REP: a link's label is outside 0..{L - 1}")
    if bool((gu == gv).any()):
        raise SystemExit("D83REP: a self-loop among the links")
    g = np.asarray(gold).reshape(-1) > 0
    if g.size != rows:
        raise SystemExit(f"D83REP: {g.size} gold flags for {rows} rows")
    key = np.unique((np.minimum(gu, gv) * rows + np.maximum(gu, gv)) * L + lab)
    l1, pair = key % L, key // L
    a, b = pair // rows, pair % rows
    touch, both = g[a] | g[b], g[a] & g[b]
    qa = np.searchsorted(off, a[both], side="right") - 1
    qk = np.unique(qa * L + l1[both])
    return {"directed": np.bincount(lab, minlength=L), "links": np.bincount(l1, minlength=L),
            "gold_touching": np.bincount(l1[touch], minlength=L), "gold_gold": np.bincount(l1[both], minlength=L),
            "questions_gold_gold": np.bincount(qk % L, minlength=L)}


def labelstats(arm, ds):
    import lean_cache as LC
    import ltwdiag as LTW
    if arm not in LS_ARMS or ds not in LS_ARMS[arm][3]:
        raise SystemExit(f"D83REP: labelstats takes an arm of {', '.join(LS_ARMS)} and its datasets")
    root_rel, carve, names, _dss = LS_ARMS[arm]
    root = ROOT / root_rel
    L = len(names)
    t0 = time.time()
    parts, recs = LC.part_dirs(ds, carve, root / "cache")
    tot = {k: np.zeros(L, np.int64) for k in ("directed", "links", "gold_touching", "gold_gold", "questions_gold_gold")}
    rows = queries = with_gold = 0
    plist = []
    for p, r in zip(parts, recs):
        A, E, lsha = LTW.part_arrays(p, r, root / "links")
        c = count_links(A["n"], A["gold"], E["q_ne"], E["e_u"], E["e_v"], E["e_fam"], L)
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
           "labels": list(names), "rows": rows, "questions": queries, "questions_with_gold": with_gold,
           **{k: [int(x) for x in v] for k, v in tot.items()}, "parts": plist,
           "rule": "links = unique unordered pairs per question and label; gold-touching = one end a gold row of its question; "
                   "gold-gold = both ends; directed = the stored entries",
           "zd83rep_sha256": sha_src(__file__), "seconds": round(time.time() - t0, 1), "utc": utc()}
    write_json(LSTATS / arm / f"{ds}.json", out)
    log(f"labelstats {arm} {ds}: links {out['links']}, gold-touching {out['gold_touching']}, gold-gold {out['gold_gold']}")
    return 0


# ── the files ────────────────────────────────────────────────────────────────


def files():
    """The rx fetch commands (run on the laptop after d83_collect.py and the labelstats rows ran on the host)."""
    rx = "python tools/rx/rx.py fetch --glob"
    for g in ("outputs/d83/reads/fits/*/*/reads/*.npz", "outputs/d83/reads/fits/*/*/*.json", "outputs/d83/fits/*.json",
              "outputs/d83/labelstats/*/*.json", "outputs/d83/collect_manifest.txt"):
        print(f'{rx} "{g}"')
    for b in BASES.values():
        for g in (f"outputs/screen/fits/{b}-s*/reads/*__s1eval.npz", f"outputs/screen/fits/{b}-s*/read.json"):
            print(f'{rx} "{g}" --into outputs/d83/rec')
    return 0


# ── self-test ────────────────────────────────────────────────────────────────


def selftest():
    import zd83m as M83
    import zd83r as R
    import zswpcr as CR
    assert len(DECIDING) == 15 and len(set(DECIDING)) == 15
    assert set(DECIDING) == {(k, ds, rg) for k in KINDS for ds in B1 for rg in regimes(ds)}
    assert regimes("2wiki") == ["in-domain"] and regimes("metaqa") == ["held-out"] and regimes("musique") == ["in-domain", "held-out"]
    assert fits_of("gnx", "musique", "in-domain") == ["scr-gnx-nx-s0-hp", "scr-gnx-nx-s1-hp", "scr-gnx-nx-s2-hp"]
    assert len(fits_of("zg6r", "2wiki", "in-domain")) == 6 and len(fits_of("zrckpn", "metaqa", "held-out")) == 6
    assert fits_of("zrckpn", "hotpotqa", "held-out", (0,)) == ["scr-ltm-zrckpn-s0-hp"]
    for ds in SIX:
        assert set(IN_DOMAIN[ds]) == set(CR.in_domain_splits(ds)), ds
    assert BASES == R.BASES and LABELS == tuple(M83.LABELS) and set(CARVE) == set(R.ARMS)
    for arm, (carve, dss) in R.ARMS.items():
        assert CARVE[arm] == carve
        for f in R.FITS:
            for ds in dss:
                assert d83_path(arm, f, ds) == R.read_path(arm, f, ds) and d83_marker(arm, f, ds) == R.marker_path(arm, f, ds)
    # the interval: zap1's own function, bit for bit
    import zap1 as AP1
    rng = np.random.default_rng(3)
    x, y = rng.random(1000), rng.random(1000)
    assert boot_diff(x, y, np.arange(1000), 1000) == AP1.boot_diff(x, y, np.arange(1000), 1000)
    assert call(0.1, 0.2) == "ABOVE" and call(-0.2, -0.1) == "BELOW" and call(-0.1, 0.1) == "LEVEL"
    # the record's b1w reads: present and equal to their filed sha256 (real files, real shapes)
    pin = pins()
    b1w = {k: h for k, h in pin.items() if k[0] == "B1V"}
    assert len(b1w) == 54, len(b1w)
    for (arm, f, ds), h in b1w.items():
        assert sha_file(rec_path(arm, f, ds)) == h, (f, ds)
    # the cells on real reads: D83 = the record (every cell LEVEL at 0, ADOPT); one cell lowered (BELOW, NOT_ADOPTED);
    # cells missing (INCOMPLETE); a read with its questions out of order (G6 stops)
    problems = {"missing": [], "g6": []}
    rec_cell = (lambda s, a, k, ds, rg, seeds: real_cell("rec", a, k, ds, rg, problems, pin, seeds) if a == "B1V" else None)

    def same(s, a, k, ds, rg, seeds):
        return rec_cell("rec", a, k, ds, rg, seeds)

    r = build(same)
    assert r["verdict"] == "ADOPT" and not problems["missing"] and not problems["g6"], (r["verdict"], problems)
    assert all(c["diff"] == 0 and c["call"] == "LEVEL" and c["seed0"]["diff"] == 0 for c in r["deciding"].values())
    assert len(r["deciding"]) == 15 and len(r["gaps"]) == 10 and not r["s1eval"]
    mq = r["by_group"]["gnx musique in-domain"]
    assert {g: c["questions"] for g, c in mq.items()} == {"2 hops": 518, "3 hops": 316, "4 hops": 166}
    assert all(c["change"]["diff"] == 0 for c in r["gaps"].values())

    def lowered(s, a, k, ds, rg, seeds):
        c = rec_cell("rec", a, k, ds, rg, seeds)
        if s == "d83" and c is not None and (k, ds, rg) == ("zrckpn", "hotpotqa", "held-out"):
            c = dict(c, m=[np.c_[m[:, :1] - 0.05 * (m[:, :1] > 0), m[:, 1:]] for m in c["m"]])
        return c

    r2 = build(lowered)
    d = r2["deciding"]["zrckpn hotpotqa held-out"]
    assert r2["verdict"] == "NOT_ADOPTED" and d["call"] == "BELOW" and d["fits_d83_below"] == 3, (r2["verdict"], d["call"])
    assert sum(c == "BELOW" for c in r2["calls"].values()) == 1

    def gone(s, a, k, ds, rg, seeds):
        return None if (s == "d83" and k == "gnx" and ds == "2wiki") else rec_cell("rec", a, k, ds, rg, seeds)

    r3 = build(gone)
    assert r3["verdict"] == "INCOMPLETE" and r3["calls"]["gnx 2wiki in-domain"] is None

    def shuffled(s, a, k, ds, rg, seeds):
        c = rec_cell("rec", a, k, ds, rg, seeds)
        if s == "d83" and c is not None and (k, ds) == ("zg6r", "musique"):
            o = np.r_[1, 0, 2:c["ids"].size]
            c = dict(c, ids=c["ids"][o], gt=c["gt"][o], m=[m[o] for m in c["m"]])
        return c

    try:
        build(shuffled)
        raise AssertionError("a read out of order was paired")
    except SystemExit as e:
        assert "G6" in str(e)
    # the label counts: both directions stored, a duplicate entry, two labels on one pair, golds; against a brute-force count
    n = np.array([4, 3])
    gold = np.array([1, 0, 1, 0, 0, 1, 1])
    q_ne = np.array([7, 3])
    e_u = np.array([0, 2, 0, 1, 0, 3, 2, 1, 2, 2])
    e_v = np.array([2, 0, 1, 0, 2, 0, 3, 2, 1, 1])
    e_lab = np.array([1, 1, 4, 4, 2, 0, 0, 1, 1, 1])
    c = count_links(n, gold, q_ne, e_u, e_v, e_lab, 5)
    off = [0, 4]
    want = {}
    for q, (lo_, hi_) in enumerate(((0, 7), (7, 10))):
        for u, v, lb in zip(e_u[lo_:hi_], e_v[lo_:hi_], e_lab[lo_:hi_]):
            want.setdefault(lb, set()).add((q, min(u, v) + off[q], max(u, v) + off[q]))
    for lb in range(5):
        s = want.get(lb, set())
        assert c["links"][lb] == len(s), (lb, c["links"], s)
        assert c["gold_touching"][lb] == sum(1 for _q, u, v in s if gold[u] or gold[v])
        assert c["gold_gold"][lb] == sum(1 for _q, u, v in s if gold[u] and gold[v])
        assert c["questions_gold_gold"][lb] == len({q for q, u, v in s if gold[u] and gold[v]})
    assert c["directed"].tolist() == [2, 5, 1, 0, 2] and c["links"].tolist() == [2, 2, 1, 0, 1]
    assert c["gold_gold"].tolist() == [0, 2, 1, 0, 0] and c["questions_gold_gold"].tolist() == [0, 2, 1, 0, 0]
    try:
        count_links(n, gold, q_ne, e_u, np.where(np.arange(10) == 9, 2, e_v), e_lab, 5)
        raise AssertionError("a self-loop passed")
    except SystemExit:
        pass
    log("selftest passed: 15 cells, the interval equal to zap1's, 54 record b1w reads at their filed sha256, ADOPT / "
        "NOT_ADOPTED / INCOMPLETE / G6 on the real reads, the label counts against a brute-force count")
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
