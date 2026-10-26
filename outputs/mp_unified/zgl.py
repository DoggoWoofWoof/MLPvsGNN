"""GL (docs/GL_GPU_LINE_ON_L40S.md): the GPU line on U2d's base build, on Modal's L40S, while the host is offline.

G6 is decided now from U2d's filed L40S fits, by G6's rule unchanged (docs/G6_FREE_GNN.md, section 4):
    zg6     scr-u2d-base-zg6, -hp, against zspk's scr-u2d-base-zspk, -hp
    pair    U2d's compares scr-u2d-base-zg6-vs-zspk, -hp (screen_pair, GL's loader)
    null    U2d's zspk null scr-u2d-null-zspk-s1, -s2, -s1-hp, -s2-hp (zu2d.u2d_null: N2's design on the L40S); no
            re-base, since the pair's base is the null's seed-0 fit
    out     outputs/screen/scr-gl-g6-zg6-pair, scr-gl-g6-zg6-pair-recall; outputs/gl/g6-verdict
The eight fits (zg6's two, zspk's two, the null's four) must share one device (zu2d.one_device) before anything is
called. R2's two arms and G6's twin run on the same build and device from 1 November (sections 3 and 4); their commands
come in a later commit.

    python outputs/mp_unified/zgl.py g6          # the pair, the re-call and the verdict
    python outputs/mp_unified/zgl.py --selftest
"""
import os
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import json  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import zu2d as Z  # noqa: E402  (U2d's fits, device check, null loader and fakes; through it zg6 and screen_pair)

G6, SP = Z.G6, Z.SP
G2 = G6.G2
S, LG, LC, SR = Z.S, Z.LG, Z.LC, Z.SR
log = Z.log

DECLARED = "docs/GL_GPU_LINE_ON_L40S.md"
STAGE = "GL"
SCREEN = Path("outputs") / "screen"            # relative: every command runs from the repository's root
OUT = Path("outputs") / "gl"
SPLITS = Z.SPLITS
SUF = Z.SUF
ARM, BASE, NULL_ARM = "zg6", "zspk", "zspk"
G6_FIT = {sp: Z.fit_name("base", ARM, 0, sp) for sp in SPLITS}          # scr-u2d-base-zg6, -hp
G6_BASE = {sp: Z.fit_name("base", BASE, 0, sp) for sp in SPLITS}        # scr-u2d-base-zspk, -hp: the null's seed 0
G6_CMP = {sp: f"scr-u2d-base-zg6-vs-zspk{SUF[sp]}" for sp in SPLITS}    # U2d's compares, filed 22 October
HOST_ZSPK = {sp: Path("outputs") / "l1" / f"base-zspk-{sp}.json" for sp in SPLITS}   # zspk's host card fits (beside)
PAIR = "scr-gl-g6-zg6-pair"
RECALL = "scr-gl-g6-zg6-pair-recall"
VERDICT = "g6-verdict"
NULL_KIND = "U2d-zspk"
NULL_IS = ("U2d's zspk null on the L40S: zspk's base-build fit trained again with seeds 1 and 2, each against its "
           "seed-0 fit (N2's design, on the device the arms ran on)")
AGAINST = ("zspk's seed-0 fit on U2d's base build (scr-u2d-base-zspk, scr-u2d-base-zspk-hp) on Modal's L40S: the same "
           "pools, KT input, card settings and device")
RULE = ("ADOPT_FOR_FULL_RUN when zg6's re-call under U2d's zspk seed null (N2's design on the L40S) has no LOSS on any "
        "of its twelve reads and at least one GAIN; NOT_ADOPTED otherwise; INCOMPLETE when that re-call is missing, is "
        "not GL's re-call of zg6 under U2d's zspk null, or does not hold the twelve reads (docs/G6_FREE_GNN.md, "
        f"section 4, with the files of {DECLARED}, section 2)")
NOT_BESIDE = {"twin": "zg6t has no L40S fit yet: it runs on workspace 3 from 1 November (section 4), filed beside",
              "oct7": "the 7 October re-call is not computed: its re-base reads the host's fits, another device "
                      "(section 2)"}
FIX_PAIR = (("R@5 of the screen's p@swa minus step 1's p@swa of the same split",
             "R@5 of zg6's p@swa minus zspk's p@swa of the same split, both U2d's base-build fits on the L40S"),)
FIX_RECALL = FIX_PAIR + (("twice the seed-only spread of seeds 1 and 2 against seed 0",
                          "twice the seed-only spread of zspk's seeds 1 and 2 against its seed-0 fit on U2d's base "
                          "build"),
                         ("D1, D2: the null's seed-only differences on the read",
                          "D1, D2: zspk's seed-only differences on the read (U2d's zspk null)"))


def rel(p):
    p = Path(p)
    return p.relative_to(ROOT).as_posix() if p.is_absolute() and p.is_relative_to(ROOT) else p.as_posix()


def fits():
    """The eight fits G6's call reads: zg6's and zspk's seed-0 fits, then zspk's four null fits."""
    return [*G6_FIT.values(), *G6_BASE.values(),
            *(Z.fit_name("base", NULL_ARM, s, sp) for sp in SPLITS for s in SR.SEEDS)]


def device(out_root=None):
    """The eight fits' one device (zu2d.one_device over their screen and read stamps); stops when two differ."""
    return Z.one_device(fits(), out_root)


def loader(dev):
    """screen_pair.load for U2d's compare of zg6 with zspk on the base build, of either split: (split, rows)."""

    def load(fn):
        p = Path(fn)
        if not p.exists():
            raise SystemExit(f"zgl: {p} missing")
        rec = json.loads(p.read_text(encoding="utf-8"))
        nn_ = SR.parts_of(rec.get("new", ""))[-1:]
        split = next((sp for sp in SPLITS if nn_ == (G6_FIT[sp],)), None)
        if split is None:
            raise SystemExit(f"zgl: {p} reads {rec.get('new')}, not zg6's seed-0 fit on U2d's base build")
        u = rec.get("u2d") or {}
        if (u.get("declared_in"), u.get("compare"), u.get("new"), u.get("bases")) != (
                Z.DECLARED, G6_CMP[split], G6_FIT[split], [G6_BASE[split]]):
            raise SystemExit(f"zgl: {p} is not U2d's compare {G6_CMP[split]} of {G6_FIT[split]} with {G6_BASE[split]}")
        if SR.parts_of((rec.get("bases") or [""])[0])[-1:] != (G6_BASE[split],):
            raise SystemExit(f"zgl: {p} is decided against {(rec.get('bases') or [None])[0]}, not {G6_BASE[split]}")
        if u.get("device") != dev:
            raise SystemExit(f"zgl: {p} was made on {u.get('device')}, not on the fits' device {dev}")
        if sorted(rec.get("trained_on", [])) != sorted(LG.FITS[split]):
            raise SystemExit(f"zgl: {p} trained on {rec.get('trained_on')}, not {split}'s {sorted(LG.FITS[split])}")
        if rec.get("candidate") != "p@swa" or rec.get("carve") != "s1eval":
            raise SystemExit(f"zgl: {p} reads {rec.get('candidate')} on {rec.get('carve')}, not p@swa on s1eval")
        rows = rec["by_base"][rec["bases"][0]]["rows"]
        if sorted(r["dataset"] for r in rows) != sorted(LG.EVAL_ORDER):
            raise SystemExit(f"zgl: {p} reads {[r['dataset'] for r in rows]}, not the six datasets")
        return split, rows

    return load


def restamp(out, kind, dev):
    """The pair's or re-call's files stamped as GL's call of G6; the md's wording says what was compared."""
    out = Path(out)
    js = out.with_suffix(".json")
    if js.exists():
        rec = json.loads(js.read_text(encoding="utf-8"))
        rec.update({"stage": STAGE, "declared_in": DECLARED, "decides": "G6", "g6_declared_in": G6.DECLARED,
                    "arm": ARM, "base_arm": BASE, "against": AGAINST, "device": dev,
                    "compares": [G6_CMP[sp] for sp in SPLITS], "zgl_sha256": LC.sha_src(__file__)})
        if kind == "recall":
            rec.update({"null_kind": NULL_KIND, "null_is": NULL_IS, "deciding": True})
        LC.write_json(js, rec)
    md = out.with_suffix(".md")
    if md.exists():
        t = md.read_text(encoding="utf-8")
        for a, b in (FIX_RECALL if kind == "recall" else FIX_PAIR):
            t = t.replace(a, b)
        head = f"GL ({DECLARED}), G6's call on U2d's L40S fits: zg6 (the free GNN) against {BASE} on U2d's base build"
        head += f", re-called under {NULL_IS}.\n\n" if kind == "recall" else ".\n\n"
        md.write_text(head + t, encoding="utf-8")


def pair(screen=None, out_root=None):
    """screen_pair.pair over U2d's two compares of zg6 with zspk, through GL's loader, after the device check."""
    dev = device(out_root)
    scr = Path(screen or SCREEN)
    out = scr / PAIR
    saved = SP.load
    SP.load = loader(dev)
    try:
        rec = SP.pair([scr / f"{G6_CMP[sp]}.json" for sp in SPLITS], out)
    finally:
        SP.load = saved
    restamp(out, "pair", dev)
    return rec


def recall(screen=None, out_root=None):
    """screen_recall.pair of GL's pair under U2d's zspk null. The pair's base is the null's seed-0 fit on both
    splits, so nothing is re-based; screen_recall checks each read's base R@5 against the null's seed 0."""
    dev = device(out_root)
    scr = Path(screen or SCREEN)
    if any(Z.fit_name("base", NULL_ARM, 0, sp) != G6_BASE[sp] for sp in SPLITS):
        raise SystemExit("zgl: the pair's base is not the null's seed-0 fit")
    files = Z.null_files(NULL_ARM, scr)
    null, sources = Z.u2d_null(NULL_ARM, files)
    pf = (scr / PAIR).with_suffix(".json")
    pr = json.loads(pf.read_text(encoding="utf-8")) if pf.exists() else {}
    if (pr.get("stage"), pr.get("declared_in"), pr.get("arm"), pr.get("device")) != (STAGE, DECLARED, ARM, dev):
        raise SystemExit(f"zgl: {pf} is not GL's pair of zg6 on the fits' device")
    out = scr / RECALL
    saved = SR.load_null
    SR.load_null = lambda _files: (null, sources)
    try:
        rec = SR.pair(files, pf, out)
    finally:
        SR.load_null = saved
    restamp(out, "recall", dev)
    return rec


# ── the verdict (G6's rule) and its beside rows ──────────────────────────────


def tag(r):
    return f"{r.get('split')} {r.get('dataset')}"


def recall_rows(rp, problems):
    rp = Path(rp)
    if not rp.exists():
        problems.append(f"{rp.name} missing")
        return [], {}
    rec = json.loads(rp.read_text(encoding="utf-8"))
    if (rec.get("stage"), rec.get("declared_in"), rec.get("decides"), rec.get("arm"), rec.get("base_arm"),
            rec.get("null_kind")) != (STAGE, DECLARED, "G6", ARM, BASE, NULL_KIND):
        problems.append(f"{rp.name} is not GL's re-call of zg6 under U2d's zspk null")
    rows = rec.get("rows", [])
    got = sorted((r.get("split"), r.get("dataset")) for r in rows)
    if got != sorted((sp, ds) for sp in SPLITS for ds in LG.EVAL_ORDER):
        problems.append(f"{rp.name}: {len(rows)} reads, not the twelve of {', '.join(SPLITS)} on the six datasets")
    return rows, rec


def r5_of(cmp_file, base_name, which):
    """{dataset: R@5} of a compare's new fit (which 'new') or of its base (which 'base'), when the compare is decided
    against base_name; {} when the file is absent or not that compare."""
    p = Path(cmp_file)
    if not p.exists():
        return {}
    c = json.loads(p.read_text(encoding="utf-8"))
    if SR.parts_of((c.get("bases") or [""])[0])[-1:] != (base_name,):
        return {}
    return {r["dataset"]: r[which][0] for r in c["by_base"][c["bases"][0]]["rows"]}


def beside_r5(scr, root):
    """R@5 per (split, dataset) of zrckp's seed-0 fit on the L40S (its null's base) and of zspk's host card fit
    (L1's compare of it with step 1's: its new fit)."""
    zrckp, host = {}, {}
    for sp in SPLITS:
        for ds, v in r5_of(Path(scr) / f"scr-u2d-null-zrckp-s1{SUF[sp]}.json", Z.fit_name("base", "zrckp", 0, sp),
                           "base").items():
            zrckp[(sp, ds)] = v
        p = Path(root) / HOST_ZSPK[sp]
        if p.exists():
            c = json.loads(p.read_text(encoding="utf-8"))
            if SR.parts_of(c.get("new", ""))[-1:] == (G6.BASE_FITS[sp][2],):
                for r in c["by_base"][c["bases"][0]]["rows"]:
                    host[(sp, r["dataset"])] = r["new"][0]
    return zrckp, host


def fmt(x, signed=False):
    if not isinstance(x, (int, float)):
        return "n/a"
    return f"{x:+.4f}" if signed else f"{x:.4f}"


def verdict(screen=None, out=None, root=None):
    """G6's verdict from zg6's re-call under U2d's zspk null (G6, sections 4 and 5). Beside, deciding nothing: zg6
    minus zspk per read (R@5 with its interval, FC@5, hit@1), R@5 of rrf, zspk, zg6 and zrckp on the L40S, the MLP
    share, and zspk's host card R@5 beside its L40S R@5."""
    scr, root = Path(screen or SCREEN), Path(root or ROOT)
    rp = (scr / RECALL).with_suffix(".json")
    problems = []
    rows, rrec = recall_rows(rp, problems)
    gains = [tag(r) for r in rows if r.get("call") == "GAIN"]
    losses = [tag(r) for r in rows if r.get("call") == "LOSS"]
    v = "INCOMPLETE" if problems else ("NOT_ADOPTED" if losses or not gains else "ADOPT_FOR_FULL_RUN")
    nxt = G6.next_form(v, gains, losses)
    vs = {sp: G2.beside_rows(scr / f"{G6_CMP[sp]}.json", G6_BASE[sp]) for sp in SPLITS}
    zrckp, host = beside_r5(scr, root)
    r5, share = {}, {}
    for r in rows:
        key = (r.get("split"), r.get("dataset"))
        r5[tag(r)] = {"read": r.get("read"), "rrf": r.get("rrf"), "zspk": r.get("base"), "zg6": r.get("new"),
                      "zrckp": zrckp.get(key), "zspk_host": host.get(key), "floor": r.get("floor"),
                      "call": r.get("call"), "call_filed": r.get("call_filed")}
        share[tag(r)] = G6.mlp_share(zrckp.get(key), r.get("new"), r.get("rrf"))
    rec = {"stage": STAGE, "declared_in": DECLARED, "decides": "G6", "g6_declared_in": G6.DECLARED, "rule": RULE,
           "verdict": v, "arm": ARM, "base_arm": BASE, "against": AGAINST, "null_kind": NULL_KIND, "null_is": NULL_IS,
           "device": rrec.get("device"), "next_form": nxt, "gains": gains, "losses": losses, "problems": problems,
           "calls": {tag(r): r.get("call") for r in rows}, "delta_r5": {tag(r): r.get("delta") for r in rows},
           "floors": {tag(r): r.get("floor") for r in rows}, "recall": rel(rp),
           "recall_sha256": LC.sha_file(rp) if rp.exists() else None,
           "beside": {"zg6_minus_zspk": vs, "r5": r5, "mlp_share": share, "not_beside": NOT_BESIDE},
           "zgl_sha256": LC.sha_src(__file__), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    md = [f"# G6's verdict on the L40S (GL): zg6 **{v}**", "", f"Rule: {RULE}.", "",
          f"- **zg6** (the free GNN: its own encoder, {G6.LAYERS} layers of {G6.D}-wide hidden-state message passing, "
          f"an unbounded readout, no MLP score) against {AGAINST}, after the re-call under {NULL_IS}: {len(gains)} "
          f"GAIN ({', '.join(gains) or 'none'}), {len(losses)} LOSS ({', '.join(losses) or 'none'})"
          + (f"; problems: {'; '.join(problems)}" if problems else "") + ".",
          f"- **Next form** (fixed in docs/G6_FREE_GNN.md, section 5, before any number): {nxt}.",
          f"- Device: {json.dumps(rrec.get('device'))}.", "",
          "Beside, deciding nothing.", "",
          f"- The twin: {NOT_BESIDE['twin']}.", f"- {NOT_BESIDE['oct7'][0].upper() + NOT_BESIDE['oct7'][1:]}.", "",
          "R@5 per read, all on the L40S except the last column; the MLP share = (zrckp - rrf) / (zg6 - rrf) where zg6's "
          f"gain over rrf exceeds {G6.SHARE_MIN}:", "",
          "| read | rrf | zspk | zg6 | zrckp (MLP) | MLP share | floor | call (filed -> re-call) | zspk, host card |",
          "|---|---:|---:|---:|---:|---:|---:|---|---:|"]
    for k, t in r5.items():
        c = t["call"] if t["call"] == t["call_filed"] else f"{t['call_filed']} -> {t['call']}"
        md.append(f"| {k} ({t['read']}) | {fmt(t['rrf'])} | {fmt(t['zspk'])} | {fmt(t['zg6'])} | {fmt(t['zrckp'])} | "
                  f"{fmt(share[k])} | {fmt(t['floor'])} | {c} | {fmt(t['zspk_host'])} |")
    md += ["", "zg6 minus zspk: R@5 with its 95% question-bootstrap interval, FC@5 and hit@1 (U2d's compares; the call "
               "at 0.0075, before the re-call):", "",
           "| split | dataset | read | delta R@5 [95% CI] | delta FC@5 | delta hit@1 | call (0.0075) |",
           "|---|---|---|---|---|---|---|"]
    for sp in SPLITS:
        if vs[sp] is None:
            md.append(f"| {sp} | (no compare {G6_CMP[sp]}) | | | | | |")
            continue
        for r in vs[sp]:
            d, lo, hi = r["delta"], r["lo"], r["hi"]
            md.append(f"| {sp} | {r['dataset']} | {r['read']} | {d[0]:+.4f} [{lo[0]:+.4f}, {hi[0]:+.4f}] | {d[1]:+.4f} | "
                      f"{d[2]:+.4f} | {r['call']} |")
    od = Path(out or OUT)
    od.mkdir(parents=True, exist_ok=True)
    LC.write_json(od / f"{VERDICT}.json", rec)
    (od / f"{VERDICT}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    log(f"zgl G6 verdict: {v} ({len(gains)} GAIN, {len(losses)} LOSS{'; ' + '; '.join(problems) if problems else ''}); "
        f"next: {nxt}")
    return rec


def g6(screen=None, out_root=None, out=None, root=None):
    pair(screen, out_root)
    recall(screen, out_root)
    return verdict(screen, out, root)


# ── selftest ─────────────────────────────────────────────────────────────────


def fake_read(td, name, dev):
    """A fit's read.json in U2d's form (the stamp zu2d.fit_device reads)."""
    d = Path(td) / "fits" / name
    d.mkdir(parents=True, exist_ok=True)
    LC.write_json(d / "read.json", {"u2d": {"declared_in": Z.DECLARED, "fit": name, "device": dev}})


def fake_g6(td, deltas, big=None, dev=None, base_r5=0.6):
    """U2d's files for G6's call in td: the two zg6-vs-zspk compares (deltas {(split, dataset): d}), zspk's and zrckp's
    nulls (big {(split, dataset): (D1, D2)} for zspk's), and the read stamps of every fit; all on dev."""
    dev = dev or Z.device_of(Z.TOY_ENV)
    for sp in SPLITS:
        p = Z.fake_cmp(td, G6_CMP[sp], G6_FIT[sp], G6_BASE[sp], {ds: d for (s, ds), d in deltas.items() if s == sp},
                       base_r5=base_r5)
        rec = json.loads(p.read_text(encoding="utf-8"))
        rec["u2d"]["device"] = dev
        p.write_text(json.dumps(rec), encoding="utf-8")
        base = Path(td) / "fits" / G6_BASE[sp]
        base.mkdir(parents=True, exist_ok=True)
        (base / "screen.json").write_text(json.dumps(
            {"arm": BASE, "split": sp, "seed": 0, "u2d": {"declared_in": Z.DECLARED, "fit": G6_BASE[sp], "build": "base",
                                                          "arm": BASE, "seed": 0, "split": sp, "device": dev}}),
            encoding="utf-8")
    Z.fake_null(td, NULL_ARM, big, base_r5=base_r5)
    Z.fake_null(td, "zrckp", base_r5=0.59)
    for n in fits():
        fake_read(td, n, dev)


def move_fit(td, name, dev):
    """Both stamps of one fit (screen.json and read.json) moved to device dev."""
    sj = Path(td) / "fits" / name / "screen.json"
    rec = json.loads(sj.read_text(encoding="utf-8"))
    rec["u2d"]["device"] = dev
    sj.write_text(json.dumps(rec), encoding="utf-8")
    fake_read(td, name, dev)


def selftest():
    t0 = time.time()
    W, GN, LS = "WITHIN", "GAIN", "LOSS"
    real = [ROOT / SCREEN / f"{PAIR}.json", ROOT / SCREEN / f"{RECALL}.json", ROOT / OUT]
    before = {str(p): p.stat().st_mtime if p.exists() else None for p in real}
    saved = SP.load, SR.load_null
    ms, hp = SPLITS
    # filed-like: GAINs on L-musique metaqa and L-hotpotqa musique; LOSSes on both webqsp reads and L-hotpotqa hotpotqa;
    # L-hotpotqa 2wiki a GAIN at 0.0075 below its raised floor (0.0139); L-hotpotqa webqsp a LOSS above its 0.0405
    dz = {(ms, "metaqa"): 0.0102, (ms, "webqsp"): -0.0795, (hp, "musique"): 0.0138, (hp, "hotpotqa"): -0.0203,
          (hp, "2wiki"): 0.0101, (hp, "webqsp"): -0.0584, (ms, "musique"): 0.0063}
    big = {(hp, "2wiki"): (0.007, -0.0069), (hp, "webqsp"): (0.02, -0.0205), (hp, "hotpotqa"): (0.0072, 0.0073),
           (ms, "musique"): (0.004, -0.0045)}
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        fake_g6(td, dz, big)
        rec = g6(td, td / "fits", td / "gl", td)
        assert rec["verdict"] == "NOT_ADOPTED", rec["verdict"]
        assert rec["gains"] == [f"{ms} metaqa", f"{hp} musique"], rec["gains"]
        assert rec["losses"] == [f"{ms} webqsp", f"{hp} hotpotqa", f"{hp} webqsp"], rec["losses"]
        assert rec["next_form"] == G6.NEXT["GAIN_AND_LOSS"]
        assert rec["calls"][f"{hp} 2wiki"] == W and rec["beside"]["r5"][f"{hp} 2wiki"]["call_filed"] == GN
        assert abs(rec["floors"][f"{hp} 2wiki"] - 0.0139) < 1e-4 and rec["floors"][f"{ms} metaqa"] == S.FLOOR
        assert rec["device"] == Z.device_of(Z.TOY_ENV) and not rec["problems"]
        r5 = rec["beside"]["r5"][f"{ms} metaqa"]
        assert (r5["zspk"], r5["zg6"], r5["zrckp"], r5["zspk_host"]) == (0.6, 0.6102, 0.59, None)
        assert rec["beside"]["mlp_share"][f"{ms} metaqa"] == round((0.59 - 0.4) / (0.6102 - 0.4), 4)
        assert all(len(rec["beside"]["zg6_minus_zspk"][sp]) == 6 for sp in SPLITS)
        pr = json.loads((td / f"{RECALL}.json").read_text(encoding="utf-8"))
        assert pr["stage"] == STAGE and pr["null_kind"] == NULL_KIND and pr["device"] == rec["device"]
        assert pr["verdict"] == "MIXED" and "U2d zspk L-musique seed 1" in pr["null"]
        md = (td / f"{RECALL}.md").read_text(encoding="utf-8")
        assert md.startswith("GL (") and "zspk's seeds 1 and 2" in md and "step 1's p@swa" not in md
        assert (td / "gl" / f"{VERDICT}.md").read_text(encoding="utf-8").startswith("# G6's verdict on the L40S")
        # the host card's zspk beside, when L1's compare is there
        l1 = td / "outputs" / "l1"
        l1.mkdir(parents=True)
        for sp in SPLITS:
            rows = [{"dataset": ds, "new": [0.55, 0, 0]} for ds in LG.EVAL_ORDER]
            LC.write_json(l1 / f"base-zspk-{sp}.json", {"new": f"outputs/screen/fits/{G6.BASE_FITS[sp][2]}",
                                                        "bases": ["b"], "by_base": {"b": {"rows": rows}}})
        assert verdict(td, td / "gl", td)["beside"]["r5"][f"{hp} squad"]["zspk_host"] == 0.55
        # INCOMPLETE: a re-call that is not GL's, one without twelve reads, none
        x = json.loads((td / f"{RECALL}.json").read_text(encoding="utf-8"))
        LC.write_json(td / f"{RECALL}.json", {**x, "null_kind": "N2"})
        assert verdict(td, td / "gl", td)["verdict"] == "INCOMPLETE"
        LC.write_json(td / f"{RECALL}.json", {**x, "rows": x["rows"][:11]})
        v = verdict(td, td / "gl", td)
        assert v["verdict"] == "INCOMPLETE" and v["next_form"] == G6.NEXT["INCOMPLETE"]
        (td / f"{RECALL}.json").unlink()
        assert verdict(td, td / "gl", td)["verdict"] == "INCOMPLETE"
    # ADOPT: GAINs and no LOSS; a LOSS inside its raised floor is WITHIN
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        fake_g6(td, {(ms, "metaqa"): 0.0102, (hp, "webqsp"): -0.0300}, {(hp, "webqsp"): (0.02, -0.0205)})
        rec = g6(td, td / "fits", td / "gl", td)
        assert rec["verdict"] == "ADOPT_FOR_FULL_RUN" and rec["next_form"] == G6.NEXT["ADOPT_FOR_FULL_RUN"]
        assert rec["calls"][f"{hp} webqsp"] == W and rec["beside"]["r5"][f"{hp} webqsp"]["call_filed"] == LS
    # NO_GAIN: every read within
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        fake_g6(td, {(ms, "squad"): 0.003})
        rec = g6(td, td / "fits", td / "gl", td)
        assert rec["verdict"] == "NOT_ADOPTED" and rec["next_form"] == G6.NEXT["NO_GAIN"] and not rec["gains"]
    # the refusals
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        fake_g6(td, {(ms, "metaqa"): 0.0102})
        dev = Z.device_of(Z.TOY_ENV)
        cmp_ = td / f"{G6_CMP[ms]}.json"
        good = cmp_.read_text(encoding="utf-8")

        def bad(**edit):
            c = json.loads(good)
            for k, val in edit.items():
                if k == "u2d":
                    c["u2d"].update(val)
                else:
                    c[k] = val
            cmp_.write_text(json.dumps(c), encoding="utf-8")
            Z.stops(pair, td, td / "fits", has="zgl:")
            cmp_.write_text(good, encoding="utf-8")

        bad(u2d={"declared_in": "docs/OTHER.md"})
        bad(u2d={"compare": "scr-u2d-new-zg6-vs-zspk"})
        bad(u2d={"device": {**dev, "gpu": "NVIDIA T4"}})
        bad(bases=["outputs/screen/fits/scr-u2d-base-zrckp"])
        bad(new=str(td / "fits" / G6_FIT[hp]))
        bad(candidate="p@last")
        bad(carve="s1sel")
        bad(trained_on=["squad"])
        c = json.loads(good)
        b0 = c["bases"][0]
        c["by_base"][b0]["rows"] = c["by_base"][b0]["rows"][:5]
        cmp_.write_text(json.dumps(c), encoding="utf-8")
        Z.stops(pair, td, td / "fits", has="six datasets")
        cmp_.write_text(good, encoding="utf-8")
        # a null fit on another image, or read on another device than it was fitted on, stops the call before anything
        # is written
        n1 = Z.fit_name("base", NULL_ARM, 2, hp)
        move_fit(td, n1, {**dev, "image": "other"})
        Z.stops(pair, td, td / "fits", has="different devices")
        Z.stops(recall, td, td / "fits", has="different devices")
        move_fit(td, n1, dev)
        fake_read(td, G6_BASE[ms], {**dev, "gpu": "NVIDIA T4"})
        Z.stops(pair, td, td / "fits", has="on the device it was fitted on")
        assert not (td / f"{PAIR}.json").exists()
        fake_read(td, G6_BASE[ms], dev)
        pair(td, td / "fits")
        # a pair that is not GL's stops the re-call
        x = json.loads((td / f"{PAIR}.json").read_text(encoding="utf-8"))
        LC.write_json(td / f"{PAIR}.json", {**x, "stage": "G6"})
        Z.stops(recall, td, td / "fits", has="not GL's pair")
    assert (SP.load, SR.load_null) == saved, "screen_pair's load and screen_recall's null are restored"
    after = {str(p): p.stat().st_mtime if p.exists() else None for p in real}
    assert after == before, "nothing is written under GL's or the screens' real folders"
    print(f"zgl selftest: OK ({time.time() - t0:.1f}s)", flush=True)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="GL: G6 decided on U2d's L40S fits (docs/GL_GPU_LINE_ON_L40S.md)")
    ap.add_argument("cmd", nargs="?", choices=("g6", "pair", "recall", "verdict"))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if not a.cmd:
        ap.error("a command (g6, pair, recall, verdict) or --selftest")
    with Z.at_root():
        if a.cmd in ("g6", "pair"):
            pair()
        if a.cmd in ("g6", "recall"):
            recall()
        if a.cmd in ("g6", "verdict"):
            return 2 if verdict()["verdict"] == "INCOMPLETE" else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
