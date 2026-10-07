"""Screens, tenth round (docs/SCREENS.md): ztop50 on rel's base, on lean_screen2's commands and lean_screen's rule,
unchanged; two fits per screen (L-musique and L-hotpotqa), each compared with rel's screen fit of its split (that
comparison decides) and with step 1's (reported); their verdict over both by screen_pair.py's rule, and its re-call
under the seed null by screen_recall.py's, each decided against rel's screen fits (pair, recall; regrade for the full
run's grade). Both files are used unchanged: only their loaders are swapped, inside against_rel().

    python outputs/mp_unified/relz.py smoke --device cuda --host
    python outputs/mp_unified/relz.py train --split L-musique --name scr-relz --arm relz --device cuda --host
    python outputs/mp_unified/relz.py read --name scr-relz --device cuda --host
    python outputs/mp_unified/relz.py compare --new outputs/screen/fits/scr-relz \\
        --base outputs/screen/fits/scr-rel,outputs/step1/fits/L-musique --out outputs/screen/scr-relz
    python outputs/mp_unified/relz.py pair --screens outputs/screen/scr-relz.json,outputs/screen/scr-relz-hp.json \\
        --out outputs/screen/scr-relz-pair
    python outputs/mp_unified/relz.py recall --null N1,N2,N3,N4 --pair outputs/screen/scr-relz-pair.json \\
        --out outputs/screen/scr-relz-pair-recall
    python outputs/mp_unified/relz.py regrade --null N1,N2,N3,N4 --grade outputs/full_relz/grade.json \\
        --out outputs/full_relz/grade-recall
    python outputs/mp_unified/relz.py --selftest
N1..N4 are the seed null's comparisons (outputs/screen/scr-null-s1, -s2, -s1-hp, -s2-hp .json). A read's floor is the
null's; its base R@5 is rel's screen fit's (outputs/screen/scr-rel.json, scr-rel-hp.json, which were compared with the
null's seed 0), in place of step 1's seed 0.

The arm:
  relz   rel's carve and block sets (relcols.py: step 1's nine blocks, then typed_rel, typed_v2 and ordered from its
         build) under ztop50's model (lean_screen4.ZTop): every block's within-pool z-score, rel's blocks' included,
         and rrf's base z-score are taken against the pool's top 50 retrieved rows by rrf (a pool with fewer than two
         such rows, or a column with no spread over them, keeps the whole-pool z-score). Both parts are used
         unchanged; no new column, block or hyperparameter. A fit that trains no typed graph drops rel's blocks by the
         dead-block rule and is ztop50's fit.
"""
import os
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import contextlib  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import relcols as R  # noqa: E402
import lean_screen4 as S4  # noqa: E402
import screen_pair as SP  # noqa: E402
import screen_recall as SR  # noqa: E402

S2, S, S3 = R.S2, R.S, S4.S3
LG, LC = S.LG, S.LC
log = S.log
ARM = "relz"
BASE_ARM = "rel"
S.ARMS.update({ARM: (S4.ZTop, R.RelCarveRel)})


# ── train and read (lean_screen2's, under rel's block sets) ──────────────────


def train(argv, split):
    arm = R.arm_of(argv)
    if arm != ARM:
        raise SystemExit(f"relz: train takes --arm {ARM}, not {arm}")
    with R.with_sets(BASE_ARM):
        rc = S2.train(argv, split)
    name, out_root = S2.where(argv)
    sj = out_root / name / "screen.json"
    if sj.exists():
        rec = json.loads(sj.read_text(encoding="utf-8"))
        rec["relz_sha256"] = LC.sha_src(__file__)
        rec["relcols_sha256"] = LC.sha_src(R.__file__)
        rec["lean_screen4_sha256"] = LC.sha_src(S4.__file__)
        rec["relz_blocks"] = list(R.ARM_BLOCKS[BASE_ARM])
        rec["relz_K"] = S4.ZTop.K
        rec["relcols_records"] = R.built_records()
        LC.write_json(sj, rec)
    return rc


def read(argv):
    name, out_root = S2.where(argv)
    arm = json.loads((out_root / name / "screen.json").read_text(encoding="utf-8"))["arm"]
    if arm != ARM:
        raise SystemExit(f"relz: {name} was trained as {arm}, not {ARM}")
    with R.with_sets(BASE_ARM):
        return S2.main(["read"] + argv)


# ── the full run's gate and grade (docs/FULL_ROUND10.md) ─────────────────────


REL_FITS = {"L-musique": ("screen", "fits", "scr-rel"), "L-hotpotqa": ("screen", "fits", "scr-rel-hp")}


def rel_fit(split):
    """rel's fit of a split, the base relz is decided against: its screen fits, else its full run's fits."""
    return REL_FITS.get(split, ("full_rel", "fits", split))


def gate(recall, rel_grade):
    """Exit 0 when relz's re-call is PROMISING and rel's re-grade is ADOPT, else 1 (the feeder then drops the run)."""
    vs = []
    for p, want in ((recall, "PROMISING"), (rel_grade, "ADOPT")):
        p = Path(p)
        v = json.loads(p.read_text(encoding="utf-8")).get("verdict") if p.exists() else None
        vs.append(v)
        log(f"relz gate: {p} {v} (needs {want})")
    return 0 if vs == ["PROMISING", "ADOPT"] else 1


def grade(full_root, reuse, out=None, check_fits=True):
    """lean_screen2.grade's rule, decided against rel's fit of each split in place of step 1's: ADOPT when a primary
    read GAINs and no read of the thirty-six LOSEs, NOT_ADOPTED otherwise; INCOMPLETE (exit 1) when a comparison is
    missing or not the declared one."""
    full_root = Path(full_root)
    rows, problems, sources = [], [], {}
    for split in S2.SPLITS:
        fn = Path(reuse.get(split) or full_root / f"compare-{split}.json")
        sources[split] = str(fn)
        if not fn.exists():
            problems.append(f"{split}: {fn} missing")
            continue
        rec = json.loads(fn.read_text(encoding="utf-8"))
        if S2.parts_of(rec["bases"][0])[-3:] != rel_fit(split):
            problems.append(f"{split}: decided against {rec['bases'][0]}, not rel's {'/'.join(rel_fit(split))}")
        if rec["candidate"] != "p@swa" or rec["carve"] != "s1eval":
            problems.append(f"{split}: {rec['candidate']} on {rec['carve']}, not p@swa on s1eval")
        if sorted(rec["trained_on"]) != sorted(LG.FITS[split]):
            problems.append(f"{split}: trained on {rec['trained_on']}, not {sorted(LG.FITS[split])}")
        if check_fits:
            sj = S2.ROOT / Path(*S2.parts_of(rec["new"])) / "screen.json"
            if not sj.exists():
                problems.append(f"{split}: {sj} missing")
            else:
                s = json.loads(sj.read_text(encoding="utf-8"))
                if s.get("arm") != ARM or s.get("split") != split:
                    problems.append(f"{split}: the fit is arm {s.get('arm')} on {s.get('split')}, not {ARM} on {split}")
        for r in rec["by_base"][rec["bases"][0]]["rows"]:
            rows.append({"split": split, "dataset": r["dataset"], "read": r["read"],
                         "primary": S2.primary(split, r["dataset"]), "base": r["base"][0], "new": r["new"][0],
                         "rrf": r["rrf"][0], "delta": r["delta"][0], "lo": r["lo"][0], "hi": r["hi"][0],
                         "delta_fc5": r["delta"][1], "delta_hit1": r["delta"][2], "call": r["call"]})
    n_prim = sum(r["primary"] for r in rows)
    if len(rows) != len(S2.SPLITS) * len(LG.EVAL_ORDER) or n_prim != 11:
        problems.append(f"{len(rows)} reads ({n_prim} primary), not 36 (11)")
    gains = [r for r in rows if r["primary"] and r["call"] == "GAIN"]
    losses = [r for r in rows if r["call"] == "LOSS"]
    v = "INCOMPLETE" if problems else "ADOPT" if gains and not losses else "NOT_ADOPTED"
    rec = {"arm": ARM, "base_arm": BASE_ARM, "verdict": v, "problems": problems, "sources": sources,
           "primary_gains": len(gains), "losses": len(losses), "rows": rows, "floor": S.FLOOR,
           "script_sha256": LC.sha_src(__file__), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    md = [f"# Full run of {ARM} against {BASE_ARM}: **{v}**", "",
          f"R@5 of {ARM}'s p@swa minus {BASE_ARM}'s p@swa of the same split, on the six s1eval carves (95% "
          f"question-bootstrap interval; call by docs/SCREENS.md's rule). Bold: step 1's eleven primary reads. zs: read "
          f"zero-shot.", "", "| split | " + " | ".join(LG.EVAL_ORDER) + " |", "|---|" + "---|" * len(LG.EVAL_ORDER)]
    for split in S2.SPLITS:
        cells = []
        for ds in LG.EVAL_ORDER:
            r = next((x for x in rows if x["split"] == split and x["dataset"] == ds), None)
            if r is None:
                cells.append("—")
                continue
            c = f"{r['delta']:+.4f} {r['call']}{' zs' if r['read'] == 'zero-shot' else ''}"
            cells.append(f"**{c}**" if r["primary"] else c)
        md.append(f"| {split} | " + " | ".join(cells) + " |")
    md += ["", f"Primary reads with a GAIN: {len(gains)} of {n_prim}. Reads with a LOSS: {len(losses)} of {len(rows)}.",
           "ADOPT when at least one primary read GAINs and none of the thirty-six LOSEs (docs/FULL_ROUND10.md)."]
    if problems:
        md += ["", "Problems:"] + [f"- {p}" for p in problems]
    if out:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        LC.write_json(out.with_suffix(".json"), rec)
        out.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    log(f"grade {ARM}: {v}; {len(gains)} primary GAIN, {len(losses)} LOSS" + (f"; problems {problems}" if problems else ""))
    return rec


# ── the pair, its re-call and the full run's re-grade, against rel's screen fits ─


REL_SCREENS = {"L-musique": "outputs/screen/scr-rel.json", "L-hotpotqa": "outputs/screen/scr-rel-hp.json"}
NULL_OF = SR.load_null          # screen_recall's own loader (against_rel swaps the module's name)


def pair_load(fn):
    """screen_pair.load's checks, the deciding base rel's screen fit of the split in place of step 1's."""
    p = Path(fn)
    if not p.exists():
        raise SystemExit(f"relz: {p} missing")
    rec = json.loads(p.read_text(encoding="utf-8"))
    b0 = S2.parts_of(rec["bases"][0])[-3:]
    split = next((sp for sp, f in REL_FITS.items() if f == b0), None)
    if split is None:
        raise SystemExit(f"relz: {p} is decided against {rec['bases'][0]}, not rel's screen fit of "
                         f"{' or '.join(REL_FITS)}")
    if sorted(rec["trained_on"]) != sorted(LG.FITS[split]):
        raise SystemExit(f"relz: {p} trained on {rec['trained_on']}, not {split}'s {sorted(LG.FITS[split])}")
    if rec["candidate"] != "p@swa" or rec["carve"] != "s1eval":
        raise SystemExit(f"relz: {p} reads {rec['candidate']} on {rec['carve']}, not p@swa on s1eval")
    rows = rec["by_base"][rec["bases"][0]]["rows"]
    if sorted(r["dataset"] for r in rows) != sorted(LG.EVAL_ORDER):
        raise SystemExit(f"relz: {p} reads {[r['dataset'] for r in rows]}, not the six datasets")
    return split, rows


def rel_null(null_files, rel_screens=None):
    """The seed null's floors (screen_recall's), each read's base R@5 rel's screen fit's in place of step 1's seed 0.
    rel's screens must be its screen fits compared with the null's seed 0."""
    null, sources = NULL_OF(null_files)
    screens = rel_screens or REL_SCREENS
    if sorted(screens) != sorted(SR.NULL_SPLITS):
        raise SystemExit(f"relz: rel's screens cover {sorted(screens)}, not {sorted(SR.NULL_SPLITS)}")
    for split, fn in screens.items():
        p = Path(fn) if Path(fn).is_absolute() else S2.ROOT / fn
        if not p.exists():
            raise SystemExit(f"relz: {p} missing")
        rec = json.loads(p.read_text(encoding="utf-8"))
        if S2.parts_of(rec["new"])[-3:] != REL_FITS[split] or \
                S2.parts_of(rec["bases"][0])[-3:] != ("step1", "fits", split):
            raise SystemExit(f"relz: {p} compares {rec['new']} with {rec['bases'][0]}, not rel's screen fit of {split} "
                             f"with step 1's")
        rows = {r["dataset"]: r for r in rec["by_base"][rec["bases"][0]]["rows"]}
        if sorted(rows) != sorted(LG.EVAL_ORDER):
            raise SystemExit(f"relz: {p} reads {sorted(rows)}, not the six datasets")
        for ds, v in null[split].items():
            if rows[ds]["base"][0] != v["seed0"]:
                raise SystemExit(f"relz: {p} {ds} is compared with seed-0 R@5 {rows[ds]['base'][0]}, the null with "
                                 f"{v['seed0']}")
            v["step1_seed0"], v["seed0"] = v["seed0"], rows[ds]["new"][0]
        sources[f"rel {split}"] = str(p)
    return null, sources


@contextlib.contextmanager
def against_rel(rel_screens=None):
    """screen_pair's load and screen_recall's null, decided against rel's screen fits; restored on the way out."""
    saved = SP.load, SR.load_null
    SP.load, SR.load_null = pair_load, (lambda files: rel_null(files, rel_screens))
    try:
        yield
    finally:
        SP.load, SR.load_null = saved


def tag(rec):
    rec.update({"arm": ARM, "base_arm": BASE_ARM, "decided_against": "rel's fit of each split",
                "relz_sha256": LC.sha_src(__file__)})
    return rec


def pair(screens, out=None):
    """screen_pair.pair over relz's fits, each decided against rel's screen fit of its split."""
    with against_rel():
        rec = tag(SP.pair(screens))
    rows = rec["rows"]
    md = [f"# Screen over {len(rec['fits'])} fits: **{rec['verdict']}**", "",
          f"R@5 of {ARM}'s p@swa minus {BASE_ARM}'s p@swa of the same split (rel's screen fits scr-rel and scr-rel-hp), "
          "on the six s1eval carves (95% question bootstrap; docs/SCREENS.md's rule over every read). zs: read "
          "zero-shot.", "",
          "| split | " + " | ".join(LG.EVAL_ORDER) + " | fit's own verdict |", "|---|" + "---|" * (len(LG.EVAL_ORDER) + 1)]
    for f in rec["fits"]:
        cells = []
        for ds in LG.EVAL_ORDER:
            r = next(x for x in rows if x["split"] == f["split"] and x["dataset"] == ds)
            cells.append(f"{r['delta']:+.4f} {r['call']}{' zs' if r['read'] == 'zero-shot' else ''}")
        md.append(f"| {f['split']} | " + " | ".join(cells) + f" | {f['verdict']} |")
    md += ["", f"Reads with a GAIN: {sum(r['call'] == 'GAIN' for r in rows)} of {len(rows)}; with a LOSS: "
               f"{sum(r['call'] == 'LOSS' for r in rows)}. PROMISING needs at least one GAIN and no LOSS."]
    SR.write(out, rec, md)
    return rec


def tail(rec):
    return ["Changed calls: " + ("; ".join(rec["changed"]) or "none") + ".",
            f"Within {SR.ROUNDING} of the floor (the call stands): " + ("; ".join(rec["near_floor"]) or "none") + "."]


def recall(null_files, pair_file, out=None, rel_screens=None):
    """screen_recall.pair against rel's screen fits: each read of relz's pair called with the seed null's floor."""
    with against_rel(rel_screens):
        rec = tag(SR.pair(null_files, pair_file))
    rows = rec["rows"]
    md = [f"# {Path(pair_file).stem} re-called under the seed null: **{rec['verdict']}** (filed: "
          f"{rec['verdict_filed']})", "",
          f"R@5 of {ARM}'s p@swa minus {BASE_ARM}'s p@swa of the same split (rel's screen fits), on the six s1eval "
          "carves (95% question bootstrap), called with each read's floor: max(0.0075, twice the seed-only spread of "
          "step 1's seeds 1 and 2 against seed 0) (docs/SCREENS.md, section 2 and the tenth round). D1, D2: the null's "
          "seed-only differences on the read. Bold: a changed call.", "",
          "| split | dataset | read | rel R@5 | relz R@5 | delta R@5 [95% CI] | D1, D2 | floor | filed | re-call |",
          "|---|---|---|---:|---:|---|---|---:|---|---|"]
    for r in rows:
        c = f"**{r['call']}**" if r["call"] != r["call_filed"] else r["call"]
        md.append(f"| {r['split']} | {r['dataset']} | {r['read']} | {r['base']:.4f} | {r['new']:.4f} | "
                  f"{r['delta']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}] | {r['null_d'][0]:+.4f}, {r['null_d'][1]:+.4f} | "
                  f"{r['floor']:.4f} | {r['call_filed']} | {c} |")
    md += ["", f"Re-call: reads with a GAIN: {sum(r['call'] == 'GAIN' for r in rows)} of {len(rows)}; with a LOSS: "
               f"{sum(r['call'] == 'LOSS' for r in rows)}. PROMISING needs at least one GAIN and no LOSS.",
           "Each fit, filed -> re-called: " + "; ".join(f"{f['split']} {f['verdict_filed']} -> {f['verdict']}"
                                                         for f in rec["fits"]) + "."] + tail(rec)
    SR.write(out, rec, md)
    return rec


def regrade(null_files, grade_file, out=None, rel_screens=None):
    """screen_recall.grade against rel's fits: the grade's reads on L-musique and L-hotpotqa (the screen's reused fits)
    called with the seed null's floors; the other splits keep 0.0075."""
    with against_rel(rel_screens):
        rec = tag(SR.grade(null_files, grade_file))
    rows = rec["rows"]
    md = [f"# Full run of {ARM} against {BASE_ARM} re-called under the seed null: **{rec['verdict']}** (filed: "
          f"{rec['verdict_filed']})", "",
          f"R@5 of {ARM}'s p@swa minus {BASE_ARM}'s p@swa of the same split. The reads on "
          f"{' and '.join(SR.NULL_SPLITS)} (the screen's reused fits, against rel's screen fits) are called with the "
          "seed null's floors (docs/SCREENS.md, section 2); the other splits keep 0.0075, since no null covers them. "
          "Bold: step 1's eleven primary reads. zs: read zero-shot. A changed call shows filed -> re-call.", "",
          "| split | " + " | ".join(LG.EVAL_ORDER) + " |", "|---|" + "---|" * len(LG.EVAL_ORDER)]
    for sp in dict.fromkeys(r["split"] for r in rows):
        cells = []
        for ds in LG.EVAL_ORDER:
            r = next((x for x in rows if x["split"] == sp and x["dataset"] == ds), None)
            if r is None:
                cells.append("—")
                continue
            c = r["call"] if r["call"] == r["call_filed"] else f"{r['call_filed']} -> {r['call']}"
            c = f"{r['delta']:+.4f} {c}{' zs' if r['read'] == 'zero-shot' else ''}"
            if r["covered"]:
                c += f" (floor {r['floor']:.4f})"
            cells.append(f"**{c}**" if r["primary"] else c)
        md.append(f"| {sp} | " + " | ".join(cells) + " |")
    md += ["", f"Primary reads with a GAIN: {rec['primary_gains']}. Reads with a LOSS: {rec['losses']} of {len(rows)}. "
               f"Reads under the null's floors: {rec['covered']}; under 0.0075: {len(rows) - rec['covered']}.",
           "ADOPT when at least one primary read GAINs and none of the thirty-six LOSEs (docs/FULL_ROUND10.md)."] + tail(rec)
    if rec.get("problems"):
        md += ["", "The filed grade's problems:"] + [f"- {x}" for x in rec["problems"]]
    SR.write(out, rec, md)
    return rec


# ── smoke ────────────────────────────────────────────────────────────────────


def smoke(device, host, out_root=None):
    """rel and relz each trained for one epoch on metaqa's select carve (relz twice: the repeat must be IDENTICAL) and
    read on it. relz's fit must hold rel's blocks, and its scores must differ from rel's (the same init and batches,
    so a difference is ztop50's forward)."""
    t0 = time.time()
    root = Path(out_root or S.OUT / "smoke10")
    h = ["--host"] if host else []
    common = ["--train", "metaqa=select", "--basis", "2wiki", "--variants", "p", "--config", "2e-3:1e-4:0.1:1:0",
              "--device", device, "--out-root", str(root)]
    rec, ok = {}, True
    for arm, mod, rep in ((BASE_ARM, R, "1"), (ARM, sys.modules[__name__], "2")):
        nm = f"smoke-{arm}"
        rc = mod.main(["train", "--name", nm, "--arm", arm, "--repeat", rep] + common + h)
        tj = json.loads((root / nm / "train.json").read_text(encoding="utf-8"))
        sj = json.loads((root / nm / "screen.json").read_text(encoding="utf-8"))
        blocks = tj["variants"]["p"]["blocks"]
        rr = mod.main(["read", "--name", nm, "--read", "metaqa=select", "--device", device, "--out-root", str(root)] + h)
        rec[arm] = {"train_rc": rc, "repeats": int(rep), "blocks": blocks, "screen_arm": sj.get("arm"), "read_rc": rr,
                    "rel_blocks_live": all(b in blocks for b in R.ARM_BLOCKS[BASE_ARM])}
        ok = ok and rc == 0 and rr == 0 and rec[arm]["rel_blocks_live"] and sj.get("arm") == arm
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
    rec[ARM]["repeat"] = "IDENTICAL" if rec[ARM]["train_rc"] == 0 else "DIFFERENT"
    fr = root / f"smoke-{BASE_ARM}" / "reads" / "metaqa__select.npz"
    fz = root / f"smoke-{ARM}" / "reads" / "metaqa__select.npz"
    if fr.exists() and fz.exists():
        a, b = np.load(fr), np.load(fz)
        same_q = [str(x) for x in a["ids"]] == [str(x) for x in b["ids"]]
        differs = bool(not np.array_equal(a["scores64"], b["scores64"]))
        rec["scores_differ_from_rel"] = differs
        rec["same_questions"] = same_q
        ok = ok and same_q and differs and rec[BASE_ARM]["blocks"] == rec[ARM]["blocks"]
    else:
        rec["scores_differ_from_rel"] = None
        ok = False
    rec.update({"ok": ok, "seconds": time.time() - t0, "script_sha256": LC.sha_src(__file__)})
    LC.write_json(root / "smoke.json", rec)
    log(f"smoke10: {json.dumps(rec)}; {'ok' if ok else 'FAILED'}")
    return 0 if ok else 1


# ── selftest ─────────────────────────────────────────────────────────────────


def selftest():
    t0 = time.time()
    LG.bind_device_ops()
    # 1. the arm is ztop50's model on rel's carve; rel's and ztop50's own arms are unchanged
    assert S.ARMS[ARM] == (S4.ZTop, R.RelCarveRel)
    assert S.ARMS[BASE_ARM] == (S.ARMS["base"][0], R.RelCarveRel) and S.ARMS["ztop50"] == (S4.ZTop, S.ARMS["base"][1])
    # 2. the block sets: rel's blocks after step 1's inside the context, step 1's sets restored after it
    saved = {k: list(v) for k, v in LG.SETS.items()}
    with R.with_sets(BASE_ARM):
        for k in saved:
            assert LG.SETS[k][:len(saved[k])] == saved[k]
            assert LG.SETS[k][len(saved[k]):] == [b for b in R.ARM_BLOCKS[BASE_ARM] if b not in saved[k]]
    assert {k: list(v) for k, v in LG.SETS.items()} == saved
    # 3. another arm is refused
    try:
        train(["train", "--name", "x", "--arm", BASE_ARM], "L-musique")
        raise AssertionError("relz trained another arm")
    except SystemExit as e:
        assert "takes --arm relz" in str(e)
    # 4. the forward on a fake rel carve: every block's z-score, rel's included, and rrf's base z-score are taken
    #    against the top-K reference rows (each call of seg_zscore_ref recorded)
    c = R.fake_carve(R.RelCarveRel)
    blocks = ["rank", "WALK"]
    qs = np.array([2, 0, 1])
    feats, nq, base_z, gold = c.batch(qs, blocks + list(R.ARM_BLOCKS[BASE_ARM]))
    m = S4.ZTop(blocks + list(R.ARM_BLOCKS[BASE_ARM]), c.widths, 8, ctx="none")
    m.eval()
    calls, orig = [], S3.seg_zscore_ref

    def spy(x, nq_, B_, ref, eps=1e-6):
        calls.append((x.shape[1], ref.clone()))
        return orig(x, nq_, B_, ref, eps)

    S3.seg_zscore_ref = spy
    try:
        for k_ in (S4.ZTop.K, 2):
            calls.clear()
            m.K = k_
            keep = torch.ones((qs.size, len(m.blocks)))
            s = m(feats, keep, nq, qs.size, base_z)
            assert s.shape == (int(c.n_np[qs].sum()),) and torch.isfinite(s).all()
            want = S4.top_ref(feats["rank"][:, S3.I_RRF], nq, qs.size, k_)
            assert [w for w, _ in calls] == [c.widths[b] for b in m.blocks] + [1]
            assert all(torch.equal(r, want) for _, r in calls)
        assert int(want.sum()) < int((feats["rank"][:, S3.I_RRF] > 0).sum())    # K=2 cuts a pool's reference
    finally:
        S3.seg_zscore_ref = orig
        m.K = S4.ZTop.K
    assert m.blocks[-len(R.ARM_BLOCKS[BASE_ARM]):] == list(R.ARM_BLOCKS[BASE_ARM])
    # 5. the grade decides against rel's fit of each split; the gate needs both verdicts
    import shutil
    import tempfile
    tmp = Path(tempfile.mkdtemp(prefix="relz_"))
    try:
        def comp(split, base, calls):
            rows = [{"dataset": ds, "read": "in-domain" if ds in LG.FITS[split] else "zero-shot",
                     "base": [0.5, 0.4, 0.3], "new": [0.5, 0.4, 0.3], "rrf": [0.1, 0.1, 0.1],
                     "delta": [d, 0.0, 0.0], "lo": [d - 0.001, 0.0, 0.0], "hi": [d + 0.001, 0.0, 0.0], "call": cl}
                    for ds, (d, cl) in zip(LG.EVAL_ORDER, calls)]
            return {"new": f"outputs/full_relz/fits/{split}", "bases": [base], "candidate": "p@swa", "carve": "s1eval",
                    "trained_on": sorted(LG.FITS[split]), "by_base": {base: {"rows": rows}}}

        def write_all(prim_gain, loss, wrong_base=None):
            reuse = {}
            for split in S2.SPLITS:
                calls = [(0.0, "WITHIN")] * len(LG.EVAL_ORDER)
                if prim_gain and split == "J5":
                    calls[0] = (0.02, "GAIN")
                if loss and split == "L-squad":
                    calls[1] = (-0.02, "LOSS")
                base = "/".join(("outputs",) + rel_fit(split))
                if split == wrong_base:
                    base = f"outputs/step1/fits/{split}"
                p = tmp / f"compare-{split}.json"
                p.write_text(json.dumps(comp(split, base, calls)), encoding="utf-8")
                if split in REL_FITS:
                    reuse[split] = str(p)
            return reuse

        assert grade(tmp, write_all(True, False), check_fits=False)["verdict"] == "ADOPT"
        assert grade(tmp, write_all(True, True), check_fits=False)["verdict"] == "NOT_ADOPTED"
        assert grade(tmp, write_all(False, False), check_fits=False)["verdict"] == "NOT_ADOPTED"
        bad = grade(tmp, write_all(True, False, wrong_base="L-2wiki"), check_fits=False)
        assert bad["verdict"] == "INCOMPLETE" and "not rel's" in bad["problems"][0]
        bad = grade(tmp, write_all(True, False, wrong_base="L-musique"), check_fits=False)
        assert bad["verdict"] == "INCOMPLETE" and "scr-rel" in bad["problems"][0]
        for rv, gv, want in (("PROMISING", "ADOPT", 0), ("PROMISING", "NOT_ADOPTED", 1), ("MIXED", "ADOPT", 1)):
            (tmp / "rc.json").write_text(json.dumps({"verdict": rv}), encoding="utf-8")
            (tmp / "gr.json").write_text(json.dumps({"verdict": gv}), encoding="utf-8")
            assert gate(tmp / "rc.json", tmp / "gr.json") == want
        assert gate(tmp / "missing.json", tmp / "gr.json") == 1
        # 6. the pair, its re-call and the re-grade decide against rel's screen fits: each read's base R@5 is rel's,
        #    its floor the null's; screen_pair's and screen_recall's own loaders are restored after each call
        null1 = SR.fake_null(tmp / "null", "n1", {("L-hotpotqa", "hotpotqa"): (-0.010, 0.012)})   # floor 0.0221
        null0 = SR.fake_null(tmp / "null", "n0")
        rels = {sp: str(SR.fake_compare(tmp / "screen", nm, sp, {ds: 0.1 for ds in LG.EVAL_ORDER}, arm=BASE_ARM))
                for sp, nm in (("L-musique", "scr-rel"), ("L-hotpotqa", "scr-rel-hp"))}
        rb = {sp: "/".join(("outputs",) + f) for sp, f in REL_FITS.items()}
        za = SR.fake_compare(tmp / "z", "scr-relz", "L-musique", {"musique": 0.0842}, arm=ARM, base_r5=0.6,
                             base=rb["L-musique"])
        zb = SR.fake_compare(tmp / "z", "scr-relz-hp", "L-hotpotqa", {"hotpotqa": -0.0147}, arm=ARM, base_r5=0.6,
                             base=rb["L-hotpotqa"])
        pr = pair([za, zb], tmp / "pz")
        assert pr["verdict"] == "MIXED" and pr["base_arm"] == BASE_ARM and (tmp / "pz.md").exists()
        assert SP.load is not pair_load and SR.load_null is NULL_OF
        SR.must_stop(pair, [za, SR.fake_compare(tmp / "z", "scr-relz-s1", "L-hotpotqa", {}, arm=ARM)])   # step 1's
        rc = recall(null1, tmp / "pz.json", tmp / "rz", rels)
        assert rc["verdict"] == "PROMISING" and rc["changed"] == ["L-hotpotqa hotpotqa LOSS -> WITHIN"], rc["changed"]
        assert all(r["base"] == 0.6 for r in rc["rows"]) and "**WITHIN**" in (tmp / "rz.md").read_text(encoding="utf-8")
        assert recall(null0, tmp / "pz.json", None, rels)["verdict"] == "MIXED"
        bad = SR.fake_compare(tmp / "bad" / "screen", "scr-rel", "L-musique", {ds: 0.1 for ds in LG.EVAL_ORDER},
                              arm=BASE_ARM, base_r5=0.55)
        SR.must_stop(recall, null1, tmp / "pz.json", None, {**rels, "L-musique": str(bad)})   # rel against another 0
        SR.must_stop(recall, null1, tmp / "pz.json", None, {"L-musique": rels["L-musique"]})  # a split missing
        zc = SR.fake_compare(tmp / "z2", "scr-relz", "L-musique", {"musique": 0.0842}, arm=ARM, base_r5=0.61,
                             base=rb["L-musique"])
        pair([zc, zb], tmp / "pz2")
        SR.must_stop(recall, null1, tmp / "pz2.json", None, rels)                  # a base R@5 that is not rel's
        full = tmp / "full"
        full.mkdir()
        for split in S2.SPLITS:
            if split not in REL_FITS:
                d = {"musique": 0.02} if split == "J5" else {}                    # J5 musique: a primary GAIN
                SR.fake_compare(full / "src", f"fit-{split}", split, d, arm=ARM,
                                base="/".join(("outputs",) + rel_fit(split))).replace(full / f"compare-{split}.json")
        g = grade(full, {"L-musique": str(za), "L-hotpotqa": str(zb)}, full / "grade", check_fits=False)
        assert g["verdict"] == "NOT_ADOPTED" and g["losses"] == 1
        rg = regrade(null1, full / "grade.json", tmp / "gz", rels)
        assert rg["verdict"] == "ADOPT" and rg["covered"] == 12 and rg["changed"] == ["L-hotpotqa hotpotqa LOSS -> WITHIN"]
        assert rg["arm"] == ARM and regrade(null0, full / "grade.json", None, rels)["verdict"] == "NOT_ADOPTED"
        assert SP.load is not pair_load and SR.load_null is NULL_OF
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    log(f"relz selftest: the arm is ztop50's model on rel's carve; rel's blocks follow step 1's inside the sets and "
        f"the sets are restored; another arm is refused; every block's z-score (rel's included) and rrf's base "
        f"z-score take the top-K reference rows; the grade decides against rel's fits (a compare against step 1's is "
        f"INCOMPLETE) and the gate needs PROMISING and ADOPT; the pair, re-call and re-grade take rel's R@5 as each "
        f"read's base and the null's floors, and refuse step 1's base, another seed 0 or a base that is not rel's "
        f"({time.time() - t0:.1f}s): ok")
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("cmd", nargs="?")
    ap.add_argument("--split", default="L-musique", choices=S2.SPLITS)
    ap.add_argument("--selftest", action="store_true")
    k, rest = ap.parse_known_args(argv)
    if k.selftest:
        return selftest()
    if k.cmd == "smoke":
        sp = argparse.ArgumentParser()
        sp.add_argument("--device", default="cpu")
        sp.add_argument("--host", action="store_true")
        sp.add_argument("--out-root")
        s = sp.parse_args(rest)
        return smoke(s.device, s.host, s.out_root)
    if k.cmd == "train":
        return train([k.cmd] + rest, k.split)
    if k.cmd == "read":
        return read(rest)
    if k.cmd == "compare":
        return S2.main([k.cmd] + rest)
    if k.cmd == "gate":
        gp = argparse.ArgumentParser()
        gp.add_argument("--recall", required=True)
        gp.add_argument("--rel-grade", required=True)
        g = gp.parse_args(rest)
        return gate(g.recall, g.rel_grade)
    if k.cmd == "grade":
        gp = argparse.ArgumentParser()
        gp.add_argument("--full-root", required=True)
        gp.add_argument("--reuse", default="")
        gp.add_argument("--out")
        g = gp.parse_args(rest)
        reuse = dict(x.split("=", 1) for x in g.reuse.split(",") if x)
        rec = grade(g.full_root, reuse, g.out or str(Path(g.full_root) / "grade"))
        return 1 if rec["verdict"] == "INCOMPLETE" else 0
    if k.cmd in ("pair", "recall", "regrade"):
        gp = argparse.ArgumentParser()
        gp.add_argument("--screens", default="")
        gp.add_argument("--null", default="")
        gp.add_argument("--pair")
        gp.add_argument("--grade")
        gp.add_argument("--out", required=True)
        g = gp.parse_args(rest)
        null = [x for x in g.null.split(",") if x]
        need = {"pair": g.screens, "recall": g.pair and len(null) == 4, "regrade": g.grade and len(null) == 4}[k.cmd]
        if not need:
            gp.error(f"{k.cmd}: pair needs --screens A,B; recall --pair and regrade --grade, with the four --null files")
        try:
            if k.cmd == "pair":
                pair([x for x in g.screens.split(",") if x], g.out)
            elif k.cmd == "recall":
                recall(null, g.pair, g.out)
            else:
                return 1 if regrade(null, g.grade, g.out)["verdict"] == "INCOMPLETE" else 0
        except SystemExit as e:
            log(f"relz {k.cmd}: {e}")
            return 2
        return 0
    raise SystemExit("relz: train, read, compare, pair, recall, gate, grade, regrade, smoke, or --selftest")


if __name__ == "__main__":
    sys.exit(main())
