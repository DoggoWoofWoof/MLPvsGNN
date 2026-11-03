"""The WWW rule (docs/WWW_SCOPE_RULE.md): until the freeze, a screen, a full run and a seed-0 stop are decided on the
in-domain reads of the WWW paper's three datasets, 2wiki, HotpotQA and MuSiQue.

The WWW 2027 paper's tables are those three datasets (paper/www2027_v2/main.tex); MetaQA, WebQSP and SQuAD are the journal
version's (paper/www2027_v2/PLAN.md section 11). The six deciding reads:

    WQ-musique   hotpotqa, 2wiki                                   (the datasets this split trains on)
    WQ-hotpotqa  musique, 2wiki, musique-3hop, musique-4hop        (and LTG's two musique slices)

Each read's call is the stage's own: the row its calls.json already holds (the same base, floor and delta). A screen: a
LOSS among the six is NOT_ADOPTED; else a GAIN is ADOPT_FOR_FULL_RUN; else NO_GAIN. A full run: ADOPT when there is no
LOSS among the six and a GAIN on one of the four dataset reads; else NOT_ADOPTED. The seed-0 stop: a LOSS on the
WQ-musique split's hotpotqa or 2wiki. Every other read (webqsp, squad, the zero-shot reads, the WQ-musique fits' musique
slices) is reported beside and decides nothing. A problem in the stage's calls (its top-level "problems") makes the WWW
verdict INCOMPLETE, as it makes the stage's own.

    python outputs/mp_unified/zwww.py call --stage gnt
        the stage's calls.json and s0stop.json under the WWW rule: outputs/www/<stage>.json. Exits 0.
    python outputs/mp_unified/zwww.py s0stop-run --stage gnt -- COMMAND ...
        COMMAND (the stage's WQ-hotpotqa fit) with its exit code, unless the stage's s0stop.json, made from its seed-0
        fit's read as it is now, has a LOSS on a WWW read: then outputs/www/<stage>_s0stop_skip.json and exit 0.
    python outputs/mp_unified/zwww.py gate --stage gnt --want ADOPT_FOR_FULL_RUN [--block full_run]
        the stage's WWW verdict: exit 0 when it is one of --want, else 1; no calls.json exits 2 (verdict_gate.py's codes).
        A current WWW stop is the screen NOT_ADOPTED.
    python outputs/mp_unified/zwww.py audit
        every stage with a calls.json or an s0stop.json under its own rule and the WWW rule: outputs/www/audit.json, .md.
    python outputs/mp_unified/zwww.py --selftest

Reads only the stages' own records (stdlib only; no stage module is imported or changed).
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DECLARED = "docs/WWW_SCOPE_RULE.md"
OUT = "outputs/www"
S0_SPLIT, HP_SPLIT = "WQ-musique", "WQ-hotpotqa"
READS = ((S0_SPLIT, "hotpotqa"), (S0_SPLIT, "2wiki"), (HP_SPLIT, "musique"), (HP_SPLIT, "2wiki"),
         (HP_SPLIT, "musique-3hop"), (HP_SPLIT, "musique-4hop"))
DATASET_READS = READS[:4]                      # the full run's GAIN: on a dataset read, as LTG's decide_full asks
S0_READS = tuple(k for k in READS if k[0] == S0_SPLIT)
GAIN, LOSS, WITHIN = "GAIN", "LOSS", "WITHIN"
ADOPT_FOR_FULL_RUN, NO_GAIN, NOT_ADOPTED, ADOPT, INCOMPLETE = (
    "ADOPT_FOR_FULL_RUN", "NO_GAIN", "NOT_ADOPTED", "ADOPT", "INCOMPLETE")
GO, STOP = "GO", "STOP"
BLOCKS = ("screen", "full_run")
assert len(set(READS)) == len(READS) == 6 and len(S0_READS) == 2


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] zwww: {msg}", flush=True)


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def write_json(p, rec):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
    os.replace(tmp, p)


def read_json(p):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def stage_out(stage, base=ROOT):
    return Path(base) / "outputs" / stage


def key_of(r):
    """(split, read) of a calls row; R2P's rows name the read "dataset"."""
    return r.get("split"), r.get("read") or r.get("dataset")


def rows_of(blk):
    """{(split, read): row} over a block's deciding and beside rows."""
    blk = blk or {}
    return {key_of(r): r for r in (blk.get("reads") or []) + (blk.get("beside") or [])}


# ── the rule ─────────────────────────────────────────────────────────────────


def screen_verdict(blk):
    """(verdict, the six rows, the missing reads) of a screen block under the WWW rule. A LOSS decides with a read
    missing (no read can undo it); anything else with a read missing is INCOMPLETE."""
    rows = rows_of(blk)
    have = [rows[k] for k in READS if k in rows]
    missing = [f"{a} {b}" for a, b in READS if (a, b) not in rows]
    calls = [r.get("call") for r in have]
    if LOSS in calls:
        return NOT_ADOPTED, have, missing
    if missing:
        return INCOMPLETE, have, missing
    return (ADOPT_FOR_FULL_RUN if GAIN in calls else NO_GAIN), have, []


def full_verdict(blk):
    """(verdict, the six rows, the missing reads) of a full-run block under the WWW rule; a LOSS decides as in
    screen_verdict."""
    rows = rows_of(blk)
    have = [rows[k] for k in READS if k in rows]
    missing = [f"{a} {b}" for a, b in READS if (a, b) not in rows]
    if any(r.get("call") == LOSS for r in have):
        return NOT_ADOPTED, have, missing
    if missing:
        return INCOMPLETE, have, missing
    ok = any(rows[k].get("call") == GAIN for k in DATASET_READS)
    return (ADOPT if ok else NOT_ADOPTED), have, []


def counts(rows):
    out = {}
    for r in rows:
        out[r.get("call")] = out.get(r.get("call"), 0) + 1
    return out


def current_stop(stage, base=ROOT, strict=True):
    """The stage's s0stop.json when it was made from its seed-0 fit's read as it is now, else None. Not strict (the
    audit, where the fit's read may not be on this machine): the record as filed."""
    rec = read_json(stage_out(stage, base) / "s0stop.json")
    if not isinstance(rec, dict) or not rec.get("s0_fit"):
        return None
    if not strict:
        return rec
    rp = Path(base) / "outputs" / "screen" / "fits" / str(rec["s0_fit"]) / "read.json"
    if not rp.is_file() or rec.get("s0_read_sha256") != sha_file(rp):
        return None
    return rec


def stop_of(rec):
    """(GO or STOP, the LOSS rows, why) of a current s0stop.json under the WWW rule. As s0stop: no record, a problem in
    it, or a WWW read missing from its rows is GO."""
    if rec is None:
        return GO, [], "no current s0stop.json"
    if rec.get("problems"):
        return GO, [], f"its check's problems {rec['problems']}"
    rows = {key_of(r): r for r in (rec.get("rows") or []) + (rec.get("beside") or [])}
    missing = [f"{a} {b}" for a, b in S0_READS if (a, b) not in rows]
    if missing:
        return GO, [], f"missing {missing}"
    loss = [rows[k] for k in S0_READS if rows[k].get("call") == LOSS]
    if loss:
        return STOP, loss, "LOSS on " + ", ".join(f"{key_of(r)[0]} {key_of(r)[1]}" for r in loss)
    return GO, [], f"no LOSS on {', '.join(b for _a, b in S0_READS)} of {S0_SPLIT}"


def own_stop(rec):
    return None if rec is None else rec.get("decision")


def call(stage, base=ROOT, write=True, strict=True):
    """The stage's records under the WWW rule (its calls.json, its s0stop.json) -> outputs/www/<stage>.json."""
    od = stage_out(stage, base)
    cp = od / "calls.json"
    c = read_json(cp) if cp.is_file() else None
    stop = current_stop(stage, base, strict)
    sd, sloss, swhy = stop_of(stop)
    rec = {"stage": stage.upper(), "declared_in": DECLARED, "rule": "WWW: the six in-domain reads of 2wiki, HotpotQA "
           "and MuSiQue decide", "deciding_reads": [f"{a} {b}" for a, b in READS],
           "stop": {"www": sd, "own": own_stop(stop), "why": swhy, "loss_rows": sloss,
                    "s0_fit": (stop or {}).get("s0_fit"), "s0_read_sha256": (stop or {}).get("s0_read_sha256")},
           "calls_sha256": sha_file(cp) if c is not None else None, "problems": [], "zwww_sha256": sha_file(__file__),
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    probs = [str(p) for p in ((c or {}).get("problems") or [])]
    rec["problems"] = probs
    for name, fn in (("screen", screen_verdict), ("full_run", full_verdict)):
        blk = (c or {}).get(name)
        if not isinstance(blk, dict):
            rec[name] = None
            continue
        v, rows, missing = fn(blk)
        if probs and v != INCOMPLETE:
            v = INCOMPLETE
        six = {key_of(r) for r in rows}
        side = [r for k, r in rows_of(blk).items() if k not in six]
        rec[name] = {"www": v, "own": blk.get("verdict"), "counts": counts(rows), "missing": missing, "reads": rows,
                     "beside": side}
    if sd == STOP:                              # the screen is decided by its seed-0 stop, as s0stop's gate reads it
        rec["screen"] = dict(rec["screen"] or {}, www=NOT_ADOPTED, by_stop=True)
    if write:
        write_json(Path(base) / OUT / f"{stage}.json", rec)
    return rec


def verdict(stage, block="screen", base=ROOT):
    """The stage's WWW verdict of block, or None when it has no calls.json (and no WWW stop for the screen)."""
    rec = call(stage, base)
    if block == "screen" and rec["stop"]["www"] == STOP:
        return NOT_ADOPTED
    if rec["calls_sha256"] is None:
        return None
    return (rec.get(block) or {}).get("www")


# ── the commands ─────────────────────────────────────────────────────────────


def s0stop_run(stage, cmd, base=ROOT):
    if not cmd:
        raise SystemExit("zwww: s0stop-run needs a command after --")
    stop = current_stop(stage, base)
    d, loss, why = stop_of(stop)
    if d == STOP:
        write_json(Path(base) / OUT / f"{stage}_s0stop_skip.json",
                   {"stage": stage.upper(), "declared_in": DECLARED, "skipped": True, "command": cmd,
                    "verdict": NOT_ADOPTED, "why": why, "loss_rows": loss, "s0_fit": stop.get("s0_fit"),
                    "s0stop_utc": stop.get("utc"), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        log(f"{stage}: the WQ-hotpotqa fit is not run: the WWW seed-0 stop ({why}); {OUT}/{stage}_s0stop_skip.json")
        return 0
    log(f"{stage}: the WQ-hotpotqa fit runs (WWW stop {d}: {why}; the stage's own {own_stop(stop)})")
    return subprocess.call(cmd)


def gate(stage, want, block="screen", base=ROOT):
    v = verdict(stage, block, base)
    if v is None:
        log(f"gate {stage}: outputs/{stage}/calls.json missing")
        return 2
    ok = v in want
    log(f"gate {stage}: the {block} is {v} under the WWW rule ({OUT}/{stage}.json); {'pass' if ok else 'stop'} "
        f"(wants {', '.join(want)})")
    return 0 if ok else 1


def audit_stages(base=ROOT):
    """Every stage folder under outputs/ with a calls.json or an s0stop.json (outputs/www itself excluded)."""
    out = []
    for p in sorted((Path(base) / "outputs").iterdir()):
        if p.is_dir() and p.name != "www" and ((p / "calls.json").is_file() or (p / "s0stop.json").is_file()):
            out.append(p.name)
    return out


def audit(base=ROOT):
    rows, md = [], ["# The WWW rule against each stage's own (docs/WWW_SCOPE_RULE.md)", "",
                    "| stage | own stop | WWW stop | own screen | WWW screen | own full run | WWW full run | changed |",
                    "|---|---|---|---|---|---|---|---|"]
    for st in audit_stages(base):
        c = read_json(stage_out(st, base) / "calls.json") or {}
        forms = c.get("forms") if isinstance(c.get("forms"), dict) else None
        if forms:                               # RWT: two forms, each its own screen block
            for k, blk in forms.items():
                v, _six, missing = screen_verdict(blk)
                r = {"stage": f"{st}.{k}", "own_stop": None, "www_stop": None, "own_screen": blk.get("verdict"),
                     "www_screen": v, "own_full_run": None, "www_full_run": None, "missing": missing}
                rows.append(r)
            continue
        rec = call(st, base, write=False, strict=False)
        scr, full = rec.get("screen") or {}, rec.get("full_run") or {}
        rows.append({"stage": st, "own_stop": rec["stop"]["own"], "www_stop": rec["stop"]["www"] if rec["stop"]["own"]
                     else None, "own_screen": scr.get("own"), "www_screen": scr.get("www"),
                     "own_full_run": full.get("own"), "www_full_run": full.get("www"),
                     "missing": scr.get("missing") or []})
    for r in rows:
        changed = ((r["own_stop"], r["own_screen"], r["own_full_run"]) != (r["www_stop"], r["www_screen"], r["www_full_run"])
                   and not (r["own_stop"] is None and r["www_stop"] is None and r["own_screen"] == r["www_screen"]
                            and r["own_full_run"] == r["www_full_run"]))
        r["changed"] = changed
        md.append(f"| {r['stage']} | {r['own_stop'] or ''} | {r['www_stop'] or ''} | {r['own_screen'] or ''} | "
                  f"{r['www_screen'] or ''} | {r['own_full_run'] or ''} | {r['www_full_run'] or ''} | "
                  f"{'yes' if changed else ''} |")
    rec = {"declared_in": DECLARED, "deciding_reads": [f"{a} {b}" for a, b in READS], "stages": rows,
           "zwww_sha256": sha_file(__file__), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    write_json(Path(base) / OUT / "audit.json", rec)
    (Path(base) / OUT / "audit.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    for line in md[2:]:
        print(line)
    return 0


# ── self-test (laptop or host, CPU, no fit) ──────────────────────────────────


def selftest():
    def row(sp, rd, c, d=0.0, key="read"):
        return {"split": sp, key: rd, "base": [0.5, 0.5, 0.5], "new": 0.5 + d, "delta": d, "floor": 0.0075, "call": c}

    ALL = [(S0_SPLIT, x) for x in ("webqsp", "squad", "hotpotqa", "2wiki")] + \
          [(HP_SPLIT, x) for x in ("webqsp", "squad", "musique", "2wiki", "musique-3hop", "musique-4hop")]

    def blk(calls, beside=()):
        return {"verdict": "OWN", "reads": [row(a, b, calls.get((a, b), WITHIN)) for a, b in ALL],
                "beside": [row(a, b, c) for (a, b), c in beside]}

    # a LOSS on webqsp alone decides nothing; a GAIN on squad or metaqa alone is no gain
    assert screen_verdict(blk({(S0_SPLIT, "webqsp"): LOSS}))[0] == NO_GAIN
    assert screen_verdict(blk({(HP_SPLIT, "squad"): GAIN}, [((HP_SPLIT, "metaqa"), GAIN)]))[0] == NO_GAIN
    assert screen_verdict(blk({(S0_SPLIT, "webqsp"): LOSS, (HP_SPLIT, "musique"): GAIN}))[0] == ADOPT_FOR_FULL_RUN
    assert screen_verdict(blk({(HP_SPLIT, "musique-4hop"): LOSS, (S0_SPLIT, "2wiki"): GAIN}))[0] == NOT_ADOPTED
    v, _rows, miss = screen_verdict({"reads": [row(S0_SPLIT, "hotpotqa", WITHIN)]})
    assert v == INCOMPLETE and len(miss) == 5, (v, miss)
    assert screen_verdict({"reads": [row(HP_SPLIT, "musique", LOSS)]})[0] == NOT_ADOPTED   # a LOSS decides alone
    assert screen_verdict({"reads": [row(HP_SPLIT, "musique", GAIN)]})[0] == INCOMPLETE
    assert full_verdict({"reads": [row(HP_SPLIT, "musique", LOSS)]})[0] == NOT_ADOPTED
    # R2P's rows name the read "dataset"
    b = blk({})
    b["reads"] = [dict(r, dataset=r.pop("read")) for r in b["reads"]]
    assert screen_verdict(b)[0] == NO_GAIN
    # the full run: a GAIN on a dataset read and no LOSS; a GAIN on a slice alone is not enough
    assert full_verdict(blk({(HP_SPLIT, "2wiki"): GAIN}))[0] == ADOPT
    assert full_verdict(blk({(HP_SPLIT, "musique-3hop"): GAIN}))[0] == NOT_ADOPTED
    assert full_verdict(blk({(HP_SPLIT, "2wiki"): GAIN, (HP_SPLIT, "musique"): LOSS}))[0] == NOT_ADOPTED
    assert full_verdict(blk({(HP_SPLIT, "2wiki"): GAIN, (S0_SPLIT, "squad"): LOSS}))[0] == ADOPT
    # the stop: webqsp's LOSS is GO, hotpotqa's STOP; a problem or a missing read is GO
    s0rows = lambda c: [row(S0_SPLIT, x, c.get(x, WITHIN)) for x in ("webqsp", "squad", "hotpotqa", "2wiki")]
    assert stop_of({"rows": s0rows({"webqsp": LOSS}), "problems": []})[0] == GO
    assert stop_of({"rows": s0rows({"hotpotqa": LOSS}), "problems": []})[0] == STOP
    assert stop_of({"rows": s0rows({"hotpotqa": LOSS}), "problems": ["x"]})[0] == GO
    assert stop_of({"rows": s0rows({})[:2], "problems": []})[0] == GO
    assert stop_of(None)[0] == GO
    # on disk: the wrapper, the gate's codes, the stale stop, the audit
    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        fits = base / "outputs" / "screen" / "fits"
        (fits / "scr-x-s0").mkdir(parents=True)
        rp = fits / "scr-x-s0" / "read.json"
        rp.write_text('{"r": 1}\n', encoding="utf-8")
        od = base / "outputs" / "xx"
        od.mkdir(parents=True)
        stop = {"stage": "XX", "s0_fit": "scr-x-s0", "s0_read_sha256": sha_file(rp), "decision": STOP, "problems": [],
                "rows": s0rows({"webqsp": LOSS}), "utc": "t"}
        write_json(od / "s0stop.json", stop)
        _, a, rest = parse(["gate", "--stage", "dep", "--want", ADOPT_FOR_FULL_RUN])   # the feeder's gate line
        assert (a.cmd, a.stage, a.want, a.block, rest) == ("gate", "dep", ADOPT_FOR_FULL_RUN, "screen", [])
        _, a, rest = parse(["s0stop-run", "--stage", "gnw", "--", "python", "f.py", "--frac", "0.8", "--", "x.py"])
        assert (a.cmd, a.stage, rest) == ("s0stop-run", "gnw", ["python", "f.py", "--frac", "0.8", "--", "x.py"])
        assert gate("xx", [ADOPT_FOR_FULL_RUN], base=base) == 2          # no calls.json
        marker = base / "ran.txt"
        cmd = [sys.executable, "-c", f"open(r'{marker}', 'w').write('1')"]
        assert s0stop_run("xx", cmd, base) == 0 and marker.is_file()     # the own STOP was webqsp: the fit runs
        marker.unlink()
        write_json(od / "s0stop.json", dict(stop, rows=s0rows({"2wiki": LOSS})))
        assert s0stop_run("xx", cmd, base) == 0 and not marker.is_file()  # a WWW LOSS: skipped
        assert (base / OUT / "xx_s0stop_skip.json").is_file()
        write_json(od / "calls.json", {"problems": [], "screen": blk({(HP_SPLIT, "musique"): GAIN})})
        assert gate("xx", [ADOPT_FOR_FULL_RUN], base=base) == 1          # the WWW stop decides the screen
        rp.write_text('{"r": 2}\n', encoding="utf-8")                    # the seed-0 read changed: the stop is stale
        assert gate("xx", [ADOPT_FOR_FULL_RUN], base=base) == 0
        assert s0stop_run("xx", cmd, base) == 0 and marker.is_file()
        write_json(od / "calls.json", {"problems": ["a stamp"], "screen": blk({(HP_SPLIT, "musique"): GAIN})})
        assert gate("xx", [ADOPT_FOR_FULL_RUN], base=base) == 1 and verdict("xx", base=base) == INCOMPLETE
        write_json(od / "calls.json", {"problems": [], "screen": blk({}), "full_run": blk({(S0_SPLIT, "2wiki"): GAIN})})
        assert gate("xx", [ADOPT], "full_run", base) == 0 and gate("xx", [ADOPT_FOR_FULL_RUN], base=base) == 1
        rw = base / "outputs" / "rw"
        rw.mkdir()
        write_json(rw / "calls.json", {"forms": {"f": dict(blk({(HP_SPLIT, "musique-3hop"): LOSS}), verdict=NOT_ADOPTED)}})
        write_json(od / "s0stop.json", stop)                             # GNW's case: the own STOP is webqsp's
        assert audit(base) == 0                                          # the audit takes the stop as filed (stale)
        a = read_json(base / OUT / "audit.json")
        got = {r["stage"]: (r["own_stop"], r["www_stop"], r["www_screen"], r["changed"]) for r in a["stages"]}
        assert got == {"rw.f": (None, None, NOT_ADOPTED, False), "xx": (STOP, GO, NO_GAIN, True)}, got
    log("selftest passed")
    return 0


def parse(argv):
    """(the parser, its args, the command after "--"). A REMAINDER positional would swallow --stage and --want too."""
    ap = argparse.ArgumentParser(description="the WWW rule: the six in-domain reads of 2wiki, HotpotQA and MuSiQue decide")
    ap.add_argument("cmd", nargs="?", choices=("call", "s0stop-run", "gate", "audit"))
    ap.add_argument("--stage")
    ap.add_argument("--want", default="")
    ap.add_argument("--block", default="screen", choices=BLOCKS)
    ap.add_argument("--selftest", action="store_true")
    argv = list(argv)
    cut = argv.index("--") if "--" in argv else len(argv)
    return ap, ap.parse_args(argv[:cut]), argv[cut + 1:]


def main(argv=None):
    ap, a, rest = parse(sys.argv[1:] if argv is None else argv)
    if a.selftest:
        return selftest()
    if a.cmd == "audit":
        return audit()
    if not a.stage:
        ap.error(f"{a.cmd} needs --stage")
    if a.cmd == "call":
        rec = call(a.stage)
        log(f"{a.stage}: stop {rec['stop']['www']} (own {rec['stop']['own']}), screen "
            f"{(rec['screen'] or {}).get('www')} (own {(rec['screen'] or {}).get('own')}), full run "
            f"{(rec['full_run'] or {}).get('www')} (own {(rec['full_run'] or {}).get('own')}); {OUT}/{a.stage}.json")
        return 0
    if a.cmd == "s0stop-run":
        if not rest:
            ap.error("s0stop-run needs a command after --")
        return s0stop_run(a.stage, rest)
    want = [w for w in a.want.split(",") if w]
    if not want:
        ap.error("gate needs --want")
    return gate(a.stage, want, a.block)


if __name__ == "__main__":
    sys.exit(main())
