"""The seed-0 stop of the GNN screens (docs/DEP_GNN_DEPTH.md section 12, docs/LTG_NEXT_HOP_GNN.md section 11,
docs/RTR_GO_BACK_ROUNDS.md section 11, docs/IRA_ITERATIVE_ALIGNMENT.md section 12, docs/GNX_NEXT_COLUMNS_IN_GNN.md): a
screen's seed-0 WQ-hotpotqa fit is not run when its seed-0 WQ-musique fit has already decided the screen NOT_ADOPTED.

Each of these screens calls one LOSS among its deciding reads NOT_ADOPTED. The deciding reads of the WQ-musique split are
fixed once that split's seed-0 fit is read: the base's three seeds are filed, and the new arm's one read of each comes
from that fit. When one of them is a LOSS, no read of the WQ-hotpotqa fit can change the screen's verdict, and that fit
(about 4.5 hours of the card, alone) is skipped. Any problem the stage's calls report, or a deciding read of the
WQ-musique split that is missing, runs the fit as before.

    python outputs/mp_unified/s0stop.py check --stage dep
        the stage's compare-all and calls, then outputs/<stage>/s0stop.json: STOP (the screen NOT_ADOPTED) or GO.
        Always exits 0; a failure is a problem, and a problem is GO.
    python outputs/mp_unified/s0stop.py run --stage dep -- COMMAND ...
        COMMAND (the stage's WQ-hotpotqa fit) with its exit code, unless the stage's s0stop.json, made from the seed-0
        fit's read as it is now, says STOP: then outputs/<stage>/s0stop_skip.json and exit 0.
    python outputs/mp_unified/s0stop.py gate --stage dep --want NO_GAIN,NOT_ADOPTED
        the screen's verdict, NOT_ADOPTED on a STOP and calls.json's screen.verdict otherwise: exit 0 when it is one of
        --want, else 1; a missing calls.json exits 2 (verdict_gate.py's codes).
    python outputs/mp_unified/s0stop.py --selftest

The check, the wrapper and the gate import no stage module; the check runs the stage's own compare-all and calls, and
takes the deciding reads from the stage's rule (zdep.read_keys, or zltg.deciding_keys) in the check's process.
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
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

S0_SPLIT, HP_SPLIT = "WQ-musique", "WQ-hotpotqa"
GO, STOP = "GO", "STOP"
LOSS, NOT_ADOPTED = "LOSS", "NOT_ADOPTED"
RULES = {"dep": "DEP's sixteen-read rule (zdep.decide_screen)", "ltg": "LTG's in-domain rule (zltg.decide_screen)"}
STAGES = {                                      # stage: (module, OUT, seed-0 WQ-musique fit, seed-0 WQ-hotpotqa fit, rule, doc)
    "dep": ("zdep", "outputs/dep", "scr-dep-l4-s0", "scr-dep-l4-s0-hp", "dep", "docs/DEP_GNN_DEPTH.md"),
    "ltg": ("zltg", "outputs/ltg", "scr-ltg-nh-s0", "scr-ltg-nh-s0-hp", "ltg", "docs/LTG_NEXT_HOP_GNN.md"),
    "rtr": ("zrtr", "outputs/rtr", "scr-rtr-r2-s0", "scr-rtr-r2-s0-hp", "dep", "docs/RTR_GO_BACK_ROUNDS.md"),
    "ira": ("zira", "outputs/ira", "scr-ira-k4-s0", "scr-ira-k4-s0-hp", "dep", "docs/IRA_ITERATIVE_ALIGNMENT.md"),
    "gnx": ("zgnx", "outputs/gnx", "scr-gnx-nx-s0", "scr-gnx-nx-s0-hp", "ltg", "docs/GNX_NEXT_COLUMNS_IN_GNN.md"),
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] s0stop: {msg}", flush=True)


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read_path(name, base=ROOT):
    return Path(base) / "outputs" / "screen" / "fits" / name / "read.json"


def deciding(rule):
    """The deciding reads of the WQ-musique split under the stage's rule, as (split, read)."""
    if rule == "dep":
        import zdep
        keys = zdep.read_keys()
    else:
        import zltg
        keys = zltg.deciding_keys()
    return [tuple(k) for k in keys if k[0] == S0_SPLIT]


def decide(rec, keys):
    """(GO or STOP, the LOSS rows, the missing reads, the problems) from a stage's calls record and the deciding reads of
    the WQ-musique split."""
    problems = [str(p) for p in rec.get("problems") or []]
    rows = {(r.get("split"), r.get("read")): r for r in (rec.get("screen") or {}).get("reads") or []}
    missing = [f"{a} {b}" for a, b in keys if (a, b) not in rows]
    loss = [rows[k] for k in keys if k in rows and rows[k].get("call") == LOSS]
    if not keys:
        problems.append("no deciding read of the WQ-musique split")
    if problems or missing:
        return GO, loss, missing, problems
    return (STOP if loss else GO), loss, missing, problems


def run_stage(module, verb, base):
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    return subprocess.run([sys.executable, str(Path(base) / "outputs" / "mp_unified" / f"{module}.py"), verb],
                          cwd=str(base), env=env).returncode


def check(stage, base=ROOT, runner=run_stage, keys_of=deciding):
    module, out, s0, hp, rule, doc = STAGES[stage]
    t0 = time.time() - 1.0
    problems, rec, keys, csha, rsha = [], {}, [], None, None
    try:
        rp = read_path(s0, base)
        if not rp.is_file():
            raise RuntimeError(f"{s0} is not read ({rp.relative_to(base).as_posix()} missing)")
        rsha = sha_file(rp)
        for verb in ("compare-all", "calls"):
            rc = runner(module, verb, base)
            if rc:
                raise RuntimeError(f"{module}.py {verb} exited {rc}")
        cp = Path(base) / out / "calls.json"
        if not cp.is_file() or cp.stat().st_mtime < t0:
            raise RuntimeError(f"{out}/calls.json was not written by this check's calls")
        csha = sha_file(cp)
        rec = json.loads(cp.read_text(encoding="utf-8"))
        if sha_file(rp) != rsha:
            raise RuntimeError(f"{s0}'s read.json changed during the check")
        keys = keys_of(rule)
    except Exception as e:                      # a failure is a problem, and a problem runs the fit
        problems.append(f"{type(e).__name__}: {e}")
    decision, loss, missing, bad = decide(rec, keys) if not problems else (GO, [], [], [])
    problems += bad
    scr = rec.get("screen") or {}
    rows = [r for r in scr.get("reads") or [] if r.get("split") == S0_SPLIT]
    side = [r for r in scr.get("beside") or [] if r.get("split") == S0_SPLIT]
    res = {"stage": stage.upper(), "declared_in": doc, "rule": RULES[rule], "s0_fit": s0, "hp_fit": hp, "split": S0_SPLIT,
           "deciding_reads": [f"{a} {b}" for a, b in keys], "decision": decision,
           "verdict": NOT_ADOPTED if decision == STOP else None, "loss_rows": loss, "missing": missing,
           "rows": rows, "beside": side, "problems": problems, "s0_read_sha256": rsha, "calls_sha256": csha,
           "calls_screen_verdict": scr.get("verdict"), "s0stop_sha256": sha_file(__file__),
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    od = Path(base) / out
    od.mkdir(parents=True, exist_ok=True)
    tmp = od / "s0stop.json.tmp"
    tmp.write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    os.replace(tmp, od / "s0stop.json")
    why = (f"LOSS on {', '.join(r['split'] + ' ' + r['read'] for r in loss)}" if decision == STOP
           else f"problems {problems}" if problems else f"missing {missing}" if missing
           else f"no LOSS on the {len(keys)} deciding reads of {S0_SPLIT}")
    log(f"{stage}: {decision} ({why}); {out}/s0stop.json")
    return 0


def stop_record(stage, base=ROOT):
    """The stage's s0stop.json when it was made from the seed-0 fit's read as it is now, else None."""
    _m, out, s0, hp, _r, _d = STAGES[stage]
    p, rp = Path(base) / out / "s0stop.json", read_path(s0, base)
    if not (p.is_file() and rp.is_file()):
        return None
    try:
        rec = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if (rec.get("s0_fit"), rec.get("hp_fit"), rec.get("s0_read_sha256")) != (s0, hp, sha_file(rp)):
        return None
    return rec


def run(stage, cmd, base=ROOT):
    _m, out, _s0, hp, _r, _d = STAGES[stage]
    if not cmd:
        raise SystemExit("s0stop: run needs a command after --")
    rec = stop_record(stage, base)
    if rec is not None and rec.get("decision") == STOP:
        skip = {"stage": stage.upper(), "hp_fit": hp, "skipped": True, "command": cmd, "verdict": NOT_ADOPTED,
                "s0stop_utc": rec.get("utc"), "loss_rows": rec.get("loss_rows"),
                "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        (Path(base) / out / "s0stop_skip.json").write_text(json.dumps(skip, indent=1) + "\n", encoding="utf-8")
        log(f"{stage}: {hp} not run: the seed-0 stop decided the screen NOT_ADOPTED ({out}/s0stop.json, {rec.get('utc')})")
        return 0
    log(f"{stage}: {hp} runs ({'decision ' + str(rec.get('decision')) if rec else 'no current s0stop.json'})")
    return subprocess.call(cmd)


def gate(stage, want, base=ROOT):
    _m, out, _s0, _hp, _r, _d = STAGES[stage]
    rec = stop_record(stage, base)
    if rec is not None and rec.get("decision") == STOP:
        v, src = NOT_ADOPTED, f"{out}/s0stop.json"
    else:
        cp = Path(base) / out / "calls.json"
        if not cp.is_file():
            log(f"gate {stage}: {out}/calls.json missing")
            return 2
        v, src = (json.loads(cp.read_text(encoding="utf-8")).get("screen") or {}).get("verdict"), f"{out}/calls.json"
    ok = isinstance(v, str) and v in want
    log(f"gate {stage}: the screen is {v} ({src}); {'pass' if ok else 'stop'} (wants {', '.join(want)})")
    return 0 if ok else 1


# ── self-test (laptop or host, CPU, no fit) ──────────────────────────────────


def selftest():
    import zml1 as RULE
    assert (RULE.LOSS, S0_SPLIT, HP_SPLIT) == (LOSS, *RULE.SPLITS)
    dk, lk = deciding("dep"), deciding("ltg")
    assert len(dk) == 8 and len(set(dk)) == 8 and all(k[0] == S0_SPLIT for k in dk), dk
    assert sorted(k[1] for k in lk) == sorted(("webqsp", "squad", "hotpotqa", "2wiki")), lk
    assert set(lk) < set(dk)
    for st, (module, out, s0, hp, rule, doc) in STAGES.items():   # each stage's names are its module's
        if not (HERE / f"{module}.py").is_file():
            raise AssertionError(f"{module}.py missing")
        m = __import__(module)
        assert (m.fit_name(0, S0_SPLIT), m.fit_name(0, HP_SPLIT), m.OUT, m.DECLARED) == (s0, hp, out, doc), st
    for st, rule in (("dep", "dep"), ("rtr", "dep"), ("ira", "dep"), ("ltg", "ltg"), ("gnx", "ltg")):
        assert STAGES[st][4] == rule

    def row(sp, rd, c):
        return {"split": sp, "read": rd, "call": c}

    keys = dk
    within = [row(*k, "WITHIN") for k in keys]
    hp_loss = within + [row(HP_SPLIT, "squad", LOSS)]
    assert decide({"screen": {"reads": within}, "problems": []}, keys)[0] == GO
    assert decide({"screen": {"reads": hp_loss}, "problems": []}, keys)[0] == GO           # WQ-hotpotqa is not the stop's
    one = [dict(r, call=LOSS) if r["read"] == "metaqa" else r for r in within]
    d, loss, miss, bad = decide({"screen": {"reads": one}, "problems": []}, keys)
    assert (d, [r["read"] for r in loss], miss, bad) == (STOP, ["metaqa"], [], [])
    assert decide({"screen": {"reads": one}, "problems": ["x"]}, keys)[0] == GO            # a problem runs the fit
    assert decide({"screen": {"reads": one[1:]}, "problems": []}, keys)[0] == GO            # a missing read runs it
    assert decide({"screen": {"reads": one}, "problems": []}, [])[0] == GO
    gain = [dict(r, call="GAIN") for r in within]
    assert decide({"screen": {"reads": gain}, "problems": []}, keys)[0] == GO
    zs = [row(*k, "WITHIN") for k in lk] + [row(S0_SPLIT, "musique", LOSS), row(S0_SPLIT, "3hop", LOSS)]
    assert decide({"screen": {"reads": zs}, "problems": []}, lk)[0] == GO                  # LTG: zero-shot is beside
    assert decide({"screen": {"reads": zs[1:] + [row(*lk[0], LOSS)]}, "problems": []}, lk)[0] == STOP

    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        _m, out, s0, hp, _r, _d = STAGES["dep"]
        rp = read_path(s0, base)

        def runner(rows, rc=0, write=True, problems=()):
            def f(module, verb, b):
                assert module == "zdep" and verb in ("compare-all", "calls") and Path(b) == base
                if verb == "calls" and write:
                    (base / out).mkdir(parents=True, exist_ok=True)
                    (base / out / "calls.json").write_text(json.dumps(
                        {"screen": {"verdict": "INCOMPLETE", "reads": rows}, "problems": list(problems)}), encoding="utf-8")
                return rc
            return f

        def rec():
            return json.loads((base / out / "s0stop.json").read_text(encoding="utf-8"))

        def keys_of(_rule):
            return keys

        check("dep", base, runner(one), keys_of)                                           # no read: GO
        assert rec()["decision"] == GO and "not read" in rec()["problems"][0]
        rp.parent.mkdir(parents=True)
        rp.write_text('{"read": 1}', encoding="utf-8")
        check("dep", base, runner(one), keys_of)
        r = rec()
        assert (r["decision"], r["verdict"], r["s0_read_sha256"]) == (STOP, NOT_ADOPTED, sha_file(rp)), r
        assert len(r["rows"]) == 8 and [x["read"] for x in r["loss_rows"]] == ["metaqa"]
        fail = [sys.executable, "-c", "raise SystemExit(7)"]
        assert run("dep", fail, base) == 0 and (base / out / "s0stop_skip.json").is_file()   # STOP: not run
        assert gate("dep", ["NO_GAIN", "NOT_ADOPTED"], base) == 0
        assert gate("dep", ["ADOPT_FOR_FULL_RUN"], base) == 1
        rp.write_text('{"read": 2}', encoding="utf-8")                                     # a new read: the record is stale
        assert stop_record("dep", base) is None and run("dep", fail, base) == 7
        assert gate("dep", ["NOT_ADOPTED"], base) == 1                                     # calls.json says INCOMPLETE
        check("dep", base, runner(within), keys_of)
        assert rec()["decision"] == GO and run("dep", fail, base) == 7
        check("dep", base, runner(one, problems=["code differs"]), keys_of)
        assert rec()["decision"] == GO and rec()["problems"] == ["code differs"]
        check("dep", base, runner(one, rc=3), keys_of)
        assert rec()["decision"] == GO and "exited 3" in rec()["problems"][0]
        os.utime(base / out / "calls.json", (time.time() - 600, time.time() - 600))
        check("dep", base, runner(one, write=False), keys_of)                              # an old calls.json: GO
        assert rec()["decision"] == GO and "not written" in rec()["problems"][0]
        (base / out / "calls.json").unlink()
        (base / out / "s0stop.json").unlink()
        assert gate("dep", ["NOT_ADOPTED"], base) == 2
        (base / out / "calls.json").write_text(json.dumps({"screen": {"verdict": "NO_GAIN"}}), encoding="utf-8")
        assert gate("dep", ["NO_GAIN", "NOT_ADOPTED"], base) == 0 and gate("dep", ["ADOPT_FOR_FULL_RUN"], base) == 1
        try:
            run("dep", [], base)
            raise AssertionError("run without a command must stop")
        except SystemExit:
            pass
    print("s0stop selftest: OK")
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    cmd = []
    if "--" in argv:
        i = argv.index("--")
        argv, cmd = argv[:i], argv[i + 1:]
    ap = argparse.ArgumentParser()
    ap.add_argument("verb", nargs="?", choices=("check", "run", "gate"))
    ap.add_argument("--stage", choices=sorted(STAGES))
    ap.add_argument("--want")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if not (a.verb and a.stage):
        ap.error("a verb and --stage are needed")
    if a.verb == "check":
        return check(a.stage)
    if a.verb == "run":
        return run(a.stage, cmd)
    if not a.want:
        ap.error("gate needs --want")
    return gate(a.stage, [w for w in a.want.split(",") if w])


if __name__ == "__main__":
    sys.exit(main())
