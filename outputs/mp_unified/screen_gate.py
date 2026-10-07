"""Gates between a screen's verdict and the work queued behind it (docs/FULL_ZRET.md, section 5).

    python outputs/mp_unified/screen_gate.py pass --screen outputs/screen/scr-zret.json
        exit 0 when the screen's verdict is PROMISING, else 1: the full run queued behind it is dropped by the feeder.

    python outputs/mp_unified/screen_gate.py chain --screen outputs/screen/scr-zret.json \\
        --wait outputs/full_zret/grade.json --flag outputs/screen/chain-go.flag --timeout-h 3
        exit 0 at once when the verdict is not PROMISING; otherwise poll every --poll-s seconds and exit 0 when the
        --wait file exists, the --flag file exists, or --timeout-h hours pass. The GNN runs queued behind it then start.
        A screen file that is missing or has no verdict exits 2.

    python outputs/mp_unified/screen_gate.py --selftest
"""
import argparse
import json
import sys
import tempfile
import time
from pathlib import Path


def verdict(screen):
    p = Path(screen)
    if not p.exists():
        raise SystemExit(f"screen_gate: {p} missing")
    v = json.loads(p.read_text(encoding="utf-8")).get("verdict")
    if not v:
        raise SystemExit(f"screen_gate: {p} has no verdict")
    return v


def gate_pass(screen):
    v = verdict(screen)
    print(f"screen_gate pass: {screen} is {v}", flush=True)
    return 0 if v == "PROMISING" else 1


def gate_chain(screen, wait, flag, timeout_h, poll_s, clock=time.time, sleep=time.sleep):
    v = verdict(screen)
    if v != "PROMISING":
        print(f"screen_gate chain: {screen} is {v}; no full run, release now", flush=True)
        return 0
    t0 = clock()
    while True:
        for p, why in ((wait, "the full run's grade exists"), (flag, "the release flag exists")):
            if p and Path(p).exists():
                print(f"screen_gate chain: {why} ({p}); release", flush=True)
                return 0
        if clock() - t0 >= timeout_h * 3600:
            print(f"screen_gate chain: {timeout_h} h passed; release", flush=True)
            return 0
        sleep(poll_s)


def selftest():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        s = td / "s.json"
        for v, want in (("PROMISING", 0), ("MIXED", 1), ("NO_GAIN", 1)):
            s.write_text(json.dumps({"verdict": v}), encoding="utf-8")
            assert gate_pass(s) == want, v
        s.write_text(json.dumps({"verdict": "MIXED"}), encoding="utf-8")
        assert gate_chain(s, td / "g.json", None, 3, 60, sleep=lambda x: 1 / 0) == 0   # never waits
        s.write_text(json.dumps({"verdict": "PROMISING"}), encoding="utf-8")
        t = [0.0]

        def tick(x):
            t[0] += x
            if t[0] >= 120:
                (td / "g.json").write_text("{}", encoding="utf-8")
        assert gate_chain(s, td / "g.json", None, 3, 60, clock=lambda: t[0], sleep=tick) == 0 and t[0] == 120
        (td / "g.json").unlink()
        t[0] = 0.0
        assert gate_chain(s, td / "g.json", None, 0.05, 60, clock=lambda: t[0],
                          sleep=lambda x: t.__setitem__(0, t[0] + x)) == 0 and t[0] == 180      # timeout
        t[0] = 0.0
        (td / "go.flag").write_text("", encoding="utf-8")
        assert gate_chain(s, td / "g.json", td / "go.flag", 3, 60, clock=lambda: t[0], sleep=lambda x: 1 / 0) == 0
        s.write_text(json.dumps({}), encoding="utf-8")
        for f in (lambda: gate_pass(s), lambda: gate_pass(td / "none.json")):
            try:
                f()
                raise AssertionError("no verdict must stop")
            except SystemExit:
                pass
    print("screen_gate selftest: OK")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", choices=("pass", "chain"))
    ap.add_argument("--screen")
    ap.add_argument("--wait")
    ap.add_argument("--flag")
    ap.add_argument("--timeout-h", type=float, default=3.0)
    ap.add_argument("--poll-s", type=float, default=60.0)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if not a.screen:
        ap.error("--screen is needed")
    if a.cmd == "pass":
        return gate_pass(a.screen)
    if a.cmd == "chain":
        try:
            return gate_chain(a.screen, a.wait, a.flag, a.timeout_h, a.poll_s)
        except SystemExit as e:
            print(e, flush=True)
            return 2
    ap.error("pass or chain")


if __name__ == "__main__":
    sys.exit(main())
