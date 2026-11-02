"""A gate between a filed verdict and the work queued behind it (first used by docs/DEP_GNN_DEPTH.md section 9).

    python outputs/mp_unified/verdict_gate.py --json outputs/dep/calls.json --key screen.verdict --want ADOPT_FOR_FULL_RUN
        exit 0 when the record's value at --key (dots step into nested objects) is one of --want (comma-separated), else 1:
        the work queued behind the gate is dropped by the feeder. A missing file or key exits 2.

    python outputs/mp_unified/verdict_gate.py --selftest
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path


def value(path, key):
    p = Path(path)
    if not p.is_file():
        raise SystemExit(f"verdict_gate: {p} missing")
    v = json.loads(p.read_text(encoding="utf-8"))
    for k in key.split("."):
        if not isinstance(v, dict) or k not in v:
            raise SystemExit(f"verdict_gate: {p} has no {key}")
        v = v[k]
    return v


def gate(path, key, want):
    v = value(path, key)
    ok = isinstance(v, str) and v in want
    print(f"verdict_gate: {path} {key} is {v}; {'pass' if ok else 'stop'} (wants {', '.join(want)})", flush=True)
    return 0 if ok else 1


def selftest():
    with tempfile.TemporaryDirectory() as td:
        s = Path(td) / "calls.json"
        for v, want in (("ADOPT_FOR_FULL_RUN", 0), ("NO_GAIN", 1), ("NOT_ADOPTED", 1), ("INCOMPLETE", 1)):
            s.write_text(json.dumps({"screen": {"verdict": v}, "full_run": None}), encoding="utf-8")
            assert gate(s, "screen.verdict", ["ADOPT_FOR_FULL_RUN"]) == want, v
        assert gate(s, "screen.verdict", ["NO_GAIN", "INCOMPLETE"]) == 0
        s.write_text(json.dumps({"verdict": "ADOPT"}), encoding="utf-8")
        assert gate(s, "verdict", ["ADOPT"]) == 0
        s.write_text(json.dumps({"screen": {"verdict": {"x": 1}}}), encoding="utf-8")
        assert gate(s, "screen.verdict", ["ADOPT_FOR_FULL_RUN"]) == 1                     # not a string
        for path, key in ((s, "full_run.verdict"), (s, "screen.counts"), (Path(td) / "none.json", "verdict")):
            try:
                gate(path, key, ["ADOPT"])
                raise AssertionError("a missing file or key must stop")
            except SystemExit:
                pass
        s.write_text(json.dumps({"full_run": None}), encoding="utf-8")
        try:
            gate(s, "full_run.verdict", ["ADOPT"])                                       # steps into null
            raise AssertionError("null must stop")
        except SystemExit:
            pass
    print("verdict_gate selftest: OK")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    ap.add_argument("--key")
    ap.add_argument("--want")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if not (a.json and a.key and a.want):
        ap.error("--json, --key and --want are needed")
    try:
        return gate(a.json, a.key, [w for w in a.want.split(",") if w])
    except SystemExit as e:
        print(e, flush=True)
        return 2


if __name__ == "__main__":
    sys.exit(main())
