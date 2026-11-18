"""ENC E2 (docs/ENC_ENCODER_COMPARISON.md section 12.11's log, 19 Nov): an arm's world carries PKO's section-5 verdicts.

    python outputs/mp_unified/zenc2k.py carry --enc mpnet     (the host, in the record's tree)
    python outputs/mp_unified/zenc2k.py selftest

PKO's section-5 checks certify its code: it rebuilds the earlier pools on WebQSP and HotpotQA (a) and U2d's SQuAD caches
(b). D83's chain refuses to build before they pass (zpko.need_checks reads <base>/check/<name>.json). An arm's world runs
that code pinned to the record's (zenc2w), so the record's verdicts carry; the checks are not re-run on the arm's vectors
(they compare against the frozen vectors' pools by design). The four verdict files are copied with a sha256 check into
<world>/outputs/pko/check, beside carried.json naming their sources. Nothing else is read or written; no number reads them.
"""
import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DECLARED = "docs/ENC_ENCODER_COMPARISON.md section 12.11 (development log, 19 Nov)"
NAMES = ("webqsp.json", "hotpotqa.json", "squad__fit.json", "squad__s1eval.json")   # zpko.checks_state's names
ARMS = ("mpnet", "qwen3e06")


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def stop(msg):
    raise SystemExit(f"zenc2k: {msg}")


def carry(enc, root=ROOT):
    root = Path(root)
    if enc not in ARMS:
        stop(f"{enc} is not an arm (the frozen world's D83 chains are the record's)")
    src = root / "outputs" / "pko" / "check"
    world = root.parent / f"e2world_{enc}"
    dst = world / "outputs" / "pko" / "check"
    if not (world / "outputs" / "pko").is_dir():
        stop(f"{world} has no outputs/pko: make the world and its pools first")
    got = []
    for n in NAMES:
        s = src / n
        if not s.is_file():
            stop(f"the record has no {s}")
        v = json.loads(s.read_text(encoding="utf-8")).get("verdict")
        if v != "PASS":
            stop(f"the record's {n} says {v}, not PASS")
        h = sha_file(s)
        d = dst / n
        if d.is_file() and sha_file(d) != h:
            stop(f"{d} holds another file than the record's {n}")
        if not d.is_file():
            d.parent.mkdir(parents=True, exist_ok=True)
            tmp = d.with_name(d.name + ".tmp")
            shutil.copy2(s, tmp)
            if sha_file(tmp) != h:
                stop(f"{tmp} differs from {s} after the copy")
            tmp.replace(d)
        got.append({"name": n, "verdict": v, "sha256": h, "from": f"outputs/pko/check/{n}"})
    rec = {"declared_in": DECLARED, "enc": enc, "carried": got, "code_sha256": sha_file(__file__),
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (dst / "carried.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    print(f"zenc2k: {enc}: carried {len(got)} section-5 verdicts into {dst}")
    return 0


def selftest():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        ws = td / "ws"
        (ws / "outputs" / "pko" / "check").mkdir(parents=True)
        for n in NAMES:
            (ws / "outputs" / "pko" / "check" / n).write_text(json.dumps({"verdict": "PASS", "n": n}), encoding="utf-8")
        try:
            carry("mpnet", ws)
            raise AssertionError("a world without pools must be refused")
        except SystemExit as e:
            assert "no outputs/pko" in str(e), e
        (td / "e2world_mpnet" / "outputs" / "pko").mkdir(parents=True)
        assert carry("mpnet", ws) == 0 and carry("mpnet", ws) == 0          # a second carry finds the same files
        d = td / "e2world_mpnet" / "outputs" / "pko" / "check"
        assert sorted(x.name for x in d.iterdir()) == sorted(NAMES + ("carried.json",))
        assert all(sha_file(d / n) == sha_file(ws / "outputs" / "pko" / "check" / n) for n in NAMES)
        (ws / "outputs" / "pko" / "check" / "webqsp.json").write_text(json.dumps({"verdict": "FAIL"}), encoding="utf-8")
        try:
            carry("mpnet", ws)
            raise AssertionError("a verdict other than PASS must be refused")
        except SystemExit as e:
            assert "not PASS" in str(e), e
        try:
            carry("frozen", ws)
            raise AssertionError("the frozen world must be refused")
        except SystemExit as e:
            assert "not an arm" in str(e), e
    print("zenc2k self-test: OK")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="zenc2k.py")
    ap.add_argument("cmd", choices=("carry", "selftest"))
    ap.add_argument("--enc")
    a = ap.parse_args(argv)
    if a.cmd == "selftest":
        return selftest()
    if not a.enc:
        stop("carry: --enc")
    return carry(a.enc)


if __name__ == "__main__":
    sys.exit(main())
