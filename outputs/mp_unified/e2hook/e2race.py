"""E2's rerun wrapper for the arm's D83 chains (systems code; no science).

    python outputs/mp_unified/e2hook/e2race.py [--tries 3] [--wait 90] -- COMMAND ...
    python outputs/mp_unified/e2hook/e2race.py selftest

host_ops/retry_cmd.py reruns a command that failed on a transient Windows file race (WinError 5 or 32) only when the
LAST 4,000 characters of its output name the race. A D83 chain runs its look shards six lanes at a time, and on a dataset
whose label sets are not on disk yet every lane writes the same labels/<dataset>/<i>.npy through os.replace. On 19 Nov
two lanes of mpnet's D83F MuSiQue chain (w2b-chain-f-musique) lost that race at 02:59 (WinError 5), the other 24 shards
ran on for two hours, and the chain ended at 04:54 with the race long out of retry_cmd's window. zd83 build skips a
step already filed and write_label_sets compares a label file already on disk instead of writing it, so the chain is
safe to rerun and a rerun finds the labels in place. This wrapper is retry_cmd's rule with the whole output scanned:
when the command exits nonzero and ANY line of its output names WinError 5 or 32, it runs again after --wait seconds,
up to --tries runs in all; any other failure ends at once with the command's code. The output passes through
unchanged. It lives in the junctioned e2hook folder (E2's own, not pinned), so it reaches both worlds without a refresh.
"""
import argparse
import subprocess
import sys
import time

RACE = (b"[WinError 5]", b"[WinError 32]")


def run(cmd, tries, wait):
    rc = 1
    for k in range(1, tries + 1):
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        seen, edge = set(), b""
        for chunk in iter(lambda: p.stdout.read1(1 << 14), b""):
            sys.stdout.buffer.write(chunk)
            sys.stdout.buffer.flush()
            buf = edge + chunk
            seen.update(r.decode() for r in RACE if r in buf)
            edge = buf[-32:]
        rc = p.wait()
        if rc == 0:
            return 0
        if not seen or k == tries:
            if seen:
                print(f"[e2race] run {k} of {tries} failed after {sorted(seen)}; no runs left (rc {rc})", flush=True)
            return rc
        print(f"[e2race] run {k} of {tries} failed after {sorted(seen)} (rc {rc}); again in {wait:g} s", flush=True)
        time.sleep(wait)
    return rc


def selftest():
    import os
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        mark = os.path.join(d, "ran")
        # the race early, 6,000 characters after it, then rc 1 on the first run and rc 0 on the second
        child = ("import os,sys\nm=sys.argv[1]\nfirst=not os.path.exists(m)\nopen(m,'a').write('x')\n"
                 "print('PermissionError: [WinError 5] Access is denied: a.tmp -> a.npy' if first else 'ok')\n"
                 "print('y'*6000)\nsys.exit(1 if first else 0)\n")
        f = os.path.join(d, "child.py")
        with open(f, "w", encoding="utf-8") as h:
            h.write(child)
        assert run([sys.executable, f, mark], 3, 0) == 0
        assert open(mark).read() == "xx", "the race run is run again once"
        os.remove(mark)
        bad = os.path.join(d, "bad.py")
        with open(bad, "w", encoding="utf-8") as h:
            h.write("import sys\nprint('ValueError: no race here')\nsys.exit(3)\n")
        assert run([sys.executable, bad], 3, 0) == 3, "any other failure ends at once with its code"
        rep = os.path.join(d, "rep.py")
        with open(rep, "w", encoding="utf-8") as h:
            h.write("import sys\nopen(sys.argv[1],'a').write('x')\nprint('[WinError 32] in use')\nsys.exit(2)\n")
        assert run([sys.executable, rep, mark], 3, 0) == 2 and open(mark).read() == "xxx", "a race on every run: --tries"
    print("e2race selftest: OK", flush=True)
    return 0


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["selftest"]:
        return selftest()
    ap = argparse.ArgumentParser()
    ap.add_argument("--tries", type=int, default=3)
    ap.add_argument("--wait", type=float, default=90.0)
    if "--" not in argv:
        ap.error("COMMAND goes after --")
    i = argv.index("--")
    a = ap.parse_args(argv[:i])
    cmd = argv[i + 1:]
    if not cmd:
        ap.error("no command")
    return run(cmd, a.tries, a.wait)


if __name__ == "__main__":
    sys.exit(main())
