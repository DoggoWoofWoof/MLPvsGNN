"""ENC E7, section 23.9: the storage call.

A form is TOLERABLE on a dataset when, on each of d5, d20 and rrf5, the lower end of the 95% interval of its gap to
F_exact (23.6's paired bootstrap, read from read.json) is at least -1.00 point. The storage call names the smallest
form (23.6's summed listed-set bytes, shared tables included, ties by name) TOLERABLE on all three datasets, else exact.

    python outputs/mp_unified/zenc7c.py call       # needs the three read.json files; writes storage_call.json / .md
    python outputs/mp_unified/zenc7c.py selftest   # the rule on made-up cells

zenc7.py is imported for OUT, FORMS and DSETS only and is not edited.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import zenc7 as Z7  # noqa: E402

TOL = -1.00
MEAS = ("d5", "d20", "rrf5")
DECLARED = "docs/ENC_ENCODER_COMPARISON.md section 23.9"


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def tolerable(cell):
    return all(float(cell["gaps"][m][1]) >= TOL for m in MEAS)


def lowest(cell):
    return min(float(cell["gaps"][m][1]) for m in MEAS)


def storage_call(reads, forms):
    tot = {f: sum(int(r["storage"][f]["bytes_set"]) for r in reads.values()) for f in forms}
    tol = {f: {ds: tolerable(r["cells"][f]) for ds, r in reads.items()} for f in forms}
    pick = next((f for f in sorted((f for f in forms if f != "exact"), key=lambda f: (tot[f], f))
                 if all(tol[f].values())), "exact")
    return pick, tot, tol


def cmd_call():
    reads, miss = {}, []
    for ds in Z7.DSETS:
        p = Z7.OUT / ds / "read.json"
        if not p.is_file():
            miss.append(ds)
            continue
        reads[ds] = json.loads(p.read_text(encoding="utf-8"))
    if miss:
        sys.exit(f"23.9: no read for {miss}")
    bad = [ds for ds, r in reads.items() if r["g_r"] != "PASS" or r.get("g_d", "PASS") != "PASS"]
    if bad:
        sys.exit(f"23.9: a gate failed on {bad}")
    forms = list(Z7.FORMS)
    pick, tot, tol = storage_call(reads, forms)
    utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    rec = {"stage": "ENC E7: storage call", "declared_in": DECLARED, "tolerance_pts": TOL, "measures": list(MEAS),
           "call": pick, "set_bytes_three": tot, "tolerable": tol,
           "lowest_lower_end": {f: {ds: lowest(r["cells"][f]) for ds, r in reads.items()} for f in forms},
           "reads": {ds: sha256(Z7.OUT / ds / "read.json") for ds in reads},
           "code_sha256": sha256(Path(__file__)), "utc": utc}
    out = Z7.OUT / "storage_call.json"
    out.write_text(json.dumps(rec, indent=1), encoding="utf-8")
    ex = tot["exact"]
    L = [f"# ENC E7: storage call (section 23.9, {utc})", "",
         f"**Storage call: {pick}** ({tot[pick]:,} bytes over the three listed sets, "
         f"{100 * tot[pick] / ex:.1f}% of exact's {ex:,}).", "",
         "| form | bytes, three sets | % of exact | " + " | ".join(f"{ds} lowest lower end" for ds in reads)
         + " | TOLERABLE on all |",
         "|---|---|---|" + "---|" * len(reads) + "---|"]
    for f in sorted(forms, key=lambda f: (tot[f], f)):
        cells = " | ".join(f"{lowest(reads[ds]['cells'][f]):+.2f}{'' if tol[f][ds] else ' x'}" for ds in reads)
        L.append(f"| {f} | {tot[f]:,} | {100 * tot[f] / ex:.1f} | {cells} | {'yes' if all(tol[f].values()) else 'no'} |")
    (Z7.OUT / "storage_call.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    return 0


def selftest():
    def cell(lo):
        return {"gaps": {m: [lo + 0.3, lo, lo + 0.6, "LEVEL"] for m in MEAS}}

    def st(b):
        return {"bytes_set": b}

    forms = ["exact", "int8", "res2", "res1"]
    reads = {"a": {"cells": {"exact": cell(0.0), "int8": cell(-0.2), "res2": cell(-0.99), "res1": cell(-1.01)},
                   "storage": {"exact": st(100), "int8": st(50), "res2": st(12), "res1": st(6)}},
             "b": {"cells": {"exact": cell(0.0), "int8": cell(-0.5), "res2": cell(-1.00), "res1": cell(-0.1)},
                   "storage": {"exact": st(100), "int8": st(50), "res2": st(12), "res1": st(6)}}}
    pick, tot, tol = storage_call(reads, forms)
    assert pick == "res2" and tot["res2"] == 24 and tol["res1"] == {"a": False, "b": True}, (pick, tot, tol)
    reads["b"]["cells"]["res2"] = cell(-1.02)
    assert storage_call(reads, forms)[0] == "int8"
    reads["a"]["cells"]["int8"] = cell(-3.0)
    assert storage_call(reads, forms)[0] == "exact"
    print("zenc7c self-test PASS")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=("call", "selftest"))
    a = p.parse_args()
    sys.exit(cmd_call() if a.cmd == "call" else selftest())
