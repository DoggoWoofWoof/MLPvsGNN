"""SEM2 R2c, Part A (docs/SEM2_EDGE_OPERATORS.md section 10): R2a's calls at 20 candidates, and the share of questions whose pool
holds every gold, read from the per-question arrays R2a already stored. No traversal runs here, no GPU, no host job.

    python outputs/mp_unified/zsem2tr.py selftest
    python outputs/mp_unified/zsem2tr.py report

What it reads: outputs/sem2/r2/read/{e3v,frozen}/<dataset>.json and .npz (R2a's records; zsem2t.py's code, never edited). A record whose
script_sha256 is not zsem2t.py's, or whose npz does not hash to the record's npz_sha256, stops the report.
Per question the stored recall is min(hits, golds) / golds; "ALL at 20" is the indicator of recall = 1, i.e. every gold of the question sits
in the top 20 of RRF(v, list, SPLADE) (NaN stays NaN: a question with no gold is left out). The calls are zsem1.boot's (paired bootstrap over
questions, 2,000 resamples, seed 0, 95% interval; ABOVE / LEVEL / BELOW), as 9.6.
Outputs: outputs/sem2/r2/r2c/report.json and report.md.
"""
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import json  # noqa: E402

import numpy as np  # noqa: E402

import zsem2t as T  # noqa: E402

S1, Z = T.S1, T.Z
OUT = T.OUT / "r2c"
DECLARED = "docs/SEM2_EDGE_OPERATORS.md section 10"
EPS = 1e-9


def all_of(r):
    """Per-question recall -> 1.0 where every gold is in the top 20, 0.0 where one is not, NaN where there is no gold."""
    r = np.asarray(r, np.float64)
    return np.where(np.isnan(r), np.nan, (r >= 1.0 - EPS).astype(np.float64))


def calls_of(z, conv):
    """One measure at 20: T1, T2, the beside calls and the means. conv maps a stored recall array to the measure."""
    g = {k: conv(z[f"r_{k}"]) for k in ("base20", "ctrl20", "trav_fused20", "trav0_fused20", "oracle_fused20")}
    c = {"T1 trav - base": S1.boot(g["trav_fused20"], g["base20"]),
         "T2 trav - trav0": S1.boot(g["trav_fused20"], g["trav0_fused20"]),
         "beside": {"trav - SEM1 control": S1.boot(g["trav_fused20"], g["ctrl20"]),
                    "trav0 - base": S1.boot(g["trav0_fused20"], g["base20"]),
                    "C oracle - trav": S1.boot(g["oracle_fused20"], g["trav_fused20"])},
         "mean": {k: round(100 * float(np.nanmean(v)), 3) for k, v in g.items()}}
    c["ADDS"] = c["T1 trav - base"]["call"] == "ABOVE" and c["T2 trav - trav0"]["call"] == "ABOVE"
    c["ROOM"] = c["beside"]["C oracle - trav"]["diff"] >= T.ROOM
    return c


def verdict(rows, have_all):
    """9.6's e3v verdict by ADDS, on one measure."""
    if not have_all:
        return "INCOMPLETE"
    if all(c["ADDS"] for c in rows.values()):
        return "HOLDS"
    if any(c["ADDS"] for c in rows.values()) and not any(c["T1 trav - base"]["call"] == "BELOW" for c in rows.values()):
        return "PARTIAL"
    return "FAILS"


def load_row(arm, ds, sha):
    p = T.OUT / "read" / arm / f"{ds}.json"
    if not p.is_file():
        return None
    r = json.loads(p.read_text(encoding="utf-8"))
    if r["script_sha256"] != sha:
        T.stop(f"{arm} {ds}: read by other code")
    q = T.OUT / "read" / arm / f"{ds}.npz"
    if Z.sha_file(q) != r["npz_sha256"]:
        T.stop(f"{arm} {ds}: npz sha256 is not its record's")
    z = np.load(q)
    for k in ("base20", "trav_fused20", "trav0_fused20"):              # G-A: the stored array is the record's mean
        m = round(float(np.nanmean(z[f"r_{k}"])), 6)
        if abs(m - r["recall"][k]) > 1e-6:
            T.stop(f"{arm} {ds}: gate G-A: r_{k} averages {m}, the record says {r['recall'][k]}")
    return r, z


def cmd_report():
    sha = T.script_sha()
    calls = {}
    for arm in T.SETS:
        for ds in T.SETS[arm]:
            row = load_row(arm, ds, sha)
            if row is None:
                continue
            r, z = row
            calls[f"{arm}/{ds}"] = {"recall at 20": calls_of(z, lambda x: np.asarray(x, np.float64)),
                                    "ALL at 20": calls_of(z, all_of),
                                    "chosen": {k: v["lambda"] for k, v in r["chosen"].items()},
                                    "questions": int(np.sum(~np.isnan(z["r_base20"])))}
    e3 = {ds: calls.get(f"e3v/{ds}") for ds in Z.SEVEN}
    have_all = all(e3.values())
    ver = {m: verdict({ds: c[m] for ds, c in e3.items() if c}, have_all) for m in ("recall at 20", "ALL at 20")}
    adds_all = [ds for ds, c in e3.items() if c and c["ALL at 20"]["ADDS"]]
    part_b = "INCOMPLETE" if not have_all else ("NAMED" if adds_all else "CLOSED")
    rep = {"stage": "SEM2 R2c Part A", "declared_in": DECLARED, "verdict_e3v": ver, "adds_all_at_20": adds_all,
           "part_b": part_b, "calls": calls, "script_sha256_of_reads": sha,
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    OUT.mkdir(parents=True, exist_ok=True)
    Z.wjson(OUT / "report.json", rep)
    lines = ["# SEM2 R2c Part A", "", f"Declared in {DECLARED}. e3v verdict by recall at 20: **{ver['recall at 20']}**; by ALL at 20: "
             f"**{ver['ALL at 20']}**; ADDS on ALL at 20: {', '.join(adds_all) or 'none'}; Part B: **{part_b}**.", ""]
    for m in ("recall at 20", "ALL at 20"):
        lines += [f"## {m} (points; paired bootstrap, 2,000 resamples, seed 0)", "",
                  "| arm/dataset | base | trav0 | trav | oracle | T1 = trav - base | T2 = trav - trav0 | trav - control | C = oracle - trav | ADDS |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for k, c in calls.items():
            x = c[m]
            mn = x["mean"]
            lines.append(f"| {k} | {mn['base20']:.2f} | {mn['trav0_fused20']:.2f} | {mn['trav_fused20']:.2f} | {mn['oracle_fused20']:.2f} | "
                         f"{S1.fmt(x['T1 trav - base'])} | {S1.fmt(x['T2 trav - trav0'])} | {S1.fmt(x['beside']['trav - SEM1 control'])} | "
                         f"{S1.fmt(x['beside']['C oracle - trav'])} | {x['ADDS']} |")
        lines.append("")
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    T.log(f"R2c Part A: recall@20 {ver['recall at 20']}, ALL@20 {ver['ALL at 20']}, ADDS on ALL {adds_all}, Part B {part_b}")
    return 0


def selftest():
    r = np.array([1.0, 0.5, np.nan, 1.0, 0.0, 2 / 3, 1.0])
    a = all_of(r)
    assert np.array_equal(np.isnan(a), np.isnan(r)), a
    assert a[~np.isnan(a)].tolist() == [1.0, 0.0, 1.0, 0.0, 0.0, 1.0], a
    rng = np.random.default_rng(0)
    n = 400
    base = np.clip(rng.integers(0, 4, n) / 3.0, 0, 1)
    up = np.minimum(1.0, base + (rng.random(n) < 0.15) * (1 / 3))        # trav: some questions gain a gold
    z = {"r_base20": base, "r_ctrl20": base, "r_trav_fused20": up, "r_trav0_fused20": base, "r_oracle_fused20": up}
    for conv in (lambda x: np.asarray(x, np.float64), all_of):
        c = calls_of(z, conv)
        assert c["T1 trav - base"]["diff"] >= 0 and c["T2 trav - trav0"]["diff"] >= 0, c
    c = calls_of(z, lambda x: np.asarray(x, np.float64))
    assert c["ADDS"] and c["T1 trav - base"]["call"] == "ABOVE", c            # planted gain, as 15 percent of 400 questions
    flat = calls_of({**z, "r_trav_fused20": base, "r_oracle_fused20": base}, all_of)
    assert not flat["ADDS"] and flat["T1 trav - base"]["call"] == "LEVEL", flat
    assert verdict({"a": c}, True) == "HOLDS" and verdict({"a": flat}, True) == "FAILS" and verdict({"a": c, "b": flat}, True) == "PARTIAL"
    assert verdict({"a": c}, False) == "INCOMPLETE"
    T.log("zsem2tr selftest: PASS")
    return 0


def main(argv):
    if len(argv) != 1 or argv[0] not in ("selftest", "report"):
        raise SystemExit("usage: zsem2tr.py selftest | report")
    return selftest() if argv[0] == "selftest" else cmd_report()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
