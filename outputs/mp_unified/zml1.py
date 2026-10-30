"""ML1's rule (docs/ML1_REFITS_ON_PKO.md, sections 3 and 4): the calls on one arm's twelve reads against its base, at three
seeds, and the arm's verdict.

This file is only the rule. The fits, the reads and the wiring to PKO's records are written when PKO's arrays exist and
their record shapes can be checked; nothing here trains or reads a model.

Commands:
  selftest        the rule on hand-built reads, and on the real shape of WQG's and WQK's filed floors
  dryrun-today    the rule applied to the seeds already filed on today's pools (zg6r against zspk). That is NOT ML1's
                  result: ML1 reads PKO's pools, fits its bases again and builds its own floors. It tests the code on
                  real numbers and shows how the rule would have called the build that exists.
"""
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FLOOR = 0.0075                      # docs/SCREENS.md: no floor below this (equals lean_screen's)
EPS = 1e-9                          # a difference exactly at its floor is at the floor
SEEDS = (0, 1, 2)
SPLITS = ("WQ-musique", "WQ-hotpotqa")
DATASETS = ("metaqa", "squad", "musique", "hotpotqa", "2wiki", "webqsp")
READS = tuple((s, d) for s in SPLITS for d in DATASETS)     # twelve per arm
GAIN, LOSS, WITHIN, UNCONFIRMED = "GAIN", "LOSS", "WITHIN", "UNCONFIRMED_GAIN"


def spread(r5):
    """An arm's spread on one read: the root mean square of its seed-1 and seed-2 differences from its own seed 0."""
    d1, d2 = r5[1] - r5[0], r5[2] - r5[0]
    return math.sqrt((d1 * d1 + d2 * d2) / 2.0)


def pair_floor(spreads):
    """WQG's pair floor: max(0.0075, 2 x the root mean square of the arms' spreads)."""
    return max(FLOOR, 2.0 * math.sqrt(sum(s * s for s in spreads) / len(spreads)))


def call(d, floor):
    """ML1 section 4. d = {seed: arm minus base at that seed}. A GAIN needs seed 0 at the floor and seeds 1 and 2 above zero;
    a seed-0 gain that they do not confirm is UNCONFIRMED_GAIN and counts as WITHIN. A LOSS stands on seed 0 alone."""
    if d[0] <= -floor + EPS:
        return LOSS
    if d[0] >= floor - EPS:
        return GAIN if d[1] > 0 and d[2] > 0 else UNCONFIRMED
    return WITHIN


def problems(new, base):
    """Why the inputs are not an arm's twelve reads at three seeds against a base's (an INCOMPLETE), or []."""
    out = []
    for nm, arm in (("arm", new), ("base", base)):
        for rd in READS:
            v = arm.get(rd)
            if not isinstance(v, dict) or sorted(v) != list(SEEDS):
                out.append(f"{nm}: {rd[0]} {rd[1]} is not read at seeds {list(SEEDS)}")
            elif any(not isinstance(x, (int, float)) or math.isnan(x) for x in v.values()):
                out.append(f"{nm}: {rd[0]} {rd[1]} has a missing R@5")
        extra = sorted(set(arm) - set(READS))
        if extra:
            out.append(f"{nm}: reads outside the twelve: {extra}")
    return out


def verdict(new, base):
    """new, base: {(split, dataset): {seed: R@5}}. Returns the reads' calls and the verdict (ML1 section 4)."""
    bad = problems(new, base)
    if bad:
        return {"verdict": "INCOMPLETE", "problems": bad, "reads": []}
    rows = []
    for rd in READS:
        d = {k: new[rd][k] - base[rd][k] for k in SEEDS}
        fl = pair_floor([spread(new[rd]), spread(base[rd])])
        rows.append({"split": rd[0], "dataset": rd[1], "delta": {k: round(d[k], 4) for k in SEEDS},
                     "floor": round(fl, 6), "call": call(d, fl)})
    n = {c: sum(1 for r in rows if r["call"] == c) for c in (GAIN, LOSS, UNCONFIRMED, WITHIN)}
    v = "ADOPT" if n[LOSS] == 0 and n[GAIN] >= 1 else "NOT_ADOPTED"
    return {"verdict": v, "counts": n, "problems": [], "reads": rows}


# ── what is filed on today's pools ──────────────────────────────────────────────────────────────────────────────


def filed_arm(path):
    """{(split, dataset): {seed: R@5}} from a floors record (seed-0 R@5 and the seed-1 and seed-2 differences, 4 places)."""
    rec = json.loads(Path(path).read_text(encoding="utf-8"))
    if rec.get("status") != "OK":
        raise SystemExit(f"{path}: status {rec.get('status')}, not OK")
    out = {}
    for x in rec["reads"]:
        out[(x["split"], x["dataset"])] = {0: x["seed0"], 1: x["seed0"] + x["d1"], 2: x["seed0"] + x["d2"]}
    return out, {(x["split"], x["dataset"]): x for x in rec["reads"]}


def today(arm):
    f = {"zg6r": "outputs/wqg/floors-zg6r.json", "zg6": "outputs/wqg/floors-zg6.json",
         "zspk": "outputs/wqk/floors-zspk.json", "zrckp": "outputs/wqk/floors-zrckp.json"}[arm]
    return filed_arm(ROOT / f)


def dryrun_today(out=None):
    new, _ = today("zg6r")
    base, _ = today("zspk")
    res = verdict(new, base)
    lines = ["ML1's rule on today's pools, zg6r against zspk. NOT ML1's result: other pools, no refit, the filed seeds.",
             f"verdict (dry run): {res['verdict']}   counts: {res.get('counts')}", ""]
    for r in res["reads"]:
        d = r["delta"]
        lines.append(f"{r['split']:12s} {r['dataset']:9s} d0 {d[0]:+.4f} d1 {d[1]:+.4f} d2 {d[2]:+.4f}  floor {r['floor']:.4f}  {r['call']}")
    txt = "\n".join(lines)
    print(txt)
    if out:
        Path(out).write_text(txt + "\n", encoding="utf-8")
    return res


# ── self-test ───────────────────────────────────────────────────────────────────────────────────────────────────


def flat(v=0.5, seeds=None):
    return {k: v + (seeds or {}).get(k, 0.0) for k in SEEDS}


def arm_of(f):
    """An arm whose every read is f(read) -> {seed: R@5}."""
    return {rd: f(rd) for rd in READS}


def selftest():
    # 1. the spread and the floor, by hand
    assert abs(spread({0: 0.5, 1: 0.5, 2: 0.5})) < 1e-15
    assert abs(spread({0: 0.5, 1: 0.53, 2: 0.46}) - math.sqrt((0.03 ** 2 + 0.04 ** 2) / 2)) < 1e-12
    assert pair_floor([0.0, 0.0]) == FLOOR                                  # never below 0.0075
    assert abs(pair_floor([0.03, 0.04]) - 2 * math.sqrt((0.03 ** 2 + 0.04 ** 2) / 2)) < 1e-12
    assert abs(pair_floor([0.02]) - 0.04) < 1e-12                           # one arm: its own floor

    # 2. the calls (floor 0.01)
    F = 0.01
    assert call({0: 0.02, 1: 0.005, 2: 0.011}, F) == GAIN
    assert call({0: 0.02, 1: 0.005, 2: -0.001}, F) == UNCONFIRMED           # seed 2 disagrees
    assert call({0: 0.02, 1: 0.0, 2: 0.011}, F) == UNCONFIRMED              # zero is not above zero
    assert call({0: 0.01, 1: 0.002, 2: 0.002}, F) == GAIN                   # exactly at the floor
    assert call({0: 0.0099, 1: 0.02, 2: 0.02}, F) == WITHIN                 # seeds 1 and 2 cannot make a seed-0 WITHIN a GAIN
    assert call({0: -0.02, 1: 0.03, 2: 0.03}, F) == LOSS                    # a LOSS stands on seed 0 alone
    assert call({0: -0.01, 1: 0.0, 2: 0.0}, F) == LOSS                      # exactly at minus the floor
    assert call({0: -0.0099, 1: -0.1, 2: -0.1}, F) == WITHIN

    # 3. verdicts on a twelve-read arm. The base is flat 0.50 at every seed (spread 0), so every floor is 0.0075.
    base = arm_of(lambda rd: flat(0.50))

    def arm(gains=(), losses=(), skip=()):
        out = {}
        for rd in READS:
            if rd in skip:
                continue
            if rd in gains:
                out[rd] = {0: 0.52, 1: 0.52, 2: 0.52}
            elif rd in losses:
                out[rd] = {0: 0.48, 1: 0.48, 2: 0.48}
            else:
                out[rd] = {0: 0.50, 1: 0.50, 2: 0.50}
        return out

    r0, r1, r2 = READS[0], READS[1], READS[7]
    v = verdict(arm(gains=[r0]), base)
    assert v["verdict"] == "ADOPT" and v["counts"][GAIN] == 1 and v["counts"][LOSS] == 0
    v = verdict(arm(gains=[r0], losses=[r2]), base)
    assert v["verdict"] == "NOT_ADOPTED" and v["counts"][LOSS] == 1                  # one LOSS blocks, whatever the GAINs
    # an unconfirmed gain needs a base that moves at one seed: against a flat base an arm's own spread already forces
    # its seed-0 difference under its floor whenever another seed is at or below the base. The base reads 0.60 at seed 2.
    nb = arm_of(lambda rd: {0: 0.50, 1: 0.50, 2: 0.60})
    na = arm_of(lambda rd: {0: 0.50, 1: 0.50, 2: 0.60})                                # follows the base: WITHIN everywhere
    na[r0] = {0: 0.70, 1: 0.70, 2: 0.59}                                               # +0.20, +0.20, -0.01
    na[r1] = {0: 0.70, 1: 0.70, 2: 0.80}                                               # +0.20 at every seed
    v = verdict(na, nb)
    assert v["counts"][UNCONFIRMED] == 1 and v["counts"][GAIN] == 1, v["counts"]
    assert [r["call"] for r in v["reads"][:2]] == [UNCONFIRMED, GAIN] and v["verdict"] == "ADOPT"
    na[r1] = {0: 0.50, 1: 0.50, 2: 0.60}
    v = verdict(na, nb)
    assert v["counts"][UNCONFIRMED] == 1 and v["counts"][GAIN] == 0 and v["verdict"] == "NOT_ADOPTED"
    v = verdict(arm(), base)
    assert v["verdict"] == "NOT_ADOPTED" and v["counts"][WITHIN] == 12
    v = verdict(arm(skip=[r2]), base)
    assert v["verdict"] == "INCOMPLETE" and any(r2[1] in p for p in v["problems"])
    bad = arm(gains=[r0])
    bad[r1] = {0: 0.5, 1: float("nan"), 2: 0.5}
    assert verdict(bad, base)["verdict"] == "INCOMPLETE"
    bad = arm(gains=[r0])
    bad[r1] = {0: 0.5, 1: 0.5}                                                       # a seed missing
    assert verdict(bad, base)["verdict"] == "INCOMPLETE"
    extra = arm(gains=[r0])
    extra[("WQ-musique", "nope")] = flat()
    assert verdict(extra, base)["verdict"] == "INCOMPLETE"
    # a noisy base raises the floor: the same +0.02 at every seed is a GAIN against a flat base, WITHIN against a noisy one
    noisy = arm_of(lambda rd: {0: 0.50, 1: 0.54, 2: 0.46})
    a = arm_of(lambda rd: {0: 0.52, 1: 0.56, 2: 0.48})
    assert verdict(a, noisy)["verdict"] == "NOT_ADOPTED"                             # delta +0.02 < floor 2 x 0.04 / sqrt(2)
    assert verdict(arm_of(lambda rd: {0: 0.52, 1: 0.52, 2: 0.52}), base)["verdict"] == "ADOPT"

    # 4. the real shape: the filed floors reproduce from the seed-0 R@5 and the two differences, for the four arms
    for nm in ("zg6r", "zspk", "zrckp", "zg6"):
        try:
            arm_r5, raw = today(nm)
        except (OSError, SystemExit) as e:
            if nm in ("zg6r", "zspk"):
                raise SystemExit(f"selftest: {nm}'s filed floors are needed: {e}")
            print(f"selftest: {nm}'s filed floors not read here ({e}); skipped")
            continue
        assert sorted(arm_r5) == sorted(READS), nm
        for rd, x in raw.items():
            sp = spread(arm_r5[rd])
            assert abs(sp - x["spread"]) < 2e-6, (nm, rd, sp, x["spread"])           # the spread the arm's own file filed
            assert abs(pair_floor([sp]) - x["floor"]) < 4e-6, (nm, rd)                # and its one-arm floor
    # 5. the compares agree with the floors records: a seed-1 compare's new R@5 is the floors record's seed0 + d1
    for arm_name, fn in (("zg6r", "outputs/screen/scr-wqg-zg6r-s1{}.json"), ("zspk", "outputs/screen/scr-wqk-zspk-s1{}.json")):
        arm_r5, _ = today(arm_name)
        for sp, sfx in (("WQ-musique", ""), ("WQ-hotpotqa", "-hp")):
            p = ROOT / fn.format(sfx)
            if not p.is_file():
                print(f"selftest: {p.name} not here; skipped")
                continue
            js = json.loads(p.read_text(encoding="utf-8"))
            rows = {r["dataset"]: r for r in js["by_base"][js["bases"][0]]["rows"]}
            assert sorted(rows) == sorted(DATASETS), p.name
            for ds, r in rows.items():
                assert abs(r["new"][0] - arm_r5[(sp, ds)][1]) < 2e-4, (p.name, ds)
                assert abs(r["base"][0] - arm_r5[(sp, ds)][0]) < 2e-4, (p.name, ds)
    # 6. the dry run runs on the real numbers and returns twelve called reads
    res = dryrun_today()
    assert res["verdict"] in ("ADOPT", "NOT_ADOPTED") and len(res["reads"]) == 12
    print("zml1 selftest: ok")


def main(argv=None):
    a = list(sys.argv[1:] if argv is None else argv)
    if a[:1] == ["selftest"]:
        return selftest()
    if a[:1] == ["dryrun-today"]:
        return dryrun_today(a[1] if len(a) > 1 else None)
    raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
