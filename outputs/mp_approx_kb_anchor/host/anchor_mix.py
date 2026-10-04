"""Design look (untracked; not a result and not filed): the anchor fits with level 8's score form.

Every anchor model fitted on metaqa itself (anchor_kbfit.py: T0, free A256(-1), txt0 / txtc, gropet, cosrot, with and
without mres or a hold) reads at or below the twin on x1 B (R@5 rho -0.15 to +0.01), while level 8's typed walks on the
same KB recover most of the GNN's gain. Two differences are the suspects:
  rule  every kbfit fit protects the twin's rank-1 node (rule p), but metaqa's GNN gain is mostly hit@1 (+11.8 points,
        against +5.8 on recall@5), which rule p cannot reach;
  form  the anchor bonus is additive, s = z + kappa sum_tau p(tau|q) 1[v in R_tau] |R_tau|^-beta, a few z units at most,
        while level 8 scores s = z + kappa log(n m(v) + eta), m(v) = sum_tau p(tau|q) 1[v in R_tau] / |R_tau| + p(null) / n
        (configs/mp_approx_l8.yaml arms.mixture; scripts/mp_approx_l8.py Fitter.mixture, n the pool size), which can lift
        a small reach set over the whole pool.

This look runs a pinned fit unchanged, with --score lmix:
  - l16_look_gate2.scores (the one scorer every anchor fit, select read and read calls) is rebound in-process: a model
    that carries log_eta is scored by the log-mixture, with kappa = exp(log_kappa) and eta = exp(log_eta) learned by the
    same loss (beta is unused there: the mixture is a distribution);
  - every model gets log_eta (starting at --eta0) registered after its construction, so its draws are the pinned fit's.
--score add registers nothing, so every model goes to the original scorer: the pinned fit, bit for bit (the smoke checks
it). The rule comes from the variant name (fit:A256-1/lin/np; p, np, mp as anchor_walk6.RULES).
Modes:
    kb    anchor_kbfit.py fit's arguments (a KB: metaqa)
    gen3  anchor_cos.py fit's arguments (anchor_gen3 with the cos models bound: 2wiki, hotpotqa, musique)
    gen   anchor_gen.py's arguments (T0 and K256 / K1024 on a passage graph)
The record is the pinned fit's (its look name and sha kept, so the summaries read it), with a 'wrapper' block (this
file's sha, the score, eta0) and each seed's eta added from its saved model; a sidecar <out>.mix.json is written
first. Nothing reads a neighbour's score or state: the mixture is a function of the query and the compiled reach sets.

    python outputs/mp_approx_kb_anchor/host/anchor_mix.py kb --score lmix --variants fit:A256-1/lin/np --out .../anchor_kbfit_mix_x.json
    python outputs/mp_approx_kb_anchor/host/anchor_mix.py gen3 --score lmix --dataset 2wiki --variants x4:A256-1/lin/mp --out .../anchor_mix_2w_x.json
"""
import os
import sys

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "4"

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
import anchor_kbfit as KF  # noqa: E402

AC, AG, G3, AW8 = KF.AC, KF.AG, KF.G3, KF.AW8
G2 = AW8.G2
PINS = {"anchor_kbfit": "c0f7eedc6d4563a4b43febe42874d26760f5563d93ac4d0544c966634bfc09f7",
        "l16_look_gate2": "3d6410c545cf5f1e5d4872e30376417e00ba1e95dcb30d8662013301a9848ec7",
        "anchor_walk8": "7e4aa326324f44b3bce7ecee2afd7e79b09c74fda8fa692509f9adb5063b5768",
        "anchor_walk6": "441d68bd85212d0bb1264b8ec269b9f5d440328b7869614c4287c67ca3ab49de"}
ORIG = {"scores": G2.scores, "ac_make_for": AC.make_for, "ag_make_for": AG.make_for, "ac_parse": AC.parse,
        "ag_parse": AG.parse, "kf_parse": KF.parse}
SCORE = {"form": None, "eta0": 0.0}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def check_pins():
    for mod, key in ((KF, "anchor_kbfit"), (G2, "l16_look_gate2"), (AW8, "anchor_walk8"), (KF.AW6, "anchor_walk6")):
        if sha(mod.__file__) != PINS[key]:
            raise SystemExit(f"{mod.__file__} is not the pinned file")


def scores_mix(model, PX):
    """l16_look_gate2.scores for a model without log_eta; the log-mixture for one with it."""
    if not hasattr(model, "log_eta"):
        return ORIG["scores"](model, PX)
    if isinstance(model, G2.LG.Gate) or hasattr(model, "gamma"):
        raise SystemExit("lmix: node gates and the spread-scaled kappa are not in this look")
    P, _X = PX
    qemb, tb, t1, t2, tmask, eq, et, en, ew, z, gold = P
    p = model(qemb, tb, t1, t2, tmask, None, None, None, z, None)        # (B, T): the row's types; the null is the rest
    fin = torch.isfinite(z)
    n = fin.sum(1).to(z.dtype)
    mix = torch.zeros_like(z).index_put((eq, en), p[eq, et] / ew, accumulate=True)
    pnull = (1.0 - p.sum(1)).clamp(min=0.0)
    nm = n[:, None] * mix + pnull[:, None]
    s = torch.where(fin, z + torch.exp(model.log_kappa) * torch.log(nm + torch.exp(model.log_eta)), z)
    return s, gold


def with_eta(make):
    if SCORE["form"] != "lmix":
        return make

    def made():
        m = make()
        if not hasattr(m, "log_eta"):
            m.register_parameter("log_eta", torch.nn.Parameter(torch.tensor(float(SCORE["eta0"]))))
        return m
    return made


def ac_make_for(sp, nt, phi, basis=None):
    return with_eta(ORIG["ac_make_for"](sp, nt, phi, basis))


def ag_make_for(sp, nt, phi):
    return with_eta(ORIG["ag_make_for"](sp, nt, phi))


def tag(sp):
    sp["score"] = SCORE["form"]
    return sp


def install(form, eta0=0.0):
    """Rebind the scorer, the model makers and the parsers (anchor_cos.bind, called by the fits, reads AC's names)."""
    if form not in ("add", "lmix"):
        raise SystemExit("--score add|lmix")
    SCORE.update(form=form, eta0=float(eta0))
    G2.scores = scores_mix
    AC.make_for, AG.make_for = ac_make_for, ag_make_for
    AC.parse = lambda name, train_looks: tag(ORIG["ac_parse"](name, train_looks))
    AG.parse = lambda name, train_looks: tag(ORIG["ag_parse"](name, train_looks))
    KF.parse = lambda name: tag(ORIG["kf_parse"](name))


def post(out, wrapper):
    """Add the wrapper block and each seed's eta (from its saved model) to the record."""
    if not out.exists():
        return
    res = json.loads(out.read_text(encoding="utf-8"))
    res["wrapper"] = wrapper
    for v in res.get("variants", {}).values():
        for s in (v.get("seeds_read") or {}).values():
            mf = Path(s["model_file"])
            mf = mf if mf.is_absolute() else ROOT / mf
            st = torch.load(mf, map_location="cpu", weights_only=False)["state_dict"]
            s["eta"] = float(torch.exp(st["log_eta"]).item()) if "log_eta" in st else None
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, out)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in ("kb", "gen3", "gen"):
        raise SystemExit("anchor_mix.py kb|gen3|gen --score add|lmix [--eta0 x] ...the fit's arguments")
    mode = argv[0]
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", required=True, choices=("add", "lmix"))
    ap.add_argument("--eta0", type=float, default=0.0)
    a, rest = ap.parse_known_args(argv[1:])
    if "--out" not in rest or "--xread" in rest:
        raise SystemExit("a fit with --out (cross reads of these models need the scorer installed: not in this look)")
    out = Path(rest[rest.index("--out") + 1])
    if "mix" not in out.name:
        raise SystemExit("--out must carry 'mix' in its file name")
    out = out if out.is_absolute() else ROOT / out
    check_pins()
    install(a.score, a.eta0)
    wrapper = {"look": "anchor_mix", "mode": mode, "script_sha256": sha(__file__), "score": a.score, "eta0": a.eta0,
               "pins": PINS, "argv": argv}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".mix.json").write_text(json.dumps(wrapper, indent=1), encoding="utf-8")
    if mode == "kb":
        KF.main(["fit"] + rest)
    elif mode == "gen3":
        AC.main(["fit"] + rest)
    else:
        AG.main(rest)
    post(out, wrapper)
    print(f"anchor_mix: {out.name} done ({a.score}, eta0 {a.eta0})", flush=True)


if __name__ == "__main__":
    main()
