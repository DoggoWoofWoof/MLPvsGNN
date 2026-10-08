"""Screens, fifteenth round (docs/SCREENS.md): rmatch's question-relation match stacked on zret's finished fit. zrs is
zrm's model (rmatch.ChainMatch over lean_screen3.ZRet, unchanged), trained in a second stage. Its zret part is zret's
fit of the split, the p@swa state, loaded and frozen. Only the match's parameters train: on the fit's typed training
carves (metaqa; webqsp never trains), against the frozen model's scores, with the fit's own config and seed and
batches of 32 questions. The base takes no gradient, so every state of the fit holds zret's p@swa state bit for bit.
On a graph without typed relations (squad, musique, hotpotqa, 2wiki) every score is zret's fit's. The match is an
objective on the existing graph: a residual learned on top of the finished model, where rmatch and zrm train the two
jointly. No new column, block or hyperparameter. Two fits (L-musique and L-hotpotqa), each compared with zret's fit of
its split (that comparison decides) and with rmatch's and step 1's (reported); the pair, its re-call under the seed
null and the full run's grade are relz.py's, run under zrs's name with zret's fits in place of rel's (zgs.py's
mapping of zret's fits).

    python outputs/mp_unified/zrs.py smoke --device cuda --host
    python outputs/mp_unified/zrs.py train --split L-musique --name scr-zrs --arm zrs --device cuda --host
    python outputs/mp_unified/zrs.py read --name scr-zrs --device cuda --host
    python outputs/mp_unified/zrs.py compare --new outputs/screen/fits/scr-zrs \\
        --base outputs/screen/fits/scr-zret,outputs/screen/fits/scr-rmatch,outputs/step1/fits/L-musique \\
        --out outputs/screen/scr-zrs
    python outputs/mp_unified/zrs.py pair --screens outputs/screen/scr-zrs.json,outputs/screen/scr-zrs-hp.json \\
        --out outputs/screen/scr-zrs-pair
    python outputs/mp_unified/zrs.py recall --null N1,N2,N3,N4 --pair outputs/screen/scr-zrs-pair.json \\
        --out outputs/screen/scr-zrs-pair-recall
    python outputs/mp_unified/zrs.py grade --full-root outputs/full_zrs \\
        --reuse L-musique=outputs/screen/scr-zrs.json,L-hotpotqa=outputs/screen/scr-zrs-hp.json
    python outputs/mp_unified/zrs.py --selftest
train takes --base-fit DIR (default: zret's fit of the split, zgs.zret_fit). N1..N4 are the seed null's comparisons
(outputs/screen/scr-null-s1, -s2, -s1-hp, -s2-hp .json). The full run's re-grade is nullx.py's regrade with zret's six
comparisons as --base-compares.

Speed (8216ffe, docs/FULL_ROUND12.md section 6): the chains are a query-local compile, so any latency figure for this
arm is cold: each question timed from scratch, the typed walk from its pool's arrays, the move of its entries to the
device and the forward, with no warm-up pass and nothing kept from an earlier question.
"""
import os
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import contextlib  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import lean_screen3 as S3  # noqa: E402
import rmatch as RM  # noqa: E402
import zgs as G  # noqa: E402
import zrm as ZM  # noqa: E402

Z, R = RM.Z, RM.R
S2, S = RM.S2, RM.S
LG, LC = S.LG, S.LC
L8 = LG.L8
log = S.log
ARM, BASE_ARM = "zrs", "zret"
ROOT = S.ROOT


class ZRS(ZM.ZRM):
    """zrs: zrm's model (rmatch's gated chain match added to zret's forward), trained by fit_stacked."""


S.ARMS.update({ARM: (ZRS, RM.ChainCarveBase)})
STACK = {"base_fit": None, "info": None}


def zret_fit_dir(split):
    """zret's fit of a split (zgs.zret_fit): its screen fit on L-musique, its full run's fit on every other split."""
    return ROOT / "outputs" / Path(*G.zret_fit(split))


def rel(p):
    """A path as recorded: relative to the repository, with forward slashes (no host path in a record)."""
    p = Path(p).resolve()
    try:
        return p.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return p.name


# ── the second stage ─────────────────────────────────────────────────────────


def load_base(fit_dir, vname, blocks, widths, cfg, seed, hidden, ctx, train, basis_sha256):
    """zret's fit: its blob and its {vname}@swa state. Refuses a fit trained as another arm, or with other blocks,
    widths, context, hidden size, seed, config, training carves or basis."""
    fit_dir = Path(fit_dir)
    arm = json.loads((fit_dir / "screen.json").read_text(encoding="utf-8")).get("arm")
    if arm != BASE_ARM:
        raise SystemExit(f"zrs: the base fit {rel(fit_dir)} was trained as {arm}, not {BASE_ARM}")
    blob = torch.load(fit_dir / "models.pt", map_location="cpu", weights_only=False)
    v = blob["variants"].get(vname) or {}
    want = {"blocks": list(blocks), "widths": dict(widths), "ctx": ctx, "hidden": hidden, "seed": seed,
            "config": dict(cfg), "train": [tuple(x) for x in train], "basis_sha256": basis_sha256}
    got = {"blocks": list(v.get("blocks", [])), "widths": dict(v.get("widths", {})), "ctx": v.get("ctx"),
           "hidden": blob.get("hidden"), "seed": blob.get("seed"), "config": dict(blob.get("config") or {}),
           "train": [tuple(x) for x in blob.get("train", [])], "basis_sha256": blob.get("basis_sha256")}
    bad = [k for k in want if want[k] != got[k]]
    if bad:
        raise SystemExit(f"zrs: the base fit {rel(fit_dir)} differs from this fit in {bad}")
    return blob, blob["states"][f"{vname}@swa"]


def fit_stacked(tr, blocks, cfg, seed, hidden, ctx, device, tag=""):
    """lean_gpu.fit_variant's form (every epoch's state and the SWA state) for the second stage: zret's fit of the
    split (STACK['base_fit']) loaded and frozen, the match trained on the typed training carves against its scores,
    lean_gpu's loss and guard, Adam on the match's parameters only. Every state holds the base bit for bit (checked)."""
    if cfg["cos"] or cfg["adamw"] or cfg["drop"]:
        raise SystemExit("zrs takes lr:wd:dropout:epochs:swa_from only")
    vname = tag.split("/")[-1].split(" ")[0]
    if vname != "p" or ctx != "none":
        raise SystemExit(f"zrs trains variant p with ctx none only, not {vname} ({ctx})")
    if STACK["base_fit"] is None:
        raise SystemExit("zrs: fit_stacked runs inside zrs.train only (no base fit)")
    widths = {b: tr[0].widths[b] for b in blocks}
    _blob, base = load_base(STACK["base_fit"], vname, blocks, widths, cfg, seed, hidden, ctx,
                            [(c.ds, c.carve) for c in tr], getattr(tr[0], "basis_sha256", None))
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = LG.LeanMLP8D(blocks, widths, hidden, dropout=cfg["dropout"], seed=seed, arm="ctl", ctx=ctx).to(device)
    if not isinstance(model, ZRS):
        raise SystemExit("zrs: fit_stacked runs under zrs's patching only")
    sd = model.state_dict()
    mk = sorted(k for k, _ in model.named_parameters() if k.startswith("cm_"))
    if sorted(set(sd) - set(base)) != mk or set(base) - set(sd):
        raise SystemExit(f"zrs: the base state's keys are not the model's without the match "
                         f"({sorted(set(sd) ^ set(base))})")
    model.load_state_dict({**sd, **base})
    for k, p in model.named_parameters():
        p.requires_grad_(k in mk)
    params = [p for k, p in model.named_parameters() if k in mk]
    opt = torch.optim.Adam(params, lr=cfg["lr"], weight_decay=cfg["wd"])
    typed = [ci for ci, c in enumerate(tr) if getattr(c, "chains", None) is not None]
    units_ci = np.concatenate([np.full(tr[ci].rows, ci, np.int64) for ci in typed] or [np.zeros(0, np.int64)])
    units_q = np.concatenate([np.arange(tr[ci].rows, dtype=np.int64) for ci in typed] or [np.zeros(0, np.int64)])
    keep_all = None
    states, curve = [], []
    model.eval()                      # the base as it reads (no dropout); the match has none
    for ep in range(cfg["epochs"]):
        order = rng.permutation(units_ci.size)
        t0 = time.time()
        tot = torch.zeros((), dtype=torch.float64, device=device)
        nb, steps = 0, 0
        guard = {"skipped_loss": 0, "skipped_grad": 0, "first": None}
        for k in range(0, order.size, 32):
            sel = order[k:k + 32]
            cis, qq = units_ci[sel], units_q[sel]
            for ci in sorted(set(cis.tolist())):
                qs = qq[cis == ci]
                feats, nq, base_z, gold = tr[ci].batch(qs, blocks)
                B = qs.size
                steps += 1
                if keep_all is None or keep_all.shape[0] != B:
                    keep_all = torch.ones((B, len(blocks)), dtype=torch.float32, device=device)
                s = model(feats, keep_all, nq, B, base_z)
                loss = LG.listwiseD(s, gold, nq, B)
                if not bool(torch.isfinite(loss).all()):
                    guard["skipped_loss"] += 1
                    if guard["first"] is None:
                        guard["first"] = {"epoch": ep, "at": "loss", "graph": tr[ci].ds, "rows": [int(v) for v in qs[:32]]}
                    continue
                opt.zero_grad()
                loss.backward()
                grads = [p.grad for p in params if p.grad is not None]
                if grads and not bool(torch.stack([torch.isfinite(g).all() for g in grads]).all()):
                    guard["skipped_grad"] += 1
                    if guard["first"] is None:
                        guard["first"] = {"epoch": ep, "at": "grad", "graph": tr[ci].ds, "rows": [int(v) for v in qs[:32]]}
                    opt.zero_grad()
                    continue
                opt.step()
                tot += loss.detach().to(torch.float64)
                nb += 1
        states.append(LG.cpu_state(model))
        rec = {"epoch": ep, "loss": float(tot) / max(nb, 1), "steps": steps, "seconds": time.time() - t0,
               "gate": [float(x) for x in model.cm_gate.detach().cpu()]}
        if guard["skipped_loss"] or guard["skipped_grad"]:
            rec["guard"] = guard
            log(f"  {tag} ep {ep}: GUARD skipped {guard['skipped_loss']} at a non-finite loss, {guard['skipped_grad']} at a "
                f"non-finite gradient; first {json.dumps(guard['first'])}")
        curve.append(rec)
        log(f"  {tag} ep {ep}: match loss {rec['loss']:.4f}, {steps} steps, gates {rec['gate']} ({rec['seconds']:.0f}s)")
    acc, n_acc = None, 0
    for ep, st in enumerate(states):
        if ep < cfg["swa_from"]:
            continue
        if acc is None:
            acc = {k: st[k].double().clone() for k in mk}
        else:
            for k in acc:
                acc[k] += st[k].double()
        n_acc += 1
    swa = {k: ((acc[k] / n_acc).to(torch.float32) if acc is not None and k in acc else states[-1][k].clone())
           for k in states[-1]}
    moved = [k for st in states + [swa] for k in LG.same_state({k: st[k] for k in base}, base)]
    if moved:
        raise SystemExit(f"zrs: the base moved in {sorted(set(moved))}")
    STACK["info"] = {"base_fit": rel(STACK["base_fit"]),
                     "base_models_sha256": LC.sha_file(Path(STACK["base_fit"]) / "models.pt"),
                     "base_candidate": f"{vname}@swa", "base_unchanged_in_every_state": True,
                     "typed_training_carves": [f"{tr[ci].ds}={tr[ci].carve}" for ci in typed],
                     "match_steps_per_epoch": curve[-1]["steps"] if curve else 0}
    return {"states": states, "swa": swa, "curve": curve, "widths": widths, "ctx_stats": None,
            "swa_epochs": list(range(cfg["swa_from"], cfg["epochs"]))}


class stacked_loop:
    """lean_gpu.fit_variant swapped for fit_stacked while a zrs fit trains."""

    def __enter__(self):
        self.saved = LG.fit_variant
        LG.fit_variant = fit_stacked
        return self

    def __exit__(self, *exc):
        LG.fit_variant = self.saved
        return False


# ── train and read (rmatch.py's, under zrs's arm) ────────────────────────────


def train(argv, split):
    """rmatch.py's train (lean_screen2's, with rmatch's settings and chain records) for zrs only, with lean_gpu's loop
    swapped for fit_stacked on zret's fit (--base-fit DIR; default zret's fit of the split)."""
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--base-fit")
    k, rest = ap.parse_known_args(argv)
    arm = R.arm_of(rest)
    if arm != ARM:
        raise SystemExit(f"zrs: train takes --arm {ARM}, not {arm}")
    base = Path(k.base_fit) if k.base_fit else zret_fit_dir(split)
    if not (base / "models.pt").exists() or not (base / "screen.json").exists():
        raise SystemExit(f"zrs: no base fit at {rel(base)}")
    STACK["base_fit"], STACK["info"] = base, None
    try:
        with stacked_loop():
            rc = RM.train(rest, split)
    finally:
        STACK["base_fit"] = None
    name, out_root = S2.where(rest)
    sj = out_root / name / "screen.json"
    if sj.exists():
        rec = json.loads(sj.read_text(encoding="utf-8"))
        rec["zrs_sha256"] = LC.sha_src(__file__)
        rec["zrm_sha256"] = LC.sha_src(ZM.__file__)
        rec["lean_screen3_sha256"] = LC.sha_src(S3.__file__)
        rec["zrs"] = {"model": "zrm.ZRM (rmatch.ChainMatch over lean_screen3.ZRet)", "carve": "rmatch.ChainCarveBase",
                      "stage": "the match trained on zret's frozen fit", **(STACK["info"] or {})}
        LC.write_json(sj, rec)
    return rc


def read(argv):
    name, out_root = S2.where(argv)
    arm = json.loads((out_root / name / "screen.json").read_text(encoding="utf-8"))["arm"]
    if arm != ARM:
        raise SystemExit(f"zrs: {name} was trained as {arm}, not {ARM}")
    return RM.read(argv)


# ── relz.py's pair, re-call and grade, decided against zret's fits ───────────


@contextlib.contextmanager
def on_zret():
    """relz.py's records under zrs's name, each read decided against zret's fit of its split (zgs.py's mapping);
    relz restored after."""
    with G.on_zret():
        Z.ARM = ARM
        yield


FIX = (("(rel's screen fits scr-rel and scr-rel-hp)", "(zret's fits: its screen fit scr-zret and its full run's "
                                                      "L-hotpotqa fit)"),
       ("(rel's screen fits)", "(zret's fits)"),
       ("| rel R@5 | relz R@5 |", "| zret R@5 | zrs R@5 |"),
       ("section 2 and the tenth round", "section 2 and the fifteenth round"),
       ("docs/FULL_ROUND10.md", "docs/FULL_ROUND15.md"))


def restamp(out):
    """A record relz.py wrote under zrs's name: zret named as the base in its md, this file's sha added."""
    out = Path(out)
    md = out.with_suffix(".md")
    if md.exists():
        t = md.read_text(encoding="utf-8")
        for a, b in FIX:
            t = t.replace(a, b)
        md.write_text(t, encoding="utf-8")
    js = out.with_suffix(".json")
    if js.exists():
        rec = json.loads(js.read_text(encoding="utf-8"))
        rec["decided_against"] = "zret's fit of each split"
        rec["zrs_sha256"] = LC.sha_src(__file__)
        LC.write_json(js, rec)


def pair(screens, out):
    with on_zret():
        rec = Z.pair(screens, out)
    restamp(out)
    return rec


def recall(null_files, pair_file, out, zret_screens=None):
    with on_zret():
        rec = Z.recall(null_files, pair_file, out, zret_screens)
    restamp(out)
    return rec


def grade(full_root, reuse, out, check_fits=True):
    with on_zret():
        rec = Z.grade(full_root, reuse, out, check_fits)
    restamp(out)
    return rec


# ── smoke ────────────────────────────────────────────────────────────────────


def carve_check(ds, carve, device):
    """One carve through rmatch's chain carve, under train's and read's flags: on its first questions zrs at its start
    scores as zret's model bit for bit (two forwards of zret's model equal first), with finite chain features on a
    typed graph."""
    LG.set_flags(device)
    LG.bind_device_ops()
    b = RM.ChainCarveBase(ds, carve, "2wiki", device)
    rec = {"typed": b.chains is not None}
    ok = rec["typed"] == (ds in RM.TYPED)
    blocks = [x for x in LG.SETS["pick"] if x in b.widths]
    widths = {x: b.widths[x] for x in blocks}
    qs = np.arange(min(16, b.rows))
    feats, nq, base_z, _gold = b.batch(qs, blocks)
    keep = torch.ones((qs.size, len(blocks)), dtype=torch.float32, device=b.device)
    nets = {}
    for k, cls in ((BASE_ARM, S3.ZRet), (ARM, ZRS)):
        torch.manual_seed(0)
        nets[k] = cls(blocks, widths, 32, seed=0).to(b.device).eval()
    with torch.no_grad():
        s = {k: m(feats, keep, nq, qs.size, base_z) for k, m in nets.items()}
        rec["zret_repeat_equal"] = bool(torch.equal(nets[BASE_ARM](feats, keep, nq, qs.size, base_z), s[BASE_ARM]))
        rec["identity_at_start"] = bool(torch.equal(s[ARM], s[BASE_ARM]))
        if b.chains is not None:
            fm, fr = nets[ARM].chain_feats(feats, nq.numel())
            rec["entries_first_questions"] = int(feats[RM.CH_KEY]["eq"].size)
            rec["features_finite"] = bool(torch.isfinite(fm).all() and torch.isfinite(fr).all())
            ok = ok and rec["features_finite"] and rec["entries_first_questions"] > 0
    ok = ok and rec["zret_repeat_equal"] and rec["identity_at_start"]
    del b, nets
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return ok, rec


def base_kept(fit_dir, base_dir):
    """Whether every state of a zrs fit holds its base fit's p@swa state bit for bit (the match's keys aside)."""
    a = torch.load(Path(fit_dir) / "models.pt", map_location="cpu", weights_only=False)
    b = torch.load(Path(base_dir) / "models.pt", map_location="cpu", weights_only=False)["states"]["p@swa"]
    return all(not LG.same_state({k: st[k] for k in b if k in st}, b) for st in a["states"].values())


def smoke(device, host, out_root=None):
    """The carve check on metaqa select, webqsp s1eval and 2wiki select; then zret once and zrs twice on zret's fit
    (the repeat must be IDENTICAL), one epoch each on metaqa's and 2wiki's select carves, each read on both. Every
    state of zrs's fit must hold zret's p@swa state bit for bit, zrs's match must move from zero and stay finite, zret's
    fit holds none, zrs's scores must equal zret's on 2wiki (untyped) bit for bit and differ on metaqa. A crash fails
    it (exit 1)."""
    t0 = time.time()
    root = Path(out_root or S.OUT / "smoke15")
    h = ["--host"] if host else []
    rec, ok = {"carves": {}}, True
    for ds, cv in RM.SMOKE_CARVES:
        good, chk = carve_check(ds, cv, device)
        rec["carves"][f"{ds}={cv}"] = chk
        ok = ok and good
    sets = "metaqa=select,2wiki=select"
    common = ["--train", sets, "--basis", "2wiki", "--variants", "p", "--config", "2e-3:1e-4:0.1:1:0", "--device",
              device, "--out-root", str(root)] + h
    runs = {BASE_ARM: (S3.train, "1", []), ARM: (train, "2", ["--base-fit", str(root / f"smoke-{BASE_ARM}")])}
    for arm, (fn, rep, extra) in runs.items():
        nm = f"smoke-{arm}"
        rc = fn(["train", "--name", nm, "--arm", arm, "--repeat", rep] + extra + common, "L-musique")
        rr = RM.read(["--name", nm, "--read", sets, "--device", device, "--out-root", str(root)] + h)
        sj = json.loads((root / nm / "screen.json").read_text(encoding="utf-8"))
        tj = json.loads((root / nm / "train.json").read_text(encoding="utf-8"))
        rec[arm] = {"train_rc": rc, "repeats": int(rep), "read_rc": rr, "screen_arm": sj.get("arm"),
                    "blocks": tj["variants"]["p"]["blocks"], "head_norms": RM.head_norms(root / nm / "models.pt"),
                    "curve": tj["variants"]["p"]["curve"], "peak_gpu_gb": tj.get("peak_gpu_gb")}
        ok = ok and rc == 0 and rr == 0 and sj.get("arm") == arm
        if arm == ARM:
            rec[arm]["stack"] = sj.get("zrs")
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
    rec[ARM]["repeat"] = "IDENTICAL" if rec[ARM]["train_rc"] == 0 else "DIFFERENT"
    rec["base_kept"] = base_kept(root / f"smoke-{ARM}", root / f"smoke-{BASE_ARM}")
    rec["match_moved"] = RM.moved(rec[ARM]["head_norms"])
    rec["zret_has_no_match"] = all(v is None for v in rec[BASE_ARM]["head_norms"].values())
    rec["same_blocks"] = rec[ARM]["blocks"] == rec[BASE_ARM]["blocks"]
    ok = ok and rec["base_kept"] and rec["match_moved"] and rec["zret_has_no_match"] and rec["same_blocks"]
    rec["reads"] = {}
    for ds in ("metaqa", "2wiki"):
        npz = {a: root / f"smoke-{a}" / "reads" / f"{ds}__select.npz" for a in runs}
        if not all(p.exists() for p in npz.values()):
            rec["reads"][ds] = None
            continue
        r = {a: np.load(p) for a, p in npz.items()}
        ids = {a: [str(x) for x in v["ids"]] for a, v in r.items()}
        k = [str(x) for x in r[ARM]["candidates"]].index("p@swa")
        rec["reads"][ds] = {"same_questions": ids[ARM] == ids[BASE_ARM],
                            "equal_to_zret": bool(np.array_equal(r[ARM]["scores64"], r[BASE_ARM]["scores64"])),
                            "p@swa_hit@1": {a: float(v["hit"][k].mean()) for a, v in r.items()}}
    rd = rec["reads"]
    ok = ok and all(v is not None and v["same_questions"] for v in rd.values())
    ok = ok and rd["2wiki"]["equal_to_zret"] and not rd["metaqa"]["equal_to_zret"]
    rec.update({"ok": ok, "seconds": time.time() - t0, "script_sha256": LC.sha_src(__file__)})
    LC.write_json(root / "smoke.json", rec)
    log(f"smoke15: {json.dumps(rec)}; {'ok' if ok else 'FAILED'}")
    return 0 if ok else 1


# ── selftest ─────────────────────────────────────────────────────────────────


class Untyped:
    """A toy carve without its chains: an untyped graph's batches (the same rows, blocks and golds)."""

    def __init__(self, c, ds):
        self.c, self.ds, self.carve, self.chains = c, ds, c.carve, None
        self.rows, self.widths, self.n_np, self.off_np = c.rows, c.widths, c.n_np, c.off_np

    def nbytes(self):
        return 0

    def batch(self, qs, blocks):
        feats, nq, bz, gold = self.c.batch(qs, blocks)
        return {k: v for k, v in feats.items() if k != RM.CH_KEY}, nq, bz, gold


def toy_base(tmp, tr, blocks, cfg, seed=0, hidden=16):
    """zret's fit of the toy carves through lean_gpu's own loop, saved as a fit directory (models.pt, screen.json)."""
    with S.patched(BASE_ARM):
        f = LG.fit_variant(tr, blocks, cfg, seed, hidden, "none", "cpu", tag="toy/p")
    d = tmp / "fits" / "toy-zret"
    d.mkdir(parents=True)
    blob = {"train": [(c.ds, c.carve) for c in tr], "basis_sha256": None, "config": cfg, "seed": seed, "hidden": hidden,
            "variants": {"p": {"set": "pick", "ctx": "none", "blocks": blocks, "widths": f["widths"]}},
            "states": {**{f"p@ep{e}": st for e, st in enumerate(f["states"])}, "p@swa": f["swa"]},
            "candidates": [f"p@ep{e}" for e in range(len(f["states"]))] + ["p@swa"]}
    torch.save(blob, d / "models.pt")
    LC.write_json(d / "screen.json", {"arm": BASE_ARM})
    return d, blob


def selftest():
    t0 = time.time()
    LG.bind_device_ops()
    base_cls, base_carve = S.ARMS["base"]
    # 1. the arm is zrm's model on rmatch's chain carve; zret's, rmatch's and zrm's own arms unchanged
    assert S.ARMS[ARM] == (ZRS, RM.ChainCarveBase) and ZRS.__mro__[1:5] == (ZM.ZRM, RM.ChainMatch, S3.ZRet, base_cls)
    assert S.ARMS[BASE_ARM] == (S3.ZRet, base_carve) and S.ARMS[ZM.ARM] == (ZM.ZRM, RM.ChainCarveBase)
    assert zret_fit_dir("L-musique") == ROOT / "outputs" / "screen" / "fits" / "scr-zret"
    assert zret_fit_dir("J5") == ROOT / "outputs" / "full_zret" / "fits" / "J5"
    tmp = Path(tempfile.mkdtemp(prefix="zrs_"))
    saved_fit = LG.fit_variant
    try:
        # 2. on rmatch's toy carve (typed) and the same carve without its chains (untyped): zret's fit through
        #    lean_gpu's loop, then the second stage on it
        rng = np.random.default_rng(14)
        RM.toy_roots(tmp, rng)
        RM.build("metaqa", "toy", out_root=tmp / "ch", cache_root=tmp / "cache", look_root=tmp / "look",
                 rel_dir=tmp / "rel")
        c = RM.chain_carve(ZM.ToyZ, tmp / "ch", tmp / "rel")("metaqa", "toy", "2wiki", "cpu", tmp / "cache")
        assert c.chains is not None
        u = Untyped(c, "2wiki")
        tr = [c, u]
        blocks = ["rank", "SEMB"]
        cfg = {"lr": 2e-2, "wd": 1e-4, "dropout": 0.1, "epochs": 3, "swa_from": 1, "cos": False, "adamw": False,
               "drop": False}
        bdir, bblob = toy_base(tmp, tr, blocks, cfg)
        base = bblob["states"]["p@swa"]
        STACK["base_fit"] = bdir
        with S.patched(ARM), stacked_loop():
            assert LG.fit_variant is fit_stacked
            f = LG.fit_variant(tr, blocks, cfg, 0, 16, "none", "cpu", tag="toy-zrs/p")
            g = LG.fit_variant(tr, blocks, cfg, 0, 16, "none", "cpu", tag="toy-zrs/p repeat")
        assert LG.fit_variant is saved_fit
        # every state holds the base bit for bit; the match moved; the repeat is identical; only the typed carve's
        # questions train (3 epochs of its 6 questions: one batch each)
        for st in f["states"] + [f["swa"]]:
            assert not LG.same_state({k: st[k] for k in base}, base)
        assert float(f["swa"]["cm_gate"].abs().sum()) > 0 and float(f["swa"]["cm_b"].abs().sum()) > 0
        assert all(not LG.same_state(a, b) for a, b in zip(f["states"] + [f["swa"]], g["states"] + [g["swa"]]))
        assert [r["steps"] for r in f["curve"]] == [1, 1, 1], f["curve"]
        info = STACK["info"]
        assert info["typed_training_carves"] == ["metaqa=toy"] and info["base_candidate"] == "p@swa"
        assert info["base_unchanged_in_every_state"] and info["base_fit"].endswith("toy-zret")
        # its scores: the base's on the untyped carve bit for bit, the base's plus the match on the typed one
        qs = np.arange(c.rows)
        nets = {}
        for k, cls, st in ((BASE_ARM, S3.ZRet, base), (ARM, ZRS, f["swa"])):
            torch.manual_seed(0)
            nets[k] = cls(blocks, c.widths, 16, dropout=0.1, seed=0).eval()
            nets[k].load_state_dict(st)
        keep = torch.ones(qs.size, 2)
        with torch.no_grad():
            for carve, same in ((u, True), (c, False)):
                feats, nq, bz, _gold = carve.batch(qs, blocks)
                assert torch.equal(nets[ARM](feats, keep, nq, qs.size, bz),
                                   nets[BASE_ARM](feats, keep, nq, qs.size, bz)) == same
        # 3. refusals: no base fit, another variant or context, a base trained as another arm or with another hidden
        #    size, training another arm, reading another arm's fit
        STACK["base_fit"] = None
        with S.patched(ARM):
            for bad in (lambda: fit_stacked(tr, blocks, cfg, 0, 16, "none", "cpu", tag="x/p"),):
                SRstop(bad)
        STACK["base_fit"] = bdir
        with S.patched(ARM):
            for bad in (lambda: fit_stacked(tr, blocks, cfg, 0, 16, "none", "cpu", tag="x/pf"),
                        lambda: fit_stacked(tr, blocks, cfg, 0, 16, "film", "cpu", tag="x/p"),
                        lambda: fit_stacked(tr, blocks, cfg, 0, 32, "none", "cpu", tag="x/p"),
                        lambda: fit_stacked(tr, blocks, dict(cfg, lr=1e-3), 0, 16, "none", "cpu", tag="x/p"),
                        lambda: fit_stacked(tr[:1], blocks, cfg, 0, 16, "none", "cpu", tag="x/p")):
                SRstop(bad)
        LC.write_json(bdir / "screen.json", {"arm": ZM.ARM})
        with S.patched(ARM):
            SRstop(lambda: fit_stacked(tr, blocks, cfg, 0, 16, "none", "cpu", tag="x/p"))
        STACK["base_fit"] = None
        SRstop(lambda: train(["train", "--name", "x", "--arm", ZM.ARM], "L-musique"))
        SRstop(lambda: train(["train", "--name", "x", "--arm", ARM, "--base-fit", str(tmp / "none")], "L-musique"))
        (tmp / "fits" / "x").mkdir(parents=True)
        LC.write_json(tmp / "fits" / "x" / "screen.json", {"arm": ZM.ARM})
        try:
            read(["--name", "x", "--out-root", str(tmp / "fits")])
            raise AssertionError("zrs read zrm's fit")
        except SystemExit as e:
            assert "trained as zrm" in str(e)
        # 4. load_models builds zrs under patching, its scores unchanged; the patching restores
        blob = {"candidates": ["p@swa"], "variants": {"p": {"blocks": blocks, "widths": c.widths, "ctx": "none"}},
                "hidden": 16, "states": {"p@swa": f["swa"]}}
        feats, nq, bz, _gold = c.batch(qs, blocks)
        with S.patched(ARM):
            assert LG.LeanMLP8D is ZRS and LG.CacheCarve is RM.ChainCarveBase
            (_name, m, bl), = LG.load_models(blob, "cpu")
            assert type(m) is ZRS and bl == blocks
            with torch.no_grad():
                assert torch.equal(m(feats, keep, nq, qs.size, bz), nets[ARM](feats, keep, nq, qs.size, bz))
        assert LG.LeanMLP8D is base_cls and LG.CacheCarve is base_carve
    finally:
        STACK["base_fit"] = None
        LG.fit_variant = saved_fit
        shutil.rmtree(tmp, ignore_errors=True)
    # 5. relz's pair, re-call and grade under zrs's name, decided against zret's fits; relz restored after;
    #    restamp names zret and this round
    saved = Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit
    with on_zret():
        assert Z.tag({})["arm"] == ARM and Z.BASE_ARM == BASE_ARM and Z.rel_fit("J5") == ("full_zret", "fits", "J5")
    assert (Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit) == saved
    SR = Z.SR
    with tempfile.TemporaryDirectory() as td:
        T = Path(td)
        null1 = SR.fake_null(T / "null", "n1", {("L-hotpotqa", "hotpotqa"): (-0.010, 0.012)})   # floor 0.0221
        zs = {sp: str(SR.fake_compare(T / G.zret_fit(sp)[0], G.zret_fit(sp)[2], sp, {ds: 0.1 for ds in LG.EVAL_ORDER},
                                      arm=BASE_ARM))
              for sp in G.ZRET_FITS}
        zb = {sp: "/".join(("outputs",) + G.zret_fit(sp)) for sp in G.ZRET_FITS}
        za = SR.fake_compare(T / "z", "scr-zrs", "L-musique", {"metaqa": 0.1268}, arm=ARM, base_r5=0.6,
                             base=zb["L-musique"])
        zh = SR.fake_compare(T / "z", "scr-zrs-hp", "L-hotpotqa", {"hotpotqa": -0.0147}, arm=ARM, base_r5=0.6,
                             base=zb["L-hotpotqa"])
        pr = pair([za, zh], T / "pz")
        assert pr["verdict"] == "MIXED" and pr["arm"] == ARM and pr["base_arm"] == BASE_ARM
        assert json.loads((T / "pz.json").read_text(encoding="utf-8"))["decided_against"] == "zret's fit of each split"
        rc = recall(null1, T / "pz.json", T / "rz", zs)
        assert rc["verdict"] == "PROMISING" and rc["changed"] == ["L-hotpotqa hotpotqa LOSS -> WITHIN"], rc["changed"]
        t = (T / "rz.md").read_text(encoding="utf-8")
        assert "| zret R@5 | zrs R@5 |" in t and "fifteenth round" in t and "rel's" not in t
        st1 = SR.fake_compare(T / "z2", "scr-zrs", "L-musique", {}, arm=ARM)       # decided against step 1's fit
        SR.must_stop(pair, [st1, zh], T / "bad")
        full = T / "full"
        full.mkdir()
        for sp in S2.SPLITS:
            if sp not in G.ZRET_FITS:
                d = {"webqsp": -0.02} if sp == "J5" else {}
                SR.fake_compare(full / "src", f"fit-{sp}", sp, d, arm=ARM,
                                base="/".join(("outputs",) + G.zret_fit(sp))).replace(full / f"compare-{sp}.json")
        g2 = grade(full, {"L-musique": str(za), "L-hotpotqa": str(zh)}, full / "grade", check_fits=False)
        assert g2["verdict"] == "NOT_ADOPTED" and g2["losses"] == 2 and g2["arm"] == ARM, (g2["verdict"], g2["losses"])
        assert "docs/FULL_ROUND15.md" in (full / "grade.md").read_text(encoding="utf-8")
    assert (Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit) == saved
    log(f"zrs selftest: the arm is zrm's model on rmatch's chain carve; on a typed toy carve and its untyped copy, zret's "
        f"fit through lean_gpu's loop, then the second stage on it: every state holds zret's p@swa state bit for bit, "
        f"only the typed carve trains, the match moves, the repeat is identical, its scores are zret's on the untyped "
        f"carve bit for bit and differ on the typed one; refusals (no base, another variant, context, hidden size, "
        f"config, carves or arm); load_models under patching; relz's pair, re-call and grade under zrs's name against "
        f"zret's fits, naming zret and this round, relz restored ({time.time() - t0:.1f}s): ok")
    return 0


def SRstop(fn):
    """fn must refuse (SystemExit)."""
    try:
        fn()
    except SystemExit:
        return
    raise AssertionError("a refusal was accepted")


# ── main ─────────────────────────────────────────────────────────────────────


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("cmd", nargs="?")
    ap.add_argument("--split", default="L-musique", choices=S2.SPLITS)
    ap.add_argument("--selftest", action="store_true")
    k, rest = ap.parse_known_args(argv)
    if k.selftest:
        return selftest()
    if k.cmd == "smoke":
        sp = argparse.ArgumentParser()
        sp.add_argument("--device", default="cpu")
        sp.add_argument("--host", action="store_true")
        sp.add_argument("--out-root")
        s = sp.parse_args(rest)
        return smoke(s.device, s.host, s.out_root)
    if k.cmd == "train":
        return train([k.cmd] + rest, k.split)
    if k.cmd == "read":
        return read(rest)
    if k.cmd == "compare":
        return S2.main([k.cmd] + rest)
    if k.cmd in ("pair", "recall", "grade"):
        gp = argparse.ArgumentParser()
        gp.add_argument("--screens", default="")
        gp.add_argument("--null", default="")
        gp.add_argument("--pair")
        gp.add_argument("--full-root")
        gp.add_argument("--reuse", default="")
        gp.add_argument("--out")
        g = gp.parse_args(rest)
        null = [x for x in g.null.split(",") if x]
        need = {"pair": g.screens and g.out, "recall": g.pair and len(null) == 4 and g.out, "grade": g.full_root}[k.cmd]
        if not need:
            gp.error(f"{k.cmd}: pair needs --screens A,B and --out; recall --pair, the four --null files and --out; "
                     "grade --full-root")
        try:
            if k.cmd == "pair":
                pair([x for x in g.screens.split(",") if x], g.out)
            elif k.cmd == "recall":
                recall(null, g.pair, g.out)
            else:
                reuse = dict(x.split("=", 1) for x in g.reuse.split(",") if x)
                v = grade(g.full_root, reuse, g.out or str(Path(g.full_root) / "grade"))["verdict"]
                return 1 if v == "INCOMPLETE" else 0
        except SystemExit as e:
            log(f"zrs {k.cmd}: {e}")
            return 2
        return 0
    raise SystemExit("zrs: train, read, compare, pair, recall, grade, smoke, or --selftest")


if __name__ == "__main__":
    sys.exit(main())
