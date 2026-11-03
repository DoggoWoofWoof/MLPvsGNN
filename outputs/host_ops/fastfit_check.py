"""fastfit's check on the host's card: one small real fit three ways, compared bit for bit.

lean_gpu's own train command (the code fastfit patches: CacheCarve.__init__, CacheCarve.batch and fit_variant) on PKO's
cache, trained on squad/fit and webqsp/fit for two epochs, under a scratch root and a name of its own (no fit of any
stage is read or written). Three runs, each in its own process under cuda_alloc as the queue runs fits:

  plain   fastfit_check.py --lg-main -- train ...                               (no fastfit)
  host    fastfit.py --no-prefetch -- fastfit_check.py --lg-main -- train ...   (X in host memory)
  both    fastfit.py -- fastfit_check.py --lg-main -- train ...                 (X in host memory, the next batch built
                                                                                 on a thread)

`--lg-main` imports lean_gpu and runs its main, so every run enters lean_gpu as the fits do, as an imported module.
fastfit makes its changes when lean_gpu is imported: run as the script itself, lean_gpu is __main__ and stays as it is.
The first check (3 November, 19:23 IST) ran it that way, and both fastfit runs logged "lean_gpu was not imported;
nothing changed". So a run's changes now count only when its log says they were made: `lean_gpu patched: host_x on`
(host) or `host_x, prefetch on` (both), and a carve with X in host memory.

Each run's models.pt is compared with plain's, tensor by tensor at the bit (every epoch's state and the swa average), and
its train.json curve (loss and steps per epoch). Each run's seconds and peak GPU memory are reported beside.

    python outputs/host_ops/fastfit_check.py            (outputs/fastfit_check/check.json and the runs' folders)
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "fastfit_check"
NAME = "ffchk"
TRAIN = "squad=fit,webqsp=fit"
CONFIG = "2e-3:1e-4:0.1:2:1"                    # lean_gpu's lr:wd:dropout, two epochs, swa from epoch 1
FRAC = "0.15"                                   # the item books 0.17 of the card
SELF = "outputs/host_ops/fastfit_check.py"
RUNS = {"plain": [], "host": ["outputs/host_ops/fastfit.py", "--no-prefetch", "--"],
        "both": ["outputs/host_ops/fastfit.py", "--"]}
MADE = {"plain": None, "host": "host_x on", "both": "host_x, prefetch on"}    # fastfit's line for each run's changes


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] fastfit_check: {msg}", flush=True)


def train_argv(out_root):
    return ["train", "--name", NAME, "--train", TRAIN, "--basis", "2wiki", "--variants", "p", "--config", CONFIG,
            "--seed", "0", "--device", "cuda", "--host", "--cache-root", "outputs/pko/cache", "--out-root", str(out_root)]


def lg_main(args):
    """lean_gpu's main on ARGS, lean_gpu imported as a module (as the fits import it), so fastfit's finder sees it."""
    sys.path.insert(0, str(ROOT / "outputs" / "mp_unified"))
    import lean_gpu
    return lean_gpu.main(args)


def patched(mode, text):
    """Whether the run's log shows the changes its mode makes, and none that it does not."""
    lines = [ln for ln in text.splitlines() if ln.startswith("[fastfit]")]
    if MADE[mode] is None:
        return not lines
    made = any(ln.endswith(f"lean_gpu patched: {MADE[mode]}") for ln in lines)
    host = any("carve(s) with X in host memory" in ln and not ln.startswith("[fastfit] 0 carve(s)") for ln in lines)
    return made and host and not any("refused" in ln or "nothing changed" in ln for ln in lines)


def run(mode):
    out_root = OUT / mode
    fd = out_root / NAME
    for f in ("models.pt", "train.json", "screen.json"):
        if (fd / f).exists():
            (fd / f).unlink()
    inner = RUNS[mode] + [SELF, "--lg-main", "--"] + train_argv(out_root)
    cmd = [sys.executable, "outputs/host_ops/cuda_alloc.py", "--frac", FRAC, "--gc", "0.9", "--"] + inner
    log(f"{mode}: {' '.join(cmd[1:])}")
    t = time.time()
    with open(out_root.parent / f"{mode}.log", "w", encoding="utf-8") as fh:
        rc = subprocess.call(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT)
    secs = time.time() - t
    text = (out_root.parent / f"{mode}.log").read_text(encoding="utf-8", errors="replace")
    tail = text.splitlines()[-12:]
    ok = patched(mode, text)
    log(f"{mode}: exit {rc} in {secs:.0f}s; fastfit's changes as the mode makes them: {ok}")
    for ln in tail:
        if "[fastfit]" in ln or "peak" in ln or "Error" in ln or "error" in ln:
            log(f"  {ln.strip()[:220]}")
    return rc, secs, tail, ok


def bits_equal(a, b):
    import torch
    if not (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)):
        return a == b
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    if a.dtype.is_floating_point:
        iv = {2: torch.int16, 4: torch.int32, 8: torch.int64}[a.element_size()]
        return torch.equal(a.contiguous().view(iv), b.contiguous().view(iv))
    return torch.equal(a, b)


def same_states(sa, sb):
    """The keys of two state dicts that differ (a key missing from either counts)."""
    keys = set(sa) | set(sb)
    return sorted(k for k in keys if k not in sa or k not in sb or not bits_equal(sa[k], sb[k]))


def compare(ref, other):
    import torch
    fa, fb = OUT / ref / NAME, OUT / other / NAME
    A = torch.load(fa / "models.pt", map_location="cpu", weights_only=False)
    B = torch.load(fb / "models.pt", map_location="cpu", weights_only=False)
    diff = {}
    for k in sorted(set(A["states"]) | set(B["states"])):
        d = same_states(A["states"].get(k, {}), B["states"].get(k, {}))
        if d:
            diff[k] = d
    meta = [k for k in ("fit", "train", "basis", "basis_sha256", "config", "seed", "hidden", "dead", "sets", "variants",
                        "candidates") if A.get(k) != B.get(k)]
    ta = json.loads((fa / "train.json").read_text(encoding="utf-8"))
    tb = json.loads((fb / "train.json").read_text(encoding="utf-8"))
    ca, cb = ta["variants"]["p"]["curve"], tb["variants"]["p"]["curve"]
    curve = [(r.get("loss"), r.get("steps")) for r in ca] == [(r.get("loss"), r.get("steps")) for r in cb]
    return {"states_compared": len(set(A["states"]) | set(B["states"])), "states_differing": diff,
            "meta_differing": meta, "curve_equal": curve,
            "identical": not diff and not meta and curve}


def summary(mode):
    tj = OUT / mode / NAME / "train.json"
    if not tj.is_file():
        return None
    t = json.loads(tj.read_text(encoding="utf-8"))
    return {"seconds": round(t.get("seconds", 0.0), 1), "variant_seconds": round(t["variants"]["p"]["seconds"], 1),
            "peak_gpu_gb": t.get("peak_gpu_gb"), "data_gb": round(t.get("data_gb", 0.0), 3),
            "questions": sum(r["questions"] for r in t["train"]), "rows": sum(r["rows"] for r in t["train"]),
            "models_sha256": t.get("models_sha256")}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--lg-main", action="store_true", help="run lean_gpu's main on the arguments after --")
    argv = list(sys.argv[1:] if argv is None else argv)
    cut = argv.index("--") if "--" in argv else len(argv)
    a, rest = ap.parse_args(argv[:cut]), argv[cut + 1:]
    if a.lg_main:
        return lg_main(rest)
    os.chdir(ROOT)
    OUT.mkdir(parents=True, exist_ok=True)
    rec = {"train": TRAIN, "config": CONFIG, "name": NAME, "frac": FRAC, "runs": {}, "compare": {}}
    for mode in RUNS:
        rc, secs, tail, made = run(mode)
        rec["runs"][mode] = {"exit": rc, "wall_seconds": round(secs, 1), "patched_as_mode": made,
                             "train": summary(mode), "log_tail": [ln[:300] for ln in tail]}
    for mode in ("host", "both"):
        if rec["runs"]["plain"]["exit"] == 0 and rec["runs"][mode]["exit"] == 0:
            rec["compare"][mode] = compare("plain", mode)
    made = all(r["patched_as_mode"] for r in rec["runs"].values())
    ok = len(rec["compare"]) == 2 and all(c["identical"] for c in rec["compare"].values())
    rec["verdict"] = ("IDENTICAL" if ok else "DIFFERENT_OR_FAILED") if made else "NOT_PATCHED"
    ok = ok and made
    rec["utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (OUT / "check.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    for mode, r in rec["runs"].items():
        s = r["train"] or {}
        log(f"{mode}: exit {r['exit']}, {s.get('variant_seconds')}s fitting, peak {s.get('peak_gpu_gb')} GB on the card, "
            f"{s.get('rows')} rows")
    for mode, c in rec["compare"].items():
        log(f"{mode} vs plain: {'IDENTICAL' if c['identical'] else 'DIFFERENT'} ({c['states_compared']} states; "
            f"differing {list(c['states_differing'])[:4]}; meta {c['meta_differing']}; curve equal {c['curve_equal']})")
    log(f"verdict {rec['verdict']}; {OUT / 'check.json'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
