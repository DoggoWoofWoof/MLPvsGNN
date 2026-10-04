"""Device path for the chainscore line (systems code; untracked, LF; no science). chainscore21's train21 and read21
copied under new names with an optional device, as src/mp_retrieval/device_placement.py copies M3B's fit and
evaluate. With device None or "cpu" the copies run chainscore21's statements; run21 --device cpu reproduces a
chainscore21 arm's rows, history and weights bit for bit (checked on the part 5 smoke). With a CUDA device:
  - the tested settings (host_gpu_det, docs/GPU_HOST_HANDOVER.md): device_placement.placement_settings("det"), with
    CUBLAS_WORKSPACE_CONFIG set in the environment before torch starts CUDA (this module sets it on import);
  - the model is made on the CPU by chainscore20's make_model (the same initial weights) and moves once, before the
    optimiser is built;
  - each batch is made on the CPU by chainscore21's make_batch21 (the frozen code) and moved to the device; the
    forward and the loss run under torch.device(device), since chainscore19/20 make their scratch tensors with no
    device argument;
  - scores come back to the CPU for the ranking; weights are pinned by state_sha256 over CPU copies;
  - a warning that an op has no deterministic implementation is kept (deterministic_warnings), so a repeat names it.
A GPU fit is a new draw: it never replaces, pairs with or averages with a CPU fit, and every arm a verdict compares
runs on one device.
    python outputs/mp_unified/cs_dev.py run21 --device cuda --out X.json --rows-out X.rows.npz -- <chainscore21 train
        arguments without --out, --rows-out and --state-out>
"""
import os
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")     # before torch starts CUDA (host_gpu_det)
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")      # chainscore19's count

import argparse  # noqa: E402
import hashlib  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import chainscore19 as C19  # noqa: E402
import chainscore20 as C20  # noqa: E402
import chainscore21 as C21  # noqa: E402

BATCH = C21.BATCH
DET_WARNINGS = []
log, sha = C21.log, C21.sha


def placed(device):
    return device is not None and str(device) != "cpu"


def _dp():
    if str(ROOT / "src") not in sys.path:
        sys.path.insert(0, str(ROOT / "src"))
    from mp_retrieval import device_placement as DP
    return DP


def settings(device, threads):
    """chainscore21's train21 settings on the CPU; host_gpu_det's on a GPU. Returns the placement block."""
    import torch
    DP = _dp()
    if placed(device):
        DP.placement_settings("det", threads)
    else:
        torch.set_num_threads(threads)
        torch.use_deterministic_algorithms(True)
    return DP.placement_block(device if placed(device) else "cpu")


def to_dev(x, dev):
    import torch
    if torch.is_tensor(x):
        return x.to(dev)
    if isinstance(x, dict):
        return {k: to_dev(v, dev) for k, v in x.items()}
    if isinstance(x, list):
        return [to_dev(v, dev) for v in x]
    if isinstance(x, tuple):
        return tuple(to_dev(v, dev) for v in x)
    return x


def state_sha(state):
    """sha256 of a state dict's CPU copy, saved by torch.save into memory (key order kept)."""
    import torch
    buf = io.BytesIO()
    torch.save({k: v.detach().to("cpu", copy=True) for k, v in state.items()}, buf)
    return hashlib.sha256(buf.getvalue()).hexdigest()


def read_dev(model, d, c, lb, st, rule, device=None, ri=0, batch=BATCH):
    """chainscore21's read21 with an optional device (the model already on it)."""
    import torch
    rank = {"kb": C19.rank_row, "passage": C21.rank_row_p}[rule]
    N = d["n"].size
    out = np.zeros((N, 3))
    L = d["level_names"].size
    ml = np.zeros(L)
    plev = d["p_lev"].astype(np.int64)
    gnn = model.gnn is not None
    with torch.no_grad():
        for b0 in range(0, N, batch):
            rows = np.arange(b0, min(N, b0 + batch))
            items = np.c_[np.zeros(rows.size, np.int64), rows, np.full(rows.size, ri)]
            bt = C21.make_batch21(items, [d], [c], None if lb is None else [lb], st, pairs=model.g is not None,
                                  train=False, gnn=gnn, att=gnn and model.att)
            if placed(device):
                with torch.device(device):
                    s, lp = C20.forward20(model, to_dev(bt, device))
            else:
                s, lp = C20.forward20(model, bt)
            s = s.double().cpu().numpy()
            Pm = None if lp is None else np.exp(lp.double().cpu().numpy())
            for j, i in enumerate(rows):
                out[i] = rank(d, i, s[j, :int(d["n"][i])])
                if Pm is not None:
                    p0, p1 = int(d["rowoff"][i]), int(d["rowoff"][i + 1])
                    if p1 > p0:
                        ml += np.bincount(plev[p0:p1], weights=Pm[j, :p1 - p0], minlength=L)
    return out, ml / max(N, 1)


def train_dev(arm, f, builds, comps, LB, groups, Sd, Sc, Slb, srule, epochs, per_epoch, batch, seed, device=None,
              threads=1, spec=None, make_batch=None):
    """chainscore21's train21 with an optional device. spec (default chainscore21's arm_spec) and make_batch (default
    make_batch21) let a later part pass its own arm table and batch maker; the training statements are train21's.
    Returns (model on the device, info with placement, state_sha256, seconds_per_batch and the deterministic
    warnings)."""
    import torch
    block = settings(device, threads)
    spec = spec or C21.arm_spec(arm, f)
    make_batch = make_batch or C21.make_batch21
    st = C19.input_stats(builds[0])
    model = C20.make_model(f, seed)
    if placed(device):
        _dp().model_to(model, device)       # once, before the optimiser is built
    opt = torch.optim.Adam(model.parameters(), lr=C19.LR, weight_decay=C19.WD)
    rng = np.random.default_rng(seed)
    hist, best, best_state = [], None, None
    tb, nbt = 0.0, 0
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for ep in range(epochs):
            t1 = time.time()
            model.train()
            items = C21.draw21(rng, builds, groups, per_epoch)
            tot, tn, tc, nb = 0.0, 0.0, 0.0, 0
            for b0 in range(0, len(items), batch):
                t2 = time.time()
                bt = make_batch(items[b0:b0 + batch], builds, comps, LB, st, pairs=spec["g"], gnn=spec["gnn"],
                                att=spec["att"])
                if placed(device):
                    bt = to_dev(bt, device)
                    with torch.device(device):
                        s, lp = C20.forward20(model, bt)
                        loss, ln, lc = C19.objective(s, lp, bt)
                else:
                    s, lp = C20.forward20(model, bt)
                    loss, ln, lc = C19.objective(s, lp, bt)
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += float(loss.detach())
                tn += ln
                tc += lc
                nb += 1
                tb += time.time() - t2
                nbt += 1
            model.eval()
            x, _ = read_dev(model, Sd, Sc, Slb, st, srule, device)
            sr5 = float(x[:, 0].mean())
            hist.append({"epoch": ep, "loss": round(tot / max(nb, 1), 5), "node_loss": round(tn / max(nb, 1), 5),
                         "chain_loss": round(tc / max(nb, 1), 5), "select_r5": round(sr5, 5),
                         "by_build": np.bincount(items[:, 0], minlength=len(builds)).tolist(),
                         **C20.coupling20(model), "seconds": round(time.time() - t1, 1)})
            log(f"    {arm} ({f}) ep {ep}: loss {tot / max(nb, 1):.4f} (node {tn / max(nb, 1):.4f}, chain "
                f"{tc / max(nb, 1):.4f}) select R@5 {sr5:.4f} {C20.coupling20(model)} {time.time() - t1:.0f}s")
            if best is None or sr5 > best[1]:
                best = (ep, sr5)
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    det = sorted({str(w.message)[:300] for w in caught if "deterministic" in str(w.message).lower()})
    DET_WARNINGS.extend(det)
    model.load_state_dict(best_state)
    model.eval()
    return model, {"history": hist, "best_epoch": best[0], "best_select_r5": round(best[1], 5), "stats": st,
                   "coupling": C20.coupling20(model), "spec": spec, "placement": block,
                   "state_sha256": state_sha(best_state), "seconds_per_batch": round(tb / max(nbt, 1), 4),
                   "deterministic_warnings": det}


def run21(a, rest):
    """A chainscore21 arm through this module: chainscore21's train_cmd with train21 and read21 replaced by the
    device copies (it saves no weights here: --state-out is refused; state_sha256 pins them)."""
    if any(x.startswith("--state-out") or x.startswith("--out") or x.startswith("--rows-out") for x in rest):
        raise SystemExit("run21 sets --out and --rows-out itself and saves no weights")
    extra = {}

    def train21(arm, f, builds, comps, LB, groups, Sd, Sc, Slb, srule, epochs, per_epoch, batch, seed, threads=1):
        model, info = train_dev(arm, f, builds, comps, LB, groups, Sd, Sc, Slb, srule, epochs, per_epoch, batch,
                                seed, device=a.device, threads=threads)
        extra.update({k: info[k] for k in ("placement", "state_sha256", "seconds_per_batch",
                                           "deterministic_warnings")})
        return model, info

    def read21(model, d, c, lb, st, rule, ri=0, batch=BATCH):
        return read_dev(model, d, c, lb, st, rule, a.device, ri=ri, batch=batch)

    C21.train21, C21.read21 = train21, read21
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    t = sub.add_parser("train")
    for k in ("--arm", "--f", "--fit", "--select", "--pfit", "--pselect"):
        t.add_argument(k, default=None)
    for k in ("--aug", "--read", "--pread", "--edges"):
        t.add_argument(k, action="append", default=[])
    t.add_argument("--epochs", type=int, default=C19.EPOCHS)
    t.add_argument("--per-epoch", type=int, default=C19.PER_EPOCH)
    t.add_argument("--batch", type=int, default=BATCH)
    t.add_argument("--read-batch", type=int, default=BATCH)
    ta = ap.parse_args(["train"] + rest)
    ta.out, ta.rows_out, ta.state_out = a.out, a.rows_out, None
    res = C21.train_cmd(ta)
    res.update({"look": "chainscore21 via cs_dev", "cs_dev_sha256": sha(__file__), "device": str(a.device), **extra})
    p = Path(a.out)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    os.replace(tmp, p)
    log(f"wrote {p}: {a.device} state {extra.get('state_sha256', '')[:16]} {extra.get('seconds_per_batch')} s/batch "
        f"det warnings {len(extra.get('deterministic_warnings', []))} ({res['seconds']}s)")
    return res


def compare(a):
    """Two run21 outputs (or a run21 output and a chainscore21 arm): rows, history and weights."""
    x, y = (json.loads(Path(p).read_text(encoding="utf-8")) for p in (a.x, a.y))
    with np.load(x["rows_out"]) as zx, np.load(y["rows_out"]) as zy:
        rows = {k: float(np.abs(zx[k].astype(np.float64) - zy[k].astype(np.float64)).max()) for k in zx.files
                if k in zy.files}
    hk = ("loss", "node_loss", "chain_loss", "select_r5")
    hist = all(all(hx[k] == hy[k] for k in hk) for hx, hy in zip(x["history"], y["history"])) and \
        len(x["history"]) == len(y["history"])
    out = {"rows_max_abs_diff": rows, "history_equal": hist, "best_epoch": [x["best_epoch"], y["best_epoch"]],
           "state_sha256": [x.get("state_sha256"), y.get("state_sha256")]}
    out["identical"] = hist and all(v == 0.0 for v in rows.values()) and \
        (None in out["state_sha256"] or out["state_sha256"][0] == out["state_sha256"][1])
    print(json.dumps(out, indent=1))
    return out


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    rest = []
    if "--" in argv:
        k = argv.index("--")
        argv, rest = argv[:k], argv[k + 1:]
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run21")
    r.add_argument("--device", default="cpu")
    r.add_argument("--out", required=True)
    r.add_argument("--rows-out", required=True)
    c = sub.add_parser("compare")
    c.add_argument("x")
    c.add_argument("y")
    a = ap.parse_args(argv)
    if a.cmd == "run21":
        run21(a, rest)
    else:
        compare(a)


if __name__ == "__main__":
    main()
