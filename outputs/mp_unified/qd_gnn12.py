"""Design look (untracked; not a result and not filed): qd_gnn11.py, pinned and unchanged, with QD's weights averaged over
training instead of taken at the best select epoch: the MLP side's weight averaging (lean_mlp5) carried to the GNN.
Averaged weights sit in flatter minima, which is what generalisation to an unseen domain asks of a model (SWAD, Cha et al.,
NeurIPS 2021); QD's open problem is the unseen graph.

--swa-from S   average the parameters at the end of epochs S .. epochs-1 (epoch mode);
--swa-step     with --swa-from, average after every optimiser step from the start of epoch S instead (SWAD's dense form).
Without --swa-from the run is qd_gnn11's, draw for draw: nothing is rebound. With it, qd_gnn2.fit_shared2 (which
qd_gnn2.run calls by name for every arm and seed) is replaced by fit_swa: its loop line for line (the same seeding,
batches, steps and select reads, so the curve and the best epoch are the ones qd_gnn11 would log) plus a running mean of
the parameters (buffers are left as the last step has them; prepare() re-sets them before every read). At the end the
model holds the averaged parameters, which the margin choice, every read, kalpha and the saved model then use.
best_epoch in the output stays the select-best epoch, recorded only; the "swa" block holds, per arm and seed in run
order, the averaged model's select score per graph beside the best epoch's.
Every arm of the run is averaged. QD arms are message passing; walk arms (W3/W4) are not; averaging changes neither.

    python outputs/mp_unified/qd_gnn12.py --selftest
    python outputs/mp_unified/qd_gnn12.py --swa-from 2 --train 2wiki=x4 --read hotpotqa=x1,musique=x1,metaqa=x1,webqsp=x1 \
        --arms QD-T0-L2-tb5-ed50,QD-TXT-L2-dr25e10-tb5-ed50 --rule both --epochs 6 --kalpha 64,256 --out outputs/mp_unified/qd/sw-2w.json
"""
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import qd_gnn11 as Q11  # noqa: E402  (imports qd_gnn10 ... qd_gnn, which sets the BLAS thread counts before numpy loads)

import numpy as np  # noqa: E402
import torch  # noqa: E402

Q10, Q7, Q6x, Q5, Q4, Q2, QG = Q11.Q10, Q11.Q7, Q11.Q6x, Q11.Q5, Q11.Q4, Q11.Q2, Q11.QG
SHAS = {**Q11.SHAS, "qd_gnn12": QG.sha(__file__)}   # at import: what ran
SWA = {"from": None, "step": False, "fits": []}
_FIT2 = Q2.fit_shared2
log = QG.log


def strip_swa(argv):
    """argv without --swa-from S and --swa-step; returns (argv, S or None, step)."""
    out, s, step, i = [], None, False, 0
    while i < len(argv):
        a = argv[i]
        if a == "--swa-from":
            if i + 1 >= len(argv):
                raise SystemExit("--swa-from needs an epoch")
            s, i = int(argv[i + 1]), i + 2
            continue
        if a.startswith("--swa-from="):
            s, i = int(a.split("=", 1)[1]), i + 1
            continue
        if a == "--swa-step":
            step, i = True, i + 1
            continue
        out.append(a)
        i += 1
    if step and s is None:
        raise SystemExit("--swa-step needs --swa-from")
    if s is not None and s < 0:
        raise SystemExit("--swa-from must be an epoch, 0 or more")
    return out, s, step


def fit_swa(arm, names, tr, sel, rule, epochs, lr, wd, seed, balance, AU):
    """qd_gnn2.fit_shared2, line for line, plus the running mean of the parameters (module docstring)."""
    if not 0 <= SWA["from"] < epochs:
        raise SystemExit(f"--swa-from {SWA['from']} is not an epoch of a {epochs}-epoch fit")
    torch.manual_seed(seed)
    model = arm.make()
    mats = [p for p in model.parameters() if p.dim() >= 2]
    rest = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": mats, "weight_decay": wd}, {"params": rest, "weight_decay": 0.0}], lr=lr)
    rng = np.random.default_rng(seed)
    streams = {g: {"perm": None, "pos": 0} for g in names}
    best, best_state, best_ep, curve, by_graph = -1.0, None, -1, [], {g: [] for g in names}
    sel_rule = "np" if rule == "both" else rule
    cols = slice(0, 2) if sel_rule == "p" else slice(0, 3)
    params = dict(model.named_parameters())
    acc = {k: torch.zeros_like(p, dtype=torch.float64) for k, p in params.items()}
    n_acc = [0]

    def add():
        with torch.no_grad():
            for k, p in params.items():
                acc[k] += p.detach().double()
        n_acc[0] += 1

    for ep in range(epochs):
        t_ep = time.time()
        batches = AU.epoch_batches(rng, names, tr, balance, streams)
        model.train()
        for g, rows in batches:
            arm.prepare(model, g)
            s, gold, z = arm.forward(model, g, rows)
            loss = Q2.step_loss2(s, gold, z, rule)
            if loss is None:
                continue
            opt.zero_grad()
            loss.backward()
            opt.step()
            if SWA["step"] and ep >= SWA["from"]:
                add()
        sc = {}
        for g in names:
            arm.prepare(model, g)
            sc[g] = float(arm.read(model, g, list(sel[g]), sel_rule)[:, cols].mean())
            by_graph[g].append(round(sc[g], 4))
        score = sc[names[0]] if len(names) == 1 else float(np.mean([sc[g] for g in names]))
        curve.append(round(score, 4))
        if score > best:
            best, best_ep, best_state = score, ep, {k: v.detach().clone() for k, v in model.state_dict().items()}
        log(f"  ep {ep}: select {round(score, 4)} " + " ".join(f"{g} {sc[g]:.4f}" for g in names) + f" ({time.time() - t_ep:.0f}s)")
        if not SWA["step"] and ep >= SWA["from"]:
            add()
    if n_acc[0] == 0:
        raise SystemExit("no step was averaged (every batch of the averaged epochs had no gold)")
    with torch.no_grad():
        for k, p in params.items():
            p.copy_((acc[k] / n_acc[0]).to(p.dtype))
    sw = {}
    for g in names:
        arm.prepare(model, g)
        sw[g] = round(float(arm.read(model, g, list(sel[g]), sel_rule)[:, cols].mean()), 4)
    SWA["fits"].append({"seed": seed, "best_epoch": best_ep, "best_select": round(best, 4),
                        "best_select_by_graph": {g: by_graph[g][best_ep] for g in names}, "swa_select_by_graph": sw,
                        "swa_select": sw[names[0]] if len(names) == 1 else round(float(np.mean([sw[g] for g in names])), 4),
                        "averaged": n_acc[0], "unit": "step" if SWA["step"] else "epoch", "from_epoch": SWA["from"],
                        "epochs": epochs})
    log(f"  swa ({n_acc[0]} {'steps' if SWA['step'] else 'epochs'} from epoch {SWA['from']}): select "
        + " ".join(f"{g} {sw[g]:.4f}" for g in names) + f" (best epoch {best_ep}: {round(best, 4)})")
    return model, best_ep, curve, by_graph


def finish(out):
    res = json.loads(out.read_text(encoding="utf-8"))
    res["look"] = "qd_gnn12"
    res["pins"]["qd_gnn11"] = SHAS["qd_gnn11"]
    res["qd_gnn12_sha256"] = SHAS["qd_gnn12"]
    fits = list(SWA["fits"])
    block = {"from_epoch": SWA["from"], "unit": "step" if SWA["step"] else "epoch", "by_arm": {}}
    if SWA["from"] is not None:
        for name, r in res.get("results", {}).items():
            for sd in r.get("seeds_read", {}):
                if not fits:
                    raise SystemExit("fewer averaged fits than arms x seeds in the output")
                f = fits.pop(0)
                if str(f["seed"]) != str(sd):
                    raise SystemExit(f"averaged fit order: seed {f['seed']} where {name} has seed {sd}")
                block["by_arm"].setdefault(name, {})[str(sd)] = f
        if fits:
            raise SystemExit(f"{len(fits)} averaged fits have no arm in the output")
    res["swa"] = block
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")


def main(selftest_run=False):
    if "--selftest" in sys.argv and not selftest_run:
        selftest()
        return
    sys.argv, s, step = strip_swa(sys.argv)
    SWA.update({"from": s, "step": step, "fits": []})
    Q2.fit_shared2 = _FIT2 if s is None else fit_swa
    _finish11 = Q11.__dict__["_finish11"]
    Q11.finish = lambda out: (_finish11(out), finish(out))   # qd_gnn11.main's finish lambda calls Q11.finish by name
    Q11.main(selftest_run=selftest_run)


Q11._finish11 = Q11.finish


# ── selftest ─────────────────────────────────────────────────────────────────


def selftest():
    sys.path.insert(0, str(QG.AU_DIR))
    sys.path.insert(0, str(QG.KB_DIR))
    import anchor_univ as AU
    A16 = AU.A16
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(2)
    rng = np.random.default_rng(0)
    K = 8
    Q = QG.synthetic_rows(60, rng, K)
    TOK = [QG.edge_labels(q) for q in Q]
    z_of = [A16.zscore(q["score"][:, 0]) for q in Q]
    phi = rng.standard_normal((QG.R_TOP, QG.TDIM)).astype(np.float32)
    G = {"g": {"phi": phi, "kind": "passage"}}
    rows = list(range(60))
    tr, sel = {"g": rows[:40]}, {"g": rows[40:]}
    Q5.bind()
    assert strip_swa(["x", "--swa-from", "2", "--swa-step", "--rule", "np"]) == (["x", "--rule", "np"], 2, True)
    assert strip_swa(["x", "--swa-from=1"]) == (["x"], 1, False) and strip_swa(["x"]) == (["x"], None, False)
    for bad in (["x", "--swa-step"], ["x", "--swa-from", "-1"]):
        try:
            strip_swa(bad)
            raise AssertionError(bad)
        except SystemExit:
            pass
    for nm, extra in (("QD-T0-L2-d16", {}), ("QD-TXT-L2-d16-K8", {"drop_row": 0.25, "drop_edge": 0.10})):
        sp = {**QG.parse_qd(nm), **extra}       # qd_gnn3.parse_arms' form of -dr25e10 (relation dropout draws)
        nm = nm + ("-dr25e10" if extra else "")
        # the loop is fit_shared2's: the curve and best epoch match, and averaging only the last epoch is its weights
        arm0 = QG.QDArm(sp, Q, TOK, G, z_of, A16, 0)
        m0, e0, c0, b0 = _FIT2(arm0, ["g"], tr, sel, "both", 3, 2e-3, 1e-4, 0, "graph", AU)
        SWA.update({"from": 2, "step": False, "fits": []})
        arm1 = QG.QDArm(sp, Q, TOK, G, z_of, A16, 0)
        m1, e1, c1, b1 = fit_swa(arm1, ["g"], tr, sel, "both", 3, 2e-3, 1e-4, 0, "graph", AU)
        assert (c0, e0, b0) == (c1, e1, b1), f"{nm}: the averaged loop's curve differs from fit_shared2's"
        last = SWA["fits"][0]
        assert last["averaged"] == 1 and last["unit"] == "epoch" and last["best_epoch"] == e0
        # a three-epoch fit from 0: the mean of the three epoch-end weights, which fit_shared2's own loop gives
        ends = []
        _read = arm0.__class__.read

        def spy(self, model, g, rr, rule, *x, **k):
            ends.append({kk: v.detach().clone() for kk, v in model.named_parameters()})
            return _read(self, model, g, rr, rule, *x, **k)

        arm0.__class__.read = spy
        try:
            arm2 = QG.QDArm(sp, Q, TOK, G, z_of, A16, 0)
            _FIT2(arm2, ["g"], tr, sel, "both", 3, 2e-3, 1e-4, 0, "graph", AU)
        finally:
            arm0.__class__.read = _read
        assert len(ends) == 3, len(ends)
        SWA.update({"from": 0, "step": False, "fits": []})
        arm3 = QG.QDArm(sp, Q, TOK, G, z_of, A16, 0)
        m3, _e, _c, _b = fit_swa(arm3, ["g"], tr, sel, "both", 3, 2e-3, 1e-4, 0, "graph", AU)
        for kk, p in m3.named_parameters():
            want = torch.stack([e[kk].double() for e in ends]).mean(0).to(p.dtype)
            assert torch.allclose(p.detach(), want, atol=1e-6), f"{nm}: {kk} is not the mean of the epoch ends"
        # step mode averages more snapshots than epochs, and the averaged model scores every pool node
        SWA.update({"from": 1, "step": True, "fits": []})
        arm4 = QG.QDArm(sp, Q, TOK, G, z_of, A16, 0)
        m4, _e, _c, _b = fit_swa(arm4, ["g"], tr, sel, "both", 3, 2e-3, 1e-4, 0, "graph", AU)
        assert SWA["fits"][0]["averaged"] >= 2 and SWA["fits"][0]["unit"] == "step"   # at least one step an epoch
        arm4.prepare(m4, "g")
        s_id = arm4.forward(m4.eval(), "g", rows[:8])[0]
        fin = torch.isfinite(s_id)
        assert all(int(fin[bi].sum()) == Q[i]["n"] for bi, i in enumerate(rows[:8])), f"{nm}: a pool node is not scored"
        print(f"selftest {nm}: fit_swa's curve == fit_shared2's; last-epoch average == its weights; epoch mean exact; "
              f"step mode averaged {SWA['fits'][0]['averaged']} steps")
    # finish maps the averaged fits onto arms x seeds in run order and refuses a mismatch
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "r.json"
        base = {"pins": {}, "results": {"A": {"seeds_read": {"0": {}, "1": {}}}, "B": {"seeds_read": {"0": {}}}}}
        out.write_text(json.dumps(base), encoding="utf-8")
        SWA.update({"from": 2, "step": False, "fits": [{"seed": 0}, {"seed": 1}, {"seed": 0}]})
        finish(out)
        r = json.loads(out.read_text(encoding="utf-8"))
        assert r["look"] == "qd_gnn12" and set(r["swa"]["by_arm"]) == {"A", "B"} and r["swa"]["by_arm"]["A"]["1"] == {"seed": 1}
        out.write_text(json.dumps(base), encoding="utf-8")
        SWA.update({"fits": [{"seed": 0}, {"seed": 0}, {"seed": 0}]})
        try:
            finish(out)
            raise AssertionError("a seed mismatch must be refused")
        except SystemExit:
            pass
    Q2.fit_shared2 = _FIT2
    # end to end, each in a fresh process: without --swa-from the run reads as qd_gnn11's; averaged runs read otherwise
    import os
    import subprocess
    with tempfile.TemporaryDirectory() as d:
        res = {}
        for mode, script in (("plain11", Q11.__file__), ("plain", __file__), ("epoch", __file__), ("step", __file__)):
            out = Path(d) / f"{mode}.json"
            argv = ([sys.executable, script, "--selftest-run", "plain", d, str(out)] if mode == "plain11"
                    else [sys.executable, script, "--selftest-run", mode, str(out)])
            r = subprocess.run(argv, env=dict(os.environ), capture_output=True, text=True, timeout=1800)
            if r.returncode != 0:
                print(r.stdout[-3000:], r.stderr[-3000:])
                raise AssertionError(f"selftest run {mode} failed")
            res[mode] = json.loads(out.read_text(encoding="utf-8"))

        def reads(r):
            return {a: {s: sv["reads"] for s, sv in v["seeds_read"].items()} for a, v in r["results"].items()}
        def states(mode):     # the saved model of each arm (qd_gnn2.run: <out stem>_models/a<arm>_s<seed>.pt)
            return [torch.load(f, weights_only=False)["state_dict"] for f in sorted((Path(d) / f"{mode}_models").glob("a*_s*.pt"))]

        def same(a, b):
            return len(a) == len(b) and all(x.keys() == y.keys() and all(torch.equal(x[k], y[k]) for k in x) for x, y in zip(a, b))
        assert reads(res["plain"]) == reads(res["plain11"]), "without --swa-from the run must read as qd_gnn11's"
        assert same(states("plain"), states("plain11")) and len(states("plain")) == 2, "without --swa-from: qd_gnn11's models"
        assert res["plain"]["look"] == "qd_gnn12" and res["plain"]["swa"]["from_epoch"] is None
        for mode in ("epoch", "step"):
            sw = res[mode]["swa"]
            assert sw["unit"] == mode and sw["from_epoch"] == 0 and set(sw["by_arm"]) == set(res[mode]["results"]), sw
            assert all(f["averaged"] >= 2 for v in sw["by_arm"].values() for f in v.values())
            assert not same(states(mode), states("plain")), f"{mode}: averaging must change the saved models"
        a0 = Q11.ARMS_ST.split(",")[0]
        print(f"selftest e2e: no --swa-from runs as qd_gnn11 bit for bit (reads and saved models); averaged runs save other "
              f"models ({a0} ID/np " + ", ".join(f"{m} {res[m]['results'][a0]['seeds_read']['0']['reads']['metaqa']['ID/np']['fit']}"
                                                 for m in ("plain", "epoch", "step")) + ")")
    print("selftest ok")


def selftest_run(mode, out):
    """One end-to-end run on qd_gnn11's synthetic graph: plain, or averaged from epoch 0 by epoch or by step (from
    epoch 1 of a 2-epoch fit the epoch mean is the last epoch's weights, which the best epoch can equal)."""
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(2)
    phi = np.random.default_rng(7).standard_normal((QG.R_TOP, QG.TDIM)).astype(np.float32)
    names = [f"rel {j}" for j in range(Q11.K_ST)]

    def fake(trains, reads, max_len, t0, AU_):
        Qf, part = Q11.synthetic()
        Gf = {"metaqa": {"kind": "kb", "role": "train", "phi": phi, "phi_sha": "synthetic", "n_rel": Q11.K_ST, "vocab": names,
                         "info": {"pruned_loader": {"synthetic": True}}}}
        return Qf, part, Gf

    Q4._load_all = fake
    sys.argv = (["qd_gnn12.py", "--train", "metaqa=fit", "--arms", Q11.ARMS_ST, "--rule", "np", "--epochs", "2", "--out", out,
                 "--kalpha", "4,8", "--kalpha-draws", "2"]
                + ([] if mode == "plain" else ["--swa-from", "0"] + (["--swa-step"] if mode == "step" else [])))
    main(selftest_run=True)


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--selftest-run":
        selftest_run(sys.argv[2], sys.argv[3])
    else:
        main()
