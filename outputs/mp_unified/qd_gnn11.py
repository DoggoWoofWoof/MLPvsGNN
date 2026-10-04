"""Design look (untracked; not a result and not filed): qd_gnn10.py, pinned and unchanged, with QD's base z taken from a
lean MLP instead of the twin, so the GNN residual runs over the same lean features (the user's "our gnn should go
through the same features or training regimes").

--lean-base TAG: after loading, each row's stored twin0 score (the column QD's z, its rank-1 protection and kalpha's
alpha0 read; nothing else in QD reads the stored scores) is replaced by a saved lean MLP's per-node score,
outputs/mp_unified/lean/scores/TAG/<ds>/<carve>.npz, written by lean_score.py. load_pruned orders a carve's rows by
sorted chunk_rows and keeps no row id, so the k-th loaded row is matched to the k-th smallest scored chunk row, and every
row's pool must equal the scored pool node for node, or it refuses. Every carve of every trained or read graph needs
scores. rho and abs stay against the stored twin0 and gnn0 metrics, as in every QD read; the lean base itself is
recorded on the same read rows (anchor_gen.Reader at QD's read seed), so QD's lift over its own base reads directly.
--base gnn is refused with --lean-base. Without --lean-base the run is qd_gnn10's, draw for draw.

The lean base must not be trained on QD's training carve: QD trains 2wiki on x4, so its lean base is trained on fit
(and selected on select, which QD also selects on: the base's block choice and QD's epoch both see select).
A QD arm over a lean z is message passing, as every QD arm is; the lean base alone is non-MP (fixed per-node counts and
means; DISTS/F and NBR2 are parameter-free propagation, flagged).

    python outputs/mp_unified/qd_gnn11.py --selftest
    python outputs/mp_unified/qd_gnn11.py --lean-base l3-2wf-pick --train 2wiki=x4 --arms QD-T0-L2,QD-TXT-L2-dr25e10 \
        --rule np --epochs 6 --out outputs/mp_unified/qd/qdl-2w.json
"""
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import qd_gnn10 as Q10  # noqa: E402  (imports qd_gnn7 ... qd_gnn, which sets the BLAS thread counts before numpy loads)

import numpy as np  # noqa: E402

Q7, Q6x, Q5, Q4, Q2, QG = Q10.Q7, Q10.Q6x, Q10.Q5, Q10.Q4, Q10.Q2, Q10.QG
SHAS = {**Q10.SHAS, "qd_gnn11": QG.sha(__file__)}   # at import: what ran
SCORES = HERE / "lean" / "scores"
LEAN = {"tag": None, "parts": {}, "read_rows": {}}
log = QG.log


def strip_arg(argv):
    """argv without --lean-base TAG, and TAG (None when absent)."""
    out, tag, i = [], None, 0
    while i < len(argv):
        a = argv[i]
        if a == "--lean-base":
            if i + 1 >= len(argv) or argv[i + 1].startswith("--"):
                raise SystemExit("--lean-base needs a tag")
            tag, i = argv[i + 1], i + 2
            continue
        if a.startswith("--lean-base="):
            tag, i = a.split("=", 1)[1], i + 1
            continue
        out.append(a)
        i += 1
    if tag is not None and (not tag or "/" in tag or "\\" in tag):
        raise SystemExit(f"--lean-base {tag!r}: a tag is one directory name under {SCORES}")
    return out, tag


def base_of(argv):
    for i, a in enumerate(argv):
        if a == "--base" and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith("--base="):
            return a.split("=", 1)[1]
    return "twin"


def lean_part(Q, rows, ds, cv, tag, A16):
    """Replace column 0 of each row's stored scores with the tag's lean scores; per-row lean and twin0 metrics."""
    f = SCORES / tag / ds / f"{cv}.npz"
    if not f.exists():
        raise SystemExit(f"--lean-base {tag}: no scores for {ds}={cv} ({f})")
    with np.load(f) as z:
        cr, n, pool, s = (z[k] for k in ("chunk_rows", "n", "pool", "score"))
    off = np.r_[0, np.cumsum(n)]
    if cr.size != len(rows) or np.unique(cr).size != cr.size or int(off[-1]) != s.size or s.size != pool.size:
        raise SystemExit(f"--lean-base {tag}: {ds}={cv} has {cr.size} scored rows ({np.unique(cr).size} distinct) "
                         f"against {len(rows)} loaded")
    order = np.argsort(cr, kind="stable")
    m_lean = np.zeros((len(rows), 3))
    m_twin = np.zeros((len(rows), 3))
    agree = 0
    for k, i in enumerate(rows):
        q, j = Q[i], int(order[k])
        a, b = int(off[j]), int(off[j + 1])
        if "pool" not in q or int(q["n"]) != int(n[j]) or not np.array_equal(np.asarray(q["pool"], np.int64), pool[a:b]):
            raise SystemExit(f"--lean-base {tag}: {ds}={cv} row {k} (chunk row {int(cr[j])}): the pool differs from the scored one")
        sc = q["score"].astype(np.promote_types(q["score"].dtype, np.float32), copy=True)
        m_twin[k] = A16.metrics_of(sc[:, 0], q["gold"], q["gt"])
        agree += int(np.array_equal(m_twin[k], np.asarray(q["metrics"])[0, A16.RI]))
        sc[:, 0] = s[a:b]
        q["score"] = sc
        m_lean[k] = A16.metrics_of(sc[:, 0], q["gold"], q["gt"])
    meta_f = f.with_suffix(".json")
    meta = json.loads(meta_f.read_text(encoding="utf-8")) if meta_f.exists() else {}
    LEAN["parts"][f"{ds}={cv}"] = {
        "scores": str(f.relative_to(SCORES)), "sha256": QG.sha(f), "rows": len(rows), "pools_matched": len(rows),
        "twin0_recomputed_equals_stored": round(agree / max(len(rows), 1), 4),
        "lean_mean": m_lean.mean(0).round(4).tolist(), "twin0_mean": m_twin.mean(0).round(4).tolist(),
        "meta": {k: meta.get(k) for k in ("models", "models_sha256", "name", "look", "store", "blocks", "fit_args", "freeze",
                                          "placement", "script_sha256")}}
    return m_lean


def apply_lean(Q, part, trains, reads, tag, AU):
    """Every loaded carve gets its lean scores; the lean base is recorded on QD's read rows and on select."""
    A16, AG = AU.A16, AU.AG
    per_row = {}
    for (ds, cv), rows in sorted(part.items()):
        m = lean_part(Q, rows, ds, cv, tag, A16)
        per_row.update({i: m[k] for k, i in enumerate(rows)})
        log(f"lean base {tag}: {ds}={cv} {len(rows)} rows matched, lean {LEAN['parts'][f'{ds}={cv}']['lean_mean']}")
    graphs = sorted({ds for ds, _cv in part})
    for g in graphs:
        if g in trains:
            sets = {"B": part[(g, "x1")][1::2], "select": part[(g, "select")]}
        else:
            rd = part[(g, Q2.READ[g])]
            sets = {"B": list(rd) if g in Q2.WHOLE else rd[1::2]}
        for nm, rows in sets.items():
            R = AG.Reader(Q, rows, QG.READ_SEED)
            m = np.stack([per_row[i] for i in rows])
            LEAN["read_rows"][f"{g}/{nm}"] = {**R.base(), "lean": R.record(m, by_type=False)}
            log(f"lean base {tag} on {g}/{nm} ({len(rows)} rows): {R.record(m, by_type=False)['fit']} "
                f"rho {R.rho(m)} (twin0 {R.base()['twin0']}, gnn0 {R.base()['gnn0']})")


def lean_load(inner):
    def load(trains, reads, max_len, t0, AU):
        Q, part, G = inner(trains, reads, max_len, t0, AU)
        if LEAN["tag"] is not None:
            apply_lean(Q, part, trains, reads, LEAN["tag"], AU)
        return Q, part, G
    return load


def finish(out):
    res = json.loads(out.read_text(encoding="utf-8"))
    res["look"] = "qd_gnn11"
    res["pins"]["qd_gnn10"] = SHAS["qd_gnn10"]
    res["qd_gnn11_sha256"] = SHAS["qd_gnn11"]
    res["lean_base"] = {"tag": LEAN["tag"], "z": "twin0's stored score" if LEAN["tag"] is None else f"lean scores {LEAN['tag']}",
                        "parts": LEAN["parts"], "read_rows": LEAN["read_rows"]}
    if LEAN["tag"] is not None:      # qd_gnn4 labels z by its --base ('twin'); here column 0 holds the lean scores
        res["base_z"] = f"lean:{LEAN['tag']}"
        for recs in (res.get("kalpha") or {}).get("reads", {}).values():
            for r in recs:
                if isinstance(r, dict) and "base_z" in r:
                    r["base_z"] = f"lean:{LEAN['tag']}"
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")


def main(selftest_run=False):
    if "--selftest" in sys.argv and not selftest_run:
        selftest()
        return
    sys.argv, tag = strip_arg(sys.argv)
    if tag is not None and base_of(sys.argv) != "twin":
        raise SystemExit("--lean-base replaces twin0's column; it cannot run with --base gnn")
    LEAN.update({"tag": tag, "parts": {}, "read_rows": {}})
    Q4._load_all = lean_load(Q4._load_all)
    _finish10 = Q10.finish
    Q10.finish = lambda out: (_finish10(out), finish(out))   # qd_gnn10.main's finish lambda calls Q10.finish by name
    Q10.main(selftest_run=selftest_run)


# ── selftest ─────────────────────────────────────────────────────────────────


K_ST = 8
ARMS_ST = "QD-T0-L2-d16,QD-TXT-L2-d16-K8-pc"


def synthetic(seed=5):
    """qd_gnn10's selftest graph: 80 synthetic KB rows, parts x1 / select / fit, a random label table."""
    Qf = Q6x.synthetic_kb(80, np.random.default_rng(seed), K_ST)
    for j, q in enumerate(Qf):
        q["pool"] = np.arange(q["n"]) + 1000 * j      # every row its own pool, so a mismatched row cannot pass
        q["score"] = q["score"].astype(np.float32)    # as the looks store it; a base equal to twin0 is then bit-equal
    part = {("metaqa", "x1"): list(range(0, 40)), ("metaqa", "select"): list(range(40, 52)), ("metaqa", "fit"): list(range(52, 80))}
    return Qf, part


def write_scores(root, tag, col, seed=11, noise=0.0):
    """Scores for the synthetic parts in a shuffled chunk order: each row's stored column col (plus noise)."""
    Qf, part = synthetic()
    rng = np.random.default_rng(seed)
    for (ds, cv), rows in part.items():
        cr_sorted = np.sort(rng.choice(10 ** 6, len(rows), replace=False))
        perm = rng.permutation(len(rows))            # the lean order: scored rows in chunk order, not sorted
        n = np.asarray([Qf[rows[k]]["n"] for k in perm], np.int64)
        pool = np.concatenate([np.asarray(Qf[rows[k]]["pool"], np.int64) for k in perm])
        s = np.concatenate([Qf[rows[k]]["score"][:, col].astype(np.float32) +
                            noise * rng.standard_normal(Qf[rows[k]]["n"]).astype(np.float32) for k in perm])
        d = Path(root) / tag / ds
        d.mkdir(parents=True, exist_ok=True)
        np.savez(d / f"{cv}.npz", chunk_rows=cr_sorted[perm], n=n, pool=pool, score=s)
        (d / f"{cv}.json").write_text(json.dumps({"models": "synthetic", "name": tag}), encoding="utf-8")


def selftest_run(mode, root, out):
    """One end-to-end run on the synthetic graph (a subprocess, so every bind starts fresh)."""
    global SCORES
    import torch
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(2)
    SCORES = Path(root)
    phi = np.random.default_rng(7).standard_normal((QG.R_TOP, QG.TDIM)).astype(np.float32)
    names = [f"rel {j}" for j in range(K_ST)]

    def fake(trains, reads, max_len, t0, AU_):
        Qf, part = synthetic()
        Gf = {"metaqa": {"kind": "kb", "role": "train", "phi": phi, "phi_sha": "synthetic", "n_rel": K_ST, "vocab": names,
                         "info": {"pruned_loader": {"synthetic": True}}}}
        return Qf, part, Gf

    Q4._load_all = fake
    sys.argv = ["qd_gnn11.py", "--train", "metaqa=fit", "--arms", ARMS_ST, "--rule", "np", "--epochs", "2", "--out", out,
                "--kalpha", "4,8", "--kalpha-draws", "2"] + ([] if mode == "plain" else ["--lean-base", mode])
    main(selftest_run=True)


def selftest():
    import tempfile
    sys.path.insert(0, str(QG.AU_DIR))
    sys.path.insert(0, str(QG.KB_DIR))
    import anchor_univ as AU
    A16 = AU.A16
    # argument handling
    assert strip_arg(["x", "--lean-base", "t1", "--epochs", "2"]) == (["x", "--epochs", "2"], "t1")
    assert strip_arg(["x", "--lean-base=t2"]) == (["x"], "t2")
    assert strip_arg(["x", "--epochs", "2"]) == (["x", "--epochs", "2"], None)
    for bad in (["--lean-base"], ["--lean-base", "--epochs"], ["--lean-base=a/b"], ["--lean-base="]):
        try:
            strip_arg(bad)
        except SystemExit:
            continue
        raise AssertionError(bad)
    assert base_of(["--base", "gnn"]) == "gnn" and base_of(["--base=gnn"]) == "gnn" and base_of([]) == "twin"
    print("selftest: --lean-base is stripped before qd_gnn10 parses; empty or path-like tags are refused")
    global SCORES
    old = SCORES
    tmp = Path(tempfile.mkdtemp(prefix="qd11_"))
    try:
        SCORES = tmp / "scores"
        write_scores(SCORES, "ident", 0)                # the twin's own column: QD must read exactly as qd_gnn10
        write_scores(SCORES, "gnnz", 3, noise=0.3)      # another base: z changes
        # unit: column 0 replaced, the rest untouched, matched through the shuffled order
        Qf, part = synthetic()
        before = [q["score"].copy() for q in Qf]
        LEAN.update({"tag": "gnnz", "parts": {}, "read_rows": {}})
        apply_lean(Qf, part, {"metaqa": ["fit"]}, [], "gnnz", AU)
        Qr, _ = synthetic()
        diffs = []
        for q, b in zip(Qf, before):
            assert np.array_equal(q["score"][:, 1:], b[:, 1:])
            diffs.append(float(np.abs(q["score"][:, 0] - b[:, 3]).max()))
        assert 0.0 < max(diffs) < 3.0, max(diffs)
        assert set(LEAN["read_rows"]) == {"metaqa/B", "metaqa/select"} and LEAN["parts"]["metaqa=x1"]["pools_matched"] == 40
        Qi, part_i = synthetic()
        apply_lean(Qi, part_i, {"metaqa": ["fit"]}, [], "ident", AU)
        assert all(np.array_equal(q["score"], b) for q, b in zip(Qi, before)), "a base equal to twin0 must leave the scores as they were"
        # refusals: a pool that differs, a missing carve, a row count that differs
        Qb, partb = synthetic()
        Qb[3]["pool"] = Qb[3]["pool"] + 1
        for Qx, px, why in ((Qb, partb, "pool"), (Qr, {**part, ("metaqa", "x4"): [0, 1]}, "missing"),
                            (Qr, {("metaqa", "x1"): list(range(0, 39))}, "count")):
            try:
                apply_lean(Qx, px, {"metaqa": ["fit"]}, [], "gnnz", AU)
            except SystemExit as e:
                print(f"selftest: refused ({why}): {str(e)[:90]}")
                continue
            raise AssertionError(why)
        # end to end, each in a fresh process: plain = qd_gnn10; ident = plain bit for bit; gnnz differs
        res = {}
        for mode in ("plain", "ident", "gnnz"):
            out = tmp / f"{mode}.json"
            env = {**os.environ, "QD11_SELFTEST_ROOT": str(SCORES)}
            r = subprocess.run([sys.executable, __file__, "--selftest-run", mode, str(SCORES), str(out)], env=env,
                               capture_output=True, text=True, timeout=1800)
            if r.returncode != 0:
                print(r.stdout[-3000:], r.stderr[-3000:])
                raise AssertionError(f"selftest run {mode} failed")
            res[mode] = json.loads(out.read_text(encoding="utf-8"))
            assert res[mode]["look"] == "qd_gnn11" and res[mode]["qd_gnn11_sha256"] == SHAS["qd_gnn11"]

        def reads(r):
            return {a: {s: sv["reads"] for s, sv in v["seeds_read"].items()} for a, v in r["results"].items()}
        assert res["plain"]["lean_base"]["tag"] is None and res["ident"]["lean_base"]["tag"] == "ident"
        assert res["plain"]["base_z"] == "twin" and res["gnnz"]["base_z"] == "lean:gnnz", (res["plain"]["base_z"], res["gnnz"]["base_z"])
        kz = {r.get("base_z") for recs in res["gnnz"]["kalpha"]["reads"].values() for r in recs if isinstance(r, dict)}
        assert kz <= {"lean:gnnz"}, kz
        assert reads(res["plain"]) == reads(res["ident"]), "a lean base equal to twin0 must read as qd_gnn10"
        assert reads(res["plain"]) != reads(res["gnnz"]), "another base must change the reads"
        lp = res["ident"]["lean_base"]["parts"]["metaqa=x1"]      # the synthetic stored metrics are random draws, so
        assert lp["lean_mean"] == lp["twin0_mean"], lp             # compare with twin0 recomputed from its scores
        a0 = ARMS_ST.split(",")[0]
        print(f"selftest e2e: --lean-base equal to twin0 reads as qd_gnn10 bit for bit; another base reads differently "
              f"({a0} ID/np {res['plain']['results'][a0]['seeds_read']['0']['reads']['metaqa']['ID/np']['fit']} -> "
              f"{res['gnnz']['results'][a0]['seeds_read']['0']['reads']['metaqa']['ID/np']['fit']}); "
              f"the lean base on B reads twin0's means when it is twin0")
    finally:
        SCORES = old
    print("selftest: all checks passed")


if __name__ == "__main__":
    if len(sys.argv) == 5 and sys.argv[1] == "--selftest-run":
        selftest_run(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        main()
