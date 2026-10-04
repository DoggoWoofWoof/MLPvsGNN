"""Design look (untracked; not a result and not filed): qd_gnn5.py, pinned and unchanged, with an unseen-relation
protocol for its QD, W3 and W4 arms: anchor_kbfit's --hold_rel carried into the shared loop.

--hold <graph>:<drop|mask>:<spec>   one training graph. spec is the graph's relation names separated by ';' (a KB's
    relation names, a passage graph's anchor phrases; each among the graph's 256 most frequent labels) or frac<F>s<S>
    (that fraction of the 256 most frequent label ranks, drawn with numpy seed S).
  drop   every held edge -- a structural pool edge whose label, in the direction qd_gnn.edge_labels reads it, is held --
         is deleted from every loaded row of the graph (fit, select and x1 rows alike), so the fit and the selection
         meet no trace of a held relation, typed or untyped. A KB row stores one walk edge per relation (kb_view), so on
         a KB this is anchor_kbfit's DROP exactly.
  mask   every held label is shown as 'other' on every loaded row (the edges stay, untyped): an unseen label.
The pool, the seeds and the twin's z were built from the full graph, as in anchor_kbfit: only the fitted model's view
changes. qd_gnn2.run's own reads of the graph are the held view (ID: DROP or MASK, NR, SHUF). On the graph's read rows
(x1's half B) the full rows, kept before the change, are read beside it:
  REV    the held labels shown as they are (text arms read them through their text, never trained on; ID arms read
         their ranks' untrained vectors)
  MASK   (drop) the held edges back, shown as 'other'
  FNR    the full rows with every label 'other' (the untyped graph)
  FULL   (T0 arms, which read no label; drop only) the held edges back
each under np, p and the run's margin (mp), on every read row and on the held rows -- the read rows where a
non-backtracking walk of up to the graph's loaded length from a seed passes a held edge and ends on a gold
(qd_gnn4.walk_types3 on a held / not-held token) -- and their one-edge subset, each paired against the held view's own
read (ID). Walk arms (W:) are refused under --hold: anchor_univ compiles their types outside the arm.

    python outputs/mp_unified/qd_gnn6.py --selftest
    python outputs/mp_unified/qd_gnn6.py --train metaqa=fit --arms QD-T0-L3,QD-TXT-L3-dr25e10,W3-TXT-L3-dr25e10 \
        --hold metaqa:drop:has_genre --rule both --epochs 6 --out outputs/mp_unified/qd/h6-mq-genre.json
(a name not in the vocabulary as written is read with each '_' as a space)
"""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import qd_gnn5 as Q5  # noqa: E402  (imports qd_gnn4 ... qd_gnn, which sets the BLAS thread counts before numpy loads)

import numpy as np  # noqa: E402
import torch  # noqa: E402

Q4, Q3, Q2, QG = Q5.Q4, Q5.Q3, Q5.Q2, Q5.QG
SHAS = {**Q5.SHAS, "qd_gnn6": QG.sha(__file__)}   # at import: what ran
HOLDRE = re.compile(r"^([a-z0-9]+):(drop|mask):(.+)$")
FRACRE = re.compile(r"^frac(0?\.[0-9]+|1(?:\.0*)?)s(\d+)$")
TOP = 256
EDGE_KEYS = ("u", "v", "fam", "fwd", "bwd", "w")
STRUCT_KEYS = ("w2_f", "w2_b", "w1_f", "w1_b")
KNOWN = set(EDGE_KEYS) | set(STRUCT_KEYS) | {"pool", "score", "gold", "proj", "metrics", "qemb", "seeds", "bucket"}
HOLD = {}     # graph, mode, spec; after the load: ranks, names
STATE = {"full": {}, "tok": {}, "held_rows": {}, "margin": None, "stats": {}, "out": None}
RES = {}      # arm name -> one entry per ID read of the held graph (one per seed)
ROWS = {}     # per-row metrics of the extra reads (saved beside the run's npz)
log = QG.log


# ── the held view ────────────────────────────────────────────────────────────


def held_edges(q, ranks):
    """Per pool edge: its label (qd_gnn.edge_labels: -1 off structural edges) is a held rank."""
    return np.isin(QG.edge_labels(q)[1], ranks)


def check_keys(q):
    E, Es = q["u"].size, int((q["fam"] == 0).sum())
    for k, val in q.items():
        if k in KNOWN or not isinstance(val, np.ndarray) or val.ndim == 0:
            continue
        if val.shape[0] in (E, Es):
            raise SystemExit(f"row key {k!r} has an edge-length first axis; the hold does not know how to cut it")


def drop_row(q, ranks):
    """(q without its held edges, how many went); the structural tables cut alike."""
    hit = held_edges(q, ranks)
    if not hit.any():
        return q, 0
    check_keys(q)
    keep = ~hit
    nq = dict(q)
    for k in EDGE_KEYS:
        nq[k] = q[k][keep]
    sm = q["fam"] == 0
    for k in STRUCT_KEYS:
        if k in q:
            nq[k] = q[k][keep[sm]]
    return nq, int(hit.sum())


def mask_row(q, ranks):
    """(q with every held rank in w2_f / w2_b shown as R_TOP, which no arm's K reaches: 'other'; its held edges)."""
    k = int(held_edges(q, ranks).sum())
    if not k:
        return q, 0
    nq = dict(q)
    for key in ("w2_f", "w2_b"):
        nq[key] = np.where(np.isin(q[key], ranks), QG.R_TOP, q[key]).astype(q[key].dtype)
    return nq, k


def resolve(vocab, spec):
    """The held ranks (sorted) and their names from a spec: names separated by ';' or frac<F>s<S>."""
    top = min(TOP, len(vocab))
    m = FRACRE.match(spec)
    if m:
        k = max(1, int(round(float(m.group(1)) * top)))
        ranks = np.sort(np.random.default_rng(int(m.group(2))).choice(top, size=k, replace=False))
    else:
        # a name not in the vocabulary as written is read with each '_' as a space ('has_genre'), so a command line
        # split on whitespace can carry it
        names = [s if s in vocab[:top] else s.replace("_", " ") for s in spec.split(";") if s]
        if not names or len(set(names)) != len(names):
            raise SystemExit(f"--hold {spec!r}: name the held labels once each")
        for nm in names:
            if nm not in vocab[:top]:
                raise SystemExit(f"--hold {nm!r} is not among the graph's {top} most frequent labels: {vocab[:min(top, 12)]} ...")
        ranks = np.asarray(sorted(vocab.index(nm) for nm in names))
    return ranks.astype(np.int64), [vocab[int(r)] for r in ranks]


def held_rows(Q, rows, ranks, L):
    """The rows where a non-backtracking walk of 1..L edges from a seed passes a held edge and ends on a gold, and
    the rows where a one-edge walk does."""
    out, out1 = [], []
    for i in rows:
        q = Q[i]
        codes, ptr, nodes = Q4.walk_types3(q, held_edges(q, ranks).astype(np.int64), 2, L)
        if codes.size == 0:
            continue
        _b, steps, ln = Q4.decode(codes, 2, L)
        thru = (steps == 1).any(1)
        one = (ln == 1) & (steps[:, 0] == 1)
        g = q["gold"]
        hit = np.asarray([bool(g[nodes[ptr[j]:ptr[j + 1]]].any()) for j in range(codes.size)])
        if (thru & hit).any():
            out.append(i)
        if (one & hit).any():
            out1.append(i)
    return out, out1


_load5 = Q4.load_all4


def load_all6(trains, reads, max_len, t0, AU):
    Q, part, G = _load5(trains, reads, max_len, t0, AU)
    if not HOLD:
        return Q, part, G
    g = HOLD["graph"]
    if g not in trains:
        raise SystemExit(f"--hold {g}: the held graph must be a training graph")
    ranks, names = resolve(list(G[g]["vocab"]), HOLD["spec"])
    HOLD.update(ranks=ranks.tolist(), names=names)
    rows = sorted({i for (gg, _lk), rr in part.items() if gg == g for i in rr})
    B = part[(g, "x1")][1::2]
    for i in B:
        STATE["full"][i] = Q[i]
    L = Q4.LOADED.get(g, max_len)
    hr, hr1 = held_rows(Q, B, ranks, L)
    STATE["held_rows"] = {"held_rows": hr, "held_rows_1edge": hr1}
    n_e, n_rows = 0, 0
    for i in rows:
        Q[i], k = (drop_row if HOLD["mode"] == "drop" else mask_row)(Q[i], ranks)
        n_e += k
        n_rows += int(k > 0)
    if sum(int(held_edges(Q[i], ranks).sum()) for i in rows):
        raise SystemExit("a held edge survived the change")
    STATE["stats"] = {"rows": len(rows), "rows_touched": n_rows, "held_edges": n_e, "read_rows": len(B), "held_rows": len(hr),
                      "held_rows_1edge": len(hr1), "loaded_len": L}
    log(f"hold {g} {HOLD['mode']} {names} (ranks {ranks.tolist()}): {STATE['stats']}")
    return Q, part, G


# ── the extra reads ──────────────────────────────────────────────────────────


def views_for(arm, rows):
    """The extra views of the held graph's read rows: name -> (Q list, TOK list, every label 'other')."""
    full, ranks = STATE["full"], np.asarray(HOLD["ranks"], dtype=np.int64)
    QF, TF = list(arm.Q), list(arm.TOK)
    for i in rows:
        QF[i] = full[i]
        if i not in STATE["tok"]:
            STATE["tok"][i] = QG.edge_labels(full[i])
        TF[i] = STATE["tok"][i]
    if arm.sp["kind"] == "T0":
        return {"FULL": (QF, TF, False)} if HOLD["mode"] == "drop" else {}
    v = {"REV": (QF, TF, False)}
    if HOLD["mode"] == "drop":
        QM, TM = list(arm.Q), list(arm.TOK)
        for i in rows:
            QM[i] = mask_row(full[i], ranks)[0]
            TM[i] = QG.edge_labels(QM[i])
        v["MASK"] = (QM, TM, False)
    v["FNR"] = (QF, TF, True)
    return v


def score_view(arm, model, g, rows, Qv, Tv, nr):
    keep = (arm.Q, arm.TOK, arm.nr, getattr(arm, "cache", None))
    arm.Q, arm.TOK, arm.nr = Qv, Tv, nr
    if keep[3] is not None:
        arm.cache = {}
    try:
        return Q5._score_rows(arm, model, g, rows)
    finally:
        arm.Q, arm.TOK, arm.nr = keep[0], keep[1], keep[2]
        if keep[3] is not None:
            arm.cache = keep[3]


def hold_reads(arm, model, g, rows, SZ):
    AU = sys.modules["anchor_univ"]
    AG, AW3 = AU.AG, AU.AW3
    mo = arm.A16.metrics_of
    name = Q5.NAMES.get(id(arm.sp), "?")
    if sorted(rows) != sorted(STATE["full"]):
        raise SystemExit("the held graph's ID read is not on its read rows")
    R = AG.Reader(arm.Q, rows, QG.READ_SEED)
    base = R.base()
    pos = {i: j for j, i in enumerate(rows)}
    rules = (("np", None), ("p", 0.0), ("mp", STATE["margin"]))
    mID = {rl: Q2.metrics_at(SZ, rows, arm.Q, mg, mo) for rl, mg in rules}
    k = len(RES.get(name, []))
    ent = {"margin": str(STATE["margin"]), "all": {}, "held": {t: {} for t in STATE["held_rows"]}}
    for rl in ("np", "mp"):
        ent["all"][f"ID/{rl}"] = Q2.with_abs(R.record(mID[rl], by_type=False), base)
        for tag, hr in STATE["held_rows"].items():
            ent["held"][tag][f"ID/{rl}"] = R.subset(mID[rl], hr)
    for vn, (Qv, Tv, nr) in views_for(arm, rows).items():
        SZv = score_view(arm, model, g, rows, Qv, Tv, nr)
        for rl, mg in rules:
            m = Q2.metrics_at(SZv, rows, arm.Q, mg, mo)
            ROWS[f"{name}|{k}|{vn}|{rl}"] = m
            ent["all"][f"{vn}/{rl}"] = Q2.with_abs(R.record(m, by_type=False), base)
            ent["all"][f"{vn} - ID/{rl}"] = AW3.boot_pair(m - mID[rl], R.W)
            for tag, hr in STATE["held_rows"].items():
                hx = np.asarray([pos[i] for i in hr], dtype=np.int64)
                ent["held"][tag][f"{vn}/{rl}"] = R.subset(m, hr)
                if hx.size:
                    ent["held"][tag][f"{vn} - ID/{rl}"] = AW3.boot_pair((m - mID[rl])[hx], R.W[:, hx])
        del SZv
    for rl in ("np", "mp"):
        ROWS[f"{name}|{k}|ID|{rl}"] = mID[rl]
    RES.setdefault(name, []).append(ent)
    hh = ent["held"].get("held_rows", {})
    log(f"hold {name}#{k} {g}: " + "; ".join(f"{key} rho {v['rho (R@5, FC@5, hit@1)']}" for key, v in ent["all"].items()
                                           if " - " not in key and key.endswith("/np")) +
        " | held rows: " + "; ".join(f"{key} rho {v.get('rho (R@5, FC@5, hit@1)')}" for key, v in hh.items()
                                     if " - " not in key and key.endswith("/np")))


_score5 = Q5.score_rows
_choose5 = Q5.choose_margin


def score_rows(arm, model, g, rows, TY=None):
    SZ = _score5(arm, model, g, rows, TY)
    if (HOLD and g == HOLD["graph"] and Q5.PHASE["now"] == "read" and TY is None and not getattr(arm, "nr", False)
            and not getattr(arm, "shuf", False)):
        try:
            hold_reads(arm, model, g, rows, SZ)
        except Exception as ex:  # noqa: BLE001 -- the run's own reads stand; the failure is recorded and logged
            RES.setdefault("_failed", []).append({"arm": Q5.NAMES.get(id(arm.sp), "?"), "error": repr(ex)})
            log(f"hold reads failed: {ex!r}")
    return SZ


def choose_margin(*args, **kwargs):
    out = _choose5(*args, **kwargs)
    STATE["margin"] = out[0]
    return out


_parse5 = Q5.parse_arms


def parse_arms(spec, names, rule, AU, AG, G3):
    out = _parse5(spec, names, rule, AU, AG, G3)
    if HOLD:
        bad = [nm for nm, sp in out.items() if sp.get("family") != "qd"]
        if bad:
            raise SystemExit(f"--hold: {bad} refused (anchor_univ compiles a walk arm's types outside the arm)")
    return out


_bind5 = Q5.bind


def bind():
    _bind5()
    Q2.parse_arms = parse_arms
    Q2.load_all = load_all6
    Q2.score_rows = score_rows
    Q2.choose_margin = choose_margin


def parse_hold(s):
    m = HOLDRE.match(s)
    if not m or m.group(1) not in Q2.READ:
        raise SystemExit(f"--hold {s!r}: <graph>:<drop|mask>:<names;...|frac<F>s<S>>, graph one of {list(Q2.READ)}")
    return {"graph": m.group(1), "mode": m.group(2), "spec": m.group(3)}


def hold_record():
    """The hold, its counts and the extra reads (each held-row record carries the twin's and the GNN's means there)."""
    return {**HOLD, "stats": STATE["stats"], "held_rows": {k: len(v) for k, v in STATE["held_rows"].items()}, "reads": RES}


def finish(out):
    res = json.loads(out.read_text(encoding="utf-8"))
    res["look"] = "qd_gnn6"
    res["pins"]["qd_gnn5"] = SHAS["qd_gnn5"]
    res["qd_gnn6_sha256"] = SHAS["qd_gnn6"]
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")


# ── selftest ─────────────────────────────────────────────────────────────────


def synthetic_kb(n_rows, rng, K):
    Q = QG.synthetic_rows(n_rows, rng, K)
    for j, q in enumerate(Q):
        q["metrics"] = rng.random((6, 14))
        q["type"] = ("a", "b")[j % 2]
        q["pool"] = np.arange(q["n"])
        q["proj"] = np.zeros((q["n"], 0), dtype=np.float32)
    return Q


def selftest():
    import tempfile
    sys.path.insert(0, str(QG.AU_DIR))
    sys.path.insert(0, str(QG.KB_DIR))
    import anchor_univ as AU
    A16 = AU.A16
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(2)
    rng = np.random.default_rng(0)
    K = 8
    ranks = np.asarray([1, 5], dtype=np.int64)
    # the row changes: drop removes exactly the held edges, mask shows them as 'other'; both keep the rest as it was
    for q in synthetic_kb(20, rng, K):
        d0, a0 = QG.edge_labels(q)
        hit = np.isin(a0, ranks)
        qd, nd = drop_row(q, ranks)
        d1, a1 = QG.edge_labels(qd)
        assert nd == int(hit.sum()) and np.array_equal(d1, d0[~hit]) and np.array_equal(a1, a0[~hit])
        assert qd["w2_f"].size == int((qd["fam"] == 0).sum()) and not held_edges(qd, ranks).any()
        assert all(np.array_equal(qd[k], q[k][~hit]) for k in EDGE_KEYS)
        qm, nm = mask_row(q, ranks)
        d2, a2 = QG.edge_labels(qm)
        assert nm == nd and np.array_equal(d2, d0) and np.array_equal(a2 >= K, (a0 >= K) | hit)
        assert np.array_equal(np.where(hit, -9, a2), np.where(hit, -9, a0)) and not held_edges(qm, ranks).any()
        assert all(np.array_equal(qm[k], q[k]) for k in EDGE_KEYS)
    print("selftest: drop deletes exactly the held edges, mask shows exactly them as 'other'")
    # the held rows on a constructed graph: 0 -(held)-> 1 -> 2, 0 -> 3
    base = {"n": 4, "u": np.asarray([0, 1, 0]), "v": np.asarray([1, 2, 3]), "fam": np.zeros(3, dtype=np.int64),
            "fwd": np.ones(3, dtype=bool), "bwd": np.zeros(3, dtype=bool), "w": np.ones(3, dtype=np.float32),
            "w2_f": np.asarray([5, 0, 0], dtype=np.int32), "w2_b": np.full(3, -1, dtype=np.int32),
            "seeds": np.asarray([0, -1]), "bucket": np.zeros(2, dtype=np.int64)}
    rows = []
    for gold_at in (2, 3, 1):
        q = dict(base)
        q["gold"] = np.zeros(4, dtype=bool)
        q["gold"][gold_at] = True
        rows.append(q)
    hr, hr1 = held_rows(rows, [0, 1, 2], ranks, 2)
    assert hr == [0, 2] and hr1 == [2], (hr, hr1)
    assert held_rows(rows, [0, 1, 2], ranks, 1) == ([2], [2])
    print("selftest: held rows are the rows a walk through a held edge reaches a gold (and the one-edge subset)")
    # spec resolution
    vocab = [f"r{j}" for j in range(300)]
    assert resolve(vocab, "r3;r1")[0].tolist() == [1, 3]
    kbv = ["directed by", "has genre", "x_y", "written by"]
    assert resolve(kbv, "has_genre")[0].tolist() == [1] and resolve(kbv, "written_by;x_y")[0].tolist() == [2, 3]
    assert resolve(kbv, "has genre")[1] == ["has genre"]
    fr = resolve(vocab, "frac0.1s7")[0]
    assert fr.size == 26 and fr.max() < 256 and np.array_equal(fr, resolve(vocab, "frac0.1s7")[0])
    for bad in ("r300", "r1;r1", "nope"):
        try:
            resolve(vocab, bad)
        except SystemExit:
            continue
        raise AssertionError(bad)
    # QD-ID under the held view: a held rank's vector gets no gradient
    Qs = synthetic_kb(24, rng, K)
    Qd = [drop_row(q, ranks)[0] for q in Qs]
    TOKd = [QG.edge_labels(q) for q in Qd]
    z_of = [A16.zscore(q["score"][:, 0]) for q in Qs]
    m = QG.QD("ID", 16, 2, K)
    torch.nn.init.normal_(m.wo.weight, 0.0, 0.5)    # w_o starts at zero, which would stop every gradient upstream
    P = QG.pack_qd(list(range(24)), Qd, TOKd, z_of, K)
    s, gold, z = QG.padded(P, m(P))
    loss = Q2.step_loss2(s, gold, z, "both")
    loss.backward()
    g_id = m.E_id.grad.abs().sum(1)
    assert float(g_id[ranks].max()) == 0.0 and float(g_id.max()) > 0, g_id
    print("selftest: under the held view a held rank's ID vector gets no gradient")
    # the full loop on a synthetic KB through qd_gnn2.run: hold reads for QD, W3 and W4 arms, in both modes
    names = [f"rel {j}" for j in range(K)]
    phi = rng.standard_normal((QG.R_TOP, QG.TDIM)).astype(np.float32)
    real = Q4._load_all

    def fake(trains, reads, max_len, t0, AU_):
        Qf = synthetic_kb(60, np.random.default_rng(5), K)
        part = {("metaqa", "x1"): list(range(0, 20)), ("metaqa", "select"): list(range(20, 32)),
                ("metaqa", "fit"): list(range(32, 60))}
        G = {"metaqa": {"kind": "kb", "role": "train", "phi": phi, "phi_sha": "synthetic", "n_rel": K, "vocab": names,
                        "info": {"pruned_loader": {"synthetic": True}}}}
        return Qf, part, G

    def call_main(argv):
        """main() as the command line runs it, its rebinding of qd_gnn2.run and qd_gnn5.bind undone afterwards."""
        keep = (Q2.run, Q5.bind, sys.argv)
        sys.argv = ["qd_gnn6.py"] + argv
        try:
            main()
        finally:
            Q2.run, Q5.bind, sys.argv = keep

    Q4._load_all = fake
    bind()
    tmp = Path(tempfile.mkdtemp(prefix="qd6_"))
    try:
        for mode in ("drop", "mask"):
            HOLD.clear()
            for key in ("full", "tok", "held_rows"):
                STATE[key] = {}
            STATE["out"] = None
            RES.clear()
            ROWS.clear()
            out = tmp / f"h_{mode}.json"
            arms = "QD-T0-L2-d16,QD-TXT-L2-d16-K8-dr25e10,QD-ID-L2-d16-K8,W3-TXT-L2-d16-K8,W4-T0-L2-d16"
            call_main(["--train", "metaqa=fit", "--arms", arms, "--rule", "both", "--epochs", "2", "--out", str(out),
                       "--hold", f"metaqa:{mode}:rel 1;rel 5"])
            res = json.loads(out.read_text(encoding="utf-8"))
            assert HOLD["ranks"] == [1, 5] and STATE["stats"]["held_edges"] > 0, STATE["stats"]
            want = {"QD-T0-L2-d16": {"FULL"} if mode == "drop" else set(),
                    "QD-TXT-L2-d16-K8-dr25e10": {"REV", "MASK", "FNR"} if mode == "drop" else {"REV", "FNR"},
                    "QD-ID-L2-d16-K8": {"REV", "MASK", "FNR"} if mode == "drop" else {"REV", "FNR"},
                    "W3-TXT-L2-d16-K8": {"REV", "MASK", "FNR"} if mode == "drop" else {"REV", "FNR"},
                    "W4-T0-L2-d16": {"FULL"} if mode == "drop" else set()}
            assert "_failed" not in RES, RES.get("_failed")
            for nm, vs in want.items():
                ent = RES[nm][0]
                got = {k.split("/")[0] for k in ent["all"] if " - " not in k} - {"ID"}
                assert got == vs, (mode, nm, got, vs)
                # the ID read here is the run's own ID/np read of the held view
                run_np = res["results"][nm]["seeds_read"]["0"]["reads"]["metaqa"]["ID/np"]["fit"]
                assert ent["all"]["ID/np"]["fit"] == run_np, (nm, ent["all"]["ID/np"]["fit"], run_np)
            # a view of the full rows read through the swap equals a fresh arm built on the full rows
            nm = "W3-TXT-L2-d16-K8"
            sp = Q4.parse_w3(nm)
            Qfull = fake(None, None, None, None, None)[0]
            rows = list(range(0, 20))[1::2]
            TOKf = [QG.edge_labels(q) for q in Qfull]
            zf = [A16.zscore(q["score"][:, 0]) for q in Qfull]
            torch.manual_seed(0)
            arm_f = QG.QDArm(sp, Qfull, TOKf, {"metaqa": {"phi": phi}}, zf, A16, 0)
            mdl = arm_f.make()
            arm_f.prepare(mdl, "metaqa")
            ref = Q5._score_rows(arm_f, mdl, "metaqa", rows)
            Qh = [(drop_row if mode == "drop" else mask_row)(q, ranks)[0] for q in Qfull]
            arm_h = QG.QDArm(sp, Qh, [QG.edge_labels(q) for q in Qh], {"metaqa": {"phi": phi}}, zf, A16, 0)
            arm_h.prepare(mdl, "metaqa")
            held_SZ = Q5._score_rows(arm_h, mdl, "metaqa", rows)
            cache = arm_h.cache
            STATE["full"] = {i: Qfull[i] for i in rows}
            STATE["tok"] = {}
            QF, TF, _nr = views_for(arm_h, rows)["REV"]
            got = score_view(arm_h, mdl, "metaqa", rows, QF, TF, False)
            assert all(np.array_equal(a[0], b[0]) for a, b in zip(got, ref)), "the swapped view differs from the full rows"
            assert arm_h.cache is cache and arm_h.Q is Qh, "the held view was not restored"
            again = Q5._score_rows(arm_h, mdl, "metaqa", rows)
            assert all(np.array_equal(a[0], b[0]) for a, b in zip(again, held_SZ)), "the held view's read changed"
            print(f"selftest {mode}: hold reads for every arm ({sorted(want)}), held rows {res['hold']['held_rows']}, "
                  f"stats {res['hold']['stats']}; the view swap reads the full rows and restores the held view")
            assert res["look"] == "qd_gnn6" and res["qd_gnn6_sha256"] == SHAS["qd_gnn6"]
            assert Path(str(out.with_suffix("")) + "_hold.npz").exists()
        # walk arms are refused, a held graph must be a training graph
        HOLD.clear()
        HOLD.update(parse_hold("metaqa:drop:rel 1"))
        AU.check_pins()
        AU.AC.bind()
        try:
            Q2.parse_arms("QD-T0-L2,W:T0", ["metaqa"], "both", AU, AU.AG, AU.G3)
        except SystemExit:
            pass
        else:
            raise AssertionError("a walk arm passed under --hold")
        for bad in ("metaqa:cut:x", "nograph:drop:x", "metaqa:drop"):
            try:
                parse_hold(bad)
            except SystemExit:
                continue
            raise AssertionError(bad)
    finally:
        Q4._load_all = real
        HOLD.clear()
    print("selftest: all checks passed")


def main():
    bind()
    if "--selftest" in sys.argv:
        selftest()
        return
    if "--hold" in sys.argv:
        k = sys.argv.index("--hold")
        HOLD.update(parse_hold(sys.argv[k + 1]))
        del sys.argv[k:k + 2]
    _run2 = Q2.run

    def run(a):
        STATE["out"] = Path(a.out)
        _run2(a)
        out = Path(a.out)
        res = json.loads(out.read_text(encoding="utf-8"))
        res["hold"] = hold_record() if HOLD else None
        out.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
        if ROWS:
            np.savez(str(out.with_suffix("")) + "_hold.npz", **{k.replace("|", "__"): v for k, v in ROWS.items()})

    Q2.run = run
    Q5.bind = bind    # qd_gnn5.main binds first (its --cap-len / --base, its output fields), then runs Q2.main
    Q5.main()
    if STATE["out"] is not None:
        finish(STATE["out"])


if __name__ == "__main__":
    main()
