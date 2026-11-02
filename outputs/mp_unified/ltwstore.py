"""LTW amendment A1 (docs/LTW_LEARNED_NEXT_HOP.md section 11): does the three-way term survive on the store the models
read? LTW's joint fit is refit with the scorer's vectors replaced by their coordinates on basis_2wiki's 256 axes (the
int8 store of lean_mlp3, decoded; both WQ splits' models read this basis), and read on LTW's three s1eval carves.

    candidate and anchor rows   x -> s * codes(x) + c       (lean_mlp3.Store.codes, then .decode's first 256 entries)
    question                    q -> V'q                    (lean_mlp3.Store.query's first 256 entries)
    residual                    r -> unit(V'q - c_a x_a')   (c_a the anchor's full cosine, x_a' its store coordinates)

Everything else is LTW's: the lists (LTW's case files, read only), the 14 scalars (from the full rows), the forms, the
seeds, the settings, the fit and the read. The rule (section 11, fixed before any read), on the three s1eval A reads:
    K = mean(T2 store.bil - T2 joint.lin) / mean(T2 joint.bil - T2 joint.lin), joint.* from section 10's record;
    STORE_KEEPS when K >= 0.5 and musique's store.bil T2 is at least its joint.lin T2; else STORE_LOSES.

    python outputs/mp_unified/ltwstore.py run --host
    python outputs/mp_unified/ltwstore.py --selftest
"""
import os
import sys

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import contextlib  # noqa: E402
import json  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (ROOT / "src", ROOT / "scripts", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402

import ltwdiag as LT  # noqa: E402  (LTW: cases, fit, read_eval, the forms; imported, never edited)

OUT = LT.OUT / "store"
NAME = "store"
BASIS = "2wiki"                                  # zwq: both WQ splits read basis_2wiki
REF = LT.OUT / "report.json"                     # section 10's record: joint.lin and joint.bil on the same lists
K_MIN = 0.5


def log(msg):
    LT.log(msg)


class StoreMap:
    """lean_mlp3.Store's coordinates on however many axes the basis holds: the int8 codes decoded, and V'q."""

    def __init__(self, basis):
        self.m, self.V = basis["m"].astype(np.float32), basis["V"].astype(np.float32)
        self.s = (4.0 * np.sqrt(np.maximum(basis["w"], 1e-12)) / 127.0).astype(np.float32)
        self.c = self.m @ self.V

    def rows(self, X):
        codes = np.clip(np.rint(((np.asarray(X, np.float32) - self.m) @ self.V) / self.s), -127, 127)
        return codes.astype(np.int8).astype(np.float32) * self.s + self.c

    def query(self, q):
        return np.asarray(q, np.float32) @ self.V


def to_store(b, sm):
    """One read_chunk's vectors on the store's axes; its scalars, free scores and identity check stay LTW's."""
    ca = np.sum(b["qv"] * b["xa"], 1, dtype=np.float32)          # the anchor's cosine, as LTW's r uses it
    qk, xa = sm.query(b["qv"]), sm.rows(b["xa"])
    out = dict(b)
    out["X16"] = sm.rows(b["X16"].astype(np.float32)).astype(np.float16)
    out["qv"], out["xa"] = qk.astype(np.float32), xa.astype(np.float32)
    out["rv"] = LT.unit(qk - ca[:, None] * xa)
    return out


@contextlib.contextmanager
def on_store(sm, cases_out=LT.OUT):
    """LTW's read_chunk with to_store after it, FITS with the store fit (the joint fit's datasets), and LTW's own case
    files whatever out a call names; restored on the way out."""
    rc, lc, fits = LT.read_chunk, LT.load_cases, dict(LT.FITS)

    def read_chunk(z, pre, nd, sel, idx):
        b, chk = rc(z, pre, nd, sel, idx)
        return to_store(b, sm), chk

    LT.read_chunk = read_chunk
    LT.load_cases = lambda out, ds, carve: lc(cases_out, ds, carve)
    LT.FITS = {NAME: fits[LT.DECIDE]}
    try:
        yield
    finally:
        LT.read_chunk, LT.load_cases, LT.FITS = rc, lc, fits


def verdict(reads, ref):
    """Section 11's rule against section 10's joint fit, and the lin repeat beside (store.lin reads no vector)."""
    rows = {}
    for ds, r in reads.items():
        a, f = r["A"]["scores"], ref["reads"][ds]["A"]["scores"]
        rows[ds] = {"joint.lin": f["joint.lin"]["T2"], "joint.bil": f["joint.bil"]["T2"],
                    "store.lin": a[f"{NAME}.lin"]["T2"], "store.bil": a[f"{NAME}.bil"]["T2"],
                    "H1_joint.bil": ref["reads"][ds]["H"]["scores"]["H1_joint.bil"]["R@5"],
                    "H1_store.bil": r["H"]["scores"][f"H1_{NAME}.bil"]["R@5"],
                    "rrf_top5": r["H"]["scores"]["rrf_top5"]["R@5"]}
    g_full = float(np.mean([x["joint.bil"] - x["joint.lin"] for x in rows.values()]))
    g_store = float(np.mean([x["store.bil"] - x["joint.lin"] for x in rows.values()]))
    k = g_store / g_full if g_full > 0 else None
    m = rows[LT.MAIN]
    keeps = k is not None and k >= K_MIN and m["store.bil"] >= m["joint.lin"]
    return {"verdict": "STORE_KEEPS" if keeps else "STORE_LOSES", "K": k, "K_min": K_MIN, "gain_full": g_full,
            "gain_store": g_store, "rows": rows,
            "lin_repeat_max_abs": max(abs(x["store.lin"] - x["joint.lin"]) for x in rows.values())}


def report_md(rep):
    v = rep["verdict"]
    k = "-" if v["K"] is None else f"{v['K']:.3f}"
    L = [f"# LTW A1: the three-way term on the store (declared in {rep['declared_in']} section 11)", "",
         f"**{v['verdict']}**: K = {k} (the store keeps this share of the vector term's mean T2 gain over "
         f"joint.lin; {v['gain_store']:+.3f} against {v['gain_full']:+.3f}); the bar is {v['K_min']}.", "",
         "| read | joint.lin | joint.bil | store.lin | store.bil | rrf top 5 | H1 joint.bil | H1 store.bil |",
         "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for ds, x in v["rows"].items():
        L.append(f"| {ds} | {x['joint.lin']:.3f} | {x['joint.bil']:.3f} | {x['store.lin']:.3f} | {x['store.bil']:.3f} "
                 f"| {x['rrf_top5']:.4f} | {x['H1_joint.bil']:.4f} | {x['H1_store.bil']:.4f} |")
    f = rep["fit"]
    L += ["", f"Fit: decision form {f['decision_form']}; " +
          "; ".join(f"{k} held-out {f['forms'][k]['best_val_T2']} (epoch {f['forms'][k]['best_epoch']} of "
                    f"{f['forms'][k]['epochs_run']}, {f['forms'][k]['parameters']} parameters)" for k in LT.FORMS),
          f"Basis: {rep['basis']['graph']} ({rep['basis']['rows']} nodes, {rep['basis']['explained']:.3f} of the variance "
          f"on {rep['basis']['axes']} axes, npz sha256 {rep['basis']['npz_sha256'][:12]}).",
          f"store.lin repeats joint.lin to within {v['lin_repeat_max_abs']:.4f} T2."]
    return "\n".join(L) + "\n"


def run(out=OUT, nodes=None, basis=None, brec=None, ref=REF, cases_out=LT.OUT, epochs=LT.EPOCHS, placement=None,
        rec_sha=None):
    t0 = time.time()
    ref_rec = json.loads(Path(ref).read_text(encoding="utf-8"))
    if ref_rec.get("verdict", {}).get("verdict") not in ("SIGNAL", "WEAK", "NO_SIGNAL"):
        raise SystemExit(f"ltw store: {ref} holds no filed LTW verdict")
    if basis is None:
        import lean_cache as LC
        basis, brec = LC.load_basis(BASIS, LT.PKO / "cache")
    sm = StoreMap(basis)
    if nodes is None:
        import lean_mlp3 as L3
        nodes, rec_sha = L3.open_nodes(list(LT.FITS[LT.DECIDE]))
    with on_store(sm, cases_out):
        fit = LT.fit(NAME, out=out, nodes=nodes, epochs=epochs, placement=placement, rec_sha=rec_sha)
        M, _dec = LT.load_fits(out, [NAME])
        reads = {}
        for ds, cv in LT.EVAL:
            reads[ds] = LT.read_eval(ds, cv, LT.load_cases(out, ds, cv), nodes[ds], M)
            LT.write_json(Path(out) / f"{ds}__{cv}.read.json", reads[ds])
    ident = all(r["identity"]["pass"] for r in reads.values()) and fit["identity"]["pass"]
    rep = {"declared_in": "docs/LTW_LEARNED_NEXT_HOP.md", "identity_pass": ident,
           "verdict": verdict(reads, ref_rec) if ident else {"verdict": "IDENTITY_FAIL"},
           "fit": {k: fit[k] for k in ("decision_form", "carves", "forms")},
           "basis": {"graph": BASIS, "rows": basis.get("rows"), "explained": basis.get("explained"),
                     "axes": int(sm.V.shape[1]), "npz_sha256": (brec or {}).get("npz_sha256", "")},
           "reference": {"path": "outputs/ltw/report.json", "sha256": LT.CH.sha_src(ref) if Path(ref).is_file() else None},
           "reads": reads, "package_record_sha256": rec_sha, "placement": placement,
           "script_sha256": LT.CH.sha_src(__file__), "ltwdiag_sha256": LT.CH.sha_src(LT.__file__),
           "seconds": time.time() - t0, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    LT.write_json(Path(out) / "report.json", rep)
    if ident:
        (Path(out) / "report.md").write_text(report_md(rep), encoding="utf-8")
    log(f"ltw store: {rep['verdict'].get('verdict')} K {rep['verdict'].get('K')}")
    return rep


# ── self-test ────────────────────────────────────────────────────────────────


def selftest():
    rng = np.random.default_rng(11)
    # 1. StoreMap is lean_mlp3.Store on a full-size basis (1536 x 256), codes and query alike
    import lean_mlp3 as L3
    Q_, _r = np.linalg.qr(rng.normal(size=(1536, 256)))
    basis = {"m": (0.01 * rng.normal(size=1536)).astype(np.float32), "V": Q_.astype(np.float32),
             "w": np.sort(rng.random(256))[::-1].astype(np.float32) * 0.01 + 1e-4}
    st, sm = L3.Store(basis), StoreMap(basis)
    X = LT.unit(rng.normal(size=(64, 1536)))
    assert np.array_equal(sm.rows(X), st.decode(st.codes(X))[:, :L3.STORE_K])
    q = LT.unit(rng.normal(size=1536))
    assert np.allclose(sm.query(q[None])[0], st.query(q)[:L3.STORE_K], atol=1e-6)
    # 2. to_store: shapes, the residual's definition, and the scalars untouched
    b = {"qv": LT.unit(rng.normal(size=(3, 1536))), "xa": LT.unit(rng.normal(size=(3, 1536))),
         "X16": LT.unit(rng.normal(size=(7, 1536))).astype(np.float16), "F": rng.random((7, len(LT.FEATS))),
         "rv": np.zeros((3, 1536), np.float32), "free": [{}] * 3, "sel": np.arange(3), "off": np.array([0, 2, 5, 7])}
    o = to_store(b, sm)
    assert o["X16"].shape == (7, 256) and o["X16"].dtype == np.float16 and o["qv"].shape == o["xa"].shape == (3, 256)
    ca = np.sum(b["qv"] * b["xa"], 1)
    want = LT.unit(sm.query(b["qv"]) - ca[:, None] * sm.rows(b["xa"]))
    assert np.allclose(o["rv"], want) and np.allclose(np.linalg.norm(o["rv"], axis=1), 1, atol=1e-5)
    assert o["F"] is b["F"] and o["free"] is b["free"] and b["X16"].shape == (7, 1536)
    # 3. on_store restores LTW's functions, FITS included
    rc, lc, fits = LT.read_chunk, LT.load_cases, dict(LT.FITS)
    with on_store(sm):
        assert LT.read_chunk is not rc and LT.FITS == {NAME: fits[LT.DECIDE]}
    assert LT.read_chunk is rc and LT.load_cases is lc and LT.FITS == fits
    # 4. end to end on LTW's toy PKO trees (d = 16, a toy basis of 8 axes): fit, reads, verdict, records
    with tempfile.TemporaryDirectory() as td:
        root, lout, sout = Path(td) / "pko", Path(td) / "ltw", Path(td) / "ltw" / "store"
        M = LT.unit(rng.normal(size=(500, 16)))
        for ds, cv in LT.FIT_CARVES + LT.EVAL:
            LT.toy_tree(root, ds, cv, M, rng)
        for ds, cv in LT.FIT_CARVES + LT.EVAL:
            LT.cases(ds, cv, root, lout)
        nodes = {ds: LT.FakeNodes(M) for ds in LT.DS_K}
        LT.fit("joint", root, lout, nodes, epochs=2)
        LT.fit("musique", root, lout, nodes, epochs=2)
        LT.report(root, lout, nodes)
        Qt, _r = np.linalg.qr(rng.normal(size=(16, 8)))
        tb = {"m": np.zeros(16, np.float32), "V": Qt.astype(np.float32), "w": np.full(8, 0.06, np.float32),
              "rows": 500, "explained": 0.5}
        rep = run(sout, nodes, tb, {"npz_sha256": "0" * 64}, lout / "report.json", lout, epochs=2)
        assert rep["identity_pass"] and rep["verdict"]["verdict"] in ("STORE_KEEPS", "STORE_LOSES")
        assert rep["fit"]["forms"]["bil"]["parameters"] == 1569 + 2 * 3 * 8 * LT.RANK
        assert rep["verdict"]["lin_repeat_max_abs"] < 1e-9, rep["verdict"]["lin_repeat_max_abs"]
        for ds, cv in LT.EVAL:
            assert (sout / f"{ds}__{cv}.read.json").exists()
            assert set(rep["reads"][ds]["A"]["scores"]) == set(LT.FREE) | {f"{NAME}.{f}" for f in LT.FORMS}
        assert (sout / "report.md").exists() and (sout / f"fit_{NAME}.json").exists()
        assert not (lout / f"fit_{NAME}.json").exists() and (lout / "work" / "cases").is_dir()
    assert LT.FITS == fits
    log("ltw store selftest: PASS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", choices=("run",))
    ap.add_argument("--host", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return 0
    if a.cmd == "run":
        placement = None
        if a.host:
            import lean_host as LH
            placement = LH.substitute()      # the verified mirror in place of the package, as ltwdiag --host
        rep = run(placement=placement)
        return 0 if rep["identity_pass"] else 1
    ap.error("a command or --selftest")


if __name__ == "__main__":
    sys.exit(main())
