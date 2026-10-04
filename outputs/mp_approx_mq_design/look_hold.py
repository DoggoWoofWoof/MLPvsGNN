"""Design look after level 15's read (2c246c1), look_ul and look_ula, on r2: the 11,920 metaqa train-split rows level 15
read once, now spent, so this is design only. It is not a result and is not filed. Level 15's fit module and every module
it imports are imported unchanged; level 15's units, r2's sidecar and level 12's carve sidecars are read in place and
never written. This look's units go to its own directory.

The question (the user, 2026-10-02): does the typed-walk model work for a relation it never trained on, and is it safe
for a relation whose name means nothing? Level 15's FZ-TW-1x (level 9's NB-hyb: e(tau) = E[s(tau)] + sum_pos
Rot(P e_r + delta_d, (pos + 1) theta), FZ's posterior, level 10's set likelihood) is refitted with one relation r* held
out of training and selection, and read on r2 with r*'s walks in each of three states:

  protocol RU (relation unseen)  every fit and select row whose true chain (named by its qtype) uses r* leaves the fit
                                 and inner sets (it is given no in-pool gold, which is level 8's rule for a row outside
                                 them), and every walk type with an r* token (3 r* + d, d fwd / bwd / both) is removed
                                 from every fit and select row. Nothing in training or selection reads r*'s edges, its
                                 rows or its text row, so E has no gradient on an r* sequence and P never sees e_r*.
  protocol QU (questions unseen) the rows leave as under RU, but the walks stay: r*'s edges are seen in training only as
                                 distractors of other questions (the usual zero-shot-relation split of KBQA).
  protocol NONE                  nothing held: the control, which must reproduce level 15's unit bit for bit (the view
                                 here gathers every entry through this file's per-type reader).

  eval DROP  r2's r* walk types removed (what the model can do without the relation's edges at all);
  eval REV   r2's walks as they are: an r* type is scored by its text code alone (E is 0 on its sequence), the
             zero-shot reading;
  eval NR    as REV, with r*'s text row replaced by a random vector of the same norm (a relation whose name carries
             nothing: the non-existent-relation reading). Training is the same as REV's: no training row has an r* type.

Arms (k 0): RU-DROP, RU-REV, RU-NR and QU-REV for each held relation, and the NONE control; level 15's own unit (every
relation trained) is ID. Read: rho_bar against the twin and the GNN (level 8's quantity) on the held rows (true chain
uses r*), the other rows and all rows, the paired rho_bar contrasts and the per-metric differences in points, the chain
agreement and the posterior mass the model puts on r*'s types, the true chain's types and the null type.

Every learned quantity is a function of the query embedding and a discrete walk type, applied once to reach sets
compiled before any fit. No row of r2 is fitted.

    python outputs/mp_approx_mq_design/look_hold.py unit --protocol RU --rel written_by --mode REV --k 0 --host
    python outputs/mp_approx_mq_design/look_hold.py unit --protocol NONE --mode REV --k 0 --host
    python outputs/mp_approx_mq_design/look_hold.py read          # -> look_hold.json
"""
import os
import sys

if __name__ == "__main__":   # level 15's fit pools: 4 threads, fixed before numpy and torch load
    for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[_v] = "4"

import argparse  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (ROOT / "src", ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import torch  # noqa: E402

import mp_approx_l15_fit as F15  # noqa: E402  (level 15's fit module, imported unchanged)

P, L0, L8, L9, F11, F12 = F15.P, F15.L0, F15.L8, F15.L9, F15.F11, F15.F12
SEEDS, FUNCS, METRIC_NAMES, RETRIEVAL, HOPS = F15.SEEDS, F15.FUNCS, F15.METRIC_NAMES, F15.RETRIEVAL, F15.HOPS
FOLD, FIT = F15.FOLD, "TW-1x"
ORIGINAL_VIEW = L9.View
RELS = ("written_by", "directed_by", "starred_actors", "has_genre", "in_language", "release_year")
PROTOCOLS = {"NONE": (False, False), "RU": (True, True), "QU": (False, True)}   # (r* types out of training, r* rows out)
MODES = ("DROP", "REV", "NR")
ARMS = (("RU", "DROP"), ("RU", "REV"), ("RU", "NR"), ("QU", "REV"))
NR_SEED = 7100
CHECK_ROWS = 400            # rows per part whose gathered entries are checked against the stored block
UNITS = HERE / "units_hold"
OUT_JSON = HERE / "look_hold.json"
CONTRASTS = {"rev_minus_drop": ("RU-REV", "RU-DROP"), "nr_minus_drop": ("RU-NR", "RU-DROP"), "rev_minus_nr": ("RU-REV", "RU-NR"),
             "id_minus_rev": ("ID", "RU-REV"), "id_minus_drop": ("ID", "RU-DROP"), "qu_minus_ru": ("QU-REV", "RU-REV"),
             "id_minus_qu": ("ID", "QU-REV"), "ctrl_minus_id": ("CTRL", "ID")}


def log(msg):
    print(f"[{L0.utc()}] {msg}", flush=True)


# ── the held relation's view ─────────────────────────────────────────────────


def held_tokens(r) -> np.ndarray:
    return np.zeros(0, dtype=np.int64) if r is None else np.arange(L8.N_DIR * r, L8.N_DIR * (r + 1), dtype=np.int64)


def chain_uses(qtype: str, r) -> bool:
    return r is not None and any(rr == r for rr, _d in L8.true_chain(qtype))


def type_hits(codes, r) -> np.ndarray:
    """Per type code: does its token sequence hold an r* token (any direction)."""
    _b, toks, _n = L8.decode_types(codes)
    return np.isin(toks, held_tokens(r)).any(1) if toks.size else np.zeros(0, dtype=bool)


def filter_layout(q_types, t_size, keep) -> tuple[np.ndarray, np.ndarray]:
    """Per query, the type and entry counts after keeping the type rows flagged in keep."""
    n_q = int(np.asarray(q_types).size)
    owner = np.repeat(np.arange(n_q, dtype=np.int64), np.asarray(q_types, dtype=np.int64))[keep]
    types = np.bincount(owner, minlength=n_q).astype(np.int64)
    entries = np.zeros(n_q, dtype=np.int64)
    np.add.at(entries, owner, np.asarray(t_size, dtype=np.int64)[keep])
    return types, entries


def gather_index(first, size) -> np.ndarray:
    """The entry rows of a run of types, type by type: first[i] .. first[i] + size[i] - 1."""
    first, size = np.asarray(first, dtype=np.int64), np.asarray(size, dtype=np.int64)
    if size.size == 0 or int(size.sum()) == 0:
        return np.zeros(0, dtype=np.int64)
    ends = np.cumsum(size)
    return np.arange(int(ends[-1]), dtype=np.int64) - np.repeat(ends - size, size) + np.repeat(first, size)


def make_hold_view(r, drop_types: bool, drop_rows: bool):
    """Level 9's View of one family, then (drop_types) without its r* walk types and (drop_rows) with every row whose true
    chain uses r* given no in-pool gold, so level 8's unit_queries keeps it out of the fit and inner sets. Its entries are
    read type by type from the stored arrays (in place), so a type removed from the middle of a block leaves no gap."""

    class HoldView(ORIGINAL_VIEW):
        hold = {"rel": None if r is None else L8.REL_ORDER[r], "drop_types": bool(drop_types), "drop_rows": bool(drop_rows)}

        def __init__(self, d, family, check=True):
            super().__init__(d, family, check)
            code = np.asarray(self.t_code, dtype=np.int64)
            hit = type_hits(code, r)
            keep = ~hit if drop_types else np.ones(code.size, dtype=bool)
            old = {"type_ptr": self.type_ptr.copy(), "entry_start": self.entry_start.copy(), "q_entries": self.q_entries.copy(),
                   "t_size": np.asarray(self.t_size).copy()}
            self.t_code = code[keep]
            self.t_size = np.asarray(self.t_size)[keep]
            self.t_gold = np.asarray(self.t_gold)[keep]
            self.t_first = np.asarray(self.t_first)[keep]
            self.q_types, self.q_entries = filter_layout(old["type_ptr"][1:] - old["type_ptr"][:-1], old["t_size"], keep)
            self.type_ptr = np.r_[0, np.cumsum(self.q_types)].astype(np.int64)
            self.entry_start = None           # entries() gathers type by type
            qts = self.meta["qtypes"]
            uses = np.asarray([chain_uses(qts[int(i)], r) for i in self.q_qtype], dtype=bool)
            gold_before = int((np.asarray(self.q_gold_in_pool) > 0).sum())
            if drop_rows:
                self.q_gold_in_pool = np.where(uses, 0, self.q_gold_in_pool).astype(np.asarray(self.q_gold_in_pool).dtype)
            self.hold_counts = {"rows": int(self.n_q), "rows_true_chain_uses_rel": int(uses.sum()),
                                "rows_with_gold_before": gold_before, "rows_with_gold_after": int((np.asarray(self.q_gold_in_pool) > 0).sum()),
                                "types_before": int(code.size), "types_with_rel": int(hit.sum()), "types_after": int(self.t_code.size),
                                "entries_before": int(old["q_entries"].sum()), "entries_after": int(self.q_entries.sum())}
            self.check_entries(old, keep)

        def entries(self, q: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            rows = self.type_rows(q)
            size = np.asarray(self.t_size[rows], dtype=np.int64)
            idx = gather_index(self.t_first[rows], size)
            tl = np.repeat(np.arange(rows.stop - rows.start, dtype=np.int64), size)
            return np.asarray(self.e_node[idx], dtype=np.int64), np.asarray(self.e_count[idx], dtype=np.int64), tl

        def check_entries(self, old: dict, keep: np.ndarray) -> None:
            """The first CHECK_ROWS rows: the gathered entries are the stored block's entries of the kept types, in order,
            with the type-local index renumbered over the kept types."""
            for q in range(min(CHECK_ROWS, self.n_q)):
                a, b = int(old["type_ptr"][q]), int(old["type_ptr"][q + 1])
                s = slice(int(old["entry_start"][q]), int(old["entry_start"][q] + old["q_entries"][q]))
                tl_old = np.repeat(np.arange(b - a, dtype=np.int64), old["t_size"][a:b].astype(np.int64))
                kq = keep[a:b]
                m = kq[tl_old]
                renum = np.cumsum(kq) - 1
                want = (np.asarray(self.e_node[s], dtype=np.int64)[m], np.asarray(self.e_count[s], dtype=np.int64)[m], renum[tl_old[m]])
                got = self.entries(q)
                if not all(np.array_equal(x, y) for x, y in zip(got, want)):
                    raise SystemExit(f"{self.dir}: row {q}'s gathered entries are not its stored block's kept entries")

    return HoldView


def hold_view(decl: dict, r, protocol: str, mode: str):
    """Level 15's full deploy view for TW-1x, built as level 15's deploy_view builds it (level 11's MultiView of r2 and
    level 12's part_view of the select and fit carves, level 12's roles), with the parts read through this file's views:
    the training parts under the protocol, r2 without r*'s types under DROP and as it is otherwise."""
    drop_t, drop_r = PROTOCOLS[protocol]
    names = F12.part_names(FIT)
    train = make_hold_view(r, drop_t, drop_r)
    ev = make_hold_view(r, mode == "DROP", False)
    with F11.rebound(L9, View=ev):
        eval_part = F15.eval_view(decl)
    with F11.rebound(L9, View=train):
        parts = [F12.part_view(decl, p) for p in names[1:]]
    merged = F12.with_roles(F11.MultiView([eval_part, *parts], names))
    return merged, {p: dict(v.hold_counts, **v.hold) for p, v in zip(names, [eval_part, *parts])}


def nr_rel_emb(rel_emb: np.ndarray, r: int) -> np.ndarray:
    out = np.array(rel_emb, dtype=np.float32, copy=True)
    g = np.random.default_rng(NR_SEED + r).standard_normal(out.shape[1])
    out[r] = (g * (float(np.linalg.norm(rel_emb[r])) / float(np.linalg.norm(g)))).astype(np.float32)
    return out


# ── a unit ───────────────────────────────────────────────────────────────────


def unit_paths(protocol: str, rel, mode: str, k: int) -> tuple[Path, Path]:
    d = UNITS / protocol / (rel or "none") / mode
    return d / f"k{k}.npz", d / f"k{k}.json"


def setup(host: bool) -> dict:
    P.HARD_STOP_DIR[0] = HERE   # a hard stop of this look lands here, never in level 15's outputs
    P.route_stops()
    decl = P.load_declaration()
    if host:
        L8.host_mode(decl, log)
    return decl


def masses(data, cap: dict, r) -> dict:
    """Per scored row, from the unit's capture (FZ's posterior before any score rule): the mass on the row's r* types, on
    its true chain's types (both buckets) and on the null type, and whether the true chain is among the row's types."""
    qts = data.meta["qtypes"]
    held = held_tokens(r)
    out = {key: np.zeros(len(cap["score_q"])) for key in ("mass_held", "mass_true", "mass_null")}
    out["true_present"] = np.zeros(len(cap["score_q"]), dtype=bool)
    for i, (q, lp) in enumerate(zip(cap["score_q"], cap["lp_score"])):
        q = int(q)
        t = int(data.q_types[q])
        codes = np.asarray(data.t_code[data.type_rows(q)], dtype=np.int64)
        _b, toks, _n = L8.decode_types(codes)
        chain = L8.chain_tokens(L8.true_chain(qts[int(data.q_qtype[q])]))
        pad = np.full(L8.MAX_L, -1, dtype=np.int64)
        pad[:len(chain)] = chain
        is_true = (toks == pad[None, :]).all(1) if t else np.zeros(0, dtype=bool)
        hit = np.isin(toks, held).any(1) if t else np.zeros(0, dtype=bool)
        p = np.exp(np.asarray(lp[:t], dtype=np.float64))
        out["mass_held"][i], out["mass_true"][i] = float(p[hit].sum()), float(p[is_true].sum())
        out["mass_null"][i] = float(np.exp(lp[t]))
        out["true_present"][i] = bool(is_true.any())
    return out


def stage_unit(decl: dict, protocol: str, rel, mode: str, k: int) -> None:
    t0 = time.time()
    L8.fit_process()
    P.HARD_STOP_DIR[0] = HERE
    P.route_stops()
    P.verify_inputs(decl)
    if protocol == "NONE":
        rel = None
        if mode != "REV":
            raise SystemExit("NONE: the control is read under REV only")
    elif rel not in RELS:
        raise SystemExit(f"{protocol}: --rel one of {RELS}")
    npz, js = unit_paths(protocol, rel, mode, k)
    if js.exists():
        log(f"{js} exists")
        return
    r = None if rel is None else L8.REL_ORDER.index(rel)
    dv, counts = hold_view(decl, r, protocol, mode)
    emb = L8.load_rel_emb(decl)
    if mode == "NR":
        emb = nr_rel_emb(emb, r)
    fx = F15.make_fitter("fz", dv, emb)
    log(f"{protocol} {rel} {mode} k{k}: {dv.n_q} queries in the deploy view ({', '.join(f'{p} {v.n_q}' for p, v in zip(dv.sources, dv.parts))}), "
        f"{fx.table.codes.size} nb walk types, V {int(fx.fz_vocab.sum())}; holds {counts}")
    kept = {}
    orig = F11.fit_and_capture

    def keep(fx_, k_, fold_, log_=print):
        a_, f_, c_ = orig(fx_, k_, fold_, log_)
        kept["cap"] = c_
        return a_, f_, c_

    with F11.rebound(F11, fit_and_capture=keep):
        arrays, flog = F15.run_unit(fx, "fz", "full", FIT, k, log)
    cap = kept["cap"]
    if not np.array_equal(cap["score_q"], arrays["q"]):
        raise SystemExit(f"{protocol} {rel} {mode} k{k}: the capture's scored rows are not the unit's")
    ms = masses(fx.data, cap, r)
    ctrl = None
    if protocol == "NONE":
        ref_npz, _ref_js = F15.unit_paths("fz", "full", FIT, k)
        with np.load(ref_npz) as z:
            common = sorted(set(z.files) & set(arrays))
            differ = [key for key in common if not (z[key].dtype == np.asarray(arrays[key]).dtype and np.array_equal(z[key], arrays[key]))]
            ctrl = {"level15_unit": str(ref_npz.relative_to(ROOT)), "compared": common, "differ": differ,
                    "only_level15": sorted(set(z.files) - set(arrays)), "only_here": sorted(set(arrays) - set(z.files))}
        log(f"NONE k{k}: against level 15's unit, {len(common)} arrays compared, differing {differ}")
    out = {key: arrays[key] for key in ("q", "argmax", *[f"metrics_{s}" for s in ("dens", "cov", "dsh", "b1d")])}
    out.update(ms)
    npz.parent.mkdir(parents=True, exist_ok=True)
    tmp = npz.with_name(npz.stem + ".tmp.npz")
    np.savez(tmp, **out)
    for attempt in range(8):
        try:
            os.replace(tmp, npz)
            break
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(5)
    keep_keys = ("kept_round", "rounds", "beta_dsh", "kappa_dsh", "eta_dsh", "beta_b1d", "kappa_b1d", "eta_b1d", "inner_mean3_dsh",
                 "inner_mean3_b1d", "b1d_mass_moved", "em_not_monotone_rounds", "fit_queries", "inner_queries", "scored_queries",
                 "rows_by_part", "unseen_sequences", *F15.FZ_KEYS)
    L8.write_json(js, {"protocol": protocol, "rel": rel, "mode": mode, "k": k, "holds": counts,
                       "nr_seed": (NR_SEED + r) if mode == "NR" else None, "fit": {key: flog.get(key) for key in keep_keys},
                       "control": ctrl, "script_sha256": L0.sha256_file(Path(__file__)), "arrays_sha256": L0.sha256_file(npz),
                       **L8.job_fields(t0)})
    log(f"{protocol} {rel} {mode} k{k}: filed in {time.time() - t0:.0f} s")


# ── read ─────────────────────────────────────────────────────────────────────


def point_diff(A: np.ndarray, B: np.ndarray, W: np.ndarray, sel: np.ndarray) -> dict:
    """Per metric, the mean over the selected rows of A - B (seeds averaged), in points, with its 95% interval."""
    out = {}
    for i, m in enumerate(RETRIEVAL):
        d = (A[:, :, i] - B[:, :, i]).mean(1)
        with np.errstate(all="ignore"):
            boot = (W[:, sel] @ d[sel]) / W[:, sel].sum(1)
        lo, hi = L0.ci(boot)
        out[m] = [round(100 * float(d[sel].mean()), 3), [round(100 * lo, 3), round(100 * hi, 3)]] if sel.any() else None
    return out


def stage_read(ks=(0,)) -> None:
    t0 = time.time()
    P.HARD_STOP_DIR[0] = HERE
    P.route_stops()
    base = L9.View(F15.DATA, "std")
    n = base.n_q
    ri = [METRIC_NAMES.index(m) for m in RETRIEVAL]
    qm = base.q_metrics
    T = qm[:, [FUNCS.index(f"twin{k}") for k in ks]][:, :, ri]
    G = qm[:, [FUNCS.index(f"gnn{k}") for k in ks]][:, :, ri]
    hop = np.asarray(base.q_hop)
    qts = base.meta["qtypes"]
    qt = [qts[int(i)] for i in base.q_qtype]
    truth = [tuple(L8.chain_tokens(L8.true_chain(x))) for x in qt]
    W = L0.boot_weights(n)

    def load(paths, level15=False):
        M, A, extra, logs = np.zeros((n, len(ks), len(RETRIEVAL))), np.zeros((n, len(ks)), dtype=np.int64), {}, []
        for i, (npz, js) in enumerate(paths):
            flog = L8.read_json(js)
            if L0.sha256_file(npz) != flog["arrays_sha256"]:
                raise SystemExit(f"{npz}: not the arrays its log records")
            with np.load(npz) as z:
                if not np.array_equal(z["q"], np.arange(n)):
                    raise SystemExit(f"{npz}: not r2's rows, each once")
                M[:, i] = z["metrics_b1d"][:, ri]
                A[:, i] = z["argmax"]
                if not level15:
                    for key in ("mass_held", "mass_true", "mass_null", "true_present"):
                        extra.setdefault(key, []).append(np.asarray(z[key]))
            logs.append(flog)
        return M, A, {key: np.stack(v, 1) for key, v in extra.items()}, logs

    def agreement(A: np.ndarray, sel: np.ndarray) -> float | None:
        if not sel.any():
            return None
        right = np.asarray([[L8.token_sequence(int(A[q, i])) == truth[q] for i in range(A.shape[1])] for q in np.flatnonzero(sel)])
        return round(float(right.mean()), 4)

    common = {}
    common["ID"] = load([F15.unit_paths("fz", "full", FIT, k) for k in ks], level15=True)
    ctrl = [unit_paths("NONE", None, "REV", k) for k in ks]
    if all(js.exists() for _n, js in ctrl):
        common["CTRL"] = load(ctrl)
    out = {"look": "look_hold", "rows": n, "seeds": list(ks), "rels": {}, "script_sha256": L0.sha256_file(Path(__file__))}
    if "CTRL" in common:
        out["control"] = {f"k{k}": lg.get("control") for k, lg in zip(ks, common["CTRL"][3])}
    for rel in RELS:
        r = L8.REL_ORDER.index(rel)
        arms = dict(common)
        for protocol, mode in ARMS:
            got = [unit_paths(protocol, rel, mode, k) for k in ks]
            if all(js.exists() for _n, js in got):
                arms[f"{protocol}-{mode}"] = load(got)
        if len(arms) <= len(common):
            log(f"{rel}: no unit filed; skipped")
            continue
        uses = np.asarray([chain_uses(x, r) for x in qt], dtype=bool)
        sels = {"held": uses, "other": ~uses, "all": np.ones(n, dtype=bool), **{f"held_hop{h}": uses & (hop == h) for h in HOPS}}
        rec = {"held_rows": int(uses.sum()), "held_by_hop": {f"hop={h}": int((uses & (hop == h)).sum()) for h in HOPS},
               "units": {a: [{key: lg.get(key) for key in ("holds", "nr_seed", "seconds", "peak_rss_bytes")} | {"fit": {key: lg["fit"].get(key) for key in (
                   "kept_round", "beta_b1d", "kappa_b1d", "eta_b1d", "inner_mean3_b1d", "fit_queries", "inner_queries", "unseen_sequences",
                   "fz_vocab", "fz_scored_rows_outside_v")}} for lg in v[3]] for a, v in arms.items() if a not in ("ID",)}, "strata": {}}
        for name, sel in sels.items():
            if not sel.any():
                continue
            den, dens, readable = L8.denominators(T, G, W, sel)
            A = {a: F15.read_arm(v[0], T, G, dens, readable, W, sel) for a, v in arms.items()}
            st = {"rows": int(sel.sum()), "readable": readable, "gnn_minus_twin": {m: den[m]["gap"] for m in RETRIEVAL},
                  "rho_bar": {a: e["rho_bar"] for a, e in A.items()},
                  "mean": {a: {m: round(100 * e["mean"][m]["arm"], 3) for m in RETRIEVAL} for a, e in A.items()},
                  "twin": {m: round(100 * next(iter(A.values()))["mean"][m]["twin"], 3) for m in RETRIEVAL},
                  "gnn": {m: round(100 * next(iter(A.values()))["mean"][m]["gnn"], 3) for m in RETRIEVAL},
                  "contrasts": {}, "points": {}, "agreement": {a: agreement(v[1], sel) for a, v in arms.items()},
                  "posterior": {a: {key: round(float(v[2][key][sel].mean()), 4) for key in v[2]} for a, v in arms.items() if v[2]}}
            for c, (a, b) in CONTRASTS.items():
                if a in A and b in A:
                    st["contrasts"][c] = F15.paired(A, a, b, readable)
                    st["points"][c] = point_diff(arms[a][0], arms[b][0], W, sel)
            rec["strata"][name] = st
        out["rels"][rel] = rec
    out["seconds"] = round(time.time() - t0, 1)
    L8.write_json(OUT_JSON, out)

    def f3(x, sign=True):
        return "  n/a " if x is None or not np.isfinite(x) else (f"{x:+.3f}" if sign else f"{x:.3f}")

    for rel, rec in out["rels"].items():
        log(f"== {rel}: held rows {rec['held_rows']} {rec['held_by_hop']}")
        for name in ("held", "other", "all"):
            st = rec["strata"].get(name)
            if not st:
                continue
            rb = "  ".join(f"{a} {f3(v['point'])}" for a, v in st["rho_bar"].items())
            log(f"   {name:5s} ({st['rows']} rows, readable {st['readable']}): rho_bar {rb}")
            for c, v in st["contrasts"].items():
                pts = st["points"][c]
                ci = v["ci"] or [None, None]
                log(f"      {c:15s} rho_bar {f3(v['point'])} [{f3(ci[0])}, {f3(ci[1])}]  pts " +
                    "  ".join(f"{m.split('@')[0][:2]} {p[0]:+.2f}[{p[1][0]:+.2f},{p[1][1]:+.2f}]" for m, p in pts.items() if p))
            log(f"      agreement {st['agreement']}  posterior {st['posterior']}")
    log(f"done in {out['seconds']} s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("unit", "read"))
    ap.add_argument("--protocol", choices=tuple(PROTOCOLS))
    ap.add_argument("--rel", choices=RELS)
    ap.add_argument("--mode", choices=MODES)
    ap.add_argument("--k", type=int, choices=SEEDS)
    ap.add_argument("--ks", default="0")
    ap.add_argument("--host", action="store_true")
    a = ap.parse_args()
    if a.stage == "unit":
        if a.protocol is None or a.mode is None or a.k is None:
            raise SystemExit("unit: --protocol, --mode and --k (and --rel unless NONE)")
        stage_unit(setup(a.host), a.protocol, a.rel, a.mode, a.k)
    else:
        setup(False)
        torch.set_num_threads(L8.FIT_THREADS)
        stage_read(tuple(int(x) for x in a.ks.split(",")))


if __name__ == "__main__":
    main()
