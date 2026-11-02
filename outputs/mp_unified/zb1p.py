"""B1P (docs/B1P_PKO_CHAIN_ON_B1.md): PKO's pools chain on B1's three 1,000-question settings (HippoRAG 2's), read by
ML1's GNN (zg6r) and MLP (zrckp) at their three seeds on both WQ splits. No LLM and no corpus hyperlink anywhere.

Two carves of each setting, from B1f's variant folders:
  b1w  U2b's W2_S rule cut to the setting's passages (B1f's hipporag2_w): the Table 1 read
  b1t  no structural family, the setting's text links only (B1f's hipporag2_t): the floor; b1w - b1t is the carry-over

    python outputs/mp_unified/zb1p.py --selftest
    python outputs/mp_unified/zb1p.py settings [--host]                 # the cut graphs and restricted types, checked
    python outputs/mp_unified/zb1p.py pools --dataset D --variant w|t [--limit Q] [--host] [--root DIR]
    python outputs/mp_unified/zb1p.py chain --dataset D --variant w|t [--host] [--root DIR]
    python outputs/mp_unified/zb1p.py step KIND --dataset D --variant V [--index i/n] [--host] [--root DIR]
    python outputs/mp_unified/zb1p.py gate [--host]
    python outputs/mp_unified/zb1p.py fork
    python outputs/mp_unified/zb1p.py read --name FIT --variant w|t      # the host's GPU (ML1's device check)
    python outputs/mp_unified/zb1p.py report
    python outputs/mp_unified/zb1p.py look-dev --dataset D --variant V --root DIR [--questions 8]   # laptop

pools  PKO's pools pass (zpko.pools_pass, one shard) on the setting: zu2d_pools.open_carve opens the package's dataset
       with every store, graph, population, first stage and name index swapped for the setting's in process (on_b1),
       the U2s record's graph and the U2m (a) / U2p types re-targeted to the cut graph (each type array restricted to
       the cut's entries, in the cut's order). Every B, I and expansion row is checked to lie inside the setting.
chain  the rest of PKO's chain on the carve, each step its own process (STEPS): merge, look shards (look_u2d, its
       population and first stage B1's), lean_cache's bases and parts, links, KT, the carve's record.
gate   zpko.gate on the six carves with PKO's own section 5 checks (outputs/pko/check): outputs/b1p/gate.json.
fork   ML1's twelve fits (zg6r, zrckp x seeds 0-2 x WQ-musique, WQ-hotpotqa) copied to outputs/b1p/fits/<v>/<fit>
       without their reads; ML1's folders are never written.
read   ML1's read of a fork (zml1s.run_arm for zg6r, zml1s.read's body for zrckp, no stamp) on outputs/b1p's roots
       and gate, with --read <setting>=<carve> for the three settings and --out-root outputs/b1p/fits/<v>.
report R@5 per setting, arm, carve and fit; the in-domain seed means against the best published rows (b1f.PUBLISHED)
       by the 95% bootstrap interval, the paired b1w - b1t difference, the zero-shot reads beside, and each setting's
       questions checked against the training carves' question ids. outputs/b1p/report.{json,md}.
"""
import os
import sys

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "2"                                  # BLAS at 2, as in every PKO process (zpko section 4)
os.environ.setdefault("NUMBA_NUM_THREADS", "2")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")   # ML1's reads (zml1s.where_ok)

from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (ROOT / "scripts", ROOT / "src", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import argparse  # noqa: E402
import contextlib  # noqa: E402
import copy  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import numpy as np  # noqa: E402

DECLARED = "docs/B1P_PKO_CHAIN_ON_B1.md"
STAGE = "B1P"
OUT = ROOT / "outputs" / "b1p"
SETTINGS = ("musique", "2wiki", "hotpotqa")
VARIANTS = ("w", "t")
CARVE = {"w": "b1w", "t": "b1t"}
GRAPH_TEXT = {"w": "B1P b1w: U2b's W2_S rule cut to the setting's passages",
              "t": "B1P b1t: no structural family (the setting's text links only)"}
ARMS = ("zg6r", "zrckp")
SEEDS = (0, 1, 2)
SPLITS = ("WQ-musique", "WQ-hotpotqa")
SUF = {"WQ-musique": "", "WQ-hotpotqa": "-hp"}               # zwq.SUF
IN_DOMAIN = {"musique": ("WQ-hotpotqa",), "hotpotqa": ("WQ-musique",), "2wiki": ("WQ-musique", "WQ-hotpotqa")}
TRAINS = {"WQ-musique": (("webqsp", "fit"), ("squad", "fit"), ("hotpotqa", "fit"), ("2wiki", "fit")),
          "WQ-hotpotqa": (("webqsp", "fit"), ("squad", "fit"), ("musique", "s1fit"), ("2wiki", "fit"))}   # zwq FIT_CARVE
FITS = tuple(f"scr-ml1-{arm}-s{seed}{SUF[split]}" for arm in ARMS for seed in SEEDS for split in SPLITS)
SCREEN_FITS = ROOT / "outputs" / "screen" / "fits"
PKO_KT = ROOT / "outputs" / "pko" / "kt"
PYLIB_RUN = ROOT / "outputs" / "host_ops" / "pylib_run.py"
STEPS = ("pools", "merge", "look", "bases", "cache", "links", "kt", "record")
NO_STEP1 = {"state": "not applicable: B1's settings have no step-1 cache (B1P, docs/B1P_PKO_CHAIN_ON_B1.md)"}


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def sha_src(p):
    return hashlib.sha256(Path(p).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def sha_arr(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def rel(p):
    p = Path(p).resolve()
    try:
        return p.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return p.as_posix()


def read_json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def write_json(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, p)


class Patches:
    """Attributes set in process and put back in reverse order (restore)."""

    def __init__(self):
        self.saved = []

    def set(self, obj, **kw):
        for k, v in kw.items():
            if not hasattr(obj, k):
                raise SystemExit(f"B1P: {getattr(obj, '__name__', type(obj).__name__)} has no attribute {k}")
            self.saved.append((obj, k, getattr(obj, k)))
            setattr(obj, k, v)

    def save(self, obj, *names):
        for k in names:
            if not hasattr(obj, k):
                raise SystemExit(f"B1P: {getattr(obj, '__name__', type(obj).__name__)} has no attribute {k}")
            self.saved.append((obj, k, getattr(obj, k)))

    def restore(self):
        while self.saved:
            obj, k, v = self.saved.pop()
            setattr(obj, k, v)


# ── modules (the host's mirror goes in before any of them loads) ──────────────

_M = {}


def mods(host=False):
    if _M:
        if host != _M["ns"].host:
            raise SystemExit("B1P: --host must be given before any module loads")
        return _M["ns"]
    mirror = None
    if host:
        import zpkoh
        mirror = zpkoh.apply_mirror()                    # S6.host_mode + step 4's package root (PKOH's, unchanged)
    import zpko as PK                                    # numba; PKO's pools pass, chain, records and gate
    import p1e_anchor_kind as K                          # S4, E (step 4e), D, G: what zu2d_pools.open_carve opens with
    import look_b1 as LB                                 # B1's setting loader, stores, first stage and population
    import look_b1u as LBU                               # before LB.BENCH moves: it keeps B1a's folder for the lists
    import b1f as B1F
    import look_u2d as LU
    LBU.VARIANTS["w"] = B1F.VAR_W                        # look_b1w's one line
    if LU.U2C is not PK.ZP.U2C or LU.LX is not LB.LX:
        raise SystemExit("B1P: look_u2d and zu2d_pools do not share zu2c and look_x_six")
    ns = SimpleNamespace(PK=PK, K=K, S4=K.S4, E=K.E, ZP=PK.ZP, U2C=PK.ZP.U2C, LB=LB, LBU=LBU, B1F=B1F, LU=LU, LX=LB.LX,
                         AT=PK.AT, P=PK.P, host=host, mirror=mirror)
    _M["ns"] = ns
    return ns


def var_dir(M, v):
    return Path(M.B1F.VAR_W) if v == "w" else Path(M.LBU.VARIANTS["t"])


def load_variant_setting(M, name, v):
    """B1a's setting files from the variant's folder (look_b1.load_setting, its build.json checks)."""
    P_ = Patches()
    try:
        P_.set(M.LB, BENCH=var_dir(M, v))
        st = M.LB.load_setting(name)
    finally:
        P_.restore()
    if (st.build.get("u1a") or {}).get("variant") != v:
        raise SystemExit(f"B1P: {var_dir(M, v)}/{name} is not B1f's variant {v}")
    return st


# ── the cut graph and the restricted types (settings) ────────────────────────

def cut_csr(ip, col, inside):
    """The symmetric CSR (ip, col) kept on rows and columns inside the mask: (ip, col int32, map), where map gives each
    kept entry's index in the original entries, so an array over the original entries restricted by map is in the
    cut's entry order (rows stay sorted)."""
    ip = np.asarray(ip, np.int64)
    keep = np.repeat(inside, np.diff(ip)) & inside[col]
    ck = np.concatenate([[0], np.cumsum(keep, dtype=np.int64)])
    return ck[ip], np.ascontiguousarray(np.asarray(col)[keep], np.int32), np.flatnonzero(keep).astype(np.int64)


def pair_keys(n, a, b):
    a, b = np.asarray(a, np.int64), np.asarray(b, np.int64)
    m = a != b
    lo, hi = np.minimum(a[m], b[m]), np.maximum(a[m], b[m])
    return np.unique(lo * np.int64(n) + hi)


def csr_keys(ip, col):
    n = int(np.asarray(ip).size - 1)
    heads = np.repeat(np.arange(n, dtype=np.int64), np.diff(np.asarray(ip, np.int64)))
    return pair_keys(n, heads, col)


def settings_paths(name, v):
    d = OUT / "settings" / v
    return d / f"{name}.npz", d / f"{name}.json"


def settings(host=False):
    M = mods(host)
    op = M.S4.Opened()
    for name in SETTINGS:
        if all(settings_paths(name, v)[1].exists() for v in VARIANTS):
            log(f"settings {name}: filed")
            continue
        n = int(op.canonical.Dataset(name, root=str(op.served)).n_nodes)
        ip, col, open_desc = M.U2C.graph_csr(name, "chosen", n)
        open_desc = json.loads(json.dumps(open_desc, default=str))
        tarr, arec = M.AT.load_types(name, open_desc)
        cls_s, _zbar, srec = M.P.load_types(name, open_desc)
        if not (np.asarray(tarr["cls"]).size == np.asarray(cls_s).size == np.asarray(col).size):
            raise SystemExit(f"B1P {name}: the types do not cover the open graph's entries")
        for v in VARIANTS:
            npz, js = settings_paths(name, v)
            if js.exists():
                continue
            st = load_variant_setting(M, name, v)
            inside = np.zeros(n, bool)
            inside[st.urows] = True
            if v == "w":
                ip_b, col_b, mp = cut_csr(ip, col, inside)
            else:
                ip_b, col_b, mp = np.zeros(n + 1, np.int64), np.zeros(0, np.int32), np.zeros(0, np.int64)
            g = st.graphs["structural"]
            want = pair_keys(n, st.rows[g["src"].astype(np.int64)], st.rows[g["dst"].astype(np.int64)])
            got = csr_keys(ip_b, col_b)
            if not np.array_equal(want, got):
                raise SystemExit(f"B1P {name}/{v}: the cut has {got.size} pairs, not B1f's {want.size} of the variant")
            if col_b.size != 2 * got.size or int(ip_b[-1]) != col_b.size or (v == "t" and want.size):
                raise SystemExit(f"B1P {name}/{v}: the cut is not a symmetric CSR without repeats")
            if col_b.size and not inside[col_b].all():
                raise SystemExit(f"B1P {name}/{v}: an entry outside the setting")
            em_b = np.ascontiguousarray(np.asarray(tarr["cls"])[mp])
            sc_b = np.ascontiguousarray(np.asarray(cls_s)[mp])
            npz.parent.mkdir(parents=True, exist_ok=True)
            tmp = npz.with_name(npz.stem + f".{os.getpid()}.tmp.npz")
            np.savez(tmp, ip=ip_b, col=col_b, map=mp)
            os.replace(tmp, npz)
            sha = sha_file(npz)
            desc_b1 = {"graph": GRAPH_TEXT[v], "setting": name, "variant": v, "settings_npz": rel(npz),
                       "settings_npz_sha256": sha, "open": open_desc}
            vd = var_dir(M, v) / name
            write_json(js, {
                "declared_in": DECLARED, "stage": STAGE, "dataset": name, "variant": v, "carve": CARVE[v], "n_nodes": n,
                "setting": {"dir": rel(vd), "build_json_sha256": sha_file(vd / "build.json"), "shas": st.shas,
                            "passages": int(st.rows.size), "distinct_passages": int(st.urows.size),
                            "questions": len(st.queries)},
                "open": open_desc, "npz": rel(npz), "npz_sha256": sha, "map_sha256": sha_arr(mp),
                "edges": {"entries": int(col_b.size), "pairs": int(got.size), "b1f_pairs": int(want.size),
                          "open_entries": int(np.asarray(col).size)},
                "types": {"em": {"file_sha256": arec["file_sha256"], "cls_b1_sha256": sha_arr(em_b), "entries": int(mp.size)},
                          "score": {"file_sha256": srec["file_sha256"], "cls_b1_sha256": sha_arr(sc_b),
                                    "entries": int(mp.size), "entries_null": int((sc_b == M.P.NULL).sum())}},
                "desc_b1": desc_b1, "script_sha256": sha_src(__file__), "utc": utc()})
            log(f"settings {name}/{v}: {got.size} pairs ({col_b.size} entries of {np.asarray(col).size}), "
                f"{st.urows.size} distinct passages")
        del ip, col, tarr, cls_s
    return 0


def load_settings(name, v):
    npz, js = settings_paths(name, v)
    if not js.exists():
        raise SystemExit(f"B1P: {rel(js)} missing (zb1p.py settings)")
    sj = read_json(js)
    if sha_file(npz) != sj["npz_sha256"]:
        raise SystemExit(f"B1P: {rel(npz)} is not the file {js.name} names")
    with np.load(npz) as z:
        ip, col, mp = z["ip"].astype(np.int64), z["col"].astype(np.int32), z["map"].astype(np.int64)
    if sha_arr(mp) != sj["map_sha256"]:
        raise SystemExit(f"B1P: {rel(npz)}'s map is not its record's")
    return sj, ip, col, mp


# ── the setting in process (on_b1) ────────────────────────────────────────────

@contextlib.contextmanager
def checks(M):
    """PKO's section 5 checks (outputs/pko/check) stand for B1P's carves: zpko.need_checks and checks_state read PKO's
    own roots. On the laptop (a development run) need_checks only says so: those records are the host's."""
    PK = M.PK
    P_ = Patches()
    orig_need, orig_state = PK.need_checks, PK.checks_state

    def need(_R):
        if M.host:
            return orig_need(PK.roots())
        log("laptop development run: PKO's section 5 checks are the host's records; not checked here")

    P_.set(PK, need_checks=need, checks_state=lambda _R: orig_state(PK.roots()))
    try:
        yield
    finally:
        P_.restore()


def make_prepare(M, m3c, st, name, carve, stats, memo, inside):
    """look_b1's prepare (its first stage, base rows, seeds and expansion, unchanged in substance) for any stores dict:
    zu2d_pools.open_carve prepares twice (the package's stores, then with the swapped graph); the first stage is
    computed once per population."""
    LB = M.LB

    def prepare(ds_, pops, construction, cfg_h, stores, m3a, m3b_contract):
        if len(pops) != 1 or pops[0].kind != carve or pops[0].dataset != name:
            raise SystemExit("B1P prepare: not the setting's one population")
        pop = pops[0]
        t = time.time()
        key = tuple(int(r) for r in pop.idx)
        if key not in memo:
            pos = {int(r): j for j, r in enumerate(st.qrows)}
            qi = np.asarray([pos[int(r)] for r in pop.idx], dtype=np.int64)
            Xd = np.asarray(ds_.embeddings("dense", "docs").read(st.rows), dtype=np.float32)
            qemb = np.asarray(ds_.embeddings("dense", "queries").read(pop.idx), dtype=np.float32)
            d_ids, d_sc, d_changed, d_swap = LB.top_lists(st, st.first["dense"][qi], qemb, Xd, "dense")
            del Xd
            if LB.splade_docs(ds_):
                Xs = ds_.embeddings("splade", "docs").read(st.rows)
                Qs = ds_.embeddings("splade", "queries").read(pop.idx)
                s_ids, s_sc, s_changed, s_swap = LB.top_lists(st, st.first["splade"][qi], Qs.astype(np.float32),
                                                              Xs.astype(np.float32), "splade")
                del Xs, Qs
                lists = "computed here"
            else:
                lz = LB.load_lists(st)
                s_ids, s_sc = lz["splade_ids"][qi], lz["splade_sc"][qi]
                s_changed = lz["record"]["splade_lists_with_repeats_dropped"]
                s_swap = lz["record"]["splade_lists_with_near_tie_swaps"]
                ties = LB.same_up_to_ties(d_ids, d_sc, lz["dense_ids"][qi], lz["dense_sc"][qi])
                d_ids, d_sc = lz["dense_ids"][qi], lz["dense_sc"][qi]
                lists = {"from": "b1_lists.npz (the laptop's, from the package)", "sha256": lz["record"]["sha256"],
                         "dense_lists_here_differing_by_near_ties": ties}
            memo[key] = (d_ids, d_sc, s_ids, s_sc, qemb)
            stats["first_stage"] = {"lists": lists, "distinct_passages": int(st.urows.size), "passages": int(st.rows.size),
                                    "dense_lists_with_repeats_dropped": d_changed,
                                    "splade_lists_with_repeats_dropped": s_changed,
                                    "dense_lists_with_near_tie_swaps": d_swap, "splade_lists_with_near_tie_swaps": s_swap,
                                    "seconds": round(time.time() - t, 1)}
        d_ids, d_sc, s_ids, s_sc, qemb = memo[key]
        constant = int(cfg_h["retrieval_pools"]["equal_rrf"]["constant"])
        regime = construction["regime"]
        fams_ = [stores[f] for f in m3c.regime_families(cfg_h, regime)] if regime != "RETRIEVAL" else None
        setting = construction.get("setting")
        mp = m3c.m3b_pools
        seeds = [mp.seeds_of(d, s) for d, s in zip(d_ids, s_ids)]
        bases = m3c.base_rows(construction, d_ids, s_ids, m3a, m3b_contract, constant)
        pools, added = [], []
        for base, seed in zip(bases, seeds):
            expansion = mp.expand_hops(seed, fams_, setting) if fams_ is not None else None
            pool, n_added = m3c.build_pool(np.asarray(base, dtype=np.int64), seed, expansion)
            pools.append(pool)
            added.append(n_added)
        out_rows = int(sum(int((~inside[np.asarray(p, np.int64)]).sum()) for p in pools))
        if out_rows:
            raise SystemExit(f"B1P {name}/{carve}: {out_rows} pool rows outside the setting's passages")
        sizes = np.asarray([np.asarray(p).size for p in pools], dtype=np.int64)
        stats.setdefault("prepared", []).append({
            "structural_entries": int(stores["structural"].col.size) if "structural" in stores else None,
            "questions": len(pools), "rows": int(sizes.sum()), "mean": round(float(sizes.mean()), 1),
            "regime": regime, "families": m3c.regime_families(cfg_h, regime) if fams_ is not None else None})
        return [m3c.Prepared(pop, d_ids, d_sc, s_ids, s_sc, qemb, seeds, pools, np.asarray(added), time.time() - t)]

    return prepare


@contextlib.contextmanager
def on_b1(M, name, v, stats, look=False):
    """The package's dataset opened as B1's setting `name` (variant v) for zu2d_pools.open_carve and zpko.open_pko, and
    (look) for look_u2d's look; every patch put back on exit (including open_carve's store loader, which it leaves)."""
    PK, S4, E, ZP, LB, LBU, LX = M.PK, M.S4, M.E, M.ZP, M.LB, M.LBU, M.LX
    carve = CARVE[v]
    sj, ip_b, col_b, mp_ = load_settings(name, v)
    desc_b1 = sj["desc_b1"]
    open_desc = sj["open"]
    n = int(sj["n_nodes"])
    P_ = Patches()
    try:
        with checks(M):
            P_.set(LB, BENCH=var_dir(M, v), CARVE=carve, load_lists=LBU.load_lists)
            st = LB.load_setting(name)
            if (st.build.get("u1a") or {}).get("variant") != v or st.shas != sj["setting"]["shas"]:
                raise SystemExit(f"B1P {name}/{v}: the setting's files are not the ones settings/{v}/{name}.json names")
            inside = np.zeros(n, bool)
            inside[st.urows] = True

            mp_mod = S4.m3b_pools

            def load_store(ds_, family, cache_dir):
                if ds_.name != name:
                    raise SystemExit(f"B1P: {ds_.name}'s stores asked for; B1P opens {name} only")
                return LB.setting_store(mp_mod, st, family, int(ds_.n_nodes))

            P_.set(mp_mod, load_or_build_store=load_store)

            def graph_csr(name_, graph, n_):
                if name_ != name or graph != "chosen" or int(n_) != n:
                    raise SystemExit(f"B1P: graph_csr({name_}, {graph}, {n_}); B1P swaps {name}'s chosen graph only")
                return ip_b.copy(), col_b.copy(), copy.deepcopy(desc_b1)

            P_.set(M.U2C, graph_csr=graph_csr)

            def u2e_graph(name_):
                if name_ != name:
                    raise SystemExit(f"B1P: u2e_graph({name_})")
                return copy.deepcopy(desc_b1)

            P_.set(ZP, u2e_graph=u2e_graph)
            orig_rec = PK.u2s_record

            def u2s_record(name_, v_=None):
                rec, npz = orig_rec(name_, v_)
                if name_ != name:
                    return rec, npz
                return dict(rec, graph_detail=copy.deepcopy(desc_b1)), npz

            P_.set(PK, u2s_record=u2s_record)
            P_.save(LX.P12, "carve_ids_of")
            P_.save(LX, "carve_population")
            P_.save(LX.S6, "pair_open")
            LB.install(name, st, stats.setdefault("look_b1", {}))

            def populations(op, _S4, ds, name_):
                if name_ != name:
                    raise SystemExit(f"B1P: populations of {name_}")
                pop, _info = LX.carve_population(op.m3b_compile, ds, None, carve, op.m3a, op.m3a.node_position_map(ds),
                                                 name)
                return [(carve, pop)]

            P_.set(E.C4, populations=populations)

            def union_pools(name_, carve_, pop, prep, bases):
                if (name_, carve_) != (name, carve):
                    raise SystemExit(f"B1P: union_pools({name_}, {carve_})")
                bs = [np.union1d(np.asarray(b, np.int64), np.asarray(s, np.int64)) for b, s in zip(bases, prep.seeds)]
                return None, None, bs, [np.asarray(p, np.int64) for p in prep.pools], None

            P_.set(E, union_pools=union_pools)
            orig_names = E.node_names

            def node_names(ds):
                if ds.name != name:
                    raise SystemExit(f"B1P: node names of {ds.name}")
                names = orig_names(ds)
                return [s if inside[i] else "" for i, s in enumerate(names)]

            P_.set(E, node_names=node_names)
            memo = {}
            base_cls = S4.Opened

            class OpenedOnB1(base_cls):
                def __init__(self):
                    super().__init__()
                    P_.set(self.m3b_compile, prepare=make_prepare(M, self.m3b_compile, st, name, carve, stats, memo, inside))

            P_.set(S4, Opened=OpenedOnB1)
            orig_at, orig_p = PK.AT.load_types, PK.P.load_types

            def at_load(name_, desc=None):
                if name_ != name or desc != desc_b1:
                    raise SystemExit(f"B1P: EM types of {name_} on {desc}")
                arr, rec = orig_at(name, open_desc)
                arr = dict(arr)
                arr["cls"] = np.ascontiguousarray(np.asarray(arr["cls"])[mp_])
                if sha_arr(arr["cls"]) != sj["types"]["em"]["cls_b1_sha256"]:
                    raise SystemExit(f"B1P {name}/{v}: the restricted EM types are not the settings record's")
                return arr, dict(rec, entries=int(mp_.size), graph_detail=copy.deepcopy(desc_b1),
                                 cls_sha256=sha_arr(arr["cls"]), b1p={"restricted_from": rec["graph_detail"],
                                                                     "map_sha256": sj["map_sha256"]})

            def p_load(d, desc=None):
                if d != name or desc != desc_b1:
                    raise SystemExit(f"B1P: score types of {d} on {desc}")
                cls, zbar, rec = orig_p(name, open_desc)
                cls = np.ascontiguousarray(np.asarray(cls)[mp_])
                if sha_arr(cls) != sj["types"]["score"]["cls_b1_sha256"]:
                    raise SystemExit(f"B1P {name}/{v}: the restricted score types are not the settings record's")
                return cls, zbar, dict(rec, entries=int(mp_.size), entries_null=int((cls == PK.P.NULL).sum()),
                                       graph_detail=copy.deepcopy(desc_b1),
                                       b1p={"restricted_from": rec["graph_detail"], "map_sha256": sj["map_sha256"]})

            P_.set(PK.AT, load_types=at_load)
            P_.set(PK.P, load_types=p_load)

            def rule(cx, ctx, js, verify=False):
                out = PK.rule_batch(cx, ctx, js, verify)
                for j, f in zip(js, out):
                    for key in ("B", "I", "exp"):
                        a = np.asarray(f[key], np.int64)
                        if a.size and not inside[a].all():
                            raise SystemExit(f"B1P {name}/{carve} question {j}: {key} leaves the setting")
                return out

            if look:
                LU = M.LU

                def ls1_install(_rec, name_):
                    if name_ != name:
                        raise SystemExit(f"B1P: look_step1's install for {name_}")
                    LU._ORIG_OPEN = LX.S6.pair_open                 # B1's pair_open, under look_u2d's

                P_.save(LU, "_ORIG_OPEN")
                P_.set(LU, CARVES=tuple(LU.CARVES) + (carve,))
                P_.set(LU.LS1, install=ls1_install)
            yield SimpleNamespace(st=st, carve=carve, inside=inside, rule=rule, desc=desc_b1, settings=sj)
    finally:
        P_.restore()


# ── the steps ─────────────────────────────────────────────────────────────────

def roots_of(M, root=None):
    return M.PK.roots(Path(root) if root else OUT)


def step_record(R, name, carve, kind, extra):
    write_json(Path(R.base) / "steps" / f"{name}__{carve}__{kind}.json",
               {"declared_in": DECLARED, "stage": STAGE, "dataset": name, "carve": carve, "step": kind,
                "zb1p_sha256": sha_src(__file__), **extra, "utc": utc()})


def pools(name, v, limit=0, host=False, root=None):
    M = mods(host)
    R = roots_of(M, root)
    stats = {}
    t = time.time()
    with on_b1(M, name, v, stats) as B:
        res = M.PK.pools_pass(name, B.carve, threads=2, limit=int(limit or 0), shard=(0, 1), R=R, rule=B.rule)
    log(f"pools {name}/{CARVE[v]}: {res} first stage {stats.get('first_stage')} prepared {stats.get('prepared')}")
    if not limit:
        step_record(R, name, CARVE[v], "pools", {"desc_b1": B.desc, "result": res, "stats": stats,
                                                 "seconds": round(time.time() - t, 1)})
    return 0


def run_step(kind, name, v, index=None, host=False, root=None):
    M = mods(host)
    PK = M.PK
    R = roots_of(M, root)
    carve = CARVE[v]
    t = time.time()
    if kind == "pools":
        return pools(name, v, 0, host, root)
    if kind == "merge":
        ent = PK.merge_carve(name, carve, 1, R)
        step_record(R, name, carve, "merge", {"questions": ent["questions"], "rows": ent["rows"],
                                              "changed": ent["changed"], "short": ent["short"]})
    elif kind == "look":
        with on_b1(M, name, v, {}, look=True):
            PK.look_shard(name, carve, index, R)
    elif kind == "bases":
        PK.copy_bases(R)
    elif kind == "cache":
        with checks(M):
            PK.cache_part(name, carve, index, R)
    elif kind == "links":
        with checks(M):
            PK.chain_build("links", name, carve, R)
    elif kind == "kt":
        PK.kt_stage(name, carve, R)
    elif kind == "record":
        rec = PK.record_carve(name, carve, R, step1=lambda *_a: dict(NO_STEP1))
        step_record(R, name, carve, "record", {"inputs": len(rec["inputs"]), "problems": rec["problems"]})
    else:
        raise SystemExit(f"B1P: step {kind}: one of {STEPS}")
    log(f"step {kind} {name}/{carve} {index or ''}: {time.time() - t:.0f} s")
    return 0


def chain(name, v, host=False, root=None):
    """The carve's pools, merge and PKO's post-pools steps, each its own process; a step already filed is skipped."""
    M = mods(host)
    PK = M.PK
    R = roots_of(M, root)
    carve = CARVE[v]
    base = ([sys.executable, str(PYLIB_RUN)] if host else [sys.executable]) + [str(Path(__file__).resolve())]
    tail = (["--host"] if host else []) + (["--root", str(root)] if root else [])
    env = dict(os.environ, PYTHONIOENCODING="utf-8", OPENBLAS_NUM_THREADS="2", OMP_NUM_THREADS="2",
               MKL_NUM_THREADS="2", NUMBA_NUM_THREADS="2")

    def run(kind, index=None):
        cmd = base + ["step", kind, "--dataset", name, "--variant", v] + \
            (["--index", f"{index[0]}/{index[1]}"] if index else []) + tail
        log(f"chain {name}/{carve}: {kind} {index or ''}")
        t = time.time()
        rc = subprocess.run(cmd, cwd=str(ROOT), env=env).returncode
        if rc:
            raise SystemExit(f"B1P chain {name}/{carve}: step {kind} {index or ''} exited {rc}")
        log(f"chain {name}/{carve}: {kind} {index or ''} done in {time.time() - t:.0f} s")

    st = PK.stem(name, carve)
    man = R.pools / f"{st}.json"
    if not man.exists():
        if not (R.parts / f"{PK.stem(name, carve, (0, 1))}.json").exists():
            run("pools")
        run("merge")
    ent = read_json(man)
    nl, nc = PK.look_parts(name, int(ent["questions"]), int(ent["rows"]))
    bad = PK.other_counts(R, name, carve, nl, nc)
    if bad:
        raise SystemExit(f"B1P chain {name}/{carve}: files of other shard or part counts: {[rel(p) for p in bad[:4]]}")
    for i in range(nl):
        if not all(p.exists() for p in PK.look_outs(R, name, carve, (i, nl))):
            clean_look(R, name, carve, nl)
            run("look", (i, nl))
    if not all((Path(R.cache) / f"basis_{b}.{x}").exists() for b in ("2wiki", "hotpotqa") for x in ("npz", "json")):
        run("bases")
    for i in range(nc):
        if not (Path(R.cache) / name / carve / f"part_{i}of{nc}" / "record.json").exists():
            run("cache", (i, nc))
    if not all((Path(R.links) / name / carve / f"part_{i}of{nc}" / "record.json").exists() for i in range(nc)):
        run("links")
    if not (Path(R.kt) / name / carve / "record.json").exists():
        run("kt")
    if not (Path(R.records) / f"{st}.json").exists():
        run("record")
    log(f"chain {name}/{carve}: filed ({ent['questions']} questions, {ent['rows']} rows, {nl} look shards, {nc} parts)")
    return 0


def clean_look(R, name, carve, nl):
    """A look shard that stopped before its record leaves chunk files no record lists: they are removed (B1P's own
    regenerable files under its root) so the shard runs again on a clean folder."""
    d = Path(R.look) / name / carve
    if not (d / "chunks").exists():
        return
    listed = set()
    for r_ in d.glob(f"record_*of{nl}.json"):
        listed |= set(read_json(r_)["chunks"])
    stray = []
    for p in (d / "chunks").glob("c*.npz"):
        m = re.fullmatch(r"c(\d+)(\.tmp)?", p.stem)    # look_x_six writes c#####.tmp.npz, then renames it
        if m and (m.group(2) or int(m.group(1)) not in listed):
            stray.append(p)
    if not Path(R.base).resolve().is_relative_to(OUT.resolve()) and not str(R.base).startswith(tempfile.gettempdir()):
        raise SystemExit(f"B1P: {rel(d)} is not under B1P's roots; nothing removed")
    for p in stray:
        p.unlink()
    if stray:
        log(f"look {name}/{carve}: {len(stray)} chunk files of an unfinished shard removed")


def gate(host=False):
    M = mods(host)
    R = roots_of(M)
    carves = [(ds, CARVE[v]) for v in VARIANTS for ds in SETTINGS]
    with checks(M):
        verdict = M.PK.gate(R, carves=carves)
    write_json(OUT / "b1p_gate.json", {"declared_in": DECLARED, "stage": STAGE, "pko_gate": rel(R.gate),
                                       "pko_gate_sha256": sha_file(R.gate), "verdict": verdict, "carves": carves,
                                       "settings": {f"{v}/{ds}": sha_file(settings_paths(ds, v)[1])
                                                    for v in VARIANTS for ds in SETTINGS},
                                       "zb1p_sha256": sha_src(__file__), "utc": utc()})
    return 0


def fork():
    """ML1's twelve fits copied for B1P's reads: everything but their reads (reads/, read.json); models.pt checked."""
    for v in VARIANTS:
        for fit in FITS:
            src, dst = SCREEN_FITS / fit, OUT / "fits" / v / fit
            want = sha_file(src / "models.pt")
            if (dst / "fork.json").exists():
                if read_json(dst / "fork.json")["models_sha256"] != want or sha_file(dst / "models.pt") != want:
                    raise SystemExit(f"B1P: {rel(dst)} holds other models than {rel(src)}")
                continue
            tmp = dst.with_name(dst.name + ".tmp")
            if tmp.exists():
                shutil.rmtree(tmp)
            shutil.copytree(src, tmp, ignore=lambda d, names, s=src: [x for x in names if Path(d) == s and x in ("reads", "read.json")])
            if sha_file(tmp / "models.pt") != want:
                raise SystemExit(f"B1P: {rel(tmp)}'s models.pt is not the source's")
            os.replace(tmp, dst)
            write_json(dst / "fork.json", {"declared_in": DECLARED, "fit": fit, "variant": v, "source": rel(src),
                                           "models_sha256": want, "zb1p_sha256": sha_src(__file__), "utc": utc()})
            log(f"fork {rel(src)} -> {rel(dst)}")
    return 0


def read_flags(v):
    return ["--read", ",".join(f"{ds}={CARVE[v]}" for ds in SETTINGS), "--out-root", rel(OUT / "fits" / v)]


def read(fit, v):
    """ML1's read of a fork on B1P's roots, gate and carves (zml1s's code, no stamp); the reads land in the fork."""
    import zml1s as ML
    import zwq as WQ
    import zp1c3 as Z3
    if fit not in FITS or ML.ALL_FITS.get(fit) is None:
        raise SystemExit(f"B1P: {fit} is not one of ML1's zg6r or zrckp fits")
    arm, _seed, _split = ML.ALL_FITS[fit]
    fdir = OUT / "fits" / v / fit
    if not (fdir / "fork.json").exists():
        raise SystemExit(f"B1P: {rel(fdir)} is not forked (zb1p.py fork)")
    outs = [fdir / "reads" / f"{ds}__{CARVE[v]}.npz" for ds in SETTINGS]
    if (fdir / "b1p_read.json").exists() and all(p.exists() for p in outs):
        log(f"read {fit} {v}: filed")
        return 0
    env = ML.device_env()
    ML.where_ok(env, "read")
    _rec, k = ML.fit_stamp(fit)
    if ML.device_of(env) != k.get("device"):
        raise SystemExit(f"B1P: {fit} was fitted on {k.get('device')}; this read runs on {ML.device_of(env)}")
    roots = {**ML.ROOTS, "pools": "outputs/b1p/pools", "plook": "outputs/b1p/look", "cache": "outputs/b1p/cache",
             "rmatch": "outputs/b1p/chains/rmatch", "zrc": "outputs/b1p/chains/zrc", "links": "outputs/b1p/links",
             "kt": "outputs/b1p/kt"}
    flags = read_flags(v)
    P_ = Patches()
    rm = Z3.RM
    orig_read = rm.read

    def rm_read(argv):
        argv = list(argv)
        for i in (0, 2):
            if flags[i] not in argv:
                argv += flags[i:i + 2]
        if argv[argv.index("--out-root") + 1] != flags[3] or argv[argv.index("--read") + 1] != flags[1]:
            raise SystemExit(f"B1P: the read's flags are not B1P's: {argv}")
        return orig_read(argv)

    t = time.time()
    try:
        P_.set(ML, ROOTS=roots, GATE="outputs/b1p/gate.json")
        P_.set(Z3, eval_carves=lambda: [(ds, CARVE[v]) for ds in SETTINGS])
        P_.set(rm, read=rm_read)
        if arm in ML.BASE_OF:
            rc, extra = ML.run_arm(fit, env, "read")
        else:
            argv = ["--name", fit, "--device", "cuda", "--host", "--cache-root", roots["cache"], "--rows-cap",
                    str(ML.ROWS_CAP[arm])] + flags
            ML._ACTIVE[0] = fit
            try:
                with ML.on_ml1():
                    WQ.cache_root_ok(roots["cache"])
                    WQ.gate_passed()
                    why = WQ.wq_why(fit)
                    if why:
                        raise SystemExit(f"B1P: {why}")
                    with WQ.on_wq() as W:
                        rc = W.Z3P.read(argv)
            finally:
                ML._ACTIVE[0] = None
            extra = {}
    finally:
        P_.restore()
    if rc:
        return rc
    miss = [rel(p) for p in outs if not p.exists()]
    if miss:
        raise SystemExit(f"B1P: the read wrote no {miss}")
    write_json(fdir / "b1p_read.json", {"declared_in": DECLARED, "stage": STAGE, "fit": fit, "arm": arm, "variant": v,
                                        "carve": CARVE[v], "reads": {rel(p): sha_file(p) for p in outs},
                                        "device": ML.device_of(env), "gate_sha256": sha_file(OUT / "gate.json"),
                                        "extra": json.loads(json.dumps(extra, default=str)),
                                        "seconds": round(time.time() - t, 1), "zb1p_sha256": sha_src(__file__),
                                        "utc": utc()})
    log(f"read {fit} {v}: {time.time() - t:.0f} s")
    return 0


# ── the report ────────────────────────────────────────────────────────────────

def setting_ids(ds):
    """The setting's question ids as the package names them (B1P's KT ids.json of each carve, the id space of PKO's KT
    files), or None where B1P's KT is not on this machine; both carves must hold one set."""
    got = {v: [str(x) for x in read_json(OUT / "kt" / ds / CARVE[v] / "ids.json")]
           for v in VARIANTS if (OUT / "kt" / ds / CARVE[v] / "ids.json").exists()}
    if len({frozenset(x) for x in got.values()}) > 1:
        raise SystemExit(f"B1P {ds}: b1w and b1t hold different questions")
    return next(iter(got.values()), None)


def overlap_with_training(ds, qids):
    """Each in-domain split's training carve of the setting's dataset (PKO's KT question ids): the setting's questions
    found there."""
    if qids is None:
        return "B1P's KT ids are not on this machine"
    out = {}
    for split in IN_DOMAIN[ds]:
        for d, c in TRAINS[split]:
            if d != ds:
                continue
            p = PKO_KT / d / c / "ids.json"
            if not p.exists():
                out[f"{split}:{d}/{c}"] = "not on this machine"
                continue
            ids = set(str(x) for x in read_json(p))
            out[f"{split}:{d}/{c}"] = int(sum(q in ids for q in qids))
    return out


def report():
    import b1b as BB
    import b1f as B1F
    bench = ROOT / "outputs" / "bench" / "hipporag2"
    rec = {"declared_in": DECLARED, "stage": STAGE, "candidate": BB.CAND, "boot": {"resamples": BB.BOOT, "seed": BB.SEED},
           "in_domain": {k: list(x) for k, x in IN_DOMAIN.items()}, "settings": {}, "missing": []}
    b1f_rep = ROOT / "outputs" / "b1f" / "report.json"
    b1f = read_json(b1f_rep) if b1f_rep.exists() else None
    for ds in SETTINGS:
        qs = read_json(bench / ds / "queries.json")
        gc = [len(q["golds"]) for q in qs]
        s = {"questions": len(qs), "overlap_with_training": overlap_with_training(ds, setting_ids(ds)), "fits": {}, "in_domain": {},
             "zero_shot": {}, "paired_w_minus_t": {}, "calls": {}}
        perq = {}
        for v in VARIANTS:
            for fit in FITS:
                p = OUT / "fits" / v / fit / "reads" / f"{ds}__{CARVE[v]}.npz"
                if not p.exists():
                    rec["missing"].append(rel(p))
                    continue
                with np.load(p) as z:
                    m = BB.per_question(z, gc)
                perq[(v, fit)] = m["R@5"]
                s["fits"][f"{CARVE[v]}/{fit}"] = {**{k: round(float(x.mean()), 4) for k, x in m.items()},
                                                  "file_sha256": sha_file(p)}
        for arm in ARMS:
            for which, splits in (("in_domain", IN_DOMAIN[ds]), ("zero_shot", tuple(x for x in SPLITS if x not in IN_DOMAIN[ds]))):
                if not splits:
                    continue
                for v in VARIANTS:
                    fits = [f"scr-ml1-{arm}-s{sd}{SUF[sp]}" for sp in splits for sd in SEEDS]
                    got = [perq[(v, f)] for f in fits if (v, f) in perq]
                    if len(got) != len(fits):
                        continue
                    mean = np.mean(np.stack(got), axis=0)          # each question's R@5, over the seeds (and splits)
                    ci = BB.boot_ci(mean)
                    s[which][f"{arm}/{CARVE[v]}"] = {"R@5": round(float(mean.mean()), 4), "R5_ci": ci, "fits": len(got),
                                                     "per_fit": [round(float(x.mean()), 4) for x in got]}
                    perq[(v, arm, which)] = mean
                if (("w", arm, which) in perq) and (("t", arm, which) in perq):
                    g = perq[("w", arm, which)] - perq[("t", arm, which)]
                    s["paired_w_minus_t"][f"{arm}/{which}"] = {"R@5": round(float(g.mean()), 4), "ci": BB.boot_ci(g)}
        s["best_published"] = {}
        for which, peer_only in (("any", False), ("peer_reviewed", True)):
            val, row = B1F.best_published(ds, peer_only)
            s["best_published"][which] = {"row": row, "R@5": val}
        s["calls"] = {key: {w: BB.call(c["R5_ci"], b["R@5"] / 100) for w, b in s["best_published"].items()}
                      for key, c in s["in_domain"].items()}
        if b1f is not None:
            s["b1f_beside"] = {cv: {k: x.get("R@5") for k, x in c.items()}
                               for cv, c in ((b1f.get("settings") or {}).get(ds) or {}).get("carves", {}).items()}
        rec["settings"][ds] = s
    rec["zb1p_sha256"] = sha_src(__file__)
    rec["utc"] = utc()
    write_json(OUT / "report.json", rec)
    L = [f"# B1P report (declared in {DECLARED})", "",
         f"R@5 x100 of the {BB.CAND} candidate on HippoRAG 2's settings (1,000 questions each), no LLM and no hyperlink: "
         "b1w = U2b's W2_S rule cut to the setting, b1t = the setting's text links only. In-domain = the seed mean (and "
         "split mean on 2wiki) of the fits that trained on the dataset; 95% bootstrap intervals over the questions; calls "
         "against the best published row (any / peer-reviewed): ABOVE, AT or BELOW.", "",
         "| setting | read | arm | b1t | b1w | w - t (CI) | best (any) | best (peer) | calls b1w (any/peer) |",
         "|---|---|---|---:|---:|---|---|---|---|"]
    for ds, s in rec["settings"].items():
        ba = s["best_published"]
        for which in ("in_domain", "zero_shot"):
            for arm in ARMS:
                cw, ct = s[which].get(f"{arm}/b1w"), s[which].get(f"{arm}/b1t")
                if cw is None and ct is None:
                    continue
                pw = s["paired_w_minus_t"].get(f"{arm}/{which}")
                cl = s["calls"].get(f"{arm}/b1w") if which == "in_domain" else None
                L.append(f"| {ds} | {which.replace('_', '-')} | {arm} | "
                         + (f"{100 * ct['R@5']:.1f}" if ct else "--") + " | "
                         + (f"{100 * cw['R@5']:.1f} ({100 * cw['R5_ci'][0]:.1f}-{100 * cw['R5_ci'][1]:.1f})" if cw else "--")
                         + " | " + (f"{100 * pw['R@5']:+.1f} ({100 * pw['ci'][0]:+.1f}, {100 * pw['ci'][1]:+.1f})" if pw else "--")
                         + f" | {ba['any']['row']} {ba['any']['R@5']} | {ba['peer_reviewed']['row']} {ba['peer_reviewed']['R@5']} | "
                         + (f"{cl['any']}/{cl['peer_reviewed']}" if cl else "--") + " |")
    L += ["", "Setting questions found in the in-domain splits' training carves (PKO's KT ids): "
          + "; ".join(f"{ds} {s['overlap_with_training']}" for ds, s in rec["settings"].items()),
          "", f"Missing reads: {len(rec['missing'])}."]
    (OUT / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    log(f"report: {rel(OUT / 'report.md')}; missing {len(rec['missing'])}")
    return 0


# ── the laptop's look development run ─────────────────────────────────────────

def look_dev(name, v, root, questions=8):
    """look_u2d --dev Q on a carve whose pools were filed under `root` (pools, then step merge, with --root): every
    look_u2d check on Q questions (B1's population, first stage and stores, the filed pools swapped in), nothing kept."""
    M = mods(False)
    PK = M.PK
    R = roots_of(M, root)
    carve = CARVE[v]
    filed, ent, msha = PK.load_filed(name, carve, R)
    LU, L8 = M.LU, M.LX.L8
    zr = PK.zp_roots(R)

    def lf(*a, **k):
        return filed, ent, msha

    with on_b1(M, name, v, {}, look=True):
        P_ = Patches()
        try:
            P_.set(PK.ZP, load_filed=lf, roots=lambda *_a, **_k: zr)
            P_.set(LU, DECLARED=PK.DECLARED)
            P_.set(L8, parse_shard=PK.accept_one_shard(L8.parse_shard))
            rc = LU.main(["--build", "new", "--dataset", name, "--carve", carve, "--full", "--dev", str(int(questions))])
        finally:
            P_.restore()
    log(f"look-dev {name}/{carve}: rc {rc}")
    return rc or 0


# ── self-test ─────────────────────────────────────────────────────────────────

def selftest():
    # 1. the cut: a symmetric CSR with sorted rows; rows and columns kept inside the mask, the map aligned
    n = 7
    pairs = [(0, 1), (0, 3), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (1, 6)]
    adj = [[] for _ in range(n)]
    for a, b in pairs:
        adj[a].append(b)
        adj[b].append(a)
    ip = np.concatenate([[0], np.cumsum([len(sorted(x)) for x in adj])]).astype(np.int64)
    col = np.concatenate([sorted(x) for x in adj]).astype(np.int32)
    lab = np.arange(col.size, dtype=np.int64) * 10                       # a per-entry array (types)
    inside = np.zeros(n, bool)
    inside[[1, 2, 3, 6]] = True
    ip_b, col_b, mp = cut_csr(ip, col, inside)
    want = pair_keys(n, *zip(*[(a, b) for a, b in pairs if inside[a] and inside[b]]))
    assert np.array_equal(csr_keys(ip_b, col_b), want), "cut pairs"
    assert col_b.size == 2 * want.size and int(ip_b[-1]) == col_b.size, "cut symmetric"
    assert np.array_equal(col[mp], col_b) and np.array_equal(lab[mp] // 10, mp), "map aligned"
    for r in range(n):
        seg = col_b[ip_b[r]:ip_b[r + 1]]
        assert (np.diff(seg) > 0).all(), "rows sorted"
        assert inside[r] or seg.size == 0, "rows outside are empty"
        heads = np.repeat(np.arange(n), np.diff(ip))[mp]
        assert np.array_equal(np.repeat(np.arange(n), np.diff(ip_b)), heads), "heads kept"
    e_ip, e_col, e_mp = cut_csr(ip, col, np.zeros(n, bool))
    assert e_col.size == 0 and e_mp.size == 0 and (e_ip == 0).all(), "empty cut"
    # 2. pair keys drop self-loops and repeats, order-free
    assert np.array_equal(pair_keys(5, [1, 2, 3, 3], [2, 1, 3, 4]), np.asarray([1 * 5 + 2, 3 * 5 + 4])), "pair keys"
    # 3. patches restore in reverse order, refusing unknown attributes
    ns = SimpleNamespace(a=1, b=2)
    P_ = Patches()
    P_.set(ns, a=10)
    P_.set(ns, a=20, b=30)
    P_.save(ns, "b")
    ns.b = 40
    P_.restore()
    assert (ns.a, ns.b) == (1, 2), "restore"
    try:
        P_.set(ns, c=1)
        raise AssertionError("unknown attribute set")
    except SystemExit:
        pass
    # 4. the read flags and the fits
    assert read_flags("w")[1] == "musique=b1w,2wiki=b1w,hotpotqa=b1w" and read_flags("t")[3].endswith("outputs/b1p/fits/t")
    assert len(FITS) == 12 == len(set(FITS)) and all(f.startswith("scr-ml1-") for f in FITS)
    assert set(IN_DOMAIN) == set(SETTINGS) and all(set(x) <= set(SPLITS) for x in IN_DOMAIN.values())
    for ds, splits in IN_DOMAIN.items():
        assert all(any(d == ds for d, _c in TRAINS[sp]) for sp in splits), "an in-domain split trains on its dataset"
    # 5. an unfinished look shard: unlisted chunks and look_x_six's temporaries go, listed chunks and other names stay
    with tempfile.TemporaryDirectory() as t:
        R_ = SimpleNamespace(look=Path(t) / "look", base=Path(t))
        ch = R_.look / "x" / "b1w" / "chunks"
        ch.mkdir(parents=True)
        for f in ("c00000.npz", "c00001.npz", "c00002.tmp.npz", "cnotes.npz"):
            (ch / f).write_bytes(b"")
        write_json(ch.parent / "record_0of1.json", {"chunks": [0]})
        clean_look(R_, "x", "b1w", 1)
        assert sorted(p.name for p in ch.iterdir()) == ["c00000.npz", "cnotes.npz"], "clean_look"
    print("zb1p self-test: OK")
    return 0


def parse_index(s):
    if s is None:
        return None
    i, k = (int(x) for x in s.split("/"))
    if not 0 <= i < k:
        raise SystemExit("--index i/n with 0 <= i < n")
    return (i, k)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?")
    ap.add_argument("kind", nargs="?")
    ap.add_argument("--dataset", choices=SETTINGS)
    ap.add_argument("--variant", choices=VARIANTS)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--index")
    ap.add_argument("--name")
    ap.add_argument("--root")
    ap.add_argument("--questions", type=int, default=8)
    ap.add_argument("--host", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest or a.cmd == "selftest":
        return selftest()
    need = {"pools": ("dataset", "variant"), "chain": ("dataset", "variant"), "step": ("dataset", "variant"),
            "read": ("name", "variant"), "look-dev": ("dataset", "variant", "root")}
    for k in need.get(a.cmd, ()):
        if getattr(a, k.replace("-", "_")) is None:
            raise SystemExit(f"{a.cmd}: --{k} is required")
    if a.root and Path(a.root).resolve() == OUT.resolve():
        a.root = None
    if a.cmd == "settings":
        return settings(a.host)
    if a.cmd == "pools":
        return pools(a.dataset, a.variant, a.limit, a.host, a.root)
    if a.cmd == "chain":
        return chain(a.dataset, a.variant, a.host, a.root)
    if a.cmd == "step":
        return run_step(a.kind, a.dataset, a.variant, parse_index(a.index), a.host, a.root)
    if a.cmd == "gate":
        return gate(a.host)
    if a.cmd == "fork":
        return fork()
    if a.cmd == "read":
        return read(a.name, a.variant)
    if a.cmd == "report":
        return report()
    if a.cmd == "look-dev":
        return look_dev(a.dataset, a.variant, a.root, a.questions)
    raise SystemExit(__doc__)


if __name__ == "__main__":
    sys.exit(main())
