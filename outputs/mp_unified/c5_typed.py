"""C5 (docs/C5_TYPED_KB.md): the cold cost table on the two typed KB graphs (metaqa, webqsp), every family at its
fastest exact form (C4's), per question from its first-stage lists, with no cache and no warm-up.

On a typed graph both MLP fits read chain entries (rmatch.ChainMatch's gated chain match): the relation chains a
question's seeds start on its pool's typed structural edges. The step-1 caches hold them built offline (zrc: zrc.build's
contribution cap; zsp: rmatch.build's mass cap). Served cold, a question has to build its own, so C5 times that stage:

  zrc5   pool, read, compile   c2_fast.shared_mlp under C4's galloping typed edges
         edges, lean           c3_fast.lean_rows3
         chains                each structural pair's first K_REL relations (universal_v2_models.relation_slots, as the
                               look stores e_rel), rmatch.row_triples, the seed buckets, zrc.contrib_entries on
                               zrc.step_of (the question's embedding against the relations on its pool), the entries
                               in the build's stored dtypes
         forward               FastZ5: c3_fast.FastZ plus the chain term (ChainMatch.chain_feats), then top 5
  zsp5   as zrc5 with rmatch.question_entries for the chains, plus c3_fast.links_fast; FastZ5 with zprop's head
  gnn5   pool, read, compile   C1.shared (the full fast compile) under the same galloping edges
         pack                  fast_pack5: lean_time.fast_pack with the structural relation slots (pack_queries_v2's)
         forward               c3_fast.FusedGNN, then top 5
Each path runs its whole chain on its own; the order rotates question by question. Each fast path also times a warm
repeat of its forward on its own inputs (beside, labelled warm).

Checks (untimed, every question): pool, seeds and buckets, the edges and their e_rel against the look's; each MLP
path's chain entries against its fit's filed build (zrc/cache, rmatch/cache part 0) bit for bit; FastZ5 within TOL
of the fit's own forward (lean_gpu's batch plus rmatch.chain_batch) on the same rows and entries, same top 5; that
forward on the step-1 cache's rows and the filed entries against FastZ5's scores (reported); the float16 rows and
codes against the step-1 cache's (reported: the look was built at the host's BLAS thread count); gnn5's scores against
the look's stored gnn0 scores, same top 5.

    python outputs/mp_unified/c5_typed.py check --dataset metaqa --queries 30
    python outputs/mp_unified/c5_typed.py run --dataset metaqa
    python outputs/mp_unified/c5_typed.py report
"""
import os
import sys
from pathlib import Path

THREADS = 1
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS",
           "LEAN_TIME_THREADS"):
    os.environ[_v] = str(THREADS)
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import json  # noqa: E402
import time  # noqa: E402

import c2_fast as C2  # noqa: E402
import c3_fast as C3  # noqa: E402
import c4_fast as C4  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402

import rmatch as RM  # noqa: E402
import zrc as ZC  # noqa: E402

C1 = C2.C1
LM, L3, LT, FF, S6, V2, GF, LG, LC = C1.LM, C1.L3, C1.LT, C1.FF, C1.S6, C1.V2, C1.GF, C1.LG, C1.LC
from mp_retrieval.m3b_models import PackedBatch  # noqa: E402
from mp_retrieval.universal_v2_models import K_REL, N_EDGE_FEATURES_V2, relation_slots  # noqa: E402
from mp_retrieval.m3b_features import FAMILIES, MAX_SEEDS  # noqa: E402

ROOT = HERE.parents[1]
OUT = ROOT / "outputs" / "c5"
PATHS = ("zrc5", "zsp5", "gnn5")
CH_ROOT = {"zrc": ZC.CH_OUT, "zsp": RM.CH_OUT}       # zrc's fit reads zrc.build's entries, zsp's rmatch.build's
TOL = C3.TOL
clock = time.perf_counter


# ── the galloping typed edges, with the compile's typed arrays kept for the chains and the pack ───────────────


class StashK(C4.GallopK):
    def __init__(self, base, box):
        super().__init__(base)
        self._box = box

    def typed_edges(self, *a):
        out = super().typed_edges(*a)
        self._box["typed"] = out
        return out


class Kernels5:
    def __init__(self, fc):
        self.box = {}
        fc.K, fc._K_small = StashK(fc.K, self.box), StashK(fc._K_small, self.box)

    def take(self):
        t = self.box.pop("typed", None)
        if t is None:
            raise SystemExit("the compile did not build its typed edges")
        return t


# ── setup: C1.Setup's, for a typed graph ─────────────────────────────────────


class Setup5(C1.Setup):
    """c1_cold.Setup's __init__ for a graph with typed relations (C1's refuses them), plus the relation tables the
    chains read and the look's e_rel and q_emb for the checks."""

    def __init__(self, name, nq):
        t0 = time.time()
        self.name = name
        self.index = {}
        S6.pair_verify(S6.SIX)
        cfg, cfg_m3b, cfg_h = V2.load_configs()
        op = S6.pair_open(S6.SIX, cfg, cfg_m3b, name)
        self.inputs, self.m3b_compile = op.inputs, op.m3b_compile
        self.context, dsh = op.contexts[name], op.handles[name]
        if self.context.rel_table is None:
            raise SystemExit(f"{name}: no typed relations; C5 serves the typed graphs (C4 the others)")
        self.gnn = S6.pair_models(S6.SIX, self.inputs, op.bank)["gnn0"]
        self.fgnn = GF.FastGNN(self.gnn)
        m3b_contract = V2.M3B_RUN.load_script("m3b_contract")
        m3a = op.pkg[0]
        _k, frozen = self.m3b_compile.frozen_contract(cfg_m3b)
        construction = frozen["per_dataset"][name]["construction"]
        positions = m3a.node_position_map(dsh)
        files, ids, head = C1.look_first(name, nq + 1)
        self.look_head = head
        self.ref = self.look_rows5(files, nq + 1)
        ids = ids[:nq + 1]
        self.ids = ids
        pop = C1.carve_population(self.m3b_compile, dsh, ids, C1.CARVE, m3a, positions, name)
        if list(pop.ids) != ids:
            raise SystemExit(f"{name}: a question of the look's first {len(ids)} has no gold in its pool's dataset")
        del positions
        self.prep = self.m3b_compile.prepare(dsh, [pop], construction, cfg_h, self.context.stores, m3a, m3b_contract)[0]
        self.builder = LT.PoolBuilder(construction, cfg_h, self.context.stores, m3a, m3b_contract, self.m3b_compile)
        self.columns = self.inputs["column_indices"]
        self.xcols, spans, ci = LC.xc_columns(head)
        self.c_rrf_x = ci["rrf"]
        self.index["pair_and_lists_s"] = time.time() - t0
        t = time.time()
        self.fc = FF.compiler_for(self.context.stores, self.context.nodes, self.context.rel_table)
        if self.fc is None:
            raise SystemExit("no fast compiler for this context")
        self.index["fast_compiler_s"] = time.time() - t
        t = time.time()
        LG.bind_device_ops()
        LG.set_flags("cpu")
        self.models, self.basis = {}, None
        for k, cls in (("zrc", C1.ZM.ZRM), ("zsp", C1.ZP.ZProp)):
            blob = torch.load(C1.FITS[k] / "models.pt", weights_only=False)
            if self.basis not in (None, blob["basis"]):
                raise SystemExit("the fits are on different bases")
            self.basis = blob["basis"]
            ms = {nm: (m, bl) for nm, m, bl in LG.load_models(blob, "cpu", cls=cls)}
            self.models[k] = ms[C1.CAND]
        self.blocks = list(self.models["zrc"][1])
        if list(self.models["zsp"][1]) != self.blocks:
            raise SystemExit("zrc and zsp read different blocks")
        self.index["models_s"] = time.time() - t
        bz, brec = LC.load_basis(self.basis)
        self.store = L3.Store(bz)
        self.basis_sha256 = brec["npz_sha256"]
        self.R = LM.projection()
        self.span, at = {}, 0
        for b in LC.XC_BLOCKS:
            self.span[b] = tuple(spans[b])
            at = spans[b][1]
        for b in ("WALK", "WALKF", "SEED", "DISTS"):
            w = LM.LEAN_W.get(b, C1.L2.NEW_W.get(b))
            self.span[b] = (at, at + w)
            at += w
        self.W = at
        self.c_rrf = self.span["rank"][0] + LM.SPLIT["rank"].index("rrf")
        # the relations: the look's record, the embeddings both chain builds hashed, the tables chain_feats reads
        t = time.time()
        rel = RM.look_relations(LM.LOOK / name / C1.CARVE)
        self.n_rel, self.k_rel, self.rel_offset = int(rel["n_relations"]), int(rel["k_rel"]), int(rel["offset"])
        if self.k_rel != K_REL or self.rel_offset != int(getattr(self.context, "rel_offset", -1)):
            raise SystemExit(f"{name}: the look's relations {rel} are not the context's")
        ef = RM.REL_DIR / f"{name}_rel_embeddings.npy"
        self.rel_sha256 = C1.sha_file(ef)
        for k, root in CH_ROOT.items():
            h = json.loads((root / name / C1.CARVE / self.part0() / "record.json").read_text(encoding="utf-8"))
            if h["relations"]["embeddings_sha256"] != self.rel_sha256 or int(h["relations"]["n_relations"]) != self.n_rel:
                raise SystemExit(f"{k}'s chain build of {name} read other relations")
        E = np.load(ef).astype(np.float64)
        self.U = E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)
        self.rel_unit = torch.from_numpy(self.U.astype(np.float32))
        self.rel_centered = torch.from_numpy((self.U - self.U.mean(0, keepdims=True)).astype(np.float32))
        self.index["relations_s"] = time.time() - t
        t = time.time()
        nodes = np.unique(np.concatenate([np.asarray(p, np.int64) for p in self.prep.pools]))
        self.codes = np.zeros((int(self.fc.n_nodes), L3.STORE_K), np.int8)
        for s0 in range(0, nodes.size, 4096):
            rr = nodes[s0:s0 + 4096]
            self.codes[rr] = self.store.codes(L3.unit(np.asarray(self.context.nodes.read(rr), np.float32)))
        self.index["store_codes_s"] = time.time() - t
        self.index["store_codes_nodes"] = int(nodes.size)
        self.index["setup_total_s"] = time.time() - t0

    def part0(self):
        parts = sorted((LC.OUT / self.name / C1.CARVE).glob("part_0of*"))
        if len(parts) != 1:
            raise SystemExit(f"{self.name}: step 1's part 0 is not on disk")
        return parts[0].name

    @staticmethod
    def look_rows5(files, nq):
        out = C1.Setup.look_rows(files, nq)
        at = 0
        for f in files:
            z = np.load(f)
            qed = z["q_edges"]
            eo = np.r_[0, np.cumsum(qed)]
            for i in range(qed.size):
                if at >= len(out):
                    return out
                out[at]["e_rel"] = z["e_rel"][eo[i]:eo[i + 1]]
                out[at]["q_emb"] = z["q_emb"][i]
                at += 1
        return out


# ── the chains ───────────────────────────────────────────────────────────────


def e_rel_of(r, typed):
    """(edges, K_REL) int16: each structural message edge's first K_REL relations (the look's e_rel), -1 elsewhere."""
    _ev, _eu, erel, _edir, pair_id, pu, _pv = typed
    ef = r["e"][2]
    s = ef == 0
    if int(s.sum()) != pu.size or (pair_id.size and int(pair_id[-1]) + 1 != pu.size):
        raise SystemExit("the structural edges are not the compile's pairs")
    slots, _tr = relation_slots(pair_id, erel, int(pu.size), 0)
    out = np.full((ef.size, K_REL), -1, np.int16)
    out[s] = slots.astype(np.int16)
    return out


def chains_of(su, k, r, e_rel, qemb):
    """One question's chain entries for fit k, in the build's stored dtypes, and its pool's relations."""
    eu, ev, ef, efw, ebw = r["e"]
    c = {"e_fam": ef, "e_u": eu, "e_v": ev, "e_fwd": efw, "e_bwd": ebw, "e_rel": e_rel}
    hl, tl, sl = RM.row_triples(c, 0, int(ef.size))
    sl0, bk = r["sl"], r["sb"]
    ok = sl0 >= 0
    seeds = [np.unique(sl0[ok & (bk == b)]).astype(np.int64) for b in range(RM.BUCKETS)]
    qr = np.unique(sl[sl >= 0])
    n = int(r["n"])
    if k == "zrc":
        (row, zz, bb, mm), _st = ZC.contrib_entries(n, hl, tl, sl, seeds, 2 * su.n_rel, ZC.step_of(qemb, su.U, qr))
    else:
        (row, zz, bb, mm), _st = RM.question_entries(n, hl, tl, sl, seeds, 2 * su.n_rel)
    return {"row": row.astype(np.int16), "z": zz.astype(np.int16), "b": bb.astype(np.int8), "m": mm.astype(np.float16),
            "q_rel": qr.astype(np.int16)}


def one_question_chains(su, ents):
    """rmatch.load_chains' dict for one question (B = 1), for rmatch.chain_batch."""
    ne, nr = int(ents["row"].size), int(ents["q_rel"].size)
    return {"q_ent": np.asarray([ne], np.int64), "q_nrel": np.asarray([nr], np.int64), "e_off": np.asarray([0, ne]),
            "r_off": np.asarray([0, nr]), "ent_row": ents["row"], "ent_z": ents["z"], "ent_b": ents["b"],
            "ent_m": ents["m"], "q_rel": ents["q_rel"], "rel_unit": su.rel_unit, "rel_centered": su.rel_centered,
            "n_rel": su.n_rel}


def filed_chains(ds, k, nq):
    """The fit's filed chain entries of the carve's first nq questions (part 0 of its build), per question."""
    root = CH_ROOT[k] / ds / C1.CARVE
    parts = sorted(root.glob("part_0of*"))
    if len(parts) != 1:
        raise SystemExit(f"{root}: part 0 is not on disk")
    p = parts[0]
    h = json.loads((p / "record.json").read_text(encoding="utf-8"))
    arr = {}
    for a in RM.ARRAYS:
        f = p / f"{a}.npy"
        if LC.sha_file(f) != h["arrays"][a]["sha256"]:
            raise SystemExit(f"{f}: sha256 is not its record's")
        arr[a] = np.load(f)
    eo, ro = np.r_[0, np.cumsum(arr["q_ent"])], np.r_[0, np.cumsum(arr["q_nrel"])]
    out = []
    for i in range(min(nq, arr["q_ent"].size)):
        a, b, c, d = eo[i], eo[i + 1], ro[i], ro[i + 1]
        out.append({"row": arr["ent_row"][a:b], "z": arr["ent_z"][a:b], "b": arr["ent_b"][a:b], "m": arr["ent_m"][a:b],
                    "q_rel": arr["q_rel"][c:d]})
    return out, {"part": p.name, "record_sha256": LC.sha_file(p / "record.json")}


def same_entries(a, b):
    return all(a[x].dtype == b[x].dtype and a[x].shape == b[x].shape and np.array_equal(C1.bits(a[x]), C1.bits(b[x]))
               for x in ("row", "z", "b", "m", "q_rel"))


# ── the fast MLP forward with the chain term ─────────────────────────────────


class FastZ5(C3.FastZ):
    """c3_fast.FastZ with ChainMatch's chain term added to the base score (zrm.ZRM's forward), before zprop's head
    (zprop.ZProp's): the same function on the same weights."""

    @torch.inference_mode()
    def __call__(self, r):
        m = self.m
        X = torch.from_numpy(r["X"]).to(torch.float32)
        R = torch.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).index_select(1, self.fixed)
        qe = torch.from_numpy(r["q_emb16"]).to(torch.float32)
        if self.semb:
            P = torch.from_numpy(r["codes"]).to(torch.float32) * self.s + self.c
            P = torch.cat([P, torch.full((P.shape[0], 1), self.kappa)], 1)
            R = torch.cat([R, (qe @ m.U) * (P @ m.V)], 1)
        ref = R[:, self.i_rrf] > 0
        Z = C3.zcols(R, ref)
        h = Fn.gelu(torch.addmm(self.b1, R, self.Wr).addmm_(Z, self.Wz))
        h = Fn.gelu(m.l2(h))
        s = m.base_w * Z[:, self.i_rrf] + m.out(h).squeeze(-1)
        n = s.shape[0]
        if r.get("chains") is not None:
            ch = RM.chain_batch(_One(n, r["chains"]), [0])
            fm, fr = m.chain_feats({RM.CH_KEY: ch, "SEMB": (qe.unsqueeze(0),)}, n)
            s = s + (m.cm_gate[0] * fm + m.cm_gate[1] * fr)
        if not self.zsp:
            return s
        _ne, a, b, f = r["links"]
        zs = C3.zcols(s.unsqueeze(1), torch.ones(n, dtype=torch.bool)).squeeze(1)
        zc = zs.clamp(-C1.ZP.ZCLIP, C1.ZP.ZCLIP).numpy()
        deg, sm, ex = C3.prop_numba(zc, a.astype(np.int64), b.astype(np.int64), f.astype(np.int64), n, C1.ZP.FAMS)
        deg, sm, ex = torch.from_numpy(deg), torch.from_numpy(sm), torch.from_numpy(ex)
        has = deg > 0
        zero = torch.zeros_like(deg)
        x = torch.cat([torch.where(has, sm / deg.clamp_min(1.0), zero),
                       torch.where(has, torch.log(ex.clamp_min(1e-30)), zero), torch.log1p(deg), zs.unsqueeze(1)], 1)
        return s + m.prop_head(x)


class _One:
    """The attributes rmatch.chain_batch reads, for one question."""

    def __init__(self, n, chains):
        self.n_np = np.asarray([n], np.int64)
        self.chains = chains
        self.device = torch.device("cpu")


class Rows5(C1.Rows):
    """c1_cold.Rows with each question's chain entries under rmatch.CH_KEY (rmatch.chain_carve's batch)."""

    def __init__(self, qrows, span, W, c_rrf, store, chains):
        super().__init__(qrows, span, W, c_rrf, store)
        self.chains = chains

    def batch(self, qs, blocks, links=False):
        feats, nq, base_z, gold = super().batch(qs, blocks, links=links)
        if self.chains is not None:
            feats[RM.CH_KEY] = RM.chain_batch(self, qs)
        return feats, nq, base_z, gold


class Fast5:
    def __init__(self, su):
        self.z = {k: FastZ5(su.models[k][0].eval(), su.models[k][1], su.span, su.store) for k in ("zrc", "zsp")}
        self.g = C3.FusedGNN(su.gnn)


# ── the six GNN's pack with the relation slots ───────────────────────────────


def fast_pack5(comp, E, qemb, columns, typed, offset):
    """lean_time.fast_pack with pack_queries_v2's structural relation slots (the bank rows of each pair's first K_REL
    relations, from the compile's typed entries)."""
    n = int(comp.pool.size)
    row = np.full(MAX_SEEDS, -1, dtype=np.int64)
    sl = np.asarray(comp.seeds_local, dtype=np.int64)
    row[:sl.size] = sl[:MAX_SEEDS]
    eis, eas = [], []
    for f_i, fam in enumerate(FAMILIES):
        u, v, attr = comp.edges[fam]
        if u.size == 0:
            continue
        eis.append(np.stack((u.astype(np.int64), v.astype(np.int64))))
        onehot = np.zeros((u.size, len(FAMILIES)), dtype=np.float32)
        onehot[:, f_i] = 1.0
        if fam == "structural":
            pair_id = typed[4]
            if pair_id.size and int(pair_id[-1]) + 1 != u.size:
                raise SystemExit("the typed entries are not the structural pairs")
            slots, _tr = relation_slots(pair_id, typed[2], int(u.size), offset)
        else:
            slots = np.full((u.size, K_REL), -1.0, dtype=np.float32)
        eas.append(np.concatenate((onehot, attr, slots), axis=1))
    ei = np.concatenate(eis, axis=1) if eis else np.empty((2, 0), dtype=np.int64)
    ea = np.concatenate(eas, axis=0) if eas else np.empty((0, N_EDGE_FEATURES_V2), dtype=np.float32)
    return PackedBatch(x=torch.from_numpy(np.ascontiguousarray(comp.scalars[:, columns])).to(torch.float32),
                       qptr=torch.tensor([0, n], dtype=torch.long), node_query=torch.zeros(n, dtype=torch.long),
                       emb=torch.from_numpy(E), qemb=torch.from_numpy(qemb[None, :]), seedw=torch.from_numpy(comp.seedw),
                       seed_nodes=torch.from_numpy(row[None, :]), edge_index=torch.from_numpy(ei),
                       edge_attr=torch.from_numpy(ea), gold=torch.zeros(n, dtype=torch.bool))


# ── the paths ────────────────────────────────────────────────────────────────


def run_path5(su, fa, ks, k, i, T):
    """One path for question i, cold, every stage of its own timed into T (a warm repeat of the forward beside);
    returns (scores, shared, rows or pack)."""
    if k == "gnn5":
        sh = C1.shared(su, i, T)
        typed = ks.take()
        c0 = clock()
        b = fast_pack5(sh["comp"], sh["E"], sh["qemb"], su.columns, typed, su.rel_offset)
        c1 = clock()
        with torch.inference_mode():
            sc = fa.g(b)
        torch.topk(sc, min(5, int(sh["pool"].size)))
        c2 = clock()
        with torch.inference_mode():
            fa.g(b)
        c3 = clock()
        T["pack"], T["forward"], T["warm_forward"] = c1 - c0, c2 - c1, c3 - c2
        sh["typed"] = typed
        return sc, sh, b
    fit = "zsp" if k == "zsp5" else "zrc"
    sh = C2.shared_mlp(su, i, T)
    typed = ks.take()
    r = C3.lean_rows3(su, sh, T)
    c0 = clock()
    e_rel = e_rel_of(r, typed)
    r["chains"] = one_question_chains(su, chains_of(su, fit, r, e_rel, sh["qemb"]))
    c1 = clock()
    if k == "zsp5":
        eu, ev, ef = r["e"][:3]
        r["links"] = C3.links_fast(r["n"], eu, ev, ef)
    c2 = clock()
    fz = fa.z[fit]
    sc = fz(r)
    torch.topk(sc, min(5, r["n"]))
    c3 = clock()
    fz(r)
    c4 = clock()
    T["chains"] = c1 - c0
    if k == "zsp5":
        T["links"] = c2 - c1
    T["forward"], T["warm_forward"] = c3 - c2, c4 - c3
    r["e_rel"] = e_rel
    return sc, sh, r


# ── the checks (untimed) ─────────────────────────────────────────────────────


class Part0:
    """Step 1's cached rows of the carve's first questions (part 0 only), as lean_gpu.CacheCarve lays them out."""

    def __init__(self, su, nq):
        p = LC.OUT / su.name / C1.CARVE / su.part0()
        r = json.loads((p / "record.json").read_text(encoding="utf-8"))
        if list(r["ids"][:nq]) != list(su.ids[:nq]):
            raise SystemExit(f"{p}: its questions are not the look's")

        def get(name):
            return np.array(LC.load_array(p, name, r, True))
        n = get("n").astype(np.int64)
        off = np.r_[0, np.cumsum(n)]
        R = int(off[min(nq, n.size)])
        X = np.empty((R, su.W), np.float16)
        X[:, :su.span[LC.XC_BLOCKS[-1]][1]] = get("xc")[:R]
        for b, name in (("WALK", "walk"), ("WALKF", "walkf"), ("SEED", f"seed_{su.basis}"), ("DISTS", f"dists_{su.basis}")):
            a, e = su.span[b]
            X[:, a:e] = get(name)[:R]
        tab, row = get(f"tab_{su.basis}"), get(f"row_{su.basis}").astype(np.int64)[:R]
        self.n, self.off, self.X, self.codes, self.q_emb = n, off, X, tab[row], get("q_emb")
        self.record_sha256 = LC.sha_file(p / "record.json")

    def rows(self, i):
        a, b = int(self.off[i]), int(self.off[i + 1])
        return {"n": b - a, "X": self.X[a:b], "codes": self.codes[a:b], "q_emb16": self.q_emb[i]}


@torch.inference_mode()
def ref_forward(su, k, r, chains):
    rows = Rows5([r], su.span, su.W, su.c_rrf, su.store, chains)
    return C1.forward(su, k, rows, 1, links=(k == "zsp"))


def check_c5(su, i, out, filed, part0):
    ref = su.ref[i]
    c = {"pool": all(bool(np.array_equal(out[k][1]["pool"], ref["pool"])) for k in PATHS)}
    rz, rs = out["zrc5"][2], out["zsp5"][2]
    c["seeds"] = all(bool(np.array_equal(r["sl"], ref["sl"]) and np.array_equal(r["sb"], ref["sb"])) for r in (rz, rs))
    c["edges"] = all(bool(all(np.array_equal(a, b) for a, b in zip(r["e"], ref["e"]))) for r in (rz, rs))
    c["e_rel"] = all(bool(np.array_equal(r["e_rel"], ref["e_rel"])) for r in (rz, rs))
    c["q_emb"] = bool(np.array_equal(np.asarray(out["zrc5"][1]["qemb"], np.float32), ref["q_emb"]))
    for k, r in (("zrc5", rz), ("zsp5", rs)):
        fit = "zsp" if k == "zsp5" else "zrc"
        got = {"row": r["chains"]["ent_row"], "z": r["chains"]["ent_z"], "b": r["chains"]["ent_b"],
               "m": r["chains"]["ent_m"], "q_rel": r["chains"]["q_rel"]}
        c[f"{k}_entries"] = bool(i < len(filed[fit]) and same_entries(got, filed[fit][i]))
        c[f"{k}_n_entries"] = int(got["row"].size)
        want = ref_forward(su, fit, r, r["chains"])
        d, top = C3.compare(want, out[k][0])
        c[f"{k}_diff"], c[f"{k}_top5_same"], c[f"{k}_within_tol"] = d, top, d <= TOL
        if part0 is not None and i + 1 < part0.off.size:
            pr = part0.rows(i)
            c[f"{k}_rows_vs_cache"] = C1.diff(r["X"], pr["X"])
            c[f"{k}_codes_vs_cache"] = int((pr["codes"] != r["codes"]).sum())
            if fit == "zsp":
                pr["links"] = C1.links_of(pr["n"], *ref["e"][:3])
            if i < len(filed[fit]):
                fc = one_question_chains(su, filed[fit][i])
                d2, top2 = C3.compare(ref_forward(su, fit, pr, fc), out[k][0])
                c[f"{k}_vs_cache"], c[f"{k}_vs_cache_top5"] = d2, top2
    g = out["gnn5"][0].numpy()
    c["gnn5_vs_look"] = float(np.abs(g - ref["gnn0"].astype(np.float32)).max()) if g.size else 0.0
    c["gnn5_top5_look"] = bool(np.array_equal(C1.top5(g), C1.top5(ref["gnn0"].astype(np.float32))))
    return c


GOOD = ("pool", "seeds", "edges", "e_rel", "zrc5_entries", "zsp5_entries", "zrc5_within_tol", "zrc5_top5_same",
        "zsp5_within_tol", "zsp5_top5_same", "gnn5_top5_look")


def summarize(checks):
    s = {k: int(sum(bool(c[k]) for c in checks)) for k in GOOD + ("q_emb",)}
    s["questions"] = len(checks)
    for k in ("zrc5_diff", "zsp5_diff", "gnn5_vs_look", "zrc5_vs_cache", "zsp5_vs_cache"):
        v = [c[k] for c in checks if k in c]
        if v:
            s[f"{k}_max"] = float(max(v))
    for k in ("zrc5_vs_cache_top5", "zsp5_vs_cache_top5"):
        v = [c[k] for c in checks if k in c]
        if v:
            s[k] = f"{int(sum(v))}/{len(v)}"
    for k in ("zrc5_rows_vs_cache", "zsp5_rows_vs_cache"):
        v = [c[k] for c in checks if k in c]
        if v:
            s[k] = {"questions_differing": int(sum(x[0] != 0 for x in v)), "elements": int(sum(x[0] for x in v)),
                    "largest": float(max(x[1] for x in v))}
    for k in ("zrc5_n_entries", "zsp5_n_entries"):
        v = [c[k] for c in checks]
        s[f"{k}_p50"] = float(np.percentile(v, 50)) if v else 0.0
    return s


def cold(T, k):
    return sum(v for s_, v in T[k].items() if s_ != "warm_forward")


# ── run, check, report ───────────────────────────────────────────────────────


def prepare(a):
    su = Setup5(a.dataset, a.queries)
    fa = Fast5(su)
    ks = Kernels5(su.fc)
    nq = len(su.prep.pools) - 1
    filed, frec = {}, {}
    for k in CH_ROOT:
        filed[k], frec[k] = filed_chains(a.dataset, k, nq)
    part0 = Part0(su, nq)
    return su, fa, ks, nq, filed, frec, part0


def run(a):
    import gc
    t_start = time.time()
    pinned = C1.pin() if not a.no_pin else {"pinned": False, "why": "--no-pin"}
    torch.set_num_threads(THREADS)
    if not a.no_pin and not pinned.get("pinned"):
        raise SystemExit("the process could not pin itself; nothing is timed unpinned")
    su, fa, ks, nq, filed, frec, part0 = prepare(a)
    t = time.time()
    for k in PATHS:
        run_path5(su, fa, ks, k, nq, {})
    su.index["jit_question_s"] = time.time() - t
    C1.log(f"{a.dataset}: {nq} measured questions, index {su.index}")
    recs, checks = [], []
    gc.collect()
    gc.disable()
    try:
        for i in range(nq):
            order = PATHS[i % len(PATHS):] + PATHS[:i % len(PATHS)]
            T, out = {"order": list(order)}, {}
            for k in order:
                T[k] = {}
                out[k] = run_path5(su, fa, ks, k, i, T[k])
            T["n"] = int(out["gnn5"][1]["pool"].size)
            recs.append(T)
            checks.append(check_c5(su, i, out, filed, part0))
            del out
            if (i + 1) % 25 == 0:
                gc.collect()
                C1.log(f"  {i + 1}/{nq}")
    finally:
        gc.enable()
    rec = {"dataset": a.dataset, "carve": C1.CARVE, "queries": nq, "declared_in": "docs/C5_TYPED_KB.md",
           "threads": THREADS, "pin": pinned, "index": su.index, "per_question": recs, "checks": checks,
           "check_summary": summarize(checks), "tol": TOL,
           "fits": {k: {"dir": str(C1.FITS[k].relative_to(ROOT)).replace("\\", "/"),
                        "models_sha256": C1.sha_file(C1.FITS[k] / "models.pt"), "candidate": C1.CAND} for k in C1.FITS},
           "chain_builds": {k: {"root": str(CH_ROOT[k].relative_to(ROOT)).replace("\\", "/"), **frec[k]} for k in frec},
           "step1_part0_record_sha256": part0.record_sha256, "relations_sha256": su.rel_sha256,
           "basis": su.basis, "basis_sha256": su.basis_sha256, "look_records": su.look_head["records"],
           "peak_rss_bytes": LC.peak_rss(), "script_sha256": C1.sha_file(__file__), "numpy": np.__version__,
           "torch": torch.__version__, "seconds": round(time.time() - t_start, 1),
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    out = Path(a.out or OUT / f"{a.dataset}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    LC.write_json(out, rec)
    C1.log(f"{a.dataset}: done in {rec['seconds']}s; checks {rec['check_summary']}; -> {out}")
    return 0


def check(a):
    """The development check: every question's checks, no timing kept."""
    C1.pin(2)
    torch.set_num_threads(THREADS)
    su, fa, ks, nq, filed, frec, part0 = prepare(a)
    bad, checks = [], []
    for i in [nq] + list(range(nq)):
        out = {k: run_path5(su, fa, ks, k, i, {}) for k in PATHS}
        if i == nq:
            continue
        c = check_c5(su, i, out, filed, part0)
        checks.append(c)
        if not all(c[k] for k in GOOD):
            bad.append((i, {k: c[k] for k in GOOD if not c[k]}, {k: c.get(k) for k in ("zrc5_diff", "zsp5_diff")}))
    C1.log(f"c5 check {a.dataset}: {nq} questions, {len(bad)} failing {bad[:3]}")
    C1.log(f"  {summarize(checks)}")
    return 0 if not bad else 1


def report():
    res = {}
    for f in sorted(OUT.glob("*.json")):
        if f.name.startswith("report"):
            continue
        r = json.loads(f.read_text(encoding="utf-8"))
        P = r["per_question"]
        good = [j for j, c in enumerate(r["checks"]) if all(c[k] for k in GOOD)]
        d = {"questions": len(P), "kept": len(good), "stopped": 1 - len(good) / max(len(P), 1) > 0.02}
        rng = np.random.default_rng(0)
        col = lambda k: np.asarray([cold(P[j], k) for j in good])  # noqa: E731
        for k in PATHS:
            x = col(k) * 1e3
            d[f"{k}_cold_ms"] = {str(p): float(np.percentile(x, p)) for p in (50, 95, 99)}
            d[f"{k}_warm_forward_ms_p50"] = float(np.percentile([P[j][k]["warm_forward"] for j in good], 50)) * 1e3
            d[f"{k}_stage_p50_ms"] = {s_: float(np.percentile([P[j][k][s_] for j in good], 50)) * 1e3
                                      for s_ in P[good[0]][k] if s_ != "warm_forward"}
        for num, den in (("gnn5", "zrc5"), ("gnn5", "zsp5"), ("zsp5", "zrc5")):
            d[f"cold_{num}_over_{den}"] = C2.ratio_ci(col(num), col(den), rng)
        wf = lambda k: np.asarray([P[j][k]["warm_forward"] for j in good])  # noqa: E731
        d["warm_forward_gnn5_over_zrc5"] = C2.ratio_ci(wf("gnn5"), wf("zrc5"), rng)
        d["pool_rows_p50"] = float(np.percentile([P[j]["n"] for j in good], 50))
        d["checks"] = r["check_summary"]
        res[r["dataset"]] = d
    LC.write_json(OUT / "report.json", res)
    f = lambda r_: f"{r_['ratio']:.2f} [{r_['ci'][0]:.2f}, {r_['ci'][1]:.2f}]"  # noqa: E731
    lines = ["Cold, batch 1, total per question, p50 ms:", "",
             "| dataset | pool rows p50 | zrc5 | zsp5 | gnn5 | **gnn5 / zrc5** | gnn5 / zsp5 | zrc5 chains p50 | "
             "warm forward gnn5 / zrc5 |",
             "| --- | ---: | ---: | ---: | ---: | --- | --- | ---: | --- |"]
    for ds, d in res.items():
        m = lambda k: f"{d[f'{k}_cold_ms']['50']:.1f}"  # noqa: E731
        lines.append(f"| {ds} | {d['pool_rows_p50']:.0f} | {m('zrc5')} | {m('zsp5')} | {m('gnn5')} | "
                     f"**{f(d['cold_gnn5_over_zrc5'])}** | {f(d['cold_gnn5_over_zsp5'])} | "
                     f"{d['zrc5_stage_p50_ms']['chains']:.2f} | {f(d['warm_forward_gnn5_over_zrc5'])} |")
    lines += ["", "Stage p50 ms:", ""]
    for ds, d in res.items():
        for k in PATHS:
            lines.append(f"- {ds} {k}: " + ", ".join(f"{s_} {v:.2f}" for s_, v in d[f"{k}_stage_p50_ms"].items()))
    lines += ["", "Checks:", ""] + [f"- {ds}: kept {d['kept']}/{d['questions']}; {d['checks']}" for ds, d in res.items()]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("run", "check", "report"))
    ap.add_argument("--dataset")
    ap.add_argument("--queries", type=int, default=200)
    ap.add_argument("--out")
    ap.add_argument("--no-pin", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "report":
        return report()
    if a.dataset not in C1.TYPED:
        raise SystemExit(f"C5 serves {C1.TYPED}")
    return run(a) if a.cmd == "run" else check(a)


if __name__ == "__main__":
    sys.exit(main())
