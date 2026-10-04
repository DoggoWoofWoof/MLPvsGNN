"""Design look (untracked; not a result and not filed): S6 part 4 of the transfer plan, the GNN in the node scorer's
place. Part 3 (chainscore19, read 16:45 on 4 Oct) coupled a node MLP f and a chain scorer g by EM; trained on metaqa
with the granularity transforms, its em-gr read +6.43 [+4.33, +8.91] R@5 over the untyped walk on webqsp selectf + fit
(1,507 rows, zero-shot: its J1, ABOVE), losing still on the one-step rows (D = 1, -10.1). Its declared next step: the
GNN takes f's place in the same coupling, with an fg-gr control for the rounds. This file adds the GNN and the control.
chainscore19 is imported unchanged and its builds are read as they are.

The GNN f, over the row's graph after its transform (rf's mediators included). Messages (src, d, dst): one per
distinct (node, direction, neighbour), d = 0 along a stored triple (head to tail) and 1 against it: the dir level's
edges, which every KB has. No relation id or name enters the GNN.
    h_0 = MLP(x)            two layers of 64, ReLU; x = chainscore19's 13 node features, standardised as there
    h_l = h_{l-1} + ReLU(W_l h_{l-1} + b_l + sum_d M_{l,d} mean_{l,d}(v) [+ log(1 + abar_{l,d}(v)) u_{l,d}]),  l = 1, 2
    f   = w_o . h_2 + b_o
mean_{l,d}(v) averages h_{l-1} over the messages into v with direction d.
    en    the plain mean (an untyped two-channel GNN)
    ena   the mean weighted by w(e) = exp(10 theta_{l,d} a_l(e)), and the bracketed term, where a_l(e) is the
          probability, under the chain posterior p, that a chain's l-th step type is one of the message's types:
              a_l(e) = sum over levels of sum over z in types(e, level) of pi_{level,l}(z),
              pi_{level,l}(z) = sum of p(c) over the row's chains c at that level whose l-th type is z,
          types(e, level) the message's types at that level (none: 0; dir: d; tt<k> and exact: 2 lab[r] + d over its
          relations r). abar_{l,d}(v) sums a_l over the messages into v with direction d. theta and u start at 0, where
          ena's forward is en's. Relation-typed attention whose types come from EM: each round reruns the GNN on that
          round's posterior.
The coupling is chainscore19's with f made per round (m_c the chain's end distribution over the row's non-seed nodes):
    s_0 = f_0 = GNN(x; a(p0)),  and for t = 1..T (T = 2), with pi = softmax(s_{t-1}) over the non-seed nodes:
        E   log p_t = log p0 + beta log(sum_v m_c(v) pi(v) + 1e-8), normalised over the row's chains
        M   s_t = f_t + gamma log(eps + sum_c p_t(c) m_c(v)),  f_t = GNN(x; a(p_t)) (ena), f_t = f_0 (en)
T = 0 is one product, s = f_0 + gamma log(eps + sum_c p0(c) m_c(v)). beta, gamma and eps as chainscore19. The loss is
chainscore19's: the gold share of softmax(s_T) plus chainscore18's chain loss on p_T.

Arms. All train on metaqa's fit carve and its four transforms (chainscore19's -gr builds: al4, sp4, mg3, rf), one seed,
the epoch chosen by R@5 on metaqa's select carve; training as chainscore19 (Adam 1e-3, weight decay 1e-4, 12 epochs of
5,960 examples in batches of 64, one thread, deterministic):
    em-gr   chainscore19's em-gr, through this file's loop (R0)
    fg-gr   T = 0 with the MLP f: one product (the control part 3 lacked)
    n-gr    the GNN alone (no g; the node loss)
    en-gr   em-gr with the untyped GNN as f
    ena-gr  em-gr with the posterior-typed GNN as f
Reads: chainscore19's three, from its builds: metaqa x1f (in-domain), webqsp selectf (zero-shot, 305 rows; the twin
valid) and webqsp selectf + fit (zero-shot, 1,507 rows read as one population; set against the walks only).

Companion builds ('edges'). A chainscore19 build keeps neither the rows' messages nor its chains' type sequences. A
companion reruns its build's own arguments (carve, transform, rows, hops, levels, chain cap) over the same chunks and
keeps the messages, each chain's type slot per step (steps 1 and 2: the GNN's two layers) and the message-type
incidence. It refuses unless every row's pool size, every chain's level and id, its end set (nodes and float32 masses)
and every node's degree feature equal the build's; training refuses a companion whose recorded build hash differs.

Verdicts, fixed at 17:20 on 4 Oct before any number from this file. Paired row bootstrap (BOOT 1000) on R@5:
    K1  (primary) webqsp selectf + fit: en-gr minus em-gr and ena-gr minus em-gr: ABOVE / AT / BELOW
    K2  metaqa x1f: the same
    K3  ena-gr minus en-gr, on each read: HELPS / HURTS / SAME (does the posterior typing add to the untyped GNN?)
    K4  webqsp selectf + fit: n-gr, en-gr and ena-gr minus none/rd (the untyped walk), on all rows and by seed-gold
        distance D (1 and 2): ABOVE / AT / BELOW
    K5  em-gr minus fg-gr, on each read: HELPS / HURTS / SAME (do the EM rounds add, with the transforms?)
    K6  n-gr minus chainscore19's f-gr, on each read: ABOVE / AT / BELOW (the GNN alone against the MLP alone)
    M1  metaqa: n-gr, en-gr and ena-gr against the twin, and their share of the twin's lead over rrf-s (HIGH if its
        interval lies at or above 0.5, LOW if below 0.25, else MID)
    Z1  webqsp selectf: the three GNN arms against l7g-j3a's band (0.20 to 0.25 R@5): ABOVE / WITHIN / BELOW
    R0  em-gr's per-row reads equal chainscore19's em-gr's (|diff| < 1e-6) on each read: REPRODUCES / DIFFERS
Decision. A GNN arm ABOVE em-gr on K1 carries into option B as f (if both are, ena-gr when K3 on webqsp selectf + fit
is HELPS, else en-gr). With none ABOVE on K1, a GNN arm ABOVE on K2 and not BELOW on K1 carries (the same tie rule on
K3 metaqa). Otherwise the MLP f carries. If en-gr and ena-gr are both BELOW on K1, the result goes to Swastik before
option B. Train-split rows throughout: a look, not a result.

    python outputs/mp_unified/chainscore20.py edges --base outputs/mp_unified/lean/cs19-mq-fit-id.npz \
        --out outputs/mp_unified/lean/cs20e-mq-fit-id.npz     (and each of chainscore19's other eight builds)
    python outputs/mp_unified/chainscore20.py train --arm ena-gr --fit outputs/mp_unified/lean/cs19-mq-fit-id.npz \
        --aug al4=outputs/mp_unified/lean/cs19-mq-fit-al4.npz (and sp4, mg3, rf) \
        --select outputs/mp_unified/lean/cs19-mq-select.npz --read metaqa=outputs/mp_unified/lean/cs19-mq-x1f.npz \
        --read webqsp=outputs/mp_unified/lean/cs19-wq.npz --read webqsp_sf=outputs/mp_unified/lean/cs19-wq-sf.npz \
        --edges fit=outputs/mp_unified/lean/cs20e-mq-fit-id.npz (and al4, sp4, mg3, rf, select, metaqa, webqsp,
        webqsp_sf; the MLP arms take none) --out outputs/mp_unified/lean/cs20-ena-gr.json \
        --rows-out outputs/mp_unified/lean/cs20-ena-gr.rows.npz --state-out outputs/mp_unified/lean/cs20-ena-gr.pt
    python outputs/mp_unified/chainscore20.py grade --arm ena-gr=outputs/mp_unified/lean/cs20-ena-gr.json (each arm) \
        --cs19 em-gr=outputs/mp_unified/lean/cs19-em-gr.json --cs19 f-gr=outputs/mp_unified/lean/cs19-f-gr.json \
        --read metaqa=... --read webqsp=... --read webqsp_sf=... --out outputs/mp_unified/lean/cs20.json
    python outputs/mp_unified/chainscore20.py --selftest
"""
import os
import sys

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")      # chainscore19's count, so a companion's transform and levels are its build's

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import shutil  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import chainpop17 as CP  # noqa: E402
import chainscore18 as CS  # noqa: E402
import chainscore19 as C19  # noqa: E402
import reltype11 as RT  # noqa: E402

LOOK, REL = RT.LOOK, RT.REL
SEED = C19.SEED
BOOT = C19.BOOT
EPOCHS, PER_EPOCH, BATCH = C19.EPOCHS, C19.PER_EPOCH, C19.BATCH
LR, WD, HID, T_EM, NEG = C19.LR, C19.WD, C19.HID, C19.T_EM, C19.NEG
NODEF, PAIRF = C19.NODEF, C19.PAIRF
NLAYER = 2
THETA_SCALE = 10.0     # ena's attention w = exp(THETA_SCALE theta a): Adam moves theta about 1e-3 a step
DEG = NODEF.index("deg")
J3A = C19.J3A
ARMS = ("em-gr", "fg-gr", "n-gr", "en-gr", "ena-gr")
GNN_ARMS = ("n-gr", "en-gr", "ena-gr")
log, unit, sha = RT.log, RT.unit, CS.sha


def arm_spec(arm):
    if arm not in ARMS:
        raise SystemExit(f"unknown arm {arm}")
    base = arm[:-3]
    if base in ("em", "fg"):
        return dict(C19.arm_spec(arm), gnn=False, att=False)
    return {"f": True, "g": base != "n", "T": 0 if base == "n" else T_EM, "gr": True, "gnn": True,
            "att": base == "ena"}


# ── companion builds: messages, chain type slots and the message-type incidence ─────────────────────────────────


def front(ba, root):
    """build19's front end on a build's own arguments: its chunks (with the carve of each), rrf column, transform and
    levels."""
    P = RT.proj_matrix()
    carves = [x for x in str(ba.carve).split("+") if x]
    paths, pcarve, infos, rec = [], [], [], None
    for k, cv in enumerate(carves):
        ps, info = RT.chunk_paths(ba.ds, cv, root, ba.limit)
        r_ = CP.read_record(ba.ds, cv, root)
        if rec is None:
            rec = r_
        elif r_ != rec:
            raise SystemExit(f"{ba.ds}={cv}: its record's rrf column or metrics differ from {carves[0]}'s")
        if infos and info["n_relations"] != infos[0]["n_relations"]:
            raise SystemExit(f"{ba.ds}={cv}: {info['n_relations']} relations, not {infos[0]['n_relations']}")
        paths += ps
        pcarve += [k] * len(ps)
        infos.append(info)
    n_rel = infos[0]["n_relations"]
    col = RT.Collector(P, (1,))
    col.add("read", paths)
    G, _rows, ginfo = col.finish()
    Pn = unit(G["proj"].astype(np.float32))
    ty0 = G["slots"][G["slots"] >= 0].astype(np.int64)
    if ty0.size and int(ty0.max()) >= n_rel:
        raise SystemExit(f"a relation id {int(ty0.max())} is past the record's {n_rel}")
    tf = C19.Transform(ba.transform, n_rel, G, Pn, np.bincount(ty0, minlength=n_rel) > 0)
    tt, present = tf.descriptors(G, Pn)
    ks = [int(k) for k in str(ba.ks).split(",") if k]
    levels = CP.hierarchy(tt, present, ks, tf.n_rel)
    return paths, pcarve, rec[0], tf, levels, ginfo


def row_messages(n, hl, tl, sl, exact):
    """The row's messages (src, d, dst), sorted, from its exact typed graph; each exact edge's message and relation."""
    gx = CP.typed_graph(n, hl, tl, sl, exact)
    src, zx, dst = gx[0], gx[1], gx[2]
    dx, rx = zx % 2, zx // 2
    ukey, e_of = np.unique((src * 2 + dx) * n + dst, return_inverse=True)
    return gx, ukey // (2 * n), (ukey // n) % 2, ukey % n, e_of.reshape(-1), rx, dx


def degree(n, es, ee):
    """chainscore19's degree feature from the messages: log(1 + distinct neighbours), self-loops left out."""
    nb = es != ee
    und = np.unique(es[nb] * n + ee[nb])
    return np.log1p(np.bincount(und // max(n, 1), minlength=n)) if n else np.zeros(0)


def slot_tables(levels, lev_off, P_, e_of, rx, dx):
    """A row's used slots (the slots of its chains' first NLAYER steps, sorted), its chains' local slot per step, the
    local none and dir slots, and the (message, local slot) incidence over the other levels, one entry per distinct
    pair."""
    used = np.unique(P_[P_ >= 0])

    def find(g):
        j = int(np.searchsorted(used, g))
        return j if j < used.size and used[j] == g else -1

    p_loc = np.where(P_ >= 0, np.searchsorted(used, np.maximum(P_, 0)), -1)
    sl_none = find(int(lev_off[0]))
    sl_dir = (find(int(lev_off[1])), find(int(lev_off[1]) + 1))
    ie, is_ = [], []
    for li in range(2, len(levels)):
        gg = lev_off[li] + levels[li].z(rx, dx)
        jj = np.searchsorted(used, gg)
        ok = jj < used.size
        ok[ok] = used[jj[ok]] == gg[ok]
        ie.append(e_of[ok])
        is_.append(jj[ok])
    S = max(int(used.size), 1)
    if ie:
        key = np.unique(np.concatenate(ie).astype(np.int64) * S + np.concatenate(is_))
    else:
        key = np.zeros(0, np.int64)
    return used, p_loc, sl_none, sl_dir, key // S, key % S


def collect_edges(paths, pcarve, levels, hops, cap, rrf_col, tf, keep_row, base, check_nodes=True):
    """collect19's pass without its features: rows, chains and their order are the build's, checked as they come."""
    L = len(levels)
    names = [lev.name for lev in levels]
    if names != [str(x) for x in base["level_names"]]:
        raise SystemExit(f"levels {names}, the build's {base['level_names'].tolist()}")
    if names[:2] != ["none", "dir"] or names[-1] != "exact":
        raise SystemExit(f"unexpected levels {names}")
    lev_off = np.r_[0, np.cumsum([lev.Kz for lev in levels])].astype(np.int64)
    Hs = min(hops, NLAYER)
    N = int(base["n"].size)
    V = [{} for _ in range(L)]
    R = {k: [] for k in ("e_src", "e_dst", "e_d", "slots", "p_slot", "i_e", "i_s")}
    cnt = {"e": [], "s": [], "i": []}
    sl_none, sl_dir = [], []
    st = {"rows": 0, "edges": 0, "pairs": 0, "slots": 0, "incidence": 0, "max_n": 0, "max_e": 0, "max_s": 0,
          "deg_max_abs_diff": 0.0, "chains_checked": 0, "nodes_checked": 0}
    seen = {}
    for ci, (p, cv) in enumerate(zip(paths, pcarve)):
        with np.load(p) as zf:
            c = {k: zf[k] for k in CP.KEYS}
        n_off = np.r_[0, np.cumsum(c["q_pool_size"])]
        e_off = np.r_[0, np.cumsum(c["q_edges"])]
        for i in range(c["q_pool_size"].size):
            gi = seen.get(cv, 0)
            seen[cv] = gi + 1
            if not keep_row(gi):
                continue
            r = st["rows"]
            if r >= N:
                raise SystemExit(f"more rows than the build's {N}")
            a, b = int(n_off[i]), int(n_off[i + 1])
            ea, eb = int(e_off[i]), int(e_off[i + 1])
            n = b - a
            hl, tl, sl, _u, _v = CP.row_triples(c, a, b, ea, eb)
            if tf.name == "rf":
                rrf = c["x"][a:b, rrf_col].astype(np.float64)
                pn = unit(c["proj"][a:b].astype(np.float32))
                n, hl, tl, sl, _pn, _rrf = tf.reify_row(n, hl, tl, sl, pn, rrf)
            elif tf.name != "id":
                pool = c["pool"][a:b].astype(np.int64)
                sl = tf.relabel(sl, pool[hl], pool[tl])
            if n != int(base["n"][r]):
                raise SystemExit(f"row {r}: pool size {n}, the build's {int(base['n'][r])}")
            sl0 = c["q_seed_local"][i]
            bk = c["q_seed_bucket"][i]
            seeds = np.unique(sl0[(sl0 >= 0) & (bk == 0)]).astype(np.int64)
            gx, es, ed, ee, e_of, rx, dx = row_messages(n, hl, tl, sl, levels[-1])
            xo = int(base["xn_off"][r])
            if n:
                dd = float(np.abs(degree(n, es, ee) - base["deg"][xo:xo + n].astype(np.float64)).max())
                if dd > 1e-2:
                    raise SystemExit(f"row {r}: degree feature differs from the build's by {dd}")
                st["deg_max_abs_diff"] = max(st["deg_max_abs_diff"], dd)
            ro0, ro1 = int(base["rowoff"][r]), int(base["rowoff"][r + 1])
            k = 0
            pslots = []
            for li, lev in enumerate(levels):
                g = gx if li == L - 1 else CP.typed_graph(n, hl, tl, sl, lev)
                ch, _capped = CP.walk(n, g, seeds, hops, cap)
                for types, nodes, mass in CP.candidates(ch, seeds):
                    key = lev.key(types)
                    x = V[li].get(key)
                    if x is None:
                        x = len(V[li])
                        V[li][key] = x
                    j = ro0 + k
                    if j >= ro1 or int(base["p_lev"][j]) != li or int(base["p_cid"][j]) != x:
                        raise SystemExit(f"row {r}, chain {k}: level {li} id {x}, not the build's")
                    no0, no1 = int(base["node_off"][j]), int(base["node_off"][j + 1])
                    if no1 - no0 != nodes.size:
                        raise SystemExit(f"row {r}, chain {k}: {nodes.size} end nodes, the build's {no1 - no0}")
                    if check_nodes:
                        if not (np.array_equal(base["node"][no0:no1], nodes.astype(np.int32))
                                and np.array_equal(base["mass"][no0:no1], mass.astype(np.float32))):
                            raise SystemExit(f"row {r}, chain {k}: end nodes or masses differ from the build's")
                        st["nodes_checked"] += int(nodes.size)
                    ps = np.full(Hs, -1, np.int64)
                    for h in range(min(Hs, len(types))):
                        ps[h] = lev_off[li] + int(types[h])
                    pslots.append(ps)
                    k += 1
            if ro0 + k != ro1:
                raise SystemExit(f"row {r}: {k} chains, the build's {ro1 - ro0}")
            st["chains_checked"] += k
            P_ = np.asarray(pslots, np.int64).reshape(-1, Hs)
            used, p_loc, s_n, s_d, ie, is_ = slot_tables(levels, lev_off, P_, e_of, rx, dx)
            R["e_src"].append(es.astype(np.int32))      # held as int32 until the row count is known
            R["e_dst"].append(ee.astype(np.int32))
            R["e_d"].append(ed.astype(np.int8))
            R["slots"].append(used.astype(np.int32))
            R["p_slot"].append(p_loc.astype(np.int32))
            R["i_e"].append(ie.astype(np.int32))
            R["i_s"].append(is_.astype(np.int32))
            sl_none.append(s_n)
            sl_dir.append(s_d)
            cnt["e"].append(es.size)
            cnt["s"].append(used.size)
            cnt["i"].append(ie.size)
            st["rows"] += 1
            st["edges"] += int(es.size)
            st["pairs"] += k
            st["slots"] += int(used.size)
            st["incidence"] += int(ie.size)
            st["max_n"] = max(st["max_n"], n)
            st["max_e"] = max(st["max_e"], int(es.size))
            st["max_s"] = max(st["max_s"], int(used.size))
        if (ci + 1) % 25 == 0 or ci + 1 == len(paths):
            log(f"  chunk {ci + 1}/{len(paths)}: {st['rows']} rows, {st['edges']} messages, {st['pairs']} chains, "
                f"{st['incidence']} incidences")
    if st["rows"] != N:
        raise SystemExit(f"{st['rows']} rows, the build's {N}")

    def small(x, m):
        return x.astype(np.uint16) if m < 65536 else x.astype(np.int32)

    arr = {"e_off": np.r_[0, np.cumsum(cnt["e"])].astype(np.int64),
           "s_off": np.r_[0, np.cumsum(cnt["s"])].astype(np.int64),
           "i_off": np.r_[0, np.cumsum(cnt["i"])].astype(np.int64),
           "e_src": small(np.concatenate(R["e_src"]), st["max_n"]),
           "e_dst": small(np.concatenate(R["e_dst"]), st["max_n"]),
           "e_d": np.concatenate(R["e_d"]).astype(np.int8),
           "slots": np.concatenate(R["slots"]).astype(np.int32),
           "p_slot": small(np.concatenate(R["p_slot"]) + 1, st["max_s"] + 1),
           "i_e": small(np.concatenate(R["i_e"]), st["max_e"]),
           "i_s": small(np.concatenate(R["i_s"]), st["max_s"]),
           "sl_none": np.asarray(sl_none, np.int32), "sl_dir": np.asarray(sl_dir, np.int32).reshape(-1, 2),
           "n": base["n"].astype(np.int32)}
    return arr, st, lev_off, Hs


def edges_cmd(a, root=LOOK):
    t0 = time.time()
    base_p = Path(a.base)
    keys = ("n", "rowoff", "p_lev", "p_cid", "node_off", "xn_off", "level_names") + (
        ("node", "mass") if not a.no_node_check else ())
    with np.load(base_p, allow_pickle=False) as z:
        meta = json.loads(str(z["meta"]))
        base = {k: z[k] for k in keys}
        base["deg"] = z["XN"][:, DEG].copy()
    if meta.get("look") != "chainscore19":
        raise SystemExit(f"{base_p}: not a chainscore19 build")
    ba = SimpleNamespace(**meta["args"])
    paths, pcarve, rrf_col, tf, levels, ginfo = front(ba, root)
    if tf.info != meta["transform"]:
        raise SystemExit(f"transform {tf.info}, the build's {meta['transform']}")
    log(f"{base_p.name}: {ba.ds}={ba.carve}, transform {ba.transform}, rows {ba.rows}, hops {ba.hops}; levels "
        + ", ".join(f"{lev.name} (Kz {lev.Kz})" for lev in levels))
    keep_row = {"all": lambda i: True, "even": lambda i: i % 2 == 0, "odd": lambda i: i % 2 == 1}[ba.rows]
    arr, st, lev_off, Hs = collect_edges(paths, pcarve, levels, ba.hops, ba.max_chains, rrf_col, tf, keep_row, base,
                                         check_nodes=not a.no_node_check)
    out_meta = {"look": "chainscore20-edges", "args": dict(vars(a)), "script_sha256": sha(__file__),
                "chainscore19_sha256": sha(C19.__file__), "chainpop17_sha256": sha(CP.__file__),
                "base": str(base_p), "base_sha256": sha(base_p), "base_args": meta["args"],
                "levels": [{"name": lev.name, "Kz": lev.Kz} for lev in levels], "lev_off": lev_off.tolist(),
                "steps": Hs, "graph": ginfo, "check": {"nodes": not a.no_node_check, **st},
                "seconds": round(time.time() - t0, 1)}
    arr["meta"] = np.asarray(json.dumps(out_meta))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + ".tmp.npz")
    np.savez(tmp, **arr)
    os.replace(tmp, out)
    log(f"edges: {st['rows']} rows, {st['edges']} messages (max {st['max_e']} a row), {st['pairs']} chains, "
        f"{st['slots']} slots, {st['incidence']} incidences; checked against the build ({st['nodes_checked']} end "
        f"nodes, degree within {st['deg_max_abs_diff']:.2g}) in {time.time() - t0:.0f}s -> {out}")
    return out_meta


def load_comp(p, d, base_sha):
    c = CS.load(p)
    if c["meta"].get("look") != "chainscore20-edges":
        raise SystemExit(f"{p}: not a chainscore20 companion")
    if c["meta"]["base_sha256"] != base_sha:
        raise SystemExit(f"{p}: built on a build with hash {c['meta']['base_sha256'][:12]}, not {base_sha[:12]}")
    if not np.array_equal(c["n"].astype(np.int64), d["n"].astype(np.int64)):
        raise SystemExit(f"{p}: its rows' pool sizes are not the build's")
    if c["p_slot"].shape[0] != int(d["rowoff"][-1]):
        raise SystemExit(f"{p}: {c['p_slot'].shape[0]} chains, the build's {int(d['rowoff'][-1])}")
    c["p_slot"] = c["p_slot"].astype(np.int64) - 1
    return c


# ── the model ───────────────────────────────────────────────────────────────────────────────────────────────────


def make_gnn(att):
    import torch
    nn = torch.nn

    class GNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.att = att
            self.inp = nn.Sequential(nn.Linear(len(NODEF), HID), nn.ReLU(), nn.Linear(HID, HID), nn.ReLU())
            self.lin = nn.ModuleList([nn.Linear(HID, HID) for _ in range(NLAYER)])
            self.msg = nn.ModuleList([nn.Linear(HID, HID, bias=False) for _ in range(2 * NLAYER)])
            self.out = nn.Linear(HID, 1)
            if att:
                self.theta = nn.Parameter(torch.zeros(NLAYER, 2))
                self.u = nn.Parameter(torch.zeros(NLAYER, 2, HID))

        def forward(self, X, E, A=None):
            h = self.inp(X)
            N = X.shape[0]
            for l_ in range(NLAYER):
                pre = self.lin[l_](h)
                for d in (0, 1):
                    src, dst, eix, cnt = E[d]
                    if src.numel() == 0:
                        continue
                    if A is None:
                        agg = torch.zeros(N, HID).index_add(0, dst, h[src]) / cnt.clamp_min(1.0)[:, None]
                    else:
                        a = A[l_][eix]
                        w = torch.exp(THETA_SCALE * self.theta[l_, d] * a)
                        num = torch.zeros(N, HID).index_add(0, dst, w[:, None] * h[src])
                        den = torch.zeros(N).index_add(0, dst, w)
                        agg = num / den.clamp_min(1e-12)[:, None]
                        abar = torch.zeros(N).index_add(0, dst, a)
                        pre = pre + torch.log1p(abar)[:, None] * self.u[l_, d]
                    pre = pre + self.msg[2 * l_ + d](agg)
                h = h + torch.relu(pre)
            return self.out(h).squeeze(-1)

    return GNN()


def make_model(arm, seed):
    """The MLP arms are chainscore19's model, made by its own code (so em-gr starts where chainscore19's did)."""
    import torch
    spec = arm_spec(arm)
    if not spec["gnn"]:
        m = C19.make_joint(C19.arm_spec(arm), seed)
        m.gnn, m.att = None, False
        return m

    class Joint(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.f = None
            self.gnn = make_gnn(spec["att"])
            self.g = (torch.nn.Sequential(torch.nn.Linear(len(PAIRF), HID), torch.nn.ReLU(), torch.nn.Linear(HID, HID),
                                          torch.nn.ReLU(), torch.nn.Linear(HID, 1)) if spec["g"] else None)
            inv1 = math.log(math.e - 1.0)
            self.beta_r = torch.nn.Parameter(torch.tensor(inv1))
            self.gamma_r = torch.nn.Parameter(torch.tensor(inv1))
            self.leps = torch.nn.Parameter(torch.tensor(math.log(C19.EPS0)))
            self.T, self.att = spec["T"], spec["att"]

    torch.manual_seed(seed)
    return Joint()


def make_batch20(items, builds, comps, st, pairs=True, train=True, gnn=False, att=False):
    """chainscore19's batch, and for the GNN the rows' messages as flat node indices split by direction (with each
    node's in-count per direction), the chains' slot per step and the incidence as flat slot indices. A sentinel slot
    (the last) holds 0 for a row with no none or dir chain."""
    bt = C19.make_batch(items, builds, st, pairs=pairs, train=train)
    if not gnn:
        return bt
    import torch
    t = torch.from_numpy
    items = np.asarray(items, np.int64).reshape(-1, 3)
    Lk = bt["Lk"]
    src, dst, dd, xn_, xd_, ie, is_ = [], [], [], [], [], [], []
    hp = [[] for _ in range(NLAYER)]
    hs = [[] for _ in range(NLAYER)]
    noff = eoff = soff = 0
    for j, (bi, i, _ri) in enumerate(items):
        d, c = builds[bi], comps[bi]
        n = int(d["n"][i])
        e0, e1 = int(c["e_off"][i]), int(c["e_off"][i + 1])
        ed = c["e_d"][e0:e1].astype(np.int64)
        src.append(c["e_src"][e0:e1].astype(np.int64) + noff)
        dst.append(c["e_dst"][e0:e1].astype(np.int64) + noff)
        dd.append(ed)
        if att:
            s0, s1 = int(c["s_off"][i]), int(c["s_off"][i + 1])
            p0, p1 = int(d["rowoff"][i]), int(d["rowoff"][i + 1])
            ps = c["p_slot"][p0:p1]
            for h in range(min(NLAYER, ps.shape[1])):
                m = ps[:, h] >= 0
                hp[h].append(j * Lk + np.flatnonzero(m))
                hs[h].append(soff + ps[m, h])
            sn = int(c["sl_none"][i])
            sdr = c["sl_dir"][i].astype(np.int64)
            xn_.append(np.full(e1 - e0, soff + sn if sn >= 0 else -1, np.int64))
            xd_.append(np.where(sdr[ed] >= 0, soff + sdr[ed], -1))
            i0, i1 = int(c["i_off"][i]), int(c["i_off"][i + 1])
            ie.append(eoff + c["i_e"][i0:i1].astype(np.int64))
            is_.append(soff + c["i_s"][i0:i1].astype(np.int64))
            soff += s1 - s0
        noff += n
        eoff += e1 - e0

    def cat(x):
        return np.concatenate(x).astype(np.int64) if x else np.zeros(0, np.int64)

    S_, D_, Dd = cat(src), cat(dst), cat(dd)
    E = {}
    for d_ in (0, 1):
        ix = np.flatnonzero(Dd == d_)
        E[d_] = (t(S_[ix]), t(D_[ix]), t(ix), t(np.bincount(D_[ix], minlength=noff).astype(np.float32)))
    bt["E"], bt["Nn"], bt["Ne"] = E, noff, eoff
    if att:
        xn, xd = cat(xn_), cat(xd_)
        xn[xn < 0] = soff
        xd[xd < 0] = soff
        bt.update({"S": soff, "hp": [t(cat(x)) for x in hp], "hs": [t(cat(x)) for x in hs], "xnone": t(xn),
                   "xdir": t(xd), "ie": t(cat(ie)), "is": t(cat(is_))})
    return bt


def marginals(lp, bt):
    """a_l(e) for l = 1..NLAYER over the batch's messages, from the chain posterior log p (rows x chains)."""
    import torch
    P = torch.exp(lp).reshape(-1)
    out = []
    for h in range(NLAYER):
        pi = torch.zeros(bt["S"] + 1)
        if bt["hp"][h].numel():
            pi = pi.index_add(0, bt["hs"][h], P[bt["hp"][h]])
        a = pi[bt["xnone"]] + pi[bt["xdir"]]
        if bt["ie"].numel():
            a = a.index_add(0, bt["ie"], pi[bt["is"]])
        out.append(a)
    return out


def forward20(model, bt):
    """s_T over each row's nodes and log p_T over its chains; the MLP arms are chainscore19's forward itself."""
    if model.gnn is None:
        return C19.forward(model, bt)
    import torch
    F_ = torch.nn.functional
    B, Ln, Lk = bt["B"], bt["Ln"], bt["Lk"]

    def fnode(A):
        h = model.gnn(bt["Xn"], bt["E"], A)
        return torch.zeros(B * Ln).index_copy(0, bt["npos"], h).reshape(B, Ln)

    if model.g is None:
        return fnode(None), None
    z = torch.full((B * Lk,), NEG)
    if bt["Xp"] is not None:
        z = z.index_copy(0, bt["ppos"], model.g(bt["Xp"]).squeeze(-1))
    lp0 = C19.masked_logsoftmax(z.reshape(B, Lk) - bt["lc"], bt["pm"])
    f0 = fnode(marginals(lp0, bt) if model.att else None)
    if bt["Xp"] is None:
        return f0 + F_.softplus(model.gamma_r) * model.leps, lp0
    gamma, eps = F_.softplus(model.gamma_r), torch.exp(model.leps)
    ip, inn, im = bt["ip"], bt["in"], bt["im"]

    def mstep(lp, f):
        agg = torch.zeros(B * Ln).index_add(0, inn, torch.exp(lp).reshape(-1)[ip] * im)
        return f + gamma * torch.log(eps + agg.reshape(B, Ln))

    if model.T == 0:
        return mstep(lp0, f0), lp0
    beta = F_.softplus(model.beta_r)
    s, lp = f0, lp0
    for _ in range(model.T):
        pi = torch.softmax(s.masked_fill(~bt["nm"], NEG), 1)
        ev = torch.zeros(B * Lk).index_add(0, ip, pi.reshape(-1)[inn] * im).reshape(B, Lk)
        lp = C19.masked_logsoftmax(lp0 + beta * torch.log(ev + 1e-8), bt["pm"])
        s = mstep(lp, fnode(marginals(lp, bt)) if model.att else f0)
    return s, lp


def coupling20(model):
    out = C19.coupling(model)
    if model.gnn is not None and model.att:
        out["theta"] = [[round(float(v), 5) for v in row] for row in model.gnn.theta.detach().tolist()]
        out["u_norm"] = [round(float(v), 5) for v in model.gnn.u.detach().norm(dim=-1).reshape(-1).tolist()]
    return out


def read20(model, d, c, st, ri=0, batch=BATCH):
    """chainscore19's read_joint through this file's batch and forward."""
    import torch
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
            bt = make_batch20(items, [d], [c], st, pairs=model.g is not None, train=False, gnn=gnn,
                              att=gnn and model.att)
            s, lp = forward20(model, bt)
            s = s.double().numpy()
            Pm = None if lp is None else np.exp(lp.double().numpy())
            for j, i in enumerate(rows):
                out[i] = C19.rank_row(d, i, s[j, :int(d["n"][i])])
                if Pm is not None:
                    p0, p1 = int(d["rowoff"][i]), int(d["rowoff"][i + 1])
                    if p1 > p0:
                        ml += np.bincount(plev[p0:p1], weights=Pm[j, :p1 - p0], minlength=L)
    return out, ml / max(N, 1)


def train20(arm, builds, comps, Sd, Sc, epochs, per_epoch, batch, seed, threads=1):
    """chainscore19's train_joint with this file's model, batch and forward (an MLP arm runs chainscore19's steps)."""
    import torch
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    spec = arm_spec(arm)
    use = list(range(len(builds))) if spec["gr"] else [0]
    st = C19.input_stats(builds[0])
    model = make_model(arm, seed)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    rng = np.random.default_rng(seed)
    hist, best, best_state = [], None, None
    for ep in range(epochs):
        t1 = time.time()
        model.train()
        items = C19.draw(rng, builds, use, per_epoch)
        tot, tn, tc, nb = 0.0, 0.0, 0.0, 0
        for b0 in range(0, len(items), batch):
            bt = make_batch20(items[b0:b0 + batch], builds, comps, st, pairs=spec["g"], gnn=spec["gnn"],
                              att=spec["att"])
            s, lp = forward20(model, bt)
            loss, ln, lc = C19.objective(s, lp, bt)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.detach())
            tn += ln
            tc += lc
            nb += 1
        model.eval()
        x, _ = read20(model, Sd, Sc, st)
        sr5 = float(x[:, 0].mean())
        hist.append({"epoch": ep, "loss": round(tot / max(nb, 1), 5), "node_loss": round(tn / max(nb, 1), 5),
                     "chain_loss": round(tc / max(nb, 1), 5), "select_r5": round(sr5, 5),
                     "by_build": np.bincount(items[:, 0], minlength=len(builds)).tolist(), **coupling20(model),
                     "seconds": round(time.time() - t1, 1)})
        log(f"    {arm} ep {ep}: loss {tot / max(nb, 1):.4f} (node {tn / max(nb, 1):.4f}, chain "
            f"{tc / max(nb, 1):.4f}) select R@5 {sr5:.4f} {coupling20(model)} {time.time() - t1:.0f}s")
        if best is None or sr5 > best[1]:
            best = (ep, sr5)
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    return model, {"history": hist, "best_epoch": best[0], "best_select_r5": round(best[1], 5), "stats": st,
                   "coupling": coupling20(model), "spec": spec}


# ── train (one arm) and grade ───────────────────────────────────────────────────────────────────────────────────


def train_cmd(a):
    t0 = time.time()
    spec = arm_spec(a.arm)
    res = {"look": "chainscore20", "arm": a.arm, "args": dict(vars(a)), "script_sha256": sha(__file__),
           "chainscore19_sha256": sha(C19.__file__), "chainscore18_sha256": sha(CS.__file__),
           "chainpop17_sha256": sha(CP.__file__), "inputs": {}, "inputs_sha256": {}, "reads": {},
           "rows_out": a.rows_out}
    edges = dict(x.split("=", 1) for x in a.edges)
    bpath = {"fit": a.fit, "select": a.select, **dict(x.split("=", 1) for x in a.aug),
             **dict(x.split("=", 1) for x in a.read)}
    if sorted(x.split("=", 1)[0] for x in a.aug) != sorted(C19.TRANSFORMS[1:]):
        raise SystemExit(f"every arm takes --aug for each of {C19.TRANSFORMS[1:]}")
    if spec["gnn"] and sorted(edges) != sorted(bpath):
        raise SystemExit(f"a GNN arm takes --edges for each of {sorted(bpath)}; given {sorted(edges)}")
    for p_ in list(bpath.values()) + (list(edges.values()) if spec["gnn"] else []):
        res["inputs_sha256"][p_] = sha(p_)

    def comp(name, d):
        if not spec["gnn"]:
            return None
        c = load_comp(edges[name], d, res["inputs_sha256"][bpath[name]])
        if not spec["att"]:     # the untyped GNN reads the messages only
            for k in ("slots", "p_slot", "i_e", "i_s"):
                c.pop(k)
        return c

    fit = C19.prep(CS.load(a.fit))
    builds, comps = [fit], [comp("fit", fit)]
    res["inputs"]["fit"] = fit["meta"]
    for x in a.aug:
        nm_, p_ = x.split("=", 1)
        d = C19.prep(CS.load(p_))
        if d["meta"]["transform"]["name"] != nm_:
            raise SystemExit(f"{p_} holds transform {d['meta']['transform']['name']}, not {nm_}")
        res["inputs"][nm_] = d["meta"]
        builds.append(d)
        comps.append(comp(nm_, d))
    Sd = CS.load(a.select)
    Sc = comp("select", Sd)
    res["inputs"]["select"] = Sd["meta"]
    res["train_rows"] = {("id" if k == 0 else b["meta"]["transform"]["name"]): int(b["train_rows"].size)
                         for k, b in enumerate(builds)}
    if spec["gnn"]:
        res["edges"] = {k: {"meta_check": c["meta"]["check"], "steps": c["meta"]["steps"]}
                        for k, c in zip(["fit"] + [x.split("=", 1)[0] for x in a.aug], comps)}
    model, info = train20(a.arm, builds, comps, Sd, Sc, a.epochs, a.per_epoch, a.batch, SEED)
    del builds, comps, fit, Sd, Sc
    res.update({"history": info["history"], "best_epoch": info["best_epoch"], "best_select_r5": info["best_select_r5"],
                "coupling": info["coupling"], "spec": info["spec"]})
    rows_out = {}
    for x in a.read:
        nm_, p_ = x.split("=", 1)
        t1 = time.time()
        d = CS.load(p_)
        c = comp(nm_, d)
        xr, ml = read20(model, d, c, info["stats"], batch=a.read_batch)
        rows_out[nm_] = xr.astype(np.float32)
        rows_out[f"{nm_}__D"] = d["D"]
        res["reads"][nm_] = C19.summarise(xr, d["D"], ml if model.g is not None else None, d["level_names"])
        res["inputs"][nm_] = d["meta"]
        log(f"  {a.arm} on {nm_}: {res['reads'][nm_]['mean']} mass {res['reads'][nm_]['mass_by_level']} "
            f"({time.time() - t1:.0f}s)")
        del d, c
    p = Path(a.rows_out)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.stem + ".tmp.npz")
    np.savez_compressed(tmp, **rows_out)
    os.replace(tmp, p)
    if a.state_out:
        import torch
        sp_ = Path(a.state_out)
        torch.save({"state": model.state_dict(), "stats": info["stats"], "spec": info["spec"], "arm": a.arm},
                   sp_.with_name(sp_.name + ".tmp"))
        os.replace(sp_.with_name(sp_.name + ".tmp"), sp_)
    res["seconds"] = round(time.time() - t0, 1)
    return res


def grade_cmd(a):
    t0 = time.time()
    arms, rows = {}, {}
    for flag, xs in (("arm", a.arm), ("cs19", a.cs19)):
        for x in xs:
            nm_, p_ = x.split("=", 1)
            js = json.loads(Path(p_).read_text(encoding="utf-8"))
            if js["arm"] != nm_:
                raise SystemExit(f"{p_} is arm {js['arm']}, not {nm_}")
            key = nm_ if flag == "arm" else f"cs19:{nm_}"
            with np.load(js["rows_out"]) as z:
                rows[key] = {k: z[k] for k in z.files}
            arms[key] = js
    refs, Ds = {}, {}
    for x in a.read:
        nm_, p_ = x.split("=", 1)
        d = CS.load(p_)
        refs[nm_] = CS.reference_arms(d)
        Ds[nm_] = d["D"]
        del d
    res = {"look": "chainscore20", "args": dict(vars(a)), "script_sha256": sha(__file__),
           "arms": {k: {"best_epoch": v["best_epoch"], "best_select_r5": v["best_select_r5"],
                        "coupling": v.get("coupling"), "reads": v["reads"]} for k, v in arms.items()},
           "refs": {nm_: {k: [round(float(u), 5) for u in x.mean(0)] for k, x in r.items()} for nm_, r in refs.items()},
           "verdicts": {}}
    hs, abv = ("HELPS", "HURTS", "SAME"), ("ABOVE", "BELOW", "AT")
    V = {"K1": {}, "K2": {}, "K3": {}, "K4": {}, "K5": {}, "K6": {}, "M1": {}, "Z1": {}, "R0": {}}

    def r5(arm, nm_):
        x = rows.get(arm, {}).get(nm_)
        return None if x is None else x[:, 0].astype(np.float64)

    def diff(x_, y_, nm_, idx, labels_):
        X_, Y_ = r5(x_, nm_), (r5(y_, nm_) if isinstance(y_, str) else y_)
        if X_ is None or Y_ is None:
            return None
        ci = CP.boot_mean(X_ - Y_, idx)
        return {"diff": ci, "verdict": C19.ci_label(ci, labels_)}

    for nm_ in refs:
        N = refs[nm_]["rrf_s"].shape[0]
        idx = np.random.default_rng(SEED).integers(0, N, size=(BOOT, N))
        for x_ in ("en-gr", "ena-gr"):
            e = diff(x_, "em-gr", nm_, idx, abv)
            if e and nm_ == "webqsp_sf":
                V["K1"][f"{x_} - em-gr"] = e
            if e and nm_ == "metaqa":
                V["K2"][f"{x_} - em-gr"] = e
        for tag, x_, y_, lab in (("K3", "ena-gr", "en-gr", hs), ("K5", "em-gr", "fg-gr", hs),
                                 ("K6", "n-gr", "cs19:f-gr", abv)):
            e = diff(x_, y_, nm_, idx, lab)
            if e:
                V[tag].setdefault(nm_, {})[f"{x_} - {y_}"] = e
        tw, fl = refs[nm_]["twin0"][:, 0], refs[nm_]["rrf_s"][:, 0]
        walk = refs[nm_]["none/rd"][:, 0]
        if nm_ == "webqsp_sf":
            for x_ in GNN_ARMS:
                X_ = r5(x_, nm_)
                if X_ is None:
                    continue
                e = {"all": diff(x_, walk, nm_, idx, abv)}
                for dd in (1, 2):
                    m = Ds[nm_] == dd
                    if m.any():
                        idd = np.random.default_rng(SEED).integers(0, int(m.sum()), size=(BOOT, int(m.sum())))
                        ci = CP.boot_mean((X_ - walk)[m], idd)
                        e[f"D{dd}"] = {"diff": ci, "verdict": C19.ci_label(ci, abv), "rows": int(m.sum())}
                V["K4"][f"{x_} - none/rd"] = e
        if nm_ == "metaqa":
            for x_ in GNN_ARMS:
                X_ = r5(x_, nm_)
                if X_ is None:
                    continue
                ci = CP.boot_mean(X_ - tw, idx)
                sh = CP.share(X_, fl, tw, idx)
                V["M1"][x_] = {"vs_twin": {"diff": ci, "verdict": C19.ci_label(ci, abv)}, "share": sh,
                               "share_verdict": "HIGH" if sh[1] >= 0.5 else "LOW" if sh[2] < 0.25 else "MID"}
        if nm_ == "webqsp":
            for x_ in GNN_ARMS:
                X_ = r5(x_, nm_)
                if X_ is None:
                    continue
                m = CP.boot_mean(X_, idx)
                sh = CP.share(X_, fl, tw, idx)
                V["Z1"][x_] = {"mean": m, "band": list(J3A),
                               "verdict": "ABOVE" if m[1] > J3A[1] else "BELOW" if m[2] < J3A[0] else "WITHIN",
                               "share": sh, "share_verdict": "HIGH" if sh[1] >= 0.5 else "LOW" if sh[2] < 0.25
                               else "MID", "minus_none_rd": CP.boot_mean(X_ - walk, idx)}
        if "em-gr" in rows and "cs19:em-gr" in rows and nm_ in rows["em-gr"] and nm_ in rows["cs19:em-gr"]:
            dif = float(np.abs(rows["em-gr"][nm_].astype(np.float64)
                               - rows["cs19:em-gr"][nm_].astype(np.float64)).max())
            V["R0"][nm_] = {"max_abs_diff": dif, "verdict": "REPRODUCES" if dif < 1e-6 else "DIFFERS"}
    res["verdicts"] = V
    res["decision"] = decide(V)
    res["seconds"] = round(time.time() - t0, 1)
    log(f"verdicts {json.dumps(V)}")
    log(f"decision {res['decision']}")
    return res


def decide(V):
    """The declared rule, from K1, K2 and K3."""
    k1 = {x: (V["K1"].get(f"{x} - em-gr") or {}).get("verdict") for x in ("en-gr", "ena-gr")}
    k2 = {x: (V["K2"].get(f"{x} - em-gr") or {}).get("verdict") for x in ("en-gr", "ena-gr")}
    k3 = {nm_: ((V["K3"].get(nm_) or {}).get("ena-gr - en-gr") or {}).get("verdict") for nm_ in ("webqsp_sf", "metaqa")}
    if None in k1.values():
        return {"carry": None, "why": "K1 incomplete", "report": False}

    def pick(cands, nm_):
        if len(cands) == 2:
            return "ena-gr" if k3[nm_] == "HELPS" else "en-gr"
        return cands[0]

    up = [x for x in ("en-gr", "ena-gr") if k1[x] == "ABOVE"]
    if up:
        return {"carry": pick(up, "webqsp_sf"), "why": "K1 ABOVE", "report": False}
    up2 = [x for x in ("en-gr", "ena-gr") if k2[x] == "ABOVE" and k1[x] != "BELOW"]
    if up2:
        return {"carry": pick(up2, "metaqa"), "why": "K2 ABOVE, K1 not BELOW", "report": False}
    both_below = all(k1[x] == "BELOW" for x in ("en-gr", "ena-gr"))
    return {"carry": "mlp", "why": "no GNN arm above em-gr" + (" (both BELOW on K1)" if both_below else ""),
            "report": both_below}


# ── selftest ────────────────────────────────────────────────────────────────────────────────────────────────────


def forward_np20(model, d, c, i, st, ri=0):
    """A float64 numpy reference of forward20 on one row of a GNN arm, from the model's weights."""
    def lin(m, X):
        y = X @ m.weight.detach().double().numpy().T
        return y + m.bias.detach().double().numpy() if m.bias is not None else y

    def mlp_np(seq, X, last_relu):
        ms = [m for m in seq if hasattr(m, "weight")]
        h = X
        for k, m in enumerate(ms):
            h = lin(m, h)
            if k < len(ms) - 1 or last_relu:
                h = np.maximum(h, 0.0)
        return h

    G = model.gnn
    a, n = int(d["xn_off"][i]), int(d["n"][i])
    Xn = ((d["XN"][a:a + n].astype(np.float32) - st["mn"]) / st["sn"]).astype(np.float32).astype(np.float64)
    e0, e1 = int(c["e_off"][i]), int(c["e_off"][i + 1])
    es, ee, ed = (c[k][e0:e1].astype(np.int64) for k in ("e_src", "e_dst", "e_d"))

    def gnn_np(A):
        h = mlp_np(G.inp, Xn, True)
        for l_ in range(NLAYER):
            pre = lin(G.lin[l_], h)
            for d_ in (0, 1):
                m = ed == d_
                if not m.any():
                    continue
                w = np.ones(int(m.sum())) if A is None else np.exp(THETA_SCALE * float(G.theta[l_, d_]) * A[l_][m])
                num = np.zeros((n, HID))
                np.add.at(num, ee[m], w[:, None] * h[es[m]])
                den = np.zeros(n)
                np.add.at(den, ee[m], w)
                pre = pre + lin(G.msg[2 * l_ + d_], num / np.maximum(den, 1e-12)[:, None])
                if A is not None:
                    ab = np.zeros(n)
                    np.add.at(ab, ee[m], A[l_][m])
                    pre = pre + np.log1p(ab)[:, None] * G.u[l_, d_].detach().double().numpy()
            h = h + np.maximum(pre, 0.0)
        return lin(G.out, h)[:, 0]

    if model.g is None:
        return gnn_np(None), None
    p0, p1 = int(d["rowoff"][i]), int(d["rowoff"][i + 1])
    Xp = ((C19.pair_raw(d, ri, p0, p1) - st["mp"]) / st["sp"]).astype(np.float32).astype(np.float64)
    z = mlp_np(model.g, Xp, False)[:, 0] - d["FI"][p0:p1, CS.LOGC].astype(np.float64)
    lp0 = z - np.logaddexp.reduce(z)
    no = d["node_off"][p0:p1 + 1]
    M = sp.csr_matrix((d["mass"][no[0]:no[-1]].astype(np.float64),
                       (np.repeat(np.arange(p1 - p0), np.diff(no)), d["node"][no[0]:no[-1]])), shape=(p1 - p0, n))
    ps = c["p_slot"][p0:p1]
    s0, s1 = int(c["s_off"][i]), int(c["s_off"][i + 1])
    i0, i1 = int(c["i_off"][i]), int(c["i_off"][i + 1])
    ie_, is_ = c["i_e"][i0:i1].astype(np.int64), c["i_s"][i0:i1].astype(np.int64)
    sn, sdr = int(c["sl_none"][i]), c["sl_dir"][i].astype(np.int64)

    def marg(lp):
        P = np.exp(lp)
        out = []
        for h in range(NLAYER):
            pi = np.zeros(s1 - s0 + 1)
            if h < ps.shape[1]:
                m = ps[:, h] >= 0
                np.add.at(pi, ps[m, h], P[m])
            av = pi[sn if sn >= 0 else -1] + pi[np.where(sdr[ed] >= 0, sdr[ed], -1)]
            np.add.at(av, ie_, pi[is_])
            out.append(av)
        return out

    sp1 = lambda x: math.log1p(math.exp(float(x)))  # noqa: E731
    gamma, eps, beta = sp1(model.gamma_r), math.exp(float(model.leps)), sp1(model.beta_r)
    f0 = gnn_np(marg(lp0) if model.att else None)
    mstep = lambda lp, f: f + gamma * np.log(eps + M.T @ np.exp(lp))  # noqa: E731
    if model.T == 0:
        return mstep(lp0, f0), lp0
    nsm = ~d["seed0"][a:a + n]
    s, lp = f0, lp0
    for _ in range(model.T):
        e = np.where(nsm, np.exp(s - s[nsm].max()), 0.0)
        lp = lp0 + beta * np.log(M @ (e / e.sum()) + 1e-8)
        lp = lp - np.logaddexp.reduce(lp)
        s = mstep(lp, gnn_np(marg(lp)) if model.att else f0)
    return s, lp


def brute_a(d, c, i, levels, hl, tl, sl, chains, P):
    """a_l(e) by its definition, with no slot table: for each message, the posterior mass of the row's chains whose
    l-th type is among the message's types at the chain's level."""
    n = int(d["n"][i])
    gx = CP.typed_graph(n, hl, tl, sl, levels[-1])
    rel = {}
    for s_, z_, t_ in zip(gx[0], gx[1], gx[2]):
        rel.setdefault((int(s_), int(z_) % 2, int(t_)), set()).add(int(z_) // 2)
    e0, e1 = int(c["e_off"][i]), int(c["e_off"][i + 1])
    msgs = list(zip(c["e_src"][e0:e1].tolist(), c["e_d"][e0:e1].tolist(), c["e_dst"][e0:e1].tolist()))
    assert set(msgs) == set(rel) and len(msgs) == len(rel)
    out = np.zeros((NLAYER, len(msgs)))
    for k, (s_, d_, t_) in enumerate(msgs):
        R_ = np.asarray(sorted(rel[(s_, d_, t_)]), np.int64)
        for li, lev in enumerate(levels):
            ty = set(int(z) for z in lev.z(R_, np.full(R_.size, d_)))
            for (cli, types), p_ in zip(chains, P):
                if cli != li:
                    continue
                for h in range(min(NLAYER, len(types))):
                    if int(types[h]) in ty:
                        out[h, k] += p_
    return out


def selftest():
    import torch
    P = RT.proj_matrix()
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        roots = {}
        for nm_, sd, rows in (("train", 11, 96), ("select", 12, 48), ("read", 13, 64)):
            root = td / nm_
            reld = td / f"{nm_}_rel"
            reld.mkdir(parents=True)
            _truth, names = CP.toy_world(root, np.random.default_rng(sd), P, rows=rows)
            np.save(reld / "toy_rel_embeddings.npy", names)
            roots[nm_] = (root, reld)

        def args(**kw):
            base = dict(ds="toy", carve="read", hops=2, ks="2", limit=None, max_chains=CP.MAX_CHAINS, regimes="1",
                        transform="id", rows="all", no_nodes=False)
            base.update(kw)
            return SimpleNamespace(**base)

        # 1. chainscore19's builds and a companion for each, checked against its build
        tag = {}
        for nm_, regs, tfn, rws, hops in (("train", "1,0.25,0", "id", "all", 2), ("select", "1", "id", "all", 2),
                                          ("read", "1", "id", "all", 3), ("train", "1,0", "al4", "even", 2),
                                          ("train", "1,0", "sp4", "odd", 2), ("train", "1,0", "mg3", "all", 3),
                                          ("train", "1,0", "rf", "even", 3)):
            root, reld = roots[nm_]
            name = nm_ if tfn == "id" else f"train-{tfn}"
            C19.build19(args(regimes=regs, transform=tfn, rows=rws, hops=hops, out=str(td / f"{name}.npz")),
                        root=root, rel_dir=reld)
            m = edges_cmd(SimpleNamespace(base=str(td / f"{name}.npz"), out=str(td / f"{name}.e.npz"),
                                          no_node_check=False), root=root)
            tag[name] = root
            ck = m["check"]
            print(f"toy {name}: {ck['rows']} rows, {ck['edges']} messages, {ck['pairs']} chains, {ck['slots']} slots, "
                  f"{ck['incidence']} incidences; steps {m['steps']}; {ck['nodes_checked']} end nodes checked")
        shutil.copytree(roots["read"][0] / "toy" / "read", roots["read"][0] / "toy" / "read2")
        C19.build19(args(carve="read+read2", hops=3, out=str(td / "read2.npz")), root=roots["read"][0],
                    rel_dir=roots["read"][1])
        edges_cmd(SimpleNamespace(base=str(td / "read2.npz"), out=str(td / "read2.e.npz"), no_node_check=False),
                  root=roots["read"][0])
        # 2. a companion refuses a build it does not match
        bad = CS.load(td / "select.npz")
        ro = bad["rowoff"]
        i = int(np.flatnonzero(np.diff(ro) >= 2)[0])
        bad["p_cid"] = bad["p_cid"].copy()
        bad["p_cid"][ro[i]], bad["p_cid"][ro[i] + 1] = bad["p_cid"][ro[i] + 1], bad["p_cid"][ro[i]]
        if bad["p_cid"][ro[i]] == bad["p_cid"][ro[i] + 1]:
            bad["p_cid"][ro[i]] += 1
        bad["meta"] = np.asarray(json.dumps(bad["meta"]))
        np.savez(td / "bad.npz", **bad)
        try:
            edges_cmd(SimpleNamespace(base=str(td / "bad.npz"), out=str(td / "bad.e.npz"), no_node_check=False),
                      root=roots["select"][0])
            raise AssertionError("a companion took a build whose chains differ")
        except SystemExit as e:
            print(f"toy: a companion refuses a mismatched build ({e})")
        try:
            load_comp(td / "select.e.npz", CS.load(td / "select.npz"), "0" * 64)
            raise AssertionError("a companion was loaded against another build's hash")
        except SystemExit as e:
            print(f"toy: training refuses a companion of another build ({str(e)[:60]}...)")
        # 3. a_l(e) through the slot tables equals its definition, on id, al4 and rf rows, at a random posterior
        rng = np.random.default_rng(7)
        worst = 0.0
        for name, nm_ in (("train", "train"), ("train-al4", "train"), ("train-rf", "train"), ("read", "read")):
            d = CS.load(td / f"{name}.npz")
            c = load_comp(td / f"{name}.e.npz", d, sha(td / f"{name}.npz"))
            ba = SimpleNamespace(**d["meta"]["args"])
            paths, pcarve, rrf_col, tf, levels, _g = front(ba, tag[name])
            keep = {"all": lambda i: True, "even": lambda i: i % 2 == 0, "odd": lambda i: i % 2 == 1}[ba.rows]
            r = gi = 0
            for p in paths:
                with np.load(p) as zf:
                    cc = {k: zf[k] for k in CP.KEYS}
                n_off = np.r_[0, np.cumsum(cc["q_pool_size"])]
                e_off = np.r_[0, np.cumsum(cc["q_edges"])]
                for q in range(cc["q_pool_size"].size):
                    gi += 1
                    if not keep(gi - 1):
                        continue
                    if r >= 6:
                        break
                    a_, b_ = int(n_off[q]), int(n_off[q + 1])
                    n = b_ - a_
                    hl, tl, sl, _u, _v = CP.row_triples(cc, a_, b_, int(e_off[q]), int(e_off[q + 1]))
                    if tf.name == "rf":
                        n, hl, tl, sl, _p, _x = tf.reify_row(n, hl, tl, sl, unit(cc["proj"][a_:b_].astype(np.float32)),
                                                             cc["x"][a_:b_, rrf_col].astype(np.float64))
                    elif tf.name != "id":
                        pool = cc["pool"][a_:b_].astype(np.int64)
                        sl = tf.relabel(sl, pool[hl], pool[tl])
                    sl0, bk = cc["q_seed_local"][q], cc["q_seed_bucket"][q]
                    seeds = np.unique(sl0[(sl0 >= 0) & (bk == 0)]).astype(np.int64)
                    chains = []
                    for li, lev in enumerate(levels):
                        ch, _cap = CP.walk(n, CP.typed_graph(n, hl, tl, sl, lev), seeds, ba.hops, ba.max_chains)
                        chains += [(li, types) for types, _nd, _m in CP.candidates(ch, seeds)]
                    k = int(d["rowoff"][r + 1] - d["rowoff"][r])
                    assert len(chains) == k
                    w = rng.random(k) ** 3
                    Pp = w / w.sum()
                    ref = brute_a(d, c, r, levels, hl, tl, sl, chains, Pp)
                    items = np.asarray([[0, r, 0]])
                    bt = make_batch20(items, [d], [c], C19.input_stats(d), pairs=True, train=False, gnn=True,
                                      att=True)
                    lp = torch.full((1, bt["Lk"]), NEG)
                    lp[0, :k] = torch.log(torch.from_numpy(Pp).float())
                    got = torch.stack(marginals(lp, bt)).double().numpy()
                    worst = max(worst, float(np.abs(got - ref).max()))
                    r += 1
        assert worst < 1e-5, worst
        print(f"toy: a_l(e) from the slot tables equals its definition (max |diff| {worst:.1e})")
        # 4. this file's loop with the MLP f is chainscore19's, step for step
        tr = C19.prep(CS.load(td / "train.npz"))
        aug = [C19.prep(CS.load(td / f"train-{t}.npz")) for t in C19.TRANSFORMS[1:]]
        sel = CS.load(td / "select.npz")
        rd = CS.load(td / "read.npz")
        for arm in ("em-gr", "fg-gr"):
            m19, i19 = C19.train_joint(arm, [tr] + aug, sel, 3, 192, 16, SEED)
            x19, _ = C19.read_joint(m19, rd, i19["stats"])
            m20, i20 = train20(arm, [tr] + aug, [None] * 5, sel, None, 3, 192, 16, SEED)
            x20, _ = read20(m20, rd, None, i20["stats"])
            strip = lambda h: [{k: v for k, v in e.items() if k != "seconds"} for e in h]  # noqa: E731
            assert strip(i19["history"]) == strip(i20["history"]) and np.array_equal(x19, x20), arm
            print(f"toy {arm}: this file's loop reproduces chainscore19's (read {np.round(x20.mean(0), 4).tolist()})")
        # 5. the GNN arms: deterministic, the loss falls, the batched forward matches a float64 reference
        comps = [load_comp(td / "train.e.npz", tr, sha(td / "train.npz"))] + [
            load_comp(td / f"train-{t}.e.npz", d_, sha(td / f"train-{t}.npz")) for t, d_ in zip(C19.TRANSFORMS[1:], aug)]
        sc = load_comp(td / "select.e.npz", sel, sha(td / "select.npz"))
        rc = load_comp(td / "read.e.npz", rd, sha(td / "read.npz"))
        worst = 0.0
        for arm in GNN_ARMS:
            model, info = train20(arm, [tr] + aug, comps, sel, sc, 4, 192, 16, SEED)
            x, _ = read20(model, rd, rc, info["stats"])
            h_ = info["history"]
            assert h_[-1]["loss"] < h_[0]["loss"], (arm, h_)
            model2, info2 = train20(arm, [tr] + aug, comps, sel, sc, 4, 192, 16, SEED)
            x2, _ = read20(model2, rd, rc, info2["stats"])
            strip = lambda h: [{k: v for k, v in e.items() if k != "seconds"} for e in h]  # noqa: E731
            assert strip(info2["history"]) == strip(h_) and np.array_equal(x, x2), arm
            rows_ = np.arange(min(12, rd["n"].size))
            items = np.c_[np.zeros(rows_.size, np.int64), rows_, np.zeros(rows_.size, np.int64)]
            with torch.no_grad():
                bt = make_batch20(items, [rd], [rc], info["stats"], pairs=model.g is not None, train=False, gnn=True,
                                  att=model.att)
                s, lp = forward20(model, bt)
            for j, i in enumerate(rows_):
                n = int(rd["n"][i])
                k = int(rd["rowoff"][i + 1] - rd["rowoff"][i])
                s_np, lp_np = forward_np20(model, rd, rc, i, info["stats"])
                ds_ = float(np.abs(s[j, :n].double().numpy() - s_np).max() / max(1.0, np.abs(s_np).max()))
                dp_ = 0.0 if lp_np is None else float(np.abs(np.exp(lp[j, :k].double().numpy()) - np.exp(lp_np)).max())
                worst = max(worst, ds_, dp_)
            print(f"toy {arm}: loss {h_[0]['loss']} -> {h_[-1]['loss']}, best epoch {info['best_epoch']}, read "
                  f"{np.round(x.mean(0), 4).tolist()}, coupling {info['coupling']}")
        assert worst < 1e-4, worst
        print(f"toy: GNN arms deterministic, their loss falls; forward vs the numpy reference {worst:.2e}")
        # 6. the commands end to end: chainscore19's em-gr and f-gr, this file's five arms, the grade
        L_ = td / "out"
        L_.mkdir()
        augf = [f"--aug {t}={td}/train-{t}.npz" for t in C19.TRANSFORMS[1:]]
        reads = [f"metaqa={td}/read.npz", f"webqsp={td}/read.npz", f"webqsp_sf={td}/read2.npz"]
        common = ["--fit", f"{td}/train.npz", *" ".join(augf).split(), "--select", f"{td}/select.npz"]
        for r_ in reads:
            common += ["--read", r_]
        for arm in ("em-gr", "f-gr"):
            assert C19.main(["train", "--arm", arm, *common, "--epochs", "2", "--per-epoch", "128", "--batch", "16",
                             "--out", f"{L_}/cs19-{arm}.json", "--rows-out", f"{L_}/cs19-{arm}.rows.npz"]) == 0
        eds = [f"fit={td}/train.e.npz", f"select={td}/select.e.npz", f"metaqa={td}/read.e.npz",
               f"webqsp={td}/read.e.npz", f"webqsp_sf={td}/read2.e.npz"] + [
            f"{t}={td}/train-{t}.e.npz" for t in C19.TRANSFORMS[1:]]
        for arm in ARMS:
            ex = []
            if arm_spec(arm)["gnn"]:
                for e_ in eds:
                    ex += ["--edges", e_]
            assert main(["train", "--arm", arm, *common, *ex, "--epochs", "2", "--per-epoch", "128", "--batch", "16",
                         "--out", f"{L_}/cs20-{arm}.json", "--rows-out", f"{L_}/cs20-{arm}.rows.npz",
                         "--state-out", f"{L_}/cs20-{arm}.pt"]) == 0
        g = ["grade"]
        for arm in ARMS:
            g += ["--arm", f"{arm}={L_}/cs20-{arm}.json"]
        g += ["--cs19", f"em-gr={L_}/cs19-em-gr.json", "--cs19", f"f-gr={L_}/cs19-f-gr.json"]
        for r_ in reads:
            g += ["--read", r_]
        assert main(g + ["--out", f"{L_}/cs20.json"]) == 0
        res = json.loads((L_ / "cs20.json").read_text(encoding="utf-8"))
        V = res["verdicts"]
        assert all(V["R0"][k]["verdict"] == "REPRODUCES" for k in ("metaqa", "webqsp", "webqsp_sf")), V["R0"]
        assert set(V["K1"]) == {"en-gr - em-gr", "ena-gr - em-gr"} and set(V["K4"]) == {f"{x} - none/rd" for x in GNN_ARMS}
        print(f"toy commands: R0 {[(k, v['verdict']) for k, v in V['R0'].items()]}, K1 "
              f"{[(k, v['verdict']) for k, v in V['K1'].items()]}, decision {res['decision']}")
    print("selftest: companions reproduce their builds' rows and chains and refuse others; a_l(e) equals its "
          "definition; the MLP arms run chainscore19's steps exactly; the GNN arms are deterministic, their loss falls "
          "and their batched forward matches a float64 reference; the commands run end to end and R0 reproduces. "
          "all checks passed")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    e = sub.add_parser("edges")
    e.add_argument("--base", required=True, help="a chainscore19 build")
    e.add_argument("--out", required=True)
    e.add_argument("--no-node-check", action="store_true", help="skip comparing chains' end nodes and masses")
    t = sub.add_parser("train")
    t.add_argument("--arm", required=True, choices=ARMS)
    t.add_argument("--fit", required=True)
    t.add_argument("--aug", action="append", default=[])
    t.add_argument("--select", required=True)
    t.add_argument("--read", action="append", default=[])
    t.add_argument("--edges", action="append", default=[], help="NAME=COMPANION for fit, select, each aug and read")
    t.add_argument("--epochs", type=int, default=EPOCHS)
    t.add_argument("--per-epoch", type=int, default=PER_EPOCH)
    t.add_argument("--batch", type=int, default=BATCH)
    t.add_argument("--read-batch", type=int, default=BATCH)
    t.add_argument("--out", required=True)
    t.add_argument("--rows-out", required=True)
    t.add_argument("--state-out")
    g = sub.add_parser("grade")
    g.add_argument("--arm", action="append", required=True)
    g.add_argument("--cs19", action="append", default=[])
    g.add_argument("--read", action="append", required=True)
    g.add_argument("--out", required=True)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.cmd == "edges":
        edges_cmd(a)
        return 0
    if a.cmd in ("train", "grade"):
        res = train_cmd(a) if a.cmd == "train" else grade_cmd(a)
        p = Path(a.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + ".tmp")
        tmp.write_text(json.dumps(res, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)),
                       encoding="utf-8")
        os.replace(tmp, p)
        return 0
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
