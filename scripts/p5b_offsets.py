"""P5b: learned offsets g(question, seed, graph features), read as a reranker on the final pools
(docs/P5B_LEARNED_OFFSETS.md). A diagnostic: no ranking model trains or reads, and nothing is adopted.

    python scripts/p5b_offsets.py base DATASET [--host] [--threads 2] [--limit Q]
    python scripts/p5b_offsets.py train [DATASET --limit Q] [--host] [--device cuda] [--threads 2]
    python scripts/p5b_offsets.py read DATASET [--host] [--device cuda] [--threads 2] [--limit Q]
    python scripts/p5b_offsets.py verdict
    python scripts/p5b_offsets.py --selftest

base: on a training dataset's fit carve, P5a's sample. For each question: its frame vector; its seeds with their
first-stage ranks and graph features (m_s on both spaces, the degrees); and its candidates (the final pool, from
P1c2's KIND_T file, minus the seeds) with the golds among them. Questions without a counted gold are skipped and
counted.
train: G on P5a's space V and G0 on its start V0, from the five base files.
read: on the s1eval sample (P5a's read1 sample), every order (F, H2, G, G0, F0, NEAR, FN, COS) and both counts (low,
pool).
verdict: section 5, on the six read records.
--limit: a development run. It writes nothing and logs no measure: only that every step ran, the checks, and that the
fit's loss is finite. Where a stage's inputs are not on disk, it builds them in memory: P5a's start from this corpus
for both spaces; today's pools for the final ones where P1c2's file is absent; and, for train and read, this dataset's
base and a one-epoch g.
"""
from __future__ import annotations

import os
import sys

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import p5a_space as A  # noqa: E402  (P5a's filed script, imported unchanged)
import step4c_walk_pools as S4C  # noqa: E402  (the pools file's loader and pool_sha, unchanged)

DECLARED = "docs/P5B_LEARNED_OFFSETS.md"
P1C2_DECLARED = "docs/P1C2_POOLS_KIND_T.md"
CONSTRUCTION = "KIND_T"
OUT = ROOT / "outputs" / "p5b"
POOLS = ROOT / "outputs" / "p1c2" / "pools"
SIX, TRAIN5 = A.SIX, A.TRAIN5
SPACES = ("V", "V0")
MODEL_OF = {"V": "G", "V0": "G0"}
DIM = A.DIM                                  # 64
HEADS = 4
WIDTH = 256
D_IN = (DIM + 1) + 3 * DIM + 3 + 2           # 262
N_WEIGHTS = 200_465
RANK_C = 60                                  # the substrate's RRF constant: the seeds' rank features and the fusion
KAPPA_PAIRS = 100_000
EPOCHS = 30
DEV_EPOCHS = 1
BATCH = 32
LR = 1e-3
MASK = -1e30                                 # a padded seed's or candidate's logit: finite, so no gradient is NaN
GRID = A.GRID["low"]                         # 5, 10, 25, 50, 100
ORDERS = ("F", "H2", "G", "G0", "F0", "NEAR", "FN", "COS")
READS = ("low", "pool")
RINGS = A.RING_CLASSES
INDEX_TIME = ("m_s (each seed's mean neighbour displacement over the three families' union) and the per-family "
              "degrees depend on the node and the graph alone: rows of an index-time table, flagged as index-time "
              "graph aggregations; nothing comes from the question's scored rows")
log, utc, sha_file, write_json, save_npz, ids_sha = A.log, A.utc, A.sha_file, A.write_json, A.save_npz, A.ids_sha


def script_sha256():
    return hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def out_paths(out=OUT):
    return {"base": out / "base", "g": out / "g.pt", "train": out / "train.json", "verdict": out / "verdict.json",
            "verdict_md": out / "verdict.md"}


# ── the space's tables ───────────────────────────────────────────────────────


def have_tables(name):
    return all(p.exists() for p in (A.FRAME, A.SPACE / f"u_{name}.npy", A.SPACE / f"u0_{name}.npy",
                                    A.SPACE / f"space_{name}.json"))


def filed_tables(name):
    """P5a's frame and node tables of a corpus (V: u, V0: u0), memory-mapped, checked by sha256 against P5a's space
    record."""
    sp = A.SPACE / f"space_{name}.json"
    rec = json.loads(sp.read_text(encoding="utf-8"))
    shas = {"frame": sha_file(A.FRAME), "V": sha_file(A.SPACE / f"u_{name}.npy"),
            "V0": sha_file(A.SPACE / f"u0_{name}.npy"), "space_record": sha_file(sp)}
    if (shas["frame"], shas["V"], shas["V0"]) != (rec["frame_sha256"], rec["u_sha256"], rec["u0_sha256"]):
        raise SystemExit(f"{name}: the frame or a node table is not P5a's filed one")
    mu, P = A.load_frame()
    tables = {"V": np.load(A.SPACE / f"u_{name}.npy", mmap_mode="r"),
              "V0": np.load(A.SPACE / f"u0_{name}.npy", mmap_mode="r")}
    return (mu, P), tables, shas


def tables_for(name, emb, limit):
    if have_tables(name):
        frame, tables, shas = filed_tables(name)
        return frame, tables, shas, "filed"
    if not limit:
        raise SystemExit(f"{name}: P5a's frame or node tables are not on disk")
    mu, P, u0 = A.dev_frame_u0(emb)
    return (mu, P), {"V": u0, "V0": u0}, {}, "development (P5a's start from this corpus stands in for both spaces)"


def kappa_of(U, pairs=KAPPA_PAIRS):
    """1 / the median squared distance between random node pairs (seed 0): a corpus's label-free scale."""
    n = U.shape[0]
    rng = np.random.default_rng(0)
    a, b = rng.integers(0, n, pairs), rng.integers(0, n, pairs)
    keep = a != b
    d2 = ((rows_of(U, a[keep], np.float64) - rows_of(U, b[keep], np.float64)) ** 2).sum(1)
    return float(1.0 / np.median(d2))


def rows_of(U, ids, dtype):
    """U's rows at ids, read in ascending order (memory-mapped tables), returned in the order of ids."""
    ids = np.asarray(ids, np.int64)
    out = np.empty((ids.size, U.shape[1]), dtype)
    o = np.argsort(ids, kind="stable")
    for lo in range(0, ids.size, 1 << 20):
        sl = o[lo:lo + (1 << 20)]
        out[sl] = U[ids[sl]]
    return out


def mean_rows(U, ids):
    """The mean of U's rows at ids (sorted, unique) in float64, read in blocks: a hub's neighbours never sit in memory
    at once."""
    s = np.zeros(U.shape[1], np.float64)
    for lo in range(0, ids.size, 1 << 18):
        s += np.asarray(U[ids[lo:lo + (1 << 18)]], np.float64).sum(0)
    return s / ids.size


# ── the final pools (P1c2's files) ───────────────────────────────────────────


def pools_file(name, carve, root=POOLS):
    """P1c2's pools file for one carve: (filed arrays, entry, manifest sha256), the manifest P1c2's KIND_T one and the
    file's sha256 the manifest's (step 4c's loader)."""
    d = Path(root) / name
    man = json.loads((d / "pools.json").read_text(encoding="utf-8"))
    if man.get("declared_in") != P1C2_DECLARED or man.get("construction") != CONSTRUCTION:
        raise SystemExit(f"{d / 'pools.json'}: not P1c2's {CONSTRUCTION} manifest")
    filed, ent, msha = S4C.load_filed(name, carve, d)
    if ent.get("construction") != CONSTRUCTION:
        raise SystemExit(f"{d}: {name}/{carve} is {ent.get('construction')}'s file, not {CONSTRUCTION}'s")
    return filed, ent, msha


def has_pools_file(name, carve, root=POOLS):
    m = Path(root) / name / "pools.json"
    if not m.exists():
        return False
    ent = json.loads(m.read_text(encoding="utf-8")).get("files", {}).get(f"{name}__{carve}")
    return ent is not None and (Path(root) / name / ent["file"]).exists()


class Finals:
    """Each question's final pool: base and seeds and the filed expansion, after section 5's checks."""

    def __init__(self, filed):
        self.f = filed
        self.at = {str(q): k for k, q in enumerate(filed["ids"].tolist())}
        self.expect = S4C.expected_rows(filed)

    def pool(self, qid, bs, inc):
        f = self.f
        k = self.at.get(str(qid))
        if k is None:
            raise SystemExit(f"{qid}: not in the pools file")
        b = f["bs_nodes"][f["bs_ptr"][k]:f["bs_ptr"][k + 1]].astype(np.int64)
        if not np.array_equal(b, np.asarray(bs, np.int64)):
            raise SystemExit(f"{qid}: base and seeds are not the filed ones")
        inc = np.asarray(inc, np.int64)
        sha = f["inc_sha"][k]
        sha = sha.decode() if isinstance(sha, bytes) else str(sha)
        if inc.size != int(f["inc_size"][k]) or S4C.pool_sha(inc) != sha:
            raise SystemExit(f"{qid}: today's pool is not the filed one")
        exp = f["nodes"][f["ptr"][k]:f["ptr"][k + 1]].astype(np.int64)
        b_q = int(f["B_q"][k])
        if b_q != inc.size - b.size or exp.size != int(f["count"][k]) or exp.size > b_q:
            raise SystemExit(f"{qid}: the expansion is not the filed length")
        if np.unique(exp).size != exp.size or np.isin(exp, b).any():
            raise SystemExit(f"{qid}: the expansion repeats a node or holds one of base and seeds")
        p = np.union1d(b, exp)
        if p.size != int(self.expect[k]):
            raise SystemExit(f"{qid}: the final pool has {p.size} nodes, not {int(self.expect[k])}")
        return p


def pools_said(info):
    return info if isinstance(info, str) else "P1c2's KIND_T file, every check held"


def finals_for(name, limit, root=POOLS):
    """(carve -> Finals, or None in a development run without P1c2's file; what each carve's pools were)."""
    info = {}

    def get(carve):
        if has_pools_file(name, carve, root):
            filed, ent, msha = pools_file(name, carve, root)
            info[carve] = {"file": ent["file"], "sha256": ent["sha256"], "manifest_sha256": msha,
                           "construction": ent["construction"], "questions": ent["questions"]}
            return Finals(filed)
        if not limit:
            raise SystemExit(f"{name}/{carve}: P1c2's pools file is not on disk")
        info[carve] = "development: today's pools stand in for the final ones"
        return None

    return get, info


# ── a question's inputs ──────────────────────────────────────────────────────


def first_ranks(lst, nodes):
    """Each node's 1-based rank in the list (its first occurrence), or the list's depth + 1 where it is absent."""
    lst = np.asarray(lst, np.int64)
    nodes = np.asarray(nodes, np.int64)
    out = np.full(nodes.size, lst.size + 1, np.int64)
    if lst.size and nodes.size:
        u, idx = np.unique(lst, return_index=True)
        k = np.minimum(np.searchsorted(u, nodes), u.size - 1)
        hit = u[k] == nodes
        out[hit] = idx[k[hit]] + 1
    return out


def seed_graph(adj, tables, seeds, cache):
    """Per seed: its degree in each family, and per space m_s, the mean of u_v - u_s over its neighbours (the three
    families' union, the seed excluded; zero without one). Both depend on the node and the graph alone: rows of an
    index-time table."""
    deg = np.zeros((seeds.size, 3), np.float32)
    m = {sp: np.zeros((seeds.size, DIM), np.float32) for sp in tables}
    for i, s in enumerate(seeds.tolist()):
        hit = cache.get(s)
        if hit is None:
            nbs, dg = [], np.zeros(3, np.float32)
            for f, (ip, col) in enumerate(adj):
                nb = np.unique(np.asarray(col[ip[s]:ip[s + 1]], np.int64))
                nb = nb[nb != s]
                dg[f] = nb.size
                nbs.append(nb)
            nb = np.unique(np.concatenate(nbs))
            ms = {}
            for sp, U in tables.items():
                ms[sp] = ((mean_rows(U, nb) - np.asarray(U[s], np.float64)).astype(np.float32)
                          if nb.size else np.zeros(DIM, np.float32))
            hit = (dg, ms)
            cache[s] = hit
        deg[i] = hit[0]
        for sp in tables:
            m[sp][i] = hit[1][sp]
    return deg, m


def features(x_rep, us, ms, deg, rd, rs):
    """g's inputs, one row per (question, seed), 262 wide: [x(q), u_s, x(q)[:64] - u_s, m_s, log(1 + degree) per
    family, 1/(60 + dense rank), 1/(60 + SPLADE rank)]."""
    x_rep = np.asarray(x_rep, np.float32)
    us = np.asarray(us, np.float32)
    f = np.concatenate([x_rep, us, x_rep[:, :DIM] - us, np.asarray(ms, np.float32),
                        np.log1p(np.asarray(deg, np.float32)),
                        (1.0 / (RANK_C + np.asarray(rd, np.float64)))[:, None].astype(np.float32),
                        (1.0 / (RANK_C + np.asarray(rs, np.float64)))[:, None].astype(np.float32)], axis=1)
    if f.shape[1] != D_IN:
        raise SystemExit(f"g's input is {f.shape[1]} wide, not {D_IN}")
    return f


def unit(j, pop, prep, pool, tables, adj, frame, cache):
    """One question's frame vector, seeds, first-stage ranks, graph features and candidates (the pool minus the
    seeds)."""
    seeds = np.unique(np.asarray(prep.seeds[j], np.int64))
    if not np.isin(seeds, pool).all():
        raise SystemExit(f"{pop.ids[j]}: a seed is outside the final pool")
    deg, m = seed_graph(adj, tables, seeds, cache)
    return {"x": A.qframe(prep.qemb[j], *frame), "seeds": seeds,
            "rd": first_ranks(prep.dense_ids[j], seeds), "rs": first_ranks(prep.splade_ids[j], seeds),
            "deg": deg, "m": m, "cand": np.setdiff1d(pool, seeds)}


def cat(xs, dtype, width=None):
    if xs:
        return np.concatenate(xs).astype(dtype)
    return np.zeros((0,) if width is None else (0, width), dtype)


def pack_units(units, golds, ids, pos):
    """A carve's questions as flat arrays: seeds and their rows, candidates and their gold flags."""
    S = [u["seeds"].size for u in units]
    C = [u["cand"].size for u in units]
    return {"ids": np.asarray([str(q) for q in ids]), "pos": np.asarray(pos, np.int64),
            "x": (np.stack([u["x"] for u in units]).astype(np.float32) if units
                  else np.zeros((0, DIM + 1), np.float32)),
            "seed_ptr": np.r_[0, np.cumsum(S)].astype(np.int64),
            "seeds": cat([u["seeds"] for u in units], np.int64),
            "rank_d": cat([u["rd"] for u in units], np.int32), "rank_s": cat([u["rs"] for u in units], np.int32),
            "deg": cat([u["deg"] for u in units], np.float32, 3),
            **{f"m_{sp}": cat([u["m"][sp] for u in units], np.float32, DIM) for sp in SPACES},
            "cand_ptr": np.r_[0, np.cumsum(C)].astype(np.int64),
            "cand": cat([u["cand"] for u in units], np.int64),
            "gold": cat([np.isin(u["cand"], g) for u, g in zip(units, golds)], bool)}


class Pack:
    """One carve's inputs on one space: g's rows, the seeds' and candidates' vectors (from the space's table), the
    gold flags and kappa."""

    def __init__(self, arr, U, kappa, sp):
        self.n = int(arr["seed_ptr"].size - 1)
        self.sp_, self.cp = arr["seed_ptr"], arr["cand_ptr"]
        reps = np.repeat(np.arange(self.n), np.diff(self.sp_))
        self.US = rows_of(U, arr["seeds"], np.float32)
        self.F = features(arr["x"][reps], self.US, arr[f"m_{sp}"], arr["deg"], arr["rank_d"], arr["rank_s"])
        self.UC = rows_of(U, arr["cand"], np.float16) if U.dtype == np.float16 else rows_of(U, arr["cand"], np.float32)
        self.gold = arr["gold"]
        self.kappa = float(kappa)


def batch(torch, dev, packs, items):
    """Padded tensors of (pack, question) items: rows, seed vectors and mask, candidate vectors and mask, gold flags,
    kappa."""
    S = [int(packs[d].sp_[i + 1] - packs[d].sp_[i]) for d, i in items]
    C = [int(packs[d].cp[i + 1] - packs[d].cp[i]) for d, i in items]
    B, Sm, Cm = len(items), max(S), max(C)
    F = np.zeros((B, Sm, D_IN), np.float32)
    US = np.zeros((B, Sm, DIM), np.float32)
    sm = np.zeros((B, Sm), bool)
    UC = np.zeros((B, Cm, DIM), np.float32)
    cm = np.zeros((B, Cm), bool)
    gm = np.zeros((B, Cm), bool)
    kap = np.zeros(B, np.float32)
    for b, (d, i) in enumerate(items):
        p = packs[d]
        s0, s1, c0, c1 = int(p.sp_[i]), int(p.sp_[i + 1]), int(p.cp[i]), int(p.cp[i + 1])
        F[b, :s1 - s0] = p.F[s0:s1]
        US[b, :s1 - s0] = p.US[s0:s1]
        sm[b, :s1 - s0] = True
        UC[b, :c1 - c0] = p.UC[c0:c1]
        cm[b, :c1 - c0] = True
        gm[b, :c1 - c0] = p.gold[c0:c1]
        kap[b] = p.kappa
    return tuple(torch.from_numpy(a).to(dev) for a in (F, US, sm, UC, cm, gm, kap))


# ── g ────────────────────────────────────────────────────────────────────────


def make_g(torch):
    nn = torch.nn

    class G(nn.Module):
        """g(q, s, graph features): per seed, HEADS offsets in the space and their weight logits."""

        def __init__(self):
            super().__init__()
            self.ln = nn.LayerNorm(D_IN)
            self.l1 = nn.Linear(D_IN, WIDTH)
            self.l2 = nn.Linear(WIDTH, WIDTH)
            self.l3 = nn.Linear(WIDTH, HEADS * (DIM + 1))
            nn.init.zeros_(self.l3.weight)
            nn.init.zeros_(self.l3.bias)
            self.beta = nn.Parameter(torch.zeros(()))

        def forward(self, f):
            act = nn.functional.gelu
            o = self.l3(act(self.l2(act(self.l1(self.ln(f)))))).unflatten(-1, (HEADS, DIM + 1))
            return o[..., :DIM], o[..., DIM]

    return G


def new_g(torch, dev):
    """g at its start (seed 0): every offset zero, equal weights, beta 0."""
    torch.manual_seed(0)
    g = make_g(torch)().to(dev)
    n = sum(p.numel() for p in g.parameters())
    if n != N_WEIGHTS:
        raise SystemExit(f"g has {n} weights, not the declared {N_WEIGHTS}")
    return g


def log_p(torch, g, F, US, sm, UC, cm, kap, parts=False):
    """log p(v | q) for every candidate: the mixture over (seed, head), weights pi, of the softmax over the
    candidates of -kappa e^beta |u_s + Delta - u_v|^2. Padded candidates hold about MASK."""
    delta, a = g(F)
    T = US[:, :, None, :] + delta
    d2 = ((T * T).sum(-1)[..., None] + (UC * UC).sum(-1)[:, None, None, :]
          - 2.0 * torch.einsum("bshd,bcd->bshc", T, UC)).clamp_min(0.0)
    lg = -(kap * torch.exp(g.beta))[:, None, None, None] * d2
    lg = torch.where(cm[:, None, None, :], lg, torch.full_like(lg, MASK))
    lsm = torch.log_softmax(lg, dim=-1)
    a = torch.where(sm[:, :, None], a, torch.full_like(a, MASK))
    B, S, H = a.shape
    lpi = torch.log_softmax(a.reshape(B, S * H), dim=1).reshape(B, S, H)
    out = torch.logsumexp(lpi[..., None] + lsm, dim=(1, 2))
    return (out, delta, lpi) if parts else out


def question_loss(torch, lp, gm):
    """-log p(g | q), averaged over each question's counted golds."""
    return -(torch.where(gm, lp, torch.zeros_like(lp)).sum(1) / gm.sum(1).clamp_min(1))


def fit_g(torch, dev, packs, names, epochs, logf=log):
    """Section 3: Adam on the dataset-balanced mean of each question's loss, in batches drawn in a seeded order.
    Returns g and the loss by epoch, overall (J) and by dataset."""
    g = new_g(torch, dev)
    opt = torch.optim.Adam(g.parameters(), lr=LR)
    items = [(d, i) for d, p in enumerate(packs) for i in range(p.n)]
    dsi = np.asarray([d for d, _i in items], np.int64)
    nd = np.asarray([packs[d].n for d in dsi], np.float64)
    omega = (len(items) / (len(packs) * nd)).astype(np.float32)     # each dataset weighs the same; mean weight 1
    rng = np.random.default_rng(0)
    hist = []
    t = time.time()
    for ep in range(epochs):
        perm = rng.permutation(len(items))
        per_q = np.zeros(len(items))
        J = 0.0
        for lo in range(0, len(items), BATCH):
            idx = perm[lo:lo + BATCH]
            F, US, sm, UC, cm, gm, kap = batch(torch, dev, packs, [items[k] for k in idx])
            ql = question_loss(torch, log_p(torch, g, F, US, sm, UC, cm, kap), gm)
            loss = (torch.from_numpy(omega[idx]).to(dev) * ql).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            per_q[idx] = ql.detach().double().cpu().numpy()
            J += loss.detach().item() * idx.size
        by = {names[d]: float(per_q[dsi == d].mean()) for d in range(len(packs))}
        hist.append({"epoch": ep + 1, "J": J / len(items), "by_dataset": by, "beta": g.beta.detach().item(),
                     "seconds": round(time.time() - t, 1)})
        if not (np.isfinite(J) and all(np.isfinite(v) for v in by.values())):
            raise SystemExit(f"the fit's loss is not finite at epoch {ep + 1}")
        logf(f"  epoch {ep + 1}/{epochs} ({time.time() - t:.0f}s)")
    return g, hist


def score_pack(torch, dev, g, pack):
    """g's log p(v | q) over each question's candidates (float64), and each question's largest weight pi and root
    mean square |Delta| over its (seed, head) pairs."""
    out, pimax, drms = [], [], []
    with torch.no_grad():
        for lo in range(0, pack.n, BATCH):
            items = [(0, i) for i in range(lo, min(pack.n, lo + BATCH))]
            F, US, sm, UC, cm, _gm, kap = batch(torch, dev, [pack], items)
            lp, delta, lpi = log_p(torch, g, F, US, sm, UC, cm, kap, parts=True)
            lp = lp.double().cpu().numpy()
            pi = torch.exp(lpi)
            for b, (_d, i) in enumerate(items):
                c = int(pack.cp[i + 1] - pack.cp[i])
                s = int(pack.sp_[i + 1] - pack.sp_[i])
                out.append(lp[b, :c].copy())
                pimax.append(float(pi[b, :s].max()))
                drms.append(float(torch.sqrt((delta[b, :s] ** 2).sum(-1).mean())))
    return out, np.asarray(pimax), np.asarray(drms)


def save_g(path, models, config):
    import torch
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.pt")
    torch.save({"models": {k: {n: v.detach().cpu() for n, v in g.state_dict().items()} for k, g in models.items()},
                "config": config}, tmp)
    os.replace(tmp, path)


def load_g(torch, dev, paths):
    trec = json.loads(paths["train"].read_text(encoding="utf-8"))
    if sha_file(paths["g"]) != trec["g_sha256"]:
        raise SystemExit(f"{paths['g']}: not the train record's g")
    blob = torch.load(paths["g"], map_location="cpu", weights_only=True)
    models = {}
    for k in ("G", "G0"):
        g = new_g(torch, dev)
        g.load_state_dict(blob["models"][k])
        g.eval()
        models[k] = g
    return models, trec


# ── orders ───────────────────────────────────────────────────────────────────


def order_by(score, cand):
    """The candidates by score, highest first, ties by node id."""
    return cand[np.lexsort((cand, -np.asarray(score, np.float64)))]


def ring_class(nodes, r1, r2):
    ring = np.full(nodes.size, 3, np.int64)
    ring[np.isin(nodes, r2)] = 2
    ring[np.isin(nodes, r1)] = 1
    return ring


def h2_low(cand, cc, r1, r2):
    """P5a's low order of H2 over all candidates: ring 1, then ring 2, then the rest, each by frozen cosine, ties by
    node id."""
    return cand[np.lexsort((cand, -cc, ring_class(cand, r1, r2)))]


def fused(a, b, cand, c=RANK_C):
    """The candidates (sorted) by 1/(c + rank in a) + 1/(c + rank in b), highest first, ties by node id; a and b each
    order all of them."""
    if not (a.size == b.size == cand.size):
        raise SystemExit("a fused order needs two orders of every candidate")
    ra = np.empty(cand.size, np.int64)
    rb = np.empty(cand.size, np.int64)
    ra[np.searchsorted(cand, a)] = np.arange(1, a.size + 1)
    rb[np.searchsorted(cand, b)] = np.arange(1, b.size + 1)
    s = 1.0 / (c + ra) + 1.0 / (c + rb)
    return cand[np.lexsort((cand, -s))]


def all_orders(cand, cc, r1, r2, lp):
    """Every order of section 4 over a question's candidates; lp: g's log p for G, G0 and NEAR."""
    o = {"H2": h2_low(cand, cc, r1, r2), "COS": cand[np.lexsort((cand, -cc))]}
    for k in ("G", "G0", "NEAR"):
        o[k] = order_by(lp[k], cand)
    o["F"] = fused(o["G"], o["H2"], cand)
    o["F0"] = fused(o["G0"], o["H2"], cand)
    o["FN"] = fused(o["NEAR"], o["H2"], cand)
    return o


# ── base ─────────────────────────────────────────────────────────────────────


def collect(op, E, name, carve, threads, limit, frame, tables, adj, get_finals):
    """Section 3's data on one carve: P5a's sample, each question's unit on its final pool; the questions without a
    counted gold skipped and counted."""
    holder = {}

    def sink(name_, carve_, pop, prep, U, BQ, bs, inc, linked, union, constant, families, construction):
        Q = len(pop.ids)
        samp = A.sample_of(Q)
        sample_sha = ids_sha([pop.ids[j] for j in samp])
        if limit:
            samp = samp[:limit]
        finals = get_finals(carve_)
        cache = {}
        units, golds, ids, pos = [], [], [], []
        skipped = 0
        t = time.time()
        for c, j in enumerate(samp.tolist()):
            pool = finals.pool(pop.ids[j], bs[j], inc[j]) if finals else np.unique(np.asarray(inc[j], np.int64))
            u = unit(j, pop, prep, pool, tables, adj, frame, cache)
            gl = np.intersect1d(np.unique(np.asarray(pop.golds[j], np.int64)), u["cand"])
            if not gl.size:
                skipped += 1
                continue
            units.append(u)
            golds.append(gl)
            ids.append(pop.ids[j])
            pos.append(j)
            if (c + 1) % 500 == 0:
                log(f"{name}/{carve_}: {c + 1}/{samp.size} questions ({time.time() - t:.0f}s)")
        holder["arr"] = pack_units(units, golds, ids, pos)
        holder["rec"] = {"carve": carve_, "carve_questions": Q, "sampled": int(samp.size), "kept": len(units),
                         "without_a_counted_gold": skipped, "carve_sha256": pop.digest,
                         "sample_ids_sha256": sample_sha, "seconds": round(time.time() - t, 1)}

    E.run_carves(op, name, {carve}, (), threads, sink)
    return holder["arr"], holder["rec"]


def base_data(name, threads, host, limit):
    """A training dataset's fit sample (section 3), with the tables it was built on."""
    if name not in TRAIN5:
        raise SystemExit(f"{name} never fits")
    E, _F, S4 = A.modules()
    op = A.opened(host)
    ds, emb = A.dense_docs(op, name)
    frame, tables, tshas, made = tables_for(name, emb, limit)
    _stores, adj = A.family_stores(S4, ds, name, A.csr_dir(op))
    if any(ip.size - 1 != emb.n_rows for ip, _c in adj):
        raise SystemExit(f"{name}: a family's store is not the corpus's size")
    get, pinfo = finals_for(name, limit)
    carve = A.fit_carve(name)
    arr, crec = collect(op, E, name, carve, threads, limit, frame, tables, adj, get)
    crec.update(pools=pinfo[carve], tables=tshas, tables_made=made, freeze_RECORD_SHA256=op.freeze["RECORD_SHA256"])
    return arr, crec, tables


def base_stage(name, threads, host, limit=None, out=OUT):
    t0 = time.time()
    arr, crec, _tables = base_data(name, threads, host, limit)
    if limit:
        log(f"{name}/{crec['carve']} development run: {crec['sampled']} sampled questions, units built for every "
            f"question with a counted gold; tables: {crec['tables_made']}; pools: {pools_said(crec['pools'])}; "
            f"{time.time() - t0:.0f}s, peak {A.peak_rss() / 2 ** 30:.1f} GB; nothing written")
        return 0
    paths = out_paths(out)
    p = paths["base"] / f"fit_{name}.npz"
    save_npz(p, **arr)
    nq = int(arr["ids"].size)
    rec = {"declared_in": DECLARED, "stage": "base", "dataset": name, **crec, "npz_sha256": sha_file(p),
           "seeds_mean": round(float(np.diff(arr["seed_ptr"]).mean()), 2) if nq else None,
           "candidates_mean": round(float(np.diff(arr["cand_ptr"]).mean()), 1) if nq else None,
           "golds_mean": round(float(arr["gold"].sum() / max(1, nq)), 2),
           "index_time": INDEX_TIME, "script_sha256": script_sha256(), "p5a_script_sha256": A.script_sha256(),
           "peak_rss_bytes": A.peak_rss(), "seconds": round(time.time() - t0, 1), "utc": utc()}
    write_json(paths["base"] / f"fit_{name}.json", rec)
    log(f"{name}: outputs/p5b/base/fit_{name}.json ({rec['seconds']:.0f}s)")
    return 0


# ── train ────────────────────────────────────────────────────────────────────


def fit_both(torch, dev, names, arrs, tables, epochs, logf=log):
    """G on V and G0 on V0 (section 3); returns {space: (g, hist)} and kappa per space and dataset."""
    fits, kap = {}, {}
    for sp in SPACES:
        kap[sp] = {n: kappa_of(tables[n][sp]) for n in names}
        packs = [Pack(arrs[n], tables[n][sp], kap[sp][n], sp) for n in names]
        logf(f"{MODEL_OF[sp]} on {sp}: {sum(p.n for p in packs)} questions")
        fits[sp] = fit_g(torch, dev, packs, list(names), epochs, logf)
        del packs
    return fits, kap


def train_stage(device, threads, host, limit=None, dataset=None, out=OUT):
    import torch
    torch.set_num_threads(threads)
    A.full_fp32(torch)
    dev = torch.device(device)
    t0 = time.time()
    paths = out_paths(out)
    names, arrs, tables, brecs = [], {}, {}, {}
    if limit:
        if dataset not in TRAIN5:
            raise SystemExit("a development train needs DATASET, one of the five training datasets")
        arr, crec, tab = base_data(dataset, threads, host, limit)
        names, arrs[dataset], tables[dataset] = [dataset], arr, tab
        made = crec["tables_made"]
    else:
        if dataset:
            raise SystemExit("train reads the five base files; DATASET is for development runs")
        for name in TRAIN5:
            bj = paths["base"] / f"fit_{name}.json"
            rec = json.loads(bj.read_text(encoding="utf-8"))
            bp = paths["base"] / f"fit_{name}.npz"
            if sha_file(bp) != rec["npz_sha256"] or rec["script_sha256"] != script_sha256():
                raise SystemExit(f"{bp}: not its record's file, or written by another script")
            with np.load(bp) as z:
                arrs[name] = {k: z[k] for k in z.files}
            _frame, tables[name], shas = filed_tables(name)
            if shas != rec["tables"]:
                raise SystemExit(f"{name}: the base ran on other tables")
            if rec["kept"] > 0:
                names.append(name)
            brecs[name] = sha_file(bj)
        made = "filed"
    names = [n for n in names if arrs[n]["ids"].size]
    if not names:
        raise SystemExit("no training question has a counted gold")
    epochs = DEV_EPOCHS if limit else EPOCHS
    fits, kap = fit_both(torch, dev, names, arrs, tables, epochs)
    if limit:
        log(f"train development run on {dataset}: both fits ran ({epochs} epoch), every loss finite; tables: {made}; "
            f"{time.time() - t0:.0f}s, gpu peak {A.gpu_peak(torch, dev) / 2 ** 30:.2f} GB; nothing written")
        return 0
    config = {"d_in": D_IN, "width": WIDTH, "heads": HEADS, "dim": DIM, "rank_c": RANK_C, "weights": N_WEIGHTS}
    save_g(paths["g"], {MODEL_OF[sp]: fits[sp][0] for sp in SPACES}, config)
    rec = {"declared_in": DECLARED, "stage": "train", "datasets": names,
           "questions": {n: int(arrs[n]["ids"].size) for n in names}, "base_records": brecs,
           "kappa": kap, "settings": {"epochs": EPOCHS, "batch": BATCH, "lr": LR, "optimizer": "Adam", "seed": 0,
                                      "precision": "float32, TF32 off", "kappa_pairs": KAPPA_PAIRS, **config},
           "fits": {MODEL_OF[sp]: {"space": sp, "hist": fits[sp][1], "beta": fits[sp][0].beta.detach().item()}
                    for sp in SPACES},
           "index_time": INDEX_TIME, "g_sha256": sha_file(paths["g"]), "script_sha256": script_sha256(),
           "p5a_script_sha256": A.script_sha256(), "device": str(dev), "gpu_peak_bytes": A.gpu_peak(torch, dev),
           "peak_rss_bytes": A.peak_rss(), "seconds": round(time.time() - t0, 1), "utc": utc()}
    write_json(paths["train"], rec)
    log(f"outputs/p5b/train.json ({rec['seconds']:.0f}s)")
    return 0


# ── read ─────────────────────────────────────────────────────────────────────


def read1_or_none(name, limit):
    """P5a's read1 on the sample (zsp's top 5 on today's pools), or None in a development run without its files."""
    n = A.READ1_PARTS[name]
    have = all((A.READ1 / f"{name}_{k}of{n}{x}").exists() for k in range(n) for x in (".json", ".npz"))
    if have:
        return A.read1_load(name)
    if not limit:
        raise SystemExit(f"{name}: P5a's read1 parts are not all on disk")
    return None


def dev_models(torch, dev, name, threads, host, limit):
    """A development read's g: this dataset's base and a one-epoch fit, or g at its start where it never fits."""
    if name not in TRAIN5:
        return {"G": new_g(torch, dev), "G0": new_g(torch, dev)}, "development: g at its start (never fits)"
    arr, _crec, tab = base_data(name, threads, host, limit)
    if not arr["ids"].size:
        return {"G": new_g(torch, dev), "G0": new_g(torch, dev)}, "development: g at its start (no counted gold)"
    fits, _kap = fit_both(torch, dev, [name], {name: arr}, {name: tab}, DEV_EPOCHS, logf=lambda *_: None)
    return ({MODEL_OF[sp]: fits[sp][0].eval() for sp in SPACES},
            "development: a one-epoch g on this dataset's base")


def read_stage(name, device, threads, host, limit=None, out=OUT):
    import torch
    torch.set_num_threads(threads)
    A.full_fp32(torch)
    E, _F, S4 = A.modules()
    op = A.opened(host)
    dev = torch.device(device)
    t0 = time.time()
    paths = out_paths(out)
    ds, emb = A.dense_docs(op, name)
    n = emb.n_rows
    frame, tables, tshas, made = tables_for(name, emb, limit)
    _stores, adj = A.family_stores(S4, ds, name, A.csr_dir(op))
    if any(ip.size - 1 != n for ip, _c in adj):
        raise SystemExit(f"{name}: a family's store is not the corpus's size")
    if paths["g"].exists() and paths["train"].exists():
        models, trec = load_g(torch, dev, paths)
        g_made = "filed"
    elif limit:
        models, g_made = dev_models(torch, dev, name, threads, host, limit)
        trec = None
    else:
        raise SystemExit("g is not on disk: run train first")
    models["NEAR"] = new_g(torch, dev).eval()
    r1 = read1_or_none(name, limit)
    get, pinfo = finals_for(name, limit)
    kap = {sp: kappa_of(tables[sp]) for sp in SPACES}
    if trec is not None and any(name in trec["kappa"][sp] and trec["kappa"][sp][name] != kap[sp] for sp in SPACES):
        raise SystemExit(f"{name}: kappa is not the fit's")
    mark = np.zeros(n, np.int8)
    buf = np.empty(n, np.int64)
    log(f"{name}: {n} nodes (tables {made}; g {g_made}; {device}); setup {time.time() - t0:.0f}s")
    holder = {}

    def sink(name_, carve, pop, prep, U, BQ, bs, inc, linked, union, constant, families, construction):
        Q = len(pop.ids)
        samp = A.sample_of(Q)
        sample_sha = ids_sha([pop.ids[j] for j in samp])
        if limit:
            samp = samp[:limit]
        ids = [pop.ids[j] for j in samp]
        if r1 is not None and (r1["sample_sha256"] != sample_sha or r1["ids"][:len(ids)] != ids
                               or r1["pos"][:len(ids)] != samp.tolist()):
            raise SystemExit(f"{name}: the sample is not read1's")
        finals = get(carve)
        cache = {}
        units, golds, checked = [], [], 0
        t = time.time()
        for qi, j in enumerate(samp.tolist()):
            pool = finals.pool(pop.ids[j], bs[j], inc[j]) if finals else np.unique(np.asarray(inc[j], np.int64))
            if r1 is not None:
                if not (np.array_equal(np.unique(np.asarray(inc[j], np.int64)), np.unique(r1["pools"][qi]))
                        and np.array_equal(np.unique(np.asarray(prep.seeds[j], np.int64)),
                                           np.unique(r1["seeds"][qi]))):
                    raise SystemExit(f"{name}: {pop.ids[j]}: today's pool or seeds are not read1's")
                checked += 1
            u = unit(j, pop, prep, pool, tables, adj, frame, cache)
            units.append(u)
            golds.append(np.intersect1d(np.unique(np.asarray(pop.golds[j], np.int64)), u["cand"]))
        log(f"{name}: {len(units)} questions' inputs ({time.time() - t:.0f}s)")
        cos = A.cosines(emb, np.stack([prep.qemb[j] for j in samp]), [u["cand"] for u in units], dev)
        arr = pack_units(units, golds, ids, samp)
        lp, pimax, drms = {}, None, None
        for k, sp in (("G", "V"), ("G0", "V0"), ("NEAR", "V")):
            pk = Pack(arr, tables[sp], kap[sp], sp)
            lp[k], pm, dr = score_pack(torch, dev, models[k], pk)
            if k == "G":
                pimax, drms = pm, dr
            del pk
        nQ, nO, nN = len(units), len(ORDERS), len(GRID)
        res = {rd: {"T": np.zeros(nQ, np.int64), "T_ring": np.zeros((nQ, RINGS), np.int64),
                    "found": np.zeros((nQ, nO, nN), np.int32), "found_ring": np.zeros((nQ, nO, nN, RINGS), np.int32),
                    "f_not_h2": np.zeros((nQ, nN), np.int32), "g_not_h2": np.zeros((nQ, nN), np.int32)}
               for rd in READS}
        sizes = np.zeros((nQ, 3), np.int64)
        for qi, u in enumerate(units):
            cand = u["cand"]
            r_1, r_2 = A.rings_of(adj, u["seeds"], mark, buf)
            o = all_orders(cand, cos[qi], r_1, r_2, {k: lp[k][qi] for k in ("G", "G0", "NEAR")})
            top = np.asarray(r1["top5"][qi], np.int64) if r1 is not None else np.empty(0, np.int64)
            gl = golds[qi]
            T = {"low": gl[~np.isin(gl, top)], "pool": gl}
            sizes[qi] = (r_1.size, r_2.size, cand.size)
            for rd in READS:
                Tr = T[rd]
                rg = ring_class(Tr, r_1, r_2)
                rr = res[rd]
                rr["T"][qi] = Tr.size
                for c in range(RINGS):
                    rr["T_ring"][qi, c] = int((rg == c + 1).sum())
                for oi, k in enumerate(ORDERS):
                    rr["found"][qi, oi] = A.found_counts(o[k], Tr, GRID)
                    for c in range(RINGS):
                        rr["found_ring"][qi, oi, :, c] = A.found_counts(o[k], Tr[rg == c + 1], GRID)
                for key, k in (("f_not_h2", "F"), ("g_not_h2", "G")):
                    rr[key][qi] = [int(np.isin(np.setdiff1d(o[k][:N], o["H2"][:N]), Tr).sum()) for N in GRID]
            if (qi + 1) % 500 == 0:
                log(f"{name}: {qi + 1}/{nQ} questions ordered ({time.time() - t:.0f}s)")
        holder.update(res=res, sizes=sizes, ids=ids, samp=samp, Q=Q, checked=checked, sample_sha=sample_sha,
                      carve_sha=pop.digest, pimax=pimax, drms=drms, seconds=time.time() - t)

    E.run_carves(op, name, {"s1eval"}, (), threads, sink)
    h = holder
    if limit:
        log(f"{name} development run: {len(h['ids'])} questions, every order and both counts ran; today's pools and "
            f"seeds checked against read1 on {h['checked']}"
            + ("" if r1 is not None else " (read1 is not on disk)")
            + f"; final pools: {pools_said(pinfo['s1eval'])}; {h['seconds']:.0f}s, gpu peak "
            f"{A.gpu_peak(torch, dev) / 2 ** 30:.2f} GB, peak {A.peak_rss() / 2 ** 30:.1f} GB; nothing written")
        return 0
    arrs = {f"{rd}|{k}": v for rd, d in h["res"].items() for k, v in d.items()}
    arrs.update(sizes=h["sizes"], pos=np.asarray(h["samp"], np.int64), ids=np.asarray([str(q) for q in h["ids"]]),
                pi_max=h["pimax"], delta_rms=h["drms"])
    p = out / f"read_{name}.npz"
    save_npz(p, **arrs)
    rec = {"declared_in": DECLARED, "stage": "read", "dataset": name, "carve": "s1eval", "carve_questions": h["Q"],
           "questions": len(h["ids"]), "carve_sha256": h["carve_sha"], "sample_sha256": h["sample_sha"],
           "checks": {"pools_and_seeds_equal_read1": h["checked"], "final_pools": "every check of section 5 held"},
           "pools": pinfo["s1eval"], "tables": tshas, "tables_made": made, "kappa": kap,
           "g_sha256": trec["g_sha256"], "train_record_sha256": sha_file(paths["train"]),
           "read1_records": r1["records"], "orders": list(ORDERS), "reads": list(READS), "grid": list(GRID),
           "index_time": INDEX_TIME, "npz_sha256": sha_file(p), "script_sha256": script_sha256(),
           "p5a_script_sha256": A.script_sha256(), "freeze_RECORD_SHA256": op.freeze["RECORD_SHA256"],
           "device": str(dev), "gpu_peak_bytes": A.gpu_peak(torch, dev), "peak_rss_bytes": A.peak_rss(),
           "read_s": round(h["seconds"], 1), "seconds": round(time.time() - t0, 1), "utc": utc()}
    write_json(out / f"read_{name}.json", rec)
    log(f"{name}: outputs/p5b/read_{name}.json ({rec['seconds']:.0f}s)")
    return 0


# ── the verdict (section 5 of the declared file) ─────────────────────────────


def verdict_stage(out=OUT):
    paths = out_paths(out)
    trec = json.loads(paths["train"].read_text(encoding="utf-8"))
    for k, f in trec["fits"].items():
        if not all(np.isfinite(e["J"]) and all(np.isfinite(v) for v in e["by_dataset"].values()) for e in f["hist"]):
            raise SystemExit(f"{k}'s loss is not finite")
    recs, data = {}, {}
    for name in SIX:
        rp = out / f"read_{name}.json"
        recs[name] = json.loads(rp.read_text(encoding="utf-8"))
        p = out / f"read_{name}.npz"
        if sha_file(p) != recs[name]["npz_sha256"]:
            raise SystemExit(f"{p}: not its record's file")
        with np.load(p) as z:
            data[name] = {k: z[k] for k in z.files}
    for key in ("script_sha256", "g_sha256"):
        if len({r[key] for r in recs.values()}) != 1:
            raise SystemExit(f"the reads ran on different {key}")
    if next(iter(recs.values()))["g_sha256"] != trec["g_sha256"]:
        raise SystemExit("the reads' g is not the train record's")
    checks = {}
    for name, r in recs.items():
        pools = r["pools"]
        ok = (r["checks"]["pools_and_seeds_equal_read1"] == r["questions"] and r["tables_made"] == "filed"
              and isinstance(pools, dict) and pools.get("construction") == CONSTRUCTION)
        checks[name] = {"questions": r["questions"], "pools_and_seeds_equal_read1": r["checks"]["pools_and_seeds_equal_read1"],
                        "tables": r["tables_made"], "pools": pools.get("construction") if isinstance(pools, dict) else pools}
        if not ok:
            raise SystemExit(f"{name}: the checks fail ({checks[name]})")
    oi = {o: k for k, o in enumerate(ORDERS)}
    calls, found, beside = {}, {}, {}
    for rd in READS:
        calls[rd], found[rd], beside[rd] = {}, {}, {}
        for name in SIX:
            D = data[name]
            T = D[f"{rd}|T"]
            keep = T > 0
            Fd = D[f"{rd}|found"][keep].astype(np.float64)
            t = T[keep]
            found[rd][name] = {"questions": int(keep.sum()), "golds": int(t.sum()),
                               **{o: [round(float(Fd[:, oi[o], k].sum() / max(1, t.sum())), 4)
                                      for k in range(len(GRID))] for o in ORDERS}}
            if not keep.any():
                calls[rd][name] = {"label": "NO_GOLDS"}
                continue
            c = {}
            for o in ORDERS:
                if o == "H2":
                    continue
                pts, signs = [], []
                for k, N in enumerate(GRID):
                    d, lo, hi = A.ratio_boot(Fd[:, oi[o], k], Fd[:, oi["H2"], k], t)
                    pts.append({"N": N, "delta": round(d, 4), "lo": round(lo, 4), "hi": round(hi, 4),
                                "counted": A.counted(d, lo, hi)})
                    signs.append(pts[-1]["counted"])
                c[o] = {"label": A.label(signs), "points": pts}
            calls[rd][name] = {"label": c["F"]["label"], **{f"{o}_against_H2": v for o, v in c.items()}}
            Tr = D[f"{rd}|T_ring"][keep]
            FR = D[f"{rd}|found_ring"][keep]
            beside[rd][name] = {
                "F_not_H2": [round(float(D[f"{rd}|f_not_h2"][keep][:, k].sum() / t.sum()), 4) for k in range(len(GRID))],
                "G_not_H2": [round(float(D[f"{rd}|g_not_h2"][keep][:, k].sum() / t.sum()), 4) for k in range(len(GRID))],
                "by_ring": {f"ring_{c_ + 1 if c_ < 2 else '3+'}": {
                    "golds": int(Tr[:, c_].sum()),
                    **{o: [round(float(FR[:, oi[o], k, c_].sum() / max(1, Tr[:, c_].sum())), 4)
                           for k in range(len(GRID))] for o in ("F", "G", "H2")}} for c_ in range(RINGS)}}
    labs = [calls["low"][n]["label"] for n in SIX if calls["low"][n]["label"] != "NO_GOLDS"]
    finds = "FINDS_MORE" if ("MORE" in labs and "FEWER" not in labs) else "DOES_NOT"
    call = "GOES_FURTHER" if finds == "FINDS_MORE" else "STOPS"
    where = None
    if call == "STOPS":
        where = {}
        for n in SIX:
            c = calls["low"][n]
            if c["label"] in ("NO_GOLDS", "MORE"):
                where[n] = "-"
                continue
            parts = [w for o, w in (("F0", "V"), ("FN", "the fit")) if c[f"{o}_against_H2"]["label"] == "MORE"]
            where[n] = " and ".join(parts) if parts else "what g reaches in this space"
    fits = {k: {"space": f["space"], "beta": round(f["beta"], 4), "J_first_last": [f["hist"][0]["J"], f["hist"][-1]["J"]],
                "by_dataset_last": f["hist"][-1]["by_dataset"]} for k, f in trec["fits"].items()}
    g_use = {n: {"pi_max_mean": round(float(data[n]["pi_max"].mean()), 4),
                 "delta_rms_mean": round(float(data[n]["delta_rms"].mean()), 4)} for n in SIX}
    rec = {"declared_in": DECLARED, "call": call, "finds": finds, "calls": calls, "found": found, "beside": beside,
           "where_it_fell_short": where, "fits": fits, "g_use": g_use, "checks": checks, "grid": list(GRID),
           "boot": A.BOOT, "tie": A.TIE, "reads": {n: sha_file(out / f"read_{n}.json") for n in SIX},
           "train_record": sha_file(paths["train"]), "script_sha256": script_sha256(), "utc": utc()}
    write_json(paths["verdict"], rec)
    paths["verdict_md"].write_text(report_md(rec), encoding="utf-8", newline="\n")
    log(f"P5b: {call}; low {finds} (" + ", ".join(f"{n} {calls['low'][n]['label']}" for n in SIX) + ")")
    return 0


def report_md(rec):
    grid = rec["grid"]
    L = [f"# P5b verdict: {rec['call']}", "", f"## The low read (the call: F against H2): {rec['finds']}", "",
         "| dataset | F | G | G0 | F0 | NEAR | FN | COS | " + " | ".join(f"F N={N}" for N in grid) + " |",
         "| --- | --- | --- | --- | --- | --- | --- | --- | " + " | ".join("---:" for _ in grid) + " |"]
    for n, c in rec["calls"]["low"].items():
        if c["label"] == "NO_GOLDS":
            L.append(f"| {n} | NO_GOLDS |" + " |" * (6 + len(grid)))
            continue
        labs = " | ".join(c[f"{o}_against_H2"]["label"] for o in ORDERS if o != "H2")
        L.append(f"| {n} | {labs} | " + " | ".join(f"{p['delta']:+.4f} [{p['lo']:+.4f}, {p['hi']:+.4f}]"
                                                   for p in c["F_against_H2"]["points"]) + " |")
    for rd in READS:
        L += ["", f"found(N), the {rd} read:", "", "| dataset | order | " + " | ".join(f"N={N}" for N in grid) + " |",
              "| --- | --- | " + " | ".join("---:" for _ in grid) + " |"]
        for n, f in rec["found"][rd].items():
            for o in ORDERS:
                L.append(f"| {n} | {o} | " + " | ".join(f"{v:.4f}" for v in f[o]) + " |")
    return "\n".join(L) + "\n"


# ── self-test ────────────────────────────────────────────────────────────────


def toy_arr(rng, units_spec, n_nodes):
    """Flat arrays for made-up questions: [(x, seeds, cand, golds)] with zero graph features and rank 1."""
    units, golds = [], []
    for x, seeds, cand, gl in units_spec:
        S = seeds.size
        units.append({"x": x.astype(np.float32), "seeds": seeds, "rd": np.ones(S, np.int64), "rs": np.ones(S, np.int64),
                      "deg": np.zeros((S, 3), np.float32), "m": {sp: np.zeros((S, DIM), np.float32) for sp in SPACES},
                      "cand": cand})
        golds.append(gl)
    return pack_units(units, golds, [f"q{i}" for i in range(len(units))], np.arange(len(units)))


def brute_log_p(torch, g, pack, i):
    """One question's log p(v | q), summed seed by seed and head by head."""
    s0, s1, c0, c1 = (int(v) for v in (pack.sp_[i], pack.sp_[i + 1], pack.cp[i], pack.cp[i + 1]))
    with torch.no_grad():
        delta, a = g(torch.from_numpy(pack.F[s0:s1]))
        delta, a = delta.double().numpy(), a.double().numpy()
    us, uc = pack.US[s0:s1].astype(np.float64), pack.UC[c0:c1].astype(np.float64)
    pi = np.exp(a - a.max())
    pi /= pi.sum()
    p = np.zeros(c1 - c0)
    for s in range(s1 - s0):
        for h in range(HEADS):
            lg = -pack.kappa * np.exp(g.beta.detach().item()) * ((us[s] + delta[s, h] - uc) ** 2).sum(1)
            e = np.exp(lg - lg.max())
            p += pi[s, h] * e / e.sum()
    return np.log(p)


def selftest():
    import torch
    A.full_fp32(torch)
    dev = torch.device("cpu")
    rng = np.random.default_rng(0)
    # features: 262 wide, blocks in the declared order
    x = rng.normal(size=(2, DIM + 1)).astype(np.float32)
    us, ms = rng.normal(size=(2, DIM)), rng.normal(size=(2, DIM))
    deg, rd, rs = np.asarray([[0, 1, 2], [3, 4, 5]]), np.asarray([1, 1001]), np.asarray([3, 5])
    f = features(x, us, ms, deg, rd, rs)
    assert f.shape == (2, D_IN) and np.allclose(f[:, :65], x) and np.allclose(f[:, 65:129], us)
    assert np.allclose(f[:, 129:193], x[:, :DIM] - us) and np.allclose(f[:, 193:257], ms)
    assert np.allclose(f[:, 257:260], np.log1p(deg)) and np.allclose(f[:, 260], 1 / (60 + rd))
    assert np.allclose(f[:, 261], 1 / (60 + rs))
    # first-stage ranks: 1-based first occurrences, depth + 1 when absent
    assert first_ranks([5, 3, 5, 9], [5, 9, 7, 3]).tolist() == [1, 4, 5, 2]
    assert first_ranks([], [4]).tolist() == [1]
    # m_s and the degrees equal a brute-force neighbour mean over the families' union, the seed excluded
    n = 12
    fam_edges = [[(0, 1), (0, 2), (0, 0), (3, 4)], [(0, 2), (0, 5), (6, 7)], [(5, 0), (8, 9)]]
    adj = []
    for edges in fam_edges:
        nb = [[] for _ in range(n)]
        for a_, b_ in edges:
            nb[a_].append(b_)
            if a_ != b_:
                nb[b_].append(a_)
        ip = np.r_[0, np.cumsum([len(v) for v in nb])].astype(np.int64)
        adj.append((ip, np.asarray([w for v in nb for w in v], np.int32)))
    U = rng.normal(size=(n, DIM)).astype(np.float16)
    dg, m = seed_graph(adj, {"V": U, "V0": U}, np.asarray([0, 10]), {})
    want = np.asarray(U[[1, 2, 5]], np.float64).mean(0) - np.asarray(U[0], np.float64)
    assert dg.tolist() == [[2, 2, 1], [0, 0, 0]] and np.allclose(m["V"][0], want, atol=1e-6) and not m["V"][1].any()
    # kappa: deterministic and positive
    assert kappa_of(U, 500) == kappa_of(U, 500) > 0
    # the batched score equals a seed-by-seed, head-by-head sum, with padding; one question alone equals its batch
    N_ = 40
    Ut = rng.normal(size=(N_, DIM)).astype(np.float32)
    spec = [(rng.normal(size=DIM + 1), np.asarray([0]), np.arange(10, 15), np.asarray([11])),
            (rng.normal(size=DIM + 1), np.asarray([1, 2, 3]), np.arange(15, 17), np.asarray([16])),
            (rng.normal(size=DIM + 1), np.asarray([4, 5]), np.arange(17, 24), np.asarray([18, 20]))]
    pk = Pack(toy_arr(rng, spec, N_), Ut, 0.05, "V")
    g = new_g(torch, dev)
    with torch.no_grad():
        g.l3.weight.normal_(0, 0.05)
        g.l3.bias.normal_(0, 0.05)
        g.beta.fill_(0.3)
    T_ = batch(torch, dev, [pk], [(0, 0), (0, 1), (0, 2)])
    with torch.no_grad():
        lp = log_p(torch, g, *T_[:5], T_[6]).double().numpy()
    for i in range(3):
        c = int(pk.cp[i + 1] - pk.cp[i])
        assert np.allclose(lp[i, :c], brute_log_p(torch, g, pk, i), atol=1e-5), i
        assert (lp[i, c:] < -1e29).all()
        with torch.no_grad():
            one = log_p(torch, g, *[t[i:i + 1] for t in T_[:5]], T_[6][i:i + 1]).double().numpy()
        assert np.allclose(one[0, :c], lp[i, :c], atol=1e-6)
    # at its start g orders the candidates by the seeds' nearness alone
    g0 = new_g(torch, dev)
    sc, _pm, _dr = score_pack(torch, dev, g0, pk)
    us2, uc2 = pk.US[pk.sp_[2]:pk.sp_[3]].astype(np.float64), pk.UC[pk.cp[2]:pk.cp[3]].astype(np.float64)
    near = np.zeros(uc2.shape[0])
    for s in range(us2.shape[0]):
        lg = -0.05 * ((us2[s] - uc2) ** 2).sum(1)
        e = np.exp(lg - lg.max())
        near += e / e.sum()
    cand2 = np.arange(17, 24)
    assert np.array_equal(order_by(sc[2], cand2), cand2[np.lexsort((cand2, -near))])
    # the fit: golds at a question-chosen offset from one of two seeds; the loss falls and g finds them over NEAR
    K, Qn = 3, 240
    R = rng.normal(size=(K, DIM)) * 1.0
    rows, spec = [], []
    for q in range(Qn):
        k = q % K
        s = rng.normal(size=(2, DIM))
        near_ = s[0] + rng.normal(0, 0.1, (5, DIM))
        gold = s[(q // K) % 2] + R[k] + rng.normal(0, 0.1, DIM)
        other = rng.normal(size=(10, DIM))
        base_ = len(rows)
        rows += list(s) + list(near_) + [gold] + list(other)
        xq = np.zeros(DIM + 1)
        xq[k] = 3.0
        xq[DIM] = 1.0
        spec.append((xq + rng.normal(0, 0.05, DIM + 1) * (np.arange(DIM + 1) < DIM), np.arange(base_, base_ + 2),
                     np.arange(base_ + 2, base_ + 18), np.asarray([base_ + 7])))
    Ug = np.asarray(rows, np.float32)
    kap = kappa_of(Ug, 5000)
    pkg = Pack(toy_arr(rng, spec, Ug.shape[0]), Ug, kap, "V")
    gfit, hist = fit_g(torch, dev, [pkg], ["toy"], 40, logf=lambda *_: None)
    assert hist[-1]["J"] < hist[0]["J"] * 0.5, (hist[0]["J"], hist[-1]["J"])

    def top1(gm):
        s_, _a, _b = score_pack(torch, dev, gm.eval(), pkg)
        return np.mean([order_by(s_[i], spec[i][2])[0] == spec[i][3][0] for i in range(Qn)])

    a_fit, a_near = top1(gfit), top1(new_g(torch, dev))
    assert a_fit > 0.8 and a_fit > a_near + 0.5, (a_fit, a_near)
    # the low order equals P5a's h2_orders; ties by node id
    pool = np.arange(0, 30)
    seeds = np.asarray([0, 1])
    r_1, r_2 = np.asarray([2, 3, 4, 40], np.int32), np.asarray([5, 6, 41], np.int32)
    need = np.union1d(np.union1d(r_1, r_2), pool)
    cosv = rng.normal(size=need.size).astype(np.float32)
    cosv[np.searchsorted(need, [7, 8])] = 0.5                         # a cosine tie: 7 before 8
    cand = np.setdiff1d(pool, seeds)
    _miss, low = A.h2_orders(r_1, r_2, need, cosv, pool, seeds, np.arange(50, 60), cand.size)
    mine = h2_low(cand, cosv[np.searchsorted(need, cand)], r_1, r_2)
    assert np.array_equal(mine, low) and mine.tolist().index(7) < mine.tolist().index(8)
    # the fusion: a brute-force sum of reciprocal ranks, ties by node id
    c3 = np.asarray([1, 2, 3])
    assert fused(np.asarray([1, 2, 3]), np.asarray([3, 2, 1]), c3).tolist() == [1, 3, 2]
    for _ in range(20):
        cand_ = np.sort(rng.choice(100, 15, replace=False))
        a_, b_ = rng.permutation(cand_), rng.permutation(cand_)
        sc_ = {int(v): 1 / (60 + list(a_).index(v) + 1) + 1 / (60 + list(b_).index(v) + 1) for v in cand_}
        assert fused(a_, b_, cand_).tolist() == sorted(sc_, key=lambda v: (-sc_[v], v))
    # the final pool: base and seeds and the expansion; each wrong input refused
    ids = ["a", "b"]
    bsl = [np.asarray([1, 2, 3]), np.asarray([4, 5])]
    incl = [np.asarray([1, 2, 3, 9, 10]), np.asarray([4, 5, 6, 7])]
    orders = [np.asarray([3, 20, 21, 22]), np.asarray([30, 31, 32])]
    orders = [outside_of(o, b) for o, b in zip(orders, bsl)]
    filed = S4C.filed_arrays(ids, bsl, incl, orders)
    fin = Finals(filed)
    assert fin.pool("a", bsl[0], incl[0]).tolist() == [1, 2, 3, 20, 21]
    assert fin.pool("b", bsl[1], incl[1]).tolist() == [4, 5, 30, 31]
    for bad in (lambda: fin.pool("a", np.asarray([1, 2]), incl[0]),
                lambda: fin.pool("a", bsl[0], np.asarray([1, 2, 3, 9, 11])),
                lambda: fin.pool("c", bsl[0], incl[0])):
        try:
            bad()
            raise AssertionError("a wrong input was not refused")
        except SystemExit:
            pass
    f2 = dict(filed)
    f2["nodes"] = filed["nodes"].copy()
    f2["nodes"][0] = 2                                                # the expansion touches base and seeds
    f3 = dict(filed)
    f3["count"] = filed["count"] - 1                                  # another size than the file's
    for ff in (f2, f3):
        try:
            Finals(ff).pool("a", bsl[0], incl[0])
            raise AssertionError("a wrong file was not refused")
        except SystemExit:
            pass
    with tempfile.TemporaryDirectory() as td:
        d = Path(td) / "toy"
        d.mkdir()
        np.savez_compressed(d / "toy__s1eval.npz", **filed)
        man = {"declared_in": P1C2_DECLARED, "construction": CONSTRUCTION,
               "files": {"toy__s1eval": {"file": "toy__s1eval.npz", "sha256": sha_file(d / "toy__s1eval.npz"),
                                         "construction": CONSTRUCTION, "questions": 2}}}
        (d / "pools.json").write_text(json.dumps(man), encoding="utf-8")
        get, info = finals_for("toy", None, Path(td))
        assert get("s1eval").pool("b", bsl[1], incl[1]).tolist() == [4, 5, 30, 31] and info["s1eval"]["questions"] == 2
        man["construction"] = "MX"
        (d / "pools.json").write_text(json.dumps(man), encoding="utf-8")
        try:
            pools_file("toy", "s1eval", Path(td))
            raise AssertionError("another construction's file was not refused")
        except SystemExit:
            pass
        get2, _i = finals_for("toy", 5, Path(td))
        assert get2("fit") is None                                    # a development run without the file
        try:
            finals_for("toy", None, Path(td))[0]("fit")
            raise AssertionError("a missing file was not refused")
        except SystemExit:
            pass
    verdict_selftest()
    log("p5b self-test passed")
    return 0


def outside_of(order, bs):
    """The order's nodes outside base and seeds, in order (P1c's outside, for the toy file)."""
    return np.asarray(order, np.int64)[~np.isin(order, bs)]


def verdict_selftest():
    rng = np.random.default_rng(1)
    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        paths = out_paths(out)
        hist = [{"epoch": 1, "J": 2.0, "by_dataset": {"metaqa": 2.0}, "beta": 0.0, "seconds": 1.0},
                {"epoch": 2, "J": 1.0, "by_dataset": {"metaqa": 1.0}, "beta": 0.1, "seconds": 2.0}]
        write_json(paths["train"], {"g_sha256": "g", "fits": {"G": {"space": "V", "hist": hist, "beta": 0.1},
                                                              "G0": {"space": "V0", "hist": hist, "beta": 0.1}}})

        def fake(name, up, down):
            nQ, nO, nN = 300, len(ORDERS), len(GRID)
            arr = {}
            for rd in READS:
                T = rng.integers(1, 4, nQ)
                Fd = np.zeros((nQ, nO, nN), np.int32)
                for k in range(nN):
                    base_ = np.minimum(T, rng.integers(0, 2, nQ))
                    Fd[:, :, k] = base_[:, None]
                    if up:
                        Fd[:, 0, k] = T                                  # F finds every gold
                    if down:
                        Fd[:, 0, k] = 0
                arr.update({f"{rd}|T": T, f"{rd}|T_ring": np.stack([T, 0 * T, 0 * T], 1),
                            f"{rd}|found": Fd, f"{rd}|found_ring": np.repeat(Fd[..., None], RINGS, 3) * 0,
                            f"{rd}|f_not_h2": np.zeros((nQ, nN), np.int32), f"{rd}|g_not_h2": np.zeros((nQ, nN), np.int32)})
            arr.update(pi_max=np.full(nQ, 0.5), delta_rms=np.full(nQ, 1.0))
            p = out / f"read_{name}.npz"
            save_npz(p, **arr)
            write_json(out / f"read_{name}.json", {
                "questions": nQ, "checks": {"pools_and_seeds_equal_read1": nQ}, "tables_made": "filed",
                "pools": {"construction": CONSTRUCTION}, "npz_sha256": sha_file(p), "script_sha256": "s",
                "g_sha256": "g"})

        for nm in SIX:
            fake(nm, nm == "musique", False)
        verdict_stage(out)
        v = json.loads(paths["verdict"].read_text(encoding="utf-8"))
        assert v["call"] == "GOES_FURTHER" and v["calls"]["low"]["musique"]["label"] == "MORE", v["finds"]
        assert v["calls"]["low"]["metaqa"]["label"] == "TIE"
        fake("webqsp", False, True)
        verdict_stage(out)
        v = json.loads(paths["verdict"].read_text(encoding="utf-8"))
        assert v["call"] == "STOPS" and v["calls"]["low"]["webqsp"]["label"] == "FEWER"
        assert v["where_it_fell_short"]["webqsp"] == "what g reaches in this space"
        assert v["where_it_fell_short"]["musique"] == "-"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", nargs="?", choices=("base", "train", "read", "verdict"))
    ap.add_argument("dataset", nargs="?")
    ap.add_argument("--host", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.stage in ("base", "read") and a.dataset not in SIX:
        raise SystemExit(f"dataset must be one of {SIX}")
    device = a.device
    if a.stage in ("train", "read") and device is None:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if a.stage == "base":
        return base_stage(a.dataset, a.threads, a.host, a.limit)
    if a.stage == "train":
        return train_stage(device, a.threads, a.host, a.limit, a.dataset)
    if a.stage == "read":
        return read_stage(a.dataset, device, a.threads, a.host, a.limit)
    if a.stage == "verdict":
        return verdict_stage()
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
