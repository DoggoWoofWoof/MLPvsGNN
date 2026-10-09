"""Serving forms of the chain match on a typed graph (docs/C6_FAST_CHAINS.md).

fast_entries      rmatch.question_entries (cap by mass) and zrc.contrib_entries (cap by w0(c) m_c(v)) with the same
                  entries bit for bit: the walk chain by chain (each chain's out-steps summed in order of appearance,
                  as np.bincount sums them, then sorted by (step, node) as np.unique sorts them), the per-chain mass
                  renormalisation in entry order, and the k_row cap row by row (a stable sort on the key within each
                  row, entries in chain order, which is lexsort's (row, -key, chain) order). The key's w0(c) is
                  computed by numpy exactly as zrc computes it.
FastChain         ChainMatch.chain_feats for one question over the relations on its pool only: the step scores of a
                  relation outside the pool never reach an entry, and the weight-only product rel_centered @ cm_b.T is
                  taken once at index time. The same function on the same weights (float32), checked to c3_fast.TOL.

    python outputs/mp_unified/chains_fast.py --selftest
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import numba  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402

import rmatch as RM  # noqa: E402
import zrc as ZC  # noqa: E402

HOPS, K_ROW, MAX_CHAINS = RM.HOPS, RM.K_ROW, RM.MAX_CHAINS


# ── the walk ─────────────────────────────────────────────────────────────────


@numba.njit(cache=True)
def _level(n, indptr, zl, dst, deg, types, off, node, mass, lvl, acc, seen, touched):
    """One hop of rmatch.walk from the chains (types, off, node, mass): the next level's chains in walk's order. Each
    chain's out-steps are summed per (step, node) in order of appearance (np.bincount's order) in a dense scratch over
    (local step, pool row), then sorted by (step, node) (np.unique's order)."""
    J = off.size - 1
    tot = 0
    for p in range(node.size):
        x = node[p]
        tot += indptr[x + 1] - indptr[x]
    hops = types.shape[1]
    nt = np.empty((max(tot, 1), hops), np.int64)
    no = np.empty(max(tot, 1) + 1, np.int64)
    nv = np.empty(max(tot, 1), np.int64)
    nm = np.empty(max(tot, 1), np.float64)
    jc = 0
    at = 0
    for j in range(J):
        cnt = 0
        for p in range(off[j], off[j + 1]):
            x = node[p]
            w = mass[p]
            for e in range(indptr[x], indptr[x + 1]):
                k = zl[e] * n + dst[e]
                if not seen[k]:
                    seen[k] = True
                    acc[k] = 0.0
                    touched[cnt] = k
                    cnt += 1
                acc[k] += w / deg[e]
        if cnt == 0:
            continue
        ks = np.sort(touched[:cnt])
        prev = -1
        for t in range(cnt):
            k = ks[t]
            z = k // n
            if z != prev:
                for h in range(hops):
                    nt[jc, h] = types[j, h]
                nt[jc, lvl] = z
                no[jc] = at
                jc += 1
                prev = z
            nv[at] = k % n
            nm[at] = acc[k]
            seen[k] = False
            at += 1
    no[jc] = at
    return nt[:jc], no[:jc + 1], nv[:at], nm[:at], tot


def walk_fast(n, g, seeds, hops, cap, uz):
    """rmatch.walk's levels (types in uz's local step ids, then mapped back), its order and masses bit for bit."""
    src, z, dst, deg, indptr = g
    levels = []
    if seeds.size == 0 or src.size == 0:
        return levels, False
    zl = np.searchsorted(uz, z).astype(np.int64)
    kl = int(uz.size)
    acc = np.empty(kl * n, np.float64)
    seen = np.zeros(kl * n, np.bool_)
    touched = np.empty(max(int(src.size), 1), np.int64)
    types = np.full((1, hops), -1, np.int64)
    off = np.array([0, seeds.size], np.int64)
    node = seeds.astype(np.int64)
    mass = np.full(seeds.size, 1.0 / seeds.size)
    total = 0
    for lvl in range(hops):
        nt, no, nv, nm, tot = _level(n, indptr, zl, dst, deg, types, off, node, mass, lvl, acc, seen, touched)
        if tot == 0 or nv.size == 0:
            break
        types, off, node, mass = nt, no, nv, nm
        levels.append((types, off, node, mass))
        total += types.shape[0]
        if total > cap:
            return levels, True
    return levels, False


# ── the entries ──────────────────────────────────────────────────────────────


@numba.njit(cache=True)
def _live(off, node, mass, seed, types, uz, step, use_step):
    """One level's chains: each chain's non-seed entries (pool row, mass over the chain's non-seed total summed in entry
    order, its live chain's id within the level), the live chains' steps (global ids, -1 padded) and, for the
    contribution cap, the sum of their step scores in step order (zrc's sum over the chain's steps)."""
    J = off.size - 1
    hops = types.shape[1]
    tot = np.zeros(J, np.float64)
    nk = 0
    nl = 0
    for j in range(J):
        alive = False
        for p in range(off[j], off[j + 1]):
            if not seed[node[p]]:
                tot[j] += mass[p]
                alive = True
                nk += 1
        if alive:
            nl += 1
    row = np.empty(nk, np.int64)
    m = np.empty(nk, np.float64)
    cid = np.empty(nk, np.int64)
    zc = np.empty((nl, hops), np.int64)
    lw = np.zeros(nl, np.float64)
    at = 0
    c = -1
    for j in range(J):
        first = True
        for p in range(off[j], off[j + 1]):
            if not seed[node[p]]:
                if first:
                    c += 1
                    first = False
                    s_ = 0.0
                    for h in range(hops):
                        t = types[j, h]
                        if t >= 0:
                            zc[c, h] = uz[t]
                            if use_step:
                                s_ += step[uz[t]]
                        else:
                            zc[c, h] = -1
                            if use_step:
                                s_ += 0.0
                    lw[c] = s_
                row[at] = node[p]
                m[at] = mass[p] / tot[j]
                cid[at] = c
                at += 1
    return row, m, cid, zc, lw


@numba.njit(cache=True)
def _cap(row, key, n, k_row):
    """lexsort((cid, -key, row))'s first k_row entries of every row, entries in chain order (cid ascending): within a
    row, every key above the k_row-th largest, then the ties at it in chain order."""
    cnt = np.zeros(n + 1, np.int64)
    for i in range(row.size):
        cnt[row[i] + 1] += 1
    for v in range(n):
        cnt[v + 1] += cnt[v]
    pos = cnt.copy()
    by = np.empty(row.size, np.int64)
    for i in range(row.size):
        by[pos[row[i]]] = i
        pos[row[i]] += 1
    keep = np.zeros(row.size, np.bool_)
    for v in range(n):
        a, b = cnt[v], cnt[v + 1]
        if b - a <= k_row:
            for t in range(a, b):
                keep[by[t]] = True
            continue
        kv = np.empty(b - a, np.float64)
        for t in range(a, b):
            kv[t - a] = -key[by[t]]
        thr = -np.partition(kv, k_row - 1)[k_row - 1]
        above = 0
        for t in range(a, b):
            if key[by[t]] > thr:
                keep[by[t]] = True
                above += 1
        left = k_row - above
        for t in range(a, b):
            if left == 0:
                break
            if key[by[t]] == thr:
                keep[by[t]] = True
                left -= 1
    return keep


def fast_entries(n, hl, tl, sl, seeds_by_bucket, kz, step=None, hops=HOPS, cap=MAX_CHAINS, k_row=K_ROW):
    """rmatch.question_entries (step None) or zrc.contrib_entries (step given): (row, z, bucket, mass), the same
    arrays bit for bit (no statistics)."""
    g = RM.typed_graph(n, hl, tl, sl, kz)
    uz = np.unique(g[1])
    use = step is not None
    st = np.asarray(step, np.float64) if use else np.zeros(1)
    l3 = np.log(1.0 / 3.0)
    rows, zs, bs, ms = [], [], [], []
    for b, seeds in enumerate(seeds_by_bucket):
        levels, _capped = walk_fast(n, g, seeds, hops, cap, uz)
        if not levels:
            continue
        seed = np.zeros(n, np.bool_)
        seed[seeds] = True
        R, M, C, Z, K = [], [], [], [], []
        base = 0
        for t, o, v, mm in levels:
            row, m, cid, zc, lw = _live(o, v, mm, seed, t, uz, st, use)
            R.append(row)
            M.append(m)
            C.append(cid + base)
            Z.append(zc)
            if use:
                K.append(np.exp(lw + l3)[cid] * m)
            base += zc.shape[0]
        row = np.concatenate(R)
        if row.size == 0:
            continue
        m = np.concatenate(M)
        cid = np.concatenate(C)
        zc = np.concatenate(Z)
        keep = _cap(row, np.concatenate(K) if use else m, n, k_row)
        rows.append(row[keep])
        zs.append(zc[cid[keep]])
        bs.append(np.full(int(keep.sum()), b, np.int64))
        ms.append(m[keep])
    if not rows:
        return np.zeros(0, np.int64), np.zeros((0, hops), np.int64), np.zeros(0, np.int64), np.zeros(0)
    return np.concatenate(rows), np.concatenate(zs), np.concatenate(bs), np.concatenate(ms)


# ── the chain term ───────────────────────────────────────────────────────────


@numba.njit(cache=True)
def _chain_sums(N, row, zloc, b, m, lp, lpi, beta, hops):
    Sm = np.zeros(N, np.float32)
    Sr = np.zeros(N, np.float32)
    for i in range(row.size):
        lw = np.float32(0.0)
        L = 0
        for k in range(hops):
            zz = zloc[i, k]
            if zz >= 0:
                lw += lp[k, zz]
                L += 1
        lw += lpi[L - 1] + beta[b[i]]
        w = np.float32(np.exp(lw))
        Sm[row[i]] += w * m[i]
        Sr[row[i]] += w
    return Sm, Sr


class FastChain:
    """ChainMatch.chain_feats for one question over its pool's relations."""

    def __init__(self, m, rel_unit, rel_centered):
        with torch.no_grad():
            self.Gb = (rel_centered @ m.cm_b.T).contiguous()          # (n_rel, RANK)
        self.rel_unit = rel_unit
        self.m = m
        self.hops = int(m.cm_a.shape[0])

    @torch.inference_mode()
    def __call__(self, q_emb, ch, N):
        """(fm, fr): log(1 + S_m / TAU), log(1 + S_r) over N rows; ch holds the question's entries in the build's
        dtypes (ent_row, ent_z, ent_b, ent_m, q_rel)."""
        m = self.m
        q = Fn.normalize(q_emb.reshape(1, -1), dim=1)
        qr = torch.from_numpy(ch["q_rel"].astype(np.int64))
        R = int(qr.numel())
        cos = (q @ self.rel_unit.index_select(0, qr).T).squeeze(0)       # (R,)
        if R >= 2:
            mean = cos.sum() / R
            var = ((cos - mean) ** 2).sum() / R
        else:
            mean = var = torch.zeros(())
        ok = (R >= 2) and bool(var > 1e-12)
        zc = (cos - mean) / var.clamp_min(1e-12).sqrt() if ok else torch.zeros_like(cos)
        P = torch.einsum("khd,d->kh", m.cm_a, q[0])                      # (hops, RANK)
        Gr = P @ self.Gb.index_select(0, qr).T                            # (hops, R)
        Gd = P @ m.cm_d.T                                                 # (hops, 2)
        h = (m.cm_kappa[:, None, None] * zc[None, :, None] + Gr[..., None] + Gd[:, None, :]
             + m.cm_dir[:, None, :] + m.cm_bias[:, None, None])
        lp = (-Fn.softplus(-h)).reshape(self.hops, 2 * R).contiguous().numpy()
        lpi = Fn.log_softmax(q @ m.cm_pi_w + m.cm_pi_b, dim=1)[0].contiguous().numpy()
        z = ch["ent_z"].astype(np.int64)
        loc = np.full(int(self.rel_unit.shape[0]), -1, np.int64)
        loc[ch["q_rel"].astype(np.int64)] = np.arange(R)
        zloc = np.where(z >= 0, 2 * loc[np.maximum(z, 0) // 2] + np.maximum(z, 0) % 2, -1)
        Sm, Sr = _chain_sums(N, ch["ent_row"].astype(np.int64), zloc, ch["ent_b"].astype(np.int64),
                             ch["ent_m"].astype(np.float32), lp, lpi, m.cm_beta.detach().numpy(), self.hops)
        return torch.log1p(torch.from_numpy(Sm) / RM.TAU), torch.log1p(torch.from_numpy(Sr))


# ── selftest ─────────────────────────────────────────────────────────────────


def _same(a, b):
    return all(x.dtype == y.dtype and x.shape == y.shape and np.array_equal(x, y) for x, y in zip(a, b))


def selftest(chunks=2):
    """Every question of the first look chunks of both typed graphs (and random toy graphs): fast_entries equal
    question_entries and contrib_entries bit for bit."""
    rng = np.random.default_rng(0)
    nq = 0
    for ds in RM.TYPED:
        d = RM.LM.LOOK / ds / "s1eval"
        if not (d / "chunks" / "c00000.npz").exists():
            print(f"selftest: {ds}'s look is not on disk; skipped")
            continue
        rel = RM.look_relations(d)
        n_rel = int(rel["n_relations"])
        E = np.load(RM.REL_DIR / f"{ds}_rel_embeddings.npy").astype(np.float64)
        U = E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)
        for ch in range(chunks):
            with np.load(d / "chunks" / f"c{ch:05d}.npz") as zf:
                c = {k: zf[k] for k in RM.KEYS + ("q_emb",)}
            eo = np.r_[0, np.cumsum(c["q_edges"].astype(np.int64))]
            for i in range(c["q_pool_size"].size):
                n = int(c["q_pool_size"][i])
                hl, tl, sl = RM.row_triples(c, int(eo[i]), int(eo[i + 1]))
                sl0, bk = c["q_seed_local"][i], c["q_seed_bucket"][i]
                ok = sl0 >= 0
                seeds = [np.unique(sl0[ok & (bk == b)]).astype(np.int64) for b in range(RM.BUCKETS)]
                step = ZC.step_of(c["q_emb"][i], U, np.unique(sl[sl >= 0]))
                a = RM.question_entries(n, hl, tl, sl, seeds, 2 * n_rel)[0]
                assert _same(a, fast_entries(n, hl, tl, sl, seeds, 2 * n_rel)), (ds, ch, i, "mass cap")
                a = ZC.contrib_entries(n, hl, tl, sl, seeds, 2 * n_rel, step)[0]
                assert _same(a, fast_entries(n, hl, tl, sl, seeds, 2 * n_rel, step)), (ds, ch, i, "contribution cap")
                nq += 1
    for t in range(300):
        n = int(rng.integers(2, 60))
        n_rel = int(rng.integers(1, 6))
        E_ = int(rng.integers(0, 4 * n))
        hl, tl = rng.integers(0, n, E_), rng.integers(0, n, E_)
        sl = np.full((E_, 4), -1, np.int64)
        k = rng.integers(1, 5, E_)
        for j in range(4):
            sl[:, j] = np.where(j < k, rng.integers(0, n_rel, E_), -1)
        seeds = [np.unique(rng.integers(0, n, int(rng.integers(0, 4)))) for _ in range(2)]
        step = rng.normal(size=2 * n_rel) - 1.0
        cap = int(rng.integers(1, 50))
        kr = int(rng.integers(1, 6))
        a = RM.question_entries(n, hl, tl, sl, seeds, 2 * n_rel, cap=cap, k_row=kr)[0]
        assert _same(a, fast_entries(n, hl, tl, sl, seeds, 2 * n_rel, cap=cap, k_row=kr)), (t, "toy mass")
        a = ZC.contrib_entries(n, hl, tl, sl, seeds, 2 * n_rel, step, cap=cap, k_row=kr)[0]
        assert _same(a, fast_entries(n, hl, tl, sl, seeds, 2 * n_rel, step, cap=cap, k_row=kr)), (t, "toy contrib")
    print(f"selftest: fast_entries equal question_entries and contrib_entries bit for bit on {nq} look questions and "
          f"300 toy graphs")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        os.environ.setdefault("NUMBA_NUM_THREADS", "1")
        sys.exit(selftest())
