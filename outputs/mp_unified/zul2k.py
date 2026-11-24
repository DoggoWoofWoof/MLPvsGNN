"""zul2k.py: zul2's key pass with a checkpoint (25 Nov 2026). A wrapper; zul2, zul2u, zul2m and every pinned module stay as filed.

Why: zul2.key_pass (R.hub's table, the 2-5 h part of every hbuild row) keeps its running table only in memory: a cancel, a preemption or a
host restart loses all of it (user, 25 Nov: "make sure losing anything means a few mins of progress, checkpoint all that can be and clean
up once done"). Top for 2Wiki is about 2.6 GB.

What it does: it replaces zul2.key_pass (the one attribute; zul2u.py and zul2m.py call L2.key_pass, zul2.py's cmd_hbuild calls the
module global, so one patch reaches all three) by a copy whose loop body is the original's, line for line (the self-test diffs the two
sources: only lines tagged `# ck` differ). Every EVERY_S seconds (300), at a loop-unit boundary (after one batch of small keys, or one block
of a big key), and at the end of the pass, it writes the whole running state: Top's b, s, tmax, the unflushed buffer (in its own order),
the counters (nbuf, merges, offered), the hub-key list, the loop position and the pass seconds. A resumed run starts from the last complete
save and gives the arrays an uninterrupted run gives, bit for bit, and the same counters (the flush schedule is not changed: nothing is
flushed early to make a save smaller).

Files: <repo>/outputs/ul2k/ckpt/<module>-<cmd>-<ds>-<enc>/{a,b}/*.npy + meta.json. Two slots, written alternately: the slot being written has its
meta.json removed first and gets a new one (tmp + os.replace) only after every .npy is written and fsync'd, so a kill during a save leaves
the other slot whole. The arrays are raw .npy (np.lib.format header + the bytes, written part by part: no compression, no concatenated copy).
The key (in meta.json) covers: the module, command, dataset and arm (from argv), the zul2 and the module's sha256, n, R, hub_lo, dfmax,
small, skip_le, threads, MERGE_AT, and a strided sample + the integer sums of names.df, names.key_ptr, names.key_node, the stored rows V, the
CSR (Np, Nc) and the exclusion pairs: a checkpoint of another key is ignored (the pass starts at 0 and overwrites it).

Cost (laptop disk, `wtime`): 2.6 GB (Top for 2Wiki: 13.3 M rows, R=24) written + fsync'd in 8.2 s (318 MB/s), 4.2 GB (with a full 1.44 GB
unflushed buffer and 0.2 GB of hub rows) in 11.8 s: under 4 % of the pass at one save per 300 s.

Clean-up: the row's folder is deleted when the module's main returns 0 (the row's outputs are written); a non-zero return or an exception
keeps it (the retry resumes). At the start of an hbuild row, the stray h_<enc>.tmp.npz of a killed run of the same row (zul2.save_npz's
temporary name, in the row's own output folder) is deleted.

Build identity: zul2.build_sha reads inspect.getsource of key_pass and Top; the replacement is declared `__wrapped__` = the original, so
inspect returns the original's source and every table built through this wrapper carries the build identity of the unwrapped code (the
self-test checks zul2's, zul2u's and zul2c's build_sha before and after the patch).

Usage (the row's own arguments follow the module name; --threads is read before numpy loads):
    python outputs/mp_unified/zul2k.py zul2  hbuild --ds 2wiki --enc e3v --host --threads 8
    python outputs/mp_unified/zul2k.py zul2u hbuild --ds metaqa --enc e3v --host --threads 8
    python outputs/mp_unified/zul2k.py zul2m hbuild --enc e3v --threads 8
    python outputs/mp_unified/zul2k.py wselftest        laptop CPU: the checks above on zul2.selftest's toy
    python outputs/mp_unified/zul2k.py wtime [--gb 2.6] timing of the save at a given size (random bytes, no model)
"""
import hashlib
import importlib
import inspect
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import zul2 as L2  # noqa: E402  (the first pinned import: --threads from sys.argv sets the BLAS threads before numpy loads)

import numpy as np  # noqa: E402

UL, Top, emit, gather_ranges, log = L2.UL, L2.Top, L2.emit, L2.gather_ranges, L2.log
R_DEPTH, HUB_LO, DFMAX, SMALL = L2.R_DEPTH, L2.HUB_LO, L2.DFMAX, L2.SMALL
ORIG = L2.key_pass

FMT = 1                                   # bump when the files or the key change
EVERY_S = 300.0
CKPT = ROOT / "outputs" / "ul2k" / "ckpt"
MODULES = ("zul2", "zul2u", "zul2m")
RUN = {"row": None, "saves": 0, "save_s": 0.0, "bytes": 0, "resumed": None}
FAULT = {"after_saves": None, "mid_save": None, "hard": False}      # test hooks; never set in a row


class Killed(Exception):
    pass


def _die():
    if FAULT["hard"]:
        os._exit(137)
    raise Killed()


# ── the files ────────────────────────────────────────────────────────────────


def write_npy(path, dtype, parts, step=1 << 27):
    """One .npy holding np.concatenate(parts) along axis 0, written part by part (no concatenated copy, no compression); fsync'd."""
    dt = np.dtype(dtype)
    parts = [np.ascontiguousarray(p, dtype=dt) for p in parts]
    total = sum(int(p.shape[0]) for p in parts)
    tail = tuple(parts[0].shape[1:]) if parts else ()
    with open(path, "wb") as f:
        np.lib.format.write_array_header_1_0(f, {"descr": np.lib.format.dtype_to_descr(dt), "fortran_order": False,
                                                 "shape": (total,) + tail})
        for p in parts:
            u = p.reshape(-1).view(np.uint8)
            for i in range(0, u.size, step):
                f.write(u[i:i + step])
        f.flush()
        os.fsync(f.fileno())
    return os.path.getsize(path)


DT = {"b": np.int32, "s": np.float32, "tmax": np.float32, "bufa": np.int32, "bufb": np.int32, "bufs": np.float32,
      "hk0": np.int32, "hk1": np.int32, "hk2": np.float32, "hk3": np.int32, "hk4": np.float32}


def fp(h, x, rows, total=False):
    """Into the hash: shape, dtype, `rows` strided entries, the last entry, and (integers) the sum: cheap, not a full read of x."""
    x = np.asarray(x)
    h.update(repr((x.shape, str(x.dtype))).encode())
    if x.size:
        step = max(1, x.shape[0] // rows)
        h.update(np.ascontiguousarray(x[::step][:rows]).tobytes())
        h.update(np.ascontiguousarray(x[-1:]).tobytes())
        if total and x.dtype.kind in "iu":
            h.update(str(int(x.sum(dtype=np.int64))).encode())


def key_of(names, V, Np, Nc, Nkeys, n, R, hub_lo, dfmax, small, skip_le, threads):
    row = RUN["row"] or {"id": "adhoc", "module_sha": "-"}
    h = hashlib.sha256()
    h.update(repr((FMT, row["id"], row["module_sha"], L2.script_sha256(), int(n), int(R), int(hub_lo), int(dfmax), int(small),
                   int(skip_le), int(threads), int(L2.MERGE_AT))).encode())
    for x in (names.df, names.key_ptr, names.key_node, Np, Nc, Nkeys):
        fp(h, x, 1_000_003, total=True)
    fp(h, V, 4001)
    return h.hexdigest()[:32]


class Store:
    def __init__(self, folder, key):
        self.dir, self.key = Path(folder), key
        self.cur, self.seq, self.foreign = None, 0, 0
        self.last = time.time()

    def _meta(self, slot):
        try:
            return json.loads((self.dir / slot / "meta.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def _whole(self, slot, m):
        for name, nbytes in m.get("files", {}).items():
            try:
                if (self.dir / slot / f"{name}.npy").stat().st_size != nbytes:
                    return False
            except OSError:
                return False
        return True

    def resume(self, top, hk):
        """Load the newest whole slot of this key into top and hk; return (position, pass seconds done). A slot of another key is ignored."""
        best = None
        for slot in ("a", "b"):
            m = self._meta(slot)
            if m is None:
                continue
            if m.get("fmt") != FMT or m.get("key") != self.key:
                self.foreign += 1
                continue
            if self._whole(slot, m) and (best is None or m["seq"] > best[1]["seq"]):
                best = (slot, m)
        if best is None:
            if self.foreign:
                log(f"zul2k: a checkpoint of another key in {self.dir.name} is ignored (the pass starts at 0)")
            return (1, 0, 0), 0.0
        slot, m = best
        d = self.dir / slot
        for name in ("b", "s", "tmax"):
            mm = np.load(d / f"{name}.npy", mmap_mode="r")
            tgt = getattr(top, name)
            assert mm.shape == tgt.shape and mm.dtype == tgt.dtype, (name, mm.shape, tgt.shape)
            tgt[...] = mm
            del mm
        a, b, s = (np.load(d / f"{x}.npy") for x in ("bufa", "bufb", "bufs"))
        top.buf = [(a, b, s)] if a.size else []
        hk.extend([tuple(np.load(d / f"hk{j}.npy") for j in range(5))] if m["nhk"] else [])
        top.nbuf, top.merges, top.offered = int(m["nbuf"]), int(m["merges"]), int(m["offered"])
        assert top.nbuf == a.size, (top.nbuf, a.size)
        self.cur, self.seq = slot, int(m["seq"])
        RUN["resumed"] = (tuple(m["pos"]), self.seq)
        self.last = time.time()
        log(f"zul2k: resumed checkpoint {self.seq} of {self.dir.name} at position {tuple(m['pos'])} "
            f"({m['elapsed']:.0f} s of the pass done, {top.nbuf:,} buffered, {m['nhk']:,} hub rows)")
        return tuple(m["pos"]), float(m["elapsed"])

    def tick(self, top, hk, pos, t0, force=False):
        if force or time.time() - self.last >= EVERY_S:
            self.save(top, hk, pos, time.time() - t0)

    def write_slot(self, d, arrs, meta_extra):
        """Write the arrays of one slot (name -> parts) then its meta.json, in that order; returns the bytes of the arrays."""
        d.mkdir(parents=True, exist_ok=True)
        mp = d / "meta.json"
        if mp.exists():
            mp.unlink()
        files = {}
        for i, name in enumerate(DT):
            files[name] = write_npy(d / f"{name}.npy", DT[name], arrs[name])
            if FAULT["mid_save"] == (RUN["saves"] + 1, i):
                _die()
        meta = dict(meta_extra, fmt=FMT, key=self.key, files=files)
        tmp = d / "meta.json.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(json.dumps(meta))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, mp)
        return sum(files.values())

    def save(self, top, hk, pos, elapsed):
        t = time.time()
        slot = "b" if self.cur == "a" else "a"
        arrs = {"b": [top.b], "s": [top.s], "tmax": [top.tmax], "bufa": [x[0] for x in top.buf], "bufb": [x[1] for x in top.buf],
                "bufs": [x[2] for x in top.buf]}
        for j in range(5):
            arrs[f"hk{j}"] = [h[j] for h in hk]
        nhk = sum(int(h[0].shape[0]) for h in hk)
        nbytes = self.write_slot(self.dir / slot, arrs, {"seq": self.seq + 1, "pos": list(pos), "elapsed": float(elapsed),
                                                         "nbuf": int(top.nbuf), "merges": int(top.merges),
                                                         "offered": int(top.offered), "nhk": nhk})
        self.cur, self.seq, self.last = slot, self.seq + 1, time.time()
        RUN["saves"] += 1
        RUN["save_s"] += time.time() - t
        RUN["bytes"] += nbytes
        if EVERY_S > 0:
            log(f"zul2k: checkpoint {self.seq} at position {tuple(pos)} ({nbytes / 1e9:.2f} GB, {time.time() - t:.1f} s, "
                f"{elapsed:.0f} s of the pass done)")
        if FAULT["after_saves"] == RUN["saves"]:
            _die()

    def clear(self):
        assert CKPT in self.dir.parents or "wselftest" in self.dir.parts or "wtime" in self.dir.parts, self.dir
        shutil.rmtree(self.dir, ignore_errors=True)
        for p in (self.dir.parent, self.dir.parent.parent):
            try:
                p.rmdir()
            except OSError:
                break


# ── the key pass: zul2.key_pass with a checkpoint (lines tagged `# ck` are the only differences) ──


def key_pass_k(names, V, Np, Nc, Nkeys, n, R=R_DEPTH, hub_lo=HUB_LO, dfmax=DFMAX, small=SMALL, skip_le=HUB_LO, threads=1, every=120):
    """zul2.key_pass with a checkpoint every EVERY_S seconds at a loop-unit boundary and at the end of the pass (module docstring)."""
    from concurrent.futures import ThreadPoolExecutor
    df = names.df
    lo_df = max(2, skip_le + 1)
    keys_in = np.flatnonzero((df >= lo_df) & (df <= dfmax))
    st = {"keys": int(keys_in.size), "skip_le": int(skip_le), "keys_skipped": int(((df >= 2) & (df < lo_df)).sum()),
          "entries": int(df[keys_in].sum()), "sum_df2": int((df[keys_in].astype(np.int64) ** 2).sum()),
          "hub_keys": int((df[keys_in] > hub_lo).sum()), "keys_over_dfmax": int((df > dfmax).sum()), "threads": int(threads)}
    top, hk = Top(n, R), []
    ck = Store(folder_of_row(), key_of(names, V, Np, Nc, Nkeys, n, R, hub_lo, dfmax, small, skip_le, threads))  # ck
    pos, done_s = ck.resume(top, hk)  # ck
    t0, last = time.time() - done_s, time.time()  # ck: was t0, last = time.time(), time.time()
    dim = V.shape[1]
    pool = ThreadPoolExecutor(max_workers=threads) if threads > 1 else None
    try:
        for m in range(lo_df, min(small, dfmax) + 1):
            if pos[0] > 1 or (pos[0] == 1 and m < pos[1]):  # ck
                continue  # ck
            km = keys_in[df[keys_in] == m]
            per = max(1, (1 << 27) // (m * m * 24 + m * dim * 4))
            for i0 in range(pos[2] if (pos[0] == 1 and m == pos[1]) else 0, km.size, per):  # ck: was for i0 in range(0, km.size, per):
                kb = km[i0:i0 + per]
                P = names.key_node[names.key_ptr[kb][:, None] + np.arange(m)].astype(np.int64)
                X = V[P.ravel()].astype(np.float32).reshape(kb.size, m, dim)
                S = np.matmul(X, X.transpose(0, 2, 1))
                del X
                lo = np.minimum(P[:, :, None], P[:, None, :])
                hi = np.maximum(P[:, :, None], P[:, None, :])
                ex = (lo == hi) | UL.member((lo * n + hi).ravel(), Nkeys).reshape(S.shape)
                del lo, hi
                S[ex] = -np.inf
                del ex
                emit(pool, top, hk, S.reshape(-1, m), P.ravel(), np.repeat(P, m, axis=0), m > hub_lo, R, threads)
                ck.tick(top, hk, (1, m, i0 + per), t0)  # ck
            if time.time() - last > every:
                log(f"  df {m}: {top.offered:,} offered, {top.merges} merges ({time.time() - t0:.0f} s)")
                last = time.time()
        big = keys_in[df[keys_in] > small]
        big = big[np.argsort(df[big], kind="stable")]
        for i, kk in enumerate(big.tolist()):
            if pos[0] == 3 or (pos[0] == 2 and i < pos[1]):  # ck
                continue  # ck
            P = names.key_node[names.key_ptr[kk]:names.key_ptr[kk + 1]].astype(np.int64)
            m = P.size
            X = V[P].astype(np.float32)
            B = max(1, min(m, (1 << 26) // m))
            for r0 in range(pos[2] if (pos[0] == 2 and i == pos[1]) else 0, m, B):  # ck: was for r0 in range(0, m, B):
                r1 = min(m, r0 + B)
                S2 = X[r0:r1] @ X.T
                S2[np.arange(r1 - r0), np.arange(r0, r1)] = -np.inf
                rep, nb = gather_ranges(Np, Nc, P[r0:r1])
                if nb.size:
                    pos_nb = np.searchsorted(P, nb)  # ck: was pos = np.searchsorted(P, nb)
                    pos_nb[pos_nb == m] = m - 1  # ck: was pos[pos == m] = m - 1
                    hit = P[pos_nb] == nb  # ck: was hit = P[pos] == nb
                    S2[rep[hit], pos_nb[hit]] = -np.inf  # ck: was S2[rep[hit], pos[hit]] = -np.inf
                emit(pool, top, hk, S2, P[r0:r1], P, m > hub_lo, R, threads)
                del S2
                ck.tick(top, hk, (2, i, r1) if r1 < m else (2, i + 1, 0), t0)  # ck
            if time.time() - last > every:
                log(f"  big key {i + 1:,}/{big.size:,} (df {m:,}): {top.offered:,} offered, {top.merges} merges ({time.time() - t0:.0f} s)")
                last = time.time()
        ck.tick(top, hk, (3, 0, 0), t0, force=True)  # ck
    finally:
        if pool is not None:
            pool.shutdown()
    top.flush()
    cat = (lambda j, dt: np.concatenate([h[j] for h in hk]).astype(dt) if hk else np.zeros(0, dt))  # noqa: E731
    H = {"hk_a": cat(0, np.int32), "hk_b1": cat(1, np.int32), "hk_s1": cat(2, np.float32), "hk_b2": cat(3, np.int32),
         "hk_s2": cat(4, np.float32)}
    st.update(offered=int(top.offered), merges=int(top.merges), seconds_pass=round(time.time() - t0, 1),
              hk_rows=int(H["hk_a"].size), truncated_rows=int(np.isfinite(top.tmax).sum()))
    return top, H, st


def folder_of_row():
    return RUN["row"]["folder"] if RUN["row"] else CKPT / "adhoc"


def install():
    """Replace zul2.key_pass (the attribute zul2u and zul2m reach as L2.key_pass and zul2's own cmd_hbuild reads as a global)."""
    key_pass_k.__wrapped__ = ORIG             # inspect.getsource(key_pass) (zul2's build identity) still reads the original's source
    L2.key_pass = key_pass_k
    assert L2.key_pass is key_pass_k, "the patch did not take"


def parse_row(modname, mod, rest):
    cmd = rest[0] if rest else "-"
    opt = {}
    for flag in ("--ds", "--enc"):
        if flag in rest and rest.index(flag) + 1 < len(rest):
            opt[flag] = rest[rest.index(flag) + 1]
    ds = opt.get("--ds") or getattr(mod, "DS", None) or "-"
    enc = opt.get("--enc", "-")
    path = Path(mod.__file__).read_bytes().replace(b"\r\n", b"\n")
    stray = None
    if cmd == "hbuild" and enc != "-":
        if modname == "zul2m":
            stray = mod.DOUT / f"h_{enc}.tmp.npz"
        elif ds != "-":
            stray = (L2.OUT if modname == "zul2" else mod.OUT) / ds / f"h_{enc}.tmp.npz"
    rid = f"{modname}-{cmd}-{ds}-{enc}"
    return {"id": rid, "folder": CKPT / rid, "module_sha": hashlib.sha256(path).hexdigest(), "stray": stray}


def drop_stray(p):
    """The temporary file of a killed zul2.save_npz of this row (h_<enc>.tmp.npz in the row's own output folder)."""
    if p is not None and p.name.startswith("h_") and p.name.endswith(".tmp.npz") and p.is_file():
        log(f"zul2k: removed the stray {p.name} ({p.stat().st_size / 1e9:.2f} GB) of a killed run of this row")
        p.unlink()


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "wselftest":
        return selftest()
    if argv and argv[0] == "wtime":
        return timing(argv[1:])
    if argv and argv[0] == "wchild":
        return child(argv[1:])
    if not argv or argv[0] not in MODULES:
        print(__doc__)
        return 2
    modname, rest = argv[0], argv[1:]
    sys.argv = [modname + ".py"] + rest
    mod = importlib.import_module(modname)
    install()
    assert getattr(mod, "L2", L2) is L2 and (modname == "zul2" or mod.L2.key_pass is key_pass_k), "the patch did not reach the module"
    row = RUN["row"] = parse_row(modname, mod, rest)
    drop_stray(row["stray"])
    t0 = time.time()
    rc = mod.main(rest)
    if RUN["saves"] or RUN["resumed"]:
        log(f"zul2k: {RUN['saves']} checkpoints ({RUN['bytes'] / 1e9:.2f} GB written, {RUN['save_s']:.1f} s of saving in {time.time() - t0:.0f} s); "
            f"resumed: {RUN['resumed']}")
    if rc == 0:
        drop_stray(row["stray"])
        Store(row["folder"], "-").clear()
        log(f"zul2k: row {row['id']} done; checkpoint folder removed")
    return rc


# ── self-test (laptop CPU, zul2.selftest's toy) ──────────────────────────────


def toy():
    rng = np.random.default_rng(0)
    n, dim, nk = 260, 6, 70
    V = rng.integers(-3, 4, size=(n, dim)).astype(np.float16)
    V[17] = V[5]
    V[200] = V[5]                                             # exact ties
    sizes = np.r_[rng.integers(2, 14, 50), rng.integers(15, 60, 18), [140, 3]]
    s, g = [], []
    for k, m in enumerate(sizes.tolist()):
        mem = rng.choice(n, size=m, replace=False)
        s += mem.tolist()
        g += [k] * m
    s, g = np.asarray(s, np.int64), np.asarray(g, np.int64)
    names = L2.MB.Names(s, g, nk, n)
    Npairs = L2.ukeys(rng.integers(0, n, 900), rng.integers(0, n, 900), n)
    L2.ukeys(rng.integers(0, n, 600), rng.integers(0, n, 600), n)
    cliq = []
    for j in np.flatnonzero((names.df >= 2) & (names.df <= 10)).tolist():
        P = names.key_node[names.key_ptr[j]:names.key_ptr[j + 1]].astype(np.int64)
        ia, ib = np.triu_indices(P.size, 1)
        cliq.append(P[ia] * n + P[ib])
    return names, V, {"N": Npairs, "C": np.union1d(Npairs, np.concatenate(cliq))}, n


CONFIGS = ((5, 8, 10, 0, 1, "N"), (8, 16, 20, 0, 3, "N"), (24, 64, 25, 0, 2, "N"), (5, 8, 10, 10, 2, "C"), (5, 8, 12, 0, 1, "C"))


def run_cfg(fn, tw, ci, V=None, **over):
    names, V0, Nk, n = tw
    R, small, hub_lo, skip, thr, nk = CONFIGS[ci]
    Np, Nc = L2.both_csr(Nk[nk], n)
    kw = dict(R=R, hub_lo=hub_lo, dfmax=100, small=small, skip_le=skip, threads=thr, every=1e9)
    kw.update(over)
    return fn(names, V0 if V is None else V, Np, Nc, Nk[nk], n, **kw)


def same(x, y, what):
    (tx, Hx, sx), (ty, Hy, sy) = x, y
    for a in ("b", "s", "tmax"):
        assert getattr(tx, a).dtype == getattr(ty, a).dtype and getattr(tx, a).tobytes() == getattr(ty, a).tobytes(), (what, a)
    assert (tx.merges, tx.offered) == (ty.merges, ty.offered), (what, "counters")
    assert Hx.keys() == Hy.keys()
    for k in Hx:
        assert Hx[k].dtype == Hy[k].dtype and Hx[k].tobytes() == Hy[k].tobytes(), (what, k)
    assert {k: v for k, v in sx.items() if k != "seconds_pass"} == {k: v for k, v in sy.items() if k != "seconds_pass"}, (what, "st")


def body_lines(src, mine):
    out = []
    lines = src.split("\n")
    i = next(j for j, x in enumerate(lines) if "from concurrent.futures import ThreadPoolExecutor" in x)
    for x in lines[i:]:
        x = x.rstrip()
        if mine:
            if x.endswith("# ck"):
                continue
            if "# ck: was " in x:
                x = x[:len(x) - len(x.lstrip())] + x.split("# ck: was ", 1)[1].strip()
        out.append(x)
    while out and not out[-1]:
        out.pop()
    return out


def selftest():
    import tempfile
    keep_merge, keep_every, keep_row = L2.MERGE_AT, EVERY_S, RUN["row"]
    base = ROOT / "outputs" / "ul2k" / "wselftest"
    shutil.rmtree(base, ignore_errors=True)
    g = globals()
    try:
        L2.MERGE_AT = 500                                     # many merges on the toy
        g["EVERY_S"] = 0.0                                    # a save at every loop unit
        # 1. the loop body is the original's, line for line (only the `# ck` lines differ)
        mine, theirs = body_lines(inspect.getsource(key_pass_k.__code__), True), body_lines(inspect.getsource(ORIG), False)
        assert mine == theirs, [(a, b) for a, b in zip(mine, theirs) if a != b][:3]
        assert key_pass_k.__defaults__ == ORIG.__defaults__
        ntag = sum(1 for x in inspect.getsource(key_pass_k.__code__).split("\n") if "# ck" in x)
        print(f"zul2k self-test 1: loop body == zul2.key_pass line for line ({len(theirs)} lines; {ntag} tagged lines differ)")
        # 2. the patch takes; build identities are those of the unwrapped code
        import zul2u
        import zul2c
        before = (L2.build_sha(L2.H_FUNCS), L2.build_sha(L2.NT_FUNCS), zul2u.build_sha(zul2u.H_FUNCS), zul2c.build_sha(zul2c.G_FUNCS))
        install()
        after = (L2.build_sha(L2.H_FUNCS), L2.build_sha(L2.NT_FUNCS), zul2u.build_sha(zul2u.H_FUNCS), zul2c.build_sha(zul2c.G_FUNCS))
        assert before == after, "build identity changed by the wrapper"
        assert zul2u.L2.key_pass is key_pass_k and inspect.getsource(L2.key_pass) == inspect.getsource(ORIG)
        print("zul2k self-test 2: L2.key_pass is the wrapper; zul2, zul2u and zul2c build_sha equal before and after the patch")
        tw = toy()
        # 3. original == uninterrupted == killed (and killed twice) and resumed, bit for bit
        tot_saves = 0
        for ci in range(len(CONFIGS)):
            orig = run_cfg(ORIG, tw, ci)
            RUN["row"] = {"id": f"wselftest-{ci}", "folder": base / f"c{ci}", "module_sha": "-"}
            folder = RUN["row"]["folder"]

            def go(fault=None, skip_keep=False):
                RUN.update(saves=0, save_s=0.0, bytes=0, resumed=None)
                FAULT.update(after_saves=fault, mid_save=None, hard=False)
                try:
                    return run_cfg(key_pass_k, tw, ci)
                except Killed:
                    return "killed"
                finally:
                    FAULT.update(after_saves=None)
            shutil.rmtree(folder, ignore_errors=True)
            r = go()
            assert RUN["resumed"] is None
            S = RUN["saves"]
            same(orig, r, f"cfg{ci} uninterrupted")
            tot_saves += S
            r = go()                                          # the folder holds the end-of-pass save: resume reads it and finishes
            assert RUN["resumed"] and RUN["resumed"][0] == (3, 0, 0), RUN["resumed"]
            same(orig, r, f"cfg{ci} from the end-of-pass save")
            for k in sorted({1, 2, S // 2, S - 1, S} & set(range(1, S + 1))):
                shutil.rmtree(folder, ignore_errors=True)
                assert go(k) == "killed"
                r = go()
                assert RUN["resumed"] is not None and RUN["resumed"][1] == k, (RUN["resumed"], k)
                same(orig, r, f"cfg{ci} killed after save {k} of {S}")
            shutil.rmtree(folder, ignore_errors=True)         # two kills, three runs
            assert go(2) == "killed"
            assert go(2) == "killed" and RUN["resumed"][1] == 2
            r = go()
            assert RUN["resumed"][1] == 4
            same(orig, r, f"cfg{ci} killed twice")
            print(f"zul2k self-test 3.{ci}: R={CONFIGS[ci][0]} small={CONFIGS[ci][1]} skip={CONFIGS[ci][3]} threads={CONFIGS[ci][4]}: "
                  f"original == uninterrupted == killed/resumed (saves {S}; kills after 1, 2, {S // 2}, {S - 1}, {S}; two kills): bit for bit")
        # 4. a checkpoint of another key is ignored
        names, V, Nk, n = tw
        RUN["row"] = {"id": "wselftest-f", "folder": base / "f", "module_sha": "-"}
        folder = RUN["row"]["folder"]

        def seed():
            shutil.rmtree(folder, ignore_errors=True)
            RUN.update(saves=0, resumed=None)
            FAULT.update(after_saves=2, hard=False)
            try:
                run_cfg(key_pass_k, tw, 0)
                raise AssertionError("no kill")
            except Killed:
                pass
            FAULT.update(after_saves=None)
            RUN.update(saves=0, resumed=None)
        V2 = V.copy()
        V2[3] = V2[3] + 1
        keep_sha = L2.script_sha256
        cases = {"V": (0, {"V": V2}, None), "R": (0, {"R": 6}, None), "small": (0, {"small": 9}, None), "hub_lo": (0, {"hub_lo": 11}, None),
                 "skip_le": (0, {"skip_le": 1}, None), "threads": (0, {"threads": 2}, None), "dfmax": (0, {"dfmax": 99}, None),
                 "other config": (1, {}, None), "module sha": (0, {}, ("module_sha", "x")), "zul2 sha": (0, {}, ("zul2", "x"))}
        for what, (ci, over, tweak) in cases.items():
            seed()
            if tweak and tweak[0] == "module_sha":
                RUN["row"]["module_sha"] = tweak[1]
            if tweak and tweak[0] == "zul2":
                L2.script_sha256 = lambda: "x"
            try:
                r = run_cfg(key_pass_k, tw, ci, **over)
            finally:
                L2.script_sha256 = keep_sha
                RUN["row"]["module_sha"] = "-"
            assert RUN["resumed"] is None, what
            same(run_cfg(ORIG, tw, ci, **over), r, f"foreign {what}")
        seed()
        r = run_cfg(key_pass_k, tw, 0)
        assert RUN["resumed"] and RUN["resumed"][1] == 2
        same(run_cfg(ORIG, tw, 0), r, "same key")
        print(f"zul2k self-test 4: a checkpoint of another key ({', '.join(cases)}) is ignored and the run equals the original; the same "
              f"key resumes")
        # 5. a hard kill (os._exit, no cleanup) after a save and in the middle of a save; the resumed run equals the original
        for ci, fault in ((0, "after:3"), (3, "mid:3:4"), (2, "after:2")):
            folder = base / f"h{ci}"
            shutil.rmtree(folder, ignore_errors=True)
            p = subprocess.run([sys.executable, str(Path(__file__)), "wchild", str(ci), str(folder), fault], capture_output=True, text=True,
                               cwd=str(ROOT))
            assert p.returncode == 137, (p.returncode, p.stdout[-300:], p.stderr[-600:])
            if fault.startswith("mid"):
                have = [s for s in ("a", "b") if (folder / s / "meta.json").exists()]
                assert have == ["b"], have                    # save 3's slot (a) died before its meta; save 2's slot (b) is whole
            RUN["row"] = {"id": f"wselftest-h{ci}", "folder": folder, "module_sha": "-"}
            RUN.update(saves=0, resumed=None)
            FAULT.update(after_saves=None, mid_save=None)
            r = run_cfg(key_pass_k, tw, ci)
            want = int(fault.split(":")[1]) if fault.startswith("after") else int(fault.split(":")[1]) - 1
            assert RUN["resumed"] and RUN["resumed"][1] == want, (RUN["resumed"], fault)
            same(run_cfg(ORIG, tw, ci), r, f"hard kill {fault}")
            print(f"zul2k self-test 5.{ci}: process killed hard ({fault}; rc 137), resumed from save {want}: equals the original bit for bit")
        folder = base / "trunc"                               # a newest slot whose file is short is not used: the older slot is
        shutil.rmtree(folder, ignore_errors=True)
        RUN["row"] = {"id": "wselftest-t", "folder": folder, "module_sha": "-"}
        RUN.update(saves=0, resumed=None)
        FAULT.update(after_saves=3, hard=False)
        try:
            run_cfg(key_pass_k, tw, 0)
            raise AssertionError("no kill")
        except Killed:
            pass
        FAULT.update(after_saves=None)
        newest = folder / "a" / "b.npy"
        newest.write_bytes(newest.read_bytes()[:-100])
        RUN.update(saves=0, resumed=None)
        r = run_cfg(key_pass_k, tw, 0)
        assert RUN["resumed"] and RUN["resumed"][1] == 2, RUN["resumed"]
        same(run_cfg(ORIG, tw, 0), r, "truncated newest slot")
        print("zul2k self-test 5.4: the newest slot with a short file is not used; the older slot resumes: equals the original")
        # 6. clean-up: the row's folder goes when the row ends with 0; a stray temporary file of the row's own is dropped
        folder = base / "done"
        folder.mkdir(parents=True)
        (folder / "a").mkdir()
        (folder / "a" / "x.npy").write_bytes(b"0")
        st = Store(folder, "-")
        (base / "h_e3v.tmp.npz").write_bytes(b"0")
        (base / "other.tmp.npz").write_bytes(b"0")
        drop_stray(base / "h_e3v.tmp.npz")
        drop_stray(base / "other.tmp.npz")
        assert not (base / "h_e3v.tmp.npz").exists() and (base / "other.tmp.npz").exists()
        st.clear()
        assert not folder.exists()
        print("zul2k self-test 6: the stray h_<enc>.tmp.npz of the row is removed (another name is not); a done row's folder is removed")
        # 7. end to end through the argv form: zul2.selftest runs on the patched key_pass and leaves no checkpoint
        p = subprocess.run([sys.executable, str(Path(__file__)), "zul2", "selftest"], capture_output=True, text=True, cwd=str(ROOT))
        assert p.returncode == 0 and "zul2 self-test: OK" in p.stdout, (p.returncode, p.stdout[-400:], p.stderr[-800:])
        assert not [x for x in CKPT.glob("zul2-selftest*")], list(CKPT.glob("zul2-selftest*"))
        print("zul2k self-test 7: `zul2k.py zul2 selftest` -> zul2 self-test: OK on the patched key_pass; its checkpoint folder is gone")
        print(f"zul2k self-test: OK ({tot_saves} saves over {len(CONFIGS)} configs)")
        return 0
    finally:
        g["EVERY_S"] = keep_every
        L2.MERGE_AT, RUN["row"] = keep_merge, keep_row
        FAULT.update(after_saves=None, mid_save=None, hard=False)
        shutil.rmtree(base, ignore_errors=True)
        try:
            base.parent.rmdir()
        except OSError:
            pass
        _ = tempfile


def child(args):
    """wchild <cfg> <folder> after:<k> | mid:<k>:<i>: a toy key pass that dies hard (rc 137) at the given save."""
    ci, folder, fault = int(args[0]), Path(args[1]), args[2].split(":")
    L2.MERGE_AT = 500
    globals()["EVERY_S"] = 0.0
    RUN["row"] = {"id": f"wselftest-h{ci}", "folder": folder, "module_sha": "-"}
    FAULT["hard"] = True
    if fault[0] == "after":
        FAULT["after_saves"] = int(fault[1])
    else:
        FAULT["mid_save"] = (int(fault[1]), int(fault[2]))
    run_cfg(key_pass_k, toy(), ci)
    return 0


# ── timing of a save at size ─────────────────────────────────────────────────


def timing(args):
    gb = float(args[args.index("--gb") + 1]) if "--gb" in args else 2.6
    buf_gb = float(args[args.index("--buf-gb") + 1]) if "--buf-gb" in args else 1.44
    folder = ROOT / "outputs" / "ul2k" / "wtime"
    shutil.rmtree(folder, ignore_errors=True)
    tile = np.random.default_rng(1).integers(0, 2 ** 31 - 1, size=25_000_000, dtype=np.int32)      # 100 MB of random bytes

    def tiles(nbytes, dt):
        out, left = [], int(nbytes) // 4
        while left > 0:
            k = min(left, tile.size)
            out.append(tile[:k].view(dt))
            left -= k
        return out
    n = int(gb * 1e9 / (24 * 8 + 4))
    st = Store(folder, "t")
    for label, nbuf_bytes in (("Top only (b, s, tmax)", 0), (f"Top + {buf_gb} GB unflushed buffer + 0.2 GB hub rows", buf_gb * 1e9)):
        arrs = {"b": tiles(n * 24 * 4, DT["b"]), "s": tiles(n * 24 * 4, DT["s"]), "tmax": tiles(n * 4, DT["tmax"])}
        for x in ("bufa", "bufb", "bufs"):
            arrs[x] = tiles(nbuf_bytes / 3, DT[x])
        for j in range(5):
            arrs[f"hk{j}"] = tiles(0.04e9 if nbuf_bytes else 0, DT[f"hk{j}"])
        t = time.time()
        nbytes = st.write_slot(folder / "a", arrs, {"seq": 1})
        dt = time.time() - t
        t = time.time()
        got = 0
        for name in DT:
            with open(folder / "a" / f"{name}.npy", "rb") as f:
                while True:
                    c = f.read(1 << 27)
                    if not c:
                        break
                    got += len(c)
        rt = time.time() - t
        print(f"zul2k timing: {label}: n={n:,} rows, R=24: {nbytes / 1e9:.2f} GB written + fsync in {dt:.1f} s ({nbytes / 1e6 / dt:.0f} MB/s); "
              f"read back {rt:.1f} s")
    shutil.rmtree(folder, ignore_errors=True)
    try:
        folder.parent.rmdir()
    except OSError:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
