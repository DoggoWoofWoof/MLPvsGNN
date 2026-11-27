"""STK1 amendment A7 (docs/STK1_STARK_PACKAGE.md section 5.7): the dense blocks of the three STaRK KBs, made on the host card.

Run as:
    python -B -E -P outputs/mp_unified/zstk1d.py selftest                      # laptop (torch needs the user site's packages: -E -P, not -I)
    python -I -B   outputs/mp_unified/zstk1d.py gate                           # host card: gate G-V on Prime's stored (CPU) vectors, and the card's rate
    python -I -B   outputs/mp_unified/zstk1d.py fill --kb mag|amazon           # host card: the missing scratch blocks of both parts' documents, last shard first

What it is. zstk1.py's `encode` makes the e3v parts (gte-modernbert-base, granite-embedding-small-english-r2) on the host CPU, block by block (4,096 rows,
`shard_<s>.blk<b>.npy`), and assembles each 40,000-row shard from its blocks. The card is free (the host restart of 27 Nov), so this module makes the same
blocks on the card: the same texts (T1, `nodes.jsonl`), the same blocks, the same file names, zenc's `encode_texts` (longest first, batches under the part's
own token budget, an out-of-memory halves it), float16 weights as E0N's encodes of the seven datasets, zenc's damaged-row rule, no text cut at the window.
It never assembles a shard, never writes a manifest and never removes a file: the CPU rows (`stk1m-enc`, `stk1a-enc`) stay the assemblers, walk the shards
forward, skip every block that exists, make any missing one themselves and write zenc's manifests, so `zenc3v.cmd_vectors` and `zstk1.py first` are unchanged.
The card's blocks go to `<block>.gtmp.npy` and are moved into place by `os.replace` (the CPU's scratch name is `.tmp.npy`: the two never meet).
`fill` needs `outputs/stark/dense_gpu/gate_v.json` with pass true. Queries are not filled (a KB has 13,323 or fewer; the CPU rows make them in minutes).
`outputs/stark/dense_gpu/fill_<kb>.json` lists every block the card made: the assembled shards' manifests say device cpu, that file is the record.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np  # noqa: E402

import zstk1 as Z  # noqa: E402  (sets CUDA_VISIBLE_DEVICES to "" at import: undone in main for gate and fill, before any card call)

E = Z.E
BLK = Z.BLK
DOC = "docs/STK1_STARK_PACKAGE.md section 5.7 (A7)"
OUT = Z.ROOT / "dense_gpu"
GATE_ROWS = 512                                    # rows read per slice of the gate
GATE_SLICES = ((0, 0), (3, 0))                     # (Prime docs shard, first row in it): the first 512 rows of shards 0 and 3
MIN_COS = float(E.G0_MIN_COS)                      # 0.99, zenc's floor for a re-encoded vector against a stored one


def log(msg):
    print(f"[zstk1d {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def stop(msg):
    raise SystemExit(f"ZSTK1D: {msg}")


def zmods():
    return Z.zmods()


def shard_starts(path, shard=None):
    """Byte offset of the first line of every shard of a jsonl file, and the number of lines (one pass, binary)."""
    shard = int(shard or E.SHARD)
    offs, n, pos = [], 0, 0
    with open(path, "rb") as f:
        for ln in f:
            if n % shard == 0:
                offs.append(pos)
            n += 1
            pos += len(ln)
    return offs, n


def shard_texts(path, off, n_lines, shard=None):
    """The texts of the shard that starts at byte `off` (at most `shard` lines), as zenc.doc_chunks reads them."""
    shard = int(shard or E.SHARD)
    out = []
    with open(path, "rb") as f:
        f.seek(off)
        for ln in f:
            out.append(json.loads(ln)["text"])
            if len(out) == min(shard, n_lines):
                break
    return out


def block_plan(n_rows, blk=None):
    """Block start rows of a shard, last first."""
    blk = int(blk or BLK)
    return list(range(0, n_rows, blk))[::-1]


def block_path(base, s, b, blk=None):
    return base / f"shard_{s:05d}.blk{b // int(blk or BLK):04d}.npy"


def gtmp_path(pp):
    return pp.with_name(pp.stem + ".gtmp.npy")


def cos_stats(a, b):
    """Row-wise cosine of two (n, d) arrays; (min, p1, median, mean)."""
    x, y = np.asarray(a, np.float32), np.asarray(b, np.float32)
    c = (x * y).sum(1) / (np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1))
    return float(c.min()), float(np.percentile(c, 1)), float(np.median(c)), float(c.mean())


def shard_done(base, s):
    jp, npy = base / f"shard_{s:05d}.json", base / f"shard_{s:05d}.npy"
    return jp.is_file() and npy.is_file() and json.loads(jp.read_text(encoding="utf-8")).get("sha256") == E.sha256(npy)


def wjson(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1), encoding="utf-8")
    os.replace(tmp, p)


def load_part(e, Z3):
    m = E.load_model(e, "cuda", Z3.pinned(e))
    if m.default_prompt_name:
        stop(f"{e}: a default prompt would prefix the documents")
    return m


def cmd_gate(a):
    import torch
    Z3, V = zmods()
    if not torch.cuda.is_available():
        stop("no card is visible")
    ds = Z.KB["prime"]
    path = Z.PKG / ds / "nodes.jsonl"
    offs, n = shard_starts(path)
    res = {"stage": "STK1 A7 gate G-V", "declared_in": DOC, "kb": "prime", "min_cos_bar": MIN_COS, "rows_per_slice": GATE_ROWS, "parts": {}}
    ok = True
    for e in V.PARTS:
        m = load_part(e, Z3)
        stored_base = E.OUT / e / ds / "docs"
        vs, ss, toks, secs = [], [], 0, 0.0
        warm = shard_texts(path, offs[0], n)[:64]
        E.encode_texts(m, e, warm, "docs")                       # warm-up, not counted
        for s, r0 in GATE_SLICES:
            texts = shard_texts(path, offs[s], n - s * E.SHARD)[r0:r0 + GATE_ROWS]
            t1 = time.time()
            v, st = E.encode_texts(m, e, texts, "docs")
            torch.cuda.synchronize()
            secs += time.time() - t1
            toks += st["tokens"]
            if st["cut"] or E.damaged(v):
                ok = False
            stored = np.load(stored_base / f"shard_{s:05d}.npy", mmap_mode="r")[r0:r0 + len(texts)]
            vs.append(v)
            ss.append(np.asarray(stored))
        cmin, cp1, cmed, cmean = cos_stats(np.concatenate(vs), np.concatenate(ss))
        rows = sum(len(x) for x in vs)
        res["parts"][e] = {"rows": rows, "cos_min": round(cmin, 6), "cos_p1": round(cp1, 6), "cos_median": round(cmed, 6), "cos_mean": round(cmean, 6),
                           "nodes_per_s": round(rows / secs, 1), "tokens_per_s": round(toks / secs), "seconds": round(secs, 2), "pass": bool(cmin >= MIN_COS)}
        log(f"{e}: {rows} rows, cosine to the stored CPU rows: min {cmin:.5f} p1 {cp1:.5f} median {cmed:.5f}; {rows / secs:.0f} nodes/s, {toks / secs:.0f} tokens/s")
        ok = ok and cmin >= MIN_COS
        del m
        torch.cuda.empty_cache()
    res["pass"] = bool(ok)
    res["utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    wjson(OUT / "gate_v.json", res)
    log(f"G-V: {'PASS' if ok else 'FAIL: the card makes no STaRK block'}")
    return 0 if ok else 1


def cmd_fill(a):
    import torch
    Z3, V = zmods()
    g = OUT / "gate_v.json"
    if not g.is_file() or not json.loads(g.read_text(encoding="utf-8")).get("pass"):
        stop("G-V has not passed (outputs/stark/dense_gpu/gate_v.json)")
    if not torch.cuda.is_available():
        stop("no card is visible")
    ds = Z.KB[a.kb]
    path = Z.PKG / ds / "nodes.jsonl"
    offs, n = shard_starts(path)
    made, t0 = [], time.time()
    rec = {"stage": "STK1 A7 fill", "declared_in": DOC, "kb": a.kb, "nodes": n, "blocks": made}
    for e in V.PARTS:
        m = load_part(e, Z3)
        base = E.OUT / e / ds / "docs"
        base.mkdir(parents=True, exist_ok=True)
        for s in reversed(range(len(offs))):
            if shard_done(base, s):
                continue
            nrows = min(E.SHARD, n - s * E.SHARD)
            texts = shard_texts(path, offs[s], n - s * E.SHARD)
            if len(texts) != nrows:
                stop(f"{e} {ds} shard {s}: {len(texts)} texts for {nrows} rows")
            for b in block_plan(nrows):
                if (base / f"shard_{s:05d}.json").is_file():       # the CPU row assembled this shard meanwhile
                    break
                pp = block_path(base, s, b)
                if pp.is_file():
                    continue
                t1 = time.time()
                v, st = E.encode_texts(m, e, texts[b:b + BLK], "docs")
                torch.cuda.synchronize()
                bad = E.damaged(v)
                if bad or st["cut"]:
                    stop(f"{e} {ds} shard {s} block {b // BLK}: {bad} damaged rows, {st['cut']} texts cut at the window")
                tmp = gtmp_path(pp)
                np.save(tmp, v)
                os.replace(tmp, pp)
                dt = time.time() - t1
                made.append({"part": e, "shard": s, "block": b // BLK, "rows": int(v.shape[0]), "seconds": round(dt, 2), "tokens": int(st["tokens"]), "ooms": int(st["ooms"])})
                log(f"{e} shard {s} block {b // BLK}: {v.shape[0]} rows in {dt:.0f} s ({v.shape[0] / dt:.0f} nodes/s); the card has made {len(made)} blocks in {time.time() - t0:.0f} s")
            rec["seconds"] = round(time.time() - t0, 1)
            wjson(OUT / f"fill_{a.kb}.json", rec)
        del m
        torch.cuda.empty_cache()
    rec["seconds"] = round(time.time() - t0, 1)
    rec["utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    wjson(OUT / f"fill_{a.kb}.json", rec)
    log(f"{a.kb}: the card made {len(made)} blocks in {rec['seconds']} s")
    return 0


def selftest():
    import tempfile
    bad = 0

    def chk(name, ok):
        nonlocal bad
        log(f"  {'ok  ' if ok else 'FAIL'} {name}")
        bad += 0 if ok else 1

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "nodes.jsonl"
        lines = [json.dumps({"text": f"node {i} é"}, ensure_ascii=False) for i in range(25)]
        p.write_bytes(("\n".join(lines) + "\n").encode("utf-8"))
        offs, n = shard_starts(p, shard=10)
        chk("25 lines in shards of 10 start at three offsets", n == 25 and len(offs) == 3 and offs[0] == 0)
        t2 = shard_texts(p, offs[2], n - 20, shard=10)
        chk("the last shard holds its 5 texts, accents intact", t2 == [f"node {i} é" for i in range(20, 25)])
        t1 = shard_texts(p, offs[1], n - 10, shard=10)
        chk("a middle shard holds exactly 10 texts and stops at the shard's end", t1 == [f"node {i} é" for i in range(10, 20)])
        crlf = Path(td) / "crlf.jsonl"
        crlf.write_bytes(("\r\n".join(lines) + "\r\n").encode("utf-8"))
        o2, n2 = shard_starts(crlf, shard=10)
        chk("a CRLF file gives the same shards and texts", n2 == 25 and shard_texts(crlf, o2[1], 15, shard=10) == t1)
        base = Path(td) / "docs"
        base.mkdir()
        pp = block_path(base, 3, 8192)
        chk("block names are zstk1.py's: shard_00003.blk0002.npy", pp.name == "shard_00003.blk0002.npy")
        chk("the card's temporary name is not the CPU's (.tmp.npy)", gtmp_path(pp).name == "shard_00003.blk0002.gtmp.npy" and not gtmp_path(pp).name.endswith(".tmp.npy"))
        chk("blocks are planned last first", block_plan(10000, blk=4096) == [8192, 4096, 0] and block_plan(4096, blk=4096) == [0] and block_plan(1, blk=4096) == [0])
        npy = base / "shard_00001.npy"
        np.save(npy, np.zeros((2, 3), np.float16))
        (base / "shard_00001.json").write_text(json.dumps({"sha256": E.sha256(npy)}), encoding="utf-8")
        chk("a shard with a matching checksum is done", shard_done(base, 1))
        (base / "shard_00001.json").write_text(json.dumps({"sha256": "0" * 64}), encoding="utf-8")
        chk("a shard with a wrong checksum is not done", not shard_done(base, 1))
        chk("a shard with no file is not done", not shard_done(base, 2))
    rng = np.random.default_rng(0)
    a = rng.normal(size=(50, 8)).astype(np.float32)
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    mn, p1, md, mean = cos_stats(a, a)
    chk("identical rows have cosine 1", abs(mn - 1) < 1e-6 and abs(md - 1) < 1e-6)
    b = a.copy()
    b[7] = -a[7]
    chk("one flipped row gives min cosine -1 and median 1", abs(cos_stats(a, b)[0] + 1) < 1e-6 and abs(cos_stats(a, b)[2] - 1) < 1e-6)
    chk("the cosine floor is zenc's 0.99", MIN_COS == 0.99 and GATE_ROWS == 512 and GATE_SLICES == ((0, 0), (3, 0)))
    if bad:
        raise SystemExit(f"zstk1d: selftest FAILED ({bad})")
    log("selftest PASS (13 checks)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")
    sub.add_parser("gate")
    f = sub.add_parser("fill")
    f.add_argument("--kb", choices=("mag", "amazon"), required=True)
    a = ap.parse_args(argv)
    if a.cmd in ("gate", "fill"):
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)               # zstk1.py hid the card at import; these rows are the card's
    if a.cmd == "selftest":
        return selftest()
    return cmd_gate(a) if a.cmd == "gate" else cmd_fill(a)


if __name__ == "__main__":
    sys.exit(main())
