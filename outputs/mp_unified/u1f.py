"""U1f: U1c's screen on GU_1024 (docs/U1F_RETRAIN_ON_GU1024.md). U1c's code, unchanged, with its roots moved:
the graph root to U1e's chosen build (outputs/u1e/c1024), every output root to outputs/u1f.

    python outputs/host_ops/pylib_run.py outputs/mp_unified/u1f.py host pools --dataset D --carves screen --threads 5 --host
    python outputs/mp_unified/u1f.py host copy-bases
    python outputs/mp_unified/u1f.py host gate --screen
    python outputs/mp_unified/u1f.py adopt --dataset metaqa|squad
    python outputs/mp_unified/u1f.py look  --dataset D --carve C --host --full [--shard i/n]     (look_u1c)
    python outputs/mp_unified/u1f.py look2 --dataset D --carve C --host --full [--shard i/n]     (look_u1c2, webqsp)
    python outputs/mp_unified/u1f.py zu build|train|read|compare|pair|recall ...                  (zu1c)
    python outputs/mp_unified/u1f.py --selftest

metaqa and squad: GU_1024 is GU (U1e's builds: no link removed on metaqa; squad has no graph), so U1c's pools, looks and
caches of those two are U1f's. `adopt` hard-links them (a copy where a link fails) into outputs/u1f after checking
U1e's build record; U1f's gate then checks them as it checks the rest. Their chain and link builds run again under
U1f's roots.
"""
import json
import os
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (HERE, ROOT / "scripts", ROOT / "src", ROOT / "outputs" / "step4f"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

OUT = ROOT / "outputs" / "u1f"
U1C = ROOT / "outputs" / "u1c"
U1D = ROOT / "outputs" / "u1d"
GRAPH = ROOT / "outputs" / "u1e" / "c1024"
CAP = 1024
SAME = ("metaqa", "squad")
SIX = ("metaqa", "musique", "squad", "2wiki", "hotpotqa", "webqsp")
DOC = "docs/U1F_RETRAIN_ON_GU1024.md"
ADOPTED = ("pools", "look", "cache")


def log(m):
    print(time.strftime("[%H:%M:%S] ") + m, flush=True)


def write_json(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1), encoding="utf-8")
    os.replace(tmp, p)


def sha_src(p):
    import hashlib
    return hashlib.sha256(Path(p).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def graph_info(ds):
    """U1e's build record of ds at c = 1,024; refuses a dataset where GU_1024 is GU (those are adopted)."""
    if ds in SAME:
        raise SystemExit(f"u1f: on {ds} GU_1024 is GU; its pools, looks and caches are U1c's (u1f.py adopt)")
    b = json.loads((GRAPH / ds / "build.json").read_text(encoding="utf-8"))
    if b.get("same_as_gu") or b.get("cap") != CAP or not b.get("npz_sha256"):
        raise SystemExit(f"u1f: {GRAPH / ds / 'build.json'} is not GU_{CAP}'s build")
    return {"variant": b["variant"], "npz_sha256": b["npz_sha256"], "cap": CAP, "edges": b.get("edges"),
            "removed_share": b.get("removed_share")}


def stamp(p, **extra):
    p = Path(p)
    if p.exists():
        rec = json.loads(p.read_text(encoding="utf-8"))
        rec["u1f"] = {"declared_in": DOC, "u1f_sha256": sha_src(__file__), **extra}
        write_json(p, rec)


# ── scripts/u1c_host.py: pools, copy-bases, gate ─────────────────────────────


def on_host_roots(ds=None):
    import u1c_host as UC
    UC.OUT, UC.POOLS, UC.LOOK, UC.CACHE = OUT, OUT / "pools", OUT / "look", OUT / "cache"
    if ds is not None:
        UC.H.U1D, UC.H.OUT = GRAPH, GRAPH                   # GU_1024's graph and its step 4h coverage record
    return UC


def cmd_host(argv):
    if not argv or argv[0] not in ("pools", "copy-bases", "gate"):
        raise SystemExit("u1f host: pools, copy-bases or gate")
    ds = argv[argv.index("--dataset") + 1] if "--dataset" in argv else None
    if argv[0] == "pools":
        info = graph_info(ds)
    UC = on_host_roots(ds if argv[0] == "pools" else None)
    saved = sys.argv
    sys.argv = ["u1c_host.py"] + list(argv)
    try:
        rc = UC.main()
    finally:
        sys.argv = saved
    if argv[0] == "pools" and rc == 0:
        stamp(OUT / "pools" / ds / "pools.json", graph=info, graph_root=GRAPH.relative_to(ROOT).as_posix())
    if argv[0] == "gate":
        stamp(OUT / ("gate-screen.json" if "--screen" in argv else "gate.json"))
    return rc


# ── metaqa and squad: U1c's files are U1f's ──────────────────────────────────


def cmd_adopt(argv):
    ds = argv[argv.index("--dataset") + 1]
    if ds not in SAME:
        raise SystemExit(f"u1f adopt: only {SAME}, where GU_1024 is GU")
    if ds != "squad":
        b = json.loads((GRAPH / ds / "build.json").read_text(encoding="utf-8"))
        u = json.loads((U1D / ds / "build.json").read_text(encoding="utf-8"))
        if not b.get("same_as_gu") or b.get("u1d_npz_sha256") != u["npz_sha256"] or b.get("mention_kept") != b.get(
                "mention_edges"):
            raise SystemExit(f"u1f adopt: {ds}'s GU_{CAP} is not GU ({b})")
    rec = {"declared_in": DOC, "dataset": ds, "from": "outputs/u1c", "linked": 0, "copied": 0, "bytes": 0, "dirs": {},
           "u1f_sha256": sha_src(__file__)}
    for sub in ADOPTED:
        src, dst = U1C / sub / ds, OUT / sub / ds
        if not src.is_dir():
            raise SystemExit(f"u1f adopt: {src} missing")
        n = 0
        for dp, _dn, fn in os.walk(src):
            for name in fn:
                if name.endswith(".tmp") or ".tmp." in name:
                    raise SystemExit(f"u1f adopt: {Path(dp) / name}: a write in progress in U1c's tree")
                s = Path(dp) / name
                d = dst / s.relative_to(src)
                d.parent.mkdir(parents=True, exist_ok=True)
                if d.exists():
                    if d.stat().st_size != s.stat().st_size or not os.path.samefile(s, d) and \
                            s.read_bytes() != d.read_bytes():
                        raise SystemExit(f"u1f adopt: {d} exists and is not {s}")
                    continue
                try:
                    os.link(s, d)
                    rec["linked"] += 1
                except OSError:
                    shutil.copy2(s, d)
                    rec["copied"] += 1
                rec["bytes"] += s.stat().st_size
                n += 1
        rec["dirs"][sub] = n
    import hashlib
    rec["pools_manifest_sha256"] = hashlib.sha256((OUT / "pools" / ds / "pools.json").read_bytes()).hexdigest()
    write_json(OUT / "adopted" / f"{ds}.json", rec)
    log(f"u1f adopt {ds}: {rec['linked']} linked, {rec['copied']} copied, {rec['bytes'] / 1e9:.2f} GB ({rec['dirs']})")
    return 0


# ── look_u1c / look_u1c2 ─────────────────────────────────────────────────────


def cmd_look(argv, kb):
    ds = argv[argv.index("--dataset") + 1]
    info = graph_info(ds)
    import look_u1c as LU
    LU.OUT, LU.POOLS, LU.LOOK, LU.U1D = OUT, OUT / "pools", OUT / "look", GRAPH
    if kb:
        import look_u1c2 as L2
        if L2.LU is not LU:
            raise SystemExit("u1f: look_u1c2 does not wrap look_u1c")
        rc = L2.main(list(argv))
    else:
        if ds in ("metaqa", "webqsp"):
            raise SystemExit("u1f: the KB datasets' looks run look_u1c2 (U1c's amendment 1): use look2")
        rc = LU.main(list(argv))
    shard = argv[argv.index("--shard") + 1] if "--shard" in argv else None
    sh = LU.LX.L8.parse_shard(shard)
    tag = f"_{sh[0]}of{sh[1]}" if sh else ""
    carve = argv[argv.index("--carve") + 1]
    stamp(OUT / "look" / ds / carve / f"pools{tag}.json", graph=info)
    return rc


# ── zu1c: builds, fits, reads and calls ──────────────────────────────────────


def on_zu_roots():
    import zu1c as ZU
    ZU.OUT = OUT
    ZU.LOOK, ZU.CACHE = OUT / "look", OUT / "cache"
    ZU.CH_RM, ZU.CH_ZRC, ZU.LINKS = OUT / "chains" / "rmatch", OUT / "chains" / "zrc", OUT / "links"
    ZU.GATE = OUT / "gate-screen.json"
    ZU.gate_passed.__defaults__ = (ZU.GATE,)
    return ZU


def cmd_zu(argv):
    ZU = on_zu_roots()
    rc = ZU.main(list(argv))
    cmd = argv[0] if argv else None
    if cmd == "train" and rc == 0:
        name, out_root = ZU.S2.where(argv)
        stamp(out_root / name / "screen.json", graph=f"GU_{CAP} (outputs/u1e/c{CAP}/<D>/graph_structural_u.npz; "
                                                     f"GU on metaqa and squad)", roots="outputs/u1f")
    if cmd in ("pair", "recall"):
        out = Path(argv[argv.index("--out") + 1])
        md = out.with_suffix(".md")
        if md.exists():
            t = md.read_text(encoding="utf-8")
            t = t.replace("U1c (docs/U1C_RETRAIN_ON_U.md): ", f"U1f ({DOC}; U1c's screen on GU_{CAP}): ", 1)
            t = t.replace("on U1d's graph and step 4h's pools", f"on GU_{CAP} and its step 4h pools", 1)
            t = t.replace("R@5 (U1c) |", "R@5 (U1f) |")
            md.write_text(t, encoding="utf-8")
        js = out.with_suffix(".json")
        if js.exists():
            r = json.loads(js.read_text(encoding="utf-8"))
            r.update({"stage": "U1f", "declared_in_u1c": r.get("declared_in")})
            r["declared_in"] = DOC
            write_json(js, r)
    return rc


# ── selftest ─────────────────────────────────────────────────────────────────


def selftest():
    global U1C, OUT
    import tempfile
    UC = on_host_roots("musique")
    assert UC.POOLS == OUT / "pools" and UC.CACHE == OUT / "cache" and UC.H.U1D == GRAPH and UC.H.OUT == GRAPH
    assert UC.CHOICE.parent.name == "step4h", UC.CHOICE              # step 4h's W1, resolved before the move
    ZU = on_zu_roots()
    assert ZU.CACHE == OUT / "cache" and ZU.gate_passed.__defaults__ == (OUT / "gate-screen.json",)
    try:
        ZU.cache_root_of(["--cache-root", "outputs/u1c/cache"])
        raise AssertionError("U1c's cache root passed")
    except SystemExit:
        pass
    ZU.cache_root_of(["--cache-root", "outputs/u1f/cache"])
    with ZU.on_u1c():
        assert ZU.RM.ChainCarveBase.CH_ROOT == OUT / "chains" / "rmatch" and ZU.ZL.LinkCarveBase.LK_ROOT == OUT / "links"
    import look_u1c as LU
    import look_u1c2 as L2
    assert L2.LU is LU
    for ds in SAME:
        try:
            graph_info(ds)
            raise AssertionError(f"{ds} passed")
        except SystemExit:
            pass
    # adopt: links (or copies) every file, refuses a different file already there and a write in progress
    saved = U1C, OUT
    tmp = Path(tempfile.mkdtemp(prefix="u1f_"))
    try:
        U1C, OUT = tmp / "u1c", tmp / "u1f"
        for sub in ADOPTED:
            (U1C / sub / "squad" / "a").mkdir(parents=True)
            (U1C / sub / "squad" / "a" / "x.npy").write_bytes(b"12345")
        (U1C / "pools" / "squad" / "pools.json").write_text("{}", encoding="utf-8")
        assert cmd_adopt(["--dataset", "squad"]) == 0
        assert (OUT / "cache" / "squad" / "a" / "x.npy").read_bytes() == b"12345"
        assert cmd_adopt(["--dataset", "squad"]) == 0                                  # a re-run is a no-op
        (OUT / "look" / "squad" / "a" / "x.npy").unlink()
        (OUT / "look" / "squad" / "a" / "x.npy").write_bytes(b"99999")
        try:
            cmd_adopt(["--dataset", "squad"])
            raise AssertionError("a different file passed")
        except SystemExit:
            pass
        (U1C / "cache" / "squad" / "a" / "y.npy.tmp").write_bytes(b"1")
        try:
            cmd_adopt(["--dataset", "squad"])
            raise AssertionError("a write in progress passed")
        except SystemExit:
            pass
    finally:
        U1C, OUT = saved
        shutil.rmtree(tmp, ignore_errors=True)
    on_host_roots("musique")
    on_zu_roots()
    log("u1f selftest: U1c's host, look and zu1c roots moved to outputs/u1f, the graph to GU_1024 (step 4h's W1 kept); "
        "metaqa and squad adopted from U1c, refusing a different file or a write in progress: ok")
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--selftest" in argv:
        return selftest()
    if not argv:
        raise SystemExit("u1f: host, adopt, look, look2 or zu")
    cmd, rest = argv[0], argv[1:]
    if cmd == "host":
        return cmd_host(rest)
    if cmd == "adopt":
        return cmd_adopt(rest)
    if cmd in ("look", "look2"):
        return cmd_look(rest, cmd == "look2")
    if cmd == "zu":
        return cmd_zu(rest)
    raise SystemExit(f"u1f: {cmd}? host, adopt, look, look2 or zu")


if __name__ == "__main__":
    sys.exit(main())
