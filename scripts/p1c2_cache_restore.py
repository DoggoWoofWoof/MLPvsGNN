"""P1c2's builds back after the host's clean-up (docs/N2_FRESH_SEED_NULL.md, section 6.1; systems code, no science).

By 27 October 2026 the host's auto clean-up (outputs/host_ops/auto_cleanup.py, its STAGE_BUILD class) had removed
files that N2's trains read, every record staying: 247 of the 663 arrays of P1c2's cache (in all 39 parts) and
the cache's basis_hotpotqa.npz; ent_row, ent_z and ent_m of both parts of webqsp's s1eval chains, in rmatch's and in
zrc's; e_u and e_v of five musique link parts; the kt.npy of three P1c3 KT carves; and step 1's two bases. Each kind is
built again by its committed builder, unchanged, into a root of its own (outputs/p1c2rb/...), and a file goes back only
where it is missing and the rebuilt file's sha256 is the one its record lists. Nothing is written over and no record
changes, so every reader's checks (each array's sha256 against its record, and the cache parts' record shas that the
chains, links and KT records name) hold as before.

    python outputs/mp_unified/lean_cache.py --make-basis hotpotqa --host --out-root outputs/p1c2rb/cache
    python scripts/p1c2_cache_restore.py plan
    python scripts/p1c2_cache_restore.py prep
    python outputs/mp_unified/lean_cache.py --dataset D --carve C --part i/n --host --look-root outputs/p1c2/look \\
        --out-root outputs/p1c2rb/cache
    python scripts/p1c2_cache_restore.py install --what cache
    python scripts/p1c2_cache_restore.py build --what rmatch|zrc|links --dataset D --carve C --host
    python scripts/p1c2_cache_restore.py install --what rmatch|zrc|links
    python outputs/host_ops/pylib_run.py scripts/p1c2_cache_restore.py kt D --carve C --threads T --host
    python scripts/p1c2_cache_restore.py install --what kt
    python scripts/p1c2_cache_restore.py verify
    python scripts/p1c2_cache_restore.py --selftest

plan     what the records list and the disk lacks: each kind's parts (the KT's carves), and both bases in P1c2's root,
         step 1's and the rebuild root. Records and names only; nothing is hashed, and only outputs/p1c2/restore/plan.json
         is written.
prep     the rebuild root's bases. basis_2wiki: P1c2's file (its sha256 its record's) and its record, linked (else
         copied). basis_hotpotqa: made there by lean_cache --make-basis (BLAS 2, as when it was made), and it must be P1c2's,
         the npz's sha256 and its record on every key but the run's own and the code's, else the restore stops (exit 2)
         before anything goes back. Then each root's missing basis goes back from a file holding its record's sha256.
         P1c2's root and the rebuild root gate (exit 1); step 1's is reported.
build    zp1c2.build's call, unchanged but for out_root (the rebuild root's). The carve's cache must be whole first, and for
         zrc rmatch's chains too: zrc reads rmatch's record of each part, as when it was built.
kt       p1c3_kt.feats_stage, unchanged, with its KT root set to the rebuild root (and set back after); the carve's cache
         must be whole first. P1c3's own carve folder is hashed before and after and must not change.
install  per part (per carve for kt) of one kind: the rebuilt record must be the original's on every key but the run's
         own (seconds, utc, placement, peak memory, row-gather time) and the arrays' sha256s; a code sha that differs is
         reported, not refused. Each missing file goes back where the rebuilt file's sha256 is the record's (a hard link,
         else a copy renamed into place); a file that exists is never touched. Each present array is counted as
         reproduced when the rebuilt record's sha256 is the original's. Then every file of the kind is hashed against its
         record. Recorded in outputs/p1c2/restore/install_<kind>.json; exit 1 unless the kind is whole.
verify   every file of every kind, and P1c2's two bases, against their records (step 1's bases reported); the gate that
         N2's trains wait on (outputs/p1c2/restore/verify.json, WHOLE or NOT_WHOLE; exit 1 unless WHOLE).
Stdlib only at import; build imports zp1c2 and kt imports p1c3_kt, each as its own items ran it.
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import time
import types
from collections import Counter
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
DECLARED = "docs/N2_FRESH_SEED_NULL.md"
# The originals and their rebuild roots, as literal paths (the clean-up pins the folders a queued job's script names).
PATHS = {"cache": "outputs/p1c2/cache", "look": "outputs/p1c2/look", "rmatch": "outputs/p1c2/chains/rmatch",
         "zrc": "outputs/p1c2/chains/zrc", "links": "outputs/p1c2/links", "kt": "outputs/p1c3/kt",
         "step1": "outputs/step1/cache", "records": "outputs/p1c2/restore"}
REBUILD = {"cache": "outputs/p1c2rb/cache", "rmatch": "outputs/p1c2rb/chains/rmatch",
           "zrc": "outputs/p1c2rb/chains/zrc", "links": "outputs/p1c2rb/links", "kt": "outputs/p1c2rb/kt"}
KINDS = ("cache", "rmatch", "zrc", "links", "kt")
BASES = ("2wiki", "hotpotqa")
RUN_KEYS = ("seconds", "utc", "placement", "peak_rss_bytes", "row_gather_s")
CODE_KEYS = ("script_sha256", "module_sha256", "lean_mlp3_sha256", "rmatch_sha256", "chaincov_sha256",
             "p1c2_script_sha256", "p1e_script_sha256", "p1d_script_sha256")
# The code each record names, against today's file (plan reports it: a build reproduces only on its own code).
MU = "outputs/mp_unified/"
CODE = {"cache": {"script_sha256": MU + "lean_cache.py"}, "rmatch": {"script_sha256": MU + "rmatch.py"},
        "zrc": {"script_sha256": MU + "zrc.py", "rmatch_sha256": MU + "rmatch.py", "chaincov_sha256": MU + "chaincov.py"},
        "links": {"script_sha256": MU + "zlink.py"},
        "kt": {"script_sha256": "scripts/p1c3_kt.py", "p1c2_script_sha256": "scripts/p1c2_pools.py",
               "p1e_script_sha256": "scripts/p1e_anchor_kind.py", "p1d_script_sha256": "scripts/p1d_free_pool.py"},
        "basis": {"script_sha256": MU + "lean_cache.py", "lean_mlp3_sha256": MU + "lean_mlp3.py"}}
PART = re.compile(r"part_(\d+)of(\d+)")
assert len(set(KINDS)) == len(KINDS) and set(KINDS) == set(REBUILD) and set(CODE) == set(KINDS) | {"basis"}
assert set(k for c in CODE.values() for k in c) <= set(CODE_KEYS)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha_file(p, block=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(block), b""):
            h.update(b)
    return h.hexdigest()


def sha_src(p):
    return hashlib.sha256(Path(p).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def read_json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def write_json(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1), encoding="utf-8")
    os.replace(tmp, p)


class Roots:
    """Where the originals and their rebuilds are: the workspace's by default, the self-test's toys under its own."""

    def __init__(self, base=ROOT):
        self.base = Path(base)
        self.of = {k: self.base / v for k, v in PATHS.items()}
        self.rb = {k: self.base / v for k, v in REBUILD.items()}

    def rel(self, p):
        try:
            return Path(p).resolve().relative_to(self.base.resolve()).as_posix()
        except ValueError:
            return Path(p).name


# ── records and files ────────────────────────────────────────────────────────


def units(kind, root):
    """The kind's folders under root, record or not: ds/carve/part_<i>of<n>, and for kt each ds/carve."""
    root = Path(root)
    if kind == "kt":
        found = (d for d in root.glob("*/*") if d.is_dir())
    else:
        found = (d for d in root.glob("*/*/part_*of*") if d.is_dir() and PART.fullmatch(d.name))
    return sorted(found, key=lambda d: d.relative_to(root).as_posix())


def listed(kind, rec):
    """The files a record lists, each with its sha256: its arrays as .npy, and for kt its ids.json."""
    files = {f"{k}.npy": v["sha256"] for k, v in rec["arrays"].items()}
    if kind == "kt":
        files["ids.json"] = rec["ids_sha256"]
    return files


def short_parts(kind, root):
    """Carves whose part folders are not part_0 .. part_<n-1> of one n (a whole part folder gone)."""
    if kind == "kt":
        return []
    by = {}
    for d in units(kind, root):
        i, n = (int(x) for x in PART.fullmatch(d.name).groups())
        by.setdefault(d.parent.relative_to(root).as_posix(), set()).add((i, n))
    bad = []
    for carve, got in sorted(by.items()):
        ns = {n for _i, n in got}
        if len(ns) != 1 or {i for i, _n in got} != set(range(ns.pop())):
            bad.append(carve)
    return bad


def check_kind(kind, root, only=None):
    """Every file the kind's records list under root, hashed against its record (only: one ds/carve)."""
    root = Path(root)
    out = {"units": 0, "files": 0, "ok": 0, "missing": [], "differs": [], "no_record": [],
           "short": [c for c in short_parts(kind, root) if only is None or c == only]}
    for u in units(kind, root):
        r = u.relative_to(root).as_posix()
        if only is not None and r != only and not r.startswith(only + "/"):
            continue
        if not (u / "record.json").is_file():
            out["no_record"].append(r)
            continue
        out["units"] += 1
        for f, sha in listed(kind, read_json(u / "record.json")).items():
            out["files"] += 1
            p = u / f
            if not p.is_file():
                out["missing"].append(f"{r}/{f}")
            elif sha_file(p) != sha:
                out["differs"].append(f"{r}/{f}")
            else:
                out["ok"] += 1
    out["whole"] = bool(out["units"]) and not (out["missing"] or out["differs"] or out["no_record"] or out["short"])
    return out


def record_diff(orig, new):
    """The keys on which two records of one build differ, leaving out the run's own and the arrays' sha256s: (the
    others, the code shas)."""
    differ, code = [], []
    for k in sorted(set(orig) | set(new)):
        if k in RUN_KEYS:
            continue
        a, b = orig.get(k), new.get(k)
        if k == "arrays":
            a, b = ({n: {x: y for x, y in v.items() if x != "sha256"} for n, v in (d or {}).items()} for d in (a, b))
        if a != b:
            (code if k in CODE_KEYS else differ).append(k)
    return differ, code


def put(src, dst, sha):
    """src back as dst, when dst is missing and src's sha256 is sha: a hard link, else a copy renamed into place. A
    file at dst is never touched."""
    src, dst = Path(src), Path(dst)
    if dst.exists():
        return "present"
    if not src.is_file():
        return "not rebuilt"
    if sha_file(src) != sha:
        return "differs"
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
        how = "linked"
    except FileExistsError:
        return "present"
    except OSError:
        tmp = dst.with_name(f"{dst.name}.restore{os.getpid()}.tmp")
        shutil.copyfile(src, tmp)
        if dst.exists():
            tmp.unlink()
            return "present"
        os.rename(tmp, dst)                 # Windows refuses a rename over a file
        how = "copied"
    if sha_file(dst) != sha:
        raise SystemExit(f"{dst}: put back, but its sha256 is not the record's")
    return how


def base_status(root, g):
    npz, js = Path(root) / f"basis_{g}.npz", Path(root) / f"basis_{g}.json"
    if not js.is_file():
        return "no record"
    if not npz.is_file():
        return "missing"
    return "ok" if sha_file(npz) == read_json(js)["npz_sha256"] else "differs"


def folder_shas(d):
    d = Path(d)
    return {f.name: sha_file(f) for f in sorted(d.iterdir()) if f.is_file()} if d.is_dir() else {}


def code_now(kind, rec, base):
    """Each code sha the record names against today's file: 'same', 'changed' or 'no file'."""
    def one(f, want):
        p = Path(base) / f
        return "no file" if not p.is_file() else "same" if sha_src(p) == want else "changed"
    out = {k: one(f, rec[k]) for k, f in CODE[kind].items() if k in rec}
    for m, s in sorted((rec.get("module_sha256") or {}).items()):
        out[f"module_sha256.{m}"] = one(f"{MU}{m}.py", s)
    return out


# ── the commands ─────────────────────────────────────────────────────────────


def plan(R):
    rec = {"declared_in": DECLARED, "kinds": {}, "bases": {}, "utc": utc(), "script_sha256": sha_src(__file__)}
    for kind in KINDS:
        root, rb = R.of[kind], R.rb[kind]
        us = units(kind, root)
        rows, code = [], Counter()
        for u in us:
            r = u.relative_to(root).as_posix()
            if not (u / "record.json").is_file():
                rows.append({"unit": r, "missing": ["record.json"], "rebuilt": False})
                continue
            rj = read_json(u / "record.json")
            code.update(f"{c} {s}" for c, s in code_now(kind, rj, R.base).items())
            miss = sorted(f for f in listed(kind, rj) if not (u / f).exists())
            if miss:
                rows.append({"unit": r, "missing": miss, "rebuilt": (rb / r / "record.json").is_file()})
        k = rec["kinds"][kind] = {"root": R.rel(root), "units": len(us), "short": short_parts(kind, root),
                                  "with_missing": rows, "files_missing": sum(len(x["missing"]) for x in rows),
                                  "code_against_today": dict(sorted(code.items()))}
        log(f"{kind} ({k['root']}): {len(us)} {'carves' if kind == 'kt' else 'parts'}, {len(rows)} with files missing, "
            f"{k['files_missing']} files; short carves {k['short']}; records' code against today's "
            f"{k['code_against_today']}")
        by = {}
        for x in rows:
            c = x["unit"] if kind == "kt" else x["unit"].rsplit("/", 1)[0]
            by.setdefault(c, Counter()).update(x["missing"])
        for c, cnt in sorted(by.items()):
            log(f"    {c}: " + ", ".join(f"{f}" + (f" x{n}" if n > 1 else "") for f, n in sorted(cnt.items())))
    for root in (R.of["cache"], R.of["step1"], R.rb["cache"]):
        for g in BASES:
            js = root / f"basis_{g}.json"
            rec["bases"][f"{R.rel(root)}/basis_{g}"] = {
                "npz": (root / f"basis_{g}.npz").is_file(), "json": js.is_file(),
                "npz_sha256": read_json(js)["npz_sha256"] if js.is_file() else None,
                "code_against_today": code_now("basis", read_json(js), R.base) if js.is_file() else None}
    for k, v in rec["bases"].items():
        log(f"{k}: npz {'present' if v['npz'] else 'MISSING'}, record {'present' if v['json'] else 'missing'}"
            + ("" if not v["json"] else f" (npz sha256 {v['npz_sha256'][:12]}; code {v['code_against_today']})"))
    write_json(R.of["records"] / "plan.json", rec)
    return 0


def prep(R):
    cache, step1, rb = R.of["cache"], R.of["step1"], R.rb["cache"]
    rec = {"declared_in": DECLARED, "rebuild_root": R.rel(rb), "steps": {}, "utc": utc(),
           "script_sha256": sha_src(__file__)}
    st = rec["steps"]
    j2 = cache / "basis_2wiki.json"
    if not j2.is_file():
        raise SystemExit(f"prep: {R.rel(j2)} is missing")
    st["rebuild basis_2wiki.npz"] = put(cache / "basis_2wiki.npz", rb / "basis_2wiki.npz", read_json(j2)["npz_sha256"])
    st["rebuild basis_2wiki.json"] = put(j2, rb / "basis_2wiki.json", sha_file(j2))
    jh, nh = rb / "basis_hotpotqa.json", rb / "basis_hotpotqa.npz"
    if not (jh.is_file() and nh.is_file()):
        raise SystemExit(f"prep: {R.rel(nh)} and its record are not made yet (lean_cache --make-basis hotpotqa)")
    orig, new = read_json(cache / "basis_hotpotqa.json"), read_json(jh)
    got = sha_file(nh)
    differ, code = record_diff(orig, new)
    rec["basis_hotpotqa"] = {"npz_sha256": got, "record_npz_sha256": orig["npz_sha256"], "record_differs_on": differ,
                             "code_differs_on": code}
    log(f"rebuilt basis_hotpotqa: npz sha256 {got[:12]}, P1c2's record {orig['npz_sha256'][:12]}; record differs on "
        f"{differ or 'nothing'} (code {code or 'nothing'})")
    if got != orig["npz_sha256"] or got != new["npz_sha256"] or differ:
        rec["verdict"] = "BASIS_NOT_REPRODUCED"
        write_json(R.of["records"] / "prep.json", rec)
        log("prep: the rebuilt basis_hotpotqa is not P1c2's; the restore stops here and nothing went back")
        return 2
    for root in (cache, step1):
        for g in BASES:
            key = f"{R.rel(root)}/basis_{g}.npz"
            js, dst = root / f"basis_{g}.json", root / f"basis_{g}.npz"
            if not js.is_file():
                st[key] = "no record"
                continue
            want = read_json(js)["npz_sha256"]
            if dst.exists():
                st[key] = "present"
                continue
            st[key] = "no source"
            for src in (cache / f"basis_{g}.npz", step1 / f"basis_{g}.npz", rb / f"basis_{g}.npz"):
                if src != dst and src.is_file() and sha_file(src) == want:
                    st[key] = put(src, dst, want)
                    break
    rec["check"] = {f"{R.rel(root)}/basis_{g}": base_status(root, g) for root in (cache, rb, step1) for g in BASES}
    gate = all(rec["check"][f"{R.rel(root)}/basis_{g}"] == "ok" for root in (cache, rb) for g in BASES)
    rec["verdict"] = "BASES_READY" if gate else "BASES_NOT_READY"
    write_json(R.of["records"] / "prep.json", rec)
    for k, v in st.items():
        log(f"{k}: {v}")
    log(f"prep: {rec['verdict']} ({', '.join(f'{k} {v}' for k, v in rec['check'].items())})")
    return 0 if gate else 1


def install(kind, R):
    root, rb = R.of[kind], R.rb[kind]
    rows = []
    for u in units(kind, root):
        r = u.relative_to(root).as_posix()
        if not (u / "record.json").is_file():
            rows.append({"unit": r, "error": "no record"})
            continue
        rec = read_json(u / "record.json")
        files = listed(kind, rec)
        missing = sorted(f for f in files if not (u / f).exists())
        b = rb / r
        row = {"unit": r, "listed": len(files), "missing": missing, "rebuilt": (b / "record.json").is_file()}
        differ = None
        if row["rebuilt"]:
            new = read_json(b / "record.json")
            differ, code = record_diff(rec, new)
            nf = listed(kind, new)
            kept = [f for f in files if f not in missing]
            row.update(record_differs_on=differ, code_differs_on=code,
                       reproduced=sum(nf.get(f) == files[f] for f in kept),
                       not_reproduced=sorted(f for f in kept if nf.get(f) != files[f]))
        row["put"] = {f: ("not rebuilt" if differ is None else "record differs" if differ else put(b / f, u / f, files[f]))
                      for f in missing}
        rows.append(row)
        if missing or row["rebuilt"]:
            got = Counter(row["put"].values())
            log(f"{kind} {r}: {len(missing)} missing ({', '.join(f'{n} {k}' for k, n in sorted(got.items())) or '-'})"
                + ("" if not row["rebuilt"] else f"; present reproduced {row['reproduced']} of "
                   f"{row['reproduced'] + len(row['not_reproduced'])}; record differs on {differ or 'nothing'}"))
    check = check_kind(kind, root)
    out = {"declared_in": DECLARED, "kind": kind, "root": R.rel(root), "rebuild_root": R.rel(rb), "units": rows,
           "put": dict(Counter(v for x in rows for v in x.get("put", {}).values())),
           "not_reproduced": sum(len(x.get("not_reproduced", [])) for x in rows),
           "check": {k: v for k, v in check.items()}, "whole": check["whole"], "utc": utc(),
           "script_sha256": sha_src(__file__)}
    write_json(R.of["records"] / f"install_{kind}.json", out)
    log(f"install {kind}: put {out['put'] or '{}'}; present arrays not reproduced {out['not_reproduced']}; check: "
        f"{check['ok']} of {check['files']} files as their records, missing {len(check['missing'])}, differs "
        f"{len(check['differs'])}, no record {len(check['no_record'])}, short {check['short']} -> "
        f"{'WHOLE' if check['whole'] else 'NOT WHOLE'}")
    return 0 if check["whole"] else 1


def verify(R):
    rec = {"declared_in": DECLARED, "kinds": {}, "bases": {}, "utc": utc(), "script_sha256": sha_src(__file__)}
    for kind in KINDS:
        c = rec["kinds"][kind] = check_kind(kind, R.of[kind])
        log(f"{kind} ({R.rel(R.of[kind])}): {c['units']} units, {c['ok']} of {c['files']} files as their records; missing "
            f"{len(c['missing'])}, differs {len(c['differs'])}, no record {len(c['no_record'])}, short {c['short']}")
    for root in (R.of["cache"], R.of["step1"]):
        for g in BASES:
            rec["bases"][f"{R.rel(root)}/basis_{g}"] = base_status(root, g)
    gate = all(c["whole"] for c in rec["kinds"].values()) and \
        all(rec["bases"][f"{R.rel(R.of['cache'])}/basis_{g}"] == "ok" for g in BASES)
    rec["verdict"] = "WHOLE" if gate else "NOT_WHOLE"
    write_json(R.of["records"] / "verify.json", rec)
    log(f"bases: {rec['bases']}")
    log(f"verify: {rec['verdict']}")
    return 0 if gate else 1


def build_cmd(what, ds, carve, host, R, z2=None):
    """zp1c2.build's call with the rebuild root for out_root."""
    if z2 is None:
        here = str(ROOT / "outputs" / "mp_unified")
        if here not in sys.path:
            sys.path.insert(0, here)
        import zp1c2 as z2
    for name, mine in (("CACHE", R.of["cache"]), ("LOOK", R.of["look"]), ("CH_RM", R.of["rmatch"]),
                       ("CH_ZRC", R.of["zrc"]), ("LINKS", R.of["links"])):
        if Path(getattr(z2, name)).resolve() != Path(mine).resolve():
            raise SystemExit(f"build: zp1c2.{name} is not {R.rel(mine)}")
    for kind in ("cache", "rmatch") if what == "zrc" else ("cache",):
        c = check_kind(kind, R.of[kind], only=f"{ds}/{carve}")
        if not c["whole"]:
            raise SystemExit(f"build {what} {ds}/{carve}: its {kind} is not whole yet (missing {c['missing'][:4]}, "
                             f"differs {c['differs'][:4]}, units {c['units']})")
    placement = {"where": "host" if host else "laptop"}
    out = R.rb[what]
    log(f"build {what} {ds}/{carve} into {R.rel(out)} (zp1c2.build's call; cache {R.rel(z2.CACHE)}, look {R.rel(z2.LOOK)})")
    if what == "rmatch":
        z2.RM.build(ds, carve, out, cache_root=z2.CACHE, look_root=z2.LOOK, placement=placement)
    elif what == "zrc":
        z2.ZC.build(ds, carve, out, cache_root=z2.CACHE, look_root=z2.LOOK, placement=placement, rm_root=z2.CH_RM)
    elif what == "links":
        z2.ZL.build(ds, carve, out, cache_root=z2.CACHE, look_root=z2.LOOK, placement=placement)
    else:
        raise SystemExit(f"build: --what rmatch, zrc or links, not {what}")
    c = check_kind(what, out, only=f"{ds}/{carve}")
    if not c["whole"]:
        raise SystemExit(f"build {what} {ds}/{carve}: the rebuilt parts are not their own records' ({c})")
    return 0


def kt_cmd(ds, carve, threads, host, R, pk=None, mirror=None):
    """p1c3_kt.feats_stage with its KT root set to the rebuild root; P1c3's own carve must not change."""
    if pk is None:
        import p1c3_kt as pk
    if Path(pk.KT).resolve() != R.of["kt"].resolve():
        raise SystemExit(f"kt: p1c3_kt.KT is not {R.rel(R.of['kt'])}")
    c = check_kind("cache", R.of["cache"], only=f"{ds}/{carve}")
    if not c["whole"]:
        raise SystemExit(f"kt {ds}/{carve}: its cache is not whole yet (missing {c['missing'][:4]}, differs "
                         f"{c['differs'][:4]}, units {c['units']})")
    mine = R.of["kt"] / ds / carve
    before = folder_shas(mine)
    if host:
        if mirror is None:
            import p1e_anchor_kind as K       # P1e's host mode, as p1c3_kt's main runs it
            mirror = K.G.on_the_mirror
        mirror()
    saved = pk.KT
    pk.KT = R.rb["kt"]
    log(f"kt {ds}/{carve} into {R.rel(R.rb['kt'])} (p1c3_kt.feats_stage, threads {threads})")
    try:
        rc = pk.feats_stage(ds, threads, 0, carve)
    finally:
        pk.KT = saved
    if folder_shas(mine) != before:
        raise SystemExit(f"kt: {R.rel(mine)} changed during the rebuild")
    c = check_kind("kt", R.rb["kt"], only=f"{ds}/{carve}")
    if not c["whole"]:
        raise SystemExit(f"kt {ds}/{carve}: the rebuilt carve is not its own record's ({c})")
    return rc or 0


# ── selftest ─────────────────────────────────────────────────────────────────


def selftest():
    t0 = time.time()

    def sha(b):
        return hashlib.sha256(b).hexdigest()

    def part(root, rel_, arrays, **extra):
        d = Path(root) / rel_
        d.mkdir(parents=True, exist_ok=True)
        for k, b in arrays.items():
            (d / f"{k}.npy").write_bytes(b)
        rec = {"rows": 5, "arrays": {k: {"dtype": "|u1", "shape": [len(b)], "sha256": sha(b)} for k, b in arrays.items()},
               "seconds": 1.0, "utc": "t0", "placement": {"where": "laptop"}, "script_sha256": "a"}
        rec.update(extra)
        write_json(d / "record.json", rec)
        return d

    def kt_carve(root, rel_, blobs, **extra):
        d = Path(root) / rel_
        d.mkdir(parents=True, exist_ok=True)
        for k, b in blobs.items():
            (d / f"{k}.npy").write_bytes(b)
        (d / "ids.json").write_text('["q1", "q2"]', encoding="utf-8")
        rec = {"threads": 3, "arrays": {k: {"sha256": sha(b), "shape": [len(b)]} for k, b in blobs.items()},
               "ids_sha256": sha_file(d / "ids.json"), "seconds": 1.0, "utc": "t0", "script_sha256": "a"}
        rec.update(extra)
        write_json(d / "record.json", rec)
        return d

    def basis(root, g, blob, npz=True, **extra):
        Path(root).mkdir(parents=True, exist_ok=True)
        if npz:
            (Path(root) / f"basis_{g}.npz").write_bytes(blob)
        rec = {"graph": g, "rows": 10, "explained": 0.5, "npz": f"basis_{g}.npz", "npz_sha256": sha(blob),
               "blas_threads": "2", "seconds": 1.0, "utc": "t0", "placement": {"where": "host"}, "script_sha256": "s"}
        rec.update(extra)
        write_json(Path(root) / f"basis_{g}.json", rec)

    def refuses(f, *a, **k):
        try:
            f(*a, **k)
        except SystemExit:
            return True
        return False

    with tempfile.TemporaryDirectory() as td:
        R = Roots(td)
        C, RB = R.of["cache"], R.rb["cache"]
        # put: never over a file; nothing from a missing or wrong source
        a, b, c = Path(td) / "a.bin", Path(td) / "b.bin", Path(td) / "c.bin"
        a.write_bytes(b"NEW")
        b.write_bytes(b"OLD")
        assert put(a, b, sha(b"NEW")) == "present" and b.read_bytes() == b"OLD"
        assert put(Path(td) / "none.bin", c, sha(b"NEW")) == "not rebuilt" and not c.exists()
        assert put(a, c, sha(b"OTHER")) == "differs" and not c.exists()
        assert put(a, c, sha(b"NEW")) in ("linked", "copied") and c.read_bytes() == b"NEW"
        # record_diff: the run's keys and the arrays' shas leave out; code shas apart; a shape counts
        o = {"rows": 1, "seconds": 1, "utc": "a", "script_sha256": "x", "arrays": {"n": {"shape": [2], "sha256": "p"}}}
        assert record_diff(o, {**o, "seconds": 2, "utc": "b", "arrays": {"n": {"shape": [2], "sha256": "q"}}}) == ([], [])
        assert record_diff(o, {**o, "script_sha256": "y"}) == ([], ["script_sha256"])
        assert record_diff(o, {**o, "rows": 2, "arrays": {"n": {"shape": [3], "sha256": "p"}}}) == (["arrays", "rows"], [])
        # the bases: a rebuilt basis that is not P1c2's stops the restore before anything goes back
        basis(C, "2wiki", b"W")
        basis(C, "hotpotqa", b"H", npz=False)
        basis(R.of["step1"], "2wiki", b"W", npz=False)
        basis(R.of["step1"], "hotpotqa", b"H", npz=False)
        assert refuses(prep, R)                                         # not made yet
        basis(RB, "hotpotqa", b"H2")
        assert prep(R) == 2 and not (C / "basis_hotpotqa.npz").exists()
        assert not (R.of["step1"] / "basis_2wiki.npz").exists()
        assert read_json(R.of["records"] / "prep.json")["verdict"] == "BASIS_NOT_REPRODUCED"
        basis(RB, "hotpotqa", b"H", explained=0.6)
        assert prep(R) == 2 and not (C / "basis_hotpotqa.npz").exists()
        basis(RB, "hotpotqa", b"H", seconds=7.0, utc="t1", placement={"where": "host2"}, script_sha256="z")
        assert prep(R) == 0
        assert (C / "basis_hotpotqa.npz").read_bytes() == b"H" and (RB / "basis_2wiki.npz").read_bytes() == b"W"
        assert (R.of["step1"] / "basis_2wiki.npz").read_bytes() == b"W"
        assert (R.of["step1"] / "basis_hotpotqa.npz").read_bytes() == b"H"
        pr = read_json(R.of["records"] / "prep.json")
        assert pr["verdict"] == "BASES_READY" and pr["basis_hotpotqa"]["code_differs_on"] == ["script_sha256"]
        assert prep(R) == 0                                             # again: every basis present
        # the cache: four parts lose files; their rebuilds are byte for byte, one differs, one's record differs, one
        # never ran
        code = Path(td) / MU / "lean_cache.py"
        code.parent.mkdir(parents=True, exist_ok=True)
        code.write_bytes(b"code\r\n")
        p0 = part(C, "metaqa/fit/part_0of2", {"xc": b"x0", "node": b"n0"}, script_sha256=sha(b"code\n"),
                  module_sha256={"lean_mlp": "m"})
        (C / "metaqa" / "fit" / "part_0of2_old").mkdir()                # not a part: never a unit
        p1 = part(C, "metaqa/fit/part_1of2", {"xc": b"x1", "node": b"n1"})
        p2 = part(C, "musique/s1eval/part_0of1", {"xc": b"m0", "node": b"mn"})
        p3 = part(C, "webqsp/s1eval/part_0of1", {"xc": b"w0", "node": b"wn"})
        for f in (p0 / "node.npy", p1 / "xc.npy", p1 / "node.npy", p2 / "node.npy", p3 / "node.npy"):
            f.unlink()
        part(RB, "metaqa/fit/part_0of2", {"xc": b"x0", "node": b"n0"}, seconds=9.0, utc="t1",
             placement={"where": "host"}, script_sha256="b")
        part(RB, "metaqa/fit/part_1of2", {"xc": b"x1", "node": b"nX"})
        part(RB, "musique/s1eval/part_0of1", {"xc": b"m0", "node": b"mn"}, rows=6)
        assert plan(R) == 0
        pl = read_json(R.of["records"] / "plan.json")
        assert pl["kinds"]["cache"]["units"] == 4 and pl["kinds"]["cache"]["files_missing"] == 5
        assert pl["kinds"]["cache"]["code_against_today"] == {"module_sha256.lean_mlp no file": 1,
                                                              "script_sha256 changed": 3, "script_sha256 same": 1}
        assert pl["bases"][f"{R.rel(C)}/basis_hotpotqa"]["npz"]
        assert pl["bases"][f"{R.rel(C)}/basis_hotpotqa"]["code_against_today"] == {"script_sha256": "changed"}
        assert install("cache", R) == 1
        rows = {x["unit"]: x for x in read_json(R.of["records"] / "install_cache.json")["units"]}
        r0, r1 = rows["metaqa/fit/part_0of2"], rows["metaqa/fit/part_1of2"]
        assert r0["put"]["node.npy"] in ("linked", "copied")
        assert r0["code_differs_on"] == ["module_sha256", "script_sha256"]
        assert r0["record_differs_on"] == [] and r0["reproduced"] == 1 and r0["not_reproduced"] == []
        assert r1["put"]["node.npy"] == "differs" and r1["put"]["xc.npy"] in ("linked", "copied")
        assert rows["musique/s1eval/part_0of1"]["put"] == {"node.npy": "record differs"}
        assert rows["musique/s1eval/part_0of1"]["record_differs_on"] == ["rows"]
        assert rows["webqsp/s1eval/part_0of1"]["put"] == {"node.npy": "not rebuilt"}
        assert (p0 / "node.npy").read_bytes() == b"n0" and (p1 / "xc.npy").read_bytes() == b"x1"
        assert not (p1 / "node.npy").exists() and not (p2 / "node.npy").exists() and not (p3 / "node.npy").exists()
        # a build or a KT run refuses a carve whose cache is not whole, before it calls anything
        calls = []

        class Fake:
            def __init__(self, what):
                self.what = what

            def build(self, ds, carve, out_root, **kw):
                calls.append((self.what, ds, carve, Path(out_root), kw))
                part(out_root, f"{ds}/{carve}/part_0of1", {"ent_row": b"e" + ds.encode()})

        z2 = types.SimpleNamespace(CACHE=C, LOOK=R.of["look"], CH_RM=R.of["rmatch"], CH_ZRC=R.of["zrc"],
                                   LINKS=R.of["links"], RM=Fake("rmatch"), ZC=Fake("zrc"), ZL=Fake("links"))
        assert refuses(build_cmd, "links", "webqsp", "s1eval", True, R, z2) and not calls
        # the fixed rebuilds go back; a second install changes nothing
        part(RB, "metaqa/fit/part_1of2", {"xc": b"x1", "node": b"n1"})
        part(RB, "musique/s1eval/part_0of1", {"xc": b"m0", "node": b"mn"})
        part(RB, "webqsp/s1eval/part_0of1", {"xc": b"w0", "node": b"wn"})
        assert install("cache", R) == 0
        assert (p1 / "node.npy").read_bytes() == b"n1" and (p3 / "node.npy").read_bytes() == b"wn"
        assert install("cache", R) == 0
        assert read_json(R.of["records"] / "install_cache.json")["put"] == {}
        # build: zp1c2's call with the rebuild root; zrc waits for rmatch's chains; another root is refused
        assert build_cmd("rmatch", "webqsp", "s1eval", True, R, z2) == 0
        assert calls[-1] == ("rmatch", "webqsp", "s1eval", R.rb["rmatch"],
                             {"cache_root": C, "look_root": R.of["look"], "placement": {"where": "host"}})
        rm = part(R.of["rmatch"], "webqsp/s1eval/part_0of1", {"ent_row": b"ewebqsp", "ent_z": b"z"})
        (rm / "ent_row.npy").unlink()
        assert refuses(build_cmd, "zrc", "webqsp", "s1eval", True, R, z2) and calls[-1][0] == "rmatch"
        assert install("rmatch", R) == 1                     # ent_z was never rebuilt by the toy builder: refused
        assert not (rm / "ent_row.npy").exists()
        part(R.rb["rmatch"], "webqsp/s1eval/part_0of1", {"ent_row": b"ewebqsp", "ent_z": b"z"})
        assert install("rmatch", R) == 0 and (rm / "ent_row.npy").read_bytes() == b"ewebqsp"
        assert build_cmd("zrc", "webqsp", "s1eval", False, R, z2) == 0
        assert calls[-1] == ("zrc", "webqsp", "s1eval", R.rb["zrc"],
                             {"cache_root": C, "look_root": R.of["look"], "placement": {"where": "laptop"},
                              "rm_root": R.of["rmatch"]})
        assert refuses(build_cmd, "links", "metaqa", "fit", True, R,
                       types.SimpleNamespace(**{**vars(z2), "CACHE": R.rb["cache"]}))
        assert build_cmd("links", "metaqa", "fit", True, R, z2) == 0 and calls[-1][3] == R.rb["links"]
        # kt: the rebuild root while it runs, the original root after; P1c3's own carve never written
        kd = kt_carve(R.of["kt"], "metaqa/fit", {"kt": b"K", "n": b"N"})
        (kd / "kt.npy").unlink()
        mirrored = []

        def feats(ds, threads, limit, carve):
            kt_carve(fake.KT, f"{ds}/{carve}", {"kt": b"K", "n": b"N"}, seconds=5.0, utc="t2")
            return 0

        fake = types.SimpleNamespace(KT=R.of["kt"], feats_stage=feats)
        assert kt_cmd("metaqa", "fit", 3, True, R, fake, lambda: mirrored.append(1)) == 0 and mirrored == [1]
        assert fake.KT == R.of["kt"] and not (kd / "kt.npy").exists()
        assert (R.rb["kt"] / "metaqa" / "fit" / "kt.npy").read_bytes() == b"K"

        def rogue(ds, threads, limit, carve):
            kt_carve(R.of["kt"], f"{ds}/{carve}", {"kt": b"K", "n": b"N"})
            return 0

        assert refuses(kt_cmd, "metaqa", "fit", 3, False, R, types.SimpleNamespace(KT=R.of["kt"], feats_stage=rogue))
        (kd / "kt.npy").unlink()                                # the rogue's write, undone for the rest
        assert refuses(kt_cmd, "musique", "s1fit", 2, False, R, fake)     # no cache for the carve
        assert install("kt", R) == 0 and (kd / "kt.npy").read_bytes() == b"K"
        # verify: every kind whole and P1c2's bases -> WHOLE; one changed byte -> NOT_WHOLE
        zr = part(R.of["zrc"], "webqsp/s1eval/part_0of1", {"ent_row": b"r"})
        lk = part(R.of["links"], "musique/s1fit/part_0of2", {"e_u": b"u"})
        assert verify(R) == 1                                   # links' carve is one part short
        assert read_json(R.of["records"] / "verify.json")["kinds"]["links"]["short"] == ["musique/s1fit"]
        part(R.of["links"], "musique/s1fit/part_1of2", {"e_u": b"u2"})
        assert verify(R) == 0 and read_json(R.of["records"] / "verify.json")["verdict"] == "WHOLE"
        (zr / "ent_row.npy").write_bytes(b"R")
        assert verify(R) == 1 and read_json(R.of["records"] / "verify.json")["kinds"]["zrc"]["differs"] == \
            ["webqsp/s1eval/part_0of1/ent_row.npy"]
        (zr / "ent_row.npy").write_bytes(b"r")
        (C / "basis_2wiki.npz").unlink()
        assert verify(R) == 1
        assert lk.is_dir()
        # no record names the temporary folder
        for f in R.of["records"].glob("*.json"):
            assert td not in f.read_text(encoding="utf-8") and Path(td).as_posix() not in f.read_text(encoding="utf-8")
    log(f"selftest: PASS ({time.time() - t0:.1f}s)")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="P1c2's builds back after the host's clean-up (see the docstring)")
    ap.add_argument("--selftest", action="store_true")
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("plan")
    sub.add_parser("prep")
    sub.add_parser("verify")
    ip = sub.add_parser("install")
    ip.add_argument("--what", required=True, choices=KINDS)
    bp = sub.add_parser("build")
    bp.add_argument("--what", required=True, choices=("rmatch", "zrc", "links"))
    bp.add_argument("--dataset", required=True)
    bp.add_argument("--carve", required=True)
    bp.add_argument("--host", action="store_true")
    kp = sub.add_parser("kt")
    kp.add_argument("dataset")
    kp.add_argument("--carve", required=True, choices=("fit", "s1fit", "s1eval"))
    kp.add_argument("--threads", type=int, default=2)
    kp.add_argument("--host", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    R = Roots()
    if a.cmd == "plan":
        return plan(R)
    if a.cmd == "prep":
        return prep(R)
    if a.cmd == "install":
        return install(a.what, R)
    if a.cmd == "verify":
        return verify(R)
    if a.cmd == "build":
        return build_cmd(a.what, a.dataset, a.carve, a.host, R)
    if a.cmd == "kt":
        return kt_cmd(a.dataset, a.carve, a.threads, a.host, R)
    ap.error("plan, prep, build, kt, install, verify, or --selftest")


if __name__ == "__main__":
    sys.exit(main())
