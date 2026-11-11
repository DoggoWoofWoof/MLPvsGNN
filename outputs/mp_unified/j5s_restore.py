"""J5S amendment A1 (docs/J5S_J5_SEEDS.md, section 10), declared before anything is rebuilt and before any three-seed
number. The look chunks and cache arrays that J5S's fits and reads load were removed by the host's auto-cleanup, every
record kept. They are built again by their own committed builders, unchanged, and a file goes back only where the
rebuilt file is the one its record lists. Systems code; no science. Nothing J5S declared changes: its fits and reads
run as declared, under new row names, once `verify` passes.

    python outputs/mp_unified/j5s_restore.py backup
    (each removed look: its original command, unchanged, run in place; LOOKS)
    python outputs/mp_unified/j5s_restore.py look-check --carve musique/s1fit
    python outputs/mp_unified/j5s_restore.py cache --carve metaqa/fit [--part 3] --host
    python outputs/mp_unified/j5s_restore.py chains --kind rmatch|zrc|links --carve metaqa/fit --host
    python outputs/mp_unified/j5s_restore.py verify
    python outputs/mp_unified/j5s_restore.py rows                 # the queue block (names j5r-*)
    python outputs/mp_unified/j5s_restore.py --selftest

backup      first, before anything runs: a copy of each rebuilt look's top-level json files (record*.json, b1*.json,
            ids.json, info.json) and of every part's record.json in the four caches, for every carve J5S loads, to
            outputs/j5s/restore/backup/<path>, their sha256s in backup.json. A backup that exists must hold the same
            bytes; nothing is overwritten.
looks       look_x_six.py, look_step1.py, look_b1.py and look_b1u.py, each with its original command and settings
            (LOOKS: the laptop's rx job list and the queue's rows). A look skips every chunk whose npz exists and writes
            the missing ones, and rewrites its shard's record (look_b1 its b1 record too; shard 0 also ids.json and
            info.json). look_b1 refuses --out, so every look runs in place.
look-check  after all of a carve's look rows: each rewritten json against its backup. ids.json and info.json byte for
            byte; each record on every key at any depth but the run's own (seconds, utc, peak_rss_bytes, placement);
            a code sha that differs (script_sha256, copied_from_sha256, b1a_script_sha256, look_b1u_sha256) is reported,
            not refused. Every chunk the records list must be on disk. Then the backup's bytes go back over each
            rewritten json, so every record that pins a look record's sha256 holds as filed. Any other difference
            refuses (exit 2) and nothing is put back. The look records list no chunk sha256s: a chunk is checked
            through the arrays built from it (cache, chains), each against its record.
cache       lean_cache.py run as a script (two BLAS threads, as filed), for each part of the carve that misses an
            array (with --part i: that part alone, so a carve's parts run side by side), with the part's own split
            (--part i/n), its bases and the default look root and check rows, into outputs/j5s/rebuild/step1 (step 1's
            bases linked there first, each its record's). Then install. A carve whose look is rebuilt waits for its
            look-check. The carve's own row, run after its part rows, builds any part still missing and gives the
            carve's verdict.
chains      rmatch.build, zrc.build or zlink.build: the call each CLI makes for `build --host`, unchanged but for
            out_root (the rebuild root). The builder is imported first, as its CLI loads it, so numpy starts with the
            row's threads (cpu 1, as the original rows). The carve's look must be checked and its step-1 cache whole
            first, and for zrc rmatch's chains too (zrc reads rmatch's record of each part). Then install.
install     per part of the carve (N2's restore, scripts/p1c2_cache_restore.py, its helpers imported, not edited): the
            rebuilt record must be the original's on every key but the run's own (seconds, utc, placement, peak memory,
            row-gather time) and the arrays' sha256s; a code sha that differs is reported, not refused. Each missing
            array goes back where the rebuilt file's sha256 is the record's (a hard link, else a copy renamed into
            place). A file that exists is never touched. Then the carve (or the part) is hashed against its records:
            outputs/j5s/restore/install_<kind>_<dataset>_<carve>[_p<i>].json; exit 1 unless it is whole.
verify      every array of every carve J5S loads, in the four caches, against its record, step 1's bases against theirs,
            and every look's check: outputs/j5s/restore/verify.json, WHOLE or NOT_WHOLE (exit 1 unless WHOLE). J5S's
            rows wait on it.
rows        the restore's queue block, one line per row (backup; 24 look shards; 8 look-checks; the caches, a part per
            row where a carve has parts; rmatch and zrc of metaqa/fit; 8 link carves; verify), each row's resources
            from its originals' rows and records.
"""
import argparse
import importlib
import inspect
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (ROOT / "scripts", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import p1c2_cache_restore as P  # noqa: E402  (N2's restore: stdlib only at import; its helpers, not edited)

DECLARED = "docs/J5S_J5_SEEDS.md"
TRAIN = ("metaqa/fit", "squad/fit", "musique/s1fit", "hotpotqa/fit", "2wiki/fit")
READ = tuple(f"{ds}/{cv}" for ds in ("musique", "2wiki", "hotpotqa") for cv in ("b1", "b1w", "b1t", "b1u"))
# Literal paths: the auto-cleanup pins the folders that a queued job's source names.
PATHS = {"look": "outputs/mp_unified/look", "cache": "outputs/step1/cache", "rmatch": "outputs/rmatch/cache",
         "zrc": "outputs/zrc/cache", "links": "outputs/zlink/cache", "records": "outputs/j5s/restore",
         "backup": "outputs/j5s/restore/backup"}
REBUILD = {"cache": "outputs/j5s/rebuild/step1", "rmatch": "outputs/j5s/rebuild/rmatch",
           "zrc": "outputs/j5s/rebuild/zrc", "links": "outputs/j5s/rebuild/zlink"}
KINDS = ("cache", "rmatch", "zrc", "links")
BUILDER = {"rmatch": "rmatch", "zrc": "zrc", "links": "zlink"}
# What J5S's fits (zrct.py: step 1 + zrc; zprop.py: step 1 + rmatch + zlink) and reads load.
NEEDED = {"cache": TRAIN + READ, "rmatch": ("metaqa/fit",), "zrc": ("metaqa/fit",), "links": TRAIN + READ}
# The carves the host's count (section 9) found missing arrays in, per kind.
MISSING = {"cache": TRAIN + tuple(f"{ds}/{cv}" for ds in ("musique", "2wiki", "hotpotqa") for cv in ("b1", "b1t", "b1u")),
           "rmatch": ("metaqa/fit",), "zrc": ("metaqa/fit",),
           "links": TRAIN + ("musique/b1", "musique/b1t", "musique/b1u")}
PARTS = {"metaqa/fit": 8, "musique/s1fit": 6}     # the step-1 carves with parts (the others have one); rows only
# The looks whose chunks were removed: each shard's original command (look_x_six's fit looks from the laptop's rx job
# list, 2-3 Oct; the others from the queue's rows lk1d-musique-s1fit-*, b1-look2-musique-*, u1a-look2-{t,u}-musique-*).
_X6 = "python outputs/mp_unified/look_x_six.py --dataset {d} --carve fit --host --shard {i}/{n} --full"
_S1 = "python outputs/mp_unified/look_step1.py --dataset musique --carve s1fit --host --full --shard {i}/6"
_B1 = "python outputs/mp_unified/look_b1.py --dataset musique --host --full --shard {i}/2"
_BU = "python outputs/mp_unified/look_b1u.py --variant {v} --dataset musique --host --full --shard {i}/2"
LOOKS = {
    "musique/s1fit": [_S1.format(i=i) for i in range(6)],
    "metaqa/fit": [_X6.format(d="metaqa", i=i, n=6) for i in range(6)],
    "musique/b1": [_B1.format(i=i) for i in range(2)],
    "musique/b1u": [_BU.format(v="u", i=i) for i in range(2)],
    "hotpotqa/fit": [_X6.format(d="hotpotqa", i=i, n=2) for i in range(2)],
    "2wiki/fit": [_X6.format(d="2wiki", i=i, n=2) for i in range(2)],
    "musique/b1t": [_BU.format(v="t", i=i) for i in range(2)],
    "squad/fit": [_X6.format(d="squad", i=i, n=2) for i in range(2)],
}
LOOK_RUN = ("seconds", "utc", "peak_rss_bytes", "placement")
LOOK_CODE = ("script_sha256", "copied_from_sha256", "b1a_script_sha256", "look_b1u_sha256")
_U = "outputs/universal_v2/six/"
_X6_IN = (["outputs/mp_unified/look_x_six.py", "outputs/step1/carves.json"]
          + [f"{_U}fits/u_gnn_v2_ef__H128__six__s{k}.pt" for k in range(3)]
          + [f"{_U}eval/{d}{s}" for d in ("metaqa", "2wiki", "squad", "hotpotqa", "musique", "webqsp")
             for s in (".npz", "_query_ids.json")])
_B1_IN = ["outputs/mp_unified/look_b1.py", "outputs/mp_unified/look_step1.py", "outputs/step1/carves.json",
          "outputs/mp_unified/look_x_six.py"]
_BENCH = ("build.json", "first_stage.npz", "graph_structural.npz", "graph_ner.npz", "graph_knn.npz", "nodes.json",
          "queries.json")
_LISTS = ["outputs/bench/hipporag2/musique/b1_lists.npz", "outputs/bench/hipporag2/musique/b1_lists.json"]
# Each look's rows: cpu as the original rows; memory the records' peak (6.17-8.10 GB) with about 0.5 GB.
LOOK_ROWS = {
    "musique/s1fit": (1.4, 8.6, ["outputs/mp_unified/look_step1.py", "outputs/step1/carves.json",
                                 "outputs/mp_unified/look_x_six.py"]),
    "metaqa/fit": (1.4, 6.8, _X6_IN), "hotpotqa/fit": (1.4, 8.0, _X6_IN), "2wiki/fit": (1.4, 8.0, _X6_IN),
    "squad/fit": (1.4, 6.7, _X6_IN),
    "musique/b1": (1.4, 7.6, _B1_IN + [f"outputs/bench/hipporag2/musique/{f}" for f in _BENCH] + _LISTS),
    "musique/b1t": (1.4, 6.8, ["outputs/mp_unified/look_b1u.py"] + _B1_IN
                    + [f"outputs/bench/hipporag2_t/musique/{f}" for f in _BENCH] + _LISTS),
    "musique/b1u": (1.4, 7.6, ["outputs/mp_unified/look_b1u.py"] + _B1_IN
                    + [f"outputs/bench/hipporag2_u/musique/{f}" for f in _BENCH] + _LISTS),
}
# lean_cache's peak per carve (the records: 0.72-4.28 GB) with about 0.5 GB; the chains as their original rows.
CACHE_MEM = {"metaqa/fit": 4.2, "squad/fit": 1.6, "musique/s1fit": 4.6, "hotpotqa/fit": 4.8, "2wiki/fit": 4.6,
             "musique/b1": 4.0, "musique/b1t": 3.2, "musique/b1u": 4.0}
CHAIN_ROWS = {"rmatch": (1, 4), "zrc": (1, 2), "links": (1, 4)}
MU = "outputs/mp_unified/"
_LEAN = [MU + f for f in ("lean_cache.py", "lean_host.py", "lean_mlp.py", "lean_mlp2.py", "lean_mlp3.py")]
_REL = [MU + "relz.py", MU + "relcols.py"]
_META = "outputs/m3b/relations/metaqa_rel_embeddings.npy"
CODE_IN = {"base": [MU + "j5s_restore.py", "scripts/p1c2_cache_restore.py"],
           "cache": _LEAN + [MU + "look_x_six.py", "outputs/step1/cache/basis_2wiki.npz",
                             "outputs/step1/cache/basis_2wiki.json", "outputs/step1/cache/basis_hotpotqa.npz",
                             "outputs/step1/cache/basis_hotpotqa.json"],
           "rmatch": [MU + "rmatch.py"] + _REL + _LEAN + [_META],
           "zrc": [MU + "zrc.py", MU + "chaincov.py", MU + "zrm.py", MU + "rmatch.py", MU + "zgs.py"] + _REL + _LEAN + [_META],
           "links": [MU + "zlink.py", MU + "zrc.py", MU + "chaincov.py", MU + "zrm.py", MU + "rmatch.py",
                     MU + "zgs.py"] + _REL + _LEAN}
assert set(LOOKS) == set(LOOK_ROWS) <= set(NEEDED["cache"]) and set(LOOKS) <= set(MISSING["cache"])
assert set(KINDS) == set(REBUILD) == set(NEEDED) == set(MISSING) and set(CACHE_MEM) <= set(MISSING["cache"])
assert all(set(MISSING[k]) <= set(NEEDED[k]) for k in KINDS) and len(set(TRAIN + READ)) == len(TRAIN + READ)
assert set(PARTS) <= set(MISSING["cache"])
log, utc, sha_file, read_json, write_json = P.log, P.utc, P.sha_file, P.read_json, P.write_json


class Roots:
    """Where the originals, their rebuilds and the records are: the workspace's, or the self-test's toys."""

    def __init__(self, base=ROOT):
        self.base = Path(base)
        self.of = {k: self.base / v for k, v in PATHS.items()}
        self.rb = {k: self.base / v for k, v in REBUILD.items()}

    def rel(self, p):
        return Path(p).resolve().relative_to(self.base.resolve()).as_posix()


def carve_units(kind, root, carve):
    return [u for u in P.units(kind, root) if u.relative_to(root).as_posix().startswith(carve + "/")]


def part_of(u):
    return int(P.PART.fullmatch(u.name).group(1))


def tag(carve):
    return carve.replace("/", "_")


# -- backup ------------------------------------------------------------------------------------------------------------


def backup_files(R):
    files = []
    for carve in LOOKS:
        d = R.of["look"] / carve
        fs = sorted(p for p in d.glob("*.json") if p.is_file())
        if not any(p.name.startswith("record") for p in fs):
            raise SystemExit(f"backup: {R.rel(d)} has no look record")
        files += fs
    for kind in KINDS:
        for carve in NEEDED[kind]:
            parts = sorted((R.of[kind] / carve).glob("part_*of*/record.json"))
            if not parts:
                raise SystemExit(f"backup: {R.rel(R.of[kind] / carve)} has no part record")
            files += parts
    return files


def backup(R):
    rec = {"declared_in": DECLARED, "files": {}, "utc": utc(), "script_sha256": P.sha_src(__file__)}
    got = Counter()
    for f in backup_files(R):
        rel, sha = R.rel(f), sha_file(f)
        dst = R.of["backup"] / rel
        if dst.exists():
            if sha_file(dst) != sha:
                raise SystemExit(f"backup: {rel} is not the file its backup holds; nothing overwritten")
            got["present"] += 1
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_name(dst.name + ".tmp")
            shutil.copyfile(f, tmp)
            os.replace(tmp, dst)
            if sha_file(dst) != sha:
                raise SystemExit(f"backup: the copy of {rel} is not the file")
            got["copied"] += 1
        rec["files"][rel] = sha
    rec["counts"] = dict(got)
    write_json(R.of["records"] / "backup.json", rec)
    log(f"backup: {len(rec['files'])} records ({dict(got)}) in {R.rel(R.of['backup'])}")
    return 0


# -- looks -------------------------------------------------------------------------------------------------------------


def _strip(x, path, code):
    """x without the run's keys at any depth; the code shas collected apart (path -> value)."""
    if isinstance(x, dict):
        out = {}
        for k, v in x.items():
            if k in LOOK_RUN:
                continue
            if k in LOOK_CODE:
                code[path + k] = v
                continue
            out[k] = _strip(v, f"{path}{k}.", code)
        return out
    if isinstance(x, list):
        return [_strip(v, path, code) for v in x]
    return x


def look_diff(old, new):
    """(keys that differ, code shas that differ) between two records of one look shard."""
    co, cn = {}, {}
    a, b = _strip(old, "", co), _strip(new, "", cn)
    return (sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k)),
            sorted(k for k in set(co) | set(cn) if co.get(k) != cn.get(k)))


def chunks_missing(d):
    recs = [read_json(r) for r in sorted(Path(d).glob("record*.json"))]
    if not recs:
        return ["no record"]
    n = int(recs[0]["n_chunks"])
    listed = set()
    for r in recs:
        listed |= {int(i) for i in r["chunks"]}
    on_disk = {int(p.stem[1:]) for p in (Path(d) / "chunks").glob("c*.npz") if p.stem[1:].isdigit()}
    return sorted(set(range(n)) - (listed & on_disk))


def look_check(carve, R):
    if carve not in LOOKS:
        raise SystemExit(f"look-check: {carve} is not a rebuilt look ({', '.join(LOOKS)})")
    d = R.of["look"] / carve
    pre = R.rel(d) + "/"
    files = {rel: sha for rel, sha in read_json(R.of["records"] / "backup.json")["files"].items()
             if rel.startswith(pre) and "/" not in rel[len(pre):]}
    if not files:
        raise SystemExit(f"look-check: backup.json holds nothing of {R.rel(d)}")
    rows, bad, code = {}, [], {}
    for rel, sha in sorted(files.items()):
        cur, old = R.base / rel, R.of["backup"] / rel
        if sha_file(old) != sha:
            raise SystemExit(f"look-check: the backup of {rel} is not the file backup.json names")
        if not cur.is_file():
            rows[rel] = "missing"
        elif sha_file(cur) == sha:
            rows[rel] = "unchanged"
        elif cur.name in ("ids.json", "info.json"):
            rows[rel] = "differs on its bytes"
        else:
            diff, cd = look_diff(read_json(old), read_json(cur))
            if cd:
                code[rel] = cd
            rows[rel] = "rewritten, same" if not diff else f"differs on {diff}"
        if rows[rel] not in ("unchanged", "rewritten, same"):
            bad.append(rel)
    extra = sorted(R.rel(p) for p in d.glob("*.json") if R.rel(p) not in files)
    miss = chunks_missing(d)
    rec = {"declared_in": DECLARED, "carve": carve, "files": rows, "code_differs": code, "new_files": extra,
           "chunks_missing": miss[:20], "n_chunks_missing": len(miss),
           "n_chunks": len(list((d / "chunks").glob("c*.npz"))), "utc": utc(), "script_sha256": P.sha_src(__file__)}
    out = R.of["records"] / f"look_{tag(carve)}.json"
    if bad or miss or extra:
        rec["verdict"] = "LOOK_DIFFERS"
        write_json(out, rec)
        log(f"look-check {carve}: REFUSED; differ {bad}; new files {extra}; chunks missing {len(miss)} (first "
            f"{miss[:5]}); nothing put back")
        return 2
    put = 0
    for rel, sha in sorted(files.items()):
        if rows[rel] == "unchanged":
            continue
        cur = R.base / rel
        tmp = cur.with_name(cur.name + ".j5sback.tmp")
        shutil.copyfile(R.of["backup"] / rel, tmp)
        os.replace(tmp, cur)
        if sha_file(cur) != sha:
            raise SystemExit(f"look-check: {rel} put back, but it is not its backup")
        put += 1
    rec["put_back"] = put
    rec["verdict"] = "LOOK_RESTORED"
    write_json(out, rec)
    log(f"look-check {carve}: {rec['n_chunks']} chunks on disk, every one the records list; every rewritten json is "
        f"its backup on all but the run's own keys (code differs: {code or 'nothing'}); {put} put back -> LOOK_RESTORED")
    return 0


def look_ready(carve, R):
    if carve in LOOKS:
        f = R.of["records"] / f"look_{tag(carve)}.json"
        if not f.is_file() or read_json(f).get("verdict") != "LOOK_RESTORED":
            raise SystemExit(f"{carve}: its look is rebuilt, and its look-check has not passed")
    miss = chunks_missing(R.of["look"] / carve)
    if miss:
        raise SystemExit(f"{carve}: {len(miss)} look chunks missing (first {miss[:5]})")


# -- install -----------------------------------------------------------------------------------------------------------


def check_units(kind, root, carve, units):
    """P.check_kind over just these part folders of one carve."""
    agg = {"units": 0, "files": 0, "ok": 0, "missing": [], "differs": [], "no_record": [], "short": []}
    for u in units:
        c = P.check_kind(kind, root, only=f"{carve}/{u.name}")
        for k in ("units", "files", "ok"):
            agg[k] += c[k]
        for k in ("missing", "differs", "no_record"):
            agg[k] += c[k]
    agg["whole"] = bool(agg["units"]) and not (agg["missing"] or agg["differs"] or agg["no_record"])
    return agg


def install(kind, carve, R, only=None):
    """N2's install (p1c2_cache_restore.install), scoped to one carve, or to one part of it (only: its index)."""
    root, rb = R.of[kind], R.rb[kind]
    rows = []
    units = [u for u in carve_units(kind, root, carve) if only is None or part_of(u) == only]
    if not units:
        raise SystemExit(f"install {kind} {carve}{'' if only is None else f' part {only}'}: nothing under {R.rel(root)}")
    for u in units:
        r = u.relative_to(root).as_posix()
        if not (u / "record.json").is_file():
            rows.append({"unit": r, "error": "no record"})
            continue
        rec = read_json(u / "record.json")
        files = P.listed(kind, rec)
        missing = sorted(f for f in files if not (u / f).exists())
        b = rb / r
        row = {"unit": r, "listed": len(files), "missing": missing, "rebuilt": (b / "record.json").is_file()}
        differ = None
        if row["rebuilt"]:
            new = read_json(b / "record.json")
            differ, code = P.record_diff(rec, new)
            nf = P.listed(kind, new)
            kept = [f for f in files if f not in missing]
            row.update(record_differs_on=differ, code_differs_on=code,
                       reproduced=sum(nf.get(f) == files[f] for f in kept),
                       not_reproduced=sorted(f for f in kept if nf.get(f) != files[f]))
        row["put"] = {f: ("not rebuilt" if differ is None else "record differs" if differ else P.put(b / f, u / f, files[f]))
                      for f in missing}
        rows.append(row)
        if missing or row["rebuilt"]:
            got = Counter(row["put"].values())
            log(f"{kind} {r}: {len(missing)} missing ({', '.join(f'{n} {k}' for k, n in sorted(got.items())) or '-'})"
                + ("" if not row["rebuilt"] else f"; present reproduced {row['reproduced']} of "
                   f"{row['reproduced'] + len(row['not_reproduced'])}; record differs on {differ or 'nothing'}"
                   + (f"; code differs on {row['code_differs_on']}" if row["code_differs_on"] else "")))
    check = P.check_kind(kind, root, only=carve) if only is None else check_units(kind, root, carve, units)
    out = {"declared_in": DECLARED, "kind": kind, "carve": carve, "part": only, "root": R.rel(root),
           "rebuild_root": R.rel(rb), "units": rows, "put": dict(Counter(v for x in rows for v in x.get("put", {}).values())),
           "not_reproduced": sum(len(x.get("not_reproduced", [])) for x in rows), "check": check,
           "whole": check["whole"], "utc": utc(), "script_sha256": P.sha_src(__file__)}
    write_json(R.of["records"] / f"install_{kind}_{tag(carve)}{'' if only is None else f'_p{only}'}.json", out)
    log(f"install {kind} {carve}{'' if only is None else f' part {only}'}: put {out['put'] or '{}'}; present arrays not "
        f"reproduced {out['not_reproduced']}; {check['ok']} of {check['files']} files as their records, missing "
        f"{len(check['missing'])}, differs {len(check['differs'])}, no record {len(check['no_record'])}, short "
        f"{check['short']} -> {'WHOLE' if check['whole'] else 'NOT WHOLE'}")
    return 0 if check["whole"] else 1


# -- cache and chains --------------------------------------------------------------------------------------------------


def prep_bases(R):
    """Step 1's bases in the rebuild root (lean_cache reads its bases from its out root): linked, else copied, each its
    record's."""
    for g in P.BASES:
        js, npz = R.of["cache"] / f"basis_{g}.json", R.of["cache"] / f"basis_{g}.npz"
        st = P.base_status(R.of["cache"], g)
        if st != "ok":
            raise SystemExit(f"cache: {R.rel(npz)} is not its record's ({st})")
        P.put(npz, R.rb["cache"] / npz.name, read_json(js)["npz_sha256"])
        P.put(js, R.rb["cache"] / js.name, sha_file(js))
        if P.base_status(R.rb["cache"], g) != "ok" or sha_file(R.rb["cache"] / js.name) != sha_file(js):
            raise SystemExit(f"cache: the rebuild root's basis_{g} is not step 1's")


def cache_cmd(carve, host, R, run=subprocess.run, only=None):
    if carve not in NEEDED["cache"]:
        raise SystemExit(f"cache: {carve} is not a carve J5S loads")
    ds, cv = carve.split("/")
    look_ready(carve, R)
    prep_bases(R)
    units = carve_units("cache", R.of["cache"], carve)
    if only is not None:
        units = [u for u in units if part_of(u) == only]
        if not units:
            raise SystemExit(f"cache: {carve} has no part {only}")
    for u in units:
        rec = read_json(u / "record.json")
        if all((u / f).exists() for f in P.listed("cache", rec)):
            continue
        b = R.rb["cache"] / u.relative_to(R.of["cache"])
        if (b / "record.json").is_file() and all((b / f).is_file() for f in P.listed("cache", read_json(b / "record.json"))):
            log(f"cache {carve} {u.name}: rebuilt already")
            continue
        cmd = [sys.executable, str(R.base / "outputs" / "mp_unified" / "lean_cache.py"), "--dataset", ds, "--carve", cv,
               "--part", f"{rec['part']}/{rec['nparts']}", "--bases", ",".join(rec["bases"]),
               "--out-root", str(R.rb["cache"])] + (["--host"] if host else [])
        log(f"cache {carve} {u.name}: lean_cache into {R.rel(R.rb['cache'])}")
        p = run(cmd, cwd=str(R.base))
        if p.returncode != 0:
            raise SystemExit(f"cache {carve} {u.name}: lean_cache exited {p.returncode}")
    return install("cache", carve, R, only)


def chains_cmd(kind, carve, host, R, build=None):
    if kind not in BUILDER:
        raise SystemExit(f"chains: --kind rmatch, zrc or links, not {kind}")
    if carve not in NEEDED[kind]:
        raise SystemExit(f"chains: J5S loads no {kind} of {carve}")
    ds, cv = carve.split("/")
    look_ready(carve, R)
    for k in ("cache", "rmatch") if kind == "zrc" else ("cache",):
        c = P.check_kind(k, R.of[k], only=carve)
        if not c["whole"]:
            raise SystemExit(f"chains {kind} {carve}: its {k} is not whole yet (missing {c['missing'][:4]}, differs "
                             f"{c['differs'][:4]}, units {c['units']})")
    if build is None:
        build = importlib.import_module(BUILDER[kind]).build     # first: numpy loads with the row's threads, as the CLI's
    sig = inspect.signature(build).parameters
    want = {"out_root": R.of[kind], "cache_root": R.of["cache"], "look_root": R.of["look"]}
    if kind == "zrc":
        want["rm_root"] = R.of["rmatch"]
    for k, v in want.items():
        if k not in sig or Path(sig[k].default).resolve() != Path(v).resolve():
            raise SystemExit(f"chains: {BUILDER[kind]}.build's {k} is not {R.rel(v)}")
    placement = {"where": "host" if host else "laptop"}
    log(f"chains {kind} {carve} into {R.rel(R.rb[kind])} (the CLI's build call, out_root the rebuild root)")
    build(ds, cv, out_root=R.rb[kind], placement=placement)
    c = P.check_kind(kind, R.rb[kind], only=carve)
    if not c["whole"]:
        raise SystemExit(f"chains {kind} {carve}: the rebuilt parts are not their own records' ({c})")
    return install(kind, carve, R)


def verify(R):
    rec = {"declared_in": DECLARED, "kinds": {}, "bases": {}, "utc": utc(), "script_sha256": P.sha_src(__file__)}
    whole = True
    for g in P.BASES:
        rec["bases"][g] = P.base_status(R.of["cache"], g)
        whole &= rec["bases"][g] == "ok"
    if rec["bases"] != {g: "ok" for g in P.BASES}:
        log(f"verify: step 1's bases: {rec['bases']}")
    for kind in KINDS:
        for carve in NEEDED[kind]:
            c = P.check_kind(kind, R.of[kind], only=carve)
            rec["kinds"][f"{kind} {carve}"] = {k: (v if not isinstance(v, list) else v[:20]) for k, v in c.items()}
            whole &= c["whole"]
            if not c["whole"]:
                log(f"verify {kind} {carve}: NOT WHOLE ({c['ok']} of {c['files']} as their records; missing "
                    f"{len(c['missing'])}, differs {len(c['differs'])}, no record {len(c['no_record'])}, short {c['short']})")
    for carve in LOOKS:
        f = R.of["records"] / f"look_{tag(carve)}.json"
        ok = f.is_file() and read_json(f).get("verdict") == "LOOK_RESTORED" and not chunks_missing(R.of["look"] / carve)
        rec["kinds"][f"look {carve}"] = {"whole": ok}
        whole &= ok
        if not ok:
            log(f"verify look {carve}: NOT RESTORED")
    rec["arrays"] = sum(v.get("files", 0) for v in rec["kinds"].values())
    rec["verdict"] = "WHOLE" if whole else "NOT_WHOLE"
    write_json(R.of["records"] / "verify.json", rec)
    log(f"verify: {rec['arrays']} arrays in {len(rec['kinds']) - len(LOOKS)} carves, {len(LOOKS)} looks -> {rec['verdict']}")
    return 0 if whole else 1


# -- rows --------------------------------------------------------------------------------------------------------------


def block():
    """The queue block: (deps, name, cpu, mem, inputs, command), in queue order."""
    me = "python outputs/mp_unified/j5s_restore.py"
    base = CODE_IN["base"]
    out = [((), "j5r-backup", 0.5, 0.5, base, f"{me} backup")]
    for carve, cmds in LOOKS.items():
        cpu, mem, ins = LOOK_ROWS[carve]
        for i, c in enumerate(cmds):
            out.append((("j5r-backup",), f"j5r-look-{tag(carve)}-{i}", cpu, mem, ins, c))
    for carve, cmds in LOOKS.items():
        out.append((tuple(f"j5r-look-{tag(carve)}-{i}" for i in range(len(cmds))), f"j5r-lookchk-{tag(carve)}", 0.5, 0.5,
                    base, f"{me} look-check --carve {carve}"))
    for carve in MISSING["cache"]:
        dep = f"j5r-lookchk-{tag(carve)}" if carve in LOOKS else "j5r-backup"
        n, mem, ins = PARTS.get(carve, 1), CACHE_MEM.get(carve, 1.4), base + CODE_IN["cache"]
        if n > 1:
            for i in range(n):
                out.append(((dep,), f"j5r-cache-{tag(carve)}-p{i}", 1.5, mem, ins,
                            f"{me} cache --carve {carve} --part {i} --host"))
            dep = tuple(f"j5r-cache-{tag(carve)}-p{i}" for i in range(n))
        out.append((dep if isinstance(dep, tuple) else (dep,), f"j5r-cache-{tag(carve)}", 1.5, mem, ins,
                    f"{me} cache --carve {carve} --host"))
    cpu, mem = CHAIN_ROWS["rmatch"]
    out.append((("j5r-cache-metaqa_fit",), "j5r-rm-metaqa_fit", cpu, mem, base + CODE_IN["rmatch"],
                f"{me} chains --kind rmatch --carve metaqa/fit --host"))
    cpu, mem = CHAIN_ROWS["zrc"]
    out.append((("j5r-rm-metaqa_fit",), "j5r-zrc-metaqa_fit", cpu, mem, base + CODE_IN["zrc"],
                f"{me} chains --kind zrc --carve metaqa/fit --host"))
    cpu, mem = CHAIN_ROWS["links"]
    for carve in MISSING["links"]:
        ins = base + CODE_IN["links"] + ([_META, "outputs/m3b/relations/webqsp_rel_embeddings.npy"] if carve == "metaqa/fit" else [])
        out.append(((f"j5r-cache-{tag(carve)}",), f"j5r-zl-{tag(carve)}", cpu, mem, ins,
                    f"{me} chains --kind links --carve {carve} --host"))
    last = [r[1] for r in out if r[1].startswith(("j5r-cache-", "j5r-rm-", "j5r-zrc-", "j5r-zl-")) and not re.search(r"-p\d+$", r[1])]
    out.append((tuple(last), "j5r-verify", 0.5, 1, base, f"{me} verify"))
    names = [r[1] for r in out]
    assert len(set(names)) == len(names)
    assert all(d in names for r in out for d in r[0])
    assert all(";" not in x and "|" not in x for r in out for x in r[4]) and all("|" not in r[5] for r in out)
    return out


def rows():
    for deps, name, cpu, mem, ins, cmd in block():
        print(f"line|x|{','.join(deps)}|{name}|{cpu:g}|{mem:g}|{';'.join(dict.fromkeys(ins))}|{cmd}|")
    return 0


# -- selftest ----------------------------------------------------------------------------------------------------------


def selftest():
    import hashlib

    def sha(b):
        return hashlib.sha256(b).hexdigest()

    def part(root, rel_, arrays, **extra):
        d = Path(root) / rel_
        d.mkdir(parents=True, exist_ok=True)
        for k, b in arrays.items():
            (d / f"{k}.npy").write_bytes(b)
        rec = {"arrays": {k: {"dtype": "int8", "shape": [len(b)], "sha256": sha(b)} for k, b in arrays.items()},
               "seconds": 1.0, "utc": "x", **extra}
        write_json(d / "record.json", rec)
        return rec

    def refuses(f, *a, **k):
        try:
            f(*a, **k)
        except SystemExit:
            return True
        return False

    def look_rec(s, n_sh, **kw):
        return {"look": "look_x_six", "n_chunks": 2 * n_sh, "chunks": [2 * s, 2 * s + 1], "seconds": 5, "utc": "u",
                "placement": {"a": 1}, "peak_rss_bytes": 9, "audit": {"every": 0, "seconds": 1.0},
                "threads": {"torch": 4, "blas": "2"}, "script_sha256": "s", "copied_from_sha256": "k", **kw}

    with tempfile.TemporaryDirectory() as td:
        R = Roots(td)
        # the looks: one record per shard, ids, info; the MuSiQue b1 looks a b1 record per shard (a nested seconds)
        for carve in LOOKS:
            d = R.of["look"] / carve
            (d / "chunks").mkdir(parents=True)
            for s in range(2):
                write_json(d / f"record_{s}of2.json", look_rec(s, 2))
                if carve.startswith("musique/b1"):
                    write_json(d / f"b1_{s}of2.json", {"pools": {"mean": 3, "seconds": 2.0}, "utc": "u",
                                                       "script_sha256": {"look_b1": "a"}})
            (d / "ids.json").write_text('["q1", "q2"]', encoding="utf-8")
            (d / "info.json").write_text('[{}, {}]', encoding="utf-8")
        for carve in READ:
            if carve not in LOOKS:
                d = R.of["look"] / carve
                (d / "chunks").mkdir(parents=True)
                write_json(d / "record.json", {"n_chunks": 1, "chunks": [0]})
                (d / "chunks" / "c00000.npz").write_bytes(b"z")
        # the caches: every needed carve one part (two for metaqa/fit), its arrays a and b
        for kind in KINDS:
            for carve in NEEDED[kind]:
                n = 2 if carve == "metaqa/fit" else 1
                for i in range(n):
                    part(R.of[kind], f"{carve}/part_{i}of{n}", {"a": f"{kind}{carve}{i}a".encode(),
                                                                "b": f"{kind}{carve}{i}b".encode()},
                         part=i, nparts=n, bases={"2wiki": {}, "hotpotqa": {}})
        for g in P.BASES:
            blob = f"basis {g}".encode()
            (R.of["cache"] / f"basis_{g}.npz").write_bytes(blob)
            write_json(R.of["cache"] / f"basis_{g}.json", {"npz_sha256": sha(blob)})
        # 1. verify: the looks still lack their chunks; then a whole tree; then a missing array
        assert verify(R) == 1
        for carve in LOOKS:
            for i in range(4):
                (R.of["look"] / carve / "chunks" / f"c{i:05d}.npz").write_bytes(b"c")
            write_json(R.of["records"] / f"look_{tag(carve)}.json", {"verdict": "LOOK_RESTORED"})
        assert verify(R) == 0 and read_json(R.of["records"] / "verify.json")["verdict"] == "WHOLE"
        for carve in LOOKS:
            for i in range(4):
                (R.of["look"] / carve / "chunks" / f"c{i:05d}.npz").unlink()
            (R.of["records"] / f"look_{tag(carve)}.json").unlink()
        (R.of["cache"] / "metaqa/fit/part_1of2/a.npy").unlink()
        assert verify(R) == 1
        # 2. backup: every record once; again, a check only; a changed file refuses
        assert backup(R) == 0
        n = len(read_json(R.of["records"] / "backup.json")["files"])
        assert n == sum(len(list((R.of["look"] / c).glob("*.json"))) for c in LOOKS) + \
            sum(len(list((R.of[k] / c).glob("part_*/record.json"))) for k in KINDS for c in NEEDED[k])
        assert backup(R) == 0 and read_json(R.of["records"] / "backup.json")["counts"] == {"present": n}
        rj = R.of["zrc"] / "metaqa/fit/part_0of2/record.json"
        keep = rj.read_bytes()
        rj.write_bytes(keep + b" ")
        assert refuses(backup, R)
        rj.write_bytes(keep)
        # 3. look-check: missing chunks refuse; a record rewritten on run keys (at any depth) and code shas goes back;
        #    content refuses; ids.json is byte for byte; a new json refuses
        lk = R.of["look"] / "metaqa/fit"
        assert look_check("metaqa/fit", R) == 2
        assert read_json(R.of["records"] / "look_metaqa_fit.json")["verdict"] == "LOOK_DIFFERS"
        for i in range(4):
            (lk / "chunks" / f"c{i:05d}.npz").write_bytes(b"c")
        orig0 = (lk / "record_0of2.json").read_bytes()
        write_json(lk / "record_0of2.json", {**look_rec(0, 2), "seconds": 9, "utc": "v", "placement": {"b": 2},
                                             "peak_rss_bytes": 7, "audit": {"every": 0, "seconds": 7.0},
                                             "script_sha256": "s2"})
        assert look_check("metaqa/fit", R) == 0
        assert (lk / "record_0of2.json").read_bytes() == orig0
        lr = read_json(R.of["records"] / "look_metaqa_fit.json")
        assert lr["put_back"] == 1 and lr["code_differs"] == {R.rel(lk / "record_0of2.json"): ["script_sha256"]}
        write_json(lk / "record_1of2.json", look_rec(1, 2, chunks=[2, 3, 4]))
        assert look_check("metaqa/fit", R) == 2
        write_json(lk / "record_1of2.json", look_rec(1, 2, threads={"torch": 4, "blas": "1"}))
        assert look_check("metaqa/fit", R) == 2
        (lk / "record_1of2.json").write_bytes((R.of["backup"] / R.rel(lk / "record_1of2.json")).read_bytes())
        (lk / "ids.json").write_text('["q1","q2"]', encoding="utf-8")
        assert look_check("metaqa/fit", R) == 2
        (lk / "ids.json").write_text('["q1", "q2"]', encoding="utf-8")
        write_json(lk / "record_9of9.json", look_rec(0, 2))
        assert look_check("metaqa/fit", R) == 2
        (lk / "record_9of9.json").unlink()
        assert look_check("metaqa/fit", R) == 0 and read_json(R.of["records"] / "look_metaqa_fit.json")["put_back"] == 0
        mb1 = R.of["look"] / "musique/b1"
        for i in range(4):
            (mb1 / "chunks" / f"c{i:05d}.npz").write_bytes(b"c")
        write_json(mb1 / "b1_0of2.json", {"pools": {"mean": 3, "seconds": 5.0}, "utc": "w",
                                          "script_sha256": {"look_b1": "b"}})
        assert look_check("musique/b1", R) == 0
        write_json(mb1 / "b1_0of2.json", {"pools": {"mean": 4, "seconds": 5.0}, "utc": "w",
                                          "script_sha256": {"look_b1": "a"}})
        assert look_check("musique/b1", R) == 2
        (mb1 / "b1_0of2.json").write_bytes((R.of["backup"] / R.rel(mb1 / "b1_0of2.json")).read_bytes())
        assert refuses(look_check, "2wiki/b1", R)
        # 4. cache: a look not checked refuses; the runner rebuilds only the parts that miss an array (one part with
        #    --part); the missing array goes back; nothing existing is touched; a whole carve runs nothing
        assert refuses(cache_cmd, "musique/s1fit", False, R, run=None)
        calls = []

        def fake_run(cmd, cwd=None, corrupt=(), edit=None):
            calls.append(cmd)
            ds, cv = cmd[cmd.index("--dataset") + 1], cmd[cmd.index("--carve") + 1]
            i, n_ = cmd[cmd.index("--part") + 1].split("/")
            orig = read_json(R.of["cache"] / ds / cv / f"part_{i}of{n_}" / "record.json")
            arrays = {k: f"cache{ds}/{cv}{i}{k}".encode() for k in ("a", "b")}
            part(cmd[cmd.index("--out-root") + 1], f"{ds}/{cv}/part_{i}of{n_}",
                 {k: (b"#" * len(v) if k in corrupt else v) for k, v in arrays.items()},
                 **{k: v for k, v in orig.items() if k not in ("arrays", "seconds", "utc")}, **(edit or {}))
            return subprocess.CompletedProcess(cmd, 0)

        b0 = R.of["cache"] / "metaqa/fit/part_0of2/b.npy"
        t0 = b0.stat().st_mtime_ns
        (R.of["cache"] / "metaqa/fit/part_0of2/a.npy").unlink()          # part 1's a.npy is missing already
        assert cache_cmd("metaqa/fit", True, R, run=fake_run, only=1) == 0
        assert len(calls) == 1 and "--host" in calls[0] and calls[0][calls[0].index("--part") + 1] == "1/2"
        assert calls[0][calls[0].index("--bases") + 1] == "2wiki,hotpotqa" and "--look-root" not in calls[0]
        assert not (R.of["cache"] / "metaqa/fit/part_0of2/a.npy").exists()
        ip = read_json(R.of["records"] / "install_cache_metaqa_fit_p1.json")
        assert ip["whole"] and ip["part"] == 1 and len(ip["units"]) == 1 and ip["not_reproduced"] == 0
        assert ip["put"] in ({"linked": 1}, {"copied": 1})
        assert all(P.base_status(R.rb["cache"], g) == "ok" for g in P.BASES)
        assert refuses(cache_cmd, "metaqa/fit", True, R, run=fake_run, only=5) and len(calls) == 1
        calls.clear()
        assert cache_cmd("metaqa/fit", True, R, run=fake_run) == 0       # the carve's own row: what is missing, the verdict
        assert len(calls) == 1 and calls[0][calls[0].index("--part") + 1] == "0/2"
        ins = read_json(R.of["records"] / "install_cache_metaqa_fit.json")
        assert ins["whole"] and ins["part"] is None and len(ins["units"]) == 2 and ins["not_reproduced"] == 0
        assert ins["put"] in ({"linked": 1}, {"copied": 1}) and b0.stat().st_mtime_ns == t0
        calls.clear()
        assert cache_cmd("metaqa/fit", True, R, run=fake_run) == 0 and not calls
        # a rebuilt array that is not the record's never goes back; nor anything of a record that differs
        a2 = R.of["cache"] / "2wiki/b1/part_0of1/a.npy"
        a2.unlink()
        assert cache_cmd("2wiki/b1", True, R, run=lambda c, cwd=None: fake_run(c, cwd, corrupt=("a",))) == 1
        assert not a2.exists() and read_json(R.of["records"] / "install_cache_2wiki_b1.json")["put"] == {"differs": 1}
        shutil.rmtree(R.rb["cache"] / "2wiki/b1")
        assert cache_cmd("2wiki/b1", True, R, run=lambda c, cwd=None: fake_run(c, cwd, edit={"look": "other"})) == 1
        assert not a2.exists() and read_json(R.of["records"] / "install_cache_2wiki_b1.json")["put"] == {"record differs": 1}
        shutil.rmtree(R.rb["cache"] / "2wiki/b1")
        assert cache_cmd("2wiki/b1", True, R, run=fake_run) == 0 and a2.is_file()
        # 5. chains: a look not checked refuses; zrc waits for rmatch; a builder whose default root is elsewhere
        #    refuses; the call carries only out_root and the placement
        built = []

        def builder(kind, roots=None):
            r = {"out_root": R.of[kind], "cache_root": R.of["cache"], "look_root": R.of["look"], **(roots or {})}

            def build(ds, carve, out_root=r["out_root"], cache_root=r["cache_root"], look_root=r["look_root"],
                      placement=None, rm_root=R.of["rmatch"]):
                built.append((kind, ds, carve, Path(out_root), placement))
                for u in carve_units(kind, R.of[kind], f"{ds}/{carve}"):
                    orig = read_json(u / "record.json")
                    part(out_root, u.relative_to(R.of[kind]).as_posix(),
                         {k: f"{kind}{ds}/{carve}{orig['part']}{k}".encode() for k in ("a", "b")},
                         **{k: v for k, v in orig.items() if k not in ("arrays", "seconds", "utc")})
            return build

        (R.of["rmatch"] / "metaqa/fit/part_0of2/a.npy").unlink()
        (R.of["zrc"] / "metaqa/fit/part_1of2/b.npy").unlink()
        assert refuses(chains_cmd, "zrc", "metaqa/fit", True, R, builder("zrc")) and not built
        assert refuses(chains_cmd, "rmatch", "squad/fit", True, R, builder("rmatch"))
        assert refuses(chains_cmd, "rmatch", "metaqa/fit", True, R, builder("rmatch", {"cache_root": R.base / "x"}))
        assert not built
        assert chains_cmd("rmatch", "metaqa/fit", True, R, builder("rmatch")) == 0
        assert chains_cmd("zrc", "metaqa/fit", False, R, builder("zrc")) == 0
        assert [x[0] for x in built] == ["rmatch", "zrc"] and built[0][3] == R.rb["rmatch"] and built[1][3] == R.rb["zrc"]
        assert built[0][4] == {"where": "host"} and built[1][4] == {"where": "laptop"}
        assert (R.of["rmatch"] / "metaqa/fit/part_0of2/a.npy").read_bytes() == b"rmatchmetaqa/fit0a"
        (R.of["links"] / "musique/b1t/part_0of1/a.npy").unlink()
        # musique/b1t's step-1 cache is whole, but its look-check has not run: the chain refuses
        assert refuses(chains_cmd, "links", "musique/b1t", True, R, builder("links"))
        mt = R.of["look"] / "musique/b1t"
        for i in range(4):
            (mt / "chunks" / f"c{i:05d}.npz").write_bytes(b"c")
        assert look_check("musique/b1t", R) == 0
        assert chains_cmd("links", "musique/b1t", True, R, builder("links")) == 0
        # 6. verify: whole once every look is checked; the queue block holds together
        assert verify(R) == 1
        for carve in LOOKS:
            for i in range(4):
                (R.of["look"] / carve / "chunks" / f"c{i:05d}.npz").write_bytes(b"c")
            assert look_check(carve, R) == 0
        assert verify(R) == 0
        write_json(R.of["cache"] / "basis_2wiki.json", {"npz_sha256": "0" * 64})
        assert verify(R) == 1 and read_json(R.of["records"] / "verify.json")["bases"]["2wiki"] == "differs"
        blk = block()
        n_cache = sum(PARTS.get(c, 1) + (PARTS.get(c, 1) > 1) for c in MISSING["cache"])
        assert sum(r[1].startswith("j5r-look-") for r in blk) == 24 and n_cache == 28
        assert len(blk) == 1 + 24 + 8 + n_cache + 2 + 8 + 1
        assert set(next(r for r in blk if r[1] == "j5r-verify")[0]) == \
            {f"j5r-cache-{tag(c)}" for c in MISSING["cache"]} | {"j5r-rm-metaqa_fit", "j5r-zrc-metaqa_fit"} | \
            {f"j5r-zl-{tag(c)}" for c in MISSING["links"]}
        assert next(r for r in blk if r[1] == "j5r-cache-musique_s1fit")[0] == tuple(f"j5r-cache-musique_s1fit-p{i}" for i in range(6))
        # no record names the temporary folder
        for f in R.of["records"].glob("*.json"):
            t = f.read_text(encoding="utf-8")
            assert td not in t and Path(td).as_posix() not in t
    log("j5s_restore selftest: PASS")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="J5S amendment A1: the removed looks and caches back (see the docstring)")
    ap.add_argument("cmd", nargs="?", choices=("backup", "look-check", "cache", "chains", "verify", "rows"))
    ap.add_argument("--carve")
    ap.add_argument("--kind", choices=tuple(BUILDER))
    ap.add_argument("--part", type=int, default=None, help="cache: this part of the carve only")
    ap.add_argument("--host", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.cmd == "rows":
        return rows()
    R = Roots()
    if a.cmd == "backup":
        return backup(R)
    if a.cmd == "verify":
        return verify(R)
    if a.cmd is None:
        ap.error("a command, or --selftest")
    if not a.carve:
        raise SystemExit(f"{a.cmd}: --carve ds/carve")
    if a.cmd == "look-check":
        return look_check(a.carve, R)
    if a.cmd == "cache":
        return cache_cmd(a.carve, a.host, R, only=a.part)
    if a.kind is None:
        raise SystemExit("chains: --kind rmatch, zrc or links")
    return chains_cmd(a.kind, a.carve, a.host, R)


if __name__ == "__main__":
    sys.exit(main())
