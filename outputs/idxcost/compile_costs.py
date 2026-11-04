"""File the measured costs of building the LLM-free link build's index components on the full corpora, one component at
a time.

Components, each its own section of the report: the W2_S rule links (U2b, Modal CPU containers; no spaCy), the ner
family's spaCy extraction (A6's re-run on the host; the original build was not timed), the laptop name index (P4a),
and, listed beside because it is not in the final system, P2c part 2's parse. The hardware lines come from the host
handover doc and from the Modal script's decorators (through git).

It reads records only. It measures nothing, fits nothing, uses no GPU and no network, and prints the files it read.
There is no total across components, no per-passage rate and no extrapolation. The only arithmetic is a sum, a minimum
or a maximum over the part records of ONE component of ONE corpus, marked in the item; where a record holds its own
number for the same thing, the two are checked (see "checks" in report.json). The script stops when a check fails.

    PYTHONIOENCODING=utf-8 python -I outputs/idxcost/compile_costs.py

Writes report.json and report.md beside this script. Names of hosts, accounts, addresses and user paths are masked in
what it writes, and it stops if one is left.
"""
import ast
import getpass
import hashlib
import importlib.util
import json
import math
import os
import re
import socket
import subprocess
import sys
import time
from datetime import datetime
from importlib import metadata
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SIB = ROOT.parent / "CRAG"                                  # the frozen sibling package, read-only
CORPORA = ("2wiki", "hotpotqa", "musique", "squad")
SCOPE = ("outputs", "docs", "src", "scripts")
MODAL = "outputs/mp_unified/zu2b_modal.py"
FEEDER = "outputs/host_ops/hostfeed2.py"
ITEMS = "outputs/host_ops/hostfeed_items.txt"
RETRY = "outputs/host_ops/retry_cmd.py"
STATE = "outputs/host_ops/hostfeed_state.json"
HOST_DOC = "docs/GPU_HOST_HANDOVER.md"
U2B_DOC = "docs/U2B_LINK_RULE.md"
U2C_DOC = "docs/U2C_POOLS_ON_CHOSEN.md"
U2L_DOC = "docs/U2L_ENTITY_MATCH.md"
U2M_DOC = "docs/U2M_POOLS_THREE_FAMILIES.md"
U2R_DOC = "docs/U2R_NAME_RULE.md"
NMD_DOC = "docs/NMD_NAME_LINK_DIAG.md"
B1_DOC = "docs/B1_HIPPORAG2_ALIGNMENT.md"
B1H_DOC = "docs/B1H_MODELS_OF_RECORD_ON_HYPERLINKS.md"
B1L_DOC = "docs/B1L_MODELS_OF_RECORD_ON_THE_LLM_GRAPH.md"
B1_BUILD = "scripts/b1_build.py"
P1E_DOC = "docs/P1E_ANCHOR_KIND.md"
STEP4_DOC = "docs/STEP4_POOL_REACH.md"
A6_DOC = "docs/A6_LINK_WORDS_INPUT.md"
P2C_DOC = "docs/P2C_LLM_FREE_TRIPLES.md"
P2C2_DOC = "docs/P2C2_FOURTH_FAMILY.md"
P4A_DOC = "docs/P4A_POOL_COLD.md"
ENC_DOC = "docs/M3A_WEBQSP_ENCODE_BUDGET.md"
POOLS = "src/mp_retrieval/m3b_pools.py"
RULE_CODE = ("outputs/mp_unified/zu2b.py", "outputs/mp_unified/u1b.py", "outputs/mp_unified/u1d.py")
RULE_HASH_KEYS = ("zu2b_sha256", "u1b_sha256", "u1d_sha256")
STAGE_FUNCS = ("stat", "merge", "link", "build", "build_part")
LABEL_W2S = "W2_S rule links (no spaCy)"
LABEL_NER = "ner family (spaCy): A6's re-run of its extraction (the original build was not timed)"
NER_DESC = "spaCy NER over all passages, one CPU reserved per part, summed over parts"


# -- git: what is tracked, and the last commit of a file ---------------------------------------------------------------

GIT_ENV = {**os.environ, "GIT_OPTIONAL_LOCKS": "0"}            # read-only: never take the index lock


def run_git(cwd, *args):
    p = subprocess.run(["git", *args], cwd=cwd, capture_output=True, env=GIT_ENV)
    if p.returncode != 0:
        raise SystemExit(f"git {' '.join(args)} failed: {p.stderr.decode('utf-8', 'replace').strip()[:300]}")
    return p.stdout


def git(*args):
    return run_git(ROOT, *args).decode("utf-8", "replace")


TRACKED = set(git("ls-files", "-z", "--", *SCOPE).split("\0"))
MODIFIED = set(git("diff", "--name-only", "-z", "HEAD", "--", *SCOPE).split("\0"))
_commit = {}


def prov(rel):
    """Tracked, with `git log -n1 --format='%h %ci'` of the file, or untracked."""
    if rel not in TRACKED:
        return {"tracked": False, "commit": "untracked"}
    if rel not in _commit:
        _commit[rel] = git("log", "-n1", "--format=%h %ci", "--", rel).strip()
    out = {"tracked": True, "commit": _commit[rel]}
    if rel in MODIFIED:
        out["modified_since_commit"] = True
    return out


def first_commit(rel):
    """The commit that added the file, as '%h %ci'."""
    rows = [r for r in git("log", "--diff-filter=A", "--format=%h %ci", "--", rel).strip().split("\n") if r]
    return rows[-1] if rows else None


def commit_of(rel, subject_start):
    """(%h, %ci) of the commit of `rel` whose subject starts with `subject_start`."""
    for row in git("log", "--format=%h|%ci|%s", "--", rel).strip().split("\n"):
        h, d, s = row.split("|", 2)
        if s.startswith(subject_start):
            return h, d
    raise SystemExit(f"no commit of {rel} starts with: {subject_start}")


# -- reading: every file read is registered with its sha256 and printed at the end -------------------------------------

READ = {}
GROUPS = {}


def note(label, data, provenance):
    READ.setdefault(label, {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data), **provenance})


def read_bytes(rel):
    data = (ROOT / rel).read_bytes()
    note(rel, data, prov(rel))
    return data


def jload(rel):
    for _ in range(5):
        data = (ROOT / rel).read_bytes()
        try:
            doc = json.loads(data.decode("utf-8"))
        except ValueError:                      # a live file caught while it is being rewritten
            time.sleep(1.0)
            continue
        READ.pop(rel, None)
        note(rel, data, prov(rel))
        return doc
    raise SystemExit(f"{rel}: not readable as JSON")


def doc_text(rel):
    return read_bytes(rel).decode("utf-8").replace("\r\n", "\n")


def doc_lines(rel):
    return doc_text(rel).split("\n")


def head_lines(rel):
    """The committed text of a file (for a file with edits in the working tree that this record does not use)."""
    data = run_git(ROOT, "show", f"HEAD:{rel}")
    note(f"{rel} (HEAD version)", data, prov(rel))
    return data.decode("utf-8").replace("\r\n", "\n").split("\n")


def find(lines, pattern):
    rx = re.compile(pattern)
    for n, ln in enumerate(lines, 1):
        m = rx.search(ln)
        if m:
            return n, m
    raise SystemExit(f"not found: {pattern}")


def find_text(text, pattern, base_line=1):
    """Like find, over the whole text (a pattern may span lines); returns the line the match starts on."""
    m = re.search(pattern, text)
    if not m:
        raise SystemExit(f"not found: {pattern}")
    return base_line + text.count("\n", 0, m.start()), m


# -- the records' numbers: each carries its record, key, tracked or untracked, commit and kind -------------------------

def item(value, unit, kind, rel, key, pv=None, **more):
    d = {"value": value, "unit": unit, "kind": kind, "record": rel, "key": key, **(pv or prov(rel))}
    d.update(more)
    return d


def absent(rel, key, held, meaning):
    return {"value": None, "unit": None, "kind": "not recorded", "record": rel, "key": key, "record_holds": held,
            "meaning": meaning, **prov(rel)}


def peak_item(v, rel, key, **more):
    if v in (0, None):
        return absent(rel, key, v, "the record holds no peak for this step")
    return item(v, "GB", "measured", rel, key, **more)


CHECKS = []


def check(name, ok, detail):
    CHECKS.append({"check": name, "ok": bool(ok), "detail": detail})
    return bool(ok)


def fsum1(xs):
    return round(math.fsum(xs), 1)


def strings(o):
    if isinstance(o, str):
        yield o
    elif isinstance(o, dict):
        for k, v in o.items():
            yield from strings(k)
            yield from strings(v)
    elif isinstance(o, (list, tuple)):
        for v in o:
            yield from strings(v)


# -- the Modal script's declared container specs and facts, by commit ---------------------------------------------------

def decorated(src):
    """{function name: (the keyword arguments of its app.function decorator, decorator line, def line)}."""
    out = {}
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.FunctionDef):
            for dec in node.decorator_list:
                if isinstance(dec, ast.Call) and getattr(dec.func, "attr", "") == "function":
                    out[node.name] = ({kw.arg: kw.value for kw in dec.keywords if kw.arg}, dec.lineno, node.lineno)
    return out


def modal_specs(src):
    out = {}
    for name, (kws, dec_line, def_line) in decorated(src).items():
        if name in STAGE_FUNCS:
            spec = {"def_line": def_line, "decorator_line": dec_line}
            for k in ("cpu", "memory", "nonpreemptible", "gpu", "timeout"):
                if k in kws:
                    spec[k] = ast.literal_eval(kws[k])
            out[name] = spec
    return out


def func_source(src, name):
    node = next((n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.FunctionDef) and n.name == name), None)
    if node is None:
        raise SystemExit(f"{MODAL}: no function {name}")
    return ast.get_source_segment(src, node)


def modal_history():
    commits = git("log", "--format=%h", "--", MODAL).split()[::-1]            # oldest first
    specs, dates, subjects, sources, gpu_any, digest = {}, {}, {}, {}, [], hashlib.sha256()
    for c in commits:
        raw = run_git(ROOT, "show", f"{c}:{MODAL}")
        src = raw.decode("utf-8").replace("\r\n", "\n")
        sources[c] = src
        specs[c] = modal_specs(src)
        gpu_any += [f"{c} {n}" for n, (kws, _a, _b) in decorated(src).items() if "gpu" in kws]
        dates[c] = git("log", "-n1", "--format=%ci", c).strip()
        subjects[c] = git("log", "-n1", "--format=%s", c).strip()
        digest.update(f"{c} {hashlib.sha256(raw).hexdigest()}\n".encode())
    GROUPS[f"{MODAL} (every committed version)"] = {
        "commits": commits, "digest_sha256": digest.hexdigest(),
        "digest_of": "'<commit> <sha256 of the file at it>' lines, oldest first", **prov(MODAL)}
    return commits, specs, dates, subjects, sources, gpu_any


def modal_facts(commits, sources):
    last = sources[commits[-1]]
    dicts = {}
    for node in ast.parse(last).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) \
                and node.targets[0].id in ("STAGE", "ORCH") and isinstance(node.value, ast.Dict):
            dicts[node.targets[0].id] = [k.value for k in node.value.keys if isinstance(k, ast.Constant)]
    fast = {c: "zu2b_fast.link_pass(" in func_source(sources[c], "link") for c in commits}
    sep = {c: "link_shards: int" in sources[c] for c in commits}
    rusage = {c: "ru_maxrss" in sources[c] for c in commits}
    pip = re.search(r"pip_install\((.*?)\)", last, re.S).group(1)
    retries = re.search(r"STAGE = \{.*?max_retries=(\d+)", last, re.S)
    first_fast = next(c for c in commits if fast[c])
    first_sep = next(c for c in commits if sep[c])
    at = commits.index(first_sep)
    earlier = func_source(sources[commits[at - 1]], "corpus") if at else ""
    later = func_source(sources[first_sep], "corpus")
    return {"stage_keys": dicts.get("STAGE", []), "orch_keys": dicts.get("ORCH", []),
            "first_fast": first_fast, "first_sep": first_sep,
            "link_shard_default": {
                "commit_before": commits[at - 1] if at else None,
                "earlier_one_link_container_per_stat_shard":
                    bool(re.search(r"link\.starmap\(\[\(d, rule_name, i, k\) for i in range\(k\)\]\)", earlier)),
                "later_unset_equals_stat": "kl = k_link or k" in later and "k_link=0" in later},
            "fast_and_sep_from_first": all(fast[c] and sep[c] for c in commits[commits.index(first_fast):]),
            "first_rusage": next((c for c in commits if rusage[c]), None),
            "pinned_packages": re.findall(r'"([A-Za-z0-9_.-]+==[0-9.]+)"', pip),
            "added_files": re.findall(r'add_local_file\("([^"]+)"', last),
            "max_retries": int(retries.group(1)) if retries else None,
            "single_use": bool(re.search(r'"single_use_containers": True', last)),
            "image": (re.search(r'debian_slim\(python_version="([\d.]+)"\)', last) or [None, None])[1],
            "threads": dict(re.findall(r'"((?:OPENBLAS|OMP|MKL)_NUM_THREADS)": "(\d+)"', last)),
            "shard_json_written": bool(re.search(r'_write_json\(od / f"s\{i\}of\{k\}\.json", rec\)', last))}


def memtxt(m):
    return f"({m[0]}, {m[1]})" if isinstance(m, (list, tuple)) else str(m)


def modal_memory_unit():
    """The unit of the decorators' memory argument, from the installed modal package's own comment (no import)."""
    spec = importlib.util.find_spec("modal")
    if not spec or not spec.submodule_search_locations:
        return {"unit": None, "why": "the modal package is not installed here"}
    p = Path(list(spec.submodule_search_locations)[0]) / "app.py"
    if not p.exists():
        return {"unit": None, "why": "the installed modal package has no app.py"}
    data = p.read_bytes()
    note("installed modal package: app.py", data, {"tracked": False, "commit": "not in this repository"})
    m = re.search(r"Specify, in (MiB), a memory request which is the minimum memory required", data.decode("utf-8"))
    try:
        ver = metadata.version("modal")
    except metadata.PackageNotFoundError:
        ver = None
    return {"unit": m.group(1) if m else None, "modal_version": ver,
            "read_from": "the comment above the memory argument of App.function in the installed modal package "
                         "(a version that may differ from the one the stages ran under; the records name none)"}


# -- P2c part 2's parse parts (beside) and the passage counts -----------------------------------------------------------

def parse_section():
    by = {ds: [] for ds in CORPORA}
    digest, names = hashlib.sha256(), []
    for f in sorted((ROOT / "outputs" / "p2c").glob("parse_*of*.json"), key=lambda p: p.name):
        m = re.fullmatch(r"parse_(\w+?)_(\d+)of(\d+)\.json", f.name)
        if not m or m.group(1) not in by:
            continue
        data = f.read_bytes()
        digest.update(f"{f.name} {hashlib.sha256(data).hexdigest()}\n".encode())
        names.append(f"outputs/p2c/{f.name}")
        by[m.group(1)].append((int(m.group(2)), int(m.group(3)), json.loads(data.decode("utf-8"))))
    if any(n in TRACKED for n in names):
        raise SystemExit("some parse parts are tracked: this script treats them as untracked part records")
    GROUPS["outputs/p2c/parse_<corpus>_<k>of<n>.json"] = {
        "files": len(names), "tracked": False, "commit": "untracked", "digest_sha256": digest.hexdigest(),
        "digest_of": "sorted '<file name> <sha256>' lines"}
    out = {}
    for ds, rows in by.items():
        rows.sort(key=lambda r: r[0])
        n = rows[0][1]
        rel = f"outputs/p2c/parse_{ds}_<k>of{n}.json"
        ok = [r[0] for r in rows] == list(range(n)) and all(r[1] == n and r[2]["part"] == [r[0], n] for r in rows)
        check(f"{ds}: the parse parts are k = 0..{n - 1} of {n}, each naming itself", ok, cnt(len(rows), "file"))
        check(f"{ds}: every parse part ran on the host", {r[2]["where"] for r in rows} == {"host"}, "key where")
        secs = [r[2]["seconds"] for r in rows]
        pv = {"tracked": False, "commit": "untracked"}
        out[ds] = {
            "parts": item(len(rows), "parts", "counted", rel, "file count (the names say k of n)", pv),
            "seconds_sum": item(fsum1(secs), "s", "measured", rel, "seconds", pv, arithmetic="sum",
                                over_records=len(rows)),
            "seconds_min": item(min(secs), "s", "measured", rel, "seconds", pv, arithmetic="min",
                                over_records=len(rows)),
            "seconds_max": item(max(secs), "s", "measured", rel, "seconds", pv, arithmetic="max",
                                over_records=len(rows)),
            "peak_gb_max": item(max(r[2]["peak_gb"] for r in rows), "GB", "measured", rel, "peak_gb", pv,
                                arithmetic="max", over_records=len(rows)),
            "passages_sum": item(sum(r[2]["passages"] for r in rows), "passages", "counted", rel, "passages", pv,
                                 arithmetic="sum", over_records=len(rows))}
    return out


def passages_section(caps, parse):
    u2 = doc_text(U2B_DOC)
    n_doc, m_doc = find_text(u2, r"of ([\d,]+) on hotpotqa and [\d,]+ of ([\d,]+) on 2wiki")
    out = {}
    for ds in CORPORA:
        rel = f"outputs/u2b/{ds}/new/capstat.json"
        n = caps[ds]["n_nodes"]
        out[ds] = item(n, "passages", "counted", rel, "n_nodes")
        check(f"{ds}: n_nodes equals the sum of passages over the parse parts", n == parse[ds]["passages_sum"]["value"],
              f"{n} and {parse[ds]['passages_sum']['value']}")
    check("hotpotqa and 2wiki: the counts in the link rule doc equal n_nodes",
          int(m_doc.group(1).replace(",", "")) == caps["hotpotqa"]["n_nodes"] and
          int(m_doc.group(2).replace(",", "")) == caps["2wiki"]["n_nodes"], f"{U2B_DOC} line {n_doc}")
    return out


# -- the W2_S rule links: U2b on Modal CPU containers ------------------------------------------------------------------

def w2s_section(caps, builds, commits, specs, subjects, sources, facts, gpu_any):
    u2 = doc_text(U2B_DOC)
    ul = u2.split("\n")
    n229, m229 = find(ul, r"From ([0-9a-f]{7,40}), 2wiki's build ran one variant per container: ([\d,]+) s in all, "
                          r"with a peak of ([\d.]+) GB")
    bc = next(c for c in commits if c.startswith(m229.group(1)) or m229.group(1).startswith(c))
    n151, _ = find(ul, r"The build stage makes every graph of a rule in one container")
    n148, m148 = find(ul, r"link stage runs in (\d+) shards")
    n_retry, _ = find_text(u2, r"every stage now runs one input per container and is\s+retried; a retry redoes nothing")
    n_pre, _ = find(ul, r"2wiki's was preempted four times")
    n_chosen, _ = find(ul, r"\*\*CHOSEN W2_S\.\*\*")
    n_rule, _ = find(ul, r"U1d's surfaces, case gate, capitalization statistic, AMBIG guard")
    n_plain, _ = find(ul, r"It is the plainest variant: the look")
    n_form_s, _ = find(ul, r"^- \*\*S:\*\* M plus sibling links, between descriptive nodes that share a title")
    n_stages, _ = find(ul, r"The stages are stat, merge, link and build, one container per shard")
    n_fast, _ = find(ul, r"The link stage runs `zu2b_fast\.link_pass`, which gives exactly")
    before = commits[:commits.index(bc)]
    first_fast, first_sep = facts["first_fast"], facts["first_sep"]
    cpu_hist = {fn: sorted({s[fn]["cpu"] for s in specs.values() if fn in s}) for fn in STAGE_FUNCS}
    same = f"as written in each committed version that has it ({len(specs)} versions)"

    def cpu_item(fn, values, key_note):
        v = values[0] if len(values) == 1 else values
        return item(v, "cpus", "declared", MODAL, f"@app.function(cpu=...) of {fn}, {key_note}",
                    commits_read=len(specs))

    # the code the records name, against the working tree; no mention of spaCy anywhere the image reads
    shas = {f: hashlib.sha256(read_bytes(f)).hexdigest() for f in RULE_CODE}
    for ds in CORPORA:
        ok = all(rec[k] == shas[f] for rec in (caps[ds], builds[ds]) for k, f in zip(RULE_HASH_KEYS, RULE_CODE))
        check(f"{ds}: the stat and build records name the rule code the working tree holds (sha256 of zu2b.py, "
              f"u1b.py, u1d.py)", ok, ", ".join(f"{f.split('/')[-1]} {shas[f][:12]}" for f in RULE_CODE))
    no_spacy = {f: not re.search("spacy", read_bytes(f).decode("utf-8"), re.I) for f in facts["added_files"]}
    check("no source file the Modal image adds mentions spaCy", all(no_spacy.values()),
          ", ".join(facts["added_files"]))
    check("the Modal image installs no spaCy", not any("spacy" in p.lower() for p in facts["pinned_packages"]),
          ", ".join(facts["pinned_packages"]))
    check("the Modal image sets OPENBLAS_NUM_THREADS, OMP_NUM_THREADS and MKL_NUM_THREADS, all to one value",
          sorted(facts["threads"]) == ["MKL_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"] and
          len(set(facts["threads"].values())) == 1, ", ".join(f"{k}={v}" for k, v in facts["threads"].items()))
    check("no function of any committed Modal script passes a gpu argument, and the stage and orchestrator "
          "settings name none", not gpu_any and "gpu" not in facts["stage_keys"] + facts["orch_keys"],
          f"{len(commits)} versions")
    check("the separate link shard count and the zu2b_fast link pass enter the script in one commit, and stay",
          first_fast == first_sep and facts["fast_and_sep_from_first"],
          f"the separate count first at {first_sep}, the zu2b_fast call first at {first_fast}")
    lsd = facts["link_shard_default"]
    check("an unset link shard count equals the stat shard count: one link container per stat shard before the "
          "separate count, and kl = k_link or k (k_link defaults to 0) from it on",
          lsd["earlier_one_link_container_per_stat_shard"] and lsd["later_unset_equals_stat"],
          f"function corpus of {MODAL} at {lsd['commit_before']} and at {first_sep}")
    bad = []
    for c in commits:
        for fn in ("stat", "link", "build") + (("build_part",) if "build_part" in specs[c] else ()):
            seg = func_source(sources[c], fn)
            t0 = seg.find("t0 = time.time()")
            rl = [x for x in (seg.find("_reload()"), seg.find("vol.reload()")) if x >= 0]
            if t0 < 0 or not rl or min(rl) > t0 or "round(time.time() - t0, 1)" not in seg:
                bad.append(f"{c} {fn}")
    check("in every committed version, stat, link, build and build_part start their clock after the volume "
          "reload and write their seconds as the time since that start", not bad, ", ".join(bad) or f"{len(commits)} versions")
    link_seg = func_source(sources[bc], "link")
    check("link's seconds start after zu2b_fast's selftest", link_seg.find("_selftest()") < link_seg.find("t0 = time.time()"),
          f"{MODAL} at {bc}")
    bp = func_source(sources[bc], "build_part")
    check("build_part's seconds run from after the container is up and the code is imported, through reading the "
          "link shards, to the end of the part", 0 <= bp.find("t0 = time.time()") < bp.find("_build_inputs("),
          f"{MODAL} at {bc}, function build_part")
    bd = func_source(sources[bc], "build")
    check("the build container's seconds include starting every part and checking every graph file's hash",
          0 <= bd.find("build_part.starmap(") < bd.find("_sha(") < bd.find("round(time.time() - t0, 1)"),
          f"{MODAL} at {bc}, function build")
    check("the build record gets its 'parts' key from the commit the doc names, and not before",
          [c for c in commits if '"parts"' in func_source(sources[c], "build")] == commits[commits.index(bc):],
          f"first at {bc}")
    check("the commit the doc names states the layouts in its own comment",
          "32 graphs in turn" in sources[bc] and "all variants at once" in sources[bc], f"{MODAL} at {bc}")
    out = {}
    for ds in CORPORA:
        cap, b = caps[ds], builds[ds]
        crel, brel = f"outputs/u2b/{ds}/new/capstat.json", f"outputs/u2b/{ds}/graphs/build_new.json"
        variants = list(b["link_stats"].keys())
        forms = sorted({g.split("_")[-1] for g in b["graphs"]})
        link_shards = len(b["link_npz_sha256"])
        leaked = [k for rec in (cap, b) for k in rec if re.search(r"cpu|mem|core", k, re.I)]
        check(f"{ds}: the stat and build records hold no CPU count or memory key", not leaked,
              ", ".join(leaked) or "no key matches cpu, mem or core")
        check(f"{ds}: the stat record holds no merge seconds", not any("merge" in k for k in cap), "capstat.json keys")
        if ds in ("2wiki", "hotpotqa"):
            check(f"{ds}: the link record names the shard count the doc states", link_shards == int(m148.group(1)),
                  f"{link_shards} link shard hashes; {U2B_DOC} line {n148} says {m148.group(1)}")
        rec = {"component": LABEL_W2S, "variants_all": variants, "graph_forms": forms,
               "graphs_in_build_record": len(b["graphs"])}
        prel = f"outputs/u2b/{ds}/graphs/part_new_W2.json"
        if (ROOT / prel).exists():
            p = jload(prel)
            spec = specs[bc]["build_part"]
            check(f"{ds}: the W2 part file equals the build record's parts.W2",
                  b["parts"]["W2"] == {"secs": p["secs"], "peak_gb": p["peak_gb"]}, f"{p['secs']} s, {p['peak_gb']} GB")
            check(f"{ds}: the W2 part builds the graph W2_S", "W2_S" in p["graphs"], ", ".join(p["graphs"]))
            rec["rule_own_build"] = {
                "what": "the W2 variant's part, the part that builds W2_S; it builds four graphs together "
                        "(" + ", ".join(p["graphs"]) + "), so W2_S has no seconds figure of its own",
                "variant": p["variant"], "graphs": list(p["graphs"]), "containers": 1,
                "seconds": item(p["secs"], "s", "measured", prel, "secs"),
                "peak_gb": item(p["peak_gb"], "GB", "measured", prel, "peak_gb"),
                "cpus": item(spec["cpu"], "cpus", "declared", MODAL, "@app.function(cpu=...) of build_part",
                             read_at_commit=bc, decorator_line=spec["decorator_line"]),
                "memory": item(list(spec["memory"]), "as written", "declared", MODAL,
                               "@app.function(memory=...) of build_part", read_at_commit=bc,
                               decorator_line=spec["decorator_line"])}
        else:
            rec["rule_own_build"] = None
            rec["rule_own_build_why_not"] = (f"no per-variant part: the build record has no 'parts' key and there is "
                                             f"no part_new_*.json for {ds}; before {bc} one container built every "
                                             f"variant's graphs ({U2B_DOC} line {n151})")
            check(f"{ds}: the build record has no per-variant parts", "parts" not in b, "key parts absent")
        # which link pass: stated by the shard count where it can be, otherwise not stated
        if link_shards != cap["shards"]:
            rec_link_pass = {"value": "zu2b_fast.link_pass", "kind": "declared",
                             "basis": f"{link_shards} link shards against {cap['shards']} stat shards; the script has a "
                                      f"link shard count of its own only from {first_sep}, which also first calls "
                                      f"zu2b_fast.link_pass in its link stage"}
        else:
            rec_link_pass = {"value": "not stated in the record", "kind": "not recorded",
                             "basis": f"{link_shards} link shard, as for stat; the build record names no link pass; "
                                      f"its link and build peaks are empty (0 or null)",
                             "build_record_first_committed": first_commit(brel)}
            check(f"{ds}: the peaks are empty and the script before {facts['first_rusage']} has no peak fallback",
                  b["link_peak_gb_max"] in (0, None) and b["peak_gb"] in (0, None) and facts["first_rusage"] is not None
                  and "ru_maxrss" not in sources[commits[0]],
                  f"link_peak_gb_max {json.dumps(b['link_peak_gb_max'])}, peak_gb {json.dumps(b['peak_gb'])}")
        rec["stat"] = {
            "scope": "one pass over the corpus's passages for the rule, shared by every variant (not per variant)",
            "shards": item(cap["shards"], "shards", "counted", crel, "shards"),
            "seconds_max_over_shards": item(cap["stat_secs_max"], "s", "measured", crel, "stat_secs_max"),
            "peak_gb_max": peak_item(cap["stat_peak_gb_max"], crel, "stat_peak_gb_max"),
            "cpus": cpu_item("stat", cpu_hist["stat"], same)}
        rec["link"] = {
            "scope": "one pass that links all variants together (" + ", ".join(variants) + "); a seconds figure for "
                     "W2 alone does not exist",
            "link_pass": rec_link_pass,
            "shards": item(link_shards, "shards", "counted", brel, "link_npz_sha256 (its length)"),
            "seconds_max_over_shards": item(b["link_secs_max"], "s", "measured", brel, "link_secs_max"),
            "peak_gb_max": peak_item(b["link_peak_gb_max"], brel, "link_peak_gb_max"),
            "cpus": cpu_item("link", cpu_hist["link"], same)}
        if "parts" in b:
            check(f"{ds}: the build stage's seconds are at least its slowest part's",
                  b["secs"] >= max(v["secs"] for v in b["parts"].values()),
                  f"{b['secs']} and {max(v['secs'] for v in b['parts'].values())}")
            check(f"{ds}: the build record's parts are its variants", list(b["parts"]) == variants, str(variants))
            check(f"{ds}: the doc's build line rounds the record's secs and peak_gb",
                  round(b["secs"]) == int(m229.group(2).replace(",", "")) and round(b["peak_gb"], 1) ==
                  float(m229.group(3)), f"{b['secs']} s, {b['peak_gb']} GB; {U2B_DOC} line {n229}")
            layout = (f"{len(b['parts'])} build_part containers, one per variant, started by one build container "
                      f"(starmap); the seconds are that build container's")
            cpus = {"parts": item(specs[bc]["build_part"]["cpu"], "cpus", "declared", MODAL,
                                  "@app.function(cpu=...) of build_part", read_at_commit=bc),
                    "orchestrator": item(specs[bc]["build"]["cpu"], "cpus", "declared", MODAL,
                                         "@app.function(cpu=...) of build", read_at_commit=bc)}
        else:
            layout = (f"one container built every graph, in turn (no 'parts' key; the script before {bc} builds "
                      f"every graph in one container, {U2B_DOC} line {n151})")
            cpus = cpu_item("build", sorted({specs[c]["build"]["cpu"] for c in before}),
                            f"as written in the versions before {bc}: {', '.join(before)}")
        rec["build_all_variants"] = {
            "scope": "every variant (" + ", ".join(variants) + "), " + str(len(b["graphs"])) + " graphs; NOT the "
                     "rule's own build",
            "layout": layout,
            "seconds": item(b["secs"], "s", "measured", brel, "secs"),
            "peak_gb": peak_item(b["peak_gb"], brel, "peak_gb"),
            "cpus": cpus}
        out[ds] = rec
    cites = {"chosen_line": n_chosen, "rule_lines": f"{n_rule} to {n_rule + 1}", "retry_line": n_retry,
             "preempt_line": n_pre, "single_container_line": n151, "build_commit": bc,
             "plain_lines": f"{n_plain} to {n_plain + 1}", "form_s_line": n_form_s, "stages_line": n_stages,
             "fast_pass_lines": f"{n_fast} to {n_fast + 1}", "first_fast": first_fast, "first_sep": first_sep,
             "first_sep_subject": subjects[first_sep]}
    return out, bc, cites


def modal_hardware(commits, specs, dates, facts, bc, unit):
    return {
        "declared_in": MODAL, **prov(MODAL),
        "by_commit": {c: {"date": dates[c], "functions": {fn: {k: (list(v) if isinstance(v, tuple) else v)
                                                                for k, v in spec.items()}
                                                           for fn, spec in specs[c].items()}} for c in commits},
        "gpu_argument_anywhere": False,
        "one_input_per_container": facts["single_use"],
        "retries_per_stage_input": facts["max_retries"],
        "image": f"Debian slim, Python {facts['image']}, packages {', '.join(facts['pinned_packages'])}",
        "pinned_packages": list(facts["pinned_packages"]),
        "threads_set_in_the_image": {k: int(v) for k, v in facts["threads"].items()},
        "link_shard_default": facts["link_shard_default"],
        "memory_unit": unit,
        "seconds_are": "wall-clock seconds in the container (time.time differences), not CPU seconds",
        "cpu_model_of_the_workers": "not recorded"}


# -- the ner family's extraction: A6's NER jobs on the host ---------------------------------------------------------------

def a6_items():
    out = {}
    for ln in head_lines(ITEMS):
        f = ln.rstrip("\r").split("|")
        if len(f) in (8, 9) and f[0] == "line" and re.fullmatch(r"a6-(ents|merge)-[\w-]+", f[3]):
            out[f[3]] = {"cpus": float(f[4]), "mem": f[5], "command": f[7], "gpu_field": f[8].strip() if len(f) == 9 else ""}
    return out


def ner_section(caps):
    state = jload(STATE)
    sent, term = state["sent"], state["terminal"]
    a6 = doc_lines(A6_DOC)
    n_one, m_one = find(a6, r"one CPU each, on the host")
    n_res, m_res = find(a6, r"114 NER parts, 1 CPU and 1\.5 GB each")
    n_same, m_same = find(a6, r"as the graph's NER family was built")
    n_cnt, m_cnt = find(a6, r"metaqa (\d+), squad (\d+), musique (\d+), webqsp (\d+), hotpotqa (\d+), 2wiki (\d+)")
    doc_parts = dict(zip(("metaqa", "squad", "musique", "webqsp", "hotpotqa", "2wiki"), map(int, m_cnt.groups())))
    n_chk, m_chk = find_text("\n".join(a6), r"bit for bit: ([\d,]+) mentions, ([\d,]+) keys, every one of the graph's\s+"
                                            r"([\d,]+) NER edges rebuilt and none extra\. (\d+) s on one laptop CPU")
    feeder = head_lines(FEEDER)
    n_use, _ = find(feeder, r'use = max\(float\(j\.get\("peak_mem_gb"\)')
    n_ran, _ = find(feeder, r'ran = \(\(j\.get\("ended"\)')
    n_items, _ = find(feeder, r"need line\|JOBSFILE\|DEPS\|name\|cpus\|mem\|inputs\|command\[\|gpu\]")
    retry_src = doc_lines(RETRY)
    n_race, m_race = find(retry_src, r'^RACE = \("\[WinError 5\]", "\[WinError 32\]"\)')
    n_rdoc, _ = find(retry_src, r"up to --tries runs in all")
    items = a6_items()
    sq = jload("outputs/a6/ents/parts/squad/part_0of1.json")
    check("squad's laptop part covers the corpus's passages", sq["nodes"] == caps["squad"]["n_nodes"],
          f"{sq['nodes']} and {caps['squad']['n_nodes']}")
    check("the doc's laptop check names the squad part file's mentions and keys, and its seconds rounded",
          int(m_chk.group(1).replace(",", "")) == sq["mentions"] and int(m_chk.group(2).replace(",", "")) == sq["keys"]
          and round(sq["seconds"]) == int(m_chk.group(4)),
          f"{A6_DOC} line {n_chk}: {m_chk.group(1)} mentions, {m_chk.group(2)} keys, {m_chk.group(4)} s; the part file: "
          f"{sq['mentions']}, {sq['keys']}, {sq['seconds']}")
    # the settings the doc calls the ner family's, against the package's own build script
    b1 = doc_lines(B1_DOC)
    n_b1, _ = find(b1, r"rebuilt on the setting's passages by CRAG's `build_ner_edges` rule")
    bsrc = read_bytes(B1_BUILD).decode("utf-8")
    consts = {}
    for node in ast.parse(bsrc).body:
        if isinstance(node, ast.Assign):
            t = node.targets[0]
            if isinstance(t, ast.Name) and t.id == "ENT_LABELS":
                consts["labels"] = set(ast.literal_eval(node.value))
            if isinstance(t, ast.Tuple) and [e.id for e in t.elts if isinstance(e, ast.Name)] == ["DF_MIN", "DF_MAX"]:
                consts["df"] = list(ast.literal_eval(node.value))
    st = sq["settings"]
    n_ner, _ = find(bsrc.split("\n"), r"^def ner_edges\(")
    same = ('spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])' in bsrc and "len(e.text) > 2" in bsrc
            and st["key"] in bsrc and f"batch_size={st['batch_size']}" in bsrc and set(st["labels"]) == consts["labels"]
            and list(st["df"]) == consts["df"] and st["min_chars"] == 3 and sorted(st["disable"]) == ["lemmatizer", "parser"])
    check("A6's settings are the ner family's: labels, document-frequency range, minimum characters, key, batch size "
          "and disabled components equal those of ner_edges in the package's build script", same,
          f"{B1_BUILD} line {n_ner}; {B1_DOC} line {n_b1}")
    check("the retry wrapper reruns only after a Windows file-lock error",
          'RACE = ("[WinError 5]", "[WinError 32]")' in "\n".join(retry_src) and "any other failure ends at once" in
          "\n".join(retry_src), f"{RETRY} lines {n_race}, {n_rdoc}")
    other = list(strings(state["retried"])) + list(strings(state["dropped"])) + list(strings(state["shed"]))
    entries, out = [], {}
    n_parts = n_merge = 0
    for ds in CORPORA:
        rows = []
        for name, jid in sent.items():
            m = re.fullmatch(rf"a6-ents-{ds}-(\d+)of(\d+)", name)
            if m:
                rows.append((int(m.group(1)), int(m.group(2)), name, jid, term[jid]))
        rows.sort()
        n = rows[0][1]
        check(f"{ds}: A6's parts are k = 0..{n - 1} of {n}, as the doc's part count",
              [r[0] for r in rows] == list(range(n)) and all(r[1] == n for r in rows) and n == doc_parts[ds],
              f"{cnt(len(rows), 'job')}; {A6_DOC} line {n_cnt}")
        check(f"{ds}: every A6 part job is done with return code 0", all(r[4][0] == "done" and r[4][1] == 0 for r in rows),
              "terminal[job id][0] and [1]")
        mname = f"a6-merge-{ds}"
        mt = term[sent[mname]]
        check(f"{ds}: A6's merge job is done with return code 0", mt[0] == "done" and mt[1] == 0, mname)
        names = [r[2] for r in rows] + [mname]
        check(f"{ds}: each A6 job id carries its job's name, and no A6 job is in the queue's retried, dropped or shed "
              f"lists", all(f"-{nm}-" in sent[nm] for nm in names) and
              not [nm for nm in names if any(s == nm or f"-{nm}-" in s for s in other)], cnt(len(names), "job"))
        reqs = {(items[r[2]]["cpus"], items[r[2]]["mem"], items[r[2]]["gpu_field"]) for r in rows}
        cmd_ok = all(re.fullmatch(rf"python outputs/host_ops/retry_cmd\.py --tries 3 --wait 60 -- python "
                                  rf"outputs/mp_unified/za6lw\.py ents --dataset {ds} --part {r[0]}/{n} --host",
                                  items[r[2]]["command"]) for r in rows)
        check(f"{ds}: every A6 part's queue item asks for 1 CPU and 1.5 GB, names no GPU, and runs part k/n of the "
              f"corpus on the host under the retry wrapper (3 runs, 60 s apart)",
              reqs == {(1.0, "1.5", "")} and cmd_ok, f"{ITEMS} (HEAD version), {cnt(len(rows), 'item line')}")
        secs = [float(r[4][3]) for r in rows]
        peaks = [float(r[4][2]) for r in rows]
        entries += [(r[2], r[3], r[4]) for r in rows] + [(mname, sent[mname], mt)]
        n_parts += len(rows)
        n_merge += 1
        kseq = f"sent['a6-ents-{ds}-<k>of{n}'] -> job id; terminal[job id]"
        mreq = items[mname]
        out[ds] = {
            "component": LABEL_NER,
            "what": NER_DESC,
            "parts": item(n, "parts", "counted", STATE, f"sent['a6-ents-{ds}-<k>of{n}'], k = 0..{n - 1}"),
            "cpus_per_part": item(1.0, "cpus", "declared", ITEMS, f"field 5 (cpus) of the item lines a6-ents-{ds}-<k>of"
                                  f"{n} (HEAD version); {A6_DOC} line {n_res}: '{m_res.group(0)}'; "
                                  f"the request, not a measurement"),
            "memory_request_per_part": item(1.5, "GB", "declared", ITEMS, f"field 6 (mem) of the same item lines "
                                            f"(HEAD version); the feeder may send a different request "
                                            f"({FEEDER} lines 1 to 20 describe measured sizing)"),
            "job_seconds_sum": item(fsum1(secs), "s", "measured", STATE, kseq + "[3] (ran_s)", arithmetic="sum",
                                    over_records=n),
            "job_seconds_min": item(round(min(secs), 1), "s", "measured", STATE, kseq + "[3] (ran_s)",
                                    arithmetic="min", over_records=n),
            "job_seconds_max": item(round(max(secs), 1), "s", "measured", STATE, kseq + "[3] (ran_s)",
                                    arithmetic="max", over_records=n),
            "peak_gb_max": item(round(max(peaks), 2), "GB", "measured", STATE, kseq + "[2]", arithmetic="max",
                                over_records=n),
            "merge_job": {"cpus": item(mreq["cpus"], "cpus", "declared", ITEMS, f"field 5 of the item line {mname} "
                                       f"(HEAD version)"),
                          "seconds": item(round(float(mt[3]), 1), "s", "measured", STATE,
                                          f"terminal[sent['{mname}']][3] (ran_s)"),
                          "peak_gb": item(round(float(mt[2]), 2), "GB", "measured", STATE,
                                          f"terminal[sent['{mname}']][2]")}}
    out["squad"]["laptop_part"] = {
        "seconds": item(sq["seconds"], "s", "measured", "outputs/a6/ents/parts/squad/part_0of1.json", "seconds",
                        where=sq["where"], utc=sq["utc"])}
    digest = hashlib.sha256(json.dumps(sorted(entries), sort_keys=True).encode()).hexdigest()
    READ.pop(STATE, None)
    READ[STATE] = {"live_file": True, **prov(STATE), "entries_used": len(entries), "entries_sha256": digest,
                   "used": f"the {n_parts} A6 part jobs and {n_merge} A6 merge jobs of the four corpora: sent[name] "
                           f"and terminal[job id]; the rest of this live file is not used and not hashed"}
    local_parts = sorted(p.name for p in (ROOT / "outputs" / "a6" / "ents" / "parts").iterdir())
    meta = {"what_the_seconds_are": "wall seconds of one host job (the job's end minus its start, as the host queue "
                                    "recorded it), one CPU reserved per job; summed over parts they are reserved "
                                    "CPU-seconds (one CPU times wall seconds), not a CPU-time reading. The job runs "
                                    "under a retry wrapper "
                                    "that reruns only after a Windows file-lock error (up to 3 runs, 60 s apart), so "
                                    "a rerun, if one happened, is inside the job's seconds; the queue state does not "
                                    "say whether any did. The job also includes its own start-up (unpacking, imports, "
                                    "model load)",
            "peak_is": "the host job record's peak memory (the larger of its peak and live readings)",
            "semantics_read_at": f"{FEEDER} (HEAD version) lines {n_use} and {n_ran}; items format at line {n_items}; "
                                 f"{RETRY} lines {n_race} and {n_rdoc}",
            "same_settings_as_the_ner_family": f"{A6_DOC} line {n_same}: '{m_same.group(0)}'",
            "settings_checked_at": f"{B1_BUILD} line {n_ner}, whose rule the doc describes at {B1_DOC} line {n_b1}",
            "faithful_rerun": f"the doc's laptop check rebuilt every one of squad's {m_chk.group(3)} NER edges from "
                              f"A6's part, none extra ({A6_DOC} lines {n_chk} to {n_chk + 1})",
            "settings": {"spacy": st["spacy"], "model_version": st["model"], "disabled": st["disable"],
                         "labels": len(st["labels"]), "min_chars": st["min_chars"], "key": st["key"],
                         "batch_size": st["batch_size"], "document_frequency_range": st["df"]},
            "settings_read_from": "outputs/a6/ents/parts/squad/part_0of1.json settings",
            "reserved_per_part": f"{A6_DOC} line {n_res}: '{m_res.group(0)}'",
            "local_part_files": local_parts,
            "plan_figure_not_used": f"A plan figure for all 114 NER parts over six datasets is in {A6_DOC} lines "
                                    f"{n_res + 1} to {n_res + 2}, declared before the runs; it is an estimate and is "
                                    "not used here"}
    return out, meta


def pools_note():
    """The pools keep the package's NER: U2m section 20 (doc and record) and the NMD doc; the models read the ner family."""
    u2m = doc_text(U2M_DOC)
    i = u2m.index("## 20. E1: filed")
    base = u2m.count("\n", 0, i) + 1
    sec = u2m[i:]
    n_head, m_head = find_text(sec, r"## 20\. E1: filed ([0-9]+ [A-Za-z]+ [0-9]+)", base)
    n_keep, _ = find_text(sec, r"The pools keep\s+the package's NER", base)
    n_code, m_code = find_text(sec, r"as committed \((\w{7})\)", base)
    n_row, m_row = find_text(sec, r"\| the I: ALL \(rows\) \| ([\d.]+) \(([\d,.]+)\) \| ([\d.]+) \(([\d,.]+)\) \|", base)
    n_why, _ = find_text(sec, r"musique's I reads the ner family \(FULL:h2_c25\)", base)
    n_all3, _ = find_text(u2m, r"the models' compiled edges read all three families\s+on every dataset")
    n_i, _ = find_text(u2m, r"- I = B \u222a `m3b_pools\.expand_hops\(seeds, FULL, setting\)`")
    fh, fd = commit_of(U2M_DOC, "U2m E1 filed")
    vrel = "outputs/u2m/e1/verdict_e1.json"
    v = jload(vrel)
    mi = v["the_I"]["musique"]
    doc_nums = [float(m_row.group(1)), float(m_row.group(2).replace(",", "")), float(m_row.group(3)),
                float(m_row.group(4).replace(",", ""))]
    rec_nums = [mi["filed"]["ALL"], mi["filed"]["size"], mi["E1"]["ALL"], mi["E1"]["size"]]
    check("the E1 verdict record says NO_GAIN, and its musique I figures equal the doc's table",
          v["verdict"] == "NO_GAIN" and doc_nums == rec_nums, f"{vrel}: {rec_nums}; {U2M_DOC} line {n_row}")
    check("the E1 verdict record was committed with the filing", prov(vrel)["commit"].startswith(fh),
          f"{prov(vrel)['commit']} and {fh}")
    cc = m_code.group(1)
    cdate = git("log", "-n1", "--format=%ci", cc).strip()
    cfiles = git("show", "--name-only", "--format=", cc).split()
    check("the code commit the doc names changed outputs/mp_unified/zu2m_e1.py, and is older than the filing commit",
          "outputs/mp_unified/zu2m_e1.py" in cfiles and cdate < fd, f"{cc} {cdate}; filing {fh} {fd}")
    ul = doc_lines(U2L_DOC)
    n_u2l, _ = find(ul, r"\*\*Filed 25 October 2026: CHOSEN E1\*\* \(IS alone\)")
    n_u2l_eq, _ = find(ul, r"By gate 1, E1 \u222a R is W2_S array for array")
    n_u2l_r, _ = find(ul, r"\*\*with R\*\*: on a passage dataset R is the sibling links")
    ur = doc_lines(U2R_DOC)
    n_ur, _ = find(ur, r"^\*\*`CHOSEN E1`\. The claim read: `NER_MATCHED_NOT_ALL`")
    stale = [n for n, ln in enumerate(ur, 1) if re.search(r"no\s+NER\s+model", ln)]
    u2r_h, u2r_d = commit_of(U2R_DOC, "U2r filed")
    check("U2r's filing is older than U2m's section 20 filing", u2r_d < fd, f"{u2r_h} {u2r_d}; {fh} {fd}")
    nmd = doc_lines(NMD_DOC)
    n_n1, m_n1 = find(nmd, r"N1 lost 2wiki's pools by ([\d.]+), so E1 was chosen and CN_u is not in PKO's links")
    n_b1h, _ = find(doc_lines(B1H_DOC), r"\*\*Carve b1\.\*\*")
    n_b1l, _ = find(doc_lines(B1L_DOC), r"\*\*b1t:\*\* the setting's text links only \(the NER and KNN families")
    b1 = doc_lines(B1_DOC)
    n_b1s, _ = find(b1, r"- \*\*ner:\*\* rebuilt on the setting's passages by CRAG's `build_ner_edges` rule:")
    n_b1sp, _ = find(b1, r"spaCy en_core_web_sm, with the parser and lemmatizer disabled")
    n_all, _ = find(doc_lines(P1E_DOC), r"ALL, the share of questions with every gold in the pool")
    h = doc_lines(B1H_DOC)
    check("the B1H doc's carve b1 holds the NER and KNN graphs", "NER and KNN graphs" in " ".join(h[n_b1h - 1:n_b1h + 2]),
          f"{B1H_DOC} line {n_b1h}")
    check("the B1 doc's ner rule is the spaCy rule, two lines on", n_b1sp == n_b1s + 1, f"{B1_DOC} lines {n_b1s}, {n_b1sp}")
    sup = ""
    if stale:
        sup = (f" Superseded for the pools: `{U2R_DOC}` lines {' and '.join(map(str, stale))} describe the pools' "
               f"entity family as the title links alone; they date from {u2r_d[:10]}, and section 20 of the pool "
               f"doc ({fd[:10]}) is the later record.")
    text = (f"Why this spaCy row is in the final system. The pools keep the package's spaCy-built ner family. U2l "
            f"graded a spaCy-free entity family, E1 (IS alone; on a passage graph E1 together with R, the sibling "
            f"links (`{U2L_DOC}` line {n_u2l_r}), is W2_S array for array, line {n_u2l_eq}), and filed CHOSEN E1 "
            f"(line {n_u2l}); U2r graded name rules "
            f"beside it and kept E1 (`{U2R_DOC}` line {n_ur}, filed in {u2r_h}). The pool test with E1 in the ner slot "
            f"was NO_GAIN (`{U2M_DOC}` section 20, line {n_head}, filed in {fh} on {fd[:10]}, code at {cc}; record "
            f"`{vrel}`, `verdict`): on musique the I (the base and seeds plus the expansion over the three families, "
            f"line {n_i}) holds every gold (ALL, the share of questions with every gold in the pool, `{P1E_DOC}` "
            f"line {n_all}) for {rec_nums[0]} of the questions with the filed families ({rec_nums[1]:,} rows) and "
            f"for {rec_nums[2]} with E1's ({rec_nums[3]:,} rows) (doc line {n_row}; record `the_I.musique`). The doc "
            f"reads this as musique's I reading the ner family, whose links carry bridge entities that W2_S does not "
            f"reach (lines {n_why} to {n_why + 1}), and it states that the pools keep the package's NER (lines "
            f"{n_keep} to {n_keep + 1}). U2r's model-free name rule (N1) was not adopted either: it lost 2wiki's pools "
            f"by {m_n1.group(1)} (`{NMD_DOC}` line {n_n1}). So the models of record read the package's spaCy-built "
            f"ner family: in their pools (the models' compiled edges read all three families on every dataset, "
            f"`{U2M_DOC}` line {n_all3}) and in the B1 carves (carve b1 holds the NER and KNN graphs, `{B1H_DOC}` "
            f"line {n_b1h}; carve b1t is the NER and KNN families alone, `{B1L_DOC}` line {n_b1l}; the carve's ner "
            f"graph is rebuilt with spaCy, `{B1_DOC}` lines {n_b1s} to {n_b1sp}). {LABEL_W2S}, section 2, uses no "
            f"spaCy.{sup}")
    items = {"filed_families_ALL": item(rec_nums[0], "share of questions", "measured", vrel, "the_I.musique.filed.ALL"),
             "filed_families_rows": item(rec_nums[1], "rows", "measured", vrel, "the_I.musique.filed.size"),
             "e1_ALL": item(rec_nums[2], "share of questions", "measured", vrel, "the_I.musique.E1.ALL"),
             "e1_rows": item(rec_nums[3], "rows", "measured", vrel, "the_I.musique.E1.size"),
             "verdict": {"value": v["verdict"], "record": vrel, "key": "verdict", **prov(vrel)}}
    return {"text": text, "musique_I": items,
            "u2m_section_20": {"heading_line": n_head, "keep_lines": f"{n_keep} to {n_keep + 1}",
                               "reading_lines": f"{n_why} to {n_why + 1}", "filed_in": f"{fh} {fd}",
                               "code_commit": f"{cc} {cdate}", "table_row_line": n_row,
                               "i_definition_line": n_i, "models_read_all_three_families_line": n_all3},
            "u2l": {"chosen_line": n_u2l, "e1_is_w2s_line": n_u2l_eq},
            "u2r": {"commit": f"{u2r_h} {u2r_d}", "chosen_line": n_ur, "lines_superseded_for_the_pools": stale},
            "nmd": {"line": n_n1, "n1_lost_2wiki_pools_by": float(m_n1.group(1))},
            "b1": {"b1h_carve_line": n_b1h, "b1l_b1t_line": n_b1l, "b1_ner_lines": f"{n_b1s} to {n_b1sp}"}}


# -- the frozen sibling package's graph manifests: what they say of the original ner and knn builds -------------------

DUR = re.compile(r"(?i)(^|_)(secs?|seconds|duration|elapsed(_s)?|wall(_s)?)$")


def walk_items(o, pre=""):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from walk_items(v, f"{pre}.{k}" if pre else str(k))
    elif isinstance(o, list):
        for i, v in enumerate(o[:50]):
            yield from walk_items(v, f"{pre}[{i}]")
    else:
        yield pre, o


def sibling_section():
    if not (SIB / ".git").exists():
        return None
    out = {}
    for ds in CORPORA:
        rel = f"data/final_canonical/{ds}/graph/GRAPH_MANIFEST.json"
        p = SIB / rel
        data = p.read_bytes()
        doc = json.loads(data.decode("utf-8"))
        tracked = bool(run_git(SIB, "ls-files", "--", rel).strip())
        pv = {"tracked": tracked, "commit": "untracked", "in": "the frozen sibling package, read-only"}
        if tracked:
            pv["commit"] = run_git(SIB, "log", "-n1", "--format=%h %ci", "--", rel).decode().strip()
        note(f"sibling package: {rel}", data, pv)
        rec = {"record": f"CRAG package: {rel}", "provenance": pv}
        for fam in ("ner", "knn"):
            fd = doc["families"][fam]
            durs = [(k, v) for k, v in walk_items(fd) if DUR.search(k.rsplit(".", 1)[-1]) and isinstance(v, (int, float))]
            built = [(k, v) for k, v in walk_items(fd) if k.rsplit(".", 1)[-1] == "built"]
            rec[fam] = {"durations": [item(v, "s", "measured", f"CRAG package: {rel}", f"families.{fam}.{k}", pv)
                                      for k, v in durs],
                        "built_timestamps": [{"key": f"families.{fam}.{k}", "value": v} for k, v in built]}
            if fam == "knn":
                rec[fam]["method"] = (fd.get("source", {}).get("source_manifest", {}) or {}).get("method")
        out[ds] = rec
    return out


# -- P4a: the laptop's process-start name index ------------------------------------------------------------------------

def p4a_section():
    p4 = doc_lines(P4A_DOC)
    n_a, _ = find(p4, r"\*\*Index time\*\* \(recorded, not timed\)")
    n_b, _ = find(p4, r"once per corpus, like C12's")
    n_pin, _ = find(p4, r"the laptop, pinned to logical processor 2")
    n_t, _ = find(p4, r"one thread; the session idle")
    check("the doc's index-time paragraph is lines %d to %d" % (n_a, n_b), 0 < n_b - n_a <= 4, f"{P4A_DOC}")
    out = {}
    for ds in CORPORA:
        rel = f"outputs/p4a/{ds}.json"
        p = jload(rel)
        ix = p["index"]
        parts = [ix[k] for k in ("pools_file_s", "lists_and_pools_s", "name_only_s", "name_index_s", "buffers_s")]
        check(f"{ds}: P4a ran one thread, pinned to a logical processor", p["threads"] == 1 and p["pin"]["pinned"], rel)
        check(f"{ds}: the record's setup_total_s is at least each itemised part it lists", ix["setup_total_s"] >= max(parts),
              f"{ix['setup_total_s']} and {max(parts)}")
        out[ds] = {
            "setup_total_s": item(ix["setup_total_s"], "s", "measured", rel, "index.setup_total_s"),
            "name_only_s": item(ix["name_only_s"], "s", "measured", rel, "index.name_only_s"),
            "name_index_s": item(ix["name_index_s"], "s", "measured", rel, "index.name_index_s"),
            "other_itemised": {k: item(ix[k], "s", "measured", rel, f"index.{k}")
                               for k in ("pools_file_s", "lists_and_pools_s", "buffers_s")},
            "threads": item(p["threads"], "threads", "declared", rel, "threads"),
            "pinned_to_logical_processor": item(p["pin"]["affinity_lp"], "logical processor", "declared", rel,
                                                "pin.affinity_lp"),
            "run_utc": p["utc"]}
    meta = {"what": "the index time of the pass: the stores and both unions, the name index and the name-only mask, "
                    "the question texts, the walk buffers and FastPool's stamps, built once per corpus and recorded, "
                    "not part of the timed per-question passes",
            "doc_lines": f"{P4A_DOC} lines {n_a} to {n_b}; the laptop, its pin and one thread at lines {n_pin} and {n_t}",
            "total_is": "the record's own total is setup_total_s; name_index_s and name_only_s are two of its itemised "
                        "parts (the record also itemises pools_file_s, lists_and_pools_s and buffers_s); they are not "
                        "re-summed here"}
    return out, meta


# -- the hardware lines and the masking of names ---------------------------------------------------------------------------

def hardware_section():
    L = doc_lines(HOST_DOC)
    start = next(i for i, ln in enumerate(L) if ln.strip() == "## The host")
    i = start + 1
    while not L[i].startswith("|"):
        i += 1
    rows, first = {}, i
    while L[i].startswith("|"):
        m = re.match(r"\| (.+?) \| (.*) \|$", L[i])
        if m:
            rows[m.group(1).strip()] = (i + 1, m.group(2).strip())
        i += 1
    table = "\n".join(L[first:i])
    sens = set(re.findall(r"DESKTOP-[A-Za-z0-9]+", table)) | set(re.findall(r"\b\d{1,3}(?:\.\d{1,3}){3}\b", table))
    sens |= set(re.findall(r"(?:account|user) `([^`]+)`", table)) | set(re.findall(r"[A-Za-z]:\\Users\\([^\\`]+)", table))
    sens |= {getpass.getuser(), socket.gethostname(), Path.home().name, str(ROOT), str(Path.home())}
    sens = {s for s in sens if len(s) >= 5 and s.lower() not in ("user", "admin", "administrator")}
    nl, ml = find(L[i:], r"The laptop is the reference machine for every filed number: (.*?of memory)")
    nl += i

    def mask(text):
        t = re.sub(r"DESKTOP-[A-Za-z0-9]+", "<host name masked>", text)
        t = re.sub(r"\b\d{1,3}(?:\.\d{1,3}){3}\b", "<address masked>", t)
        t = re.sub(r"[A-Za-z]:\\[^\s`|]+", "<path masked>", t)
        for s in sorted(sens, key=len, reverse=True):
            t = re.sub(re.escape(s), "<name masked>", t, flags=re.I)
        return t.replace("`", "")

    def row(label, trim=None):
        n, text = rows[label]
        text = re.split(trim, text)[0] if trim else text
        return item(mask(text), "text", "declared", HOST_DOC, f"line {n} (row '{label}')", line=n)

    n_mach, t_mach = rows["machine"]
    shared = re.match(r"[^,]+, (.*)$", t_mach)
    host = {"cpu": row("CPU"), "memory": row("memory"), "gpu": row("GPU", r", about "),
            "scheduler_capacity": row("scheduler capacity")}
    if shared:
        host["machine"] = item(mask(shared.group(1)), "text", "declared", HOST_DOC,
                               f"line {n_mach} (row 'machine', the text after the host's name)", line=n_mach)
    laptop = item(mask(ml.group(1)), "text", "declared", HOST_DOC, f"line {nl}", line=nl)
    check("the host and laptop lines sit within lines 21 to 35 of the handover doc",
          all(21 <= v["line"] <= 35 for v in list(host.values()) + [laptop]),
          ", ".join(f"{k} line {v['line']}" for k, v in host.items()) + f", laptop line {nl}")
    return {"host": host, "laptop": laptop,
            "host_used_for": "A6's NER part jobs (one CPU reserved per job, no GPU named in their queue items) and, "
                             "beside, P2c part 2's parse parts",
            "laptop_used_for": "P4a's index setup (pinned, one thread)"}, mask, sens


# -- the gaps, each checked against the repository ------------------------------------------------------------------------

def gaps_section(link, facts, local_parts, sib):
    enc = doc_lines(ENC_DOC)
    n_no, m_no = find(enc, r"no shard status JSON in the package records a timing -- they carry rows, sha, dtype and "
                           r"dim only")
    n_dnm, _ = find(enc, r"DECLARED_NOT_MEASURED")
    n_enc, _ = find(doc_lines(B1_DOC), r"ours: dense \(gte-Qwen2-1\.5B, frozen\)")
    n_csr, _ = find(doc_lines(STEP4_DOC), r"\(`outputs/m3b/csr`, through `m3b_pools\.load_or_build_store`\)")
    pools = read_bytes(POOLS).decode("utf-8")
    fn = next(n for n in ast.walk(ast.parse(pools)) if isinstance(n, ast.FunctionDef) and n.name == "load_or_build_store")
    body = ast.get_source_segment(pools, fn)
    check("load_or_build_store builds a missing store and takes no timing",
          "if path.exists():" in body and "FamilyStore.from_graph(" in body and "store.save(path)" in body and
          not re.search(r"\btime\b|perf_counter|monotonic", body), f"{POOLS} lines {fn.lineno} to {fn.end_lineno}")
    stores = [f"outputs/m3b/csr/{ds}_{fam}.npz" for ds in CORPORA for fam in ("structural", "ner", "knn")]
    present = [s for s in stores if (ROOT / s).is_file()]
    check("the stores of the four corpora are present in this working tree and none is tracked",
          len(present) == len(stores) and not any(s in TRACKED for s in stores), f"{len(present)} of {len(stores)}")
    shard_files = {ds: sorted(p.name for p in (ROOT / f"outputs/u2b/{ds}/new").rglob("*") if p.is_file())
                   for ds in CORPORA}
    check("no per-shard stat or link record is kept in this repository, though the script writes one per shard",
          all(v == ["capstat.json"] for v in shard_files.values()) and facts["shard_json_written"],
          "; ".join(f"outputs/u2b/{ds}/new holds {', '.join(v)}" for ds, v in shard_files.items()))
    dense_keys = {}
    for ds in CORPORA:
        for f in ("dense_verified.json", "dense_reread.json"):
            dense_keys[f"{ds}/{f}"] = sorted(jload(f"outputs/u2b/{ds}/{f}").keys())
    key_sets = {tuple(v) for v in dense_keys.values()}
    check("the Modal dense stage's records hold no seconds key, and all eight hold the same keys",
          len(key_sets) == 1 and all(not DUR.search(k) for ks in key_sets for k in ks),
          f"{', '.join(next(iter(key_sets)))} in all {len(dense_keys)} files")
    no_w2 = [ds for ds in CORPORA if link[ds]["rule_own_build"] is None]
    have_w2 = [ds for ds in CORPORA if link[ds]["rule_own_build"] is not None]
    missing = []
    for ds in CORPORA:
        r = link[ds]
        for step in ("stat", "link"):
            if r[step]["peak_gb_max"]["value"] is None:
                missing.append(f"{ds} {step}")
        if r["build_all_variants"]["peak_gb"]["value"] is None:
            missing.append(f"{ds} build")
    # the sibling package's manifests
    if sib is None:
        orig = {"component": "the original ner and knn builds, every corpus",
                "missing": "no duration is in the records compiled here; the frozen sibling package was not present "
                           "on this machine when the script ran, so its manifests were not read",
                "evidence": "section 3 re-runs the ner family's extraction; it is not the original build, and nothing "
                            "re-runs the knn build"}
    else:
        ner_d = [ds for ds in CORPORA if sib[ds]["ner"]["durations"]]
        knn_d = {ds: sib[ds]["knn"]["durations"] for ds in CORPORA if sib[ds]["knn"]["durations"]}
        knn_b = [f"{ds} ({sib[ds]['knn']['built_timestamps'][0]['value']})" for ds in CORPORA
                 if sib[ds]["knn"]["built_timestamps"] and ds not in knn_d]
        none_k = [ds for ds in CORPORA if ds not in knn_d and not sib[ds]["knn"]["built_timestamps"]]
        meth = [sib[ds]["knn"].get("method") for ds in knn_d if sib[ds]["knn"].get("method")]
        orig = {"component": "the original ner and knn builds, every corpus",
                "missing": ("ner: no duration in any of the four corpora's graph manifests in the frozen sibling "
                            f"package{' (found for ' + ', '.join(ner_d) + ')' if ner_d else ''}. knn: a duration for "
                            + (", ".join(f"{ds} ({v[0]['value']} s, key {v[0]['key']})" for ds, v in knn_d.items()) or "none")
                            + "; " + ("a build timestamp and no duration for " + ", ".join(knn_b) if knn_b else "no timestamp-only manifest")
                            + ("; neither a duration nor a timestamp for " + ", ".join(none_k) if none_k else "")),
                "evidence": "read at run time from the sibling package's data/final_canonical/<corpus>/graph/"
                            "GRAPH_MANIFEST.json (untracked there; its data folder is git-ignored), by scanning the "
                            "ner and knn family entries for a seconds-like key. "
                            + ("That manifest's knn method line says: " + meth[0] + ". " if meth else "")
                            + "The package's other records and logs were not searched. Section 3 re-runs the ner "
                              "family's extraction; it is not the original build, and nothing re-runs the knn build"}
    return [
        {"component": "embeddings (the frozen dense encoder, gte-Qwen2-1.5B), every corpus",
         "missing": "no wall time of any encode run is in the records compiled here",
         "evidence": f"the encoder is named at {B1_DOC} line {n_enc}. {ENC_DOC} line {n_no} says "
                     f"'{m_no.group(0)}' (the doc's subject is one corpus; its encode rate is declared and marked "
                     f"DECLARED_NOT_MEASURED, line {n_dnm}, and none of its figures is used here). The package's own "
                     "logs, if they hold a timing, were not searched"},
        orig,
        {"component": "the CSR family stores (outputs/m3b/csr/<corpus>_<family>.npz)",
         "missing": "no build time of its own is recorded: a missing store is built by whatever job first opens it, "
                    "and no record read here separates that time from the job's",
         "evidence": f"`{POOLS}` lines {fn.lineno} to {fn.end_lineno}: load_or_build_store builds a missing store and "
                     "saves it, and takes no timing (checked at run time). The stores are read from outputs/m3b/csr "
                     f"through it (`{STEP4_DOC}` line {n_csr}); the {len(present)} stores of the four corpora (three "
                     "families each) are present in this working tree, untracked"},
        {"component": "Modal container-second totals (stat, link, build) and the merge stage",
         "missing": "only the slowest shard's seconds per stage are kept, not the sum over shard containers; the merge "
                    "stage's seconds are in no record; the time lost in failed or preempted attempts is in none either",
         "evidence": "the Modal script writes each shard's record to the Modal volume, and this repository holds the "
                     "merged capstat.json and the build record only (both checked at run time)"},
        {"component": "W2 alone, where there is no part record",
         "missing": "the build seconds of W2 for " + andj(no_w2) + " (no per-variant parts; the figures are for "
                    "all variants), and the stat and link seconds of W2 alone for every corpus (one pass serves all "
                    "variants)",
         "evidence": f"section 2; only {', '.join(have_w2)} has a W2 part record, and that part builds four graphs "
                     "together, so W2_S alone has no figure there either"},
        {"component": "peak memory of some Modal steps",
         "missing": "not recorded (the record holds 0 or null) for " + andj(missing),
         "evidence": "marked 'not recorded' in the rows of section 2"},
        {"component": "the Modal workers' CPU model",
         "missing": "not recorded; the result records hold neither CPU count nor memory, and the script's decorators "
                    "declare the CPU request and memory only",
         "evidence": "section 5"},
        {"component": "the Modal dense stage (the frozen vectors the rule's context choice reads, placed on the volume)",
         "missing": "no seconds: its records (dense_verified.json and dense_reread.json, per corpus) hold "
                    + ", ".join(next(iter(key_sets))) + " only",
         "evidence": "keys checked at run time in all eight files"},
        {"component": "A6's per-part files for " + andj(ds for ds in CORPORA if ds not in local_parts),
         "missing": "this working tree holds the A6 part file of " + andj(local_parts) + " only (a laptop run); "
                    "for the other corpora the host queue's job seconds are used, which include each job's own "
                    "start-up and any rerun by the retry wrapper",
         "evidence": "section 3"},
        {"component": "a per-corpus end-to-end index figure",
         "missing": "not recorded and not computed here, by design: the components have different hardware and "
                    "different stages",
         "evidence": "no total appears in this record"}]


# -- the report files ----------------------------------------------------------------------------------------------------

def pv(it):
    if not it["tracked"]:
        return "untracked" + (f" ({it['in']})" if it.get("in") else "")
    return f"tracked, `{it['commit']}`" + (" (working tree differs)" if it.get("modified_since_commit") else "")


def rk(it):
    return f"`{it['record']}` `{it['key']}`"


def num(x, nd=1):
    if x is None:
        return "not recorded"
    return f"{x:,.{nd}f}" if isinstance(x, float) else f"{x:,}"


def up(s):
    return s[:1].upper() + s[1:]


def andj(xs):
    """'a', 'a and b', 'a, b and c'."""
    xs = list(xs)
    return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1]


def cnt(n, word):
    return f"{n} {word}" + ("" if n == 1 else "s")


PATH_RE = re.compile(r"(?<![\w/`])((?:docs|outputs|src|scripts|data)/[A-Za-z0-9_<>/.\-]*[A-Za-z0-9_>])")


def code_paths(line):
    """Put every repository path of a markdown line into backticks (so '<corpus>' is not read as a tag); spans already in
    backticks are left alone."""
    parts = re.split(r"(`[^`]*`)", line)
    return "".join(p if p.startswith("`") else PATH_RE.sub(lambda m: f"`{m.group(1)}`", p) for p in parts)


def table(head, rows):
    return ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"] + \
           ["| " + " | ".join(r) + " |" for r in rows]


def shard_scope(n):
    return "the one shard container" if n == 1 else f"slowest of {n} shard containers"


def peak_cell(it):
    if it["value"] is None:
        held = "null" if it["record_holds"] is None else str(it["record_holds"])
        return f"not recorded (the record holds {held})"
    return num(it["value"], 2)


def md_report(R):
    C = R["corpora"]
    H = R["hardware"]["modal_containers"]
    X = R["w2s_notes"]
    variants = ", ".join(C["2wiki"]["w2s_rule_links"]["variants_all"])
    L = []
    add = L.append
    add(f"# {R['title']}")
    add("")
    add(f"Read on {R['read_on']} at HEAD {R['head_at_read']} by `{R['script']}` (`{R['command']}`). It reads every "
        "number below from its record at run time and lists the files it read (`report.json`, `files_read`); nothing "
        "was measured for this file. One row is one component of one corpus. There is no total across components, no "
        "per-passage rate and no extrapolation. The only arithmetic is a sum, a minimum or a maximum over the part "
        "records of one component of one corpus, marked (summed), (min) or (max); where a record holds its own number "
        "for the same thing, the script checks the two (section 8).")
    add("")
    add(f"The components are separate. Section 2, **{LABEL_W2S}**, builds the `structural` family. Section 3, "
        f"**{LABEL_NER}**, is the `ner` family, a different family of the package (`{U2C_DOC}` lines "
        f"{X['u2c_lines']}).")
    add("")
    add("**Kinds.** *measured*: a clock or memory reading the run wrote. *counted*: a tally in a record. *declared*: a "
        "setting or a description written in a record, a script or a doc (a request, not a measurement). No estimated "
        "number is used. **Provenance.** `tracked, <commit>` is `git log -n1 --format='%h %ci'` of the file; "
        "`untracked` files are not in git (`outputs/` is ignored, so these records exist on the compiling machine "
        "only; this file carries their values and `report.json` their sha256). Peak figures are the records' `peak_gb`: "
        "the process's peak resident set in binary GB (kB divided by 1,048,576).")
    add("")
    add("## 1. Passages per corpus")
    add("")
    rows = [[ds, num(C[ds]["passages"]["value"]), "counted", rk(C[ds]["passages"]), pv(C[ds]["passages"])]
            for ds in CORPORA]
    L += table(["corpus", "passages", "kind", "record and key", "provenance"], rows)
    add("")
    add("`n_nodes` is the node count, one node per passage. The sum of `passages` over the parse parts of section 6 "
        "(untracked) equals it for each corpus, and the link rule doc's own counts for hotpotqa and 2wiki equal it "
        "(section 8).")
    add("")
    add(f"## 2. {LABEL_W2S}")
    add("")
    add(f"W2_S is the chosen link rule of U2b (`{U2B_DOC}` line {X['chosen_line']}): variant W2, the plainest variant "
        f"(the look boundary alone, lines {X['plain_lines']}), in its S form, which adds the sibling links between "
        f"descriptive nodes that share a title (line {X['form_s_line']}). It reads title surfaces, a case gate, the "
        f"corpus's capitalization statistic, an ambiguity guard and a cosine context choice over the frozen dense "
        f"vectors (lines {X['rule_lines']}). No source file the Modal image adds mentions spaCy, and the image "
        f"installs {', '.join(H['pinned_packages'])} only (section 8). It ran in four stages (stat, merge, link, "
        f"build), one container per shard, on Modal CPU containers (line {X['stages_line']}). The result records "
        "hold no CPU count or memory limit; the CPU requests and memory below are the values declared in the Modal "
        "script's decorators, read from git (table at the end of this section). Line numbers in this section are "
        f"those of `{U2B_DOC}` unless another file is named.")
    add("")
    have = [ds for ds in CORPORA if C[ds]["w2s_rule_links"]["rule_own_build"]]
    lack = [ds for ds in CORPORA if ds not in have]
    add(f"**2a. The rule's own build.** A figure for W2 alone exists for {', '.join(have)} only: variant W2's part, "
        "the part that builds W2_S. It builds four graphs together (W2_M, W2_D, W2_S and W2_DS) in one container, so "
        f"W2_S has no seconds figure of its own. {andj(lack)} {'has' if len(lack) == 1 else 'have'} no per-variant "
        f"part (their build records have no `parts` key and there is no `part_new_*.json`; before {X['build_commit']} "
        f"one container built every variant's graphs, line {X['single_container_line']}); their figures are in 2b, "
        "labelled all variants.")
    add("")
    rows = []
    for ds in have:
        own = C[ds]["w2s_rule_links"]["rule_own_build"]
        rows.append([LABEL_W2S, ds, num(own["seconds"]["value"]), num(own["peak_gb"]["value"], 2),
                     ", ".join(own["graphs"]), f"{own['cpus']['value']}", memtxt(own["memory"]["value"]),
                     f"{rk(own['seconds'])} and `peak_gb`", pv(own["seconds"])])
    L += table(["component", "corpus", "seconds", "peak GB", "graphs in the container", "CPU request (declared)",
                "memory as written (declared)", "record and key", "provenance"], rows)
    add("")
    w = C[have[0]]["w2s_rule_links"]["rule_own_build"]
    unit = H["memory_unit"]
    add(f"*measured*: seconds and peak. *declared*: the CPU request and memory, read from the `build_part` decorator "
        f"(line {w['cpus']['decorator_line']}) of `{MODAL}` at {w['cpus']['read_at_commit']}, the commit the doc "
        f"names for this build. The memory argument is (request, limit) in "
        f"{unit['unit'] or 'a unit not checked here'}"
        + (f" (the installed modal {unit['modal_version']}'s own comment; the version that ran is not recorded)"
           if unit.get("unit") else "") +
        ". The result records carry the rule code's hash (`zu2b_sha256`, that of `zu2b.py`), not the Modal script's, "
        f"so the script version that ran is stated only for the {have[0]} build.")
    add("")
    add(f"**2b. Steps whose records cover all variants.** All variants: {variants} "
        f"({C['2wiki']['w2s_rule_links']['graphs_in_build_record']} graphs: each variant in the forms "
        f"{', '.join(C['2wiki']['w2s_rule_links']['graph_forms'])}). These are not the rule's own build.")
    add("")
    allv = f"all variants ({variants})"
    rows = []
    for ds in CORPORA:
        r = C[ds]["w2s_rule_links"]
        s = r["stat"]
        rows.append([LABEL_W2S, ds, f"stat, one pass shared by {allv}", shard_scope(s["shards"]["value"]),
                     num(s["seconds_max_over_shards"]["value"]), peak_cell(s["peak_gb_max"]),
                     f"{s['cpus']['value']}", rk(s["seconds_max_over_shards"]), pv(s["seconds_max_over_shards"])])
        s = r["link"]
        rows.append([LABEL_W2S, ds, f"link, one pass over {allv}", shard_scope(s["shards"]["value"]),
                     num(s["seconds_max_over_shards"]["value"]), peak_cell(s["peak_gb_max"]),
                     f"{s['cpus']['value']}", rk(s["seconds_max_over_shards"]), pv(s["seconds_max_over_shards"])])
        a = r["build_all_variants"]
        cpus = (f"parts {a['cpus']['parts']['value']}, build container {a['cpus']['orchestrator']['value']}"
                if "parts" in a["cpus"] else f"{a['cpus']['value']}")
        rows.append([LABEL_W2S, ds, f"build, {allv}", a["layout"], num(a["seconds"]["value"]),
                     peak_cell(a["peak_gb"]), cpus, rk(a["seconds"]), pv(a["seconds"])])
    L += table(["component", "corpus", "step", "scope", "seconds", "peak GB", "CPU request (declared)",
                "record and key", "provenance"], rows)
    add("")
    n_before = list(H["by_commit"]).index(X["build_commit"])
    add("*measured*: seconds and peak GB (for stat and link, the largest over the shard containers); *declared*: the "
        "CPU request, one number where every committed version that could have run the step declares the same (stat "
        f"and link: all {len(H['by_commit'])} versions; the single-container build: the {n_before} versions before "
        f"{X['build_commit']}). The memory arguments changed between versions and the stat, link and "
        f"build records do not name the version that ran, so no memory is stated for these rows; the {have[0]} "
        "build's (section 2a) is stated because the doc names its version.")
    add("")
    add(f"- **What the seconds are.** Wall seconds in the container, not CPU seconds: from the function's "
        f"`t0 = time.time()` (after the volume reload; for link, also after the selftest) to the write of its record "
        f"(for `build_part`, after the container is up and the code is imported, and including the read of the link "
        f"shards; for `build`, including the start of every part and the hash check of every graph file; checked, "
        f"section 8). They are those of the attempt that wrote the record: stage inputs are retried (up to "
        f"{H['retries_per_stage_input']} times in the script's `STAGE`) and a retry returns an existing record "
        f"without redoing the work (`{U2B_DOC}` line {X['retry_line']}); 2wiki's earlier build was preempted four "
        f"times (line {X['preempt_line']}). The time lost in failed attempts is not in these records.")
    add(f"- **What the peak is.** The container process's peak resident set, as the record writes it (`_peak_gb`: "
        f"`VmHWM`, with a `getrusage` fallback from {X['first_rusage']}). 'Not recorded' means the record holds 0 or "
        "null; the stat records hold 0 for every corpus.")
    add("")
    add("**Which link pass.**")
    add("")
    rows = []
    for ds in CORPORA:
        lk = C[ds]["w2s_rule_links"]["link"]
        rows.append([ds, str(C[ds]["w2s_rule_links"]["stat"]["shards"]["value"]), str(lk["shards"]["value"]),
                     lk["link_pass"]["value"] + (" (inferred)" if lk["link_pass"]["kind"] == "declared" else ""),
                     lk["link_pass"]["basis"]])
    L += table(["corpus", "stat shards", "link shards", "link pass", "basis"], rows)
    add("")
    unsettled = [ds for ds in CORPORA if C[ds]["w2s_rule_links"]["link"]["link_pass"]["kind"] == "not recorded"]
    add(f"The link stage runs `zu2b_fast.link_pass`, which gives exactly the edges and statistics of the earlier pass, "
        f"faster (`{U2B_DOC}` lines {X['fast_pass_lines']}), so a link figure from the earlier pass and one from the "
        f"fast pass are not the same cost. The script calls it, and has a link shard count of its own, from "
        f"{X['first_sep']} on (that commit's subject: '{X['first_sep_subject']}'). The records of "
        f"{andj(unsettled)} do not settle which pass ran: the build record names none; one link shard equals the stat "
        f"shard count, which the script before {X['first_sep']} always gave (one link container per stat shard) and "
        f"the script from {X['first_sep']} on gives when its link shard count is left unset (`kl = k_link or k` in "
        f"its `corpus` function); and their empty link and build peaks agree with the script before "
        f"{X['first_rusage']}, whose peak reader has no `getrusage` fallback, but do not prove it (section 8).")
    add("")
    commits = list(H["by_commit"])
    rows = []
    for fn in STAGE_FUNCS:
        cells = []
        for c in commits:
            f = H["by_commit"][c]["functions"].get(fn)
            cells.append("-" if f is None else f"cpu={f['cpu']}, memory={memtxt(f['memory'])}" +
                         (", nonpreemptible" if f.get("nonpreemptible") else ""))
        rows.append([f"`{fn}`"] + cells)
    L += table(["declared in the Modal script"] + [f"{c} ({H['by_commit'][c]['date'][:16]})" for c in commits], rows)
    add("")
    add(f"Declared in `{MODAL}` ({pv(H)}). `memory` is one number or a (request, limit) pair, as written in the "
        f"decorator. No function of any committed version passes a GPU argument; "
        f"{'one input per container' if H['one_input_per_container'] else 'containers take several inputs'}; "
        f"{H['image']}; the image sets "
        f"{', '.join(f'{k}={v}' for k, v in H['threads_set_in_the_image'].items())}. The CPU model of the workers is "
        "not recorded.")
    add("")
    add(f"## 3. {LABEL_NER}")
    add("")
    meta = R["ner_meta"]
    add(meta["pools_note"]["text"])
    add("")
    st = meta["settings"]
    add(f"What the row is: {NER_DESC}. Hardware: the host (section 5), one CPU reserved per part "
        f"({meta['reserved_per_part']}; also the queue items' own field, *declared*). {up(meta['what_the_seconds_are'])}. "
        f"Peak memory is {meta['peak_is']}. Settings: spaCy {st['spacy']}, en_core_web_sm {st['model_version']}, "
        f"{', '.join(st['disabled'])} disabled, {st['labels']} labels, at least {st['min_chars']} characters, key "
        f"`{st['key']}`, document frequency {st['document_frequency_range'][0]} to {st['document_frequency_range'][1]}, "
        f"batch size {st['batch_size']}. These are the ner family's settings: the doc says so "
        f"({meta['same_settings_as_the_ner_family']}), the script checks them against `ner_edges` in the package's "
        f"build script ({meta['settings_checked_at']}), and {meta['faithful_rerun']}. {meta['plan_figure_not_used']}.")
    add("")
    rows = []
    for ds in CORPORA:
        n = C[ds]["ner_family_rerun"]
        rows.append([LABEL_NER, ds, num(n["parts"]["value"]), num(n["job_seconds_sum"]["value"]) + " (summed)",
                     f"{num(n['job_seconds_min']['value'])} (min), {num(n['job_seconds_max']['value'])} (max)",
                     num(n["peak_gb_max"]["value"], 2) + " (max)",
                     f"{num(n['merge_job']['seconds']['value'])} s, {num(n['merge_job']['peak_gb']['value'], 2)} GB",
                     rk(n["job_seconds_sum"]), pv(n["job_seconds_sum"])])
    L += table(["component", "corpus", "parts", "CPU-seconds (job wall seconds, 1 CPU reserved per job)", "per part",
                "peak GB", "merge job (one per corpus)", "record and key", "provenance"], rows)
    add("")
    sq = C["squad"]["ner_family_rerun"]["laptop_part"]["seconds"]
    add(f"All *measured*, except the part counts (*counted*). The same squad extraction also ran as one part on the "
        f"laptop: {num(sq['value'])} s ({rk(sq)}, {pv(sq)}, run {sq['utc']}); the host job above is a separate run. "
        f"Each part has one job, all done with return code 0, and none is in the host queue's retried, dropped or shed "
        f"lists (section 8), so the queue state shows no failed attempt of these jobs. Semantics read at "
        f"{meta['semantics_read_at']}.")
    add("")
    add("## 4. The laptop name index (P4a, one thread)")
    add("")
    pm = R["p4a_meta"]
    add(f"Hardware: the laptop (section 5), pinned to a logical processor, one thread. {up(pm['what'])} "
        f"({pm['doc_lines']}). {up(pm['total_is'])}.")
    add("")
    rows = []
    for ds in CORPORA:
        p = C[ds]["name_index_p4a"]
        rows.append([ds, num(p["setup_total_s"]["value"], 2), num(p["name_only_s"]["value"], 2),
                     num(p["name_index_s"]["value"], 2),
                     f"1 thread, logical processor {p['pinned_to_logical_processor']['value']}",
                     rk(p["setup_total_s"]), pv(p["setup_total_s"])])
    L += table(["corpus", "setup_total_s", "name_only_s (the name-only mask)", "name_index_s (the name index)", "run",
                "record and key", "provenance"], rows)
    add("")
    add("*measured* (seconds, rounded to two decimals here; `report.json` holds the record's values); the thread and "
        "pin settings are *declared*.")
    add("")
    add("## 5. Hardware lines")
    add("")
    hw = R["hardware"]
    for label, key in (("CPU", "cpu"), ("memory", "memory"), ("GPU", "gpu"), ("scheduler capacity", "scheduler_capacity"),
                       ("machine", "machine")):
        if key in hw["host"]:
            it = hw["host"][key]
            add(f"- Host, {label}: {it['value']} (`{it['record']}` {it['key']}; {pv(it)}; *declared*)")
    add(f"- Host, used for: {hw['host_used_for']}.")
    add(f"- Laptop: {hw['laptop']['value']} (`{hw['laptop']['record']}` {hw['laptop']['key']}; {pv(hw['laptop'])}; "
        f"*declared*). Used for: {hw['laptop_used_for']}.")
    add(f"- Modal CPU containers: one container per shard (`{U2B_DOC}` line {X['stages_line']}); the CPU request and "
        f"memory are the declared values of section 2, and no result record states them; {H['seconds_are']}; the "
        f"CPU model of the workers is {H['cpu_model_of_the_workers']}.")
    add("")
    add("## 6. Beside, not in the final system: P2c part 2's parse")
    add("")
    add(f"{R['beside']['why_beside']} Hardware: the host (the part files' `where` says `host`). *measured*, *summed* "
        "over the part files; the peak is the maximum over parts.")
    add("")
    rows = []
    for ds in CORPORA:
        b = R["beside"]["spacy_parse"][ds]
        rows.append([ds, num(b["parts"]["value"]), num(b["seconds_sum"]["value"]) + " (summed)",
                     f"{num(b['seconds_min']['value'])} (min), {num(b['seconds_max']['value'])} (max)",
                     num(b["peak_gb_max"]["value"], 2) + " (max)", rk(b["seconds_sum"]), pv(b["seconds_sum"])])
    L += table(["corpus", "parts", "seconds", "per part", "peak GB", "record and key", "provenance"], rows)
    add("")
    add("## 7. The gaps")
    add("")
    for g in R["gaps"]:
        add(f"- **{g['component']}**: {g['missing'].rstrip('.')}. {up(g['evidence']).rstrip('.')}.")
    add("")
    add("## 8. What the script checked")
    add("")
    for c in R["checks"]:
        add(f"- {'ok' if c['ok'] else 'FAILED'}: {c['check']} ({c['detail']})")
    add("")
    return "\n".join(code_paths(x) for x in L)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    caps = {ds: jload(f"outputs/u2b/{ds}/new/capstat.json") for ds in CORPORA}
    builds = {ds: jload(f"outputs/u2b/{ds}/graphs/build_new.json") for ds in CORPORA}
    parse = parse_section()
    passages = passages_section(caps, parse)
    commits, specs, dates, subjects, sources, gpu_any = modal_history()
    facts = modal_facts(commits, sources)
    link, bc, cites = w2s_section(caps, builds, commits, specs, subjects, sources, facts, gpu_any)
    ner, ner_meta = ner_section(caps)
    ner_meta["pools_note"] = pools_note()
    sib = sibling_section()
    p4a, p4a_meta = p4a_section()
    hw, mask, sens = hardware_section()
    hw["modal_containers"] = modal_hardware(commits, specs, dates, facts, bc, modal_memory_unit())
    gaps = gaps_section(link, facts, ner_meta["local_part_files"], sib)
    p2c = doc_lines(P2C_DOC)
    n_pipe, _ = find(p2c, r"spaCy 3\.8\.15 with en_core_web_sm 3\.8\.0: a CNN pipeline \(tagger, dependency parser")
    p2c2 = doc_lines(P2C2_DOC)
    n_ng, m_ng = find(p2c2, r"NO_GAIN: the fourth family is not adopted, and no full run follows")
    n_fam, _ = find(p2c2, r"\*\*The parse\*\* is part 1's version 1, run on the host over every passage")
    u2c = doc_text(U2C_DOC)
    n_struct, _ = find_text(u2c, r"the `structural` family is U2b's chosen graph")
    n_unch, _ = find_text(u2c, r"ner and knn are\s+unchanged on every dataset")
    now = datetime.now().astimezone()
    R = {
        "title": "The cost of the LLM-free link build: measured index-build records",
        "read_on": now.strftime("%Y-%m-%d %H:%M %z"),
        "head_at_read": git("log", "-n1", "--format=%h %ci").strip(),
        "script": "outputs/idxcost/compile_costs.py",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "command": "PYTHONIOENCODING=utf-8 python -I outputs/idxcost/compile_costs.py",
        "components": {"w2s_rule_links": LABEL_W2S, "ner_family": LABEL_NER,
                       "name_index_p4a": "the laptop name index (P4a, one thread)",
                       "beside": "not in the final system, beside: P2c part 2's parse"},
        "rules": ["per component and per corpus; no total across components; no per-passage rate; no extrapolation",
                  "the only arithmetic is a sum, minimum or maximum over the part records of one component of one "
                  "corpus, marked in the item (arithmetic, over_records)",
                  "every item carries its record, key, tracked or untracked, the file's last commit and its kind"],
        "kinds": {"measured": "a clock or memory reading the run wrote",
                  "counted": "a tally in a record",
                  "declared": "a setting or description written in a record, a script or a doc; not a measurement",
                  "not recorded": "the record holds nothing (or 0 or null) for this"},
        "corpora": {ds: {"passages": passages[ds], "w2s_rule_links": link[ds], "ner_family_rerun": ner[ds],
                         "name_index_p4a": p4a[ds]} for ds in CORPORA},
        "w2s_notes": {**cites, "first_rusage": facts["first_rusage"],
                      "u2c_lines": f"{n_struct} to {n_unch}"},
        "modal_script_facts": facts,
        "ner_meta": ner_meta,
        "p4a_meta": p4a_meta,
        "sibling_package_graph_manifests": sib,
        "hardware": hw,
        "beside": {"why_beside": f"Not in the final system: `{P2C2_DOC}` line {n_ng} says '{m_ng.group(0)}'. The "
                                 f"fourth family's parse is part 1's version 1, run on the host over every passage "
                                 f"(`{P2C2_DOC}` line {n_fam}), with the full spaCy pipeline including its dependency "
                                 f"parser (`{P2C_DOC}` line {n_pipe}).",
                   "spacy_parse": parse},
        "gaps": gaps,
        "checks": CHECKS,
        "files_read": READ,
        "file_groups_read": GROUPS}
    failed = [c for c in CHECKS if not c["ok"]]
    for c in CHECKS:
        print(("ok     " if c["ok"] else "FAILED ") + c["check"] + "  [" + c["detail"] + "]")
    if failed:
        raise SystemExit(f"{len(failed)} check(s) failed; nothing written")
    js = json.dumps(R, indent=1, ensure_ascii=False) + "\n"
    md = md_report(R)
    for text in (js, md):
        low = text.lower()
        for s in sens:
            if s.lower() in low:
                raise SystemExit("a masked name is still in the output; nothing written")
    (HERE / "report.json").write_text(js, encoding="utf-8", newline="\n")
    (HERE / "report.md").write_text(md, encoding="utf-8", newline="\n")
    print("files read:")
    for label, r in READ.items():
        print(f"  {label}  [{pv(r)}]")
    for label, g in GROUPS.items():
        print(f"  {label}  [{g['files'] if 'files' in g else len(g['commits'])} files or versions]")
    print(f"filed report.json and report.md ({len(CHECKS)} checks ok)")


if __name__ == "__main__":
    main()
