"""Archive the host's rx workspace outputs to private Hugging Face dataset repos, and restore them onto a host, with no
credential on the host (docs/GPU_HOST_HANDOVER.md: the lab machine is shared). Systems only: it moves and verifies bytes.

The token holder (the laptop) asks the hub for each object's signed upload links; the host streams the bytes to those links
and finishes each object only when the bytes it sent hash to the planned sha256; the laptop then commits the objects as
LFS files at ws/<rel> and checks the repo's tree. A restore is the reverse: the laptop resolves signed download links,
the host downloads and hashes.

Laptop modes (a token the user stored with `hf auth login`, chosen by --token-name, else the active one; it is never
printed, logged or written, and it is sent only to huggingface.co):
  plan     the host's manifest of --include/--exclude (rx's manifest op: size, mtime and sha256, hashed on the host), then
           the repo's upload links for every object it does not hold -> outputs/host_archive/wave_<tag>.json (links only,
           host-bound) and plan_<tag>.json (no links)
  commit   the wave's uploaded and already-held objects as LFS files at ws/<rel>, in commits of --per-commit, then
           manifests/<tag>.json; the tree is re-read and every file checked by size and sha256 -> commit_<tag>.json
  links    a restore's signed download links for manifests/<tag>.json (or --manifest-from a local file), the redirect
           followed by hand -> outputs/host_archive/links_<tag>.json (host-bound, short-lived)
  status   each manifest in the repo against its tree
  drive    plan, push, the host's upload as an rx job, fetch, commit, for one tag
  drive-restore  links, the host's restore as an rx job (into --dest, '.' = the workspace itself), fetch its record
Host modes (no token of any kind: HF_TOKEN is removed and huggingface_hub is never imported):
  upload   each planned file streamed to its links, hashed as sent -> outputs/host_archive/done_<tag>.json
  restore  each linked file into --dest/<rel> via <rel>.part, hashed as written -> outputs/host_archive/restore_<tag>.json;
           a file already there with other bytes is a conflict and is left alone unless --overwrite
"""
from __future__ import annotations

import argparse
import base64
import fnmatch
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "host_archive"
REPO_PREFIX = "ws/"
LFS_HEADERS = {"Accept": "application/vnd.git-lfs+json", "Content-Type": "application/vnd.git-lfs+json"}
HUB = "https://huggingface.co"
CHUNK = 8 * (1 << 20)
BATCH = 100
NEVER = ("outputs/host_archive/**",)      # this tool's own files are never archived


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def atomic_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def check_tag(tag: str) -> str:
    if not tag or not all(c.isalnum() or c in "-_." for c in tag):
        raise SystemExit(f"bad tag {tag!r}: letters, digits, '-', '_' and '.' only")
    return tag


def safe_rel(rel: str) -> str:
    parts = rel.split("/")
    if rel.startswith("/") or ":" in rel or "\\" in rel or any(p in ("", ".", "..") for p in parts):
        raise SystemExit(f"unsafe path {rel!r}")
    return rel


# ── laptop ───────────────────────────────────────────────────────────────────


def stored_token(name: str | None) -> str | None:
    """The token the user stored under this name with `hf auth login`, or None for the active one. It is handed only to
    huggingface_hub and never printed, logged or written."""
    if name is None:
        return None
    from huggingface_hub.utils import get_stored_tokens
    tokens = get_stored_tokens()
    if name not in tokens:
        raise SystemExit(f"no stored token named {name!r} (see `hf auth list`)")
    return tokens[name]


def hf_session(token_name: str | None):
    """(session, headers): the headers carry the token and are sent to huggingface.co only."""
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    from huggingface_hub.utils import build_hf_headers, get_session
    return get_session(), build_hf_headers(token=stored_token(token_name))


def hf_api(token_name: str | None):
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    from huggingface_hub import HfApi
    return HfApi(token=stored_token(token_name))


def private_repo(api, repo: str) -> None:
    api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
    if api.repo_info(repo, repo_type="dataset").private is not True:
        raise SystemExit(f"REFUSING: {repo} is not private")


def raise_for(r, what: str) -> None:
    if r.status_code >= 400:
        body = r.text[:300] if r.text else ""
        raise SystemExit(f"{what}: HTTP {r.status_code} {body}")


def host_manifest(include: list[str], exclude: list[str]) -> list[dict]:
    """rx's own manifest op on the host workspace: [rel, bytes, mtime_ns, sha256], hashed on the host."""
    sys.path.insert(0, str(ROOT / "tools" / "rx"))
    import rx as RX
    proj = RX.Project(RX.find_project(str(ROOT)))
    remote = RX.Remote(RX.load_host(proj.host))
    r = remote.call({"op": "manifest", "project": proj.name, "ws": "ws", "include": include,
                     "exclude": list(exclude) + list(NEVER)}, retry_for=600)
    if r.get("truncated"):
        raise SystemExit("the host manifest was truncated; narrow --include")
    rows = [{"rel": safe_rel(rel), "bytes": size, "mtime_ns": mt, "sha256": sha} for rel, size, mt, sha in r["files"]]
    bad = [x["rel"] for x in rows if not x["sha256"] or len(x["sha256"]) != 64]
    if bad:
        raise SystemExit(f"{len(bad)} files came without a sha256 (first {bad[:3]})")
    return rows


def parse_action(obj: dict) -> dict:
    """One LFS batch object -> {'present': True} or the host-bound upload plan (links only, no header beyond the parts)."""
    if obj.get("error"):
        raise SystemExit(f"LFS batch error for {obj.get('oid')}: {obj['error']}")
    acts = obj.get("actions") or {}
    if "upload" not in acts:
        return {"present": True}
    up = acts["upload"]
    hdr = up.get("header") or {}
    if any(k.lower() == "authorization" for k in hdr):
        raise SystemExit("an upload action carries an Authorization header; refusing to send it to the host")
    href = up["href"]
    if "chunk_size" in hdr:
        parts = [hdr[k] for k in sorted((k for k in hdr if k.isdigit()), key=int)]
        cs = int(hdr["chunk_size"])
        if not parts or cs <= 0 or (len(parts) - 1) * cs >= int(obj["size"]) or len(parts) * cs < int(obj["size"]):
            raise SystemExit(f"multipart plan does not cover {obj.get('oid')}: {len(parts)} parts of {cs}")
        out = {"href": href, "chunk_size": cs, "parts": parts}
    else:
        out = {"href": href}
    if "verify" in acts:
        out["verify_href"] = acts["verify"]["href"]  # called by the token holder after the upload, never by the host
    out["expires_in"] = up.get("expires_in")
    return out


def lfs_batch(session, headers, repo: str, rows: list[dict]) -> list[dict]:
    r = session.post(f"{HUB}/datasets/{repo}.git/info/lfs/objects/batch", headers={**LFS_HEADERS, **headers},
                     json={"operation": "upload", "transfers": ["basic", "multipart"], "hash_algo": "sha256",
                           "ref": {"name": "main"}, "objects": [{"oid": x["sha256"], "size": x["bytes"]} for x in rows]},
                     timeout=300)
    raise_for(r, "LFS batch")
    objs = {o["oid"]: o for o in r.json()["objects"]}
    return [parse_action(objs[x["sha256"]]) for x in rows]


def stage_plan(repo: str, tag: str, include: list[str], exclude: list[str], token_name: str | None) -> dict:
    api = hf_api(token_name)
    private_repo(api, repo)
    session, headers = hf_session(token_name)
    t0 = time.time()
    rows = host_manifest(include, exclude)
    if not rows:
        raise SystemExit("nothing matched")
    print(f"plan {tag}: host manifest {len(rows)} files, {sum(x['bytes'] for x in rows) / 1e9:.2f} GB ({time.time() - t0:.0f}s)",
          flush=True)
    owner, uniq = {}, []
    for x in rows:                                   # an object two paths share is uploaded once, by the first path
        if x["sha256"] not in owner:
            owner[x["sha256"]] = x["rel"]
            uniq.append(x)
    acts = {}
    for i in range(0, len(uniq), BATCH):
        chunk = uniq[i:i + BATCH]
        for x, a in zip(chunk, lfs_batch(session, headers, repo, chunk)):
            acts[x["sha256"]] = a
    wave, plan = [], []
    for x in rows:
        a = acts[x["sha256"]]
        if a.get("present"):
            plan.append({**x, "upload": False})
        elif owner[x["sha256"]] == x["rel"]:
            plan.append({**x, "upload": True, **({"verify_href": a["verify_href"]} if "verify_href" in a else {})})
            wave.append({**x, **{k: a[k] for k in ("href", "chunk_size", "parts") if k in a}})
        else:
            plan.append({**x, "upload": False, "dup_of": owner[x["sha256"]]})
    acts = list(acts.values())
    lifetimes = [a.get("expires_in") for a in acts if a.get("expires_in")]
    common = {"repo": repo, "tag": tag, "include": include, "exclude": exclude, "generated_utc": utc(),
              "generated_epoch": int(time.time()), "link_lifetime_s": min(lifetimes) if lifetimes else None}
    atomic_json(OUT / f"wave_{tag}.json", {**common, "files": wave})
    atomic_json(OUT / f"plan_{tag}.json", {**common, "files": plan})
    n_up = sum(p["upload"] for p in plan)
    print(f"plan {tag}: {n_up} to upload ({sum(p['bytes'] for p in plan if p['upload']) / 1e9:.2f} GB), "
          f"{len(plan) - n_up} already held; links live {common['link_lifetime_s']} s", flush=True)
    return common


def commit_lines(entries: list[dict], summary: str, regular: dict[str, bytes] | None = None) -> bytes:
    lines = [{"key": "header", "value": {"summary": summary, "description": ""}}]
    for e in entries:
        lines.append({"key": "lfsFile", "value": {"path": REPO_PREFIX + safe_rel(e["rel"]), "algo": "sha256",
                                                  "oid": e["sha256"], "size": e["bytes"]}})
    for path, content in (regular or {}).items():
        lines.append({"key": "file", "value": {"content": base64.b64encode(content).decode("ascii"), "path": path,
                                               "encoding": "base64"}})
    return "\n".join(json.dumps(x) for x in lines).encode("utf-8")


def post_commit(session, headers, repo: str, body: bytes, tries: int = 6) -> str:
    for attempt in range(tries):
        r = session.post(f"{HUB}/api/datasets/{repo}/commit/main", headers={**headers, "Content-Type": "application/x-ndjson"},
                         data=body, timeout=600)
        if r.status_code < 400:
            return r.json().get("commitOid", "")
        if r.status_code in (429, 500, 502, 503, 504) and attempt + 1 < tries:
            time.sleep(min(600, 30 * 2 ** attempt))
            continue
        raise_for(r, "commit")
    raise SystemExit("commit: out of retries")


def repo_tree(api, repo: str, prefix: str = "") -> dict:
    from huggingface_hub.errors import EntryNotFoundError
    out = {}
    try:
        for e in api.list_repo_tree(repo, repo_type="dataset", recursive=True, path_in_repo=prefix or None):
            if hasattr(e, "size"):
                lfs = getattr(e, "lfs", None)
                out[e.path] = (e.size, getattr(lfs, "sha256", None) if lfs else None)
    except EntryNotFoundError:                       # the prefix holds no file yet (a first commit)
        return {}
    return out


def stage_commit(repo: str, tag: str, token_name: str | None, per_commit: int) -> dict:
    plan = read_json(OUT / f"plan_{tag}.json")
    if plan["repo"] != repo:
        raise SystemExit(f"plan_{tag} is for {plan['repo']}, not {repo}")
    done_p = OUT / f"done_{tag}.json"
    done = {d["rel"]: d for d in read_json(done_p)["files"]} if done_p.exists() else {}
    session, headers = hf_session(token_name)
    api = hf_api(token_name)
    sent = {}
    for p in plan["files"]:
        if p["upload"]:
            d = done.get(p["rel"])
            sent[p["rel"]] = bool(d and d["status"] == "uploaded" and d["sha256"] == p["sha256"] and d["bytes"] == p["bytes"])
            if sent[p["rel"]] and p.get("verify_href"):
                vr = session.post(p["verify_href"], headers={**LFS_HEADERS, **headers},
                                  json={"oid": p["sha256"], "size": p["bytes"]}, timeout=120)
                raise_for(vr, f"verify {p['rel']}")
    ready, missing = [], []
    for p in plan["files"]:
        ok = sent[p["rel"]] if p["upload"] else sent.get(p["dup_of"], False) if p.get("dup_of") else True
        if ok:
            ready.append(p)
        else:
            d = done.get(p.get("dup_of") or p["rel"]) or {}
            missing.append([p["rel"], d.get("status", "not uploaded"), d.get("error")])
    tree = repo_tree(api, repo, REPO_PREFIX.rstrip("/")) if ready else {}
    todo = [p for p in ready if tree.get(REPO_PREFIX + p["rel"]) != (p["bytes"], p["sha256"])]
    commits = []
    for i in range(0, len(todo), per_commit):
        part = todo[i:i + per_commit]
        oid = post_commit(session, headers, repo, commit_lines(part, f"archive {tag}: {i + len(part)}/{len(todo)} "
                                                                     f"files of the host workspace"))
        commits.append(oid)
        print(f"   commit {len(commits)}: {len(part)} files -> {oid[:12]}", flush=True)
    man = {"tag": tag, "repo": repo, "created_utc": utc(), "prefix": REPO_PREFIX, "include": plan["include"],
           "exclude": plan["exclude"], "files": [[p["rel"], p["bytes"], p["sha256"]] for p in ready],
           "bytes": sum(p["bytes"] for p in ready), "not_archived": missing}
    commits.append(post_commit(session, headers, repo, commit_lines(
        [], f"archive {tag}: manifest", {f"manifests/{tag}.json": json.dumps(man, indent=0).encode("utf-8")})))
    tree = repo_tree(api, repo, REPO_PREFIX.rstrip("/"))
    bad = [p["rel"] for p in ready if tree.get(REPO_PREFIX + p["rel"]) != (p["bytes"], p["sha256"])]
    rec = {"tag": tag, "repo": repo, "utc": utc(), "files": len(plan["files"]), "archived": len(ready) - len(bad),
           "archived_bytes": sum(p["bytes"] for p in ready if p["rel"] not in set(bad)), "tree_mismatch": bad,
           "not_uploaded": missing, "commits": commits, "status": "VERIFIED" if not bad and not missing else "INCOMPLETE"}
    atomic_json(OUT / f"commit_{tag}.json", rec)
    print(f"commit {tag}: {rec['archived']}/{rec['files']} files in the tree with the planned size and sha256 "
          f"({rec['archived_bytes'] / 1e9:.2f} GB), {len(missing)} not uploaded, {len(bad)} mismatched; {rec['status']}",
          flush=True)
    return rec


def resolve(session, headers, url: str):
    """('url', signed link) or ('inline', bytes). The token goes to huggingface.co only."""
    for _ in range(6):
        host = urllib.parse.urlparse(url).hostname or ""
        on_hub = host == "huggingface.co" or host.endswith(".huggingface.co")
        r = session.get(url, headers=headers if on_hub else {}, allow_redirects=False, timeout=60, stream=True)
        if r.status_code in (301, 302, 303, 307, 308):
            url = urllib.parse.urljoin(url, r.headers["Location"])
            r.close()
            h2 = urllib.parse.urlparse(url).hostname or ""
            if not (h2 == "huggingface.co" or h2.endswith(".huggingface.co")):
                return "url", url
            continue
        if r.status_code == 200:
            b = r.content
            r.close()
            return "inline", b
        raise_for(r, f"resolve {url.split('?')[0]}")
    raise SystemExit("too many redirects")


def stage_links(repo: str, tag: str, token_name: str | None, include: list[str], manifest_from: str | None) -> None:
    session, headers = hf_session(token_name)
    if manifest_from:
        man = read_json(Path(manifest_from))
    else:
        kind, v = resolve(session, headers, f"{HUB}/datasets/{repo}/resolve/main/manifests/{tag}.json")
        if kind == "url":
            with urllib.request.urlopen(urllib.request.Request(v, headers={"User-Agent": "mpr-host-archive"}), timeout=120) as r:
                v = r.read()
        man = json.loads(v.decode("utf-8"))
    rows = [{"rel": r, "bytes": n, "sha256": s} for r, n, s in man["files"]
            if not include or any(fnmatch.fnmatch(r, g) for g in include)]
    out, lifetime = [], None
    for x in rows:
        kind, v = resolve(session, headers, f"{HUB}/datasets/{repo}/resolve/main/{REPO_PREFIX}{urllib.parse.quote(x['rel'])}")
        if kind == "url":
            q = urllib.parse.parse_qs(urllib.parse.urlparse(v).query)
            if "X-Amz-Expires" in q:
                lifetime = int(q["X-Amz-Expires"][0])
            elif "Expires" in q:
                lifetime = int(q["Expires"][0]) - int(time.time())
            out.append({**x, "url": v})
        else:
            if len(v) != x["bytes"] or hashlib.sha256(v).hexdigest() != x["sha256"]:
                raise SystemExit(f"{x['rel']}: the inline bytes are not the manifest's")
            out.append({**x, "inline_b64": base64.b64encode(v).decode("ascii")})
    atomic_json(OUT / f"links_{tag}.json", {"repo": repo, "tag": tag, "generated_utc": utc(),
                                             "generated_epoch": int(time.time()), "link_lifetime_s": lifetime, "files": out})
    print(f"links {tag}: {len(out)} files ({sum(x['bytes'] for x in out) / 1e9:.2f} GB), lifetime {lifetime} s", flush=True)


def stage_status(repo: str, token_name: str | None) -> None:
    api = hf_api(token_name)
    session, headers = hf_session(token_name)
    tree = repo_tree(api, repo)
    for path in sorted(p for p in tree if p.startswith("manifests/") and p.endswith(".json")):
        kind, v = resolve(session, headers, f"{HUB}/datasets/{repo}/resolve/main/{path}")
        if kind == "url":
            with urllib.request.urlopen(urllib.request.Request(v, headers={"User-Agent": "mpr-host-archive"}), timeout=120) as r:
                v = r.read()
        man = json.loads(v.decode("utf-8"))
        ok = sum(1 for r, n, s in man["files"] if tree.get(REPO_PREFIX + r) == (n, s))
        print(f"{path}: {ok}/{len(man['files'])} files present with their size and sha256, {man['bytes'] / 1e9:.2f} GB, "
              f"{len(man.get('not_archived') or [])} not archived", flush=True)


def rx(*args: str, check: bool = True) -> str:
    p = subprocess.run([sys.executable, str(ROOT / "tools" / "rx" / "rx.py"), *args], cwd=ROOT, capture_output=True,
                       text=True, encoding="utf-8", errors="replace")
    out = (p.stdout or "") + (p.stderr or "")
    if check and p.returncode != 0:
        raise SystemExit(f"rx {' '.join(args[:3])} failed rc={p.returncode}: {out[-800:]}")
    return out


def stage_drive(repo: str, tag: str, include: list[str], exclude: list[str], token_name: str | None, workers: int,
                cpus: float, mem: float, per_commit: int, env: str = "mpr-cpu") -> None:
    stage_plan(repo, tag, include, exclude, token_name)
    plan = read_json(OUT / f"plan_{tag}.json")
    rel = lambda p: p.relative_to(ROOT).as_posix()                                        # noqa: E731
    if any(p["upload"] for p in plan["files"]):
        out = rx("run", "--name", f"archive-{tag}", "--env", env, "--cpus", str(cpus), "--mem", str(mem), "-w",
                 "--inputs", rel(Path(__file__).resolve()), "--inputs", rel(OUT / f"wave_{tag}.json"),
                 "--", "python", rel(Path(__file__).resolve()), "upload", "--tag", tag, "--workers", str(workers), check=False)
        print(out[-1500:], flush=True)
        rx("fetch", "--glob", f"outputs/host_archive/done_{tag}.json", "--overwrite")
    (OUT / f"wave_{tag}.json").unlink(missing_ok=True)                                    # spent links stay nowhere
    stage_commit(repo, tag, token_name, per_commit)


def stage_drive_restore(repo: str, tag: str, include: list[str], token_name: str | None, dest: str, workers: int,
                        cpus: float, mem: float, env: str, overwrite: bool) -> dict:
    stage_links(repo, tag, token_name, include, None)
    rel = lambda p: p.relative_to(ROOT).as_posix()                                        # noqa: E731
    try:
        out = rx("run", "--name", f"restore-{tag}", "--env", env, "--cpus", str(cpus), "--mem", str(mem), "-w",
                 "--inputs", rel(Path(__file__).resolve()), "--inputs", rel(OUT / f"links_{tag}.json"),
                 "--", "python", rel(Path(__file__).resolve()), "restore", "--tag", tag, "--dest", dest,
                 "--workers", str(workers), *(["--overwrite"] if overwrite else []), check=False)
        print(out[-1500:], flush=True)
        rx("fetch", "--glob", f"outputs/host_archive/restore_{tag}.json", "--overwrite")
    finally:
        (OUT / f"links_{tag}.json").unlink(missing_ok=True)                               # short-lived links stay nowhere
    rec = read_json(OUT / f"restore_{tag}.json")
    print(f"drive-restore {tag}: {rec['status']}, {len(rec['files']) - len(rec['failed'])}/{len(rec['files'])} verified, "
          f"{len(rec.get('conflicts') or [])} conflicts", flush=True)
    return rec


# ── host ─────────────────────────────────────────────────────────────────────


def no_token() -> None:
    for k in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_API_TOKEN"):
        os.environ.pop(k, None)
    if "huggingface_hub" in sys.modules:
        raise SystemExit("huggingface_hub is imported in a host process; refusing")


def http(method: str, url: str, data: bytes | None = None, headers: dict | None = None, tries: int = 5):
    last = None
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, data=data, method=method, headers={"User-Agent": "mpr-host-archive",
                                                                                **(headers or {})})
            with urllib.request.urlopen(req, timeout=600) as r:
                return r.status, dict(r.headers), r.read()
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code}: {e.read()[:200]!r}"
            if e.code in (400, 401, 403, 404, 410):
                break
        except Exception as e:                      # noqa: BLE001 -- a transfer error is retried, then reported
            last = f"{type(e).__name__}: {e}"
        time.sleep(min(60, 2 ** attempt))
    raise RuntimeError(last)


def upload_one(ws: Path, f: dict) -> dict:
    t0 = time.time()
    rec = {"rel": f["rel"], "bytes": f["bytes"], "sha256": None, "status": "failed"}
    p = ws / safe_rel(f["rel"])
    try:
        st = p.stat()
        if st.st_size != f["bytes"] or st.st_mtime_ns != f["mtime_ns"]:
            return {**rec, "status": "changed", "error": f"size {st.st_size} mtime_ns {st.st_mtime_ns}", "seconds": 0.0}
        h = hashlib.sha256()
        if "chunk_size" in f:
            cs, etags = f["chunk_size"], []
            with open(p, "rb") as fi:
                for url in f["parts"]:
                    b = fi.read(cs)
                    h.update(b)
                    _, hdr, _ = http("PUT", url, data=b)
                    etag = hdr.get("ETag") or hdr.get("etag")
                    if not etag:
                        raise RuntimeError("a part came back without an ETag")
                    etags.append(etag)
                if fi.read(1):
                    raise RuntimeError("the file is longer than its parts")
            rec["sha256"] = h.hexdigest()
            if rec["sha256"] != f["sha256"]:
                return {**rec, "status": "hash_mismatch", "seconds": round(time.time() - t0, 1)}   # never completed
            body = json.dumps({"oid": f["sha256"], "parts": [{"partNumber": i + 1, "etag": e}
                                                              for i, e in enumerate(etags)]}).encode("utf-8")
            http("POST", f["href"], data=body, headers=LFS_HEADERS)
        else:                                        # one PUT makes the object: hash first, then send
            with open(p, "rb") as fi:
                for b in iter(lambda: fi.read(CHUNK), b""):
                    h.update(b)
            rec["sha256"] = h.hexdigest()
            if rec["sha256"] != f["sha256"]:
                return {**rec, "status": "hash_mismatch", "seconds": round(time.time() - t0, 1)}
            with open(p, "rb") as fi:
                http("PUT", f["href"], data=fi, headers={"Content-Length": str(f["bytes"])}, tries=1)
        return {**rec, "status": "uploaded", "seconds": round(time.time() - t0, 1)}
    except Exception as e:                          # noqa: BLE001 -- reported per file; the laptop re-plans
        return {**rec, "status": "failed", "error": f"{type(e).__name__}: {e}"[:300], "seconds": round(time.time() - t0, 1)}


def stage_upload(tag: str, workers: int) -> dict:
    no_token()
    ws = Path.cwd().resolve()
    wave_p = OUT_HOST(ws) / f"wave_{tag}.json"
    wave = read_json(wave_p)
    left = (wave.get("generated_epoch") or 0) + (wave.get("link_lifetime_s") or 0) - time.time()
    print(f"upload {tag}: {len(wave['files'])} files, links live {left / 60:.0f} min more", flush=True)
    t0, recs = time.time(), []
    with ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(upload_one, ws, f) for f in wave["files"]]
        for fu in as_completed(futs):
            r = fu.result()
            recs.append(r)
            if r["status"] not in ("uploaded", "present") or len(recs) % 100 == 0:
                print(f"   {len(recs)}/{len(futs)} {r['rel']}: {r['status']} {r.get('error') or ''}", flush=True)
    recs.sort(key=lambda r: r["rel"])
    by: dict[str, int] = {}
    for r in recs:
        by[r["status"]] = by.get(r["status"], 0) + 1
    sent = sum(r["bytes"] for r in recs if r["status"] == "uploaded")
    out = {"tag": tag, "utc": utc(), "files": recs, "by_status": by, "uploaded_bytes": sent,
           "seconds": round(time.time() - t0, 1), "mb_per_s": round(sent / 1e6 / max(1e-9, time.time() - t0), 1)}
    atomic_json(OUT_HOST(ws) / f"done_{tag}.json", out)
    wave_p.unlink()                                  # the links are spent, or the laptop re-plans; they do not stay here
    print(f"upload {tag}: {by} in {out['seconds']:.0f}s, {out['mb_per_s']} MB/s", flush=True)
    return out


def OUT_HOST(ws: Path) -> Path:                     # noqa: N802 -- the workspace's own copy of OUT
    return ws / "outputs" / "host_archive"


def fetch_one(dest_root: Path, x: dict, overwrite: bool = False) -> dict:
    t0 = time.time()
    dest = dest_root / safe_rel(x["rel"])
    if dest.exists():
        if dest.is_file() and dest.stat().st_size == x["bytes"]:
            h = hashlib.sha256()
            with open(dest, "rb") as fi:
                for b in iter(lambda: fi.read(CHUNK), b""):
                    h.update(b)
            if h.hexdigest() == x["sha256"]:
                return {"rel": x["rel"], "how": "already_present", "verified": True, "seconds": 0.0}
        if not overwrite or not dest.is_file():       # newer work on a live host is never replaced by default
            return {"rel": x["rel"], "how": "conflict", "verified": False, "error": "a different file is there",
                    "seconds": round(time.time() - t0, 1)}
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    last = None
    for attempt in range(4):
        h, n = hashlib.sha256(), 0
        try:
            if "inline_b64" in x:
                b = base64.b64decode(x["inline_b64"])
                h.update(b)
                n = len(b)
                part.write_bytes(b)
            else:
                req = urllib.request.Request(x["url"], headers={"User-Agent": "mpr-host-archive"})
                with urllib.request.urlopen(req, timeout=120) as r, open(part, "wb") as fo:
                    for b in iter(lambda: r.read(CHUNK), b""):
                        h.update(b)
                        fo.write(b)
                        n += len(b)
            if n == x["bytes"] and h.hexdigest() == x["sha256"]:
                os.replace(part, dest)
                return {"rel": x["rel"], "how": "download", "verified": True, "seconds": round(time.time() - t0, 1)}
            last = f"got {n} bytes, sha256 {h.hexdigest()[:12]}"
        except Exception as e:                      # noqa: BLE001
            last = f"{type(e).__name__}: {e}"
            if isinstance(e, urllib.error.HTTPError) and e.code in (401, 403, 404):
                break
        time.sleep(min(30, 2 ** attempt))
    if part.exists():
        part.unlink()
    return {"rel": x["rel"], "how": "failed", "verified": False, "error": last, "seconds": round(time.time() - t0, 1)}


def stage_restore(tag: str, dest: str, workers: int, overwrite: bool = False) -> dict:
    no_token()
    ws = Path.cwd().resolve()
    links_p = OUT_HOST(ws) / f"links_{tag}.json"
    links = read_json(links_p)
    dest_root = Path(dest).resolve()
    t0, recs = time.time(), []
    with ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(fetch_one, dest_root, x, overwrite) for x in links["files"]]
        for fu in as_completed(futs):
            r = fu.result()
            recs.append(r)
            if not r["verified"] or len(recs) % 100 == 0:
                print(f"   {len(recs)}/{len(futs)} {r['rel']}: {r['how']} {r.get('error') or ''}", flush=True)
    recs.sort(key=lambda r: r["rel"])
    bad = [r["rel"] for r in recs if not r["verified"]]
    out = {"tag": tag, "utc": utc(), "dest": dest_root.as_posix(), "files": recs, "failed": bad,
           "conflicts": [r["rel"] for r in recs if r["how"] == "conflict"],
           "status": "VERIFIED" if not bad else "INCOMPLETE", "seconds": round(time.time() - t0, 1)}
    atomic_json(OUT_HOST(ws) / f"restore_{tag}.json", out)
    if not bad:
        links_p.unlink()
    print(f"restore {tag}: {len(recs) - len(bad)}/{len(recs)} verified into {dest_root}, {out['seconds']:.0f}s, {out['status']}",
          flush=True)
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("mode", choices=["plan", "commit", "links", "status", "drive", "drive-restore", "upload", "restore"])
    ap.add_argument("--repo")
    ap.add_argument("--tag")
    ap.add_argument("--include", action="append", default=[])
    ap.add_argument("--exclude", action="append", default=[])
    ap.add_argument("--token-name", default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--per-commit", type=int, default=200)
    ap.add_argument("--cpus", type=float, default=2)
    ap.add_argument("--mem", type=float, default=2)
    ap.add_argument("--dest")
    ap.add_argument("--overwrite", action="store_true", help="restore: replace a file that differs")
    ap.add_argument("--env", default="mpr-cpu", help="drive/drive-restore: the rx environment of the host job")
    ap.add_argument("--manifest-from")
    a = ap.parse_args(argv)
    if a.mode in ("upload", "restore"):
        tag = check_tag(a.tag)
        if a.mode == "upload":
            stage_upload(tag, a.workers)
        else:
            if not a.dest:
                ap.error("restore needs --dest")
            stage_restore(tag, a.dest, a.workers, a.overwrite)
        return
    if not a.repo:
        ap.error(f"{a.mode} needs --repo")
    if a.mode == "status":
        stage_status(a.repo, a.token_name)
        return
    tag = check_tag(a.tag)
    if a.mode == "plan":
        stage_plan(a.repo, tag, a.include, a.exclude, a.token_name)
    elif a.mode == "commit":
        stage_commit(a.repo, tag, a.token_name, a.per_commit)
    elif a.mode == "links":
        stage_links(a.repo, tag, a.token_name, a.include, a.manifest_from)
    elif a.mode == "drive":
        if not a.include:
            ap.error("drive needs --include")
        stage_drive(a.repo, tag, a.include, a.exclude, a.token_name, a.workers, a.cpus, a.mem, a.per_commit, a.env)
    elif a.mode == "drive-restore":
        if not a.dest:
            ap.error("drive-restore needs --dest ('.' restores into the workspace itself)")
        stage_drive_restore(a.repo, tag, a.include, a.token_name, a.dest, a.workers, a.cpus, a.mem, a.env, a.overwrite)


if __name__ == "__main__":
    main()
