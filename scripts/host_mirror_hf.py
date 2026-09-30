"""The six-dataset host mirror of the served package, through a private Hugging Face dataset repo
(configs/host_mirror_six.yaml). Systems only: it moves and verifies bytes, and no stage reads the mirror until a
later file places that stage on the host.

Laptop modes (the token the user stored with `hf auth login`: this script never reads, prints or writes it, and it
is sent only to huggingface.co):
  manifest  the declared files, each read in place and hashed; every sha256 the package itself pins (DATASET.json,
            the shard manifests, GRAPH_MANIFEST.json) is cross-checked -> outputs/host_mirror_six/manifest.json
  upload    commit the declared files to the private repo, read in place: no byte is written under the package root
            (no staging copy, no upload_large_folder, which keeps a cache inside the folder it uploads); a file the
            repo already holds with the manifest's hash, or one the host survey found at the host source, is skipped
  status    the repo against the manifest
  urls      each needed file's signed download link, the redirect followed by hand so that no Authorization header
            reaches a host that is not huggingface.co -> outputs/host_mirror_six/_urls.json (git-ignored, short-lived)
Host modes (no token of any kind: HF_TOKEN is removed and huggingface_hub is never imported):
  survey    hash the declared files found at the host source (another project's workspace, read only)
            -> outputs/host_mirror_six/host_sources.json
  download  every declared file of --datasets into the mirror root, copied from the host source when its sha256 equals
            the manifest's, else fetched by its signed link; hashed as written -> outputs/host_mirror_six/download_<tag>.json
  verify    re-hash the mirror against the manifest, then the served-freeze check of m3b_compile.open_package with
            substrate.package_root substituted in memory -> outputs/host_mirror_six/verify.json
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "host_mirror_six.yaml"
OUT = ROOT / "outputs" / "host_mirror_six"
MANIFEST = OUT / "manifest.json"
URLS = OUT / "_urls.json"
SERVED_REL = "data/final_canonical"
REPO_PREFIX = "final_canonical/"
GIT_SHA1_MAX = 10 * (1 << 20)          # below this size the hub may keep a file in git: its blob id is a git sha1
INLINE_MAX = 20 * (1 << 20)
CHUNK = 8 * (1 << 20)


def load_decl() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def atomic_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def file_hashes(p: Path) -> tuple[int, str, str | None]:
    """(bytes, sha256, git blob sha1 for a file below GIT_SHA1_MAX else None), one read."""
    n = p.stat().st_size
    h = hashlib.sha256()
    g = hashlib.sha1(b"blob %d\0" % n) if n < GIT_SHA1_MAX else None
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(CHUNK), b""):
            h.update(b)
            if g is not None:
                g.update(b)
    return n, h.hexdigest(), (g.hexdigest() if g is not None else None)


# ── the declared selection ───────────────────────────────────────────────────


def selected(served: Path, sel: dict) -> list[str]:
    """Served-relative paths of the declared files, in upload order: the package's loader files, then per dataset (in
    the declared order) every file directly in the dataset folder and every file under the declared subfolders."""
    out = []
    for f in sel["package_files"]:
        if not (served / f).is_file():
            raise SystemExit(f"{served / f}: a declared package file is missing")
        out.append(f)
    for ds in sel["datasets"]:
        base = served / ds
        rels = sorted(p.name for p in base.iterdir() if p.is_file())
        for sub in sel["dataset_subfolders"]:
            d = base / sub
            if not d.is_dir():
                raise SystemExit(f"{d}: a declared subfolder is missing")
            rels += sorted(p.relative_to(base).as_posix() for p in d.rglob("*") if p.is_file())
        out += [f"{ds}/{r}" for r in rels]
    if len(out) != len(set(out)):
        raise SystemExit("the selection names a file twice")
    return out


def package_pins(served: Path, datasets) -> dict:
    """Every (file, sha256) the package pins in DATASET.json, the embedding manifests and GRAPH_MANIFEST.json, as
    served-relative path -> sha256."""
    pins: dict[str, str] = {}

    def walk(o, here: str):
        if isinstance(o, dict):
            f, h = o.get("file"), o.get("sha256")
            if isinstance(f, str) and isinstance(h, str) and len(h) == 64:
                rel = f[len(SERVED_REL) + 1:] if f.startswith(SERVED_REL + "/") else f"{here}/{f}"
                if pins.get(rel, h) != h:
                    raise SystemExit(f"the package pins two hashes for {rel}")
                pins[rel] = h
            for v in o.values():
                walk(v, here)
        elif isinstance(o, list):
            for v in o:
                walk(v, here)

    for ds in datasets:
        docs = [served / ds / "DATASET.json", served / ds / "graph" / "GRAPH_MANIFEST.json"]
        docs += sorted((served / ds / "embeddings").glob("*/*/manifest.json"))
        for p in docs:
            if p.is_file():
                walk(json.loads(p.read_text(encoding="utf-8")), p.parent.relative_to(served).as_posix())
    return pins


# ── laptop ───────────────────────────────────────────────────────────────────


def laptop_served(decl: dict) -> Path:
    cfg = yaml.safe_load((ROOT / decl["package"]["root_from"]).read_text(encoding="utf-8"))
    served = Path(cfg["substrate"]["package_root"]) / SERVED_REL
    if not served.is_dir():
        raise SystemExit(f"{served}: not the served package")
    return served


def stage_manifest(decl: dict, threads: int = 3) -> dict:
    served = laptop_served(decl)
    sel = decl["selection"]
    rels = selected(served, sel)
    pins = package_pins(served, sel["datasets"])
    t0 = time.time()
    rows: dict[str, dict] = {}
    with ThreadPoolExecutor(threads) as ex:
        futs = {ex.submit(file_hashes, served / r): r for r in rels}
        for fu in as_completed(futs):
            n, h, g = fu.result()
            rows[futs[fu]] = {"bytes": n, "sha256": h, "git_sha1": g}
    files, bad = [], []
    for r in rels:
        row = {"rel": r, "dataset": r.split("/")[0] if "/" in r else None, **rows[r], "package_pin": pins.get(r)}
        if row["package_pin"] is not None and row["package_pin"] != row["sha256"]:
            bad.append(r)
        files.append(row)
    if bad:
        raise SystemExit(f"{len(bad)} files differ from the package's own pins (first: {bad[:3]}); refusing to mirror")
    freeze = json.loads((served / "CANONICAL_FREEZE.json").read_text(encoding="utf-8"))
    out = {"utc": utc(), "served": served.as_posix(), "selection": sel, "freeze_RECORD_SHA256": freeze["RECORD_SHA256"],
           "files": files, "n_files": len(files), "bytes": sum(f["bytes"] for f in files),
           "pinned_by_package": sum(1 for f in files if f["package_pin"] is not None),
           "pins_mismatched": 0, "seconds": round(time.time() - t0, 1)}
    atomic_json(MANIFEST, out)
    print(f"manifest: {out['n_files']} files, {out['bytes'] / 1e9:.2f} GB, {out['pinned_by_package']} matched against the "
          f"package's own pins, 0 mismatched, {out['seconds']:.0f}s -> {MANIFEST}", flush=True)
    return out


def hf_api():
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    from huggingface_hub import HfApi
    return HfApi()


def private_repo(api, repo: str) -> None:
    api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
    if api.repo_info(repo, repo_type="dataset").private is not True:
        raise SystemExit(f"REFUSING: {repo} is not private")


def repo_state(api, repo: str) -> dict:
    """repo path -> (bytes, sha256 or None, git blob id or None)."""
    out = {}
    for e in api.list_repo_tree(repo, repo_type="dataset", recursive=True):
        if hasattr(e, "size"):
            lfs = getattr(e, "lfs", None)
            out[e.path] = (e.size, getattr(lfs, "sha256", None) if lfs else None, getattr(e, "blob_id", None))
    return out


def in_repo(row: dict, state: dict) -> bool:
    got = state.get(REPO_PREFIX + row["rel"])
    if got is None or got[0] != row["bytes"]:
        return False
    return got[1] == row["sha256"] or (got[1] is None and row["git_sha1"] is not None and got[2] == row["git_sha1"])


def host_has(row: dict, survey: dict | None) -> bool:
    return survey is not None and survey.get("files", {}).get(row["rel"], {}).get("sha256") == row["sha256"]


def load_survey() -> dict | None:
    p = OUT / "host_sources.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def stage_upload(decl: dict, datasets: list[str], group_bytes: float, group_files: int) -> None:
    """Dataset by dataset in the declared order; the repo state and the host survey are read again before each, so a
    survey that lands while the upload runs still spares the files it found."""
    from huggingface_hub import CommitOperationAdd
    man = json.loads(MANIFEST.read_text(encoding="utf-8"))
    api = hf_api()
    repo = decl["transport"]["repo"]
    private_repo(api, repo)
    served = Path(man["served"])
    sent, t_all = 0, time.time()
    for ds in [None] + [d for d in decl["selection"]["datasets"] if d in datasets]:
        survey = load_survey()
        state = repo_state(api, repo)
        want = [f for f in man["files"] if f["dataset"] == ds]
        todo = [f for f in want if not in_repo(f, state) and not host_has(f, survey)]
        at_host = sum(1 for f in want if host_has(f, survey) and not in_repo(f, state))
        print(f"{utc()} {ds or 'package'}: {len(want)} files, {len(want) - len(todo) - at_host} already in the repo, "
              f"{at_host} at the host source, {len(todo)} to send ({sum(f['bytes'] for f in todo) / 1e9:.2f} GB)", flush=True)
        groups, cur, cur_b = [], [], 0
        for f in todo:
            if cur and (cur_b + f["bytes"] > group_bytes or len(cur) >= group_files):
                groups.append(cur)
                cur, cur_b = [], 0
            cur.append(f)
            cur_b += f["bytes"]
        if cur:
            groups.append(cur)
        for i, g in enumerate(groups):
            for f in g:                              # the bytes about to be read must still be the manifest's
                p = served / f["rel"]
                if p.stat().st_size != f["bytes"]:
                    raise SystemExit(f"{p}: size changed since the manifest; refusing")
            ops = [CommitOperationAdd(path_in_repo=REPO_PREFIX + f["rel"], path_or_fileobj=str(served / f["rel"])) for f in g]
            for f, op in zip(g, ops):
                if op.upload_info.sha256.hex() != f["sha256"]:
                    raise SystemExit(f"{f['rel']}: sha256 changed since the manifest; refusing")
            b = sum(f["bytes"] for f in g)
            t0 = time.time()
            api.create_commit(repo_id=repo, repo_type="dataset", operations=ops,
                              commit_message=f"mirror {ds or 'package'}: {len(g)} files, {b / 1e9:.2f} GB")
            dt = time.time() - t0
            sent += b
            print(f"{utc()} {ds or 'package'} group {i + 1}/{len(groups)} ({len(g)} files, {b / 1e9:.2f} GB) in {dt:.0f}s = "
                  f"{b / 1e6 / max(dt, 1e-9):.2f} MB/s; {sent / 1e9:.2f} GB sent, "
                  f"{sent / 1e6 / (time.time() - t_all):.2f} MB/s overall", flush=True)
        print(f"{utc()} {ds or 'package'}: in the repo", flush=True)
    print(f"{utc()} upload done: {sent / 1e9:.2f} GB in {time.time() - t_all:.0f}s", flush=True)


def stage_status(decl: dict) -> dict:
    man = json.loads(MANIFEST.read_text(encoding="utf-8"))
    api = hf_api()
    state = repo_state(api, decl["transport"]["repo"])
    survey = load_survey()
    by = {}
    for f in man["files"]:
        k = f["dataset"] or "package"
        d = by.setdefault(k, {"files": 0, "bytes": 0, "in_repo": 0, "in_repo_bytes": 0, "at_host_source": 0})
        d["files"] += 1
        d["bytes"] += f["bytes"]
        if in_repo(f, state):
            d["in_repo"] += 1
            d["in_repo_bytes"] += f["bytes"]
        elif host_has(f, survey):
            d["at_host_source"] += 1
    for k, d in by.items():
        print(f"{k:9s} {d['in_repo']:4d}/{d['files']:4d} files in the repo ({d['in_repo_bytes'] / 1e9:6.2f} of "
              f"{d['bytes'] / 1e9:6.2f} GB), {d['at_host_source']} more at the host source", flush=True)
    return by


def resolve(url: str):
    """('url', signed link) or ('inline', bytes). The Authorization header goes to huggingface.co only."""
    from huggingface_hub.utils import build_hf_headers, get_session
    s = get_session()
    for _ in range(6):
        host = urllib.parse.urlparse(url).hostname or ""
        hdr = build_hf_headers() if host == "huggingface.co" or host.endswith(".huggingface.co") else {}
        r = s.get(url, headers=hdr, allow_redirects=False, timeout=60, stream=True)
        if r.status_code in (301, 302, 303, 307, 308):
            url = urllib.parse.urljoin(url, r.headers["Location"])
            r.close()
            h2 = urllib.parse.urlparse(url).hostname or ""
            if not (h2 == "huggingface.co" or h2.endswith(".huggingface.co")):
                return "url", url
            continue
        if r.status_code == 200:
            n = int(r.headers.get("Content-Length", "0"))
            if n > INLINE_MAX:
                raise SystemExit(f"no redirect for a {n}-byte file: {url.split('?')[0]}")
            b = r.content
            r.close()
            return "inline", b
        r.raise_for_status()
    raise SystemExit("too many redirects")


def stage_urls(decl: dict, datasets: list[str]) -> None:
    from huggingface_hub import hf_hub_url
    man = json.loads(MANIFEST.read_text(encoding="utf-8"))
    api = hf_api()
    repo = decl["transport"]["repo"]
    state = repo_state(api, repo)
    survey = load_survey()
    want = [f for f in man["files"] if f["dataset"] is None or f["dataset"] in datasets]
    out, lifetime = [], None
    for f in want:
        if host_has(f, survey):
            continue
        if not in_repo(f, state):
            raise SystemExit(f"{f['rel']}: neither in the repo nor at the host source; upload first")
        kind, v = resolve(hf_hub_url(repo, REPO_PREFIX + f["rel"], repo_type="dataset"))
        if kind == "url":
            q = urllib.parse.parse_qs(urllib.parse.urlparse(v).query)
            if "X-Amz-Expires" in q:
                lifetime = int(q["X-Amz-Expires"][0])
            elif "Expires" in q:
                lifetime = int(q["Expires"][0]) - int(time.time())
            out.append({"rel": f["rel"], "url": v})
        else:
            if len(v) != f["bytes"] or hashlib.sha256(v).hexdigest() != f["sha256"]:
                raise SystemExit(f"{f['rel']}: the inline bytes are not the manifest's")
            out.append({"rel": f["rel"], "inline_b64": base64.b64encode(v).decode("ascii")})
    atomic_json(URLS, {"repo": repo, "datasets": datasets, "generated_utc": utc(), "generated_epoch": int(time.time()),
                       "url_lifetime_s": lifetime, "files": out})
    print(f"{len(out)} links ({sum(1 for o in out if 'url' in o)} signed, {sum(1 for o in out if 'inline_b64' in o)} inline), "
          f"lifetime {lifetime} s -> {URLS}", flush=True)


# ── host ─────────────────────────────────────────────────────────────────────


def no_token() -> None:
    for k in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_API_TOKEN"):
        os.environ.pop(k, None)
    if "huggingface_hub" in sys.modules:
        raise SystemExit("huggingface_hub is imported in a host process; refusing")


def host_paths(decl: dict) -> tuple[Path, list[Path]]:
    h = decl["host"]
    mirror = Path(h["mirror_root"]) / SERVED_REL
    ws = Path.cwd().resolve()
    if Path(h["mirror_root"]).resolve().is_relative_to(ws):
        raise SystemExit("the mirror root is inside the workspace; refusing")
    return mirror, [Path(s) / SERVED_REL for s in h["read_only_sources"]]


def stage_survey(decl: dict) -> None:
    no_token()
    man = json.loads(MANIFEST.read_text(encoding="utf-8"))
    _, sources = host_paths(decl)
    t0, found = time.time(), {}
    for f in man["files"]:
        for src in sources:
            p = src / f["rel"]
            if p.is_file() and p.stat().st_size == f["bytes"]:
                n, h, _ = file_hashes(p)
                found[f["rel"]] = {"source": src.as_posix(), "bytes": n, "sha256": h, "matches": h == f["sha256"]}
                break
    good = {k: v for k, v in found.items() if v["matches"]}
    atomic_json(OUT / "host_sources.json", {"utc": utc(), "manifest_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
                                            "sources": [s.as_posix() for s in sources], "files": good,
                                            "size_equal_but_hash_differs": sorted(k for k, v in found.items() if not v["matches"]),
                                            "seconds": round(time.time() - t0, 1)})
    print(f"survey: {len(good)} declared files at the host source with the manifest's sha256 "
          f"({sum(v['bytes'] for v in good.values()) / 1e9:.2f} GB); {len(found) - len(good)} same size, other hash", flush=True)


def fetch_one(row: dict, src: dict | None, dest: Path, tries: int = 4) -> dict:
    """One file into dest (via dest.part), hashed as written; returns its record."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size == row["bytes"] and file_hashes(dest)[1] == row["sha256"]:
        return {"rel": row["rel"], "how": "already_present", "verified": True, "seconds": 0.0}
    part = dest.with_name(dest.name + ".part")
    t0 = time.time()
    last = None
    for attempt in range(tries):
        h = hashlib.sha256()
        n = 0
        try:
            if src is not None and "local" in src:
                how = "host_copy"
                with open(src["local"], "rb") as fi, open(part, "wb") as fo:
                    for b in iter(lambda: fi.read(CHUNK), b""):
                        h.update(b)
                        fo.write(b)
                        n += len(b)
            elif src is not None and "inline_b64" in src:
                how = "inline"
                b = base64.b64decode(src["inline_b64"])
                h.update(b)
                n = len(b)
                part.write_bytes(b)
            elif src is not None and "url" in src:
                how = "download"
                req = urllib.request.Request(src["url"], headers={"User-Agent": "mpr-host-mirror"})
                with urllib.request.urlopen(req, timeout=120) as r, open(part, "wb") as fo:
                    for b in iter(lambda: r.read(CHUNK), b""):
                        h.update(b)
                        fo.write(b)
                        n += len(b)
            else:
                return {"rel": row["rel"], "how": "no_source", "verified": False, "seconds": 0.0}
            ok = n == row["bytes"] and h.hexdigest() == row["sha256"]
            if ok:
                os.replace(part, dest)
                return {"rel": row["rel"], "how": how, "verified": True, "bytes": n, "seconds": round(time.time() - t0, 1)}
            last = f"got {n} bytes, sha256 {h.hexdigest()[:12]}"
        except Exception as e:                      # noqa: BLE001 -- a transfer error is retried, then reported
            last = f"{type(e).__name__}: {e}"
            if isinstance(e, urllib.error.HTTPError) and e.code in (401, 403, 404):
                break
        time.sleep(min(30, 2 ** attempt))
    if part.exists():
        part.unlink()
    return {"rel": row["rel"], "how": "failed", "verified": False, "error": last, "seconds": round(time.time() - t0, 1)}


def stage_download(decl: dict, datasets: list[str], workers: int) -> None:
    no_token()
    man = json.loads(MANIFEST.read_text(encoding="utf-8"))
    mirror, _ = host_paths(decl)
    survey = load_survey()
    links = json.loads(URLS.read_text(encoding="utf-8")) if URLS.exists() else {"files": []}
    by_rel = {o["rel"]: o for o in links["files"]}
    if links.get("generated_epoch") and links.get("url_lifetime_s"):
        left = links["generated_epoch"] + links["url_lifetime_s"] - time.time()
        print(f"signed links: {left / 60:.0f} min of life left", flush=True)
    want = [f for f in man["files"] if f["dataset"] is None or f["dataset"] in datasets]
    jobs = []
    for f in want:
        if host_has(f, survey):
            src = {"local": str(Path(survey["files"][f["rel"]]["source"]) / f["rel"])}
        else:
            src = by_rel.get(f["rel"])
        jobs.append((f, src))
    t0, recs = time.time(), []
    with ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(fetch_one, f, src, mirror / f["rel"]) for f, src in jobs]
        for fu in as_completed(futs):
            r = fu.result()
            recs.append(r)
            if not r["verified"] or len(recs) % 25 == 0:
                print(f"   {len(recs)}/{len(jobs)} {r['rel']}: {r['how']} {'ok' if r['verified'] else r.get('error')}", flush=True)
    recs.sort(key=lambda r: r["rel"])
    bad = [r["rel"] for r in recs if not r["verified"]]
    tag = "_".join(datasets)
    size = {f["rel"]: f["bytes"] for f in want}
    by_how: dict[str, int] = {}
    for r in recs:
        by_how[r["how"]] = by_how.get(r["how"], 0) + size[r["rel"]]
    out = {"utc": utc(), "datasets": datasets, "mirror": mirror.as_posix(), "files": recs, "n_files": len(recs),
           "failed": bad, "status": "VERIFIED" if not bad else "INCOMPLETE", "bytes_by_how": by_how,
           "seconds": round(time.time() - t0, 1), "manifest_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest()}
    atomic_json(OUT / f"download_{tag}.json", out)
    if URLS.exists() and not bad:
        URLS.unlink()                                # the signed links are spent; they do not stay in the workspace
    print(f"download {tag}: {len(recs) - len(bad)}/{len(recs)} verified, {out['seconds']:.0f}s, status {out['status']}", flush=True)
    if bad:
        raise SystemExit(f"{len(bad)} files not verified (first: {bad[:3]})")


def stage_verify(decl: dict, datasets: list[str]) -> None:
    no_token()
    man = json.loads(MANIFEST.read_text(encoding="utf-8"))
    mirror, _ = host_paths(decl)
    want = [f for f in man["files"] if f["dataset"] is None or f["dataset"] in datasets]
    t0 = time.time()
    bad = []
    with ThreadPoolExecutor(4) as ex:
        futs = {ex.submit(file_hashes, mirror / f["rel"]): f for f in want if (mirror / f["rel"]).is_file()}
        for fu in as_completed(futs):
            f = futs[fu]
            n, h, _ = fu.result()
            if n != f["bytes"] or h != f["sha256"]:
                bad.append(f["rel"])
    missing = [f["rel"] for f in want if not (mirror / f["rel"]).is_file()]
    sys.path.insert(0, str(ROOT / "scripts"))
    import m3b_compile                                # noqa: E402 -- imported after the hash pass, as the stages do
    cfg = yaml.safe_load((ROOT / decl["package"]["root_from"]).read_text(encoding="utf-8"))
    cfg["substrate"]["package_root"] = str(Path(decl["host"]["mirror_root"]))       # in memory only
    _, canonical, served, freeze = m3b_compile.open_package(cfg)
    loader_ok = Path(canonical.__file__).resolve().parent == served.resolve()
    out = {"utc": utc(), "datasets": datasets, "mirror": mirror.as_posix(), "files_checked": len(want) - len(missing),
           "missing": missing, "hash_mismatch": sorted(bad), "freeze_RECORD_SHA256": freeze["RECORD_SHA256"],
           "freeze_matches_declared": freeze["RECORD_SHA256"] == cfg["substrate"]["freeze_RECORD_SHA256_expected"],
           "loader_imported_from_mirror": loader_ok, "seconds": round(time.time() - t0, 1),
           "manifest_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest()}
    out["status"] = "VERIFIED" if not missing and not bad and out["freeze_matches_declared"] and loader_ok else "FAILED"
    atomic_json(OUT / f"verify_{'_'.join(datasets)}.json", out)
    print(f"verify: {out['files_checked']} files, {len(missing)} missing, {len(bad)} mismatched, freeze "
          f"{'ok' if out['freeze_matches_declared'] else 'DIFFERS'}, loader from the mirror {loader_ok} -> {out['status']}", flush=True)
    if out["status"] != "VERIFIED":
        raise SystemExit(1)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("mode", choices=("manifest", "upload", "status", "urls", "survey", "download", "verify"))
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--group-gb", type=float, default=1.0)
    ap.add_argument("--group-files", type=int, default=40)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args(argv)
    decl = load_decl()
    ds = a.datasets or list(decl["selection"]["datasets"])
    unknown = set(ds) - set(decl["selection"]["datasets"])
    if unknown:
        raise SystemExit(f"not declared: {sorted(unknown)}")
    if a.mode == "manifest":
        stage_manifest(decl)
    elif a.mode == "upload":
        stage_upload(decl, ds, a.group_gb * 1e9, a.group_files)
    elif a.mode == "status":
        stage_status(decl)
    elif a.mode == "urls":
        stage_urls(decl, ds)
    elif a.mode == "survey":
        stage_survey(decl)
    elif a.mode == "download":
        stage_download(decl, ds, a.workers)
    elif a.mode == "verify":
        stage_verify(decl, ds)


if __name__ == "__main__":
    main()
