"""scripts/host_archive_hf.py: link parsing, the host's upload (multipart and single) against a local stand-in for the
signed links, the hash guard that never completes a wrong object, plan/commit bookkeeping, and the restore."""
import base64
import hashlib
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import host_archive_hf as HA  # noqa: E402


class Store:
    def __init__(self):
        self.parts, self.objects, self.completions, self.fail_get = {}, {}, [], set()


def serve(store: Store):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _body(self):
            n = int(self.headers.get("Content-Length") or 0)
            return self.rfile.read(n)

        def do_PUT(self):
            b = self._body()
            seg = self.path.strip("/").split("/")
            if seg[0] == "part":
                store.parts[(seg[1], int(seg[2]))] = b
                self.send_response(200)
                self.send_header("ETag", '"%s"' % hashlib.md5(b).hexdigest())
            else:                                      # single/<oid>
                store.objects[seg[1]] = b
                self.send_response(200)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_POST(self):
            oid = self.path.strip("/").split("/")[1]
            j = json.loads(self._body())
            store.completions.append(j)
            data = b"".join(store.parts[(oid, p["partNumber"])] for p in j["parts"])
            ok = hashlib.sha256(data).hexdigest() == oid and all(
                p["etag"] == '"%s"' % hashlib.md5(store.parts[(oid, p["partNumber"])]).hexdigest() for p in j["parts"])
            if ok:
                store.objects[oid] = data
            self.send_response(200 if ok else 400)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_GET(self):
            oid = self.path.strip("/").split("/")[1]
            b = store.objects.get(oid)
            if oid in store.fail_get:
                b = b"wrong bytes"
            if b is None:
                self.send_response(404)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


@pytest.fixture()
def server():
    store = Store()
    srv, base = serve(store)
    yield store, base
    srv.shutdown()


def write(ws: Path, rel: str, data: bytes) -> dict:
    p = ws / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    st = p.stat()
    return {"rel": rel, "bytes": st.st_size, "mtime_ns": st.st_mtime_ns, "sha256": hashlib.sha256(data).hexdigest()}


def multipart(base: str, row: dict, cs: int) -> dict:
    n = max(1, -(-row["bytes"] // cs))
    return {**row, "href": f"{base}/complete/{row['sha256']}", "chunk_size": cs,
            "parts": [f"{base}/part/{row['sha256']}/{i + 1}" for i in range(n)]}


def test_parse_action():
    assert HA.parse_action({"oid": "a", "size": 3}) == {"present": True}
    hdr = {"chunk_size": "16", "00002": "u2", "00010": "u10", "00001": "u1", "00003": "u3", "00004": "u4", "00005": "u5",
           "00006": "u6", "00007": "u7", "00008": "u8", "00009": "u9"}
    a = HA.parse_action({"oid": "a", "size": 150, "actions": {"upload": {"href": "h", "header": hdr, "expires_in": 36000},
                                                               "verify": {"href": "v"}}})
    assert a["parts"] == [f"u{i}" for i in range(1, 11)] and a["chunk_size"] == 16 and a["verify_href"] == "v"
    assert a["expires_in"] == 36000 and "header" not in a
    with pytest.raises(SystemExit, match="Authorization"):
        HA.parse_action({"oid": "a", "size": 3, "actions": {"upload": {"href": "h", "header": {"Authorization": "x"}}}})
    with pytest.raises(SystemExit, match="does not cover"):
        HA.parse_action({"oid": "a", "size": 100, "actions": {"upload": {"href": "h", "header": {"chunk_size": "16", "00001": "u"}}}})
    with pytest.raises(SystemExit, match="LFS batch error"):
        HA.parse_action({"oid": "a", "size": 3, "error": {"code": 422, "message": "bad"}})
    assert HA.parse_action({"oid": "a", "size": 3, "actions": {"upload": {"href": "h"}}})["href"] == "h"


def test_commit_lines():
    body = HA.commit_lines([{"rel": "outputs/a/b.npz", "bytes": 5, "sha256": "f" * 64}], "s", {"manifests/t.json": b"{}"})
    lines = [json.loads(x) for x in body.decode().split("\n")]
    assert lines[0] == {"key": "header", "value": {"summary": "s", "description": ""}}
    assert lines[1] == {"key": "lfsFile", "value": {"path": "ws/outputs/a/b.npz", "algo": "sha256", "oid": "f" * 64, "size": 5}}
    assert lines[2]["key"] == "file" and base64.b64decode(lines[2]["value"]["content"]) == b"{}"
    for bad in ("../x", "/abs", "a//b", "C:/x", "a\\b"):
        with pytest.raises(SystemExit, match="unsafe"):
            HA.commit_lines([{"rel": bad, "bytes": 1, "sha256": "0" * 64}], "s")


def test_upload_multipart_single_changed_and_mismatch(tmp_path, server):
    store, base = server
    ws = tmp_path
    a = write(ws, "outputs/x/a.npz", os.urandom(40))
    r = HA.upload_one(ws, multipart(base, a, 16))
    assert r["status"] == "uploaded" and r["sha256"] == a["sha256"] and store.objects[a["sha256"]] == (ws / a["rel"]).read_bytes()
    assert len(store.completions) == 1 and [p["partNumber"] for p in store.completions[0]["parts"]] == [1, 2, 3]
    # exactly one chunk long, and an empty file
    e = write(ws, "outputs/x/e.npz", os.urandom(16))
    assert HA.upload_one(ws, multipart(base, e, 16))["status"] == "uploaded"
    z = write(ws, "outputs/x/z.npz", b"")
    assert HA.upload_one(ws, multipart(base, z, 16))["status"] == "uploaded" and store.objects[z["sha256"]] == b""
    # single PUT
    s = write(ws, "outputs/x/s.json", b'{"k": 1}')
    r = HA.upload_one(ws, {**s, "href": f"{base}/single/{s['sha256']}"})
    assert r["status"] == "uploaded" and store.objects[s["sha256"]] == b'{"k": 1}'
    # changed since the plan: nothing is sent
    c = write(ws, "outputs/x/c.npz", os.urandom(20))
    n_parts = len(store.parts)
    r = HA.upload_one(ws, {**multipart(base, c, 16), "mtime_ns": c["mtime_ns"] + 1})
    assert r["status"] == "changed" and len(store.parts) == n_parts
    # bytes that do not hash to the planned oid: the parts go up but the object is never completed
    m = write(ws, "outputs/x/m.npz", os.urandom(20))
    wrong = "0" * 64
    n_done = len(store.completions)
    r = HA.upload_one(ws, multipart(base, {**m, "sha256": wrong}, 16))
    assert r["status"] == "hash_mismatch" and r["sha256"] == m["sha256"] and len(store.completions) == n_done
    assert wrong not in store.objects
    r = HA.upload_one(ws, {**m, "sha256": wrong, "href": f"{base}/single/{wrong}"})
    assert r["status"] == "hash_mismatch" and wrong not in store.objects
    # a missing file is a failure, not a crash
    assert HA.upload_one(ws, {**a, "rel": "outputs/x/gone.npz"})["status"] == "failed"


def test_stage_upload_and_restore(tmp_path, server, monkeypatch):
    store, base = server
    ws = tmp_path / "ws"
    rows = [write(ws, f"outputs/d/f{i}.npy", os.urandom(10 + 7 * i)) for i in range(5)]
    HA.atomic_json(ws / "outputs" / "host_archive" / "wave_t1.json",
                   {"generated_epoch": 0, "link_lifetime_s": 0, "files": [multipart(base, r, 16) for r in rows]})
    monkeypatch.chdir(ws)
    monkeypatch.delitem(sys.modules, "huggingface_hub", raising=False)
    monkeypatch.setenv("HF_TOKEN", "should-be-removed")
    out = HA.stage_upload("t1", workers=3)
    assert os.environ.get("HF_TOKEN") is None
    assert out["by_status"] == {"uploaded": 5} and not (ws / "outputs/host_archive/wave_t1.json").exists()
    done = json.loads((ws / "outputs/host_archive/done_t1.json").read_text())
    assert [d["rel"] for d in done["files"]] == sorted(r["rel"] for r in rows)
    # restore into a fresh root: one by link, one inline, one whose served bytes are wrong
    dest = tmp_path / "newhost"
    store.fail_get.add(rows[2]["sha256"])
    files = [{**r, "url": f"{base}/obj/{r['sha256']}"} for r in rows]
    files[1] = {**rows[1], "inline_b64": base64.b64encode((ws / rows[1]["rel"]).read_bytes()).decode()}
    HA.atomic_json(ws / "outputs/host_archive/links_t1.json", {"files": files})
    monkeypatch.setattr(HA.time, "sleep", lambda s: None)
    out = HA.stage_restore("t1", str(dest), workers=2)
    assert out["status"] == "INCOMPLETE" and out["failed"] == [rows[2]["rel"]]
    for i, r in enumerate(rows):
        p = dest / r["rel"]
        assert p.exists() == (i != 2)
        assert not p.with_name(p.name + ".part").exists()
        if i != 2:
            assert hashlib.sha256(p.read_bytes()).hexdigest() == r["sha256"]
    assert (ws / "outputs/host_archive/links_t1.json").exists()      # kept for a retry while a file failed
    store.fail_get.clear()
    out = HA.stage_restore("t1", str(dest), workers=2)
    assert out["status"] == "VERIFIED" and not (ws / "outputs/host_archive/links_t1.json").exists()
    assert sum(1 for f in out["files"] if f["how"] == "already_present") == 4
    # a different file already at a destination is a conflict and stays, unless --overwrite
    p0 = dest / rows[0]["rel"]
    p0.write_bytes(b"newer work")
    HA.atomic_json(ws / "outputs/host_archive/links_t1.json", {"files": files})
    out = HA.stage_restore("t1", str(dest), workers=2)
    assert out["status"] == "INCOMPLETE" and out["conflicts"] == [rows[0]["rel"]] and p0.read_bytes() == b"newer work"
    out = HA.stage_restore("t1", str(dest), workers=2, overwrite=True)
    assert out["status"] == "VERIFIED" and hashlib.sha256(p0.read_bytes()).hexdigest() == rows[0]["sha256"]


def test_upload_refuses_with_huggingface_hub_imported(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace())
    with pytest.raises(SystemExit, match="refusing"):
        HA.stage_upload("t", 1)


class FakeApi:
    def __init__(self, tree):
        self.tree = tree

    def create_repo(self, *a, **k):
        pass

    def repo_info(self, *a, **k):
        return SimpleNamespace(private=True)

    def list_repo_tree(self, repo, repo_type, recursive, path_in_repo=None):
        from huggingface_hub.errors import EntryNotFoundError
        hits = [(p, v) for p, v in self.tree.items() if path_in_repo is None or p.startswith(path_in_repo + "/")]
        if path_in_repo and not hits:                 # the hub answers 404 for a folder with no file yet
            raise EntryNotFoundError(f"{path_in_repo} does not exist on main")
        for path, (size, sha) in hits:
            yield SimpleNamespace(path=path, size=size, lfs=SimpleNamespace(sha256=sha) if sha else None)


class FakeSession:
    def __init__(self, tree, fail_commit_paths=()):
        self.tree, self.bodies, self.verified = tree, [], []

    def post(self, url, headers=None, json=None, data=None, timeout=None):
        if url.endswith("/commit/main"):
            lines = [__import__("json").loads(x) for x in data.decode().split("\n")]
            self.bodies.append(lines)
            for l in lines:
                if l["key"] == "lfsFile":
                    v = l["value"]
                    self.tree[v["path"]] = (v["size"], v["oid"])
                elif l["key"] == "file":
                    self.tree[l["value"]["path"]] = (len(base64.b64decode(l["value"]["content"])), None)
            return SimpleNamespace(status_code=200, json=lambda: {"commitOid": "c%03d" % len(self.bodies)}, text="")
        self.verified.append(url)
        return SimpleNamespace(status_code=200, text="")


def test_plan_and_commit(tmp_path, monkeypatch):
    monkeypatch.setattr(HA, "OUT", tmp_path / "ha")
    sha = lambda s: hashlib.sha256(s.encode()).hexdigest()            # noqa: E731
    rows = [{"rel": "outputs/a/1.npz", "bytes": 3, "mtime_ns": 1, "sha256": sha("one")},
            {"rel": "outputs/a/2.npz", "bytes": 3, "mtime_ns": 1, "sha256": sha("two")},
            {"rel": "outputs/b/2copy.npz", "bytes": 3, "mtime_ns": 1, "sha256": sha("two")},      # same bytes as 2.npz
            {"rel": "outputs/a/3.npz", "bytes": 5, "mtime_ns": 1, "sha256": sha("three")},       # already held
            {"rel": "outputs/a/4.npz", "bytes": 4, "mtime_ns": 1, "sha256": sha("four")}]
    batches = []

    def fake_batch(session, headers, repo, chunk):
        batches.append([x["rel"] for x in chunk])
        return [{"present": True} if x["rel"].endswith("3.npz") else
                {"href": "h" + x["rel"], "chunk_size": 16, "parts": ["p"], "expires_in": 36000,
                 **({"verify_href": "v" + x["rel"]} if x["rel"].endswith("1.npz") else {})} for x in chunk]

    tree = {}
    sess = FakeSession(tree)
    monkeypatch.setattr(HA, "host_manifest", lambda inc, exc, src="ws": [dict(r) for r in rows])
    monkeypatch.setattr(HA, "lfs_batch", fake_batch)
    monkeypatch.setattr(HA, "hf_api", lambda name: FakeApi(tree))
    monkeypatch.setattr(HA, "hf_session", lambda name: (sess, {"authorization": "secret"}))
    HA.stage_plan("u/r", "t2", ["outputs/**"], [], None)
    assert batches == [["outputs/a/1.npz", "outputs/a/2.npz", "outputs/a/3.npz", "outputs/a/4.npz"]]   # 2copy deduped
    wave = json.loads((tmp_path / "ha/wave_t2.json").read_text())
    plan = json.loads((tmp_path / "ha/plan_t2.json").read_text())
    assert [w["rel"] for w in wave["files"]] == ["outputs/a/1.npz", "outputs/a/2.npz", "outputs/a/4.npz"]
    assert not any("verify_href" in w or "secret" in json.dumps(w) for w in wave["files"])
    pl = {p["rel"]: p for p in plan["files"]}
    assert pl["outputs/b/2copy.npz"]["dup_of"] == "outputs/a/2.npz" and not pl["outputs/a/3.npz"]["upload"]
    assert pl["outputs/a/1.npz"]["verify_href"] == "voutputs/a/1.npz" and wave["link_lifetime_s"] == 36000
    # the host: 1 and 4 went up, 2 failed -> 2 and its copy are not archived; 3 (held) is
    HA.atomic_json(tmp_path / "ha/done_t2.json", {"files": [
        {"rel": "outputs/a/1.npz", "bytes": 3, "sha256": sha("one"), "status": "uploaded"},
        {"rel": "outputs/a/2.npz", "bytes": 3, "sha256": None, "status": "failed", "error": "boom"},
        {"rel": "outputs/a/4.npz", "bytes": 4, "sha256": sha("four"), "status": "uploaded"}]})
    assert HA.repo_tree(FakeApi(tree), "u/r", "ws") == {}             # a first commit: no ws/ folder yet
    tree["ws/outputs/a/3.npz"] = (5, sha("three"))                  # committed by an earlier wave: not re-committed
    rec = HA.stage_commit("u/r", "t2", None, per_commit=1)
    assert sess.verified == ["voutputs/a/1.npz"]
    lfs_commits = [b for b in sess.bodies if any(l["key"] == "lfsFile" for l in b)]
    assert [[l["value"]["path"] for l in b if l["key"] == "lfsFile"] for b in lfs_commits] == \
        [["ws/outputs/a/1.npz"], ["ws/outputs/a/4.npz"]]
    man = json.loads(base64.b64decode([l for l in sess.bodies[-1] if l["key"] == "file"][0]["value"]["content"]))
    assert sorted(r for r, _, _ in man["files"]) == ["outputs/a/1.npz", "outputs/a/3.npz", "outputs/a/4.npz"]
    assert sorted(m[0] for m in man["not_archived"]) == ["outputs/a/2.npz", "outputs/b/2copy.npz"]
    assert rec["status"] == "INCOMPLETE" and rec["archived"] == 3 and not rec["tree_mismatch"]
    with pytest.raises(SystemExit, match="not u/other"):
        HA.stage_commit("u/other", "t2", None, per_commit=1)


def test_src_ws_mirror(tmp_path, server, monkeypatch):
    """A wave from a sibling workspace (the host mirror): the host reads ws/../mirror/<rel>, the repo path is mirror/<rel>."""
    store, base = server
    proj = tmp_path / "proj"
    ws = proj / "ws"
    (ws / "outputs").mkdir(parents=True)
    row = write(proj / "mirror", "CRAG/webqsp/x.bin", os.urandom(40))
    asked = []
    monkeypatch.setattr(HA, "OUT", tmp_path / "ha")
    monkeypatch.setattr(HA, "host_manifest", lambda inc, exc, src="ws": asked.append(src) or [dict(row)])
    monkeypatch.setattr(HA, "lfs_batch", lambda s, h, repo, chunk: [
        {"href": f"{base}/complete/{x['sha256']}", "chunk_size": 16,
         "parts": [f"{base}/part/{x['sha256']}/{i + 1}" for i in range(3)], "expires_in": 36000} for x in chunk])
    tree = {}
    sess = FakeSession(tree)
    monkeypatch.setattr(HA, "hf_api", lambda name: FakeApi(tree))
    monkeypatch.setattr(HA, "hf_session", lambda name: (sess, {}))
    HA.stage_plan("u/r", "m1", ["CRAG/webqsp/*"], [], None, src_ws="mirror")
    assert asked == ["mirror"]
    wave = json.loads((tmp_path / "ha/wave_m1.json").read_text())
    assert wave["src_ws"] == "mirror"
    HA.atomic_json(ws / "outputs/host_archive/wave_m1.json", wave)
    monkeypatch.chdir(ws)
    monkeypatch.delitem(sys.modules, "huggingface_hub", raising=False)
    out = HA.stage_upload("m1", workers=1)
    assert out["by_status"] == {"uploaded": 1} and store.objects[row["sha256"]] == (proj / "mirror" / row["rel"]).read_bytes()
    HA.atomic_json(tmp_path / "ha/done_m1.json", json.loads((ws / "outputs/host_archive/done_m1.json").read_text()))
    rec = HA.stage_commit("u/r", "m1", None, per_commit=10)
    assert rec["status"] == "VERIFIED" and ("mirror/CRAG/webqsp/x.bin") in tree and not any(p.startswith("ws/") for p in tree)
    man = json.loads(base64.b64decode([l for l in sess.bodies[-1] if l["key"] == "file"][0]["value"]["content"]))
    assert man["prefix"] == "mirror/"


def test_check_tag_and_safe_rel():
    assert HA.check_tag("anchors-hotpot.v1") == "anchors-hotpot.v1"
    for bad in ("", "a/b", "a b", "../x"):
        with pytest.raises(SystemExit):
            HA.check_tag(bad)
    assert HA.safe_rel("outputs/a.npz") == "outputs/a.npz"


def test_drive_sends_the_laptops_copies_and_leaves_the_rest_to_the_host(tmp_path, server, monkeypatch):
    """--local-root: a planned file the laptop holds with the host's sha256 goes up from the laptop (under any of the
    roots); one it holds with other bytes, or not at all, is left to the host job. With every file here, no host job."""
    store, base = server
    lap, routed = tmp_path / "lap", tmp_path / "routed"
    a = write(lap, "outputs/x/a.json", b"alpha" * 10)
    write(lap, "outputs/x/b.json", b"BRAVO!")                       # the host's b.json: same size, other bytes
    b = {"rel": "outputs/x/b.json", "bytes": 6, "mtime_ns": 1, "sha256": hashlib.sha256(b"bravo!").hexdigest()}
    write(lap, "outputs/l12/c.npz", b"x" * 40)                      # a laptop file of the same size, other bytes ...
    c = write(routed, "outputs/l12/c.npz", os.urandom(40))         # ... and the host's copy under the second root
    d = {"rel": "outputs/x/d.rows.npz", "bytes": 9, "mtime_ns": 1, "sha256": hashlib.sha256(b"d" * 9).hexdigest()}
    rows = [dict(a, mtime_ns=5), b, dict(c, mtime_ns=7), d]         # the host's mtimes are not the laptop's
    monkeypatch.setattr(HA, "OUT", tmp_path / "ha")
    monkeypatch.setattr(HA, "host_manifest", lambda inc, exc, src="ws": [dict(r) for r in rows])
    monkeypatch.setattr(HA, "lfs_batch", lambda s, h, repo, chunk: [
        {"present": True} if x["sha256"] in store.objects else
        {"href": f"{base}/single/{x['sha256']}", "expires_in": 3600} for x in chunk])
    tree = {}
    sess = FakeSession(tree)
    monkeypatch.setattr(HA, "hf_api", lambda name: FakeApi(tree))
    monkeypatch.setattr(HA, "hf_session", lambda name: (sess, {}))
    calls = []

    def fake_host(tag, workers, cpus, mem, env):                     # the host: every TLS connection fails
        calls.append(tag)
        wave = json.loads((tmp_path / f"ha/wave_{tag}.json").read_text())
        assert sorted(f["rel"] for f in wave["files"]) == ["outputs/x/b.json", "outputs/x/d.rows.npz"]
        HA.atomic_json(tmp_path / f"ha/done_{tag}.json", {"tag": tag, "files": [
            {"rel": f["rel"], "bytes": f["bytes"], "sha256": None, "status": "failed", "error": "SSL"}
            for f in wave["files"]]})

    monkeypatch.setattr(HA, "host_upload", fake_host)
    HA.stage_drive("u/r", "t5", ["outputs/**"], [], None, workers=2, cpus=1, mem=0.5, per_commit=10,
                   local_roots=[lap, routed])
    assert calls == ["t5"] and not (tmp_path / "ha/wave_t5.json").exists()
    assert store.objects[a["sha256"]] == b"alpha" * 10 and store.objects[c["sha256"]] == (routed / c["rel"]).read_bytes()
    assert b["sha256"] not in store.objects                           # the laptop's bytes differ: never sent
    done = json.loads((tmp_path / "ha/done_t5.json").read_text())
    assert done["from_laptop"] == 2 and {f["rel"]: f["status"] for f in done["files"]} == {
        "outputs/l12/c.npz": "uploaded", "outputs/x/a.json": "uploaded", "outputs/x/b.json": "failed",
        "outputs/x/d.rows.npz": "failed"}
    rec = json.loads((tmp_path / "ha/commit_t5.json").read_text())
    assert rec["status"] == "INCOMPLETE" and rec["archived"] == 2
    assert sorted(m[0] for m in rec["not_uploaded"]) == ["outputs/x/b.json", "outputs/x/d.rows.npz"]
    assert tree["ws/outputs/x/a.json"] == (a["bytes"], a["sha256"]) and tree["ws/outputs/l12/c.npz"][1] == c["sha256"]
    # the laptop now holds the host's bytes of b and d: no host job, VERIFIED
    write(lap, "outputs/x/b.json", b"bravo!")
    write(lap, "outputs/x/d.rows.npz", b"d" * 9)
    calls.clear()
    HA.stage_drive("u/r", "t6", ["outputs/**"], [], None, workers=2, cpus=1, mem=0.5, per_commit=10,
                   local_roots=[lap, routed])
    assert calls == [] and json.loads((tmp_path / "ha/commit_t6.json").read_text())["status"] == "VERIFIED"
    assert json.loads((tmp_path / "ha/done_t6.json").read_text())["from_laptop"] == 2      # a and c were held already
