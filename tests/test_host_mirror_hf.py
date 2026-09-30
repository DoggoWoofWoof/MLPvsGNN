"""The six-dataset host mirror (configs/host_mirror_six.yaml): the declared selection, the package's own pins, the host
copy and its hash check, the refusal of a mirror inside the workspace, the token rule of the link resolver, and the
declaration's shape."""

from __future__ import annotations

import hashlib
import json
import sys
import types
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "scripts",):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import host_mirror_hf as HM  # noqa: E402


def _write(p: Path, b: bytes) -> str:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b)
    return hashlib.sha256(b).hexdigest()


def _package(tmp: Path) -> Path:
    served = tmp / "data" / "final_canonical"
    for f in ("canonical.py", "pointer_resolver.py", "CANONICAL_FREEZE.json", "VERIFICATION.json", "HANDOFF.md"):
        _write(served / f, f.encode())
    h_nodes = _write(served / "ds1" / "nodes.jsonl", b'{"node_id": "a"}\n')
    _write(served / "ds1" / "queries" / "train.jsonl", b"{}\n")
    _write(served / "ds1" / "queries" / "lanes" / "LEGACY_CONTINUITY.jsonl", b"{}\n")
    h_shard = _write(served / "ds1" / "embeddings" / "dense" / "docs" / "shard_00000.npy", b"x" * 64)
    _write(served / "ds1" / "embeddings" / "dense" / "docs" / "manifest.json",
           json.dumps({"shards": [{"file": "shard_00000.npy", "sha256": h_shard}]}).encode())
    _write(served / "ds1" / "embeddings" / "splade" / "docs" / "shard_00000.npz", b"s" * 64)
    _write(served / "ds1" / "graph" / "structural.npz", b"g")
    _write(served / "ds1" / "retrieval_cache" / "dense_top1000.npz", b"c")
    _write(served / "ds1" / "_acquisition" / "raw.tgz", b"r")
    _write(served / "ds1" / "DATASET.json",
           json.dumps({"nodes": {"file": "data/final_canonical/ds1/nodes.jsonl", "sha256": h_nodes}}).encode())
    return served


SEL = {"datasets": ["ds1"], "package_files": ["canonical.py", "pointer_resolver.py", "CANONICAL_FREEZE.json", "VERIFICATION.json"],
       "dataset_subfolders": ["queries", "embeddings/dense", "graph", "retrieval_cache"]}


def test_selection_keeps_the_declared_reads_and_nothing_else(tmp_path):
    served = _package(tmp_path)
    rels = HM.selected(served, SEL)
    assert rels[:4] == SEL["package_files"]
    assert "ds1/nodes.jsonl" in rels and "ds1/DATASET.json" in rels
    assert "ds1/queries/lanes/LEGACY_CONTINUITY.jsonl" in rels
    assert "ds1/embeddings/dense/docs/shard_00000.npy" in rels and "ds1/embeddings/dense/docs/manifest.json" in rels
    assert not any("splade/docs" in r or "_acquisition" in r or r == "HANDOFF.md" for r in rels)


def test_a_missing_declared_subfolder_refuses(tmp_path):
    served = _package(tmp_path)
    with pytest.raises(SystemExit):
        HM.selected(served, {**SEL, "dataset_subfolders": SEL["dataset_subfolders"] + ["nope"]})


def test_package_pins_resolve_both_path_styles(tmp_path):
    served = _package(tmp_path)
    pins = HM.package_pins(served, ["ds1"])
    assert pins["ds1/nodes.jsonl"] == hashlib.sha256(b'{"node_id": "a"}\n').hexdigest()
    assert pins["ds1/embeddings/dense/docs/shard_00000.npy"] == hashlib.sha256(b"x" * 64).hexdigest()


def test_file_hashes_gives_the_git_blob_id_of_a_small_file(tmp_path):
    p = tmp_path / "f.json"
    p.write_bytes(b"hello\n")
    n, h, g = HM.file_hashes(p)
    assert (n, h) == (6, hashlib.sha256(b"hello\n").hexdigest())
    assert g == "ce013625030ba8dba906f756967f9e9ca394464a"      # `git hash-object` of "hello\n"


def test_fetch_one_copies_verifies_and_refuses_a_wrong_hash(tmp_path):
    src = tmp_path / "src.bin"
    src.write_bytes(b"abc" * 1000)
    row = {"rel": "ds1/x.bin", "bytes": 3000, "sha256": hashlib.sha256(b"abc" * 1000).hexdigest()}
    dest = tmp_path / "mirror" / "ds1" / "x.bin"
    r = HM.fetch_one(row, {"local": str(src)}, dest)
    assert r["verified"] and r["how"] == "host_copy" and dest.read_bytes() == b"abc" * 1000
    assert HM.fetch_one(row, {"local": str(src)}, dest)["how"] == "already_present"
    bad = dict(row, rel="ds1/y.bin", sha256="0" * 64)
    r2 = HM.fetch_one(bad, {"local": str(src)}, tmp_path / "mirror" / "ds1" / "y.bin", tries=1)
    assert not r2["verified"] and not (tmp_path / "mirror" / "ds1" / "y.bin").exists()
    assert not (tmp_path / "mirror" / "ds1" / "y.bin.part").exists()


def test_a_mirror_inside_the_workspace_is_refused(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit):
        HM.host_paths({"host": {"mirror_root": str(tmp_path / "mirror"), "read_only_sources": []}})
    ws = tmp_path / "ws"
    ws.mkdir()
    monkeypatch.chdir(ws)
    mirror, sources = HM.host_paths({"host": {"mirror_root": str(tmp_path / "mirror"), "read_only_sources": [str(tmp_path / "o")]}})
    assert mirror == tmp_path / "mirror" / "data" / "final_canonical"
    assert sources == [tmp_path / "o" / "data" / "final_canonical"]


def test_the_resolver_sends_authorization_to_huggingface_only(monkeypatch):
    seen = []

    class Resp:
        def __init__(self, code, loc=None, body=b""):
            self.status_code, self.headers, self.content = code, ({"Location": loc} if loc else {"Content-Length": str(len(body))}), body

        def close(self):
            pass

        def raise_for_status(self):
            raise RuntimeError(self.status_code)

    chain = {"https://huggingface.co/datasets/u/r/resolve/main/a": Resp(302, "https://cas-bridge.xethub.hf.co/x?X-Amz-Expires=3600"),
             "https://huggingface.co/datasets/u/r/resolve/main/b": Resp(200, body=b"tiny")}

    class Sess:
        def get(self, url, headers, **kw):
            seen.append((url, "authorization" in {k.lower() for k in headers}))
            return chain[url]

    fake = types.ModuleType("huggingface_hub.utils")
    fake.get_session = lambda: Sess()
    tokens_seen = []
    fake.build_hf_headers = lambda token=None: tokens_seen.append(token) or {"authorization": "Bearer <never printed>"}
    monkeypatch.setitem(sys.modules, "huggingface_hub.utils", fake)
    assert HM.resolve("https://huggingface.co/datasets/u/r/resolve/main/a") == ("url", "https://cas-bridge.xethub.hf.co/x?X-Amz-Expires=3600")
    assert HM.resolve("https://huggingface.co/datasets/u/r/resolve/main/b", "named") == ("inline", b"tiny")
    assert tokens_seen == [None, "named"]                    # the named token is the one handed to huggingface.co
    assert all(auth for url, auth in seen if url.startswith("https://huggingface.co/"))
    assert not any(auth for url, auth in seen if not url.startswith("https://huggingface.co/"))


def test_the_host_process_refuses_huggingface_hub(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "x")
    monkeypatch.setitem(sys.modules, "huggingface_hub", types.ModuleType("huggingface_hub"))
    with pytest.raises(SystemExit):
        HM.no_token()
    import os
    assert "HF_TOKEN" not in os.environ


def test_the_declaration_names_a_private_repo_a_mirror_outside_the_workspace_and_a_read_only_package():
    decl = yaml.safe_load((ROOT / "configs" / "host_mirror_six.yaml").read_text(encoding="utf-8"))
    assert decl["status"] in ("DECLARED_NOT_RUN", "RUN")
    assert decl["selection"]["datasets"] == ["squad", "musique", "metaqa", "webqsp", "hotpotqa", "2wiki"]
    assert "embeddings/splade" not in decl["selection"]["dataset_subfolders"]
    assert "/ws" not in decl["host"]["mirror_root"] and decl["host"]["mirror_root"].endswith("/mirror/CRAG")
    assert decl["package"]["root_from"] == "configs/m3b_controlled_comparison.yaml"
    assert len(decl["authorization"]["quotes"]) == 5


def test_only_a_declared_repo_is_used_and_a_named_token_must_be_stored(monkeypatch):
    decl = {"transport": {"repo": "u/one"}, "transport_amendment_1": {"route": "rx"},
            "transport_amendment_2": {"repo": "k/two", "token_name": "mpr"}}
    assert HM.declared_repo(decl, None) == "u/one" and HM.declared_repo(decl, "k/two") == "k/two"
    with pytest.raises(SystemExit):
        HM.declared_repo(decl, "someone/else")
    fake = types.ModuleType("huggingface_hub.utils")
    fake.get_stored_tokens = lambda: {"mpr": "<a stored token>"}
    monkeypatch.setitem(sys.modules, "huggingface_hub.utils", fake)
    assert HM.stored_token(None) is None and HM.stored_token("mpr") == "<a stored token>"
    with pytest.raises(SystemExit):
        HM.stored_token("missing")


def test_amendment_2_names_a_second_private_repo_and_writes_no_token():
    text = (ROOT / "configs" / "host_mirror_six.yaml").read_text(encoding="utf-8")
    decl = yaml.safe_load(text)
    am = decl["transport_amendment_2"]
    assert am["repo"] == "KK9895/mpr-six-mirror" and am["token_name"] == "mpr" and am["datasets"] == ["hotpotqa", "2wiki"]
    assert "hf_" not in text.replace("hf_hub", "").replace("host_mirror_hf", "")
