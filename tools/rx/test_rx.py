"""Tests for rx: the stream format, resume, globbing, path safety, the scheduler,
fetch conflict rules, and an end-to-end run against a local pseudo-host.

    python -m pytest tools/rx/test_rx.py -q
"""
import io
import json
import os
import subprocess
import sys
import time

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rx  # noqa: E402
import rx_agent as A  # noqa: E402


def write(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data if isinstance(data, bytes) else data.encode())


# ── globs and walking ──

@pytest.mark.parametrize("pat,path,ok", [
    ("*.py", "a.py", True), ("*.py", "src/deep/a.py", True), ("src/*.py", "src/a.py", True),
    ("src/*.py", "src/x/a.py", False), ("src/**", "src/x/a.py", True), ("src/", "src/x/a.py", True),
    ("**/b.txt", "b.txt", True), ("**/b.txt", "x/y/b.txt", True), ("a/**/c", "a/c", True),
    ("a/**/c", "a/b/b/c", True), ("[ab].txt", "b.txt", True), ("[ab].txt", "c.txt", False),
    ("outputs/**", "outputs", False), ("out?.log", "out1.log", True),
])
def test_glob(pat, path, ok):
    assert bool(A.Glob(pat).match(path)) is ok


def test_walk_files_prunes_and_skips_parts(tmp_path):
    for rel in ["a.py", "src/b.py", "src/__pycache__/c.pyc", "data/big.bin", "src/d.py.rxpart", "keep.txt"]:
        write(str(tmp_path / rel), "x")
    got = sorted(r for r, _, _ in A.walk_files(str(tmp_path), ["**"], ["__pycache__", "data/"]))
    assert got == ["a.py", "keep.txt", "src/b.py"]
    got = sorted(r for r, _, _ in A.walk_files(str(tmp_path), ["src", "keep.txt"]))
    assert got == ["keep.txt", "src/__pycache__/c.pyc", "src/b.py"]


@pytest.mark.parametrize("rel", ["../x", "/abs", "a/../b", "C:/x", "a\\b", "con.txt", "a/nul", "x.", "x ",
                                 "a|b", "f.rxpart", "", "a\x01b"])
def test_safe_rel_rejects(rel):
    with pytest.raises(A.RxError):
        A.safe_rel(rel)


def test_safe_rel_accepts():
    assert A.safe_rel("a/b-c_d.e/f g.txt") == "a/b-c_d.e/f g.txt"


# ── the stream format ──

def make_stream(root, rels, level=6, offsets=None):
    buf = io.BytesIO()
    w = A.StreamWriter(buf, level=level)
    for rel in rels:
        w.send_file(root, rel, (offsets or {}).get(rel, 0), A.sha256_path(os.path.join(root, rel)))
    w.end()
    return buf.getvalue(), w


def test_stream_round_trip(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    write(str(src / "text.txt"), "hello world\n" * 50000)          # compresses
    write(str(src / "rand.bin"), os.urandom(3 * A.CHUNK + 17))    # does not
    write(str(src / "empty"), b"")
    blob, w = make_stream(str(src), ["text.txt", "rand.bin", "empty"])
    assert w.wire < w.raw                                            # the text compressed
    res = A.receive_stream(A.StreamReader(io.BytesIO(blob)), str(dst))
    assert all(r["ok"] for r in res), res
    for rel in ["text.txt", "rand.bin", "empty"]:
        assert A.sha256_path(str(dst / rel)) == A.sha256_path(str(src / rel))
        assert os.stat(str(dst / rel)).st_mtime_ns // 10**6 == os.stat(str(src / rel)).st_mtime_ns // 10**6
    assert not [n for n in os.listdir(str(dst)) if A.PART in n]


def test_stream_resume_after_cut(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    write(str(src / "f.bin"), os.urandom(5 * A.CHUNK + 123))
    sha = A.sha256_path(str(src / "f.bin"))
    blob, _ = make_stream(str(src), ["f.bin"], level=0)
    with pytest.raises(EOFError):
        A.receive_stream(A.StreamReader(io.BytesIO(blob[:len(blob) // 2])), str(dst))
    off = A.partial_offset(str(dst / "f.bin"), sha)
    assert 0 < off < 5 * A.CHUNK
    assert not os.path.exists(str(dst / "f.bin"))
    blob2, w = make_stream(str(src), ["f.bin"], level=0, offsets={"f.bin": off})
    assert w.raw == os.path.getsize(str(src / "f.bin")) - off
    res = A.receive_stream(A.StreamReader(io.BytesIO(blob2)), str(dst))
    assert res[0]["ok"] and A.sha256_path(str(dst / "f.bin")) == sha


def test_stream_corruption_is_discarded(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    write(str(src / "f.bin"), os.urandom(2 * A.CHUNK))
    blob = bytearray(make_stream(str(src), ["f.bin"], level=0)[0])
    blob[len(blob) // 2] ^= 0xFF                                    # flip one payload byte
    res = A.receive_stream(A.StreamReader(io.BytesIO(bytes(blob))), str(dst))
    assert not res[0]["ok"] and "mismatch" in res[0]["error"] or "changed" in res[0]["error"]
    assert not os.path.exists(str(dst / "f.bin"))
    assert not os.path.exists(str(dst / "f.bin") + A.PART)


def test_stream_existing_file_replaced_atomically(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    write(str(src / "f.txt"), "new")
    write(str(dst / "f.txt"), "old")
    blob, _ = make_stream(str(src), ["f.txt"])
    A.receive_stream(A.StreamReader(io.BytesIO(blob)), str(dst))
    assert open(str(dst / "f.txt")).read() == "new"


# ── the scheduler ──

CAP = {"cpus": 32, "mem_gb": 120, "reserve_cpus": 2, "reserve_mem_gb": 8, "hold_after_s": 1800,
       "gpus": [{"index": 0}]}


def entry(key, enq, cpus=1, mem=1, gpus=0, assign=None):
    return {"key": key, "id": key, "enq": enq, "req": {"cpus": cpus, "mem_gb": mem, "gpus": gpus},
            **({"assign": assign} if assign else {})}


def test_decide_impossible():
    ok, _, why = A.decide(entry("a", 0, cpus=64), CAP, [], [], 10)
    assert not ok and why.startswith("impossible")
    ok, _, why = A.decide(entry("a", 0, gpus=2), CAP, [], [], 10)
    assert not ok and why.startswith("impossible")


def test_decide_admits_and_waits():
    ok, a, _ = A.decide(entry("a", 0, cpus=8, mem=16, gpus=1), CAP, [], [], 10)
    assert ok and a["gpus"] == [0]
    active = [dict(entry("a", 0, cpus=8, mem=16, gpus=1), assign=a)]
    ok, _, why = A.decide(entry("b", 1, gpus=1), CAP, active, [], 10)
    assert not ok and "waiting" in why
    ok, b, _ = A.decide(entry("b", 1, cpus=22, mem=96), CAP, active, [], 10)
    assert ok


def test_decide_fractional_gpu_sharing():
    ok, a, _ = A.decide(entry("a", 0, gpus=0.5), CAP, [], [], 10)
    assert ok and a["gpu_shares"] == {"0": 0.5}
    active = [dict(entry("a", 0, gpus=0.5), assign=a)]
    assert A.decide(entry("b", 1, gpus=0.5), CAP, active, [], 10)[0]
    assert not A.decide(entry("c", 1, gpus=0.75), CAP, active, [], 10)[0]


def test_decide_fifo_then_hold():
    active = [dict(entry("a", 0, cpus=20), assign={"cpus": 20, "mem_gb": 1, "gpus": [], "gpu_shares": {}})]
    big = entry("big", 1, cpus=20)                  # cannot fit now
    small = entry("small", 2, cpus=4)
    ok, _, _ = A.decide(small, CAP, active, [big, small], now=100)
    assert ok                                       # backfill while the big job is young
    ok, _, why = A.decide(small, CAP, active, [big, small], now=1 + 1801)
    assert not ok and "holding" in why              # the big job has waited long enough


# ── fetch conflict rules (no network: fetch_files decides before it connects) ──

class NoRemote:
    cfg = {}
    name = "none"

    def call(self, *a, **k):
        raise AssertionError("should not connect")


class P:
    def __init__(self, root):
        self.root = str(root)
        self.state = os.path.join(self.root, ".rx")
        self.name = "p"


def test_fetch_leaves_user_files_alone(tmp_path):
    proj = P(tmp_path)
    write(str(tmp_path / "out/same.txt"), "same")
    write(str(tmp_path / "out/mine.txt"), "edited by me")
    same_sha = A.sha256_path(str(tmp_path / "out/same.txt"))
    files = [["out/same.txt", 4, 0, same_sha], ["out/mine.txt", 5, 0, "0" * 64]]
    fetched, have, conflicts = rx.fetch_files(proj, NoRemote(), files, quiet=True)
    assert (fetched, have, conflicts) == (0, 1, ["out/mine.txt"])
    assert open(str(tmp_path / "out/mine.txt")).read() == "edited by me"


# ── end to end against a local pseudo-host ──

@pytest.fixture
def local_project(tmp_path, monkeypatch):
    cfgdir = tmp_path / "rxcfg"
    monkeypatch.setenv("RX_CONFIG_DIR", str(cfgdir))
    proj = tmp_path / "proj"
    proj.mkdir()
    write(str(proj / "rx.toml"), '[project]\nname = "t"\nhost = "loc"\n[push]\ninclude = ["git:tracked"]\n'
          '[run]\ncpus = 1\nmem_gb = 1\noutputs = ["out/**"]\n')
    write(str(proj / "job.py"), "import os, sys, time\nos.makedirs('out', exist_ok=True)\n"
          "open('out/result.json','w').write('{\"x\": %s}' % sys.argv[1])\n"
          "print('cpus', os.environ.get('RX_CPUS'), 'omp', os.environ.get('OMP_NUM_THREADS'))\n"
          "time.sleep(float(sys.argv[2]) if len(sys.argv) > 2 else 0)\nprint('finished')\n")
    write(str(proj / "untracked.txt"), "not pushed")
    for c in (["init", "-q"], ["add", "rx.toml", "job.py"],
              ["-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "init"]):
        subprocess.run(["git", "-C", str(proj)] + c, check=True, capture_output=True)
    assert rx.main(["setup", "loc", "--local", "--home", str(tmp_path / "home")]) == 0
    return proj, tmp_path / "home"


def test_end_to_end(local_project, capsys):
    proj, home = local_project
    assert rx.main(["-C", str(proj), "push"]) == 0
    ws = home / "projects" / "t" / "ws"
    assert (ws / "job.py").exists() and not (ws / "untracked.txt").exists()

    assert rx.main(["-C", str(proj), "run", "-w", "--", "python", "job.py", "7"]) == 0
    jid = capsys.readouterr().out.strip().splitlines()[-1]
    assert rx.main(["-C", str(proj), "logs", jid]) == 0
    out = capsys.readouterr().out
    assert "finished" in out and "cpus 1" in out and "omp 1" in out

    assert rx.main(["-C", str(proj), "fetch", jid]) == 0
    assert json.load(open(str(proj / "out/result.json"))) == {"x": 7}
    assert rx.main(["-C", str(proj), "fetch", jid]) == 0            # idempotent

    # a second run changes the output; rx's own earlier copy may be replaced
    assert rx.main(["-C", str(proj), "run", "-w", "--no-push", "--", "python", "job.py", "8"]) == 0
    jid2 = capsys.readouterr().out.strip().splitlines()[-1]
    assert rx.main(["-C", str(proj), "fetch", jid2]) == 0
    assert json.load(open(str(proj / "out/result.json"))) == {"x": 8}

    # a locally edited output is never clobbered
    write(str(proj / "out/result.json"), "mine")
    assert rx.main(["-C", str(proj), "fetch", jid]) == 2
    assert open(str(proj / "out/result.json")).read() == "mine"

    # cancel a long job
    assert rx.main(["-C", str(proj), "run", "--no-push", "--", "python", "job.py", "1", "120"]) == 0
    jid3 = capsys.readouterr().out.strip().splitlines()[-1]
    t0 = time.time()
    while time.time() - t0 < 30:
        r = rx.Remote(rx.load_host("loc")).call({"op": "status", "project": "t", "ids": [jid3]})
        if r["jobs"][0]["state"] == "running":
            break
        time.sleep(0.3)
    assert rx.main(["-C", str(proj), "cancel", jid3]) == 0
    assert rx.main(["-C", str(proj), "wait", jid3, "--interval", "0.5"]) == 1
    r = rx.Remote(rx.load_host("loc")).call({"op": "status", "project": "t", "ids": [jid3]})
    assert r["jobs"][0]["state"] == "cancelled"
    assert rx.main(["monitor", "--once"]) == 0
    assert "t" in capsys.readouterr().out
