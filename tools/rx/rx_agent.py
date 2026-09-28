#!/usr/bin/env python3
"""rx_agent -- the remote half of rx. See README.md beside this file.

Standard library only; Python 3.8 or newer; Windows or POSIX.

rx.py runs this file over ssh as `python rx_agent.py rpc`. stdin carries one
JSON request line, optionally followed by a binary stream; stdout carries one
JSON response line, optionally followed by a binary stream. Every operation is
idempotent, so the answer to a dropped connection is to ask again.

Jobs are not children of the ssh session. `launch` starts a detached
supervisor -- through WMI on Windows, where OpenSSH kills the whole process
tree of a session when it closes, and through setsid on POSIX. The supervisor
queues the job, runs it inside a Windows Job Object (or its own process
group), writes a heartbeat every ten seconds and records the job's outputs
when it ends. The laptop can be gone for the whole run and collect the
results afterwards.

rx.py imports this module for the stream format, the glob matcher and the
scheduler, so both ends of a transfer run one implementation.
"""
from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import platform
import random
import re
import shutil
import stat
import string
import subprocess
import sys
import threading
import time
import traceback
import zlib

PROTOCOL = 1
IS_WIN = os.name == "nt"
MAGIC = b"RXS1\n"
CHUNK = 1 << 20                 # stream chunk, 1 MiB
SYNC_EVERY = 64 << 20           # fsync a partial file every 64 MiB
HEARTBEAT_S = 10.0
LOST_AFTER_S = 120.0
ACTIVE_STATES = ("launching", "queued", "running", "finishing")
TERMINAL_STATES = ("done", "failed", "cancelled", "lost", "error")
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,119}$")
TMP_RE = re.compile(r"\.tmp-\d+-[a-z0-9]{6}$")
THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
               "NUMEXPR_NUM_THREADS", "NUMEXPR_MAX_THREADS", "VECLIB_MAXIMUM_THREADS",
               "RAYON_NUM_THREADS", "NUMBA_NUM_THREADS")
OUTPUT_EXCLUDES = ("__pycache__", "*.pyc", ".git", ".pytest_cache", ".ruff_cache", ".mypy_cache")
PART = ".rxpart"
SIDE = ".rxpart.json"

_AGENT_SHA = None


def agent_sha():
    """sha256 of this file's bytes: the version the laptop compares before it uploads."""
    global _AGENT_SHA
    if _AGENT_SHA is None:
        with open(os.path.abspath(__file__), "rb") as f:
            _AGENT_SHA = hashlib.sha256(f.read()).hexdigest()
    return _AGENT_SHA


class RxError(Exception):
    def __init__(self, msg, kind="error"):
        super().__init__(msg)
        self.kind = kind


class ProtocolError(RxError):
    def __init__(self, msg):
        super().__init__(msg, "protocol")


# ─────────────────────────────── small helpers ───────────────────────────────

def _rand(n=6):
    return "".join(random.choice(string.ascii_lowercase + string.digits) for _ in range(n))


def read_json(path, default=None):
    """A missing or unreadable file is `default`. Writers always rename into
    place, so a reader never sees half a file; PermissionError on Windows means
    a rename is in flight, and is retried."""
    for i in range(20):
        try:
            with open(path, "rb") as f:
                data = f.read()
            return json.loads(data.decode("utf-8")) if data.strip() else default
        except FileNotFoundError:
            return default
        except PermissionError:
            time.sleep(0.03 * (i + 1))
        except (ValueError, UnicodeDecodeError):
            return default
    return default


def replace_retry(src, dst, tries=40):
    """os.replace, retried: on Windows a reader holding `dst` open makes the rename fail briefly."""
    for i in range(tries):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if i == tries - 1:
                raise
            time.sleep(0.02 * (i + 1))


def write_json(path, obj, fsync=False):
    data = json.dumps(obj, indent=1, sort_keys=True).encode("utf-8")
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    tmp = "%s.tmp-%d-%s" % (path, os.getpid(), _rand())
    with open(tmp, "wb") as f:
        f.write(data)
        if fsync:
            f.flush()
            os.fsync(f.fileno())
    try:
        replace_retry(tmp, path)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def remove_quiet(path):
    try:
        os.remove(path)
        return True
    except OSError:
        return False


def sha256_path(path, start=0, h=None):
    h = h or hashlib.sha256()
    with open(path, "rb") as f:
        if start:
            f.seek(start)
        while True:
            b = f.read(CHUNK)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def check_name(name, what="name"):
    if not isinstance(name, str) or not NAME_RE.match(name):
        raise RxError("invalid %s %r (letters, digits, '.', '_', '-'; up to 120)" % (what, name), "name")
    return name


_WIN_RESERVED = {"CON", "PRN", "AUX", "NUL"} | {"COM%d" % i for i in range(1, 10)} | \
    {"LPT%d" % i for i in range(1, 10)}


def safe_rel(rel):
    """A relative path with '/' separators that cannot leave its root."""
    bad = (not isinstance(rel, str) or not rel or "\x00" in rel or "\\" in rel
           or rel.startswith("/") or rel.endswith(PART) or rel.endswith(SIDE))
    if not bad:
        for p in rel.split("/"):
            if p in ("", ".", "..") or ":" in p or any(ord(c) < 32 for c in p):
                bad = True
                break
            if IS_WIN and (p.split(".")[0].upper() in _WIN_RESERVED or p[-1] in ". "
                           or any(c in p for c in '<>"|?*')):
                bad = True
                break
    if bad:
        raise RxError("unsafe or reserved path %r" % (rel,), "path")
    return rel


def join_rel(root, rel):
    return os.path.join(root, *safe_rel(rel).split("/"))


def fmt_bytes(n):
    n = float(n or 0)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024 or unit == "TB":
            return ("%.0f %s" if unit == "B" else "%.1f %s") % (n, unit)
        n /= 1024.0
    return "%.1f TB" % n


# ─────────────────────────────── file locks ───────────────────────────────

class FileLock:
    """An exclusive advisory lock on a file; the OS drops it when the holder dies."""

    def __init__(self, path, timeout=120.0):
        self.path = path
        self.timeout = timeout
        self.fd = None

    def acquire(self, blocking=True):
        d = os.path.dirname(self.path)
        if d:
            os.makedirs(d, exist_ok=True)
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
        deadline = time.time() + self.timeout
        while True:
            try:
                if IS_WIN:
                    import msvcrt
                    os.lseek(fd, 0, 0)
                    msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                self.fd = fd
                return True
            except OSError:
                if not blocking or time.time() > deadline:
                    os.close(fd)
                    if not blocking:
                        return False
                    raise RxError("timed out waiting for the lock %s" % self.path, "lock")
                time.sleep(0.05 + random.random() * 0.1)

    def release(self):
        if self.fd is None:
            return
        try:
            if IS_WIN:
                import msvcrt
                os.lseek(self.fd, 0, 0)
                try:
                    msvcrt.locking(self.fd, msvcrt.LK_UNLCK, 1)
                except OSError:
                    pass
            else:
                import fcntl
                fcntl.flock(self.fd, fcntl.LOCK_UN)
        finally:
            os.close(self.fd)
            self.fd = None

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *exc):
        self.release()


# ─────────────────────────────── globs and walking ───────────────────────────────

_GLOB_CHARS = set("*?[")


def _seg_regex(seg):
    out, i, n = [], 0, len(seg)
    while i < n:
        c = seg[i]
        if c == "*":
            out.append("[^/]*")
        elif c == "?":
            out.append("[^/]")
        elif c == "[":
            j = i + 1
            if j < n and seg[j] in "!^":
                j += 1
            if j < n and seg[j] == "]":
                j += 1
            while j < n and seg[j] != "]":
                j += 1
            if j >= n:
                out.append(re.escape(c))
            else:
                body = seg[i + 1:j]
                if body[:1] in ("!", "^"):
                    body = "^" + body[1:]
                out.append("[" + body.replace("\\", "\\\\") + "]")
                i = j
        else:
            out.append(re.escape(c))
        i += 1
    return "".join(out)


def _segs_regex(segs):
    rx = []
    for k, s in enumerate(segs):
        last = k == len(segs) - 1
        if s == "**":
            rx.append(".*" if last else "(?:[^/]+/)*")
        else:
            rx.append(_seg_regex(s) + ("" if last else "/"))
    return re.compile("^" + "".join(rx) + "$", re.S)


class Glob:
    """Path globs over '/'-separated relative paths.

    `*` and `?` stay inside one path segment, `**` as a whole segment spans any
    number of them, `[...]` is a character class. A pattern with no '/' matches
    the basename at any depth (as in .gitignore), and a trailing '/' means
    everything under that directory."""

    def __init__(self, pat):
        pat = pat.strip().replace("\\", "/")
        while pat.startswith("./"):
            pat = pat[2:]
        if pat.endswith("/"):
            pat += "**"
        self.pat = pat
        segs = [s for s in pat.split("/") if s != ""] or ["**"]
        self.basename_only = len(segs) == 1 and "/" not in pat
        self.literal_prefix = []
        for s in segs:
            if any(ch in _GLOB_CHARS for ch in s):
                break
            self.literal_prefix.append(s)
        self.is_literal = len(self.literal_prefix) == len(segs)
        self.rx = _segs_regex(segs)
        self.dir_rx = _segs_regex(segs[:-1]) if len(segs) >= 2 and segs[-1] == "**" else None

    def match(self, rel):
        if self.basename_only:
            return bool(self.rx.match(rel.rsplit("/", 1)[-1]))
        return bool(self.rx.match(rel))

    def prunes_dir(self, rel_dir):
        """True when every path under this directory matches (so an exclude can skip the walk)."""
        if self.basename_only:
            return bool(self.rx.match(rel_dir.rsplit("/", 1)[-1]))
        return bool(self.dir_rx is not None and self.dir_rx.match(rel_dir))


def _is_link_dir(entry):
    if entry.is_symlink():
        return True
    if IS_WIN:  # junctions and other reparse points: never descend (loops, cloud placeholders)
        try:
            return bool(entry.stat(follow_symlinks=False).st_file_attributes & 0x400)
        except (OSError, AttributeError):
            return True
    return False


def _walk(base_abs, base_rel, exc):
    stack = [(base_abs, base_rel)]
    while stack:
        d, drel = stack.pop()
        try:
            it = os.scandir(d)
        except OSError:
            continue
        with it:
            for e in it:
                rel = e.name if not drel else drel + "/" + e.name
                try:
                    if e.is_dir(follow_symlinks=False):
                        if _is_link_dir(e) or any(x.prunes_dir(rel) for x in exc):
                            continue
                        stack.append((e.path, rel))
                        continue
                    if e.is_symlink():
                        st = os.stat(e.path)
                        if not stat.S_ISREG(st.st_mode):
                            continue
                    elif e.is_file(follow_symlinks=False):
                        st = e.stat(follow_symlinks=False)
                    else:
                        continue
                except OSError:
                    continue
                if rel.endswith(PART) or rel.endswith(SIDE) or TMP_RE.search(e.name):
                    continue
                if any(x.match(rel) for x in exc):
                    continue
                yield rel, e.path, st


def walk_files(root, includes, excludes=()):
    """Yield (rel, abs_path, stat) for regular files under root matching an include
    and no exclude. Each file is yielded once."""
    exc = [Glob(p) for p in excludes]
    seen = set()
    for pat in includes:
        g = Glob(pat)
        if g.is_literal:
            rel = "/".join(g.literal_prefix)
            ap = os.path.join(root, *g.literal_prefix)
            if os.path.isfile(ap):
                if rel not in seen and not any(x.match(rel) for x in exc):
                    seen.add(rel)
                    yield rel, ap, os.stat(ap)
                continue
            if not os.path.isdir(ap):
                continue
            g = Glob(rel + "/**")
        base_rel = "/".join(g.literal_prefix) if not g.basename_only else ""
        base_abs = os.path.join(root, *base_rel.split("/")) if base_rel else root
        if base_rel and any(x.prunes_dir(base_rel) for x in exc):
            continue
        for rel, ap, st in _walk(base_abs, base_rel, exc):
            if rel not in seen and g.match(rel):
                seen.add(rel)
                yield rel, ap, st


# ─────────────────────────────── the stream format ───────────────────────────────
#
#   MAGIC, then records. A record is one JSON line, sometimes followed by a payload:
#     {"t":"f","path","size","offset","mtime_ns","sha256"}   a file starts at `offset`
#     {"t":"c","n":raw}            + n raw bytes            a chunk
#     {"t":"c","n":raw,"z":clen}   + clen zlib bytes        a compressed chunk
#     {"t":"e","sha256"}                                    the file ends; sha of all of it as read
#     {"t":"missing","path"}                                the sender has no such file
#     {"t":"end"}                                           the stream ends
#
# The receiver writes `<path>.rxpart` (plus a `.rxpart.json` sidecar naming the
# target sha) and renames it into place only after the size and the sha agree,
# so an interrupted transfer leaves a prefix that the next attempt resumes.

class StreamWriter:
    def __init__(self, fp, level=6, progress=None):
        self.fp = fp
        self.level = level
        self.progress = progress
        self.raw = 0
        self.wire = 0
        self._put(MAGIC)

    def _put(self, b):
        self.fp.write(b)
        self.wire += len(b)

    def record(self, obj, payload=b""):
        self._put((json.dumps(obj, separators=(",", ":")) + "\n").encode("utf-8"))
        if payload:
            self._put(payload)

    def send_file(self, root, rel, offset=0, expect_sha=None):
        path = join_rel(root, rel)
        try:
            f = open(path, "rb")
        except (FileNotFoundError, IsADirectoryError, NotADirectoryError, PermissionError):
            self.record({"t": "missing", "path": rel})
            return None
        with f:
            st = os.fstat(f.fileno())
            size = st.st_size
            offset = int(offset or 0)
            if offset < 0 or offset > size:
                offset = 0
            h = hashlib.sha256()
            left = offset
            while left:
                b = f.read(min(CHUNK, left))
                if not b:
                    break
                h.update(b)
                left -= len(b)
            self.record({"t": "f", "path": rel, "size": size, "offset": offset,
                         "mtime_ns": st.st_mtime_ns, "sha256": expect_sha})
            left = size - offset
            misses, trying = 0, self.level > 0
            while left > 0:
                b = f.read(min(CHUNK, left))
                if not b:
                    break
                left -= len(b)
                h.update(b)
                z = None
                if trying and len(b) >= 256:
                    c = zlib.compress(b, self.level)
                    if len(c) < 0.9 * len(b):
                        z, misses = c, 0
                    else:
                        misses += 1
                        trying = misses < 4
                if z is not None:
                    self.record({"t": "c", "n": len(b), "z": len(z)}, z)
                else:
                    self.record({"t": "c", "n": len(b)}, b)
                self.raw += len(b)
                if self.progress:
                    self.progress(len(b))
            sha = h.hexdigest()
            self.record({"t": "e", "sha256": sha})
            return sha

    def end(self):
        self.record({"t": "end"})
        self.fp.flush()


class StreamReader:
    def __init__(self, fp):
        self.fp = fp
        m = self.read_exact(len(MAGIC))
        if m != MAGIC:
            raise ProtocolError("not an rx stream (magic %r)" % (m,))

    def read_exact(self, n):
        parts, got = [], 0
        while got < n:
            b = self.fp.read(n - got)
            if not b:
                raise EOFError("stream ended after %d of %d bytes" % (got, n))
            parts.append(b)
            got += len(b)
        return b"".join(parts)

    def record(self):
        line = self.fp.readline(1 << 20)
        if not line:
            raise EOFError("stream ended")
        if not line.endswith(b"\n"):
            raise EOFError("stream ended inside a record")
        try:
            return json.loads(line.decode("utf-8"))
        except ValueError:
            raise ProtocolError("bad record %r" % line[:80])


def partial_offset(dest, sha):
    """Bytes of `dest` already received for content `sha`, from an earlier interrupted transfer."""
    meta = read_json(dest + SIDE)
    if not meta or not sha or meta.get("sha256") != sha:
        return 0
    try:
        return os.path.getsize(dest + PART)
    except OSError:
        return 0


def _discard(reader, until_end_of_file=True):
    """Consume the rest of the current file's records."""
    while True:
        c = reader.record()
        if c.get("t") == "c":
            reader.read_exact(c["z"] if c.get("z") is not None else c["n"])
        elif c.get("t") == "e":
            return


def receive_stream(reader, root, on_done=None, replace_tries=10):
    """Receive every file of a stream into `root`. Returns one result per file.

    A file whose bytes do not hash to what the sender promised is discarded,
    never renamed into place. EOFError propagates with the partial file kept."""
    results = []
    while True:
        h = reader.record()
        t = h.get("t")
        if t == "end":
            return results
        if t == "missing":
            results.append({"path": h.get("path"), "ok": False, "error": "missing at the sender"})
            continue
        if t != "f":
            raise ProtocolError("expected a file record, got %r" % (t,))
        rel = safe_rel(h["path"])
        dest = join_rel(root, rel)
        size, offset = int(h["size"]), int(h.get("offset") or 0)
        expect, mtime_ns = h.get("sha256"), h.get("mtime_ns")
        part, side = dest + PART, dest + SIDE
        os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
        hsh = hashlib.sha256()
        if offset:
            usable = (expect and partial_offset(dest, expect) >= offset)
            if not usable:
                _discard(reader)
                results.append({"path": rel, "ok": False, "error": "resume offset no longer valid"})
                continue
            f = open(part, "r+b")
            left = offset
            while left:
                b = f.read(min(CHUNK, left))
                if not b:
                    break
                hsh.update(b)
                left -= len(b)
            f.truncate(offset)
            f.seek(offset)
        else:
            write_json(side, {"sha256": expect, "size": size, "mtime_ns": mtime_ns,
                              "started": time.time()})
            f = open(part, "wb")
        received, since_sync, trailer = offset, 0, None
        try:
            while True:
                c = reader.record()
                ct = c.get("t")
                if ct == "c":
                    n, z = int(c["n"]), c.get("z")
                    payload = reader.read_exact(int(z) if z is not None else n)
                    data = zlib.decompress(payload) if z is not None else payload
                    if len(data) != n:
                        raise ProtocolError("chunk of %s decoded to %d bytes, not %d" % (rel, len(data), n))
                    f.write(data)
                    hsh.update(data)
                    received += n
                    since_sync += n
                    if since_sync >= SYNC_EVERY:
                        f.flush()
                        os.fsync(f.fileno())
                        since_sync = 0
                elif ct == "e":
                    trailer = c.get("sha256")
                    break
                else:
                    raise ProtocolError("unexpected record %r inside %s" % (ct, rel))
            f.flush()
            os.fsync(f.fileno())
        finally:
            f.close()
        got = hsh.hexdigest()
        if received != size or got != trailer or (expect and got != expect):
            remove_quiet(part)
            remove_quiet(side)
            why = ("size %d != %d" % (received, size)) if received != size else \
                  ("changed while it was read" if got != trailer else "sha mismatch")
            results.append({"path": rel, "ok": False, "error": why})
            continue
        if mtime_ns:
            os.utime(part, ns=(time.time_ns() if hasattr(time, "time_ns") else int(time.time() * 1e9),
                               int(mtime_ns)))
        try:
            replace_retry(part, dest, tries=replace_tries)
        except OSError as e:
            results.append({"path": rel, "ok": False, "error": "could not replace (in use?): %s" % e})
            continue
        remove_quiet(side)
        st = os.stat(dest)
        if on_done:
            on_done(rel, st.st_size, st.st_mtime_ns, got)
        results.append({"path": rel, "ok": True, "size": size, "sha256": got})


def copy_via_part(src, dest, mtime_ns=None):
    """Copy a file into place the same way a transfer lands: part file, then rename."""
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    part = dest + PART
    shutil.copyfile(src, part)
    if mtime_ns:
        os.utime(part, ns=(int(time.time() * 1e9), int(mtime_ns)))
    replace_retry(part, dest)
    remove_quiet(dest + SIDE)


# ─────────────────────────────── the remote home ───────────────────────────────

def home_dir():
    h = os.environ.get("RX_HOME")
    if h:
        return os.path.abspath(h)
    here = os.path.dirname(os.path.abspath(__file__))
    if os.path.basename(here) == "agent":
        return os.path.dirname(here)
    return os.path.join(os.path.expanduser("~"), "rx")


class Home:
    """RX_HOME/
         agent/rx_agent-<sha12>.py   immutable agent versions
         python/<version>/           standalone interpreters
         envs/<name>.json            pointer to envs/<name>@<hash12>/
         host.json                   capacity the scheduler shares out
         sched.lock  queue/  active/
         projects/<p>/<ws>/          workspaces (default `ws`)
         projects/<p>/index-<ws>.json  size, mtime, sha, pushed-flag per file
         projects/<p>/jobs/<id>/     spec, status, heartbeat, output.log, outputs"""

    def __init__(self, root=None):
        self.root = root or home_dir()

    def p(self, *parts):
        return os.path.join(self.root, *parts)

    def project(self, name):
        return self.p("projects", check_name(name, "project"))

    def ws(self, project, ws="ws"):
        return os.path.join(self.project(project), check_name(ws or "ws", "workspace"))

    def jobs(self, project):
        return os.path.join(self.project(project), "jobs")

    def jobdir(self, project, jid):
        return os.path.join(self.jobs(project), check_name(jid, "job id"))

    def projects(self):
        try:
            return sorted(n for n in os.listdir(self.p("projects")) if NAME_RE.match(n))
        except OSError:
            return []


class Index:
    """Per-workspace file index: rel -> [size, mtime_ns, sha256, pushed]. A cached sha is
    trusted only while size and mtime_ns still match the file on disk."""

    def __init__(self, home, project, ws="ws"):
        self.path = os.path.join(home.project(project), "index-%s.json" % check_name(ws or "ws"))
        self.root = home.ws(project, ws)

    def load(self):
        d = read_json(self.path, {}) or {}
        return d.get("files", {})

    def update(self, changes=None, removals=()):
        if not changes and not removals:
            return
        with FileLock(self.path + ".lock"):
            files = self.load()
            files.update(changes or {})
            for r in removals:
                files.pop(r, None)
            write_json(self.path, {"v": 1, "files": files})


def cached_sha(files_idx, rel, path, st):
    e = files_idx.get(rel)
    if e and e[0] == st.st_size and e[1] == st.st_mtime_ns and e[2]:
        return e[2], False
    return sha256_path(path), True


# ─────────────────────────────── Windows plumbing ───────────────────────────────

if IS_WIN:
    import ctypes
    from ctypes import wintypes

    _k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _ntdll = ctypes.WinDLL("ntdll")

    class _FILETIME(ctypes.Structure):
        _fields_ = [("lo", wintypes.DWORD), ("hi", wintypes.DWORD)]

    class _MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [("dwLength", wintypes.DWORD), ("dwMemoryLoad", wintypes.DWORD),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

    class _IO_COUNTERS(ctypes.Structure):
        _fields_ = [(n, ctypes.c_ulonglong) for n in (
            "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
            "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

    class _JOB_BASIC_LIMIT(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                    ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                    ("SchedulingClass", wintypes.DWORD)]

    class _JOB_EXT_LIMIT(ctypes.Structure):
        _fields_ = [("BasicLimitInformation", _JOB_BASIC_LIMIT), ("IoInfo", _IO_COUNTERS),
                    ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

    class _JOB_ACCOUNTING(ctypes.Structure):
        _fields_ = [("TotalUserTime", ctypes.c_int64), ("TotalKernelTime", ctypes.c_int64),
                    ("ThisPeriodTotalUserTime", ctypes.c_int64), ("ThisPeriodTotalKernelTime", ctypes.c_int64),
                    ("TotalPageFaultCount", wintypes.DWORD), ("TotalProcesses", wintypes.DWORD),
                    ("ActiveProcesses", wintypes.DWORD), ("TotalTerminatedProcesses", wintypes.DWORD)]

    class _JOB_PIDS(ctypes.Structure):
        _fields_ = [("NumberOfAssignedProcesses", wintypes.DWORD),
                    ("NumberOfProcessIdsInList", wintypes.DWORD),
                    ("ProcessIdList", ctypes.c_size_t * 2048)]

    class _PMC_EX(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
                    ("PrivateUsage", ctypes.c_size_t)]

    _k32.CreateJobObjectW.restype = wintypes.HANDLE
    _k32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    _k32.SetInformationJobObject.restype = wintypes.BOOL
    _k32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    _k32.QueryInformationJobObject.restype = wintypes.BOOL
    _k32.QueryInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p,
                                               wintypes.DWORD, ctypes.POINTER(wintypes.DWORD)]
    _k32.AssignProcessToJobObject.restype = wintypes.BOOL
    _k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    _k32.TerminateJobObject.restype = wintypes.BOOL
    _k32.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
    _k32.CloseHandle.argtypes = [wintypes.HANDLE]
    _k32.OpenProcess.restype = wintypes.HANDLE
    _k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    _k32.K32GetProcessMemoryInfo.restype = wintypes.BOOL
    _k32.K32GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
    _k32.GetTickCount64.restype = ctypes.c_ulonglong
    _k32.SetThreadExecutionState.restype = wintypes.DWORD
    _k32.SetThreadExecutionState.argtypes = [wintypes.DWORD]
    _k32.GetSystemTimes.restype = wintypes.BOOL
    _k32.GetSystemTimes.argtypes = [ctypes.POINTER(_FILETIME)] * 3
    _k32.GlobalMemoryStatusEx.restype = wintypes.BOOL
    _k32.GlobalMemoryStatusEx.argtypes = [ctypes.POINTER(_MEMORYSTATUSEX)]
    _k32.GetCurrentProcess.restype = wintypes.HANDLE
    _k32.IsProcessInJob.restype = wintypes.BOOL
    _k32.IsProcessInJob.argtypes = [wintypes.HANDLE, wintypes.HANDLE, ctypes.POINTER(wintypes.BOOL)]
    _ntdll.NtResumeProcess.restype = ctypes.c_long
    _ntdll.NtResumeProcess.argtypes = [wintypes.HANDLE]

    CREATE_SUSPENDED = 0x00000004
    CREATE_NO_WINDOW = 0x08000000
    DETACHED_PROCESS = 0x00000008
    CREATE_NEW_PROCESS_GROUP = 0x00000200
    CREATE_BREAKAWAY_FROM_JOB = 0x01000000
    JOB_LIMIT_KILL_ON_CLOSE = 0x00002000
    JOB_LIMIT_JOB_MEMORY = 0x00000200

    class WinJob:
        """A Job Object holding a job's whole process tree: one call kills all of
        it, and the kernel keeps its CPU time and peak memory."""

        def __init__(self, mem_limit_bytes=None):
            self.h = _k32.CreateJobObjectW(None, None)
            if not self.h:
                raise ctypes.WinError(ctypes.get_last_error())
            info = _JOB_EXT_LIMIT()
            info.BasicLimitInformation.LimitFlags = JOB_LIMIT_KILL_ON_CLOSE
            if mem_limit_bytes:
                info.BasicLimitInformation.LimitFlags |= JOB_LIMIT_JOB_MEMORY
                info.JobMemoryLimit = int(mem_limit_bytes)
            if not _k32.SetInformationJobObject(self.h, 9, ctypes.byref(info), ctypes.sizeof(info)):
                raise ctypes.WinError(ctypes.get_last_error())
            self.lock = threading.Lock()

        def assign_and_resume(self, proc_handle):
            try:
                if not _k32.AssignProcessToJobObject(self.h, int(proc_handle)):
                    raise ctypes.WinError(ctypes.get_last_error())
            finally:
                _ntdll.NtResumeProcess(int(proc_handle))

        def pids(self):
            buf = _JOB_PIDS()
            with self.lock:
                if not self.h:
                    return []
                if not _k32.QueryInformationJobObject(self.h, 3, ctypes.byref(buf), ctypes.sizeof(buf), None):
                    return []
            return [int(buf.ProcessIdList[i]) for i in range(buf.NumberOfProcessIdsInList)]

        def sample(self):
            acc, ext = _JOB_ACCOUNTING(), _JOB_EXT_LIMIT()
            with self.lock:
                if not self.h:
                    return {}
                _k32.QueryInformationJobObject(self.h, 1, ctypes.byref(acc), ctypes.sizeof(acc), None)
                _k32.QueryInformationJobObject(self.h, 9, ctypes.byref(ext), ctypes.sizeof(ext), None)
            rss = 0
            for pid in self.pids():
                h = _k32.OpenProcess(0x1000 | 0x0010, False, pid)  # QUERY_LIMITED | VM_READ
                if not h:
                    continue
                try:
                    pmc = _PMC_EX()
                    pmc.cb = ctypes.sizeof(pmc)
                    if _k32.K32GetProcessMemoryInfo(h, ctypes.byref(pmc), pmc.cb):
                        rss += pmc.WorkingSetSize
                finally:
                    _k32.CloseHandle(h)
            return {"cpu_s": round((acc.TotalUserTime + acc.TotalKernelTime) / 1e7, 1),
                    "procs": int(acc.ActiveProcesses), "procs_total": int(acc.TotalProcesses),
                    "mem_gb": round(rss / 2 ** 30, 2),
                    "peak_commit_gb": round(ext.PeakJobMemoryUsed / 2 ** 30, 2)}

        def terminate(self):
            with self.lock:
                if self.h:
                    _k32.TerminateJobObject(self.h, 1)

        def close(self):
            with self.lock:
                if self.h:
                    _k32.CloseHandle(self.h)
                    self.h = None

    def process_in_job():
        r = wintypes.BOOL()
        _k32.IsProcessInJob(_k32.GetCurrentProcess(), None, ctypes.byref(r))
        return bool(r.value)


def keep_awake(on):
    """Ask Windows not to sleep while a job runs (per-thread; ends with the thread)."""
    if IS_WIN:
        try:
            _k32.SetThreadExecutionState(0x80000000 | (0x00000001 if on else 0))
        except Exception:
            pass


def boot_time():
    try:
        if IS_WIN:
            return time.time() - _k32.GetTickCount64() / 1000.0
        with open("/proc/uptime") as f:
            return time.time() - float(f.read().split()[0])
    except Exception:
        return None


def mem_info():
    try:
        if IS_WIN:
            m = _MEMORYSTATUSEX()
            m.dwLength = ctypes.sizeof(m)
            _k32.GlobalMemoryStatusEx(ctypes.byref(m))
            return {"mem_total_gb": round(m.ullTotalPhys / 2 ** 30, 1),
                    "mem_avail_gb": round(m.ullAvailPhys / 2 ** 30, 1)}
        vals = {}
        with open("/proc/meminfo") as f:
            for line in f:
                k, v = line.split(":", 1)
                vals[k] = int(v.split()[0]) * 1024
        return {"mem_total_gb": round(vals["MemTotal"] / 2 ** 30, 1),
                "mem_avail_gb": round(vals.get("MemAvailable", vals.get("MemFree", 0)) / 2 ** 30, 1)}
    except Exception:
        return {}


def _cpu_times():
    if IS_WIN:
        i, k, u = _FILETIME(), _FILETIME(), _FILETIME()
        _k32.GetSystemTimes(ctypes.byref(i), ctypes.byref(k), ctypes.byref(u))
        f = lambda x: (x.hi << 32) | x.lo  # noqa: E731
        return f(i), f(k) + f(u)            # kernel time includes idle time
    with open("/proc/stat") as fh:
        vals = [int(x) for x in fh.readline().split()[1:]]
    return vals[3] + (vals[4] if len(vals) > 4 else 0), sum(vals)


def cpu_percent(interval=0.3):
    try:
        i0, t0 = _cpu_times()
        time.sleep(interval)
        i1, t1 = _cpu_times()
        return round(100.0 * (1.0 - (i1 - i0) / float(t1 - t0)), 1) if t1 > t0 else 0.0
    except Exception:
        return None


def _num(s):
    s = s.strip()
    try:
        return float(s)
    except ValueError:
        return None


def nvidia_smi():
    exe = shutil.which("nvidia-smi")
    if not exe and IS_WIN:
        cand = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "nvidia-smi.exe")
        exe = cand if os.path.exists(cand) else None
    if not exe:
        return []
    q = "index,name,memory.total,memory.used,utilization.gpu,temperature.gpu,power.draw,uuid"
    try:
        p = subprocess.run([exe, "--query-gpu=" + q, "--format=csv,noheader,nounits"],
                           capture_output=True, timeout=20, stdin=subprocess.DEVNULL)
    except Exception:
        return []
    out = []
    for line in p.stdout.decode("utf-8", "replace").splitlines():
        f = [x.strip() for x in line.split(",")]
        if len(f) < 8:
            continue
        out.append({"index": int(f[0]), "name": f[1],
                    "mem_total_gb": round((_num(f[2]) or 0) / 1024, 1),
                    "mem_used_gb": round((_num(f[3]) or 0) / 1024, 1),
                    "util_pct": _num(f[4]), "temp_c": _num(f[5]), "power_w": _num(f[6]), "uuid": f[7]})
    return out


def sysinfo(home):
    info = {"host": platform.node(), "os": "windows" if IS_WIN else "posix",
            "platform": platform.platform(), "cpus": os.cpu_count(), "python": sys.version.split()[0],
            "boot_time": boot_time(), "cpu_pct": cpu_percent()}
    info.update(mem_info())
    try:
        info["disk_free_gb"] = round(shutil.disk_usage(home.root).free / 1e9, 1)
    except OSError:
        pass
    info["gpus"] = nvidia_smi()
    info["now"] = time.time()
    return info


def auto_capacity():
    cpus = os.cpu_count() or 1
    mem = mem_info().get("mem_total_gb") or 4.0
    gpus = [{"index": g["index"], "name": g["name"], "mem_gb": g["mem_total_gb"]} for g in nvidia_smi()]
    return {"v": 1, "cpus": cpus, "mem_gb": mem, "gpus": gpus,
            "reserve_cpus": max(1, cpus // 16), "reserve_mem_gb": max(2.0, round(mem * 0.06)),
            "hold_after_s": 1800, "updated": time.time()}


def load_capacity(home):
    cap = read_json(home.p("host.json"))
    if not cap:
        cap = auto_capacity()
        write_json(home.p("host.json"), cap)
    return cap


# ─────────────────────────────── the scheduler ───────────────────────────────
#
# Decentralised: each queued supervisor takes sched.lock, reads host.json and
# the live entries of active/ and queue/, and admits itself if decide() says so.
# FIFO with backfill -- a younger job may start in resources an older one
# cannot use yet, but never if that would stop an older job that fits now, and
# once the head of the queue has waited `hold_after_s` nothing overtakes it.

def job_request(spec):
    return {"cpus": float(spec.get("cpus") or 1), "mem_gb": float(spec.get("mem_gb") or 1),
            "gpus": float(spec.get("gpus") or 0)}


def _fit(req, cpu, mem, gfree):
    if req["cpus"] > cpu + 1e-9 or req["mem_gb"] > mem + 1e-9:
        return None
    g = req.get("gpus") or 0
    shares = {}
    if g > 0:
        if g < 1:
            cands = sorted((v, i) for i, v in gfree.items() if v >= g - 1e-9)
            if not cands:
                return None
            shares = {cands[0][1]: g}          # best fit keeps whole GPUs whole
        else:
            n = int(round(g))
            full = sorted(i for i, v in gfree.items() if v >= 1 - 1e-9)
            if len(full) < n:
                return None
            shares = {i: 1.0 for i in full[:n]}
    return {"cpus": req["cpus"], "mem_gb": req["mem_gb"], "gpus": sorted(shares),
            "gpu_shares": {str(i): s for i, s in shares.items()}}


def _take(pool, assign):
    cpu, mem, g = pool
    g = dict(g)
    for i, s in (assign.get("gpu_shares") or {}).items():
        g[int(i)] = g.get(int(i), 0) - s
    return cpu - assign["cpus"], mem - assign["mem_gb"], g


def decide(me, cap, active, queue, now):
    """(admit, assignment, reason) for queue entry `me`."""
    req = me["req"]
    tot_cpu = cap["cpus"] - cap.get("reserve_cpus", 0)
    tot_mem = cap["mem_gb"] - cap.get("reserve_mem_gb", 0)
    gpus = {int(g["index"]): 1.0 for g in cap.get("gpus") or []}
    if _fit(req, tot_cpu, tot_mem, gpus) is None:
        return False, None, ("impossible: asks %g cpus, %g GB, %g gpus; the host offers %g cpus, "
                             "%g GB, %d gpus" % (req["cpus"], req["mem_gb"], req["gpus"],
                                                 tot_cpu, tot_mem, len(gpus)))
    pool = (tot_cpu, tot_mem, gpus)
    for a in active:
        pool = _take(pool, a.get("assign") or {"cpus": a["req"]["cpus"], "mem_gb": a["req"]["mem_gb"]})
    if _fit(req, *pool) is None:
        return False, None, "waiting for resources (free: %g cpus, %.0f GB, gpu %s)" % (
            pool[0], pool[1], ",".join("%d:%.2g" % (i, v) for i, v in sorted(pool[2].items())) or "-")
    hold = cap.get("hold_after_s", 1800)
    mine_key = (me["enq"], me["key"])
    for q in sorted(queue, key=lambda q: (q["enq"], q["key"])):
        if (q["enq"], q["key"]) >= mine_key:
            continue
        qa = _fit(q["req"], *pool)
        if qa is not None:
            pool = _take(pool, qa)       # it goes first; see what is left for me
        elif now - q["enq"] > hold:
            return False, None, "holding for older job %s (queued %.0f min)" % (q.get("id"), (now - q["enq"]) / 60)
    mine = _fit(req, *pool)
    if mine is None:
        return False, None, "waiting behind older queued jobs"
    return True, mine, None


def heartbeat_alive(jd, now=None, boot=None):
    now = now or time.time()
    hb = read_json(os.path.join(jd, "heartbeat.json"))
    if not hb:
        return False
    t = hb.get("t", 0)
    return now - t <= LOST_AFTER_S and not (boot and t < boot - 5)


def live_entries(home, kind, now, boot):
    d = home.p(kind)
    out = []
    try:
        names = os.listdir(d)
    except OSError:
        return out
    for n in names:
        if not n.endswith(".json"):
            continue
        path = os.path.join(d, n)
        e = read_json(path)
        if not e:
            continue
        if heartbeat_alive(e.get("jd", ""), now, boot):
            out.append(e)
        elif now - e.get("enq", 0) > 30:     # its supervisor is gone: free what it held
            remove_quiet(path)
    return out


def sched_release(home, key):
    with FileLock(home.p("sched.lock")):
        remove_quiet(home.p("active", key + ".json"))
        remove_quiet(home.p("queue", key + ".json"))


# ─────────────────────────────── job status ───────────────────────────────

_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def last_line(path, nbytes=4096):
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - nbytes))
            tail = f.read().decode("utf-8", "replace")
    except OSError:
        return ""
    for piece in reversed(re.split(r"[\r\n]+", tail)):
        piece = _ANSI.sub("", piece).strip()
        if piece:
            return piece[-240:]
    return ""


def job_status(home, project, jid, now=None, boot=None, with_line=False):
    jd = home.jobdir(project, jid)
    st = read_json(os.path.join(jd, "status.json"))
    if not st:
        return None
    st = dict(st)
    now = now or time.time()
    hb = read_json(os.path.join(jd, "heartbeat.json"))
    if hb:
        st["heartbeat"] = hb.get("t")
        for k in ("cpu_s", "mem_gb", "peak_commit_gb", "procs"):
            if k in hb:
                st["live_" + k] = hb[k]
    if st.get("state") in ACTIVE_STATES:
        t = (hb or {}).get("t") or st.get("created") or 0
        if now - t > LOST_AFTER_S or (boot and t < boot - 5):
            st["state_recorded"] = st.get("state")
            st["state"] = "lost"
            st["lost_reason"] = ("the host restarted after the last heartbeat" if boot and t < boot - 5
                                 else "no heartbeat for %.0f s" % (now - t))
    try:
        st["log_bytes"] = os.path.getsize(os.path.join(jd, "output.log"))
    except OSError:
        st["log_bytes"] = 0
    if with_line and st.get("state") in ("running", "finishing"):
        st["last_line"] = last_line(os.path.join(jd, "output.log"))
    st["cancel_requested"] = os.path.exists(os.path.join(jd, "cancel.request"))
    return st


def list_jobs(home, project, limit=None, ids=None, now=None, boot=None, with_line=False):
    now = now or time.time()
    try:
        names = [n for n in os.listdir(home.jobs(project)) if NAME_RE.match(n)]
    except OSError:
        names = []
    if ids:
        names = [n for n in names if n in set(ids)]
    out = []
    for n in names:
        s = job_status(home, project, n, now, boot, with_line)
        if s:
            out.append(s)
    out.sort(key=lambda s: s.get("created") or 0, reverse=True)
    if limit:
        active = [s for s in out if s.get("state") in ACTIVE_STATES]
        rest = [s for s in out if s.get("state") not in ACTIVE_STATES][:max(0, limit - len(active))]
        out = sorted(active + rest, key=lambda s: s.get("created") or 0, reverse=True)
    return out


# ─────────────────────────────── launching ───────────────────────────────

def _ps_quote(s):
    return "'" + s.replace("'", "''") + "'"


def spawn_detached(argv, cwd):
    """Start argv so that it outlives this ssh session. Returns its pid."""
    mode = os.environ.get("RX_LAUNCH") or ("wmi" if IS_WIN else "setsid")
    if mode == "wmi":
        # Windows OpenSSH puts every session in a Job Object that is killed when
        # the session closes; Start-Process and plain background processes die
        # with it. A process created by WMI's Win32_Process.Create is a child of
        # the WMI service, outside that job, and survives.
        cmdline = subprocess.list2cmdline(argv)
        script = ("$ProgressPreference='SilentlyContinue';"
                  "$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create "
                  "-Arguments @{CommandLine=%s; CurrentDirectory=%s};"
                  "Write-Output ('RXPID {0} {1}' -f $r.ReturnValue, $r.ProcessId)"
                  % (_ps_quote(cmdline), _ps_quote(cwd)))
        enc = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
        p = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-ExecutionPolicy",
                            "Bypass", "-EncodedCommand", enc], capture_output=True, timeout=180,
                           stdin=subprocess.DEVNULL)
        m = re.search(rb"RXPID (\d+) (\d+)", p.stdout)
        if not m or m.group(1) != b"0":
            raise RxError("WMI launch failed (rc %s): %s %s" % (
                p.returncode, p.stdout.decode("utf-8", "replace")[-400:],
                p.stderr.decode("utf-8", "replace")[-400:]), "launch")
        return int(m.group(2))
    if mode == "systemd-run" and not IS_WIN:
        unit = "rx-%s" % _rand(10)
        subprocess.run(["systemd-run", "--user", "--collect", "--quiet", "--unit", unit,
                        "--working-directory", cwd, "--"] + list(argv), check=True,
                       stdin=subprocess.DEVNULL, capture_output=True, timeout=60)
        return 0
    kw = dict(cwd=cwd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
              stderr=subprocess.DEVNULL, close_fds=True)
    if IS_WIN:
        try:
            p = subprocess.Popen(argv, creationflags=DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
                                 | CREATE_BREAKAWAY_FROM_JOB, **kw)
        except OSError:
            p = subprocess.Popen(argv, creationflags=DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP, **kw)
    else:
        p = subprocess.Popen(argv, start_new_session=True, **kw)
    return p.pid


def spawn_supervisor(home, jd):
    return spawn_detached([sys.executable, os.path.abspath(__file__), "supervise",
                           "--home", home.root, jd], jd)


def env_dir(home, name):
    if not name:
        return None
    ptr = read_json(home.p("envs", check_name(name, "env") + ".json"))
    if not ptr or not ptr.get("dir"):
        raise RxError("environment %r is not built on this host (rx env build %s)" % (name, name), "env")
    d = home.p("envs", ptr["dir"])
    if not os.path.isdir(d):
        raise RxError("environment %r points at a missing directory %s" % (name, d), "env")
    return d


def env_python(d):
    return os.path.join(d, "python.exe") if IS_WIN else os.path.join(d, "bin", "python")


def find_git_bash():
    for c in (os.path.join(os.environ.get("ProgramFiles", r"C:\Program Files"), "Git", "bin", "bash.exe"),
              r"C:\Program Files\Git\bin\bash.exe"):
        if os.path.exists(c):
            return c
    raise RxError("Git Bash is not installed on this host", "shell")


def build_env(home, spec, assign, jd, ws):
    env = {k: v for k, v in os.environ.items()
           if k.upper() not in ("PYTHONHOME", "PYTHONPATH", "PYTHONSTARTUP", "VIRTUAL_ENV",
                                "CONDA_PREFIX", "RX_LAUNCH", "PYTHONUTF8")}
    d = env_dir(home, spec.get("env"))
    base = os.path.dirname(sys.executable)
    pre = []
    if d:
        pre = [d, os.path.join(d, "Scripts")] if IS_WIN else [os.path.join(d, "bin")]
    elif spec.get("internal") is None:
        pre = [base, os.path.join(base, "Scripts")] if IS_WIN else []
    pkey = next((k for k in env if k.upper() == "PATH"), "PATH")
    env[pkey] = os.pathsep.join(pre + [env.get(pkey, "")])
    threads = str(max(1, int(math.floor(assign["cpus"]))))
    for k in THREAD_VARS:
        env[k] = threads
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in assign.get("gpus") or [])
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env.update({"RX_JOB_ID": spec["id"], "RX_PROJECT": spec["project"], "RX_JOB_DIR": jd,
                "RX_WORKSPACE": ws, "RX_HOME": home.root, "RX_CPUS": threads,
                "RX_GPUS": env["CUDA_VISIBLE_DEVICES"]})
    for k, v in (spec.get("env_vars") or {}).items():
        env[str(k)] = str(v)
    if spec.get("shell") == "wsl":
        names = ["RX_JOB_ID/u", "RX_PROJECT/u", "RX_CPUS/u", "RX_GPUS/u", "RX_JOB_DIR/pu",
                 "RX_WORKSPACE/pu", "CUDA_VISIBLE_DEVICES/u", "CUDA_DEVICE_ORDER/u",
                 "PYTHONUNBUFFERED/u", "PYTHONIOENCODING/u"] + ["%s/u" % k for k in THREAD_VARS]
        names += ["%s/u" % k for k in (spec.get("env_vars") or {})]
        env["WSLENV"] = ":".join([x for x in [env.get("WSLENV", "")] if x] + names)
    return env, d


def build_command(home, spec, env, envd, cwd):
    """The child's command: a string for CreateProcess on Windows, an argv list on POSIX."""
    shell = spec.get("shell") or "exec"
    if spec.get("internal") == "envbuild":
        argv = [sys.executable, os.path.abspath(__file__), "envbuild", "--home", home.root,
                os.path.join(env["RX_JOB_DIR"], "env-spec.json")]
        return subprocess.list2cmdline(argv) if IS_WIN else argv
    if shell == "exec":
        argv = list(spec["argv"])
        a0 = argv[0]
        py = env_python(envd) if envd else sys.executable
        if a0 in ("python", "python3", "py"):
            argv[0] = py
        elif a0 in ("pip", "pip3"):
            argv = [py, "-m", "pip"] + argv[1:]
        elif not os.path.isabs(a0) and "/" not in a0 and "\\" not in a0:
            pkey = next((k for k in env if k.upper() == "PATH"), "PATH")
            argv[0] = shutil.which(a0, path=env.get(pkey)) or a0
        return subprocess.list2cmdline(argv) if IS_WIN else argv
    cmd = spec["command"]
    if IS_WIN:
        if shell == "cmd":
            return 'cmd.exe /d /s /c "%s"' % cmd
        if shell == "powershell":
            script = "$ProgressPreference='SilentlyContinue'\n%s\nexit $LASTEXITCODE\n" % cmd
            return ("powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass -EncodedCommand "
                    + base64.b64encode(script.encode("utf-16-le")).decode("ascii"))
        if shell == "bash":
            return subprocess.list2cmdline([find_git_bash(), "-c", cmd])
        if shell == "wsl":
            distro = spec.get("wsl_distro") or "Ubuntu-24.04"
            return subprocess.list2cmdline(["wsl.exe", "-d", distro, "--cd", cwd, "--exec",
                                            "bash", "-lc", cmd])
    else:
        if shell == "bash":
            return ["/bin/bash", "-lc", cmd]
        if shell == "sh":
            return ["/bin/sh", "-c", cmd]
    raise RxError("shell %r is not available on this host" % shell, "shell")


def collect_outputs(home, spec, ws, started):
    inc = spec.get("outputs") or ["**"]
    exc = list(spec.get("outputs_exclude") or []) + list(OUTPUT_EXCLUDES)
    idx = Index(home, spec["project"], spec.get("ws") or "ws")
    files_idx = idx.load()
    out, changes, total = [], {}, 0
    for rel, ap, st in walk_files(ws, inc, exc):
        if st.st_mtime < started - 2:
            continue
        sha, fresh = cached_sha(files_idx, rel, ap, st)
        e = files_idx.get(rel)
        if e and e[3] and e[2] == sha:
            continue                  # a pushed input whose content did not change
        out.append([rel, st.st_size, st.st_mtime_ns, sha])
        total += st.st_size
        if fresh:
            changes[rel] = [st.st_size, st.st_mtime_ns, sha, e[3] if e else 0]
    idx.update(changes)
    out.sort()
    return {"v": 1, "files": out, "bytes": total, "started": started, "globs": inc,
            "generated": time.time()}


class Supervisor:
    def __init__(self, home, jd, spec):
        self.home, self.jd, self.spec = home, jd, spec
        self.key = "%s--%s" % (spec["project"], spec["id"])
        self.status = read_json(os.path.join(jd, "status.json"), {}) or {}
        self.stop = threading.Event()
        self.win_job = None
        self.proc = None
        self.peak_mem = 0.0
        self.last_sample = {}

    def set(self, **kw):
        self.status.update(kw)
        self.status["updated"] = time.time()
        write_json(os.path.join(self.jd, "status.json"), self.status)

    def sample(self):
        s = {}
        try:
            if self.win_job is not None:
                s = self.win_job.sample()
            elif self.proc is not None and not IS_WIN:
                s = _posix_group_sample(self.proc.pid)
        except Exception:
            s = {}
        if s.get("mem_gb") is not None:
            self.peak_mem = max(self.peak_mem, s["mem_gb"])
            s["peak_mem_gb"] = round(self.peak_mem, 2)
        if s:
            self.last_sample = s
        return s

    def beat(self):
        hb = {"t": time.time(), "pid": os.getpid(), "state": self.status.get("state")}
        hb.update(self.sample())
        write_json(os.path.join(self.jd, "heartbeat.json"), hb)

    def _hb_loop(self):
        while not self.stop.wait(HEARTBEAT_S):
            try:
                self.beat()
            except Exception:
                pass

    def cancelled(self):
        return os.path.exists(os.path.join(self.jd, "cancel.request"))

    def run(self):
        now = time.time()
        self.set(state="queued", supervisor_pid=os.getpid(), queued_at=now, host=platform.node(),
                 agent=agent_sha()[:12], in_job=(process_in_job() if IS_WIN else None))
        self.beat()
        threading.Thread(target=self._hb_loop, daemon=True).start()
        keep_awake(True)
        try:
            assign = self.admit()
            if assign is None:
                self.set(state="cancelled", ended=time.time(), note="cancelled while queued")
                return
            self.run_child(assign)
        except Exception as e:
            self.set(state="error", error="%s: %s" % (type(e).__name__, e),
                     trace=traceback.format_exc()[-4000:], ended=time.time())
        finally:
            try:
                sched_release(self.home, self.key)
            except Exception:
                pass
            self.stop.set()
            try:
                self.beat()
            except Exception:
                pass
            if self.win_job is not None:
                self.win_job.close()
            keep_awake(False)

    def admit(self):
        entry = {"key": self.key, "project": self.spec["project"], "id": self.spec["id"],
                 "jd": self.jd, "req": job_request(self.spec), "enq": self.status["queued_at"]}
        qpath = self.home.p("queue", self.key + ".json")
        write_json(qpath, entry)
        last = None
        while True:
            if self.cancelled():
                sched_release(self.home, self.key)
                return None
            now = time.time()
            boot = boot_time()
            with FileLock(self.home.p("sched.lock")):
                cap = load_capacity(self.home)
                active = live_entries(self.home, "active", now, boot)
                queue = live_entries(self.home, "queue", now, boot)
                if not any(q.get("key") == self.key for q in queue):
                    queue.append(entry)
                ok, assign, reason = decide(entry, cap, active, queue, now)
                if ok:
                    entry["assign"] = assign
                    entry["admitted"] = now
                    write_json(self.home.p("active", self.key + ".json"), entry)
                    remove_quiet(qpath)
                    return assign
            if reason != last:
                self.set(waiting=reason)
                last = reason
            if reason and reason.startswith("impossible"):
                raise RxError(reason, "resources")
            time.sleep(2.0 + random.random())

    def kill_tree(self, graceful):
        if IS_WIN:
            if self.win_job is not None:
                self.win_job.terminate()
            elif self.proc is not None:
                self.proc.kill()
            return
        import signal
        if self.proc is None:
            return
        try:
            pg = os.getpgid(self.proc.pid)
        except OSError:
            pg = None
        if pg is None:
            return
        try:
            os.killpg(pg, signal.SIGTERM if graceful else signal.SIGKILL)
        except OSError:
            return
        if graceful:
            for _ in range(100):
                if self.proc.poll() is not None:
                    break
                time.sleep(0.1)
            try:
                os.killpg(pg, signal.SIGKILL)
            except OSError:
                pass

    def run_child(self, assign):
        spec = self.spec
        ws = self.home.ws(spec["project"], spec.get("ws") or "ws")
        os.makedirs(ws, exist_ok=True)
        cwd = join_rel(ws, spec["cwd"]) if spec.get("cwd") not in (None, "", ".") else ws
        env, envd = build_env(self.home, spec, assign, self.jd, ws)
        cmd = build_command(self.home, spec, env, envd, cwd)
        started = time.time()
        log = open(os.path.join(self.jd, "output.log"), "ab")
        self.set(state="running", started=started, assign=assign, waiting=None,
                 command_line=cmd if isinstance(cmd, str) else subprocess.list2cmdline(cmd))
        cancelled, rc = False, None
        try:
            if IS_WIN:
                mem_hard = spec.get("mem_hard_gb")
                self.win_job = WinJob(int(float(mem_hard) * 2 ** 30) if mem_hard else None)
                self.proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                                             stdout=log, stderr=subprocess.STDOUT,
                                             creationflags=CREATE_SUSPENDED | CREATE_NO_WINDOW)
                self.win_job.assign_and_resume(self.proc._handle)
            else:
                self.proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                                             stdout=log, stderr=subprocess.STDOUT,
                                             start_new_session=True)
            self.set(pid=self.proc.pid)
            timeout = spec.get("timeout_s")
            while True:
                try:
                    rc = self.proc.wait(timeout=1.0)
                    break
                except subprocess.TimeoutExpired:
                    pass
                if not cancelled and self.cancelled():
                    cancelled = True
                    self.set(cancelling=time.time())
                    self.kill_tree(graceful=True)
                if not cancelled and timeout and time.time() - started > float(timeout):
                    cancelled = True
                    self.set(cancelling=time.time(), note="timed out after %s s" % timeout)
                    self.kill_tree(graceful=True)
        finally:
            log.close()
        final_sample = self.sample()
        leftovers = final_sample.get("procs") or 0
        if leftovers:
            self.kill_tree(graceful=False)       # background children of a finished job
        ended = time.time()
        sched_release(self.home, self.key)
        self.set(state="finishing", rc=rc, ended=ended, leftover_procs=leftovers,
                 cpu_s=final_sample.get("cpu_s"), peak_mem_gb=round(self.peak_mem, 2),
                 peak_commit_gb=final_sample.get("peak_commit_gb"))
        try:
            outs = collect_outputs(self.home, spec, ws, started)
        except Exception as e:
            outs = {"v": 1, "files": [], "bytes": 0, "error": "%s: %s" % (type(e).__name__, e)}
        write_json(os.path.join(self.jd, "outputs.json"), outs)
        state = "cancelled" if cancelled else ("done" if rc == 0 else "failed")
        self.set(state=state, finished=time.time(), outputs=len(outs["files"]),
                 outputs_bytes=outs.get("bytes", 0))


def _posix_group_sample(pid):
    try:
        pg = os.getpgid(pid)
    except OSError:
        return {}
    page = os.sysconf("SC_PAGE_SIZE")
    tick = float(os.sysconf("SC_CLK_TCK"))
    rss, cpu, n = 0, 0.0, 0
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            with open("/proc/%s/stat" % d) as f:
                s = f.read()
            fields = s[s.rindex(")") + 2:].split()
            if int(fields[2]) != pg:
                continue
            cpu += (int(fields[11]) + int(fields[12])) / tick
            rss += int(fields[21]) * page
            n += 1
        except (OSError, ValueError, IndexError):
            continue
    return {"cpu_s": round(cpu, 1), "mem_gb": round(rss / 2 ** 30, 2), "procs": n}


def supervise_main(jd):
    home = Home()
    try:
        sys.stderr = open(os.path.join(jd, "supervisor.log"), "a", buffering=1, encoding="utf-8")
    except OSError:
        pass
    spec = read_json(os.path.join(jd, "spec.json"))
    if not spec:
        return 2
    lock = FileLock(os.path.join(jd, "supervisor.lock"))
    if not lock.acquire(blocking=False):
        return 0                      # another supervisor already owns this job
    try:
        st = read_json(os.path.join(jd, "status.json"), {}) or {}
        if st.get("state") in TERMINAL_STATES:
            return 0
        Supervisor(home, jd, spec).run()
    finally:
        lock.release()
    return 0


# ─────────────────────────────── environments ───────────────────────────────

def base_python(home, version=None):
    if version:
        exe = home.p("python", version, "python.exe" if IS_WIN else os.path.join("bin", "python3"))
        if os.path.exists(exe):
            return exe
        if not IS_WIN:
            found = shutil.which("python" + ".".join(version.split(".")[:2]))
            if found:
                return found
        raise RxError("python %s is not installed under %s (rx setup installs it)" % (
            version, home.p("python")), "env")
    return sys.executable


def _run_logged(argv, **kw):
    print("[rx] $ %s" % subprocess.list2cmdline(argv), flush=True)
    p = subprocess.run(argv, stdin=subprocess.DEVNULL, **kw)
    if p.returncode != 0:
        raise RxError("command failed with exit code %d" % p.returncode, "env")
    return p


def envbuild_main(spec_path):
    home = Home()
    spec = read_json(spec_path)
    name, h = check_name(spec["name"], "env"), spec["hash"]
    d = home.p("envs", "%s@%s" % (name, h[:12]))
    ptr_path = home.p("envs", name + ".json")
    ptr = read_json(ptr_path) or {}
    if ptr.get("hash") == h and os.path.exists(os.path.join(home.p("envs", ptr.get("dir", "")), "rx-env.json")):
        print("[rx] environment %s is already built (%s)" % (name, ptr["dir"]), flush=True)
        return 0
    base = base_python(home, spec.get("python"))
    print("[rx] building environment %s from %s into %s" % (name, base, d), flush=True)
    if os.path.exists(d):
        shutil.rmtree(d)
    os.makedirs(home.p("envs"), exist_ok=True)
    if IS_WIN:
        # A standalone CPython directory is relocatable: copying it gives an
        # interpreter of its own, with no launcher stub between it and a Job Object.
        shutil.copytree(os.path.dirname(base), d, ignore=shutil.ignore_patterns("__pycache__"))
    else:
        _run_logged([base, "-m", "venv", d])
    py = env_python(d)
    if subprocess.run([py, "-m", "pip", "--version"], capture_output=True).returncode != 0:
        _run_logged([py, "-m", "ensurepip", "--upgrade"])
    req = os.path.join(d, "rx-requirements.txt")
    with open(req, "w", encoding="utf-8") as f:
        f.write("\n".join(spec.get("requirements") or []) + "\n")
    args = [py, "-m", "pip", "install", "--no-input", "--disable-pip-version-check",
            "--progress-bar", "off"]
    if spec.get("index_url"):
        args += ["--index-url", spec["index_url"]]
    for u in spec.get("extra_index_urls") or []:
        args += ["--extra-index-url", u]
    for u in spec.get("find_links") or []:
        args += ["--find-links", u]
    if spec.get("constraints"):
        con = os.path.join(d, "rx-constraints.txt")
        with open(con, "w", encoding="utf-8") as f:
            f.write("\n".join(spec["constraints"]) + "\n")
        args += ["-c", con]
    args += ["-r", req] + list(spec.get("pip_args") or [])
    t0 = time.time()
    _run_logged(args)
    freeze = subprocess.run([py, "-m", "pip", "freeze", "--all"], capture_output=True).stdout
    with open(os.path.join(d, "rx-freeze.txt"), "wb") as f:
        f.write(freeze)
    for check in spec.get("verify") or []:
        _run_logged([py, "-c", check])
    ver = subprocess.run([py, "-c", "import sys;print(sys.version)"], capture_output=True).stdout
    meta = {"name": name, "hash": h, "dir": os.path.basename(d), "built": time.time(),
            "build_s": round(time.time() - t0, 1), "python": ver.decode().strip(),
            "freeze_sha256": hashlib.sha256(freeze).hexdigest(), "spec": spec}
    write_json(os.path.join(d, "rx-env.json"), meta)
    write_json(ptr_path, {"name": name, "hash": h, "dir": os.path.basename(d), "built": meta["built"]})
    print("[rx] environment %s ready: %s (%d packages)" % (
        name, d, len([x for x in freeze.splitlines() if x.strip()])), flush=True)
    return 0


def list_envs(home):
    out = []
    try:
        names = sorted(n for n in os.listdir(home.p("envs")) if n.endswith(".json"))
    except OSError:
        return out
    for n in names:
        ptr = read_json(home.p("envs", n)) or {}
        meta = read_json(home.p("envs", ptr.get("dir", "-"), "rx-env.json")) or {}
        out.append({"name": ptr.get("name"), "hash": ptr.get("hash"), "dir": ptr.get("dir"),
                    "built": ptr.get("built"), "python": meta.get("python"),
                    "build_s": meta.get("build_s"), "freeze_sha256": meta.get("freeze_sha256")})
    return out


# ─────────────────────────────── the RPC operations ───────────────────────────────

def _frame(out, obj, payload=b""):
    out.write((json.dumps(obj, separators=(",", ":")) + "\n").encode("utf-8"))
    if payload:
        out.write(payload)
    out.flush()


def op_hello(req, home, inp, out):
    return {"ok": True, "protocol": PROTOCOL, "agent": agent_sha(), "python": sys.version,
            "executable": sys.executable, "os": "windows" if IS_WIN else "posix",
            "platform": platform.platform(), "host": platform.node(), "home": home.root,
            "now": time.time(), "boot_time": boot_time(), "pid": os.getpid(),
            "in_job": process_in_job() if IS_WIN else None}


def op_sysinfo(req, home, inp, out):
    return {"ok": True, "sysinfo": sysinfo(home), "capacity": load_capacity(home), "envs": list_envs(home)}


def op_capacity(req, home, inp, out):
    with FileLock(home.p("sched.lock")):
        cap = None if req.get("reset") else read_json(home.p("host.json"))
        cap = cap or auto_capacity()
        for k in ("cpus", "mem_gb", "reserve_cpus", "reserve_mem_gb", "hold_after_s"):
            if req.get(k) is not None:
                cap[k] = float(req[k])
        if req.get("gpus") is not None:
            cap["gpus"] = req["gpus"]
        cap["updated"] = time.time()
        write_json(home.p("host.json"), cap)
    return {"ok": True, "capacity": cap}


def op_plan(req, home, inp, out):
    """Make the workspace hold the listed files where it can without a transfer
    (already there, or a copy of the same bytes elsewhere in the project), and
    report what still has to be sent, with the offset to resume from."""
    project, ws = req["project"], req.get("ws") or "ws"
    root = home.ws(project, ws)
    os.makedirs(root, exist_ok=True)
    idx = Index(home, project, ws)
    files_idx = idx.load()
    by_sha = {}
    try:
        others = [n for n in os.listdir(home.project(project))
                  if n.startswith("index-") and n.endswith(".json")]
    except OSError:
        others = []
    for n in others:
        w = n[len("index-"):-len(".json")]
        if not NAME_RE.match(w):
            continue
        wroot = home.ws(project, w)
        entries = files_idx if w == ws else (read_json(os.path.join(home.project(project), n), {}) or {}).get("files", {})
        for rel, e in entries.items():
            if e and e[2]:
                by_sha.setdefault(e[2], []).append((wroot, rel, e))
    need, changes, have, copied, nbytes = [], {}, 0, 0, 0
    for item in req["files"]:
        rel, size, sha = safe_rel(item[0]), int(item[1]), item[2]
        mtime_ns = int(item[3]) if len(item) > 3 and item[3] else None
        dest = join_rel(root, rel)
        try:
            st = os.stat(dest)
        except OSError:
            st = None
        if st is not None and stat.S_ISREG(st.st_mode) and st.st_size == size:
            cur, _ = cached_sha(files_idx, rel, dest, st)
            if cur == sha:
                if mtime_ns and st.st_mtime_ns != mtime_ns:
                    os.utime(dest, ns=(st.st_atime_ns, mtime_ns))
                    st = os.stat(dest)
                changes[rel] = [size, st.st_mtime_ns, sha, 1]
                have += 1
                continue
        src = None
        for wroot, orel, e in by_sha.get(sha, ()):
            p = join_rel(wroot, orel)
            try:
                ost = os.stat(p)
            except OSError:
                continue
            if ost.st_size == e[0] == size and ost.st_mtime_ns == e[1] and p != dest:
                src = p
                break
        if src:
            copy_via_part(src, dest, mtime_ns)
            st = os.stat(dest)
            changes[rel] = [size, st.st_mtime_ns, sha, 1]
            copied += 1
            continue
        off = partial_offset(dest, sha)
        need.append([rel, off])
        nbytes += size - off
    deleted, kept = [], []
    if req.get("prune"):
        keep = set(item[0] for item in req["files"])
        for rel, e in list(files_idx.items()):
            if not e[3] or rel in keep or rel in changes:
                continue
            p = join_rel(root, rel)
            try:
                st = os.stat(p)
            except OSError:
                deleted.append(rel)
                continue
            if st.st_size == e[0] and st.st_mtime_ns == e[1]:
                if remove_quiet(p):
                    deleted.append(rel)
            else:
                kept.append(rel)          # changed on the host since it was pushed: not ours to delete
    idx.update(changes, deleted)
    return {"ok": True, "need": need, "need_bytes": nbytes, "have": have, "copied": copied,
            "deleted": deleted, "kept_changed": kept}


def op_receive(req, home, inp, out):
    project, ws = req["project"], req.get("ws") or "ws"
    root = home.ws(project, ws)
    os.makedirs(root, exist_ok=True)
    idx = Index(home, project, ws)
    pending, last = {}, [time.time()]

    def on_done(rel, size, mtime_ns, sha):
        pending[rel] = [size, mtime_ns, sha, 1]
        if time.time() - last[0] > 3:
            idx.update(dict(pending))
            pending.clear()
            last[0] = time.time()

    try:
        results = receive_stream(StreamReader(inp), root, on_done)
    finally:
        if pending:
            idx.update(pending)
    bad = [r for r in results if not r.get("ok")]
    return {"ok": True, "received": len(results) - len(bad), "errors": bad}


def op_manifest(req, home, inp, out):
    project, ws = req["project"], req.get("ws") or "ws"
    root = home.ws(project, ws)
    idx = Index(home, project, ws)
    files_idx = idx.load()
    since, want_hash, limit = req.get("since"), req.get("hash", True), int(req.get("limit") or 500000)
    files, changes, total = [], {}, 0
    for rel, ap, st in walk_files(root, req.get("include") or ["**"], req.get("exclude") or []):
        if since and st.st_mtime < since:
            continue
        sha = None
        if want_hash:
            sha, fresh = cached_sha(files_idx, rel, ap, st)
            if fresh:
                e = files_idx.get(rel)
                changes[rel] = [st.st_size, st.st_mtime_ns, sha, e[3] if e else 0]
        files.append([rel, st.st_size, st.st_mtime_ns, sha])
        total += st.st_size
        if len(files) >= limit:
            break
    idx.update(changes)
    files.sort()
    return {"ok": True, "files": files, "bytes": total, "truncated": len(files) >= limit}


def op_send(req, home, inp, out):
    project, ws = req["project"], req.get("ws") or "ws"
    root = home.ws(project, ws)
    _frame(out, {"ok": True, "count": len(req["files"])})
    w = StreamWriter(out, level=int(req.get("level", 6)))
    for item in req["files"]:
        rel, off = item[0], int(item[1] or 0)
        sha = item[2] if len(item) > 2 else None
        w.send_file(root, safe_rel(rel), off, sha)
    w.end()
    return None


def op_launch(req, home, inp, out):
    spec = dict(req["spec"])
    project, jid = check_name(spec["project"], "project"), check_name(spec["id"], "job id")
    jobs = home.jobs(project)
    os.makedirs(jobs, exist_ok=True)
    jd = os.path.join(jobs, jid)
    if os.path.exists(os.path.join(jd, "spec.json")):
        st = job_status(home, project, jid, boot=boot_time())
        if st and st.get("state") == "launching" and not st.get("heartbeat") and \
                time.time() - (st.get("created") or 0) > 30:
            spawn_supervisor(home, jd)            # the first supervisor never started
        return {"ok": True, "existing": True, "job": job_status(home, project, jid, boot=boot_time())}
    shell = spec.get("shell") or "exec"
    allowed = ("exec", "cmd", "powershell", "bash", "wsl") if IS_WIN else ("exec", "bash", "sh")
    if shell not in allowed:
        raise RxError("shell %r is not available on this host (%s)" % (shell, ", ".join(allowed)), "shell")
    if spec.get("internal") != "envbuild":
        env_dir(home, spec.get("env"))
    cap = load_capacity(home)
    ok, _, reason = decide({"req": job_request(spec), "enq": 0, "key": ""}, cap, [], [], time.time())
    if not ok and reason.startswith("impossible"):
        raise RxError(reason, "resources")
    spec["created"] = time.time()
    spec["agent"] = agent_sha()[:12]
    tmp = os.path.join(jobs, ".tmp-%s-%s" % (jid, _rand()))
    os.makedirs(tmp)
    write_json(os.path.join(tmp, "spec.json"), spec)
    if spec.get("internal") == "envbuild":
        write_json(os.path.join(tmp, "env-spec.json"), spec["env_spec"])
    write_json(os.path.join(tmp, "status.json"), {
        "id": jid, "project": project, "name": spec.get("name"), "state": "launching",
        "created": spec["created"], "display": spec.get("display"), "env": spec.get("env"),
        "req": job_request(spec), "shell": shell, "ws": spec.get("ws") or "ws"})
    try:
        os.rename(tmp, jd)
    except OSError:
        shutil.rmtree(tmp, ignore_errors=True)
        if os.path.exists(os.path.join(jd, "spec.json")):
            return {"ok": True, "existing": True, "job": job_status(home, project, jid)}
        raise
    pid = spawn_supervisor(home, jd)
    return {"ok": True, "existing": False, "supervisor_pid": pid,
            "job": job_status(home, project, jid, boot=boot_time())}


def op_status(req, home, inp, out):
    now, boot = time.time(), boot_time()
    jobs = list_jobs(home, req["project"], req.get("limit"), req.get("ids"), now, boot,
                     with_line=req.get("with_line", False))
    return {"ok": True, "now": now, "jobs": jobs}


def op_outputs(req, home, inp, out):
    jd = home.jobdir(req["project"], req["id"])
    return {"ok": True, "job": job_status(home, req["project"], req["id"], boot=boot_time()),
            "outputs": read_json(os.path.join(jd, "outputs.json"))}


def op_overview(req, home, inp, out):
    now, boot = time.time(), boot_time()
    limit = int(req.get("limit") or 12)
    projects = {}
    for p in home.projects():
        projects[p] = list_jobs(home, p, limit, None, now, boot, with_line=True)
    return {"ok": True, "now": now, "sysinfo": sysinfo(home), "capacity": load_capacity(home),
            "projects": projects, "envs": list_envs(home),
            "queue": live_entries(home, "queue", now, boot), "active": live_entries(home, "active", now, boot)}


def op_tail(req, home, inp, out):
    project, jid = req["project"], req["id"]
    jd = home.jobdir(project, jid)
    path = os.path.join(jd, "output.log")
    follow = bool(req.get("follow"))
    limit_s = float(req.get("timeout") or 3600)
    try:
        size = os.path.getsize(path)
    except OSError:
        size = 0
    offset = int(req.get("offset") or 0)
    if req.get("tail_bytes") is not None:
        offset = max(0, size - int(req["tail_bytes"]))
    st = job_status(home, project, jid, boot=boot_time())
    if st is None:
        raise RxError("no job %s in project %s" % (jid, project), "notfound")
    _frame(out, {"ok": True, "size": size, "offset": offset, "state": st.get("state")})
    t0 = last_write = time.time()
    f = None
    try:
        while True:
            if f is None and os.path.exists(path):
                f = open(path, "rb")
            data = b""
            if f is not None:
                f.seek(offset)
                data = f.read(256 << 10)
            if data:
                _frame(out, {"t": "d", "n": len(data)}, data)
                offset += len(data)
                last_write = time.time()
                continue
            st = job_status(home, project, jid, boot=boot_time())
            if not follow or st.get("state") in TERMINAL_STATES or time.time() - t0 > limit_s:
                if st.get("state") in TERMINAL_STATES and f is not None:
                    f.seek(offset)
                    rest = f.read()
                    if rest:
                        _frame(out, {"t": "d", "n": len(rest)}, rest)
                        offset += len(rest)
                break
            if time.time() - last_write > 15:
                _frame(out, {"t": "k"})
                last_write = time.time()
            time.sleep(0.5)
    finally:
        if f is not None:
            f.close()
    _frame(out, {"t": "end", "offset": offset, "state": st.get("state"), "rc": st.get("rc")})
    return None


def op_cancel(req, home, inp, out):
    project, jid = req["project"], req["id"]
    jd = home.jobdir(project, jid)
    st = job_status(home, project, jid, boot=boot_time())
    if st is None:
        raise RxError("no job %s in project %s" % (jid, project), "notfound")
    if st["state"] in TERMINAL_STATES and st["state"] != "lost":
        return {"ok": True, "job": st, "note": "already %s" % st["state"]}
    with open(os.path.join(jd, "cancel.request"), "w") as f:
        f.write(str(time.time()))
    if st["state"] == "lost":
        raw = read_json(os.path.join(jd, "status.json"), {}) or {}
        raw.update({"state": "cancelled", "ended": raw.get("ended") or time.time(),
                    "note": "cancelled after its supervisor was lost (%s)" % st.get("lost_reason")})
        write_json(os.path.join(jd, "status.json"), raw)
        sched_release(home, "%s--%s" % (project, jid))
    return {"ok": True, "job": job_status(home, project, jid, boot=boot_time())}


def op_exec(req, home, inp, out):
    """Run a short command now, streaming its output. It is tied to this ssh
    session: a dropped connection kills it. Anything long belongs in `launch`."""
    project = req.get("project")
    cwd = home.ws(project, req.get("ws") or "ws") if project else home.root
    os.makedirs(cwd, exist_ok=True)
    spec = {"id": "exec", "project": project or "none", "shell": req.get("shell") or "exec",
            "argv": req.get("argv"), "command": req.get("command"), "env": req.get("env"),
            "env_vars": req.get("env_vars")}
    assign = {"cpus": float(req.get("cpus") or os.cpu_count() or 1), "gpus": req.get("gpus") or []}
    env, envd = build_env(home, spec, assign, home.root, cwd)
    if req.get("gpus") is None:
        env.pop("CUDA_VISIBLE_DEVICES", None)
    cmd = build_command(home, spec, env, envd, cwd)
    kw = {"creationflags": CREATE_NO_WINDOW} if IS_WIN else {}
    p = subprocess.Popen(cmd, cwd=cwd, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, **kw)
    _frame(out, {"ok": True, "pid": p.pid})
    deadline = time.time() + float(req.get("timeout") or 3600)
    fd = p.stdout.fileno()
    killed = False
    timer = threading.Timer(max(0.0, deadline - time.time()), p.kill)
    timer.daemon = True
    timer.start()
    try:
        while True:
            data = os.read(fd, 65536)
            if not data:
                break
            _frame(out, {"t": "d", "n": len(data)}, data)
        rc = p.wait()
        killed = time.time() >= deadline
    finally:
        timer.cancel()
    _frame(out, {"t": "end", "rc": rc, "timed_out": killed})
    return None


def op_gc(req, home, inp, out):
    days, keep, dry = float(req.get("days", 14)), int(req.get("keep", 30)), bool(req.get("dry_run"))
    now = time.time()
    removed_jobs, removed_parts, removed_envs = [], [], []
    for p in ([req["project"]] if req.get("project") else home.projects()):
        jobs = list_jobs(home, p, None, None, now, boot_time())
        done = [j for j in jobs if j.get("state") in TERMINAL_STATES]
        for j in done[keep:]:
            if now - (j.get("created") or now) > days * 86400:
                removed_jobs.append("%s/%s" % (p, j["id"]))
                if not dry:
                    shutil.rmtree(home.jobdir(p, j["id"]), ignore_errors=True)
        try:
            names = os.listdir(home.project(p))
        except OSError:
            names = []
        for n in names:
            if not NAME_RE.match(n) or n == "jobs" or not os.path.isdir(os.path.join(home.project(p), n)):
                continue
            for rel, ap, st in _walk_parts(os.path.join(home.project(p), n)):
                if now - st.st_mtime > 7 * 86400:
                    removed_parts.append("%s/%s/%s" % (p, n, rel))
                    if not dry:
                        remove_quiet(ap)
                        if ap.endswith(PART):
                            remove_quiet(ap[:-len(PART)] + SIDE)
    live = set()
    try:
        for n in os.listdir(home.p("envs")):
            if n.endswith(".json"):
                live.add((read_json(home.p("envs", n)) or {}).get("dir"))
        for n in os.listdir(home.p("envs")):
            full = home.p("envs", n)
            if "@" in n and os.path.isdir(full) and n not in live and now - os.path.getmtime(full) > 86400:
                removed_envs.append(n)
                if not dry:
                    shutil.rmtree(full, ignore_errors=True)
    except OSError:
        pass
    return {"ok": True, "jobs": removed_jobs, "parts": removed_parts, "envs": removed_envs, "dry_run": dry}


def _walk_parts(root):
    for dp, dns, fns in os.walk(root):
        for fn in fns:
            if fn.endswith(PART) or fn.endswith(SIDE):
                ap = os.path.join(dp, fn)
                try:
                    yield os.path.relpath(ap, root).replace(os.sep, "/"), ap, os.stat(ap)
                except OSError:
                    continue


def op_spec(req, home, inp, out):
    jd = home.jobdir(req["project"], req["id"])
    return {"ok": True, "spec": read_json(os.path.join(jd, "spec.json"))}


def op_speedtest(req, home, inp, out):
    n = int(req.get("bytes") or (8 << 20))
    if req.get("direction") == "up":
        t0, got = time.time(), 0
        while got < n:
            b = inp.read(min(CHUNK, n - got))
            if not b:
                break
            got += len(b)
        return {"ok": True, "bytes": got, "seconds": time.time() - t0}
    _frame(out, {"ok": True, "bytes": n})
    block = os.urandom(CHUNK)
    sent = 0
    while sent < n:
        k = min(CHUNK, n - sent)
        out.write(block[:k])
        sent += k
    out.flush()
    return None


def op_wsl(req, home, inp, out):
    if not IS_WIN:
        return {"ok": True, "distros": [], "note": "not a Windows host"}
    try:
        p = subprocess.run(["wsl.exe", "-l", "-v"], capture_output=True, timeout=60, stdin=subprocess.DEVNULL)
    except Exception as e:
        return {"ok": True, "distros": [], "error": str(e)}
    text = p.stdout.decode("utf-16-le", "replace").replace("\x00", "")
    distros = []
    for line in text.splitlines()[1:]:
        f = line.replace("*", " ").split()
        if len(f) >= 3:
            distros.append({"name": f[0], "state": f[1], "version": f[2], "default": line.strip().startswith("*")})
    return {"ok": True, "distros": distros}


OPS = {"hello": op_hello, "sysinfo": op_sysinfo, "capacity": op_capacity, "plan": op_plan,
       "receive": op_receive, "manifest": op_manifest, "send": op_send, "launch": op_launch,
       "status": op_status, "spec": op_spec, "outputs": op_outputs, "overview": op_overview, "tail": op_tail,
       "cancel": op_cancel, "exec": op_exec, "gc": op_gc, "speedtest": op_speedtest, "wsl": op_wsl}


def rpc_main():
    inp, out = sys.stdin.buffer, sys.stdout.buffer
    line = inp.readline(64 << 20)
    try:
        req = json.loads(line.decode("utf-8"))
        op = OPS.get(req.get("op"))
        if op is None:
            raise RxError("unknown operation %r" % req.get("op"), "protocol")
        res = op(req, Home(), inp, out)
    except RxError as e:
        res = {"ok": False, "error": str(e), "kind": e.kind}
    except Exception as e:
        res = {"ok": False, "error": "%s: %s" % (type(e).__name__, e), "kind": "internal",
               "trace": traceback.format_exc()[-3000:]}
    if res is not None:
        _frame(out, res)
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--home" in argv:
        i = argv.index("--home")
        os.environ["RX_HOME"] = argv[i + 1]
        del argv[i:i + 2]
    if not argv:
        print("usage: rx_agent.py rpc | supervise JOBDIR | envbuild SPEC | version", file=sys.stderr)
        return 2
    cmd = argv[0]
    if cmd == "rpc":
        return rpc_main()
    if cmd == "supervise":
        return supervise_main(argv[1])
    if cmd == "envbuild":
        return envbuild_main(argv[1])
    if cmd == "version":
        print(agent_sha())
        return 0
    print("unknown command %r" % cmd, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
