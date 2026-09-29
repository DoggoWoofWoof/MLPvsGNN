#!/usr/bin/env python3
"""rx -- run a project's work on a remote machine over ssh and bring the results back.

  rx setup HOST [--route SSH-DEST]    one time: interpreter, agent, capacity on the host
  rx doctor [--speed MB]              check the route, the agent, the host, the throughput
  rx push [--dry-run] [--prune]       sync the files rx.toml declares (content-addressed, resumable)
  rx run [opts] -- CMD ...            push, then run CMD detached on the host
  rx ls / rx status JOB               jobs and their states
  rx logs JOB [-f]                    output; -f follows and reconnects by itself
  rx wait JOB... [--fetch]            block until jobs end; rides out network outages
  rx fetch JOB [--into DIR]           bring a job's outputs back (resumable, never clobbers)
  rx cancel JOB... / rx rerun JOB
  rx env build NAME / rx env ls       build a pinned Python environment on the host
  rx exec -- CMD                      a short command now (dies with the connection)
  rx monitor / rx dashboard           one view over every host, project and job
  rx watch                            fetch finished jobs whenever the host is reachable
  rx gc / rx capacity / rx hosts

Standard library only (Python 3.11+ here; the agent needs 3.8+ on the host).
See README.md beside this file for the protocol, the failure modes and the recovery playbook.
"""
from __future__ import annotations

import argparse
import hashlib
import http.server
import json
import os
import re
import secrets
import shlex
import shutil
import socket
import subprocess
import sys
import threading
import time
import tomllib
import urllib.parse

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import rx_agent as A  # noqa: E402

AGENT_FILE = os.path.join(HERE, "rx_agent.py")
TERMINAL = A.TERMINAL_STATES
ACTIVE = A.ACTIVE_STATES


def rx_dir():
    return os.environ.get("RX_CONFIG_DIR") or os.path.join(os.path.expanduser("~"), ".rx")


class TransportError(Exception):
    """The connection failed; the same request may be sent again."""


class RemoteError(Exception):
    def __init__(self, msg, kind="error"):
        super().__init__(msg)
        self.kind = kind


class Die(Exception):
    pass


def die(msg):
    raise Die(msg)


def say(msg, end="\n"):
    sys.stderr.write(msg + end)
    sys.stderr.flush()


def fmt_bytes(n):
    return A.fmt_bytes(n)


def fmt_dur(s):
    if s is None:
        return "-"
    s = int(max(0, s))
    if s < 60:
        return "%ds" % s
    if s < 3600:
        return "%dm%02ds" % (s // 60, s % 60)
    if s < 86400:
        return "%dh%02dm" % (s // 3600, (s % 3600) // 60)
    return "%dd%02dh" % (s // 86400, (s % 86400) // 3600)


def file_sha(path):
    return A.sha256_path(path)


# ─────────────────────────────── hosts and routes ───────────────────────────────

def host_path(name):
    return os.path.join(rx_dir(), "hosts", A.check_name(name, "host") + ".json")


def load_host(name):
    cfg = A.read_json(host_path(name))
    if not cfg:
        die("host %r is not set up here; run: rx setup %s" % (name, name))
    return cfg


def save_host(cfg):
    A.write_json(host_path(cfg["name"]), cfg)


def all_hosts():
    d = os.path.join(rx_dir(), "hosts")
    try:
        names = sorted(n[:-5] for n in os.listdir(d) if n.endswith(".json") and not n.endswith(".state.json"))
    except OSError:
        return []
    return [h for h in (A.read_json(os.path.join(d, n + ".json")) for n in names) if h]


def default_ssh():
    if os.name == "nt":
        git_ssh = os.path.join(os.environ.get("ProgramFiles", r"C:\Program Files"), "Git", "usr", "bin", "ssh.exe")
        if os.path.exists(git_ssh):
            return git_ssh
    return shutil.which("ssh") or "ssh"


class Remote:
    """One host. Every request is one ssh session running the agent; transport
    failures are retried with backoff, over the next route when there are several."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.name = cfg["name"]
        self.local = cfg.get("transport") == "local"
        self._agent_checked = False
        self.state_path = os.path.join(rx_dir(), "hosts", self.name + ".state.json")

    # -- routes
    def routes(self):
        return self.cfg.get("routes") or [[self.name]]

    def pick_route(self):
        routes = self.routes()
        bad = (A.read_json(self.state_path, {}) or {}).get("bad_until", {})
        now = time.time()
        for i, r in enumerate(routes):
            if bad.get(str(i), 0) < now:
                return i, r
        return 0, routes[0]

    def mark(self, i, ok):
        if len(self.routes()) < 2:
            return
        st = A.read_json(self.state_path, {}) or {}
        bad = st.setdefault("bad_until", {})
        if ok:
            if bad.pop(str(i), None) is None and st.get("last_good") == i:
                return
            st["last_good"] = i
        else:
            bad[str(i)] = time.time() + 120
        A.write_json(self.state_path, st)

    def ssh_base(self, route):
        opts = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=%d" % int(self.cfg.get("connect_timeout", 15)),
                "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=4", "-o", "Compression=no",
                "-o", "LogLevel=ERROR", "-T"]
        return [self.cfg.get("ssh") or default_ssh()] + opts + list(self.cfg.get("ssh_options") or []) + list(route)

    def windows(self):
        return self.cfg.get("os") == "windows"

    def remote_cmd(self, argv):
        """A command line for the host's login shell (cmd.exe on Windows OpenSSH)."""
        if self.windows():
            return " ".join('"%s"' % a if (" " in a or "\\" in a or not a) else a for a in argv)
        return " ".join(shlex.quote(a) for a in argv)

    def argv(self, route):
        if self.local:
            return [sys.executable, self.cfg.get("agent_path") or AGENT_FILE, "rpc"]
        return self.ssh_base(route) + [self.remote_cmd([self.cfg["python"], self.cfg["agent_path"], "rpc"])]

    def popen_env(self):
        if not self.local:
            return None
        return dict(os.environ, RX_HOME=self.cfg["home"], RX_LAUNCH=self.cfg.get("launch", "popen"))

    # -- the agent
    def ensure_agent(self, force=False):
        if self._agent_checked and not force:
            return
        with open(AGENT_FILE, "rb") as f:
            data = f.read()
        sha = hashlib.sha256(data).hexdigest()
        if force or self.cfg.get("agent_sha") != sha:
            path = self.agent_remote_path(sha)
            if self.local:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                shutil.copyfile(AGENT_FILE, path)
            else:
                self.upload_bootstrap(data, path)
            self.cfg["agent_sha"], self.cfg["agent_path"] = sha, path
            save_host(self.cfg)
        self._agent_checked = True

    def agent_remote_path(self, sha):
        sep = "\\" if self.windows() else "/"
        return self.cfg["home"].rstrip("\\/") + sep + "agent" + sep + "rx_agent-%s.py" % sha[:12]

    def upload_bootstrap(self, data, path):
        code = ("import sys,os,hashlib;p=sys.argv[1];d=sys.stdin.buffer.read();"
                "os.makedirs(os.path.dirname(p),exist_ok=True);t=p+'.part';"
                "f=open(t,'wb');f.write(d);f.close();os.replace(t,p);"
                "print(hashlib.sha256(d).hexdigest())")
        want = hashlib.sha256(data).hexdigest()
        last = None
        for attempt in range(4):
            i, route = self.pick_route()
            cmd = self.remote_cmd([self.cfg["python"], "-c", code, path])
            try:
                p = subprocess.run(self.ssh_base(route) + [cmd], input=data, capture_output=True, timeout=300)
            except subprocess.TimeoutExpired:
                last = "timed out"
                continue
            got = p.stdout.decode("utf-8", "replace").strip()
            if p.returncode == 0 and got == want:
                return
            last = "exit %d: %s %s" % (p.returncode, got[-200:], p.stderr.decode("utf-8", "replace")[-400:])
            if p.returncode != 255:
                break
            time.sleep(2 * (attempt + 1))
        die("could not install the agent on %s: %s" % (self.name, last))

    # -- requests
    def call(self, req, send=None, recv=None, retry_for=60.0, quiet=False):
        """Send one request. `send(fp)` streams after the request line; `recv(resp, fp)`
        reads a stream after the response line and its value is returned."""
        self.ensure_agent()
        deadline = time.time() + retry_for
        delay, reinstalled, warned = 1.0, False, False
        while True:
            i, route = self.pick_route()
            try:
                out = self._once(route, req, send, recv)
                self.mark(i, True)
                if warned and not quiet:
                    say("[rx] %s: reconnected" % self.name)
                return out
            except AgentMissing:
                if reinstalled:
                    raise RemoteError("the agent on %s cannot be started" % self.name, "agent")
                self.ensure_agent(force=True)
                reinstalled = True
            except TransportError as e:
                self.mark(i, False)
                if time.time() + delay > deadline:
                    raise
                if not quiet:
                    say("[rx] %s: %s -- retrying in %.0fs" % (self.name, str(e).strip()[:160], delay))
                    warned = True
                time.sleep(delay)
                delay = min(delay * 2, 30.0)

    def _once(self, route, req, send, recv):
        kw = {}
        if os.name == "nt" and self.cfg.get("no_window"):
            kw["creationflags"] = 0x08000000
        try:
            p = subprocess.Popen(self.argv(route), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE, env=self.popen_env(), **kw)
        except OSError as e:
            raise TransportError("cannot start ssh: %s" % e)
        err = []
        t = threading.Thread(target=lambda: err.append(p.stderr.read()), daemon=True)
        t.start()
        resp, result, broken = None, None, None
        try:
            try:
                p.stdin.write((json.dumps(req, separators=(",", ":")) + "\n").encode("utf-8"))
                if send:
                    send(p.stdin)
                p.stdin.close()
            except OSError:
                pass                            # the far end went away; its reply (if any) says why
            line = p.stdout.readline()
            if line:
                try:
                    resp = json.loads(line.decode("utf-8"))
                except ValueError:
                    broken = "unreadable reply %r" % line[:120]
                if resp is not None and resp.get("ok") and recv:
                    try:
                        result = recv(resp, p.stdout)
                    except (EOFError, A.ProtocolError, OSError, ValueError) as e:
                        broken = "stream interrupted: %s" % e
        except BaseException:
            p.kill()
            raise
        finally:
            if broken or resp is None:
                try:
                    p.kill()
                except OSError:
                    pass
        rc = p.wait()
        t.join(5)
        stderr = b"".join(x for x in err if x).decode("utf-8", "replace").strip()
        if resp is None or broken:
            if "can't open file" in stderr or "No such file or directory" in stderr and "rx_agent" in stderr:
                raise AgentMissing(stderr)
            if resp is None and rc not in (255, None) and stderr and not self.local and "Traceback" in stderr:
                raise RemoteError("the agent failed (exit %s): %s" % (rc, stderr[-1500:]), "agent")
            raise TransportError(broken or stderr or "no reply (exit %s)" % rc)
        if not resp.get("ok"):
            raise RemoteError(resp.get("error") or "failed", resp.get("kind") or "error")
        return result if recv else resp


class AgentMissing(Exception):
    pass


# ─────────────────────────────── projects ───────────────────────────────

def find_project(start):
    d = os.path.abspath(start)
    while True:
        if os.path.isfile(os.path.join(d, "rx.toml")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            return None
        d = parent


class Project:
    def __init__(self, root):
        self.root = root
        with open(os.path.join(root, "rx.toml"), "rb") as f:
            self.cfg = tomllib.load(f)
        p = self.cfg.get("project", {})
        self.name = A.check_name(p.get("name") or os.path.basename(root), "project")
        self.host = p.get("host")
        self.state = os.path.join(root, ".rx")
        self.push_cfg = self.cfg.get("push", {})
        self.run_cfg = self.cfg.get("run", {})
        self.profiles = self.cfg.get("profiles", {})
        self.envs = self.cfg.get("envs", {})

    def jobs_path(self):
        return os.path.join(self.state, "jobs.json")

    def jobs(self):
        return (A.read_json(self.jobs_path(), {}) or {}).get("jobs", {})

    def record_job(self, jid, **kw):
        with A.FileLock(self.jobs_path() + ".lock"):
            d = A.read_json(self.jobs_path(), {}) or {}
            jobs = d.setdefault("jobs", {})
            jobs.setdefault(jid, {"id": jid}).update(kw)
            A.write_json(self.jobs_path(), d)


def register_project(proj):
    path = os.path.join(rx_dir(), "projects.json")
    with A.FileLock(path + ".lock"):
        reg = A.read_json(path, {}) or {}
        cur = reg.get(proj.root, {})
        if cur.get("name") != proj.name or cur.get("host") != proj.host or time.time() - cur.get("seen", 0) > 3600:
            reg[proj.root] = {"name": proj.name, "host": proj.host, "seen": time.time()}
            A.write_json(path, reg)


def registered_projects():
    reg = A.read_json(os.path.join(rx_dir(), "projects.json"), {}) or {}
    out = []
    for root in reg:
        if os.path.isfile(os.path.join(root, "rx.toml")):
            try:
                out.append(Project(root))
            except Exception:
                continue
    return out


def git(root, *args):
    return subprocess.run(["git", "-C", root] + list(args), capture_output=True, check=True).stdout


def git_info(root):
    try:
        head = git(root, "rev-parse", "HEAD").decode().strip()
        dirty = bool(git(root, "status", "--porcelain", "--untracked-files=no").strip())
        return {"head": head, "dirty": dirty}
    except Exception:
        return None


class HashCache:
    """rel -> [size, mtime_ns, sha]; a sha is reused only while size and mtime still match."""

    def __init__(self, path):
        self.path = path
        self.d = (A.read_json(path, {}) or {}).get("files", {})
        self.dirty = False

    def sha(self, rel, ap, st):
        e = self.d.get(rel)
        if e and e[0] == st.st_size and e[1] == st.st_mtime_ns:
            return e[2]
        s = file_sha(ap)
        self.d[rel] = [st.st_size, st.st_mtime_ns, s]
        self.dirty = True
        return s

    def save(self):
        if self.dirty:
            A.write_json(self.path, {"v": 1, "files": self.d})
            self.dirty = False


def local_manifest(proj, extra=()):
    """{rel: (abs_path, stat)} for everything the project pushes."""
    inc = list(proj.push_cfg.get("include") or ["git:tracked"]) + list(extra or [])
    exc = list(proj.push_cfg.get("exclude") or []) + [".rx/**", ".git/**"]
    excg = [A.Glob(x) for x in exc]
    files, globs = {}, []
    for pat in inc:
        if pat in ("git:tracked", "git:untracked"):
            args = ["ls-files", "-z"] + (["--others", "--exclude-standard"] if pat == "git:untracked" else [])
            try:
                out = git(proj.root, *args)
            except Exception as e:
                die("%s needs a git repository at %s (%s)" % (pat, proj.root, e))
            for rel in out.decode("utf-8", "surrogateescape").split("\0"):
                if not rel or rel in files or any(g.match(rel) for g in excg):
                    continue
                ap = os.path.join(proj.root, *rel.split("/"))
                try:
                    st = os.stat(ap)
                except OSError:
                    continue                 # tracked but deleted in the working tree
                if os.path.isfile(ap):
                    files[rel] = (ap, st)
        else:
            globs.append(pat)
    for rel, ap, st in A.walk_files(proj.root, globs, exc):
        files.setdefault(rel, (ap, st))
    bad = []
    for rel in list(files):
        try:
            A.safe_rel(rel)
        except A.RxError:
            bad.append(rel)
            del files[rel]
    if bad:
        say("[rx] skipping %d path(s) the host cannot hold, e.g. %s" % (len(bad), bad[0]))
    return files


# ─────────────────────────────── progress ───────────────────────────────

class Progress:
    def __init__(self, total, label, enabled=True):
        self.total, self.label = total, label
        self.done = 0
        self.t0 = time.time()
        self.lock = threading.Lock()
        self.enabled = enabled and sys.stderr.isatty()
        self._stop = threading.Event()
        if self.enabled:
            threading.Thread(target=self._loop, daemon=True).start()

    def add(self, n):
        with self.lock:
            self.done += n

    def _loop(self):
        while not self._stop.wait(0.5):
            self._print()

    def rate(self):
        return self.done / max(time.time() - self.t0, 1e-6)

    def _print(self, final=False):
        r = self.rate()
        eta = (self.total - self.done) / r if r > 0 else 0
        line = "  %s %s / %s  %s/s  eta %s" % (self.label, fmt_bytes(self.done), fmt_bytes(self.total),
                                                 fmt_bytes(r), fmt_dur(eta))
        sys.stderr.write("\r" + line[:100].ljust(100) + ("\n" if final else ""))
        sys.stderr.flush()

    def close(self):
        self._stop.set()
        if self.enabled:
            self._print(final=True)


def partition(items, k):
    """Split (key, bytes) items into k groups of similar total bytes."""
    groups = [[] for _ in range(k)]
    loads = [0] * k
    for key, n in sorted(items, key=lambda x: -x[1]):
        i = loads.index(min(loads))
        groups[i].append(key)
        loads[i] += n
    return [g for g in groups if g]


def run_parallel(fns):
    """Run callables in threads; return a list of (value, exception)."""
    out = [None] * len(fns)

    def wrap(i, fn):
        try:
            out[i] = (fn(), None)
        except BaseException as e:  # noqa: BLE001 - reported to the caller
            out[i] = (None, e)

    ts = [threading.Thread(target=wrap, args=(i, fn), daemon=True) for i, fn in enumerate(fns)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    return out


# ─────────────────────────────── push ───────────────────────────────

def push(proj, remote, extra=(), ws="ws", dry=False, prune=False, streams=None, force=False,
         retry_for=1800.0, quiet=False):
    """Make the host's workspace hold the declared files. Returns a summary dict
    including `digest`, a sha over the (path, sha) list that was pushed."""
    t0 = time.time()
    files = local_manifest(proj, extra)
    cache = HashCache(os.path.join(proj.state, "hashcache.json"))
    entries, sizes = [], {}
    for rel in sorted(files):
        ap, st = files[rel]
        entries.append([rel, st.st_size, cache.sha(rel, ap, st), st.st_mtime_ns])
        sizes[rel] = st.st_size
    cache.save()
    digest = hashlib.sha256("\n".join("%s %s" % (e[0], e[2]) for e in entries).encode()).hexdigest()
    total = sum(sizes.values())
    max_mb = float(proj.push_cfg.get("max_mb", 2048))
    k_max = int(streams or remote.cfg.get("streams") or 4)
    level = int(remote.cfg.get("compress_level", 6))
    deadline = time.time() + retry_for
    sent_raw = sent_wire = 0
    last_errors, rounds, delay = None, 0, 2.0
    summary = {}
    while True:
        rounds += 1
        plan = remote.call({"op": "plan", "project": proj.name, "ws": ws, "files": entries,
                            "prune": bool(prune and rounds == 1 and not dry)},
                           retry_for=max(30.0, deadline - time.time()), quiet=quiet)
        need = plan["need"]
        if rounds == 1:
            summary = {"files": len(entries), "bytes": total, "have": plan["have"], "copied": plan["copied"],
                       "deleted": len(plan.get("deleted") or []), "need": len(need), "need_bytes": plan["need_bytes"]}
            if not quiet:
                say("[rx] push %s -> %s:%s  %d files, %s  (%d unchanged, %d copied on host, %d to send: %s)%s" % (
                    proj.name, remote.name, ws, len(entries), fmt_bytes(total), plan["have"], plan["copied"],
                    len(need), fmt_bytes(plan["need_bytes"]),
                    ", %d pruned" % summary["deleted"] if summary["deleted"] else ""))
                if plan.get("kept_changed"):
                    say("[rx] kept %d pushed file(s) that changed on the host (not pruned)" % len(plan["kept_changed"]))
            if dry:
                for rel, off in need:
                    say("    would send %s (%s%s)" % (rel, fmt_bytes(sizes[rel] - off), ", resuming" if off else ""))
                summary["digest"] = digest
                return summary
            if plan["need_bytes"] > max_mb * 2 ** 20 and not force:
                die("this push would send %s, over push.max_mb=%g in rx.toml; pass --force, or narrow the "
                    "inputs (large data is usually cheaper to fetch or build on the host)" % (
                        fmt_bytes(plan["need_bytes"]), max_mb))
        if not need:
            break
        if rounds > 6:
            die("files keep failing to land on %s: %s" % (remote.name, last_errors))
        byrel = dict((rel, off) for rel, off in need)
        nbytes = sum(sizes[r] - o for r, o in need)
        k = max(1, min(k_max, len(need), int(nbytes / (4 << 20)) + 1))
        groups = partition([(rel, sizes[rel] - byrel[rel]) for rel in byrel], k)
        prog = Progress(nbytes, "push", enabled=not quiet)
        writers = []
        sha_of = dict((e[0], e[2]) for e in entries)

        def stream_group(group):
            def send(fp):
                w = A.StreamWriter(fp, level=level, progress=prog.add)
                writers.append(w)
                for rel in group:
                    w.send_file(proj.root, rel, byrel[rel], sha_of[rel])
                w.end()
            return remote.call({"op": "receive", "project": proj.name, "ws": ws}, send=send,
                               retry_for=0, quiet=True)

        results = run_parallel([(lambda g=g: stream_group(g)) for g in groups])
        prog.close()
        sent_raw += sum(w.raw for w in writers)
        sent_wire += sum(w.wire for w in writers)
        failed = [e for _, e in results if e is not None]
        errors = [x for v, _ in results if v for x in v.get("errors") or []]
        for e in failed:
            if not isinstance(e, TransportError):
                raise e
        if errors:
            last_errors = "; ".join("%s: %s" % (x["path"], x["error"]) for x in errors[:5])
            if not quiet:
                say("[rx] %d file(s) did not land (%s); retrying" % (len(errors), last_errors))
        if failed:
            if time.time() > deadline:
                die("push interrupted and the host stayed unreachable; run the same command again to resume")
            if not quiet:
                say("[rx] connection lost during the push; resuming in %.0fs" % delay)
            time.sleep(delay)
            delay = min(delay * 2, 60)
    el = time.time() - t0
    summary.update({"digest": digest, "sent_raw": sent_raw, "sent_wire": sent_wire, "seconds": el})
    if not quiet and sent_raw:
        say("[rx] pushed %s (%s on the wire) in %s, %s/s" % (
            fmt_bytes(sent_raw), fmt_bytes(sent_wire), fmt_dur(el), fmt_bytes(sent_wire / max(el, 1e-6))))
    return summary


# ─────────────────────────────── fetch ───────────────────────────────

class Receipts:
    """What rx fetch wrote locally: abs path -> [size, mtime_ns, sha]. A local file
    that still matches its receipt was written by rx and may be replaced; any other
    local file is the user's and is never overwritten without --overwrite."""

    def __init__(self, proj_state):
        self.path = os.path.join(proj_state, "receipts.json")
        self.d = (A.read_json(self.path, {}) or {}).get("files", {})
        self.lock = threading.Lock()

    @staticmethod
    def key(p):
        return os.path.normcase(os.path.abspath(p))

    def get(self, p):
        return self.d.get(self.key(p))

    def set(self, p, size, mtime_ns, sha):
        with self.lock:
            self.d[self.key(p)] = [size, mtime_ns, sha]

    def save(self):
        with self.lock:
            A.write_json(self.path, {"v": 1, "files": self.d})


def fetch_files(proj, remote, files, ws="ws", into=None, overwrite=False, streams=None,
                retry_for=1800.0, quiet=False):
    """files: [rel, size, mtime_ns, sha]. Returns (fetched, have, conflicts)."""
    dest_root = os.path.abspath(into) if into else proj.root
    os.makedirs(proj.state, exist_ok=True)
    rec = Receipts(proj.state)
    want = {f[0]: f for f in files}
    conflicts, have = [], 0
    need = []
    for rel, (r, size, mtime_ns, sha) in sorted(want.items()):
        A.safe_rel(rel)
        dest = os.path.join(dest_root, *rel.split("/"))
        if os.path.isfile(dest):
            st = os.stat(dest)
            e = rec.get(dest)
            ours = bool(e and e[0] == st.st_size and e[1] == st.st_mtime_ns)
            local_sha = e[2] if ours else (file_sha(dest) if st.st_size == size else None)
            if local_sha == sha:
                have += 1
                if not ours:
                    rec.set(dest, st.st_size, st.st_mtime_ns, sha)
                continue
            if not (ours or overwrite):
                conflicts.append(rel)
                continue
        elif os.path.exists(dest):
            conflicts.append(rel)
            continue
        need.append(rel)
    if conflicts and not quiet:
        say("[rx] %d file(s) differ locally and were not written by rx fetch; left untouched "
            "(use --overwrite, or --into DIR):" % len(conflicts))
        for rel in conflicts[:10]:
            say("    %s" % rel)
        if len(conflicts) > 10:
            say("    ... and %d more" % (len(conflicts) - 10))
    fetched = 0
    deadline = time.time() + retry_for
    delay = 2.0
    k_max = int(streams or remote.cfg.get("streams") or 4)
    level = int(remote.cfg.get("compress_level", 6))
    while need:
        todo = []
        for rel in need:
            dest = os.path.join(dest_root, *rel.split("/"))
            todo.append([rel, A.partial_offset(dest, want[rel][3]), want[rel][3]])
        nbytes = sum(want[r][1] - o for r, o, _ in todo)
        k = max(1, min(k_max, len(todo), int(nbytes / (4 << 20)) + 1))
        groups = partition([(tuple(t), want[t[0]][1] - t[1]) for t in todo], k)
        prog = Progress(nbytes, "fetch", enabled=not quiet)
        done = []

        def fetch_group(group):
            items = [list(x) for x in group]

            def recv(resp, fp):
                def on_done(rel, size, mtime_ns, sha):
                    rec.set(os.path.join(dest_root, *rel.split("/")), size, mtime_ns, sha)
                    done.append(rel)
                    prog.add(size)
                return A.receive_stream(A.StreamReader(fp), dest_root, on_done)
            return remote.call({"op": "send", "project": proj.name, "ws": ws, "files": items,
                                "level": level}, recv=recv, retry_for=0, quiet=True)

        results = run_parallel([(lambda g=g: fetch_group(g)) for g in groups])
        prog.close()
        rec.save()
        fetched += len(done)
        failed = [e for _, e in results if e is not None]
        for e in failed:
            if not isinstance(e, TransportError):
                raise e
        errs = [x for v, _ in results if v for x in v if not x.get("ok")]
        need = [r for r in need if r not in set(done)]
        missing = [x["path"] for x in errs if "missing" in x.get("error", "")]
        need = [r for r in need if r not in set(missing)]
        if missing and not quiet:
            say("[rx] %d file(s) no longer exist on the host, e.g. %s" % (len(missing), missing[0]))
        if not need:
            break
        if not failed and errs and delay > 30:
            die("fetch keeps failing: %s" % "; ".join("%s: %s" % (x["path"], x["error"]) for x in errs[:5]))
        if time.time() > deadline:
            die("fetch interrupted and the host stayed unreachable; run the same command again to resume")
        if not quiet:
            say("[rx] %s; resuming %d file(s) in %.0fs" % (
                "connection lost" if failed else "some files did not verify", len(need), delay))
        time.sleep(delay)
        delay = min(delay * 2, 60)
    return fetched, have, conflicts


def fetch_job(proj, remote, jid, into=None, overwrite=False, include=None, exclude=None, quiet=False,
              retry_for=1800.0):
    r = remote.call({"op": "outputs", "project": proj.name, "id": jid}, retry_for=min(retry_for, 300))
    job, outs = r.get("job") or {}, r.get("outputs")
    if outs is None:
        if job.get("state") in ACTIVE:
            die("%s is still %s; its outputs are listed when it ends (rx fetch --glob PATTERN pulls "
                "files now)" % (jid, job.get("state")))
        die("%s has no outputs record" % jid)
    files = outs.get("files") or []
    if include:
        g = [A.Glob(x) for x in include]
        files = [f for f in files if any(x.match(f[0]) for x in g)]
    if exclude:
        g = [A.Glob(x) for x in exclude]
        files = [f for f in files if not any(x.match(f[0]) for x in g)]
    total = sum(f[1] for f in files)
    if not quiet:
        say("[rx] fetch %s: %d output file(s), %s%s" % (jid, len(files), fmt_bytes(total),
                                                       " into %s" % into if into else ""))
    fetched, have, conflicts = fetch_files(proj, remote, files, ws=job.get("ws") or "ws", into=into,
                                           overwrite=overwrite, quiet=quiet, retry_for=retry_for)
    if not quiet:
        say("[rx] fetched %d, already here %d, conflicts %d" % (fetched, have, len(conflicts)))
    proj.record_job(jid, fetched=time.time(), fetch_conflicts=len(conflicts))
    return fetched, have, conflicts


# ─────────────────────────────── jobs ───────────────────────────────

def make_id(name):
    name = re.sub(r"[^A-Za-z0-9._-]+", "-", name or "job").strip("-._")[:40] or "job"
    return "%s-%s-%s" % (time.strftime("%y%m%d-%H%M%S"), name, secrets.token_hex(2))


def derive_name(argv):
    for a in argv:
        if a.endswith(".py") or a.endswith(".sh") or a.endswith(".ps1"):
            return os.path.splitext(os.path.basename(a.replace("\\", "/")))[0]
    if len(argv) >= 3 and argv[1] == "-m":
        return argv[2].split(".")[-1]
    return os.path.splitext(os.path.basename(argv[0].replace("\\", "/")))[0] if argv else "job"


def resolve_job(proj, remote, ref):
    local = sorted(proj.jobs().values(), key=lambda j: j.get("created") or 0)
    ids = [j["id"] for j in local]
    if ref in (None, "", "last", "@"):
        if ids:
            return ids[-1]
        r = remote.call({"op": "status", "project": proj.name, "limit": 1})
        if r["jobs"]:
            return r["jobs"][0]["id"]
        die("no jobs yet")
    if ref in ids:
        return ref
    m = [i for i in ids if ref in i]
    if len(m) == 1:
        return m[0]
    if len(m) > 1:
        die("%r matches %d jobs: %s" % (ref, len(m), ", ".join(m[-5:])))
    r = remote.call({"op": "status", "project": proj.name, "limit": 500})
    m = [j["id"] for j in r["jobs"] if ref == j["id"] or ref in j["id"]]
    if len(m) == 1:
        return m[0]
    die("no single job matches %r" % ref if not m else "%r matches %d jobs" % (ref, len(m)))


def res_str(j):
    req = j.get("req") or {}
    parts = ["%g cpu" % req.get("cpus", 0), "%gG" % req.get("mem_gb", 0)]
    if req.get("gpus"):
        parts.append("%g gpu" % req["gpus"])
    return " ".join(parts)


def job_line(j, now):
    st = j.get("state", "?")
    start = j.get("started") or j.get("queued_at") or j.get("created")
    end = j.get("ended") or now
    dur = end - start if (start and st not in ("launching", "queued")) else None
    return "%-34s %-10s %7s %8s  %-18s %4s  %s" % (
        j.get("id", "?")[:34], st, fmt_dur(now - (j.get("created") or now)), fmt_dur(dur) if dur else "-",
        res_str(j), "" if j.get("rc") is None else j.get("rc"), (j.get("display") or "")[:60])


def print_jobs(jobs, now, header=True):
    if header:
        print("%-34s %-10s %7s %8s  %-18s %4s  %s" % ("JOB", "STATE", "AGE", "RUNTIME", "RESOURCES", "RC", "COMMAND"))
    for j in jobs:
        print(job_line(j, now))
        if j.get("state") in ("running", "finishing") and j.get("last_line"):
            print("%-34s   > %s" % ("", j["last_line"][:110]))
        elif j.get("state") == "queued" and j.get("waiting"):
            print("%-34s   ~ %s" % ("", j["waiting"][:110]))
        elif j.get("state") == "lost":
            print("%-34s   ! %s" % ("", j.get("lost_reason", "")))
        elif j.get("state") == "error" and j.get("error"):
            print("%-34s   ! %s" % ("", j["error"][:110]))


def follow_logs(proj, remote, jid, follow=True, tail_kb=None, out=None):
    out = out or sys.stdout.buffer
    offset, first, delay, offline = 0, True, 2.0, None
    while True:
        req = {"op": "tail", "project": proj.name, "id": jid, "offset": offset, "follow": follow,
               "timeout": 1800}
        if first and tail_kb is not None:
            req["tail_bytes"] = int(tail_kb * 1024)

        def recv(resp, fp):
            nonlocal offset
            offset = resp["offset"]
            while True:
                line = fp.readline()
                if not line:
                    raise EOFError("log stream ended")
                fr = json.loads(line.decode("utf-8"))
                if fr["t"] == "d":
                    data = fp.read(fr["n"])
                    if len(data) != fr["n"]:
                        raise EOFError("log stream ended inside a frame")
                    out.write(data)
                    out.flush()
                    offset += fr["n"]
                elif fr["t"] == "end":
                    return fr
        try:
            end = remote.call(req, recv=recv, retry_for=0 if follow else 60, quiet=True)
            first = False
            if offline is not None:
                say("[rx] reconnected after %s" % fmt_dur(time.time() - offline))
                offline = None
        except TransportError as e:
            if not follow:
                raise
            first = False
            if offline is None:
                offline = time.time()
                say("\n[rx] connection lost (%s); the job keeps running on %s -- reconnecting" % (
                    str(e).strip()[:120], remote.name))
            time.sleep(delay)
            delay = min(delay * 2, 30)
            continue
        delay = 2.0
        if not follow or end.get("state") in TERMINAL:
            return end


def wait_jobs(proj, remote, ids, interval=15.0, fetch=False, into=None, quiet=False):
    pending = list(ids)
    last_lines, offline, finished = {}, None, {}
    while pending:
        try:
            r = remote.call({"op": "status", "project": proj.name, "ids": pending, "with_line": True},
                            retry_for=0, quiet=True)
        except TransportError as e:
            if offline is None:
                offline = time.time()
                say("[rx] %s unreachable (%s); jobs keep running there -- waiting" % (remote.name, str(e)[:120]))
            time.sleep(min(60.0, interval * 2))
            continue
        if offline is not None:
            say("[rx] %s reachable again after %s" % (remote.name, fmt_dur(time.time() - offline)))
            offline = None
        seen = set()
        for j in r["jobs"]:
            seen.add(j["id"])
            st = j.get("state")
            if st in TERMINAL:
                finished[j["id"]] = j
                pending.remove(j["id"])
                proj.record_job(j["id"], state=st, rc=j.get("rc"))
                if not quiet:
                    say("[rx] %s %s%s (runtime %s)" % (j["id"], st, "" if j.get("rc") is None else " rc=%s" % j["rc"],
                                                      fmt_dur((j.get("ended") or r["now"]) - (j.get("started") or j.get("created") or r["now"]))))
                if fetch and st in ("done", "failed", "cancelled"):
                    try:
                        fetch_job(proj, remote, j["id"], into=into, quiet=quiet)
                    except Die as e:
                        say("[rx] fetch of %s: %s" % (j["id"], e))
            elif not quiet:
                line = j.get("last_line") or j.get("waiting") or st
                if last_lines.get(j["id"]) != line:
                    last_lines[j["id"]] = line
                    say("[rx] %s %s: %s" % (j["id"][-24:], st, line[:120]))
        for jid in [p for p in pending if p not in seen]:
            say("[rx] %s is not known on %s" % (jid, remote.name))
            pending.remove(jid)
        if pending:
            time.sleep(interval)
    return finished


# ─────────────────────────────── envs ───────────────────────────────

def env_spec(proj, name, host_cfg):
    if name not in proj.envs:
        die("no [envs.%s] in rx.toml" % name)
    e = proj.envs[name]
    cons = list(e.get("constraints") or [])
    if e.get("constraints_file"):
        with open(os.path.join(proj.root, e["constraints_file"]), encoding="utf-8") as f:
            cons += [x.strip() for x in f if x.strip() and not x.strip().startswith("#")]
    spec = {"name": name, "python": e.get("python") or host_cfg.get("python_version"),
            "requirements": list(e.get("requirements") or []), "constraints": cons,
            "index_url": e.get("index_url"), "extra_index_urls": list(e.get("extra_index_urls") or []),
            "find_links": list(e.get("find_links") or []), "pip_args": list(e.get("pip_args") or []),
            "verify": list(e.get("verify") or [])}
    spec["hash"] = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    return spec


# ─────────────────────────────── commands ───────────────────────────────

def context(args, need_project=True):
    root = find_project(args.chdir or os.getcwd())
    proj = Project(root) if root else None
    if need_project and proj is None:
        die("no rx.toml here or above (tools/rx/example.rx.toml is a template)")
    host = args.host or (proj.host if proj else None)
    if not host:
        die("no host: pass --host NAME or set [project] host in rx.toml")
    remote = Remote(load_host(host))
    if proj:
        os.makedirs(proj.state, exist_ok=True)
        register_project(proj)
    return proj, remote


def cmd_setup(args):
    name = A.check_name(args.name, "host")
    old = A.read_json(host_path(name)) or {}
    if args.local:
        home = os.path.abspath(args.home or os.path.join(rx_dir(), "local-home"))
        cfg = {"name": name, "transport": "local", "os": "windows" if os.name == "nt" else "posix",
               "home": home, "python": sys.executable, "launch": args.launch or "popen", "routes": [["local"]]}
        save_host(cfg)
        r = Remote(cfg)
        hello = r.call({"op": "hello"})
        cap = r.call({"op": "capacity"})["capacity"]
        say("[rx] local host %s ready at %s (%s cpus, %s GB)" % (name, home, cap["cpus"], cap["mem_gb"]))
        return 0
    routes = [shlex.split(x) for x in (args.route or [])] or old.get("routes") or [[name]]
    cfg = {"name": name, "routes": routes, "ssh": args.ssh or old.get("ssh") or default_ssh(),
           "streams": old.get("streams", 4), "compress_level": old.get("compress_level", 6),
           "connect_timeout": old.get("connect_timeout", 15)}
    r = Remote(cfg)
    base = r.ssh_base(routes[0])
    say("[rx] probing %s over %s" % (name, " ".join(routes[0])))
    p = subprocess.run(base + ["echo %OS%& echo %USERPROFILE%"], capture_output=True, timeout=120)
    if p.returncode == 255:
        die("ssh to %s failed: %s" % (name, p.stderr.decode("utf-8", "replace").strip()))
    lines = [x.strip() for x in p.stdout.decode("utf-8", "replace").splitlines() if x.strip()]
    if lines and lines[0] == "Windows_NT":
        cfg["os"] = "windows"
        profile = lines[1]
        cfg["home"] = args.home or old.get("home") or profile.rstrip("\\") + "\\rx"
        ver = args.python_version
        if args.python:
            cfg["python"] = args.python
        else:
            cfg["python"] = cfg["home"] + "\\python\\%s\\python.exe" % ver
            install_windows_python(r, cfg, ver)
        cfg["python_version"] = ver
    else:
        cfg["os"] = "posix"
        p = subprocess.run(base + ['echo "$HOME"; command -v python3 || true'], capture_output=True, timeout=120)
        lines = [x.strip() for x in p.stdout.decode("utf-8", "replace").splitlines() if x.strip()]
        cfg["home"] = args.home or old.get("home") or lines[0].rstrip("/") + "/rx"
        cfg["python"] = args.python or (lines[1] if len(lines) > 1 else None)
        if not cfg["python"]:
            die("no python3 on %s; install Python 3.8+ there or pass --python PATH" % name)
        cfg["python_version"] = None
    save_host(cfg)
    r = Remote(cfg)
    r.ensure_agent(force=True)
    hello = r.call({"op": "hello"})
    info = r.call({"op": "sysinfo"})
    cap = info["capacity"]
    si = info["sysinfo"]
    say("[rx] %s ready: %s, python %s, agent %s" % (name, si["platform"], hello["python"].split()[0],
                                                   hello["agent"][:12]))
    say("[rx] cpus %s, memory %s GB, gpus: %s" % (si.get("cpus"), si.get("mem_total_gb"),
                                                ", ".join("%d %s %.0fGB" % (g["index"], g["name"], g["mem_total_gb"])
                                                          for g in si.get("gpus") or []) or "none"))
    say("[rx] scheduler capacity: %g cpus (reserve %g), %g GB (reserve %g), %d gpu(s)" % (
        cap["cpus"], cap.get("reserve_cpus", 0), cap["mem_gb"], cap.get("reserve_mem_gb", 0), len(cap.get("gpus") or [])))
    return 0


PS_INSTALL = r"""
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$ver = '__VER__'
$root = '__HOME__'
$dest = Join-Path $root "python\$ver"
if (Test-Path "$dest\python.exe") { "RXPY present $dest"; exit 0 }
$dl = Join-Path $root 'downloads'
New-Item -ItemType Directory -Force $dl | Out-Null
$pkg = Join-Path $dl "python.$ver.nupkg"
if (-not (Test-Path $pkg)) { curl.exe -sSL --fail -o $pkg "https://api.nuget.org/v3-flatcontainer/python/$ver/python.$ver.nupkg" }
$tmp = Join-Path $dl "unpack.$ver"
if (Test-Path $tmp) { Remove-Item -Recurse -Force $tmp }
New-Item -ItemType Directory -Force $tmp | Out-Null
tar.exe -xf $pkg -C $tmp
New-Item -ItemType Directory -Force (Split-Path $dest) | Out-Null
Move-Item (Join-Path $tmp 'tools') $dest
Remove-Item -Recurse -Force $tmp
$sig = Get-AuthenticodeSignature "$dest\python.exe"
if ($sig.Status -ne 'Valid') { throw "python.exe signature is $($sig.Status)" }
"RXPY installed $dest signed by $($sig.SignerCertificate.Subject)"
"""


def install_windows_python(r, cfg, ver):
    script = PS_INSTALL.replace("__VER__", ver).replace("__HOME__", cfg["home"].replace("'", "''"))
    enc = __import__("base64").b64encode(script.encode("utf-16-le")).decode()
    p = subprocess.run(r.ssh_base(r.routes()[0]) + ["powershell -NoProfile -NonInteractive -EncodedCommand " + enc],
                       capture_output=True, timeout=1800)
    out = p.stdout.decode("utf-8", "replace")
    m = re.search(r"RXPY (present|installed) .*", out)
    if p.returncode != 0 or not m:
        die("installing Python %s on %s failed: %s %s" % (ver, cfg["name"], out[-800:],
                                                       p.stderr.decode("utf-8", "replace")[-800:]))
    say("[rx] python %s: %s" % (ver, m.group(0)[5:].strip()))


def cmd_hosts(args):
    for h in all_hosts():
        print("%-12s %-8s %-40s routes: %s" % (h["name"], h.get("os"), h.get("home"),
                                             " | ".join(" ".join(r) for r in h.get("routes") or [])))
    return 0


def cmd_doctor(args):
    proj, remote = context(args, need_project=False)
    say("[rx] local: python %s, ssh %s" % (sys.version.split()[0], remote.cfg.get("ssh") or default_ssh()))
    ok = True
    for i, route in enumerate(remote.routes()):
        t0 = time.time()
        try:
            r = Remote(dict(remote.cfg, routes=[route]))
            h = r.call({"op": "hello"}, retry_for=0, quiet=True)
            dt = time.time() - t0
            skew = h["now"] - (t0 + dt / 2)
            say("[rx] route %d (%s): ok in %.2fs, clock skew %+.1fs, agent %s%s" % (
                i, " ".join(route), dt, skew, h["agent"][:12],
                "" if h["agent"] == remote.cfg.get("agent_sha") else " (stale -- will be replaced)"))
        except (TransportError, RemoteError) as e:
            ok = False
            say("[rx] route %d (%s): FAILED %s" % (i, " ".join(route), str(e)[:200]))
    try:
        info = remote.call({"op": "sysinfo"}, retry_for=10)
    except (TransportError, RemoteError) as e:
        die("host unreachable: %s" % e)
    si, cap = info["sysinfo"], info["capacity"]
    say("[rx] host %s: %s, python %s, up %s" % (si["host"], si["platform"], si["python"],
                                               fmt_dur(si["now"] - (si.get("boot_time") or si["now"]))))
    say("[rx] cpu %s%% of %s, memory %s free of %s GB, disk %s GB free" % (
        si.get("cpu_pct"), si.get("cpus"), si.get("mem_avail_gb"), si.get("mem_total_gb"), si.get("disk_free_gb")))
    for g in si.get("gpus") or []:
        say("[rx] gpu %d %s: %s%% util, %s/%s GB, %s C, %s W" % (
            g["index"], g["name"], g["util_pct"], g["mem_used_gb"], g["mem_total_gb"], g["temp_c"], g["power_w"]))
    say("[rx] capacity: %g cpus - %g reserved, %g GB - %g reserved, gpus %s" % (
        cap["cpus"], cap.get("reserve_cpus", 0), cap["mem_gb"], cap.get("reserve_mem_gb", 0),
        [g["index"] for g in cap.get("gpus") or []]))
    for e in info.get("envs") or []:
        say("[rx] env %s: %s (%s)" % (e["name"], e["dir"], (e.get("python") or "?").split()[0]))
    if remote.windows():
        try:
            w = remote.call({"op": "wsl"}, retry_for=10)
            for d in w.get("distros") or []:
                say("[rx] wsl %s: %s, WSL%s%s" % (d["name"], d["state"], d["version"], " (default)" if d["default"] else ""))
        except (TransportError, RemoteError):
            pass
    if args.speed:
        n = int(args.speed * 2 ** 20)
        t0 = time.time()
        got = remote.call({"op": "speedtest", "direction": "down", "bytes": n},
                          recv=lambda resp, fp: len(fp.read(resp["bytes"])), retry_for=0)
        down = got / (time.time() - t0)
        blob = os.urandom(1 << 20)

        def send(fp):
            left = n
            while left > 0:
                fp.write(blob[:min(left, len(blob))])
                left -= len(blob)
        t0 = time.time()
        remote.call({"op": "speedtest", "direction": "up", "bytes": n}, send=send, retry_for=0)
        up = n / (time.time() - t0)
        say("[rx] throughput (one stream, incompressible, including session setup): down %s/s, up %s/s" % (
            fmt_bytes(down), fmt_bytes(up)))
    return 0 if ok else 1


def cmd_push(args):
    proj, remote = context(args)
    push(proj, remote, extra=args.inputs, ws=args.ws or proj.run_cfg.get("ws") or "ws", dry=args.dry_run,
         prune=args.prune, streams=args.streams, force=args.force)
    return 0


def run_defaults(proj, profile):
    d = dict(proj.run_cfg)
    if profile:
        if profile not in proj.profiles:
            die("no [profiles.%s] in rx.toml" % profile)
        d.update(proj.profiles[profile])
    return d


def parse_vars(items):
    out = {}
    for x in items or []:
        if "=" not in x:
            die("--var takes KEY=VALUE, got %r" % x)
        k, v = x.split("=", 1)
        out[k] = v
    return out


def build_spec(proj, args, argv):
    d = run_defaults(proj, args.profile)
    shell = args.shell or d.get("shell") or "exec"
    gpus = args.gpus if args.gpus is not None else d.get("gpus", 0)
    spec = {
        "project": proj.name, "id": make_id(args.name or d.get("name") or derive_name(argv)),
        "name": args.name or derive_name(argv), "ws": args.ws or d.get("ws") or "ws", "shell": shell,
        "argv": argv if shell == "exec" else None,
        "command": None if shell == "exec" else " ".join(argv),
        "env": args.env if args.env is not None else d.get("env"),
        "cpus": float(args.cpus if args.cpus is not None else d.get("cpus", 1)),
        "mem_gb": float(args.mem if args.mem is not None else d.get("mem_gb", 2)),
        "gpus": float(gpus or 0),
        "mem_hard_gb": args.mem_hard if args.mem_hard is not None else d.get("mem_hard_gb"),
        "outputs": args.outputs or d.get("outputs") or ["**"],
        "outputs_exclude": list(d.get("outputs_exclude") or []),
        "env_vars": dict(d.get("env_vars") or {}, **parse_vars(args.var)),
        "cwd": args.cwd or d.get("cwd") or ".", "timeout_s": args.timeout or d.get("timeout_s"),
        "wsl_distro": d.get("wsl_distro"),
        "display": " ".join(argv), "code": git_info(proj.root), "submitted_from": socket.gethostname(),
    }
    if spec["env"] == "":
        spec["env"] = None
    return spec


def launch(proj, remote, spec, quiet=False):
    r = remote.call({"op": "launch", "spec": spec}, retry_for=600)
    j = r["job"]
    proj.record_job(spec["id"], host=remote.name, created=spec.get("created") or time.time(),
                    display=spec.get("display"), fetch=spec.get("_fetch", False), state=j.get("state"))
    if not quiet:
        g = "%g gpu" % spec["gpus"] if spec.get("gpus") else "no gpu"
        say("[rx] %s %s on %s (%g cpus, %g GB, %s%s)" % (
            spec["id"], "already exists" if r.get("existing") else "launched", remote.name, spec["cpus"],
            spec["mem_gb"], g, ", env %s" % spec["env"] if spec.get("env") else ""))
    return j


def cmd_run(args):
    proj, remote = context(args)
    argv = list(args.cmd)
    if argv and argv[0] == "--":
        argv = argv[1:]
    if not argv:
        die("nothing to run: rx run [options] -- COMMAND ...")
    spec = build_spec(proj, args, argv)
    if not args.no_push:
        s = push(proj, remote, extra=args.inputs, ws=spec["ws"])
        spec["code"] = dict(spec.get("code") or {}, pushed_digest=s["digest"], pushed_files=s["files"])
    spec["_fetch"] = bool(args.fetch)
    spec["created"] = time.time()
    launch(proj, remote, spec)
    jid = spec["id"]
    print(jid)
    if args.follow:
        end = follow_logs(proj, remote, jid)
        say("[rx] %s %s rc=%s" % (jid, end.get("state"), end.get("rc")))
        if args.fetch:
            wait_jobs(proj, remote, [jid], fetch=True, quiet=True)
            fetch_job(proj, remote, jid)
        return 0 if end.get("state") == "done" else 1
    if args.wait:
        fin = wait_jobs(proj, remote, [jid], fetch=args.fetch)
        return 0 if fin.get(jid, {}).get("state") == "done" else 1
    say("     rx logs %s -f    rx wait %s --fetch" % (jid, jid))
    return 0


def cmd_ls(args):
    proj, remote = context(args, need_project=not args.all)
    if args.all:
        r = remote.call({"op": "overview", "limit": args.n}, retry_for=30)
        for p, jobs in r["projects"].items():
            print("== %s" % p)
            print_jobs(jobs, r["now"])
        return 0
    r = remote.call({"op": "status", "project": proj.name, "limit": args.n, "with_line": True}, retry_for=30)
    print_jobs(r["jobs"], r["now"])
    return 0


def cmd_status(args):
    proj, remote = context(args)
    jid = resolve_job(proj, remote, args.job)
    r = remote.call({"op": "status", "project": proj.name, "ids": [jid], "with_line": True}, retry_for=30)
    if not r["jobs"]:
        die("no job %s on %s" % (jid, remote.name))
    j = r["jobs"][0]
    if args.json:
        print(json.dumps(j, indent=1, sort_keys=True))
        return 0
    print_jobs([j], r["now"])
    for k in ("command_line", "waiting", "error", "note", "outputs", "outputs_bytes", "peak_mem_gb", "cpu_s",
              "leftover_procs", "host", "os_priority"):
        if j.get(k) not in (None, "", 0) or k in ("outputs",) and j.get(k) == 0 and j.get("state") in TERMINAL:
            print("  %-15s %s" % (k, j.get(k)))
    return 0


def cmd_logs(args):
    proj, remote = context(args)
    jid = resolve_job(proj, remote, args.job)
    end = follow_logs(proj, remote, jid, follow=args.follow, tail_kb=args.tail)
    if args.follow:
        say("[rx] %s %s%s" % (jid, end.get("state"), "" if end.get("rc") is None else " rc=%s" % end.get("rc")))
    return 0


def cmd_wait(args):
    proj, remote = context(args)
    ids = [resolve_job(proj, remote, j) for j in args.jobs]
    fin = wait_jobs(proj, remote, ids, interval=args.interval, fetch=args.fetch, into=args.into)
    return 0 if all(j.get("state") == "done" for j in fin.values()) else 1


def cmd_fetch(args):
    proj, remote = context(args)
    if args.glob:
        r = remote.call({"op": "manifest", "project": proj.name, "ws": args.ws or "ws", "include": args.glob,
                         "exclude": args.exclude or []}, retry_for=120)
        say("[rx] %d file(s), %s match on %s" % (len(r["files"]), fmt_bytes(r["bytes"]), remote.name))
        fetched, have, conflicts = fetch_files(proj, remote, r["files"], ws=args.ws or "ws", into=args.into,
                                               overwrite=args.overwrite)
        say("[rx] fetched %d, already here %d, conflicts %d" % (fetched, have, len(conflicts)))
        return 0 if not conflicts else 2
    jid = resolve_job(proj, remote, args.job)
    _, _, conflicts = fetch_job(proj, remote, jid, into=args.into, overwrite=args.overwrite,
                                include=args.include, exclude=args.exclude)
    return 0 if not conflicts else 2


def cmd_cancel(args):
    proj, remote = context(args)
    for ref in args.jobs:
        jid = resolve_job(proj, remote, ref)
        r = remote.call({"op": "cancel", "project": proj.name, "id": jid}, retry_for=60)
        say("[rx] %s: %s%s" % (jid, r["job"].get("state"), " (%s)" % r["note"] if r.get("note") else
                               ", cancel requested" if r["job"].get("state") in ACTIVE else ""))
    return 0


def cmd_rerun(args):
    proj, remote = context(args)
    jid = resolve_job(proj, remote, args.job)
    old = remote.call({"op": "spec", "project": proj.name, "id": jid}, retry_for=60)["spec"]
    if not old:
        die("no spec recorded for %s" % jid)
    spec = {k: v for k, v in old.items() if k not in ("created", "agent")}
    spec["id"] = make_id(old.get("name") or "rerun")
    spec["rerun_of"] = jid
    if not args.no_push and not old.get("internal"):
        s = push(proj, remote, ws=spec.get("ws") or "ws")
        spec["code"] = dict(git_info(proj.root) or {}, pushed_digest=s["digest"], pushed_files=s["files"])
    spec["created"] = time.time()
    launch(proj, remote, spec)
    print(spec["id"])
    if args.follow:
        follow_logs(proj, remote, spec["id"])
    return 0


def cmd_gc(args):
    proj, remote = context(args, need_project=False)
    r = remote.call({"op": "gc", "project": proj.name if proj and not args.all else None, "days": args.days,
                     "keep": args.keep, "dry_run": args.dry_run}, retry_for=60)
    verb = "would remove" if args.dry_run else "removed"
    say("[rx] %s %d job dir(s), %d stale partial file(s), %d old env build(s)" % (
        verb, len(r["jobs"]), len(r["parts"]), len(r["envs"])))
    for x in r["jobs"][:20]:
        say("    %s" % x)
    return 0


def cmd_env(args):
    proj, remote = context(args, need_project=args.env_cmd == "build")
    if args.env_cmd == "ls":
        info = remote.call({"op": "sysinfo"}, retry_for=30)
        for e in info.get("envs") or []:
            print("%-16s %-32s built %s  (%s s)  %s" % (
                e["name"], e["dir"], time.strftime("%Y-%m-%d %H:%M", time.localtime(e.get("built") or 0)),
                e.get("build_s"), (e.get("python") or "").split(" ")[0]))
        if proj:
            for n in proj.envs:
                spec = env_spec(proj, n, remote.cfg)
                mark = next((e for e in info.get("envs") or [] if e["name"] == n), None)
                state = "missing" if not mark else ("current" if mark.get("hash") == spec["hash"] else "STALE (rx.toml changed)")
                print("rx.toml env %-16s %s" % (n, state))
        return 0
    spec = env_spec(proj, args.name, remote.cfg)
    gpus = float(proj.envs[args.name].get("gpus") or 0)      # reserved so `verify` can use a GPU
    job = {"project": proj.name, "id": make_id("env-" + args.name), "name": "env-" + args.name,
           "internal": "envbuild", "env_spec": spec, "cpus": float(args.cpus), "mem_gb": 4.0, "gpus": gpus,
           "outputs": [".rx-none"], "display": "env build %s (%s)" % (args.name, spec["hash"][:12]),
           "created": time.time()}
    launch(proj, remote, job)
    print(job["id"])
    if args.follow:
        end = follow_logs(proj, remote, job["id"])
        say("[rx] %s %s rc=%s" % (job["id"], end.get("state"), end.get("rc")))
        return 0 if end.get("state") == "done" else 1
    return 0


def cmd_exec(args):
    root = find_project(args.chdir or os.getcwd())
    proj = Project(root) if root else None
    host = args.host or (proj.host if proj else None)
    if not host:
        die("no host: pass --host NAME")
    remote = Remote(load_host(host))
    argv = list(args.cmd)
    if argv and argv[0] == "--":
        argv = argv[1:]
    if not argv:
        die("nothing to run")
    shell = args.shell or "exec"
    req = {"op": "exec", "project": proj.name if (proj and not args.no_ws) else None, "shell": shell,
           "argv": argv if shell == "exec" else None, "command": " ".join(argv) if shell != "exec" else None,
           "env": args.env, "timeout": args.timeout, "gpus": [int(x) for x in args.gpus.split(",")] if args.gpus else None}
    out = sys.stdout.buffer

    def recv(resp, fp):
        while True:
            line = fp.readline()
            if not line:
                raise EOFError("output ended")
            fr = json.loads(line.decode("utf-8"))
            if fr["t"] == "d":
                out.write(fp.read(fr["n"]))
                out.flush()
            elif fr["t"] == "end":
                return fr
    end = remote.call(req, recv=recv, retry_for=30)
    return int(end.get("rc") or 0)


def parse_priority(items):
    """["mpr=10", "crag=0,jigsaw="] -> {"mpr": 10.0, "crag": None, "jigsaw": None}; 0 or empty resets to the default."""
    out = {}
    for item in items:
        for part in item.split(","):
            if not part.strip():
                continue
            name, eq, val = part.partition("=")
            name = name.strip()
            if not eq or not A.NAME_RE.match(name):
                die("--priority wants PROJECT=NUMBER, got %r" % part)
            try:
                num = float(val) if val.strip() else 0.0
            except ValueError:
                die("--priority wants PROJECT=NUMBER, got %r" % part)
            out[name] = None if num == 0 else num
    return out


def cmd_capacity(args):
    proj, remote = context(args, need_project=False)
    req = {"op": "capacity", "cpus": args.cpus, "mem_gb": args.mem, "reserve_cpus": args.reserve_cpus,
           "reserve_mem_gb": args.reserve_mem, "hold_after_s": args.hold, "reset": args.reset}
    if args.priority:
        req["priority"] = parse_priority(args.priority)
    cap = remote.call(req, retry_for=30)["capacity"]
    print(json.dumps(cap, indent=1, sort_keys=True))
    return 0


# ─────────────────────────────── monitor / watch / dashboard ───────────────────────────────

def poll_hosts(hosts, limit=12):
    out = {}
    for h in hosts:
        try:
            out[h["name"]] = {"ok": True, "t": time.time(),
                              "data": Remote(h).call({"op": "overview", "limit": limit}, retry_for=0, quiet=True)}
        except (TransportError, RemoteError, Die) as e:
            out[h["name"]] = {"ok": False, "t": time.time(), "error": str(e).strip()[:300]}
    return out


def bar(frac, width=16):
    frac = max(0.0, min(1.0, frac or 0.0))
    n = int(round(frac * width))
    return "[" + "#" * n + "." * (width - n) + "]"


def render_monitor(state, last_ok):
    lines = ["rx monitor  %s   (Ctrl-C quits)" % time.strftime("%Y-%m-%d %H:%M:%S")]
    for name, s in state.items():
        if not s["ok"]:
            since = last_ok.get(name)
            lines.append("")
            lines.append("%s  UNREACHABLE (%s)%s -- jobs there keep running" % (
                name, s["error"][:90], "; last seen %s ago" % fmt_dur(time.time() - since) if since else ""))
            continue
        d = s["data"]
        si, cap = d["sysinfo"], d["capacity"]
        up = fmt_dur(si["now"] - (si.get("boot_time") or si["now"]))
        mem_used = (si.get("mem_total_gb") or 0) - (si.get("mem_avail_gb") or 0)
        lines.append("")
        lines.append("%s  %s  up %s   cpu %s %5.1f%% of %s   mem %s %.1f/%.1f GB   disk %s GB free" % (
            name, si["host"], up, bar((si.get("cpu_pct") or 0) / 100), si.get("cpu_pct") or 0, si.get("cpus"),
            bar(mem_used / max(si.get("mem_total_gb") or 1, 1)), mem_used, si.get("mem_total_gb") or 0,
            si.get("disk_free_gb")))
        for g in si.get("gpus") or []:
            lines.append("  gpu%d %-28s util %s %3.0f%%   mem %s %.1f/%.1f GB   %s C  %s W" % (
                g["index"], g["name"][:28], bar((g.get("util_pct") or 0) / 100, 10), g.get("util_pct") or 0,
                bar((g.get("mem_used_gb") or 0) / max(g.get("mem_total_gb") or 1, 1), 10),
                g.get("mem_used_gb") or 0, g.get("mem_total_gb") or 0, g.get("temp_c"), g.get("power_w")))
        used_c = sum(a["req"]["cpus"] for a in d["active"])
        used_m = sum(a["req"]["mem_gb"] for a in d["active"])
        used_g = sum(float(v) for a in d["active"] for v in ((a.get("assign") or {}).get("gpu_shares") or {}).values())
        lines.append("  scheduler: %g/%g cpus, %g/%g GB, %g/%d gpu reserved; %d running, %d queued" % (
            used_c, cap["cpus"] - cap.get("reserve_cpus", 0), used_m, cap["mem_gb"] - cap.get("reserve_mem_gb", 0),
            used_g, len(cap.get("gpus") or []), len(d["active"]), len(d["queue"])))
        if cap.get("priority"):
            lines.append("  priority: %s; every other project 0; higher goes first" % ", ".join(
                "%s %g" % kv for kv in sorted(cap["priority"].items(), key=lambda kv: (-float(kv[1]), kv[0]))))
        rows = [(p, j) for p, jobs in d["projects"].items() for j in jobs]
        rows.sort(key=lambda pj: (pj[1].get("state") not in ACTIVE, -(pj[1].get("created") or 0)))
        for p, j in rows[:18]:
            start = j.get("started") or j.get("created") or d["now"]
            dur = (j.get("ended") or d["now"]) - start
            live = ""
            if j.get("state") in ("running", "finishing"):
                live = "cpu %ss mem %sG" % (j.get("live_cpu_s", "-"), j.get("live_mem_gb", "-"))
            lines.append("  %-8s %-34s %-10s %8s  %-16s %-18s %s" % (
                p[:8], j["id"][:34], j["state"], fmt_dur(dur), res_str(j)[:16], live[:18],
                (j.get("last_line") or j.get("waiting") or j.get("error") or j.get("lost_reason") or
                 (j.get("display") or ""))[:70]))
    return "\n".join(lines)


def cmd_monitor(args):
    hosts = [load_host(h) for h in args.hosts] if args.hosts else all_hosts()
    if not hosts:
        die("no hosts are set up (rx setup HOST)")
    if os.name == "nt":
        os.system("")                          # turn on ANSI escapes in the Windows console
    last_ok = {}
    try:
        while True:
            state = poll_hosts(hosts)
            for n, s in state.items():
                if s["ok"]:
                    last_ok[n] = s["t"]
            text = render_monitor(state, last_ok)
            if args.once:
                print(text)
                return 0
            sys.stdout.write("\x1b[H\x1b[2J" + text + "\n")
            sys.stdout.flush()
            time.sleep(args.interval)
    except KeyboardInterrupt:
        return 0


def cmd_watch(args):
    """Fetch the outputs of jobs launched with --fetch as soon as they end, whenever
    the host is reachable. Safe to stop and start; state lives in each project's .rx/."""
    say("[rx] watching %d project(s); Ctrl-C stops" % len(registered_projects()))
    try:
        while True:
            for proj in registered_projects():
                pend = [j for j in proj.jobs().values() if j.get("fetch") and not j.get("fetched")]
                by_host = {}
                for j in pend:
                    by_host.setdefault(j.get("host") or proj.host, []).append(j["id"])
                for host, ids in by_host.items():
                    try:
                        remote = Remote(load_host(host))
                        r = remote.call({"op": "status", "project": proj.name, "ids": ids}, retry_for=0, quiet=True)
                    except (TransportError, RemoteError, Die):
                        continue
                    for j in r["jobs"]:
                        if j["state"] in TERMINAL:
                            proj.record_job(j["id"], state=j["state"], rc=j.get("rc"))
                            if j["state"] == "lost":
                                proj.record_job(j["id"], fetched=-1)
                                say("[rx] %s was lost (%s); not fetching" % (j["id"], j.get("lost_reason")))
                                continue
                            try:
                                fetch_job(proj, remote, j["id"], retry_for=120)
                            except (TransportError, Die) as e:
                                say("[rx] fetch %s postponed: %s" % (j["id"], e))
            if args.once:
                return 0
            time.sleep(args.interval)
    except KeyboardInterrupt:
        return 0


DASH_HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>rx Job Board</title>
<style>
:root{--bg:#f3f5f7;--panel:#ffffff;--ink:#17202b;--muted:#5d6b7a;--line:#dde3ea;--accent:#0e7490;
--run:#1d4ed8;--queue:#a16207;--done:#15803d;--fail:#b91c1c;--gone:#7e22ce;--idle:#64748b;--track:#e7ecf1;
--mono:ui-monospace,"Cascadia Mono",Consolas,"SF Mono",Menlo,monospace;--sans:"Segoe UI Variable","Segoe UI",system-ui,-apple-system,sans-serif}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#0d1319;--panel:#141c24;--ink:#e3e9ef;--muted:#90a0b1;
--line:#243140;--accent:#22b8cf;--run:#60a5fa;--queue:#fbbf24;--done:#4ade80;--fail:#f87171;--gone:#c084fc;--idle:#94a3b8;--track:#1f2a36}}
:root[data-theme="dark"]{--bg:#0d1319;--panel:#141c24;--ink:#e3e9ef;--muted:#90a0b1;--line:#243140;--accent:#22b8cf;
--run:#60a5fa;--queue:#fbbf24;--done:#4ade80;--fail:#f87171;--gone:#c084fc;--idle:#94a3b8;--track:#1f2a36}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 var(--sans)}
header{display:flex;align-items:baseline;gap:16px;padding:14px 20px;border-bottom:1px solid var(--line);background:var(--panel)}
header h1{font-size:17px;margin:0;letter-spacing:.02em}header .meta{color:var(--muted);font-size:12.5px}
main{padding:16px 20px;display:grid;gap:16px;max-width:1500px}
.host{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:14px 16px}
.host h2{font-size:15px;margin:0 0 10px;display:flex;gap:10px;align-items:baseline;flex-wrap:wrap}
.host h2 small{color:var(--muted);font-weight:400;font-size:12.5px}
.meters{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:10px 18px;margin-bottom:12px}
.m{font-size:12.5px;color:var(--muted)}.m b{color:var(--ink);font-variant-numeric:tabular-nums;font-weight:600}
.track{height:6px;background:var(--track);border-radius:3px;margin-top:4px;overflow:hidden}.fill{height:100%;background:var(--accent)}
.down{color:var(--fail);font-weight:600}
.tbl{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:13px}
th{text-align:left;color:var(--muted);font-weight:500;font-size:11.5px;text-transform:uppercase;letter-spacing:.05em;padding:6px 8px;border-bottom:1px solid var(--line)}
td{padding:7px 8px;border-bottom:1px solid var(--line);vertical-align:top;font-variant-numeric:tabular-nums}
tr.job{cursor:pointer}tr.job:hover td{background:color-mix(in srgb,var(--accent) 6%,transparent)}tr.sel td{background:color-mix(in srgb,var(--accent) 12%,transparent)}
td.id{font-family:var(--mono);font-size:12px;white-space:nowrap}td.line{font-family:var(--mono);font-size:12px;color:var(--muted);max-width:520px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.pill{display:inline-block;padding:1px 8px;border-radius:999px;font-size:11.5px;font-weight:600;border:1px solid currentColor}
.s-running,.s-finishing{color:var(--run)}.s-queued,.s-launching{color:var(--queue)}.s-done{color:var(--done)}.s-failed,.s-error{color:var(--fail)}.s-lost{color:var(--gone)}.s-cancelled{color:var(--idle)}
#log{background:var(--panel);border:1px solid var(--line);border-radius:8px;display:none}
#log .bar{display:flex;gap:12px;align-items:center;padding:10px 14px;border-bottom:1px solid var(--line);flex-wrap:wrap}
#log .bar code{font-family:var(--mono);font-size:12.5px}#log pre{margin:0;padding:12px 14px;max-height:55vh;overflow:auto;font:12px/1.45 var(--mono);white-space:pre-wrap;word-break:break-word}
button{font:inherit;font-size:12.5px;padding:4px 12px;border-radius:6px;border:1px solid var(--line);background:var(--panel);color:var(--ink);cursor:pointer}
button:hover{border-color:var(--accent)}button.danger{color:var(--fail)}button:focus-visible,tr.job:focus-visible{outline:2px solid var(--accent);outline-offset:1px}
.empty{color:var(--muted);font-size:13px;padding:8px 0}
</style></head><body>
<header><h1>rx job board</h1><span class="meta" id="meta">connecting...</span></header>
<main><div id="hosts"></div>
<section id="log"><div class="bar"><strong>Output</strong><code id="logid"></code><span class="meta" id="logmeta"></span>
<span style="flex:1"></span><button id="cancel" class="danger">Cancel job</button><button id="close">Close</button></div><pre id="logtext"></pre></section></main>
<script>
const TOKEN="__TOKEN__";let sel=null,stick=true;
const $=id=>document.getElementById(id);
const esc=s=>String(s??"").replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
function dur(s){if(s==null||s<0)return"-";s=Math.floor(s);if(s<60)return s+"s";if(s<3600)return Math.floor(s/60)+"m"+String(s%60).padStart(2,"0")+"s";
if(s<86400)return Math.floor(s/3600)+"h"+String(Math.floor(s%3600/60)).padStart(2,"0")+"m";return Math.floor(s/86400)+"d"+String(Math.floor(s%86400/3600)).padStart(2,"0")+"h"}
function meter(label,val,frac){return `<div class="m">${label} <b>${val}</b><div class="track"><div class="fill" style="width:${Math.max(0,Math.min(100,frac*100)).toFixed(1)}%"></div></div></div>`}
const ACTIVE=["launching","queued","running","finishing"];
function render(st){const out=[];let n=0;
for(const [name,h] of Object.entries(st.hosts)){
 if(!h.ok){out.push(`<div class="host"><h2>${esc(name)} <span class="down">unreachable</span><small>${esc(h.error)}</small></h2><div class="empty">Jobs on this host keep running; they will show again when it is reachable.</div></div>`);continue}
 const d=h.data,si=d.sysinfo,cap=d.capacity;const used=(si.mem_total_gb||0)-(si.mem_avail_gb||0);
 let m=meter("CPU",`${si.cpu_pct??"-"}% of ${si.cpus}`,(si.cpu_pct||0)/100)+meter("Memory",`${used.toFixed(1)} / ${si.mem_total_gb} GB`,used/(si.mem_total_gb||1));
 for(const g of si.gpus||[]){m+=meter(`GPU ${g.index}`,`${g.util_pct??"-"}% &middot; ${g.temp_c??"-"} C &middot; ${g.power_w??"-"} W`,(g.util_pct||0)/100)+meter(`GPU ${g.index} memory`,`${g.mem_used_gb} / ${g.mem_total_gb} GB`,g.mem_used_gb/(g.mem_total_gb||1))}
 const rc=d.active.reduce((a,x)=>a+x.req.cpus,0),rm=d.active.reduce((a,x)=>a+x.req.mem_gb,0);
 m+=meter("Reserved by jobs",`${rc}/${cap.cpus-(cap.reserve_cpus||0)} cpus &middot; ${rm}/${cap.mem_gb-(cap.reserve_mem_gb||0)} GB`,rc/Math.max(1,cap.cpus-(cap.reserve_cpus||0)));
 const rows=[];for(const [p,jobs] of Object.entries(d.projects))for(const j of jobs)rows.push([p,j]);
 rows.sort((a,b)=>(ACTIVE.includes(b[1].state)-ACTIVE.includes(a[1].state))||((b[1].created||0)-(a[1].created||0)));
 let t=`<div class="tbl"><table><thead><tr><th>Project</th><th>Job</th><th>State</th><th>Runtime</th><th>Resources</th><th>Now</th></tr></thead><tbody>`;
 for(const [p,j] of rows.slice(0,40)){n++;const start=j.started||j.created||d.now,du=(j.ended||d.now)-start;
  const r=j.req||{};const res=`${r.cpus} cpu &middot; ${r.mem_gb} GB${r.gpus?` &middot; ${r.gpus} gpu`:""}`;
  const line=j.last_line||j.waiting||j.error||j.lost_reason||j.display||"";const isSel=sel&&sel.host==name&&sel.project==p&&sel.id==j.id;
  t+=`<tr class="job${isSel?" sel":""}" tabindex="0" data-h="${esc(name)}" data-p="${esc(p)}" data-j="${esc(j.id)}" data-s="${esc(j.state)}"><td>${esc(p)}</td><td class="id">${esc(j.id)}</td><td><span class="pill s-${esc(j.state)}">${esc(j.state)}${j.rc!=null?" "+esc(j.rc):""}</span></td><td>${dur(du)}</td><td>${res}</td><td class="line" title="${esc(line)}">${esc(line)}</td></tr>`}
 t+=`</tbody></table></div>`;if(!rows.length)t=`<div class="empty">No jobs yet. Launch one with rx run -- COMMAND.</div>`;
 out.push(`<div class="host"><h2>${esc(name)}<small>${esc(si.host)} &middot; ${esc(si.platform)} &middot; up ${dur(si.now-(si.boot_time||si.now))} &middot; ${si.disk_free_gb} GB free</small></h2><div class="meters">${m}</div>${t}</div>`)}
 $("hosts").innerHTML=out.join("");$("meta").textContent=`${Object.keys(st.hosts).length} host(s) · ${n} job(s) · updated ${new Date(st.t*1000).toLocaleTimeString()}`;
 document.querySelectorAll("tr.job").forEach(tr=>{const go=()=>pick(tr.dataset.h,tr.dataset.p,tr.dataset.j,tr.dataset.s);tr.onclick=go;tr.onkeydown=e=>{if(e.key=="Enter")go()}})}
async function refresh(){try{const r=await fetch("/api/state");render(await r.json())}catch(e){$("meta").textContent="dashboard server unreachable"}}
function pick(h,p,j,s){sel={host:h,project:p,id:j,state:s};stick=true;$("log").style.display="block";$("logid").textContent=j;$("logtext").textContent="loading...";loadLog();refresh()}
async function loadLog(){if(!sel)return;try{const q=new URLSearchParams({host:sel.host,project:sel.project,id:sel.id,kb:"256"});
 const r=await fetch("/api/log?"+q);const t=await r.text();const pre=$("logtext");stick=pre.scrollTop+pre.clientHeight>=pre.scrollHeight-40;
 pre.textContent=t||"(no output yet)";if(stick)pre.scrollTop=pre.scrollHeight;$("logmeta").textContent="last 256 KB, refreshed "+new Date().toLocaleTimeString()}catch(e){}}
$("close").onclick=()=>{sel=null;$("log").style.display="none";refresh()};
$("cancel").onclick=async()=>{if(!sel||!confirm("Cancel "+sel.id+"?"))return;const r=await fetch("/api/cancel",{method:"POST",headers:{"X-RX-Token":TOKEN,"Content-Type":"application/json"},body:JSON.stringify(sel)});
 $("logmeta").textContent=r.ok?"cancel requested":"cancel failed: "+await r.text();refresh()};
refresh();setInterval(refresh,5000);setInterval(loadLog,4000);
</script></body></html>"""


def cmd_dashboard(args):
    hosts = [load_host(h) for h in args.hosts] if args.hosts else all_hosts()
    if not hosts:
        die("no hosts are set up (rx setup HOST)")
    token = secrets.token_urlsafe(18)
    shared = {"t": time.time(), "hosts": {h["name"]: {"ok": False, "error": "polling..."} for h in hosts}}
    lock = threading.Lock()
    by_name = {h["name"]: h for h in hosts}

    def poller():
        while True:
            s = poll_hosts(hosts, limit=20)
            with lock:
                for n, v in s.items():
                    shared["hosts"][n] = v
                shared["t"] = time.time()
            time.sleep(args.interval)

    threading.Thread(target=poller, daemon=True).start()
    port = args.port
    page = DASH_HTML.replace("__TOKEN__", token).encode("utf-8")

    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _host_ok(self):
            return self.headers.get("Host") in ("127.0.0.1:%d" % port, "localhost:%d" % port)

        def _send(self, code, body, ctype="application/json"):
            self.send_response(code)
            self.send_header("Content-Type", ctype + "; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if not self._host_ok():
                return self._send(403, b"bad host")
            u = urllib.parse.urlparse(self.path)
            if u.path == "/":
                return self._send(200, page, "text/html")
            if u.path == "/api/state":
                with lock:
                    body = json.dumps(shared).encode()
                return self._send(200, body)
            if u.path == "/api/log":
                q = urllib.parse.parse_qs(u.query)
                try:
                    h = by_name[q["host"][0]]
                    buf = []

                    def recv(resp, fp):
                        while True:
                            line = fp.readline()
                            if not line:
                                raise EOFError
                            fr = json.loads(line)
                            if fr["t"] == "d":
                                buf.append(fp.read(fr["n"]))
                            elif fr["t"] == "end":
                                return fr
                    Remote(h).call({"op": "tail", "project": q["project"][0], "id": q["id"][0], "follow": False,
                                    "tail_bytes": int(q.get("kb", ["256"])[0]) * 1024}, recv=recv, retry_for=0, quiet=True)
                    return self._send(200, b"".join(buf), "text/plain")
                except Exception as e:  # noqa: BLE001
                    return self._send(502, str(e).encode(), "text/plain")
            return self._send(404, b"not found", "text/plain")

        def do_POST(self):
            if not self._host_ok() or self.headers.get("X-RX-Token") != token:
                return self._send(403, b"forbidden", "text/plain")
            if urllib.parse.urlparse(self.path).path != "/api/cancel":
                return self._send(404, b"not found", "text/plain")
            try:
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n))
                r = Remote(by_name[body["host"]]).call({"op": "cancel", "project": body["project"], "id": body["id"]},
                                                       retry_for=10, quiet=True)
                return self._send(200, json.dumps(r["job"]).encode())
            except Exception as e:  # noqa: BLE001
                return self._send(502, str(e).encode(), "text/plain")

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", port), H)
    say("[rx] dashboard on http://127.0.0.1:%d/ (Ctrl-C stops)" % port)
    if args.open:
        import webbrowser
        webbrowser.open("http://127.0.0.1:%d/" % port)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


# ─────────────────────────────── the parser ───────────────────────────────

def build_parser():
    ap = argparse.ArgumentParser(prog="rx", description="Run a project's work on a remote host over ssh.")
    ap.add_argument("-C", dest="chdir", help="act as if started in this directory")
    ap.add_argument("--host", help="host name (default: [project] host in rx.toml)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("setup", help="one-time setup of a host")
    s.add_argument("name")
    s.add_argument("--route", action="append", help="ssh destination and options, e.g. 'gpu' or "
                   "'-o HostName=10.1.12.23 gpu'; repeat for fallbacks, the first is preferred")
    s.add_argument("--python-version", default="3.13.5", help="standalone CPython to install on a Windows host")
    s.add_argument("--python", help="use this interpreter on the host instead")
    s.add_argument("--home", help="rx home on the host (default: ~/rx)")
    s.add_argument("--ssh", help="local ssh client to use")
    s.add_argument("--local", action="store_true", help="a pseudo-host on this machine (tests, local queues)")
    s.add_argument("--launch", help=argparse.SUPPRESS)
    s.set_defaults(fn=cmd_setup)

    s = sub.add_parser("hosts", help="list configured hosts")
    s.set_defaults(fn=cmd_hosts)

    s = sub.add_parser("doctor", help="check routes, agent, host and throughput")
    s.add_argument("--speed", type=float, metavar="MB", help="also measure throughput with MB megabytes")
    s.set_defaults(fn=cmd_doctor)

    s = sub.add_parser("push", help="sync the declared files to the host")
    s.add_argument("--dry-run", action="store_true")
    s.add_argument("--prune", action="store_true", help="delete host files rx pushed earlier that are no longer declared")
    s.add_argument("--inputs", action="append", metavar="GLOB", help="extra files to push (repeatable)")
    s.add_argument("--streams", type=int)
    s.add_argument("--ws", help="workspace name on the host (default ws)")
    s.add_argument("--force", action="store_true", help="ignore push.max_mb")
    s.set_defaults(fn=cmd_push)

    s = sub.add_parser("run", help="push, then run a command detached on the host")
    s.add_argument("-n", "--name")
    s.add_argument("-p", "--profile", help="a [profiles.NAME] table from rx.toml")
    s.add_argument("--env", help="environment built with rx env build ('' for the bare interpreter)")
    s.add_argument("--cpus", type=float)
    s.add_argument("--mem", type=float, help="GB to reserve")
    s.add_argument("--mem-hard", type=float, help="GB at which the job is stopped (Windows Job Object limit)")
    s.add_argument("--gpus", type=float, help="GPUs to reserve; a fraction shares one GPU")
    s.add_argument("--gpu", dest="gpus", action="store_const", const=1.0, help="same as --gpus 1")
    s.add_argument("--shell", choices=["exec", "cmd", "powershell", "bash", "wsl", "sh"])
    s.add_argument("--cwd", help="working directory inside the workspace")
    s.add_argument("--ws", help="workspace name on the host")
    s.add_argument("--inputs", action="append", metavar="GLOB", help="extra files to push for this job")
    s.add_argument("--outputs", action="append", metavar="GLOB", help="what counts as this job's outputs")
    s.add_argument("--var", action="append", metavar="KEY=VALUE", help="environment variable for the job")
    s.add_argument("--timeout", type=float, help="seconds before the job is stopped")
    s.add_argument("--no-push", action="store_true")
    s.add_argument("-f", "--follow", action="store_true", help="stream the output until the job ends")
    s.add_argument("-w", "--wait", action="store_true", help="wait until the job ends")
    s.add_argument("--fetch", action="store_true", help="fetch outputs when it ends (with -w/-f, or later by rx watch)")
    s.add_argument("cmd", nargs=argparse.REMAINDER)
    s.set_defaults(fn=cmd_run)

    s = sub.add_parser("ls", help="list jobs")
    s.add_argument("-n", type=int, default=20)
    s.add_argument("--all", action="store_true", help="every project on the host")
    s.set_defaults(fn=cmd_ls)

    s = sub.add_parser("status", help="one job in detail")
    s.add_argument("job", nargs="?")
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_status)

    s = sub.add_parser("logs", help="a job's output")
    s.add_argument("job", nargs="?")
    s.add_argument("-f", "--follow", action="store_true")
    s.add_argument("--tail", type=float, metavar="KB", help="start this many KB from the end")
    s.set_defaults(fn=cmd_logs)

    s = sub.add_parser("wait", help="wait for jobs to end")
    s.add_argument("jobs", nargs="+")
    s.add_argument("--fetch", action="store_true")
    s.add_argument("--into")
    s.add_argument("--interval", type=float, default=15.0)
    s.set_defaults(fn=cmd_wait)

    s = sub.add_parser("fetch", help="bring outputs back")
    s.add_argument("job", nargs="?")
    s.add_argument("--glob", action="append", metavar="GLOB", help="fetch workspace files by pattern instead")
    s.add_argument("--include", action="append", metavar="GLOB")
    s.add_argument("--exclude", action="append", metavar="GLOB")
    s.add_argument("--into", help="write under this directory instead of the project")
    s.add_argument("--overwrite", action="store_true", help="replace local files that differ")
    s.add_argument("--ws")
    s.set_defaults(fn=cmd_fetch)

    s = sub.add_parser("cancel", help="stop jobs")
    s.add_argument("jobs", nargs="+")
    s.set_defaults(fn=cmd_cancel)

    s = sub.add_parser("rerun", help="launch a job again with the same spec")
    s.add_argument("job")
    s.add_argument("--no-push", action="store_true")
    s.add_argument("-f", "--follow", action="store_true")
    s.set_defaults(fn=cmd_rerun)

    s = sub.add_parser("gc", help="remove old job directories and stale partial files")
    s.add_argument("--days", type=float, default=14)
    s.add_argument("--keep", type=int, default=30)
    s.add_argument("--all", action="store_true")
    s.add_argument("--dry-run", action="store_true")
    s.set_defaults(fn=cmd_gc)

    s = sub.add_parser("env", help="environments on the host")
    es = s.add_subparsers(dest="env_cmd", required=True)
    b = es.add_parser("build")
    b.add_argument("name")
    b.add_argument("--cpus", type=float, default=4)
    b.add_argument("-f", "--follow", action="store_true")
    es.add_parser("ls")
    s.set_defaults(fn=cmd_env)

    s = sub.add_parser("exec", help="run a short command now (tied to the connection)")
    s.add_argument("--env")
    s.add_argument("--shell", choices=["exec", "cmd", "powershell", "bash", "wsl", "sh"])
    s.add_argument("--gpus", help="comma list of GPU indices to expose (default: all)")
    s.add_argument("--no-ws", action="store_true", help="run in the rx home instead of the workspace")
    s.add_argument("--timeout", type=float, default=600)
    s.add_argument("cmd", nargs=argparse.REMAINDER)
    s.set_defaults(fn=cmd_exec)

    s = sub.add_parser("capacity", help="show or set what the scheduler shares out")
    s.add_argument("--cpus", type=float)
    s.add_argument("--mem", type=float)
    s.add_argument("--reserve-cpus", type=float)
    s.add_argument("--reserve-mem", type=float)
    s.add_argument("--hold", type=float, help="seconds before the queue head stops younger jobs overtaking")
    s.add_argument("--priority", action="append", metavar="PROJECT=N",
                   help="project priority, higher first (default 0; PROJECT=0 resets); repeatable or comma-separated")
    s.add_argument("--reset", action="store_true")
    s.set_defaults(fn=cmd_capacity)

    s = sub.add_parser("monitor", help="live view of every host, project and job")
    s.add_argument("--interval", type=float, default=5)
    s.add_argument("--hosts", action="append")
    s.add_argument("--once", action="store_true")
    s.set_defaults(fn=cmd_monitor)

    s = sub.add_parser("watch", help="auto-fetch jobs launched with --fetch")
    s.add_argument("--interval", type=float, default=30)
    s.add_argument("--once", action="store_true")
    s.set_defaults(fn=cmd_watch)

    s = sub.add_parser("dashboard", help="the job board in a browser (127.0.0.1)")
    s.add_argument("--port", type=int, default=8765)
    s.add_argument("--interval", type=float, default=5)
    s.add_argument("--hosts", action="append")
    s.add_argument("--open", action="store_true")
    s.set_defaults(fn=cmd_dashboard)
    return ap


def main(argv=None):
    try:
        sys.stdout.reconfigure(errors="replace")
        sys.stderr.reconfigure(errors="replace")
    except Exception:
        pass
    args = build_parser().parse_args(argv)
    try:
        return args.fn(args) or 0
    except Die as e:
        say("rx: %s" % e)
        return 1
    except RemoteError as e:
        say("rx: %s: %s" % (e.kind, e))
        return 1
    except TransportError as e:
        say("rx: cannot reach the host: %s" % str(e).strip()[:400])
        return 3
    except KeyboardInterrupt:
        say("rx: interrupted (anything already launched keeps running on the host)")
        return 130


if __name__ == "__main__":
    sys.exit(main())
