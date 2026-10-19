"""Watchdog for mpr's host controllers (systems code; no science).

    python outputs/host_ops/watchdog.py loop [--agent-python PY] [--every-min 10]   # host, as an rx job named 'watchdog'
    python outputs/host_ops/watchdog.py selftest                                  # anywhere: a fake rx, no launches

The user (20 Oct): "keep the queue filled for atleast 24hrs of downtime even in case of errors too". The queue lives in
two controller jobs, the feeder (hostfeed2) and the clean-up loop (auto-cleanup); when either ended, nothing ran until I
relaunched it by hand. Every --every-min the watchdog looks at mpr's jobs through rx's agent (as the feeder does) and:

1. relaunches a controller whose newest job ended without being cancelled (failed, error, done, or lost with a
   heartbeat older than --lost-grace), with the old job's spec and arguments (the feeder's one-off --wait-gone,
   --unsend and --undrop-prefix dropped): it carries on from its own state file, as after a hand relaunch. A cancel is
   someone's decision and is left alone. At most --max-relaunch relaunches of one controller per --window-h, so a
   controller that cannot start does not loop;
2. starts `disk_squeeze.py rescue` (lossless NTFS compression; every reader sees the same bytes) as an rx job when the
   drive's free space is below --squeeze-below-gb and no squeeze ran within --squeeze-every-h. It deletes nothing.

One watchdog at a time: a second one exits at start. Records: outputs/host_ops/watchdog_state.json (what it launched
and why) and its job log (a line per event, and an hourly line of the controllers' states and the free space).
A host reboot ends the watchdog too: rx does not restart jobs, so that still needs a hand relaunch.
Stdlib only; no credential and no network.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

PROJECT = "mpr"
CONTROLLERS = ("hostfeed2", "auto-cleanup")
ALIVE = ("running", "finishing", "launching", "queued", "pending")
ONE_OFF = ("--wait-gone", "--unsend", "--undrop-prefix")
SPEC_KEYS = ("env", "cpus", "mem_gb", "gpus", "shell", "cwd", "ws", "timeout_s", "mem_hard_gb", "outputs",
             "outputs_exclude", "env_vars", "wsl_distro", "_fetch")
SQUEEZE = "disk-squeeze"

a = None
HOME = AGENT = None
WS = Path.cwd()


def log(*x):
    print(time.strftime("%H:%M:%S"), *x, flush=True)


def agent(req, timeout=300):
    env = {k: v for k, v in os.environ.items()
           if k.upper() not in ("RX_LAUNCH", "PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "PYTHONSTARTUP")}
    env["RX_HOME"] = HOME
    p = subprocess.run([a.agent_python, AGENT, "rpc"], input=(json.dumps(req, separators=(",", ":")) + "\n").encode(),
                       capture_output=True, timeout=timeout, env=env, cwd=HOME)
    line = p.stdout.split(b"\n", 1)[0]
    try:
        r = json.loads(line.decode("utf-8"))
    except ValueError:
        raise RuntimeError(f"agent rc {p.returncode}: {p.stdout[-300:]!r} {p.stderr[-600:]!r}")
    if not r.get("ok"):
        raise RuntimeError(f"agent: {r.get('kind')}: {r.get('error')}")
    return r


def make_id(name):
    name = re.sub(r"[^A-Za-z0-9._-]+", "-", name or "job").strip("-._")[:40] or "job"
    return "%s-%s-%s" % (time.strftime("%y%m%d-%H%M%S"), name, secrets.token_hex(2))


def alive(j, now):
    s = j.get("state")
    return s in ALIVE or (s == "lost" and now - float(j.get("heartbeat") or 0) < a.lost_grace)


def newest(jobs, name):
    """mpr's newest job of this name (the agent lists newest first; 'created' decides when it is there)."""
    js = [j for j in jobs if j.get("name") == name]
    return max(js, key=lambda j: float(j.get("created") or 0)) if js and all(j.get("created") for j in js) else \
        (js[0] if js else None)


def spec_of(jid):
    return json.loads((Path(HOME) / "projects" / PROJECT / "jobs" / jid / "spec.json").read_text(encoding="utf-8"))


def clean_argv(argv):
    out, skip = [], False
    for x in argv:
        if skip:
            skip = False
            continue
        if x in ONE_OFF:
            skip = True
            continue
        out.append(x)
    return out


def new_spec(old, name, argv, why):
    s = {k: old.get(k) for k in SPEC_KEYS}
    s.update({"project": PROJECT, "id": make_id(name), "name": name, "argv": argv, "command": None,
              "display": " ".join(argv), "submitted_from": socket.gethostname(), "created": time.time(),
              "code": {"via": "watchdog", "watchdog_job": os.environ.get("RX_JOB_ID"), "why": why}})
    return s


def launch(spec):
    try:
        r = agent({"op": "launch", "spec": spec}, timeout=600)
        return (r.get("job") or {}).get("id") or spec["id"]
    except Exception as e:     # the job may exist anyway: look before deciding
        try:
            got = agent({"op": "status", "project": PROJECT, "ids": [spec["id"]]}, timeout=120)["jobs"]
        except Exception:
            got = None
        if got:
            return spec["id"]
        log(f"launch of {spec['name']} failed: {e!r}"[:400])
        return None


def budget_left(st, name, now):
    recent = [r for r in st["relaunched"].get(name, []) if now - r["t"] < a.window_h * 3600]
    return a.max_relaunch - len(recent)


def cycle(st, now, jobs, free_gb):
    """One look: the launches it made, as (name, new id, why)."""
    done = []
    for name in CONTROLLERS:
        j = newest(jobs, name)
        if j is None or alive(j, now):
            continue
        if j.get("state") == "cancelled":
            key = f"cancelled:{j['id']}"
            if key not in st["said"]:
                st["said"][key] = now
                log(f"{name}'s newest job {j['id']} was cancelled: left alone (a cancel is someone's decision)")
            continue
        why = f"{j['id']} ended {j.get('state')} rc {j.get('rc')}"
        if budget_left(st, name, now) <= 0:
            key = f"budget:{j['id']}"
            if key not in st["said"]:
                st["said"][key] = now
                log(f"{name}: {why}; {a.max_relaunch} relaunches within {a.window_h:g} h already: not again")
            continue
        try:
            old = spec_of(j["id"])
        except Exception as e:
            log(f"{name}: {why}; its spec cannot be read ({e!r}): not relaunched"[:300])
            continue
        argv = clean_argv(old.get("argv") or [])
        if not argv:
            continue
        jid = launch(new_spec(old, name, argv, why))
        if jid:
            st["relaunched"].setdefault(name, []).append({"t": now, "old": j["id"], "new": jid, "why": why})
            log(f"RELAUNCH {name}: {why} -> {jid}")
            done.append((name, jid, why))
    if free_gb is not None and free_gb < a.squeeze_below_gb:
        sq = [j for j in jobs if (j.get("name") or "").startswith(SQUEEZE)]
        last = max([float(r["t"]) for r in st["squeezes"]] + [0.0])
        if not any(alive(j, now) for j in sq) and now - last >= a.squeeze_every_h * 3600:
            argv = ["python", "outputs/host_ops/disk_squeeze.py", "rescue"]
            old = {"env": "mpr-cpu", "cpus": 1.0, "mem_gb": 1.0, "gpus": 0.0, "shell": "exec", "cwd": ".", "ws": "ws",
                   "timeout_s": None, "mem_hard_gb": None, "outputs": ["outputs/**", "logs/**"],
                   "outputs_exclude": [], "env_vars": {"PYTHONHASHSEED": "0"}, "wsl_distro": None, "_fetch": False}
            why = f"drive free {free_gb:.1f} GB < {a.squeeze_below_gb:g}"
            jid = launch(new_spec(old, SQUEEZE, argv, why))
            if jid:
                st["squeezes"].append({"t": now, "new": jid, "why": why})
                log(f"SQUEEZE: {why} -> {jid}")
                done.append((SQUEEZE, jid, why))
    return done


def save(st):
    p = Path(a.state)
    tmp = p.with_name(p.name + ".tmp")
    try:
        tmp.write_text(json.dumps(st, indent=1), encoding="utf-8")
        os.replace(tmp, p)
    except OSError as e:
        log(f"state not saved ({e!r}); kept in memory"[:200])


def load():
    try:
        st = json.loads(Path(a.state).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        st = {}
    for k in ("relaunched", "said"):
        st.setdefault(k, {})
    st.setdefault("squeezes", [])
    return st


def feeder_agent_python():
    """The --agent-python of mpr's newest feeder job on disk (job ids start with their launch time)."""
    for d in sorted((Path(HOME) / "projects" / PROJECT / "jobs").glob("*-hostfeed2-*"), reverse=True):
        try:
            argv = json.loads((d / "spec.json").read_text(encoding="utf-8")).get("argv") or []
        except (OSError, ValueError):
            continue
        if "--agent-python" in argv:
            return argv[argv.index("--agent-python") + 1]
    raise SystemExit("no feeder job names an --agent-python: pass it")


def loop():
    global HOME, AGENT
    HOME = os.environ.get("RX_HOME") or ""
    if not HOME:
        raise SystemExit("RX_HOME is not set: run this as an rx job on the host")
    a.agent_python = a.agent_python or feeder_agent_python()
    spec0 = json.loads((Path(os.environ["RX_JOB_DIR"]) / "spec.json").read_text(encoding="utf-8"))
    AGENT = a.agent or str(Path(HOME) / "agent" / f"rx_agent-{spec0['agent']}.py")
    me = os.environ.get("RX_JOB_ID") or spec0.get("id")
    jobs = agent({"op": "status", "project": PROJECT, "limit": 5000}, timeout=180)["jobs"]
    now = time.time()
    others = [j["id"] for j in jobs if j.get("name") == "watchdog" and j.get("id") != me and alive(j, now)]
    if others:
        log("another watchdog runs:", " ".join(others), "- exiting")
        return 0
    st = load()
    log(f"watchdog: every {a.every_min:g} min; controllers {', '.join(CONTROLLERS)}; at most {a.max_relaunch} "
        f"relaunches each per {a.window_h:g} h; squeeze below {a.squeeze_below_gb:g} GB free")
    beat = 0.0
    while True:
        try:
            now = time.time()
            jobs = agent({"op": "status", "project": PROJECT, "limit": 5000}, timeout=180)["jobs"]
            free = shutil.disk_usage(WS).free / 1e9
            if cycle(st, now, jobs, free):
                save(st)
            if now - beat >= 3600:
                beat = now
                bits = []
                for name in CONTROLLERS:
                    j = newest(jobs, name)
                    bits.append(f"{name} {j['id'] if j else '-'} {j.get('state') if j else 'none'}")
                log("ok:", "; ".join(bits), f"; drive free {free:.1f} GB")
                save(st)
        except Exception as e:     # one bad look never ends the watchdog
            log("cycle failed:", repr(e)[:300])
        time.sleep(a.every_min * 60)


def selftest():
    global a, HOME, AGENT, agent, spec_of
    import tempfile
    tmp = Path(tempfile.mkdtemp())
    a = argparse.Namespace(lost_grace=1800.0, max_relaunch=3, window_h=6.0, squeeze_below_gb=60.0, squeeze_every_h=2.0,
                           state=str(tmp / "st.json"), every_min=10.0, agent=None, agent_python=sys.executable)
    launched = []

    def fake_agent(req, timeout=300):
        if req["op"] == "launch":
            launched.append(req["spec"])
            return {"ok": True, "job": {"id": req["spec"]["id"]}}
        raise AssertionError(req)

    specs = {"f1": {"argv": ["python", "outputs/host_ops/hostfeed2.py", "items", "state", "--agent-python", "PY",
                             "--wait-gone", "f0", "--cap-mem", "140", "--unsend", "", "--undrop-prefix", ""],
                    "env": "mpr-cpu", "cpus": 0.2, "mem_gb": 0.5, "gpus": 0.0},
             "c1": {"argv": ["python", "outputs/host_ops/auto_cleanup.py", "run", "--delete", "--loop"],
                    "env": "mpr-cpu", "cpus": 0.2, "mem_gb": 0.5, "gpus": 0.0}}
    agent = fake_agent
    spec_of = lambda jid: specs[jid]  # noqa: E731
    now = 1_000_000.0
    st = load()
    run = [{"id": "f1", "name": "hostfeed2", "state": "running", "created": 5}, {"id": "c1", "name": "auto-cleanup",
                                                                                 "state": "running", "created": 4}]
    assert cycle(st, now, run, 80.0) == [] and not launched                       # all alive: nothing
    lost_young = [dict(run[0], state="lost", heartbeat=now - 60), run[1]]
    assert cycle(st, now, lost_young, 80.0) == []                                  # lost but beating: alive
    failed = [dict(run[0], state="failed", rc=1), run[1]]
    got = cycle(st, now, failed, 80.0)
    assert [g[0] for g in got] == ["hostfeed2"], got
    argv = launched[-1]["argv"]
    assert "--wait-gone" not in argv and "--unsend" not in argv and "--undrop-prefix" not in argv, argv
    assert argv[:6] == ["python", "outputs/host_ops/hostfeed2.py", "items", "state", "--agent-python", "PY"], argv
    assert argv[-2:] == ["--cap-mem", "140"] and launched[-1]["cpus"] == 0.2 and launched[-1]["name"] == "hostfeed2"
    canc = [dict(run[0], state="cancelled"), dict(run[1], state="done", rc=0)]
    got = cycle(st, now, canc, 80.0)
    assert [g[0] for g in got] == ["auto-cleanup"], got                           # a cancel is left alone
    assert launched[-1]["argv"] == specs["c1"]["argv"]
    n = len(launched)
    for k in range(3):                                                            # the budget: 3 per window in all
        cycle(st, now + k, [dict(run[0], id="f1", state="error"), run[1]], 80.0)
    assert len(launched) == n + 2, len(launched) - n                              # 1 + 2 more = 3, then none
    cycle(st, now + a.window_h * 3600 + 10, [dict(run[0], state="error"), run[1]], 80.0)
    assert len(launched) == n + 3                                                 # a new window: one more
    n = len(launched)
    got = cycle(st, now, run, 41.0)
    assert [g[0] for g in got] == [SQUEEZE] and launched[-1]["argv"][-1] == "rescue"
    assert cycle(st, now + 60, run, 41.0) == []                                   # not again within the interval
    assert cycle(st, now + 3 * 3600, run + [{"id": "q", "name": SQUEEZE, "state": "running"}], 41.0) == []
    assert len(cycle(st, now + 3 * 3600, run, 41.0)) == 1                         # after it: again
    assert cycle(st, now + 9 * 3600, run, 61.0) == []                             # enough room: none
    shutil.rmtree(tmp, ignore_errors=True)
    print("watchdog selftest OK:", len(launched), "fake launches")
    return 0


def main(argv=None):
    global a
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    lp = sub.add_parser("loop")
    lp.add_argument("--agent-python", default=None,
                    help="the python rx's agent runs under (default: the newest feeder job's --agent-python)")
    lp.add_argument("--agent", default=None)
    lp.add_argument("--every-min", type=float, default=10.0)
    lp.add_argument("--lost-grace", type=float, default=1800.0)
    lp.add_argument("--max-relaunch", type=int, default=3)
    lp.add_argument("--window-h", type=float, default=6.0)
    lp.add_argument("--squeeze-below-gb", type=float, default=60.0)
    lp.add_argument("--squeeze-every-h", type=float, default=2.0)
    lp.add_argument("--state", default="outputs/host_ops/watchdog_state.json")
    sub.add_parser("selftest")
    a = ap.parse_args(argv)
    return selftest() if a.cmd == "selftest" else loop()


if __name__ == "__main__":
    sys.exit(main())
