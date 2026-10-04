"""Host-side feeder: send mpr's pending items to rx from ON the host, through the host's own rx agent, so the queue keeps
flowing while the laptop is off.

    python outputs/host_ops/hostfeed.py ITEMS STATE --agent-python PY [--agent AGENT] [--cap-cpus 26] [--cap-mem 104]
        [--cap-gpus 1] [--poll 120] [--hol-min 5] [--min-run 1500] [--max-h 60] [--linger-h 0] [--adopt-h 48] [--dry]

It runs as a small rx job of mpr (0.2 CPU) in mpr's workspace. rx's agent is a stdlib script that the laptop's ssh
session starts once per request (`python rx_agent-<sha>.py rpc`: one JSON line in, one out); this feeder starts it the
same way, as a local process of the job's own user, so no credential and no network is involved. RX_LAUNCH is unset
inside a job, so the agent starts each launched job's supervisor through WMI, outside this job's process tree: a job
sent from here outlives this feeder.

The rules are the laptop feeder's (scratchpad feeder.py). A job is sent only when rx can admit it at once (it fits the
free pool now) and mpr's whole reservation with it stays within the caps, so mpr never has a job waiting in rx's queue
and the rest of the pool stays open to other projects. A ready item that does not fit lets later items go until it has
waited --hol-min minutes; then nothing behind it is sent until it is, except that an item waiting only on the GPU (its
CPUs and memory fit) holds only the GPU items behind it: items without a GPU go on. An item whose dep or REF ends
badly is dropped.
STATE holds sent, dropped, measured and terminal, so a restart is safe.

ITEMS is re-read every poll (the laptop may push a longer list while this runs). One item per line, '#' comments:
    line|JOBSFILE|DEPS|name|cpus|mem|inputs|command[|gpu]
JOBSFILE is accepted for the laptop format and ignored. DEPS: comma-separated job names, each done with rc 0 first. mem:
a number, or @REF*A+B (REF's measured use: max(peak, live) once done rc 0, or after --min-run s running with a nonzero
reading; mem = ceil(1.15 x (A x use + B), 0.1)). gpu: 'gpu' sends the job under rx.toml's [profiles.gpu] (env
mpr-cu128, 1 GPU); absent or empty = [run] (mpr-cpu). inputs (';'-separated) are not pushed: each must already be in the
workspace (push it from the laptop first); an item with a missing input waits, and the feeder says so once.
A name resolves to the id this feeder sent for it, else to the newest mpr job of that exact name in rx. An unsent item
whose name already has an mpr job created within --adopt-h hours is adopted, not sent again (so items the laptop
feeder already sent are never doubled).
"""
import argparse
import hashlib
import json
import math
import os
import re
import secrets
import socket
import subprocess
import sys
import time
import tomllib
from pathlib import Path

REF = re.compile(r"@([A-Za-z0-9_.-]+)\*([0-9.]+)\+([0-9.]+)")
BAD = ("failed", "cancelled", "lost", "error")
LIVE = ("running", "queued", "finishing", "launching", "starting")

ap = argparse.ArgumentParser()
ap.add_argument("items")
ap.add_argument("state")
ap.add_argument("--agent-python", required=True, help="the python rx's agent runs under (the host cfg's 'python')")
ap.add_argument("--agent", default="", help="the agent script; default: the one that launched this job")
ap.add_argument("--cap-cpus", type=float, default=26.0)
ap.add_argument("--cap-mem", type=float, default=104.0)
ap.add_argument("--cap-gpus", type=float, default=1.0)
ap.add_argument("--poll", type=float, default=120.0)
ap.add_argument("--hol-min", type=float, default=5.0)
ap.add_argument("--min-run", type=float, default=1500.0)
ap.add_argument("--max-h", type=float, default=60.0)
ap.add_argument("--linger-h", type=float, default=0.0, help="once every item is handled, keep re-reading ITEMS this long")
ap.add_argument("--adopt-h", type=float, default=48.0)
ap.add_argument("--dry", action="store_true")
a = ap.parse_args()

WS = Path.cwd()
HOME = os.environ.get("RX_HOME") or ""
if not HOME:
    raise SystemExit("RX_HOME is not set: run this as an rx job on the host")
if a.agent:
    AGENT = a.agent
else:
    spec0 = json.loads((Path(os.environ["RX_JOB_DIR"]) / "spec.json").read_text(encoding="utf-8"))
    AGENT = str(Path(HOME) / "agent" / f"rx_agent-{spec0['agent']}.py")
if not Path(AGENT).exists() or not Path(a.agent_python).exists():
    raise SystemExit(f"agent or its python not found: {AGENT} {a.agent_python}")
RXT = tomllib.loads((WS / "rx.toml").read_text(encoding="utf-8"))
PROJECT = RXT["project"]["name"]


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


def parse_items(p):
    out = []
    for ln in Path(p).read_text(encoding="utf-8").splitlines():
        ln = ln.rstrip("\r")
        if not ln.strip() or ln.startswith("#"):
            continue
        f = ln.split("|")
        if f[0] != "line" or len(f) not in (8, 9):
            raise ValueError(f"need line|JOBSFILE|DEPS|name|cpus|mem|inputs|command[|gpu]: {ln[:120]}")
        g = f[8].strip() if len(f) == 9 else ""
        mg = re.fullmatch(r"gpu(?::([0-9.]+))?", g)
        if g not in ("", "cpu") and not mg:
            raise ValueError(f"9th field must be gpu, gpu:FRACTION, cpu or empty: {ln[:120]}")
        gpus = (float(mg.group(1)) if mg.group(1) else 1.0) if mg else 0.0
        if not 0.0 <= gpus <= 1.0:
            raise ValueError(f"gpu fraction must be in [0, 1]: {ln[:120]}")
        mem = f[5]
        m = REF.fullmatch(mem)
        if mem.startswith("@") and not m:
            raise ValueError(f"bad memory expression {mem!r} ({f[3]})")
        if not m:
            float(mem)
        out.append({"deps": [d for d in f[2].split(",") if d], "name": f[3], "cpus": float(f[4]), "mem": mem,
                    "ref": m, "inputs": [x for x in f[6].split(";") if x], "command": f[7], "gpu": gpus})
    names = [i["name"] for i in out]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate item names: {sorted({n for n in names if names.count(n) > 1})}")
    return out


state_p = Path(a.state)
st = json.loads(state_p.read_text(encoding="utf-8")) if state_p.exists() else {}
for k in ("sent", "dropped", "measured", "terminal", "said"):
    st.setdefault(k, {})


def save():
    tmp = state_p.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, indent=1), encoding="utf-8")
    for _ in range(60):
        try:
            tmp.replace(state_p)
            return
        except PermissionError:
            time.sleep(0.5)
    tmp.replace(state_p)


def say_once(key, msg):
    if st["said"].get(key) != msg:
        st["said"][key] = msg
        save()
        log(msg)


def overview():
    d = agent({"op": "overview", "limit": 300}, timeout=120)
    cap = d["capacity"]
    act = d["active"]
    mine = [x for x in act if x.get("project") == PROJECT]
    gpus_total = len(cap.get("gpus") or [])
    jobs = agent({"op": "status", "project": PROJECT, "limit": 5000}, timeout=180)["jobs"]
    newest = {}
    for j in jobs:            # newest first
        newest.setdefault(j.get("name"), j)
    return {"now": d.get("now") or time.time(),
            "free_c": cap["cpus"] - cap.get("reserve_cpus", 0) - sum(float(x["req"]["cpus"]) for x in act),
            "free_m": cap["mem_gb"] - cap.get("reserve_mem_gb", 0) - sum(float(x["req"]["mem_gb"]) for x in act),
            "free_g": gpus_total - sum(float(x["req"].get("gpus") or 0) for x in act),
            "mpr_c": sum(float(x["req"]["cpus"]) for x in mine), "mpr_m": sum(float(x["req"]["mem_gb"]) for x in mine),
            "mpr_g": sum(float(x["req"].get("gpus") or 0) for x in mine),
            "queued_mpr": [q["id"] for q in d["queue"] if q.get("project") == PROJECT],
            "by_id": {j["id"]: j for j in jobs}, "newest": newest}


def job_state(name, items_by, ov, exact=False):
    """(state, rc, use_gb, ran_s) of the job a name resolves to; 'pending' for an unsent item of ITEMS."""
    if name in items_by and name not in st["sent"]:
        return ("failed", None, None, 0.0) if name in st["dropped"] else ("pending", None, None, 0.0)
    jid = st["sent"].get(name) or (ov["newest"].get(name) or {}).get("id")
    if jid is None:
        return ("unknown", None, None, 0.0)
    if jid in st["terminal"]:
        return tuple(st["terminal"][jid])
    j = ov["by_id"].get(jid)
    if exact or j is None:
        r = agent({"op": "status", "project": PROJECT, "ids": [jid]}, timeout=120)["jobs"]
        j = r[0] if r else None
    if j is None:
        return ("unknown", None, None, 0.0)
    use = max(float(j.get("peak_mem_gb") or 0.0), float(j.get("live_mem_gb") or 0.0))
    ran = ((j.get("ended") or ov["now"]) - float(j["started"])) if j.get("started") else 0.0
    out = (j.get("state"), j.get("rc"), use, ran)
    if out[0] in BAD or out[0] == "done":
        st["terminal"][jid] = list(out)
    return out


def is_bad(s):
    return s[0] in BAD or (s[0] == "done" and s[1] != 0)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def submit(it, mem):
    d = dict(RXT.get("run") or {})
    if it["gpu"]:
        d.update((RXT.get("profiles") or {})["gpu"])
    argv = it["command"].split()
    spec = {"project": PROJECT, "id": make_id(it["name"]), "name": it["name"], "ws": d.get("ws") or "ws",
            "shell": "exec", "argv": argv, "command": None, "env": d.get("env"), "cpus": float(it["cpus"]),
            "mem_gb": float(mem), "gpus": float(it["gpu"]), "mem_hard_gb": d.get("mem_hard_gb"),
            "outputs": d.get("outputs") or ["**"], "outputs_exclude": list(d.get("outputs_exclude") or []),
            "env_vars": {str(k): str(v) for k, v in (d.get("env_vars") or {}).items()}, "cwd": d.get("cwd") or ".",
            "timeout_s": d.get("timeout_s"), "wsl_distro": d.get("wsl_distro"), "display": " ".join(argv),
            "code": {"via": "hostfeed", "feeder_job": os.environ.get("RX_JOB_ID"),
                     "inputs_sha256": {x: sha(WS / x) for x in it["inputs"]}},
            "submitted_from": socket.gethostname(), "_fetch": False, "created": time.time()}
    if a.dry:
        log("DRY", spec["id"], spec["env"], spec["cpus"], spec["mem_gb"], spec["gpus"], spec["display"][:160])
        return None
    st["sent"][it["name"]] = spec["id"]           # recorded first: a launch is idempotent per id, so a retry is safe
    save()
    try:
        r = agent({"op": "launch", "spec": spec}, timeout=600)
        return r["job"]["id"] if r.get("job") else spec["id"]
    except Exception as e:  # the job may exist anyway: look it up before deciding
        try:
            got = agent({"op": "status", "project": PROJECT, "ids": [spec["id"]]}, timeout=120)["jobs"]
        except Exception:
            got = None
        if got:
            return spec["id"]
        if got is not None:
            del st["sent"][it["name"]]
            save()
            log(f"launch of {it['name']} failed and no job exists; will retry: {e!r}"[:400])
            return None
        log(f"launch of {it['name']} unclear ({e!r}); keeping {spec['id']} as sent"[:400])
        return spec["id"]


t0 = time.time()
done_since = None
ready_since = {}
items = []
log(f"hostfeed: ws {WS}, agent {AGENT}, caps {a.cap_cpus:g}c/{a.cap_mem:g}G/{a.cap_gpus:g}g, poll {a.poll:g}s")
while True:
    try:
        items = parse_items(a.items)
    except Exception as e:  # a half-written list: keep the last good one
        log("ITEMS unreadable, keeping the last good list:", repr(e)[:300])
    by = {i["name"]: i for i in items}
    try:
        ov = overview()
    except Exception as e:  # a transient agent failure: try again next poll
        log("overview failed:", repr(e)[:300])
        time.sleep(a.poll)
        continue
    for i in items:            # adopt what the laptop feeder (or an earlier run) already sent
        n = i["name"]
        if n not in st["sent"] and n not in st["dropped"] and n in ov["newest"]:
            j = ov["newest"][n]
            if ov["now"] - float(j.get("created") or 0) < a.adopt_h * 3600:
                st["sent"][n] = j["id"]
                save()
                log(f"ADOPT {n} -> {j['id']} ({j.get('state')})")
    left = [i for i in items if i["name"] not in st["sent"] and i["name"] not in st["dropped"]]
    if not left:
        done_since = done_since or time.time()
        if time.time() - done_since >= a.linger_h * 3600:
            log("every item handled:", len(st["sent"]), "sent,", len(st["dropped"]), "dropped")
            break
        time.sleep(a.poll)
        continue
    done_since = None
    if ov["queued_mpr"]:
        log(f"mpr has {len(ov['queued_mpr'])} job(s) queued in rx ({ov['queued_mpr'][:3]}); sending nothing this poll")
        time.sleep(a.poll)
        continue
    free_c, free_m, free_g = ov["free_c"], ov["free_m"], ov["free_g"]
    mpr_c, mpr_m, mpr_g = ov["mpr_c"], ov["mpr_m"], ov["mpr_g"]
    sent_now, blocked_head, gpu_head = [], None, None
    for it in left:
        n = it["name"]
        try:
            sts = {d: job_state(d, by, ov) for d in it["deps"]}
        except Exception as e:
            log(f"dep check for {n} failed: {e!r}"[:300])
            continue
        if any(is_bad(s) for s in sts.values()):
            st["dropped"][n] = f"dep ended badly: {[(d, s[:2]) for d, s in sts.items() if is_bad(s)]}"
            save()
            log("DROP", n, st["dropped"][n])
            continue
        if any(s[0] == "unknown" for s in sts.values()):
            say_once(n, f"{n}: unknown dep(s) {[d for d, s in sts.items() if s[0] == 'unknown']}; waiting")
            continue
        if not all(s[0] == "done" and s[1] == 0 for s in sts.values()):
            continue
        miss = [x for x in it["inputs"] if not (WS / x).exists()]
        if miss:
            say_once(n, f"{n}: inputs missing in the workspace {miss[:4]}; waiting")
            continue
        mem = it["mem"]
        if it["ref"]:
            r = it["ref"].group(1)
            if r not in st["measured"]:
                s = job_state(r, by, ov)
                if s[0] == "running" and s[3] >= a.min_run:
                    s = job_state(r, by, ov, exact=True)
                if is_bad(s):
                    st["dropped"][n] = f"REF {r} ended badly {s[:2]}"
                    save()
                    log("DROP", n, st["dropped"][n])
                    continue
                if (s[0] == "done" and s[1] == 0) or (s[0] == "running" and s[3] >= a.min_run and (s[2] or 0) > 0):
                    st["measured"][r] = s[2]
                    save()
                    log(f"REF {r} {s[0]} after {s[3] / 60:.0f} min: {s[2]:.2f} GB")
            if r not in st["measured"]:
                continue
            mem = f"{math.ceil(1.15 * (float(it['ref'].group(2)) * st['measured'][r] + float(it['ref'].group(3))) * 10) / 10:.1f}"
        c, m, g = it["cpus"], float(mem), it["gpu"]
        if gpu_head and g > 0:
            continue
        ready_since.setdefault(n, time.time())
        fits_cm = (mpr_c + c <= a.cap_cpus + 1e-9 and mpr_m + m <= a.cap_mem + 1e-9 and c <= free_c + 1e-9
                   and m <= free_m + 1e-9)
        fits = fits_cm and mpr_g + g <= a.cap_gpus + 1e-9 and g <= free_g + 1e-9
        if not fits:
            if time.time() - ready_since[n] > a.hol_min * 60:
                if fits_cm and g > 0:
                    gpu_head = n
                    continue
                blocked_head = n
                break
            continue
        jid = submit(it, mem)
        if jid:
            mpr_c, mpr_m, mpr_g, free_c, free_m, free_g = mpr_c + c, mpr_m + m, mpr_g + g, free_c - c, free_m - m, free_g - g
            sent_now.append(f"{n}({c:g}c/{m:g}G{f'/{g:g}gpu' if g else ''})")
    if sent_now:
        log(f"sent {len(sent_now)}: {' '.join(sent_now)} | mpr now {mpr_c:.1f}c/{mpr_m:.1f}G/{mpr_g:g}g, "
            f"free {free_c:.1f}c/{free_m:.1f}G/{free_g:g}g")
    if blocked_head:
        say_once("_head", f"head {blocked_head} waited over {a.hol_min:g} min; holding later items for it")
    if gpu_head:
        say_once("_gpu_head", f"GPU head {gpu_head} waited over {a.hol_min:g} min on the GPU; holding later GPU items "
                              f"for it, the others go on")
    if time.time() - t0 > a.max_h * 3600:
        log("max time reached; pending:", [i["name"] for i in left][:30])
        sys.exit(1)
    if a.dry:
        break
    time.sleep(a.poll)
