"""Stop a Modal app that is billing without producing output.

The failure this exists for is not a crash. A crashed job stops billing. What
costs money is a job that stays healthy, holds its containers and writes
nothing: the substrate audit held eight CPUs and 32 GB for two hours, left no
file behind, and the only thing that stopped it was the workspace hitting its
spend limit.

Liveness is therefore the wrong signal -- the run that burned the workspace was
alive the whole time. This watches *output* instead, and asks two separate
questions with two separate thresholds:

``--first-output-deadline``  how long a healthy run may take to write anything
                             at all. Must exceed one indivisible unit of work,
                             or the watchdog kills runs that are working; pass
                             ``--unit-seconds`` and it enforces that itself.
``--stall-hours``            how long a run that has already written something
                             may go without writing more.

Progress is a change in the *fingerprint* of the output tree -- every path with
its size and timestamp -- not a file count. The substrate audit rewrites a
single ``substrate.json`` once per family, so a count would read 1 forever and a
count-based watchdog would kill it at the deadline while it was working
perfectly. Size and mtime catch both new files and rewritten ones.

Stopping is the point. Reporting a stall while the containers keep running is
the behaviour that lost the workspace in the first place.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

HEALTHY = "HEALTHY"
NO_OUTPUT = "NO_OUTPUT"
STALLED = "STALLED"
FINISHED = "FINISHED"

MAX_DEPTH = 4


def classify(
    *,
    tasks: int,
    produced: bool,
    elapsed: float,
    seconds_since_change: float,
    first_output_deadline: float,
    stall_seconds: float,
) -> tuple[str, str]:
    """Decide what a run is doing, from output growth rather than liveness.

    Kept free of I/O so the decision is testable directly; the polling loop only
    gathers readings and applies this to them.
    """
    if tasks == 0:
        if produced:
            return FINISHED, "no tasks left and output was written"
        return NO_OUTPUT, "no tasks left and nothing was written"

    if not produced:
        if elapsed > first_output_deadline:
            return NO_OUTPUT, (
                f"{elapsed/3600:.2f} h elapsed with {tasks} task(s) running and nothing "
                f"written; the deadline was {first_output_deadline/3600:.2f} h"
            )
        return HEALTHY, f"{elapsed/3600:.2f} h elapsed, nothing written yet (within deadline)"

    if seconds_since_change > stall_seconds:
        return STALLED, (
            f"{seconds_since_change/3600:.2f} h since the last write with {tasks} "
            f"task(s) still running"
        )
    return HEALTHY, f"last write {seconds_since_change/60:.0f} min ago, {tasks} task(s) running"


def _run(args: list[str], profile: str) -> subprocess.CompletedProcess:
    env = dict(os.environ, MODAL_PROFILE=profile, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    return subprocess.run(args, capture_output=True, text=True, encoding="utf-8", env=env)


def _listing(volume: str, path: str, profile: str) -> list[dict] | None:
    """None distinguishes could-not-read from read-and-empty."""
    result = _run(["modal", "volume", "ls", volume, path, "--json"], profile)
    if result.returncode != 0:
        return None
    try:
        entries = json.loads(result.stdout)
    except ValueError:
        return None
    return entries if isinstance(entries, list) else None


def snapshot(
    volume: str, prefix: str, profile: str, *, max_depth: int = MAX_DEPTH
) -> frozenset | None:
    """Fingerprint the output tree as {(path, size, timestamp)}.

    Returns None if the listing could not be read at all, so a transient CLI
    failure reads as unknown rather than as the output having vanished -- the
    latter would silently reset the baseline and hide a stall forever.
    """
    root = _listing(volume, prefix, profile)
    if root is None:
        return None

    fingerprint: set[tuple[str, str, str]] = set()
    pending = [(root, 0)]
    while pending:
        entries, depth = pending.pop()
        for entry in entries:
            name = str(entry.get("Filename", ""))
            if entry.get("Type") == "dir":
                if depth < max_depth:
                    nested = _listing(volume, name, profile)
                    if nested is not None:
                        pending.append((nested, depth + 1))
                continue
            fingerprint.add(
                (name, str(entry.get("Size", "")), str(entry.get("Created/Modified", "")))
            )
    return frozenset(fingerprint)


def running_tasks(app_id: str, profile: str) -> int:
    """Task count for one app; -1 means the answer is unknown.

    Unknown must never be read as zero: zero tasks is a terminal verdict here,
    so a hiccup in the CLI would otherwise be reported as a finished run.
    """
    result = _run(["modal", "app", "list", "--json"], profile)
    if result.returncode != 0:
        return -1
    try:
        apps = json.loads(result.stdout)
    except ValueError:
        return -1
    for app in apps:
        if app.get("App ID") == app_id:
            try:
                return int(str(app.get("Tasks", "0")).strip() or 0)
            except ValueError:
                return -1
    return 0


def stop_app(app_id: str, profile: str) -> bool:
    return _run(["modal", "app", "stop", app_id], profile).returncode == 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--app-id", required=True)
    parser.add_argument("--volume", default="message-passing-retrieval-data")
    parser.add_argument("--prefix", required=True, help="volume path whose growth means progress")
    parser.add_argument(
        "--first-output-deadline",
        type=float,
        default=5.0,
        help="hours a run may take to write its first output",
    )
    parser.add_argument(
        "--unit-seconds",
        type=float,
        default=None,
        help="one indivisible work unit; a deadline shorter than this is refused",
    )
    parser.add_argument("--stall-hours", type=float, default=2.0)
    parser.add_argument("--poll-seconds", type=float, default=300.0)
    parser.add_argument("--no-stop", action="store_true", help="report only, do not stop the app")
    args = parser.parse_args(argv)

    deadline = args.first_output_deadline * 3600
    stall = args.stall_hours * 3600

    if args.unit_seconds is not None:
        # Both thresholds have to clear a unit, and the stall one is the easier
        # to get wrong: a run that writes once per unit -- the substrate audit
        # writes one substrate.json per graph family, 3.31 h apart -- looks
        # stalled for almost the whole of every unit it is working through.
        too_small = [
            (name, value)
            for name, value in (("first-output deadline", deadline), ("stall window", stall))
            if value <= args.unit_seconds
        ]
        if too_small:
            for name, value in too_small:
                print(
                    f"REFUSED: {name} of {value/3600:.2f} h is shorter than one work unit "
                    f"({args.unit_seconds/3600:.2f} h), so it would stop a run that is working.",
                    file=sys.stderr,
                )
            return 4

    baseline = snapshot(args.volume, args.prefix, args.profile)
    if baseline is None:
        print(f"REFUSED: cannot read {args.prefix} on {args.profile}", file=sys.stderr)
        return 4

    started = time.time()
    last_change = started
    last_seen = baseline
    print(
        f"watching {args.app_id} on {args.profile}: "
        f"{len(baseline)} file(s) under {args.prefix}",
        flush=True,
    )
    print(
        f"  deadline {args.first_output_deadline:g} h | stall {args.stall_hours:g} h "
        f"| poll {args.poll_seconds:g} s",
        flush=True,
    )

    while True:
        time.sleep(args.poll_seconds)
        current = snapshot(args.volume, args.prefix, args.profile)
        tasks = running_tasks(args.app_id, args.profile)
        now = time.time()

        if current is None or tasks < 0:
            print(f"[{(now-started)/60:6.0f}m] UNKNOWN: could not read state, will retry", flush=True)
            continue

        if current != last_seen:
            last_change = now
            last_seen = current

        state, why = classify(
            tasks=tasks,
            produced=current != baseline,
            elapsed=now - started,
            seconds_since_change=now - last_change,
            first_output_deadline=deadline,
            stall_seconds=stall,
        )
        print(f"[{(now-started)/60:6.0f}m] {state}: {why}", flush=True)

        if state == FINISHED:
            return 0
        if state in (NO_OUTPUT, STALLED):
            if args.no_stop:
                print(f"{state} -- leaving {args.app_id} running; --no-stop was given", flush=True)
                return 2
            stopped = stop_app(args.app_id, args.profile)
            print(
                f"{state} -- {'stopped' if stopped else 'FAILED TO STOP'} {args.app_id}",
                flush=True,
            )
            return 2 if stopped else 3


if __name__ == "__main__":
    raise SystemExit(main())
