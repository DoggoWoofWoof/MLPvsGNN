"""After the host is lost: put back in the feeder's queue every item whose job had not finished, so the feeder sends it
again on the new host. Stop the feeder first. Dry run by default; --apply rewrites STATE (the old file is kept beside it).

The last job table host_sync.py saved (outputs/host_job_logs/status_latest.json) decides:
  - a sent item whose job that table shows done, failed, cancelled or lost keeps its place, and its state goes into
    STATE's terminal record, so items that depend on it read it there instead of asking the new host (which has never
    seen the id);
  - a sent item whose job is running or queued in that table, or whose id is newer than every id in it, goes back to the
    queue (its name leaves STATE's sent record);
  - a sent item older than the table's window is taken as finished, as it was when the feeder last looked;
  - a 'rerun' item cannot run on a new host (rx rerun needs the old host's stored spec): it is listed, to be rewritten as
    a 'line' item, and left alone.

  python outputs/host_ops/requeue_after_host_loss.py outputs/host_ops/scratchpad/feeder_items.txt \
         outputs/host_ops/scratchpad/feeder_state.json [--apply]
"""
import argparse
import json
import os
import time
from pathlib import Path

ROOT = Path("C:/Users/Swastik/Desktop/message-passing-retrieval")
TABLE = ROOT / "outputs" / "host_job_logs" / "status_latest.json"
ACTIVE = ("running", "queued", "waiting", "starting", "launching", "finishing")
FINISHED = ("done", "failed", "cancelled", "lost", "error")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("items")
    ap.add_argument("state")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    kinds = {}
    for ln in Path(a.items).read_text(encoding="utf-8").splitlines():
        if ln.strip() and not ln.startswith("#"):
            f = ln.split("|")
            kinds[f[3]] = f[0]
    st_p = Path(a.state)
    st = json.loads(st_p.read_text(encoding="utf-8"))
    for k in ("sent", "terminal"):
        st.setdefault(k, {})
    tab = json.loads(TABLE.read_text(encoding="utf-8"))
    jobs = {j["id"]: j for j in tab["jobs"]}
    newest, oldest = max(jobs), min(jobs)
    back, kept, marked, reruns, old = [], [], [], [], []
    for name, jid in sorted(st["sent"].items()):
        j = jobs.get(jid)
        if j is not None and j.get("state") in FINISHED:
            kept.append(name)
            if jid not in st["terminal"]:
                use = max(float(j.get("peak_mem_gb") or 0.0), float(j.get("live_mem_gb") or 0.0))
                ran = (float(j.get("ended") or 0.0) - float(j["started"])) if j.get("started") and j.get("ended") else 0.0
                st["terminal"][jid] = [j["state"], j.get("rc"), use, ran]
                marked.append(name)
        elif (j is not None and j.get("state") in ACTIVE) or (j is None and jid > newest):
            (reruns if kinds.get(name) == "rerun" else back).append(name)
        else:
            old.append(name)
            if jid not in st["terminal"] and jid < oldest:
                st["terminal"][jid] = ["done", 0, None, 0.0]
                marked.append(name)
    print(f"job table of {tab.get('utc')}: {len(jobs)} jobs, ids {oldest} .. {newest}")
    print(f"back to the queue ({len(back)}): {back}")
    print(f"'rerun' items to rewrite as 'line' items ({len(reruns)}): {reruns}")
    print(f"finished, kept ({len(kept)}); older than the table, taken as finished ({len(old)}); terminal records added: {len(marked)}")
    if not a.apply:
        print("dry run: nothing written (add --apply)")
        return
    for name in back:
        del st["sent"][name]
    bak = st_p.with_name(st_p.name + time.strftime(".before_requeue_%Y%m%d-%H%M%S"))
    bak.write_bytes(st_p.read_bytes())
    tmp = st_p.with_name(st_p.name + ".tmp")
    tmp.write_text(json.dumps(st, indent=1), encoding="utf-8")
    os.replace(tmp, st_p)
    print(f"written: {st_p} (the old state is {bak.name})")


if __name__ == "__main__":
    main()
