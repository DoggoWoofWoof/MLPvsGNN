"""Host cleanup: remove the files a delete list names, and nothing else. Run on the host, in the rx workspace.

  python outputs/host_cleanup/host_cleanup.py --list outputs/host_cleanup/delete_2026-10-03.json            # dry run
  python outputs/host_cleanup/host_cleanup.py --list outputs/host_cleanup/delete_2026-10-03.json --delete   # delete

A file is removed only if its path is under one of the list's directories (and that directory is in ALLOWED below), its
extension is one ALLOWED gives that directory (None: any file, used only for a test restore's copy), and its size and mtime
equal the list's (taken from the host's own manifest). Anything else is skipped and reported. Directories left empty are
removed; non-empty ones are left alone. The record goes to outputs/host_cleanup/<dry_run|deleted>_<stamp>.json. Stdlib only.
"""
import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

ARRAYS = (".npz", ".npy")
ALLOWED = {d: ARRAYS for d in (
    "outputs/mp_approx_l5/", "outputs/mp_approx_l8/", "outputs/mp_approx_l9/", "outputs/mp_approx_l10/",
    "outputs/mp_approx_l11/", "outputs/mp_approx_l13/", "outputs/mp_approx_l14/", "outputs/mp_approx_l15/",
    "outputs/mp_approx_l14_diag/", "outputs/mp_approx_l12_diag/", "outputs/mp_approx_l10_diag/",
    "outputs/mp_approx_mq_design/", "outputs/mp_approx_l3/", "outputs/mp_approx_l4/", "outputs/mp_approx_l6/",
    "outputs/mp_approx_l7/")}
ALLOWED.update({"outputs/cpu_gpu_equivalence/": (".pt",) + ARRAYS, "outputs/gpu_task_qualification/": (".pt",) + ARRAYS,
                "outputs/host_archive/restore_test_o-small/": None})


def check(ws: Path, doc: dict):
    dirs = set(doc["dirs"])
    if not dirs <= set(ALLOWED):
        raise SystemExit(f"the list names directories outside ALLOWED: {sorted(dirs - set(ALLOWED))}")
    ok, skipped = [], []
    for rel, size, mtime in doc["files"]:
        why = None
        under = [d for d in dirs if rel.startswith(d)]
        if ".." in rel.split("/") or rel.startswith("/") or ":" in rel:
            why = "bad path"
        elif not under:
            why = "outside the list's directories"
        elif ALLOWED[under[0]] is not None and not rel.lower().endswith(ALLOWED[under[0]]):
            why = f"not {'/'.join(ALLOWED[under[0]])}"
        else:
            p = ws / rel
            try:
                st = p.stat()
            except FileNotFoundError:
                why = "missing"
            else:
                if not p.is_file():
                    why = "not a file"
                elif st.st_size != size:
                    why = f"size {st.st_size} != {size}"
                elif int(st.st_mtime) != mtime:
                    why = f"mtime {int(st.st_mtime)} != {mtime}"
        (skipped if why else ok).append([rel, size] + ([why] if why else []))
    return ok, skipped


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", required=True)
    ap.add_argument("--delete", action="store_true", help="remove the checked files (default: dry run)")
    a = ap.parse_args(argv)
    ws = Path.cwd().resolve()
    if not (ws / "outputs").is_dir():
        raise SystemExit(f"{ws} has no outputs/: run this in the rx workspace")
    doc = json.loads((ws / a.list).read_text(encoding="utf-8"))
    free0 = shutil.disk_usage(str(ws)).free
    ok, skipped = check(ws, doc)
    removed, failed = [], []
    if a.delete:
        for rel, size in ok:
            try:
                os.remove(ws / rel)
                removed.append([rel, size])
            except OSError as e:
                failed.append([rel, size, f"{type(e).__name__}: {e}"])
        parents = sorted({(ws / rel).parent for rel, _ in removed}, key=lambda p: -len(p.parts))
        for d in parents:
            while d != ws and d.is_relative_to(ws / "outputs") and d != ws / "outputs":
                try:
                    d.rmdir()            # only an empty directory goes
                except OSError:
                    break
                d = d.parent
    free1 = shutil.disk_usage(str(ws)).free
    rec = {"mode": "delete" if a.delete else "dry_run", "list": a.list, "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "listed": len(doc["files"]), "checked_ok": len(ok), "checked_ok_bytes": sum(s for _, s in ok),
           "removed": len(removed), "removed_bytes": sum(s for _, s in removed), "failed": failed, "skipped": skipped,
           "disk_free_gb_before": round(free0 / 1e9, 2), "disk_free_gb_after": round(free1 / 1e9, 2),
           "removed_files": [r for r, _ in removed]}
    # "deleted_", never "delete_": delete_*.json names a delete list (host_sync.py reads every one of them)
    out = ws / "outputs" / "host_cleanup" / f"{'deleted' if a.delete else 'dry_run'}_{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(rec, indent=1), encoding="utf-8")
    os.replace(tmp, out)
    print(f"{rec['mode']}: listed {rec['listed']}, ok {rec['checked_ok']} ({rec['checked_ok_bytes'] / 1e9:.2f} GB), "
          f"skipped {len(skipped)}, removed {rec['removed']} ({rec['removed_bytes'] / 1e9:.2f} GB), failed {len(failed)}; "
          f"free {rec['disk_free_gb_before']} -> {rec['disk_free_gb_after']} GB; record {out.relative_to(ws).as_posix()}")
    for s in skipped[:20]:
        print("  skipped", s)
    for f in failed[:20]:
        print("  failed", f)
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
