"""Build the final host delete list, outputs/host_cleanup/delete_final_<date>.json, in host_cleanup.py's format plus a
reason per group. Runs on the laptop; on the host it only lists files (one manifest call, no hashing, nothing written).

A file goes on the list only in one of these groups:
  A  the 490 files of delete_2026-10-03.json: big arrays of finished, filed MP-approx levels that no queued job reads.
     They are kept nowhere on purpose (each level's committed script regenerates them from the mirror); every level's
     records and results are on the laptop and the hub;
  B  the placement tests' bundles and arm outputs (cpu_gpu_equivalence, gpu_task_qualification), whose verdicts are filed;
  C  the oracle-probe arrays of levels 3, 4, 6 and 7 (filed; only levels 4 to 7 themselves read level 3);
  D  my test restore of the tag o-small (verified 819/819 on 2026-10-03), a copy whose originals stay on the host and hub.
B and C files must be 1 MB or more, be held by a VERIFIED hub tag at the host's current size and mtime (sha256-checked when
archived), and be on the laptop at the same size. Anything that fails a condition stays on the host and is reported.

  python outputs/host_cleanup/make_final_list.py            # writes the list and prints the groups
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path("C:/Users/Swastik/Desktop/message-passing-retrieval")
sys.path.insert(0, str(ROOT / "outputs" / "host_ops"))
import host_sync as HS  # noqa: E402

OUT = ROOT / "outputs" / "host_cleanup"
OLD = OUT / "delete_2026-10-03.json"
MB = 1_000_000
GROUPS = {
    "B": {"dirs": ["outputs/cpu_gpu_equivalence/", "outputs/gpu_task_qualification/"], "exts": (".pt", ".npz", ".npy"),
          "why": "Placement tests, finished and filed: the host CPU's EQUIVALENT verdict (configs/cpu_gpu_equivalence.yaml) "
                 "and the GPU task qualification. These are the laptop-built input bundles and the host arms' score "
                 "arrays. No queued job reads them. A placement test on a new machine pushes the bundles again from the "
                 "laptop or restores them from the hub."},
    "C": {"dirs": ["outputs/mp_approx_l3/", "outputs/mp_approx_l4/", "outputs/mp_approx_l6/", "outputs/mp_approx_l7/"],
          "exts": (".npz", ".npy"),
          "why": "Oracle-probe levels 3, 4, 6 and 7, finished and filed: halo, path and probe arrays. Only levels 4 to 7 "
                 "read level 3's arrays. No queued or later stage reads any of them (levels 10 to 15 read level 0, which "
                 "stays)."},
    "D": {"dirs": ["outputs/host_archive/restore_test_o-small/"], "exts": None,
          "why": "My test restore of the hub tag o-small into a scratch folder (819/819 files verified by sha256, "
                 "2026-10-03 23:22). It is a copy. The originals stay on the host and in JGY9895/mpr-host-archive."},
}
KEPT = [
    ["outputs/mp_approx_l0/", "Levels 10 to 15 read metaqa/qids.json and q_row.npy here in place."],
    ["outputs/mp_approx_l12/", "Levels 13 to 15 and later levels read its carves and dev sidecar."],
    ["outputs/mp_approx_l16_design/, outputs/mp_approx_*_anchor/",
     "look_x_six.py and qd_six.py join new looks to these looks; the anchors feed the relation-transfer jobs."],
    ["outputs/mp_approx_six_base/", "The six twin's and GNN's inputs for QD and the lean MLP."],
    ["outputs/m3b/", "CSR edge stores and relation files that queued lean and QD jobs read."],
    ["outputs/mp_unified/", "The active track (QD, the lean MLP, EM): its looks, stores and running jobs."],
    ["outputs/universal_v2/", "Small (0.5 GB); the pilot's inputs."],
    ["the mirror workspace (70 GB)", "Every host stage reads the six datasets there."],
    ["every file under 1 MB, and every record, script and result", "They are small and are what the levels are cataloged by."],
]


def main() -> None:
    old = json.loads(OLD.read_text(encoding="utf-8"))
    proj, r = HS.remote()
    dirs = [d for g in GROUPS.values() for d in g["dirs"]]
    host = HS.listing(proj, r, "ws", [d + "**" for d in dirs])
    hub = HS.hub_index()
    out_groups, files, held_back = [], [], defaultdict(list)
    out_groups.append({"group": "A", "dirs": old["dirs"], "files": len(old["files"]), "bytes": old["bytes"],
                       "why": "Intermediates of finished, filed MP-approx levels: L5, L8 to L11, L13 to L15, their "
                              "diagnostics (L10, L12, L14) and the metaqa design. These are big arrays only (.npz/.npy, "
                              "1 MB or more). No queued job reads them, and the host dry run of 2026-10-03 23:27 found all "
                              "490 unchanged. They are kept nowhere on purpose: each level's committed script regenerates "
                              "them from the mirror. The levels' records, results and small files are on the laptop and "
                              "the hub.",
                       "copies": "none (regenerable); records and results on the laptop and the hub"})
    files += [list(x) for x in old["files"]]
    for name, g in GROUPS.items():
        rows = []
        for rel, size, mtime_ns, _ in host:
            if not rel.startswith(tuple(g["dirs"])):
                continue
            if g["exts"] is not None:
                if not rel.lower().endswith(g["exts"]) or size < MB:
                    continue
                h = hub.get(("ws", rel))
                if h is None or h[0] != size or h[1] != mtime_ns:
                    held_back[name].append([rel, size, "not on the hub at this size and mtime"])
                    continue
                if not HS.on_laptop(rel, size):
                    held_back[name].append([rel, size, "not on the laptop at this size"])
                    continue
            rows.append([rel, size, mtime_ns // 1_000_000_000])
        tags = sorted({hub[("ws", rel)][3] for rel, _, _ in rows if ("ws", rel) in hub})
        out_groups.append({"group": name, "dirs": g["dirs"], "files": len(rows), "bytes": sum(s for _, s, _ in rows),
                           "why": g["why"],
                           "copies": ("the laptop's outputs/ and the hub, JGY9895/mpr-host-archive tag(s) " + ", ".join(tags))
                           if g["exts"] is not None else "the originals on the host and the hub"})
        files += rows
    doc = {"created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "dirs": sorted({d for g in out_groups for d in g["dirs"]}),
           "rule": "groups A to D below; host_cleanup.py removes a file only if its size and mtime still equal this list's",
           "groups": out_groups, "kept": KEPT, "held_back": dict(held_back),
           "files": files, "bytes": sum(f[1] for f in files)}
    dst = OUT / f"delete_final_{time.strftime('%Y-%m-%d')}.json"
    HS.atomic_write(dst, HS.jdump(doc))
    for g in out_groups:
        print(f"{g['group']}: {g['files']} files, {g['bytes'] / 1e9:.2f} GB  [{', '.join(g['dirs'][:4])}"
              f"{' ...' if len(g['dirs']) > 4 else ''}]  copies: {g['copies']}")
    for k, v in held_back.items():
        print(f"held back from {k}: {len(v)} files, {sum(x[1] for x in v) / 1e9:.2f} GB (first {v[:2]})")
    print(f"total {len(files)} files, {doc['bytes'] / 1e9:.2f} GB -> {dst.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
