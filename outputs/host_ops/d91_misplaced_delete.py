"""Delete outputs/d91_misplaced/b1x: the 173 names D91's first B1 rows wrote into AP2's outputs/b1x, moved there on 17 Nov
(d91_b1x_misplaced.py, moved.json). The user's OK: "Yes, delete them" (17 Nov).

    python outputs/host_ops/d91_misplaced_delete.py check     (read only: the folder holds exactly moved.json's names; every
                                                                hard link still has another name, so no byte is lost)
    python outputs/host_ops/d91_misplaced_delete.py delete    (the check, then os.remove of each name, the empty folders, and
                                                                moved.json; refuses if the check finds anything else)
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "outputs" / "d91_misplaced" / "b1x"
B1V = ROOT / "outputs" / "b1v"
OWN_TOPS = ("links", "records", "steps")


def check():
    mv = json.loads((DEST / "moved.json").read_text(encoding="utf-8"))
    moved = set(mv["moved"])
    found = set()
    for dp, _dn, fn in os.walk(DEST):
        for f in fn:
            rp = (Path(dp) / f).relative_to(DEST).as_posix()
            if rp != "moved.json":
                found.add(rp)
    extra, missing = sorted(found - moved), sorted(moved - found)
    last, own, linked, nbytes_last = [], 0, 0, 0
    for rp in sorted(found & moved):
        p = DEST / rp
        st = os.stat(p)
        if rp.split("/")[0] in OWN_TOPS:
            own += 1
            nbytes_last += st.st_size
            continue
        b = B1V / rp
        if st.st_nlink < 2 or not (b.exists() and os.path.samefile(p, b)):
            last.append((rp, st.st_nlink, st.st_size))
        else:
            linked += 1
    out = {"moved_json": mv.get("count"), "found": len(found), "extra": extra[:20], "missing": missing[:20],
           "d91_own_files": own, "d91_own_bytes": nbytes_last, "links_with_b1v_name": linked, "last_name": last[:20],
           "n_last_name": len(last)}
    print(json.dumps(out, indent=1))
    return out


def delete():
    c = check()
    if c["extra"] or c["missing"] or c["n_last_name"]:
        raise SystemExit("d91_misplaced_delete: the folder is not exactly moved.json's names, or a link is the file's last name: "
                         "nothing deleted")
    n = 0
    for rp in json.loads((DEST / "moved.json").read_text(encoding="utf-8"))["moved"]:
        os.remove(DEST / rp)
        n += 1
    for dp, _dn, _fn in sorted(os.walk(DEST), key=lambda x: -len(Path(x[0]).parts)):
        p = Path(dp)
        if p != DEST and not os.listdir(p):
            os.rmdir(p)
    os.remove(DEST / "moved.json")
    os.rmdir(DEST)
    par = DEST.parent
    if not os.listdir(par):
        os.rmdir(par)
    print(f"deleted {n} names and moved.json; outputs/d91_misplaced exists: {par.exists()}")
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1:2]
    if cmd == ["check"]:
        check()
    elif cmd == ["delete"]:
        delete()
    else:
        raise SystemExit(__doc__)
