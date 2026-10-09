"""Run a script with the laptop's numba 0.67.0 and llvmlite 0.49.0 (cp313, win_amd64) on sys.path, on the host, whose
rx envs have no numba (and whose pip cannot reach the index past the lab's TLS-inspecting firewall). The rx env is not
touched: the packages are unpacked once into outputs/host_ops/pylib/cp313 from the zip pushed beside this file
(installed files copied from the laptop's site-packages, tests left out), checked by the zip's sha256.

    python outputs/host_ops/pylib_run.py SCRIPT.py [args...]
"""
import hashlib
import os
import runpy
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ZIP = HERE / "pylib" / "numba067_llvmlite049_cp313_win_amd64.zip"
DEST = HERE / "pylib" / "cp313"


def unpack():
    sha = hashlib.sha256(ZIP.read_bytes()).hexdigest()
    mark = DEST / "unpacked_from.sha256"
    if mark.exists() and mark.read_text().strip() == sha:
        return sha
    tmp = DEST.with_name(DEST.name + f".tmp{os.getpid()}")
    with zipfile.ZipFile(ZIP) as z:
        z.extractall(tmp)
    (tmp / "unpacked_from.sha256").write_text(sha)
    if DEST.exists():
        if (DEST / "unpacked_from.sha256").exists() and (DEST / "unpacked_from.sha256").read_text().strip() == sha:
            return sha                       # another run unpacked it meanwhile
        raise SystemExit(f"{DEST} holds another unpack: move it aside first")
    os.replace(tmp, DEST)
    return sha


if __name__ == "__main__":
    if sys.version_info[:2] != (3, 13):
        raise SystemExit(f"the packages are cp313, this is {sys.version}")
    sha = unpack()
    sys.path.insert(0, str(DEST))
    import numba  # noqa: E402
    print(f"pylib_run: numba {numba.__version__} from the pushed zip {sha[:12]}", flush=True)
    script = sys.argv[1]
    sys.argv = sys.argv[1:]
    sys.path.insert(0, str(Path(script).resolve().parent))
    runpy.run_path(script, run_name="__main__")
