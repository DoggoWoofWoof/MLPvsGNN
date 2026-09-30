"""Transport amendment 1 of configs/host_mirror_six.yaml: the manifest's files for the named datasets go to the host over
rx's own channel (laptop to host, no third party) after the hub's private-storage limit stopped the HF route.

tools/rx/rx.py is imported unchanged. Its push runs with the file list replaced by the manifest's rows, read in place
from the package root: nothing is written under it, and rx's hash cache lives in outputs/host_mirror_six/rx_state. The
files go to rx workspace "mirror" of project mpr, the folder that holds host.mirror_root, so each lands at its declared
path. Prune is off. Before anything is sent, every file's size and rx's sha256 of it must equal the manifest's.
Afterwards the host verify stage of scripts/host_mirror_hf.py runs unchanged for the dataset.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "scripts", ROOT / "tools" / "rx"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import host_mirror_hf as HM  # noqa: E402

WS = "mirror"
STATE = HM.OUT / "rx_state"
RETRY_FOR = 6 * 3600.0          # rx's push gives up once a transport failure outlasts this; the loop below starts it again
ATTEMPTS = 20
LOG_EVERY = 300.0              # seconds between progress lines (rx's own bar draws only on a terminal)


def layout(decl: dict) -> tuple[Path, str]:
    """(local base, rel prefix): the local path of a manifest row is base / prefix / rel and its host path is
    <mpr project>/mirror / prefix / rel = mirror_root / data/final_canonical / rel. Refuses unless the package root's
    folder name equals the mirror root's and the mirror root sits directly in the project's 'mirror' folder."""
    served = HM.laptop_served(decl)
    package_root = served.parents[len(Path(HM.SERVED_REL).parts) - 1]
    mirror_root = Path(decl["host"]["mirror_root"])
    if mirror_root.parent.name != WS or mirror_root.parent.parent.name != "mpr" or mirror_root.parents[2].name != "projects":
        raise SystemExit(f"{mirror_root}: not <rx home>/projects/mpr/{WS}/<name>; refusing")
    if package_root.name != mirror_root.name:
        raise SystemExit(f"the package folder {package_root.name!r} is not the mirror's {mirror_root.name!r}; refusing")
    return package_root.parent, f"{mirror_root.name}/{HM.SERVED_REL}"


def rows_for(man: dict, datasets: list[str]) -> list[dict]:
    rows = [r for r in man["files"] if r["dataset"] in datasets]
    if not rows:
        raise SystemExit(f"the manifest holds no rows for {datasets}")
    return rows


def file_list(rows: list[dict], base: Path, prefix: str) -> dict:
    """rx's {rel: (abs path, stat)}, each file's size checked against the manifest (its hash is checked after rx's)."""
    files = {}
    for r in rows:
        rel = f"{prefix}/{r['rel']}"
        ap = base.joinpath(*rel.split("/"))
        st = os.stat(ap)
        if st.st_size != r["bytes"]:
            raise SystemExit(f"{ap}: {st.st_size} bytes, the manifest says {r['bytes']}; refusing")
        files[rel] = (str(ap), st)
    return files


def checked_hashes(rx, files: dict, rows: list[dict], prefix: str) -> None:
    """rx's own sha256 of every file (its hash cache, kept outside the package) must equal the manifest's."""
    STATE.mkdir(parents=True, exist_ok=True)
    cache = rx.HashCache(str(STATE / "hashcache.json"))
    want = {f"{prefix}/{r['rel']}": r["sha256"] for r in rows}
    bad = [rel for rel, (ap, st) in files.items() if cache.sha(rel, ap, st) != want[rel]]
    cache.save()
    if bad:
        raise SystemExit(f"{len(bad)} file(s) differ from the manifest, e.g. {bad[0]}; refusing")


def log_progress(rx):
    """rx's Progress with its terminal bar replaced by a line every LOG_EVERY seconds, for a log file."""
    class LogProgress(rx.Progress):
        def __init__(self, total, label, enabled=True):
            super().__init__(total, label, enabled=False)
            self.enabled = enabled
            if enabled:
                threading.Thread(target=self._every, daemon=True).start()

        def _every(self):
            while not self._stop.wait(LOG_EVERY):
                self._line()

        def _line(self):
            r = self.rate()
            print(f"{HM.utc()}   {self.label} {self.done / 1e9:.2f} of {self.total / 1e9:.2f} GB, {r / 1e6:.2f} MB/s, "
                  f"eta {(self.total - self.done) / r / 3600 if r > 0 else float('nan'):.1f} h", flush=True)

        def close(self):
            self._stop.set()
            if self.enabled:
                self._line()
    return LogProgress


def send(datasets: list[str], streams: int | None) -> dict:
    import rx                                         # tools/rx/rx.py, unchanged
    decl = HM.load_decl()
    man = json.loads(HM.MANIFEST.read_text(encoding="utf-8"))
    base, prefix = layout(decl)
    rows = rows_for(man, datasets)
    files = file_list(rows, base, prefix)
    t0 = time.time()
    checked_hashes(rx, files, rows, prefix)
    print(f"{HM.utc()} {datasets}: {len(files)} files, {sum(r['bytes'] for r in rows) / 1e9:.2f} GB, sizes and sha256 "
          f"equal the manifest ({time.time() - t0:.0f}s)", flush=True)
    proj = rx.Project(str(ROOT))
    proj.root, proj.state = str(base), str(STATE)     # this process only: rx reads base/rel and keeps its cache here
    rx.local_manifest = lambda _proj, _extra=(): dict(files)
    rx.Progress = log_progress(rx)
    remote = rx.Remote(rx.load_host(proj.host))
    summaries, t1 = [], time.time()
    for attempt in range(1, ATTEMPTS + 1):
        try:
            summaries.append(rx.push(proj, remote, ws=WS, prune=False, force=True, streams=streams, retry_for=RETRY_FOR))
            break                                     # push returns only once the host holds every file
        except rx.Die as e:
            print(f"{HM.utc()} attempt {attempt}: {e}; starting again in 60s", flush=True)
            time.sleep(60)
    else:
        raise SystemExit(f"{datasets}: the push did not finish in {ATTEMPTS} attempts")
    confirm = rx.push(proj, remote, ws=WS, dry=True, prune=False, streams=streams, quiet=True)
    if confirm["need"]:
        raise SystemExit(f"{datasets}: the host still lacks {confirm['need']} file(s) after the push")
    out = {"utc": HM.utc(), "datasets": datasets, "route": "rx workspace " + WS, "files": len(files),
           "bytes": sum(r["bytes"] for r in rows), "seconds": round(time.time() - t1, 1), "attempts": len(summaries),
           "sent_raw": sum(s.get("sent_raw", 0) for s in summaries), "sent_wire": sum(s.get("sent_wire", 0) for s in summaries),
           "manifest_sha256": HM.hashlib.sha256(HM.MANIFEST.read_bytes()).hexdigest()}
    out["mb_per_s_wire"] = round(out["sent_wire"] / max(out["seconds"], 1e-6) / 1e6, 3)
    HM.atomic_json(HM.OUT / f"rx_push_{'_'.join(datasets)}.json", out)
    print(f"{HM.utc()} {datasets}: pushed {out['sent_raw'] / 1e9:.2f} GB ({out['sent_wire'] / 1e9:.2f} GB on the wire) in "
          f"{out['seconds'] / 3600:.2f} h, {out['mb_per_s_wire']} MB/s", flush=True)
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--streams", type=int, default=None)
    a = ap.parse_args(argv)
    decl = HM.load_decl()
    unknown = set(a.datasets) - set(decl["selection"]["datasets"])
    if unknown:
        raise SystemExit(f"not declared: {sorted(unknown)}")
    send(a.datasets, a.streams)


if __name__ == "__main__":
    main()
