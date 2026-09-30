"""Transport amendment 1 of configs/host_mirror_six.yaml: the path layout that makes rx workspace 'mirror' land each file
at its declared mirror path, the size and hash refusals, and the amendment's shape."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "scripts", ROOT / "tools" / "rx"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import host_mirror_rx as HR  # noqa: E402
import rx  # noqa: E402

MIRROR = "C:/Users/Student2/rx/projects/mpr/mirror/CRAG"


def _decl(mirror_root: str = MIRROR) -> dict:
    return {"host": {"mirror_root": mirror_root, "read_only_sources": []}, "package": {"root_from": "unused"}}


def test_layout_maps_the_package_onto_the_mirror_workspace(tmp_path, monkeypatch):
    served = tmp_path / "Desktop" / "CRAG" / "data" / "final_canonical"
    served.mkdir(parents=True)
    monkeypatch.setattr(HR.HM, "laptop_served", lambda decl: served)
    base, prefix = HR.layout(_decl())
    assert base == tmp_path / "Desktop" and prefix == "CRAG/data/final_canonical"
    assert base.joinpath(*f"{prefix}/2wiki/x.npy".split("/")) == served / "2wiki" / "x.npy"
    # the host side: <rx home>/projects/mpr/<ws 'mirror'>/<prefix>/<rel> is mirror_root/data/final_canonical/<rel>
    assert Path(MIRROR).parent.name == HR.WS and (Path(MIRROR).parent / prefix) == Path(MIRROR) / "data" / "final_canonical"


def test_layout_refuses_a_name_or_placement_that_would_land_elsewhere(tmp_path, monkeypatch):
    served = tmp_path / "Desktop" / "OTHER" / "data" / "final_canonical"
    served.mkdir(parents=True)
    monkeypatch.setattr(HR.HM, "laptop_served", lambda decl: served)
    with pytest.raises(SystemExit):
        HR.layout(_decl())
    served2 = tmp_path / "Desktop2" / "CRAG" / "data" / "final_canonical"
    served2.mkdir(parents=True)
    monkeypatch.setattr(HR.HM, "laptop_served", lambda decl: served2)
    with pytest.raises(SystemExit):
        HR.layout(_decl("C:/Users/Student2/rx/projects/mpr/ws/CRAG"))
    with pytest.raises(SystemExit):
        HR.layout(_decl("C:/Users/Student2/rx/projects/crag/mirror/CRAG"))


def _files(tmp_path):
    base = tmp_path / "base"
    p = base / "CRAG" / "data" / "final_canonical" / "ds1" / "a.bin"
    p.parent.mkdir(parents=True)
    p.write_bytes(b"abc" * 100)
    row = {"rel": "ds1/a.bin", "dataset": "ds1", "bytes": 300, "sha256": hashlib.sha256(b"abc" * 100).hexdigest()}
    return base, row


def test_file_list_checks_sizes_and_the_hash_check_refuses_a_difference(tmp_path, monkeypatch):
    base, row = _files(tmp_path)
    prefix = "CRAG/data/final_canonical"
    files = HR.file_list([row], base, prefix)
    assert list(files) == [f"{prefix}/ds1/a.bin"]
    with pytest.raises(SystemExit):
        HR.file_list([dict(row, bytes=301)], base, prefix)
    monkeypatch.setattr(HR, "STATE", tmp_path / "state")
    HR.checked_hashes(rx, files, [row], prefix)                    # equal: passes
    assert (tmp_path / "state" / "hashcache.json").is_file()
    assert not any(p.name == "hashcache.json" for p in base.rglob("*"))   # nothing written beside the package's files
    with pytest.raises(SystemExit):
        HR.checked_hashes(rx, files, [dict(row, sha256="0" * 64)], prefix)


def test_rows_for_refuses_an_empty_selection():
    man = {"files": [{"rel": "a", "dataset": "ds1"}, {"rel": "b", "dataset": None}]}
    assert [r["rel"] for r in HR.rows_for(man, ["ds1"])] == ["a"]
    with pytest.raises(SystemExit):
        HR.rows_for(man, ["ds2"])


def test_the_amendment_names_the_rx_route_and_leaves_the_hub_alone():
    decl = yaml.safe_load((ROOT / "configs" / "host_mirror_six.yaml").read_text(encoding="utf-8"))
    am = decl["transport_amendment_1"]
    assert 'rx workspace "mirror"' in am["route"] and "Prune is off" in am["route"]
    assert "does not delete files from the hub" in am["why"]
    assert Path(decl["host"]["mirror_root"]).parent.name == HR.WS
