"""scripts/host_restore.py: the order of a new host's restore (no network)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import host_restore as HR  # noqa: E402


def tag(repo, name, prefix, utc):
    return {"repo": repo, "token_name": "x", "tag": name, "prefix": prefix, "created_utc": utc, "files": 1, "bytes": 1}


def test_mirror_first_then_workspace_tags_newest_first():
    tags = [tag("K/archive", "anc-x", "ws/", "2026-10-03T15:00:00Z"),
            tag("J/archive", "o-inc-2", "ws/", "2026-10-04T05:00:00Z"),
            tag("K/archive", "t0", "ws/", "2026-10-03T14:00:00Z"),
            tag("K/archive", "mirror-spared", "mirror/", "2026-10-03T16:30:00Z"),
            tag("J/archive", "o-small", "ws/", "2026-10-03T17:26:00Z")]
    steps = HR.order_steps(tags)
    assert steps[0]["key"] == "tag:K/archive:mirror-spared" and steps[0]["dest"] == "../mirror"
    assert [s["kind"] for s in steps[1:6]] == ["mirror"] * 5
    assert [s["key"] for s in steps[6:]] == ["tag:J/archive:o-inc-2", "tag:J/archive:o-small", "tag:K/archive:anc-x"]
    assert all(s["dest"] == "." for s in steps[6:])
    assert not any(s.get("tag") == "t0" for s in steps)


def test_mirror_groups_cover_the_six_datasets_once():
    ds = [d for group, _, _ in HR.MIRROR_GROUPS for d in group]
    assert sorted(ds) == sorted(["squad", "musique", "metaqa", "webqsp", "hotpotqa", "2wiki"])
    assert {repo for _, repo, _ in HR.MIRROR_GROUPS} == {"Swastik9895/mpr-six-mirror", "KK9895/mpr-six-mirror"}
