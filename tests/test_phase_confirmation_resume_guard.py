"""The E2 runner must never retrain a model-seed it has already recorded.

E2 is resumed rather than restarted: 48 of its 96 cells were complete before
the workspace migration, and ten more were partial with eight or nine of their
ten model-seeds already trained. Resuming is driven by what a cell's
``result.json`` records, not by a cell's position in a job list. The guard that
enforces it is two lines inside the seed loop of ``run_phase_confirmation.py``:

    if seed_key in result["models"][model_name]["seeds"]:
        continue

If that guard is removed, reordered after training, or weakened, a resumed cell
silently retrains model-seeds that were already frozen and overwrites them with
fresh numbers. Nothing downstream would notice -- the cell would still look
complete, with a full seed set and a valid contract -- so the corruption would
reach the paper as a result rather than as an error.

Structural rather than behavioural: running the real loop needs a GPU, a
dataset and a trained screen checkpoint. What is checked here is that the guard
is present, that it is the first thing the loop body does, and that it skips
rather than merely warning.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

RUNNER = (
    Path(__file__).resolve().parents[1] / "scripts" / "run_phase_confirmation.py"
)


@pytest.fixture(scope="module")
def seed_loop() -> ast.For:
    """The ``for seed in args.seeds`` loop inside the runner."""

    tree = ast.parse(RUNNER.read_text(encoding="utf-8"))
    loops = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id == "seed"
    ]
    assert len(loops) == 1, (
        f"expected exactly one `for seed in ...` loop, found {len(loops)}; "
        "this test locates the guard by that loop and cannot disambiguate."
    )
    return loops[0]


def _guard(loop: ast.For) -> ast.If:
    """The first ``if`` in the loop body -- the already-recorded check."""

    for node in loop.body:
        if isinstance(node, ast.If):
            return node
    raise AssertionError("the seed loop has no guard at all")


def test_the_seed_loop_opens_with_a_membership_check(seed_loop: ast.For) -> None:
    test = _guard(seed_loop).test
    assert isinstance(test, ast.Compare), "guard is not a comparison"
    assert len(test.ops) == 1 and isinstance(test.ops[0], ast.In), (
        "guard must test membership of the seed in the recorded seed set; "
        "a different comparison would skip on the wrong condition."
    )


def test_the_guard_reads_the_recorded_seeds_not_the_checkpoint_directory(
    seed_loop: ast.For,
) -> None:
    # Checkpoint files on disk are not the record of what was trained: a cell
    # can carry a checkpoint whose metrics never reached result.json. The
    # recorded seed set is the authority.
    # ast.unparse normalises string quoting, so compare against its own form.
    source = ast.unparse(_guard(seed_loop).test)
    assert source == "seed_key in result['models'][model_name]['seeds']", source


def test_the_guard_skips_rather_than_logs(seed_loop: ast.For) -> None:
    body = _guard(seed_loop).body
    assert len(body) == 1 and isinstance(body[0], ast.Continue), (
        "the guard must `continue`. Falling through after a warning would "
        "retrain the seed it just identified as already recorded."
    )


def test_nothing_in_the_loop_does_real_work_before_the_guard(
    seed_loop: ast.For,
) -> None:
    """Only trivial assignments may precede the check.

    `seed_key = str(seed)` legitimately comes first -- it computes the very key
    the guard tests. What must not creep in above the guard is work: seeding
    the RNG, building a model, loading a checkpoint. All of that is wasted for
    a seed about to be skipped, and `seed_everything` in particular would
    perturb global RNG state for the seeds that are actually trained.
    """

    guard = _guard(seed_loop)
    for node in seed_loop.body[: seed_loop.body.index(guard)]:
        calls = [c for c in ast.walk(node) if isinstance(c, ast.Call)]
        offenders = sorted(
            ast.unparse(c.func) for c in calls if ast.unparse(c.func) != "str"
        )
        assert not offenders, (
            "work performed before the already-recorded guard: "
            + ", ".join(offenders)
        )


def test_seed_zero_reuses_the_screen_checkpoint_under_a_verified_hash() -> None:
    # The other half of "never retrain what exists": seed 0 is not trained at
    # all, it is loaded from E1 -- and only after both the file and the loaded
    # state hash match what E1 recorded. Dropping either check would let a
    # corrupted transfer become a silent scientific result.
    source = RUNNER.read_text(encoding="utf-8")
    assert 'if _sha256(checkpoint_path) != source["checkpoint_file_sha256"]:' in source
    assert 'if _state_sha256(state) != source["checkpoint_sha256"]:' in source
    # Both checks must raise rather than warn-and-continue.
    for check in ("checkpoint failed SHA-256", "state failed SHA-256"):
        assert f'raise ValueError("Validation-screen seed-0 {check}")' in source
