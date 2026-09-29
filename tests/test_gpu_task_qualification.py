"""configs/gpu_task_qualification.yaml: the criteria on toy scores, the verdict ladder, the probe window and the pins."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import cpu_gpu_equivalence as CGE  # noqa: E402
import gpu_task_qualification as G  # noqa: E402


def _metrics(raw: np.ndarray, gold: list) -> np.ndarray:
    m = CGE.rank_metrics(raw, np.asarray(gold, dtype=np.int64), len(gold))
    return np.asarray([m[n] for n in CGE.METRIC_NAMES], dtype=np.float64)


# ── the criteria ─────────────────────────────────────────────────────────────


def test_order_breaks_ties_by_pool_position_as_rank_metrics_does():
    raw = np.asarray([1.0, 2.0, 2.0, 0.5], dtype=np.float32)
    assert G.order_of(raw).tolist() == [1, 2, 0, 3]
    assert G.top_k(raw, 1) == frozenset({1}) and G.top_k(raw, 5) == frozenset({0, 1, 2, 3})
    m = CGE.rank_metrics(raw, np.asarray([2]), 1)
    assert m["hit@1"] == 0.0 and m["first_gold_rank"] == 2.0


def test_a_near_tie_pair_is_twice_the_forward_tolerance_at_its_size():
    assert G.near_tie_pair(1.0, 1.0 - 1e-5) and not G.near_tie_pair(1.0, 1.0 - 1e-4)
    assert G.near_tie_pair(0.0, 0.0) and G.near_tie_pair(0.3, 0.30001)        # equal, and a reversed pair
    assert G.near_tie_pair(100.0, 100.0 - 1.5e-3) and not G.near_tie_pair(100.0, 100.0 - 3e-3)      # 2 * (1e-5 + 1e-3)


def test_a_swap_between_near_tied_candidates_is_excused_and_any_other_change_is_not():
    ref = np.asarray([1.0, 1.0 - 5e-6, 0.3, 0.2, 0.1, 0.0, -0.1], dtype=np.float32)
    cen = ref - ref.mean()
    swapped = ref.copy()
    swapped[[0, 1]] = swapped[[1, 0]]
    r = G.draw_reading(swapped, ref, cen, _metrics(swapped, [0]), _metrics(ref, [0]))
    assert r["top1_differs"] and not r["top5_differs"]
    assert r["excused"] and not r["rank_difference"] and r["metric_differs"] and not r["metric_violation"]
    far = ref.copy()
    far[2] = 2.0     # candidate 2 jumps from rank 3 to rank 1: the reference's near tie at rank 1 does not excuse it
    r = G.draw_reading(far, ref, cen, _metrics(far, [0]), _metrics(ref, [0]))
    assert r["top1_differs"] and r["rank_difference"] and not r["excused"] and r["metric_violation"]
    same = G.draw_reading(ref.copy(), ref, cen, _metrics(ref, [0]), _metrics(ref, [0]))
    assert not any(same[k] for k in ("top1_differs", "top5_differs", "rank_difference", "excused", "metric_differs", "metric_violation"))
    few = np.asarray([1.0, 0.0, -1.0], dtype=np.float32)
    assert G.boundary(few[::-1].copy(), few, few, 5) == (False, False)          # k >= pool size: the whole pool


def test_a_top5_boundary_swap_needs_the_swapped_pair_near_tied():
    ref = np.asarray([5.0, 4.0, 3.0, 2.0, 1.0, 0.0, -1.0], dtype=np.float32)
    cen = ref - ref.mean()
    x = ref.copy()
    x[[4, 5]] = x[[5, 4]]              # ranks 5 and 6 swap across a gap of 1.0
    r = G.draw_reading(x, ref, cen, _metrics(x, [4]), _metrics(ref, [4]))
    assert r["top5_differs"] and r["rank_difference"] and r["metric_violation"]
    tied = np.asarray([5.0, 4.0, 3.0, 2.0, 1.0, 1.0 - 5e-6, -1.0], dtype=np.float32)
    y = tied.copy()
    y[[4, 5]] = y[[5, 4]]              # ranks 5 and 6 swap across a near tie
    r = G.draw_reading(y, tied, tied - tied.mean(), _metrics(y, [4]), _metrics(tied, [4]))
    assert r["top5_differs"] and r["excused"] and not r["rank_difference"] and not r["metric_violation"]


def test_compare_counts_cells_ranks_and_excuses_on_toy_arms():
    ref = np.asarray([1.0, 1.0 - 5e-6, 0.2, 0.1, 0.0, -0.1, 2.0, 1.0, 0.5], dtype=np.float32)
    ptr = np.asarray([0, 6, 9])
    x = ref.copy()
    x[[0, 1]] = x[[1, 0]]              # draw 0: an excused near-tie swap, within score tolerance of its own size
    x[6], x[7] = 0.9, 1.1              # draw 1: a real ranking difference, far outside score tolerance

    def t1(raw):
        cen = np.concatenate([raw[s:e] - raw[s:e].mean() for s, e in zip(ptr[:-1], ptr[1:])]).astype(np.float32)
        mets = np.stack([_metrics(raw[s:e], [0]) for s, e in zip(ptr[:-1], ptr[1:])])
        return {"raw__m": raw, "centred__m": cen, "metrics__m": mets}

    c = G.compare(t1(x), t1(ref), ["m"], ptr, ["a", "b"])
    assert c["rank_fail"] == {("m", 1)} and c["excused"] == {("m", 0)} and ("m", 1) in c["score_fail"]
    m = c["models"]["m"]
    assert m["top1_differs"] == 2 and m["rank_differences"] == [1] and m["excused"] == [0]
    assert m["metric_deltas"]["b"]["hit@1"] == -1.0 and m["metric_deltas"]["all"]["draws_differing"] == 2


def test_the_verdict_ladder_in_its_declared_order():
    assert G.evaluation_verdict(False, True, set(), set()) == "NOT_REPRODUCIBLE"
    assert G.evaluation_verdict(True, True, set(), set()) == "SCORE_EQUIVALENT"
    assert G.evaluation_verdict(True, False, set(), {("m", 1)}) == "TASK_EQUIVALENT"
    assert G.evaluation_verdict(True, False, {("m", 1)}, set()) == "RANKING_DIFFERS"
    assert G.evaluation_verdict(True, False, {("m", 1)}, {("m", 1), ("m", 2)}) == "INCONCLUSIVE_FLOOR"
    assert G.evaluation_verdict(True, False, {("m", 1), ("m", 3)}, {("m", 1)}) == "RANKING_DIFFERS"
    with pytest.raises(SystemExit):
        G.evaluation_verdict(True, True, {("m", 1)}, set())


def test_training_reproducibility_needs_every_loss_the_state_and_the_scores():
    a = {"loss__m": np.asarray([1.0, 0.5], dtype=np.float32), "centred__m": np.asarray([0.1, -0.1], dtype=np.float32)}
    rows = {"m": {"state_sha256": "x"}}
    assert G.training_verdict(a, dict(a), rows, {"m": {"state_sha256": "x"}})[0] == "TRAINING_REPRODUCIBLE"
    b = dict(a, loss__m=np.asarray([1.0, 0.5000001], dtype=np.float32))
    assert G.training_verdict(a, b, rows, rows)[0] == "TRAINING_NOT_REPRODUCIBLE"
    assert G.training_verdict(a, dict(a), rows, {"m": {"state_sha256": "y"}})[0] == "TRAINING_NOT_REPRODUCIBLE"


# ── the probe ────────────────────────────────────────────────────────────────


def test_the_probe_starts_after_the_equivalence_probe():
    eq = yaml.safe_load((ROOT / "configs" / "cpu_gpu_equivalence.yaml").read_text(encoding="utf-8"))
    assert eq["run_record_cpu_gpu_equivalence_2026_09_29"]["probe"]["batches"] == G.EQ_BATCHES == G.FIRST == 11
    assert (G.N_BATCHES, G.MIN_BATCHES, G.BUNDLE_CAP_BYTES) == (16, 8, 1.0e9)


def test_overlap_and_replay_checks_on_toy_draws():
    earlier = [{"queries": [{"dataset": "a", "fit_index": 1}, {"dataset": "b", "fit_index": 7}]}]
    parts_all = [[("a", np.asarray([1])), ("b", np.asarray([7]))], [("a", np.asarray([2, 3]))]]
    assert G.replay_problem(parts_all, earlier) is None
    assert G.overlap_with(parts_all[1:], earlier) == []
    assert G.overlap_with([[("b", np.asarray([7, 8]))]], earlier) == [("b", 7)]
    assert G.replay_problem([[("a", np.asarray([1])), ("b", np.asarray([8]))]], earlier) is not None


def test_the_size_cap_keeps_the_largest_prefix_that_fits():
    assert G.keep_under_cap([100] * 16, 50, cap=1650, least=8) == 16
    assert G.keep_under_cap([100] * 16, 50, cap=1000, least=8) == 9
    assert G.keep_under_cap([100] * 16, 50, cap=800, least=8) is None


# ── the pins ─────────────────────────────────────────────────────────────────


def test_the_imported_script_is_the_pinned_bytes_and_the_declaration_quotes_them():
    assert G.eq_script_problem() is None
    decl = (ROOT / "configs" / "gpu_task_qualification.yaml").read_text(encoding="utf-8")
    assert G.EQ_SCRIPT_SHA256 in decl


def test_the_shared_arms_keep_the_equivalence_files_settings():
    for arm in ("laptop_cpu_t8", "laptop_cpu_t4", "host_gpu_det", "host_gpu_det_r2", "host_cpu_t8"):
        for k in ("host", "device", "threads", "mode", "env"):
            assert G.ARMS[arm][k] == CGE.ARMS[arm][k], (arm, k)
        assert G.ARMS[arm]["tf32"] is False
    tf32 = G.ARMS["host_gpu_tf32"]
    assert tf32["tf32"] is True and tf32["role"] == "diagnosis" and tf32["mode"] == "det"
    assert "host_gpu_default" not in G.ARMS


def test_the_tolerance_and_task_metrics_are_the_declared_ones():
    assert G.ATOL == G.RTOL == 1e-5
    assert G.TASK_METRICS == ("recall@5", "full_coverage@5", "hit@1")
    decl = yaml.safe_load((ROOT / "configs" / "gpu_task_qualification.yaml").read_text(encoding="utf-8"))
    assert decl["status"] in ("DECLARED_NOT_RUN", "RUN")
    assert set(decl["verdicts"]["evaluation"]) == {"NOT_REPRODUCIBLE", "SCORE_EQUIVALENT", "TASK_EQUIVALENT", "RANKING_DIFFERS", "INCONCLUSIVE_FLOOR"}


def test_imported_helpers_stop_into_this_phases_outputs(monkeypatch):
    monkeypatch.setattr(CGE, "hard_stop", CGE.hard_stop)       # restored after the test
    assert CGE.hard_stop is not G.hard_stop                    # importing changed nothing
    G.bind_stops()
    assert CGE.hard_stop is G.hard_stop
