"""Which of the ten query-local columns is exactly bare retrieval-seed membership.

D3 splits QLS-v1's local block into seed identity, seed geometry, and everything
else. That split is only meaningful if one column is *exactly* `I[d in Sq]` and
not a proxy for it, so this file proves the identity against the historical
`_seed_indicator` rather than assuming the distance-zero bucket must be it.

The proof drives the real feature path -- `build_local_features` -> the shipped
`_local_feature_chunk` kernel -> `candidate_readout` -- so it fails if any of
those three ever stops preserving the identity. The same invariant is asserted
again at full scale inside the D3 runner, on real queries.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, REPO_ROOT / "src", REPO_ROOT / "tests"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from mp_retrieval.graph_context import NORMALISED_COLUMNS, candidate_readout  # noqa: E402
from mp_retrieval.linear_control import LOCAL_FEATURE_NAMES  # noqa: E402
from scripts.run_sa_mlp_confirmation import _seed_indicator  # noqa: E402
from test_graph_context_d1 import dataset as build_dataset, local_block  # noqa: E402

#: The column D3 treats as bare seed membership, and the group it belongs to.
SEED_ID_COLUMN = 0
DISTANCE_COLUMNS = (0, 1, 2, 3)


@pytest.fixture(scope="module")
def data():
    return build_dataset()


@pytest.fixture(scope="module")
def cand_block(data):
    """The historical CAND local block, exactly as D1 and D2 build it."""

    local, ptr = local_block(data, "CAND")
    return np.asarray(local, dtype=np.float32), ptr


def historical_indicator(queries) -> np.ndarray:
    lengths = [int(query.candidate_index.numel()) for query in queries]
    return _seed_indicator(queries, lengths, torch.device("cpu")).numpy()[:, 0]


# --- the schema -----------------------------------------------------------------


def test_the_ten_column_schema_is_what_d3_assumes():
    assert LOCAL_FEATURE_NAMES == (
        "distance_0",
        "distance_1",
        "distance_2",
        "distance_3_plus_or_unreachable",
        "seed_connections",
        "paths_length_1",
        "paths_length_2",
        "paths_length_3",
        "personalized_pagerank",
        "common_out_neighbors_with_seed_neighborhood",
    )
    assert LOCAL_FEATURE_NAMES[SEED_ID_COLUMN] == "distance_0"
    assert tuple(LOCAL_FEATURE_NAMES[i] for i in DISTANCE_COLUMNS) == LOCAL_FEATURE_NAMES[:4]


def test_the_distance_buckets_are_never_rescaled():
    """`candidate_readout` must not touch columns 0-3, or column 0 stops being a bit."""

    assert NORMALISED_COLUMNS == (4, 5, 6, 7, 8, 9)
    assert not set(NORMALISED_COLUMNS) & set(DISTANCE_COLUMNS)
    block = np.arange(40, dtype=np.float32).reshape(4, 10)
    before = block[:, DISTANCE_COLUMNS].copy()
    assert np.array_equal(candidate_readout(block)[:, DISTANCE_COLUMNS], before)


# --- the identity ---------------------------------------------------------------


def test_the_distance_zero_column_is_exactly_the_historical_seed_indicator(data, cand_block):
    """The claim D3 rests on, checked elementwise over every candidate row."""

    local, _ = cand_block
    expected = historical_indicator(data.queries)
    observed = local[:, SEED_ID_COLUMN]
    assert observed.shape == expected.shape
    assert np.array_equal(observed, expected), (
        "distance_0 is no longer exactly I[d in Sq]; D3's SEED_ID_ONLY arm is invalid"
    )
    assert expected.any(), "no seeds in the fixture, so this proved nothing"
    assert not expected.all(), "every candidate is a seed, so this proved nothing"


def test_the_identity_holds_query_by_query(data, cand_block):
    """A global match could hide two errors that cancel. This cannot."""

    local, ptr = cand_block
    for position, query in enumerate(data.queries):
        rows = local[ptr[position] : ptr[position + 1], SEED_ID_COLUMN]
        seeds = query.retrieval_seed_local.numpy()
        expected = np.zeros(rows.shape[0], dtype=np.float32)
        expected[seeds] = 1.0
        assert np.array_equal(rows, expected), position


def test_the_column_is_binary(cand_block):
    local, _ = cand_block
    assert set(np.unique(local[:, SEED_ID_COLUMN]).tolist()) <= {0.0, 1.0}


def test_no_other_column_coincides_with_seed_membership(data, cand_block):
    """If a second column were also the indicator, the decomposition would leak."""

    local, _ = cand_block
    expected = historical_indicator(data.queries)
    for column in range(1, len(LOCAL_FEATURE_NAMES)):
        assert not np.array_equal(local[:, column], expected), LOCAL_FEATURE_NAMES[column]


def test_the_distance_group_is_a_complete_one_hot(cand_block):
    """So zeroing 4-9 leaves geometry intact, and keeping only 0 leaves membership."""

    local, _ = cand_block
    buckets = local[:, DISTANCE_COLUMNS]
    assert np.array_equal(buckets.sum(axis=1), np.ones(buckets.shape[0], dtype=np.float32))
    assert set(np.unique(buckets).tolist()) <= {0.0, 1.0}


def test_keeping_only_column_zero_reproduces_the_indicator_and_nothing_else(data, cand_block):
    """Exactly what `SEED_ID_ONLY` hands the model: one bit, no geometry."""

    local, _ = cand_block
    masked = np.zeros_like(local)
    masked[:, SEED_ID_COLUMN] = local[:, SEED_ID_COLUMN]
    assert np.array_equal(masked[:, SEED_ID_COLUMN], historical_indicator(data.queries))
    assert not masked[:, 1:].any()


# --- the precondition -----------------------------------------------------------


def test_the_identity_needs_unique_candidates_and_the_loader_guarantees_them(data):
    """The one way this could break, named rather than left implicit.

    `build_local_features` maps seeds to *global node ids* and reads features
    back by value, while `_seed_indicator` marks *local positions*. If a query
    listed the same node twice, with one occurrence a seed, the two would
    disagree. The frozen loader builds `candidate_index` with `_stable_union`,
    which deduplicates, so this cannot arise -- but it is the precondition, and
    the D3 runner re-checks it on real data rather than trusting this fixture.
    """

    from mp_retrieval.complete_data import _stable_union

    duplicated = np.array([7, 3, 7, 5], dtype=np.int64)
    assert _stable_union(duplicated).numpy().tolist() == [7, 3, 5]

    for query in data.queries:
        candidates = query.candidate_index.numpy()
        assert np.unique(candidates).size == candidates.size, int(query.query_index)
