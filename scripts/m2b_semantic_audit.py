#!/usr/bin/env python
"""Count M2B's three semantic candidates by building them, not by quoting them.

M2B compares S2, S3 and a projection-style control at the payload this track
actually runs on. Every number the declaration files about those candidates --
input width, semantic output width, semantic parameters, scorer parameters,
total -- comes from this script, and this script gets them from live
``nn.Module`` instances via ``numel()``.

The reason is specific and already cost this track something. ``qls_v2_semantic``
carries ``V1_SEMANTIC_PARAMETERS = 2 * EMBEDDING_DIM * 64 = 98304``, computed
from that module's 768-dimensional default, and "the historical projection is
about 98K parameters" has been repeated in planning prose ever since. The
payload is 1536-dimensional. Writing 98K into M2B's declaration would have
understated the control by exactly a factor of two, and every reduction ratio
computed from it would have been wrong in the direction that flatters S3.

So: no parameter total in this file is a literal. The widths come from the
declaration and the dataset; the counts come from the models.

Reads only. Instantiates models on CPU, runs no data through them beyond a
shape probe, trains nothing, and touches no dataset unless --data is given.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from mp_retrieval.m1a_screen import HEAD_WIDTH, M1AScorer  # noqa: E402
from mp_retrieval.m2b_semantic_control import (  # noqa: E402
    PROJECTION_DIM,
    ProjectionSemanticHead,
)
from mp_retrieval.qls_v2_semantic import (  # noqa: E402
    RUNG_FEATURES,
    V1_SEMANTIC_PARAMETERS,
    SemanticHead,
)

M2_DECLARATION = REPO_ROOT / "configs" / "m2_qls_v2_freeze.yaml"
OUTPUT_PATH = REPO_ROOT / "outputs" / "m2b_semantic_minimality" / "semantic_audit.json"

#: The three candidates, in the order the declaration lists them. "S4" is not a
#: member of qls_v2_semantic.RUNGS on purpose: that tuple is a strictly nested
#: frontier (each rung's feature set contains the one below), and a projection
#: control shares no column with it. Filing it as a fourth rung there would
#: break the nesting invariant that module's tests assert, and would claim a
#: monotone relationship that does not exist.
CANDIDATES = ("S2", "S3", "S4")

PROBE_CANDIDATES = 7


def _frozen_embedding_dim(declaration: dict[str, Any]) -> int:
    """The width M2 fit at, read out of M2's own declaration two independent ways.

    Taken from the declaration rather than from a constant here, because the
    whole failure this script exists to prevent is a width that is true of some
    other payload. ``architecture.semantic_head`` states it; ``parameter_count``
    implies it, since S3 holds exactly two vectors of the embedding width. If
    those two disagree the declaration is internally inconsistent and no
    parameter total derived from either is worth filing.
    """

    universal = declaration["qls_universal"]
    stated = re.search(r"dim=(\d+)", universal["architecture"]["semantic_head"])
    if not stated:
        raise SystemExit(
            "qls_universal.architecture.semantic_head no longer names its dim; the audit "
            "will not guess the width it is auditing"
        )
    semantic = int(universal["parameter_count"]["semantic"])
    if semantic % 2:
        raise SystemExit(
            f"cannot derive the embedding width from a semantic count of {semantic}; "
            "S3 holds two vectors of it"
        )
    width, implied = int(stated.group(1)), semantic // 2
    if width != implied:
        raise SystemExit(
            f"the declaration states dim={width} but its own S3 parameter count implies "
            f"{implied}; fix the declaration before counting anything against it"
        )
    return width


def build_head(candidate: str, dim: int) -> torch.nn.Module:
    if candidate == "S4":
        return ProjectionSemanticHead(dim=dim)
    return SemanticHead(rung=candidate, dim=dim)


def audit_candidate(candidate: str, *, dim: int, precomputed_width: int,
                    dropout: float, temperature: float) -> dict[str, Any]:
    head = build_head(candidate, dim)
    model = M1AScorer(
        precomputed_width=precomputed_width,
        semantic_rung=candidate,
        dropout=dropout,
        temperature=temperature,
        embedding_dim=dim,
        semantic_head=head,
    )

    # A shape probe rather than a trusted formula: the number of columns the
    # branch really emits is what the scorer's first Linear has to accept, and a
    # mismatch between the two is exactly the bug a hand-written width hides.
    with torch.no_grad():
        emitted = head(torch.randn(dim), torch.randn(PROBE_CANDIDATES, dim))
    semantic_width = int(emitted.shape[1])
    if semantic_width != len(head.feature_names):
        raise SystemExit(
            f"{candidate}: the branch emits {semantic_width} columns but names "
            f"{len(head.feature_names)}"
        )
    scorer_input = model.scorer[0].in_features
    if scorer_input != precomputed_width + semantic_width:
        raise SystemExit(
            f"{candidate}: scorer takes {scorer_input} columns, branch and structure "
            f"supply {precomputed_width + semantic_width}"
        )

    return {
        "candidate": candidate,
        "implementation": type(head).__name__,
        "module": type(head).__module__,
        "raw_embedding_dimension": dim,
        "semantic_output_width": semantic_width,
        "projection_dim": getattr(head, "projection_dim", None),
        "scorer_input_width": scorer_input,
        "precomputed_width": precomputed_width,
        "semantic_trainable_parameters": model.semantic_parameter_count(),
        "scorer_trainable_parameters": model.scorer_parameter_count(),
        "total_trainable_parameters": model.trainable_parameter_count(),
        "counted_from": "live nn.Module instances via numel(), not from any filed constant",
    }


def build(*, dim: int | None = None) -> dict[str, Any]:
    declaration = yaml.safe_load(M2_DECLARATION.read_text(encoding="utf-8"))
    universal = declaration["qls_universal"]
    schema = universal["feature_schema"]
    hyperparameters = universal["hyperparameters"]
    width = dim if dim is not None else _frozen_embedding_dim(declaration)
    precomputed_width = int(schema["precomputed_width"])

    rows = [
        audit_candidate(
            candidate,
            dim=width,
            precomputed_width=precomputed_width,
            dropout=float(hyperparameters["dropout"]),
            temperature=float(hyperparameters["temperature"]),
        )
        for candidate in CANDIDATES
    ]
    by_candidate = {row["candidate"]: row for row in rows}

    incumbent = by_candidate["S3"]
    filed = universal["parameter_count"]
    agrees = (
        incumbent["semantic_trainable_parameters"] == int(filed["semantic"])
        and incumbent["scorer_trainable_parameters"] == int(filed["scorer"])
        and incumbent["total_trainable_parameters"] == int(filed["total"])
    )
    if not agrees:
        raise SystemExit(
            "S3 rebuilt here does not reproduce the parameter count M2 filed and fit "
            f"({incumbent['total_trainable_parameters']} vs {filed['total']}); the audit "
            "cannot be trusted about the other two candidates until it reproduces the one "
            "with 15 completed fits behind it"
        )

    historical = {
        "qls_v2_semantic.V1_SEMANTIC_PARAMETERS": V1_SEMANTIC_PARAMETERS,
        "the_width_that_constant_assumes": V1_SEMANTIC_PARAMETERS // (2 * PROJECTION_DIM),
        "the_width_this_track_actually_runs": width,
        "measured_here_for_S4": by_candidate["S4"]["semantic_trainable_parameters"],
        "ratio_measured_over_historical": (
            by_candidate["S4"]["semantic_trainable_parameters"] / V1_SEMANTIC_PARAMETERS
        ),
        "verdict": (
            "HISTORICAL_EXACT"
            if by_candidate["S4"]["semantic_trainable_parameters"] == V1_SEMANTIC_PARAMETERS
            else "CURRENT_DIMENSION_PROJECTION_CONTROL"
        ),
        "why": (
            "The constant is 2 * 768 * 64, a fact about a 768-dimensional payload. This "
            "track's node embeddings are asserted at load time to be the width above. The "
            "projection formulas are v1's; the input width and the downstream scorer are "
            "M2's, so the control is not a reconstruction of v1 and is not labelled as one."
        ),
    }

    return {
        "status": "M2B_SEMANTIC_AUDIT_COMPLETE",
        "what_this_is_not": (
            "A fit, a selection, or an authorisation. It counts parameters on models that "
            "have been instantiated and never trained."
        ),
        "declaration_read": "configs/m2_qls_v2_freeze.yaml#qls_universal",
        "frozen_embedding_dim": width,
        "precomputed_width": precomputed_width,
        "head_width": HEAD_WIDTH,
        "scorer_is_shared": (
            "One M1AScorer class, one head width, one dropout and one temperature across all "
            "three candidates; only the semantic branch is swapped. The scorer's input width "
            "follows from the branch, which is the only thing that may differ."
        ),
        "candidates": rows,
        "s3_reproduces_the_filed_m2_count": agrees,
        "s2_feature_names": list(RUNG_FEATURES["S2"]),
        "s3_feature_names": list(RUNG_FEATURES["S3"]),
        "historical_projection_accounting": historical,
        "parameter_ladder_total": {
            row["candidate"]: row["total_trainable_parameters"] for row in rows
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-dim", type=int, default=None,
                        help="override the width derived from the M2 declaration")
    parser.add_argument("--out", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args(argv)

    report = build(dim=args.embedding_dim)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
