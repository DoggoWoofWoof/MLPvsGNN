#!/usr/bin/env python
"""Export verified edge-family sidecars from a read-only CRAG substrate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from mp_retrieval.edge_provenance import (
    reconstruct_edge_families,
    save_edge_families,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--master", type=Path, required=True)
    parser.add_argument("--baseline-graph", type=Path, required=True)
    parser.add_argument("--ner", type=Path, required=True)
    parser.add_argument("--node-ids", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    families, metadata = reconstruct_edge_families(
        dataset=args.dataset,
        master_path=args.master,
        baseline_graph_path=args.baseline_graph,
        ner_path=args.ner,
        expected_node_ids_path=args.node_ids,
    )
    save_edge_families(families, metadata, args.output)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
