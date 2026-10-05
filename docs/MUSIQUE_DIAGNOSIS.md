# MuSiQue diagnosis

**Declared 5 October 2026, before any number below was computed.** This file states the question, the inputs, the
slices and the metrics. The script is `scripts/musique_diagnosis.py`. Results are appended under "Results" after the
declaration is committed.

## Question

On musique the universal GAT reaches recall@5 0.521, against a pool ceiling@5 of 0.921. Where is that gap, and what
does the published number we calibrate against measure?

## Inputs

These are read-only, as filed by M3B (closed at 0bbb3c4). The script checks each sha256 against the value in
`docs/M3B_RESULTS.md` and refuses on a mismatch.

| file | sha256 |
| --- | --- |
| `outputs/m3b/eval/musique.npz` | `554b7a20605b6d965f3af62477103dc7c66e05079b716d781d2fa3d3408ee06c` |
| `outputs/m3b/eval/musique_query_ids.json` | `4412638ce9a31e412545fb750130f843f5abbdc8e52849cfe6dea840acbd1024` |

## Slices

- **Hop count**, from the MuSiQue question id prefix: `2hop` is 2, `3hop1` and `3hop2` are 3, and `4hop1`, `4hop2` and
  `4hop3` are 4.
- **Composition type**: the prefix itself.
- **All** 2,417 eval queries.

## Metrics

For each slice:

- n, the mean number of gold passages, the share of queries with every gold in the pool, and the pool ceiling@5. The
  ceiling is the mean of min(5, golds in the pool) / golds.
- recall@5, full_coverage@5 and hit@1 for each arm: fixed `rrf`, QLS-U, GAT-NO-MP and the universal GAT. Seed 0 is the
  row, and the mean over seeds 0 to 2 is shown beside it.
- δ_MP = GAT − GAT-NO-MP (seed 0) on recall@5 and full_coverage@5, with a paired bootstrap 95% interval over queries
  (2,000 resamples, seed 0).

The calibration is like for like. GraphER's PR@K is set coverage: 1 only if every gold passage is in the top K
(`docs/M3A_SOTA_ARCHAEOLOGY.md`). So our full_coverage@5 is placed next to GraphER's PR@5, not our recall@5.

## What this does not do

It trains nothing, selects nothing, and changes no filed record. It is a diagnosis, not a result of M3B, and M3B's
filed reading stands.
