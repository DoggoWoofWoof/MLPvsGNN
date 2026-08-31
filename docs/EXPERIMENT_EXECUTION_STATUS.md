# Experiment execution status

Last updated: 2026-08-31.

This is an operational checkpoint, not a result table. No partial seed or
validation-cell outcome has been interpreted here.

## Completed and closed

- The six-dataset four-model fairness confirmation is complete and immutable.
- Package A1/A2/A3 is complete and closed to further test-driven tuning.
- The original C-RAG repository remains read-only. All new code, caches,
  protocols, outputs, commits, and tags belong to this standalone repository.

## Active frozen packages

| Package | Frozen checkpoint | Current execution boundary |
|---|---|---|
| B — edge provenance | `edge-provenance-protocol-v2` | Four new graph families × six datasets × QLS/GNN × five seeds; sealed A is reused |
| C — candidate budget | `candidate-budget-protocol-v1` | Budgets 50/100/200/400 × six datasets × QLS/GNN × five seeds |
| D — uncached systems | `online-systems-protocol-v1` | Code and parity gates complete; waits for Package C budget-400 checkpoints |
| E1 — phase screen | `phase-screen-protocol-v1` | Four axes × four rates × six datasets, seed 0, validation only |

The active Modal run pages are:

- Package B: `ap-u43K1tZADRHNKZqHdg4Yv0`
- Package C: `ap-7RMOj5HqgPgMXfUV6Z91X3`
- Package E1: `ap-ydAGFwev7ToWAJGw1JCdcS`

All jobs write disjoint, fingerprinted paths and checkpoint after each model
seed. A Windows UTF-8 launcher issue and a Modal Volume refresh conflict were
fixed without changing candidates, graphs, models, seeds, metrics, or any other
scientific condition. Interrupted jobs resume from their committed seed
records.

## Analysis gates already implemented

- Package B compiler requires all registered families, exact packed-query
  hashes/order, all five seeds, paired seed/query intervals, and Holm correction.
- Package C compiler requires all four budgets and forbids a test-selected
  budget.
- Package E1 compiler reads validation only and applies the preregistered
  endpoint/crossover-bracket rule. It generates a proposed confirmation config
  that must be committed and tagged before any selected test cell runs.
- Package D compiler keeps cached operator timing and uncached post-retrieval
  timing separate.

The five-seed phase-confirmation runner is prepared. It reuses the untouched
validation-selected seed-0 screen checkpoint, trains seeds 1–4, and evaluates
test only after the generated rate configuration is frozen. Clean rate zero is
reused from the sealed confirmation rather than rerun four times.

## Next automatic gates

1. Wait for complete Package B/C/E1 artifacts; do not interpret individual
   seeds or incomplete datasets.
2. Compile B and C only after every registered output is present.
3. Launch Package D after all six budget-400 result/checkpoint/cache contracts
   pass.
4. Run the E1 validation-only compiler, review its deterministic selected-rate
   union, then commit and tag the E2 confirmation protocol.
5. Launch E2 five-seed confirmation. Do not build a utility predictor unless
   E2 contains reproducible message-passing help, neutral, and harm regimes.
6. Keep Package F unopened until Packages A–E and the final external contract
   are frozen.

## Verification

The full local suite passes 101 tests after the Package D protocol, analysis
gates, and phase-confirmation runner were added. Changed Python files also pass
their scoped Ruff checks. The working tree was clean at this status update.
