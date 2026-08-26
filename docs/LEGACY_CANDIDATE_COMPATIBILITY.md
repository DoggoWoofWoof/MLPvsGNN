# 2Wiki/MuSiQue candidate-contract compatibility note

The frozen 2Wiki and MuSiQue baseline artifacts were produced before optional
query-hop metadata was added to the candidate-contract digest. Their candidate
construction is unchanged: stable dense-top-200 then SPLADE-top-200 union.

The compatibility layer reproduces the exact legacy digest over query ID,
split byte, candidate IDs in stable order, and gold IDs. It may omit only the
later hop-metadata field. Any candidate addition, removal, reordering, query
reordering, split change, or gold-ID change fails the check.

Before either GPU job is submitted, a CPU-only Modal preflight must return
`BIT_EXACT_FROZEN_CANDIDATE_EQUIVALENCE`. The GPU runner reconstructs the proof
independently and requires the proof SHA-256 to match. Candidate generation,
graphs, features, labels, splits, model definitions, and hyperparameters are
not regenerated or changed.
