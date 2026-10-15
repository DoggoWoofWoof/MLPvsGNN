# M1: where the lean pair loses golds, by hop and by link distance

Declared 15 October 2026, before any M1 number exists. Stage 2 of docs/PROGRAM_2026_10_15.md. Code:
`scripts/m1_musique_hops.py`. Status: declared; results in section 6.

## 1. Question

musique is the dataset where both lean models sit lowest (R@5 0.573 MLP, 0.580 GNN, J5 in-domain) and where the GNN's
share of the gain is not shown (P3c). docs/MUSIQUE_DIAGNOSIS.md split musique by hop count on the pools and models of
5 October. Its table predates KIND_T's pools, the KT input and the lean pair. M1 asks the same question of today's
models, and a sharper one: **for every gold the models miss, why is it missed?** It does this on all six datasets, so
the answer says where to work next (pools, message passing, or links) and whether that place is the same everywhere.

## 2. Inputs (read only)

- The filed fits of P3c (docs/P3C_CLAIM1_INTERVALS.md), candidate p@swa:
  - J5 (in-domain on five datasets; webqsp zero-shot): `outputs/full_zrckp/fits/J5`, `outputs/full_zspk/fits/J5`;
  - L-musique (musique zero-shot): `outputs/screen/fits/scr-p1c3p-zrckp`, `outputs/screen/fits/scr-p1c3-zspk`;
  - L-metaqa (metaqa and webqsp zero-shot, the KB transfer gap): `outputs/full_zrckp/fits/L-metaqa`,
    `outputs/full_zspk/fits/L-metaqa`.
- P1c2's s1eval caches with P1c3's KT input and zlink's pool edges (`outputs/p1c2/cache`, `outputs/p1c3/kt`,
  `outputs/p1c2/links`), as each fit's reads used them.
- musique's query records (`queries/train.jsonl` and `queries/dev.jsonl` of the frozen package, on the host mirror):
  the `question_decomposition` names each sub-question's supporting paragraph, so each gold passage gets its hop. The
  test split is never opened. An s1eval id that is in neither file stops the script.

## 3. What is computed

**Scoring.** Each fit's p@swa model scores every s1eval row on the host CPU (float32, lean_gpu's flags), with rrf's
base score beside. **Check:** per question, the top-5 gold count from these scores must match the filed read's
(`reads/{ds}__s1eval.npz`) on at least 99% of questions, and the mean R@5 must be within 0.002 of the filed one;
otherwise the script stops and nothing is tabled. On musique each row's node is checked against the package: a row is
gold exactly when its node is one of the question's gold nodes.

**Ranking** follows lean_gpu.top_hit: descending score, NaN last, ties by pool position.

**Each gold, for each model (rrf, zrckp, zspk), gets one class:**

| class | meaning | what would fix it |
| --- | --- | --- |
| `top5` | ranked in the top five | nothing |
| `d1` | in the pool, missed, one pool edge from a gold the model found | ranking along links (message passing) |
| `d2` | in the pool, missed, two edges from a found gold | deeper message passing |
| `far` | in the pool, missed, three or more edges from every found gold, or unlinked | the score itself (features) |
| `no_anchor` | in the pool, missed, and the model found no gold for this question | the first hop |
| `out` | not in the pool | the pool |

Distances use zlink's pool edges (all three families, both directions). For `d1` golds, the family of the link is
reported (structural, named-entity, kNN; a gold can count under more than one). Beside: the missed in-pool golds' rank
(6-10, 11-20, 21-50, beyond 50), and the golds one model finds that the other misses, by the other's class.

**Buckets:** every dataset as a whole; musique also by hop count (2, 3, 4, from the id prefix) and by the hop of each
gold (sub-question 1 to 4; the last sub-question's paragraph holds the answer). The hop mix of musique's training
carve (s1fit) is reported beside the s1eval mix.

## 4. What it decides

Nothing is adopted or refused. M1 is a diagnosis; it trains nothing and changes no filed record. What it points to,
stated before the numbers:
- **`out` dominates** a bucket: the pool is the limit there; P1f (one pool-size rule) and the walk's reach are where
  to work.
- **`d1` and `d2` dominate:** the gold is next to a found gold and the model does not lift it; that is message
  passing's job (G2, then E1's latent paths).
- **`far` dominates:** the gold is in the pool but nothing links it to what was found; features or links, not message
  passing.
- **`no_anchor` dominates:** the first hop fails, and everything after it; the first stage's limit (D4, B1's 13.3
  points below NV-Embed-v2).

The same reading is made on every dataset, so a fix chosen from it is universal.

## 5. Running it

- **Selftest** (`python scripts/m1_musique_hops.py --selftest`, passed before this commit): the ranking against
  lean_gpu.top_hit on random scores with ties, NaN and signed zeros; the classes on a hand-built pool graph; the hop
  of each gold on a hand-built decomposition; the test split refused; the score check refusing a mismatch.
- **Host items** (`m1-*`): six scoring items on the CPU (J5 on six datasets per arm, L-musique on musique, L-metaqa on
  metaqa and webqsp; four threads each), then one table item. No GPU.
- **Disk:** per-row scores and the edges once per dataset, about 1 GB at most (`outputs/m1/scores`).
- **ETA:** about an hour for the scoring (zspk's forward on the CPU), minutes for the table.

## 6. Results

Not run yet.
