# M1: where the lean pair loses golds, by hop and by link distance

Declared 15 October 2026, before any M1 number exists. Stage 2 of docs/PROGRAM_2026_10_15.md. Code:
`scripts/m1_musique_hops.py`. Status: RUN 15 October 2026; results in section 6.

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

Run 15 October 2026 on the host CPU (`m1-*` items; tables `outputs/m1/m1.md`, numbers `outputs/m1/m1.json`). The check
held on every read: the CPU scores reproduce the filed top-5 gold counts on every question (agreement 1.0). Every
musique s1eval id resolved in train or dev, and every gold resolved to its sub-question (0 unresolved).

**Where the missed golds sit (J5, in-domain; shares of the missed golds, zspk; zrckp within 0.03):**

| dataset | golds missed | d1 | d2 | far | no_anchor | out |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| musique | 0.453 | 0.26 | 0.30 | 0.18 | 0.16 | 0.10 |
| hotpotqa | 0.096 | 0.50 | 0.00 | 0.02 | 0.23 | 0.25 |
| 2wiki | 0.072 | 0.62 | 0.07 | 0.05 | 0.01 | 0.25 |
| squad | 0.090 | 0 | 0 | 0 | 0.78 | 0.22 |
| webqsp (zero-shot) | 0.914 | 0.03 | 0.25 | 0.03 | 0.35 | 0.33 |

metaqa is left out of that reading. Its questions hold 7.4 golds on average, and the top five can hold at most five,
so most of its "misses" are golds the top five has no room for (R@5 0.80, ceiling 0.83).

**What it points to (section 4's readings, stated before the numbers):**
- **Message passing is where the in-domain misses are on every multi-hop passage dataset.**
  - Missed golds one or two links from a found gold: 0.56 of musique's misses, 0.50 of hotpotqa's, 0.69 of 2wiki's.
  - Today's GNN does not take them: zspk lifts about as many golds as the MLP as it drops.
    - 2wiki: zspk finds 473 golds zrckp misses (417 of them d1 for zrckp) and misses 362 that zrckp finds (318 d1),
      a net 111 of 30,654 golds.
    - hotpotqa: a net −31 of 14,810.
    - musique: a net +46 of 6,404.
  - Its one propagation step does not use the link that is there. That is G2's question (hidden states passed over
    three layers) and then E1's.
- **The pool is the second limit.**
  - 0.25 of the misses on hotpotqa and 2wiki are golds outside the pool (pools hold every gold for 0.959 of
    questions).
  - musique's 4-hop questions: only 0.637 of pools hold every gold, and 0.22 of the last sub-question's golds are out.
  - That is P1f's and P1f-b's job.
- **The first stage limits squad and the KB zero-shot reads.**
  - squad: no_anchor is 0.78 of its misses. Its one gold is simply not ranked, and its pools have no links.
  - webqsp zero-shot: 0.35 no_anchor and 0.33 out.
  - metaqa zero-shot from L-metaqa: no_anchor is 0.70 of the golds (0.74 of the misses). The models put no answer of the
    topic entity in the top five, the first relation-typed step the passage-trained models never saw.
- **The missed golds are near misses on hotpotqa and 2wiki, and deep on musique.**
  - hotpotqa and 2wiki: 54% and 71% of the missed in-pool golds sit at ranks 6–10.
  - musique: 40% sit beyond rank 50 of its ~2,090-node pools.

**musique by hop (J5, zspk; top-five share of each sub-question's golds):**

| hop count (questions) | sub-question 1 | 2 | 3 | 4 | the next miss is mostly |
| --- | ---: | ---: | ---: | ---: | --- |
| 2-hop (1,252) | 0.840 | 0.516 | | | d1 (0.238 of sub-question 2's golds) |
| 3-hop (760) | 0.811 | 0.392 | 0.404 | | d1 and d2 (0.207, 0.237), then far (0.197) |
| 4-hop (405) | 0.696 | 0.235 | 0.284 | 0.225 | d1/d2, then out (0.217 on sub-question 4) |

The first sub-question's paragraph is found (0.70–0.84). The chain breaks at the bridge: the second paragraph is
usually one link from the found first one and is not ranked. A message-passing step should lift exactly that one.
The 4-hop chains also run out of pool.

**Zero-shot musique (L-musique) leans the other way.** Trained on the other four datasets (all two-hop or single-hop),
both models lift the 2-hop bridge (sub-question 2: rrf 0.328, zrckp 0.415, zspk 0.416). They push the answer
paragraph of longer chains below rrf: 3-hop sub-question 3 falls from 0.363 to 0.270 and 0.267, and 4-hop
sub-question 4 from 0.212 to 0.161 and 0.151. musique's own training carve is also two-hop-heavy (s1fit 72% 2-hop,
22% 3-hop, 6% 4-hop, against s1eval's 52%, 31% and 17%). Step 2 (eval-mix weights) re-weighted the fit questions to
the eval mix and found no gain, so the mix alone is not the fix. The models learn "find the bridge" and not "follow
the chain".
