# A6e: the link words as edge weights in the GNN's step, and on wider anchors in the MLP

Declared 18 October 2026, before any number. The next form A6's verdict named for both arms
(docs/A6_LINK_WORDS_INPUT.md, sections 5 and 7), under docs/SCREENS.md section 3b and docs/VERDICT_AUDIT_2026_10_17.md.
Code: `outputs/mp_unified/za6e.py`. Filed 19 October: both arms NOT_ADOPTED (section 7).

## 1. Why this, now

**D4** (13 October) scored a link by its words: the largest summed IDF of the question's stems, less both ends' title
stems, in one window of the links between two rows (`linkwords.link_score`). On musique the score ranked the neighbour a
question follows only a little above chance (le2 0.261 against 0.225).

**A6** (17 October) gave that score to both lean models as six columns, each row's best link to the first stage's top
five rows, through R1's pointwise head. Both arms were NOT_ADOPTED with no GAIN. Every loss was on webqsp zero-shot
(MLP −0.019 and −0.026, GNN −0.011), where the structural words are the relation names in an edge's slots.

That closes only A6's form. Two things were left untried, and A6's sections 5 and 7 fixed both as the next form:
- **The GNN never read the score where it acts, on the edges.** In A6, zspk saw the link words only as columns on its
  row. Its propagation still weighted every link of a family the same.
- **The MLP's anchors were narrow, and its structural columns mixed two readings.** Five anchors reach few of a large
  pool's rows. A KB's relation names and a passage graph's title windows shared one set of weights, which is where
  webqsp zero-shot lost.

## 2. The form

One build gives both inputs. It runs per step-1 cache part over P1c2's look: pools, rrf, and the raw edges with their
relation slots.

**The pair score.** A worded pair is a family (structural or NER) and two pool rows joined by at least one raw edge of
that family, with no self-loop. Each pair is scored once with D4's link score, over all its raw edges (on a typed
graph, every relation slot of every raw edge). The kNN family has no words.

### The MLP (zrckpn)

Eight columns per row, from the first stage's top **ten** rows by rrf (A6 used five). For each family the row takes its
best pair with an anchor other than itself:

| column | what it holds |
| --- | --- |
| log1p_w_title | log1p of the best structural score, on a graph without typed relations (title windows) |
| log1p_w_rel | log1p of the best structural score, on a typed graph (relation names) |
| log1p_w_ner | log1p of the best NER score (shared entities) |
| share_title, share_rel, share_ner | the same scores over the question's total stem IDF |
| has_struct, has_ner | 1 when the row has a window with an anchor in the family |

**The split.** On a typed graph the title columns are 0; on the others the relation columns are 0. So the KB's words
cannot move the weights the passage graphs read. The graph's kind is its own property: it has relation slots exactly
when its relations are typed.

The columns enter through R1's pointwise head: Linear(8, 16), GELU, Linear(16, 1). That is 161 parameters, drawn from
seed + 6601, with the last layer zero at the start. The head's output adds to zrckp's score.

**The MLP rule.** Each row reads only its own columns. The anchors come from rrf, the first stage's fused rank, before
any row is scored. No model score enters a column, and nothing is normalised over the question's scores.

### The GNN (zspkl)

Each pool edge zlink stores (P1c2's links: undirected, both directions, unique, no self-loops) takes its pair's
weight:
- **weight:** log1p of the pair's score;
- **kNN edges:** weight 0;
- **symmetry:** the weight is the same in both directions.

zsp's step reads ten inputs per row: per edge family, the mean and soft-max of its neighbours' z-scores, and log1p of
its degree, plus its own z. zspkl adds six:
- per word family (title windows, relation names, shared entities), the **link-weighted mean** of the row's
  neighbours' z-scores (zsp's z: the detached score's z-score in its pool, clipped to ±8);
- per word family, **log1p of the summed weight**;
- 0 where the row has no worded neighbour of the family.

The six enter the first layer of zsp's head through their own columns, `ew_w1` (32 × 6 = 192 parameters, **zero at
the start**), beside the ten inputs' columns. The head's hidden units and output layer are zsp's own.

So the step reads a message along each worded link in proportion to its words, and it can learn to leave them. At the
start zspkl is zspk bit for bit, scores and every shared gradient. The new columns get a gradient from the first step
on.

**Message passing.** zspkl's step passes messages, as zsp's does. The MLP passes none.

### Both

- **No parameter, label or model score enters a column or a weight.** The mentions are A6's (`outputs/a6/ents`).
- **Edge for edge with zlink.** The build recomputes `zlink.chunk_edges` per chunk and stops unless the edges equal the
  stored part's, array for array. Each weights part records the zlink record it was built on, and the carve refuses
  any other.
- **Serving cost.** The pair scores come from the pool's own text. The cost is a query-local compile, timed in its own
  file if an arm is adopted.

## 3. The arms

| arm | model | carve |
| --- | --- | --- |
| zrckpn | zrckp's model plus the head over the eight columns (MRO: ZRCKPN, ZRCKP, ...) | zrckp's (`zp1c3.KTChainCarveZRC`) with the columns beside it |
| zspkl | zspk's model with the six weighted inputs in zsp's step (MRO: ZSPKL, ZSPK, ZProp, KTMix, ZRM, ...) | zspk's (`zp1c3.KTLinkCarveBase`) with the edge weights beside its edges |

## 4. The fits and the call

A6's, unchanged:
- **Fits:** one card fit per arm on L-musique and one on L-hotpotqa (`scr-a6e-<arm>`, `-hp`), on P1c2's roots with
  P1c3's KT input, at the card fits' settings (variant p, config 2e-3:1e-4:0.1:8:2, seed 0, hidden 128).
- **Reads:** each fit on the six s1eval carves at p@swa.
- **Compares:** each fit against its base arm's card fit of the same split (zrckp: scr-p1c3p-zrckp and -hp; zspk:
  scr-p1c3-zspk and -hp), with step 1's fit beside. The re-call's base screens are L1's.
- **Call:** relz's pair and its re-call under the seed null (the four null files of docs/SCREENS.md), per arm.

### The rule, fixed before any number

Per arm:
- **ADOPT_FOR_FULL_RUN:** its re-call has no LOSS on any of its twelve reads and at least one GAIN;
- **NOT_ADOPTED:** otherwise;
- **INCOMPLETE:** its re-call is missing, is not A6e's re-call of that arm, or does not hold the twelve reads.

The verdict closes only this form (SCREENS.md section 3b), and it names the next form (section 5).

**Reported beside, deciding nothing:**
- musique (D4's target), webqsp (A6's loss) and hotpotqa;
- musique in domain (Claim 1's read), both arms side by side;
- hit@1 and FC@5 per read;
- the build's counts: worded pairs, the share with a word, and the share of rows with a window.

When the results are filed, A6's deltas are set beside each read.

## 5. What each outcome leads to (fixed now)

| its re-call | the next form |
| --- | --- |
| ADOPT_FOR_FULL_RUN | the arm fitted again on P1g-b's pools (columns and weights built again on P1g-b's looks), then its full run |
| NOT_ADOPTED, with at least one GAIN and a LOSS | A6ep, the passage-graph back-off: the same arms with the typed graphs' relation-name words left out (relation columns and weights 0), so only title windows and shared entities move either model; on both models |
| NOT_ADOPTED, with no GAIN | A6ex, the link words on the adopted arms (G5's zspx with the edge weights, R1's zrckpr with the eight columns), on both models; the link-word idea closes as a model input only if A6ex finds no GAIN either |
| INCOMPLETE | the missing re-call is run; nothing else changes |

No model change is adopted on today's pools (the user's rule of 13 October): an adopt is refitted on P1g-b's pools
before its full run.

## 6. How it runs

**Self-test.** `python outputs/mp_unified/za6e.py --selftest` passes before this commit (11.2 s). On rmatch's typed toy
look, with pools, an rrf column and A6's toy corpus and mentions, read once as a passage graph and once as a KB, it
checks:
- **the arms:** the MROs; the base arms unchanged; 161 and 192 added parameters;
- **the build against brute force:** the eight columns against every (row, anchor, family), and each zlink edge's
  weight against its pair's score (within float16's error), on both readings:
  - the split fills only its reading's columns;
  - the KB reading moves only the structural values, the NER values unchanged;
  - kNN edges weigh 0;
  - ten anchors reach rows that five do not;
  - the nodes read are exactly the worded edges' ends;
- **building one part at a time** gives the whole build's arrays byte for byte;
- **the build's refusals:** a missing question, a slot past the names, a part from another look, a part out of range,
  zlink edges built on another part, and zlink edges that are not the look's;
- **a batch** holds its rows' own columns, and its edges' own weights in zlink's order with their word families. The
  carve refuses weights built on other zlink edges;
- **the models:**
  - from the same seed, each arm is its base's state plus its added parameters, with no draw moved;
  - at the start each arm is its base bit for bit, in eval and in training: scores and every shared gradient (zsp's
    output layer set nonzero, so the step acts);
  - the six weighted inputs equal a loop over the edges;
  - with the added parameters set, the head (MLP) or the weighted columns (GNN) move the score by exactly their term;
  - without its side key each arm is its base;
- **toy fits** through lean_gpu's loop repeat bit for bit, and the added parameters learn. On the carve without the
  side key every state is the base fit's bit for bit;
- the train and read refusals; relz's pair and re-call per arm, stamped A6e's; the verdict's outcomes and next forms.

**Host items** (56, `a6e-*`), right after A5c's:
- **Builds:** 39, one per step-1 cache part, on A6's merged mentions. Memory is A6's plus the stored and rebuilt edges
  and the weights: metaqa and squad 4 GB, musique 6, webqsp 7, hotpotqa and 2wiki 12. The cost is about 3.6 CPU-hours
  (every worded pool pair is scored once, at A6's measured rate), and about 2 GB on disk under `outputs/a6e`, which
  stays off git.
- **The screen:** 17 items. Trains and reads use A6's `--frac` (L-musique 0.20, L-hotpotqa and reads 0.27); zspkl gets
  0.03 more for its per-edge tensors. Memory: zrckpn as A6's MLP arm; zspkl trains 5.5 GB and 8.5 GB (L-hotpotqa) and
  reads 7.5 GB, for the host-side weights.

## 7. Results

Filed 19 October 2026 from the host's records (`outputs/a6e/screen-verdict.json`, written 2026-10-18T21:16:55Z; the
re-calls `outputs/screen/scr-a6e-zrckpn-pair-recall.json` and `-zspkl-`, host paths made relative). **Both arms
NOT_ADOPTED:** no GAIN and no LOSS after the seed-null re-call.

- **zrckpn: 0 GAIN, 0 LOSS.** hotpotqa zero-shot rises +0.0086 [+0.0055, +0.0116] on L-hotpotqa: filed GAIN, re-called
  WITHIN against its 0.02 floor. hotpotqa in domain on L-musique (+0.0054) and musique in domain on L-hotpotqa (+0.0057)
  rise too, WITHIN.
- **zspkl: 0 GAIN, 0 LOSS.** hotpotqa zero-shot rises +0.0077 [+0.0046, +0.0109] (filed GAIN, re-called WITHIN); musique
  zero-shot falls 0.0094 on L-musique (filed LOSS, re-called WITHIN against its 0.072 floor); musique and 2wiki in
  domain on L-hotpotqa rise +0.0071 and +0.0070, WITHIN.
- **Beside:** webqsp zero-shot, where all of A6's losses were, moves -0.0026 and -0.0067 (MLP), -0.0079 and +0.0078
  (GNN), all WITHIN; A6 had -0.0188 and -0.0264 (MLP), -0.0107 and -0.0021 (GNN). A6's webqsp losses are gone in this
  form, where the KB's relation words have weights of their own. musique in domain (L-hotpotqa, Claim 1's read): +0.0057
  (MLP) and +0.0071 (GNN).
- **The builds:** on squad, hotpotqa and 2wiki 0.41 to 0.53 of the linked pairs carry a word (musique 0.28), and about a
  fifth to a quarter of the rows have a structural window (a tenth on musique); on the KBs fewer pairs carry a word
  (metaqa 0.15 of the linked pairs, webqsp 0.025) and a relation word sits on 0.003 of the rows (table below).

The link words lift hotpotqa a little on both models (up on all four fits), within the floors. This closes only this
form (the link score as an edge weight in zspk's step; eight split columns to the top ten rows on zrckp). The next form,
fixed in section 5 and recorded in the verdict, is **A6ex**: the link words on the adopted arms (G5's zspx with the edge
weights, R1's zrckpr with the eight columns), on both models, in its own declared file. The link-word idea closes as a
model input only if A6ex finds no GAIN either.

Each arm's twelve reads after the seed-null re-call (delta = the arm's R@5 minus its base's; a call is shown with its
filed call where the re-call changed it; base R@5 shown as zrckp's / zspk's):

| split | dataset | read | base R@5 | zrckpn: delta [95% CI], call | zspkl: delta [95% CI], call |
| --- | --- | --- | --- | --- | --- |
| L-musique | metaqa | in-domain | 0.8019 / 0.8009 | -0.0006 [-0.0015, +0.0003], WITHIN | +0.0011 [+0.0002, +0.0020], WITHIN |
| L-musique | squad | in-domain | 0.9116 / 0.9114 | +0.0028 [+0.0009, +0.0045], WITHIN | -0.0003 [-0.0025, +0.0019], WITHIN |
| L-musique | musique | zero-shot | 0.5121 / 0.5012 | -0.0034 [-0.0085, +0.0018], WITHIN | -0.0094 [-0.0149, -0.0042], WITHIN (filed LOSS) |
| L-musique | hotpotqa | in-domain | 0.9065 / 0.9080 | +0.0054 [+0.0033, +0.0076], WITHIN | +0.0013 [-0.0009, +0.0036], WITHIN |
| L-musique | 2wiki | in-domain | 0.9347 / 0.9383 | +0.0026 [+0.0011, +0.0042], WITHIN | -0.0052 [-0.0067, -0.0035], WITHIN |
| L-musique | webqsp | zero-shot | 0.3627 / 0.3229 | -0.0026 [-0.0117, +0.0069], WITHIN | -0.0079 [-0.0167, +0.0012], WITHIN |
| L-hotpotqa | metaqa | in-domain | 0.8044 / 0.8040 | -0.0011 [-0.0019, -0.0003], WITHIN | +0.0011 [+0.0001, +0.0022], WITHIN |
| L-hotpotqa | squad | in-domain | 0.9127 / 0.9135 | +0.0013 [-0.0007, +0.0031], WITHIN | +0.0001 [-0.0019, +0.0020], WITHIN |
| L-hotpotqa | musique | in-domain | 0.5745 / 0.5754 | +0.0057 [+0.0009, +0.0108], WITHIN | +0.0071 [+0.0014, +0.0123], WITHIN |
| L-hotpotqa | hotpotqa | zero-shot | 0.8513 / 0.8499 | +0.0086 [+0.0055, +0.0116], WITHIN (filed GAIN) | +0.0077 [+0.0046, +0.0109], WITHIN (filed GAIN) |
| L-hotpotqa | 2wiki | in-domain | 0.9350 / 0.9333 | +0.0020 [+0.0006, +0.0035], WITHIN | +0.0070 [+0.0054, +0.0086], WITHIN |
| L-hotpotqa | webqsp | zero-shot | 0.3475 / 0.2680 | -0.0067 [-0.0158, +0.0028], WITHIN | +0.0078 [-0.0028, +0.0186], WITHIN |

Beside, deciding nothing: each read's FC@5 and hit@1 deltas against the same base, from the compare screens
(`outputs/screen/scr-a6e-zrckpn.json` and the others, `-hp` for L-hotpotqa); and zrckpw's R@5 delta (A6, its own
re-call) and zspkw's R@5 delta (A6, its own re-call), on the same read:

| split | dataset | zrckpn FC@5 | zrckpn hit@1 | zspkl FC@5 | zspkl hit@1 | zrckpw R@5 (A6) | zspkw R@5 (A6) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| L-musique | metaqa | +0.0003 | -0.0054 | +0.0004 | +0.0114 | +0.0001 | +0.0017 |
| L-musique | squad | +0.0028 | +0.0029 | -0.0003 | +0.0011 | +0.0002 | +0.0012 |
| L-musique | musique | -0.0037 | +0.0008 | +0.0004 | -0.0360 | -0.0011 | -0.0031 |
| L-musique | hotpotqa | +0.0104 | -0.0023 | +0.0007 | +0.0101 | +0.0025 | +0.0031 |
| L-musique | 2wiki | +0.0077 | -0.0118 | -0.0150 | -0.0119 | +0.0011 | -0.0020 |
| L-musique | webqsp | -0.0086 | +0.0007 | -0.0053 | -0.0033 | -0.0188 LOSS | -0.0107 LOSS |
| L-hotpotqa | metaqa | -0.0008 | -0.0039 | +0.0009 | +0.0046 | -0.0022 | +0.0003 |
| L-hotpotqa | squad | +0.0013 | +0.0024 | +0.0001 | -0.0001 | +0.0007 | -0.0008 |
| L-hotpotqa | musique | +0.0083 | -0.0037 | +0.0074 | -0.0091 | +0.0006 | -0.0006 |
| L-hotpotqa | hotpotqa | +0.0161 | +0.0046 | +0.0162 | +0.0003 | +0.0024 | +0.0046 |
| L-hotpotqa | 2wiki | +0.0070 | -0.0107 | +0.0211 | -0.0010 | +0.0018 | -0.0058 |
| L-hotpotqa | webqsp | -0.0093 | -0.0053 | +0.0040 | -0.0007 | -0.0264 LOSS | -0.0021 |

The builds (each carve's parts summed; a linked pair is two rows joined by a raw edge of a worded family, 'with an
anchor' those with one of the first stage's top ten rows at an end; the row shares weighted by rows):

| dataset | carve | rows | linked pairs | with an anchor | share with a word | rows with a structural window | rows with an NER window | rows with a title word | rows with a relation word | rows with an NER word |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| metaqa | fit | 12,021,002 | 37,416,754 | 189,953 | 0.145 | 0.013 | 0.001 | 0.000 | 0.003 | 0.000 |
| metaqa | s1eval | 19,742,463 | 61,137,136 | 310,439 | 0.150 | 0.013 | 0.001 | 0.000 | 0.003 | 0.000 |
| squad | fit | 292,994 | 1,207,549 | 544,233 | 0.410 | 0.261 | 0.278 | 0.196 | 0.000 | 0.180 |
| squad | s1eval | 594,063 | 2,167,829 | 986,360 | 0.424 | 0.253 | 0.254 | 0.179 | 0.000 | 0.175 |
| musique | s1eval | 5,057,436 | 77,090,387 | 1,230,424 | 0.275 | 0.100 | 0.047 | 0.044 | 0.000 | 0.027 |
| musique | s1fit | 9,623,206 | 149,845,441 | 2,313,487 | 0.279 | 0.106 | 0.042 | 0.044 | 0.000 | 0.023 |
| hotpotqa | fit | 557,195 | 1,364,550 | 435,255 | 0.529 | 0.275 | 0.190 | 0.153 | 0.000 | 0.146 |
| hotpotqa | s1eval | 693,828 | 1,653,299 | 521,946 | 0.496 | 0.270 | 0.183 | 0.143 | 0.000 | 0.136 |
| 2wiki | fit | 628,354 | 1,351,430 | 328,753 | 0.488 | 0.190 | 0.181 | 0.096 | 0.000 | 0.117 |
| 2wiki | s1eval | 1,328,715 | 3,123,312 | 828,310 | 0.434 | 0.206 | 0.209 | 0.092 | 0.000 | 0.130 |
| webqsp | s1eval | 3,186,165 | 7,445,358 | 382,505 | 0.025 | 0.104 | 0.006 | 0.000 | 0.003 | 0.000 |
