# P5c: the learned offsets as a fixed input to both models

Declared 17 October 2026, before any number. Code: `scripts/p5c_offset_cols.py` builds the columns, and
`outputs/mp_unified/zp5c.py` trains and reads the arms. Two arms, one per model. Both read the same four columns,
computed before either model runs. This is reopening #1 of [VERDICT_AUDIT_2026_10_17.md](VERDICT_AUDIT_2026_10_17.md).
Filed 19 October: both arms NOT_ADOPTED (section 7).

## 1. Why this, now

- **g found golds the models leave low.** P5b's g sets a target u_s + Δ(q, s) from the question, the seed and the
  seed's index-time graph features. P5b2 read it with a held-out stop
  ([P5B2_HELD_OUT_STOP.md](P5B2_HELD_OUT_STOP.md), section 7). Fused with H2, it found more of zsp's low golds on
  metaqa, hotpotqa and 2wiki. It tied on squad and found fewer on musique. On golds that no fit question had, at
  N = 10 it found about twice H2's share: 2wiki 0.316 against 0.156, metaqa 0.017 against 0.007. Those golds are two
  or more links from the seeds, where the rings cannot order them. P5b3's gated form found more on metaqa, hotpotqa,
  2wiki and webqsp (zero-shot), tied on squad and found fewer on musique.
- **Those were diagnostics.** g was read alone, fused with H2 by reciprocal rank. No ranking model read it. Each stage
  stopped on its rule's "fewer on none", which musique's −0.005 to −0.011 broke. Under the user's rule of 17 October
  (docs/SCREENS.md section 3b), a diagnostic closes no idea. An idea closes only when its signal has reached both
  models as an input through the screen. The audit reopened the offsets (its row 1) and declared this form.
- **Why a fixed input, and not g inside the model.** zof trained an offset end to end on zrc, and it harmed, so that
  form is closed. As a fixed input, each model's head weighs g's columns against what the model already reads, row by
  row. Its zero start means it can also leave them out where they mislead, as on musique.
- **What the MLP may read.** The MLP's scoring has no message passing. No row's score reaches another row, and no
  input is a score normalised over the question's rows (the rules of 8 and 14 October). P5c's columns are pointwise.
  Each row's four values come from that row's vector, the question and its seeds, before any row is scored.
- **Both models get it** (F1, docs/F1_ONE_INPUT_SET.md). Each arm is decided against its own model's card fit. No
  GNN number chooses whether the MLP keeps it.

## 2. The input

### g

g is P5b's g, unchanged in form ([P5B_LEARNED_OFFSETS.md](P5B_LEARNED_OFFSETS.md), section 2).
- **The space** is P5a's V. These are label-free node vectors, fit at index time to each corpus's graph. κ, the
  scale, is label-free too.
- **Per question q and seed s, g reads:**
  - the question in P5a's frame, x(q);
  - u_s;
  - x(q) − u_s;
  - m_s, the seed's mean displacement to its neighbours;
  - the seed's degree per link family;
  - 1/(60 + rank) of the seed in the dense and SPLADE lists.
- **What comes out:** for each of four heads h, an offset Δ_h(q, s) and a weight logit. g has 200,465 weights.
- **The seeds** are the dense top 5, then the SPLADE top 5.

### Refit per split, cross-fitted

- **Why refit.** P5b's filed g was fit on all five training datasets, musique and hotpotqa among them. So neither
  screen split could read it without g having trained on its held-out dataset.
- **What each split fits on.** P5c refits g per split, on that split's four training datasets' fit carves only:
  - L-musique: metaqa, squad, hotpotqa and 2wiki;
  - L-hotpotqa: metaqa, squad, musique and 2wiki.

  webqsp never fits.
- **Two folds.** fold(q) is sha256("dataset:question id") read as an integer, mod 2.
  - g_f fits on fold f's questions that have a counted gold, that is, a gold among the candidates.
  - It uses P5b2's held-out stop inside the fold: 20% held out, split seed 0. It keeps the epoch with the lowest loss
    on held-out new golds, at most 30. Epoch 0 is g's start.
  - Otherwise it is P5b's fit: Adam 1e-3, batches of 32, float32 with TF32 off, and EM over the latent (seed, head).
- **Every question q in fold f takes its columns from g_{1−f}.**
  - A training question gets its columns from the g that never saw it. So a model fits on columns of the same kind
    it reads on s1eval.
  - The s1eval questions and the held-out dataset's questions are fitted by neither g. They follow the same parity
    rule.
- **Every question is built, not P5b's sample.** P5b took 2,000 questions per dataset. P5c builds every question of
  each carve (`units`), on P1c2's KIND_T pools. On each training dataset's fit carve, every question in P5b's filed
  base must come out as P5b's:
  - its seeds, ranks, degrees, candidates and golds exactly;
  - x and m_s within 1e-5.

### The four columns

Let ℓ_{s,h}(v) = −κ e^β |u_s + Δ_h(q, s) − u_v|², which is P5b's score of row v from seed s and head h. Let π be the
softmax of g's weight logits over the question's (seed, head) pairs. With S the number of seeds:

    pt  = log Σ_{s,h} π_{s,h} exp ℓ_{s,h}(v)         P5b's mixture, pointwise
    mx  = max_{s,h} ℓ_{s,h}(v)                         the nearest target
    pt0 = log (1/S) Σ_s exp(−κ |u_s − u_v|²)           pt at g's start: no offset, equal weights, β = 0
    mx0 = max_s −κ |u_s − u_v|²                        mx at g's start: the nearest seed

- **pt0 and mx0 do not depend on g.** They let the head tell what g learned from what the space alone says.
- **Storage.** Each column is clipped to [−40, 40], divided by 10 and stored as float16. There is one file per P1c2
  cache part, per split.
- **Edge cases.** A seed row gets the same formulas, so its mx0 is 0. A question with no seed gets −4 in every column.
- **What is not a column.** P5b's own log p is its softmax over the candidates. That normalises over the question's
  rows, which the MLP may not read. It is computed only on each part's first two questions, as a check that g, its
  inputs and the distances are P5b's (within 2e-3).

**A worked example, with made-up numbers.** Take "Who is the spouse of the director of *Film X*?" Let *Film X*'s
page be the only seed, with one head, κ = 1 and e^β = 1.5.
- v₁ is a page about the film's music: |u_s − u_v₁|² = 0.8, and its distance² to g's target is 1.2.
- v₂ is the director's page: |u_s − u_v₂|² = 1.0, and its distance² to the target is 0.2.

| row | pt0 = mx0 | pt = mx | stored (pt, mx, pt0, mx0) |
| --- | ---: | ---: | --- |
| v₁ (the music) | −0.8 | −1.8 | −0.18, −0.18, −0.08, −0.08 |
| v₂ (the director) | −1.0 | −0.3 | −0.03, −0.03, −0.10, −0.10 |

The space alone puts v₁ nearer the seed. g's offset reverses that. The head sees both, before any score exists.

**The check, in one line.** Each row's columns come from that row's vector, the question and its seeds alone. The
self-test computes every row on its own and gets the same columns. No row's score is read, and nothing is normalised
over the question's rows.

## 3. The arms

| arm | model | carve | its base |
| --- | --- | --- | --- |
| zrckpo | zrckp plus the P5c head (`ZRCKPO`) | zrckp's (`KTChainCarveZRC`) with P5c's columns beside | `scr-p1c3p-zrckp`, `-hp` |
| zspko | zspk plus the P5c head, added with the KT head before zsp's propagation reads the score (`ZSPKO`) | zspk's (`KTLinkCarveBase`) with P5c's columns beside | `scr-p1c3-zspk`, `-hp` |

- **The head** is the same in both: Linear(4, 16), GELU, Linear(16, 1), over the row's own four columns, added to the
  score. That is 97 parameters. It has no score input, so it is pointwise in both models.
- Its last layer starts at zero, so each arm starts as its base bit for bit. Its first layer is drawn from its own
  generator (seed + 5701).
- **Each split's fits read that split's columns.** A fit of L-musique never sees a g that trained on musique, and a
  fit of L-hotpotqa never sees one that trained on hotpotqa.
- Everything else is the base's, unchanged: the carve, KT input, settings, seeds, batches, loss and loop. Neither arm
  adds a graph, an encoder, text or a message. The MLP's carve gets the four columns and no edge.

## 4. The fits and the call

- **Fits:** one card fit per arm on L-musique and one on L-hotpotqa (`scr-p5c-<arm>`, `-hp`). They run on P1c2's
  roots, P1c3's KT input and P5c's columns of that split.
- **Reads:** each fit on the six s1eval carves at p@swa.
- **Compares:** each fit against its base arm's card fit of the same split, which decides. Step 1's fit is reported
  beside. The re-call's base screens are L1's (`outputs/l1/base-<base arm>-<split>.json`).
- **Call:** relz's pair and its re-call under the seed null (the four null files of docs/SCREENS.md), per arm. GAIN,
  LOSS and WITHIN are docs/SCREENS.md's, on R@5.

### The rule, fixed before any number

Per arm:
- **ADOPT_FOR_FULL_RUN:** its re-call has no LOSS on any of its twelve reads and at least one GAIN;
- **NOT_ADOPTED:** otherwise;
- **INCOMPLETE:** its re-call is missing, is not P5c's re-call of that arm, or does not hold the twelve reads.

The verdict closes only this form (SCREENS.md section 3b), and it names the next form (section 5).

**Reported beside, deciding nothing:**
- webqsp (zero-shot in both splits) and metaqa (in-domain), per split and arm, with their calls and differences;
- musique's in-domain read (L-hotpotqa) for both arms side by side, the MLP's change and the GNN's. Claim 1's musique
  share depends on it;
- hit@1 and FC@5 per read, as the compares give them. Era 1's offset screen gained on R@1 while losing on R@5, so
  hit@1 is watched;
- each part's column quantiles and clipped share, and each g fit's kept epoch.

**Flagged.**
- **Graph aggregates.** g reads m_s and the degrees. These are index-time aggregations over a seed's graph neighbours,
  which P5b flagged. They are the same for both models and computed before any row is scored, like KIND_T's walks.
- **π.** π is a softmax over the question's seeds of g's own logits. It is not a softmax over rows' scores. The
  seeds' first-stage ranks enter as P5b's 1/(60 + rank) inputs, values both models already read through RRF.
- **Serving cost.** It is one g forward per (question, seed): 10 seeds, about 2 million multiply-adds. Then per row,
  40 distances in 64 dimensions: about 2,600 multiply-adds. Both models pay it. It is timed cold beside P4's timings
  before any latency claim.

## 5. What each outcome leads to (fixed now)

Per arm, the next form, each in its own file declared before its numbers:

| its re-call | the next form |
| --- | --- |
| ADOPT_FOR_FULL_RUN | the arm fitted again on P1g-b's pools (units and columns built on P1g-b's parts), then its full run |
| NOT_ADOPTED, with at least one GAIN and a LOSS | the back-off form: P5b3's gate, refit per split as g is, as a fifth column, so the head can drop g's columns on the questions where the gate says g misleads |
| NOT_ADOPTED, with no GAIN | five folds instead of two, so each g fits on four fifths of the questions (the cross-fit's halved data is the first suspect). For zspko alone, P5b's normalised log p as a fifth column: the GNN may read a value normalised over the question's rows, and the MLP may not |
| INCOMPLETE | the missing re-call is run; nothing else changes |

- **If both arms pass**, the input is the lean pair's, read by both.
- **If only one passes**, its full run is declared. The next file decides whether F1's one-input-set rule keeps the
  input.
- **What does not happen.** No model change is adopted on today's pools (the user's rule of 13 October).

## 6. How it runs

- **Self-tests.** Both pass before this commit:
  - `python scripts/p5c_offset_cols.py --selftest` checks the following:
    - folds are deterministic and balanced;
    - the four columns match a float64 reference built seed by seed, head by head and row by row;
    - each row computed alone gives the same columns;
    - at g's start, pt is pt0 and mx is mx0;
    - P5b's log p, the check, is `B.log_p`'s;
    - pools of seeds only work;
    - the stored transform.
  - The builder's self-test also runs end to end on a toy:
    - the P5b check of the units, and its refusal of a moved candidate;
    - four g fits that learn (the kept epoch is above 0);
    - both splits' columns, from the other fold's g, equal the reference;
    - the cross-fit folds differ;
    - refusals: a carve no split reads, and a cache row outside the final pool.
  - `python outputs/mp_unified/zp5c.py --selftest` checks the following:
    - the arms are zrckp's and zspk's models plus the head, and the base arms are unchanged;
    - each arm starts as its base bit for bit;
    - without the columns, each arm's toy fit equals its base's on every shared key;
    - train refuses another arm, split, fit name or cache root;
    - read refuses a fit of another split, or one that is not P5c's;
    - relz's pair and re-call per arm, stamped P5c's;
    - the verdict's outcomes;
    - the builder's settings are the ones the models check.
- **Development run.** `units musique s1eval --limit 20` ran on the laptop on P1c2's own pool file: 20 questions and
  42,570 candidates. Every check held, in 26 s at a peak of 0.9 GB, and it wrote nothing.
- **Host items** (`p5c-*`). They are placed after P1g-b's screen verdict in the feeder's order, so P1g-b's ready items
  go first. P5c's ready items fill the CPU and the card otherwise.
  - on the CPU: 11 `units`, one per carve the splits read, 2 CPUs each. Memory follows P5b's and P5b3's peaks, plus
    room;
  - on the card at share 0.12 (P5b's), 2 CPUs: four `fitg`, one per split and fold. Each runs after its split's four
    fit-carve units;
  - on the CPU: 11 `cols`, each after its carve's units and the fits of every split that reads it;
  - on the card: four trains, then four reads, at `--frac 0.30` and share 0.32 (L1's and R1's);
  - on the CPU: four compares, two pairs, two re-calls and the verdict.
- **Disk.**
  - units: about 0.5 GB, one candidate id per pool row;
  - columns: about 95 million row-splits × 4 columns × 2 bytes, about 0.8 GB;
  - g's four fits: under 4 MB.

  About 1.3 GB in all.
- **ETA.**
  - units: about 2 to 15 minutes each;
  - fitg: P5b2's two fits took 86 minutes together, on about 4,600 training questions each. A P5c fold holds about
    5,000, so each fit takes about 45 to 60 minutes, four at once, once R1's trains leave room on the card;
  - cols: about 5 to 20 minutes each;
  - trains and reads: about 2.5 to 3 hours, as R1's.

  The verdict is due about 16:30 on 17 October. It moves later if P1g-b's fits take the card first.
- **Fixed rules.** The encoder stays frozen. webqsp never trains. Test splits are never read. No LLM or transformer.

## 7. Results

Filed 19 October 2026 from the host's records (`outputs/p5c/screen-verdict.json`, written 2026-10-18T12:34:32Z; the
re-calls `outputs/screen/scr-p5c-zrckpo-pair-recall.json` and `-zspko-`, host paths made relative). **Both arms
NOT_ADOPTED:** neither has a GAIN; the GNN's arm has one LOSS.

- **zrckpo: 0 GAIN, 0 LOSS.** musique in domain (L-hotpotqa, Claim 1's read) rises +0.0062 [+0.0016, +0.0112], WITHIN;
  webqsp zero-shot on L-hotpotqa falls 0.0121 [-0.0219, -0.0016], filed LOSS and re-called WITHIN against its 0.0233
  floor.
- **zspko: 0 GAIN, 1 LOSS.** 2wiki in domain falls 0.0111 on L-hotpotqa [-0.0126, -0.0094].
- **Beside:** webqsp zero-shot -0.0084 and -0.0121 (MLP), -0.0066 and +0.0040 (GNN); metaqa in domain within 0.0015 on
  all four fits; musique in domain side by side: the MLP +0.0062, the GNN +0.0008.
- **The offsets:** the g fits kept epochs 2 to 4; on 2wiki the pt column sits at its clip on 0.18 to 0.19 of the fit
  carve's rows (0.09 to 0.10 of s1eval's) and mx on 0.14 to 0.16 (0.07); on every other carve pt clips on at most 0.11
  of the rows (metaqa) and mx on at most 0.085; pt0 and mx0 never clip (tables below).

The learned offsets, read as four fixed columns, gain nothing past the seed null on either model, and cost the GNN 2wiki
in domain on L-hotpotqa. This closes only this form (two folds; the four columns through a pointwise head). The next
form, fixed in section 5 for no GAIN: **five folds instead of two**, so each g fits on four fifths of the questions (the
cross-fit's halved data is the first suspect), on both models; for zspko alone, **P5b's normalised log p as a fifth
column** (the GNN may read a value normalised over the question's rows; the MLP may not). One declared file.

Each arm's twelve reads after the seed-null re-call (delta = the arm's R@5 minus its base's; a call is shown with its
filed call where the re-call changed it; base R@5 shown as zrckp's / zspk's):

| split | dataset | read | base R@5 | zrckpo: delta [95% CI], call | zspko: delta [95% CI], call |
| --- | --- | --- | --- | --- | --- |
| L-musique | metaqa | in-domain | 0.8019 / 0.8009 | +0.0007 [-0.0002, +0.0015], WITHIN | +0.0014 [+0.0006, +0.0023], WITHIN |
| L-musique | squad | in-domain | 0.9116 / 0.9114 | +0.0003 [-0.0015, +0.0021], WITHIN | +0.0015 [-0.0005, +0.0034], WITHIN |
| L-musique | musique | zero-shot | 0.5121 / 0.5012 | +0.0017 [-0.0027, +0.0062], WITHIN | -0.0013 [-0.0069, +0.0047], WITHIN |
| L-musique | hotpotqa | in-domain | 0.9065 / 0.9080 | -0.0001 [-0.0021, +0.0018], WITHIN | -0.0011 [-0.0034, +0.0009], WITHIN |
| L-musique | 2wiki | in-domain | 0.9347 / 0.9383 | +0.0006 [-0.0009, +0.0021], WITHIN | -0.0037 [-0.0050, -0.0022], WITHIN |
| L-musique | webqsp | zero-shot | 0.3627 / 0.3229 | -0.0084 [-0.0183, +0.0008], WITHIN | -0.0066 [-0.0156, +0.0021], WITHIN |
| L-hotpotqa | metaqa | in-domain | 0.8044 / 0.8040 | -0.0009 [-0.0017, -0.0002], WITHIN | +0.0004 [-0.0005, +0.0014], WITHIN |
| L-hotpotqa | squad | in-domain | 0.9127 / 0.9135 | -0.0005 [-0.0023, +0.0013], WITHIN | +0.0001 [-0.0019, +0.0021], WITHIN |
| L-hotpotqa | musique | in-domain | 0.5745 / 0.5754 | +0.0062 [+0.0016, +0.0112], WITHIN | +0.0008 [-0.0039, +0.0057], WITHIN |
| L-hotpotqa | hotpotqa | zero-shot | 0.8513 / 0.8499 | +0.0027 [+0.0001, +0.0056], WITHIN | +0.0020 [-0.0009, +0.0047], WITHIN |
| L-hotpotqa | 2wiki | in-domain | 0.9350 / 0.9333 | -0.0023 [-0.0039, -0.0007], WITHIN | -0.0111 [-0.0126, -0.0094], **LOSS** |
| L-hotpotqa | webqsp | zero-shot | 0.3475 / 0.2680 | -0.0121 [-0.0219, -0.0016], WITHIN (filed LOSS) | +0.0040 [-0.0039, +0.0121], WITHIN |

Beside, deciding nothing: each read's FC@5 and hit@1 deltas against the same base, from the compare screens
(`outputs/screen/scr-p5c-zrckpo.json` and the others, `-hp` for L-hotpotqa):

| split | dataset | zrckpo FC@5 | zrckpo hit@1 | zspko FC@5 | zspko hit@1 |
| --- | --- | ---: | ---: | ---: | ---: |
| L-musique | metaqa | +0.0004 | +0.0056 | -0.0002 | +0.0095 |
| L-musique | squad | +0.0003 | -0.0019 | +0.0015 | -0.0021 |
| L-musique | musique | +0.0037 | +0.0095 | +0.0054 | -0.0240 |
| L-musique | hotpotqa | +0.0020 | -0.0054 | -0.0035 | +0.0061 |
| L-musique | 2wiki | +0.0018 | -0.0027 | -0.0088 | -0.0037 |
| L-musique | webqsp | -0.0153 | +0.0013 | -0.0073 | -0.0093 |
| L-hotpotqa | metaqa | -0.0007 | -0.0012 | +0.0003 | -0.0006 |
| L-hotpotqa | squad | -0.0005 | +0.0002 | +0.0001 | +0.0001 |
| L-hotpotqa | musique | +0.0074 | -0.0041 | -0.0070 | +0.0029 |
| L-hotpotqa | hotpotqa | +0.0057 | -0.0105 | +0.0045 | +0.0128 |
| L-hotpotqa | 2wiki | -0.0064 | -0.0018 | -0.0351 | -0.0035 |
| L-hotpotqa | webqsp | -0.0120 | +0.0000 | +0.0033 | -0.0073 |

Each g fit (`outputs/p5c/g/<split>_f<fold>.json`):

| split | fold | held out | kept epoch | beta |
| --- | ---: | --- | ---: | ---: |
| L-musique | 0 | musique | 2 | 0.0237 |
| L-musique | 1 | musique | 3 | 0.0811 |
| L-hotpotqa | 0 | hotpotqa | 4 | 0.1876 |
| L-hotpotqa | 1 | hotpotqa | 2 | 0.0917 |

Each carve's columns (`outputs/p5c/cols/...`; clipped shares weighted by rows; pt and mx clip at -4 after the scale):

| split | dataset | carve | parts | rows | pt clipped | mx clipped | pt0, mx0 clipped | pt median (parts) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| L-musique | metaqa | fit | 8 | 12,021,002 | 0.0470 | 0.0287 | 0.0000 | -1.41 to -1.00 |
| L-musique | metaqa | s1eval | 13 | 19,742,463 | 0.0469 | 0.0277 | 0.0000 | -1.58 to -0.84 |
| L-musique | squad | fit | 1 | 292,994 | 0.0080 | 0.0064 | 0.0000 | -0.90 |
| L-musique | squad | s1eval | 1 | 594,063 | 0.0098 | 0.0072 | 0.0000 | -0.87 |
| L-musique | musique | s1eval | 3 | 5,057,436 | 0.0105 | 0.0056 | 0.0000 | -1.27 to -0.94 |
| L-musique | hotpotqa | fit | 1 | 557,195 | 0.0043 | 0.0032 | 0.0000 | -1.09 |
| L-musique | hotpotqa | s1eval | 1 | 693,828 | 0.0057 | 0.0034 | 0.0000 | -1.08 |
| L-musique | 2wiki | fit | 1 | 628,354 | 0.1765 | 0.1362 | 0.0000 | -1.91 |
| L-musique | 2wiki | s1eval | 2 | 1,328,715 | 0.0900 | 0.0659 | 0.0000 | -1.61 to -1.61 |
| L-musique | webqsp | s1eval | 2 | 3,186,165 | 0.0107 | 0.0077 | 0.0000 | -0.83 to -0.80 |
| L-hotpotqa | metaqa | fit | 8 | 12,021,002 | 0.1071 | 0.0848 | 0.0000 | -1.87 to -1.26 |
| L-hotpotqa | metaqa | s1eval | 13 | 19,742,463 | 0.1062 | 0.0780 | 0.0000 | -2.54 to -1.10 |
| L-hotpotqa | squad | fit | 1 | 292,994 | 0.0131 | 0.0088 | 0.0000 | -1.12 |
| L-hotpotqa | squad | s1eval | 1 | 594,063 | 0.0204 | 0.0149 | 0.0000 | -1.13 |
| L-hotpotqa | musique | s1eval | 3 | 5,057,436 | 0.0016 | 0.0005 | 0.0000 | -1.22 to -1.10 |
| L-hotpotqa | musique | s1fit | 6 | 9,623,206 | 0.0038 | 0.0025 | 0.0000 | -1.25 to -1.02 |
| L-hotpotqa | hotpotqa | s1eval | 1 | 693,828 | 0.0112 | 0.0067 | 0.0000 | -1.08 |
| L-hotpotqa | 2wiki | fit | 1 | 628,354 | 0.1891 | 0.1560 | 0.0000 | -1.85 |
| L-hotpotqa | 2wiki | s1eval | 2 | 1,328,715 | 0.0950 | 0.0737 | 0.0000 | -1.53 to -1.50 |
| L-hotpotqa | webqsp | s1eval | 2 | 3,186,165 | 0.0171 | 0.0119 | 0.0000 | -1.02 to -0.97 |
