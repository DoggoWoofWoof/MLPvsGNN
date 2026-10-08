# Full run: zlk, a row's link to the pool's leading rows on zrm, only if its screen's re-call is PROMISING and zrm is the base

Declared 8 October 2026 at about 18:15, before any number of round twenty-two exists (docs/SCREENS.md, twenty-second
round). zrm is the base (docs/BASE_ZRM_ZRS.md, re-graded ADOPT at 10:05). If the screen passes, this run decides whether
zlk's gain over zrm holds on every split, and whether anything else loses. These are development numbers. The paper's
numbers come from one declared confirmation run.

## 1. Question

zlk is zrm plus a small head added to its score. The head reads, for each row:

- its edges from zrm's top five rows and from zrm's top row, per edge family;
- its two-step paths from the top five;
- its top-five flag;
- the z-score of zrm's score in its pool.

The edges are each question's own pool graph, read undirected. Does a row's link to the rows zrm ranks highly lift
zrm's reads on every split? And does nothing else lose? The splits are J5 (in-domain on all five training datasets)
and each leave-one-out fit on its held-out dataset.

Why this question (docs/DIAG_BRIDGE.md):
- **The missed golds are linked to the found ones.** In partly-found questions, the golds zrm misses sit next to a
  found gold: 2wiki 0.77, hotpotqa 0.92 to 0.93, musique 0.38 (0.77 within two hops). That is 6 to 14 times the rate
  of non-gold rows. The reach bound is +0.06 to +0.12 R@5 on the multi-hop passage reads.
- **No fixed re-rank uses it.** Even anchored on a known gold, filling ranks 4 and 5 with the anchor's best-scored
  neighbours loses R@5 on musique and 2wiki. So the link must be weighed inside the score, not swapped in.
- **The per-row inputs hold no more.** Boosted trees on zrm's inputs plus zrm's own score add at most +0.006 R@5 in
  domain (FEATURE_LIMIT). What they lack is relational: every input is computed before any row is scored, and the graph
  inputs tie a row only to the retrieval's fixed seeds.
- **The KB answer sets are not chains.** On metaqa only 0.075 of missed golds sit next to a found one, with a lift of
  1.4. The head may learn little there, and it is graded there all the same.
- **The change is general.** The same edges rule, the same leaders rule and the same head on all six datasets. The
  leaders come from zrm's own scores, so the inputs are query-local and label-free.

## 2. Fits

- **What trains.** The arm zlk, (zlink.ZLink, zlink.LinkCarveBase over rmatch.ChainCarveBase), through
  `outputs/mp_unified/zlink.py train`:
  - zrm's model plus the link head: `lk_w1`, `lk_b1`, `lk_w2` and `lk_b2`, 353 new weights;
  - zrm's settings and training (rmatch.py's train), over rmatch's builds and zlink's edge builds;
  - seed 0, as zrm's fits.
- **The screen's fits are this run's L-musique and L-hotpotqa fits** (`outputs/screen/fits/scr-zlk` and
  `scr-zlk-hp`).
- The run trains four fits, in the order L-2wiki, L-squad, J5, L-metaqa: `zlink.py train --split S --name S --arm zlk
  --out-root outputs/full_zlk/fits`.
- **The head trains on every graph.** No split keeps it at zero, so no split is zrm's bit for bit. The identity at the
  start is checked by zlink.py's selftest and by `zlk-check-hotpotqa` and `zlk-check-webqsp` on real carves.
- **On the card:**
  - each fit 0.30 (share 0.32) and 11 GB, L-metaqa's 0.20 (share 0.22);
  - each read 0.30 (share 0.32) and 8 GB.

## 3. Reads and comparison

- Each fit is read on the six s1eval carves, with rmatch's chain entries and zlink's edges (`zlink.py read --name S
  --out-root outputs/full_zlk/fits`).
- **Each read is compared, question by question, with zrm's fit of the same split,** p@swa, on the same carves. That
  comparison decides. zret's fit and step 1's are reported beside it.
  - zrm's fits: `outputs/full_zrm/fits/S`; for L-musique and L-hotpotqa, zrm's screen fits `scr-zrm` and `scr-zrm-hp`.
  - zret's fits: `outputs/full_zret/fits/S`; for L-musique, zret's screen fit `scr-zret`.
  - The comparison is lean_screen's: the R@5 difference with a 2,000-resample question bootstrap, called GAIN, LOSS or
    WITHIN with the floor 0.0075.
  - hit@1 is reported beside R@5.
  - Twenty-four reads come from these fits and twelve from the screen: 36 in all.
- **Primary reads** are step 1's eleven: J5 on all six datasets (five in-domain, webqsp zero-shot), and each leave-out
  fit on its held-out dataset.

## 4. Verdict

- `zlink.py grade --full-root outputs/full_zlk --reuse
  L-musique=outputs/screen/scr-zlk.json,L-hotpotqa=outputs/screen/scr-zlk-hp.json` files the grade: relz.py's grade,
  decided against zrm's fits, under zlk's name (zrc.py's mapping).
- **The re-grade decides.** `nullx.py regrade` re-grades all 36 reads under the seed null over every split into
  `outputs/full_zlk/grade-nullx.{md,json}`. Each read's base R@5 is zrm's (`--base-compares`: zrm's six fits compared
  with step 1's, `outputs/zrc/base-zrm-S.json`).
- **ADOPT:** at least one primary read GAINs, and none of the 36 reads LOSEs.
- **NOT_ADOPTED:** otherwise.
- **INCOMPLETE:** a comparison is missing or is not the declared one (exit 1).

**What ADOPT means here.**
- zlk ADOPT makes zlk the base of every later screen and run.
- If another round (eighteen, nineteen, twenty or twenty-one) is ADOPTED against zrm as well, the base is decided in a
  file declared before that decision, as docs/BASE_ZRM_ZRS.md decided between zrm and zrs. Combinations are rounds of
  their own.
- If another arm becomes the base before this run is graded, this run still decides zlk against zrm's fits, as
  declared. A zlk ADOPT then leads to a combination with that base in a later round.

**Every fit runs to its end.** A LOSS filed at the floor 0.0075 may turn WITHIN under the null, so nothing is stopped
early.

## 5. Gates and order of work (8 October)

- **Before the screen's fits:** the eleven edge builds (`zlk-build-<dataset>-<carve>`: the five training datasets' fit
  carves, musique's s1fit, and the six s1eval carves) and the two checks (`zlk-check-hotpotqa`, `zlk-check-webqsp`).
  A check exits 1 if zlk's forward at the start is not zrm's, and the fits wait on it.
- **`flk-gate`** (`zlink.py gate`) exits 0 only when both of these hold:
  - the screen's re-call (`outputs/screen/scr-zlk-pair-recall.json`) is PROMISING;
  - zrm is the base: docs/BASE_ZRM_ZRS.md's re-grade (`outputs/zbase/grade-nullx.json`) is ADOPT. It is (10:05).

  Otherwise the feeder drops the four fits with their reads, comparisons and grades.
- **The fits** wait on the gate. Each read waits for its fit. Each comparison waits for its read.
- **The items.** The screen's GPU items go ahead of every waiting GPU item but round twenty's two reads. The full run
  goes at the end of the feeder's list. Nothing is preempted.
- **ETAs.** The screen's re-call lands about 20:15 to 21:00. If it passes, the four fits queue behind the rounds
  already waiting, and the grade lands late on 9 October, depending on the card.

## 6. What this does not do

- No new graph, column, text, encoder or model. The edges are the pools' own, as the look built them for the cache.
  The encoder and the substrate embeddings stay frozen.
- No new seed, variant or hyperparameter of training. zrm's model, settings and training, and rmatch's walk, chain cap
  and relation tables, are used unchanged. The 353 head weights are the only new weights.
- The leaders are zrm's own scores. No label, gold flag or dataset name enters the inputs.
- Nothing is written into zrm's, zrs's or zret's fit folders.
- webqsp never trains. Test splits are never read.
- **Speed.** Per question, one sort of its pool's scores and one pass over its pool's edges, after zrm's forward. Any
  latency figure for this arm is cold (8216ffe): each question timed from scratch, with the walk, the move of its
  entries and edges to the device and the forward included, no warm-up pass, and nothing kept from an earlier
  question. Its latency stage is declared in its own file.

## Results
