# Full run: zkind, a head for graphs with typed relations on zrm, only if its screen's re-call is PROMISING and zrm is the base

Declared 8 October 2026 at about 11:00, before any number of round nineteen exists (docs/SCREENS.md, nineteenth
round). zrm is the base (docs/BASE_ZRM_ZRS.md, re-graded ADOPT at 10:05). This run decides, if the screen passes,
whether zkind's gain over zrm holds on every split, and whether anything else loses. Numbers here are development
numbers; the paper's numbers come from one declared confirmation run.

## 1. Question

zkind is zrm whose output layer (its weights and bias) and rrf's base weight take learned offsets on a graph with typed
relations, zero at the start. A graph has typed relations when its batches carry rmatch's chain entries: metaqa's and
webqsp's. Every other batch's forward is zrm's. Does a separate read of the shared hidden units on the typed graphs lift
zrm's reads on every split, above all webqsp's, which never trains? And does nothing else lose? The splits are J5
(in-domain on all five training datasets) and each leave-one-out fit on its held-out dataset.

Why this question:
- **The two kinds of graph want different things from a seed.**
  - On metaqa and webqsp a seed is the question's topic entity. It holds only 0.066 of webqsp's in-pool golds and 0.005
    of metaqa's (outputs/diag/hopdiag-J5).
  - On the passage graphs a seed is a retrieved passage, and often gold.
- **What the MLPs learn of metaqa does not carry to webqsp.** step 1's and rel's J5 fits rank a seed first for 0.9454
  and 0.5948 of webqsp's questions, against 0.0262 and 0.0198 of metaqa's (outputs/diag/hopdiag-J5). hopdiag reads only
  step 1's and rel's fits, so zrm's rate is not measured.
- **zrm's reads** (outputs/zbase/grade-nullx.json):
  - metaqa in-domain: 0.78 R@5 in each of the five fits that train it;
  - webqsp read zero-shot: 0.24 to 0.31 in those five fits, and 0.10 in L-metaqa's, which trains no typed graph.
- **One output layer serves both kinds of graph.** The offsets give the typed graphs their own read of the shared hidden
  units.
- **The change is general.** A graph's kind (typed relations or not) is the graph's own property, read from its batches,
  never from a dataset's name. So the head applies to any typed graph, seen in training or not, and every dataset is
  graded.

## 2. Fits

- **What trains.** The arm zkind, (zkind.ZKind, rmatch.ChainCarveBase), through `outputs/mp_unified/zkind.py train`:
  - zrm's model with three offsets (`kd_w`, `kd_b`, `kd_base`: 130 new weights at zrm's 128 hidden units);
  - zrm's settings and training (rmatch.py's train), over rmatch's builds;
  - seed 0, as zrm's fits.
- **The screen's fits are this run's L-musique and L-hotpotqa fits** (`outputs/screen/fits/scr-zkind` and
  `scr-zkind-hp`).
- The run trains four fits, in the order L-2wiki, L-squad, J5, L-metaqa: `zkind.py train --split S --name S --arm zkind
  --out-root outputs/full_zkind/fits`.
- **An offset takes a gradient only from a typed graph's batch.** metaqa trains in every split but L-metaqa. webqsp
  never trains.
- **L-metaqa's fit trains no typed graph.** Its offsets stay at zero and its training is zrm's, so its six reads must be
  zrm's bit for bit. zkind.py's selftest checks this on toy fits; here L-metaqa's comparison checks it (section 3).
- **On the card:**
  - each fit 0.26 (share 0.28) and 9 GB, L-metaqa's 0.18 (share 0.20);
  - each read 0.30 (share 0.32) and 6 GB, as round eighteen's.

## 3. Reads and comparison

- Each fit is read on the six s1eval carves with rmatch's chain entries, as zrm's fits were (`zkind.py read --name S
  --out-root outputs/full_zkind/fits`).
- **Each read is compared, question by question, with zrm's fit of the same split,** p@swa, on the same carves. That
  comparison decides. zret's fit and step 1's are reported beside it.
  - zrm's fits: `outputs/full_zrm/fits/S`; for L-musique and L-hotpotqa, zrm's screen fits `scr-zrm` and `scr-zrm-hp`.
  - zret's fits: `outputs/full_zret/fits/S`; for L-musique, zret's screen fit `scr-zret`.
  - The comparison is lean_screen's: the R@5 difference with a 2,000-resample question bootstrap, called GAIN, LOSS or
    WITHIN with the floor 0.0075.
  - **L-metaqa's comparison also files every read's arrays against zrm's** (`compare-L-metaqa-same.md`). Every read
    must be IDENTICAL. If one is not, it is a bug: the comparison exits 1, the grade is not filed as a result, and the
    round stops.
  - hit@1 is reported beside R@5.
  - Twenty-four reads come from these fits and twelve from the screen: 36 in all.
- **Primary reads** are step 1's eleven: J5 on all six datasets (five in-domain, webqsp zero-shot), and each leave-out
  fit on its held-out dataset.
- **Where a change can show.**
  - metaqa's and webqsp's reads, through the offsets.
  - The passage datasets' reads use zrm's forward. But the shared weights they read through train beside the offsets,
    so they may move too.

## 4. Verdict

- `zkind.py grade --full-root outputs/full_zkind --reuse
  L-musique=outputs/screen/scr-zkind.json,L-hotpotqa=outputs/screen/scr-zkind-hp.json` files the grade: relz.py's
  grade, decided against zrm's fits, under zkind's name (zrc.py's mapping).
- **The re-grade decides.** `nullx.py regrade` re-grades all 36 reads under the seed null over every split into
  `outputs/full_zkind/grade-nullx.{md,json}`. Each read's base R@5 is zrm's (`--base-compares`: zrm's six fits compared
  with step 1's, `outputs/zrc/base-zrm-S.json`).
- **ADOPT:** at least one primary read GAINs, and none of the 36 reads LOSEs.
- **NOT_ADOPTED:** otherwise.
- **INCOMPLETE:** a comparison is missing or is not the declared one (exit 1).

**What ADOPT means here.**
- zkind ADOPT makes zkind the base of every later screen and run.
- If round eighteen's zrc is ADOPTED against zrm as well, the base is decided in a file declared before that decision,
  as docs/BASE_ZRM_ZRS.md decided between zrm and zrs. The two changes do not overlap: zrc changes which entries a typed
  row keeps, and zkind the head that reads them. So their combination would be a round of its own.
- If another arm becomes the base before this run is graded, this run still decides zkind against zrm's fits, as
  declared. A zkind ADOPT then leads to a combination with that base in a later round (docs/BASE_ZRM_ZRS.md, section 4).

**Every fit runs to its end.** A LOSS filed at the floor 0.0075 may turn WITHIN under the null, so nothing is stopped
early.

## 5. Gates and order of work (8 October)

- **`fkd-gate`** (`zkind.py gate`) exits 0 only when both of these hold:
  - the screen's re-call (`outputs/screen/scr-zkind-pair-recall.json`) is PROMISING;
  - zrm is the base: docs/BASE_ZRM_ZRS.md's re-grade (`outputs/zbase/grade-nullx.json`) is ADOPT. It is (10:05).

  Otherwise the feeder drops the four fits with their reads, comparisons and grades.
- **The fits** wait on the gate. Each read waits for its fit. Each comparison waits for its read.
- **The items.** The screen's items go after round eighteen's screen items, ahead of the queued full runs' fits. The
  full run goes at the end of the feeder's list. Nothing is preempted.
- **ETAs.**
  - The screen's two fits start when round eighteen's reads leave room on the card, about 11:10, and take about 40
    minutes each. The re-call lands about 11:55 to 12:15.
  - If it passes, the four fits queue behind round eighteen's full run, if that runs. The grade lands about 14:00 to
    16:00, depending on the card.

## 6. What this does not do

- No new column, block, carve, graph, text, seed, variant or hyperparameter. zrm's model, settings and training, and
  rmatch's walk, chain cap and relation tables, are used unchanged. The 130 offsets are the only new weights.
- The graph's kind is read from the batch (whether it carries chain entries), never from a dataset's name.
- No new graph, text, encoder or model. The encoder and the substrate embeddings stay frozen.
- Nothing is written into zrm's, zrs's or zret's fit folders.
- webqsp never trains. Test splits are never read.
- **Speed.** On a typed graph the offsets add one sum per weight, once per batch, and no work per row. Any latency
  figure for this arm is cold (8216ffe): each question timed from scratch, with the walk, the move of its entries to the
  device and the forward included, no warm-up pass, and nothing kept from an earlier question. Its latency stage is
  declared in its own file.

## Results
