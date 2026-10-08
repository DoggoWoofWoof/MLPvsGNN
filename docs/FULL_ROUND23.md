# Full run: zsp, each row's neighbours' scores on zrm, on the host's CPU, only if its screen's re-call is PROMISING and zrm is the base

Declared 8 October 2026 at about 20:05, before any number of round twenty-three exists (docs/SCREENS.md,
twenty-third round). zrm is the base (docs/BASE_ZRM_ZRS.md, re-graded ADOPT at 10:05). If the screen passes, this run
decides whether zsp's gain over zrm holds on every split, and whether anything else loses. These are development
numbers. The paper's numbers come from one declared confirmation run.

## 1. Question

zsp is zrm plus a small head added to its score. For each row the head reads, per edge family of the question's own
pool graph:

- the mean of its neighbours' z-scored zrm scores;
- their soft maximum;
- its degree;

and the row's own z-score. Does a row's neighbours' standing lift zrm's reads on every split, and does nothing else
lose? The splits are J5 (in-domain on all five training datasets) and each leave-one-out fit on its held-out dataset.

Why this question: docs/DIAG_BRIDGE.md (D1, D2) and docs/SCREENS.md's twenty-third round. The golds zrm misses sit
next to the rows it ranks highly; the per-row inputs hold no more; round twenty-two cuts the link at zrm's top five,
and this round reads every neighbour's score.

## 2. Fits, all on the host's CPU

- **What trains.** The arm zsp, (zprop.ZProp, zlink.LinkCarveBase over rmatch.ChainCarveBase), through
  `outputs/mp_unified/zprop.py train --device cpu`:
  - zrm's model plus the head: `sp_w1`, `sp_b1`, `sp_w2` and `sp_b2`, 385 new weights;
  - zrm's settings and training (rmatch.py's train), over rmatch's builds and zlink's edge builds;
  - seed 0, as zrm's fits.
- **zrm's base fits are trained on the CPU too** (`zprop.py base`, zrm.py's train with `--device cpu`), so every
  comparison has the same device on both sides.
- **The screen's fits are this run's L-musique and L-hotpotqa fits:** zsp's `outputs/screen/fits/scr-zsp` and
  `scr-zsp-hp`, against zrm's `scr-zrm-cpu` and `scr-zrm-cpu-hp`.
- The run trains eight fits on the CPU, 6 threads each:
  - zsp's four: L-2wiki, L-squad, J5 and L-metaqa (`zprop.py train --split S --name S --arm zsp --out-root
    outputs/full_zsp/fits --device cpu --threads 6`);
  - zrm's four on the same splits (`zprop.py base --split S --name S --out-root outputs/full_zsp/zrm --threads 6`).
- **The head trains on every graph.** No split keeps it at zero, so no split is zrm's bit for bit. The identity at the
  start is checked by zprop.py's selftest and by `zsp-check-hotpotqa` and `zsp-check-webqsp` on real carves.
- **On the host.** Each fit and read takes 6 CPUs and 16 GB; no card. The feeder starts them in file order as its CPU
  cap allows (about four at a time): L-2wiki, L-squad, L-metaqa, J5, each zsp fit beside zrm's of the same split.

## 3. Reads and comparison

- Each fit is read on the six s1eval carves on the CPU (`zprop.py read` for zsp, `zrm.py read --device cpu` for zrm).
- **Each zsp read is compared, question by question, with zrm's CPU fit of the same split,** p@swa, on the same
  carves. That comparison decides. zrm's card fit (`outputs/full_zrm/fits/S`; on L-musique and L-hotpotqa, `scr-zrm` and
  `scr-zrm-hp`) and step 1's are reported beside.
  - The comparison is lean_screen's: the R@5 difference with a 2,000-resample question bootstrap, called GAIN, LOSS or
    WITHIN with the floor 0.0075.
  - hit@1 is reported beside R@5.
  - Twenty-four reads come from these fits and twelve from the screen: 36 in all.
- **Each zrm CPU fit is also compared with step 1's fit of its split** (`outputs/full_zsp/base-zrm-cpu-S.json`; on
  L-musique and L-hotpotqa, `outputs/zprop/base-zrm-cpu-S.json`). These give each read's base R@5 for the re-grade,
  and, beside, zrm's CPU fit against its card fit: the device's spread, reported only.
- **Primary reads** are step 1's eleven: J5 on all six datasets (five in-domain, webqsp zero-shot), and each leave-out
  fit on its held-out dataset.

## 4. Verdict

- `zprop.py grade --full-root outputs/full_zsp --reuse
  L-musique=outputs/screen/scr-zsp.json,L-hotpotqa=outputs/screen/scr-zsp-hp.json` files the grade: relz.py's grade,
  decided against zrm's CPU fits, under zsp's name. It refuses a comparison whose new fit or deciding base is not a
  CPU fit.
- **The re-grade decides.** `nullx.py regrade` re-grades all 36 reads under the seed null over every split into
  `outputs/full_zsp/grade-nullx.{md,json}`. Each read's base R@5 is zrm's CPU fit's (`--base-compares`, the six
  comparisons above).
- **ADOPT:** at least one primary read GAINs, and none of the 36 reads LOSEs.
- **NOT_ADOPTED:** otherwise.
- **INCOMPLETE:** a comparison is missing or is not the declared one (exit 1).

**What ADOPT means here.**
- zsp ADOPT makes zsp the base of every later screen and run, its fits the CPU's.
- If another round (eighteen, nineteen, twenty-one or twenty-two) is ADOPTED against zrm as well, the base is decided
  in a file declared before that decision, as docs/BASE_ZRM_ZRS.md decided between zrm and zrs. Combinations are rounds
  of their own.
- If another arm becomes the base before this run is graded, this run still decides zsp against zrm's CPU fits, as
  declared. A zsp ADOPT then leads to a combination with that base in a later round.

**Every fit runs to its end.** A LOSS filed at the floor 0.0075 may turn WITHIN under the null, so nothing is stopped
early.

## 5. Gates and order of work (8 October)

- **Before the screen's fits:** the two checks (`zsp-check-hotpotqa`, `zsp-check-webqsp`). A check exits 1 if zsp's
  forward at the start is not zrm's, or an input is not finite, and the zsp fits wait on it. zrm's two CPU refits wait
  on nothing.
- **`fsp-gate`** (`zprop.py gate`) exits 0 only when both of these hold:
  - the screen's re-call (`outputs/screen/scr-zsp-pair-recall.json`) is PROMISING;
  - zrm is the base: docs/BASE_ZRM_ZRS.md's re-grade (`outputs/zbase/grade-nullx.json`) is ADOPT. It is (10:05).

  Otherwise the feeder drops the eight fits with their reads, comparisons and grades.
- **The fits** wait on the gate. Each read waits for its fit. Each comparison waits for its read and zrm's.
- **The items** are CPU items at the end of the feeder's list; they take nothing from the card's queue. Nothing is
  preempted.
- **ETAs.** A CPU fit's pace is measured on the screen's first epoch. If the screen passes, the eight fits take about
  one to two days on the CPU, J5's the longest.

## 6. What this does not do

- No new graph, column, text, encoder or model. The edges are round twenty-two's, the pools' own. The encoder and the
  substrate embeddings stay frozen.
- No new seed, variant or hyperparameter of training. zrm's model, settings and training, and rmatch's walk, chain cap
  and relation tables, are used unchanged. The 385 head weights are the only new weights.
- The neighbours' scores are zrm's own. No label, gold flag or dataset name enters the inputs.
- Nothing is written into zrm's, zrs's or zret's fit folders; zrm's CPU fits are new folders of their own.
- webqsp never trains. Test splits are never read.
- **Speed.** Per question, one pass over its pool's edges, after zrm's forward. Any latency figure for this arm is cold
  (8216ffe): each question timed from scratch, with the walk, the move of its entries and edges to the device and the
  forward included, no warm-up pass, and nothing kept from an earlier question. Its latency stage is declared in its
  own file.

## Results
