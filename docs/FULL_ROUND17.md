# Full run: zrm's fits read with chain entries kept by what they add to the match (zrc), only if its screen's re-call is PROMISING

Declared 8 October 2026 at about 09:10, before any number of round seventeen exists (docs/SCREENS.md, seventeenth
round). zrm's screen and its re-call are PROMISING against zret's fits, and its full run is under way. This run
decides, if the screen passes, whether keeping rmatch's chain entries by what they add to the match at its start, in
place of by walk mass, lifts zrm's reads on every split, and whether anything else loses. Numbers here are development
numbers; the paper's numbers come from one declared confirmation run.

## 1. Question

zrc is zrm with rmatch's chain entries kept, per row and seed bucket, by w0(c)·m_c(v) in place of m_c(v). If its
screen's re-call against zrm's fits is PROMISING: does its gain over zrm hold on every split, and does nothing else
lose? The splits are J5 (in-domain on all five training datasets) and each leave-one-out fit on its held-out dataset.

## 2. Fits

- **No fit is trained.** zrm's fits are zrc's when the identity gate holds (docs/SCREENS.md, seventeenth round, 'Why
  nothing is trained'): `outputs/zrc/identity.json` must be IDENTICAL.
- **The screen's fits are this run's L-musique and L-hotpotqa fits** (`outputs/screen/fits/scr-zrc` and `scr-zrc-hp`,
  forks of `scr-zrm` and `scr-zrm-hp`).
- The run forks zrm's four full-run fits, in the order L-2wiki, L-squad, J5, L-metaqa, through `outputs/mp_unified/zrc.py
  fork --src outputs/full_zrm/fits/S --name S --out-root outputs/full_zrc/fits`. Each fork waits for its zrm fit to
  end, and copies its models and training record unchanged. zrm's folders are never written.
- **L-metaqa's fit trains no typed graph,** so its match's gates stay at zero (docs/FULL_ROUND14.md, section 2), and
  its reads must be zrm's bit for bit.
- **The entries** are zrc's builds of metaqa's and webqsp's s1eval carves (`outputs/zrc/cache`), from rmatch's walk
  with only the kept set changed.
- **On the card:** reads 0.30 (share 0.32), 6 GB.

## 3. Reads and comparison

- Each fork is read on the six s1eval carves with zrc's entries (`zrc.py read --name S --out-root
  outputs/full_zrc/fits`).
- Each read is compared, question by question, with zrm's fit of the same split, p@swa, on the same carves. That
  comparison decides. zret's fit and step 1's are reported beside it.
  - zrm's fits: `outputs/full_zrm/fits/S`. For L-musique and L-hotpotqa they are zrm's screen fits, `scr-zrm` and
    `scr-zrm-hp`. Each comparison waits for zrm's read of its split.
  - zret's fits: `outputs/full_zret/fits/S`. For L-musique it is zret's screen fit, `outputs/screen/fits/scr-zret`.
  - The comparison is lean_screen's: the R@5 difference with a 2,000-resample question bootstrap, called GAIN, LOSS or
    WITHIN with the floor 0.0075.
  - **Each comparison also files every read's arrays against zrm's** (`compare-S-same.md`). The reads of squad,
    musique, hotpotqa and 2wiki, and every read of L-metaqa's fit, must be IDENTICAL. If one is not, it is a bug: the
    grade is not filed as a result, and the round stops.
  - Twenty-four reads come from these fits and twelve from the screen: 36 in all.
- **Primary reads** are step 1's eleven: J5 on all six datasets (five in-domain, webqsp zero-shot), and each leave-out
  fit on its held-out dataset.

## 4. Verdict

- `zrc.py grade --full-root outputs/full_zrc --reuse
  L-musique=outputs/screen/scr-zrc.json,L-hotpotqa=outputs/screen/scr-zrc-hp.json` files the grade: relz.py's grade,
  decided against zrm's fits, under zrc's name.
- **The re-grade decides.** `nullx.py regrade` re-grades all 36 reads under the seed null over every split into
  `outputs/full_zrc/grade-nullx.{md,json}`. Each read's base R@5 is zrm's (`--base-compares`: zrm's six fits compared
  with step 1's, `outputs/zrc/base-zrm-S.json`).
- **ADOPT:** at least one primary read GAINs, and none of the 36 reads LOSEs.
- **NOT_ADOPTED:** otherwise.
- **INCOMPLETE:** a comparison is missing or is not the declared one (exit 1).

**What ADOPT means here.** zrc's grade is relative to zrm.
- If zrm's own re-grade (docs/FULL_ROUND14.md) is ADOPT, zrc ADOPT makes zrc the base of every later screen and run.
- If zrm is NOT_ADOPTED, zrc's grade decides nothing about the base. A grade of zrc against zret's fits would be
  declared in a later round, before its numbers.
- If another arm of rounds thirteen to sixteen is ADOPTED too, each re-grade is filed and the next base is declared in
  a later round, before its numbers.

**Every read runs to its end.** A LOSS filed at the floor 0.0075 may turn WITHIN under the null, so nothing is stopped
early.

## 5. Gates and order of work (8 October)

- **`zrc-identity`** exits 0 only on IDENTICAL. Every fork waits on it.
- **`fzc-gate`** (`screen_gate.py pass` on `outputs/screen/scr-zrc-pair-recall.json`) exits 0 only when the re-call is
  PROMISING. The four forks wait on it and on their zrm fits (`fzm-train-S-b`). On any other result the feeder drops
  them with their reads, comparisons and grades.
- **The base comparisons** (`zrc-base-S`: zrm's fit of each split against step 1's, 1 CPU) wait for zrm's reads.
- **The items** go after round fifteen's in the feeder's list, ahead of zgs's re-queued screen (docs/SCREENS.md,
  seventeenth round, 'Caps and order').
- **ETAs if it passes:** the re-call lands about 09:45 to 10:30. Each fork waits for its zrm fit, and each read takes
  about 3 minutes once the card has room. The grade lands about 10:30 to 11:30.

## 6. What this does not do

- No training, and no new carves, seeds, variants, columns or hyperparameters of the fit. zrm's model and weights, and
  rmatch's walk, chain cap and relation tables, are used unchanged. Only which 64 entries a row keeps changes.
- No new graph, text, encoder or model. The encoder and the substrate embeddings stay frozen. w0 uses the frozen
  encoder's question and relation embeddings, already on disk.
- webqsp never trains. Test splits are never read.
- No speed figure rests on entries built ahead of time. Any latency figure for this arm is cold (8216ffe): each question
  from scratch, the walk, the selection and the forward included. Its latency stage is declared in its own file.

## Results
