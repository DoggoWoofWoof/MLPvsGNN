# LTA: zrckpn with its next-hop columns over every pool row (linked to the found passages or not), on PKO's pools, on the host's CPU

Declared 3 November 2026, while LTS's fits run and before any LTA list, scorer fit, column or MLP fit. Stage modules,
both committed with this file:

- `outputs/mp_unified/ltacols.py` (the A lists, the scorer's fits and the columns);
- `outputs/mp_unified/zlta.py` (the MLP's fits, compares and calls).

They import `ltmcols.py`, `ltwdiag.py`, `chndiag.py`, `zltm.py`, `zwq.py`, `zml1.py`, `zml1s.py`, `zdob.py`, `zp1c3p.py`,
`zdep.py` and `zltg.py`, and edit none of them. LTM's lists, fold fits, columns, MLP fits and compares are read only, as
are PKO's cache and links. LTA writes only under `outputs/lta` and under its own fit and compare names (`scr-lta-*`).

## 1. The question

- **Where NEXT stops.** LTM's NEXT columns (docs/LTM_NEXT_HOP_MLP.md) score, for each of the first stage's top five
  rows, only the rows linked to it. On musique those lists hold 0.045 of the rows. A gold that no found passage links to
  gets no next-hop term at all.
- **LTM section 8 (c)** names the form: "the term over every pool row against each anchor, linked or not, since
  musique's lists hold 0.045 of its rows."
- **Why now, and beside LTS.** LTS (docs/LTS_SPAN_RESIDUAL_MLP.md, 36c5e51) changes the question's residual inside the
  same term. Its section 7 makes this form its next on NO_GAIN or NOT_ADOPTED. It is declared now, before LTS's screen,
  because the host's CPU has free cores while LTS's fits run (19.5 of 30 CPUs reserved at 06:26 IST). Each form is
  screened against zrckpn on its own. Neither is built on the other, so neither verdict waits on the other.
- **The user's direction (3 November):** "can we finish these any faster, and iterate much faster ... we need to close
  the gaps that we have no matter what."
- **So the question is:** with NEXT's term read over every row outside the top five, linked or not, does zrckp's
  in-domain R@5 rise over zrckpn's, above all on musique's 3- and 4-hop questions?

## 2. The columns (`ltacols.py`)

- **Questions and folds are LTM's**, read only: the questions, their cache parts, rrf's top five rows, LTM's five folds
  (training carves) and its drawn folds (other carves), and the question vectors. Each list set is checked against LTM's
  lists of its carve, key by key, and refused on any difference.
- **The lists.** For each question, the anchors are rrf's top five rows. Anchor k's candidates are every row of the
  question outside the top five. A candidate's link bits to its anchor are PKO's; a row not linked to the anchor has
  none (0).
- **The A lists (the scorer's training lists).** A question with a gold among the top five and a gold outside them is
  an A case:
  - the anchor is the first gold among the top five;
  - every row outside the top five is a candidate, and every gold outside them a target.

  An A list is stored capped as ltmcols' fit caps a long list: every target, then negatives drawn without replacement up
  to LTW's cap (256), from the stream ltmcols' fit draws from (`default_rng([SEED, dataset, 1])`, in question order).
  So the fit's own draw finds nothing to cap. Each list's candidate count before the cap is kept.
- **The scorer** is LTW's bil form, refitted per split on LTA's A lists of its four training carves with LTM's folds
  (ltmcols' fit, unchanged: five folds, LTM's epochs, stops and MIN_LISTS).
- **The columns** are LTM's seven, with LTM's layout:
  - next_1 to next_5: a row's log-ratio to uniform among anchor k's candidates, clipped to +-8, and -8 on the top five
    rows;
  - next_max: the largest of the five;
  - next_n: the number of anchors the row is linked to (LTM's own column).
- **Scoring.** Out of fold on a training carve, the drawn fold on any other carve. The lists are read through ltwdiag's
  own `read_chunk`, `items_of` and `score_items`, one chunk of questions at a time (about 40,000 candidate rows), so no
  list of every row is held at once.
- **Checks in every build.** Besides LTM's (the cache parts by sha256, each list's cosine identity):
  - the cache parts and their links are the ones the carve's lists were built on, and each question's row count and
    part are its lists' own;
  - next_n equals LTM's next_n, bit for bit, on every part where LTM's columns are on disk;
  - a fit and its columns carry the ltacols.py that built them, and a column set is refused unless its split's fit is
    LTA's under the present ltacols.py;
  - lists built with a limit or another cap are refused by the fit and the columns.

## 3. What runs

1. Eleven list sets, one per carve of LTM's lists.
2. Two scorer fits, one per split, on its four training carves.
3. Twenty column sets, ten per split (its four training carves and the six s1eval carves). Then the report, which
   decides nothing.
4. zrckpa at seed 0 on both splits: zltm's fit of zrckpn with LTA's columns in NEXT's place. The model, chain carve,
   pick, trainer, read cap, device and threads are zrckpn's. zrckpa has zrckpn's inputs and shapes, so at a seed both
   start from the same draw.
5. The compares (zrckpa against zrckpn of the same seed and split), the calls, and the gate.
6. On ADOPT_FOR_FULL_RUN, the full run: seeds 1 and 2 on both splits.

## 4. The reads

DEP's section 3, unchanged, as in LTM's section 5:

- ML1's twelve reads;
- musique's 3-hop and 4-hop slices on both splits.

## 5. The rule (fixed now)

LTM's section 6, with zrckpa in zrckpn's place and LTM's zrckpn (seeds 0, 1 and 2) in DOB's base's place.

- **Deciding (ten):** the eight reads of datasets their split trains on, and the WQ-hotpotqa fits' musique 3-hop and
  4-hop slices.
- **Beside (six):** the four zero-shot reads and WQ-musique's slices.
- **The screen (seed 0).** Each deciding read is called on its own:
  - the floor is max(0.0075, 2 x zrckpn's spread over its three seeds), the spread being the root mean square of its
    seed-1 and seed-2 differences from its seed 0;
  - LOSS: zrckpa - zrckpn at seed 0 is at or below minus the floor. GAIN: at or above the floor;
  - NOT_ADOPTED: any LOSS. ADOPT_FOR_FULL_RUN: no LOSS and at least one GAIN. NO_GAIN: every read WITHIN.
- **The full run (seeds 1 and 2).** Each deciding read is called by `zml1.call` on its three seed deltas, with the pair
  floor of both arms' spreads.
  - ADOPT: no LOSS among the ten, and a GAIN on at least one of the eight dataset reads. A slice GAIN alone does not
    adopt, and UNCONFIRMED_GAIN does not count.
  - NOT_ADOPTED otherwise.
- **INCOMPLETE** holds when a deciding read is missing, and also when:
  - a fit names a gate other than PKO's present gate;
  - its device, threads, pick or read cap are not those of LTM's fit of its seed and split, or it is not stamped against
    that fit;
  - its train or read loaded columns other than LTA's, or of more than one scorer, or of a scorer other than the split's
    present LTA fit;
  - a training carve's columns are missing from its train, or an eval carve's from its read;
  - a column record changed after the fit read it;
  - its code differs from LTM's fit's in any file that fit stamped, or zlta.py and ltacols.py are not stamped;
  - its scorer and columns were built by an ltacols.py other than the one its code stamp names;
  - a compare's base R@5 at a seed is not zrckpn's at that seed, or musique's read file does not give its compare's R@5.

A verdict closes only the form it tested.

## 6. Reported beside, deciding nothing

- The six beside reads, called the same way.
- Per carve, the A cases' T2 under bil among every row outside the top five, by hop count on musique:
  - its linked and unlinked targets apart;
  - rrf's T2 on the same cases, as the first stage's own order of those rows;
  - LTM's A T2 among linked rows beside (LTM's WQ-hotpotqa musique/s1eval: 0.493; 2-hop 0.549, 3-hop 0.420, 4-hop
    0.387). Its lists are of other lengths, so the two are not one measure.
- LTA's lists: A cases, their targets' linked share, candidates per list, and lists capped.
- The rows with an anchor's link (next_n above 0), and next_max's mean on gold and other rows.
- LTS's filed zrckps - zrckpn at seed 0, once filed.
- Seconds per scorer fit, column set and MLP fit.

## 7. What follows (fixed now)

| outcome | what follows |
| --- | --- |
| ADOPT_FOR_FULL_RUN | The full run, queued now behind `lta-gate`, then section 5's full-run rule. **On ADOPT,** zrckpa is the MLP candidate in zrckpn's place. It is measured again for Claim 1 (C1R's intervals), and Claim 3's cold timing counts its columns' cost (every pool row scored against five anchors, and the scorer), measured, not assumed. The next form, in its own file: LTA's rows with LTS's span residual if LTS's full run adopts, else LTA's rows with each A list's negatives drawn from rrf's leading rows outside the top five. **On NOT_ADOPTED,** as the next row. |
| NO_GAIN or NOT_ADOPTED | The term over every row with uniform negatives closes. The next form, in its own file: LTA's rows with each A list's negatives drawn from rrf's leading rows outside the top five, not uniformly. |
| INCOMPLETE | What is missing is fixed in an amendment, before any number is read. |

- **If LTS's and LTA's full runs both adopt,** the MLP candidate is the one with the larger mean delta against zrckpn
  over the eight dataset reads and three seeds, until the next form's own file is read.
- A full run not read by the freeze (6 to 7 November) leaves the candidate as it stands then.
- No other stage waits on LTA: LTS's, RWT's and STK's items do not name it.

## 8. Risks, stated before any number

- **Easy negatives.** An A list's negatives are drawn uniformly from thousands of rows (musique: about 5,450 rows per
  question). Most lie far from the question, so the scorer may learn to tell golds from far rows and not from the rows
  that compete at the top. This is the next form's change.
- **No link bits off the links.** An unlinked candidate's term rests on the vectors and the first stage's ranks and
  scores. Those are NEXT's own inputs, but on linked rows the link bits carry much of LTM's signal.
- **The scale of the log-ratio.** Among about 5,000 candidates, uniform is ln 5,000 = 8.5, so the +-8 clip binds on
  the highest-scored rows. The columns keep LTM's clip, so the MLP reads the same layout.
- **The scorer.** It is refitted on longer lists with LTM's folds. Its stops may fall at other epochs than LTM's.
- **Seed 0 only in the screen.** The floor is zrckpn's seed spread although both fits start from one draw, so the screen
  is conservative.
- **Time.**
  - Lists: minutes per carve. Scorer fits: LTM's took 250 s (WQ-musique) and 268 s (WQ-hotpotqa); LTA's lists are about
    seven times as long, so 30 to 60 minutes.
  - Columns: LTM's took 19 to 104 s per carve over its linked rows. LTA scores every row outside the top five against
    five anchors (about 120 million rows per split), so tens of minutes per carve; the first build measures it.
  - zrckpa's fits should take as long as zrckpn's, about 3.5 to 4 h each, and start when LTS's or RWT's fits free their
    memory. The screen should be read on 3 November, the full run, if the screen passes, by 4 to 5 November, before the
    freeze.
- **Memory.** Lists at LTM's lists' measured peaks plus 1 GB; scorer fits at 13 GB (about 7,300 lists x 256 rows of
  float16 vectors per split, about 5.7 GB); columns at LTM's measured peaks plus 3 GB; the MLP fits at zrckpn's measured
  peaks (the feeder adds its 15% margin to each).
- **Disk.** 218 million rows x 7 float16 columns: about 3 GB under `outputs/lta` on the host (75.2 GB free at this
  declaration).

## 9. How it runs

Feeder items on the host's CPU. None touches the GPU. They sit after LTS's items, ahead of DOB's remaining x-fits.

1. `lta-selftest-cols` (`ltacols.py --selftest`, 1 CPU, 3 GB) and `lta-selftest` (`zlta.py --selftest`, 1 CPU, 4 GB).
2. The eleven list sets, `lta-lists-<dataset>-<carve>`, after `lta-selftest-cols` and LTM's lists of the carve (1 CPU,
   at the LTM lists item's measured peak plus 1 GB).
3. The scorer fits, `lta-fit-wqm` and `lta-fit-wqh`, after their split's four training list sets (2 CPUs, 13 GB).
4. The twenty column sets, `lta-cols-<wqm|wqh>-<dataset>-<carve>`, after their split's fit, the carve's lists and LTM's
   same column set (2 CPUs, at its measured peak plus 3 GB). Then `lta-report`.
5. The screen's fits, after `pkoh-gate`, `lta-selftest` and their split's ten column sets (6 CPUs, at zrckpn's peaks):
   `lta-zrckpa-s0-hp`, then `lta-zrckpa-s0`.
6. `lta-compare`, then `lta-calls`, then `lta-gate` (`outputs/lta/calls.json`, `screen.verdict` ADOPT_FOR_FULL_RUN).
7. The full run, after `pkoh-gate` and `lta-gate`: `lta-zrckpa-s1-hp`, `-s2-hp`, `-s1` and `-s2`, at zrckpn's peaks of
   the same seed. Then `lta-compare-full` and `lta-calls-full`.

```
python outputs/mp_unified/ltacols.py lists --dataset musique --carve s1fit
python outputs/host_ops/retry_cmd.py --tries 3 --wait 90 -- python outputs/mp_unified/ltacols.py fit --split WQ-hotpotqa --host
python outputs/host_ops/retry_cmd.py --tries 3 --wait 90 -- python outputs/mp_unified/ltacols.py cols --split WQ-hotpotqa --dataset musique --carve s1eval --host
python outputs/mp_unified/ltacols.py report                         (outputs/lta/report.json and .md)
python outputs/host_ops/retry_cmd.py --tries 3 --wait 90 -- python outputs/mp_unified/zlta.py fit --name scr-lta-zrckpa-s0-hp
python outputs/mp_unified/zlta.py compare-all
python outputs/mp_unified/zlta.py calls                             (outputs/lta/calls.json and .md)
```

Only `outputs/lta/calls.json`, `calls.md`, `report.json` and `report.md` are committed. The lists, fits and columns stay
on the host.

`ltacols.py --selftest` passed on the laptop before this commit. It covers, on toy carves of several cache parts with
pools of unequal size:

- LTM's lists, then LTA's: the A cases by hand (anchor, candidates, targets), and the stored cap equal to LTW's own
  draw (`lists_of`) on the same lists;
- the refusals: LTM's lists with a fold changed, or missing;
- LTA's fold fits under LTA's names, every patched name put back, also on a failure;
- LTM's fits and columns of the same toys, then LTA's columns of three carves at chunk sizes 120, 1,000 and 40,000:
  - each list's scores by hand from the fold that scores it;
  - next_n equal to LTM's;
  - the columns the same at every chunk size;
- zltm's column loader: LTA's columns read and checked under LTA's root and stage, refused outside it;
- the refusals: a changed next_n, a carve outside the split, a fit without LTA's block, lists built with a limit;
- the report.

## 10. zlta.py

zlta.py runs zltm.py's fit of zrckpn under LTA's names:

- zltm's fits, declaration, stage, stamp key, naming and code list become LTA's (`on_zltm`);
- ltmcols' output root, stage and declaration become LTA's (`on_cols`), so zltm's carve loads and checks LTA's columns
  under `outputs/lta` as it loads LTM's;
- zml1s's stamp is wrapped (`lta_stamp`): LTM's fit of the seed and split is what an LTA fit is against, and a train or
  read whose NEXT block is not LTA's stops unstamped;
- every name is put back afterwards, also on a failure.

Its compares: zrckpa against zrckpn of the same seed and split (`scr-lta-zrckpa-vs-zrckpn-s<seed>[-hp]`), through zwq's
run_compare, stamped with LTA's block.

Its calls (`outputs/lta/calls.json` and `.md`): zrckpn's three seeds from LTM's compares and musique's read files;
section 5's screen and full run (zltm's rule) with section 5's INCOMPLETE checks; section 7's next form; the beside of
section 6.

`zlta.py --selftest` passed on the laptop's CPU before this commit. It covers:

- names; LTA's fits are none of ML1's or LTM's, and each is against LTM's fit of its seed and split;
- the patches and their restoring, also on a failure; the code stamp with zlta.py and ltacols.py beside LTM's files;
- the stamp: against and naming set, and its refusals (another stage's NEXT block, a block without a scorer, a name not
  LTA's);
- zltm's column loader at call time: LTA's toy columns read and checked under LTA's root, refused outside it, and LTM's
  refused inside it;
- on made compares and read files: zrckpn's three seeds, LTA's reads, the slices, a read file off its compare, a compare
  of another base, and the calls with every table drawn;
- the rule on made numbers (NO_GAIN, ADOPT_FOR_FULL_RUN, NOT_ADOPTED, ADOPT) and the next form.
