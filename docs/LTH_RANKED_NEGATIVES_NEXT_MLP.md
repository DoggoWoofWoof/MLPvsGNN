# LTH: zrckpn with LTA's next-hop columns, the scorer's negatives drawn by rrf's rank, on PKO's pools, on the host's CPU

Declared 3 November 2026. This is after LTA's screen was filed (6b8b786, NOT_ADOPTED) and GNX's (ec09b5a, NO_GAIN), and
before any LTH list, scorer fit, column or MLP fit. Stage modules, both committed with this file:

- `outputs/mp_unified/lthcols.py` (the A lists, the scorer's fits and the columns). It is ltacols.py with LTA's names
  made LTH's and one change, the A lists' draw (section 2).
- `outputs/mp_unified/zlth.py` (the MLP's fits, compares and calls). It is zlta.py with LTA's names made LTH's, plus
  section 7's next form and LTA's filed screen beside.

They import `ltmcols.py`, `ltwdiag.py`, `chndiag.py`, `zltm.py`, `zwq.py`, `zml1.py`, `zml1s.py`, `zdob.py`, `zp1c3p.py`,
`zdep.py` and `zltg.py`, and edit none of them. `ltacols.py` and `zlta.py` are neither imported nor edited.

The following are read only: LTM's lists, fold fits, columns, MLP fits and compares; LTA's scorer fits, column records
and calls (for the report and the calls' beside); PKO's cache and links. LTH writes only under `outputs/lth` and under
its own fit and compare names (`scr-lth-*`).

## 1. The question

- **LTA's screen** (6b8b786): NOT_ADOPTED. It had 0 GAIN, 1 LOSS and 9 WITHIN. The LOSS was WQ-hotpotqa's musique 4-hop
  slice, -0.0111 against a 0.0075 floor.
- **LTA's report: where its term stops.** T2 is the share of the A cases' targets that the scorer ranks in its top two
  among every row outside rrf's top five (LTA section 6):

  | carve (split) | LTA bil | linked targets | unlinked targets | rrf | LTM, among linked rows |
  | --- | ---: | ---: | ---: | ---: | ---: |
  | musique s1eval (WQ-hotpotqa) | 0.179 | 0.346 | 0.084 | 0.067 | 0.493 |
  | musique s1fit (WQ-hotpotqa) | 0.174 | 0.331 | 0.096 | 0.066 | 0.542 |
  | musique s1eval (WQ-musique) | 0.172 | 0.356 | 0.067 | 0.067 | 0.454 |

  On hotpotqa and 2wiki, LTA's bil T2 is 0.47 to 0.64 against rrf's 0.05 to 0.10, and even their unlinked targets
  rise (0.12 to 0.21). On musique, where 64 to 67% of the
  targets are not linked to the anchor, the scorer ranks unlinked targets about as rrf does (0.067 to 0.096 against
  0.066 to 0.067). Over unlinked rows the term adds almost nothing to the first stage's own order.
- **LTA section 8 named the likely cause** ("Easy negatives"). A list's negatives are drawn uniformly from every row
  outside the top five. On musique that is about 5,450 rows per question, and every musique A list is capped (1,748 of
  1,748 on s1eval, 3,022 of 3,022 on s1fit). So about 250 rows fill a list, and almost all of them lie far below the
  top. The scorer learns to tell golds from far rows, not from the rows that compete with them near the top.
- **LTA section 7 names this form:** "LTA's rows with each A list's negatives drawn from rrf's leading rows outside the
  top five, not uniformly."
- **The user's direction (3 November):** "dont keep training and wasting time uselessly unless we have identified why
  we arent able to be better and we have directed training rounds for it". This round is directed by the diagnosis
  above. It changes one thing, which negatives the scorer is trained against.
- **So the question is:** with the scorer's negatives drawn toward rrf's leading rows outside the top five, does
  zrckp's in-domain R@5 rise over zrckpn's, above all on musique's 3- and 4-hop questions?

## 2. The columns (`lthcols.py`)

LTA's section 2 applies unchanged, except for the A lists' draw.

- **The draw.** An A list is stored capped as LTA's is: every target, then negatives drawn without replacement up to
  LTW's cap (256). They come from the same stream (`default_rng([SEED, dataset, 1])`, in question order).
  - A negative's weight is 1 / r, where r is its rank by rrf among the question's rows. Every candidate's r is 6 or
    more, since the top five are the anchors.
  - The draw is `Generator.choice(negatives, m, replace=False, p=w / w.sum())` (`draw()`).
  - A list with no more candidates than the cap is LTA's own (every row), so the draw does not touch it.
- **What the draw does**, on made lists of musique's size (5,450 candidates, 250 drawn; 400 draws, no data read):

  | rrf rank | 6-10 | 11-20 | 21-40 | 41-100 | 101-300 | 301-1,000 | past 1,000 |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | drawn share | 1.00 | 0.95 | 0.80 | 0.52 | 0.24 | 0.08 | 0.02 |

  A uniform draw takes 0.046 everywhere. The drawn ranks' median is 422, against a uniform draw's 2,730. The tail is
  thinned but not cut, so the scorer still sees far rows, but most of its negatives are the rows rrf puts just below
  the top five.
- **The rest is LTA's:**
  - the questions, folds and question vectors (LTM's, checked against LTM's lists);
  - the scorer (LTW's bil, refitted per split on the A lists of its four training carves with LTM's folds);
  - the columns (LTM's seven, LTM's layout, scored over every row outside the top five);
  - the chunked scoring and every build's checks.

  Only the scorer's training lists change. The scored rows and the columns' layout do not.
- **The census adds** the stored negatives' rrf rank (its distribution over each carve's lists).

## 3. What runs

1. Eleven list sets, one per carve of LTM's lists.
2. Two scorer fits, one per split, on its four training carves.
3. Twenty column sets, ten per split: its four training carves and the six s1eval carves. Then the report, which decides
   nothing.
4. zrckph at seed 0 on both splits: zltm's fit of zrckpn with LTH's columns in NEXT's place.
   - The model, chain carve, pick, trainer, read cap, device and threads are zrckpn's.
   - zrckph has zrckpn's inputs and shapes, so at a seed both start from the same draw.
5. The compares (zrckph against zrckpn of the same seed and split), the calls, and the gate.
6. On ADOPT_FOR_FULL_RUN, the full run: seeds 1 and 2 on both splits.

## 4. The reads

DEP's section 3, unchanged, as in LTM's section 5 and LTA's section 4:

- ML1's twelve reads;
- musique's 3-hop and 4-hop slices on both splits.

## 5. The rule (fixed now)

LTA's section 5, with zrckph in zrckpa's place. The base is LTM's zrckpn (seeds 0, 1 and 2), not LTA's zrckpa.

- **Deciding (ten):** the eight reads of datasets their split trains on, plus the WQ-hotpotqa fits' musique 3-hop and
  4-hop slices.
- **Beside (six):** the four zero-shot reads and WQ-musique's slices.
- **The screen (seed 0).** Each deciding read is called on its own:
  - the floor is max(0.0075, 2 x zrckpn's spread over its three seeds), the spread being the root mean square of its
    seed-1 and seed-2 differences from its seed 0;
  - LOSS: zrckph - zrckpn at seed 0 is at or below minus the floor. GAIN: it is at or above the floor;
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
  - its train or read loaded columns other than LTH's, or of more than one scorer, or of a scorer other than the split's
    present LTH fit;
  - a training carve's columns are missing from its train, or an eval carve's from its read;
  - a column record changed after the fit read it;
  - its code differs from LTM's fit's in any file that fit stamped, or zlth.py and lthcols.py are not stamped;
  - its scorer and columns were built by an lthcols.py other than the one its code stamp names;
  - a compare's base R@5 at a seed is not zrckpn's at that seed, or musique's read file does not give its compare's R@5.

A verdict closes only the form it tested.

## 6. Reported beside, deciding nothing

- The six beside reads, called the same way.
- Per carve, the A cases' T2 under bil among every row outside the top five, by hop count on musique:
  - its linked and unlinked targets apart;
  - LTA's T2 on the same cases (uniform negatives), the same way;
  - rrf's T2 on the same cases, as the first stage's own order of those rows;
  - LTM's A T2 among linked rows. Its lists are of other lengths, so the two are not one measure.
- LTH's lists: A cases, their targets' linked share, candidates per list, lists capped, and the negatives' rrf rank.
- The fold fits' stop T2 of LTM, LTA and LTH.
- The rows with an anchor's link (next_n above 0), and next_max's mean on gold and other rows.
- LTA's filed zrckpa - zrckpn at seed 0.
- Seconds per scorer fit, column set and MLP fit.

## 7. What follows (fixed now)

| outcome | what follows |
| --- | --- |
| ADOPT_FOR_FULL_RUN | The full run, queued now behind `lth-gate`, then section 5's full-run rule. **On ADOPT,** zrckph is the MLP candidate in zrckpn's place. It is measured again for Claim 1 (C1R's intervals), and Claim 3's cold timing counts its columns' cost (every pool row scored against five anchors, and the scorer), measured, not assumed. The next form, in its own file: LTH's draw with the term on two-link rows (LTM's lists widened by one link: each anchor's term on the rows two links from it). **On NOT_ADOPTED,** as the next row. |
| NO_GAIN or NOT_ADOPTED | The ranked negatives close. The next form, in its own file: the term on two-link rows (LTM's lists widened by one link: each anchor's term on the rows two links from it). |
| INCOMPLETE | What is missing is fixed in an amendment, before any number is read. |

- A full run not read by the freeze (6 to 7 November) leaves the candidate as it stands then.
- No other stage waits on LTH. MPP's, DOB's and STK's items do not name it, and the GNN's forms do not read its columns.

## 8. Risks, stated before any number

- **Hard negatives may hide golds.** Rows ranked just below the top five include the dataset's chosen distractors. On
  musique, these were picked to look like the question's passages. They are real negatives under the dataset's labels,
  but some may answer a hop the labels do not mark. The scorer then learns to push down rows the MLP would rather see.
  The MLP still reads rrf's own order beside the columns.
- **A biased draw.** The scorer is trained on lists weighted toward rrf's top and scores every row outside the top five.
  So its scores on far rows rest on fewer negatives than LTA's did, and the log-ratio's scale on those rows may shift.
  The columns keep LTM's clip (+-8), and the MLP's own fit decides how much it uses them.
- **The same scorer form.** If unlinked golds are not told apart by the vectors, ranks and scores at all, harder
  negatives will not change that. A NO_GAIN would say the term's inputs, not its negatives, are where it stops.
  Section 7's next form then adds link bits from one more link.
- **Seed 0 only in the screen.** The floor is zrckpn's seed spread although both fits start from one draw, so the screen
  is conservative.
- **Time.** Measured on LTA's items, which have the same sizes:
  - Lists: LTA's eleven started 06:36 to 06:41 IST and were done before its fits started at 07:03.
  - Scorer fits: at most 12 minutes (WQ-hotpotqa) and at most 42 minutes (WQ-musique), from their start to their
    split's first column set.
  - Columns: LTA's twenty started from 07:15 to 08:24 and were done before its report started at 09:20.
  - zrckpa's fits: s0 took 2,518 s to train and 3,537 s to read; s0-hp took 4,740 s and 3,271 s.

  LTH's screen should be read late on 3 November or early on 4 November. Its MLP fits start when MPP's two fits free
  their memory (12 CPUs and about 71 GB at this declaration). The full run, if the screen passes, should be read by
  5 November, before the freeze.
- **Memory.**
  - Lists: LTA's lists' measured peaks plus 1 GB.
  - Scorer fits: 13 GB, as LTA's.
  - Columns: LTA's measured peaks plus 3 GB.
  - Screen MLP fits: LTA's zrckpa fits' measured peaks. Full-run fits: zrckpn's measured peaks of the same seed, as LTA's
    were.

  The feeder adds its 15% margin to each.
- **Disk.** About 3 GB under `outputs/lth` on the host, LTA's size. 60.8 GB was free at this declaration.

## 9. How it runs

Feeder items on the host's CPU; none touches the GPU. They sit after MPP's items and ahead of DOB's remaining x-fits
(`dob-xseed-s0-hp` and after), so LTH's lists, scorer fits and columns use the CPUs that MPP's fits leave free.

1. `lth-selftest-cols` (`lthcols.py --selftest`, 1 CPU, 3 GB) and `lth-selftest` (`zlth.py --selftest`, 1 CPU, 4 GB).
2. The eleven list sets, `lth-lists-<dataset>-<carve>`. They run after `lth-selftest-cols` and LTM's lists of the carve,
   with 1 CPU at the LTA lists item's measured peak plus 1 GB.
3. The scorer fits, `lth-fit-wqm` and `lth-fit-wqh`. They run after their split's four training list sets, with 2 CPUs
   and 13 GB.
4. The twenty column sets, `lth-cols-<wqm|wqh>-<dataset>-<carve>`. They run after their split's fit, the carve's lists
   and LTM's same column set, with 2 CPUs at LTA's same column set's measured peak plus 3 GB. Then `lth-report`.
5. The screen's fits, `lth-zrckph-s0-hp`, then `lth-zrckph-s0`. They run after `pkoh-gate`, `lth-selftest` and their
   split's ten column sets, with 6 CPUs at LTA's zrckpa fits' peaks.
6. `lth-compare`, then `lth-calls`, then `lth-gate` (`outputs/lth/calls.json`, `screen.verdict` ADOPT_FOR_FULL_RUN).
7. The full run, after `pkoh-gate` and `lth-gate`: `lth-zrckph-s1-hp`, `-s2-hp`, `-s1` and `-s2`, at zrckpn's peaks of
   the same seed. Then `lth-compare-full` and `lth-calls-full`.

```
python outputs/mp_unified/lthcols.py lists --dataset musique --carve s1fit
python outputs/host_ops/retry_cmd.py --tries 3 --wait 90 -- python outputs/mp_unified/lthcols.py fit --split WQ-hotpotqa --host
python outputs/host_ops/retry_cmd.py --tries 3 --wait 90 -- python outputs/mp_unified/lthcols.py cols --split WQ-hotpotqa --dataset musique --carve s1eval --host
python outputs/mp_unified/lthcols.py report                         (outputs/lth/report.json and .md)
python outputs/host_ops/retry_cmd.py --tries 3 --wait 90 -- python outputs/mp_unified/zlth.py fit --name scr-lth-zrckph-s0-hp
python outputs/mp_unified/zlth.py compare-all
python outputs/mp_unified/zlth.py calls                             (outputs/lth/calls.json and .md)
```

Only `outputs/lth/calls.json`, `calls.md`, `report.json` and `report.md` are committed. The lists, fits and columns stay
on the host.

`lthcols.py --selftest` passed on the laptop before this commit. It covers LTA's self-test under LTH's names, plus:

- every capped list re-drawn by hand with 1 / rank weights from the same stream, equal to LTH's stored list;
- every capped list unlike LTW's uniform draw (`lists_of`) on the same list, with its kept negatives' mean rrf rank
  below the uniform draw's;
- the census's negatives' rrf rank;
- the record's draw (`1/rank`);
- the report with and without LTA's records: LTA's T2 beside LTH's and LTM's, and the fold-stop table of all three.

## 10. zlth.py

zlth.py runs zltm.py's fit of zrckpn under LTH's names, as zlta.py does under LTA's (LTA section 10):

- zltm's fits, declaration, stage, stamp key, naming and code list become LTH's (`on_zltm`);
- ltmcols' output root, stage and declaration become LTH's (`on_cols`), so zltm's carve loads and checks LTH's columns
  under `outputs/lth`;
- zml1s's stamp is wrapped (`lth_stamp`): LTM's fit of the seed and split is what an LTH fit is against, and a train or
  read whose NEXT block is not LTH's stops unstamped;
- every name is put back afterwards, also on a failure.

Its compares: zrckph against zrckpn of the same seed and split (`scr-lth-zrckph-vs-zrckpn-s<seed>[-hp]`), through zwq's
run_compare, stamped with LTH's block.

Its calls (`outputs/lth/calls.json` and `.md`) hold:

- zrckpn's three seeds, from LTM's compares and musique's read files;
- section 5's screen and full run (zltm's rule), with section 5's INCOMPLETE checks;
- section 7's next form;
- the beside of section 6, with LTA's T2 and LTA's filed screen.

`zlth.py --selftest` passed on the laptop's CPU before this commit. It covers zlta.py's self-test under LTH's names, plus:

- LTA's filed screen beside: absent, then made, read back as filed and drawn in the md;
- section 7's next form for each verdict.
