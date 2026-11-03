# LTS: zrckpn with its next-hop columns on the span residual (the question's part off all five found passages), on PKO's pools, on the host's CPU

Declared 3 November 2026, after LTC's screen was filed (NO_GAIN, 50a2c0e) and before any LTS top row, scorer fit, column
or MLP fit. Stage modules, both committed with this file:

- `outputs/mp_unified/ltscols.py` (the top rows, the scorer's fits and the columns);
- `outputs/mp_unified/zlts.py` (the MLP's fits, compares and calls).

They import `ltmcols.py`, `ltwdiag.py`, `chndiag.py`, `zltm.py`, `zwq.py`, `zml1.py`, `zml1s.py`, `zdob.py`, `zp1c3p.py`,
`zdep.py` and `zltg.py`, and edit none of them. LTM's lists, fold fits, columns, MLP fits and compares are read only, as
are PKO's cache and links. LTS writes only under `outputs/lts` and under its own fit and compare names (`scr-lts-*`).

## 1. The question

- **Where LTC stops.** LTC (docs/LTC_CHAINED_BEAM_MLP.md) chained LTM's next-hop scorer two and three links out. Its
  screen is NO_GAIN (0 GAIN, 0 LOSS, 10 WITHIN against DOB's base), and zrckpc - zrckpn on WQ-hotpotqa's musique is
  -0.0055 at seed 0.
- **What its rule names next.** LTC's chain report gave depth 2's AUC 0.8638 against depth 1's 0.8506, so its
  section 7 (a) does not apply and (b) does: "the next form is the cumulative residual as columns (LTM section 8 (b)):
  each hop's query takes out the span of every passage on the path."
- **LTM section 8 (b)** words the form at hop one: "each anchor's query takes out the span of all five anchors, not its
  own passage alone."
- **Why hop one, and why against zrckpn.** LTM's full run is ADOPT, so zrckpn is the MLP candidate. LTS changes one thing
  in it: the question's residual inside NEXT's scorer. The chain is not carried, since it did not gain; putting the span
  into the chain at once would test two changes against zrckpn.
- **The user's direction (3 November):** "we need to close the gaps that we have no matter what", after "try whatever
  can fix it, KL div, ELBO, EM, iterative refinement of the vectors". Taking the found passages out of the question
  refines the question's vector by what the first stage already holds.
- **So the question is:** with NEXT's scorer reading the question's part off the span of the first stage's top five rows,
  does zrckp's in-domain R@5 rise over zrckpn's, above all on musique's 3- and 4-hop questions?

## 2. The columns (`ltscols.py`)

- **Lists and folds are LTM's**, read only: each question's top five rows by rrf; the T lists (each top row as the
  anchor, its linked rows outside the top five as candidates); the A lists (the first gold among the top five as the
  anchor). LTM's five folds and its cross-fitting are kept.
- **The top rows** of each question are read again from PKO's cache parts in rrf's order, as LTM reads them (the parts'
  rrf column, `chndiag.order_of`). They are checked against LTM's lists: the questions, the parts, every T list's anchor
  by position and node, and every A list's anchor among its question's top rows. A carve's top rows are refused once
  LTM's lists of it change.
- **The span.** For a list with anchor a, the path is a, then the question's other top rows in rrf's order (the frozen
  encoder's unit rows). E is their orthonormal basis: Gram-Schmidt in float64, each row's part off the basis taken twice,
  and a part of norm at most 1e-6 adds nothing. Then p = q - E^T E q.
- **Two of LTW's inputs change:**
  - res = x_c . p, in place of LTW's x_c . q - (x_a . q)(x_a . x_c) (the residual against the anchor alone);
  - r = p / |p|, in place of the anchor-alone residual, normalised.

  Every other input is LTW's. A basis of one row keeps LTW's own res and r, bit for bit.
- **One p per question.** The span of a question's top rows is the same for each of its anchors, so res no longer varies
  with the anchor. The anchor still enters through LTW's other inputs.
- **The scorer** is LTW's bil form, refitted per split on LTM's lists and folds with the span's inputs (ltmcols' fit,
  unchanged: five folds, LTM's epochs, stops and MIN_LISTS).
- **The columns** are LTM's seven, with LTM's layout and checks: next_1 to next_5, next_max and next_n; the log-ratio to
  uniform clipped to +-8, and -8 off a list; out of fold on a training carve, the drawn fold on any other carve.
- **Checks in every build.** Besides LTM's (the cache parts and LTM's lists by sha256, each list's cosine identity):
  - the span's read takes ltwdiag's own read of the chunk first and must rebuild its anchor rows and float16 candidates
    exactly;
  - every anchor must be one of its question's top rows;
  - a fit and its columns carry the ltscols.py that built them, and a column set is refused unless its split's fit is
    LTS's under the present ltscols.py.

## 3. What runs

1. Eleven top-row sets, one per carve of LTM's lists.
2. Two scorer fits, one per split, on its four training carves.
3. Twenty column sets, ten per split (its four training carves and the six s1eval carves). Then the report, which
   decides nothing.
4. zrckps at seed 0 on both splits: zltm's fit of zrckpn with LTS's columns in NEXT's place. The model, chain carve,
   pick, trainer, read cap, device and threads are zrckpn's. zrckps has zrckpn's inputs and shapes, so at a seed both
   start from the same draw.
5. The compares (zrckps against zrckpn of the same seed and split), the calls, and the gate.
6. On ADOPT_FOR_FULL_RUN, the full run: seeds 1 and 2 on both splits.

## 4. The reads

DEP's section 3, unchanged, as in LTM's section 5:

- ML1's twelve reads;
- musique's 3-hop and 4-hop slices on both splits.

## 5. The rule (fixed now)

LTM's section 6, with zrckps in zrckpn's place and LTM's zrckpn (seeds 0, 1 and 2) in DOB's base's place.

- **Deciding (ten):** the eight reads of datasets their split trains on, and the WQ-hotpotqa fits' musique 3-hop and
  4-hop slices.
- **Beside (six):** the four zero-shot reads and WQ-musique's slices.
- **The screen (seed 0).** Each deciding read is called on its own:
  - the floor is max(0.0075, 2 x zrckpn's spread over its three seeds), the spread being the root mean square of its
    seed-1 and seed-2 differences from its seed 0;
  - LOSS: zrckps - zrckpn at seed 0 is at or below minus the floor. GAIN: at or above the floor;
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
  - its train or read loaded columns other than LTS's, or of more than one scorer, or of a scorer other than the split's
    present LTS fit;
  - a training carve's columns are missing from its train, or an eval carve's from its read;
  - a column record changed after the fit read it;
  - its code differs from LTM's fit's in any file that fit stamped, or zlts.py and ltscols.py are not stamped;
  - its scorer and columns were built by an ltscols.py other than the one its code stamp names;
  - a compare's base R@5 at a seed is not zrckpn's at that seed, or musique's read file does not give its compare's R@5.

A verdict closes only the form it tested.

## 6. Reported beside, deciding nothing

- The six beside reads, called the same way.
- Per carve, the A lists' T2 under bil, LTS's scorer beside LTM's, and by hop count on musique (LTM's WQ-hotpotqa
  musique/s1eval: 0.493; 2-hop 0.549, 3-hop 0.420, 4-hop 0.387).
- LTS's lists by the size of their basis.
- LTC's filed zrckpc - zrckpn per read.
- Seconds per fit.

## 7. What follows (fixed now)

| outcome | what follows |
| --- | --- |
| ADOPT_FOR_FULL_RUN | The full run, queued now behind `lts-gate`, then section 5's full-run rule. **On ADOPT,** zrckps is the MLP candidate in zrckpn's place. It is measured again for Claim 1 (C1R's intervals), and Claim 3's cold timing counts its columns' cost (the top rows, the span and the scorer), measured, not assumed. The next form, in its own file: the span at every hop of LTC's chain, each expansion's question taking out the span of the path so far. **On NOT_ADOPTED,** as the next row. |
| NO_GAIN or NOT_ADOPTED | The span at hop one closes. The next form, in its own file, is LTM section 8 (c): "the term over every pool row against each anchor, linked or not, since musique's lists hold 0.045 of its rows." |
| INCOMPLETE | What is missing is fixed in an amendment, before any number is read. |

- A full run not read by the freeze (6 to 7 November) leaves zrckpn the MLP candidate.
- No other stage waits on LTS: RWT's and STK's items do not name it.

## 8. Risks, stated before any number

- **One p per question.** res is the same for a row under each of its question's anchors. If the next hop is told apart
  by the anchor it follows, the span loses that, and only LTW's other inputs carry it.
- **Hop one only.** A 3- or 4-hop question's last passage may be two links from a found one. This form does not reach
  it (LTC's chain did, without gain).
- **Near-repeats among the top rows.** A top row inside the span of those before it adds nothing (norm at most 1e-6),
  so some lists' basis is smaller than five. The lists by basis size are reported.
- **The scorer.** It is refitted on the same lists and folds. Its stops may fall at other epochs than LTM's.
- **Coverage.** The columns change only rows on LTM's lists: about 0.045 of musique's rows, as for NEXT.
- **Seed 0 only in the screen.** The floor is zrckpn's seed spread although both fits start from one draw, so the screen
  is conservative.
- **Time.**
  - Top rows: minutes per carve. Scorer fits: LTM's took 250 s (WQ-musique) and 268 s (WQ-hotpotqa), and the span adds a
    Gram-Schmidt per list. Columns: LTM's took 19 to 104 s per carve.
  - zrckps's fits should take as long as zrckpn's, about 3.5 to 4 h each. They start when RWT's two fits free their
    memory (about 08:00 IST, 3 November), so the screen should be read about 12:00 to 13:00 IST.
  - If the screen passes, the full run should be read by 4 November, before the freeze.
- **Memory.** Top rows at LTM's lists' measured peaks; scorer fits and columns at LTM's measured peaks plus 0.5 GB; the
  MLP fits at zrckpn's measured peaks (the feeder adds its 15% margin to each).
- **Disk.** 218 million rows x 7 float16 columns: about 3 GB under `outputs/lts` on the host (78.4 GB free at this
  declaration).

## 9. How it runs

Feeder items on the host's CPU. None touches the GPU. They sit ahead of DOB's remaining x-fits.

1. `lts-selftest-cols` (`ltscols.py --selftest`, 1 CPU, 3 GB) and `lts-selftest` (`zlts.py --selftest`, 1 CPU, 4 GB).
2. The eleven top-row sets, `lts-tops-<dataset>-<carve>`, after `lts-selftest-cols` and LTM's lists of the carve
   (1 CPU, at the lists item's measured peak).
3. The scorer fits, `lts-fit-wqm` and `lts-fit-wqh`, after their split's four training top-row sets (2 CPUs, at LTM's
   fit's peak plus 0.5 GB).
4. The twenty column sets, `lts-cols-<wqm|wqh>-<dataset>-<carve>`, after their split's fit and the carve's top rows
   (2 CPUs, at LTM's same column set's peak plus 0.5 GB). Then `lts-report`.
5. The screen's fits, after `pkoh-gate`, `lts-selftest` and their split's ten column sets (6 CPUs, at zrckpn's peaks):
   `lts-zrckps-s0-hp`, then `lts-zrckps-s0`.
6. `lts-compare`, then `lts-calls`, then `lts-gate` (`outputs/lts/calls.json`, `screen.verdict` ADOPT_FOR_FULL_RUN).
7. The full run, after `pkoh-gate` and `lts-gate`: `lts-zrckps-s1-hp`, `-s2-hp`, `-s1` and `-s2`, at zrckpn's peaks of
   the same seed. Then `lts-compare-full` and `lts-calls-full`.

```
python outputs/mp_unified/ltscols.py tops --dataset musique --carve s1eval
python outputs/host_ops/retry_cmd.py --tries 3 --wait 90 -- python outputs/mp_unified/ltscols.py fit --split WQ-hotpotqa --host
python outputs/host_ops/retry_cmd.py --tries 3 --wait 90 -- python outputs/mp_unified/ltscols.py cols --split WQ-hotpotqa --dataset musique --carve s1eval --host
python outputs/mp_unified/ltscols.py report                         (outputs/lts/report.json and .md)
python outputs/host_ops/retry_cmd.py --tries 3 --wait 90 -- python outputs/mp_unified/zlts.py fit --name scr-lts-zrckps-s0-hp
python outputs/mp_unified/zlts.py compare-all
python outputs/mp_unified/zlts.py calls                             (outputs/lts/calls.json and .md)
```

Only `outputs/lts/calls.json`, `calls.md`, `report.json` and `report.md` are committed. The top rows, fits and columns
stay on the host.

`ltscols.py --selftest` passed on the laptop before this commit. It covers:

- the basis: orthonormal and in order, its projection equal to least squares', and a row in the span or a repeat adding
  nothing;
- on LTW's toy PKO tree, after LTM's lists of every WQ-hotpotqa carve:
  - the top rows in rrf's order, by hand, and against LTM's lists;
  - the span's read: a basis of the anchor alone (and a path of repeats of it) is LTW's read bit for bit; the full span
    changes res and r alone, with r unit and off every path row, and res and r equal to least squares' on the path;
  - an A list's read; an anchor off its question's top rows refused;
- every patched name put back, also on a failure; building lists refused inside the span;
- LTS's fold fits and columns of three carves: LTS's records and blocks under LTS's root, LTM's toy files unchanged, a
  question's columns by hand from the fold without it, and LTM's scorer and columns of the same toys differing;
- the refusals: columns from a fit without LTS's block; top rows whose LTM lists changed; a carve without top rows;
- the report.

## 10. zlts.py

zlts.py runs zltm.py's fit of zrckpn under LTS's names:

- zltm's fits, declaration, stage, stamp key, naming and code list become LTS's (`on_zltm`);
- ltmcols' output root, stage and declaration become LTS's (`on_cols`), so zltm's carve loads and checks LTS's columns
  under `outputs/lts` as it loads LTM's;
- zml1s's stamp is wrapped (`lts_stamp`): LTM's fit of the seed and split is what an LTS fit is against, and a train or
  read whose NEXT block is not LTS's stops unstamped;
- every name is put back afterwards, also on a failure.

Its compares: zrckps against zrckpn of the same seed and split (`scr-lts-zrckps-vs-zrckpn-s<seed>[-hp]`), through zwq's
run_compare, stamped with LTS's block.

Its calls (`outputs/lts/calls.json` and `.md`): zrckpn's three seeds from LTM's compares and musique's read files;
section 5's screen and full run (zltm's rule) with section 5's INCOMPLETE checks; section 7's next form; the beside of
section 6.

`zlts.py --selftest` passed on the laptop's CPU before this commit. It covers:

- names; LTS's fits are none of ML1's or LTM's, and each is against LTM's fit of its seed and split;
- the patches and their restoring, also on a failure; the code stamp with zlts.py and ltscols.py beside LTM's files;
- the stamp: against and naming set, and its refusals (another stage's NEXT block, a block without a scorer, a name not
  LTS's);
- zltm's column loader at call time: LTS's toy columns read and checked under LTS's root, refused outside it, and LTM's
  refused inside it;
- on made compares and read files: zrckpn's three seeds, LTS's reads, the slices, a read file off its compare, a compare
  of another base, and the calls with every table drawn;
- the rule on made numbers (NO_GAIN, ADOPT_FOR_FULL_RUN, NOT_ADOPTED, ADOPT) and the next form.
