# LT2: zrckpn with next-hop columns on the rows within two links of each anchor, the two links as the scorer's inputs, on PKO's pools, on the host's CPU

Declared 3 November 2026. This is after L2D was filed (5f8da44, SIGNAL on musique/s1fit) and before any LT2 list, scorer
fit, column or MLP fit. LTH's MLP fits had not started (they wait for memory), so no LTH number had been read. Stage
modules, both committed with this file:

- `outputs/mp_unified/lt2cols.py` (the A lists within two links, the scorer's fits and the columns). It takes LTH's
  question reader, A cases, LTM-list checks and toy carves from lthcols.py, and LTM's fit, fold loading, log-ratio and
  records from ltmcols.py. Section 2 says what is new.
- `outputs/mp_unified/zlt2.py` (the MLP's fits, compares and calls). It is zlth.py with LTH's names made LT2's, plus
  section 7's next form and LTA's and LTH's filed screens beside.

They import `lthcols.py`, `ltmcols.py`, `ltwdiag.py`, `chndiag.py`, `zltm.py`, `zwq.py`, `zml1.py`, `zml1s.py`,
`zdob.py`, `zp1c3p.py`, `zdep.py` and `zltg.py`, and edit none of them. The self-tests also import `l2ddiag.py`
(lt2cols: its depths, middle counts and thinned links) and `zlth.py` (zlt2: LT2's names are apart from LTH's), and edit
neither.

The following are read only: LTM's lists, fold fits, columns, MLP fits and compares; LTA's census (the A cases' check);
LTA's and LTH's scorer fits, column records and calls (for the report and the calls' beside); PKO's cache and links. LT2
writes only under `outputs/lt2` and under its own fit and compare names (`scr-lt2-*`).

## 1. The question

- **Where LTA's term stops** (LTH section 1). Over every row outside the top five, LTA's scorer ranks musique's unlinked
  targets about as rrf does (T2 0.067 to 0.096 against 0.066 to 0.067). LTH tests harder negatives on the same rows.
- **Where those targets are** (L2D, 5f8da44). On musique/s1fit, with 3,022 A cases, 3,798 targets and 0.332 of the
  targets linked to their anchor:
  - 0.569 of the 2,536 unlinked targets are two links from their anchor, against 0.234 of the unlinked candidate rows.
    The lift against rows of the same rrf decile is 2.42 (bar 1.5), and the middle count's matched AUC is 0.670
    (bar 0.65).
  - With the linked targets, 0.712 of the targets lie within two links of the anchor. A case has a median of 1,059
    two-link rows (p90 2,615), against a median of 5,018 unlinked rows.
  - By hop, the two-link share and lift are 0.689 and 2.96 at 2 hops, 0.510 and 2.11 at 3 hops, and 0.350 and 1.65 at
    4 hops.
  - By family, with each family's links alone: two links reach 0.441 of the targets that family leaves unlinked
    (struct, lift 2.56), 0.090 (ner, lift 4.34) and 0.054 (knn, lift 19.6).
  - The middle count's lift is not monotone: 2.15 with one middle row, 3.46 with two or three, 2.15 with four or more.
  - A fixed order that puts the two-link rows first raises the unlinked targets' T2 from 0.097 (rrf) to 0.129.
- **Beside, not musique.** On hotpotqa/fit, 0.564 of the unlinked targets are two links out, but the matched lift is
  1.25 (matched AUC 0.612). On 2wiki/fit the share is 0.175 and the matched lift 0.45, so there two-link rows are less
  likely to be targets than rows of the same rrf decile. The two-first order lowers both datasets' unlinked T2 (hotpotqa
  0.206 to 0.128, 2wiki 0.112 to 0.019).
- **So the term changes in two ways.**
  - Its candidates become the rows within two links of the anchor. On musique that is about a fifth of LTA's rows, and
    it holds 0.71 of the targets.
  - Its scorer reads the two links as inputs: the families, the middle count, and the best middle row's rrf rank. The
    scorer is fitted per split on all its training datasets, so it can learn where two links help (musique) and where
    they do not (2wiki), with no per-dataset setting.
- **The user's direction (3 November):** "dont keep training and wasting time uselessly unless we have identified why
  we arent able to be better and we have directed training rounds for it". L2D is the identification. This round is
  directed by it.
- **So the question is:** with each anchor's next-hop term on the rows within two links of it, and the two links as the
  scorer's inputs, does zrckp's in-domain R@5 rise over zrckpn's, above all on musique's 3- and 4-hop questions?

## 2. The columns (`lt2cols.py`)

- **The questions and folds are LTM's**, read only: the questions, their cache parts, rrf's top five rows, LTM's five
  folds (training carves) and its drawn folds (other carves), and the question vectors. Each list set is checked
  against LTM's lists of its carve, key by key, and refused on any difference.
- **The links** are PKO's links of the question, taken undirected. Self-links are dropped, and each linked pair is kept
  once in each direction with the OR of its links' family bits (L2D's `sym_edges`, with the bits).
- **Within two links of anchor a** means outside the top five and either:
  - linked to a (one link); or
  - linked to a row that is linked to a (two links).

  A row's middle rows are the rows linked to both a and the row. They may be top-five rows.
- **The packed field.** One int64 per candidate. It is the lists' and the scorer's family field, so ltwdiag reads its
  first three bits as LTW's own link bits, unchanged:

  | bits | what |
  | --- | --- |
  | 0-2 | the families of the row's links to the anchor (LTW's `fam_bits`) |
  | 3 | two links from the anchor and not linked to it |
  | 4-6 | the families of the first links (the anchor to a middle row), over the row's middle rows |
  | 7-9 | the families of the second links (a middle row to the row), over the row's middle rows |
  | 10-29 | the number of middle rows (capped at 2^20 - 1) |
  | 30-49 | the best (lowest) rrf rank among the middle rows, 1-based; the pool size when there is none |

  The field is 0 on the anchor and on rows beyond two links.
- **The scorer's nine new inputs** sit beside LTW's fourteen:
  - `two_link` (bit 3);
  - `l1_struct`, `l1_ner`, `l1_knn` and `l2_struct`, `l2_ner`, `l2_knn` (bits 4-9);
  - `log1p_mid`: log1p of the middle count;
  - `log1p_mid_rank`: log1p of the best middle row's rank.

  They come from links and rrf ranks only. No gold enters them. They reach ltwdiag's own `read_chunk`, `collate`,
  `build_model` and training through a patch of its `FEATS` and `feature_rows` (`on_feats`), and the patch is put back
  afterwards, also on a failure.
- **The A lists.** An A case is LTA's: a question with a gold among the top five and a gold outside them, whose anchor
  is the first gold among the top five.
  - Its candidates are the rows within two links of the anchor. Its targets are the golds among them.
  - A case with no target within two links holds no list. It is counted, and its targets rank last beside (section 6).
  - Each carve's cases, targets and linked targets are checked against LTA's census of the carve. A difference is
    refused.
- **The draw.** The lists are stored whole. ltmcols' fit is unchanged (LTM's folds, epochs, stops and MIN_LISTS). It
  keeps every target, then draws negatives uniformly up to LTW's cap of 256 from its own stream: LTA's draw.
  - A median musique list holds about 1,100 rows, so its 256 keep about a quarter of its negatives. LTA's kept 0.046.
  - Every negative now lies within two links of the anchor.
  - **This departs from L2D section 5**, which gave this term LTH's draw if LTH's full run ADOPTs. That verdict cannot
    land before 5 November. Waiting for it would put this screen at the freeze. So LT2 runs LTA's draw now, and LTH's
    draw on LT2's lists follows in its own file if LTH's full run ADOPTs (section 7).
- **The columns** are LTM's seven, in LTM's layout:
  - `next_1` to `next_5`: a row's log-ratio to uniform among anchor k's candidates (the rows within two links of it),
    clipped to +-8. The value is -8 on the top five rows and on rows beyond two links of anchor k;
  - `next_max`: the largest of the five;
  - `next_n`: LTM's own column, the number of anchors the row is linked to.
- **Scoring.** Out of fold on a training carve, and the drawn fold on any other carve. Rows are read through ltwdiag's
  own `read_chunk`, `items_of` and `score_items` with LT2's inputs, about 40,000 candidate rows at a time. Each
  question's link pairs are built once and serve its five anchors.
- **Checks in every build.** LTA's and LTH's checks run under LT2's names:
  - the cache parts and links by sha256, and each question's rows and part;
  - each list's cosine identity;
  - `next_n` equal to LTM's, bit for bit, on every part where LTM's columns are on disk;
  - one lt2cols.py for a fit and its columns, and the split's fit present.

  LT2 adds:
  - the fit refuses lists that are not LT2's (by their stage), limited lists, and lists with a cap;
  - a column set refuses a fit record without its `lt2` block.

## 3. What runs

1. Eleven list sets, one per carve of LTM's lists.
2. Two scorer fits, one per split, on its four training carves.
3. Twenty column sets, ten per split: its four training carves and the six s1eval carves. Then the report, which decides
   nothing.
4. zrckp2 at seed 0 on both splits: zltm's fit of zrckpn with LT2's columns in NEXT's place.
   - The model, chain carve, pick, trainer, read cap, device and threads are zrckpn's.
   - zrckp2 has zrckpn's inputs and shapes, so at a seed both start from the same draw.
5. The compares (zrckp2 against zrckpn of the same seed and split), the calls, and the gate.
6. On ADOPT_FOR_FULL_RUN, the full run: seeds 1 and 2 on both splits.

## 4. The reads

DEP's section 3, unchanged, as in LTM's section 5 and LTH's section 4:

- ML1's twelve reads;
- musique's 3-hop and 4-hop slices on both splits.

## 5. The rule (fixed now)

LTH's section 5, with zrckp2 in zrckph's place. The base is LTM's zrckpn (seeds 0, 1 and 2).

- **Deciding (ten):** the eight reads of datasets their split trains on, plus the WQ-hotpotqa fits' musique 3-hop and
  4-hop slices.
- **Beside (six):** the four zero-shot reads and WQ-musique's slices.
- **The screen (seed 0).** Each deciding read is called on its own:
  - the floor is max(0.0075, 2 x zrckpn's spread over its three seeds), the spread being the root mean square of its
    seed-1 and seed-2 differences from its seed 0;
  - LOSS: zrckp2 - zrckpn at seed 0 is at or below minus the floor. GAIN: it is at or above the floor;
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
  - its train or read loaded columns other than LT2's, or of more than one scorer, or of a scorer other than the split's
    present LT2 fit;
  - a training carve's columns are missing from its train, or an eval carve's from its read;
  - a column record changed after the fit read it;
  - its code differs from LTM's fit's in any file that fit stamped, or zlt2.py, lt2cols.py and lthcols.py are not
    stamped;
  - its scorer and columns were built by an lt2cols.py other than the one its code stamp names;
  - a compare's base R@5 at a seed is not zrckpn's at that seed, or musique's read file does not give its compare's R@5.

A verdict closes only the form it tested.

## 6. Reported beside, deciding nothing

- The six beside reads, called the same way.
- Per carve, the A cases' T2 under bil among the rows within two links of the anchor. Targets beyond two links rank last
  (at the pool size). On musique it is also given by hop count. Beside it:
  - its linked and two-link targets apart;
  - rrf's T2 on the same cases and rows, as the first stage's own order of those rows;
  - LTH's and LTA's T2 among every row outside the top five, and LTM's among linked rows. Their lists are of other
    lengths, so these are not one measure.
- **Section 7's read:** on WQ-hotpotqa's musique/s1eval (musique is in-domain there), the two-link targets' T2 under
  LT2's scorer against rrf's among the same rows.
- LT2's lists:
  - A cases, and cases without a target within two links;
  - the targets' share within two links, linked and two-link apart;
  - candidates per list, and the two-link share of candidates;
  - middle rows per two-link row;
  - the negatives' rrf rank.
- The fold fits' stop T2 of LTM, LTA, LTH and LT2.
- Rows scored, rows within two links of an anchor, and the targets' depth shares (one link, two links, beyond).
- LTA's and LTH's filed screens against zrckpn at seed 0.
- Seconds per scorer fit, column set and MLP fit.

## 7. What follows (fixed now)

The names used below:

- **The beam through the middle.** Each two-link row's best middle row is scored by its anchor's one-link term (LTM's
  scorer), and that score is an input to the two-link term: one round of refinement.
- **The path's vectors.** Each two-link row's middle row nearest the question, with its cosines to the question, the
  anchor and the row, as inputs beside LT2's nine.

| outcome | what follows |
| --- | --- |
| ADOPT_FOR_FULL_RUN | The full run, queued now behind `lt2-gate`, then section 5's full-run rule. **On ADOPT,** zrckp2 is the MLP candidate in zrckpn's place. It is measured again for Claim 1 (C1R's intervals), and Claim 3's cold timing counts its columns' cost (each question's link pairs, five two-link fields and the scorer), measured, not assumed. The next form, in its own file: the beam through the middle on LT2's lists; and, if LTH's full run ADOPTs, LTH's draw on LT2's lists. **On NOT_ADOPTED,** as the next row. |
| NO_GAIN or NOT_ADOPTED | The two-link lists close as inputs. The next form, in its own file, follows section 6's read. If LT2's scorer ranks the two-link targets above rrf among the same rows, the term is there but did not lift the MLP, and the next form is the beam through the middle. If not, the two links' shape does not separate them, and the next form is the path's vectors. |
| INCOMPLETE | What is missing is fixed in an amendment, before any number is read. |

- **If LTH's and LT2's full runs both ADOPT,** the MLP candidate is the one with more GAIN reads among the ten deciding
  reads. A tie goes to the larger mean delta over the eight dataset reads.
- A full run not read by the freeze (6 to 7 November) leaves the candidate as it stands then.
- No other stage waits on LT2. DOB's and STK's items do not name it, and the GNN's forms do not read its columns.

## 8. Risks, stated before any number

- **2wiki and hotpotqa.** On 2wiki, two-link rows are less likely to be targets than rows of the same rrf decile
  (matched lift 0.45), and 0.32 of its targets lie beyond two links of their anchor (0.42 of its unlinked targets are
  three links out and 0.40 farther). Those targets get -8 in their anchor's column. LTM's columns already give -8 there
  to the 0.39 of 2wiki's targets not linked to their anchor, so LT2 gives it to fewer, not more. Still, 2wiki's and hotpotqa's two-link rows may teach
  the scorer little, and the MLP may learn to discount the columns there. Their dataset reads decide as musique's do.
- **A shared scorer.** One scorer per split serves every dataset, with no per-dataset setting. If musique's two-link
  pattern and 2wiki's pull its weights apart, the scorer may settle between them. Its inputs let it tell the datasets'
  link shapes apart, since the families and middle counts differ between them, but nothing forces it to.
- **Hub rows.** On musique a row has about 33 links on average (the median case), so a few hub rows put many rows two
  links out. The
  middle count's lift falls back from 3.46 (two or three middles) to 2.15 (four or more), and log1p of the count is the
  scorer's only handle on hubs.
- **The uniform draw.** A list keeps every target and about a quarter of its negatives. The draw is not toward rrf's top,
  so the scorer may still learn far-from-top rows first. LTH's verdict says whether the draw matters, and section 7 then
  carries it to LT2's lists.
- **Depth three is excluded.** On musique, 0.335 of the unlinked targets are three links out. They rank last in this
  term, as before. A three-link term would hold most rows again, which is LTA's case.
- **musique's 4-hop slice.** Its floor is the 0.0075 minimum over few questions, and it is the read LTA's screen lost
  on (-0.0111). L2D puts the fewest 4-hop targets two links out (0.350, lift 1.65), so the term helps least there. A
  small shift in how the MLP weighs the columns can still call a LOSS there. (zrckpn's webqsp floors are wide, 0.0146
  and 0.0191.)
- **Seed 0 only in the screen.** The floor is zrckpn's seed spread although both fits start from one draw, so the screen
  is conservative.
- **Time.** Measured on LTH's items, which read the same parts:
  - Lists: LTH's eleven were sent from 13:32 to 13:44 IST and were done by 13:58, when its second scorer fit was sent.
    LT2's lists add one two-link field per A case.
  - Scorer fits: about 13 minutes each, from the fit being sent to its split's first column set.
  - Columns: LTH's twenty were sent from 13:58 to 15:19, sharing the CPUs, and its report started at 16:29. LT2 scores
    only the rows within two links, about a fifth of LTA's on musique and about a third on 2wiki and hotpotqa.
  - zrckpa's fits: s0 took 2,518 s to train and 3,537 s to read; s0-hp took 4,740 s and 3,271 s.

  LT2's screen fits start after LTH's two screen fits free their memory. So its screen should be read late on
  3 November or early on 4 November. The full run, if the screen passes, should be read on 5 November, before the
  freeze.
- **Memory.**
  - Lists: LTA's lists' measured peaks plus 1 GB.
  - Scorer fits: 13 GB, as LTH's.
  - Columns: LTA's measured peaks plus 3 GB.
  - Screen MLP fits: LTA's zrckpa fits' measured peaks. Full-run fits: zrckpn's measured peaks of the same seed.

  The feeder adds its 15% margin to each.
- **Disk.** About 3 GB under `outputs/lt2` on the host, LTH's size. 66.5 GB was free at this declaration.

## 9. How it runs

Feeder items on the host's CPU; none touches the GPU.

1. `lt2-selftest-cols` (`lt2cols.py --selftest`, 1 CPU, 3 GB) and `lt2-selftest` (`zlt2.py --selftest`, 1 CPU, 4 GB).
2. The eleven list sets, `lt2-lists-<dataset>-<carve>`. They run after `lt2-selftest-cols` and LTM's lists of the carve,
   with 1 CPU at LTA's lists item's measured peak plus 1 GB.
3. The scorer fits, `lt2-fit-wqm` and `lt2-fit-wqh`. They run after their split's four training list sets, with 2 CPUs
   and 13 GB.

   Items 1 to 3 sit ahead of LTH's screen fits, which wait for memory. So they use the CPUs that are free now, and
   should end within the hour.
4. The twenty column sets, `lt2-cols-<wqm|wqh>-<dataset>-<carve>`. They run after their split's fit, the carve's lists
   and LTM's same column set, with 2 CPUs at LTA's same column set's measured peak plus 3 GB. Then `lt2-report`.
5. The screen's fits, `lt2-zrckp2-s0-hp`, then `lt2-zrckp2-s0`. They run after `pkoh-gate`, `lt2-selftest` and their
   split's ten column sets, with 6 CPUs at LTA's zrckpa fits' peaks.
6. `lt2-compare`, then `lt2-calls`, then `lt2-gate` (`outputs/lt2/calls.json`, `screen.verdict` ADOPT_FOR_FULL_RUN).

   Items 4 to 6 sit just after LTH's two screen fits, so LTH's fits take the memory that DOB's x-fits free first. A
   screen goes before a full run, so these also sit ahead of LTH's full run.
7. The full run, after `pkoh-gate` and `lt2-gate`: `lt2-zrckp2-s1-hp`, `-s2-hp`, `-s1` and `-s2`, at zrckpn's peaks of
   the same seed. Then `lt2-compare-full` and `lt2-calls-full`. These sit after LTH's full run and ahead of DOB's
   remaining x-fits.

```
python outputs/mp_unified/lt2cols.py lists --dataset musique --carve s1fit
python outputs/host_ops/retry_cmd.py --tries 3 --wait 90 -- python outputs/mp_unified/lt2cols.py fit --split WQ-hotpotqa --host
python outputs/host_ops/retry_cmd.py --tries 3 --wait 90 -- python outputs/mp_unified/lt2cols.py cols --split WQ-hotpotqa --dataset musique --carve s1eval --host
python outputs/mp_unified/lt2cols.py report                         (outputs/lt2/report.json and .md)
python outputs/host_ops/retry_cmd.py --tries 3 --wait 90 -- python outputs/mp_unified/zlt2.py fit --name scr-lt2-zrckp2-s0-hp
python outputs/mp_unified/zlt2.py compare-all
python outputs/mp_unified/zlt2.py calls                             (outputs/lt2/calls.json and .md)
```

Only `outputs/lt2/calls.json`, `calls.md`, `report.json` and `report.md` are committed. The lists, fits and columns stay
on the host.

`lt2cols.py --selftest` passed on the laptop before this commit. It covers:

- **Made graphs** (60 random graphs, three anchors each):
  - every row's packed field equals a by-hand field built from dicts and sets;
  - bits 0-2 equal LTW's `fam_bits`;
  - the rows within two links, and the two-link bit, agree with L2D's depths;
  - the middle count equals L2D's `through` on the rows within two links;
  - two-link rows, rows beyond two links and pairs linked by several families all occur.
- **The feature decode** on a made field, its refusal outside `on_feats`, and the patch put back after a failure.
- **Toy carves** (LTH's `toy_carve`, with links thinned by L2D's `thin_links` so every depth occurs):
  - LT2's lists against LTH's census, equal, with cases = lists + cases without a target within two links;
  - every musique/s1fit question checked by hand;
  - refusals of a census off by one list, a bad LTM fold, and missing LTM lists.
- **The fit:**
  - three datasets, two epochs, with its record;
  - a 23-wide input in the fold checkpoint;
  - every patched name put back, also after a failure.
- **Columns** on three carves at three chunk sizes:
  - the identity, the fit's sha and the training flag;
  - `next_n` against LTM's columns written by hand, and where they are absent;
  - each list's log-ratio by hand (within 8e-3);
  - each anchor's candidates equal to the by-hand field's rows within two links;
  - -8 beyond two links and on the top five;
  - `next_max` as the row maximum;
  - the same columns at 10^9 and at 1,000 rows a chunk (within 4e-3).
- **Refusals:** LTM's `next_n` tampered; a carve outside the split; a fit record without its `lt2` block; limited lists;
  LTH's lists.
- **The report** with LTM's fit beside.

## 10. zlt2.py

zlt2.py runs zltm.py's fit of zrckpn under LT2's names, as zlth.py does under LTH's (LTH section 10):

- zltm's fits, declaration, stage, stamp key, naming and code list become LT2's (`on_zltm`);
- ltmcols' output root, stage and declaration become LT2's (`on_cols`), so zltm's carve loads and checks LT2's columns
  under `outputs/lt2`;
- zml1s's stamp is wrapped (`lt2_stamp`). LTM's fit of the seed and split is what an LT2 fit is against. A train or read
  whose NEXT block is not LT2's (LTM's or LTH's, say) stops unstamped;
- every name is put back afterwards, also on a failure.

Its compares: zrckp2 against zrckpn of the same seed and split (`scr-lt2-zrckp2-vs-zrckpn-s<seed>[-hp]`), through
zwq's run_compare, stamped with LT2's block.

Its calls (`outputs/lt2/calls.json` and `.md`) hold:

- zrckpn's three seeds, from LTM's compares and musique's read files;
- section 5's screen and full run (zltm's rule), with section 5's INCOMPLETE checks;
- section 7's next form and section 6's read;
- the beside of section 6, with LTH's and LTA's T2 and their filed screens.

`zlt2.py --selftest` passed on the laptop's CPU before this commit. It covers zlth.py's self-test under LT2's names,
plus:

- LT2's names apart from LTH's;
- the stamp's refusal of LTH's NEXT block;
- LTA's and LTH's filed screens beside: absent, then made, read back as filed and drawn in the md;
- section 6's read from a made LT2 column record, and the beside table's row;
- section 7's next form for each verdict and each answer of section 6's read.
