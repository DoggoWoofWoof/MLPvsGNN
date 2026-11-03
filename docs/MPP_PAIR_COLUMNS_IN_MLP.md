# MPP: zrckpn with GNP's six pair columns beside its NEXT columns, on PKO's pools, on the host's CPU

Declared 3 November 2026, after PFS A3's filing and GNP's declaration (2f6aaf7) and STK amendment A3 (f7309df). It comes
before any MPP fit and before any GNP fit has been read. Stage module, committed with this file:
`outputs/mp_unified/zmpp.py` (the MLP's fits, compares and calls).

zmpp.py imports `zltm.py`, `ltmcols.py`, `gnpcols.py`, `zpfp.py`, `zml1.py`, `zml1s.py`, `zdob.py`, `zp1c3p.py`,
`zdep.py` and `zltg.py`, and edits none of them. LTM's columns, fits and compares are read only, as are GNP's columns and
PKO's cache and links. MPP writes only under `outputs/mpp` and under its own fit and compare names (`scr-mpp-*`).

## 1. The question

- **What PFS A3 found.** PFS A3 (docs/PKO_FEATURE_SCREEN.md section 12) put six free pair scores of each pool row
  against rrf's top five beside the 172 lean columns, in a logistic probe. It gave PAIR_ADDS on four of five reads:
  - musique +0.0162 R@5 (400 questions, all 2-hop);
  - hotpotqa +0.0170, 2wiki +0.0118, webqsp +0.0380;
  - squad +0.0040, which is PAIR_NOTHING.
- **The order.** PFS A3's call put the GNN first (GNP) and the MLP second, on the CPU, in its own file. This is that file.
- **Two models, not one built on the other.** GNP puts the six columns into zg6r's encoder. MPP puts them into zrckpn,
  the MLP of record (LTM, ADOPT 7874e4d). The two share the columns only. Neither is built on the other, and neither
  verdict waits on the other.
- **No message passing.** The six columns come from three things fixed before any row is scored:
  - the frozen encoder's vectors of the row and of the question;
  - rrf's top five rows;
  - gnpcols.py's rule, which fits nothing.

  The MLP reads them as it reads any cached column, and passes no message.
- **The user's direction (3 November):** "can we finish these any faster, and iterate much faster ... we need to close
  the gaps that we have no matter what."
- **So the question is:** with GNP's six pair columns beside NEXT's seven, does zrckp's in-domain R@5 rise over
  zrckpn's, above all on musique?

## 2. The columns

No column is built here. Both sets are read as they are on the host.

- **LTM's seven**, under `outputs/ltm/cols/<split>/<dataset>/<carve>/` (docs/LTM_NEXT_HOP_MLP.md): next_1 to next_5,
  next_max and next_n, scored by the split's LTM scorer.
- **GNP's six**, under `outputs/gnp/cols/<split>/<dataset>/<carve>/` (docs/GNP_PAIR_COLUMNS_IN_GNN.md section 2): pair_max,
  res_max, pair_mean, res_mean, pair_1 and res_1. The rule fits nothing, so every record's fit_sha256 is one sha256 over
  the rule's code (gnpcols.py, zpfp.py, ltwdiag.py and chndiag.py).
- **The join.** For each carve, each set is loaded and checked under its own stage's names by zltm's own loader. That
  covers each record's stage, declaration, split, carve, columns, training flag and identity, and each part's cache
  record, rows, scorer and file. Then the two are joined row by row, LTM's seven first: 13 float16 columns in zltm's
  NEXT block.
  - Both sets must give the same rows and the same training flag.
  - The block's fit_sha256 is the two scorers' joined by "+", LTM's first. Each carve's record sha256 is the two records'
    joined the same way.
  - A set missing, under another stage's names, or of other rows is refused.

## 3. What runs

1. `zmpp.py --selftest`.
2. zrckpp at seed 0 on both splits. This is zltm's fit of zrckpn with the joined 13 columns in NEXT's place. The model,
   chain carve, pick, trainer, read cap, device and threads are zrckpn's. The first layer takes six more inputs, so
   zrckpp and zrckpn do not start from one draw at a seed.
3. The compares (zrckpp against zrckpn of the same seed and split), the calls, and the gate.
4. On ADOPT_FOR_FULL_RUN, the full run: seeds 1 and 2 on both splits.

## 4. The reads

DEP's section 3, unchanged, as in LTM's section 5 and LTA's section 4:

- ML1's twelve reads;
- musique's 3-hop and 4-hop slices on both splits.

## 5. The rule (fixed now)

LTA's section 5 (LTM's section 6), with zrckpp in zrckpa's place and LTM's zrckpn (seeds 0, 1 and 2) as the base.

- **Deciding (ten):** the eight reads of datasets their split trains on, and the WQ-hotpotqa fits' musique 3-hop and
  4-hop slices.
- **Beside (six):** the four zero-shot reads and WQ-musique's slices.
- **The screen (seed 0).** Each deciding read is called on its own:
  - the floor is max(0.0075, 2 x zrckpn's spread over its three seeds), the spread being the root mean square of its
    seed-1 and seed-2 differences from its seed 0;
  - LOSS: zrckpp - zrckpn at seed 0 is at or below minus the floor. GAIN: at or above the floor;
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
  - its train or read loaded columns other than MPP's join, or its train and read differ in either scorer;
  - the join's LTM scorer is not the split's present LTM fit;
  - the join's GNP rule is not the one that the code stamp's gnpcols.py, zpfp.py, ltwdiag.py and chndiag.py make;
  - a training carve's columns are missing from its train, or an eval carve's from its read;
  - either set's column record changed after the fit read it;
  - its code differs from LTM's fit's in any file that fit stamped, or zmpp.py, gnpcols.py and zpfp.py are not stamped;
  - a compare's base R@5 at a seed is not zrckpn's at that seed, or musique's read file does not give its compare's R@5.

A verdict closes only the form it tested.

## 6. Reported beside, deciding nothing

- The six beside reads, called the same way.
- GNP's, LTA's and LTS's screen verdicts, as filed.
- Seconds per MLP fit.

## 7. What follows (fixed now)

| outcome | what follows |
| --- | --- |
| ADOPT_FOR_FULL_RUN | The full run, queued now behind `mpp-gate`, then section 5's full-run rule. **On ADOPT,** zrckpp is the MLP candidate in zrckpn's place. C1R's intervals measure it again for Claim 1. Claim 3's cold timing counts its columns' cost (six scores of every pool row against five anchors), measured, not assumed. The next form, in its own file: the six pair columns beside the MLP candidate's own NEXT columns, if LTA's or LTS's full run adopts. **On NOT_ADOPTED,** as the next row. |
| NO_GAIN or NOT_ADOPTED | The six pair columns beside NEXT close for the MLP. The next form, in its own file: the same six scores against rrf's top ten rows. |
| INCOMPLETE | What is missing is fixed in an amendment, before any number is read. |

- **If more than one of LTS's, LTA's and MPP's full runs adopt,** the MLP candidate is the one with the largest mean delta
  against zrckpn over the eight dataset reads and three seeds, until the next form's own file is read.
- A full run not read by the freeze (6 to 7 November) leaves the candidate as it stands then.
- No other stage waits on MPP. GNP's, STK's, LTA's and LTS's items do not name it.

## 8. Risks, stated before any number

- **NEXT may carry the same signal.** NEXT's columns and the pair columns both score a row against rrf's top five.
  PFS A3's probe read the pair columns beside the 172 lean columns, without NEXT. Beside NEXT, the gain may be smaller.
- **A probe is not the MLP.** PFS's logistic probe read 400 to 1,000 questions of one training carve per dataset. zrckp
  reads every question of the six s1eval carves, through its own model and pick.
- **musique's probe read was 2-hop only.** No 3-hop or 4-hop question was read, so the deciding slices have no prior.
- **squad.** Its probe delta was PAIR_NOTHING (+0.0040). squad is a deciding read on both splits.
- **Another draw at a seed.** The first layer is wider, so the seed-0 delta carries draw noise as well as the columns'
  effect. The floor's seed spread covers draw noise of that size.
- **Time.** zrckpn's fits took 1 h 43 min (WQ-musique) and 2 h 29 min (WQ-hotpotqa) on the host. zrckpp's should take
  about as long. They start when LTA's two screen fits free their memory. So the screen should be read in the evening of
  3 November. The full run, if the screen passes, should be read by 4 to 5 November, before the freeze.
- **Memory.** The fits get zrckpn's measured peaks plus 2 GB, and the feeder adds its 15% margin. The extra memory
  comes from 13 float16 columns in place of 7: about 0.2 GB more over the training carves, about 0.5 GB more on
  metaqa's s1eval carve (42.8 million rows), and the join's copy.
- **Disk.** Nothing new.

## 9. How it runs

Feeder items on the host's CPU. None touches the GPU.

1. `mpp-selftest` (`zmpp.py --selftest`, 1 CPU, 4 GB), before LTA's first screen fit, so it runs at once.
2. The screen's fits, right after LTA's two: `mpp-zrckpp-s0-hp`, then `mpp-zrckpp-s0`.
   - They run after `pkoh-gate`, `mpp-selftest`, `gnp-check-cols` and `ltm-calls`.
   - Each takes 6 CPUs, at zrckpn's peak of the same split plus 2 GB.
3. `mpp-compare`, then `mpp-calls`, then `mpp-gate` (`outputs/mpp/calls.json`, `screen.verdict` ADOPT_FOR_FULL_RUN),
   right after `lta-gate`.
4. The full run, after `pkoh-gate` and `mpp-gate`, right after LTA's full run: `mpp-zrckpp-s1-hp`, `-s2-hp`, `-s1` and
   `-s2`, at zrckpn's peaks of the same seed plus 2 GB. Then `mpp-compare-full` and `mpp-calls-full`.

```
python outputs/host_ops/retry_cmd.py --tries 3 --wait 90 -- python outputs/mp_unified/zmpp.py fit --name scr-mpp-zrckpp-s0-hp
python outputs/mp_unified/zmpp.py compare-all
python outputs/mp_unified/zmpp.py calls                             (outputs/mpp/calls.json and .md)
python outputs/mp_unified/zmpp.py --selftest
```

Only `outputs/mpp/calls.json` and `calls.md` are committed. The fits stay on the host.

## 10. zmpp.py

zmpp.py runs zltm.py's fit of zrckpn under MPP's names.

- zltm's fits, declaration, stage, stamp key, naming and code list become MPP's (`on_zltm`).
- zltm's column loader becomes the join (`joined_load`). ltmcols' column names and width become the join's 13 for the
  length of the fit. Each set is loaded under its own names inside the join.
- zml1s's stamp is wrapped (`mpp_stamp`). LTM's fit of the seed and split is what an MPP fit is against. A train or read
  whose NEXT block is not MPP's join stops unstamped.
- Every name is put back afterwards, also on a failure.

Its compares are zrckpp against zrckpn of the same seed and split (`scr-mpp-zrckpp-vs-zrckpn-s<seed>[-hp]`), through
zwq's run_compare, stamped with MPP's block.

Its calls (`outputs/mpp/calls.json` and `.md`) hold:

- zrckpn's three seeds, from LTM's compares and musique's read files;
- section 5's screen and full run (zltm's rule), with section 5's INCOMPLETE checks;
- section 7's next form;
- section 6's beside.

`zmpp.py --selftest` passed on the laptop's CPU before this commit. It covers:

- **Names.** MPP's fits are none of ML1's, LTM's or LTA's. Each is against LTM's fit of its seed and split.
- **The patches.** They are made and put back, also on a failure. The code stamp has zmpp.py, gnpcols.py and zpfp.py
  beside LTM's files.
- **The stamp.** Against and naming are set. It refuses another stage's NEXT block, a block of one scorer, and a name
  not MPP's.
- **The join, on made LTM and GNP columns of a toy carve.**
  - It gives 13 columns: LTM's seven, then GNP's six, each equal to its own set's.
  - It carries the joined scorers and record sha256.
  - It is refused when GNP's set is missing, under LTM's names, or of other rows.
  - Outside the fit, zltm's loader reads LTM's columns alone.
- **The compares and the calls, on made records.** zrckpn's three seeds, MPP's reads, the slices, a read file off its
  compare and a compare of another base are read. The calls are drawn with every table.
- **The INCOMPLETE checks on made stamps.** A wrong LTM scorer, a GNP rule other than the stamp's, a changed column
  record, and a missing stamped file each make a problem.
- **The rule on made numbers** (NO_GAIN, ADOPT_FOR_FULL_RUN, NOT_ADOPTED, ADOPT), and the next form.

## 11. Amendment A1, 3 November 2026, after PFS amendment A4's calls and before any MPP fit

**Why.** PFS amendment A4 (docs/PKO_FEATURE_SCREEN.md section 13) tested the next form named in section 7: the same six pair scores against rrf's top ten rows instead of its top five. It used A3's logistic probe on the five fit carves. Its rule, fixed before any number: TOP10_NOTHING or TOP10_HURTS on at least three of the five withdraws that form by an MPP amendment before MPP's verdict, and the pair columns' line for the MLP ends at MPP. Three reads already give TOP10_NOTHING. The other two cannot change the count:

| dataset | questions | recall@5, lean + P5 | lean + P10 | d10 | call |
| --- | --- | --- | --- | --- | --- |
| hotpotqa | 1,000 | 0.8223 | 0.8178 | -0.0045 | TOP10_NOTHING |
| squad | 1,000 | 0.9558 | 0.9588 | +0.0030 | TOP10_NOTHING |
| 2wiki | 1,000 | 0.8203 | 0.8185 | -0.0018 | TOP10_NOTHING |
| musique (first 400, all 2-hop) | 400 | 0.6663 | 0.6625 | -0.0038 | TOP10_NOTHING; counted under section 14's rule once its 3-hop and 4-hop read is in |

Every read's machinery check is exact (rrf's top five give A3's twelve columns, 0.0e+00). Each read reproduces A3's lean and lean + P5 recall@5 to 0.0. Section 13's full table, with musique's 3-hop and 4-hop read (section 14) and webqsp, is filed in its own section of that file.

**What changes.**
- **Section 7, the NO_GAIN or NOT_ADOPTED row.** Its next form, the same six scores against rrf's top ten rows, is withdrawn. On NO_GAIN or NOT_ADOPTED, the six pair columns close for the MLP and no further pair form follows. The MLP's next forms are the ones LTS's and LTA's declarations name.
- **Section 7, the ADOPT_FOR_FULL_RUN row,** stands as written. Its next form joins the adopted columns to another adopted candidate and reads no new anchors.
- **Nothing else changes:** not zmpp.py, not MPP's items, not its rule, not its reads.
