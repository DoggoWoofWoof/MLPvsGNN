# D3: is the MLP's base limited by its training data?

Declared 10 October 2026, about 04:30, before any number. A diagnostic: it decides whether larger fit carves get built,
and nothing else.

## Why

Every fit carve is capped at 6,000 questions per dataset (outputs/m3b/carves.json, `fit.size_cap`; its own note calls the
cap a CPU-era compute bound). The train splits are much larger:

| dataset | train pool | fit carve |
| --- | ---: | ---: |
| metaqa | 329,282 | 5,960 |
| 2wiki | 167,454 | 5,928 |
| squad | 130,319 | 5,856 |
| hotpotqa | 90,447 | 5,930 |
| musique | 19,938 | 4,601 |

Rounds 26–32 changed the inputs or the head and found nothing (D2: FEATURE_LIMIT on zrc's inputs). One lever no round
has tried is more training questions. Building a second fit carve per dataset (about 26,000 new questions) means:
- looks, lean caches and chain builds for all of them;
- 8–18 GB of host disk;
- CPU and memory that the link and pool stages (U1d, step 4h) hold now.

D3 asks first, from the other side, whether more data would help.

## What runs

`outputs/mp_unified/zds.py` (new; frozen once run). It refits zrc, the MLP's base, on a stride of its own fit carves:
- **every 2nd** question of each training dataset (half);
- **every 4th** (a quarter).

The stride is the carve rule's own, in carve order. Everything else is zrc's:
- zrm.ZRM over zrc.ChainCarveZRC;
- rmatch.py's train, zrm's settings, seed 0, p@swa;
- the same number of epochs, so a half fit takes half the steps. Doubling the carve under the same recipe would double
  them too, so this is the comparison that predicts it.

**Fits.** The screens' two splits, L-musique and L-hotpotqa: four GPU fits in all.

**Reads.** Each fit is read on the six s1eval carves, against zrc's own fit of its split (scr-zrct, scr-zrct-hp). Pair
and re-call follow the screens' protocol under the seed null (docs/SCREENS.md, section 2). Here a LOSS says the full
carve beats its subset by more than seed noise.

zds.py's selftest checks:
- every 1 is zrc's fit bit for bit;
- every 2 asks the carve only for the kept questions and moves the fit;
- repeats are identical.

## The rule

| half fits (12 reads, seed null) | verdict |
| --- | --- |
| 2 or more reads LOSS | **DATA_LIMITED** |
| otherwise | **NOT_DATA_LIMITED** |

- **DATA_LIMITED:** halving the data costs more than seed noise. A second fit carve per dataset is then declared as its
  own screen: group-disjoint from the select, eval and reserved carves, built by the same rule, with zrc trained on
  both. That build waits for host disk (two times its estimate free above the 100 GB floor) and for U1c's stages ahead
  of it.
- **NOT_DATA_LIMITED:** no second carve is built, and data scale is not the next lever for zrc.

The quarter fits are reported only, for the curve's shape. The record is `outputs/screen/dsx-decide.{json,md}` (`zds.py
decide`).

## What it does not say

- It reads the MLP's base only. The GNN track (zsp) is not refit.
- A NOT_DATA_LIMITED verdict at seed 0 on two splits does not rule out a gain from a different data mix: for example,
  more musique-like multi-hop questions where they exist.

## Result (filed 10 October 2026, about 07:20)

**DATA_LIMITED.** The half fits LOSE on 4 of 12 reads under the seed null, with no GAIN; the quarter fits LOSE on 10
(outputs/screen/dsx-decide.{json,md}, dsx-{h2,q4}{,-hp}{,-pair,-pair-recall}.{json,md}).

| split | dataset | read | zrc R@5 | half delta | half call | quarter delta | quarter call |
| --- | --- | --- | ---: | ---: | --- | ---: | --- |
| L-musique | metaqa | in-domain | 0.7782 | -0.0043 | WITHIN | -0.0259 | LOSS |
| L-musique | squad | in-domain | 0.9105 | -0.0019 | WITHIN | -0.0084 | LOSS |
| L-musique | musique | zero-shot | 0.5235 | -0.0071 | WITHIN | -0.0180 | WITHIN |
| L-musique | hotpotqa | in-domain | 0.9041 | -0.0068 | WITHIN | -0.0146 | LOSS |
| L-musique | 2wiki | in-domain | 0.8733 | -0.0078 | **LOSS** | -0.0175 | LOSS |
| L-musique | webqsp | zero-shot | 0.3384 | -0.0310 | **LOSS** | -0.0813 | LOSS |
| L-hotpotqa | metaqa | in-domain | 0.7793 | -0.0064 | WITHIN | -0.0298 | LOSS |
| L-hotpotqa | squad | in-domain | 0.9112 | -0.0015 | WITHIN | -0.0089 | LOSS |
| L-hotpotqa | musique | in-domain | 0.5645 | +0.0036 | WITHIN | -0.0093 | LOSS |
| L-hotpotqa | hotpotqa | zero-shot | 0.8451 | -0.0018 | WITHIN | -0.0084 | WITHIN |
| L-hotpotqa | 2wiki | in-domain | 0.8731 | -0.0118 | **LOSS** | -0.0242 | LOSS |
| L-hotpotqa | webqsp | zero-shot | 0.2865 | -0.0381 | **LOSS** | -0.0542 | LOSS |

Mean delta: half -0.0096, quarter -0.0250.

### What it says

- The curve is still rising at the full carve: each halving costs about 0.01-0.015 R@5 on average. **webqsp read
  zero-shot is the most data-hungry read** (-0.031 / -0.038 at half), then 2wiki in-domain. musique, the B1 gap, is
  not: its in-domain read is within noise at half.
- **A confound the rule did not separate:** a half fit runs zrc's epochs, so it also takes half the optimiser steps.
  Round 33 (docs/SCREENS.md, zep) separates them before any carve is built:
  - zrc at twice its epochs on its full carves (the screen);
  - zrc at twice its epochs on the half (as many steps as zrc; D3's step control).
- **Under the rule, a second fit carve per dataset is declared as its own screen** (docs/FIT2_CARVES.md, to be written
  once round 33's step control reads). Its build waits for U1c's stages and for disk, as above. Measured on the host
  today: the step-1 looks and caches of the five fit carves total about 14 GB (metaqa 7.1, musique 6.2, the other three
  under 0.6 each), before the chain builds (measured when the file is written). A second carve needs at least twice
  that free above the 100 GB floor.

