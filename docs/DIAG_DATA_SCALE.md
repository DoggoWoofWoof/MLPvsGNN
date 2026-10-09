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
