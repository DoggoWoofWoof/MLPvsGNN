# L1: the lean pair trained with each gold against the non-gold rows (zsep's loss)

Declared 15 October 2026, before any number of this stage. Code: `outputs/mp_unified/zl1sep.py` (the objective is
`outputs/mp_unified/zsep.py`, imported unchanged). Numbers here are development numbers. Status: declared.

## 1. Why this, now

This is the first item of the ledger of verdicts that were NOT_ADOPTED although they showed gains (the user, 13
October: "we shud check for other verdicts of not adopted even after having gains and see what we can imprve and take
from them").

- **What zsep is.** lean_gpu's loss is, per question, the mean over its golds of −log softmax over every row of the
  pool. A question's other golds therefore sit in each gold's denominator. Once one gold leads, the loss pushes it down
  against its sibling golds, and the non-gold rows above a trailing gold get little push. zsep's loss is the mean over
  golds of −log(e^{s_g} / (e^{s_g} + Σ_n e^{s_n})), with the sum over the non-gold rows only. A question with one gold
  trains exactly as before.
- **What it did before.** On zret (the sixteenth round, docs/FULL_ROUND16.md) the screen's re-call was NO_GAIN: no R@5
  GAIN and one LOSS (musique in L-hotpotqa's fit, −0.0101). hit@1 rose on metaqa, hotpotqa and 2wiki in-domain (+0.012
  to +0.022). That gain is the one the ledger names. P1c3's full run lowered hotpotqa's hit@1 under the KT input.
- **Why it may differ now.** The lean pair (zrckp, zspk) is not zret. It reads KIND_T's pools and P1c3's KT input,
  where the misses that remain are mostly the second gold of a multi-gold question: musique's chains, and hotpotqa and
  2wiki's pairs (docs/D5_GOLD_REACH.md, M1). Those are the questions whose terms zsep changes.

## 2. The arms

Each arm is a lean pair arm unchanged (model, carve, KT input, settings, seeds, batches, P1c2's pools and caches), with
`zsep.listwise_sep` in place of lean_gpu's loss **in training only**:

| arm | model and carve | its base |
| --- | --- | --- |
| zrckps | zrckp: zrc + the pointwise KT head (161 weights), no message passing (`zp1c3p.py`) | `scr-p1c3p-zrckp`, `-hp` |
| zspks | zspk: zsp's propagation head + the KT head (`zp1c3.py`) | `scr-p1c3-zspk`, `-hp` |

Reads and serving are the base model's, so neither arm changes a latency.

- **The MLP rule holds.** zrckps scores each row from inputs computed before any row is scored. The objective changes
  only which rows a gold is trained against.
- Each fit records, per training carve, how many questions have two or more golds in the pool (`l1.census` in
  screen.json). Only those questions' terms change.

## 3. The fits and the call

- **Fits:** one card fit per arm on L-musique and L-hotpotqa (`scr-l1-zrckps`, `-hp`; `scr-l1-zspks`, `-hp`), read on
  the six s1eval carves at p@swa, on P1c2's roots and P1c3's KT input, exactly as the bases were.
- **Compares:** each fit against its base arm's card fit of the same split, which decides. Step 1's fit is reported
  beside. The re-call's base screens are each base fit compared with step 1's fit of the split
  (`outputs/l1/base-<arm>-<split>.json`), as G2's were.
- **Call:** relz's pair and its re-call under the seed null (the four null files of docs/SCREENS.md). GAIN, LOSS and
  WITHIN are docs/SCREENS.md's, on R@5.

### The rule, fixed before any number

Per arm:
- **ADOPT_FOR_FULL_RUN:** its re-call has no LOSS on any of its twelve reads and at least one GAIN;
- **NOT_ADOPTED:** otherwise;
- **INCOMPLETE:** its re-call is missing, is not L1's re-call of that arm, or does not hold the twelve reads.

The two arms are decided separately. The objective is a training choice, so the MLP and the GNN may each keep or drop
it. A full run of an adopted arm is its own file (six splits under grade-nullx), and only then does the arm replace its
base in the lean pair.

**Reported beside, deciding nothing:**
- hit@1 and FC@5 per read, as the compares give them. The seed null's own hit@1 differences (its third column) are
  shown beside each hit@1 difference, because the ledger names hit@1. hit@1 does not decide.
- the census per training carve.

## 4. How it runs

- **Self-test** (`python outputs/mp_unified/zl1sep.py --selftest`, passes before this commit):
  - the arms are the lean pair's classes;
  - zsep's loss is swapped in for training only and restored after;
  - with one gold, it equals lean_gpu's loss in float32. With several golds, it matches zsep's float64 reference and
    differs from lean_gpu's;
  - train refuses another arm, split, fit name, cache root or device; read refuses a fit that is not L1's;
  - relz's pair and re-call run against each base arm's card fits, are stamped L1's, and relz is restored after;
  - the verdict's three outcomes per arm.
- **Host items** (`l1-*`, 21 in all):
  - on the card, two at a time, as P1c3's fits ran (`--frac 0.30`, gpu share 0.32): four trains, then four reads;
  - on the CPU: four base compares (no training), four compares, two pairs, two re-calls, and the verdict.
- **Disk:** four fit folders of a few MB each. The caches are P1c2's, read only.
- **ETA:** P1c3's card fits took about 25 to 40 minutes each, and reads about 15. With two on the card at once, the
  verdict should land about 2 hours after the items start.
- The encoder stays frozen. webqsp never trains. Test splits are never read.

## 5. Results

Filed 15 October 2026, 22:17 on the host (`outputs/l1/screen-verdict.json`; fetched 17 October after the host's
outage). **zrckps NOT_ADOPTED, zspks NOT_ADOPTED.**
- **zrckps** against zrckp's card fits (`scr-p1c3p-zrckp`), after the seed-null re-call: 0 GAIN, 0 LOSS in its twelve
  reads.
- **zspks** against zspk's card fits (`scr-p1c3-zspk`): 0 GAIN, 0 LOSS.

zsep's separation term trained on the lean pair moves no read beyond the seed null. It is not taken further.
