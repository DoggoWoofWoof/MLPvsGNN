# T1: S6's granularity transforms as a training augmentation of the lean pair

Declared 16 October 2026, before any number of this stage. Code: `outputs/mp_unified/zt1gr.py` (S6's transforms are
`outputs/mp_unified/chainscore19.py`, imported unchanged). Numbers here are development numbers. Status: declared.

## 1. Why this, now

The user (15 October): "why isnt the transforms and offsets working". The offsets were tried in the models and closed
(P5b to P5b3). **The transforms were never tried in the models** (docs/PROGRAM_2026_10_15.md, section 1.6). This stage
tries them.

- **What S6 showed.** S6's chain scorer (`chainscore19.py`) was trained on metaqa with its relation layer rewritten:
  - merged three at a time (mg3);
  - split into aliases or by tail cluster (al4, sp4);
  - half reified through mediator nodes (rf).

  As design looks, webqsp zero-shot rose 6.43 R@5 above the untyped walk, and 10.7 above the same scorer without the
  transforms. metaqa in-domain fell 4.8.
- **Why the lean pair may need it.** Both models read rmatch's and zrc's chain entries, and their chain heads learn
  from metaqa alone (the only typed graph in training). metaqa has 9 relation names and no mediators. webqsp, read
  zero-shot, has many relations, and most of its facts pass through CVT mediators. A chain head that has only seen
  metaqa's granularity reads webqsp's chains at a granularity it never saw.

## 2. The augmentation

Only **metaqa's fit carve** changes, and **only in training**. Each of its questions takes exactly one regime, fixed
by a hash of its id, so every question keeps its weight in training:

| regime | share | what changes |
| --- | --- | --- |
| id | about 1/3 | nothing: the question as filed |
| mg3 | about 1/3 | chainscore19's mg3: relation r becomes perm(r) // 3 (3 merged relations), each named by the unit sum of its members' names |
| rf | about 1/3 | chainscore19's rf on this question: about half its relations, each by its own hash, are reified. (h, r, t) becomes (h, r, m) and (m, r′, t), with one mediator m per distinct (h, r, t) and r′ named as r |

- **Mediators are not rows.** A mediator is numbered after the pool's rows. The walk passes through it, its entries are
  dropped, and it is never gold. A chain's end nodes are either all mediators or all rows, so no row's mass changes
  because of a mediator.
- **One relation table.** It holds metaqa's 9 names as filed, the 3 merged names, then the 9 second halves (21 rows).
  Its centred form (the chain head's `rel_centered`) is centred on the filed 9's mean, not the 21's. So an id
  question's entries, relation rows and chain features are today's (entries array for array; features to float32
  rounding, at most 5e-7 on the toy).
- **Unchanged:**
  - every other carve, including metaqa's s1sel and s1eval;
  - reads and serving;
  - both models, their parameters, settings, seeds and batches;
  - P1c2's pools and caches, P1c3's KT input, and zspk's propagation edges (which carry no relation).
- **How it differs from S6:**
  - **al4 and sp4 are left out.** They keep a relation's name, and the lean pair matches relations by their name's
    text, so they would change nothing.
  - **rf's half is drawn per question,** not fixed per carve (S6 rewrote one union graph).
  - **The walk keeps rmatch's 3 hops.** A chain that reification stretches past 3 steps is lost, as a long CVT path is
    on webqsp today.
- **The MLP rule holds.** zrckpt scores each row from inputs computed before any row is scored. Only its training
  inputs on one carve change.

## 3. The fits and the call

| arm | model and carve | its base |
| --- | --- | --- |
| zrckpt | zrckp (zrc + the pointwise KT head, no message passing) | `scr-p1c3p-zrckp`, `-hp` |
| zspkt | zspk (zsp's propagation head + the KT head) | `scr-p1c3-zspk`, `-hp` |

- **Fits:** one card fit per arm on L-musique and L-hotpotqa (`scr-t1-zrckpt`, `-hp`; `scr-t1-zspkt`, `-hp`). Both
  splits train metaqa. Each fit is read on the six s1eval carves at p@swa, on P1c2's roots and P1c3's KT input, as the
  bases were.
- **Compares:** each fit against its base arm's card fit of the same split, which decides. Step 1's fit is reported
  beside. The re-call's base screens are L1's (`outputs/l1/base-<arm>-<split>.json`): the same base fits compared with
  step 1's.
- **Call:** relz's pair and its re-call under the seed null (the four null files of docs/SCREENS.md). GAIN, LOSS and
  WITHIN are docs/SCREENS.md's, on R@5.

### The rule, fixed before any number

Per arm:
- **ADOPT_FOR_FULL_RUN:** its re-call has no LOSS on any of its twelve reads and at least one GAIN;
- **NOT_ADOPTED:** otherwise;
- **INCOMPLETE:** its re-call is missing, is not T1's re-call of that arm, or does not hold the twelve reads.

The two arms are decided separately. metaqa in-domain is one of the twelve reads, so S6's in-domain cost counts
against an arm here. Under the user's rule of 13 October, no model change is adopted on today's pools: an arm that
passes is fitted again on P1g's final pools before its full run (six splits under grade-nullx, its own file).

**Reported beside, deciding nothing:**
- webqsp (the target, zero-shot in both splits) and metaqa (in-domain), per split, with their calls and differences;
- each regime's question count and the mediators (`outputs/t1/build.json`);
- hit@1 and FC@5 per read, as the compares give them.

## 4. How it runs

- **Self-test** (`python outputs/mp_unified/zt1gr.py --selftest`, passes before this commit):
  - the arms are the lean pair's classes;
  - the union table keeps the filed rows exact, and its merged names are chainscore19's `Transform.names`;
  - on a toy typed carve (12 questions, all three regimes), every question's rmatch and zrc entries equal those of
    chainscore19's own mg3 relabel and rf `reify_row` on its triples, kept to the pool's rows. An id question's
    entries equal the filed build's, array for array;
  - loading: the union tables' filed rows load exactly as the filed tables, so an id question's chain features are the
    filed carve's to float32 rounding. Only the named carve is swapped, a root T1 does not replace is refused, and the
    loader is restored after;
  - train refuses another arm, split, fit name, cache root or device; read refuses a fit that is not T1's; a union
    table on disk is never overwritten;
  - relz's pair and re-call against each base arm's card fits, stamped T1's; the verdict's three outcomes per arm.
- **Host items** (`t1-*`, 18 in all):
  - `t1-build` on the CPU (1 CPU, 8 GB, as P1c2's chain builds of the same carve): both chain trees and
    `outputs/t1/build.json`. The train items wait on it;
  - on the card, as L1's fits ran (`--frac 0.30`, gpu share 0.32): four trains, then four reads;
  - on the CPU: four compares, two pairs, two re-calls and the verdict.
- **Disk:** two chain trees of metaqa's fit carve (about the size of P1c2's for that carve) and four fit folders of a
  few MB each.
- **ETA:** the build takes about as long as P1c2's two chain builds of metaqa's fit carve. The fits and reads then take
  about 2 hours on the card, after L1's and L3's items.
- The encoder stays frozen. webqsp never trains. Test splits are never read.

## 5. Results

Not run yet.
