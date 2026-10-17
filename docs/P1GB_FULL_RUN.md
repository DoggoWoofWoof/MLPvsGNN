# P1g-b's full run: the lean pair on P1g-b's pools, six splits

Declared 17 October 2026, before any number of P1g-b's screen. Code: `outputs/mp_unified/zp1gbf.py`. It runs only if
P1g-b's screen (docs/P1GB_POOLS_REFIT.md) is ADOPT_FOR_FULL_RUN.

## 1. Why

P1g-b refits the lean pair, unchanged, on P1g2's chosen pools: ALL ≥ 0.95 on every dataset (musique 0.885 → 0.951),
about 3.4 times today's rows. Its screen fits two splits per arm. Its section 5 says what follows a pass: "the full
run, six splits per arm under grade-nullx, in its own file". This is that file.

It is declared now, ahead of the screen's verdict, so that the card has work queued behind P1g-b's fits (the user, 17
October: "stuff should always be queued for any such downtime so that system never is idle"). Its first item is a gate
on the screen's verdict. If the screen is not ADOPT_FOR_FULL_RUN, nothing of this run trains or is read.

## 2. The fits

| arm | model | its four new fits | its screen fits (reused) | decided against (P1c2's pools) |
| --- | --- | --- | --- | --- |
| zrckp | the MLP, as filed | `outputs/full_p1gb_zrckp/fits/S` | `scr-p1gb-zrckp`, `-hp` | `scr-p1c3p-zrckp`, `-hp`; `outputs/full_zrckp/fits/S` |
| zspk | the GNN, as filed | `outputs/full_p1gb_zspk/fits/S` | `scr-p1gb-zspk`, `-hp` | `scr-p1c3-zspk`, `-hp`; `outputs/full_zspk/fits/S` |

- **The four new splits:** J5, L-2wiki, L-squad and L-metaqa. With the screen's L-musique and L-hotpotqa, that makes
  six per arm.
- **Everything is P1g-b's:** its looks, caches, chains, links and KT input (`outputs/p1gb/*`), its pool gate, and
  zp1gb.py's train and read, imported unchanged. Each arm's settings, seeds, batches, loss and loop are unchanged.
- **Reads:** each fit on the six s1eval carves of P1g-b's pools at p@swa. R@5 is over the question's golds, so a gold
  a pool misses counts as a miss.
- **Compares:** each fit against the same arm's fit of the split on P1c2's pools, which decides. Step 1's fit is
  reported beside.
- **Base comparisons for the null:** each base fit compared with step 1's fit of its split. These are P1g-b's
  `outputs/p1gb/base-<arm>-<split>.json` for the screen's splits, and four new ones per arm,
  `outputs/p1gbf/base-<arm>-<split>.json` (`full_<arm>/fits/S` against `step1/fits/S`).

## 3. The rule, fixed before any number

Each arm's six comparisons go through relz's grade, then nullx's re-grade under the seed null over all six splits
(the twelve null files of docs/SCREENS.md), with that arm's base comparisons. GAIN, LOSS and WITHIN are
docs/SCREENS.md's, on R@5; step 1's eleven primary reads are docs/SCREENS.md's.

- **ADOPT:** neither arm's re-grade has a LOSS on any of its 36 reads, and the two arms together have at least one
  primary GAIN. P1g-b's pools become the lean pair's pools.
- **NOT_ADOPTED:** otherwise. The lean pair stays on P1c2's pools.
- **INCOMPLETE:** a re-grade is missing, is not that arm's, was not made with the declared base comparisons, or does
  not hold the 36 reads with 11 primary.

Both arms read the same pools, so pools that cost either arm are not adopted (P1c2's rule).

**Reported beside, deciding nothing:**
- each arm's own re-grade verdict;
- R@5 on P1g-b's pools of both arms on all 36 reads, and Claim 1's share (zrckp − RRF) / (zspk − RRF) as point
  estimates (no share where the GNN gains less than 0.01). Claim 1's intervals on these pools are P3c's, run again
  after this verdict;
- hit@1 and FC@5 per read, as the compares give them.

**What follows ADOPT:** P1g-b's pools are the pools of P3's gap ledger, G2r, and the refits of the screens adopted
on P1c2's pools (E1, T1 and G3, if they pass). The new rows' cost is timed cold beside P4's timings before any
latency claim.

## 4. How it runs

- **Self-test** (`python outputs/mp_unified/zp1gbf.py --selftest`, passes before this commit):
  - the arms are the lean pair as filed. Each arm's bases are its own fits on P1c2's pools, and its base comparisons
    are the six declared ones;
  - the gate opens on P1g-b's ADOPT_FOR_FULL_RUN only. NOT_ADOPTED, INCOMPLETE, another stage's record or a missing
    record keep it shut;
  - trains, reads and comparisons stop while it is shut. read refuses P1g-b's screen fits;
  - train refuses the screen's splits, another arm, another folder and P1c2's caches. P1c2's roots are restored after;
  - each arm's grade decides against its own bases, and another arm's comparisons make it INCOMPLETE;
  - one verdict for the pair (five cases), six INCOMPLETEs, and Claim 1's shares beside.
- **Host items** (`p1gbf-*`, 38 in all, placed after P1g-b's verdict):
  - eight base comparisons on the CPU, which can run any time;
  - the gate, after P1g-b's verdict;
  - on the card, behind the gate: eight trains and eight reads at `--frac 0.92`, share 0.94, one at a time, as
    P1g-b's were. Memory is P1c3's and P1c3p's full-run items scaled as P1g-b's were (×3.15). The feeder resizes on
    the measured peaks;
  - on the CPU: eight compares, two grades, two re-grades under the null, and the verdict.
- **Disk:** a fit with its reads takes about 0.02 GB (measured on P1c3's fits), so about 0.4 GB in all.
- **ETA:** P1g-b's screen fits and reads take about 1.5 hours per fit, one at a time. Eight fits are about 12 to 13
  hours on the card after P1g-b's verdict, which is due about 16:00 on 17 October. The verdict is due about 05:00 on
  18 October.
- The encoder stays frozen. webqsp never trains. Test splits are never read. No LLM or transformer.

## 5. Results

Not run: the gate is SHUT (`outputs/p1gb/full-gate.json`). P1g-b's screen is NOT_ADOPTED
(docs/P1GB_POOLS_REFIT.md, section 7), so no train, read, compare or grade of this file ran.
