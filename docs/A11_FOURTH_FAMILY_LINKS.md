# A11: our LLM-free phrase links as a fourth family in the GNN's step

Declared 17 October 2026, before any number. The eleventh reopening of docs/VERDICT_AUDIT_2026_10_17.md (section 2,
row 11), under docs/SCREENS.md section 3b. Code: `outputs/mp_unified/za11p4.py` (the build, the arm, the screen and
the call). Not run yet.

## 1. Why this, now

**Where the idea came from.** P2c built a triples graph from our own parse, without an LLM. Part 1 read it on B1's
small corpora, where it added more in the pool walk than every LLM-built graph (+8.4 R@5 on musique against +3.7 to
+4.8). Part 2 (13 October; docs/P2C2_FOURTH_FAMILY.md) read its P form on our full corpora: two passages are linked
when they share a phrase that 2 to 25 passages hold.

**What P2c2 found.** NO_GAIN on pool coverage. As a fourth family in the pool's walks, it moved the pool's ALL by
−0.007 to +0.003. Most of its edges are new (66% to 74% of its pairs are in none of the three families), but under the
pool's rule they did not move golds into the pool's head.

**Why that closes only its form.** P2c2 asked whether the family puts more golds into the pool. It never asked whether
the GNN can use it once the pool is fixed: the family never reached the GNN's links. The GNN's step reads each row's
neighbours' scores per family, so a link that does not change who is in the pool can still change how a row inside it
is ranked. Under section 3b, the idea stays open until it reaches a model.

**The base is zspk,** as the audit row names: the family enters zspk's propagation step, beside its three families.

## 2. The family

**The build is P2c2's code, unchanged** (outputs/p2c/dev/p2c2_proto.py: `mentions` and `p_form`):
- the parse is P2c's version 1, already on the host for all four passage corpora (P2c2's items);
- each phrase is keyed by the first 8 bytes of blake2b over its UTF-8 bytes, the same key in every part;
- two passages are linked when they share a phrase that 2 to 25 passages hold, once, undirected, with no self-loop.

The corpus's passage count comes from the parse records: their line ranges must tile the corpus in order, and
`mentions` checks the parts hold exactly that many passages.

**Each pool's edges.** For each question of a P1c2 cache part, every pair of its pool rows whose passages the P form
links becomes two directed edges, u to v and v to u. They are stored sorted by (u, v), int16 and pool-local, tied to
the part's step-1 record. Each build item rebuilds its corpus's P form in memory (P2c2 measured 1 s on squad to 481 s
on 2wiki), so no corpus-wide graph is stored.

**The knowledge bases** (metaqa, webqsp) hold no passages, so they have no fourth-family edge and no build, by rule.
There zspk4 is zspk bit for bit.

No new constant: the 2-to-25 window and the phrase key are P2c's. No LLM, no encoder change, no new text.

## 3. The arm

**zspk4** is zspk (zp1c3.ZSPK) whose step head also reads the fourth family:

    zspk:   s + W2 gelu(W1 x10 + b1) + b2
    zspk4:  s + W2 gelu(W1 x10 + b1 + W4 x4) + b2

- **x10** is zspk's ten step inputs: per family the mean and soft maximum of the neighbours' z-scores and log1p of the
  degree, and the row's own z-score.
- **x4** is the same three columns for the fourth family, computed by zprop.prop_inputs's own code on its edges.
- **W1 and b1** are zspk's own draws. **W4** (32 × 3, 96 parameters) is drawn from its own generator (seed + 1101), at
  W1's scale.
- A row with no fourth-family neighbour reads x4 = 0. So zspk4 is zspk bit for bit at the start (W2 is zero), on the
  knowledge bases, and on any row the family does not reach.

**GNN only.** The MLP takes no message. A gain here brings the family to the MLP next as a fixed per-row column
(section 5).

| arm | the model | the carve |
| --- | --- | --- |
| zspk4 | zspk with W4 over the fourth family's columns | zspk's (P1c3's KT link carve) with the fourth family's edges beside |

## 4. The fits and the call

- **Fits:** one card fit on L-musique and one on L-hotpotqa (`scr-a11-zspk4`, `-hp`), on P1c2's roots with P1c3's
  KT input. Card fit settings: variant p, config 2e-3:1e-4:0.1:8:2, seed 0, hidden 128 (zspk's).
- **Reads:** each fit on the six s1eval carves at p@swa.
- **Compares:** each fit against zspk's card fit of the same split (scr-p1c3-zspk, scr-p1c3-zspk-hp), which decides.
  Step 1's fit is reported beside.
- **The re-call's base screens:** L1's, zspk's card fits compared with step 1's of the same split
  (outputs/l1/base-zspk-<split>).
- **Call:** relz's pair and its re-call under the seed null (the four null files of docs/SCREENS.md). GAIN, LOSS and
  WITHIN are docs/SCREENS.md's, on R@5.

### The rule, fixed before any number

- **ADOPT_FOR_FULL_RUN:** zspk4's re-call has no LOSS on any of its twelve reads and at least one GAIN;
- **NOT_ADOPTED:** otherwise;
- **INCOMPLETE:** its re-call is missing, is not A11's re-call of zspk4, or does not hold the twelve reads.

The verdict closes only this form (SCREENS.md section 3b), and it names the next form (section 5).

**Reported beside, deciding nothing:**
- webqsp and metaqa, both splits (the KBs, where zspk4 is zspk by rule: any move there comes from the fit's other
  datasets);
- musique in domain (L-hotpotqa): Claim 1's read;
- hit@1 and FC@5 per read;
- each part's build counts: edges per row, and the share of rows with a fourth-family neighbour.

## 5. What each outcome leads to (fixed now)

| its re-call | the next form |
| --- | --- |
| ADOPT_FOR_FULL_RUN | the fourth family built on P1g-b's pools and zspk4 fitted again there, then its full run; and A11m, the family's per-row degree (log1p) as a fixed column in the MLP (zrckp) through R1's pointwise head (each its own declared file) |
| NOT_ADOPTED, with at least one GAIN and a LOSS | A11d, the back-off form: the fourth family under edge-family dropout over the four families (FD 0.25, G2's rule), so the step cannot lean on it where its mix differs; and A11m for the MLP (one declared file) |
| NOT_ADOPTED, with no GAIN | A11x, the fourth family in all three of zspx's steps (the adopted GNN, where depth paid) (one declared file) |
| INCOMPLETE | the missing re-call is run; nothing else changes |

No model change is adopted on today's pools (the user's rule of 13 October).

## 6. How it runs

**Self-test, passed before this commit.** `python outputs/mp_unified/za11p4.py --selftest` (10 s) checks:
- **The constants.** The arm is zspk plus W4 on the KT link carve with the fourth family; zspk's and zsp's arms are
  unchanged. The passage corpora and the KBs split the six. The committed parse records tile each corpus at P2c2's
  passage counts (20,233; 117,534; 5,233,329; 5,989,847).
- **The edges.** `induce` equals a brute-force pair set on random graphs and pools, including rows that share a
  passage.
- **The build** (on a toy cache with step 1's node arrays and a toy P form): it refuses a KB and a missing part. Each
  part's edges are the brute force's, tied to its record, and the carve's batch rows are the brute force's. A KB
  carve gets no edge and no record.
- **Bit-for-bit cases.**
  - zspk4's state is zspk's plus W4.
  - At the start, zspk4 is zspk.
  - With W2 moved, zspk4 moves only the rows with a fourth-family neighbour. It is zspk bit for bit on the KB,
    without the family's key, and without links.
- **The columns** match a float64 brute force (z-scores, clip, mean, log-sum-exp, log1p degree); rows without a
  neighbour read 0.
- **Toy fits through lean_gpu's loop.** A repeat is bit for bit; W4 learns; on a KB the fit is zspk's bit for bit on
  every shared key.
- **The rest.** The train and read refusals; relz's pair and re-call stamped A11's; the verdict's outcomes and next
  forms.

**Host items** (`a11-*`), after A10's verdict in the file, each starting when its inputs are ready:
- **16 builds,** one per passage cache part of the carves the screen reads:
  - the fit carves: squad 1, hotpotqa 1, 2wiki 1, musique's s1fit 6;
  - the s1eval carves: squad 1, hotpotqa 1, 2wiki 2, musique 3.

  Each takes 1 CPU, through pylib_run (numba is imported by P2c2's module). Memory, from P2c2's measured peaks:
  2wiki 12 GB, hotpotqa 8 GB, musique 3 GB, squad 2 GB. The edges are int16 pairs, a fraction of the 3.8 GB that
  zlink's three families take on the same parts, so the items go under the feeder's small-writer names.
- **The screen:**
  - 2 trains and 2 reads, GPU sized from zspk's runs with a margin for the family: `--frac 0.32` (share 0.34);
    memory: trains 4 GB (-hp 12 GB), reads 5.5 GB;
  - the trains wait on their split's fit-carve builds, the reads on the s1eval builds;
  - 2 compares, the pair, the re-call and the verdict.
