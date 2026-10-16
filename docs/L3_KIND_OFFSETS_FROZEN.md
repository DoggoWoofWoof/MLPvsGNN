# L3: zkind's offsets on the frozen lean pair

Declared 15 October 2026, before any number of this stage. Code: `outputs/mp_unified/zl3ck.py`. Numbers here are
development numbers. Status: declared.

## 1. Why this, now

This is the third item of the ledger of stages NOT_ADOPTED although they showed gains (docs/PROGRAM_2026_10_15.md, item
8). The first item is L1 (docs/L1_SEP_LEAN_PAIR.md). The second, RRF5's position as a ninth KT column, needs a CPU
build of the column first and gets its own file.

- **What zck was** (docs/SCREENS.md, round twenty-eight): zrc's chain entries under zkind's head. zkind's head adds
  three offsets, zero at the start, which act only on a batch with chain entries, i.e. a KB graph (metaqa or webqsp).
  They are added to the output layer's weight and bias and to rrf's base weight. That gives the typed graphs their own
  read of the shared hidden units.
- **What it did.** webqsp read zero-shot gained in both fits: +0.0160 [+0.0042, +0.0270] and +0.0402 [+0.0257,
  +0.0557]. Its one LOSS was 2wiki in L-hotpotqa's fit (−0.0082). 2wiki has no chain entries, so the offsets never
  acted there: the loss came through the shared weights, trained jointly with the offsets.
- **What changes here.** Freeze the shared weights, and the passage graphs cannot move. The question becomes: does a
  typed-graph read of the lean pair's own hidden units, learned from metaqa alone, carry to webqsp? This is the user's
  webqsp question of 15 October ("check what helps webqsp and see how can we use to help other datasets").

## 2. The arms

| arm | frozen fit (its p@swa state) | trainable |
| --- | --- | --- |
| zrckpo | zrckp's card fit of the split: `scr-p1c3p-zrckp`, `-hp` | zkind's three offsets |
| zspko | zspk's card fit of the split: `scr-p1c3-zspk`, `-hp` | zkind's three offsets |

**The model.** The frozen fit's model with its fitted state loaded and every one of its parameters frozen, plus the
offsets:
- **kd_w** on the output layer's weight (128 weights);
- **kd_b** on its bias;
- **kd_base** on rrf's base weight.

That is 130 trainable weights per arm. On a batch with chain entries, the forward is the frozen fit's with
`out.weight + kd_w`, `out.bias + kd_b` and `base_w + kd_base`. On any other batch it is the frozen fit's forward.

**Training.** lean_gpu's loop, unchanged: its settings, seed, loss, batches and SWA. An untyped carve gives the offsets
no gradient, so the loop runs over the training carves with chain entries only. On both splits that is metaqa's fit
carve. Every carve is still loaded, as in the frozen fit. webqsp never trains.

**What cannot move:**
- squad, musique, hotpotqa and 2wiki read through the frozen fit's forward. Their reads change only by the card's
  floating-point nondeterminism, and those R@5 differences are reported beside.
- After training, the fit's every epoch state and its SWA state must equal the frozen fit's p@swa state bit for bit
  outside the offsets. Its training carves, basis, config, seed and hidden must also equal the frozen fit's. If not,
  the train item exits 1, and nothing that waits on it runs.
- The MLP rule holds: zrckpo's offsets read only the row's own hidden units and the dataset's kind.

## 3. The fits and the call

- **Fits:** one card fit per arm on L-musique and L-hotpotqa (`scr-l3-zrckpo`, `-hp`; `scr-l3-zspko`, `-hp`), read on
  the six s1eval carves at p@swa, on P1c2's roots and P1c3's KT input.
- **Compares:** each fit against the frozen fit, which decides. Step 1's fit is reported beside. The re-call's base
  screens are L1's (`outputs/l1/base-<arm>-<split>.json`): each frozen fit compared with step 1's fit of the split.
- **Call:** relz's pair and its re-call under the seed null (the four null files of docs/SCREENS.md). GAIN, LOSS and
  WITHIN are docs/SCREENS.md's, on R@5.

### The rule, fixed before any number

Per arm:
- **ADOPT_FOR_FULL_RUN:** its re-call has no LOSS on any of its twelve reads and at least one GAIN;
- **NOT_ADOPTED:** otherwise;
- **INCOMPLETE:** its re-call is missing, is not L3's re-call of that arm, or does not hold the twelve reads.

The two arms are decided separately. An adopted arm's full run is its own file (six splits under grade-nullx), and it
replaces its base only after that.

**Reported beside, deciding nothing:** hit@1 and FC@5 per read; the offsets' norms; the four untyped datasets' R@5
differences (expected 0, or within the card's float noise).

## 4. How it runs

- **Self-test** (`python outputs/mp_unified/zl3ck.py --selftest`, passes before this commit), on rmatch's toy chain
  carve and an untyped copy, for both arms:
  - built under a frozen fit, the model holds its state bit for bit, with every one of its parameters frozen. It
    refuses other blocks or widths;
  - with zero offsets, its forward equals the frozen fit's bit for bit, typed or not, in eval and in training;
  - with the offsets set, a typed batch equals the frozen fit with the offsets added, and an untyped batch is unchanged;
  - toy fits through lean_gpu's loop train on the typed carve only. They keep the frozen state bit for bit in every
    epoch and in SWA, move the offsets, repeat identically, and refuse when no carve is typed;
  - the check catches a moved weight and another fit's settings;
  - train, read and the frozen fit's loader refuse what is not L3's;
  - relz's pair and re-call run against the frozen fits, stamped L3's;
  - the verdict per arm.
- **Host items** (`l3-*`, 17 in all): four trains and four reads on the card at P1c3's share (0.32); four compares,
  two pairs, two re-calls and the verdict on the CPU. The base screens are L1's, already filed. The trains queue behind
  L1's card items.
- **Disk:** four fit folders of a few MB each.
- **ETA:** each train runs one carve with 130 trainable weights. Loading the carves dominates, so each should take
  about 10 minutes, and each read about 15. The verdict should land about an hour after the trains start.
- The encoder stays frozen. webqsp never trains. Test splits are never read.

## 5. Results

Filed 15 October 2026, 22:31 on the host (`outputs/l3/screen-verdict.json`; fetched 17 October after the host's
outage). **zrckpo NOT_ADOPTED, zspko NOT_ADOPTED.**
- **zrckpo** against the frozen zrckp fits: 0 GAIN, 1 LOSS (webqsp zero-shot on L-musique's fit).
- **zspko** against the frozen zspk fits: 0 GAIN, 0 LOSS.
- The untyped reads' R@5 differences are 0.0 on all eight, for both arms: the frozen forward is unchanged there, as
  declared.

zck's per-kind offsets on a frozen base gain nothing and cost webqsp zero-shot once. They are not taken further.
