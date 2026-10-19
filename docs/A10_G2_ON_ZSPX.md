# A10: G2's hidden-state layers on the adopted GNN

Declared 17 October 2026, before any number. The tenth reopening of docs/VERDICT_AUDIT_2026_10_17.md (section 2,
row 10), under docs/SCREENS.md section 3b. Code: `outputs/mp_unified/za10g2x.py` (the arm, the screen and the call).
Filed 19 October: zg2x NOT_ADOPTED (section 7).

## 1. Why this, now

**Where the idea came from.** G2 (15 October; docs/G2_HIDDEN_STATE_GNN.md) put three hidden-state message-passing
layers per link family on top of zspk: 64-wide states built from the encoder's hidden state, per-family gated means
and an attention over all edges, with zspk's z-score and the KT columns fed at every layer, and a readout bounded at
0.5. Before it, zg1 (10 October) was NO_GAIN.

**What G2 found.** 0 GAIN, 0 LOSS after the seed-null re-call. Its filed GAINs, musique zero-shot (+0.0099) and webqsp
zero-shot (+0.0210), sat inside the seed null's floors. Beside it, zg2 minus its no-edge twin had one GAIN (webqsp
zero-shot from musique's split, +0.0131) and two LOSSes.

**Why that closes only its form.** G2 ran over zspk's one message step, with the stack reading zspk's score. G4 later
found that depth pays when the step acts on the score itself (zspk2, zspkg), and G5 combined those steps into zspx
(ADOPT_FOR_FULL_RUN; 2 GAIN on L-hotpotqa, 0 LOSS). The stack has never sat over the adopted GNN's score. Under
section 3b, the idea stays open until it does.

## 2. The form

zg2x is zspx (outputs/mp_unified/zg5mix.py) with G2's stack on top, unchanged:

    s3  = zspx's score: zspk's scorer and KT head, its step with the gradient kept, G3's residual message, G4's
          second step
    out = s3 + 0.5 tanh(r / 0.5),   r = G2's stack reading s3's z-score

**The stack is G2's code, unchanged** (zg2.G2Mix, mixed over zg5mix.ZSPX):
- 64-wide states from the encoder's hidden state (the output layer's input), three layers;
- per layer: a gated mean per link family and an attention over all edges, with s3's z-score and the eight KT
  columns fed at every layer;
- edge-family dropout inside the stack in training (FD 0.25, G2's rule), drawn from the stack's own generator;
- the readout starts at zero, so zg2x starts as zspx bit for bit.

**What the stack reads.** s3's z-score is detached, as in G2: the stack learns on top of zspx's score, and zspx's
steps learn through their own path.

**What zspx's steps read.** Every edge, in training and at read. Only the stack drops families, as in G2.

**Parameters added:** G2's stack, 121,418 at hidden 128. No new graph, column, encoder or text. Settings, seeds,
batches, KT input, loss and loop are all zspx's.

**Bit-for-bit anchors.** With the stack's readout at zero, zg2x is zspx. With zspx's three step heads at zero, zg2x
is zg2. Without links, zg2x is zspx's scorer.

**GNN only.** The MLP takes no message.

## 3. The arm

| arm | the model | the carve |
| --- | --- | --- |
| zg2x | zspx's, then G2's stack over its score | zspx's (G3's: zspk's with c and s beside) |

## 4. The fits and the call

- **Fits:** one card fit on L-musique and one on L-hotpotqa (`scr-a10-zg2x`, `-hp`), on P1c2's roots with P1c3's KT
  input. Card fit settings: variant p, config 2e-3:1e-4:0.1:8:2, seed 0, hidden 128 (zspx's).
- **Reads:** each fit on the six s1eval carves at p@swa.
- **Compares:** each fit against zspx's card fit of the same split (scr-g5-zspx, scr-g5-zspx-hp), which decides.
  Step 1's fit is reported beside.
- **The re-call's base screens:** A9's, zspx's card fits compared with step 1's of the same split
  (outputs/a9/base-zspx-<split>).
- **Call:** relz's pair and its re-call under the seed null (the four null files of docs/SCREENS.md). GAIN, LOSS and
  WITHIN are docs/SCREENS.md's, on R@5.

### The rule, fixed before any number

- **ADOPT_FOR_FULL_RUN:** zg2x's re-call has no LOSS on any of its twelve reads and at least one GAIN;
- **NOT_ADOPTED:** otherwise;
- **INCOMPLETE:** its re-call is missing, is not A10's re-call of zg2x, or does not hold the twelve reads.

The verdict closes only this form (SCREENS.md section 3b), and it names the next form (section 5).

**Reported beside, deciding nothing:**
- webqsp and metaqa, both splits (the KB graphs: webqsp is the zero-shot KB);
- musique in domain (L-hotpotqa): Claim 1's read;
- hit@1 and FC@5 per read.

## 5. What each outcome leads to (fixed now)

| its re-call | the next form |
| --- | --- |
| ADOPT_FOR_FULL_RUN | zg2x fitted again on P1g-b's pools, then its full run beside zspx's (its own declared file) |
| NOT_ADOPTED, with at least one GAIN and a LOSS | A10b, the back-off form: the stack's readout bounded at 0.2 instead of 0.5 (the MP-unified track's 0.2 tanh), so the layers can move each score less (one declared file) |
| NOT_ADOPTED, with no GAIN | A10e, the stack end to end: its z-score of zspx's score with the gradient kept (as G4's zspkg found depth pays when the step acts on the score), everything else G2's (one declared file) |
| INCOMPLETE | the missing re-call is run; nothing else changes |

No model change is adopted on today's pools (the user's rule of 13 October).

## 6. How it runs

**Self-test, passed before this commit.** `python outputs/mp_unified/za10g2x.py --selftest` (8 s) checks, on a toy
carve with G3's residuals and the KT input:
- **The arm.** G2Mix over zspx, on G3's carve; zspx's, zg2's and zspk's arms are unchanged; A9's base screens are the
  re-call's.
- **The weights.** zg2x's state is zspx's plus G2's stack, each drawn as in its own arm from the same seed; the
  stack holds G2's parameter count.
- **Bit-for-bit cases.**
  - At the start (readout zero), zg2x is zspx, with zspx's step heads moved so all three steps act.
  - With the readout moved, zg2x moves off zspx by at most the bound 0.5.
  - With zspx's three step heads at zero, zg2x is zg2.
  - Without links, zg2x is zspx's scorer.
  - In training, the stack's dropout draws from its own generator, and with the readout at zero zg2x is zspx under
    the same global seed.
- **Toy fits through lean_gpu's loop.** A repeat is bit for bit; the stack's readout learns; without links the fit is
  zspx's bit for bit on every shared key, the readout staying at zero.
- **The rest.** The train and read refusals; relz's pair and re-call stamped A10's; the verdict's outcomes and next
  forms.

**Host items** (`a10-*`), after A9's verdict (9 items):
- 2 trains and 2 reads. GPU: G2's stack takes most of the memory and zspx's three steps add a little, so G2's
  measured `--frac 0.45` gets a margin: `--frac 0.48` (share 0.50). Memory: trains 6 GB (-hp 15 GB), reads 6 GB
  (zspx's sizes, which exceed G2's);
- the trains wait on A9's rebuild of G3's residual arrays for their split's fit carves, the reads on the six s1eval
  carves';
- 2 compares, the pair, the re-call (after A9's base compares) and the verdict.

## 7. Results

Filed 19 October 2026 from the host's records (`outputs/a10/screen-verdict.json`, written 2026-10-18T07:57:02Z; the
re-calls `outputs/screen/scr-a10-zg2x-pair-recall.json`, host paths made relative). **zg2x NOT_ADOPTED:** no GAIN and no
LOSS after the seed-null re-call.

- **zg2x: 0 GAIN, 0 LOSS.** 2wiki in domain moves -0.0056 [-0.0071, -0.0041] on L-musique and +0.0054 [+0.0039, +0.0068]
  on L-hotpotqa: opposite signs, both WITHIN. Every other read is within 0.007.
- **Beside:** webqsp zero-shot +0.0050 (L-musique) and -0.0062 (L-hotpotqa); metaqa in domain +0.0012 and +0.0001;
  musique in domain (L-hotpotqa, Claim 1's read) -0.0033, all WITHIN.

G2's three hidden-state layers over zspx's detached z-score add nothing past the seed null. This closes only this form.
The next form, fixed in section 5 and recorded in the verdict, is **A10e**, the stack end to end: its z-score of zspx's
score with the gradient kept (G4's zspkg found that depth pays when the step acts on the score), everything else G2's;
one declared file.

Each arm's twelve reads after the seed-null re-call (delta = the arm's R@5 minus its base's; a call is shown with its
filed call where the re-call changed it; base R@5 shown as zspx's):

| split | dataset | read | base R@5 | zg2x: delta [95% CI], call |
| --- | --- | --- | --- | --- |
| L-musique | metaqa | in-domain | 0.8030 | +0.0012 [+0.0002, +0.0021], WITHIN |
| L-musique | squad | in-domain | 0.9092 | +0.0019 [-0.0001, +0.0038], WITHIN |
| L-musique | musique | zero-shot | 0.5073 | +0.0011 [-0.0042, +0.0063], WITHIN |
| L-musique | hotpotqa | in-domain | 0.9074 | -0.0018 [-0.0041, +0.0004], WITHIN |
| L-musique | 2wiki | in-domain | 0.9351 | -0.0056 [-0.0071, -0.0041], WITHIN |
| L-musique | webqsp | zero-shot | 0.3204 | +0.0050 [-0.0015, +0.0119], WITHIN |
| L-hotpotqa | metaqa | in-domain | 0.8039 | +0.0001 [-0.0007, +0.0009], WITHIN |
| L-hotpotqa | squad | in-domain | 0.9129 | +0.0013 [-0.0006, +0.0031], WITHIN |
| L-hotpotqa | musique | in-domain | 0.5901 | -0.0033 [-0.0082, +0.0016], WITHIN |
| L-hotpotqa | hotpotqa | zero-shot | 0.8547 | -0.0014 [-0.0043, +0.0016], WITHIN |
| L-hotpotqa | 2wiki | in-domain | 0.9312 | +0.0054 [+0.0039, +0.0068], WITHIN |
| L-hotpotqa | webqsp | zero-shot | 0.2914 | -0.0062 [-0.0144, +0.0022], WITHIN |

Beside, deciding nothing: each read's FC@5 and hit@1 deltas against the same base, from the compare screens
(`outputs/screen/scr-a10-zg2x.json` and the others, `-hp` for L-hotpotqa):

| split | dataset | zg2x FC@5 | zg2x hit@1 |
| --- | --- | ---: | ---: |
| L-musique | metaqa | +0.0018 | +0.0028 |
| L-musique | squad | +0.0019 | +0.0005 |
| L-musique | musique | -0.0070 | +0.0323 |
| L-musique | hotpotqa | -0.0032 | +0.0038 |
| L-musique | 2wiki | -0.0149 | +0.0086 |
| L-musique | webqsp | +0.0053 | -0.0007 |
| L-hotpotqa | metaqa | -0.0009 | -0.0012 |
| L-hotpotqa | squad | +0.0013 | -0.0023 |
| L-hotpotqa | musique | -0.0074 | -0.0091 |
| L-hotpotqa | hotpotqa | -0.0034 | +0.0036 |
| L-hotpotqa | 2wiki | +0.0135 | +0.0013 |
| L-hotpotqa | webqsp | -0.0013 | +0.0000 |
