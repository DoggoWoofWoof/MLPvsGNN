# G4: deeper and end-to-end score propagation for the GNN

Declared 17 October 2026, before any number. Code: `outputs/mp_unified/zg4prop.py`. Two arms, both the GNN's. The
MLP takes no message (the user's rule of 8 October).

## 1. Why this, now

The user (15 October): "why cant the gnn be even better", and the GNN "should exploit message passing and other stuff
as much as possible". zspk, the lean pair's GNN, is zrc's scorer, the KT input and **one** step of score propagation
(zprop): per link family, the mean and soft maximum of its neighbours' z-scored scores, its degree and its own z, into
a 385-parameter head added to the score. Two things about that step limit what message passing can do:

- **It is one step.** A row reads its direct neighbours' scores. A gold two links from the found paragraph sees only
  the score of the middle node, which is usually not high, so it is not lifted. M1 (docs/M1_MUSIQUE_HOPS.md) found the
  next paragraph one *or two* links from the found one (0.56 of musique's missed golds; 0.50 on hotpotqa, 0.69 on
  2wiki), and D1 found the missed golds linked to found ones (REACHABLE_NOT_RANKED).
- **It carries no gradient into the scorer.** zprop's inputs are computed under `no_grad` from a detached score. The
  scorer is trained only to rank each row alone; it is never told that a higher score on a bridge would lift the gold
  behind it. The step can only re-weight what the scorer already emits.

G2 gave the GNN three layers of hidden-state messages with attention. It was not adopted (0 GAIN, 0 LOSS against
zspk). G4 keeps zspk's own message, the score, which was adopted, and changes only its depth or its training. Each is
tested on its own, so a result says which of the two matters.

## 2. The arms

| arm | model | parameters added | its base |
| --- | --- | --- | --- |
| zspk2 | zspk + a second step over zspk's full score (`ZSPK2`) | 385 | `scr-p1c3-zspk`, `-hp` |
| zspkg | zspk with its one step trained end to end (`ZSPKG`) | 0 | `scr-p1c3-zspk`, `-hp` |

**zspk2.** With s1 zspk's full score (its scorer, KT head and first step):

    s2 = s1 + h2(prop_inputs(s1))

- prop_inputs is zprop's, unchanged: per family the mean and soft maximum of the neighbours' z (clipped to ±8), log1p
  of the degree, and the row's own z; ten inputs, no gradient.
- h2 is zprop's head shape: Linear(10, 32), GELU, Linear(32, 1). Its last layer starts at zero, so zspk2 starts as
  zspk bit for bit. Its first layer is drawn from its own generator (seed + 4401).
- After the first step a bridge whose neighbour is found has been lifted, so in the second step the row behind the
  bridge reads a lifted score. A gold two links from a found row can now move.

**zspkg.** zspk's model and weights, and zspk's forward bit for bit for the same weights. Its one step's inputs are
computed with the gradient kept (no `no_grad`, no detach). The loss on a row now reaches its neighbours' scores, so the
scorer can learn to score a row in a way that helps the rows linked to it. At the start (the head's last layer at zero)
the gradient is zspk's; it differs as soon as the head moves.

Everything else is zspk's, unchanged: the carve, KT input, settings, seeds, batches, loss and loop. Neither arm adds a
graph, an input column, an encoder or text.

## 3. The fits and the call

- **Fits:** one card fit per arm on L-musique and one on L-hotpotqa (`scr-g4-<arm>`, `-hp`), on P1c2's roots and
  P1c3's KT input, as zspk's were.
- **Reads:** each fit on the six s1eval carves at p@swa.
- **Compares:** each fit against zspk's card fit of the same split, which decides. Step 1's fit is reported beside.
  The re-call's base screens are L1's (`outputs/l1/base-zspk-<split>.json`).
- **Call:** relz's pair and its re-call under the seed null (the four null files of docs/SCREENS.md), per arm. GAIN,
  LOSS and WITHIN are docs/SCREENS.md's, on R@5.

### The rule, fixed before any number

Per arm:
- **ADOPT_FOR_FULL_RUN:** its re-call has no LOSS on any of its twelve reads and at least one GAIN;
- **NOT_ADOPTED:** otherwise;
- **INCOMPLETE:** its re-call is missing, is not G4's re-call of that arm, or does not hold the twelve reads.

Under the user's rule of 13 October, no model change is adopted on today's pools. An arm that passes is fitted again on
P1g-b's pools before its full run (six splits under grade-nullx, its own file). If both pass, the two are combined in
that refit's file (a second step, both steps end to end), declared there before its numbers.

**Reported beside, deciding nothing:**
- webqsp (zero-shot in both splits) and metaqa (in-domain), per split and arm, with their calls and differences;
- hit@1 and FC@5 per read, as the compares give them.

## 4. How it runs

- **Self-test** (`python outputs/mp_unified/zg4prop.py --selftest`, passes before this commit):
  - the arms are zspk's model with a second step and with its step end to end, on zspk's carve. zspk's arm is
    unchanged;
  - `prop_inputs_grad` gives zprop's `prop_inputs` values bit for bit, and carries a gradient that zprop's drops;
  - zspk2 starts as zspk bit for bit (all of zspk's weights equal, the new head's last layer zero). With its head
    moved, the difference is that head over zprop's inputs of zspk's score. Without links, it is zspk's forward;
  - zspkg's forward equals zspk's bit for bit for the same weights, with zspk's head moved too. Its scorer's gradients
    equal zspk's while the propagation head's last layer is zero, and differ once it moves;
  - toy fits through lean_gpu's loop: a repeat fit is bit-identical; zspk2's second head learns; on the carve without
    links, zspk2's fit equals zspk's bit for bit on every shared key and zspkg's fit equals zspk's; with links,
    zspkg's fit moves away from zspk's;
  - train refuses another arm, split, fit name, cache root or device. read refuses a fit that is not G4's;
  - relz's pair and re-call per arm against zspk's card fits, stamped G4's; the verdict's outcomes per arm.
- **Host items** (`g4-*`, 17 in all, placed after G3's verdict and ahead of P1g-b's fits):
  - on the card: four trains, then four reads (`--frac 0.40`, gpu share 0.42, two at a time). They need nothing but
    P1c2's roots and P1c3's KT input, which are on the host;
  - on the CPU: four compares, two pairs, two re-calls and the verdict.
- **Disk:** about 0.1 GB (fits and reads, as zspk's).
- **ETA:** E1's fit of zspk's model on L-musique took 47 minutes on the card at share 0.32; with L-hotpotqa's larger
  fit and the reads, about 1 to 1.5 hours per fit. zspkg keeps the propagation's graph for the
  backward pass, so it may be slower. The card has room once T1's trains end (about 05:00); four fits two at a time
  take about 3 to 4 hours. The verdict is due about 09:00 to 10:00 on 17 October, before P1g-b's fits need the card.
- The encoder stays frozen. webqsp never trains. Test splits are never read. No LLM or transformer.

## 5. Results

Not run yet.
