# A4fx: A4f's family dropout on the adopted arms, in both models' training

Declared 19 October 2026, before any number. The next form A4f's verdict named for zrckpf and zspkf
(docs/A4F_FAMILY_DROPOUT.md, section 5; outputs/a4f/screen-verdict.json), under docs/SCREENS.md section 3b and
docs/VERDICT_AUDIT_2026_10_17.md. Code: `outputs/mp_unified/za4fx.py`.

## 1. Why this, now

A4f dropped, per question in training, each of zret's blocks, the KT head's term and rmatch's chain term with
probability 0.2, on zrckp's and zspk's models. Both arms were NOT_ADOPTED with no GAIN. The reads that moved, after the
seed-null re-call (outputs/screen/scr-a4f-zrckpf-pair-recall.json, outputs/screen/scr-a4f-zspkf-pair-recall.json):

| arm, against its base | read | delta R@5 [95% CI] | floor | call |
| --- | --- | --- | ---: | --- |
| zrckpf, against zrckp | L-musique musique (zero-shot) | −0.1147 [−0.1246, −0.1048] | 0.0720 | LOSS |
| zrckpf, against zrckp | L-hotpotqa webqsp (zero-shot) | −0.0236 [−0.0358, −0.0115] | 0.0233 | LOSS |
| zspkf, against zspk | L-musique musique (zero-shot) | −0.2264 [−0.2372, −0.2158] | 0.0720 | LOSS |
| zspkf, against zspk | L-musique webqsp (zero-shot) | −0.0125 [−0.0243, −0.0010] | 0.0102 | LOSS |
| zspkf, against zspk | L-hotpotqa webqsp (zero-shot) | −0.0128 [−0.0248, −0.0010] | 0.0233 | WITHIN (filed LOSS) |

The other nineteen reads were WITHIN, none beyond ±0.0057. So dropping the two terms as well did not undo A4's
musique zero-shot loss (A4: zrckpd −0.1195, zspkd −0.1394, against the same bases).

Since A4f was declared, two arms passed their screens and are now the adopted arms:
- **G5's zspx** against zspk (docs/G5_COMBINED_GNN.md): 2 GAIN, 0 LOSS. L-hotpotqa musique +0.0147 [+0.0095, +0.0198]
  (floor 0.0075) and L-hotpotqa webqsp +0.0235 [+0.0133, +0.0327] (floor 0.0233).
- **R1's zrckpr** against zrckp (docs/R1_RESIDUAL_INPUT.md): 1 GAIN, 0 LOSS. L-musique webqsp +0.0135
  [+0.0032, +0.0239] (floor 0.0102). Its filed GAIN on L-musique musique (+0.0076 [+0.0028, +0.0125]) was WITHIN after
  the re-call (floor 0.0720).

A4f's section 5 fixed A4fx as the next form after no GAIN: the family form on these two arms, with R1's columns as one
more cell. The reading behind A4f still applies: a term that is never dropped is one a model can lean on. On zrckpr the
R1 head is a third input above zret, so it gets its own cell, as the KT and chain terms did in A4f.

## 2. The form

In a training forward, each question draws len(blocks) + 2 cells (zspxf) or len(blocks) + 3 cells (zrckprf) from the
model's own generator (A4f's seed, seed + 101), each kept with probability **0.8** (A4f's rate):

| cell | when it is dropped |
| --- | --- |
| the first len(blocks) | multiply the keep flags: A4 and A4f exactly (a dropped block's columns and flag are 0) |
| the KT cell | the KT head's term is 0 on the question's rows (A4f) |
| the chain cell | rmatch's chain term is 0 on the question's rows (A4f) |
| the R1 cell (zrckprf only) | R1's head's term is 0 on the question's rows |

- **A4f's rate, generator and order.** The rate (0.2) and the seed offset (101) are imported from za4f.py, not copied.
  The cells are drawn in the order of the table. With A4f's two terms the cells are A4f's bit for bit (the self-test
  checks this against za4f's own arms).
- **The R1 cell drops the head's term, not the columns.** Zeroing the nine columns would leave the head's bias on. With
  the term at 0 the question sees zrckpr's score without R1's input, the same treatment A4f gives the KT term.
- **The chain term keeps SEMB's gate.** At a read it needs SEMB, so it is off when either its own cell or SEMB's is.
- **No rescaling.** A kept term is exactly the read's term.
- **Reads keep everything.** Both models serve exactly as their bases do.
- **Nothing else changes.** No new parameter, no score read; the global random stream, dropout's masks included, is
  the base's.

The masks act inside the models' own forwards. A4FXTop wraps the KT head (`kt_head`) and rmatch's chain features
(`chain_feats`); on zrckprf, R1Cell also wraps R1's head (`r1_head`). Each multiplies its term by the question's cell
on each row. On zspx both terms are added before its three increments read the score (zspk's step with the gradient
kept, G3's residual message, G4's second step), so a dropped term is dropped for all three. zspx has no R1 columns, so
zspxf draws A4f's cells. The increments' edge families are A9's (docs/A9_EDGE_FAMILY_DROPOUT.md), not this file's.

**The MLP rule.** zrckprf changes only which inputs a training forward sees. The R1 cell multiplies the head's output
by 0 or 1 per question, drawn from the generator. It reads no score and normalises nothing over the question's scores.

## 3. The arms

| arm | model | carve |
| --- | --- | --- |
| zrckprf | zrckpr's model with R1Cell and A4FXTop at the top (MRO: R1Cell, A4FXTop, ZRCKPR, ZRCKP, KTMixP, ZRM, ChainMatch, ZRet) | zrckpr's (`zr1res.R1KTChainCarveZRC`) |
| zspxf | zspx's model with A4FXTop at the top (MRO: A4FXTop, ZSPX, ZSPR, ZSPK, ZProp, KTMix, ZRM, ChainMatch, ZRet) | zspx's (`zg3res.ResKTLinkCarveBase`) |

A4FXTop is A4f's A4FTop with its term cells listed per class; R1Cell adds the R1 cell to it.

## 4. The fits and the call

A4f's, on the adopted arms:
- **Fits:** one card fit per arm on L-musique and one on L-hotpotqa (`scr-a4fx-<arm>`, `-hp`), on P1c2's roots with
  P1c3's KT input and the base arm's own columns (R1's for zrckpr, G3's c and s for zspx), at the card fits' settings
  (variant p, config 2e-3:1e-4:0.1:8:2, seed 0, hidden 128).
- **Reads:** each fit on the six s1eval carves at p@swa.
- **Compares:** each fit against its base arm's card fit of the same split (zrckpr: scr-r1-zrckpr and -hp; zspx:
  scr-g5-zspx and -hp), step 1's fit beside.
- **Base screens for the re-call:** each base fit against step 1's. For zrckpr they are P1g-c's
  (`outputs/p1gc/base-pools-zrckpr-<split>.json`); for zspx, A9's (`outputs/a9/base-zspx-<split>.json`).
- **Call:** relz's pair and its re-call under the seed null (the four null files of docs/SCREENS.md), per arm.

### The rule, fixed before any number

Per arm:
- **ADOPT_FOR_FULL_RUN:** its re-call has no LOSS on any of its twelve reads and at least one GAIN;
- **NOT_ADOPTED:** otherwise;
- **INCOMPLETE:** its re-call is missing, is not A4fx's re-call of that arm, or does not hold the twelve reads.

The verdict closes only this form (SCREENS.md section 3b), and it names the next form (section 5).

**Reported beside, deciding nothing:**
- musique on both splits (A4f's loss zero-shot, Claim 1's read in domain);
- webqsp zero-shot (zspx's and zrckpr's gains);
- metaqa;
- hit@1 and FC@5 per read;
- per fit, the share of cells dropped and how often each term was dropped.

When the results are filed, A4f's deltas (section 1) are set beside each read.

**Serving cost: none.** Reads keep everything.

## 5. What each outcome leads to (fixed now)

| its re-call | the next form |
| --- | --- |
| ADOPT_FOR_FULL_RUN | the arm fitted again on P1g-b's pools, then its full run, its own declared file |
| NOT_ADOPTED, with at least one GAIN and a LOSS | A4fxh, the family form at half the rate (0.1 per question on each cell) on the adopted arms, its own declared file |
| NOT_ADOPTED, with no GAIN | A4fxt, the term cells alone on the adopted arms (the KT, chain and, on zrckpr, R1 terms dropped at 0.2 per question; zret's blocks never), its own declared file |
| INCOMPLETE | the missing re-call is run; nothing else changes |

Why these two forms:
- **A4fxh.** A GAIN is signal, so the form backs off instead of closing. Half the rate is the back-off A4f named for
  itself.
- **A4fxt.** Every musique zero-shot loss so far came with zret's block cells on: A4 dropped the blocks only, A4f the
  blocks and the terms. The term cells have never run alone. A4fxt tests them alone, on the same arms.

No model change is adopted on today's pools (the user's rule of 13 October): an adopt is refitted on P1g-b's pools
before its full run.

## 6. How it runs

```
python outputs/mp_unified/za4fx.py train --split L-musique --name scr-a4fx-zrckprf --arm zrckprf --device cuda --host --cache-root outputs/p1c2/cache
python outputs/mp_unified/za4fx.py read --name scr-a4fx-zrckprf --device cuda --host --cache-root outputs/p1c2/cache
python outputs/mp_unified/za4fx.py compare --new outputs/screen/fits/scr-a4fx-zrckprf --base outputs/screen/fits/scr-r1-zrckpr,outputs/step1/fits/L-musique --out outputs/screen/scr-a4fx-zrckprf
python outputs/mp_unified/za4fx.py pair --arm zrckprf --screens outputs/screen/scr-a4fx-zrckprf.json,outputs/screen/scr-a4fx-zrckprf-hp.json --out outputs/screen/scr-a4fx-zrckprf-pair
python outputs/mp_unified/za4fx.py recall --arm zrckprf --null outputs/screen/scr-null-s1.json,outputs/screen/scr-null-s2.json,outputs/screen/scr-null-s1-hp.json,outputs/screen/scr-null-s2-hp.json --pair outputs/screen/scr-a4fx-zrckprf-pair.json --out outputs/screen/scr-a4fx-zrckprf-pair-recall
python outputs/mp_unified/za4fx.py verdict
```

The same for zspxf (base fit scr-g5-zspx), and for L-hotpotqa with `--split L-hotpotqa` and the `-hp` names.

**Self-test.** `python outputs/mp_unified/za4fx.py --selftest` passes before this commit (5.6 s). On toy carves that
carry the KT input, rmatch's chains, zlink's edges, G3's c and s and R1's columns, it checks:
- the arms' MROs, with A4FXTop at the top (R1Cell above it on zrckprf); the base arms and A4f's arms unchanged; A4f's
  rate and seed;
- a draw of 100,000 cells keeps 0.8 within 0.01;
- A4FXTop with A4f's two terms, over zrckp's and zspk's models, is za4f's zrckpf and zspkf in training, bit for bit:
  scores, every gradient, the generators' states and the counts, over three forwards;
- each arm against its base (the KT head's last layer, rmatch's gate, R1's head's last layer and zspx's three heads set
  nonzero, so every term moves scores):
  - in eval mode bit for bit, with no draw, and the row masks cleared;
  - in training at a rate of 0 bit for bit: scores, every gradient, and the global stream;
  - with fixed cells (a block, SEMB, each term, and pairs and triples of them off on different questions), each
    question's rows equal the base model's forward on keep times its block cells, with the KT head's last layer at zero
    where its KT cell is off, rmatch's gate at zero where its chain cell is off and R1's head's last layer at zero where
    its R1 cell is off, bit for bit; the scores move only on those questions, each term's own cell moves its question,
    and the counts are the cells drawn;
  - a question whose SEMB is dropped loses the chain term;
- toy fits through lean_gpu's loop repeat bit for bit, equal the base arm's at a rate of 0, and differ with A4fx; the
  R1 cell drops on zrckprf only;
- the refusals; relz's pair and re-call per arm, stamped A4fx's; the verdict's outcomes and next forms, and a re-call
  with eleven reads is INCOMPLETE.

**Host items** (17, `a4fx-*`), sized as the base arms' own screens:
- zrckprf at R1's sizes: `--frac 0.30`, GPU share 0.32, 4.5 GB for the trains and 6 GB for the reads.
- zspxf at A9's sizes for zspx: `--frac 0.40`, GPU share 0.42, 6 GB, and 15 GB for the L-hotpotqa train.
- A4fx adds no parameter and no memory.
- zrckprf's trains and reads wait on the n19-r1c-* rebuilds of R1's columns, by R1's carve lists. zspxf's wait on the
  n19-res-* rebuilds of G3's c and s, by A9's lists. Both sets are the shared rebuilds of 19 October; A4fx adds no
  rebuild item.
- zrckprf's re-call waits on P1g-c's base screens (p1gc-base-pools-zrckpr-L-musique and -L-hotpotqa); zspxf's waits
  on A9's (a9-base-zspx and -hp).
- Compares take 1 CPU and 4 GB; pairs, re-calls and the verdict take 1 CPU and 2 GB.

**ETA.** G3's and G4's fits took 47 to 54 minutes each on the card, and their reads about 5 minutes
(docs/G5_COMBINED_GNN.md). E1's fit of zspk's model took 47 minutes alone on the card (docs/R1_RESIDUAL_INPUT.md).

The encoder stays frozen. webqsp never trains. Test splits are never read. No LLM or transformer, and no new graph.

## 7. Results

Not run yet.
