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

Filed 27 October 2026 from the host's records (`outputs/a4fx/screen-verdict.json`, written 2026-10-20T12:26:12Z; the
re-calls `outputs/screen/scr-a4fx-zrckprf-pair-recall.json` and `scr-a4fx-zspxf-pair-recall.json`, host paths made
relative). **Both arms NOT_ADOPTED, with no GAIN:** A4f's family dropout in the adopted arms' training moves no read up
past its floor on either model. Both lose musique zero-shot on L-musique by more than twice its floor, and the MLP's arm
loses two more reads.

- **zrckprf: 0 GAIN, 3 LOSS.** musique zero-shot on L-musique: -0.1663 [-0.1766, -0.1560] against its 0.0720 floor, with
  FC@5 -0.0956 and hit@1 -0.3240. webqsp zero-shot on L-musique: -0.0220 [-0.0345, -0.0100] against its 0.0102 floor,
  with FC@5 -0.0220. musique in domain on L-hotpotqa: -0.0077 [-0.0134, -0.0020] against its 0.0075 floor, with FC@5
  -0.0108. Seven more intervals exclude 0, all inside their floors: on L-musique, metaqa -0.0018 [-0.0032, -0.0004],
  squad -0.0030 [-0.0051, -0.0010], hotpotqa -0.0045 [-0.0069, -0.0022] and 2wiki -0.0051 [-0.0067, -0.0035]; on
  L-hotpotqa, metaqa -0.0073 [-0.0088, -0.0059], hotpotqa zero-shot +0.0054 [+0.0022, +0.0086] and 2wiki -0.0050
  [-0.0068, -0.0032]. No call changed under the re-call.
- **zspxf: 0 GAIN, 1 LOSS.** musique zero-shot on L-musique: -0.1938 [-0.2051, -0.1830] against its 0.0720 floor, with
  FC@5 -0.1134 and hit@1 -0.3471. Four more intervals exclude 0, all inside their floors: on L-musique, metaqa -0.0034
  [-0.0048, -0.0021] and 2wiki -0.0041 [-0.0056, -0.0025]; on L-hotpotqa, metaqa -0.0025 [-0.0039, -0.0012] and 2wiki
  -0.0043 [-0.0060, -0.0027]. No call changed under the re-call.
- **Beside: the adopted arms lose what A4f's lean arms lost.** On the lean pair (A4f), the same dropout lost musique
  zero-shot on L-musique on both models (zrckpf -0.1147, zspkf -0.2264), with webqsp zero-shot on L-hotpotqa on the MLP
  (-0.0236) and on L-musique on the GNN (-0.0125). On the adopted arms, musique zero-shot on L-musique falls -0.1663 on
  the MLP and -0.1938 on the GNN.

Not adopted. This closes only this form: A4f's family dropout (each of zret's blocks and the KT, chain and, on zrckpr,
R1 terms dropped at 0.2 per question in training) on the adopted arms (docs/SCREENS.md section 3b). The next form, fixed
in section 5 and recorded in the verdict for both arms, is A4fxt, the term cells alone on the adopted arms (the KT,
chain and, on zrckpr, R1 terms dropped at 0.2 per question; zret's blocks never), its own declared file.

The floors are the seed null of 7 October (docs/SCREENS.md section 2: `outputs/screen/scr-null-s1.json`,
`scr-null-s2.json` and their `-hp` files), not N2's fresh floors (docs/N2_FRESH_SEED_NULL.md).

Records, filed with this section: the verdict `outputs/a4fx/screen-verdict.json` and `.md`; for each arm (`zrckprf`,
`zspxf`), the compares `outputs/screen/scr-a4fx-<arm>.json` and `-hp.json`, the pair `-pair.json` and its re-call
`-pair-recall.json`, each with its `.md`. The host paths were made relative in each re-call's two `rel` entries and
nowhere else, so the `recall_sha256` each verdict holds is the host file's, from before that change. The re-calls' nulls
also read the base screens `outputs/a9/base-zspx-L-musique.json` and `outputs/a9/base-zspx-L-hotpotqa.json`, fetched
from the host with this filing (A9's own filing had left them there); and
`outputs/p1gc/base-pools-zrckpr-L-musique.json` and `outputs/p1gc/base-pools-zrckpr-L-hotpotqa.json`, filed with P1g-c's
results.

Each arm's twelve reads after the seed-null re-call (delta = the arm's R@5 minus its base's, the base's card fit of the
split; a call is shown with its filed call where the re-call changed it):

| split | dataset | read | zrckpr R@5 | zrckprf: delta [95% CI], call | zspx R@5 | zspxf: delta [95% CI], call |
| --- | --- | --- | --- | --- | --- | --- |
| L-musique | metaqa | in-domain | 0.8012 | -0.0018 [-0.0032, -0.0004], WITHIN | 0.8030 | -0.0034 [-0.0048, -0.0021], WITHIN |
| L-musique | squad | in-domain | 0.9127 | -0.0030 [-0.0051, -0.0010], WITHIN | 0.9092 | -0.0011 [-0.0033, +0.0011], WITHIN |
| L-musique | musique | zero-shot | 0.5197 | -0.1663 [-0.1766, -0.1560], **LOSS** | 0.5073 | -0.1938 [-0.2051, -0.1830], **LOSS** |
| L-musique | hotpotqa | in-domain | 0.9083 | -0.0045 [-0.0069, -0.0022], WITHIN | 0.9074 | -0.0006 [-0.0028, +0.0018], WITHIN |
| L-musique | 2wiki | in-domain | 0.9361 | -0.0051 [-0.0067, -0.0035], WITHIN | 0.9351 | -0.0041 [-0.0056, -0.0025], WITHIN |
| L-musique | webqsp | zero-shot | 0.3762 | -0.0220 [-0.0345, -0.0100], **LOSS** | 0.3204 | -0.0057 [-0.0176, +0.0058], WITHIN |
| L-hotpotqa | metaqa | in-domain | 0.8040 | -0.0073 [-0.0088, -0.0059], WITHIN | 0.8039 | -0.0025 [-0.0039, -0.0012], WITHIN |
| L-hotpotqa | squad | in-domain | 0.9114 | -0.0005 [-0.0029, +0.0018], WITHIN | 0.9129 | -0.0010 [-0.0029, +0.0009], WITHIN |
| L-hotpotqa | musique | in-domain | 0.5779 | -0.0077 [-0.0134, -0.0020], **LOSS** | 0.5901 | -0.0049 [-0.0100, +0.0003], WITHIN |
| L-hotpotqa | hotpotqa | zero-shot | 0.8558 | +0.0054 [+0.0022, +0.0086], WITHIN | 0.8547 | +0.0028 [-0.0005, +0.0063], WITHIN |
| L-hotpotqa | 2wiki | in-domain | 0.9331 | -0.0050 [-0.0068, -0.0032], WITHIN | 0.9312 | -0.0043 [-0.0060, -0.0027], WITHIN |
| L-hotpotqa | webqsp | zero-shot | 0.3411 | +0.0120 [-0.0015, +0.0250], WITHIN | 0.2914 | +0.0087 [-0.0009, +0.0187], WITHIN |

Beside, deciding nothing: each read's FC@5 and hit@1 deltas against the same base, from the compare screens
(`outputs/screen/scr-a4fx-<arm>.json` and `-hp.json`); and A4f's R@5 deltas with the same dropout on the lean pair
(zrckpf against zrckp, zspkf against zspk; their own re-calls, `outputs/screen/scr-a4f-<arm>-pair-recall.json`), on the
same read (a call is shown where it is not WITHIN):

| split | dataset | zrckprf FC@5 | zrckprf hit@1 | zspxf FC@5 | zspxf hit@1 | zrckpf (A4f) R@5 | zspkf (A4f) R@5 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| L-musique | metaqa | -0.0017 | -0.0250 | -0.0040 | -0.0164 | -0.0038 | -0.0039 |
| L-musique | squad | -0.0030 | -0.0039 | -0.0011 | -0.0045 | -0.0008 | -0.0009 |
| L-musique | musique | -0.0956 | -0.3240 | -0.1134 | -0.3471 | -0.1147 LOSS | -0.2264 LOSS |
| L-musique | hotpotqa | -0.0077 | -0.0070 | +0.0001 | -0.0059 | -0.0022 | -0.0023 |
| L-musique | 2wiki | -0.0149 | -0.0041 | -0.0136 | +0.0069 | -0.0024 | -0.0054 |
| L-musique | webqsp | -0.0220 | -0.0226 | -0.0067 | -0.0200 | +0.0026 | -0.0125 LOSS |
| L-hotpotqa | metaqa | -0.0097 | -0.0273 | -0.0044 | -0.0160 | -0.0057 | -0.0017 |
| L-hotpotqa | squad | -0.0005 | -0.0006 | -0.0010 | -0.0040 | +0.0003 | -0.0022 |
| L-hotpotqa | musique | -0.0108 | -0.0062 | -0.0108 | -0.0066 | -0.0035 | +0.0021 |
| L-hotpotqa | hotpotqa | +0.0107 | +0.0014 | +0.0069 | -0.0061 | -0.0024 | +0.0039 |
| L-hotpotqa | 2wiki | -0.0111 | -0.0068 | -0.0142 | -0.0052 | -0.0051 | +0.0030 |
| L-hotpotqa | webqsp | +0.0053 | -0.0060 | +0.0060 | +0.0133 | -0.0236 LOSS | -0.0128 |
