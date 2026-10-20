# G6: the free GNN, its own model on the lean inputs

Declared 20 October 2026, before any number. The user's direction of 20 October (the GNN is not to be built on the MLP). Code: `outputs/mp_unified/zg6.py`. Not run yet.

## 1. Why this, now

Every GNN screened since 15 October starts as the MLP and moves a little from it:
- **zspk**, the lean GNN, is zrckp's network plus one propagation step over the neighbours' z-scored scores (P1c3).
- **G2's zg2 and A10's zg2x** put three hidden-state layers over zspk's and zspx's score, the readout bounded to ±0.5
  around it. Neither found a GAIN after the seed-null re-call (G2: 0 GAIN, 0 LOSS, docs/G2_HIDDEN_STATE_GNN.md
  section 7; A10: 0 GAIN, 0 LOSS, every read within 0.007, docs/A10_G2_ON_ZSPX.md section 7).
- **G5's zspx**, the adopted GNN, is zspk with its step trained end to end and two more steps on the score, each
  starting at zero, so it starts as zspk (2 GAIN, 0 LOSS, docs/G5_COMBINED_GNN.md section 5).

So the GNN and the MLP stay close. On the twelve screen reads, zspk is within 0.004 R@5 of zrckp, the MLP, on every
in-domain read, and below it on all four zero-shot reads (R@5 at p@swa on s1eval, from L1's compares
`outputs/l1/base-zspk-<split>.json` and `base-zrckp-<split>.json`):

| split | dataset | read | rrf | zspk | zrckp | zspk − zrckp |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| L-musique | metaqa | in-domain | 0.0047 | 0.8009 | 0.8019 | −0.0010 |
| L-musique | squad | in-domain | 0.9054 | 0.9114 | 0.9116 | −0.0002 |
| L-musique | musique | zero-shot | 0.4734 | 0.5012 | 0.5121 | −0.0109 |
| L-musique | hotpotqa | in-domain | 0.6846 | 0.9080 | 0.9065 | +0.0015 |
| L-musique | 2wiki | in-domain | 0.6079 | 0.9383 | 0.9347 | +0.0036 |
| L-musique | webqsp | zero-shot | 0.0535 | 0.3229 | 0.3627 | −0.0398 |
| L-hotpotqa | metaqa | in-domain | 0.0047 | 0.8040 | 0.8044 | −0.0004 |
| L-hotpotqa | squad | in-domain | 0.9054 | 0.9135 | 0.9127 | +0.0008 |
| L-hotpotqa | musique | in-domain | 0.4734 | 0.5754 | 0.5745 | +0.0009 |
| L-hotpotqa | hotpotqa | zero-shot | 0.6846 | 0.8499 | 0.8513 | −0.0014 |
| L-hotpotqa | 2wiki | in-domain | 0.6079 | 0.9333 | 0.9350 | −0.0017 |
| L-hotpotqa | webqsp | zero-shot | 0.0535 | 0.2680 | 0.3475 | −0.0795 |

Claim 1 (the MLP keeps most of the GNN's gain) is only fair when both models are at their best. The lean MLP keeps at
least 0.968 of the lean GNN's gain over the bar on R@5 and FC@5 on all six datasets (docs/P3A_SOTA_RANKERS.md). That
says little when the GNN is the MLP plus a small step. The universal GNN, gnn0, has its own network and passes hidden
states over three relation-aware layers, and there a gap shows: its MLP twin (twin0) keeps 0.596 of its R@5 gain on
musique and 0.852 on 2wiki, and gnn0 leads zsp on 2wiki (0.891 against 0.873, the same file). On HippoRAG 2's musique
questions, zsp's R@5 (×100) is 66.3 [64.5, 68.1] against HippoRAG 2's 74.7 (docs/B1B_MODELS_ON_HIPPORAG2.md).

**The question:** does a GNN that is its own model (its own weights, hidden states passed along the pool's edges, an
unbounded score, trained from scratch) do better than zspk on the same inputs?

## 2. The model

**zg6** has no MLP inside it. It does not read zspk's or zrckp's score, it is not added to one, and nothing bounds it.

- **Encoder.** zrm's block encoder up to the input of its output layer: each block's raw, z-scored and mask columns
  through two 128-wide layers (GELU, dropout 0.1). The inputs are the lean pair's; no new column, graph or text. The
  weights are zg6's own, drawn from seed 0: they start where zrm's draws start, and nothing trained is copied. zrm's
  output layer, its rrf skip and rmatch's chain gate are deleted, so no MLP score is ever formed.
- **Boundary.** xb = the row's eight KT columns (P1c3's KT input) and rmatch's two chain features (0 on a graph
  without chains). In zrm the chain features enter the score through the deleted gate; here they are inputs, and
  rmatch's chain-match weights train with the rest. h0 = GELU(W_in [H, xb] + b), 64 wide.
- **Context.** s0 = w0 · h0 for each row, z-scored over the question's rows and clipped to ±8. The softmax of that
  weights the question's current states into its context c, again at each layer. The gradient runs through s0: the
  model learns which rows make the context.
- **Three layers**, G2's layer op for op, over zlink's pool edges (both directions; the structural, named-entity and
  kNN families):
  - for an edge u → v of family f: a logit l = ⟨W_a h_u, c⟩ / 8 + β_f, clipped to ±8, and a message m = W_f h_u (one
    64 × 64 map per family);
  - at v: per family, the mean of sigmoid(l) · m over v's edges of that family; over all of v's edges, the attention
    average Σ exp(l) m / Σ exp(l). A row without edges gets 0;
  - the update h ← h + GELU(layer_norm(W_s h + W_mean [the three means] + W_att att + W_x xb + b)). xb enters every
    layer (NBFNet's boundary condition).
- **Readout.** score = w_o · h3 + w_s · h0 + b. Unbounded. w_o and w_s are random at the start, not zero, so zg6 does
  not start as any other model.
- **Edge-family dropout** (G2's fd25 rule), in training only: each question drops each family it has with probability
  0.25, keeping its structural family (or its lowest) when all would go. The draws come from the module's own
  generator.
- **Parameters.** G6's own: 121,802 at hidden 128 (the input map, context, three layers and readout; G2's stack had
  121,418). With the encoder and the chain match, about 346,500 in all (an estimate from zrckp's saved card fit, whose
  encoder input is 309 wide).

**zg6t, the twin,** is zg6 with no edges reaching its layers: the same parameters and readout. Every mean and attention
average is 0, so the context has no effect (it enters only through the edges' logits) and each row's score depends
only on its own inputs. zg6 minus zg6t is what the edges add to the same model.

**Training.** rmatch.py's train exactly as zspk's card fits: the listwise loss, eight epochs, config 2e-3:1e-4:0.1:8:2,
the weight average read as p@swa, seed 0, hidden 128, variant p. P1c3's carves: P1c2's pools at each question's frozen
size, with the KT input beside. On the card.

## 3. What is read

Four card fits:

| arm | L-musique | L-hotpotqa |
| --- | --- | --- |
| zg6 | scr-g6-zg6 | scr-g6-zg6-hp |
| zg6t | scr-g6-zg6t | scr-g6-zg6t-hp |

- **Reads:** each fit on the six s1eval carves at p@swa, in batches of at most 65,536 rows (`--rows-cap 65536`, whole
  questions). G2's stack ran out of card memory at the default 262,144 (docs/G2_HIDDEN_STATE_GNN.md section 7). Only
  the batch size changes.
- **The base, which decides:** zspk's P1c3 card fit of the same split (scr-p1c3-zspk, scr-p1c3-zspk-hp): the same
  pools, KT input, card and settings.
- **Compares:** zg6 against zspk, its twin and step 1's fit; the twin against zspk and step 1's fit.
- **Call:** relz's pair and its re-call, as in every screen since the tenth round. GAIN, LOSS and WITHIN are
  docs/SCREENS.md's, on R@5.

**Beside, deciding nothing:**
- the twin's re-call under the same null;
- zg6's re-call under the 7 October null (section 4), so its calls sit beside every earlier screen's;
- zg6 minus zg6t and zg6 minus zspk on each read: R@5 with its 95% question-bootstrap interval, FC@5 and hit@1;
- R@5 per read of rrf, zspk, zg6, zg6t, zrckp (L1's compares), G5's zspx and G2's zg2 (their filed re-calls);
- the MLP share, (zrckp − rrf) / (zg6 − rrf), where zg6's R@5 gain over rrf is above 0.002.

## 4. The rule, fixed before any number

**The null that decides is N2's.** The floors so far come from 7 October: step 1's MLP fitted again with seeds 1 and
2 (docs/SCREENS.md section 2). N2 (docs/N2_FRESH_SEED_NULL.md, declared beside this file) fits zspk itself again with
seeds 1 and 2 on both splits and compares each with its seed-0 card fit (`outputs/screen/scr-n2-zspk-s1`, `-s2`,
`-s1-hp`, `-s2-hp`). For each read, with D1 and D2 the two seeds' R@5 differences, the floor is
max(0.0075, 2 · sqrt((D1² + D2²) / 2)). A filed GAIN stays a GAIN only when its difference reaches the floor; the same
for a LOSS.

- **ADOPT_FOR_FULL_RUN:** zg6's re-call under N2's null has no LOSS on any of its twelve reads (six datasets × two
  fits; webqsp zero-shot in both, and each fit's held-out dataset zero-shot) and at least one GAIN;
- **NOT_ADOPTED:** otherwise;
- **INCOMPLETE:** that re-call is missing, is not G6's re-call of zg6 under N2's null, or does not hold the twelve
  reads.

**Under the 7 October null, beside.** zg6's re-call with the 7 October null, re-based to zspk's card fits through L1's
compares (`outputs/l1/base-zspk-<split>.json`), as G5's was. It decides nothing.

**A limit, disclosed.** N2 measures zspk's seed spread, not zg6's. A model trained from scratch may vary more between
seeds. That is why an ADOPT here leads to a refit and a full run, not to a claim.

## 5. What each outcome leads to (fixed now)

| zg6's re-call under N2's null | what follows |
| --- | --- |
| ADOPT_FOR_FULL_RUN | zg6 fitted again on P1g-b's pools, then its full run on all six splits beside zspk's, then Claim 1 (P3c's intervals) and Claim 3 (cold latency, the C stages' timing) measured again; each its own declared file |
| NOT_ADOPTED, with at least one GAIN and a LOSS | G6b, a per-question back-off: zg6's score mixed with its twin's through a learned gate on the question's edge-family shares, at equal weight at the start (one declared file) |
| NOT_ADOPTED, with no GAIN | G6r, relation-aware messages from text: the relation embeddings (`outputs/m3b/relations/metaqa_rel_embeddings.npy`, `webqsp_rel_embeddings.npy`) through one shared low-rank map in place of the per-family maps on the KB datasets, the encoder started from zspk's trained encoder, still unbounded with its own readout (one declared file) |
| INCOMPLETE | the missing re-call is run; nothing else changes |

No verdict here closes the free GNN (docs/SCREENS.md section 3b): each outcome names its next form.

No model change is adopted on today's pools (the user's rule of 13 October).

## 6. Running it

**Self-test, passed before this commit.** `python outputs/mp_unified/zg6.py --selftest` (5 s) checks, on a toy carve
with the KT input and chains:
- **The arms.** zg6 and zg6t on P1c3's link carve. Their class builds on zrm's for the encoder only: zspk's, zprop's,
  the KT mix's and G2's classes are not in their class order. Earlier arms unchanged; 121,802 of G6's parameters at
  hidden 128.
- **The weights.** From seed 0 the encoder's draws are zrm's, less its output layer, rrf skip and chain gate; the
  readout starts nonzero; both arms start from the same state.
- **No MLP score.** With every ancestor's forward made to raise and the rrf base set to NaN, the score is the same.
- **The forward.** The encoder equals the input of zrm's output layer (eval and training); the chain features are
  rmatch's; G6's layers are G2's bit for bit; each arm equals a by-hand loop over questions and edges; the twin is the
  arm with its edges removed, bit for bit, in eval and training; the readout is unbounded (×100 in, ×100 out).
- **The dropout.** Its draws come from the module's own generator, in training only, by hand; never all of a
  question's families (501 draws).
- **The gradient** reaches the encoder, the chain match, every layer and the readout; the twin's edge maps get
  exactly 0.
- **Toy fits** repeat bit for bit and move.
- **The refusals.** train refuses another arm, split or fit name, a changed card setting, no cache root and the CPU.
  read refuses a fit that is not G6's (or not its own arm's and split's), a changed read setting and another cache
  root.
- **The call.** relz's pair and re-calls under N2's and the 7 October nulls, stamped G6's, with N2's loader refusals
  and relz restored after. The verdict's outcomes and next forms, its beside rows and the MLP share. Nothing is
  written under G6's roots.

**No development run on real data.** P1c2's caches and P1c3's KT files are on the host only.

**Host items** (`g6-*`, 18 in all):
- 4 trains on the card, the L-hotpotqa fits first, sized as G2's: `--frac 0.45`, share 0.47, two at a time;
- 4 reads after their fits, at `--rows-cap 65536`;
- on the CPU: 4 compares (the twin's first), 2 pairs, 3 re-calls and the verdict;
- memory by G2's measured peaks (each item's REF to its G2 counterpart, plus 0.5 GB, times 1.15): trains 3.5 GB
  (L-musique) and 5.3 GB (L-hotpotqa), reads 4.9 to 5.5 GB, the CPU items 0.6 GB;
- the two re-calls under N2's null wait for N2's four zspk compares (`n2-compare-zspk-s1`, `-s2`, `-s1-hp`,
  `-s2-hp`); the 7 October re-call waits for L1's base compares (`l1-base-zspk`, `-hp`), as G5's did; the verdict
  waits for all three.

No build: zg6 and zg6t read zspk's carve.

The commands (the L-hotpotqa fit is the same with `--split L-hotpotqa` and `-hp` names; zg6t with `--arm zg6t`):

    python outputs/mp_unified/zg6.py train --split L-musique --name scr-g6-zg6 --arm zg6 --device cuda --host \
        --cache-root outputs/p1c2/cache
    python outputs/mp_unified/zg6.py read --name scr-g6-zg6 --device cuda --host --cache-root outputs/p1c2/cache \
        --rows-cap 65536
    python outputs/mp_unified/zg6.py compare --new outputs/screen/fits/scr-g6-zg6 \
        --base outputs/screen/fits/scr-p1c3-zspk,outputs/screen/fits/scr-g6-zg6t,outputs/step1/fits/L-musique \
        --out outputs/screen/scr-g6-zg6
    python outputs/mp_unified/zg6.py pair --arm zg6 \
        --screens outputs/screen/scr-g6-zg6.json,outputs/screen/scr-g6-zg6-hp.json --out outputs/screen/scr-g6-zg6-pair
    python outputs/mp_unified/zg6.py recall --arm zg6 --null-kind N2 --null outputs/screen/scr-n2-zspk-s1.json,\
    outputs/screen/scr-n2-zspk-s2.json,outputs/screen/scr-n2-zspk-s1-hp.json,outputs/screen/scr-n2-zspk-s2-hp.json \
        --pair outputs/screen/scr-g6-zg6-pair.json --out outputs/screen/scr-g6-zg6-pair-recall
    python outputs/mp_unified/zg6.py recall --arm zg6 --null-kind oct7 --null outputs/screen/scr-null-s1.json,\
    outputs/screen/scr-null-s2.json,outputs/screen/scr-null-s1-hp.json,outputs/screen/scr-null-s2-hp.json \
        --pair outputs/screen/scr-g6-zg6-pair.json --out outputs/screen/scr-g6-zg6-pair-recall-oct7
    python outputs/mp_unified/zg6.py verdict

**Disk** (an estimate from zrckp's card fit): each fit writes about 12.5 MB of weights (nine states of about 346,500
parameters) and each read about 13.5 MB (six s1eval score files). The compares, pairs, re-calls and verdict are under
0.1 MB each. About 0.1 GB in all.

**ETA** (an estimate from G2's measured runs; G2's stack is the same size): G2's fits took 89 and 51 minutes on
L-hotpotqa and 56 and 38 on L-musique; its reads took 15.5 minutes at the cap and about 6 at the default. Two at a time
on a free card, about 2 hours 45 minutes from the first fit to the last read. The deciding re-call then waits for N2's
zspk fits.

The text encoder stays frozen. webqsp never trains. Test splits are never read. No LLM or transformer, and no new
graph.

## 7. Results

Not run yet.
