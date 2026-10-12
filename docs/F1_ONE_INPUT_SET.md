# F1: one input set for the MLP and the GNN, and each model's own fastest set

Declared 12 October 2026, about 13:15, before any number of the stage. (Times in this file are corrected to the
commit times: they had been written ahead of the clock.) It sits under docs/PROGRAM_2026_10_09.md's
claims 2 and 3. Step 1 only reads existing fits; nothing trains before step 1's records exist.

## Why

The user, 12 October:
- The MLP and the GNN should read the **same inputs**, made of the families **both** actually use. Then the gap between
  them, in speed and in accuracy, is message passing's cost and value, and nothing else.
- Then each model on **its own** used families, at its fastest form. That is the ceiling: how fast each domain can be.
- Computing a column that no model uses wastes cold time.

Today the inputs differ (read from the code):

| model | inputs |
| --- | --- |
| zrc, the MLP's base | the pick: rank, dense_cos, topo_STRUCT, depth_STRUCT, SEMB, SEED, WALK, DISTS, WALKF; on typed graphs zrc's chain entries |
| zsp, the GNN track's base (claims 1 and 2, B1b) | the same pick; rmatch's chain entries; the pool edges, for one propagation step over zrm's scores |
| zgn (round 25) | the pick, the seven look blocks of round 24, rmatch's chain entries, the pool edges for two layers over hidden states |
| the six GNN (C1-C6's speed comparison) | the 197-column compile and the 1,536-wide embedding, two message-passing layers |

C3 measured zsp at 1.08-1.14× zrc's cold cost, and the six GNN at 1.6-3.0× zsp's. The filed speed ratio is taken
against the six GNN, whose inputs are not the MLP's. That ratio mixes the cost of more inputs with the cost of message
passing.

## Step 1: which families each model uses (no training)

- **Models.** p@swa of the screen fits on the card, seed 0:
  - zrc: scr-zrct (L-musique), scr-zrct-hp (L-hotpotqa);
  - zsp: scr-zspg, scr-zspg-hp;
  - zgn: scr-zgn, scr-zgn-hp (beside: message passing over hidden states with the MLP's inputs). Dropped by the
    amendment below: its caches are gone.
- **Reads.** The select carves of the five training datasets, as rounds 24-25 read them: ten reads per model. Each fit's
  held-out dataset is read zero-shot: musique for L-musique, hotpotqa for L-hotpotqa. No s1eval, test or B1 carve is
  read.
- **Families.**
  - every block the model reads;
  - `chains` on the typed graphs: each entry moves with its row;
  - `links` (zsp, zgn): both ends of each pool edge move. The graph keeps its shape, but its rows are the wrong ones.
- **Permutation.** Within each question, rows are shuffled by a fixed generator, seeded by (seed, the batch's first
  question), for seeds 0 and 1. The family's values follow the shuffle; every other input and the gold labels stay.
  Every model sees the same shuffles on the same carve.
- **Drop** = R@5 unshuffled minus R@5 shuffled, the mean of the two seeds. FC@5 and hit@1 are reported beside.
- **USED** by a model: a drop of at least 0.002 (round 24's MIN_DROP) on at least one of its ten reads. **UNUSED**: a
  drop under 0.002 on all ten.
  - A family any read needs, zero-shot included, is kept.
- `rank` holds rrf, the source of every model's base score. Its drop is reported, and it is never cut.

## Step 2: the sets (from step 1's records alone)

- **S_both**, the controlled set: the families USED by both zrc and zsp.
- **S_mlp**: those USED by zrc. **S_gnn**: those USED by zsp.
- zsp is the GNN because it is the GNN of claims 1 and 2. Every heavier message passing tried so far (zdp, zg1, zgn,
  the six GNN) is no more accurate. zgn's and the six GNN's usage and cost are reported beside; they are not chosen.
- A family in neither set is removed from both models' compile.
- **Chains.** zrc and zsp build their chain entries by different rules: zrc's contribution rule, and rmatch's mass cap.
  If chains are in S_both, both models of the controlled pair read zrc's entries, and zsp on them is one of step 2's
  fits.
- **links** is the message passing itself. It is reported and never enters a set. In the controlled pair, the GNN is
  zsp on S_both plus its links.
- **Training only where a set differs from a model's current inputs.** The fit uses the screens' rules: L-musique and
  L-hotpotqa, seed 0, the model's own config, read on the six s1eval carves. It is compared, question by question, with
  the model's current fit of the split, with the pair and the re-call under the seed null.
  - A set whose fit has **no LOSS** stands.
  - A LOSS restores the lost family to that set, and the next step reports it.
  - Where a set equals a model's current inputs, its current fits stand, and nothing trains.

## Step 3: C8, the cold cost of each

Timed as C6 (laptop, pinned, one thread, cold, the first 200 s1eval questions of each dataset, every check of C1-C6),
each at its fastest exact form, each compiling only its own set's columns:

- **message passing's cost:** zrc on S_both against zsp on S_both;
- **the ceilings:** zrc on S_mlp, zsp on S_gnn;
- the six GNN beside, as filed.

Claim 3 is stated from the controlled pair, with the ceilings beside it. It is stated whatever the ratio turns out to
be.

**Amended 12 October 2026, about 13:25, before any number of F1.** The user asked for the MLP's lead cold and warm.
- C8 also times each form **warm**, as C1-C6 did: the graph index and every per-question input built beforehand, so
  only the forward is timed.
- Warm sits beside cold, labelled. The headline stays cold.
- What is already known, and why the controlled pair matters: C3 filed zsp at 1.08-1.14× zrc's cold cost
  (docs/C3_FASTEST_FORMS.md). The 1.6-3.0× lead is over the six GNN, whose inputs are not the MLP's.

**Amended 12 October 2026, about 13:50, before any number of F1.** zgn's step-1 read is dropped.
- Its select-carve caches (outputs/zfeat/cache) were removed in clean-up 9, after rounds 24-25 filed NO_SELECTION.
  The read failed on the first missing array (`f1-read-zgn`).
- zgn is reported beside and chooses nothing, so rebuilding those caches would spend host hours and disk on a read
  that decides nothing.
- Step 2's sets read zrc's and zsp's records only (`f1-sets2`). C8 can still time zgn beside, from its per-question
  compile.

**Amended 12 October 2026, about 14:00, after step 1's records and before any number of step 2 or C8.** The user,
12 October: two model families, universal and universal-lean; the MLP against the GNN within each family, every model at
its fastest, cold first; everything else stays the same, including the links, the pools and the edges.
- Step 2 trains nothing. zsp keeps its own chain entries (rmatch's mass cap); zsp on zrc's entries is not fitted.
- C8 is C9's lean pair (docs/C9_UNIVERSAL_PAIR.md): zrc against zsp, each on its current inputs, at its fastest exact
  form, cold, with the warm forward beside. zgn is in neither pair and is not timed.
- On the typed graphs, the lean ratio therefore includes each model's own chain rule. Each chain build is timed as its
  own stage and reported, so its share of the ratio is visible.

## What this does not do

- No new column, objective, graph or encoder. The encoder and the substrate embeddings stay frozen.
- One rule for all six datasets; nothing is chosen per dataset. webqsp never trains. Test splits are never read.
- Step 1 writes only its own records under outputs/f1.

## Files

    outputs/mp_unified/f1_usage.py            step 1 and the sets (new; frozen once run)
    outputs/f1/usage-<model>.json, .md        step 1, per model
    outputs/f1/sets.json, sets.md             step 2's sets

    python outputs/mp_unified/f1_usage.py --selftest
    python outputs/mp_unified/f1_usage.py read --model zrc --fits outputs/screen/fits/scr-zrct,outputs/screen/fits/scr-zrct-hp \
        --out outputs/f1/usage-zrc --device cuda --host
    python outputs/mp_unified/f1_usage.py sets --mlp outputs/f1/usage-zrc.json --gnn outputs/f1/usage-zsp.json \
        --out outputs/f1/sets

## Results

**Step 1, filed 12 October 2026, about 14:00.** Records: `outputs/f1/usage-zrc.{json,md}`, `usage-zsp.{json,md}`.
Each model has 2 shuffle seeds × 10 reads on the select carves. Drops are in R@5; `chains` is read on metaqa only (webqsp
never trains).

| family | zrc: largest drop | zrc: mean | zsp: largest drop | zsp: mean |
| --- | ---: | ---: | ---: | ---: |
| rank (never cut) | +0.5376 | +0.3192 | +0.5537 | +0.3145 |
| dense_cos | +0.2732 | +0.1068 | +0.3332 | +0.1128 |
| SEMB | +0.1144 | +0.0365 | +0.1198 | +0.0325 |
| SEED | +0.0667 | +0.0296 | +0.0706 | +0.0296 |
| WALK | +0.0647 | +0.0219 | +0.0473 | +0.0176 |
| DISTS | +0.0350 | +0.0158 | +0.0284 | +0.0133 |
| WALKF | +0.0288 | +0.0086 | +0.0177 | +0.0080 |
| topo_STRUCT | +0.0160 | +0.0056 | +0.0165 | +0.0047 |
| depth_STRUCT | +0.0068 | +0.0027 | +0.0081 | +0.0042 |
| chains (metaqa) | +0.7209 | +0.7086 | +0.7443 | +0.7120 |
| links (message passing, beside) | | | +0.0167 | +0.0068 |

- Every family is USED by both models. The smallest largest-drop is depth_STRUCT's: +0.0068 for zrc and +0.0081 for
  zsp, over the 0.002 bar.
- **Step 2's sets** (`outputs/f1/sets.{json,md}`): S_both = S_mlp = S_gnn = every family both models read today.
  - Nothing is cut.
  - No family is used by one model only.
- So no column is computed for nothing, and neither model's cold cost comes down by computing less. It comes down only
  through faster exact forms of the same stages. With amendment 3, C8 is C9's lean pair.
