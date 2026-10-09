# G1a: does B1d's one universal bridge rule carry to our six datasets, untuned?

Declared 9 October 2026, before any number. This stage follows from the user's rulings of the same day:
- "every fix universal";
- "better graph representation and better pools while trying to improve ranking";
- "keep getting results, identifying gaps and fixing, while staying on our claims".

## Why

B1d chose one rule for all three of HippoRAG 2's settings, on their even questions. The rule is `s2m3/structural/Q`:
- keep the first stage's top 3;
- fill slots 4 and 5 with the best document-link neighbours of its top 2, by cosine with the question.

It was ABOVE R0 on 2wiki (+21.2) and hotpotqa (+10.0), and SAME on musique (+1.1).

A universal rule must also hold on our own six datasets: the open corpora (5–6 million passages on 2wiki and
hotpotqa), the KB graphs (metaqa, webqsp) and squad. This stage applies the rule exactly as chosen. Nothing about it
is chosen on these datasets.

## The base stays fixed

Unchanged:
- the frozen encoder and the package's vectors;
- the M3B graph stores (every family symmetrised, with per-family capped neighbourhoods, as the walks read them);
- step 1's carves;
- the models.

Nothing is trained or re-encoded. The first stage is each question's dense and SPLADE top-1000 lists from M3B's
prepare. They are fused by equal RRF with the package's constant (`retrieval_pools.equal_rrf.constant`), ties going
by node position.

## Arms

| arm | what it is | role |
| --- | --- | --- |
| R0 | the RRF list | the base |
| U | B1d's rule on the `structural` family, the dataset's own links: hyperlinks on 2wiki and hotpotqa, title mentions on musique, KB triples on metaqa and webqsp | the call |
| U-ner, U-knn, U-all | the same rule on another family, or on all three | reported only, to inform the universal graph's design; never chosen |

Where a family has no store, the arm equals R0 and is marked `no_family`. squad is expected to have no structural
store.

## Carves and the call

- **Carves.** Every dataset's s1eval carve, and s1sel where it exists.
- **The call.** U's R@5 gain over R0 on s1eval, with a 95% bootstrap interval (2,000 resamples, seed 0), is ABOVE,
  SAME or BELOW.
- **Also filed:**
  - R@2, R@5 and R@10 of every arm, and full_coverage@5;
  - the gain by question kind (hop count, or question type).

## What it decides

- **U ABOVE or SAME on all six,** with no BELOW: the rule is universal at the first stage. G1b makes its quantities
  per-row inputs for both models. These are:
  - the bridge score, meaning the best cosine among the top 2's neighbours;
  - whether the row is such a neighbour, per family;
  - the rank of its source.

  The quantities are computed from the first stage, before any row is scored, so the MLP may read them. G1b is
  declared in its own file and graded on all six datasets, zero-shot included.
- **Any BELOW:** the rule is not universal as it stands. The BELOW datasets name the graph property that breaks it,
  for example a KB's hub degree or squad's lack of links. That property goes into the universal graph's design,
  which is the next declared stage.

## Running it

    python scripts/g1a_universal_bridge.py run [--datasets ...] [--threads 4]   -> outputs/g1a/g1a_<dataset>.json
    python scripts/g1a_universal_bridge.py report                               -> outputs/g1a/report.json, report.md
    python scripts/g1a_universal_bridge.py --selftest

## Results

(filed after the run)
