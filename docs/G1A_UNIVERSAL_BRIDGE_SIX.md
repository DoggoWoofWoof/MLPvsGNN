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

**Run (9 October 16:11): NOT UNIVERSAL. ABOVE on four datasets, SAME on musique, BELOW on squad.** Records:
`outputs/g1a/report.md`, `outputs/g1a/report.json` and `outputs/g1a/g1a_<dataset>.json`. The table gives R@5 on s1eval
(×100); the gain is U minus R0 with its 95% bootstrap interval.

| dataset | R0 | U | gain [95% CI] | call | zsp (J5) |
| --- | ---: | ---: | --- | --- | ---: |
| metaqa | 0.5 | 11.8 | +11.3 [+10.8, +11.9] | ABOVE | 77.9 |
| squad | 90.5 | 88.3 | -2.2 [-2.5, -1.9] | **BELOW** | 91.0 |
| musique | 47.3 | 47.6 | +0.2 [-0.6, +1.1] | SAME | 56.6 |
| hotpotqa | 68.5 | 80.1 | +11.6 [+11.1, +12.2] | ABOVE | 90.5 |
| 2wiki | 60.8 | 68.4 | +7.6 [+7.1, +8.0] | ABOVE | 87.3 |
| webqsp | 5.3 | 12.2 | +6.9 [+5.7, +8.1] | ABOVE | 26.8 (zero-shot) |

The s1sel carves give the same calls.

- **Not universal as it stands.** squad is BELOW, so the declared outcome is "any BELOW".
- **What breaks it is the question, not the graph.** The rule gives up slots 4 and 5 on every question. Where the
  golds are already in the first stage's top five, that costs:

  | dataset | questions | R0 → U |
  | --- | --- | --- |
  | squad | all (single-hop) | 0.905 → 0.883 |
  | hotpotqa | comparison | 0.897 → 0.845 |
  | 2wiki | comparison | 0.850 → 0.776 |
  | musique | 3-hop | 0.437 → 0.406 |
  | musique | 4-hop | 0.269 → 0.246 |

  Bridge questions gain: hotpotqa bridge 0.631 → 0.790, 2wiki compositional 0.558 → 0.722, metaqa 1-hop
  0.007 → 0.410. A fixed slot rule cannot tell the two kinds apart.
- **The trained models are far above the rule on all six.** zsp's J5 R@5 is above U's by:

  | dataset | zsp − U |
  | --- | ---: |
  | metaqa | +66 |
  | squad | +2.7 |
  | musique | +9.0 |
  | hotpotqa | +10.4 |
  | 2wiki | +18.9 |
  | webqsp | +14.6 |

  The bridge signal is real, but on our datasets the models already take most of it.
- **Families, reported only:**
  - U-ner is the best arm on musique (49.1, +1.8 over R0). Title mentions are sparse there and entity overlap
    carries the chain.
  - U-all is below U wherever U gains: mixing families dilutes the hop, as step 4's union walk did.
- **Next.**
  - **B1b** (docs/B1B_MODELS_ON_HIPPORAG2.md): the models on HippoRAG 2's settings, Claim 2's critical path.
  - **G1b:** the rule's quantities as pre-scoring per-row inputs, so the model decides per question whether a bridge
    row beats the first stage's row. These are the bridge score, membership per family and the source's rank. G1b
    goes in its own file and is graded on all six datasets, zero-shot included.
