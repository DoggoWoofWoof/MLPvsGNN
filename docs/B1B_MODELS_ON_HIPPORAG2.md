# B1b: our trained models on HippoRAG 2's three settings

Declared 9 October 2026, before any number. This stage follows from the user's rulings of the same day:
- Claim 2's headline is HippoRAG 2's published R@5;
- every fix is universal;
- "we have better features and models is our claim".

## Why

B1d's training-free bridge rule (`s2m3/structural/Q`, one arm for all three settings) gives:

| setting | B1d universal R@5 | call against HippoRAG 2 |
| --- | ---: | --- |
| musique | 57.0 | BELOW (74.7) |
| 2wiki | 92.8 | ABOVE (90.4) |
| hotpotqa | 95.2 | BELOW (96.3) |

On our own six datasets the trained models far exceed any training-free rule. These are R@5 figures on s1eval
(outputs/full_zsp/grade.json, J5 fit):

| dataset | RRF | zrm | zsp |
| --- | ---: | ---: | ---: |
| musique | 0.473 | 0.573 | 0.566 |
| hotpotqa | 0.685 | 0.903 | 0.905 |
| 2wiki | 0.608 | 0.868 | 0.873 |

Next to that, G1a's rule adds +0.002 on musique and +0.076 on 2wiki. So the models on HippoRAG 2's settings are
Claim 2's critical path. This stage reads them there, with no training and nothing chosen per setting.

## The held-out check

- HippoRAG 2's questions are rows of each dataset's dev split (validation on hotpotqa), mapped by B1a.
- M3B's eval population is the same split. Every fit trains only on train-split carves (s1sel, s1fit, fit).
- No B1 question has entered any fit, any selection or any screen.
- B1a found the package's golds equal to the setting's on every question of all three settings (`gold_agrees_with_package`
  1.0), so the package's golds are the benchmark's.

## What runs

The same pipeline and the same fits for all three settings.

1. **The look** (`outputs/mp_unified/look_b1.py`, new): look_x_six's pass, unchanged except for three things that come
   from the setting, as B1a built it (scripts/b1_build.py, outputs/bench/hipporag2/<dataset>):
   - **questions:** the carve `b1`, the setting's 1,000 questions as rows of M3B's eval population, in the setting's
     order;
   - **first stage:** each question's dense and SPLADE top-1000 over the setting's distinct passages, each score the
     inner product of the frozen vectors. Ties go by the passage's first position in the setting, as in B1a's lists.
     B1a's list, mapped to package rows with repeats dropped, must be a prefix of this list. On 2wiki and hotpotqa it
     equals it. MuSiQue's corpus repeats two passages (no gold among them), and each pair shares one package row,
     counted once.

     *Amended 16:55, before any number:* the host's mirror carries no SPLADE document vectors, and the first looks
     stopped there. So the lists are computed on the laptop from the package (`look_b1.py lists`), each checked
     against B1a's as above, and saved per setting (outputs/bench/hipporag2/<dataset>/b1_lists.npz with a sha256
     record). The host's look reads them, and its own dense lists must equal them up to near-ties (scores within 1e-5
     at each rank where the rows differ). The count of such lists is reported. Near-tie swaps against B1a's lists:
     dense 39 / 31 / 33 lists (2wiki / hotpotqa / musique), SPLADE none;
   - **graph:** the context's three family stores replaced by the setting's graph, as package rows. structural is the
     package's family induced on the setting's rows; ner and knn are rebuilt on the setting by the package's rules.
     A repeated passage's edges land on its one row: self-loops are dropped, and a repeated edge is kept once with its
     largest weight.

   Everything else is look_x_six's: the frozen construction, the pool rules (base, seeds, hops over the stores), the
   129 compiled columns, the six pair's scores, and --full's edges and projections. A pool row outside the setting's
   passages stops the look.
2. **The feature cache:** lean_cache.py on carve `b1`, as on every carve.
3. **zsp's edges:** zlink.py build on carve `b1`, as on every carve.
4. **The reads.**
   - **Arms and fits:**

     | arm | role | J5 fit | L-2wiki fit | L-musique fit | L-hotpotqa fit |
     | --- | --- | --- | --- | --- | --- |
     | zrc | the MLP's base | outputs/full_zrct/fits/J5 | outputs/full_zrct/fits/L-2wiki | outputs/screen/fits/scr-zrct | outputs/screen/fits/scr-zrct-hp |
     | zsp | the GNN track's base | outputs/full_zsp/fits/J5 | outputs/full_zsp/fits/L-2wiki | outputs/screen/fits/scr-zsp | outputs/screen/fits/scr-zsp-hp |

     These are the fits their grades read.
   - **Where they read.** Each fit is copied to outputs/b1b/fits/<arm>/<split>, leaving out its reads, and the copy's
     models.pt is checked against the source's sha256. The copy is read with `--read musique=b1,2wiki=b1,hotpotqa=b1`.
     Nothing is written into a fit's own folder.
   - **Candidate:** `p@swa`, the one every grade reads. zrc reads on the card and zsp on the CPU, as their grades did.

## The calls

- **Metric.** R@5, the share of a question's golds among its five highest-scored rows, averaged over the 1,000
  questions. The reads keep per question the golds in the top five, so R@5, FC@5 and hit@1 are read; R@2 is not.
- **Claim 2 (the GNN, primary).** zsp's J5 fit, per setting: ABOVE, AT or BELOW HippoRAG 2's published R@5. The call
  uses the 95% bootstrap interval of our R@5 (2,000 resamples of the questions, seed 0):
  - ABOVE when the interval's bottom is above the published number;
  - BELOW when its top is below it;
  - AT otherwise.

  This is docs/PAPER_CLAIMS_2026_10_09.md's rule. J5 is trained on the five datasets' train questions and read on dev
  questions it never saw, the protocol of the trained graph retrievers we compare with (GFM-RAG trains on the same
  three datasets' train splits).
- **The MLP.** The same call for zrc's J5 fit. The share it keeps of the GNN's gain is reported:
  (zrc − RRF) / (zsp − RRF) on R@5.
- **Zero-shot (reported).** Each setting read by the fit that held its dataset out (L-musique, L-2wiki, L-hotpotqa),
  both arms, with the same call.
- **Also reported:**
  - against NV-Embed-v2 dense (69.7 / 76.5 / 94.5), our RRF (the look's own `rrf` reference) and B1d's universal arm;
  - **E1:** our lift over our first stage, against HippoRAG 2's lift over NV-Embed-v2 (+5.0 / +13.9 / +1.8);
  - **the pool's ceiling:** the share of golds in each question's pool. This splits a BELOW into what the pool never
    holds and what the ranking misses.
- **The hyperlink caveat stays.** On 2wiki and hotpotqa the structural family is the documents' own hyperlinks. They
  are question-independent and LLM-free, but HippoRAG 2 does not use them. Reported beside every call there.

## What it decides

- **ABOVE or AT on a setting:** Claim 2 holds there for zsp, with the MLP's share beside it.
- **BELOW:** the pool's ceiling and the ranking's miss name the gap. A pool gap goes to the pool track (step 4g/4h); a
  ranking gap goes to G1b and the universal graph, each declared in its own file and graded on all six datasets.
- **Nothing here is tuned on B1.** No arm, fit, rule or threshold is chosen on these questions. A later stage that
  changes the models reads B1 again only through a declared re-run of this stage, under the same calls.

## Running it

    python outputs/mp_unified/look_b1.py --dataset 2wiki --host --full            -> outputs/mp_unified/look/2wiki/b1
    python outputs/mp_unified/lean_cache.py --dataset 2wiki --carve b1 --part 0/1 --host
    python outputs/mp_unified/zlink.py build --dataset 2wiki --carve b1 --host
    python outputs/mp_unified/b1b.py fork --arm zsp --split J5 --src outputs/full_zsp/fits/J5
    python outputs/mp_unified/zprop.py read --name J5 --out-root outputs/b1b/fits/zsp --read musique=b1,2wiki=b1,hotpotqa=b1 --device cpu --threads 6 --host
    python outputs/mp_unified/b1b.py report                                         -> outputs/b1b/report.json, report.md

## Results

(filed after the run)
