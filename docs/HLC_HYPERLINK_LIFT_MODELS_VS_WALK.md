# HLC: the hyperlinks' lift to our trained models against their lift to HippoRAG 2's walk

Declared 4 November 2026 (about 01:50 IST), before any HLC number. Module: `outputs/mp_unified/hlcdiag.py`. It imports
`hlwdiag`, `b1b` and `scripts/b1c_walks.py` unchanged. Records: `outputs/hlc/report.json` and `report.md`.

Nothing trains and nothing is chosen. HLC reads filed read files and reports, and walks B1's graphs with HLW's walk.

## 1. Question

HLW (docs/HLW_HYPERLINK_LIFT_OF_PPR.md, section 6) measured what the corpus hyperlinks add to HippoRAG 2's walk on
HippoRAG 2's three settings. It listed the trained models' lift from the same links beside, deciding nothing. Its next
measure was a paired test of the models' lift against the walk's. This file is that test.

The statement under test is comparative: **on the same questions, adding the links lifts our trained models more than
it lifts HippoRAG 2's walk.** A statement that the walk gets "no lift" is not tested here.

## 2. The measure

For each question i of a setting, in queries.json's order (B1's 1,000 questions per setting):
- the model lift m_i is R@5_i of a model on the graph with the links, minus R@5_i of the same model on the graph without
  them;
- the walk lift w_i is R@5_i of the A1 walk with the links, minus R@5_i of the A1 walk without them;
- D_i = m_i − w_i.

**R@5 per question** is the share of the question's golds in the top 5. The golds are queries.json's.
- For the models it comes from the read file's top count, through `b1b.per_question` (candidate p@swa).
- For the walk it comes from the walk's ranked list, through `b1c_walks.recall`.

**The call per cell** uses the mean of D with B1b's paired interval (`b1b.boot_ci`: 2,000 resamples, seed 0, 95%):
- GREATER when the interval's lower end is above 0;
- LESS when its upper end is below 0;
- SAME otherwise.

**The walk is HLW's**, unchanged:
- restart A1, HippoRAG 2's own rule: half on the dense and SPLADE top-5 seeds, half on their RRF weights;
- B1C's kernel (α = 0.5, 50 iterations);
- ties broken by R0's rank.

"Without the links" is the text graph NK (ner + knn); "with the links" adds the links' family to it.

## 3. Two families

### Family H: the corpus's own hyperlinks

- **Models.** U1a's reads (docs/U1A_B1_WITHOUT_HYPERLINKS.md): B1b's fits, read on b1 (structural = the corpus's
  hyperlinks) and on b1t (structural empty). The arms are zsp (the GNN track's base) and zrc (the MLP's base).
  - The J5 fit decides; its reads are in-domain.
  - Each setting's held fit (L-<setting>) is beside.
- **Walk.** A1 on SNK minus A1 on NK, on B1's graph. This is HLW's ADD.
- **Deciding settings:** 2wiki and hotpotqa. musique is beside, because its structural family is title mentions, not
  hyperlinks.
- **Deciding cells:** 2 settings × 2 arms = 4.

### Family W: the recreated links (W2_S, LLM-free)

- **Models.** B1P's reads (`outputs/mp_unified/zb1p.py`): zg6r (GNN) and zrckp (MLP).
  - The fits ran under WQ-musique and WQ-hotpotqa with seeds 0 to 2. They were read on b1w (structural = U2b's W2_S,
    cut to the setting's nodes) and on b1t.
  - Each question's R@5 is the mean over the in-domain fits: the seeds, and both splits on 2wiki. This is how B1P's
    report takes it.
  - The in-domain reads decide; the zero-shot reads are beside.
- **Walk.** A1 on W2_S + NK minus A1 on NK.
  - B1F's W2_S bench (`outputs/bench/hipporag2_w`) lists every W2_S pair twice, once in each direction. B1's hyperlinks
    are listed once per link.
  - B1C's `family_matrix` adds both directions of each listed row, so the stored file would weight every W2_S pair
    twice.
  - The deciding walk therefore takes each W2_S pair once. The walk on the stored file is beside.
- **Deciding settings:** all three.
- **Deciding cells:** 3 settings × 2 arms = 6.

**The stage call per family:**
- GREATER_EVERYWHERE when every deciding cell is GREATER;
- LESS_SOMEWHERE when any deciding cell is LESS;
- MIXED otherwise.

## 4. Checks

Any failure stops the run, and nothing is written.

- **The walk is HLW's.**
  - R0, A1/NK and A1/SNK on B1's graph equal HLW's filed arms (R@2, R@5, R@10 and FC@5) within 5e-5.
  - The walk's lift equals HLW's filed ADD, interval included.
- **The reads are the filed ones.**
  - Every model read's sha256 equals the one U1a or B1P filed.
  - Its question ids equal queries.json's, in order. U1a's and B1P's reports checked only the gold counts.
- **The model lifts are the filed ones**, within 5e-5:
  - U1a's filed b1t − b1, negated, interval included;
  - B1P's filed paired_w_minus_t, interval included;
  - B1P's mean per variant, and its fit count.
- **The W2_S bench is B1's elsewhere.** It shares B1's ner and knn families, first stage, questions and nodes byte for
  byte, so its NK walk is B1's.

**The self-test** (`python outputs/mp_unified/hlcdiag.py selftest`) passed before this commit. It checks:
- the calls and the stage calls;
- that a file listing every pair in both directions, taken once, gives `family_matrix` on the distinct pairs, and that
  the stored file gives twice that;
- the cell's lifts, difference and levels on made-up arrays;
- that B1b's interval negates exactly under negation, which the U1a check relies on;
- that a read in another question order is refused.

## 5. Also recorded (deciding nothing)

- **The levels:** R0, the walk with and without the links, and the model with and without them.
- **Per cell, two paired gaps:**
  - the model with the links minus the walk with the links;
  - the model without the links minus the walk with them.
- **Family W against the walk on the stored file.**

## 6. What a sentence may say (fixed now)

- **GREATER_EVERYWHERE in a family.** The sentence may say: on HippoRAG 2's settings, adding the [corpus hyperlinks /
  recreated links] lifts our trained models more than it lifts HippoRAG 2's walk on our graph, question for question.
  It gives the cells' differences and intervals.
- **MIXED.** Name the cells where the difference is GREATER and the cells where it is SAME. No general sentence is
  written.
- **LESS_SOMEWHERE.** Report the cells. The comparative sentence is not written.

In every case the walk is HippoRAG 2's walk on our LLM-free graph from its own restart rule, not HippoRAG 2 as released
with its LLM-built graph. The reads are on HippoRAG 2's 1,000-question settings.

## 7. What follows

- The models of record (zg6r and zrckpn) have not been read on the corpus-hyperlink carve b1. A zb1p amendment adds
  those reads. Family H is then read again with them, in a dated section of this file.
- HippoRAG 2's walk at full scale is HLW's part B (P3a's orders). It is not paired here.
- The verdict closes only this measure.

## 8. Running

```
python outputs/mp_unified/hlcdiag.py selftest
python outputs/mp_unified/hlcdiag.py run          (laptop, minutes)
```
