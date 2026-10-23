# U2j: link precision (step 1 of the hyperlink-gap plan)

Declared 23 October 2026, before any number. It follows [U2I_CORPUS_ALIASES.md](U2I_CORPUS_ALIASES.md), sections 7
and 8, and the user's instruction of 23 October: "yeah do the 3 steps to fix". Code: `outputs/mp_unified/zu2j.py`.
Outputs: only under `outputs/u2j/`. Nothing trains, and no filed graph, pool, fit or record changes.

## 1. The question

A hyperlink is a link an editor chose. An LLM-free mention link is any place a title occurs, so W2_S's rows hold far
more neighbours than the hyperlinks' rows (mean degree 12.3 on hotpotqa and 23.9 on 2wiki, with hubs of 372k and
621k). Two of the losses U2g and U2a measured come from that:
- **CAP_CUT:** the frozen I takes each seed's first 25 neighbours in node-id order, so a gold linked to a seed can sit
  past the cap. That is 0.31 of 2wiki's lost golds (U2a). The frozen I's ALL on hotpotqa falls from 0.968 with the
  hyperlinks to 0.888 with W2_S (U2g, s1eval).
- **The bridge:** the second gold drops out of the top five (U2g, section 10). A link that is precise should rank
  that gold near the first gold.

Step 1 asks whether rule-based evidence about each link puts the links an editor would choose first. The evidence is
written by U2i's pass. No question, gold or hyperlink enters any order; the hyperlinks enter only the beside-measures
of section 5.

## 2. The evidence of a link

For a mention link u → v (u's text names v's title, or one of v's aliases once U2i adopts it), from U2i's records:

| feature | what |
| --- | --- |
| lead | the first mention is in u's first sentence (U2i's sentence index 0) |
| pos | the first mention's character offset over u's text length |
| count | the mention's occurrences in u, as log(1 + n) |
| rare | −log(1 + v's mention links in): rarer titles first |
| caps | the surface's capitalization statistic, cap / (cap + lower + 1), over the corpus |
| capmid | the surface written exactly as titled and capitalized, mid-sentence, somewhere in u |
| tokens | the surface's token count |
| alias | 1 for a kind-4 surface (U2i), 0 for a title |

A W2_S row of u holds v when u names v (*out*), when v names u (*in*), or when they share a title (*sibling*). A pair
can be several of these at once.

## 3. The orders (parameter-free)

Each candidate reorders every W2_S row. The graph's links stay the same, so B ∪ X keeps its reach; what changes is
which neighbours fall under a cap. Ties always go to node id.

| order | key, first to last |
| --- | --- |
| R0 | node id (today's: m3b_pools.FamilyStore's order) |
| R1 | the pair's class: mutual (out and in), then out, then in, then sibling only |
| R2 | R1's class, then lead (for an out-link, u's mention of v is in u's first sentence; for an in-link, v's mention of u is in v's; a mutual pair is lead if either is) |
| R3 | rare: v's mention links in, fewest first |
| R4 | R1's class, then lead, then rare |

Declared order: R1, R2, R3, R4.

## 4. The grading

Per passage dataset, on s1sel, over W2_S (U2i's gate 1 must hold for every dataset graded: its rebuild equals the
filed W2_S array for array). Each link's evidence is read from U2i's chunk files, and the W2 links rebuilt from those
hits must equal U2i's `links.npz` exactly, else REFUSED. Every neighbour in a reordered row must be a mention link or a
sibling, else REFUSED.
- **The rows reordered:** every row on musique (two hops: the frontier's rows are read too); on hotpotqa, 2wiki and squad
  (one hop or none), the rows of the seeds and of the golds in B, which are all the rows the I and the bridge read.
  R0 must reproduce the filed rows, else REFUSED.
- **The frozen I** over the reordered graph (m3b_pools.expand_hops with the construction, unchanged; zu2i's loop). Its
  ALL, recall and mean pool. squad's construction reads no links (RETRIEVAL), so squad has no I and is reported for the
  bridge only.
- **The bridge at r**, for r = 5 and 25: over the (question, gold) pairs whose gold is outside B, the share whose gold
  is within the first r of some gold-in-B's row, in the graph's row order. PKG's rows are in node-id order, as the
  frozen construction reads them.
- **The gap share** on every dataset: `(m(order) − m(R0)) / (m(PKG) − m(R0))` for I's ALL and the bridge at 25, shown
  when the denominator is at least 0.005 (it matters on hotpotqa and 2wiki).

## 5. Beside, deciding nothing

Each feature's AUC against the dataset's own hyperlinks, on every passage dataset (`beside`). The label of a directed
W2 mention link is whether its pair is in PKG. For the yes/no features, the hyperlinked share when 1 and when 0 is
reported too. This is the measure step 2 (U2k) learns from; here it only describes.

## 6. The choice

- **Score:** the mean I's ALL over musique, hotpotqa and 2wiki, on s1sel.
- **Best:** the order with the highest score; among orders within 0.002 of it, the first in the declared order.
- **ADOPT_FOR_POOLS** when the best's score is at least R0's plus 0.002, and no dataset's I's ALL falls below R0's by
  more than 0.002.
- **NO_GAIN** otherwise. It closes only these four orders as the I's row order.
- **REFUSED** when U2i's gate 1 fails on a dataset graded, or a grade is missing.

## 7. What follows (fixed now)

- **Either way:** the eight features go into both models in their own file, once GPU credit allows. The idea reaches
  both models before it closes (SCREENS 3b). The GNN reads them on its edges; the MLP reads, for each row,
  summaries over its links to the question's first-stage rows, computed before any row is scored.
- **ADOPT_FOR_POOLS:** the chosen order also becomes the row order of the I inside the pools, rebuilt on the W2_S
  build, in its own file.
- **Always:** step 2 (U2k) learns one score from these features on 2wiki's hyperlinks. If the score helps, it orders
  the rows in place of any order here.

## 8. How it runs (laptop)

    python outputs/mp_unified/zu2j.py --selftest
    python outputs/mp_unified/zu2j.py grade --dataset D        # after zu2i.py pass --dataset D
    python outputs/mp_unified/zu2j.py beside --dataset D
    python outputs/mp_unified/zu2j.py choose
    python outputs/mp_unified/zu2j.py report

## 9. Results

Filed after the run: `outputs/u2j/<D>/grade_s1sel.json`, `outputs/u2j/<D>/beside.json`, `outputs/u2j/choice.json`,
`outputs/u2j/report.md`.
