# D5: where the golds today's pools miss sit in the graph

Declared 15 October 2026. Code: `scripts/d5_gold_reach.py`. A diagnostic: it reads golds and decides nothing. No
model trains or reads. Status: **declared**.

## 1. Why

The user's target (15 October): **ALL ≥ 0.95 on every dataset, coverage first, then the fewest rows.** P1f filed
NONE_MEETS: no size rule over KIND_T's order stays within today's rows and every dataset's floor. Its numbers also say
size alone cannot reach 0.95 on two datasets:
- **musique:** KIND_T's 7,000 nodes hold every gold of 0.950 of s1sel's questions (s1eval 0.941). Today's pools give
  0.885.
- **webqsp:** 0.889 at 7,000 on s1eval (beside only), 0.841 today.
- Across all 23 filed orders, musique's per-question best reaches 0.95 only at about 3,900 nodes, with a ceiling of
  0.972. Mixing today's orders will not do it.

So the missing golds need **reach**, not size. Before a rule is designed (P1g), this file measures where those golds
sit relative to each question's base and seeds.

## 2. What it measures

Per question, with **B** = its base and seeds (step 4d's, as every pool stage opens them) and the union of the three
link families (structural, ner, knn) as the walks see it:
- **N1:** B's one-link neighbours outside B. **N2:** the nodes two links from B (expanded only when a gold is in
  neither B nor N1, and only when N1's links total at most 5,000,000 entries).

Per gold:
- **hop:** 0 in B; 1 in N1; 2 in N2; 3 = further, or N2 not expanded.
- **fam1** (for a gold in N1): which families carry a link from B to it.
- **cnt** (for a gold in N1): how many nodes of B link to it.
- **Its rank in N1** under two parameter-free orders, ties by node id:
  - **by count:** the number of B-nodes linking to it;
  - **by weight:** the sum of those B-nodes' first-stage weights, w = 1/(c + dense rank) + 1/(c + splade rank). This is
    P1e's steer, read through `p1e_anchor_kind.steer` unchanged.
- **today:** whether today's pool (P1c2's: KIND_T at |I_q|) holds it, from P1f's filed depths.

Summaries per carve:
- golds by KIND_T depth bucket and hop;
- today's missed golds by hop, family and rank;
- ALL if every gold within 0, 1 or 2 links;
- **ALL and mean rows of B ∪ N1's first k**, in each order, k from 50 to 7,000;
- ALL of today's pool ∪ N1's first k by weight (rows not counted; an upper bound on what a union would cost).

## 3. Scope and checks

- **Carves:** s1sel of the five training datasets. webqsp's s1eval beside only, since webqsp golds never design a rule.
  s1eval is not read for the training datasets.
- **Checks** (the stage stops on any failure):
  - P1f's gold pointers equal the carve's golds;
  - |B| equals P1f's filed `bs`;
  - B ∪ N1's first k, taken from the orders themselves at k = 100, 300, 1,000 and 3,000, gives the coverage the ranks
    give.
- **Selftest:** a toy graph with hand-computed hops, families, counts, both orders and N2 sizes; the steer read at B;
  the summary on two hand-built questions.

## 4. Before this file

Development runs on the first 150 s1sel questions of musique and metaqa ran before this commit, to check the APIs and
the speed (about 4 seconds per 150 questions after the graph loads). They decide nothing, and P1g will be declared on
its own and chosen on s1sel. What they showed:
- metaqa: every one of those golds is within one link of B.
- musique: most misses are one link out, but deep in N1 (N1 holds about 5,700 nodes at the median).

## 5. Results

(filled from `outputs/d5/` when the run lands)
