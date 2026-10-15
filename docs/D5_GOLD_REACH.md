# D5: where the golds today's pools miss sit in the graph

Declared 15 October 2026. Code: `scripts/d5_gold_reach.py`. A diagnostic: it reads golds and decides nothing. No
model trains or reads. Status: **RUN** (section 5); D5b declared (section 6).

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

Filed 15 October about 19:22 (`outputs/d5/reach_*.json`, `outputs/d5/table.md`). It ran on the laptop: each carve took
10-131 seconds after its graph loaded. Every check held.

**Where the golds sit.** "Within k" is the share of questions whose golds are all within k links of B:

| carve | ALL today | within 0 / 1 / 2 links | golds today misses: 1 / 2 / 3+ links | N1 p50 / p90 |
| --- | ---: | --- | --- | --- |
| metaqa s1sel | 0.952 | 0.019 / 0.400 / 0.839 | 31 / 203 / 668 of 902 | 526 / 695 |
| squad s1sel | 0.979 | 0.979 / 0.992 / 0.999 | 19 / 10 / 2 of 31 | 1,734 / 2,759 |
| musique s1sel | 0.885 | 0.502 / 0.884 / 0.995 | 101 / 86 / 5 of 192 | 6,456 / 12,432 |
| hotpotqa s1sel | 0.956 | 0.663 / 0.989 / 0.997 | 55 / 15 / 5 of 75 | 887 / 1,584 |
| 2wiki s1sel | 0.953 | 0.348 / 0.953 / 0.978 | 25 / 30 / 20 of 75 | 1,400 / 1,856 |
| webqsp s1eval (beside) | 0.841 | 0.279 / 0.909 / 0.978 | 1,351 / 1,629 / 203 of 3,183 | 1,996 / 6,401 |

(3+ also counts a gold whose question's two-link neighbourhood was over the budget; that happens on webqsp.)

**What it says, dataset by dataset:**
- **musique:** reach is not the problem. 99.5% of questions have every gold within two links of B. The missed golds
  are mostly one link out (101) or two (86). But B's one-link neighbourhood is large (6,456 nodes at the median), and a
  missed gold ranks about 2,350th in it by first-stage weight (p50; 5,389 at p90). Today's KIND_T fill spends about
  1,900 rows and still misses them.
- **metaqa:** the misses are the 3-hop answers, 3+ links out (668 of 902). B ∪ N1 can never pass 0.400 there.
  KIND_T's walk already reaches them (the 7,000 ceiling is 0.990); what costs is the 2,022 rows it takes.
- **hotpotqa and 2wiki:** already at 0.95. N1 by weight finds their missed one-link golds early (p50 117 and 127).
- **webqsp (beside):** like musique, nearly all within two links (0.978) but deep in N1 (p50 1,444).

**The two parameter-free N1 orders**, B plus N1's first k (ALL / mean rows):

| carve | by weight, k = 100 | 300 | 1,000 | 3,000 | today ∪ the first k by weight, k = 50 | 300 | 1,000 | 3,000 |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| metaqa | 0.354 / 150 | 0.377 / 350 | 0.400 / 588 | 0.400 / 590 | 0.953 | 0.954 | 0.954 | 0.954 |
| squad | 0.983 / 150 | 0.985 / 350 | 0.989 / 1,025 | 0.992 / 1,829 | 0.982 | 0.985 | 0.989 | 0.992 |
| musique | 0.532 / 300 | 0.588 / 500 | 0.734 / 1,200 | 0.826 / 3,194 | 0.885 | 0.887 | 0.894 | 0.923 |
| hotpotqa | 0.926 / 150 | 0.979 / 349 | 0.988 / 876 | 0.989 / 1,053 | 0.968 | 0.985 | 0.990 | 0.990 |
| 2wiki | 0.763 / 150 | 0.906 / 350 | 0.947 / 1,019 | 0.953 / 1,451 | 0.956 | 0.965 | 0.968 | 0.969 |
| webqsp | 0.610 / 301 | 0.769 / 500 | 0.856 / 1,190 | 0.888 / 2,294 | 0.865 | 0.884 | 0.904 | 0.918 |

By count (how many B-nodes link to it) is worse than by weight on every dataset (`outputs/d5/table.md`). The weight
order lifts hotpotqa and 2wiki cheaply (today's 94 and 105 rows plus 50 more give 0.968 and 0.956). It does not fix
musique: even 3,000 more rows give 0.923.

## 6. D5b: ordering the neighbours by what they say

Declared 15 October, after D5's numbers and before D5b's. Code: `scripts/d5b_neighbour_orders.py`. Also a diagnostic:
it decides nothing. Same carves and checks as D5.

**Why.** On 150 s1sel musique questions (a development run, section 4), **the golds today misses are in neither
first-stage list**: 0 of them are in the dense or splade top 1,000, while 98% of the golds today holds are in the dense
list. They are bridge passages that share little with the question's words. So the question alone cannot rank them.
What can is the node that links to them, together with the part of the question that node does not already answer.

**Three more N1 orders** (ê = the frozen encoder's dense embeddings at unit length, the ones the dense list is ranked
by; ties by node id):
- **cos:** by ê_q · ê_v.
- **rrf:** RRF of the cos order and D5's weight order, with the stage's constant c.
- **res:** by the best parent's residual, the largest over v's B-parents p of cos(q − (q·p̂)p̂, v): the question with
  what p already says taken out. This is one parameter-free offset.

**Per-parent beams.** Take B's first P nodes by first-stage weight. From each, keep its first m neighbours outside B,
by cos or by res with that node as the parent. That is **beam1**. **beam2** adds, from each beam1 node, its first m
neighbours outside B by the same score, with the beam1 node as the parent. The grid:
- P ∈ {10, 20, 50, 100, 200} and m ∈ {1, 2, 3, 5, 10};
- beam2 only for P ≤ 50 and m ≤ 5, where its cost stays small.

Per question: whether B ∪ beam holds every gold, and the rows |B ∪ beam|. Beside: today ∪ beam (rows not counted).

**Per gold, also:**
- its rank in N1 under each order;
- for a gold two links out, the best rank of an N1 node linking to it (its bridge);
- its own rank in the dense and splade lists (0 if not in the 1,000).

**Checks.** D5b recomputes D5's measures with D5's own function on the host (D5's arrays are not there). The run stops
if its N1 and two orders differ from that function's. After the fetch, `check` compares D5b's hops, today flags, N1
sizes and both ranks with D5's filed arrays on the laptop. They must be equal. The selftest covers:
- the three orders on a hand-built 3-d embedding;
- the bridge ranks;
- one- and two-step beams with their rows;
- the summary.

**What it is for.** P1g, declared on its own, will choose one universal order or beam plus a size rule on s1sel. Its
target is the user's: ALL ≥ 0.95 on every dataset (or each dataset's ceiling where that is lower), then the fewest
rows. D5b says which family is worth declaring.

**Before this file.** Development runs of the beams on the first 150 s1sel musique questions ran while the code was
written. Nothing is chosen from them.

**Launch fix (15 October, about 19:50).** The first six host items stopped in two seconds, at the package import:
the script had no `--host` switch to P1e's mirror mode (the one P1f's stats and P1c3's features use). The switch
was added (`run DATASET --host` calls `p1e_anchor_kind.G.on_the_mirror()` first) and the items re-queued as
`d5b-*-r2`. Nothing measured changed; the selftest passes.

### D5b results

Filed 15 October about 21:10 (`outputs/d5/orders_*.json`). It ran on the host, one item per dataset (6 to 76
minutes). On every dataset, `check` found D5b's hops, today flags, N1 sizes and both ranks equal to D5's filed arrays.

**Today ∪ N1's first k** under each order (ALL; s1sel, webqsp's s1eval beside):

| carve | today | weight, k = 50 / 1,000 / 7,000 | cos, k = 50 / 1,000 / 7,000 | res, k = 50 / 1,000 / 7,000 |
| --- | ---: | --- | --- | --- |
| metaqa | 0.952 | 0.953 / 0.954 / 0.954 | 0.952 / 0.954 / 0.954 | 0.953 / 0.954 / 0.954 |
| squad | 0.979 | 0.982 / 0.989 / 0.992 | 0.989 / 0.992 / 0.992 | 0.983 / 0.991 / 0.992 |
| musique | 0.885 | 0.885 / 0.894 / 0.940 | 0.885 / 0.913 / 0.945 | 0.885 / 0.913 / 0.943 |
| hotpotqa | 0.956 | 0.968 / 0.990 / 0.991 | 0.960 / 0.991 / 0.991 | 0.967 / 0.989 / 0.991 |
| 2wiki | 0.953 | 0.956 / 0.968 / 0.969 | 0.953 / 0.965 / 0.969 | 0.954 / 0.963 / 0.969 |
| webqsp (beside) | 0.841 | 0.865 / 0.903 / 0.928 | 0.841 / 0.868 / 0.928 | 0.854 / 0.901 / 0.928 |

**The best beam** (today ∪ the beam; the beam alone with B, ALL / rows, in brackets):

| carve | best beam | today ∪ beam | B ∪ beam |
| --- | --- | ---: | --- |
| metaqa | res, P 50, m 5, two steps | 0.957 | 0.548 / 1,155 |
| squad | res, P 50, m 5, two steps | 0.993 | 0.993 / 409 |
| musique | res, P 200, m 10, one step | 0.906 | 0.748 / 1,260 |
| hotpotqa | res, P 50, m 5, two steps | 0.981 | 0.931 / 855 |
| 2wiki | cos, P 50, m 10, one step | 0.962 | 0.774 / 453 |
| webqsp (beside) | res, P 50, m 5, two steps | 0.883 | 0.664 / 742 |

**What it says:**
- **The question's own words do not find musique's missed golds early.** cos and res beat weight at 1,000 (0.913
  against 0.894), but all three need thousands of rows to pass 0.94. No beam lifts today's 0.885 above 0.906. The
  golds today misses are rarely in either first-stage list (dense 3.7% and splade 5.7% within the top 1,000).
- **On squad, cos is the better order** (0.989 at 50 more rows against weight's 0.982). Its missed golds are in the
  dense list (87%).
- **hotpotqa and 2wiki are already over 0.95.** Weight stays the cheapest order there. 2wiki's one-link ceiling is
  0.969, so its other misses are two or more links out.
- **metaqa's misses are three links out.** No N1 order moves it (0.954 at most), and two-step beams add 0.005.
- **For P1g:** musique decides. Only KIND_T's longer fill (its 7,000 ceiling is 0.950) or the largest N1 cuts can
  reach 0.95 there, and either is applied to every dataset.
