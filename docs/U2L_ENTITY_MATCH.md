# U2l: one entity-link family (EL) on all six datasets, graded as U2b graded graphs

Declared 25 October 2026, before any number. It is stage U2l of [PROGRAM_2026_10_24.md](PROGRAM_2026_10_24.md)
(sections 2 and 3), the user's ruling of 24 October:

> i think ner and links do the same thing which is entity match and can be unified into a better ner itself, and cant
> we use this on the KB too if it needs to be a universal rule

Code: `outputs/mp_unified/zu2l_el.py` (new). Outputs: only under `outputs/u2l/el/`. Nothing trains. No filed graph,
pool, fit or record changes: every graph here is new, beside today's. The U2l diagnostics (`zu2l.py`, deciding nothing)
stay as filed.

## 1. Why

Two of today's three package families are entity matches (the program, section 1):
- a W2_S link (docs/U2B_LINK_RULE.md) means passage u's text holds node v's title: v *is* the entity;
- an NER edge (CRAG's `ner_edges`) means u and v both hold a spaCy span of one entity, with corpus df in [2, 25],
  weighted by the sum of 1/df.

They are kept apart today, with different caps, and the KBs get neither from their names: U2b's descriptive switch
gives metaqa and webqsp no mention links. EL is one family with two edge kinds, built by one rule on all six datasets.
On musique, NER edges reach 152 golds that no link reaches (the U2l diagnostics), so the co-mention kind is kept.

## 2. Entities and mentions

**An entity** is a title group: U2b's group, the title with underscores read as spaces, stripped. On a KB a node's
name is its title, so a group is the nodes of one name.

**A node mentions a group** when U2b's scan finds one of the group's surfaces in the node's text under W2's gates,
unchanged: the look boundary, the case gate, the capitalization statistic over the whole corpus, the AMBIG guard (50)
and the context choice (the frozen dense vectors). The surfaces are:
- the title (kind 0, W2's);
- U2i's aliases (kind 4), for exactly the cue set U2i's filed choice names: none if U2i chooses W2_S, else the
  candidate it chooses (`outputs/u2i/choice.json`). U2l reads that file and refuses to build before it exists.

**The texts.** On a passage dataset, the descriptive passages are read, as U2b and U2i read them. On a KB, each node's
name is read as its text: U2b's descriptive switch is off for EL, and only for EL. The scan matches every run of a
name's words, so a name that holds another name ("Harry Potter and the Goblet of Fire" holds "Harry Potter") mentions
it. A node's match to its own group links nothing. webqsp's CVT mediators (`V1_NODE_KIND_FROZEN.json`; 1,588,085 of
2,592,894 nodes) neither mention nor own: their text is written from their neighbours' names, so their links would only
restate the R family.

**spaCy's entities** are CRAG's: spaCy 3.8.15 with en_core_web_sm 3.8.0, the nine labels, keys lowercased and stripped,
more than 2 characters. The package stores their co-mention edges on all six datasets, but their keys only where
`linkwords.py ents` built them (musique and squad). So spaCy's entities enter EL through the package's edges, on every
dataset alike, and a spaCy key is not merged with a title group anywhere (merging them is named in section 8).

## 3. The edges

| kind | between | kept | weight |
| --- | --- | --- | --- |
| **IS** | u mentions group E, v is in E (v ≠ u) | at any df, as links are today | 1 |
| **CO_t** | u and w both mention group E | df_t(E) in [2, 25] | Σ over shared E of 1/df_t(E) |
| **CO_s** | the package's NER edges, as stored | df in [2, 25] (CRAG's) | the package's |

- df_t(E) is the number of distinct nodes that mention E. It is NER's cap, applied to the titles.
- On a passage dataset, IS's title part is exactly U2i's W2 links (`outputs/u2i/<D>/links.npz`, `w2_src`, `w2_dst`),
  and its alias part is U2i's chosen cue's links. U2i built both by U2b's scan; U2l reads them and rebuilds nothing.
  CO_t is computed from the same directed IS links, grouped by the target's group (`evidence.npz`, `group`).
- On a KB, IS and CO_t come from the scan of section 2: zu2b's `stat_pass` and `link_pass`, unchanged, with variant
  W2 under `Rule("look", True, True, False)`. That is U2b's `NEW_RULE` with the descriptive switch off, the form
  zu2b's own selftest runs. CVT mediators are left out of the texts and of the owners before the passes.
- The package's NER edge counts: musique 800,278; squad 156,130; hotpotqa 20,369,074; 2wiki 34,067,241; metaqa 5,949;
  webqsp 2,994,802.

The grading (section 5) ranks X by the question's cosine, so it reads edge sets only. The weights are filed for the
walks (U2m) and the models (U2n).

## 4. The candidates

The four forms of EL, in declared order:

| form | edges |
| --- | --- |
| E1 | IS |
| E2 | IS ∪ CO_t |
| E3 | IS ∪ CO_s |
| E4 | IS ∪ CO_t ∪ CO_s |

Each form is graded in three unions, the way the families serve a pool:
- **alone**;
- **with R**: on a passage dataset R is the sibling links (passages of one article, U2b's S part); on a KB it is the
  stored triples (PKG);
- **with R and kNN** (the three families): kNN is the package's `knn.npz`, unchanged.

References, beside: PKG; W2_S (U2b's chosen graph); PKG ∪ NER ∪ kNN (the package's three families); W2_S ∪ NER ∪ kNN
(the families U2e's walk reads). That is 12 candidates and 4 references on each dataset. All graphs are undirected and
collapsed, with rows in node-id order, built by `zu2b.undirected_csr`.

## 5. The grading

This is U2b's section 3, unchanged, on every graph above. For each question, B is the frozen construction's base pool
with its seeds (U2b's question files, `outputs/u2b/<D>/questions_<carve>.npz`, sha256 checked against their records).
X is B's one-link neighbours outside B, ranked by cos(q, v) with the frozen dense vectors, ties broken by node id. For
k in 25, 50, 100, 250, 500, 1000, 2000 and 5000, U2l records ALL, recall and rows of B ∪ X[:k]. Beside: all of X and B
alone. Scores are computed once per question over the union of the graphs' X's (`zu2b.reach_curves`), so every graph
ranks by the same numbers.

**The frozen I, beside.** The frozen I over a candidate is `m3b_pools.expand_hops` with the dataset's setting, the
candidate's EL ∪ R in place of the structural family, and the package's NER and kNN families unchanged. The program
named "frozen I's ALL" as the deciding read. U2b's amendment of U2A (docs/U2B_LINK_RULE.md, section 3) applies here as
well: I's first hop takes each seed's first 25 neighbours in node-id order, so on a denser graph an order unrelated to
the question cuts its reach. U2l therefore decides by U2b's score and files I's ALL beside, on every candidate.

**Carves.** s1sel on the five training datasets (musique, squad, hotpotqa, 2wiki, metaqa); webqsp's s1eval beside. On
the KBs, U2b graded only PKG; EL is new there, so U2l grades them like the passage datasets. No test split is read.

**The question's links, beside (deciding nothing).** The question's text is scanned with the same surfaces and gates.
The capitalization statistic is the corpus's. Q(q) is the nodes of every group the question mentions. On the chosen
form's three-family graph, U2l files the reach of (B ∪ Q(q)) ∪ X'[:k], where X' is the one-link neighbours of B ∪ Q(q),
and |Q(q)|. A pool that seeds from Q(q) is U2m's to declare.

## 6. The choice

- **Score:** per dataset, U2b's choice score (`zu2b.choice_score`, the mean ALL of B ∪ X(k) over k in 100, 250, 500,
  1000 and 2000). The choice reads the mean over the five s1sel datasets.
- **What is chosen:** EL's form, read on its three-family union. Among E1 to E4, the pick is the earliest form whose
  score is within 0.002 of the best form's.
- **Universal:** the same form on all six datasets. A per-dataset best is reported beside, as an upper bound only.
  webqsp's s1eval is reported beside and never moves the pick.
- **Outcome:**
  - `CHOSEN <form>` when gates 1 to 3 hold and every candidate is graded on all five datasets;
  - otherwise `REFUSED`, naming what is missing.
- **Beside:** the alone and with-R unions of every form, the references, and the frozen I.

A pick closes only the forms graded here (docs/SCREENS.md, section 3b).

## 7. Gates

1. **Selftest** (`zu2l_el.py --selftest`), on toy passages and toy names:
   - CO_t's pairs and weights equal a brute-force count;
   - the KB scan equals `zu2b.scan` on the same texts with the descriptive switch off;
   - a name's own group links nothing, and a CVT neither mentions nor owns.
2. **Gate 1 (before any grade).** On each passage dataset: U2i's `links.npz` and `evidence.npz` carry the sha256 its
   `pass.json` names, and U2i's gate 1 is PASS. E1 with the sibling links equals U2i's graph for the chosen cue set
   (W2_S, or the chosen A graph), array for array.
3. **Gate 2.** CO_s is the package's `ner.npz`, its sha256 checked against `GRAPH_MANIFEST.json`. Over PKG, the frozen
   I equals U2b's filed I for PKG on every dataset and carve graded (U2b's gate 2). A dataset that fails a gate is
   REFUSED, and none of its numbers is read.

## 8. What follows (fixed now)

- **U2m** reads the chosen EL in place of W2_S and NER as its entity family. Each use is declared in U2m's own
  follow-ups, which this file does not set.
- **U2n's models** read EL's kinds (IS, CO_t, CO_s) as edge types, with their weights.
- If no form beats W2_S ∪ NER ∪ kNN, the next forms are, in order:
  - the merged vocabulary: a spaCy key equal to a group's lowercased title is that group. This needs spaCy's keys on
    every corpus, about 18 CPU-hours each on hotpotqa and 2wiki (docs/DIAG_LINK_WORDS.md);
  - IS ordered by U2j's link precision, then by U2k's scorer, as their filed choices allow.

## 9. How it runs (laptop)

- `zu2l_el.py build --dataset D`: EL's kinds and the candidate graphs, with a record per graph.
- `zu2l_el.py grade --dataset D --carve C`: section 5.
- `zu2l_el.py choose`, then `zu2l_el.py report`.
- Order: metaqa, squad and musique first, then webqsp, hotpotqa and 2wiki (2wiki after U2i's 2wiki pass). It runs on
  the laptop after U2m's laptop runs, or beside them when memory allows.
- Outputs: `outputs/u2l/el/<D>/build.json`, `grade_<carve>.json`, `choice.json` and `report.md`. Arrays (npz) stay
  uncommitted.

## 10. Results

None yet.
