# UL1: one rule-built NER store inside R on the six training corpora, graded against the oracle pairs before any refit

Declared 16 November 2026, before `outputs/mp_unified/zul1.py` reads any real file. The user, 16 November 2026 (verbatim): "i do
not think we need to unnecessarily train models after every improvement, like if we created better links we can train a model
after verifying those links improved connectivity etc by comparing with the oracle links we have and across datasets etc and
unifying everything, properly check the after paper stuff, spacy removal, musique links etc"; and, while this page was being
written: "make sure the SRK and the related family rulings are properly followed".

**Nothing trains, no model reads anything, and no filed graph, pool, fit or record changes.** UL1 is graph level. It decides
one thing: whether a rule-built NER store is carried, as R's NER store on all six datasets, into ONE later refit (section 7).

## 0. The family rulings this page follows (docs/CANONICAL_LINK_AND_POOL_LAYOUT.md)

- **S, R, K** (8 Nov): S = the original structural edges the dataset ships with; R = our rule links (U2b's W2_S) plus the NER
  store, the same rules on every dataset, as one family; K = the 3 nearest neighbours. Priority S > R > K.
- **A1** (10 Nov): S only where the data has structure. 2Wiki and HotpotQA: two domains, S + R + K and R + K. MuSiQue and
  SQuAD: R + K only (their structural store, title mentions, is not S and is not read here). MetaQA and WebQSP: one domain,
  S + R + K, where R's rule part is ABL-U's rule graph on the entity names (`outputs/abl/u/<d>/W2_S.npz`) and S, the triples,
  is never removed.
- **A2** (11 Nov): the pools of record stay as built (PKO's pick O on each domain's graph); a change to a family reuses the
  pools, caches, looks and fits by sha256 and recomputes only what reads the changed edges, named before it runs. Only a
  ruling of the user's that names A2 changes the pool standard.
- **Section 6 item 2** of the layout: the rule-based NER store replaces spaCy's inside R on all six datasets. "R's
  definition does not change when the store does." This page is that item's graph-level check.

## 1. Why, and what is already known

- AP1, AP1b and AP2 tested the replacement on B1's three 1,000-question settings only: U2r's CN_all links as many gold pairs
  as spaCy's family (LEVEL on 2Wiki and MuSiQue, ABOVE on HotpotQA; AP1), AP1b's F0T (CN_all plus each passage's own title
  as a name) links 71.18% of MuSiQue's chain slots against N_sp's 68.26%, and the models of record read F0T's carve within
  0.03-0.26 points of spaCy's (AP2, REPLACE; a read-time shift).
- The six training corpora (the package's `data/final_canonical/<d>`, the substrate the pools of record, the caches and
  the fits are built on) still hold spaCy's store inside R. Nothing has graded a rule-built store against gold pairs on
  these corpora; U2r graded name forms there by a pool-reach score (U2r section 10: N2 = IS + CN_all against E3 = IS +
  spaCy within 0.002 on four of five datasets and -0.0026 on MetaQA), not by gold pairs.
- So the refit the user asked to be earned (D83) waits on this check.

## 2. What is read (all read only)

For each dataset d in {musique, squad, hotpotqa, 2wiki} (the passage corpora) and {metaqa, webqsp} (the KBs):
- the package's nodes (`nodes.jsonl`, checked against `DATASET.json`), the NER store N_sp and K (`graph/ner.npz`,
  `graph/knn.npz`), and S (`graph/structural.npz`) on 2Wiki, HotpotQA and the KBs only, each checked against
  `GRAPH_MANIFEST.json`; the served freeze checked by `u1b.package_root`;
- R's rule part: U2b's W2_S on the passage corpora (`outputs/u2i/<d>/ref/`, checked against `outputs/u2b/<d>/graphs/
  build_new.json`), ABL-U's on the KBs (`outputs/abl/u/<d>/W2_S.npz`, checked against its `build.json`);
- U2r's names (`outputs/u2r/<d>/names.npz`: CN_all, checked against `names.json`);
- the questions of the carves s1sel and s1eval (`outputs/u2b/<d>/questions_<carve>.npz`: gold positions; webqsp has s1eval
  only) and, for MuSiQue, each question's `question_decomposition` from the package's queries, matched by the carve's ids
  (`outputs/step1/carves.json`).

## 3. The NER-slot forms (one inside R at a time) and the domains

| form | what R's NER store holds |
| --- | --- |
| none | nothing (R = its rule part alone) |
| N_sp | spaCy's store of record |
| CN_all | U2r's co-mention pairs over its rule names, df in [2, 25] (AP1's carried form) |
| F0T | CN_all's names plus each read passage's own title runs as names of that passage, co-mention pairs at df [2, 25], unioned with CN_all (AP1b's carried form, built by zap1b's `runs_of` and `title_stripped` and zu2l_el's `co_mentions`, unchanged) |

On the KBs a node's title is its text, so F0T adds only the runs of the stripped title (a parenthetical or comma tail removed)
that the name's own runs lack; F0T's size there is reported, not gated.

The domains graded, per A1:

| dataset | domains | union |
| --- | --- | --- |
| 2Wiki, HotpotQA | SRK and RK | S + (W2_S + form) + K; (W2_S + form) + K |
| MuSiQue, SQuAD | RK | (W2_S + form) + K |
| MetaQA, WebQSP | SRK | S + (ABL-U's W2_S + form) + K |

**S > R > K.** The priority changes which family holds a pair, not the union's pairs, so no reach number depends on it. Each
union's pairs are reported family by family under it: S's pairs, R's pairs not in S, K's pairs in neither.

**Edge labels.** The user, 16 November 2026 (verbatim): "the edges should also have labels based on how it was created so that
we can unify with KB labels and make the model train and understand better rather than just giving family names". Each pair
carries the label of every rule that made it, in one vocabulary with the KB relations:

| label | made by |
| --- | --- |
| S.<relation> | the structural store's own relation vocabulary (MetaQA 9 relations, WebQSP 7,058; 2Wiki and HotpotQA one) |
| R.mention | R's rule links (W2_S: a passage or entity name mentions another node's title or name) |
| R.ner_spacy | spaCy's NER store (N_sp) |
| R.name_comention | two nodes share a rule name at df [2, 25] (CN_all) |
| R.title_comention | F0T's pairs not in CN_all: a node's own title run shared with another node's name |
| K.knn | the 3 nearest neighbours |

A pair keeps every label it carries: S > R > K decides the family that holds a pair, not the labels it keeps. W2_S is stored
without the kind of match that made each link (exact title, stripped title, alias, sibling), so R.mention is one label here;
splitting it by kind needs U2b's link pass to keep the kind (a later step, named here and not run). Reported per label, for the
unions with N_sp and with F0T: its pairs, the oracle slots carrying it, the slots carrying it alone, and slots per million
of its pairs. Descriptive: the labels decide nothing in section 6.

## 4. Slots (the oracle pairs) and measures

- **GP:** every unordered pair of distinct gold passages of a question, one slot each (questions with at least two golds).
  Counted on the carves before this declaration (gold counts only, no link read): SQuAD's questions have one gold each, so
  SQuAD has no GP slot; MuSiQue has 3,783 (s1sel) and 5,962 (s1eval) GP slots, HotpotQA 1,508 and 7,405, 2Wiki 3,131 and
  26,331. A KB question's golds are its answer set (up to 742 on WebQSP), not a chain, so on the KBs GP is read on
  questions with 2 to 10 golds only, beside.
- **CH (MuSiQue):** oraclepairs' ORC rule applied to the training corpus's questions: for each sub-question j, a slot
  between the supporting paragraph of j and that of every earlier sub-question its text names ("#1", "#2"); the hop nodes
  are `context_node_ids[paragraph_support_idx]`, mapped to node positions by the nodes file. A question whose hop nodes are
  not its gold set, or whose hops do not form one connected chain, stops the run (G3).
- **CC (MuSiQue), chain connected:** a question whose CH slots are all linked; reported by hops (2, 3, 4), with F0T's and
  CN_all's share minus N_sp's over questions (paired interval), beside.
- A slot is linked when its pair is an edge of the union. A share's difference between two forms is the ratio estimator over
  questions with the paired 95% interval of 2,000 question resamples, seed 0 (`zap1.boot_diff`); ABOVE / BELOW / LEVEL as
  `zap1.call`.
- **Price:** the union's distinct pairs; the ratio of a form's union to N_sp's union in the same domain; pairs added per slot
  added over the union without a NER store.
- Beside, deciding nothing: on 2Wiki and HotpotQA, the share of S's pairs the RK union holds with each form; on the KBs,
  the GP rows.

## 5. Gates (any failure stops that dataset; none of its numbers is read)

- G1: the served freeze is the declared one; every input file's sha256 equals its record's; U2r's names were built on
  the same nodes and freeze.
- G2: the rebuilt CN_all (the text runs alone) equals U2r's `names.npz` pairs exactly (pairs and weights).
- G3: every carve question's gold positions are in range; MuSiQue's hop nodes equal each question's gold set and form one
  connected chain; the carve's ids match `carves.json` in order.
- G4: the self-test passes before every real command: the streamed build equals zap1b's level-0 forms (F0 and F0T) and
  U2r's CN_all on toy corpora, a titled hub joins only in F0T, the keys, membership, union sizes and S > R > K counts equal
  a brute force, the chain slots and an unconnected chain's stop, and the rule on synthetic grades (a KB's BELOW on GP
  decides nothing).

## 6. The rule (fixed now)

For the forms in the order F0T, then CN_all, the first form that holds all three is **CARRIED**:
1. GP: form minus N_sp is not BELOW on every passage dataset with GP slots (MuSiQue, HotpotQA, 2Wiki), in each domain
   section 3 grades, on both carves;
2. MuSiQue's CH: form minus N_sp is not BELOW on both carves;
3. size, on all six datasets and every domain (one construction): the form's union has at most 1.30 times N_sp's union's
   pairs (AP1's B1 ratios were 1.11-1.18; the bound keeps the pool walk's degree near the record's).

If neither form holds, the verdict is **NOT_CARRIED**, naming each failing dataset, carve, domain and condition; no refit is
declared, and the next stage is the loss class (the slots N_sp links and the form does not).

## 7. What follows (fixed now)

- **CARRIED X:** X becomes R's NER store on all six datasets (R = its rule part + X; the same rule everywhere), the unified
  link set for ONE refit. D83 is declared in its own file before it runs, under A2: the pools of record stay as built and
  are checked by sha256; only what reads R's NER store is recomputed (the graph the models read, the family inputs and
  any look or cache computed from walks on those edges), each file named before it runs; then ONE refit of zg6r, zrckpn
  and GNX at three seeds on A1's domains, read on B1's settings with X's carve. Whether the S > R > K deduplication
  (layout section 6 item 4) is implemented in the models' inputs before that refit, and whether the pools are rebuilt on
  the new store (that needs a ruling naming A2), are the user's calls, asked with D83; neither is assumed. The models'
  inputs carry the edge labels of section 3 in place of family names alone (the user's direction of 16 November); how each
  model reads them (the GNNs per label in message passing, the MLP as per-label counts in its family inputs, which keep it
  free of message passing) is a model change, declared with D83 before it runs.
- **NOT_CARRIED:** no refit; the loss class is read, then a rule for it is declared, and UL1's check runs again on it.
- Either way, MuSiQue's chain reach on the training questions (CH, CC by hops) is the record for "the MuSiQue links" on the
  training corpus, beside B1's.

## 8. Outputs and runs

`outputs/ul1/<d>/f0t.npz` and `build.json` (the build; the arrays uncommitted), `outputs/ul1/<d>/grade.json`, and
`outputs/ul1/report.json` and `report.md` (committed). 2Wiki, HotpotQA and WebQSP (6.0 and 5.2 million nodes; WebQSP's ABL-U
rule graph holds about 500 million pairs) build and grade on the lab host, which holds the package's mirror and ABL-U's
WebQSP graph; MuSiQue, SQuAD and MetaQA may run on the laptop, by the same code on the same freeze (G1 checks it either
way). Commands: `python outputs/mp_unified/zul1.py selftest | build --dataset D [--host] | grade --dataset D [--host] |
report`.
