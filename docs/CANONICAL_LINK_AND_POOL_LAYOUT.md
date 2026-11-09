# Canonical link and pool layout (8 November 2026, 02:35 IST)

This page fixes what the link families are called, what each holds on every dataset, and how pools and graphs are
built from them. It is the one place to check before using the words "hyperlinks", "structural", "text links" or
"rule links". It declares no stage, runs nothing and changes no number; every later stage starts from it.

The user, 8 November 2026 (early hours, IST), in their words, shortened:
- keep the original edges of each graph as the structural family: the hyperlinks for 2Wiki and HotpotQA, the triples
  for MetaQA and WebQSP;
- the rule-built links and a better NER family are applied by the same rules on every dataset;
- the families are S, R, K; edges are deduplicated by priority S > R > K, so the original edge is kept first, then the
  rule edge, then the kNN edge;
- structural links are never removed again;
- from these clean pools, generalize and improve with the models we have (the universal and fast families);
- the pools already cover the golds; what is left is fewer nodes;
- what was done so far is not continued; this page is the clarification, and the after-paper program starts from it;
- another multi-hop dataset, named by the guide, is added.

## 1. The three families

| family | name in older records | what it holds | per dataset |
| --- | --- | --- | --- |
| **S** | `structural` | the original edges the dataset's store ships with; never rebuilt, never removed | 2Wiki, HotpotQA: the corpus's own Wikipedia hyperlinks. MuSiQue, SQuAD: title mentions (the package's rule, which U1a reproduces exactly on MuSiQue). MetaQA, WebQSP: the knowledge-base triples. |
| **R** | W2_S rule links (U2b) plus the `ner` store | edges built by our LLM-free rules, the same rules on every dataset, as ONE family | a passage links to the passages whose titles or aliases it mentions (W2_S), plus the NER store. On the KBs the rule runs on an entity's name and its KB neighbours' names (ABL-U). |
| **K** | `knn` | the 3 nearest neighbours by cosine of the frozen encoder's vectors, one row per unordered pair | the same on every dataset |

- **R's NER store.** Today the store inside R is the package's NER family (built with spaCy; ML1, PKO and ABL read it).
  The user's plan is to replace it with a rule-based NER built the same way on all six datasets (N1 of U2r, or a
  successor; the 5 Nov ruling "drop spaCy NER"). That gets its own declared stage. R's definition does not change when
  the store does.
- **Deduplication, S > R > K.** An edge between the same two nodes that sits in more than one family is kept once, in
  the highest family. The details (directed against undirected pairs, weights, relation types, what a model sees of
  the dropped copies) are fixed in the stage that implements it. The ablation's modules (`zabl*.py`) have no such step
  today; they keep each family as built.
- **Never remove S.** A pool or graph is S plus R plus K, and an arm without S is a read-time experiment (section 4),
  never a model of record or a headline.
- **One construction for all six datasets.** No family is chosen, dropped or rebuilt for one dataset. The three
  per-dataset choices still in the pipeline (pool sizes, SQuAD's question links, the two-name list of typed KBs) are
  listed in PROJECT_STORY section 5.6 and stay open items, not precedents.

## 2. Words that have meant two things

| the word | where it was used | canonical reading |
| --- | --- | --- |
| "hyperlinks" | records before 6 Nov; the 9 Oct ruling | S on 2Wiki and HotpotQA only |
| "hyperlinks" | the 6 Nov thread and the user's 8 Nov messages | our rule edges (R) |
| "text links", b1t | B1P, U1a | NER and kNN with S emptied: **a graph with no title-link family and no S** |
| b1 (B1b) | 9 Oct | S + NER + kNN, corpus hyperlinks on 2Wiki and HotpotQA |
| b1u | U1a | S replaced by the first title-mention rule |
| b1w | B1P | S replaced by W2_S (R's rule part) with NER and kNN: **R + K, no S** |
| "links add +18" | guide tables, the paper's step sentences | the model trained WITH a title-link family, read with it removed: a read-time shift, not a refit |

**Use S, R and K in new text and avoid the bare word "hyperlinks".**

## 3. What each filed record is, in this layout

| record | families in its training pools | notes |
| --- | --- | --- |
| zrc, zsp J5 and the B1b reads (9 Oct) | S + NER + K by M3A regime | S is the corpus hyperlinks on 2Wiki and HotpotQA |
| the lean pair zrckp, zspk (14 Oct) | S + NER + K by M3A regime (STRUCT: MetaQA, HotpotQA, 2Wiki; FULL: MuSiQue, WebQSP; none: SQuAD) | seed 0 |
| ML1 refits zg6r, zrckp; B1P reads; the WWW draft v4's headline rows | R + K on the passage datasets, **S removed** (PKO pools) | not canonical: S is missing. Three seeds. |
| ABL arm SRK | S + R + K | the canonical layout; its reads are carry-over reads of the ML1 fits, no refit |
| ABL arms S, SR, SK, R, RK, K, 0 | subsets | read-time experiments, never fitted on their pools |

## 4. Read-time against trained

- A fit is **trained on** the families its pools held.
- An arm that **removes** a family at read time (b1t, ABL arms without S) is a shift: the model sees inputs it never
  trained on, so its drop overstates what a model trained without the family would lose. ZN (12 Oct) is the one
  trained edge-input-free MLP on record.
- A **refit** on an arm's own pools is ABL amendment A2 and the user's call. It has not been called.

## 5. What this changes

- **For the models of record.** Nothing moves by itself. The universal and fast families, zrckp, zrckpn, zg6r and GNX
  stay what they are. Refitting them on S + R + K pools is a stage of the after-paper program (section 6), not
  done.
- **For the WWW 2027 draft (local, never committed).** Its headline rows are the ML1 and B1P fits on R + K pools, the
  9 Oct "headline without the datasets' hyperlinks" ruling. This page supersedes that ruling for every later stage.
  Whether the draft changes before 9 and 16 Nov is the user's call (open item 1).
- **For the ablation.** Its SRK arm is the canonical layout. Nothing further is queued by this page; the queue on the
  host is left as it is.

## 6. The after-paper program starts here

The order is the user's of 5 Nov (docs in memory: project-after-paper-roadmap), read against this layout. Nothing below
is declared or queued before the user says "go" (D19).

1. **Pools on S + R + K, same coverage at fewer nodes.** The pools reach 97.8 to 98.7% of the golds in 0.013 to 0.018%
   of the full corpora (COV, 25ce7d7), at a row price on the unified pools (U2z, 68f9e2b). The aim is the same
   coverage at fewer rows, with S in.
2. **NER unification.** The rule-based NER store replaces spaCy's inside R, on all six datasets.
3. **Links on MuSiQue and WebQSP.** A reach diagnosis first: each missed gold placed as no link path, a path beyond
   the row budget, or reached but ranked low.
4. **Deduplication S > R > K** implemented and read.
5. **Refit the models on the canonical pools** (the user's call), then generalize with them: zero-shot and the
   leave-out splits as before.
6. **A new multi-hop dataset** (open item 2).
7. Then the encoder gap, the graph vectors in a semantic space, the smaller-encoder comparison, and the paper
   session's 4a to 4e (all in the roadmap).

## 7. Open items (the user's call)

1. **The paper.** The user said the current draft "can be a paper" but is not what they wanted to submit. The draft's
   headline rows lack S (section 5). The records cut off 6 Nov; the abstract is due 9 Nov and the paper 16 Nov
   (project notes). Default: the draft stays as it is, with the shift disclosed, until the user rules.
2. **The extra dataset.** The guide's message named a multi-hop dataset that the repository does not record by name.
   The guide's 6 Nov review asked about Wikipedia scope against the paper's "Web" title and the draft lists "test the
   rule beyond Wikipedia" as a next step; MultiHop-RAG (news articles) would fit that point, but this is a guess and
   unverified. The user names the dataset; a stage file declares the build.
3. **The running queue.** The ablation rows on the host stay as queued unless the user cancels them. The host has been
   unreachable since about 00:30 on 8 Nov.
4. **The refit** of the models on S + R + K pools (section 6, item 5).

## Sources

docs/ABL_LINK_FAMILY_ABLATION.md (arms, carry-over, A2, A3); docs/B1P_PKO_CHAIN_ON_B1.md (b1t, b1w);
docs/U1A_B1_WITHOUT_HYPERLINKS.md (b1, b1t, b1u); docs/U2B_LINK_RULE.md (W2_S); docs/U2R_NAME_RULE.md (N1);
docs/P2A_LINK_FAMILIES.md (the old S, N, K families and regimes); docs/ML1_REFITS_ON_PKO.md; docs/ZN_LEAN_NO_MP.md;
docs/PROJECT_STORY.md sections 2.4 and 5; docs/COV_PICK_O_COVERAGE_AND_LOCALIZATION.md.
