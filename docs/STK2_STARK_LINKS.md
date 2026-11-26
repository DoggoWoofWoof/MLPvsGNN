# STK2: the seven's link layout on the three STaRK knowledge bases, built the same way

Declared 26 Nov 2026, 23:38 IST, before any STK2 link number exists. Code: `outputs/mp_unified/zstk2.py` (a new module, committed with this page before any link row runs; it imports `zstk1.py`, `zmhr.py`, `zd84.py`, `zul1.py`, `zu2b.py`, `zenc1.py` and `zenc1b2.py` and edits none of them). Follows `docs/STK1_STARK_PACKAGE.md` (the packages, the first-stage read) and `docs/ING1_INGESTION_AND_STAGE_REGISTRY.md` (the loader `zworld.py`, G-W). The layout it applies is the one that stands on the seven: B0 = S | W2_S | F0T | K_e3v (`docs/UL2_LAYOUT_ON_E3V.md` section 14.5).

## 1. Why this page exists

The user's words (26 Nov): **"deal with all the datasets the same way no matter what"**, **"all 10"**, and on the repository: **"whatever dataset in future we will get make sure we can properly process it ... cant reprocess everything again and again and keep changing rules."** Earlier (21 Nov): "we need to have universal rules". STK1's section 2 named STK2 and left open which of the seven's rules need what from a node. This page closes that, before a link is built.

What it does: builds the seven's link families on Prime, MAG and Amazon with the seven's own builders, labels every edge by the rule that made it, deduplicates S > R > K, and writes sizes. What it does not do: no retrieval number, no pool, no model, no change to a rule or a constant, no choice made for one KB only.

## 2. What each rule needs from a node

| family | the seven's rule | needs from a node |
| --- | --- | --- |
| S | the dataset's own structure | endpoints only: the benchmark's edges (`graph/s_edges.npz`, relation per edge), kept whole, labelled `S.<relation name>` |
| R.mention (W2_S) | U2b's chosen rule (`zu2b.link_pass`, same-name siblings) | a **title** and a **text** |
| R.name_comention and R.title_comention (F0, F0T) | UL1's build (`zd84.rule_forms`) | a **title** and a **body** |
| K | the 3 nearest neighbours by cosine of the e3v rows (`zenc1`, `zenc1b2`) | the node's e3v row |

STK1's package keeps only T1's text (type head and every field flattened). It has no title. So the one thing STK2 adds is a title, and that is a rule of its own:

## 3. N1: a node's name

**N1.** A node's name is the first string among the top-level keys `name`, `title`, `DisplayName` of its raw record that, with its spaces collapsed, is not empty, is not `nan` and holds a letter (T1's own test for a usable string, so a benchmark placeholder such as `-1` is no name), read from the benchmark's `node_info.pkl` through `zstk1.safe_load` (the restricted unpickler). It is the same ordered key list on all three KBs; no per-type or per-KB choice. A node with none of the three has **no name**. A nameless node keeps its text and its vectors, takes part in S and K and in the scan as a source, and is given a title no text contains (`"\x00" + node_id`), which is unique, so it is nobody's target and no node's sibling. (Without this, every nameless node would share the empty title and the same-name sibling pairs would be quadratic in them.)

**Looked at before this page, disclosed.** The raw key lists were read once with `scripts/stark/stark_topkeys.py` (top-level keys per type; no link, no question, no gold read): Prime has `name` on 100 percent of its nodes in all ten types; MAG's papers have `title` (100 percent), its authors `DisplayName` (92.5 percent, so 82,853 authors are nameless), its fields of study and institutions `DisplayName` (100 percent). Amazon's raw file was not read (5.2 GB pickle; read when its package row has run and memory allows); N1 is applied to Amazon as written, and if Amazon has none of the three on a type that is reported, not worked around.

**What the title is not.** N1 does not clean, split or enrich a name; the seven's builders do their own title reading (`zu2b.norm_title`, `forms_of`). A name is the benchmark's own field, which is what the seven's titles are (MuSiQue's and Wikipedia's titles, MetaQA's and WebQSP's entity names).

## 4. The layout

B0 = S | W2_S | F0T | K_e3v, built per KB:

- **S**: every edge of `graph/s_edges.npz`, as the unordered pair, with the set of relation ids that join it (a `uint32` mask over the 18 relations of Prime; the masks of MAG and Amazon fit the same way if their relation counts are at most 32, else the page gets an amendment first).
- **R**: W2_S (`zmhr.w2s_keys`: U2b's rule, variant W2, form S, plus same-title sibling pairs between descriptive nodes; here every named node is descriptive, since its T1 text is longer than its name) and F0T (`zd84.rule_forms` over `{title, body = T1 text}`), both over the node text as the seven read it. The choice among same-named targets uses the e3v rows (the seven used the frozen rows because their W2_S was built before e3v; the STaRK KBs have no frozen arm, so the one arm is used for both and this is a difference to read, not a hidden one).
- **K**: the 3 nearest neighbours on the e3v document rows, `zenc1.nn_top` and `zenc1b2.k_edges` as `zmhr.k_family` calls them, with the same search depth.
- **Dedup and labels** as UL2 section 8: S > R > K; label bits 0 S, 1 R.mention, 2 R.name_comention, 3 R.title_comention, 4 K.knn (D83's vocabulary); family 0 S, 1 R, 2 K; plus the relation mask for the S pairs.

## 5. Gates (a failure stops that KB; none of its sizes is read)

- **L1 names**: the sidecar has one row per node in node order, its count equals `n_nodes`, the raw file has the same node count and keys 0 to N-1, each node id is `<dataset>:<position>`, on 1,000 evenly spaced nodes every name sits inside the node's own T1 text (T1 keeps every field; nodes cut at the cap are skipped), and the per-type named share and the key each name came from are written.
- **L2 package**: `zworld.load` reads the package, its record hash is the freeze's, and the vector shard count equals `ceil(n_nodes / 40,000)` for both parts (G-W's W7, now with the vectors written).
- **L3 builders**: the toy self-test of `zstk2.py` (twelve nodes, hand-checked pairs: a named pair, a nameless node, two same-named nodes, a mention in a body, a kNN edge) passes before any real row.
- **L4 size stop**: the same-name sibling pair count (the sum over name groups of k(k-1)) and the W2_S and F0T counts are computed first; **if the sibling count exceeds 50,000,000 pairs, or any family's keys exceed 2^31, the row stops and reports the group sizes**. It does not cap, cut or sample: the rule is changed only by an amendment on this page. (No seven-dataset group comes near this; MAG's authors are the case this stop exists for.)
- **L5 identity**: the union of the families equals the dedup store's keys; every S pair carries a non-zero relation mask; the family array is the highest family of each key's label bits.

## 6. Order, size and the subset rule

Prime first (129,375 nodes; its e3v vectors come from STK1 row `stk1p-vec`), then MAG, then Amazon: size only, as STK1. The user's subset rule applies: each KB's **names** step is read first on its own (cheap, a package observation); a **links** row asks for memory from the previous KB's measured peak working set scaled by node count, and MAG's and Amazon's first links row is a dry pricing pass (the L4 counts, the surface table size and the K cost from a 2,048-row block) that stops before building anything and is read before the full row is queued. K's cost on MAG (1,872,968 rows against 1,152 columns, all pairs) is priced from Prime's row by n^2 scaling and, if it exceeds 3 days on 8 threads, this page gets an amendment on the thread count or the search method **before** the row is queued (STK1's X6 form). No GPU row.

## 7. What is read

Sizes only: per KB the number of S, W2_S, F0T and K pairs, their overlaps, the per-label counts and the per-relation S counts, the named share per type, the largest name groups, the L4 counts, peak working set and seconds. **No retrieval number, no pool, no call.** Whether R and K add anything to S for these KBs is read in STK3 at equal distinct nodes against the first stage's flat top-N. One reading is fixed here so it cannot be chosen later: the R and K families are judged on STaRK the way they were on the seven, by what they add at equal pool size, never by their size.

## 8. Code, rows and outputs

- `outputs/mp_unified/zstk2.py`: `names --kb K`, `links --kb K [--price]`, `selftest`. The sidecar is `outputs/stark/<kb>/names.jsonl` (`node_id`, `name`, `key`) with its record `outputs/stark/<kb>/names.json`; the build is `outputs/stark/<kb>/links/` (`store.npz`: `keys`, `labels`, `family`, `rel_mask`; the three family files) with `outputs/stark/<kb>/links.json`. Both are checkpointed (the names in one pass; the links write each family when it is built and `links.json` last).
- The package is not edited (STK1's record, the freeze and the e3v shards stay as filed). `zstk1.py` is not edited.
- Rows go through the feeder in the classes `stk2p`, `stk2m`, `stk2a`, behind the KB's vectors. The host's `mpr-cpu` environment has no numba; the self-test (L3) is run on the host as well as on the laptop, so that a module that needs numba is found before a row is queued, and if one is, the row goes through `outputs/host_ops/pylib_run.py --host` as the pko-based rows do.

## 9. Results

(Not yet read. Written after the self-test and the names step; the links rows follow the KB's vectors.)
