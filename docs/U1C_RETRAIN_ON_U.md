# U1c: the two bases retrained on the universal graph and the precise pools

Declared 10 October 2026, about 02:30, before any number of the stage. It follows U1d (docs/U1D_PRECISE_LINKS.md: one
LLM-free link rule for all six datasets) and step 4h (docs/STEP4H_U_POOLS.md: pools on that graph, one configuration
for all six).

## Why

The user, 10 October: links first, then pools, then one retrain on both. The headline must not use the 2wiki and
hotpotqa hyperlinks (docs/PAPER_CLAIMS_2026_10_09.md). Today's models were trained and read on today's graph, which
holds those hyperlinks, and on step 1's frozen pools. U1c asks what the two bases do when everything below them is the
universal construction:

- **the MLP base, zrc** (zrm's model over zrc's chain entries; no message passing);
- **the GNN base, zsp** (zrm plus one propagation head over each row's neighbours' scores, on zlink's pool edges).

Neither model, nor its settings, seeds or batches, changes. The encoder is unchanged.

## What changes

1. **The graph.** In every look, the dataset's `structural` family is U1d's structural_U
   (outputs/u1d/<D>/graph_structural_u.npz, the sha256 of U1d's build.json). The KB triples of metaqa and webqsp are
   kept. The 2wiki and hotpotqa hyperlinks are replaced by U1d's links. squad has no graph and keeps its stores. ner and
   knn are unchanged. The frozen construction (m3b_compile.prepare) runs on that graph, so its pools change too.
2. **The pools.** Each question's pool is step 4h's chosen configuration on GU (outputs/step4h/choice.json), built
   by scripts/u1c_host.py in step 4e's file format. Each file's ALL and mean pool size on s1sel and s1eval must equal
   step 4h's coverage record for that configuration.
3. **Everything computed from them.**
   - Looks: outputs/mp_unified/look_u1c.py, which is look_step4e with the graph replaced.
   - Caches: lean_cache into outputs/u1c/cache.
   - Chain entries: rmatch's and zrc's builds into outputs/u1c/chains.
   - Pool edges: zlink's build into outputs/u1c/links. The pool edges now come from structural_U inside each pool.

The models, the training code and the reads are today's (outputs/mp_unified/zu1c.py calls rmatch's train and read with
U1c's roots).

## The screen (this file)

- **Fits:** four, on the host's card: zrc and zsp, each on L-musique and L-hotpotqa (scr-u1c-zrc, scr-u1c-zrc-hp,
  scr-u1c-zsp, scr-u1c-zsp-hp).
- **Reads:** each fit on the six s1eval carves of U1c's pools, p@swa.
- **Bases:** each arm's own card fit of the split on today's graph and pools, trained the same way on the same device.
  - zrc: scr-zrct and scr-zrct-hp (round eighteen).
  - zsp: scr-zspg and scr-zspg-hp (round thirty-one's base fits).
  - The same questions and gold totals. R@5 is over the question's golds, so a gold the pools lose counts as a miss.
- **Call:** relz's pair and its re-call under the seed null, each arm decided against its own base. GAIN, LOSS and
  WITHIN are docs/SCREENS.md's.

### The rule, fixed before any number

On 2wiki and hotpotqa the bases read graphs that hold the corpus hyperlinks. U1c's reads do not. Those reads therefore
measure the cost of dropping the hyperlinks (net of U1d's links and the new pools), not a choice the headline could
make. The rule:

- **ADOPT_FOR_FULL_RUN:** for each arm, the re-call has no LOSS on metaqa, musique, webqsp or squad (in-domain or
  zero-shot), and the two arms together have at least one GAIN anywhere.
- **NOT_ADOPTED:** otherwise.
- The 2wiki and hotpotqa reads are reported in full, every call shown. A LOSS there is not hidden. It is the measured
  price of the hyperlink-free headline and goes to the gap ledger.

One construction and one rule for all six datasets. Nothing is chosen per dataset.

## What follows

- **The full run:** six splits per arm under grade-nullx, then B1 read again beside it (never used to choose). It goes
  in its own file, and only after ADOPT_FOR_FULL_RUN.
- **Latency:** step 4h's pools are at most step 4e's mean size, but the per-question compile reads structural_U. Any
  latency figure is cold, timed in its own stage.

## How it runs

- `scripts/u1c_host.py pools --dataset D --carves screen` (under the host's pylib), once step 4h's `choose` has run.
- Looks: look_u1c.py, one host job per shard, with step 4e's shard counts.
- Caches: lean_cache with `--look-root outputs/u1c/look --out-root outputs/u1c/cache`, then `copy-bases`.
- Builds: zu1c.py's rmatch, zrc and links builds.
- `scripts/u1c_host.py gate --screen` checks that every look ran on the filed pools and every cache row count is the
  filed pool size. Training needs its PASS.
- The four fits and reads on the card, then the compares, the base compares of scr-zrct against step 1
  (outputs/u1c/base-zrct-<split>.json, for relz's null), the pairs and the re-calls.
- Disk: step 4e's caches, chains and looks for every carve were about 70 GB. The screen's 11 carves at no more than
  step 4e's pool size need less. The stage is queued only with at least twice that free, under the 100 GB floor.

## Amendment 1: the mention relation on the KB datasets (declared 10 October 2026, about 07:25, before any number)

U1d adds its mention links on metaqa and webqsp as one new relation id after the KB vocabulary (metaqa 9, webqsp
7,058). The relation table has no row for that id, so all 64 of metaqa's fit looks stopped at the compile
(`relcos[erel]`: index 9 out of bounds for 9 relations). No number was read.

The fix, one rule for all six datasets: **a mention link is a relation with no text.**
- **The relation table** gets one zero row for the mention id. Its compatibility cos(q, e_r) is 0, the value an
  untyped pair already gets. Its ief uses M3B's formula on structural_U's count. The KB relations' ief must equal
  today's table bit for bit, or the look refuses.
- **The relation slots** take no bank row for a mention entry. A pair with only a mention link has every slot −1,
  as an untyped pair has, so rmatch's and zrc's chains skip it. A pair's mention entry sorts last, so its KB slots
  are unchanged.
- **The topology keeps every mention link**: pools, STRUCT columns, reach, depth, message edges.

On the four datasets without a relation table it changes nothing, so their looks stand. It runs as
outputs/mp_unified/look_u1c2.py, which is look_u1c with the two changes. Its selftest checks the zero row, the ief bit
for bit, the refusals and the KB-only slots. The metaqa and webqsp looks and everything after them are requeued
under new item names. The screen's rule, fits and reads are unchanged.

## Amendment 2: the chain builds read amendment 1's table (declared 12 October 2026, about 19:05, before any number)

Amendment 1 extended the relation table in the look's memory only. rmatch's and zrc's chain builds read the relation
table from its file (outputs/m3b/relations/<D>_rel_embeddings.npy, the KB rows only) and refuse a table with fewer rows
than the look has relations. All three of U1c's rmatch builds stopped at that check (metaqa fit and s1eval, webqsp
s1eval: "(7058, 1536), the look has 7059 relations"). No chain was built and no number was read. Everything after
the builds (the zrc builds, the fits, reads, compares, pairs and re-calls) was dropped with them.

The fix applies amendment 1's rule to the file:
- **The table.** `zu1c.py relations --dataset D` writes outputs/u1c/relations/<D>_rel_embeddings.npy. It holds the KB
  table's rows bit for bit (float16, as filed), then one zero row for the mention id. It refuses unless the mention id
  equals the KB table's row count (outputs/u1d/<D>/build.json). The record (`<D>_record.json`) gives both files'
  sha256. A second run must find the same bytes, or it refuses.
- **The chain builds.** U1c's rmatch and zrc builds read that table.
- **The chain carves.** At training and reading time, U1c's chain carves read the same table: zu1c's on_u1c swaps
  their table directory together with their chain roots.
- **U1f.** U1f runs zu1c on its own roots and reads the same table. U1e's GU_1024 has U1d's relations and the same
  mention id (metaqa 9, webqsp 7,058).

What it changes:
- **The chain entries: nothing.** A mention entry takes no relation slot, so no chain step and no question relation is
  the mention id. Both builds use the table only for its shape, its sha256 and (zrc) the z-scored cosines of the
  pool's own relations.
- **One thing in the model.** rmatch's centred relation table (rel_centered = U − mean U, used by both arms' chain
  match) is centred over the table it reads, and that table now includes the zero row:
  - the mean is 9/10 of the KB rows' mean on metaqa and 7,058/7,059 of it on webqsp;
  - every relation's centred vector moves by the same amount, so at each hop every step's logit gets the same
    question-dependent offset;
  - the per-question z-scored cosines are over the pool's relations only, which never include the mention id, so
    they are unchanged.
- **The four datasets without a relation table:** nothing.

The relation tables are built first (u1c-rel-metaqa, u1c-rel-webqsp). U1c's rmatch and zrc builds and everything
after them are requeued under new item names (u1c-*3-*). U1f's chain builds, not yet sent, wait for the same two items.
The screen's rule, fits and reads are unchanged.

## Results

Not yet run.
