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

## Amendment 1: the mention relation on the KB datasets (declared 10 October 2026, about 07:40, before any number)

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

## Results

Not yet run.
