# ZN: the lean MLP without message passing

Declared 12 October 2026, about 15:15, before any number of the stage.

## Why

The user, 12 October:
- "mlp should not have any message passing"
- "i want the lean mlp to be 2-3x times faster"
- "universal have same input features for both and lean can have different depending on what is required by the
  model but overall, mlp should be 2-3x faster at the minimum from the fastest gnn"

C9 (docs/C9_UNIVERSAL_PAIR.md) timed zsp at 0.94-1.21× zrc's cold time. That is because zrc and zsp read the same inputs.

Five of zrc's nine families, and the chain entries on the KB graphs, are computed over the pool graph's edges, starting
from the seeds:
- walks;
- distances;
- topology;
- depth.

That is message passing by fixed rules. Building these families is most of zrc's cold time, and zsp needs them too. With
these inputs, no MLP can be 2-3× faster than zsp.

F1's permutation test (docs/F1_ONE_INPUT_SET.md) found every family used. Dropping these families therefore costs
accuracy, and this stage measures how much.

## What "no message passing" means here

The lean MLP reads no edge of the graph.
- None of its inputs is computed over edges.
- No row reads another row's inputs or score, with two exceptions:
  - the question's seed rows, which retrieval names (not the graph);
  - the pool's own statistics of a column (the within-pool z-score).

The pool is built as before. It is shared with the GNN and unchanged: candidate generation sits upstream of both models.

| family | reads edges? | in zn |
| --- | --- | --- |
| rank (dense_rr, splade_rr, rrf, agreement, is_seed) | no: the retrieval lists | yes |
| dense_cos | no: the row's own embedding against the question's | yes |
| SEMB | no: the question's embedding against the row's stored projection (learned maps) | yes |
| SEED | no: the row's stored projection against the seed rows' (mean, max, max over bucket 0) | yes |
| topo_STRUCT | yes: topology over the pool graph | no |
| depth_STRUCT | yes: multi-hop reach from the seeds | no |
| WALK, WALKF | yes: walk counts from the seeds | no |
| DISTS | yes: distances from the seeds | no |
| chains (metaqa, webqsp) | yes: typed relation paths from the seeds | no |

## The model and its training

zn is zret's model (lean_screen3.ZRet) on rank, dense_cos, SEMB and SEED, over the base carve. It has no chain entries
and no chain match.

Everything else is zrc's:
- lean_screen2's train;
- config 2e-3:1e-4:0.1:8:2, hidden 128, seed 0, p@swa;
- lean_gpu's flags (deterministic algorithms, TF32 off, 2 torch threads);
- the same carves and caches.

On the four passage graphs, zrc's model already is ZRet: its chain match is zero without typed relations. On those
graphs, zn differs from zrc only by the five families.

## The screen

- **Fits:** scr-zn (L-musique) and scr-zn-hp (L-hotpotqa), on the host's card, after a smoke. The smoke trains one epoch
  on 2wiki's select carve twice; a repeat must be IDENTICAL.
- **Reads:** the six s1eval carves.
- **Reported against** zrc (scr-zrct, scr-zrct-hp) and zsp (scr-zspg, scr-zspg-hp), read by read:
  - R@5, with the paired difference and its 95% question-bootstrap interval;
  - FC@5 and hit@1 beside;
  - rrf's R@5 (plain retrieval).
- **Claim 2's measure:** the share of zsp's gain over rrf that zn keeps, (zn − rrf) / (zsp − rrf), per read. It is not
  computed where zsp − rrf is under 0.01.
- **No adoption call.** The user's constraints decide the MLP's inputs; this screen measures what they cost.
  - The reads go to the gap ledger.
  - The next stages work on the gaps without message passing, each declared in its own file before its numbers.

## Speed

The next stage, C10, times zn's cold time against zsp's at its fastest exact form, as C9 did. It is declared in its own
file before any timed number.

## What this does not do

- No new column, objective, graph or encoder. The encoder and the substrate embeddings stay frozen.
- zsp, the pools, the edges and the links are unchanged.
- One rule for all six datasets. webqsp never trains. Test splits are never read. B1 is not used.

## Files

    outputs/mp_unified/zn.py                      the arm, train, read, compare, summary, smoke, selftest (new)
    outputs/screen/fits/scr-zn, scr-zn-hp         the fits and their reads (host)
    outputs/zn/compare-<split>.{json,md}          zn against zrc and zsp, read by read
    outputs/zn/summary.{json,md}                  both splits, with the share of zsp's gain

    python outputs/mp_unified/zn.py --selftest
    python outputs/mp_unified/zn.py smoke --device cuda --host
    python outputs/mp_unified/zn.py train --split L-musique --name scr-zn --arm zn --device cuda --host
    python outputs/mp_unified/zn.py read --name scr-zn --device cuda --host
    python outputs/mp_unified/zn.py compare --split L-musique --new outputs/screen/fits/scr-zn --out outputs/zn/compare-L-musique
    python outputs/mp_unified/zn.py summary --out outputs/zn/summary

## Results

Not yet run.
