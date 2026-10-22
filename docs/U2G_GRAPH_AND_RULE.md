# U2g: the graph and the pool rule, told apart

Declared 23 October 2026, after U2d's verdict (A, B, C and G NOT_ADOPTED, 103ab28) and before any U2g number. Code:
`outputs/mp_unified/zu2g.py`, and Y's stages added to `outputs/mp_unified/zu2d_modal.py` in a follow-up commit, as the
hit@1 diagnostic's were. Everything else is imported unchanged: zu2d_pools.py, look_u2d.py, zu2d.py, lean_cache,
zlink, zp1c3, zp1c3p and zg6. Numbers here are development numbers.

## 1. The question

U2d's decision A changed two things at once: the graph (today's to W2_S, the LLM-free links) and the pool rule
(P1c2's to U2e's pick). Both lean models lost on hotpotqa and 2wiki. U2d's section 9 named the next form: the 2 × 2.

| graph | P1c2's rule | U2e's rule |
| --- | --- | --- |
| today's | base (U2d, filed) | X |
| W2_S | Y | new (U2d, filed) |

- **Y against base** is the graph, at P1c2's rule.
- **new against Y** is the rule, on W2_S.
- **X against base** is the rule on today's graph, and **new against X** the graph at U2e's rule.

Y alone splits U2d's A along one path: new − base = (Y − base) + (new − Y). X gives the other path, which shows
whether the two effects interact.

**Why Y first.** It is the cheaper cell, and the credit decides:
- Y's pools follow P1c2's sizes: 1.3 times base's rows on hotpotqa and 1.5 times on 2wiki.
- X's pools follow U2e's: 5.9 times base's rows on hotpotqa and 8.0 times on 2wiki. Its features cost about what U2d's
  new build cost.
- On Modal, workspace 3 holds U2d's base build and has about $11 left. Y fits in it; X does not.
- X runs when credit or the host allows. Workspaces 6, 7, 8 and 10 need a payment method, which is the user's to add;
  workspace 9 is at its limit.

Y also tests the user's view of 21 October, "the links arent bad as well", at the rule the lean pair was tuned on.

## 2. Cell Y

**The rule.** P1c2's, unchanged: B, then KIND_T's order outside B, its first |I_q| − |B| nodes, where I_q is the
frozen pool on the build's graph. On W2_S this expansion is exactly what U2d calls the new build's `fill`
(zu2d_pools.question). So Y's pool is B ∪ fill, where U2d's new pool is B ∪ U2e's expansion.

**The graph.** W2_S, swapped in as U2d's new build swaps it: `zu2d_pools.open_carve("new")`, with U2c's graph_csr and
csr_store unchanged. The graph's description is checked against the one U2e's pick ran on, and the seeds against
today's.

**What is built, and what is taken as filed:**
- **Built on Y's own roots:** squad, musique, hotpotqa and 2wiki (`outputs/u2g/y` on the laptop, `/out/y` on mpr-u2d).
  - squad's pools are B only on both graphs (50 rows), but its walks read the graph. U2d's new looks swapped W2_S in on
    squad (their records say `swapped: true`). So squad is built too.
- **Taken from U2d's base build, as filed:** metaqa and webqsp. Their graph is the KB on both builds, and Y's rule is
  base's, so Y's cell on them is the base build. Y uses base's pools, looks, caches, chains, links, KT and gate there,
  under U2d's declaration.
- **The reads on metaqa and webqsp can still move**, since every fit trains on Y's squad, musique, hotpotqa and 2wiki
  carves.

**The code path.** zu2g.py registers a third build, `y`, in memory. It gives the build:
- its roots;
- its construction, `U2G:y:KIND_T`, and its rule, P1c2's;
- a carve opener: `zu2d_pools.open_carve("new")`, with the question's branch set to P1c2's rule.

Everything else is U2d's code, unchanged: the pools in shards, the merge, the looks, the bases, the caches, zlink's
links, the KT columns and the gate.

Y's records name U2g's declaration, build `y` and construction `U2G:y:KIND_T`. Their field `stage` reads "U2d", since
zu2d_pools.py writes it as a literal: it names the code, not the stage.

Y has no via columns (U2d's B was NOT_ADOPTED). It has no rmatch or zrc chains either: those are built only for the
typed datasets, metaqa and webqsp, which are base's.

## 3. The checks

Any failure stops the item.
- **Per question** (the pools): B is step 4d's; the seeds are the same on both graphs; the graph is the one U2e's pick
  ran on.
- **Per carve, after the merge** (`outputs/u2g/y/pools/<D>/same_as_new.json`). Y's manifest is checked against U2d's
  new manifest, which is committed. These must be equal:
  - questions and the carve ids;
  - |I_q| − |B| (mean and maximum), the mean frozen pool and the mean of today's frozen pool;
  - the entity and route shares;
  - ALL and recall of I, the caps of I, and the graph's description.

  Also, 12 × Y's mean expansion must equal new's mean `fill_want` within 0.05 (its rounding), because new's budget is
  12 × |fill|. These checks cover totals. The per-question equality holds by construction: one function computes both.
- **The looks:** look_u2d's own checks. The graph's description equals the manifest's, and questions, changed and
  short equal the manifest's.
- **The gate:** `zu2d_pools.gate` over Y's eight carves. Every row count must be as filed, every look must be on the
  filed pools, and the KT input must be on the caches. same_as_new must PASS on all four datasets.
- **Before any fit:** Y's gate must PASS (U2g's), and so must base's (U2d's). Both builds' caches must also hold the
  same step-1 bases: their `bases_copied.json` must name one source and the same files. Every fit reads carves of both
  builds, and each carve is normalised by the basis in its own cache root.

## 4. The fits

The models, card settings, splits and reads are U2d's (its section 5): variant p, config `2e-3:1e-4:0.1:8:2`,
hidden 128, threads 2 and seed 0, on L-musique and L-hotpotqa. Each fit is read with p@swa on the six s1eval carves.

| fits | arms | seed | count |
| --- | --- | --- | --- |
| `scr-u2g-y-<arm>[-hp]` | zrckp and zspk, then zg6 | 0 | 6 |

**Each carve comes from its dataset's build.** squad, musique, hotpotqa and 2wiki come from Y; metaqa and webqsp
from base.
- L-musique trains on metaqa (base) and on squad, hotpotqa and 2wiki (Y).
- L-hotpotqa trains on metaqa (base) and on squad, musique and 2wiki (Y).

The carve classes are U2d's arms'. Each is opened on its dataset's roots and checks the KT records against that
build's declaration. Each fit's record names, per dataset, the build, the cache root, the KT, chain and link records,
and both gates.

**One device.** U2d's: an L40S, from the image `u2d-gpu:py3.13:torch2.8.0+cu128:numpy2.3.2`, in workspace 3, where
U2d's base build and its 14 fits ran. So every decision compares fits from one device, one image and one workspace's
builds.

**The order.** zrckp's and zspk's four fits run first, and zg6's two last. If the workspace's limit stops zg6, YG is
INCOMPLETE until credit allows.

## 5. The null and the decisions

**The null** is U2d's own, filed: seeds 1 and 2 on the base build. zrckp's null calls zrckp; zspk's calls zspk and
zg6. The device, the workspace and the base fits are the same, so it applies unchanged.
- The pair base of zrckp and zspk is U2d's seed-0 base fit, the null's own.
- zg6's pair base is U2d's base zg6 fit. It is re-based through U2d's filed `scr-u2d-base-zg6-vs-zspk[-hp]`, as U2d
  re-based G.

GAIN, LOSS and WITHIN are docs/SCREENS.md's, as in U2d. A GAIN needs delta R@5 ≥ 0.0075 with the 2,000-resample
bootstrap's 95% interval above 0. The re-call turns a GAIN or LOSS below the read's floor into WITHIN.

**The decisions:**

| decision | what it asks | arms (Y's fit against base's) |
| --- | --- | --- |
| Y | W2_S at P1c2's rule, for the lean pair | zrckp; zspk |
| YG | the same, for zg6 | zg6 |

**The rule, fixed before any number.** Over a decision's arms and their twelve reads each:
- **ADOPT_FOR_FULL_RUN:** no LOSS on any read, and at least one GAIN (P1g-b's rule, as U2d's);
- **NO_LOSS:** no LOSS on any read, and no GAIN;
- **NOT_ADOPTED:** at least one LOSS;
- **INCOMPLETE:** a re-call is missing, is not U2g's, does not hold twelve reads, or compares fits from different
  devices.

**Why NO_LOSS is its own outcome.** U2d's rule asks a change to show a gain. Y changes only the graph, and the graph is
not chosen for the screen's gain. The headline wants the LLM-free graph: one LLM-free link rule for all six datasets
("a few edges suffice, no LLM graph", 9 October). A graph that costs the models nothing is enough for that.

| Y | what follows |
| --- | --- |
| ADOPT_FOR_FULL_RUN or NO_LOSS | the lean pair's full run on Y's build (W2_S, P1c2's rule), declared in its own file: six splits per arm, under grade-nullx |
| NOT_ADOPTED | no full run: the lean pair stays on today's graph |
| INCOMPLETE | nothing until the missing re-call is run |

YG decides only zg6's graph, the same way. G6's own verdict (zg6 against zspk) stays with its stage.

## 6. Reported beside, deciding nothing

- **The rule on W2_S:** U2d's new fits against Y's (`scr-u2g-rule-<arm>[-hp]`), with the calls under U2d's null as
  readings. The two fits come from different workspaces: U2d's new build ran in workspaces 4 and 5. U2d's section 11
  found float32 rounding differences between two copies of one build (score2, at most 2.3 × 10⁻⁵) and in zrc's chains
  on webqsp. So these are readings, not decisions.
- zg6 against zspk on Y;
- each carve's ALL, recall, mean rows and caps on base, Y and new;
- zrckp's share of the GNN's gain over RRF on Y;
- FC@5 and hit@1 for every read, and hit@1 as the headline on metaqa and webqsp.

## 7. What follows (fixed now)

Each verdict closes only the form it tests (docs/SCREENS.md, 3b).
- **Y NOT_ADOPTED, losing on hotpotqa or 2wiki** (where today's graph holds the datasets' hyperlinks): the graph costs
  at P1c2's rule too. The next form is the links: a link rule chosen by the lean pair's R@5 on this screen, not by
  coverage alone (U2b chose W2_S by the coverage of B ∪ X(k)). Then X.
- **Y NOT_ADOPTED, losing elsewhere:** X, then the links as above.
- **Y ADOPT_FOR_FULL_RUN or NO_LOSS:** the full run, then X. If new − Y also loses, the rule (U2e's larger pools) is
  what lost in U2d's A, and the next form is the stop: U2f's learned cut, after the pools are final.
- **YG NOT_ADOPTED:** zg6 stays on today's graph. G's declared next form (zg6v) stands.
- In every case, X runs when credit or the host allows.

## 8. Cell X (declared now, built later)

- **The rule:** U2e's pick, unchanged: zu2d_pools.new_expansion over kt_cos from zu2e.question_orders, on today's
  graph (no swap).
- **Built:** musique, hotpotqa and 2wiki.
  - X's squad is base's (today's graph, B only).
  - X's metaqa and webqsp are U2d's new build's (the KB, U2e's rule). They sit in workspaces 4 and 5, so X's fits need
    a copy checked by sha256, as U2d's workspace copies were, or a rebuild.
- **Decided:** X against base, by section 5's rule. X changes the rule, not the graph, so NO_LOSS sends nothing to a
  full run there. Beside: new against X, the graph at U2e's rule.
- **The code** comes in a follow-up commit before any X number: a carve opener on today's stores with U2e's question
  branch, and a graph for look_u2d that is today's stores.

## 9. How it runs

On Modal, in workspace 3, from zu2d_modal.py's images and its volume mpr-u2d. That volume holds U2d's base build at
`/out/base` and the fits at `/out/fits`. Y's stages:
- `--stage y-pools`: Y's pools in shards, then each dataset's merge and its same_as_new check. The shard counts are
  U2d base's: squad 2 + 3, musique 2 + 1, hotpotqa 9 + 11 and 2wiki 10 + 21. The volume holds no copy of U2d's new
  build, so U2d's new manifests (committed) reach Modal inside the uploaded code archive, and same_as_new reads them
  there, by their sha256.
- `--stage y-features`: the bases, then each carve's looks, caches, links and KT, then the gate.
- `--stage y-fits [--fits N,...]`: each fit trains and reads in one L40S container.
- `--stage fetch-y` and `--stage fetch-y-fits`: the records and the reads come to the laptop.

Then, on the laptop: `python outputs/mp_unified/zu2g.py compare-all`, then `decide`.

**The cost**, from U2d's bills and part seconds:
- Y's pools: about $1.5 to $3;
- its features: about $3 to $4;
- the four lean fits: about $2.2;
- zg6's two fits: about $1.2 to $2.

That is about $7 to $10 in all, against about $11 left in workspace 3.

## 10. Results

Filed 23 October 2026. Development numbers. The files:
- the verdict: `outputs/u2g/screen-verdict.json` and `.md`;
- the 14 compares, the three pairs and their re-calls: `outputs/screen/scr-u2g-*`;
- Y's build records: `outputs/u2g/y`.

The fits' own records and reads (`outputs/screen/fits/scr-u2g-y-*`) stay off the repository, as U2d's do.

**The run.** All of Y ran on Modal in workspace 3, from 02:57 to 04:27 IST on 23 October, for about $7 (section 9
estimated $7 to $10).
- **The pools:** 59 shards in 715 s. same_as_new passed on all four datasets: Y's manifests equal U2d's new ones, and
  twelve times Y's mean fill_k equals new's fill_want_mean to the record's rounding (within 0.046).
- **The features:** 1,415 s. The gate passed.
- **The fits:** all six trained and read in one L40S container each, from U2d's image. So every compare's two fits
  share the device.

**The verdicts.**

| decision | what it asks | GAIN | LOSS | verdict |
| --- | --- | --- | --- | --- |
| Y | W2_S in place of today's graph, at P1c2's rule, for the lean pair | 2 | 8 | NOT_ADOPTED |
| YG | the same, for zg6 | 1 | 4 | NOT_ADOPTED |

So there is no full run (section 7). The lean pair and zg6 stay on today's graph.

**Y: the graph.** Delta R@5, Y against the base, with the re-called calls:

| arm | split | metaqa | squad | musique | hotpotqa | 2wiki | webqsp |
| --- | --- | --- | --- | --- | --- | --- | --- |
| zrckp | L-musique | 0.000 | +0.001 | +0.016 GAIN | −0.054 LOSS | −0.017 LOSS | −0.004 |
| zrckp | L-hotpotqa | +0.001 | +0.002 | +0.007 | −0.048 LOSS | −0.009 LOSS | 0.000 |
| zspk | L-musique | 0.000 | −0.001 | +0.014 GAIN | −0.069 LOSS | −0.027 LOSS | −0.015 LOSS |
| zspk | L-hotpotqa | +0.001 | +0.003 | +0.001 | −0.055 LOSS | −0.008 | −0.010 |

- **hotpotqa loses on every arm and split,** by 0.048 to 0.069.
- **2wiki loses on three of four,** by 0.009 to 0.027. The fourth, zspk on L-hotpotqa (−0.008), is within its floor
  (0.0139) after the re-call.
- **webqsp:** zspk loses on L-musique (−0.015, floor 0.0105). On L-hotpotqa its −0.010 is within the floor (0.0405).
- **musique gains on L-musique** for both arms (+0.016 and +0.014). On L-hotpotqa, +0.007 and +0.001 are within.
- **metaqa and squad** stay within.

**Where hotpotqa's loss sits.** At P1c2's rule, W2_S's pools hold fewer golds in more rows. On s1eval:

| carve | rows, base / Y | I's ALL, base / Y | pool ALL, base / Y | R@5 cap, base / Y |
| --- | --- | --- | --- | --- |
| hotpotqa | 93.7 / 118.8 | 0.968 / 0.888 | 0.959 / 0.911 | 0.976 / 0.952 |
| 2wiki | 105.7 / 163.5 | 0.906 / 0.852 | 0.959 / 0.941 | 0.984 / 0.979 |
| musique | 2,092.4 / 2,060.7 | 0.803 / 0.809 | 0.900 / 0.901 | 0.963 / 0.964 |

squad's pools are the same 50 rows on both builds.
- **Reach explains under half of it.** hotpotqa's R@5 cap falls by 0.024, so the lost golds explain at most 0.024 of
  zrckp's 0.054 on L-musique. The rest, at least 0.030, is ranking: zrckp reaches 0.928 of its cap on the base build
  and 0.894 on Y.
- **The second gold is what goes.** On hotpotqa, FC@5 falls by 0.094 to 0.137 while hit@1 moves by −0.013 to +0.007.
  The first gold still comes first about as often, and the second drops out of the top five. That is the bridge,
  which W2_S's links reach less often than the dataset's hyperlinks do (I's ALL, above).
- **musique's gain is ranking.** Its pools and caps are the same on both graphs.

**U2d's A, split along Y.** For every read, (Y − base) + (new − Y) equals U2d's new − base to 0.0001. new − Y crosses
workspaces, so it is a reading. For the lean pair:

| dataset | the graph: Y − base | the rule: new − Y |
| --- | --- | --- |
| hotpotqa | −0.069 to −0.048 | −0.003 to +0.009 |
| 2wiki | −0.027 to −0.008 | −0.019 to +0.012 |
| musique | +0.001 to +0.016 | −0.013 to −0.003 |
| webqsp | −0.015 to 0.000 | +0.021 to +0.050 |
| metaqa | 0.000 to +0.001 | −0.005 to −0.001 |
| squad | −0.001 to +0.003 | −0.002 to +0.003 |

- **The graph** carries all of hotpotqa's loss in A, and musique's gain.
- **The rule,** U2e's larger pools on W2_S, carries A's webqsp gain. On 2wiki it helps on L-musique (+0.003, +0.012)
  and costs on L-hotpotqa (−0.019, −0.012). So 2wiki's loss in A is the graph's on L-musique and mostly the rule's on
  L-hotpotqa. The rule also gives back most of zrckp's musique gain on L-musique (−0.013 of +0.016).
- Read under U2d's null, the rule has five GAINs (webqsp three times, and zspk's hotpotqa and 2wiki on L-musique) and
  one LOSS (zrckp's 2wiki on L-hotpotqa).

**YG: zg6.** zg6 follows the lean pair:
- it gains musique on L-musique (+0.021);
- it loses hotpotqa (−0.055 and −0.046), and on L-musique 2wiki (−0.015) and webqsp (−0.041);
- on L-hotpotqa its 2wiki (−0.012) is within its floor (0.0139) after the re-call, and webqsp holds (0.000).

The rule, new − Y, costs zg6 webqsp on both splits (−0.040, −0.041), a LOSS on each read.

**Beside, deciding nothing.**
- **The full table** of R@5, FC@5 and hit@1 per fit and build is in `screen-verdict.md`.
- **zg6 against zspk on Y:**
  - above on musique (+0.013, +0.018), and on L-musique on metaqa (+0.009), hotpotqa (+0.018) and 2wiki (+0.020);
  - below on webqsp (−0.105, −0.048) and on hotpotqa on L-hotpotqa (−0.012).

  That is the pattern U2d filed on its new build, the other build on W2_S.
- **The MLP's share** of the GNN's gain over RRF on Y:
  - against zspk, 0.96 to 1.08, except squad on L-musique (1.39, where zspk's gain over RRF is 0.006) and webqsp (1.21,
    1.32);
  - against zg6, 0.83 to 1.24, except webqsp (2.10, 1.71), where zg6 is below the MLP. musique is the low end (0.85,
    0.83).
- **hit@1, the headline on metaqa and webqsp** (Y reads the base build's carves on both):
  - metaqa: 0.915 to 0.954, against a cap of 0.998;
  - webqsp: 0.133 to 0.179, against a cap of 0.967.
- **U2d's new build, beside Y:** at U2e's rule, W2_S's pools cover about as many golds as today's graph on hotpotqa,
  and more on 2wiki, at 5.9 and 8.0 times the rows. On s1eval: hotpotqa 555.7 rows (ALL 0.953, cap 0.976), 2wiki
  848.3 (0.967, 0.989), musique 5,217.2 (0.945, 0.982).

**What follows** (section 7, fixed before any number):
- **The links:** a link rule chosen by the lean pair's R@5 on this screen, not by coverage alone. Then X (section 8),
  when credit or the host allows.
- **zg6** stays on today's graph. G's declared next form, zg6v, stands.

Y gains musique and loses hotpotqa and 2wiki. docs/SCREENS.md (3b) calls for a form that can back off, per question or
per dataset. That form is declared in its own file, docs/U2H_W2S_BESIDE.md: W2_S's KT columns beside today's graph, as
an input to both lean models.
