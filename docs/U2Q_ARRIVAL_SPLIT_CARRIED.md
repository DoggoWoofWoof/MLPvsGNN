# U2q: the mass split by arrival type, on the carried walk

Declared 25 October 2026, before U2p's verdict and before any U2q code or number. U2m's (a) filing says (d)'s arrays
are computed again on the carried walk for U2n, in their own file ([U2M_POOLS_THREE_FAMILIES.md](U2M_POOLS_THREE_FAMILIES.md),
section 18). This is that file. Code: `outputs/mp_unified/zu2q.py`, which imports zu2m.py, zu2m_b.py, zu2m_d.py,
zu2m_a.py, zu2m_a_types.py and zu2p.py unchanged. Outputs go only under `outputs/u2q/`. Nothing trains and nothing is
fit. Only the beside table reads golds. U2q has no verdict: its arrays are an input for U2n's models, graded there,
never a pool rule.

## 1. What it computes

U2m (d)'s arrays (section 13), in the same form: for the carried pick's rows on every question of the carve, and for
each of the five walks `kind_t` reads (A1st, A1all, Kst, KTst and KTall), a row's mass p(v) split into its arrival
types (restart, structural, ner, knn), and on metaqa and webqsp the structural part split further by (relation,
direction). Only the walk changes: (d) split the walks of U2e's pick at β = 0; U2q splits the carried walk's.

## 2. The carried walk

It is fixed by U2p's filed verdict (`outputs/u2p/verdict.json`), whatever it says, and U2q's stats refuse to run
before it is filed:
- **U2p RAISED or CHEAPER:** U2p's pick, its point and its β. On the passage graphs each structural entry weighs
  ω_a,q(EM type) × ω(score type) × s(v), U2p's pair weight (U2p, section 2).
- **U2p NO_GAIN:** (a)'s pick, `own/kind_t/kt_cos:ada_f12_k0@0.85` at G = 0 with RWe at β = 0.5. This is U2p's weight at
  β = 0, which U2p's self-test and stats show to be (a)'s walk bit for bit.
- **The KBs** take (a)'s RWe at β = 0.5 in both cases, on the pick's point, since U2p does not walk them.

One code path serves both cases: U2p's pair weight at the β its verdict carries (0 when NO_GAIN) on the passage
graphs, and (a)'s weight on the KBs.

## 3. The kernel

The split kernel is (a)'s relational kernel (U2m, section 14) with (d)'s bookkeeping (section 13):
- the same statements on p and r, in the same order, as (a)'s push: within a family, u's mass goes to v in proportion
  to w = ω(entry) × s(v) (ω = 1 on ner and knn);
- an arrival at a pool row adds base × w to its type's pending part, and on the KB's structural family to its
  (relation, direction) entries for u in v's typed row, split equally; a push of the row settles α × each pending part.

So p and every order are (a)'s kernel's, and with every ω = 1 the kernel is (d)'s split kernel.

## 4. The rows

The rows of the carried pick on every question: B and order[:keep] (G = 0, so nothing is restored), with the order
rebuilt on the carried walk as U2p's stats rebuild it.

## 5. The arrays

As (d)'s: the rows with a question pointer; per row and walk the four parts and p (float32); each walk's total over
its top NMAX nodes; on the KB, sparse (row, walk, relation, direction, mass) entries. They go to
`outputs/u2q/stats/<D>__<carve>.npz` and `.json`, and stay uncommitted.

## 6. Checks

Any failure stops the dataset, and nothing is written.
- **The walk.** U2p's and (a)'s verdicts are the committed scripts' filed ones; U2p's score types and (a)'s EM types
  are the files their stats read.
- **Orders and masses.** For every question and walk, the split kernel's top-NMAX order and raw masses equal (a)'s
  kernel's (`zu2m_a.rwalk`) with the same weights, bit for bit.
- **Parts.** In float64, |Σ parts − p(v)| ≤ 1e-9 × p(v) + 1e-15 on every row and walk; on the st walks the ner and knn
  parts are 0; on the KB the (relation, direction) parts meet the same bound against the structural part.
- **Rows.** Each question's row count and held golds equal the carried pick's filed ones: U2p's arrays at its pick on
  the passage graphs, and (a)'s at RWe@0.5 on the pick's point on the KBs.
- **Self-test** (`--selftest`), before the commit that adds the code. On toy graphs, typed and untyped, steered and
  unsteered: with random ω the split kernel's order and masses equal `zu2m_a.rwalk` bit for bit, and its parts equal
  a per-arrival log settled push by push in plain Python within 1e-12; with every ω = 1 its p, parts and relation
  slots equal (d)'s split kernel bit for bit.
- **Development runs** (`--limit`) write nothing and log no number. 30-question runs on metaqa and musique pass before
  the code's commit. Before U2p's verdict is filed, a development run may read (a)'s pick (`--dev-a`, U2p's β = 0);
  once it is filed, `--dev-a` is refused.

## 7. Beside, deciding nothing

Per dataset and walk, (d)'s table on the carried walk: the mean share of each type in a row's mass over gold rows and
over the other rows, and on the KB the five (relation, direction) pairs with the largest share of the gold rows'
structural mass. Beside it, (d)'s filed shares, so the change the carried walk makes is visible. webqsp's s1eval is
beside, as everywhere.

## 8. Where it runs

The laptop, one job at a time, after U2p's verdict: squad, musique, hotpotqa, metaqa, webqsp, then 2wiki ((d) took
428 s, 1,066 s, 1,747 s, 1,283 s, 1,489 s and 4,313 s; U2q walks each question twice, once with (a)'s kernel for the
orders and once split, so about twice that). The lab host takes them if it is reachable first. `--shard I/K` and
`merge D --parts K` split a dataset into question ranges and join them into the file one run writes.

U2n computes the same arrays on its fit and evaluation carves with this kernel, since the kernel is the form. If a
later pool verdict changes the carried walk, the arrays are computed again on it, in that verdict's follow-up file.
