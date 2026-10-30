# PKO A2: check (b) compares U2d's manifest at U2d's own definitions and rounding

This amends `outputs/mp_unified/zpko.py`, PKO's own file (docs/PKO_PICK_O_POOLS_UNDER_WQ.md, amended once before by
docs/PKO_A1_AMENDMENT.md), before any further pick-O number is built. It changes no number any build step computes: the pools
pass, the look and cache steps and the tail are untouched, and only `check_squad` and its self-test change. It is declared
with its code. No sha-locked module is touched. Nothing here is a result.

## 1. What was found, and how

On 30 Oct 2026 a rehearsal of PKOM's `run` on the smallest carve (squad/fit, workspace 3; docs/PKOM_PKO_ON_MODAL.md
section 8) built its two pools shards and the merge, then **check 5 (b) FAILED** (`check/squad__fit.json`, filed
verdict FAIL). Its record says the pool sizes and the sorted pool rows hash to U2d new's (`n_sha256` true, `node_sha256`
true), `fill_want` and `count` are zero (`k_q_zero` true), and four manifest fields differ: `today_inc_mean`, `pool_mean`,
`B_q` and `rows`. A FAIL of check (b) stops every carve's build (`need_checks`), so this would have stopped the whole run.

The cause is in the check, and was read, not guessed. The merged arrays were copied from the volume (sha256 01a3d8fb... as
the manifest names it) and the check's summary was recomputed locally against U2d new's real manifest entry
(`outputs/u2d/new/pools/squad/pools.json`, `squad__fit`). Three things differ in how the two sides are written:

1. **Rounding.** U2d files its means at one decimal (`zu2d_pools.carve_summary`, lines 578 to 583: `round(x, 1)`). PKO
   compared its exact means (292,994 / 5,856 = 50.0331) with U2d's 50.0 at a relative 1e-9. `today_inc_mean`, `pool_mean`
   and `rows.mean` could not match, whatever the pools.
2. **What `B_q` is.** U2d's `B_q` is `inc_size - bs_size`, the rows added to the base set (the fill), 0 for squad. PKO's
   check put the base set's own size (about 50) there. The doc's section 5 (b) says "the maximum and mean of |B|" and does not
   define it; U2d's definition is the one the comparison has to use.
3. **The toy fixture.** The self-test built its own "U2d manifest" from the same formulas the check used (exact floats,
   `B_q` = the base set), so it passed on a manifest no U2d run ever wrote. This is the failure that the self-test's toy
   manifest hid: a guard that passes a self-built toy and fails the first real record.

With the rule below, the real squad/fit arrays differ from U2d's real entry in no field; under A1's rule the same arrays
differ in exactly the four fields the container reported. Both runs are in the session record (the local script reads the
volume's copy; nothing is written back).

## 2. The rule

Check (b) is unchanged in what it requires (the pool sizes and the sorted pool rows equal U2d's records', `k_q` is 0,
the question set is U2d's) and in what the manifest comparison covers (questions, changed, short, `budget_mean`,
`fill_want_mean`, `today_inc_mean`, `graph_detail`, `pool_mean`, `B_q` and `rows` as max and mean, and the sum of the
parts' rows). The summary PKO computes for it is now written the way U2d writes its own:

- every mean is `round(x, 1)`, as U2d files it;
- `B_q` is the pool size minus the base-set size, which is `count`: `{"max": int(max), "mean": round(mean, 1)}`;
- `rows` and `pool_mean` are the pool size's max and rounded mean; `parts_rows` stays the exact sum.

Nothing else of the check moves: the question set, the record hashes and `k_q = 0` are compared exactly, as before, and
`close` still compares the rounded numbers to 1e-9. A pool that differs by one row changes the sha256 of the sizes or of the
nodes, and the check FAILs there, as before.

## 3. A FAIL filed under A1's rule

`check_squad` refuses to overwrite a check file (`exists`). The record of the first rehearsal (`verdict` FAIL, no `rule` key)
is kept. When `check_squad` finds a FAIL without `rule` "A2" at `check/squad__<carve>.json` it moves that file to
`check/squad__<carve>.a1-fail.json` (bytes unchanged, never overwritten; if that name exists the check stops) and runs
again, and every record it writes carries `"rule": "A2"`. A FAIL carrying `"rule": "A2"` is final, as before: it needs a
decision, not a rerun. The ledger's FAILED entry for `check-squad:<carve>` is run again with `--retry`, as every FAILED
entry is (PKOM's `run --retry`, docs/PKOM_PKO_ON_MODAL.md section 10).

## 4. What A2 does not touch

- The pools shards built before A2 stand. `ZPKO_SAME` gains A1's sha (5306512e...), so shards written by A0's, A1's and
  A2's file join in one merge, as A0's and A1's did; any other zpko.py sha, or any other script's, still makes two shards
  differ.
- The webqsp/s1eval shard on the laptop (A0's file) and the first two Modal squad shards (A1's file) are not rebuilt.
- Check (a) (webqsp and hotpotqa/s1sel against U2s's chain) is not changed. webqsp's already passes on real data inside the
  shard (`check` IDENTICAL, section 5 (a)); **hotpotqa/s1sel's has never run on real data**, and its self-test is also a
  toy. It is the next place this fault could be: it is named here, and nothing is assumed about it.
- Section 5 (b)'s requirements, the other checks, the picks, the types and every sha-locked module: unchanged.

## 5. Self-test

`squad_summary` is a function of the merged arrays and the manifest entry. The self-test gains:

- the summary of arrays with squad/fit's real totals (5,856 questions, 292,994 rows, maximum 54, no fill) against the numbers
  U2d filed for squad__fit (50.0, 50.0, B_q 0 and 0.0, rows 50.0 and 54), pasted as literals from the real manifest;
- the toy U2d manifest written as U2d writes one (rounded means, `B_q` the fill), with a pool mean that does not round to
  itself, so that A1's rule would fail it;
- a FAIL under "A2" is final; a FAIL without it is moved aside and the check runs again; a second move over an existing
  aside file stops.

## 6. After A2

Commit and push (zpko.py, this file, the input map for the new zpko.py sha), then PKOM's `upload` and `check` for the new
commit, then the squad/fit rehearsal again (`run --carves squad/fit --retry`), which continues from its ledger: the two
shards and the merge are filed OK and are not rebuilt. What the rehearsal does after the check (look, cache, the tail and
the prune of squad/fit) has not run yet anywhere; it is what it is for.
