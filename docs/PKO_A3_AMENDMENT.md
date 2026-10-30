# PKO A3: a carve with one look shard is looked at with `--shard 0/1`, which the committed look scripts refuse

This amends `outputs/mp_unified/zpko.py`, PKO's own file (docs/PKO_PICK_O_POOLS_UNDER_WQ.md, amended before by
docs/PKO_A1_AMENDMENT.md and docs/PKO_A2_AMENDMENT.md), before any look is built. It changes no number any build step
computes: the pools pass, check (b), the cache step and the tail are untouched, and only `look_shard` and its self-test
change. It is declared with its code. No committed look script is edited. Nothing here is a result.

## 1. What was found, and how

On 30 Oct 2026 (17:39 to 17:45, workspace 3, 0.03 USD) `run --carves squad/fit --lanes 4 --retry` ran PKOM's chain on
the smallest carve with all four section-5 checks filed PASS. It is the first time a look step ran anywhere. The look
`look:squad/fit:0/1` **failed after 222 s (peak 4.68 GiB) with rc 1**, and the three steps after it were dropped
(cache, tail, the carve's prune; the gate: "a carve's tail is not done"). The last line of the step's log
(`/pko/logs/look_squad_fit_0_1.log` on the volume) is:

    --shard 0/1: needs 0 <= i < n and n >= 2

The cause was read in the code, not guessed. `zpko.look_shard` calls `look_u2d.main` with `--shard i/nl`, and a carve gets
`nl = min(chunks, ceil(rows / LOOK_ROWS))` look shards (`look_parts`). `look_u2d.py` line 215 and `look_x_six.py` line 262
both parse the text with `mp_approx_l8.parse_shard` (scripts/mp_approx_l8.py lines 230 to 239), which raises for any
`n < 2`. The same text, from the same module, was reproduced on the real modules before this amendment was written:

| text | `parse_shard` (committed) |
|---|---|
| `0/1` | SystemExit: needs 0 <= i < n and n >= 2 |
| `1/1`, `2/2` | SystemExit (same) |
| `0/2` | (0, 2) |

The parse is reached inside `look_x_six.main`, after the stats were loaded (the log shows squad's 5,856 ids at 12:14:44 UTC
and U2d's pools on W2_S at 12:15:04) and before `shard_chunks` and any chunk, so the look built nothing; the 222 s
(container clock is UTC, 12:11:22 to 12:15:04) were the setup before the parse.

**Which carves.** PKO's carve table (docs/PKO_PICK_O_POOLS_UNDER_WQ.md section 6) puts nine of the eleven carves above
1,000,000 rows (webqsp/fit 5.77M, hotpotqa/fit 4.23M, 2wiki/fit 6.35M, musique/s1fit 25.4M, metaqa/s1eval 42.8M,
musique/s1eval 13.3M, hotpotqa/s1eval 5.28M, 2wiki/s1eval 13.5M, webqsp/s1eval 7.22M), so they have two or more look shards
and never meet the limit. **squad/fit (293,027 rows) and squad/s1eval (594,110 rows) have one look shard each, and would
have failed the same way on 1 Nov.** A failed look stops its carve's cache and tail, and the gate needs every carve's tail,
so the package could not have been filed.

**Why it was not found sooner.** The self-test builds toy carves and calls `post_steps` for the step list, never the
look scripts' parser; the look step had run nowhere. This is the same shape as A2's finding: a guard that was never
shown a real record. This document is written after the failure and after reading the data, and says so.

## 2. The rule

`look_shard` patches `mp_approx_l8.parse_shard` for the duration of the look (the same in-process patch it already makes
for `load_filed`, `roots` and `DECLARED`), with `accept_one_shard(parse)`:

- the text `0/1` returns `(0, 1)`;
- any other text goes to the committed parser, so `1/1`, `2/2`, `0/0` and every `n >= 2` behave as before.

Nothing else changes, and the committed scripts are not edited. The consequences of `(0, 1)` were read in the code:
`shard_chunks(n, (0, 1))` selects `ci % 1 == 0`, which is every chunk, the same chunks a look without `--shard` takes;
`look_x_six.py` line 426 writes the tag `_0of1`, so the files are `record_0of1.json` and `pools_0of1.json`, the names
`look_outs` already expects; the first-shard branch (`shard[0] == 0`, line 428) writes the shared files a no-shard look
writes. The record carries `"shard": [0, 1]`. `lean_cache.py` already accepts `--part 0/1` (it is the default), so the
cache step needs no change.

`ZPKO_SAME` gains A2's file sha (`6af776be...`, commit 893ea05), so shards the A2 file built (squad/fit's two pools shards)
and shards this file builds join in one merge. `outputs/pkoh/inputs.json` is regenerated with `zpkoh.py inputs`; only
zpko.py's entry and the time change. `zpkoh.py` is not edited.

## 3. Self-test

One check is added, and it uses the real modules, not a toy: it imports `look_u2d`, applies `accept_one_shard` to its
`LX.L8.parse_shard`, and asserts the table above with the patch (`0/1` gives `(0, 1)`; `1/1`, `2/2` refused; `0/2` gives
`(0, 2)`), that the committed parser still refuses `0/1` without it (so a later change of the committed script is seen),
that `look_x_six.L8` is the patched module object, and that `look_outs` names `record_0of1.json` and `pools_0of1.json`.
The zpko self-test goes from 20 to 21 checks.

## 4. What follows

1. A commit, a new Modal gate for it (a code commit), and `run --carves squad/fit --lanes 4 --retry --code <commit>`,
   the same rehearsal as 30 Oct 17:39. The ledger holds `look:squad/fit:0/1` as FAILED for the reason in section 1, so
   `--retry` is what runs it again. Cost: a gate build (0.04 to 0.05 USD) and the rehearsal (0.03 USD for the part that
   ran; the look, cache, tail and prune are not priced). Month to date 29.547 USD of an assumed 30; the stop line stays
   29.9.
2. What the rehearsal can still show: the look, cache, tail and prune steps have not run anywhere, so another fault
   after the look is possible and would be filed the same way. A pass would show that the single-shard look builds
   and that the chain files squad/fit's tail. It would not show squad/s1eval's (same path, not run) or anything about
   the other nine carves' looks, which are larger and use shards `n >= 2`.
3. The first full run on 1 Nov waits on nothing else from this amendment.
