# P1g-c's look alias: P1g-b's rebuilt looks checked against its caches

Written 18 October 2026 and committed 19 October, before any verification ran. It amends how P1g-c's builds read P1g-b's rebuilt looks
(docs/P1GC_ADOPTED_ON_P1GB.md, section 3); nothing in P1g-c's fits, calls or rules changes. Code:
`outputs/mp_unified/zp1gcla.py`.

## 1. What was found

P1g-c rebuilds P1g-b's looks with P1g-b's own commands (177 shards), then its links, chains and G3's residuals from
them. Each look shard writes a record, and the record holds its run's `seconds`, `utc` and `peak_rss_bytes`. So a
rebuilt record's sha256 is not the one P1g-b's cache parts stamped (`look.records` in every part's record).

Five builders read a look beside a cache, and each stops when the two sha256s differ ("built from another look"):

| builder | stops at |
| --- | --- |
| zlink | `zlink.build` |
| rmatch | `rmatch.build` |
| zrc | `zrc.build` |
| zg3res | `zg3res.build` |
| A7's scorer builds | `scripts/a7_s6_cols.py`, `sbuild` and `cols` |

Every one of P1g-c's link, chain and residual builds would therefore have stopped. P1g-c's gate would have stayed shut,
and none of its fits would have run.

**They did stop.** Between this file's writing and its commit, the host finished all 177 look shards. It then sent
P1g-c's 11 link builds and 3 rmatch builds (18 October, 15:36 to 16:01 IST). Each stopped within 8 s (rc 1) at the
first cache part:

    outputs/p1gb/cache/2wiki/fit/part_0of15: built from another look than outputs/p1gb/look/2wiki/fit

Nothing was written. P1g-c's 3 zrc builds, its gate and its 140 G3 builds had not been sent: they wait on those 14.

**A probe of the host at 02:33 IST** compared each look record on disk with the cache's stamp:

| carve | rebuilt so far | still the stamped record |
| --- | ---: | ---: |
| 2wiki fit | 15 of 15 | 0 |
| 2wiki s1eval | 23 of 29 | 6 |
| hotpotqa fit | 7 of 12 | 5 |
| metaqa fit | 1 of 14 | 13 |
| the other seven carves | 0 | all |

**At 02:36 a one-off host job copied every record still equal to its stamp** to `outputs/p1gc/p1gb_look_records`,
with a list `saved.json`:
- 129 records saved, each checked against its stamped sha256 as it was copied;
- 48 records were already overwritten (the list names them as lost): 2wiki fit 15, 2wiki s1eval 25, hotpotqa fit 7,
  metaqa fit 1;
- no other copy exists: neither the archive index nor the laptop holds P1g-b's look records.

**Which fields differ.** For hotpotqa fit, a saved original against a rebuilt record of the same carve: every field is
equal except `seconds`, `utc` and `peak_rss_bytes`. The records hold no per-chunk sha256, so the record alone cannot
show that the rebuilt chunks are P1g-b's.

**What the saved originals agree on.** All 129 hold the same value of every carve-independent field:
- the look's name, pair and its model keys, functions and metrics;
- `--full` and the projection (dim 128, seed 20261001);
- the threads (torch 4, BLAS 2), no limit;
- the source copied, its sha256 and look_x_six's sha256;
- the placement.

Within each carve they agree on every carve field. Every record on disk, rebuilt or not, holds those same global values.

## 2. The check: the cache built again from the rebuilt look

**The evidence the stamp stood for.** A cache part is built from its look's chunks. So the right test is whether the
rebuilt look gives P1g-b's cache: each part built again from the rebuilt look must equal P1g-b's stored part, array
for array.

**`verify`, one item per P1g-b cache part (140):**
- **the build:** `lean_cache.build_part`, unchanged, on the rebuilt look, with:
  - the part's own BLAS threads (its record's `blas_threads`, set before numpy loads);
  - numba off, as P1g-b's parts were (it refuses otherwise);
  - the same two bases, each checked against the part's record;
  - no Carve3 check (the comparison below is stronger);
- **no array is written:** `save_npy` hashes each array in memory, the bytes `np.save` would write;
- **compared with P1g-b's stored part record:**
  - every array's sha256, dtype and shape (17 arrays a part);
  - the part's ids, question and row ranges, chunks, counts, column spans, bases and odd code rows;
  - the look head's carve fields and record names;
- **verdict:** IDENTICAL when all of these are the same, DIFFERENT otherwise;
- **the record:** `outputs/p1gc/look_alias/parts/<dataset>/<carve>/<part>.json`, holding the look records on disk at
  the time.

**`alias`, one item per carve (11).** ALIAS when all of these hold, REFUSED otherwise:
- every part of the carve was verified IDENTICAL, against the look records on disk now, and no stored part record
  changed since;
- **each saved original:**
  - its copy's sha256 is the stamped one, and the list names it;
  - its rebuild equals it in every field but `seconds`, `utc` and `peak_rss_bytes`;
- **each lost original:**
  - its rebuild holds the saved originals' common value of every carve-independent field;
  - it holds its carve's single value of every carve field;
- the saved originals agree among themselves on every carve-independent field.

The record (`outputs/p1gc/look_alias/<dataset>/<carve>.json`) maps each look record's name to its rebuilt and its
stamped sha256, and says whether its original was saved or lost.

**What reading under the alias means.** Inside `zp1gcla.py run`, `lean_cache.look_chunks` on P1g-b's look root returns
the stamped sha256s in place of the rebuilt ones. It does so only when:
- the carve's alias is ALIAS;
- the alias's rebuilt sha256s are the records on disk at that moment.

Any other root, a record changed after its alias, a missing alias or a REFUSED one: the head is the look's own, and
the builder stops as before. Nothing else in any builder changes. The builders' outputs say where they came from:
- each `run` logs the alias it read;
- the alias records name both sha256s;
- P1g-c's gate still compares every rebuilt link and chain record's arrays with P1g-b's (its section 3), which checks
  the look's edges and relations as well.

**The hook.** `run` imports nothing heavy before the builder. lean_cache is patched by an import hook as the builder
imports it, so the builder's own thread settings stand: zp1gc's one thread on its builds, and pylib_run's numba
loaded first for G3's.

## 3. The items

**New, after the carve's last look shard:**

| items | count | sizes |
| --- | ---: | --- |
| `p1gc-lav-<dataset>-<carve>-<part>` | 140 | each after every look shard of its carve; 1.5 CPUs; memory the part's own measured peak + 1.5 GB (at most 9.4) |
| `p1gc-la-<dataset>-<carve>` | 11 | each after its carve's verifications; 1 CPU, 1 GB |

**Sent again under new names.** The feeder does not resend a failed item, so the 14 builds that stopped (section 1)
run again as:
- `p1gc-lkb2-<dataset>-<carve>` (11);
- `p1gc-rmb2-<dataset>-<carve>` (3).

**Changed (not yet sent):**
- **P1g-c's 3 zrc builds:** each runs through `zp1gcla.py run --`, waits for its carve's alias and for `p1gc-rmb2`
  in place of the failed `p1gc-rmb`;
- **P1g-c's gate:** waits for the `lkb2` and `rmb2` items in place of the failed ones;
- **P1g-c's 140 G3 builds:** run through `zp1gcla.py run --` behind pylib_run, and wait for their carve's alias;
- their commands are otherwise P1g-b's.
- **R1's column builds** read no look and are unchanged.

**The sizes:**
- each `lkb2` and `rmb2` item has its failed item's sizes and its carve's alias as a dependency;
- every item that reads a look under the alias lists `zp1gcla.py` among its inputs.

**Cost.** The verifications repeat P1g-b's cache builds: 28,047 s in all, at most 1,341 s a part (musique). They write
no array; the records are text.

## 4. What each outcome leads to (fixed now)

| alias | what follows |
| --- | --- |
| ALIAS on all 11 carves | P1g-c runs as declared: the builds read the rebuilt looks under the stamped sha256s, then its gate decides as before |
| REFUSED on a carve | that carve's builds stop, and P1g-c's gate stays shut. The next form is **P1g-b2**: P1g-b's caches built again from the rebuilt looks into `outputs/p1gb2/cache`, P1g-b's lean fits refitted on them, and P1g-c's calls made on those roots, its own declared file. An alias refused for a record field alone, with every part IDENTICAL, is reported as such in that file |

**A7p** (A7's next form) reads P1g-b's looks beside its caches. It builds under the same alias, in the same carves.

## 5. Self-test

`python outputs/mp_unified/zp1gcla.py --selftest` passes before this commit (17.8 s). It runs on A7's toy looks (a KB
name and a passage name, two shard records each) and their cache parts.

**The rebuild's effect:** after a rebuild (new seconds, time and memory), A7's check stops.

**verify:**
- with the build stood in for, so the hashing is the one tested: the in-memory sha256 of an array equals the stored
  file's;
- no array is written, and its temporary folder is removed;
- a changed array is DIFFERENT (`arrays.n`);
- it refuses other BLAS threads, numba, another basis, or a part out of range.

**alias:**
- REFUSED with a DIFFERENT part;
- ALIAS when every part is IDENTICAL (two saved originals, or one saved and one lost);
- REFUSED when:
  - a saved original differs outside the run keys;
  - a lost one holds another global value, or another carve value;
  - a part was verified against older records, or a part is not verified;
  - a saved copy is not the stamped record.

**The alias in a process:**
- under it, A7's check passes; after it, the function is restored;
- another root, a record changed after its alias, and a REFUSED alias are each read as they are;
- the import hook patches lean_cache in a fresh process that loaded no numpy before it;
- `run` takes only zp1gb.py, zp1gc.py and a7_s6_cols.py.

## 6. The rebuilt looks deleted, and the redo (19 October, before any verification)

**What happened.** At 22:28 IST on 18 October the host's clean-up (`outputs/host_ops/auto_cleanup.py`, emergency
tier at 34.8 GB free) removed the chunks of all 177 rebuilt look shards (54 GB) and P1g-c's rebuilt KT columns. The
records, which are text, stayed. The 14 builds of section 1 had failed and the feeder had dropped the ~300 items behind
them, so no pending item named `outputs/p1gb/` any more, and nothing pinned it. On 19 October a first verification
stopped at once ("13 of 13 chunks are not listed or not on disk"). No part was verified and no alias was made, so
nothing in sections 2 to 4 changes.

The clean-up now keeps a failed item's folders, and those of the items dropped behind it, for 48 h after its launch
(301905a).

**The redo.** Every item keeps its section 3 role. Nothing in any builder, fit, call or rule changes.

| items | count | after | what |
| --- | ---: | --- | --- |
| `p1gc-look2-<dataset>-<carve>-<shard>` | 177 | as P1g-c's look items | the look shards again, with P1g-c's own look commands and sizes. A shard rebuilds the chunks that are missing and rewrites its record |
| `p1gc-kt2-<dataset>-<carve>` | 11 | as P1g-c's KT items | P1g-b's KT columns again, with P1g-c's own KT commands (they read no look) |
| `p1gc-lav-...`, `p1gc-la-...` | 140, 11 | `lav` after every `look2` shard of its carve | section 3's verifications and aliases |
| `p1gc-lkb2-...`, `p1gc-rmb2-...` | 11, 3 | their carve's alias | section 3 |
| zrc builds, gate, G3 builds | 3, 1, 140 | section 3, with `lkb2`, `rmb2` and `kt2` in place of the failed or deleted items | section 3 |
| `p1gc-release-look-<dataset>` | 6 | every verification, alias, link, chain, zrc and G3 build of its dataset | `auto_cleanup.py release` (a4e0cd2) gives up the clean-up's family and source pins on `outputs/p1gb/look/<dataset>` |

A7p's builds read P1g-b's looks. Once declared, they are added to the release items' dependencies.

**The disk.** The redo needs about 88 GB at once, until the G3 builds end:

| part | GB |
| --- | ---: |
| looks | 54 |
| KT | 3 |
| links | 15 |
| chains | 11 |
| residuals | 5.5 |

The drive had 98.6 GB free. Two releases came before the redo: L12's carves and units with L16's design looks, and
the closed tracks' carves of `outputs/mp_unified/look` (fit, select, x1, x1f, fitf). These files are archived on HF
byte for byte, have been idle 7 to 15 days, and are read by no look build with `--full`; together they free 25.5 GB.
Step 1's looks (s1fit, s1sel, s1eval) and M3B's CSR stores, which the look builds read, stay.

Verifications, aliases and releases write only records, and a G3 build writes about 40 MB a part. The feeder sends
all of them below its disk floor (30 GB), so a release never waits behind the space it frees.

**The saved originals are unaffected.** The 129 originals in `outputs/p1gc/p1gb_look_records`, and the list naming
the 48 lost ones, are untouched. The alias compares them with the records the redo writes.

## 7. Results

Filed 27 October 2026 from the host's records: the eleven alias records
`outputs/p1gc/look_alias/<dataset>/<carve>.json`, written 2026-10-19T20:07:54Z to 20:11:27Z. **ALIAS on all 11 carves,
with no problem in any record.** Every one of P1g-b's 140 cache parts was built again from the rebuilt look and verified
IDENTICAL; each alias record names its parts' verification records by sha256. Of the 177 look records, 129 originals
were saved, and each rebuild equals its original in every field but `seconds`, `utc` and `peak_rss_bytes` (none byte for
byte, as section 1 found). The 48 lost originals' rebuilds hold the saved originals' common value of every
carve-independent field and their carve's single value of every carve field.

| dataset | carve | parts verified | look records | originals saved | originals lost |
| --- | --- | ---: | ---: | ---: | ---: |
| 2wiki | fit | 15 | 15 | 0 | 15 |
| 2wiki | s1eval | 29 | 29 | 4 | 25 |
| hotpotqa | fit | 12 | 12 | 5 | 7 |
| hotpotqa | s1eval | 12 | 12 | 12 | 0 |
| metaqa | fit | 14 | 14 | 13 | 1 |
| metaqa | s1eval | 22 | 22 | 22 | 0 |
| musique | s1eval | 10 | 23 | 23 | 0 |
| musique | s1fit | 19 | 41 | 41 | 0 |
| squad | fit | 1 | 1 | 1 | 0 |
| squad | s1eval | 1 | 1 | 1 | 0 |
| webqsp | s1eval | 5 | 7 | 7 | 0 |
| all | | 140 | 177 | 129 | 48 |

What follows, fixed in section 4: P1g-c runs as declared: the builds read the rebuilt looks under the stamped sha256s,
then its gate decides as before. Its gate opened at 2026-10-19T20:51:42Z with no problems, and P1g-c is filed in
docs/P1GC_ADOPTED_ON_P1GB.md, section 8: pools and input both NOT_ADOPTED. A7p, which reads the same looks (section 4),
is filed in docs/A7P_P1GB_POOLS.md, section 7.

Records, filed with this section: the eleven alias records. The 140 per-part verification records
(`outputs/p1gc/look_alias/parts/<dataset>/<carve>/<part>.json`) and the 129 saved originals
(`outputs/p1gc/p1gb_look_records`, with `saved.json`) stay on the host; each alias record holds the sha256 of every
part's verification record and of the saved list.
