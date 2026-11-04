# B1L: the models of record, fitted without LLM links, read on an LLM-built graph (b1o)

Declared 5 November 2026, before any B1L read. Follows [B1P_PKO_CHAIN_ON_B1.md](B1P_PKO_CHAIN_ON_B1.md) (PKO's chain on b1w and
b1t, read by zg6r; filed in d9570e4), [B1M_MLP_OF_RECORD_ON_B1.md](B1M_MLP_OF_RECORD_ON_B1.md) (zrckpn on the same carves; filed in
272f8e9) and [P2C_LLM_FREE_TRIPLES.md](P2C_LLM_FREE_TRIPLES.md) sections 2 to 4 (the references matched to B1's passages; f5b24ad).
Its design is B1H's ([B1H_MODELS_OF_RECORD_ON_HYPERLINKS.md](B1H_MODELS_OF_RECORD_ON_HYPERLINKS.md), declared and deferred, not run):
`zb1h.py` is reused unchanged by import. Code: `outputs/mp_unified/zb1l.py` (new). Outputs only under `outputs/b1l/`.

Nothing trains. ML1's and LTM's fits, PKO's chain code and B1P's and B1M's reads are read and never written. B1L decides nothing
about any model, pool, rule or column. It is a measurement.

## 1. Why

The thesis is that the links graph RAG builds with an LLM can be built without one. MuSiQue has no hyperlinks, so HippoRAG 2's
LLM-built graph is the only reference graph there to set the rule's links against. The question, on HippoRAG 2's MuSiQue setting
(B1, 1,000 questions): **what do the models of record, zg6r (ML1's GNN) and zrckpn (LTM's MLP), fitted without LLM links, get on
three link sets, with no new fit?**

- **b1t:** the setting's text links only (the NER and KNN families; no structural family). B1P's and B1M's reads.
- **b1w:** b1t plus the rule's links, W2_S cut to the setting (41,417 pairs). B1P's and B1M's reads.
- **b1o:** b1t plus the P pairs of the LLM-built reference (below). New reads.

It measures b1o's R@5, the differences b1o - b1t, b1w - b1t and b1w - b1o paired by question, and the share of the LLM-built step
that the rule's step recovers where that is defined. Block 2, only if cheap and after block 1 is queued: the same on 2wiki with
SR (SiReRAG's propositions) in the place of H4o.

## 2. What is read

- **Carve b1o.** B1a's six files for the setting byte for byte, except the structural family, which holds the P pairs of the
  reference (section 3). The folder is `outputs/b1l/bench/hipporag2_o/<setting>/`, built by `zb1l.py bench` (B1f's variant layout:
  its build.json names the structural sha it replaced, which `look_b1u.load_lists` checks, so B1a's lists serve it).
- **The reference.** H4o: HippoRAG 2's released OpenIE for MuSiQue with gpt-4o-mini, matched to B1's passages position by position
  in P2c part 1 (11,656 items, 106,847 phrases, 145,948 mentions, 127,315 relation pairs). The file is read only, from the CRAG
  repository's `baselines/repos`; it and its phrases are never committed (counts and per-question numbers only). `bench` re-reads
  it and requires its phrase, entity and relation sets to equal P2c's stored ones (as strings per passage; the vocabulary's ids
  follow each process's set order) and its P-pair count to equal P2c's filed 98,243.
- **The fits.** The twelve fits of record, forked once to `outputs/b1l/fits/h/<fit>` without their reads, `models.pt`
  sha-checked: zg6r `scr-ml1-zg6r-s0|s1|s2` and the same with `-hp`; zrckpn `scr-ltm-zrckpn-s0|s1|s2` and the same with `-hp`. The
  suffix names the split (none: WQ-musique; `-hp`: WQ-hotpotqa).
- **b1t and b1w.** B1P's reads of zg6r and B1M's reads of zrckpn, read only. The report recomputes their rows from the
  per-question files and checks each against B1M's filed rows (section 7), as B1H's does.

## 3. The graph and the types of its entries

**The P form** (P2c, `scripts/p2c_triples.py`: `p_edges`): two passages are joined when they share a phrase that occurs in 2 to 25
passages. Each pair once each way in the file; the setting's store then collapses a repeated passage's rows and self-loops. This is
a passage-level projection of the LLM-built graph (HippoRAG 2 itself walks the phrase graph with its triples; this projection
does not reproduce that).

**The weights do not enter.** The setting's structural family is unweighted by contract (`look_b1.setting_store` stops on a
weighted one), so only H4o's **pair set** enters and the 1/df weights (the sum of 1/df over the shared phrases) are saved beside
(`outputs/b1l/bench/p_weights_<setting>.npz`, not committed) with their distribution recorded (section 7, `links`). The report says
so.

**The types** are B1H's D1 and D2, unchanged: an entry's EM type is U2m (a)'s rule with its filed centres on the entry's endpoint
vectors; its score type is the filed U2p type of the W2_S entry of the same pair where the pair is one, **else U2p's null type
(64)**. Checks as B1H's, no tolerance: the EM types recomputed on all W2_S entries inside the setting equal the filed ones; the
EM types of the P entries that are W2_S entries equal the filed ones; the P CSR equals `setting_store`'s structural store.

**The confound, in the stage's own words.** b1o is NULL-heavy (87.3 % of its entries have no W2_S type) while b1w is W2_S-typed,
and the twelve fits were fitted on W2_S types. So **b1o - b1t is a lower bound on what these models can take from an LLM-built
graph**, and **b1w - b1o mixes the pair set with the typing**. No call of this stage reads as "the rule's links beat the LLM's".
The wording is: the models of record, fitted without LLM links, read on ...

The counts, which depend on no model (`zb1l.py bench`, laptop, 5 November 03:21 IST; `settings` recounts the types and the
record carries them):

| | MuSiQue, H4o |
|---|---:|
| passages (positions; distinct package rows) | 11,656 (11,654) |
| P pairs over positions | 98,243 |
| P pairs over package rows (self-loops and repeats dropped) | 98,211 |
| entries (each pair once each way) | 196,422 |
| passages with a P edge; degree mean, median, p95, max over those | 10,764; 18.2, 14, 50, 435 |
| phrases with df 1 / 2 to 25 / over 25 | 88,639 / 11,927 / 258 |
| W2_S pairs in the setting (entries) | 41,417 (82,834) |
| P entries that are W2_S entries (**typed**) | 25,014 (12.7 %), 12,507 pairs |
| P entries that are not (**NULL**) | 171,408 (87.3 %), 85,704 pairs |
| W2_S entries that are P entries | 25,014 (30.2 %) |
| weights, 5 / 25 / 50 / 75 / 95 / 99 % and max (beside, not entered) | 0.042 / 0.053 / 0.083 / 0.167 / 0.50 / 0.79; 11.77 |

A typed entry whose filed U2p score type is itself null counts as NULL in the types' own record (`entries_null`); `settings` reports
the number.

**The equal-typing control is not run.** b1w and b1o with every entry typed NULL would compare the two pair sets under identical
typing, but the types enter the pool walks (`zpko.rule_batch`: types, walks, pools, KT columns), so it needs two new full pool
chains, not just the typing and columns step.

## 4. What runs (`zb1l.py`)

Each step is zb1h's, run on B1L's roots, carve and folder (the module rebinds zb1h's globals; `chain` spawns this file). Only what
differs is new: `bench`, the lists patch (`on_b1o`), the settings record's label and the gate's file name, the report.

| step | what |
|---|---|
| `bench --dataset D` | Section 2's folder (laptop). |
| `settings` | B1H's section 3 on the folder: `outputs/b1l/settings/<setting>.{npz,json}`. |
| `chain --dataset D` | PKO's chain on carve b1o, each step its own process, BLAS at 2 threads: pools, merge, look shards, bases, cache parts, links, KT, record. |
| `gate --tag musique` | `zpko.gate` on the tag's carve (`outputs/b1l/gate_musique.json`, `b1l_gate_musique.json`). |
| `fork` | The twelve fits, once. |
| `read-gnn`, `lists`, `cols`, `read-mlp` | B1H's, on carve b1o: ML1's read of zg6r on the host's GPU; LTM's lists and columns for zrckpn; zltm's read on the host's CPU. |
| `report` | Sections 5 and 7 (laptop). |

## 5. The report (fixed now)

- **Metrics.** B1b's candidate p@swa, R@5 per question, FC@5 and hit@1 beside; 95% bootstrap intervals over the 1,000 questions
  (2,000 resamples, seed 0).
- **In-domain read** (B1P's and B1M's convention): musique by the WQ-hotpotqa fits (three seeds per model), each question's R@5
  averaged over them; 2wiki by both splits (six fits). **The held-out read** beside (musique by the WQ-musique fits).
- **Paired differences**, question by question: b1o - b1t, b1w - b1t, b1w - b1o, per model, in-domain and held-out.
- **Calls, by the interval** (declared here): for b1o - b1t and b1w - b1t, **UP** when the bottom is above 0, **DOWN** when the top
  is below 0, **FLAT** otherwise. For **b1w - b1o: GREATER** when the bottom is above 0, **LESS** when the top is below 0, **SAME**
  otherwise (the interval holds 0, or touches it).
- **The setting's call**: the two models' in-domain b1w - b1o words: GREATER, SAME or LESS where both agree; **NOT_LESS** where
  they differ and neither is LESS; **MIXED** where one is LESS and the other is not; INCOMPLETE where a read is missing. The
  held-out words are beside and decide nothing.
- **The share of the LLM-built step that the rule's step recovers**, (b1w - b1t) / (b1o - b1t), **only where b1o - b1t is UP**
  (its interval's bottom above 0), with its interval over the resamples whose denominator is above 0 (their count beside). Where
  b1o - b1t is not UP the report says so and gives b1w - b1o.
- **zg6r - zrckpn** on each carve, in-domain and held-out, with the label by the interval: **GNN_ABOVE**, **MLP_ABOVE**, **LEVEL**
  (B1M's).
- **Beside, deciding nothing:** the link sets' counts (section 3), the weights' distribution, the rows against the best published
  rows (`b1f.PUBLISHED`, ABOVE / AT / BELOW by `b1b.call`).
- **What a call means.** It says how two models fitted without LLM links read on the rule's links (W2_S-typed) and on the
  LLM-built pair set (mostly NULL-typed). It is never a claim beyond these twelve fits on these three graphs, and never "the rule's
  links beat the LLM's".

## 6. Deviations, stated before any number

1. **The pair set only.** The 1/df weights do not enter (section 3); they are the P form's other half.
2. **The typing confound** (section 3): 87.3 % of b1o's entries are NULL, the fits were fitted on W2_S types, b1w is W2_S-typed.
3. **A shift of graph family**, as B1H's: the fits are LLM-free and hyperlink-free. They meet a third family on b1o.
4. **A projection.** b1o is H4o's passage-pair projection, not its phrase graph.
5. **Reuse by rebinding.** Folder and file names inside `outputs/b1l/` that zb1h writes keep its labels: the forks sit under
   `fits/h/`, the read markers are `b1h_read_<tag>.json`, the step records name `zb1h_sha256`, the settings record's descriptor is
   relabelled variant `o` with the reference named. `zb1h.py` is never edited, and B1H's own folders and records are not touched.
6. **The columns carry over** (B1M's deviations 1 and 2); each model is read on its own device at the threads B1P and B1M used; the
   chain's steps run at 2 threads; B1's pools are smaller than s1eval's (B1P).
7. **b1t and b1w are recomputed from B1P's and B1M's own per-question files**, and checked against B1M's filed rows.

## 7. The record's layout (fixed now)

`outputs/b1l/report.json` (and `report.md`): B1M's keys for the reads and B1F's for the pairs, with `{model}` in {`zg6r`,
`zrckpn`} and `{carve}` in {`b1t`, `b1w`, `b1o`}. Numbers are fractions of 1 to 4 places, except `gap_points` (points) and the
share (3 places).

    top level            declared_in, stage, candidate, boot, in_domain, settings, missing, checks {equal_to_b1m {checked, bad}},
                         inputs_sha256, zb1l_sha256, zb1h_sha256, utc
    settings[ds]         questions, graph, reference, fits['{model}/{carve}/{fit}'], in_domain['{model}/{carve}'],
                         zero_shot[...], gnn_minus_mlp['in_domain|zero_shot/{carve}'], best_published, calls, gap_points
                         paired['b1o-b1t {model}'], ['b1w-b1t {model}'], ['b1w-b1o {model}']   {R@5, ci}   (paired_zero_shot)
                         calls_vs_llm_graph {in_domain|zero_shot: {model: {b1o-b1t, b1w-b1t, b1w-b1o}}}
                         share_of_llm_step[model] {share, ci, resamples, of, b1o-b1t} | {share: null, reason, b1o-b1t}
                         call (the setting's), types {edges, em, score, checks, content_sha256}, links (section 3's counts)

`outputs/b1l/calls.json` holds the calls, the shares and the zg6r - zrckpn labels.

## 8. Risks

- **A failing check in `settings`** stops the stage before any pool. Reported, not patched.
- **A lists check** (`look_b1u.load_lists`: B1a's lists name the structural sha this folder replaced) stops the first pool.
- **A device mismatch** stops a read before any number. **A zg6r read out of GPU memory:** B1H's cap and share (0.13 / 0.15; B1P's
  measured peak 1.51 GB); a read that dies on the cap runs again with B1P's, disclosed in section 12.
- **Code fixes** before any read number are allowed and disclosed in section 12. After the first read number only an amendment
  changes this file.

## 9. Commit policy

Commit only this file, `zb1l.py`, `outputs/b1l/report.json`, `report.md` and `calls.json`, masked with `mask_host.py`, with
`git add -f`. Never commit the folder, the settings arrays, pools, look chunks, caches, lists, columns, forks, reads, the weights
or any file or phrase of the references.

## 10. Files

    outputs/mp_unified/zb1l.py                          the stage (new)
    outputs/b1l/bench/hipporag2_o/<setting>/            b1o's folder; outputs/b1l/bench/p_weights_<setting>.npz
    outputs/b1l/settings/<setting>.{npz,json}           the P CSR, its types, counts and checks
    outputs/b1l/{pools,look,cache,links,kt,records}/    PKO's chain on carve b1o
    outputs/b1l/gate_musique.json, b1l_gate_musique.json
    outputs/b1l/ltm/{work/lists,lists,cols}/            LTM's lists and columns on carve b1o
    outputs/b1l/fits/h/<fit>/                           the forks, their reads and b1h_read_<tag>.json
    outputs/b1l/report.json, report.md, calls.json

## 11. Queue (the host)

Written 5 November 03:35 IST, after this file's first commit, before any read number. Two block files, `block1_chain.txt` (4
items) and `block2_reads.txt` (16 items), `b1l-` names, kept in the session's scratch and never committed. Every row carries the
same 118 inputs: B1M's rows' inputs, the import closure of `zb1h.py` and `zb1l.py`, `scripts/p2c_triples.py` (the self-test's P
pairs; its top-level imports are the standard library and numpy), this file, and b1o's seven files under
`outputs/b1l/bench/hipporag2_o/musique/` (8.8 MB: B1a's files and the structural family of section 3). The reference files and the
weights are not inputs. Nothing is downloaded and no credential goes to the host.

| item | CPUs | GB | GPU share | after | runs (`zb1l.py ...`) |
|---|---:|---:|---:|---|---|
| `b1l-selftest` | 2 | 2 | | | `selftest` |
| `b1l-settings` | 2 | 4 | | selftest | `settings --dataset musique --host` |
| `b1l-fork` | 1 | 2 | | selftest | `fork` |
| `b1l-chain-musique` | 2 | 16 | | `pkoh-gate`, settings | `chain --dataset musique --host` |
| `b1l-gate-musique` | 1 | 2 | | chain | `gate --tag musique --host` |
| `b1l-lists-musique` | 1 | 2 | | chain | `lists --dataset musique` |
| `b1l-cols-wqm-musique`, `-wqh-` | 2 | 2 | | lists | `cols --split WQ-musique` / `WQ-hotpotqa --dataset musique --host` |
| `b1l-read-gnn-musique-s{0,1,2}[-hp]` (6) | 1.1 | 4 | 0.15 (cap 0.13) | gate, fork | `read-gnn --name scr-ml1-zg6r-s.. --tag musique` |
| `b1l-read-mlp-musique-s{0,1,2}[-hp]` (6) | 6 | 4 | | gate, fork, its split's columns | `read-mlp --name scr-ltm-zrckpn-s.. --tag musique` |

**Reservations** are the peaks the host measured for the same steps: B1P's settings 2.33 GB; its musique chains 6.04 GB (b1t,
2,515 s) and 6.33 GB (b1w, 2,953 s); its zg6r reads 1.4 to 1.5 GB (113 to 190 s); B1M's zrckpn reads 1.9 GB (156 to 190 s) and
columns up to 0.53 GB. The chain is reserved at **16 GB, b1w's 6.33 GB x 2.5** (the coordinator's rule: b1o's structural family has
2.4 x b1w's entries); the host's own measured peak replaces it in the record. The GPU reads carry B1H's cap and share (0.13 / 0.15).

**No laptop development pass.** `settings` needs about 3.5 GB on the laptop, which had 2.9 GB free (held by other applications), so
`selftest` and `settings` are the block's first host items and the chain waits for `settings`. A failing item is fixed, and the fix
is disclosed in section 12 with its commit, before any read number.

**Time.** B1P's b1t and b1w chains took 2,515 s and 2,953 s: 438 s for 82,834 more entries. b1o's 196,422 entries over b1t
extrapolate to about 3,550 s (a projection, not a measurement); the reads take about three minutes each.

## 12. What comes next, named now

- **The union carve** (the rule's pairs plus H4o's, to read what each adds beyond the other) is declared in section 15, after block 1
  was queued and its cost told to the coordinator. No chain step is reusable (every step is per carve): it is one more full chain, a
  gate, lists, two column sets and twelve reads.
- **Block 2** (2wiki with SR; the same P form, rules and report): declared in section 14, after block 1 was queued.
- **Development log** (disclosed here before any read number):
  - `zb1l.py bench --dataset musique` (laptop, 5 November 03:21 IST): the folder and the counts of section 3; no model, no fit.
  - `zb1l.py bench --dataset 2wiki` (laptop, 5 November 03:37 IST): the folder and the counts of section 14; no model, no fit.
  - `zb1lu.py bench --dataset musique` (laptop, 5 November 03:50 IST; run again at 03:52 after a wording change in the report's
    header, so that the folder's recorded script sha is the committed file's): the folder and the counts of section 15; no model, no
    fit.
  - `zb1l.py report` and `zb1lu.py report` were each run once on a temporary fixture (a copy of B1H's laptop settings record, and
    b1o's reads, for the union also b1u's, made as copies of b1w's reads) to check the code end to end. Nothing there is a result;
    the folders were removed.
  - Code fixes made after the first commit and before any read number are listed here with their commit.

## 13. Results

Not yet filed.

## 14. Amendment 1: block 2, 2wiki with SR

Declared 5 November 03:40 IST, after block 1 was queued and before any 2wiki settings, pool or read number. Sections 1 to 13 hold
for 2wiki with the substitutions below, and nothing else changes.

- **The reference** is SR (SiReRAG's propositions; its knowledge-graph file for 2wiki, read only from the CRAG repository's
  `baselines/repos` and never committed), matched to B1's 6,119 passages in P2c part 1 (6,119 items, 29,282 phrases, 83,971
  relation pairs, 42,453 mentions, 27,959 propositions). `zb1l.py bench --dataset 2wiki` (laptop, 5 November 03:37 IST) re-read it
  and required its sets and counts to equal P2c's, as for H4o. The P form, the pair set only (weights beside), the typing, the
  calls, the share and the bootstrap are sections 3 and 5's.
- **The counts**, which depend on no model:

| | 2wiki, SR |
|---|---:|
| passages | 6,119 |
| P pairs (positions = package rows) | 27,364 |
| entries (each pair once each way) | 54,728 |
| passages with a P edge; degree mean, median, p95, max over those | 4,807; 11.4, 7, 35, 143 |
| phrases with df 1 / 2 to 25 / over 25 | 24,361 / 4,869 / 52 |
| W2_S pairs in the setting (entries) | 2,030 (4,060) |
| P entries that are W2_S entries (**typed**) | 3,524 (6.4 %), 1,762 pairs |
| P entries that are not (**NULL**) | 51,204 (93.6 %), 25,602 pairs |
| W2_S entries that are P entries | 3,524 (86.8 %) |
| weights, 5 / 25 / 50 / 75 / 95 / 99 % and max (beside, not entered) | 0.040 / 0.056 / 0.100 / 0.243 / 0.50 / 1.21; 5.66 |

- **The confound is stronger here.** 93.6 % of b1o's entries are NULL, and the rule's W2_S links in 2wiki's setting are 2,030
  pairs, so b1w differs from b1t by 4,060 entries only. b1o - b1t is a lower bound as in section 3; b1w - b1o compares 2,030 typed
  pairs with 27,364 mostly-NULL ones. The wording is section 3's: the models of record, fitted without LLM links, read on ...
- **Reads.** 2wiki's in-domain reads are both splits' fits, six per model (section 5); there is no held-out read. Twelve reads.
- **Queue.** `block3_2wiki.txt`, 18 items, `b1l-` names, the same inputs as block 1's plus 2wiki's seven bench files (125): the
  settings (2 CPUs, 4 GB, after `b1l-selftest`), the chain (2 CPUs, **12 GB** = B1P's 2wiki b1w chain 4.77 GB x 2.5; B1P measured
  545 and 607 s), the gate, the lists, the two column sets and the twelve reads, with block 1's reservations, cap and share.

## 15. Amendment 2: the union carve b1u (MuSiQue)

Declared 5 November 03:54 IST, after blocks 1 and 3 were queued, with the coordinator's go, and before any union settings, pool or
read number. Sections 1 to 14 hold for MuSiQue with the substitutions below, and nothing else changes. **It is filed beside B1L's
report, never in place of it, and B1L's filing (b1t, b1w and b1o on MuSiQue and 2wiki) does not wait for it.**

- **What it reads.** The same twelve fits (zg6r `scr-ml1-zg6r-s{0,1,2}[-hp]`, zrckpn `scr-ltm-zrckpn-s{0,1,2}[-hp]`), no new fit, on
  a fourth link set **b1u**: B1a's NER and KNN families and, as the structural family, the **union** of the rule's W2_S pairs in the
  setting (b1w's: B1f's cut) and the P pairs of H4o (b1o's), each pair once each way, unweighted (section 3). Typing is section 3's:
  an entry is W2_S-typed where it is a W2_S entry of the setting, NULL otherwise. So b1u's rule entries are typed as in b1w and its
  H4o-only entries are NULL as in b1o.
- **Why.** b1w - b1o (section 5) mixes two things, the pair set (the rule's against H4o's) and the typing (W2_S against mostly
  NULL). The union gives each pair set the other's pairs, so it reads what each adds beyond the other. **b1u - b1w** is what H4o's
  pairs that the rule lacks add to the rule's graph; **b1u - b1o** is what the rule's pairs that H4o lacks add to H4o's. Neither is
  a comparison of the rule with the LLM, and neither is worded as one.
- **The counts**, which depend on no model (`zb1lu.py bench --dataset musique`, over package rows):

| | MuSiQue |
|---|---:|
| H4o's P pairs (entries), b1o | 98,211 (196,422) |
| the rule's W2_S pairs in the setting (entries), b1w | 41,417 (82,834) |
| both: pairs (entries) | 12,507 (25,014) |
| the rule only: pairs (entries) | 28,910 (57,820) |
| H4o only: pairs (entries) | 85,704 (171,408) |
| **b1u: pairs (entries)** | **127,121 (254,242)** |
| b1u entries that are the rule's (**typed**) | 82,834 (32.6 %) |
| b1u entries that are H4o's alone (**NULL**) | 171,408 (67.4 %) |
| b1u entries against b1w's / b1o's | 3.07 x / 1.29 x |

- **Calls** (declared here, before any union number). Per model, in-domain and held-out, for b1u - b1w, b1u - b1o and b1u - b1t:
  **UP** when the interval's bottom is above 0, **DOWN** when its top is below 0, **FLAT** otherwise (section 5's words). The
  setting's call is made separately for b1u - b1w and for b1u - b1o, from the two models' in-domain words: **ADDS** where both are
  UP, **NOTHING** where both are FLAT, **HURTS** where both are DOWN; **NOT_DOWN** where they differ and neither is DOWN;
  **MIXED** otherwise; INCOMPLETE where a read is missing. The held-out words and zg6r - zrckpn on b1u are beside and decide
  nothing. There is no share.
- **What a call means.** It says what one pair set adds to the other for two models fitted without LLM links, with the typing
  confound of section 3 (the fits were fitted on W2_S types; b1u is two thirds NULL). It is never a claim beyond these twelve fits on
  these four graphs, never "the rule's links beat the LLM's", and never a statement about H4o's phrase graph.
- **Reads.** MuSiQue's in-domain read is the WQ-hotpotqa fits (section 5), the held-out read beside. b1t and b1w are B1P's and B1M's
  own per-question files, b1o's are B1L's (read only), b1u's are this stage's: twelve reads. 2wiki gets no union (its W2_S pairs are
  2,030 and b1o's NULL share is 93.6 %); it would need a later amendment.
- **Code.** `outputs/mp_unified/zb1lu.py`: `zb1l.py`, and through it `zb1h.py`, rebound to the union's roots, carve, folder and read
  paths. Neither is edited. `bench` (laptop) builds the union's folder from b1o's and B1f's w-structural files and B1a's other
  files, each sha-checked; `settings` (relabelled variant `u`), `chain` (its steps run `zb1lu.py step`), `gate`, `lists`, `cols`,
  `read-gnn` and `read-mlp` are zb1l's and zb1h's on the union's roots; `report` is the union's, with the keys listed next.
- **Records.** `outputs/b1l/u/report.json`, `report.md`, `calls.json`: `stage` B1L-U, per setting the links block (the counts
  above), `levels` for b1t, b1w, b1o, b1u (in-domain and held-out, per model, with intervals), `paired` (`b1u-b1w`, `b1u-b1o`,
  `b1u-b1t`), `calls`, `call`, `gnn_minus_mlp`, the reads' file shas for b1o and b1u, `types`, and the shas of the three scripts.
  Section 9's commit policy holds, with `zb1lu.py` and those three files added.
- **Queue.** Written 5 November 03:54 IST. `block4_union.txt`, 20 items, `b1l-u-` names (so the host watcher for `b1l-` reports
  them), in the session's scratch and never committed; every row carries 126 inputs: block 1's 118, `zb1lu.py` and b1u's seven
  files under `outputs/b1l/u/bench/hipporag2_u/musique/` (8.9 MB). Block 1's items with `b1l-u-` names and block 1's reservations,
  cap and share: `selftest` (2 CPUs, 2 GB), `settings` (2, 4, after the self-test), `fork` (1, 2, after the self-test), `chain` (2,
  **16 GB** = B1P's b1w chain 6.33 GB x 2.5, after `pkoh-gate`, the settings and **`b1l-chain-musique`**, so that block 1 keeps
  the host's CPU), `gate`, `lists`, the two `cols`, six `read-gnn` (1.1, 4, GPU 0.15, cap 0.13) and six `read-mlp` (6, 4).
  Projection, not a measurement: b1w's chain took 2,953 s and 438 s were 82,834 entries, so b1u's 171,408 entries over b1w extrapolate
  to about 3,860 s.
