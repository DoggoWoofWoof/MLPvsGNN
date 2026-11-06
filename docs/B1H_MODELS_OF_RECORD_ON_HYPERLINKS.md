# B1H: the models of record (zg6r, zrckpn) on HippoRAG 2's settings with the corpus hyperlinks (b1)

Declared 5 November 2026, before any B1H read. (Two development runs on the laptop came before it, into a scratch folder with no fit
and no read: section 12.) Follows [B1P_PKO_CHAIN_ON_B1.md](B1P_PKO_CHAIN_ON_B1.md) (PKO's chain on
b1w and b1t, read by zg6r and zrckp; filed in d9570e4) and [B1M_MLP_OF_RECORD_ON_B1.md](B1M_MLP_OF_RECORD_ON_B1.md) (zrckpn on the
same carves; filed in 272f8e9). B1M's section 10 named this stage: PKO's chain needs U2m (a)'s EM type and U2p's score type on every
structural entry, and no stage had computed them for hyperlink entries. Code: `outputs/mp_unified/zb1h.py` (new). Outputs only
under `outputs/b1h/`.

Nothing trains. ML1's and LTM's fits, PKO's chain code and B1P's and B1M's reads are read and never written. B1 is never used for
selection: B1H decides nothing about any model, pool, rule or column. It is a measurement.

## 1. Why

The ladder of carves on HippoRAG 2's three 1,000-question settings is b1t (text links only) -> b1w (U2b's W2_S rule, cut to the
setting) -> b1 (B1a's own structural graph: the corpus hyperlinks on 2wiki and hotpotqa; musique's is the package's title-mention
graph, not hyperlinks). B1P and B1M read both models of record on b1t and b1w. Their b1 rows exist only for the J5 fits (B1b, U1a,
B1F, HLC), which trained on hyperlinks. B1H measures four things for **zg6r** (ML1's GNN) and **zrckpn** (LTM's MLP):

1. **The ladder:** in-domain R@5 on b1t, b1w and b1, and the paired differences b1w - b1t, b1 - b1w and b1 - b1t, with the share of
   the hyperlink gain that b1w recovers.
2. **Table 1's b1 row:** each model's in-domain b1 R@5 against the best published rows (`b1f.PUBLISHED`), by interval.
3. **The two models of record on b1:** zg6r - zrckpn, paired by question.
4. **HLC's family H for the models of record:** the model's hyperlink lift (b1 - b1t) against the lift of HippoRAG 2's walk, question
   for question.

## 2. What is read

- **Carve b1.** B1a's own folders, `outputs/bench/hipporag2/<setting>`, unchanged: queries, first stage, NER and KNN graphs, and the
  structural graph. These are the files B1b, U1a, B1F, HLC and HLW read, so the walk's with-links and without-links arms of HLC are
  the same files here.
- **The fits.** The twelve fits of record, forked once to `outputs/b1h/fits/h/<fit>` without their reads, `models.pt` sha-checked
  against the source, as B1P and B1M did:
  - zg6r (ML1): `scr-ml1-zg6r-s0`, `-s1`, `-s2`, and the same three with the suffix `-hp`;
  - zrckpn (LTM): `scr-ltm-zrckpn-s0`, `-s1`, `-s2`, and the same three with `-hp`.

  The suffix names the split: none is WQ-musique, `-hp` is WQ-hotpotqa.
- **b1t and b1w.** B1P's reads of zg6r (`outputs/b1p/fits/<w|t>/<fit>/reads`) and B1M's reads of zrckpn
  (`outputs/b1m/fits/<w|t>/<fit>/reads`), read only. The report recomputes their rows from the per-question files and checks each
  against B1M's filed rows (section 7).

## 3. The hyperlink graph and the types of its entries

**The graph.** For each setting, B1a's structural graph on the setting's passages, as a symmetric CSR over the package's nodes:
each undirected pair once in each direction, self-loops dropped, repeats merged, rows and columns ascending. This is what
`FamilyStore.from_graph` builds, and `look_b1.setting_store` hands the pools. The step `settings` builds it from the graph file and
**stops** if it is not equal to `setting_store`'s own structural store, entry for entry.

**The types.** PKO's pair types come from two per-entry types: U2m (a)'s EM type (64 centres, null = 64) and U2p's score type (64
cuts, null = 64). They exist for W2_S's entries and for no hyperlink. The rule, neutral and fixed here, uses no b1 number:

- **EM type (D2).** U2m (a)'s own code (`zu2m_a_types.node_scores`, `entry_types`) with its filed per-dataset centres `mu`, on the
  entry's two endpoint vectors (the frozen encoder's), computed over the setting's nodes. The entry is null where the two vectors are
  identical, as in U2m (a).
- **Score type (D1).** The filed U2p type of the W2_S entry for the same pair, where the pair is an entry of W2_S's CSR cut to the
  setting; **else the null score type (64)**, U2p's own "no score" class.

**Checks, with no tolerance** (a failing check stops the step; nothing is tuned):

1. The EM types recomputed by the rule on **all** W2_S entries inside the setting equal U2m (a)'s filed types.
2. The EM types of the hyperlink entries that are W2_S entries equal the filed types of those entries.
3. The hyperlink CSR equals `setting_store`'s structural store.

**The consequence of D1, stated before any read.** A hyperlink that W2_S did not recover gets the null score type. W2_S's recall of
the hyperlinks is about 0.84 on 2wiki, 0.80 on hotpotqa and 0.89 on musique, so 11 to 20 % of the entries are untyped by score. The
null score type is not new to the models: U2p's filed types hold it on 245,854 of 4,687,840 entries of musique (5.2 %) and on
627,522 of 1,756,400 of squad (35.7 %), and both WQ splits train on squad. But on 2wiki and hotpotqa W2_S's own pools it hardly
occurs (4 of 142,967,492 entries and 0 of 64,318,458), so on those two settings a sixth of the entries sit in a class the
models met only on other datasets' pools. B1H's b1 rows therefore state what zg6r and zrckpn do with hyperlinks that carry W2_S's
types where W2_S has them and no score type elsewhere. They are not what a model that was fitted with hyperlink types would do.

The step `settings` was run on the laptop on 5 November between 01:12 and 01:24 IST, twice (the second time after the arrays were
compressed and given a content sha, section 4), **before any fit or read**, to test its three checks. All passed with 0 differing
entries both times, and the counts were the same. The counts, which depend on no model:

| setting | graph | passages | pairs | entries | W2_S entries inside the setting | entries that are W2_S entries | entries with the null score type |
|---|---|---:|---:|---:|---:|---:|---:|
| 2wiki | corpus hyperlinks | 6,119 | 1,984 | 3,968 | 4,060 | 3,348 (84.4 %) | 620 (15.6 %) |
| hotpotqa | corpus hyperlinks | 9,811 | 4,400 | 8,800 | 8,630 | 7,074 (80.4 %) | 1,726 (19.6 %) |
| musique | title mentions | 11,656 | 42,823 | 85,646 | 82,834 | 76,508 (89.3 %) | 9,806 (11.4 %) |

On musique 668 of the null entries are W2_S entries whose own filed score type is null. The record reports these shares per setting
(section 7, `types`).

## 4. What runs (`zb1h.py`)

Each step is B1P's or B1M's on carve b1, in new code that imports the frozen modules and writes only under `outputs/b1h/`.

| step | what |
|---|---|
| `settings` | Section 3: `outputs/b1h/settings/<setting>.npz` (compressed) and `.json` (the CSR, both types, the counts, the checks, the shas). The graph's description, which the pools' records carry, holds the arrays' own shas (`content_sha256`), not the zip file's, so a rebuild gives the same description. |
| `chain --dataset D` | PKO's chain on carve b1 of D, each step its own process with BLAS at 2 threads (`zb1p.chain`'s): pools pass, merge, look shards, bases, cache parts, links, KT, record. A step with a filed record is skipped. `zpko`'s open calls see the setting, the hyperlink CSR and these types (`zb1p.on_b1`, with B1a's own folder and the types above). The pools' open stops if the structural store it reads is not the settings' CSR, entry for entry. |
| `gate --tag main` and `--tag musique` | `zpko.gate` on the tag's carves: 2wiki and hotpotqa first (`outputs/b1h/gate.json`), musique on its own (`gate_musique.json`). |
| `fork` | The twelve fits, once. |
| `read-gnn --name FIT --tag T` | ML1's read of zg6r on the host's GPU, B1P's body (ML1's device check, no stamp), PKO's roots and the tag's gate. |
| `lists`, `cols` | LTM's lists and columns for zrckpn on carve b1, as `zb1m.py` (the same patches, restored after). |
| `read-mlp --name FIT --tag T` | zltm's read of zrckpn on the host's CPU (six threads, GPU hidden), B1M's body. |
| `report` | Sections 5 and 7, on the laptop (family H runs HLW's walks, minutes per setting). |

The tags split the reads so that 2wiki and hotpotqa proceed without waiting for musique's chain (about 3.3 M pool rows).

## 5. The report (fixed now)

- **Metrics.** B1b's candidate p@swa, R@5 per question, with FC@5 and hit@1 beside. 95% bootstrap intervals over the setting's 1,000
  questions (B1b's: 2,000 resamples, seed 0), as B1P and B1M.
- **In-domain read of a setting** (B1P's and B1M's):
  - musique: WQ-hotpotqa;
  - hotpotqa: WQ-musique;
  - 2wiki: both splits.

  Per model, each question's R@5 is averaged over the in-domain fits (three seeds; six fits on 2wiki). **The held-out read**,
  beside: the fits that did not train on the setting (hotpotqa by WQ-hotpotqa, musique by WQ-musique). It comes free with the same
  reads and is reported for hotpotqa and musique.
- **Deciding settings.** 2wiki and hotpotqa. Musique is beside in every table: its b1 is not a hyperlink graph, and its chain is run
  last.
- **Table 1's b1 row.** Each model's in-domain b1 R@5 against the best published row of the setting, of any kind and peer-reviewed
  only: ABOVE, AT or BELOW by the interval (`b1b.call`), with the gap in points. The b1t and b1w rows are called the same way.
- **The ladder.** Paired by question, each question's in-domain mean: b1w - b1t, b1 - b1w, b1 - b1t, with intervals, and **the share
  of the hyperlink gain that b1w recovers**, (b1w - b1t) / (b1 - b1t), computed only where b1 - b1t > 0 (as B1F's, without its b1u).
- **The two models of record.** zg6r - zrckpn on each carve, in-domain and held-out, with the label by the interval: **GNN_ABOVE**
  when its bottom is above 0, **MLP_ABOVE** when its top is below 0, **LEVEL** otherwise (B1M's), and each fit pair's difference.
- **HLC's family H** for the models of record (the same method as HLC's section on family H). For each model, per question:
  the model's lift, R@5(b1) - R@5(b1t), against the walk's lift, A1 on structural + NER + KNN minus A1 on NER + KNN (HLW's walk,
  restart A1, the same files), and the difference of the two with its paired interval. The call is **GREATER** when the
  interval's bottom is above 0, **LESS** when its top is below 0, **SAME** otherwise. The deciding cells are 2wiki and hotpotqa,
  in-domain, for zg6r and zrckpn; the stage call is **GREATER_EVERYWHERE** when all four are GREATER, **LESS_SOMEWHERE** when one is
  LESS, **MIXED** otherwise. Musique and the held-out read are beside. HLC's family H rows for the J5 fits stay as filed and are named
  beside, never mixed with these.
- **What a call means.** A model's lift larger than the walk's says the model gets more out of the links than the walk does. The
  call never says that the links carry the model: its R@5 on b1t is the baseline.
- **Beside, deciding nothing.** The NULL share of each setting's entries, and the J5 fits' rows (B1F's, HLC's).

## 6. Deviations, stated before any number

1. **A shift of graph family.** zg6r and zrckpn were fitted on pools that W2_S's links built. The fits are hyperlink-free (B1P).
   On b1 they meet another family of links. The J5 fits of B1b and U1a, which trained on hyperlinks, stay beside and are never
   mixed with B1H's rows.
2. **The types of hyperlink entries** are D1's and D2's (section 3), and 11 to 20 % of the entries carry the null score type.
3. **Musique's b1** is the package's own title-mention graph, 42,823 pairs, not a hyperlink graph. It is run last and reported
   beside.
4. **The columns carry over** (B1M's deviations 1 and 2): LTM's scorers were fitted on the full corpora's training carves. The
   column records keep LTM's stage and declaration.
5. **Each model is read on its own device** (B1P's GPU for zg6r, B1M's CPU for zrckpn), at the threads those stages used. The
   chain's steps run at 2 threads.
6. **B1's pools are smaller** than s1eval's (B1P, section 7).
7. **The b1t and b1w rows are recomputed, not copied.** The report loads B1P's and B1M's own per-question files and checks every
   row against B1M's filed rows (section 7). The pairs need the per-question values.

## 7. The record's layout (fixed now)

`outputs/b1h/report.json` (and `report.md`) is built so that one reader of B1M's, B1F's and HLC's filed records reads it too: the
same keys, with `{model}` in {`zg6r`, `zrckpn`} and `{carve}` in {`b1t`, `b1w`, `b1`}. Nothing filed by B1M, B1F or HLC changes.
Numbers are fractions of 1, rounded to 4 places as theirs, except `gap_points` (points, 2 places) and the share (3 places).

    top level            declared_in, stage, candidate, boot {resamples, seed}, in_domain {setting: [splits]}, settings,
                         missing, checks, deciding_cells {H: ["2wiki zg6r", "2wiki zrckpn", "hotpotqa zg6r", "hotpotqa zrckpn"]},
                         call_H, call_H_j5_beside, inputs_sha256, zb1h_sha256, utc
    settings[ds]         ds in {2wiki, hotpotqa, musique}; musique beside

    B1M's shape (the reads):
      questions, graph
      fits['{model}/{carve}/{fit}']            {R@5, FC@5, hit@1, file_sha256}
      in_domain['{model}/{carve}']             {R@5, R5_ci, FC@5, hit@1, fits, per_fit}
      zero_shot['{model}/{carve}']             the same, the held-out fits; empty on 2wiki
      gnn_minus_mlp['in_domain/{carve}']       {R@5, ci, label, per_fit_pair: [{gnn, mlp, R@5}]}
      gnn_minus_mlp['zero_shot/{carve}']       the same
      best_published                           {any, peer_reviewed}: {row, R@5 (points)}
      calls['{model}/{carve}']                 {any, peer_reviewed}: ABOVE | AT | BELOW (in-domain row)
      gap_points['{model}/{carve}']            {any, peer_reviewed}: points
    The rows for b1t and b1w sit under the same keys, recomputed from B1P's and B1M's read files; checks.equal_to_b1m
    {checked, bad: []} counts every read's sha256 and every fit, in_domain and zero_shot row compared with B1M's filed record.

    B1F's shape (the pairs, no '/J5'; in-domain, the held-out read in paired_zero_shot with the same keys):
      paired['b1w-b1t {model}'], ['b1-b1w {model}'], ['b1-b1t {model}']      {R@5, ci}
      share_of_hyperlink_gain['{model}']       {b1w}   only where b1 - b1t > 0 (zero-shot: share_of_hyperlink_gain_zero_shot)

    HLC's shape (family H; in-domain, the held-out read in H_zero_shot):
      H['{model}']       {model ("GNN" | "MLP"), read, levels {model_with, model_without, walk_with, walk_without, R0},
                          model_lift {R@5, ci}, walk_lift, difference, call, model_with_minus_walk_with,
                          model_without_minus_walk_with}
      H_j5_beside        {source, cells: {zsp/J5, zrc/J5, zsp/L-<ds>, zrc/L-<ds>}} as HLC filed them

    beside, deciding nothing:
      types              {edges {entries, pairs, ...}, em, score {entries_w2s, entries_w2s_share, entries_null,
                          entries_null_share, ...}, checks, npz_sha256, content_sha256}
      walk_arms          the R0 and A1 arms' means

`outputs/b1h/calls.json` holds `call_H`, each cell's call, the calls against the best published rows and the zg6r - zrckpn labels.

## 8. Risks

- **A failing check in `settings`** (section 3) stops the stage before any pool. It is reported, not patched.
- **An identity failure in `cols`** (B1M's) stops that carve's columns. It is reported, not patched.
- **A device mismatch** (the host's CPU or GPU torch changed since the fits) stops the read before any number.
- **A zg6r read out of GPU memory.** The cap and share of section 11 are twice B1P's measured peak on three settings (1.51 GB);
  the first read's peak sizes the rest, and a read that dies on the cap is run again with B1P's cap (0.27), disclosed in section 12.
- **Code fixes** before any read number are allowed and disclosed in section 12. After the first read number only an amendment
  changes this file.

## 9. Commit policy

Commit only this file, `zb1h.py`, `outputs/b1h/report.json`, `report.md` and `calls.json`, masked with `mask_host.py`, with
`git add -f`. Never commit the settings arrays, pools, look chunks, caches, lists, columns, forks or reads.

## 10. Files

    outputs/mp_unified/zb1h.py                         the stage (new)
    outputs/b1h/settings/<setting>.{npz,json}          the hyperlink CSR, its types, counts and checks
    outputs/b1h/{pools,look,cache,links,kt,records}/   PKO's chain on carve b1 (roots as zpko.roots)
    outputs/b1h/gate.json, gate_musique.json           zpko.gate on the two tags
    outputs/b1h/ltm/{work/lists,lists,cols}/           LTM's lists and columns on carve b1
    outputs/b1h/fits/h/<fit>/                          the forks, their reads and b1h_read_<tag>.json
    outputs/b1h/report.json, report.md, calls.json     sections 5 and 7

## 11. Queue (the host)

Three blocks of host items, as B1P's and B1M's: block 1 (self-test, settings, fork, the 2wiki and hotpotqa chains), block 2 (the
`main` tag's gate, lists, columns and twelve reads), block 3 (musique, last). Each item carries the memory its analogue measured on
the host (the feeder's own record of B1P's and B1M's items), not the earlier tables' reservations; the host is shared, so no row
asks for more.

| items | CPUs | GB | GPU | after | measured by |
|---|---:|---:|---:|---|---|
| `b1h-selftest` | 2 | 2 | | - | (numba's two threads need two CPUs) |
| `b1h-settings` | 2 | 6 | | the self-test | 3.47 GB, a fresh laptop build |
| `b1h-fork` | 1 | 2 | | the self-test | B1P's fork, 2 s |
| `b1h-chain-2wiki`, `b1h-chain-hotpotqa` | 2 | 6 | | `pkoh-gate`, `b1h-settings` | B1P's chains, 4.5 - 5.2 GB, 9 - 10 min |
| `b1h-gate-main` | 1 | 2 | | both chains | B1P's gate, 4 s |
| `b1h-lists-<dataset>` (two) | 1 | 2 | | the dataset's chain | B1M's lists, 2 - 7 s |
| `b1h-cols-<wqm or wqh>-<dataset>` (four) | 2 | 2 | | the dataset's lists | B1M's columns, <= 0.53 GB |
| `b1h-read-gnn-main-<fit>` (six) | 1.1 | 6 | 0.15 | `b1h-gate-main`, `b1h-fork` | B1P's reads, GPU peak 1.51 GB: cap 0.13 (3.1 GB), share 0.15 |
| `b1h-read-mlp-main-<fit>` (six) | 6 | 4 | | `b1h-gate-main`, `b1h-fork`, the fit's split's two column sets | B1M's reads, 1.9 GB |
| `b1h-chain-musique` | 2 | 8 | | `pkoh-gate`, `b1h-settings` | B1P's chains, 6.0 - 6.3 GB, 42 - 49 min |
| `b1h-gate-musique`, `b1h-lists-musique`, `b1h-cols-<split>-musique` (two), the twelve `b1h-read-{gnn,mlp}-musique-<fit>` | as above | | | the musique chain |

Musique's chain waits for nothing of 2wiki's or hotpotqa's: its block comes last in the queue and the host's CPUs are free for
it.

The report runs on the laptop after `rx.py fetch` of the b1h reads and of B1M's zrckpn b1t and b1w reads.

## 12. What comes next, named now

- **Hyperlink-typed fits.** A fit whose pools carried the hyperlinks' own EM and score types is a different model. It is not
  declared here.
- **Development log** (disclosed here before any read number):
  - the `settings` runs of section 3 (laptop, 5 November, 01:12 to 01:24 IST);
  - the pools pass on the first eight questions of 2wiki's carve b1 (laptop, 01:13 to 01:25 IST, into a scratch folder, not
    `outputs/b1h`; no fit, no read, no number kept): it opened the setting through `on_b1h`, computed the first-stage lists of the 1,000 questions, prepared the structural
    family (3,968 entries, 53,885 rows) and ran the pass on eight questions; exit 0. The self-test (laptop, 01:25 IST) passed
    after the last code change. That change, the check that the structural store equals the settings' CSR at the pools' open (two
    lines), came after the development run began, so no run has exercised it yet; the first chain does, and a failure there stops
    before any pool row.
  - The queue of section 11 was written after the declaration's first commit (29c2553): the table's reservations are the measured
    peaks of the analogous B1P and B1M items, in place of the first table's, and the musique chain no longer waits for the other
    two (nothing in the rule changes).
  - Code fixes made after the first commit and before any read number are listed here with their commit.

## 13. Results

Filed 6 November 20:14 IST. Records: `outputs/b1h/report.json`, `report.md` and `calls.json` (masked with `mask_host.py`; counts and per-question
statistics only). `zb1h.py` is the file committed with the declaration (29c2553) and ran unchanged on the host and on the laptop: its
sha256 (`33b0e49554e2...`) is the one in every gate record, read marker and `report.json`, so **no code fix was made and no deviation
was added to section 6.** All 36 read arrays (12 fits x 3 settings) and 24 read markers (12 fits x 2 tags) were fetched, none missing;
both gates (`b1h_gate_main.json`, `b1h_gate_musique.json`) read READY; each setting's type checks hold (`w2s_em_recomputed_mismatches` 0
and `overlap_em_mismatches` 0 on 2wiki, hotpotqa and musique); the settings JSONs the host built carry the same `content_sha256` as
the laptop's three; the b1t and b1w rows recomputed from B1P's and B1M's read files equal B1M's filed rows (92 checked, 0 differ). R@5 x100
of the p@swa candidate, 1,000 questions per setting; the intervals are 95% bootstrap intervals over the questions (2,000 resamples, seed
0), the differences paired. **The two models were fitted on W2_S pools, never on a hyperlink graph: b1 is a shift of graph family for
both (section 6), and every word below is the declared rule's and nothing more.**

**Queue history** (disclosed, no rule or reservation changed). Block 1 (self-test, settings, fork, the three chains) was sent from
01:50 IST on 5 November; the chains ended rc 0 (2wiki 466 s, peak 4.62 GB; hotpotqa 451 s, 4.83 GB; musique 2,202 s, 6.26 GB) and
`b1h-lists-hotpotqa` ran once. At 02:00 the other 35 rows were taken out of the feeder's queue (no read was sent then). On 6 November
at 10:49 IST, after the user asked that the host not idle, the 35 went back, behind the end of FSV's last M4 pass (`fsvm4b-2wiki`: the
five root rows waited on it, so the timing pass had a quiet host) and after ISB's rows; at 10:51 the twelve zg6r read rows were moved
ahead of ISB's fits. `b1h-lists-hotpotqa` carries the suffix `-r2` in the second queue because the feeder held its first name as sent
(same command; it ended rc 0 in 1.9 s). The gates ran at 11:39 IST and the reads between 11:41 and 14:20 IST, all 24 jobs rc 0. Measured on the host (the
jobs whose ids resolved: 8 of 12 zg6r reads, 6 of 12 zrckpn reads): zg6r main tag 16-18 s, 1.15-1.18 GB; zg6r musique 649 s, 1.51-1.52 GB;
zrckpn main tag 8-9 s, under 0.1 GB; zrckpn musique 162 s, 1.91 GB; every request was above its peak. Report:
`zb1h.py report` on the laptop, 20:07 IST.

**The ladder, in-domain** (2wiki: both splits, six fits; hotpotqa: the WQ-musique fits; musique, beside: the WQ-hotpotqa fits; each
question's R@5 averaged over the fits):

| setting | model | b1t | b1w | b1 | b1w - b1t | b1 - b1w | b1 - b1t | share of the gain at b1w | b1 calls (any / peer) | b1 gap (any / peer) |
|---|---|---:|---:|---|---|---|---|---:|---|---|
| 2wiki | zg6r | 78.1 (76.7-79.5) | 94.9 (94.1-95.7) | 95.4 (94.6-96.1) | +16.8 (+15.5, +18.2) | +0.4 (-0.3, +1.1) | +17.3 (+16.0, +18.6) | 0.974 | BELOW / BELOW | -3.1 / -2.8 |
| 2wiki | zrckpn | 77.0 (75.6-78.4) | 94.1 (93.2-94.9) | 94.4 (93.5-95.2) | +17.1 (+15.8, +18.4) | +0.3 (-0.4, +0.9) | +17.4 (+16.1, +18.7) | 0.984 | BELOW / BELOW | -4.1 / -3.8 |
| hotpotqa | zg6r | 91.1 (90.0-92.3) | 94.3 (93.3-95.3) | 96.5 (95.7-97.3) | +3.1 (+2.4, +4.0) | +2.2 (+1.5, +2.9) | +5.3 (+4.4, +6.2) | 0.589 | BELOW / BELOW | -2.3 / -0.9 |
| hotpotqa | zrckpn | 90.4 (89.2-91.6) | 93.4 (92.4-94.5) | 95.6 (94.8-96.5) | +3.0 (+2.2, +3.9) | +2.2 (+1.5, +2.9) | +5.2 (+4.3, +6.2) | 0.580 | BELOW / BELOW | -3.1 / -1.8 |
| musique | zg6r | 66.0 (64.3-67.7) | 69.0 (67.2-70.7) | 68.6 (66.9-70.4) | +3.0 (+1.8, +4.0) | -0.4 (-0.9, +0.2) | +2.6 (+1.6, +3.7) | 1.135 | BELOW / BELOW | -12.1 / -9.7 |
| musique | zrckpn | 64.5 (62.8-66.1) | 67.4 (65.7-69.2) | 67.3 (65.6-69.0) | +2.9 (+1.9, +4.0) | -0.1 (-0.6, +0.4) | +2.9 (+1.8, +4.0) | 1.032 | BELOW / BELOW | -13.3 / -11.0 |

Best published rows: 2wiki EviReform 98.45 / G-reasoner 98.2; hotpotqa BridgeRAG 98.75 / PropRAG 97.4; musique SAG2 80.68 / PropRAG 78.3

**The held-out read** (hotpotqa by the WQ-hotpotqa fits, musique by the WQ-musique fits; decides nothing):

| setting | model | b1t | b1w | b1 | b1w - b1t | b1 - b1w | b1 - b1t |
|---|---|---:|---:|---|---|---|---|
| hotpotqa | zg6r | 89.4 (88.2-90.6) | 92.9 (91.9-94.0) | 94.7 (93.8-95.6) | +3.5 (+2.6, +4.4) | +1.8 (+1.1, +2.5) | +5.3 (+4.3, +6.3) |
| hotpotqa | zrckpn | 88.6 (87.4-89.9) | 92.2 (91.2-93.4) | 93.9 (92.9-94.9) | +3.6 (+2.8, +4.5) | +1.6 (+0.9, +2.3) | +5.3 (+4.4, +6.2) |
| musique | zg6r | 62.1 (60.4-63.7) | 60.8 (59.1-62.5) | 59.1 (57.4-60.9) | -1.3 (-2.6, +0.1) | -1.7 (-2.3, -1.0) | -2.9 (-4.3, -1.6) |
| musique | zrckpn | 60.0 (58.4-61.7) | 61.8 (60.1-63.6) | 61.1 (59.3-62.9) | +1.8 (+0.5, +3.1) | -0.8 (-1.3, -0.1) | +1.1 (-0.1, +2.3) |

**zg6r - zrckpn**, paired by question, on each carve:

| setting | read | carve | zg6r - zrckpn (CI) | label |
|---|---|---|---|---|
| 2wiki | in-domain | b1t | +1.1 (+0.7, +1.6) | GNN_ABOVE |
| 2wiki | in-domain | b1w | +0.8 (+0.5, +1.2) | GNN_ABOVE |
| 2wiki | in-domain | b1 | +1.0 (+0.6, +1.4) | GNN_ABOVE |
| hotpotqa | in-domain | b1t | +0.8 (+0.2, +1.4) | GNN_ABOVE |
| hotpotqa | in-domain | b1w | +0.9 (+0.4, +1.4) | GNN_ABOVE |
| hotpotqa | in-domain | b1 | +0.9 (+0.4, +1.3) | GNN_ABOVE |
| hotpotqa | zero-shot | b1t | +0.8 (+0.0, +1.5) | GNN_ABOVE |
| hotpotqa | zero-shot | b1w | +0.6 (+0.1, +1.2) | GNN_ABOVE |
| hotpotqa | zero-shot | b1 | +0.8 (+0.2, +1.4) | GNN_ABOVE |
| musique | in-domain | b1t | +1.5 (+0.6, +2.4) | GNN_ABOVE |
| musique | in-domain | b1w | +1.5 (+0.7, +2.4) | GNN_ABOVE |
| musique | in-domain | b1 | +1.3 (+0.4, +2.1) | GNN_ABOVE |
| musique | zero-shot | b1t | +2.0 (+1.1, +2.9) | GNN_ABOVE |
| musique | zero-shot | b1w | -1.1 (-1.9, -0.1) | MLP_ABOVE |
| musique | zero-shot | b1 | -1.9 (-2.8, -1.1) | MLP_ABOVE |

**HLC's family H on the models of record** (stage call MIXED; HLC's rows for the J5 fits beside, as filed: GREATER_EVERYWHERE):

Model lift = R@5(b1) - R@5(b1t); walk lift = A1 on structural + NER + KNN minus A1 on NER + KNN (HLW's walk); difference = model lift - walk lift, paired by question. Deciding cells (in-domain): 2wiki zg6r, 2wiki zrckpn, hotpotqa zg6r, hotpotqa zrckpn.

| setting | model | read | model lift (CI) | walk lift (CI) | difference (CI) | call |
|---|---|---|---|---|---|---|
| 2wiki | zg6r | in-domain | +17.3 (+16.0, +18.6) | -0.1 (-0.9, +0.6) | +17.4 (+15.9, +18.9) | GREATER |
| 2wiki | zrckpn | in-domain | +17.4 (+16.1, +18.7) | -0.1 (-0.9, +0.6) | +17.5 (+16.0, +19.1) | GREATER |
| hotpotqa | zg6r | in-domain | +5.3 (+4.4, +6.2) | +5.1 (+4.1, +6.3) | +0.2 (-1.2, +1.6) | SAME |
| hotpotqa | zrckpn | in-domain | +5.2 (+4.3, +6.2) | +5.1 (+4.1, +6.3) | +0.1 (-1.4, +1.6) | SAME |
| hotpotqa | zg6r | held-out | +5.3 (+4.3, +6.3) | +5.1 (+4.1, +6.3) | +0.1 (-1.3, +1.6) | SAME |
| hotpotqa | zrckpn | held-out | +5.3 (+4.4, +6.2) | +5.1 (+4.1, +6.3) | +0.1 (-1.4, +1.6) | SAME |
| musique | zg6r | in-domain | +2.6 (+1.6, +3.7) | +0.8 (-0.1, +1.8) | +1.8 (+0.4, +3.1) | GREATER |
| musique | zrckpn | in-domain | +2.9 (+1.8, +4.0) | +0.8 (-0.1, +1.8) | +2.0 (+0.6, +3.4) | GREATER |
| musique | zg6r | held-out | -2.9 (-4.3, -1.6) | +0.8 (-0.1, +1.8) | -3.8 (-5.3, -2.2) | LESS |
| musique | zrckpn | held-out | +1.1 (-0.1, +2.3) | +0.8 (-0.1, +1.8) | +0.2 (-1.2, +1.7) | SAME |

**The link sets**, by entry type (section 3):

| setting | hyperlink pairs | entries | entries that are W2_S entries | entries with the null score type | W2_S pairs in the setting |
|---|---:|---:|---:|---:|---:|
| 2wiki | 1984 | 3968 | 3348 (84.4%) | 620 (15.6%) | 2030 |
| hotpotqa | 4400 | 8800 | 7074 (80.4%) | 1726 (19.6%) | 4315 |
| musique | 42823 | 85646 | 76508 (89.3%) | 9806 (11.5%) | 41417 |

Check: the b1t and b1w rows recomputed from B1P's and B1M's read files equal B1M's filed rows (92 checked, 0 differ). Missing reads: 0.

**What the reads say, under the rules declared above and nothing more.**

- **The stage call is MIXED.** The four deciding cells are in-domain on 2wiki and hotpotqa. 2wiki is GREATER for both models:
  the difference of the model's lift and the walk's is +17.4 (+15.9, +18.9) points for zg6r and +17.5 (+16.0, +19.1) for zrckpn (the
  models' lifts are +17.3 (+16.0, +18.6) and +17.4 (+16.1, +18.7), the walk's -0.1 (-0.9, +0.6)). Hotpotqa is SAME for both: +0.2 (-1.2, +1.6)
  and +0.1 (-1.4, +1.6), the models' lifts +5.3 (+4.4, +6.2) and +5.2 (+4.3, +6.2) against the walk's +5.1 (+4.1, +6.3). HLC's rows for the
  J5 fits stay as filed (their stage call is GREATER_EVERYWHERE) and are not mixed with these. The call says that the models get more out
  of the links than the walk does on 2wiki and no more than the walk does on hotpotqa; it never says that the links carry the model
  (section 5).
- **Every b1 row is BELOW the best published row**, for both lists, on all three settings and both models; the gaps (any / peer-reviewed)
  are -3.1 / -2.8 (zg6r) and -4.1 / -3.8 (zrckpn) on 2wiki, -2.3 / -0.9 and -3.1 / -1.8 on hotpotqa, and -12.1 / -9.7 and -13.3 / -11.0
  on musique (beside; its b1 is title mentions).
- **The ladder.** b1 - b1t is +17.3 / +17.4 points on 2wiki, +5.3 / +5.2 on hotpotqa and +2.6 (+1.6, +3.7) / +2.9 (+1.8, +4.0) on musique.
  The share of that gain that b1w recovers (section 5) is 0.974 / 0.984 on 2wiki, 0.589 / 0.580 on hotpotqa and 1.135 / 1.032 on musique,
  where b1 - b1w is -0.4 (-0.9, +0.2) / -0.1 (-0.6, +0.4). On 2wiki b1 - b1w is +0.4 (-0.3, +1.1) / +0.3 (-0.4, +0.9), both intervals
  holding 0; on hotpotqa it is +2.2 (+1.5, +2.9) for both models.
- **zg6r - zrckpn.** GNN_ABOVE on every in-domain carve of all three settings (on b1: +1.0 (+0.6, +1.4) on 2wiki, +0.9 (+0.4, +1.3) on
  hotpotqa, +1.3 (+0.4, +2.1) on musique) and on all three hotpotqa held-out carves. The musique held-out read changes sign: GNN_ABOVE on
  b1t (+2.0 (+1.1, +2.9)), MLP_ABOVE on b1w (-1.1 (-1.9, -0.1)) and on b1 (-1.9 (-2.8, -1.1)). The held-out words decide nothing.
- **Held-out reads, beside.** Hotpotqa: b1 - b1t is +5.3 (+4.3, +6.3) for zg6r and +5.3 (+4.4, +6.2) for zrckpn. Musique: -2.9
  (-4.3, -1.6) for zg6r (R@5 on b1 below b1t) and +1.1 (-0.1, +2.3) for zrckpn. Family H on the held-out cells: hotpotqa SAME for both;
  musique LESS for zg6r (-3.8 (-5.3, -2.2)) and SAME for zrckpn.
- **Musique, beside.** Family H in-domain is GREATER for both models (+1.8 (+0.4, +3.1) and +2.0 (+0.6, +3.4)); musique's b1 is title
  mentions, not corpus hyperlinks (section 4), and the stage call does not use it.

**What this does not say.** The records compare nothing with an LLM-built graph (B1L's records are its own and stay as filed), no model
was fitted on the hyperlinks (section 12 names that as not declared here), and a model's gain from b1 over b1t is not a statement about
what the links carry. The musique chain and gate ran as declared; nothing was run twice except the one 2-second lists row named above.
