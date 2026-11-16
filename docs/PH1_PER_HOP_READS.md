# PH1: R@5 and FC@5 by number of hops (descriptive, post hoc)

Declared 16 November 2026, before `outputs/mp_unified/zph1.py` reads any real file. Requested on 16 Nov 2026: "with three
relevant paragraphs all three must be in the top 5; validate by number of hops (MuSiQue has this)". **Nothing trains, no
model reads anything new, and no filed read, fit or record changes.** PH1 splits reads already filed by the kind of question.
It has no verdict and no call; it decides nothing.

## 1. What is read

- The questions: B1a's three settings (`outputs/bench/hipporag2/<d>/queries.json`, 1,000 each), with each question's `kind`
  and gold count.
- The per-question reads of record (each an npz with `ids`, `gold_total`, `top` per candidate, `ref_top` for RRF), the
  candidate `p@swa`, the files the filed records pin (section 4, G2):
  - GNN zg6r and MLP zrckpn on b1w (B1P's and B1M's reads) and b1 (B1H's), and GNX on b1w (IDPA's) and b1 (IDPB's): every
    fit at seeds 0-2 on both training splits (WQ-musique; WQ-hotpotqa, suffix `-hp`).
  - J5 zrc and zsp on b1 (B1b's) and b1w (B1F's) at seed 0, and seeds 1 and 2 from J5S (`outputs/j5s/fits/<carve>/<arm>/J5-s1`,
    `J5-s2`; read only, under the J5S hold), each file checked against `outputs/j5s/report.json`'s `files_sha256[seed]`.
  - RRF: the `ref_top` row "rrf" of the reads (equal in every read of a setting, checked).
- b1t (the graph without a title-link family) is not read: it is a read-time shift, not a domain.

## 2. Groups

| setting | groups (from `kind`; sizes) |
| --- | --- |
| MuSiQue | hops 2 (`2hop`, 518), 3 (`3hop1`, `3hop2`: 316), 4 (`4hop1`-`4hop3`: 166); the six kinds beside |
| 2Wiki | compositional (413, 2 golds), inference (108, 2), comparison (244, 2), bridge_comparison (235, 4) |
| HotpotQA | bridge (811), comparison (189) |

The two passages of a comparison question need no link between them (AP1 A1 (i)); the chain kinds are MuSiQue's hops,
2Wiki's compositional and inference, and HotpotQA's bridge.

## 3. The numbers

- Per question: R@5 = golds in the top 5 / gold count; FC@5 = 1 when every gold is in the top 5 (b1b.per_question's
  definitions).
- A row's per-question value is the mean over its fits: the in-domain fits (zb1h.IN_DOMAIN: MuSiQue reads the WQ-hotpotqa
  fits, HotpotQA the WQ-musique fits, 2Wiki both) at seeds 0-2 for zg6r, zrckpn and GNX; seeds 0-2 of the one J5 fit for
  zrc and zsp. Seed 0's value is reported beside (on 2Wiki the mean of the two splits' seed-0 fits), per the three-seed
  record rule.
- Per group: the mean of R@5 and of FC@5 with a 95% interval of 2,000 question resamples (b1b.boot_ci, seed 0), and the
  row minus RRF on the same questions with the paired interval.
- Held-out reads are not split (they are not the record).

## 4. Gates (the run stops on any failure)

- G1: group sizes as in section 2 (sum 1,000 per setting); every read holds the setting's questions in B1a's order with
  B1a's gold counts; the RRF row is equal in every read of a setting.
- G2: every read's sha256 equals the file_sha256 its filed record holds (B1H's fits for zg6r and zrckpn on b1 and b1w;
  IDPA's for GNX on b1w; J5S's `files_sha256[seed]` for zrc and zsp, seeds 0-2). IDPB files no per-file sha256 for GNX on
  b1: those files' sha256 are recorded here and they pass G3 fit by fit.
- G3: each row's all-question R@5 (and FC@5 where filed) equals the filed record's, to 4 decimals: B1H in-domain zg6r and
  zrckpn on b1w and b1; IDPA in-domain GNX on b1w; IDPB in-domain GNX on b1 (R@5 and each fit's R@5); J5S's three-seed means
  for zrc and zsp on b1 and b1w.
- The self-test (toy reads with hand-worked groups, means and differences) passes before the real run.

## 5. Disclosure

Post hoc and descriptive. Earlier stages already split MuSiQue by kind or by hop depth (B1C's R@5 by kind for an older
model, AP2's and AP3's differences by hop depth, where AP3 saw the MLP's fall on three-hop questions). No group, model or
setting here was chosen after reading this table. A difference between groups is not a test of anything: the groups differ
in their questions as well as in their hops.

## 6. Outputs

`outputs/ph1/report.json` and `report.md` (committed), with the script's sha256, every read's sha256 and the gates' results.

## 7. Results (filed 16 November 2026; `outputs/ph1/report.json`, `report.md`)

**The run.** `zph1.py run` at the declared commit (eac51a1): 108 reads (30 on MuSiQue and HotpotQA, 48 on 2Wiki), G1, G2 and
G3 held on all three settings (every row's all-question R@5 and FC@5 equal to B1H, IDPA, IDPB or J5S to 4 decimals; GNX's b1
files, which IDPB does not pin, matched its per-fit R@5). Descriptive: no call.

**MuSiQue (in-domain, three-seed means; R@5 / FC@5 in %; FC@5 minus RRF in points, paired 95% interval):**

| row | 2 hops (518) | 3 hops (316) | 4 hops (166) | all |
| --- | ---: | ---: | ---: | ---: |
| RRF | 65.2 / 36.7 | 52.8 / 7.3 | 35.7 / 0.0 | 56.4 / 21.3 |
| GNN zg6r, b1w | 79.0 / 61.7 (+25.0 [+20.6, +29.6]) | 64.2 / 22.8 (+15.5 [+11.5, +19.9]) | 46.6 / 2.6 (+2.6 [+0.8, +5.0]) | 69.0 / 39.6 |
| MLP zrckpn, b1w | 77.9 / 60.3 (+23.6) | 62.8 / 18.6 (+11.3) | 43.5 / 3.4 (+3.4) | 67.4 / 37.7 |
| GNX, b1w | 79.1 / 62.3 (+25.6) | 64.3 / 22.5 (+15.2) | 46.6 / 3.8 (+3.8) | 69.1 / 40.0 |
| J5 zrc, b1 | 76.4 / 57.6 (+20.9) | 61.7 / 17.4 (+10.1) | 43.1 / 1.6 (+1.6) | 66.3 / 35.6 |

- R@5 minus RRF shrinks with hops in every row: zg6r and GNX +13.6 to +14.3 points at 2 hops, +10.8 to +11.5 at 3 and
  +10.1 to +10.9 at 4; zrckpn +12.7 / +9.5 to +9.9 / +7.7 to +7.8; J5 zrc and zsp +10.5 to +11.7 / +7.9 to +8.9 / +6.2 to
  +7.4 (all rows on b1w and b1 in `report.md`). The share of questions with every gold in the top 5 falls from about 60% at
  2 hops to about 20% at 3 and 2-4% at 4. A 4-hop question has 4 golds, so the top 5 leaves one slot spare.
- **Correction (16 Nov, before any use):** the first filing (cafed99, its commit message too) said every row lifts R@5 by
  about 11-14 points at each hop count; that held for zg6r and GNX only, and the bullet above replaces it.
- **2Wiki** (b1w, zg6r): compositional 94.7 / 89.4, inference 93.8 / 87.6, comparison 100.0 / 100.0, bridge_comparison (4
  golds) 90.5 / 70.0; RRF's bridge_comparison FC@5 is 0.4%.
- **HotpotQA** (zg6r): bridge 93.0 / 86.4 on b1w and 95.7 / 91.7 on b1; comparison 99.9 / 99.8 on both.
- The full tables (every row on b1w and b1, seed 0 beside, MuSiQue's six kinds) are in `report.md`.

What it says, read and not tested: the gap left on these settings is the third and fourth hop of MuSiQue's chains (and
2Wiki's four-gold bridge_comparison), which is where chain-completing selection (the plan's item 4a) would act. It decides
nothing.
