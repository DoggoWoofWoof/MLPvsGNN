# B1M: the MLP of record (zrckpn) on HippoRAG 2's settings, beside the GNN of record (zg6r)

Declared 4 November 2026, before any B1M read. Follows [B1P_PKO_CHAIN_ON_B1.md](B1P_PKO_CHAIN_ON_B1.md) (PKO's chain on B1's
three settings, read by ML1's zg6r and zrckp; filed in d9570e4) and [LTM_NEXT_HOP_MLP.md](LTM_NEXT_HOP_MLP.md) (zrckpn: zrckp with
LTW's seven next-hop columns, ADOPT in 7874e4d). B1P's section 12 set the rule: a model that replaces zg6r or zrckp before the
6-7 November freeze is read on B1 with B1P's chain, in a stage of its own. zrckpn replaced zrckp. Code:
`outputs/mp_unified/zb1m.py` (new). Outputs only under `outputs/b1m/`.

Nothing trains. LTM's scorers and zrckpn's six fits are read and never written. B1P's chain, gate and reads are read and never
written. B1 is never used for selection: B1M decides nothing about any model, pool, rule or column. It is a measurement.

## 1. Why

B1P read zrckp, not zrckpn. So Table 1 has no row for the MLP of record, and the two models of record have not been compared on
B1. B1M measures four things:

1. **zrckpn's Table 1 row:** its in-domain R@5 on b1w for each setting, against the best published rows.
2. **The two models of record on B1:** zg6r - zrckpn, paired by question, in R@5 points.
3. **The rule's carry-over for zrckpn:** the paired b1w - b1t.
4. **Beside, deciding nothing:** zrckpn - zrckp on b1w, what LTM's columns add on B1.

## 2. What is read

- **Carves.** B1P's six carves: b1w and b1t of musique, 2wiki and hotpotqa (1,000 questions each). B1P's chain, roots and gate are
  used on the host, unchanged: `outputs/b1p/{cache,links,kt,look,pools,records}` and `outputs/b1p/gate.json` (READY).
- **The MLP.** LTM's six zrckpn fits: `scr-ltm-zrckpn-s0`, `-s1` and `-s2`, each under WQ-musique and WQ-hotpotqa (suffix `-hp`).
  Each is forked to `outputs/b1m/fits/<w|t>/<fit>` without its reads, with `models.pt` sha-checked against the source.
- **The GNN and zrckp.** B1P's reads of zg6r and zrckp (`outputs/b1p/fits/<w|t>/scr-ml1-{zg6r,zrckp}-s*/reads/`), read only.

## 3. What runs (`zb1m.py`)

| step | what |
|---|---|
| `lists` | `ltmcols.lists` on each B1P carve, with root `outputs/b1p` and out `outputs/b1m/ltm`. These are LTM's T and A lists, built from B1P's cache and links parts. |
| `cols` | `ltmcols.cols` for each split and carve, out `outputs/b1m/ltm`, with two in-process patches restored afterwards. `col_carves` also names the B1 carve, and `load_folds` reads LTM's own five fold fits per split (`outputs/ltm`, sha-checked against LTM's fit record). A B1 carve is no training carve, so each question's columns come from the fold fit its index draws (`Q_hfold`), as on s1eval. The scorer and every setting are LTM's, unchanged. |
| `fork` | The six fits, once per variant: twelve forks. |
| `read` | zltm.read's body under `zltm.on_ltm` (LTM's names and stamp key; DOB's CPU device: six threads, the GPU hidden, torch 2.8.0+cpu), with zb1p.read's patches. ML1's roots and gate become B1P's. zp1c3's eval carves become the variant's three B1 carves. rmatch's read takes `--read musique=c,2wiki=c,hotpotqa=c --out-root outputs/b1m/fits/<v>` (c = b1w or b1t). `zltm.on_split(split, the fit's scorer, out=outputs/b1m/ltm)` makes the carves load B1M's columns, checked by `zltm.load_next` against B1P's cache parts and the scorer the fit was trained with. The device check is LTM's: the fit's own device. No LTM or ML1 folder is written: no `zml1s.stamp` is called. `b1m_read.json` is written beside the reads. |
| `report` | Section 4: `outputs/b1m/report.json` and `report.md`. |

## 4. The report (fixed now)

- **Metrics.** B1b's candidate p@swa, as B1P: R@5 per question, with FC@5 and hit@1 beside. 95% bootstrap intervals over the
  setting's 1,000 questions (B1b's: 2,000 resamples, seed 0).
- **In-domain read of a setting** (B1P's):
  - musique: WQ-hotpotqa;
  - hotpotqa: WQ-musique;
  - 2wiki: both splits.

  Per model, each question's R@5 is averaged over the in-domain fits (three seeds; six fits on 2wiki), and the interval is taken
  over the questions. Zero-shot reads are beside: musique by WQ-musique, hotpotqa by WQ-hotpotqa.
- **Table 1 row.** zrckpn's in-domain b1w R@5 against the best published row of the setting (`b1f.PUBLISHED`), of any kind and
  peer-reviewed only: ABOVE, AT or BELOW by the interval (`b1b.call`), with the gap in points.
- **The two models of record.** zg6r - zrckpn, each question's in-domain mean against the other's over the same splits and seeds,
  with its interval. The label is by the interval: **GNN_ABOVE** when its bottom is above 0, **MLP_ABOVE** when its top is below 0,
  **LEVEL** otherwise. Reported on b1w (the Table 1 carve) and b1t, in-domain, with the zero-shot reads beside. Beside it, each fit
  pair's difference (zg6r against zrckpn at the same seed and split), as C1R reports seeds.
- **Carry-over.** zrckpn's paired b1w - b1t, in-domain and zero-shot, with its interval.
- **Beside, deciding nothing.** zrckpn - zrckp on b1w and b1t, paired, in-domain.
- **Training overlap.** B1P found 0 of each setting's questions in the in-domain training carves. zrckpn trains on the same
  carves (WQ's), so the report copies B1P's counts and checks that LTM's fits name WQ's training carves.

## 5. Deviations, stated before any number

1. **The columns carry over.** LTM's scorers were fitted on the full corpora's training carves. B1's pools are smaller (B1P,
   section 7), so the columns see other anchors and candidates than on s1eval.
2. **The column records keep LTM's stage and declaration.** `ltmcols` writes them, and `zltm.load_next` refuses any other.
   B1M's own record (`outputs/b1m/cols.json`) names each column record's sha256 under this declaration.
3. **Each model is read on its own device.** zrckpn's fits were trained on the host's CPU (DOB's device), so they are read there.
   B1P read zg6r on the host's GPU.
4. **Threads as LTM's.** The lists and columns run as LTM's items did: one and two CPUs, BLAS at rx's per-job value, LTM's two
   torch threads. The reads run at zltm's six threads.

## 6. Risks

- **An identity failure in `cols`** (B1P's cached dense_cos against the frozen vectors) stops that carve. It is reported, not
  patched.
- **A device mismatch** (the host's CPU torch changed since LTM's fits) stops the read before any number.
- **Code fixes** before any read number are allowed and are disclosed in section 10. After the first read number, only an
  amendment changes this file.

## 7. Commit policy

Commit only `outputs/b1m/report.json`, `report.md` and `cols.json`, masked with `mask_host.py`, with `git add -f`. Never commit the
lists, columns, forks or reads.

## 8. Files

    outputs/mp_unified/zb1m.py                         the stage (new)
    outputs/b1m/ltm/{work/lists,lists,cols}/           LTM's lists and columns on B1P's carves
    outputs/b1m/cols.json                              the column records' shas, under this declaration
    outputs/b1m/fits/<w|t>/<fit>/                      the forks, their reads and b1m_read.json
    outputs/b1m/report.json, report.md                 section 4's tables

## 9. Queue (the host)

| items | CPUs | GB | after |
|---|---:|---:|---|
| `b1m-selftest` | 1 | 4 | - |
| `b1m-lists-<dataset>-<carve>` (six) | 1 | 6 | the self-test |
| `b1m-cols-<wqm or wqh>-<dataset>-<carve>` (twelve) | 2 | 6 | the carve's lists |
| `b1m-fork` | 1 | 4 | the self-test |
| `b1m-read-<fit>-<w or t>` (twelve) | 6 | 12 | the fork, and the fit's split's three column sets of the variant |
| `b1m-report` | 1 | 6 | every read |

## 10. What comes next, named now

- **The hyperlink carve.** The models of record on b1, with the setting's hyperlinks as the structural graph (B1a's). PKO's chain
  needs U2m (a)'s EM and U2p's score type on every structural entry, and no stage has computed them for hyperlink entries. That is a
  stage of its own if declared. Until then, the hyperlink rows stay B1b's and B1f's (J5 fits), and HLC's.
- **Development fixes** made before any read number are listed here.
  - *Lists' memory (4 Nov).* The six `b1m-lists-*` items reserve 2 GB on the host, not section 9's 6: LTM's own lists peaked at
    1.3 GB.
  - *B1P's removed files restored (4 Nov).* On 3 Nov at 19:31 the host's clean-up (an urgent pass at 59.7 GB free) removed 240 of
    B1P's regenerable files: every look chunk of the six carves, and 28 cache arrays (musique b1w's `node` and hotpotqa-store
    arrays in all four parts; the hotpotqa store's `dists` and `tab` in the other four carves). B1P's reads had ended and B1M was
    not yet queued, so no queued job pinned them. `b1m-lists-musique-b1w` then failed on the missing `node` array, twice with the
    feeder's retry, and its eight dependents dropped. B1P's chain skips any step whose record exists, so running it again rebuilds
    nothing. `outputs/mp_unified/b1prepair.py` runs B1P's own steps (look, bases, cache; `--host`, the chain's 2-thread
    environment) on a scratch root under `outputs/b1m/repair`. It copies back exactly the removed files, and only if all of these
    hold: every rebuilt cache array of the six carves has the file sha256 that B1P's part record pins; every rebuilt look record
    agrees with B1P's on every field but time, seconds and memory; every chunk B1P still holds is byte-identical to its rebuilt
    twin; and nothing under `outputs/b1p` changed during the rebuild. These are B1M's only writes outside `outputs/b1m`. Each
    restored file is checked after the copy, no existing file is overwritten and no record changes, so section 2's roots are again
    what B1P's records say. B1M's six t-reads wait on the restore. The musique-b1w lists, two column sets and six w-reads run again
    under new item names (`-r`), and `b1m-report-r` reads all twelve.
