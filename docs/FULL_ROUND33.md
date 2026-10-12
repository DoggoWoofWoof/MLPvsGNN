# Full run: twice the training steps, for the MLP's base and the GNN track's base

Declared 10 October 2026 at about 08:55. Round thirty-three's screen re-called PROMISING at 08:47 (docs/SCREENS.md,
thirty-third round). No fit, read or number of this run exists yet. These are development numbers. The paper's numbers
come from one declared confirmation run.

## 1. Question

Round thirty-three's rule: a PROMISING zep earns a full run, and that run applies the same doubling to the GNN track's
base (zsp), so neither model gets more training than the other.

Both arms below train their base's model on their base's carves, at twice the base's epochs:
- the config changes from `2e-3:1e-4:0.1:8:2` to `2e-3:1e-4:0.1:16:4` (lr:wd:dropout:epochs:swa_from);
- SWA starts at twice the base's first SWA epoch, so the same share of the run is averaged.

Does the doubled run lift each base's reads on every split, with nothing else losing? The splits are J5 (in-domain on
all five training datasets) and each leave-one-out fit on its held-out dataset.

## 2. Arms and fits, all on the card

All commands are `outputs/mp_unified/zepf.py` (new; frozen once run).

**zep: the MLP's base zrc, doubled.** This is zep.py's arm, unchanged (zrm.ZRM over zrc.ChainCarveZRC). No message
passing.
- Four fits: L-2wiki, L-squad, J5 and L-metaqa.
  `zepf.py train --arm zep --split S --name S --out-root outputs/full_zep/fits --device cuda --host`.
- The screen's fits, `scr-zep` and `scr-zep-hp`, are its L-musique and L-hotpotqa fits.

**zspx: the GNN track's base zsp, doubled.** zprop.ZProp over zlink.LinkCarveBase: zsp's model, carve and edges.
- Six fits, one per split.
  `zepf.py train --arm zspx --split S --name S --out-root outputs/full_zspx/fits --device cuda --host`.

**zsp's base fits on the card.** zsp's full run trained on the host's CPU (docs/FULL_ROUND23.md). Round thirty-one
refit it on the card on L-musique and L-hotpotqa only (`scr-zspg`, `scr-zspg-hp`).
- So that both sides of every zspx comparison run on one device, zsp at its own config is refit on the card on the
  other four splits (zg1.py's base).
  `zepf.py base --split S --name S --out-root outputs/full_zspg/fits --device cuda --host`.

**Common to every fit:**
- seed 0;
- p@swa;
- rmatch.py's train;
- each model's own builds;
- no new row, column, pool, graph, edge or score.

## 3. Reads and comparison

- Each fit is read on the six s1eval carves (`zepf.py read`).
- **Each read is compared, question by question, with its base's fit of the same split.** p@swa, on the same carves.
  That comparison decides.
  - **zep against zrc:** `outputs/full_zrct/fits/S`, and `scr-zrct` / `scr-zrct-hp` on L-musique / L-hotpotqa.
  - **zspx against zsp's card fits:** `outputs/full_zspg/fits/S`, and `scr-zspg` / `scr-zspg-hp` on L-musique /
    L-hotpotqa.
  - Step 1's fit is reported beside each.
- **The comparison** is lean_screen's: the R@5 difference with a 2,000-resample question bootstrap, called GAIN, LOSS
  or WITHIN with the floor 0.0075. hit@1 is reported beside it.
- **zsp's four new card fits are also compared with step 1's fits** (`outputs/full_zspg/base-zspg-S.json`). These give
  the re-grade its base R@5 (as `outputs/zg1/base-zspg-S.json` does for round thirty-one's two).
- Each arm has 36 reads. Primary reads are step 1's eleven: J5 on all six datasets (five in-domain, webqsp zero-shot),
  and each leave-out fit on its held-out dataset.

## 4. Verdict, for each arm on its own

- **`zepf.py grade --arm A`** files relz.py's grade, decided against the arm's base fits:
  - for zep, `--reuse L-musique=outputs/screen/scr-zep.json,L-hotpotqa=outputs/screen/scr-zep-hp.json`;
  - for zspx, its six comparisons.
- **The re-grade decides.** `nullx.py regrade` re-grades all 36 reads under the seed null over every split into
  `outputs/full_zep/grade-nullx` and `outputs/full_zspx/grade-nullx`. Each read's base R@5 is its base fit's:
  - zrc's `outputs/zbase2/base-zrc-S.json`;
  - zsp's `outputs/zg1/base-zspg-S.json` (L-musique, L-hotpotqa) and `outputs/full_zspg/base-zspg-S.json`.
- **ADOPT:** at least one primary read GAINs, and none of the 36 reads LOSEs.
- **NOT_ADOPTED:** otherwise.
- **INCOMPLETE:** a comparison is missing or is not the declared one (exit 1).

**What ADOPT means.**
- **Each model takes its own verdict.** zep ADOPT makes 16 epochs zrc's config, and zspx ADOPT makes it zsp's. Each
  model then trains at the step count that its own grade found best: both are at their peak, and neither is handed
  steps the other was refused.
- Whatever the verdicts, the paper states each model's epochs.
- **Rounds and runs declared before this verdict keep their config** and their decisions as declared: U1c, the U1c
  rerun on GU_1024, and round thirty-three's own records.
  - A later round that trains zrc or zsp uses the adopted config.
  - A combination of an adopted config with U1c's graph is a run of its own.

**Every fit runs to its end.** A LOSS filed at the floor 0.0075 may turn WITHIN under the null, so nothing is stopped
early.

## 5. Gates and order of work

- **`fep-gate`** (`zepf.py gate --recall outputs/screen/scr-zep-pair-recall.json`) exits 0 only when round
  thirty-three's re-call is PROMISING. It is (08:47). Every fit waits on it.
- **Each read waits for its fit. Each comparison waits for its read and its base's read.**
- **The items go at the end of the feeder's list, after U1c's.**
  - The card is idle now: U1c's fits wait on its looks and caches. These fits start at once.
  - U1c's fits stay ahead in file order when they are ready.
  - The MLP's items come before the GNN track's.
- **Caps.** Every fit and read runs under cuda_alloc (frac 0.3, share 0.32), so three share the card. The screen's
  torch peaks were 5.16 GB at most.
- **ETAs.** Fourteen fits.
  - The screen's doubled fits took 1 h 17 min (L-musique) and 1 h 16 min (L-hotpotqa); J5 should take about 1.5 h.
  - zsp's card fits took 28 to 38 min at its own config, so about an hour doubled.
  - With three on the card: the zep grade around 13:00, the zspx grade around 16:00 to 17:00.

## 6. What this does not do

- No new graph, column, text, encoder or model. The encoder and the substrate embeddings stay frozen.
- No new seed, variant or hyperparameter other than the epoch count and SWA start, which keep their ratio.
- Nothing is written into zrc's, zsp's or step 1's fit folders. zsp's card fits are new folders of their own.
- webqsp never trains. Test splits are never read.
- **Speed.** A model trained longer has the same forward. Its cold latency is the base's, from the same code.

## Results

### zep (zrc, 16 epochs): NOT_ADOPTED (12 October 2026, 13:02)

The filed grade (outputs/full_zep/grade.json) is ADOPT: 2 of 11 primary reads GAIN, 0 of 36 LOSE. The re-grade under
the seed null over every split decides (section 4), and it is **NOT_ADOPTED** (outputs/full_zep/grade-nullx.json):

- **0 primary GAINs, 0 LOSSes in 36.** All 36 reads sit under the null's floors.
- Five calls change from GAIN to WITHIN: webqsp zero-shot on J5 (+0.020, floor 0.030), L-squad (+0.015, floor 0.026),
  L-hotpotqa (+0.023, floor 0.023) and L-2wiki (+0.012, floor 0.047), and metaqa zero-shot on L-metaqa (+0.012,
  floor 0.017).
- L-musique's webqsp zero-shot read stays a GAIN (+0.012 over a floor of 0.010). It is not a primary read.
- In-domain reads move by at most 0.007 either way.

**So zrc keeps 8 epochs.** Doubling the steps never hurts, and it lifts webqsp zero-shot on every split (+0.002 to
+0.023), but no lift clears what reseeding the same fit moves. webqsp zero-shot goes to the gap ledger as a direction
(training length helps transfer to the large KB), not as a result.

### zspx (zsp, 16 epochs): NOT_ADOPTED (12 October 2026, 18:03)

The filed grade against zsp's card fits (outputs/full_zspx/grade.json) is NOT_ADOPTED: 2 of 11 primary reads GAIN, and
4 of 36 LOSE. The re-grade under the seed null over every split (outputs/full_zspx/grade-nullx.json) is also
**NOT_ADOPTED**:

- **0 primary GAINs, 1 LOSS in 36.** The LOSS is webqsp zero-shot on L-musique (-0.0125, floor 0.0102). It is not a
  primary read.
- Six calls change, all to WITHIN:
  - the two primary GAINs: metaqa zero-shot on L-metaqa (+0.0130, floor 0.0168) and hotpotqa zero-shot on L-hotpotqa
    (+0.0086, floor 0.0200);
  - three LOSSes: webqsp zero-shot on J5 (-0.0099, floor 0.0295) and on L-squad (-0.0187, floor 0.0260), and musique
    zero-shot on L-musique (-0.0086, floor 0.0720);
  - one GAIN: webqsp zero-shot on L-hotpotqa (+0.0174, floor 0.0233).
- L-musique's 2wiki read stays a GAIN (+0.0083 over a floor of 0.0075). It is not a primary read.
- The 25 in-domain reads move by -0.003 to +0.008. The largest is that 2wiki GAIN.

**So zsp keeps 8 epochs.** Unlike zrc's, the GNN's webqsp zero-shot reads go both ways when the steps double (-0.019
to +0.017). For the GNN, training length is not a direction for transfer to the large KB.

### What round 33 settles

Both bases keep their 8-epoch configs: zrc (the MLP's) and zsp (the GNN track's). Neither doubling moves a primary
read past what reseeding the same fit moves. The MLP's webqsp zero-shot lift from longer training goes to the gap
ledger as a direction, not a result.
