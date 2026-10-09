# Where the space head's harm comes from: zgf's and zgr's fits read with the head left out

Declared 9 October 2026 at about 10:30, after rounds twenty-six and twenty-seven were filed (docs/SCREENS.md) and before
any read below exists. A diagnostic, not a round: nothing here is adopted, and no screen or run is decided by it.

## 1. Question

Round twenty-six (zgf: zrm plus a head over a learned 64-wide space, shaped by the graph's edges in training) and round
twenty-seven (zgr: zgf plus relation offsets from the relations' names) both lose on the passage graphs in L-hotpotqa's
fit, against zrm's card fit:

| read | zgf | zgr |
|---|---:|---:|
| musique in-domain | -0.0599 | -0.0474 |
| hotpotqa read zero-shot | -0.0437 | -0.0263 |
| squad in-domain | -0.0143 | -0.0141 |

zgr's L-musique fit also loses on squad (-0.0118). Two rounds show it, so it is not one fit's accident.

There are two places the harm can come from:
- **At read.** The head's score is added to zrm's, and on these graphs it ranks the gold rows lower.
- **In training.** zrm's own weights trained beside the head (the listwise loss now sees zrm's score plus the head's),
  and they drifted. The edge and chain losses touch only the head's weights, so any drift is co-adaptation.

## 2. What is read

- Nothing trains. Each fit is read on the CPU (4 threads) on the six s1eval carves, p@swa.
- **off:** the fit's own weights with the space head left out of the forward, which is zrm's forward on the fit's zrm
  weights. No weight is changed.
- **on:** one fit (zgf's L-hotpotqa) read as trained, head included, as the device control. The filed reads were on the
  card.
- **Code:** `outputs/mp_unified/zgfoff.py`; its selftest passes. On a toy carve, off scores exactly as zrm's forward on
  the same weights, and differs from on once the head's last layer is not zero. `stage` copies a fit's models.pt,
  screen.json and train.json bit for bit to `outputs/diag/zgfoff/fits/<name>`. It never writes into the fit's folder.
- **Four reads:**
  - `scr-zgf-hp-off`
  - `scr-zgr-hp-off`
  - `scr-zgr-off` (L-musique; its squad LOSS)
  - `scr-zgf-hp-on` (the control)
- **Comparisons:** lean_screen's (R@5 difference, 2,000-resample question bootstrap, filed floor 0.0075), written to
  `outputs/diag/zgfoff/<name>`.
  - Each off read against zrm's card fit of its split and against its own fit as filed.
  - The control against zgf's filed reads.

## 3. How it is called

The calls are made for each fit on the reads it lost (section 1), comparing off with zrm's card fit.
- **HEAD_AT_READ:** every one of those reads is WITHIN or GAIN. Leaving out the head recovers zrm.
- **DRIFT:** every one of those reads is still a LOSS. zrm's own weights moved.
- **BOTH:** anything else.
- **Reported beside, for each lost read:** the share of the loss that the head carries,
  (on − off) / (on − zrm), from the filed on reads.
- **The device control** must read within ±0.002 of zgf's filed L-hotpotqa reads on all six datasets. If it does not,
  the CPU/card difference is named beside every call, and the calls still stand.

## 4. What follows

- **HEAD_AT_READ:** the head needs a bound or a gate on passage graphs. Any such change is a later round, declared before
  its numbers.
- **DRIFT:** a head trained on frozen zrm weights (zrs's pattern, round fifteen) is the candidate, again a later round.
- **Neither result changes a filed verdict.** zgf and zgr stay NO_GAIN.

## 5. Order and caps

- Items `zgo-*`:
  - four stages (1 CPU, 2 GB);
  - four CPU reads (4 CPUs, 11 GB each; a CPU read peaked at 9.8 GB and 3.5 CPUs, round twenty-three);
  - four comparisons (1 CPU, 4 GB).
- They go at the top of the feeder's list, ahead of step 4e's looks. The card stays with round twenty-one's full run
  (fzrg).
- **ETA:** each read takes about 25 minutes. The calls land about 11:15 if the host's memory admits two reads at once.
- No training, no new carve, graph, text, encoder, seed or setting. The encoder and substrate embeddings stay frozen.
  webqsp never trains. Test splits are never read.

## Results
