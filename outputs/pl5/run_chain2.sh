#!/bin/bash
# Committed copy: the laptop checkout path in the wait command below is shown as <repo>; nothing else differs from the script that ran.
# PL5 laptop chain, second version (25 Nov 15:45): the coordinator asked for MuSiQue's dev grid right after WebQSP's, so the host's PL6 rows can start sooner.
# The first chain (run_chain.sh) was stopped while WebQSP's three grid parts were running (the parts were left running); this one waits for them, then:
#   WebQSP select (seconds) -> MuSiQue dev grid + select (no G-R: it PASSED at 13:34; no confirmation yet) -> WebQSP frozen confirmation -> MuSiQue frozen confirmation
#   -> SQuAD dev grid + select (its frozen confirmation is run by hand as confirm-par, 4 slices).
# No number, rule or cell changes: only the order of the same commands.
set -u
cd "$(dirname "$0")/../.." || exit 2
export PYTHONIOENCODING=utf-8
Z=outputs/mp_unified/zpl5g.py
L=outputs/pl5/run_chain.log
{
  echo "[$(date +%H:%M:%S)] chain2 start (waiting for WebQSP's three grid parts)"
  bash tools/laptop/bounded_wait.sh pl5-chain2-wq 14400 30 -- bash -c 'cd <repo>; [ -f outputs/pl5/webqsp/grid_dev_frozen.part0of3.json ] && [ -f outputs/pl5/webqsp/grid_dev_frozen.part1of3.json ] && [ -f outputs/pl5/webqsp/grid_dev_frozen.part2of3.json ]' \
    || { echo "[$(date +%H:%M:%S)] WebQSP parts not there in time"; exit 4; }
  echo "[$(date +%H:%M:%S)] webqsp grid parts done"
  python $Z select --ds webqsp || { echo "[$(date +%H:%M:%S)] webqsp select FAILED"; exit 5; }
  bash outputs/pl5/run_ds.sh musique no-gr no-confirm
  [ -f outputs/pl5/musique/select.json ] || { echo "[$(date +%H:%M:%S)] musique select missing"; exit 6; }
  echo "[$(date +%H:%M:%S)] musique grid and select done"
  bash outputs/pl5/run_ds.sh webqsp
  bash outputs/pl5/run_ds.sh musique no-gr
  bash outputs/pl5/run_ds.sh squad no-confirm
  echo "[$(date +%H:%M:%S)] chain2 end"
} >> "$L" 2>&1
