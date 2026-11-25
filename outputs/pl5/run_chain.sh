#!/bin/bash
# PL5 laptop chain (one heavy dataset at a time): MuSiQue's G-R first (the host rows wait for gr_musique.json), then the short datasets, then the long ones.
set -u
cd "$(dirname "$0")/../.." || exit 2
export PYTHONIOENCODING=utf-8
L=outputs/pl5/run_chain.log
{
  echo "[$(date +%H:%M:%S)] chain start"
  python outputs/mp_unified/zpl5g.py gr --ds musique || { echo "musique G-R FAILED"; exit 3; }
  bash outputs/pl5/run_ds.sh multihoprag
  bash outputs/pl5/run_ds.sh metaqa
  bash outputs/pl5/run_ds.sh webqsp
  bash outputs/pl5/run_ds.sh musique no-gr
  bash outputs/pl5/run_ds.sh squad no-confirm
  echo "[$(date +%H:%M:%S)] chain end"
} >> "$L" 2>&1
