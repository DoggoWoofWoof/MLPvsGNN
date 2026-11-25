#!/bin/bash
# PL5 laptop driver (docs/PL5_COVERAGE_ROUND4.md section 11): one dataset's G-R, dev grid in three parallel parts (frozen lists), select, frozen confirmation (and e3v for MultiHop-RAG).
# usage: bash outputs/pl5/run_ds.sh DATASET [no-gr] [no-confirm]
set -u
cd "$(dirname "$0")/../.." || exit 2
ds=$1
shift
export PYTHONIOENCODING=utf-8
Z=outputs/mp_unified/zpl5g.py
L=outputs/pl5/run_${ds}.log
{
  echo "[$(date +%H:%M:%S)] $ds start"
  if [[ " $* " != *" no-gr "* ]]; then
    python $Z gr --ds "$ds" || { echo "[$(date +%H:%M:%S)] $ds G-R FAILED"; exit 3; }
  fi
  if [ ! -f outputs/pl5/$ds/select.json ]; then
    for i in 0 1 2; do
      python $Z grid --ds "$ds" --carve dev --first frozen --part $i 3 > outputs/pl5/run_${ds}_g$i.log 2>&1 &
    done
    wait
    for i in 0 1 2; do
      [ -f outputs/pl5/$ds/grid_dev_frozen.part${i}of3.json ] || { echo "[$(date +%H:%M:%S)] $ds part $i missing"; exit 4; }
    done
    echo "[$(date +%H:%M:%S)] $ds grid parts done"
    python $Z select --ds "$ds" || { echo "[$(date +%H:%M:%S)] $ds select FAILED"; exit 5; }
  fi
  if [[ " $* " != *" no-confirm "* ]]; then
    python $Z confirm --ds "$ds" --first frozen || { echo "[$(date +%H:%M:%S)] $ds confirm FAILED"; exit 6; }
    if [ "$ds" = "multihoprag" ]; then
      python $Z confirm --ds "$ds" --first e3v || { echo "[$(date +%H:%M:%S)] $ds e3v FAILED"; exit 7; }
    fi
  fi
  echo "[$(date +%H:%M:%S)] $ds done"
} >> "$L" 2>&1
