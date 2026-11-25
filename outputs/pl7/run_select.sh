#!/bin/bash
# PL7 laptop chain: the rule at every ceiling on dev, one dataset at a time (G-C controls are printed first by select itself).
cd "$(dirname "$0")/../.." || exit 1
export PYTHONIOENCODING=utf-8
for d in metaqa musique webqsp multihoprag squad hotpotqa 2wiki; do
  echo "=== select $d $(date +%H:%M:%S)" >> outputs/pl7/run_select.log
  python outputs/mp_unified/zpl7g.py select --ds $d > outputs/pl7/select_$d.log 2>&1
  echo "=== $d exit $? $(date +%H:%M:%S)" >> outputs/pl7/run_select.log
done
echo "=== all done $(date +%H:%M:%S)" >> outputs/pl7/run_select.log
