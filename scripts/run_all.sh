#!/bin/sh
# Pilot -> learning rate by the pilot rule -> 5 main seeds -> tables and figures.
# About 3 hours on an Apple M6 (pilot ~25 min, base scoring ~15 min, ~27 min per seed).
set -e
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
$PY scripts/run_experiments.py --pilot --lr 1e-4
LR=$($PY -c "import json; print(json.load(open('results/pythia-160m-deduped-lr0.0001-pilot/pilot_lr0.0001.json'))['decision_lr'])")
echo "main runs with lr $LR"
$PY scripts/run_experiments.py --lr "$LR" --seeds 0 1 2 3 4
TAG=$($PY -c "print(f'pythia-160m-deduped-lr{float(\"$LR\"):g}')")
$PY scripts/analyze.py --tag "$TAG"
