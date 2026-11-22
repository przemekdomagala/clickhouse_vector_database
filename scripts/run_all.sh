#!/usr/bin/env bash
# Populate each dataset and run goals 1, 2, 7 (+ file check) and 8 through the standard compose services.
# Usage: scripts/run_all.sh [dataset ...]   (default: the five non-uform datasets)
set -uo pipefail
cd "$(dirname "$0")/.."

datasets=("$@")
[ ${#datasets[@]} -eq 0 ] && datasets=(cohere-small-100k nytimes-256-angular glove-200-angular pubmed-arxiv ccnews-nomic-768-normalized)
declare -A radius=([cohere-small-100k]=0.142 [nytimes-256-angular]=0.604 [glove-200-angular]=0.563
                   [pubmed-arxiv]=0.269 [ccnews-nomic-768-normalized]=0.198)

status=0
run() { echo "+ DATASET=$DATASET $*"; docker compose run --rm -T "$@" || { echo "FAILED: $*"; status=1; }; }

for DATASET in "${datasets[@]}"; do
    export DATASET
    echo "===== $DATASET ====="
    [ "${SKIP_POPULATE:-0}" = 1 ] || run populate
    run benchmark db 1 1000 10
    run benchmark db 2 1000 "${radius[$DATASET]}"
    run benchmark db 7 1000 98999
    run benchmark python3 /app/scripts/check_goal7.py
    run benchmark db 8 1000 10
    run benchmark db 8 1000 100
done
exit $status
