#!/usr/bin/env bash
set -euo pipefail

ADA_EF_ROOT="${1:-${ADA_EF_ROOT:-$(pwd)}}"
EXPERIMENTS_ROOT="${ADA_EF_ROOT}/experiments"

mkdir -p "$EXPERIMENTS_ROOT"

dirs=(
  data
  index
  estimation_table
  sampling
  statistics
  ablation_distance_size
  ablation_sampling_size
  ablation_decay_func
  incremental_update
  incremental_deletion
)

for dir in "${dirs[@]}"; do
  mkdir -p "${EXPERIMENTS_ROOT}/${dir}"
done

for batch in 10percent 50percent; do
  mkdir -p "${EXPERIMENTS_ROOT}/incremental_update/${batch}"
  mkdir -p "${EXPERIMENTS_ROOT}/incremental_deletion/${batch}"
done

printf 'Experiment folders prepared under %s\n' "$EXPERIMENTS_ROOT"