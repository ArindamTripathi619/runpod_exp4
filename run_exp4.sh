#!/bin/bash
# Wrapper script for running Experiment 4 (Layer Ablation Study)

set -e

TRIALS=${TRIALS:-5}
DB_PATH=${DB_PATH:-exp4_results.db}
LOG_FILE=${LOG_FILE:-exp4_execution.log}
OUTPUT_DIR=${OUTPUT_DIR:-results}
SEED=${SEED:-42}

echo "=========================================="
echo "Running Experiment 4: Layer Ablation Study"
echo "=========================================="
echo "Trials: $TRIALS"
echo "Database: $DB_PATH"
echo "Log file: $LOG_FILE"
echo "Output directory: $OUTPUT_DIR"
echo "Random seed: $SEED"
echo "=========================================="

python3 run_experiments_corrected.py \
    --experiment 4 \
    --trials $TRIALS \
    --db-path $DB_PATH \
    --log-file $LOG_FILE \
    --output-dir $OUTPUT_DIR \
    --seed $SEED \
    --no-confirm
