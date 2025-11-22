#!/bin/bash
# run_batch_training.sh - Wrapper for batch training

CONFIG_FILE=${1:-"configs/informer.yaml"}
NUM_GPUS=${2:-8}
OUTPUT_DIR=${3:-"./batch_results"}

echo "=========================================="
echo "Batch Training: 12-Month Models"
echo "=========================================="
echo "Config: $CONFIG_FILE"
echo "GPUs: $NUM_GPUS"
echo "Output: $OUTPUT_DIR"
echo "=========================================="

python batch_train.py \
    --config $CONFIG_FILE \
    --num_gpus $NUM_GPUS \
    --output_dir $OUTPUT_DIR \
    --year 2023

echo "=========================================="
echo "Batch training complete!"
echo "=========================================="