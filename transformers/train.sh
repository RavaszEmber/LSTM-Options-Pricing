#!/bin/bash

# Multi-GPU training script for transformer-based option pricing

# Parse arguments
CONFIG_FILE=${1:-"configs/informer.yaml"}
NUM_GPUS=${2:-8}
MASTER_PORT=${3:-29500}

echo "Starting multi-GPU training..."
echo "Config file: $CONFIG_FILE"
echo "Number of GPUs: $NUM_GPUS"
echo "Master port: $MASTER_PORT"

# Extract batch size from config for display
BATCH_SIZE=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG_FILE'))['training']['batch_size'])")
echo "Batch size per GPU: $BATCH_SIZE"
echo "Effective batch size: $((NUM_GPUS * BATCH_SIZE))"

torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=$MASTER_PORT \
    train.py \
    --config $CONFIG_FILE

echo "Training complete!"