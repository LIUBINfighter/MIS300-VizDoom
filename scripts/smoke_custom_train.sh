#!/usr/bin/env bash
# Lightweight smoke test for custom training
# Usage: ./scripts/smoke_custom_train.sh [ENV_NAME] [TRAIN_DIR] [STEPS] [NUM_WORKERS] [NUM_ENVS] [DEVICE]
set -euo pipefail

ENV_NAME=${1:-custom_doom_basic}
TRAIN_DIR=${2:-/tmp/smoke_train}
STEPS=${3:-1000}
NUM_WORKERS=${4:-1}
NUM_ENVS=${5:-2}
DEVICE=${6:-cpu}

echo "Running smoke custom training: env=$ENV_NAME steps=$STEPS workers=$NUM_WORKERS envs=$NUM_ENVS device=$DEVICE"
mkdir -p "$TRAIN_DIR"

# Run training with minimal settings and short duration
PYTHONPATH=. python src/train_custom.py \
  --env $ENV_NAME \
  --num_workers $NUM_WORKERS \
  --num_envs_per_worker $NUM_ENVS \
  --train_for_env_steps $STEPS \
  --device $DEVICE \
  --train_dir $TRAIN_DIR \
  --save_every_sec 300 \
  --with_wandb False

EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
  echo "Smoke training finished successfully"
else
  echo "Smoke training failed with exit code $EXIT_CODE"
fi
exit $EXIT_CODE
