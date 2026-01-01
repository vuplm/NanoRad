#!/bin/bash

set -e

# Always run from project root
cd "$(dirname "$0")/.." || exit 1

export CUDA_VISIBLE_DEVICES=0,1
export TOKENIZERS_PARALLELISM=false

python train_advanced/train.py \
  --strategy ddp \
  --accelerator gpu \
  --devices 2 \
  --max_epochs 15 \
  --accumulate_grad_batches 1 \
  --ckpt_file ./outputs/phase1_advanced_model/checkpoints/best_ckpt.ckpt \
  --llm_use_lora True \
  --val_check_interval 1.0 \
  --limit_train_batches 1.0 \
  --limit_val_batches 1.0 \
  --savedmodel_path ./outputs/phase2_advanced_model \