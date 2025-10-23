#!/bin/bash
set -e
set -o pipefail
export CUDA_VISIBLE_DEVICES=0

# 基本参数
MAX_LOAD_NUM=10000
EPOCH=40
EXP_NUM=200
NUM__PROCESSES=1
BATCH=2

BASE="models"
NAME="${MAX_LOAD_NUM}_ep${EPOCH}"


accelerate launch --mixed_precision 'fp16' \
  --num_processes ${NUM__PROCESSES} --num_machines 1 --dynamo_backend 'no' train.py \
  --config config/train_config.yaml \
  --output_dir "${BASE}/${NAME}" \
  --num_train_epochs ${EPOCH} \
  --validation_steps 1000 \
  --max_load_num ${MAX_LOAD_NUM} \
  --train_batch_size ${BATCH} \
  --id_cos_loss_weight 1 \
  --age_loss_weight 0.75 \
  --age_loss_2_weight 1 \
  --pixel_mse_loss_weight 0.05 \
  --ssim_loss_weight 0.1 \
  --g_loss_weight 0.01 \
  --lpips_loss_weight 1 \
  --t1 10 \
  --t2 30 




echo "🎯 所有权重遍历实验跑完！"
