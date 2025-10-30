#!/bin/bash
# --- 配置 ---
# 总共要启动的并行工作进程数量
NUM_WORKERS=4
# !!! 总共希望处理的图像数量
FAKE_NUM=5000 
INPUT_DIR="dataset/faces_webface_112x112_labeled"
FAKE_DIR="dataset/faces_webface_112x112_labeled_fake"
COMBINE_DIR="dataset/faces_webface_112x112/imgs"

# --- !!! 新增：计算每个worker的工作量 ---
# 使用 ceiling division (向上取整)，确保所有 3000 个都被覆盖
# ( (3000 + 4 - 1) / 4 ) = (3003 / 4) = 750 (在bash整数除法中)
# 如果 FAKE_NUM=3001, (3004 / 4) = 751.
FAKE_NUM_PER_WORKER=$(( (FAKE_NUM + NUM_WORKERS - 1) / NUM_WORKERS ))
echo "总目标 ${FAKE_NUM} 张图片, 分配给 ${NUM_WORKERS} 个进程."
echo "每个进程将处理 ${FAKE_NUM_PER_WORKER} 张图片."
# -------------------------------------

echo "启动 $NUM_WORKERS 个并行工作进程..."

# --- GPU检测与分配 ---
# (这部分无需更改)
NUM_GPUS=$(nvidia-smi -L | wc -l)
if [ "$NUM_GPUS" -eq 0 ]; then
  echo "警告: 未检测到NVIDIA GPU，将尝试使用CPU。这可能会非常慢。"
  DEVICE_PREFIX="cpu"
else
  echo "检测到 $NUM_GPUS 个GPU，将把工作进程分配到这些GPU上。"
  DEVICE_PREFIX="cuda"
fi

# 循环启动每一个工作进程
for i in $(seq 0 $(($NUM_WORKERS - 1)))
do
  if [ "$DEVICE_PREFIX" == "cuda" ]; then
    DEVICE_ID=$(($i % $NUM_GPUS))
    DEVICE="cuda:$DEVICE_ID"
  else
    DEVICE="cpu"
  fi

  echo "启动 Worker $i, 分配到设备: $DEVICE"

  python -m lib.eval.gen_fake_fast \
    --input_image_folder ${INPUT_DIR} \
    --models_dir "models/Cradle2Cane" \
    --output_dir ${FAKE_DIR} \
    --min_target_age 15 \
    --num_ages_per_id 50 \
    --device "$DEVICE" \
    --weight_dtype "float16" \
    --seed 42 \
    --num_workers $NUM_WORKERS \
    --worker_rank $i \
    --num_ids_to_process ${FAKE_NUM_PER_WORKER} &> "outputs/worker_$i.log" & # !!! 关键修改

done

# 等待所有在后台启动的进程执行完毕
wait

python -m lib.eval.combine_data \
  --source ${FAKE_DIR} \
  --dest ${COMBINE_DIR} 

echo "所有工作进程已完成。"