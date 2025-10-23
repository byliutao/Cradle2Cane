#!/bin/bash
# --- 配置 ---
# 总共要启动的并行工作进程数量
NUM_WORKERS=1
FAKE_NUM=2000
INPUT_DIR="dataset/faces_webface_112x112_labeled"
FAKE_DIR="dataset/faces_webface_112x112_labeled_fake"
COMBINE_DIR="dataset/faces_webface_112x112/imgs"

echo "启动 $NUM_WORKERS 个并行工作进程..."

# --- GPU检测与分配 ---
# 自动检测可用的NVIDIA GPU数量
NUM_GPUS=$(nvidia-smi -L | wc -l)
if [ "$NUM_GPUS" -eq 0 ]; then
  echo "警告: 未检测到NVIDIA GPU，将尝试使用CPU。这可能会非常慢。"
  # 如果没有GPU，所有worker都使用CPU
  DEVICE_PREFIX="cpu"
else
  echo "检测到 $NUM_GPUS 个GPU，将把工作进程分配到这些GPU上。"
  DEVICE_PREFIX="cuda"
fi

# 循环启动每一个工作进程
# 循环范围是从 0 到 (NUM_WORKERS - 1)
for i in $(seq 0 $(($NUM_WORKERS - 1)))
do
  # 为每个进程分配设备
  if [ "$DEVICE_PREFIX" == "cuda" ]; then
    # 使用取模运算(%)将进程轮流分配给每个GPU
    # 例如，worker 0 分配给 cuda:0, worker 1 分配给 cuda:1, ..., worker 4 分配给 cuda:0
    DEVICE_ID=$(($i % $NUM_GPUS))
    DEVICE="cuda:$DEVICE_ID"
  else
    DEVICE="cpu"
  fi

  echo "启动 Worker $i, 分配到设备: $DEVICE"

  # 在后台运行Python命令，并将日志输出到 worker_i.log 文件
  python -m lib.eval.gen_fake_fast \
    --input_image_folder ${INPUT_DIR} \
    --models_dir "models/Cradle2Cane" \
    --output_dir ${FAKE_DIR} \
    --min_target_age 15 \
    --num_ages_per_id 50 \
    --device "$DEVICE" \
    --weight_dtype "float16" \
    --num_ids_to_process ${FAKE_NUM} \
    --default_source_age 30 \
    --default_source_race "person" \
    --default_source_gender "person" \
    --seed 42 \
    --num_workers $NUM_WORKERS \
    --worker_rank $i &> "outputs/worker_$i.log" &

done

# 等待所有在后台启动的进程执行完毕
wait

python -m lib.eval.combine_data \
  --source ${FAKE_DIR} \
  --dest ${COMBINE_DIR} \
  --num ${FAKE_NUM}


echo "所有工作进程已完成。"