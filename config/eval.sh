#!/bin/bash

export PYTHONUNBUFFERED=1

# ================================
# 用户配置区域
# ================================
# get you api key and secret from https://www.faceplusplus.com.cn/
api_key="AkBbh8beATErbj63k5u5oZY3XULJEAZK" 
api_secret="xnJh2VLgNQPRQzW8AMTB1qDR0oiHrCYL"
# get your api key from https://bailian.console.aliyun.com/?spm=5176.29597918.J_bNSze_09Z5SDm7ZHdpz3Y.1.1a6c7b08AxHIIg&tab=model#/model-market/detail/qwen2.5-vl-32b-instruct
api_key_qwen="sk-1eb2b4da66b34348adb30596b14824b7"
models_dir="models/Cradle2Cane"
arcface_weight="models/backbone.pth"
lpips_model="models/alex.pth"

# 输入输出文件夹统一变量
eval_folder1="celeba-200"
eval_folder2="agedb-400"

input_base="dataset/eval"
output_base="outputs/Cradle2Cane_eval_046"

input_folder1="${input_base}/${eval_folder1}"
output_folder1="${output_base}/${eval_folder1}"

input_folder2="${input_base}/${eval_folder2}"
output_folder2="${output_base}/${eval_folder2}"

log_dir="${output_base}/logs"

# ================================
# 初始化
# ================================
mkdir -p ${output_folder1} ${output_folder2} ${log_dir}

echo "==== 开始运行推理与评估流程 ===="
echo "输入输出统一为:"
echo "  CelebA: ${input_folder1} -> ${output_folder1}"
echo "  AgeDB:  ${input_folder2} -> ${output_folder2}"
echo "日志目录: ${log_dir}"
echo "==============================="
sleep 1

# ================================
# 推理阶段
# ================================

# echo ">>> 1. CelebA 推理中..."
# python -m lib.eval.infer_dataset \
#   --models_dir "${models_dir}" \
#   --input_folder "${input_folder1}" \
#   --output_dir "${output_folder1}" 

# echo ">>> 2. AgeDB 推理中..."
# python -m lib.eval.infer_dataset \
#   --models_dir "${models_dir}" \
#   --input_folder "${input_folder2}" \
#   --output_dir "${output_folder2}" 

# ================================
# 评估阶段
# ================================

echo ">>> 3. ArcFace 评估中..."
python -m lib.eval.arcface_eval \
  --folder1 "${input_folder1}" \
  --folder2 "${output_folder1}" \
  --weight "${arcface_weight}" \
  2>&1 | tee "${log_dir}/arcface_eval.log"

echo ">>> 4. LPIPS 评估中..."
python -m lib.eval.lpips_eval \
  --folder1 "${input_folder2}" \
  --folder2 "${output_folder2}" \
  --model_path "${lpips_model}" \
  2>&1 | tee "${log_dir}/lpips_eval.log"

# echo ">>> 5. Face++ 评估中..."
# python -m lib.eval.face++_eval \
#   --folder1="${input_folder1}" \
#   --folder2="${output_folder1}" \
#   --api_key="${api_key}" \
#   --api_secret="${api_secret}" \
#   2>&1 | tee "${log_dir}/facepp_eval.log"

# echo ">>> 6. Age Accuracy 评估中..."
# python -m lib.eval.age_eval \
#   --base_dir="${output_folder1}" \
#   --api_key="${api_key_qwen}" \
#   2>&1 | tee "${log_dir}/age_eval.log"



echo "==== 所有任务完成，日志已保存至 ${log_dir}/ 目录 ===="
