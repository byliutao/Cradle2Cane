import os
import csv
import re
from PIL import Image
from tqdm import tqdm
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor


# ==== 简洁肤色关键词提取 ====
def extract_coarse_skin_color(text):
    color_keywords = ['White', 'Black', 'Asian', 'Indian']
    text_lower = text.lower()
    for word in color_keywords:
        if re.search(rf'\b{re.escape(word)}\b', text_lower):
            return word
    return "unknown"


# ==== 构造 inference 输入 ====
def process(processor, images: list, querys: list, device):
    messages_list = []
    for image, text in zip(images, querys):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {"type": "text", "text": text},
                ],
            }
        ]
        messages_list.append(messages)

    texts = [
        processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        for messages in messages_list
    ]

    inputs = processor(
        text=texts,
        images=images,
        videos=None,
        padding=True,
        return_tensors="pt",
    ).to(device)

    return inputs


# ==== 推理 ====
def infer(model, processor, inputs, **kwargs):
    output = model.generate(**inputs, **kwargs)
    generated_ids = output["sequences"] if "return_dict_in_generate" in kwargs else output
    generated_ids_trimmed = [
        out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)
    ]
    return processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )


# ==== 批量推理 ====
def batch_infer(query_list, image_list, model, processor, device, **kwargs):
    inputs = process(processor, image_list, query_list, device)
    return infer(model, processor, inputs, **kwargs)


# ==== 加载 Qwen 模型 ====
def load_qwen(qwen_path, device, weight_dtype, min_pixels, max_pixels):
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        qwen_path,
        dtype=weight_dtype,
        device_map=device,
    )
    processor = AutoProcessor.from_pretrained(
        qwen_path,
        use_fast=True,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        padding_side='left'
    )
    return model, processor


# ==== 主函数 ====
def predict_skin_color_for_folder(
    image_folder,
    output_csv,
    qwen_model,
    qwen_processor,
    device
):
    result_rows = []
    all_filenames = sorted([
        fname for fname in os.listdir(image_folder)
        if fname.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    for fname in tqdm(all_filenames, desc="Processing images"):
        image_path = os.path.join(image_folder, fname)
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error loading {fname}: {e}")
            continue

        query = (
            "What is this person's race? "
            "Please respond with only one of the following options: "
            "'White', 'Black', 'Asian', or 'Indian'."
        )
        try:
            output = batch_infer([query], [image], qwen_model, qwen_processor, device)
            color = extract_coarse_skin_color(output[0])
            result_rows.append([fname, color, output[0]])
        except Exception as e:
            print(f"Error processing {fname}: {e}")
            result_rows.append([fname, "error", str(e)])

    # 写入 CSV 文件
    with open(output_csv, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "skin_color", "raw_output"])
        writer.writerows(result_rows)


# ==== 执行入口 ====
if __name__ == "__main__":
    image_folder = "others/cele_test_wo_bg"  # 修改为你的图片目录
    output_csv = "others/cele_test_wo_bg/race_predictions.csv"
    model_path = "/home/u2120240694/data/model/Qwen2.5-VL-7B-Instruct"

    device = "cuda:0"
    weight_dtype = torch.bfloat16
    min_pixels = 256 * 28 * 28
    max_pixels = 1280 * 28 * 28

    model, processor = load_qwen(model_path, device, weight_dtype, min_pixels, max_pixels)
    predict_skin_color_for_folder(image_folder, output_csv, model, processor, device)
