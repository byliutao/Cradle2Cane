from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
import torch
import re
from PIL import Image


def extract_age(text_or_list):
    """
    从文本或文本列表中提取年龄数字。
    支持字符串或字符串列表作为输入：
    如果是字符串，返回匹配到的第一个年龄数字；
    如果是字符串列表，则返回与每个字符串对应的年龄数字列表。
    
    如果无法匹配到年龄，直接抛出异常。
    """
    
    def _extract_from_string(single_text):
        age_patterns = [
            r'age is (\d+)',
            r'appears to be (\d+)',
            r'approximately (\d+)',
            r'about (\d+)',
            r'(\d+) years old',
            r'(\d+)-year-old',
            r'(\d+) year old',
            r'age: (\d+)',
            r'(\d+)'  # 最后尝试匹配任何数字
        ]
        
        for pattern in age_patterns:
            matches = re.search(pattern, single_text.lower())
            if matches:
                return int(matches.group(1))
        raise ValueError("Age prediction failed")
    
    if isinstance(text_or_list, str):
        return _extract_from_string(text_or_list)
    elif isinstance(text_or_list, list):
        return [_extract_from_string(item) for item in text_or_list]
    else:
        raise TypeError("Input must be a string or a list of strings.")


def process(processor, images: list, querys: list, device):
    messages_list = []
    for image_path, text in zip(images, querys):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": f"",
                    },
                    {"type": "text", "text": f"{text}"},
                ],
            }
        ]
        messages_list.append(messages)

    # Preparation for inference
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
    )
    inputs = inputs.to(device)

    return inputs


def infer(model: Qwen2_5_VLForConditionalGeneration, processor: AutoProcessor, inputs, **kwargs):
    model_output = model.generate(**inputs, **kwargs)
    
    if "output_hidden_states" in kwargs and kwargs["output_hidden_states"] is True:
        return model_output["hidden_states"]
    else:
        if "return_dict_in_generate" in kwargs and kwargs["return_dict_in_generate"] is True:
            generated_ids = model_output["sequences"]
        else:
            generated_ids = model_output

        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text


def batch_infer(query_list: list, image_list: list, qwen_model, qwen_processor, device, **kwargs):
    inputs = process(qwen_processor, image_list, query_list, device)
    output_text = infer(qwen_model, qwen_processor, inputs, **kwargs)
    
    return output_text


def age_predict(query_list: list, image_list: list, qwen_model, qwen_processor, device,):
    output_text = batch_infer(query_list, image_list, qwen_model, qwen_processor, device,)
    output_age = extract_age(output_text)

    return output_age


def face_describe(query_list: list, image_list: list, qwen_model, qwen_processor, device,):
    output_text = batch_infer(query_list, image_list, qwen_model, qwen_processor, device,)
    return output_text


def face_aging_describe(target_face_ages, input_face_image_list: list, qwen_model, qwen_processor, device, **kwargs):
    query_list = [(
        f"You are a face transformation expert."
        f"Descibe this person's face at age {target_age}. Just give me a description of the transformed face."
    ) for target_age in target_face_ages]
    
    output_text = batch_infer(query_list, input_face_image_list, qwen_model, qwen_processor, device, **kwargs)
    return output_text


def face_aging_describe_w_input_age(input_face_ages, target_face_ages, input_face_image_list: list, qwen_model, qwen_processor, device, **kwargs):
    query_list = [(
        f"You are a face transformation expert."
        f"Please generate a description of how this face would look if the person were {target_age} years older."
    ) for input_age, target_age in zip(input_face_ages, target_face_ages)]
    
    output_text = batch_infer(query_list, input_face_image_list, qwen_model, qwen_processor, device, **kwargs)
    return output_text


def load_qwen(qwen_path, device, weight_dtype, min_pixels, max_pixels, use_flash=True):
    if use_flash:
        qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            qwen_path, dtype=weight_dtype,
            attn_implementation="flash_attention_2",
            device_map=device,
        )
    else:
        qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            qwen_path, dtype=weight_dtype,
            device_map=device,
        )        
    qwen_processor = AutoProcessor.from_pretrained(qwen_path, use_fast=True, min_pixels=min_pixels, max_pixels=max_pixels, padding_side='left')
    
    qwen_model.requires_grad_ = False

    return qwen_model, qwen_processor


if __name__ == "__main__":

    # default: Load the model on the available device(s)
    # model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    #     "/data/model/Qwen2.5-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
    # )

    # We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
    model, processor = load_qwen("/data/model/Qwen2.5-VL-7B-Instruct", "cuda", torch.bfloat16, min_pixels=256*28*28, max_pixels=1280*28*28)


    # The default range for the number of visual tokens per image in the model is 4-16384.
    # You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
    # min_pixels = 256*28*28
    # max_pixels = 1280*28*28
    # processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)
    output_text = age_predict(["You are a face age estimation expert. Looking at this face image, "
        "how old is the person in this image? Provide ONLY the specific age number, with no additional text."], 
        [Image.open("/data/dataset/AgeDB/8290_RichardChamberlain_55_m.jpg")],model,processor,"cuda")
    
    print(output_text)

    inputs = process(processor, [Image.open("/data/dataset/AgeDB/8290_RichardChamberlain_55_m.jpg")], ["describe this image"], "cuda")

    # Inference: Generation of the output
    output_text = infer(model, processor, inputs)
    print(output_text)
