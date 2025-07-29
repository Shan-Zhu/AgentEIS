import os
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from peft import PeftModel, LoraConfig, TaskType

prompt = "You are an expert in the field of electrochemistry and need to conduct impedance equivalent circuit analysis. What is the equivalent circuit of this EIS?"
local_model_path = "./Qwen/Qwen2___5-VL-3B-Instruct"
lora_model_path = "./output/Qwen2___5-VL-7B-EIS/checkpoint-33"

# 新增功能：批量处理测试文件夹
test_folder = "D:/AgentEIS_3/EIS_dataset_test"
supported_ext = ['.jpg', '.jpeg', '.png', '.bmp']  # 支持的图片格式

# 移除 LoRA 配置代码
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=True,
    r=4,  # Lora 秩
    lora_alpha=16,  # Lora alaph，具体作用参见 Lora 原理
    lora_dropout=0.05,  # Dropout 比例
    bias="none",
)

# default: Load the model on the available device(s)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    local_model_path, torch_dtype="auto", device_map="auto"
)

# 移除加载 LoRA 模型的代码
model = PeftModel.from_pretrained(model, model_id=f"{lora_model_path}", config=config)
processor = AutoProcessor.from_pretrained(local_model_path)

# 遍历测试文件夹
for filename in os.listdir(test_folder):
    if os.path.splitext(filename)[1].lower() in supported_ext:
        img_path = os.path.join(test_folder, filename)
        # 获取同名的 txt 文件路径
        txt_filename = os.path.splitext(filename)[0] + '.txt'
        txt_path = os.path.join(test_folder, txt_filename)
        
        # 尝试读取 txt 文件内容作为 prompt
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                prompt = f.read().strip()
        except FileNotFoundError:
            print(f"未找到 {txt_filename} 文件，将使用默认 prompt。")
            # 这里可以保留原来的默认 prompt 或者设置新的默认值
            prompt = "You are an expert in the field of electrochemistry and need to conduct impedance equivalent circuit analysis. What is the equivalent circuit of this EIS?"
        
        # 构建消息
        messages = [{
            "role": "user", 
            "content": [
                {
                    "type": "image", 
                    "image": img_path,
                    "resized_height": 512,
                    "resized_width": 512,   
                },
                {"type": "text", "text": prompt}
            ]
        }]
        
        # 处理推理（复用原有逻辑）
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")
        
        # Inference: Generation of the output
        generated_ids = model.generate(**inputs, max_new_tokens=8192)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        # 保存结果到同名_result.txt
        output_path = os.path.join(test_folder, f"{os.path.splitext(filename)[0]}_result-Tuned-7B.txt")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(output_text[0])

print("批量处理完成！结果已保存至原文件夹")
