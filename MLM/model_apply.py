import os
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from peft import PeftModel, LoraConfig, TaskType

prompt = "You are an expert in the field of electrochemistry and need to conduct impedance equivalent circuit analysis. What is the equivalent circuit of this EIS?"
local_model_path = "./Qwen/Qwen2___5-VL-3B-Instruct"
lora_model_path = "./output/Qwen2___5-VL-7B-EIS/checkpoint-33"

test_folder = "./EIS_dataset_test"
supported_ext = ['.jpg', '.jpeg', '.png', '.bmp']  

config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=True,
    r=4,  
    lora_alpha=16,  
    lora_dropout=0.05, 
    bias="none",
)

# default: Load the model on the available device(s)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    local_model_path, torch_dtype="auto", device_map="auto"
)

model = PeftModel.from_pretrained(model, model_id=f"{lora_model_path}", config=config)
processor = AutoProcessor.from_pretrained(local_model_path)

for filename in os.listdir(test_folder):
    if os.path.splitext(filename)[1].lower() in supported_ext:
        img_path = os.path.join(test_folder, filename)
        txt_filename = os.path.splitext(filename)[0] + '.txt'
        txt_path = os.path.join(test_folder, txt_filename)
        
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                prompt = f.read().strip()
        except FileNotFoundError:
            print(f"if no {txt_filename} ，using default prompt。")
            prompt = "You are an expert in the field of electrochemistry and need to conduct impedance equivalent circuit analysis. What is the equivalent circuit of this EIS?"
        
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
        
        output_path = os.path.join(test_folder, f"{os.path.splitext(filename)[0]}_result-Tuned-7B.txt")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(output_text[0])

print("Done")
