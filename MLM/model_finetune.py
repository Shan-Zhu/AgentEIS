import csv
import torch
from datasets import Dataset
from modelscope import snapshot_download, AutoTokenizer
from swanlab.integration.transformers import SwanLabCallback
from qwen_vl_utils import process_vision_info
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
)
import swanlab
import json
import os

swanlab.login(api_key="...")

prompt = "You are an expert in the field of electrochemistry and need to conduct impedance equivalent circuit analysis. What is the equivalent circuit of the electrochemical impedance in this picture?"
model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
local_model_path = "./Qwen/Qwen2___5-VL-7B-Instruct"
train_dataset_json_path = "eis_dataset_train.json"
val_dataset_json_path = "eis_dataset_val.json"
output_dir = "./output/Qwen2___5-VL-7B-EIS"
MAX_LENGTH = 100000

def process_func(example):
    """
    data_preprocess
    """
    input_ids, attention_mask, labels = [], [], []
    conversation = example["conversations"]
    image_file_path = conversation[0]["value"]
    output_content = conversation[1]["value"]
    
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": f"{image_file_path}",
                    "resized_height": 512,
                    "resized_width": 512,
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]
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
    inputs = {key: value.tolist() for key, value in inputs.items()} #tensor -> list,
    instruction = inputs

    response = tokenizer(f"{output_content}", add_special_tokens=False)


    input_ids = (
            instruction["input_ids"][0] + response["input_ids"] + [tokenizer.pad_token_id]
    )

    attention_mask = instruction["attention_mask"][0] + response["attention_mask"] + [1]
    labels = (
            [-100] * len(instruction["input_ids"][0])
            + response["input_ids"]
            + [tokenizer.pad_token_id]
    )
    if len(input_ids) > MAX_LENGTH:  
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]

    input_ids = torch.tensor(input_ids)
    attention_mask = torch.tensor(attention_mask)
    labels = torch.tensor(labels)
    inputs['pixel_values'] = torch.tensor(inputs['pixel_values'])
    inputs['image_grid_thw'] = torch.tensor(inputs['image_grid_thw']).squeeze(0)  
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels,
            "pixel_values": inputs['pixel_values'], "image_grid_thw": inputs['image_grid_thw']}


def predict(messages, model):
    
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

    
    generated_ids = model.generate(**inputs, max_new_tokens=MAX_LENGTH)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    return output_text[0]


model_dir = snapshot_download(model_id, cache_dir="./", revision="master")

tokenizer = AutoTokenizer.from_pretrained(local_model_path, use_fast=False, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(local_model_path)

origin_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    local_model_path, 
    device_map="cuda",  
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
).to("cuda")  
origin_model.enable_input_require_grads() 

train_ds = Dataset.from_json(train_dataset_json_path)
train_dataset = train_ds.map(process_func)

config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False,  
    r=4,  
    lora_alpha=16,  # Lora alaph，
    lora_dropout=0.05,  
    bias="none",
)

train_peft_model = get_peft_model(origin_model, config)

args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    logging_steps=10,
    logging_first_step=10,
    num_train_epochs=1,
    save_steps=100,
    learning_rate=1e-5,
    save_on_each_node=True,
    gradient_checkpointing=True,
    report_to="none",
)
    
swanlab_callback = SwanLabCallback(
    project="Qwen2.5-VL-eis",
    experiment_name="eis",
    config={
        "model": "https://modelscope.cn/models/Qwen/Qwen25-VL-7B-Instruct",
        "dataset": "https://modelscope.cn/datasets/AI-ModelScope/eis/summary",
        # "github": "https://github.com/datawhalechina/self-llm",
        "model_id": model_id,
        "train_dataset_json_path": train_dataset_json_path,
        "val_dataset_json_path": val_dataset_json_path,
        "output_dir": output_dir,
        "prompt": prompt,
        "train_data_number": len(train_ds),
        "token_max_length": MAX_LENGTH,
        "lora_rank": 4,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
    },
)


trainer = Trainer(
    model=train_peft_model,
    args=args,
    train_dataset=train_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    callbacks=[swanlab_callback],
)


trainer.train()

# ====================test===================

val_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=True,  
    r=4,  
    lora_alpha=16,  
    lora_dropout=0.05,  
    bias="none",
)

load_model_path = f"{output_dir}/checkpoint-{max([int(d.split('-')[-1]) for d in os.listdir(output_dir) if d.startswith('checkpoint-')])}"
print(f"load_model_path: {load_model_path}")
val_peft_model = PeftModel.from_pretrained(origin_model, model_id=load_model_path, config=val_config)

with open(val_dataset_json_path, "r") as f:
    test_dataset = json.load(f)

test_image_list = []
results = []

for item in test_dataset:
    image_file_path = item["conversations"][0]["value"]
    label = item["conversations"][1]["value"]
    
    messages = [{
        "role": "user", 
        "content": [
            {
            "type": "image", 
            "image": image_file_path,
            "resized_height": 512,
            "resized_width": 512,   
            },
            {
            "type": "text",
            "text": prompt,
            }
        ]}]
    
    response = predict(messages, val_peft_model)
    
    print(f"predict:{response}")
    print(f"gt:{label}\n")

    results.append({
        "image_path": image_file_path,
        "predict": response,
        "gt": label
    })
    
    test_image_list.append(swanlab.Image(image_file_path, caption=response))

csv_path = "./prediction_results_qwenVL2_5-7B.csv"
with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['image_path', 'predict', 'gt']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)

swanlab.log({"Prediction": test_image_list})

swanlab.finish()
