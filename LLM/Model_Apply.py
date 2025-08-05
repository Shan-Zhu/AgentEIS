import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import pandas as pd
import os

def predict(messages, model, tokenizer):
    device = "cuda"

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=512)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response


# 加载原下载路径的tokenizer和model
tokenizer = AutoTokenizer.from_pretrained("./Qwen/Qwen3-8B", use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("./Qwen/Qwen3-8B", device_map="auto", torch_dtype=torch.bfloat16)

# 加载训练好的Lora模型，将下面的checkpointXXX替换为实际的checkpoint文件名名称
model = PeftModel.from_pretrained(model, model_id="./output/Qwen3-8B-small/checkpoint-67")

# 加载测试数据集（替换原有测试数据）
test_df = pd.read_json("D:/AgentEIS_4/new_test.jsonl", lines=True)

# 批量预测函数
def batch_predict(df, model, tokenizer):
    results = []
    from tqdm import tqdm
    # 创建 output-small 文件夹
    output_folder = "output-small-8B"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        messages = [
            {"role": "system", "content": row['instruction']},
            {"role": "user", "content": row['input']}
        ]
        prediction = predict(messages, model, tokenizer)
        results.append({
            "Original": row['output'],
            "Prediction": prediction
        })
        # 将每个 Prediction 保存为 txt 文件
        file_path = os.path.join(output_folder, f"prediction_{index}.txt")
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(prediction)
    
    return pd.DataFrame(results)

# 执行批量预测并保存结果
results_df = batch_predict(test_df, model, tokenizer)
results_df.to_csv(
    "D:/AgentEIS_4/test_results-8B-small.csv", 
    index=False,
    encoding='utf-8-sig'
)
