from modelscope import snapshot_download, AutoTokenizer
from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq, Qwen2_5_VLForConditionalGeneration, AutoProcessor
import torch

# 在modelscope上下载Qwen2.5-VL模型到本地目录下（修改模型名称）
model_dir = snapshot_download("Qwen/Qwen2.5-VL-3B-Instruct", cache_dir="./", revision="master")

# 使用Transformers加载模型权重（调整模型路径）
tokenizer = AutoTokenizer.from_pretrained("./Qwen/Qwen2.5-VL-3B-Instruct/", use_fast=False, trust_remote_code=True)
# 特别的，Qwen2.5-VL-3B-Instruct模型需要使用Qwen2VLForConditionalGeneration来加载（调整模型路径）
model = Qwen2_5_VLForConditionalGeneration.from_pretrained("./Qwen/Qwen2.5-VL-3B-Instruct/", device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True,)
model.enable_input_require_grads()  # 开启梯度检查点时，要执行该方法
