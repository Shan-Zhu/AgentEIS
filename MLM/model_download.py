from modelscope import snapshot_download, AutoTokenizer
from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq, Qwen2_5_VLForConditionalGeneration, AutoProcessor
import torch

# download Qwen2.5-VL
model_dir = snapshot_download("Qwen/Qwen2.5-VL-3B-Instruct", cache_dir="./", revision="master")
tokenizer = AutoTokenizer.from_pretrained("./Qwen/Qwen2.5-VL-3B-Instruct/", use_fast=False, trust_remote_code=True)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained("./Qwen/Qwen2.5-VL-3B-Instruct/", device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True,)
model.enable_input_require_grads()  
