import os
import base64
import json  # 新增：导入 json 模块
import requests  # 用于调用ollama API

# ollama配置（需提前在ollama中加载支持多模态的模型，如llava:7b）
OLLAMA_MODEL = "llama3.2-vision:11b"
OLLAMA_API_URL = "http://localhost:11434/api/generate"  # ollama默认API地址

# 将图片转换为Base64编码的函数
def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# 批量处理测试文件夹
test_folder = "D:/AgentEIS_3/EIS_dataset_test"
supported_ext = ['.jpg', '.jpeg', '.png', '.bmp']  # 支持的图片格式

# 遍历测试文件夹
for filename in os.listdir(test_folder):
    if os.path.splitext(filename)[1].lower() in supported_ext:
        img_path = os.path.join(test_folder, filename)
        txt_filename = os.path.splitext(filename)[0] + '.txt'
        txt_path = os.path.join(test_folder, txt_filename)
        
        # 读取或使用默认 prompt
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                prompt = f.read().strip()
        except FileNotFoundError:
            print(f"未找到 {txt_filename} 文件，将使用默认 prompt。")
            prompt = "You are an expert in the field of electrochemistry and need to conduct impedance equivalent circuit analysis. What is the equivalent circuit of this EIS?"
        
        # 构造ollama多模态请求体
        request_body = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "images": [image_to_base64(img_path)],  # 图片通过Base64传递
            "options": {
                "max_tokens": 8192
            },
            "stream": False 
        }

        # 调用ollama API（关键修改：处理流式响应）
        try:
            response = requests.post(OLLAMA_API_URL, json=request_body, stream=True)  # 启用流式响应
            response.raise_for_status()
            output_text = ""
            # 逐行解析流式响应（每行是一个JSON对象）
            for line in response.iter_lines():
                if line:
                    chunk = line.decode('utf-8')
                    # 跳过可能的空行或非JSON行
                    try:
                        chunk_json = json.loads(chunk)
                        output_text += chunk_json.get("response", "")
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"调用ollama API失败: {str(e)}")
            output_text = f"推理失败: {str(e)}"
        
        # 保存结果
        output_path = os.path.join(test_folder, f"{os.path.splitext(filename)[0]}_result-llama3_2-vision-11b.txt")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(output_text)

print("批量处理完成！结果已保存至原文件夹")