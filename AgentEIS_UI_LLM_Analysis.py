import os
import sys
import tkinter as tk
from tkinter import scrolledtext, messagebox
from langchain_ollama import ChatOllama

# 定义对话历史变量
conversation_history = []


# 定义LLM分析函数
def LLM_analysis(input_text):
    # 将对话历史和当前输入拼接成一个字符串
    history_text = "\n".join(conversation_history)
    combined_input = f"{history_text}\nUser: {input_text}"

    # llm = ChatOllama(model="llama3.2", temperature=0.7)
    llm = ChatOllama(model="llama3.2:1b", temperature=0.7)
    # llm = ChatOllama(model="llama3.1", temperature=0.7)
    # llm = ChatOllama(model="qwen2.5:7b", temperature=0.7)
    # llm = ChatOllama(model="qwen2.5:3b", temperature=0.7)
    # llm = ChatOllama(model="qwen2.5:1.5b", temperature=0.7)
    # llm = ChatOllama(model="qwen2.5:0.5b", temperature=0.7)
    # llm = ChatOllama(model="gemma2:9b", temperature=0.7)
    # llm = ChatOllama(model="gemma2:2b", temperature=0.7)
    response = llm.invoke(combined_input)

    # 提取模型的实际回复
    output = response['message']['content'] if 'message' in response and 'content' in response['message'] else str(
        response)

    # 记录用户输入和LLM输出到对话历史中
    conversation_history.append(f"User: {input_text}")
    conversation_history.append(f"LLM: {output}")

    return output


# 创建主窗口
root = tk.Tk()
root.title("LLM_Analysis")

# 创建输入文本框
input_label = tk.Label(root, text="Input")
input_label.pack(pady=5)
input_text = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=80, height=10)
input_text.pack(pady=5)

# 创建输出文本框
output_label = tk.Label(root, text="Output")
output_label.pack(pady=5)
output_text = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=80, height=10)
output_text.pack(pady=5)


# 创建新的窗口来显示对话历史记录
def show_conversation_history():
    history_window = tk.Toplevel(root)
    history_window.title("Conversation History")

    history_text = scrolledtext.ScrolledText(history_window, wrap=tk.WORD, width=80, height=40)
    history_text.pack(pady=5)

    full_conversation = "\n".join(conversation_history)
    history_text.insert(tk.END, full_conversation)
    history_text.config(state=tk.DISABLED)  # 禁用编辑


# 创建分析按钮的回调函数
def analyze_text():
    try:
        input_data = input_text.get("1.0", tk.END).strip()
        if not input_data:
            messagebox.showwarning("Error_input", "Please input")
            return

        output_data = LLM_analysis(input_data)

        # 更新输出文本框，只显示最新的对话内容
        output_text.delete("1.0", tk.END)
        latest_conversation = f"User: {input_data}\nLLM: {output_data}"
        output_text.insert(tk.END, latest_conversation)

        # 清空输入框
        input_text.delete("1.0", tk.END)
    except Exception as e:
        messagebox.showerror("Error", str(e))


# 创建分析按钮
analyze_button = tk.Button(root, text="LLM_Analysis", command=analyze_text)
analyze_button.pack(pady=10)

# 创建显示对话历史的按钮
history_button = tk.Button(root, text="Show Conversation History", command=show_conversation_history)
history_button.pack(pady=5)

# Populate input_text with circuit_string if provided as an argument
if len(sys.argv) > 1:
    circuit_string = sys.argv[1]
    input_text.insert(tk.END, circuit_string)

# 运行主循环
root.mainloop()

'''
import os
import sys
import tkinter as tk
from tkinter import scrolledtext, messagebox
from langchain_ollama import ChatOllama

# 定义对话历史变量
conversation_history = []


# 定义LLM分析函数
def LLM_analysis(input_text):
    # 将对话历史和当前输入拼接成一个字符串
    history_text = "\n".join(conversation_history)
    combined_input = f"{history_text}\nUser: {input_text}"

    llm = ChatOllama(model="llama3.1", temperature=0.7)
    output = llm.invoke(combined_input)

    # 记录用户输入和LLM输出到对话历史中
    conversation_history.append(f"User: {input_text}")
    conversation_history.append(f"LLM: {output}")

    return output


# 创建主窗口
root = tk.Tk()
root.title("LLM_Analysis")

# 创建输入文本框
input_label = tk.Label(root, text="Input")
input_label.pack(pady=5)
input_text = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=80, height=10)
input_text.pack(pady=5)

# 创建输出文本框
output_label = tk.Label(root, text="Output")
output_label.pack(pady=5)
output_text = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=80, height=10)
output_text.pack(pady=5)


# 创建新的窗口来显示对话历史记录
def show_conversation_history():
    history_window = tk.Toplevel(root)
    history_window.title("Conversation History")

    history_text = scrolledtext.ScrolledText(history_window, wrap=tk.WORD, width=80, height=40)
    history_text.pack(pady=5)

    full_conversation = "\n".join(conversation_history)
    history_text.insert(tk.END, full_conversation)
    history_text.config(state=tk.DISABLED)  # 禁用编辑


# 创建分析按钮的回调函数
def analyze_text():
    try:
        input_data = input_text.get("1.0", tk.END).strip()
        if not input_data:
            messagebox.showwarning("Error_input", "Please input")
            return

        output_data = LLM_analysis(input_data)

        # 更新输出文本框，只显示最新的对话内容
        output_text.delete("1.0", tk.END)
        latest_conversation = f"User: {input_data}\nLLM: {output_data}"
        output_text.insert(tk.END, latest_conversation)

        # 清空输入框
        input_text.delete("1.0", tk.END)
    except Exception as e:
        messagebox.showerror("Error", str(e))


# 创建分析按钮
analyze_button = tk.Button(root, text="LLM_Analysis", command=analyze_text)
analyze_button.pack(pady=10)

# 创建显示对话历史的按钮
history_button = tk.Button(root, text="Show Conversation History", command=show_conversation_history)
history_button.pack(pady=5)

# Populate input_text with circuit_string if provided as an argument
if len(sys.argv) > 1:
    circuit_string = sys.argv[1]
    input_text.insert(tk.END, circuit_string)

# 运行主循环
root.mainloop()
'''