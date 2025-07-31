import os
import sys
import tkinter as tk
from tkinter import scrolledtext, messagebox
from langchain_ollama import ChatOllama

conversation_history = []

def LLM_analysis(input_text):
    history_text = "\n".join(conversation_history)
    combined_input = f"{history_text}\nUser: {input_text}"
    
    llm = ChatOllama(model="llama3.2", temperature=0.7)
    # llm = ChatOllama(model="qwen3:8b", temperature=0.7)
    response = llm.invoke(combined_input)

    output = response['message']['content'] if 'message' in response and 'content' in response['message'] else str(
        response)

    conversation_history.append(f"User: {input_text}")
    conversation_history.append(f"LLM: {output}")

    return output

root = tk.Tk()
root.title("LLM_Analysis")

input_label = tk.Label(root, text="Input")
input_label.pack(pady=5)
input_text = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=80, height=10)
input_text.pack(pady=5)

output_label = tk.Label(root, text="Output")
output_label.pack(pady=5)
output_text = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=80, height=10)
output_text.pack(pady=5)

def show_conversation_history():
    history_window = tk.Toplevel(root)
    history_window.title("Conversation History")

    history_text = scrolledtext.ScrolledText(history_window, wrap=tk.WORD, width=80, height=40)
    history_text.pack(pady=5)

    full_conversation = "\n".join(conversation_history)
    history_text.insert(tk.END, full_conversation)
    history_text.config(state=tk.DISABLED)  


def analyze_text():
    try:
        input_data = input_text.get("1.0", tk.END).strip()
        if not input_data:
            messagebox.showwarning("Error_input", "Please input")
            return

        output_data = LLM_analysis(input_data)

        output_text.delete("1.0", tk.END)
        latest_conversation = f"User: {input_data}\nLLM: {output_data}"
        output_text.insert(tk.END, latest_conversation)
        
        input_text.delete("1.0", tk.END)
    except Exception as e:
        messagebox.showerror("Error", str(e))

analyze_button = tk.Button(root, text="LLM_Analysis", command=analyze_text)
analyze_button.pack(pady=10)

history_button = tk.Button(root, text="Show Conversation History", command=show_conversation_history)
history_button.pack(pady=5)

# Populate input_text with circuit_string if provided as an argument
if len(sys.argv) > 1:
    circuit_string = sys.argv[1]
    input_text.insert(tk.END, circuit_string)

root.mainloop()
