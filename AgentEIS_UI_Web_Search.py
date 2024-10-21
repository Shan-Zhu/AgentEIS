import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox
import os
from langchain import hub
from langchain_ollama import ChatOllama
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain.agents import initialize_agent, AgentType, create_tool_calling_agent, AgentExecutor
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool
from langchain_core.tools import Tool

# 定义对话历史变量
conversation_history = []


def initialize_tavily_search(max_results=2):
    """Initialize the Tavily search tool."""
    return TavilySearchResults(max_results=max_results)


def create_agent_executor(model, tools, prompt):
    """
    Create an agent executor.

    :param model: Chat model.
    :param tools: List of tools to bind to the model.
    :param prompt: Prompt template to be used.
    :return: Configured agent executor.
    """
    agent = create_tool_calling_agent(model, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools)


def advanced_search(input_text):
    """Main function to set up and execute the agent."""
    Tavily_search = initialize_tavily_search(max_results=2)
    tools = [Tavily_search]
    model = ChatOllama(model="llama3.1")
    prompt = hub.pull("hwchase17/openai-functions-agent")
    agent_executor = create_agent_executor(model, tools, prompt)
    input = input_text
    output = agent_executor.invoke({"input": input})

    return output


def LLM_analysis(input_text):
    # Combine history and current input
    history_text = "\n".join(conversation_history)
    combined_input = f"{history_text}\nUser: {input_text}"

    # llm = ChatOllama(model="llama3.1", temperature=0.7)
    # llm = ChatOllama(model="llama3.2:1b", temperature=0.7)
    llm = ChatOllama(model="qwen2.5:0.5b", temperature=0.7)
    response = llm.invoke(combined_input)

    # Extract the actual response from the model
    output = response['message']['content'] if 'message' in response and 'content' in response['message'] else str(
        response)

    # Append user input and model output to conversation history
    conversation_history.append(f"User: {input_text}")
    conversation_history.append(f"LLM: {output}")

    return output


class Application(tk.Tk):
    def __init__(self, circuit_string=""):
        super().__init__()
        self.title("Web Search")

        # Set the circuit_string
        self.circuit_string = circuit_string

        # Create UI elements
        self.create_widgets()

    def create_widgets(self):
        # Tavily API
        self.tavily_api_label = tk.Label(self, text="Tavily API:")
        self.tavily_api_label.grid(row=0, column=0, sticky='e')
        self.tavily_api_entry = tk.Entry(self, width=50)
        self.tavily_api_entry.grid(row=0, column=1, columnspan=2)
        self.tavily_api_entry.insert(0, "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")  # 设置默认密钥

        # LangSmith API
        self.api_label = tk.Label(self, text="LangSmith API:")
        self.api_label.grid(row=1, column=0, sticky='e')
        self.api_entry = tk.Entry(self, width=50)
        self.api_entry.grid(row=1, column=1, columnspan=2)
        self.api_entry.insert(0, "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")  # 设置默认密钥

        # Input text
        self.input_label = tk.Label(self, text="Input:")
        self.input_label.grid(row=2, column=0, sticky='ne')
        self.input_text = scrolledtext.ScrolledText(self, width=50, height=10)
        self.input_text.grid(row=2, column=1, columnspan=2)
        self.input_text.insert(tk.END, self.circuit_string)  # 将circuit_string设置为默认输入内容

        # Output text
        self.output_label = tk.Label(self, text="Output:")
        self.output_label.grid(row=3, column=0, sticky='ne')
        self.output_text = scrolledtext.ScrolledText(self, width=50, height=10)
        self.output_text.grid(row=3, column=1, columnspan=2)

        # Buttons
        self.run_search_button = tk.Button(self, text="Run Web Search", command=self.run_search)
        self.run_search_button.grid(row=4, column=1, sticky='e')

        # 创建显示对话历史的按钮
        self.history_button = tk.Button(self, text="Show Conversation History", command=self.show_conversation_history)
        self.history_button.grid(row=4, column=2, sticky='w')

    def run_search(self):
        default_tavily_api_key = "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"
        tavily_api_key = self.tavily_api_entry.get().strip() or default_tavily_api_key
        os.environ["TAVILY_API_KEY"] = tavily_api_key

        default_langsmith_api_key = "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"
        langsmith_api_key = self.api_entry.get().strip() or default_langsmith_api_key
        os.environ["LANGCHAIN_API_KEY"] = langsmith_api_key

        input_text = self.input_text.get("1.0", tk.END).strip()
        if not input_text:
            messagebox.showerror("Error", "Please enter input text for the search.")
            return

        output_data = LLM_analysis(input_text)

        # 更新输出文本框，只显示最新的对话内容
        self.output_text.delete("1.0", tk.END)
        latest_conversation = f"User: {input_text}\nLLM: {output_data}"
        self.output_text.insert(tk.END, latest_conversation)

        # 清空输入框
        self.input_text.delete("1.0", tk.END)

    def show_conversation_history(self):
        history_window = tk.Toplevel(self)
        history_window.title("Conversation History")

        history_text = scrolledtext.ScrolledText(history_window, wrap=tk.WORD, width=80, height=40)
        history_text.pack(pady=5)

        full_conversation = "\n".join(conversation_history)
        history_text.insert(tk.END, full_conversation)
        history_text.config(state=tk.DISABLED)  # 禁用编辑


if __name__ == "__main__":
    import sys

    circuit_string = sys.argv[1] if len(sys.argv) > 1 else ""

    app = Application(circuit_string=circuit_string)
    app.mainloop()
