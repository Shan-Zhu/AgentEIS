Data and code for "Integrating Large Language and Multimodal Models with Machine Learning for Equivalent-Circuit Analysis of Electrochemical Impedance Spectroscopy".

#LLM

This part employed multiple open-source LLMs, such as Qwen3, Deepseek-r1, Llama3.1, and Llama3.2. We elaborate on its application in the analysis of equivalent circuits for EIS, including the process of fine-tuning the model via Low-Rank Adaptation (LoRA) parameter adjustment

#MLM

Several MLMs were applied in this study, including Gamma3-4B, Llama-7B, Minicpm-v-8B, Qwen2.5-VL-3B, and Qwen2.5-VL-7B. Prior to inference, a dedicated multimodal database is constructed by converting raw EIS spectra into Nyquist plots and by generating a text file that serves as a analysis prompt. 
The automatic EIS analysis pipeline is illustrated with Qwen2.5-VL-7B as the base model.

#AgentEIS

The code for AgentEIS, which is an AI-enpowered EIS anlysis system

How to use:
1. Download all code into a single folder.
2. Install all necessary Python prerequisite packages.
3. Obtain API keys for Langsmith (https://www.langchain.com/langsmith), and input it in the appropriate places in the code.
4. Download and install Ollama (https://ollama.com/), and set up open-source models such as Llama and Qwen.
5. Run AgentEIS.py.
