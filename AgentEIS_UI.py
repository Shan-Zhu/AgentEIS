# coding: utf-8

from tkinter import *
from tkinter import filedialog, ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from AgentEIS_ML_Classification import *
from subprocess import Popen


class EIS_GUI:
    def __init__(self, master):
        self.master = master
        master.title("AgentEIS")
        master.geometry("800x1200")
        master.minsize(800, 1200)  # Set minimum size of the window

        # 创建主框架
        main_frame = ttk.Frame(master, padding="10")
        main_frame.grid(row=0, column=0, sticky=(N, S, E, W))

        # 创建输入框和按钮
        self.file_path = StringVar()
        self.path_label = ttk.Label(main_frame, text="EIS csv file：")
        self.path_label.grid(row=0, column=0, sticky=W, padx=5, pady=5)
        self.path_entry = ttk.Entry(main_frame, textvariable=self.file_path, width=50)
        self.path_entry.grid(row=0, column=1, padx=5, pady=5)
        self.browse_button = ttk.Button(main_frame, text="Input EIS data", command=self.browse_file)
        self.browse_button.grid(row=0, column=2, padx=5, pady=5)

        # 创建显示结果的窗口
        self.result_frame = ttk.LabelFrame(main_frame, text="Results", padding="10")
        self.result_frame.grid(row=2, column=0, columnspan=3, sticky=(N, S, E, W), pady=10)

        # 创建显示图片的窗口
        self.plot_frame = ttk.LabelFrame(main_frame, text="Plot", padding="10")
        self.plot_frame.grid(row=3, column=0, columnspan=3, sticky=(N, S, E, W), pady=10)
        self.plot_frame.columnconfigure(0, weight=1)
        self.plot_frame.rowconfigure(0, weight=1)

        # 创建运行按钮
        self.run_button = ttk.Button(main_frame, text="Find Equivalent Circuit", command=self.run_eis)
        self.run_button.grid(row=1, column=0, columnspan=3, pady=10)

        # 创建Chatbot按钮
        self.chatbot_frame = ttk.Frame(main_frame, padding="10")
        self.chatbot_frame.grid(row=4, column=0, columnspan=3, pady=10)

        self.chatbot_button1 = ttk.Button(self.chatbot_frame, text="LLM Analysis", command=self.open_LLM_Analysis)
        self.chatbot_button1.grid(row=0, column=0, padx=5, pady=5)
        self.chatbot_button2 = ttk.Button(self.chatbot_frame, text="RAG Analysis", command=self.open_RAG_Analysis)
        self.chatbot_button2.grid(row=0, column=1, padx=5, pady=5)
        self.chatbot_button3 = ttk.Button(self.chatbot_frame, text="Web Search", command=self.open_Web_Search)
        self.chatbot_button3.grid(row=0, column=2, padx=5, pady=5)

        # 使得主框架可扩展
        for i in range(3):
            main_frame.columnconfigure(i, weight=1)
        main_frame.rowconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)

        # Initialize variable to store the circuit string
        self.circuit_string = ""

    def browse_file(self):
        # 打开文件选择对话框
        file_path = filedialog.askopenfilename(filetypes=[("CSV file", "*.csv")])
        self.file_path.set(file_path)

    def run_eis(self):
        # 清空之前的结果和图片
        for widget in self.result_frame.winfo_children():
            widget.destroy()
        for widget in self.plot_frame.winfo_children():
            widget.destroy()

        # 读取文件并进行EIS分析
        sample_path = self.file_path.get()
        sample_data = load_eis_data(sample_path)
        eis_type, eis_type_top3 = eis_ML_select(sample_data)

        fig, axs = plt.subplots(1, 3, figsize=(24, 6))
        fig_width, fig_height = fig.get_size_inches() * fig.dpi

        for i, et in enumerate(eis_type_top3[0][0]):
            circuit_model, initial_guess = get_model_and_guess(et)
            circuit = CustomCircuit(circuit_model, initial_guess=initial_guess)

            freq, Z = readFile(sample_path)
            freq, Z = ignoreBelowX(freq, Z)
            circuit.fit(freq, Z)

            circuit_pred = circuit.predict(freq)

            # Create a new frame for each plot and its button
            plot_frame = ttk.Frame(self.plot_frame)
            plot_frame.pack(side=LEFT, fill=BOTH, expand=True)

            # 在结果窗口中显示circuit.fit(freq, Z)结果
            result_text = Text(self.result_frame, wrap="word", height=10)
            result_text.insert(END, "Probability {}\n".format(eis_type_top3[0][1][i]))
            result_text.insert(END, str(circuit))
            result_text.pack(side=LEFT, fill=BOTH, expand=True)

            fig, ax = plt.subplots(figsize=(8, 6))
            plt.subplots_adjust(left=0.18,right=0.82)  # 调整左边边距
            plot_nyquist_SZ(Z, fmt='o', markerfacecolor='none', color='#6495ED', ax=ax)
            plot_nyquist_SZ(circuit_pred, fmt='-', color='#F08080', ax=ax)
            canvas = FigureCanvasTkAgg(fig, master=plot_frame)
            canvas.get_tk_widget().grid(row=0, column=0, sticky='nsew')

             # Create a button under each plot
            button = ttk.Button(plot_frame, text="Select this type", command=lambda c=str(circuit): self.select_this_type(c))
            button.grid(row=1, column=0, sticky=N)

        entry_height = self.path_entry.winfo_reqheight()
        result_height = self.result_frame.winfo_reqheight()
        chatbot_height = self.chatbot_frame.winfo_reqheight()
        # Calculate the total required height
        total_height = int(fig_height + entry_height + result_height + chatbot_height)

        # Resize the window to fit the plot
        self.master.geometry("%dx%d" % (int(fig_width), total_height))


    def open_LLM_Analysis(self):
        Popen(["python", "AgentEIS_UI_LLM_Analysis.py", self.circuit_string])

    def open_RAG_Analysis(self):
        Popen(["python", "AgentEIS_UI_RAG_Analysis.py", self.circuit_string])

    def open_Web_Search(self):
        Popen(["python", "AgentEIS_UI_Web_Search.py", self.circuit_string])

    def select_this_type(self, circuit):
        # 清除上次的输入
        self.circuit_string = ""

        # 找到"Circuit string"在字符串中的位置
        index = circuit.find("Circuit string: ")

        # 判断"Circuit string"是否存在
        if index != -1:
            # "Circuit string"存在，取出"Circuit string"后面的部分
            circuit = circuit[index + len("Circuit string: "):]

        # 使用split函数切割字符串并取出第一部分
        circuit = circuit.split('\n', 1)[0]

        self.circuit_string += "Please analyze the following information and provide your expert insights. We are testing the impedance of an electrochemical device and the equivalent circuit obtained is \"" + circuit + "\"."

root = Tk()
eis_gui = EIS_GUI(root)
root.mainloop()
