# Import necessary modules
import os  # Add operating system module
import argparse
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

if __name__ == "__main__":
    # Configure command line arguments
    parser = argparse.ArgumentParser(description='Batch text analysis tool using Qwen3 model')
    parser.add_argument('-i', '--input_dir', 
                    default=r'D:\AgentEIS-5\new_test',  # 添加原始字符串前缀r
                    help='Path to the folder containing text files to analyze')
    parser.add_argument('-o', '--output_dir', 
                    default=r'D:\AgentEIS-5\new_test_response-deepseek-r1-1_5b',  # 使用原始字符串前缀r0.6b #llama3_1-8b
                    help='Path to save analysis results')
    args = parser.parse_args()

    try:
        # Create output directory if it doesn't exist
        os.makedirs(args.output_dir, exist_ok=True)

        # Initialize Qwen3 model
        llm = ChatOllama(model="deepseek-r1:1.5b") #llama3.1:8b
        
        # Process text directory
        if not os.path.isdir(args.input_dir):
            raise FileNotFoundError(f"Directory {args.input_dir} not found")
            
        # Get all text files
        text_files = [f for f in os.listdir(args.input_dir) 
                      if f.lower().endswith('.txt')]
        
        if not text_files:
            print("Error: No text files (.txt) found in the input folder")
            exit(1)

        # Batch processing
        for txt_file in text_files:
            input_path = os.path.join(args.input_dir, txt_file)
            output_path = os.path.join(args.output_dir, txt_file)
            
            # Read text content
            try:
                with open(input_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
            except Exception as e:
                print(f"Error reading {txt_file}: {str(e)}")
                continue
                
            # Build message
            messages = [HumanMessage(content=content)]
            
            # Get response from Qwen3
            try:
                response = llm.invoke(messages)
            except Exception as e:
                print(f"Error processing {txt_file}: {str(e)}")
                continue
                
            # Save results
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(f"Analysis results for {txt_file}:\n\n")
                f.write(response.content)
                
            print(f"Processed: {txt_file} → {output_path}")

    except Exception as e:
        print(f"Processing failed: {str(e)}")
