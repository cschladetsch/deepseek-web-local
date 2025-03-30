import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import os
import sys
import platform
import traceback
from pathlib import Path

# Global variables for model and tokenizer
model = None
tokenizer = None

# Get system info for display
def get_system_info():
    try:
        gpu_info = "No GPU detected"
        if torch.cuda.is_available():
            gpu_info = f"GPU: {torch.cuda.get_device_name(0)}"
            gpu_info += f" | Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
        
        return {
            "Platform": platform.platform(),
            "Python": platform.python_version(),
            "Torch": torch.__version__,
            "GPU": gpu_info
        }
    except Exception as e:
        return {"Error": str(e)}

# Setup logging
def log(message, level="INFO"):
    print(f"[{level}] {message}")

# Configure and load model
def load_model():
    global model, tokenizer
    log("Loading model...")
    
    # Get model name from environment variable or find it in models directory
    model_name = os.environ.get('MODEL_NAME')
    
    if not model_name:
        # Try to find a model directory
        models_dir = Path("./models")
        if models_dir.exists():
            model_dirs = [d for d in models_dir.iterdir() if d.is_dir()]
            if model_dirs:
                model_name = model_dirs[0].name
                log(f"Found model directory: {model_name}")
            else:
                log("No model directories found in ./models", "ERROR")
                return False
        else:
            log("Models directory not found", "ERROR")
            return False
    
    model_path = f"./models/{model_name}"
    log(f"Using model: {model_name} at {model_path}")
    
    try:
        # Load tokenizer first as it's usually smaller
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        log("Tokenizer loaded successfully")
        
        # Configure model loading based on available hardware
        if torch.cuda.is_available():
            log("Loading model with GPU acceleration...")
            
            # Try to load with 4-bit quantization first (more memory efficient for 8GB GPUs)
            try:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    quantization_config=quantization_config,
                    device_map="auto",
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    max_memory={0: "7GiB", "cpu": "16GiB"}  # Optimize for 8GB GPU
                )
                log("Model loaded with 4-bit quantization (optimized for 8GB GPU)")
            except Exception as e:
                log(f"4-bit quantization failed: {str(e)}", "WARNING")
                log("Falling back to CPU-only mode (will be slower)...")
                
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    device_map="cpu",  # Force CPU-only as fallback
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
                log("Model loaded on CPU only (slower but more compatible)")
        else:
            log("Loading model on CPU (will be slow)...")
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            log("Model loaded on CPU")
        
        return True
    
    except Exception as e:
        log(f"Error loading model: {str(e)}", "ERROR")
        log(traceback.format_exc(), "ERROR")
        return False

# Get the maximum supported length for the model
def get_max_length(tokenizer):
    if hasattr(tokenizer, "model_max_length"):
        return min(tokenizer.model_max_length, 2048)
    return 1024

# Process file content
def process_files(files):
    file_context = ""
    
    for file in files:
        try:
            file_name = os.path.basename(file.name)
            file_size = os.path.getsize(file.name)
            
            # First try to read as text
            try:
                with open(file.name, "r", encoding="utf-8") as f:
                    content = f.read()
                    file_context += f"\nFile: {file_name}\nSize: {file_size} bytes\nContent:\n{content}\n\n"
            except UnicodeDecodeError:
                # Try to handle binary files
                file_context += f"\nBinary File: {file_name}\nSize: {file_size} bytes\n"
                
                # Check if it's a common binary type we can read
                if file_name.endswith(('.png', '.jpg', '.jpeg', '.gif', '.pdf')):
                    file_context += f"File type: {file_name.split('.')[-1]}\n"
                    
        except Exception as e:
            file_context += f"\nError reading file {file_name}: {str(e)}\n"
    
    return file_context

# Simple chatbot function
def chat(message, history, files):
    global model, tokenizer
    
    if not message.strip():
        return history
    
    if model is None or tokenizer is None:
        return history + [(message, "Error: Model not loaded. Please check console for errors.")]
    
    # Process file content if files are uploaded
    file_context = process_files(files) if files else ""
    
    # Combine file context with user message
    if file_context:
        input_text = f"I've uploaded the following files (processed locally):\n{file_context}\n\nMy question is: {message}"
    else:
        input_text = message
    
    # Generate response
    try:
        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(model.device)
        max_length = min(input_ids.shape[1] + 1024, get_max_length(tokenizer))
        
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_length=max_length,
                temperature=0.7,
                top_p=0.9,
                top_k=40,
                do_sample=True,
                num_beams=1,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Extract only the response part (not the input)
        if response.startswith(input_text):
            response = response[len(input_text):].strip()
        
        # If response is empty or just whitespace, provide a fallback
        if not response or response.isspace():
            response = "I couldn't generate a meaningful response. Please try rephrasing your question."
        
        return history + [(message, response)]
        
    except Exception as e:
        print(f"Error generating response: {str(e)}")
        return history + [(message, f"An error occurred: {str(e)}")]

# Function to clear chat history and textbox
def clear_chat_and_textbox():
    return None, ""

# Main function
def main():
    # Load the model and tokenizer
    if not load_model():
        print("Failed to load model. Exiting.")
        sys.exit(1)
    
    # Get system info
    system_info = get_system_info()
    info_text = "\n".join([f"**{k}**: {v}" for k, v in system_info.items()])
    
    # Create Gradio interface
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown(f"# Christian's DeepSeek Local Interface - {os.environ.get('MODEL_NAME', 'Unknown Model')}")
        gr.Markdown("All processing occurs locally on your machine. Uploaded files are never sent over the internet.")
        
        with gr.Accordion("System Information", open=False):
            gr.Markdown(info_text)
            gr.Markdown(f"**Model**: {os.environ.get('MODEL_NAME', 'Unknown')}")
            gr.Markdown("**Privacy Notice**: All processing occurs locally. Files and queries never leave your machine.")
        
        files = gr.File(file_count="multiple", label="Upload Files (Processed Locally)")
        
        chatbot = gr.Chatbot(label="Chat", height=500)
        
        with gr.Row():
            with gr.Column(scale=6):
                msg = gr.Textbox(
                    label="Enter your query",
                    placeholder="Type your question here and press Enter to submit",
                    lines=3,
                    show_label=True,
                )
            with gr.Column(scale=1):
                submit_btn = gr.Button("Submit")
                clear_btn = gr.Button("Clear")
        
        # Hook up the chat function to ensure Enter key works
        msg.submit(chat, [msg, chatbot, files], [chatbot])
        submit_btn.click(chat, [msg, chatbot, files], [chatbot])
        
        # Add clear button functionality that also clears the textbox
        clear_btn.click(clear_chat_and_textbox, None, [chatbot, msg])
    
    # Get port from environment variable or use default
    port = int(os.environ.get('DEEPSEEK_PORT', 7860))
    
    # Launch the interface
    demo.launch(server_name="127.0.0.1", server_port=port, share=False)

if __name__ == "__main__":
    main()
