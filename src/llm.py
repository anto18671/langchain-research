from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Initialize local Qwen model
def load_local_llm():
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("../models/qwen3")

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        "../models/qwen3",
        torch_dtype=torch.float16,
        device_map="auto"
    )

    # Return pipeline components
    return tokenizer, model
