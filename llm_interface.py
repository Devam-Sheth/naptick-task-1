from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from config import LLM_MODEL_NAME, MAX_NEW_TOKENS_LLM 
import torch

def get_llm():
        device_option = 0 if torch.cuda.is_available() else -1 # 0 for first GPU, -1 for CPU
        device_name = "cuda" if device_option == 0 else "cpu"
        print(f"Attempting to load LLM '{LLM_MODEL_NAME}' on device: {device_name}")
        hf_pipeline = HuggingFacePipeline.from_model_id(
            model_id=LLM_MODEL_NAME,  
            task="text-generation",    
            device=device_option,      
            pipeline_kwargs={         
                "max_new_tokens": MAX_NEW_TOKENS_LLM, 
                "pad_token_id": 50256
            },
        )
        print(f"Successfully initialized LLM pipeline with model: {LLM_MODEL_NAME} on {device_name}")
        return hf_pipeline

if __name__ == '__main__':
    print("\n--- Testing LLM Interface Initialization ---")
    llm = get_llm()
    if llm:
        print("LLM pipeline loaded successfully.")
    print("\n--- Testing Complete ---")