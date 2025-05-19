# llm_interface.py
# Handles the initialization of the Language Model (LLM).

from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from config import LLM_MODEL_NAME, MAX_NEW_TOKENS_LLM # Import configurations
import torch # PyTorch is a dependency for HuggingFace transformers

def get_llm():
    """
    Initializes and returns a HuggingFace LLM pipeline for text generation.

    This function sets up a local LLM using the HuggingFace `transformers` library.
    The model specified in config.py (LLM_MODEL_NAME) will be used.
    It's configured for text generation tasks.

    Returns:
        HuggingFacePipeline: An instance of the LLM pipeline ready for generation.
                             Returns None if initialization fails.
    """
    try:
        # Determine the device to run the model on (GPU if available, else CPU)
        device_option = 0 if torch.cuda.is_available() else -1 # 0 for first GPU, -1 for CPU
        device_name = "cuda" if device_option == 0 else "cpu"
        print(f"Attempting to load LLM '{LLM_MODEL_NAME}' on device: {device_name}")

        # Initialize the HuggingFace pipeline for text generation
        # `from_model_id` handles downloading the model if not already cached.
        hf_pipeline = HuggingFacePipeline.from_model_id(
            model_id=LLM_MODEL_NAME,  # The Hugging Face Hub model ID
            task="text-generation",    # Specify the task for the pipeline
            device=device_option,      # Device to run on (-1 for CPU, 0 for GPU 0, etc.)
            pipeline_kwargs={          # Arguments passed to the underlying HuggingFace pipeline
                "max_new_tokens": MAX_NEW_TOKENS_LLM, # Max tokens to generate
                # For GPT-2, eos_token_id is often the same as pad_token_id
                # Transformers library often handles this, but explicit setting can avoid warnings
                "pad_token_id": 50256 # GPT-2's pad_token_id, also its eos_token_id
            },
            # model_kwargs can be used for specific model loading options if needed
            # e.g., model_kwargs={"torch_dtype": torch.float16} for GPU memory saving
        )
        print(f"Successfully initialized LLM pipeline with model: {LLM_MODEL_NAME} on {device_name}")
        return hf_pipeline
    except Exception as e:
        # Catch errors during LLM initialization (e.g., model not found, OOM errors)
        print(f"Error initializing LLM '{LLM_MODEL_NAME}': {e}")
        print("Ensure 'transformers', 'torch', and 'accelerate' are installed.")
        print("If using a large model on CPU, ensure you have enough RAM.")
        return None

if __name__ == '__main__':
    # Example usage for testing this module directly
    print("\n--- Testing LLM Interface Initialization ---")
    llm = get_llm()
    if llm:
        print("LLM pipeline loaded successfully.")
        # Optionally test generating text
        try:
            sample_prompt = "Once upon a time"
            print(f"Testing LLM with prompt: '{sample_prompt}'")
            # LangChain LLMs expect a string prompt for `invoke`
            response = llm.invoke(sample_prompt)
            print(f"LLM Response: {response}")
        except Exception as e:
            print(f"Error during LLM test generation: {e}")
    else:
        print("Failed to load LLM pipeline.")
    print("\n--- Testing Complete ---")