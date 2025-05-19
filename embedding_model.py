# embedding_model.py
# Handles the initialization of the embedding model.

from langchain_huggingface import HuggingFaceEmbeddings
from config import EMBEDDING_MODEL_NAME # Import the chosen model name

def get_embedding_function():
    """
    Initializes and returns the HuggingFace embedding function.

    This function loads a pre-trained model from Hugging Face (specified in config.py)
    that can convert text into numerical vectors (embeddings).
    These embeddings are used by the vector store for similarity searches.

    Returns:
        HuggingFaceEmbeddings: An instance of the embedding model.
                                Returns None if initialization fails.
    """
    try:
        # Initialize HuggingFaceEmbeddings with the specified model name.
        # model_kwargs can be used to specify the device (e.g., {'device': 'cpu'} or {'device': 'cuda'})
        # By default, it will try to use CUDA if available, otherwise CPU.
        embedding_function = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={'device': 'cpu'} # Explicitly set to CPU for wider compatibility
                                          # Change to 'cuda' if you have a compatible GPU setup
        )
        print(f"Successfully initialized embedding model: {EMBEDDING_MODEL_NAME}")
        return embedding_function
    except Exception as e:
        # Catch any errors during initialization (e.g., model not found, network issues)
        print(f"Error initializing embedding model '{EMBEDDING_MODEL_NAME}': {e}")
        return None

if __name__ == '__main__':
    # Example usage for testing this module directly
    print("\n--- Testing Embedding Model Initialization ---")
    embeddings = get_embedding_function()
    if embeddings:
        print("Embedding function loaded successfully.")
        # Optionally test creating an embedding for a sample text
        try:
            sample_text = "This is a test sentence."
            sample_embedding = embeddings.embed_query(sample_text)
            print(f"Successfully embedded sample text. Vector dimension: {len(sample_embedding)}")
        except Exception as e:
            print(f"Error embedding sample text: {e}")
    else:
        print("Failed to load embedding function.")
    print("\n--- Testing Complete ---")