from langchain_huggingface import HuggingFaceEmbeddings
from config import EMBEDDING_MODEL_NAME # Import the chosen model name

def get_embedding_function():
        embedding_function = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={'device': 'cpu'} # Explicitly set to CPU for wider compatibility
                                          # Change to 'cuda' if you have a compatible GPU setup
        )
        print(f"Successfully initialized embedding model: {EMBEDDING_MODEL_NAME}")
        return embedding_function
    
if __name__ == '__main__':
    print("\n--- Testing Embedding Model Initialization ---")
    embeddings = get_embedding_function()
    if embeddings:
        print("Embedding function loaded successfully.")
    print("\n--- Testing Complete ---")