# vector_store_manager.py
# Manages creation, loading, and retrieval from ChromaDB vector stores.

import os
import shutil # For deleting directories
from langchain_chroma import Chroma
from config import CHROMA_PERSIST_DIR_BASE # Import base directory for persistence
# Import document processing functions
from document_processing import load_single_document, split_documents_into_chunks

def initialize_collection_vectorstore(
        file_path: str,
        collection_name: str,
        embedding_function,
        force_recreate: bool = False # Set to True to always re-index
    ):
    """
    Initializes a ChromaDB vector store for a specific data collection.
    It loads documents, splits them, embeds them, and stores them.
    If a persisted store already exists, it loads it unless force_recreate is True.

    Args:
        file_path (str): Path to the data file for this collection.
        collection_name (str): Name for the ChromaDB collection.
        embedding_function: The embedding function to use (e.g., HuggingFaceEmbeddings instance).
        force_recreate (bool): If True, deletes any existing persisted store and re-indexes.

    Returns:
        Chroma: An instance of the Chroma vector store for the collection, or None if failed.
    """
    # Construct the persistence directory path for this specific collection
    persist_directory = os.path.join(CHROMA_PERSIST_DIR_BASE, collection_name)
    print(f"\n--- Initializing Vector Store for Collection: {collection_name} ---")
    print(f"Persistence directory: {persist_directory}")

    # If force_recreate is True and the directory exists, delete it
    if force_recreate and os.path.exists(persist_directory):
        print(f"Force recreate: Deleting existing vector store at {persist_directory}")
        try:
            shutil.rmtree(persist_directory)
        except Exception as e:
            print(f"Error deleting directory {persist_directory}: {e}")
            # Continue, try to create anyway or fail if it affects creation

    vectorstore = None
    # Check if the vector store already exists and we are not forcing recreate
    if os.path.exists(persist_directory) and not force_recreate:
        print(f"Loading existing vector store for '{collection_name}' from {persist_directory}")
        try:
            vectorstore = Chroma(
                collection_name=collection_name,
                persist_directory=persist_directory,
                embedding_function=embedding_function
            )
            print(f"Successfully loaded '{collection_name}' vector store.")
        except Exception as e:
            # Catch errors if loading fails (e.g., corrupted store, version mismatch)
            print(f"Error loading existing vector store for '{collection_name}': {e}")
            print("Attempting to re-index...")
            # If loading fails, try to delete and re-index
            try: shutil.rmtree(persist_directory)
            except: pass # Ignore if deletion fails, from_documents might still work or fail cleanly
            vectorstore = None # Ensure it's None to trigger re-indexing
    
    # If vector store doesn't exist (or loading failed/force_recreate was True), create it
    if vectorstore is None:
        print(f"Creating new vector store for '{collection_name}'. Indexing data from: {file_path}")
        
        # 1. Load documents
        documents = load_single_document(file_path)
        if not documents:
            print(f"No documents loaded for {collection_name}. Vector store cannot be created.")
            return None

        # 2. Split documents into chunks
        chunks = split_documents_into_chunks(documents)
        if not chunks:
            print(f"No chunks created for {collection_name}. Vector store cannot be created.")
            return None

        # 3. Create Chroma vector store from chunks
        try:
            print(f"Indexing {len(chunks)} chunks into '{collection_name}'...")
            # Ensure the base directory exists for persistence
            os.makedirs(CHROMA_PERSIST_DIR_BASE, exist_ok=True)
            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embedding_function,
                collection_name=collection_name,
                persist_directory=persist_directory # Persist to this directory
            )
            # Explicitly persist after creation from documents
            vectorstore.persist()
            print(f"Successfully created and indexed '{collection_name}' vector store.")
        except Exception as e:
            # Catch errors during ChromaDB creation/indexing
            print(f"Error creating vector store for '{collection_name}': {e}")
            return None
            
    return vectorstore

def get_retriever_from_vectorstore(vectorstore, search_k=3):
    """
    Creates a retriever from a given Chroma vector store instance.

    Args:
        vectorstore (Chroma): The initialized Chroma vector store.
        search_k (int): The number of top relevant documents to retrieve.

    Returns:
        VectorStoreRetriever: A LangChain retriever object, or None if input is invalid.
    """
    if not vectorstore:
        print("Error: Cannot create retriever from None vector store.")
        return None
    
    try:
        # Create a retriever from the vector store
        # search_kwargs specifies how many documents (k) to retrieve.
        retriever = vectorstore.as_retriever(search_kwargs={"k": search_k})
        print(f"Retriever created for vector store, k={search_k}.")
        return retriever
    except Exception as e:
        print(f"Error creating retriever: {e}")
        return None

if __name__ == '__main__':
    # Example usage for testing this module directly
    from config import CHAT_HISTORY_PATH, HISTORY_COLLECTION_NAME, SEARCH_K
    from embedding_model import get_embedding_function

    print("\n--- Testing Vector Store Manager ---")
    embedding_func = get_embedding_function()

    if embedding_func:
        print(f"\nUsing embedding function: {embedding_func.model_name}")
        # Test initializing the chat_history collection
        # Set force_recreate=True if you want to re-index from scratch for testing
        history_vs = initialize_collection_vectorstore(
            file_path=CHAT_HISTORY_PATH,
            collection_name=HISTORY_COLLECTION_NAME,
            embedding_function=embedding_func,
            force_recreate=False # Set to True to test re-indexing
        )

        if history_vs:
            print(f"Vector store for '{HISTORY_COLLECTION_NAME}' initialized/loaded.")
            # Test creating a retriever
            history_retriever = get_retriever_from_vectorstore(history_vs, search_k=SEARCH_K)
            if history_retriever:
                print("Retriever created successfully.")
                # Test a sample retrieval
                try:
                    sample_query = "deep sleep improvement"
                    print(f"Testing retriever with query: '{sample_query}'")
                    retrieved_docs = history_retriever.invoke(sample_query)
                    print(f"Retrieved {len(retrieved_docs)} documents:")
                    for i, doc in enumerate(retrieved_docs):
                        print(f"Doc {i+1}: {doc.page_content[:150]}...") # Print snippet
                except Exception as e:
                    print(f"Error during sample retrieval: {e}")
        else:
            print(f"Failed to initialize vector store for '{HISTORY_COLLECTION_NAME}'.")
    else:
        print("Failed to load embedding function, cannot test vector store manager.")
    print("\n--- Testing Complete ---")