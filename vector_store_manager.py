import os
import shutil 
from langchain_chroma import Chroma
from config import CHROMA_PERSIST_DIR_BASE, COLLECTION_CONFIGS, SEARCH_K
from document_processing import load_single_document, split_documents_into_chunks

def initialize_collection_vectorstore(
        file_path: str,
        collection_name: str,
        embedding_function,
        force_recreate: bool = False # Set to True to always re-index
    ):
    persist_directory = os.path.join(CHROMA_PERSIST_DIR_BASE, collection_name)
    print(f"\n--- Initializing Vector Store for Collection: {collection_name} ---")
    print(f"Persistence directory: {persist_directory}")

    if force_recreate and os.path.exists(persist_directory):
        print(f"Force recreate: Deleting existing vector store at {persist_directory}")
        shutil.rmtree(persist_directory)
        
    vectorstore = None
    if os.path.exists(persist_directory) and not force_recreate:
        print(f"Loading existing vector store for '{collection_name}' from {persist_directory}")
        vectorstore = Chroma(
                collection_name=collection_name,
                persist_directory=persist_directory,
                embedding_function=embedding_function
            )
        print(f"Successfully loaded '{collection_name}' vector store.")
    
    if vectorstore is None:
        print(f"Creating new vector store for '{collection_name}'. Indexing data from: {file_path}")
        
        documents = load_single_document(file_path)
        if not documents:
            print(f"No documents loaded for {collection_name}. Vector store cannot be created.")
            return None

        chunks = split_documents_into_chunks(documents)
        if not chunks:
            print(f"No chunks created for {collection_name}. Vector store cannot be created.")
            return None

        print(f"Indexing {len(chunks)} chunks into '{collection_name}'...")
        os.makedirs(CHROMA_PERSIST_DIR_BASE, exist_ok=True) # Ensure base directory exists
        vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embedding_function,
                collection_name=collection_name,
                persist_directory=persist_directory # Persistence handled here
            )
        print(f"Successfully created and indexed '{collection_name}' vector store. Data should be persisted to {persist_directory}.")
            
    return vectorstore

def get_retriever_from_vectorstore(vectorstore, search_k=3):
    if not vectorstore:
        print("Error: Cannot create retriever from None vector store.")
        return None

    retriever = vectorstore.as_retriever(search_kwargs={"k": search_k})
    print(f"Retriever created for vector store, k={search_k}.")
    return retriever

if __name__ == '__main__':
    
    from config import COLLECTION_CONFIGS, SEARCH_K 
    from embedding_model import get_embedding_function

    test_collection_key = "history"
    test_file_path = COLLECTION_CONFIGS[test_collection_key]["path"]
    test_collection_name = COLLECTION_CONFIGS[test_collection_key]["collection_name"]

    print("\n--- Testing Vector Store Manager ---")
    embedding_func = get_embedding_function()

    if embedding_func:
        print(f"\nUsing embedding function (model: ensure config.EMBEDDING_MODEL_NAME is set)")
        
        history_vs = initialize_collection_vectorstore(
            file_path=test_file_path,
            collection_name=test_collection_name,
            embedding_function=embedding_func,
            force_recreate=False 
        )

        if history_vs:
            print(f"Vector store for '{test_collection_name}' initialized/loaded.")
            history_retriever = get_retriever_from_vectorstore(history_vs, search_k=SEARCH_K)
            if history_retriever:
                print("Retriever created successfully.")
                sample_query = "deep sleep improvement"
                print(f"Testing retriever with query: '{sample_query}'")
                retrieved_docs = history_retriever.invoke(sample_query)
                print(f"Retrieved {len(retrieved_docs)} documents:")
                for i, doc in enumerate(retrieved_docs):
                        print(f"Doc {i+1}: {doc.page_content[:150]}...")
    print("\n--- Testing Complete ---")