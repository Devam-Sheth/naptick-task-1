import os
import traceback
import config
from document_processing import load_single_document, split_documents_into_chunks
from embedding_model import get_embedding_function
from llm_interface import get_llm
from vector_store_manager import initialize_collection_vectorstore, get_retriever_from_vectorstore
from memory_manager import get_conversation_memory
from rag_chain_builder import create_rag_chain

def run_rag_chatbot():
    
    print("--- Starting RAG Chatbot Initialization ---")

    print("\nStep 1: Initializing Embedding Model...")
    embedding_function = get_embedding_function()

    print("\nStep 2: Initializing Language Model (LLM)...")
    llm = get_llm()

    print("\nStep 3: Initializing Conversation Memory...")
    conversation_memory = get_conversation_memory()

    print("\nStep 4: Initializing Vector Stores and Retrievers for Data Collections...")
    retrievers = {}

    for key, collection_info in config.COLLECTION_CONFIGS.items(): 
        print(f"  Processing collection: {collection_info['collection_name']}")
        vectorstore = initialize_collection_vectorstore(
            file_path=collection_info['path'],
            collection_name=collection_info['collection_name'],
            embedding_function=embedding_function,
            force_recreate=False
        )
        if vectorstore:
            retriever = get_retriever_from_vectorstore(vectorstore, search_k=config.SEARCH_K)
            if retriever:
                retrievers[collection_info['collection_name']] = retriever
                print(f"  Successfully initialized retriever for {collection_info['collection_name']}.")

    history_key_in_config = "history"
    if history_key_in_config in config.COLLECTION_CONFIGS:
        chosen_retriever_key = config.COLLECTION_CONFIGS[history_key_in_config]["collection_name"]
    active_retriever = retrievers[chosen_retriever_key]
    print(f"Using retriever for '{chosen_retriever_key}' for RAG chain.")
    print("\nStep 5: Creating RAG Chain...")
    rag_chain = create_rag_chain(
        retriever=active_retriever, 
        llm=llm,
        memory=conversation_memory
    )

    print("\n--- RAG Chatbot Setup Complete ---")
    print("You can now ask questions. Type 'exit' or 'quit' to end.")

    while True:
        try:
            user_question = input("You: ")
            if user_question.lower().strip() in ["exit", "quit"]:
                print("Exiting chatbot...")
                break
            if not user_question.strip():
                continue
            chain_input = {"question": user_question}
            
            print("Assistant processing...")
            result = rag_chain.invoke(chain_input)
            ai_answer = result

            conversation_memory.save_context(
                {config.MEMORY_INPUT_KEY: user_question}, 
                {config.MEMORY_OUTPUT_KEY: ai_answer}
            )

            print(f"Assistant: {ai_answer}")

        except KeyboardInterrupt:
            print("\nExiting chatbot due to user interruption...")
            break

if __name__ == "__main__":
    run_rag_chatbot()