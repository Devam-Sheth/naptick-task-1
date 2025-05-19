# app.py
# Main Command Line Interface (CLI) application for the RAG-based Chatbot.

import os
import traceback # For detailed error printing

# --- Import configurations and module functions ---
import config # To access paths, model names, etc.
from document_processing import load_single_document, split_documents_into_chunks
from embedding_model import get_embedding_function
from llm_interface import get_llm
from vector_store_manager import initialize_collection_vectorstore, get_retriever_from_vectorstore
from memory_manager import get_conversation_memory
from rag_chain_builder import create_rag_chain

def run_rag_chatbot():
    """
    Initializes all components and runs the main chat loop for the RAG chatbot.
    """
    print("--- Starting RAG Chatbot Initialization ---")

    # 1. Initialize Embedding Function
    print("\nStep 1: Initializing Embedding Model...")
    embedding_function = get_embedding_function()
    if not embedding_function:
        print("CRITICAL: Failed to initialize embedding function. Exiting.")
        return

    # 2. Initialize LLM
    print("\nStep 2: Initializing Language Model (LLM)...")
    llm = get_llm()
    if not llm:
        print("CRITICAL: Failed to initialize LLM. Exiting.")
        return

    # 3. Initialize Conversation Memory
    print("\nStep 3: Initializing Conversation Memory...")
    conversation_memory = get_conversation_memory()
    if not conversation_memory:
        print("CRITICAL: Failed to initialize conversation memory. Exiting.")
        return

    # 4. Initialize Vector Stores and Retrievers for all collections
    print("\nStep 4: Initializing Vector Stores and Retrievers for Data Collections...")
    retrievers = {} # Dictionary to hold retrievers for each collection
    all_retrievers_initialized = True

    # Loop through each collection defined in config.py
    for key, collection_info in config.COLLECTION_CONFIGS.items():
        print(f"  Processing collection: {collection_info['collection_name']}")
        # Initialize (load or create) the vector store for this collection
        vectorstore = initialize_collection_vectorstore(
            file_path=collection_info['path'],
            collection_name=collection_info['collection_name'],
            embedding_function=embedding_function,
            # Set force_recreate=True if you want to re-index on every run (for debugging/development)
            force_recreate=False
        )
        if vectorstore:
            # Create a retriever from the vector store
            retriever = get_retriever_from_vectorstore(vectorstore, search_k=config.SEARCH_K)
            if retriever:
                retrievers[collection_info['collection_name']] = retriever
                print(f"  Successfully initialized retriever for {collection_info['collection_name']}.")
            else:
                print(f"  WARNING: Failed to create retriever for {collection_info['collection_name']}.")
                all_retrievers_initialized = False # Mark that at least one failed
        else:
            print(f"  WARNING: Failed to initialize vector store for {collection_info['collection_name']}.")
            all_retrievers_initialized = False

    if not retrievers: # If no retrievers were successfully initialized
        print("CRITICAL: No retrievers were initialized. Cannot proceed with RAG. Exiting.")
        return
    
    # --- IMPORTANT: Multi-Collection Retrieval Strategy ---
    # For this starter code, we will just use ONE retriever (e.g., from chat_history).
    # A full solution for Task 1 needs a strategy to query across MULTIPLE retrievers
    # or select the appropriate one(s) based on the user's question.
    # This might involve:
    #   - LangChain Agents with tools (each retriever as a tool).
    #   - LangChain Router Chains (e.g., MultiRetrievalQAChain).
    #   - Custom logic to query all/selected retrievers and merge results.
    # This part is a significant extension point for Task 1.
    print("\nNOTE: This starter uses a single retriever for simplicity.")
    print("      Full multi-collection retrieval needs to be implemented.")
    
    # Using the 'chat_history' retriever as an example.
    # Replace this with your multi-collection retrieval logic.
    chosen_retriever_key = config.HISTORY_COLLECTION_NAME # Default to history
    if chosen_retriever_key not in retrievers:
        # Fallback if default key isn't available, pick first available
        if retrievers:
            chosen_retriever_key = list(retrievers.keys())[0]
            print(f"Warning: Default retriever '{config.HISTORY_COLLECTION_NAME}' not found. Using '{chosen_retriever_key}' instead.")
        else:
            # This case should have been caught by 'if not retrievers:' above, but defensive check.
            print("CRITICAL: No retrievers available at all. Exiting.")
            return
            
    active_retriever = retrievers[chosen_retriever_key]
    print(f"Using retriever for '{chosen_retriever_key}' for RAG chain.")
    # ------------------------------------------------------

    # 5. Create the RAG Chain
    print("\nStep 5: Creating RAG Chain...")
    rag_chain = create_rag_chain(
        retriever=active_retriever, # Pass the chosen (single) retriever
        llm=llm,
        memory=conversation_memory
    )
    if not rag_chain:
        print("CRITICAL: Failed to create RAG chain. Exiting.")
        return

    print("\n--- RAG Chatbot Setup Complete ---")
    print("You can now ask questions. Type 'exit' or 'quit' to end.")

    # 6. Start CLI Chat Loop
    while True:
        try:
            user_question = input("You: ")
            if user_question.lower().strip() in ["exit", "quit"]:
                print("Exiting chatbot...")
                break

            if not user_question.strip():
                continue # Skip empty input

            # Prepare input for the chain (as expected by memory and chain structure)
            chain_input = {"question": user_question}
            
            print("Assistant processing...")
            # Invoke the RAG chain
            result = rag_chain.invoke(chain_input)
            ai_answer = result # In this basic LCEL chain, the direct output is the answer string

            # Save context to memory
            # The memory object itself is updated by the chain if configured internally,
            # or we save it manually if our chain requires it.
            # Our ConversationBufferMemory needs manual saving.
            conversation_memory.save_context(
                {config.MEMORY_INPUT_KEY: user_question}, # Inputs to memory
                {config.MEMORY_OUTPUT_KEY: ai_answer}    # Outputs to memory
            )

            print(f"Assistant: {ai_answer}")

            # Optional: Print retrieved context for debugging (if chain returns it)
            # if "context" in result:
            #     print("\n--- Retrieved Context (for debugging) ---")
            #     for i, doc in enumerate(result['context']):
            #         print(f"Doc {i+1}: {doc.page_content[:100]}...")
            #     print("--------------------------------------")

        except KeyboardInterrupt:
            print("\nExiting chatbot due to user interruption...")
            break
        except Exception as e:
            print(f"\nAn error occurred in the chat loop: {e}")
            traceback.print_exc()
            # Optionally, reset memory or take other recovery actions
            # For now, we just continue the loop
            print("Please try again or type 'exit' to quit.")

if __name__ == "__main__":
    run_rag_chatbot()