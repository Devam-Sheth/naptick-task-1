# rag_chain_builder.py
# Constructs the RAG (Retrieval-Augmented Generation) chain using LangChain Expression Language (LCEL).

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from config import MEMORY_KEY # Import memory key for MessagesPlaceholder

def format_docs(docs: list) -> str:
    """
    Helper function to concatenate the page_content of retrieved documents
    into a single string, separated by double newlines.

    Args:
        docs (list): A list of LangChain Document objects.

    Returns:
        str: A single string containing all document contents.
    """
    if not docs:
        return "No context documents found."
    return "\n\n".join(doc.page_content for doc in docs)

def create_rag_chain(retriever, llm, memory):
    """
    Creates and returns a RAG chain.

    The chain performs the following steps:
    1. Retrieves relevant context documents based on the user's question.
    2. Loads conversation history from memory.
    3. Constructs a prompt using the question, retrieved context, and chat history.
    4. Sends the prompt to the LLM to generate an answer.
    5. Parses the LLM's output into a string.

    Args:
        retriever: An initialized LangChain retriever object (e.g., from ChromaDB).
        llm: An initialized LangChain LLM or ChatModel object.
        memory: An initialized LangChain memory object (e.g., ConversationBufferMemory).

    Returns:
        Runnable: A LangChain runnable (the RAG chain).
    """
    if not all([retriever, llm, memory]):
        print("Error: Retriever, LLM, or Memory not provided for RAG chain creation.")
        return None

    # --- Prompt Template ---
    # This template structures the input to the LLM.
    # It includes system instructions, placeholders for retrieved context,
    # chat history (from memory), and the current user question.
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", (
                "You are a helpful assistant. Answer the user's question based on the "
                "following retrieved context and the chat history. "
                "If the context doesn't directly answer the question, use your general knowledge "
                "but indicate that the information wasn't found in the provided documents. "
                "Keep your answers concise and relevant.\n\n"
                "Retrieved Context:\n{context}\n\n"
                "Chat History (recent first):"
            )),
            MessagesPlaceholder(variable_name=MEMORY_KEY), # Where memory messages will be injected
            ("human", "{question}"), # The user's current question
        ]
    )

    # --- RAG Chain Construction using LCEL ---

    # This function is used to load chat history from the memory object.
    # It's part of the chain that prepares inputs for the main prompt.
    def load_memory_for_chain(input_dict):
        # `load_memory_variables` returns a dict, we need the value of `MEMORY_KEY`
        return memory.load_memory_variables(input_dict).get(MEMORY_KEY, [])

    # This part of the chain prepares the inputs for the prompt_template.
    # It runs in parallel:
    #   - "context": Invokes the retriever with the user's question.
    #   - "question": Passes the original question through.
    #   - "chat_history": Loads the conversation history using the memory object.
    contextualize_question_chain = RunnableParallel(
        context=lambda x: retriever.invoke(x["question"]), # Retrieve context based on question
        question=lambda x: x["question"],                 # Pass question through
        chat_history=lambda x: load_memory_for_chain(x)   # Load chat history
    )

    # This is the main generation chain.
    # It takes the output from contextualize_question_chain (which includes context, question, chat_history),
    # formats the retrieved documents, pipes everything into the prompt template,
    # then to the LLM, and finally parses the LLM output to a string.
    rag_chain = (
        contextualize_question_chain # Output: {"context": docs, "question": str, "chat_history": messages}
        | RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"]))) # Format the retrieved docs
        | prompt_template # Input: {"context": str, "question": str, "chat_history": messages}
        | llm             # Input: PromptValue (formatted prompt)
        | StrOutputParser() # Output: string (the LLM's answer)
    )

    print("RAG chain created successfully.")
    return rag_chain


if __name__ == '__main__':
    # Example usage for testing this module directly
    # This requires setting up dummy retriever, llm, and memory components.
    from langchain_community.vectorstores import FAISS # Using a simple in-memory store for test
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
    from langchain.schema import Document
    from memory_manager import get_conversation_memory
    from config import EMBEDDING_MODEL_NAME as TEST_EMBEDDING_MODEL_NAME
    from config import LLM_MODEL_NAME as TEST_LLM_MODEL_NAME

    print("\n--- Testing RAG Chain Builder ---")

    # 1. Setup dummy components
    print("Setting up dummy components for testing...")
    try:
        test_embeddings = HuggingFaceEmbeddings(model_name=TEST_EMBEDDING_MODEL_NAME, model_kwargs={'device':'cpu'})
        # Create a dummy document and vector store
        dummy_docs = [Document(page_content="LangChain is a framework for LLMs.")]
        dummy_vectorstore = FAISS.from_documents(dummy_docs, test_embeddings)
        test_retriever = dummy_vectorstore.as_retriever()
        print("Dummy retriever created.")

        test_llm = HuggingFacePipeline.from_model_id(
            model_id=TEST_LLM_MODEL_NAME, task="text-generation", device=-1,
            pipeline_kwargs={"max_new_tokens": 50, "pad_token_id": 50256}
        )
        print(f"Dummy LLM ({TEST_LLM_MODEL_NAME}) created.")

        test_memory = get_conversation_memory()
        print("Dummy memory created.")

    except Exception as e:
        print(f"Error setting up dummy components: {e}")
        print("Skipping RAG chain test.")
        test_retriever, test_llm, test_memory = None, None, None

    if test_retriever and test_llm and test_memory:
        # 2. Create RAG chain
        chain = create_rag_chain(test_retriever, test_llm, test_memory)

        if chain:
            print("RAG chain test instance created.")
            # 3. Test invocation
            try:
                test_question = "What is LangChain?"
                print(f"\nInvoking chain with question: '{test_question}'")
                
                # Simulate invoking the chain as `app.py` would (input is a dict)
                response = chain.invoke({"question": test_question})
                print(f"Chain Response: {response}")

                # Test memory saving
                test_memory.save_context({"question": test_question}, {"answer": response})
                loaded_memory = test_memory.load_memory_variables({})
                print(f"Memory after first turn: {loaded_memory}")

                # Test a follow-up question
                follow_up_question = "Tell me more."
                print(f"\nInvoking chain with follow-up: '{follow_up_question}'")
                response_follow_up = chain.invoke({"question": follow_up_question})
                print(f"Chain Follow-up Response: {response_follow_up}")

            except Exception as e:
                print(f"Error invoking RAG chain: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("Failed to create RAG chain for testing.")
    else:
        print("Could not initialize all components for RAG chain testing.")
    print("\n--- Testing Complete ---")