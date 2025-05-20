from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from config import MEMORY_KEY # Import memory key for MessagesPlaceholder

def format_docs(docs: list) -> str:
    if not docs:
        return "No context documents found."
    return "\n\n".join(doc.page_content for doc in docs)

def create_rag_chain(retriever, llm, memory):
   
    if not all([retriever, llm, memory]):
        print("Error: Retriever, LLM, or Memory not provided for RAG chain creation.")
        return None
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
            MessagesPlaceholder(variable_name=MEMORY_KEY),
            ("human", "{question}"),
        ]
    )
    def load_memory_for_chain(input_dict):
        return memory.load_memory_variables(input_dict).get(MEMORY_KEY, [])

    contextualize_question_chain = RunnableParallel(
        context=lambda x: retriever.invoke(x["question"]), 
        question=lambda x: x["question"],                 
        chat_history=lambda x: load_memory_for_chain(x)   
    )
    rag_chain = (
        contextualize_question_chain
        | RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"]))) 
        | prompt_template
        | llm             
        | StrOutputParser() 
    )

    print("RAG chain created successfully.")
    return rag_chain


if __name__ == '__main__':
    from langchain_community.vectorstores import FAISS # Using a simple in-memory store for test
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
    from langchain.schema import Document
    from memory_manager import get_conversation_memory
    from config import EMBEDDING_MODEL_NAME as TEST_EMBEDDING_MODEL_NAME
    from config import LLM_MODEL_NAME as TEST_LLM_MODEL_NAME

    print("\n--- Testing RAG Chain Builder ---")

    print("Setting up dummy components for testing...")
    test_embeddings = HuggingFaceEmbeddings(model_name=TEST_EMBEDDING_MODEL_NAME, model_kwargs={'device':'cpu'})
    print("\n--- Testing Complete ---")
