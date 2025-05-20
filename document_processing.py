# document_processing.py
# Handles loading documents from files and splitting them into chunks.

import os
from langchain_community.document_loaders import (
    TextLoader,
    CSVLoader,
    JSONLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP

def load_single_document(file_path: str):
    """
    Loads a single document from the given file_path.
    Determines the loader based on the file extension.

    Args:
        file_path (str): The path to the document file.

    Returns:
        list: A list of LangChain Document objects, or None if loading fails.
    """
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    if ext == ".txt":
        loader = TextLoader(file_path=file_path, encoding="utf-8")
    elif ext == ".csv":
        loader = CSVLoader(file_path=file_path, encoding="utf-8")
    elif ext == ".json" or ext == ".jsonl": # Handling both .json and .jsonl
        # <<< CORRECTED jq_schema HERE >>>
        loader = JSONLoader(
            file_path=file_path,
            jq_schema='.text_content_for_rag', # Correct schema for JSONL when processing line by line
            text_content=True,                 # Uses the selected content as the document's page_content
            json_lines=True                    # Reads the file as JSON Lines
        )
    else:
        print(f"Warning: Unsupported file type '{ext}' for {file_path}. Cannot load.")
        return None

    try:
        documents = loader.load()
        if not documents:
            print(f"Warning: No documents loaded from {file_path} (possibly empty or schema mismatch).")
            return None
        documents = [doc for doc in documents if doc.page_content is not None and doc.page_content.strip() != ""]
        if not documents:
            print(f"Warning: All documents from {file_path} had null or empty content after schema application.")
            return None
        print(f"Successfully loaded {len(documents)} document(s) from {file_path}.")
        return documents
    except Exception as e:
        print(f"Error loading document {file_path}: {e}")
        import traceback
        traceback.print_exc() # Print full traceback for debugging
        return None

def split_documents_into_chunks(documents: list):
    """
    Splits a list of LangChain Document objects into smaller chunks.
    """
    if not documents:
        print("Warning: No documents provided for splitting.")
        return []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    try:
        chunks = text_splitter.split_documents(documents)
        print(f"Split documents into {len(chunks)} chunks.")
        return chunks
    except Exception as e:
        print(f"Error splitting documents: {e}")
        return []

if __name__ == '__main__':
    from config import CHAT_HISTORY_PATH # Using CHAT_HISTORY_PATH from config

    print("\n--- Testing Document Loading and Splitting (with JSONL focus) ---")

    print(f"\nLoading CHAT_HISTORY_PATH ('{CHAT_HISTORY_PATH}') (JSONL expected)...")
    jsonl_docs = load_single_document(CHAT_HISTORY_PATH)
    if jsonl_docs:
        print(f"Successfully loaded {len(jsonl_docs)} documents from chat history.")
        # Print the page_content of the first loaded document to verify
        if jsonl_docs[0].page_content:
            print(f"Content of first loaded document: '{jsonl_docs[0].page_content[:200]}...'")
        else:
            print("First loaded document has no page_content.")

        jsonl_chunks = split_documents_into_chunks(jsonl_docs)
        if jsonl_chunks:
            print(f"First chunk from JSONL: {jsonl_chunks[0].page_content[:100]}...")
        else:
            print("No chunks created from JSONL.")
    else:
        print(f"Failed to load documents from {CHAT_HISTORY_PATH}")

    print("\n--- Testing Complete ---")
    