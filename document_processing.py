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

    # Add specific handling for .jsonl if desired, or let .json handle it
    if ext == ".txt":
        loader = TextLoader(file_path=file_path, encoding="utf-8")
    elif ext == ".csv":
        loader = CSVLoader(file_path=file_path, encoding="utf-8")
    elif ext == ".json" or ext == ".jsonl": # Handling both .json and .jsonl
        # <<< MODIFIED HERE for JSONL structure with 'text_content_for_rag' >>>
        # jq_schema points to the field containing the text you want to make searchable.
        # text_content=True ensures the content of that field becomes Document.page_content.
        # json_lines=True tells it to read one JSON object per line.
        loader = JSONLoader(
            file_path=file_path,
            jq_schema='.[].text_content_for_rag', # Selects the value of "text_content_for_rag" for each object
            text_content=True,                   # Uses the selected content as the document's page_content
            json_lines=True                      # Reads the file as JSON Lines
        )
        # Note: If your JSONL objects do NOT have a "text_content_for_rag" field,
        # and you want the entire JSON object (as a string) to be the content,
        # you would use: jq_schema='.', text_content=False, json_lines=True
    else:
        print(f"Warning: Unsupported file type '{ext}' for {file_path}. Cannot load.")
        return None

    try:
        documents = loader.load()
        if not documents:
            print(f"Warning: No documents loaded from {file_path} (possibly empty or schema mismatch).")
            return None
        # Filter out documents that might have loaded with None as page_content if jq_schema didn't find the field
        documents = [doc for doc in documents if doc.page_content is not None]
        if not documents:
            print(f"Warning: All documents from {file_path} had null content after schema application.")
            return None
        print(f"Successfully loaded {len(documents)} document(s) from {file_path}.")
        return documents
    except Exception as e:
        print(f"Error loading document {file_path}: {e}")
        return None

def split_documents_into_chunks(documents: list):
    """
    Splits a list of LangChain Document objects into smaller chunks.
    (Logic remains the same as previous version)
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

# (The if __name__ == '__main__': block for testing can remain similar,
# but ensure your test files are .jsonl if you update the paths there)
if __name__ == '__main__':
    from config import CHAT_HISTORY_PATH, USER_PROFILE_PATH, WEARABLE_DATA_PATH

    print("\n--- Testing Document Loading and Splitting (with JSONL focus) ---")

    # Test loading a .jsonl file
    print("\nLoading CHAT_HISTORY_PATH (JSONL expected)...")
    jsonl_docs = load_single_document(CHAT_HISTORY_PATH)
    if jsonl_docs:
        jsonl_chunks = split_documents_into_chunks(jsonl_docs)
        if jsonl_chunks:
            print(f"First chunk from JSONL: {jsonl_chunks[0].page_content[:100]}...")
        else:
            print("No chunks created from JSONL.")
    else:
        print(f"Failed to load documents from {CHAT_HISTORY_PATH}")

    print("\n--- Testing Complete ---")