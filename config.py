# config.py
# This file stores configurations for the RAG application.

import os

# --- Data Paths ---
DATA_DIR = "dummy datasets"
# <<< MODIFIED HERE: Changed file extensions to .jsonl >>>
WEARABLE_DATA_PATH = os.path.join(DATA_DIR, "wearable_data.jsonl")
CHAT_HISTORY_PATH = os.path.join(DATA_DIR, "chat_history.jsonl")
USER_PROFILE_PATH = os.path.join(DATA_DIR, "user_profile.jsonl")
LOCATION_DATA_PATH = os.path.join(DATA_DIR, "location_data.jsonl")
CUSTOM_NOTES_PATH = os.path.join(DATA_DIR, "custom_notes.jsonl") # Or whatever you named your custom collection file

# --- Model Names ---
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_NAME = "gpt2"

# --- Vector Store Configuration ---
CHROMA_PERSIST_DIR_BASE = "chroma_db_store"

# Collection names for ChromaDB
COLLECTION_CONFIGS = {
    # <<< MODIFIED HERE: Ensure keys match new file paths if needed for clarity, but paths are primary >>>
    "wearable": {"path": WEARABLE_DATA_PATH, "collection_name": "wearable_data_jsonl"},
    "history": {"path": CHAT_HISTORY_PATH, "collection_name": "chat_history_jsonl"},
    "profile": {"path": USER_PROFILE_PATH, "collection_name": "user_profile_jsonl"},
    "location": {"path": LOCATION_DATA_PATH, "collection_name": "location_data_jsonl"},
    "custom": {"path": CUSTOM_NOTES_PATH, "collection_name": "custom_notes_jsonl"},
}

# --- Text Splitting Parameters ---
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# --- Retriever Parameters ---
SEARCH_K = 3

# --- LLM Generation Parameters ---
MAX_NEW_TOKENS_LLM = 150

# --- Memory Configuration ---
MEMORY_KEY = "chat_history"
MEMORY_INPUT_KEY = "question"
MEMORY_OUTPUT_KEY = "answer"