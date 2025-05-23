# Naptick AI Challenge - Task 1: Multi-Collection RAG System with Memory

**GitHub Repository:** [https://github.com/Devam-Sheth/naptick-task-1](https://github.com/Devam-Sheth/naptick-task-1) 

## Task Description

The objective of Task 1 is to design and build an AI chatbot capable of retrieving information from multiple structured and semi-structured data collections. The system must implement a Retrieval-Augmented Generation (RAG) pipeline, include a memory layer to maintain context over multiple queries, and be demonstrable via a simple CLI.

This project implements a modular RAG system using LangChain, local embedding models, a local LLM, and a local vector store (ChromaDB) to query across five distinct (dummy) data collections.

## Features Implemented

* **Multi-Collection Data Handling:**
    * Loads data from 5 distinct dummy collections: Wearable Data, Chat History, User Profile, Location Data, and Custom Notes (as `.jsonl` files).
    * Creates and persists a separate vector store (ChromaDB collection) for each data source.
* **RAG Pipeline:**
    * **Loading:** Loads `.jsonl` files, extracting relevant text content.
    * **Chunking:** Splits documents into manageable chunks using `RecursiveCharacterTextSplitter`.
    * **Embedding:** Uses `sentence-transformers/all-MiniLM-L6-v2` for generating text embeddings locally.
    * **Storage:** Stores embeddings in persistent ChromaDB collections.
    * **Retrieval:** Fetches relevant chunks from the vector stores. *(Note: The current `app.py` starter uses a single retriever as a placeholder; full multi-collection retrieval logic is a primary extension point).*
    * **Generation:** Uses a local `gpt2` model via `HuggingFacePipeline` to generate answers based on retrieved context and chat history.
* **Memory Layer:** Implements `ConversationBufferMemory` from LangChain to store and recall chat history, providing context to the LLM for follow-up questions.
* **CLI Interface:** A simple command-line interface (`app.py`) allows users to interact with the chatbot.

## Technology Stack

* **Core AI/NLP Framework:** LangChain
* **LLM:** `gpt2` (via Hugging Face `transformers` and `HuggingFacePipeline`)
* **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2` (via `langchain-huggingface`)
* **Vector Store:** `ChromaDB` (via `langchain-chroma`)
* **Core Libraries:** `torch`, `datasets` (for potential future use, not strictly in current loader), `os`, `shutil`, `traceback`.

## Setup and Installation Instructions

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/Devam-Sheth/naptick-task-1.git
    cd naptick-task-1
    ```

2.  **Create and Activate Virtual Environment (Highly Recommended):**
    ```bash
    python -m venv .venv
    # On Windows (PowerShell/CMD):
    .\.venv\Scripts\activate
    # On macOS/Linux (Bash/Zsh):
    # source .venv/bin/activate
    ```

3.  **Install Python Dependencies:**
    Install all required Python libraries using the `requirements.txt` and `requirements1.txt` file (to eradicate the error of 'resolution-too-deep' error).
    ```bash
    # Ensure pip is up-to-date within the venv
    python -m pip install --upgrade pip
    # Install requirements
    pip install -r requirements.txt
    pip install -r requirements1.txt
    ```
## Running the RAG Chatbot

1.  Navigate to the root of the cloned repository in your terminal (where `app.py` is located), with your virtual environment activated.
2.  Execute the main application script:
    ```bash
    python app.py
    ```
    *(Or `py -3.10 app.py` if you need to specify your Python 3.10 interpreter explicitly).*

3.  **First Run:**
    * The application will download the embedding model (`sentence-transformers/all-MiniLM-L6-v2`) and the LLM (`gpt2`) from Hugging Face Hub if they are not already cached locally. This might take a few minutes depending on your internet connection.
    * It will then process each of your 5 dummy dataset files: loading, chunking, embedding, and indexing them into separate ChromaDB collections. This data will be persisted in a `chroma_db_store` directory. This initial indexing process can also take some time.
    * You will see log messages indicating the progress of these steps.

4.  **Subsequent Runs:**
    * Model downloads will be skipped if models are cached.
    * Vector stores will be loaded from the `chroma_db_store` directory, making startup much faster.

5.  **Interacting with the Chatbot:**
    * Once initialization is complete, you will see:
        ```
        --- RAG Chatbot Setup Complete ---
        You can now ask questions. Type 'exit' or 'quit' to end.
        You:
        ```
    * Type your question and press Enter. The assistant will process it and provide a response.
    * To end the session, type `exit` or `quit`.

## File Structure (Key Files in Repository)

* `app.py`: Main CLI application; orchestrates all components.
* `config.py`: Stores all configurations (data paths, model names, vector store settings).
* `document_processing.py`: Handles loading of data files and splitting them into chunks.
* `embedding_model.py`: Initializes the sentence transformer embedding model.
* `llm_interface.py`: Initializes the local LLM (GPT-2) pipeline.
* `vector_store_manager.py`: Manages the creation, persistence, and loading of ChromaDB vector stores for each data collection and provides retriever objects.
* `memory_manager.py`: Initializes and manages conversation history.
* `rag_chain_builder.py`: Defines and constructs the RAG chain using LangChain Expression Language (LCEL).
* `dummy_datasets/`: Folder containing the 5 `.jsonl` dummy data files.
    * `wearable_data.jsonl`
    * `chat_history.jsonl`
    * `user_profile.jsonl`
    * `location_data.jsonl`
    * `custom_notes.jsonl`
* `requirements.txt` and `requirements1.txt`: Lists Python dependencies.
* `README.md`: This file.

## Excluded Files (Not in GitHub Repository)

* `.venv/`: Python virtual environment directory.
* `chroma_db_store/`: Locally persisted ChromaDB vector stores.
* `__pycache__/`: Python bytecode cache directories.

## Limitations

* **LLM Quality:** Uses `gpt2`, a small base model. Responses will be basic and may lack the nuanced understanding or instruction-following capabilities of larger or API-based models. The focus here is on the RAG architecture.
* **Multi-Collection Retrieval:** As noted, the current chain uses a single retriever. Implementing robust multi-collection retrieval is the primary next step.
* **Proactive Suggestions (Bonus):** Not implemented in this starter code. This would require an additional layer of logic to analyze context and trigger suggestions.
