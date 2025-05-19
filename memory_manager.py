# memory_manager.py
# Handles the initialization of conversation memory.

from langchain.memory import ConversationBufferMemory
from config import MEMORY_KEY, MEMORY_INPUT_KEY, MEMORY_OUTPUT_KEY # Import memory config keys

def get_conversation_memory():
    """
    Initializes and returns a ConversationBufferMemory instance.

    ConversationBufferMemory stores previous turns of the conversation
    and can make them available to the LLM, usually via a prompt template.

    Returns:
        ConversationBufferMemory: An instance of the memory buffer.
    """
    try:
        # Initialize ConversationBufferMemory.
        # - `memory_key`: The key in the chain's input/output dictionary where the
        #                 chat history messages will be stored and read from.
        # - `input_key`: The key in the chain's input dictionary that contains the
        #                current user question. Used by memory to know what the input was.
        # - `output_key`: The key in the chain's output dictionary that contains the
        #                 AI's answer. Used by memory to know what the output was.
        # - `return_messages=True`: Configures the memory to return a list of
        #                           ChatMessage objects (BaseMessage instances),
        #                           which is compatible with ChatPromptTemplate's
        #                           MessagesPlaceholder.
        memory = ConversationBufferMemory(
            memory_key=MEMORY_KEY,
            input_key=MEMORY_INPUT_KEY,
            # output_key=MEMORY_OUTPUT_KEY, # output_key is not directly used by ConversationBufferMemory
                                          # when saving context manually via save_context,
                                          # but good to be aware of if using chains that auto-populate memory.
            return_messages=True
        )
        print("ConversationBufferMemory initialized successfully.")
        return memory
    except Exception as e:
        # Catch any errors during memory initialization
        print(f"Error initializing conversation memory: {e}")
        return None

if __name__ == '__main__':
    # Example usage for testing this module directly
    print("\n--- Testing Memory Manager ---")
    conv_memory = get_conversation_memory()
    if conv_memory:
        print("Memory object created.")
        # Test saving and loading context
        try:
            # Simulate a turn
            inputs = {"question": "Hello there!"} # Corresponds to MEMORY_INPUT_KEY
            outputs = {"answer": "Hi! How can I help?"} # 'answer' key matches chain output

            conv_memory.save_context(inputs, outputs)
            print("Saved context to memory.")

            # Load memory variables (chat_history)
            loaded_vars = conv_memory.load_memory_variables({}) # Empty dict for simple buffer
            print(f"Loaded memory variables: {loaded_vars}")
            # Expected: {'chat_history': [HumanMessage(content='Hello there!'), AIMessage(content='Hi! How can I help?')]}

            # Check if history is as expected
            if MEMORY_KEY in loaded_vars and len(loaded_vars[MEMORY_KEY]) == 2:
                print("Memory save and load test successful.")
            else:
                print("Memory save and load test FAILED or produced unexpected result.")

        except Exception as e:
            print(f"Error testing memory: {e}")
    else:
        print("Failed to create memory object.")
    print("\n--- Testing Complete ---")