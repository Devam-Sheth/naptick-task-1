from langchain.memory import ConversationBufferMemory
from config import MEMORY_KEY, MEMORY_INPUT_KEY, MEMORY_OUTPUT_KEY

def get_conversation_memory():
        memory = ConversationBufferMemory(
            memory_key=MEMORY_KEY,
            input_key=MEMORY_INPUT_KEY,
            return_messages=True
        )
        print("ConversationBufferMemory initialized successfully.")
        return memory

if __name__ == '__main__':
   
    print("\n--- Testing Memory Manager ---")
    conv_memory = get_conversation_memory()
    if conv_memory:
        print("Memory object created.")        
        inputs = {"question": "Hello there!"} 
        outputs = {"answer": "Hi! How can I help?"} 
        conv_memory.save_context(inputs, outputs)
        print("Saved context to memory.")
        loaded_vars = conv_memory.load_memory_variables({}) 
        print(f"Loaded memory variables: {loaded_vars}")
        if MEMORY_KEY in loaded_vars and len(loaded_vars[MEMORY_KEY]) == 2:
                print("Memory save and load test successful.")
        else:
                print("Memory save and load test FAILED or produced unexpected result.")

    print("\n--- Testing Complete ---")