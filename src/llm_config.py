from langchain_groq import ChatGroq
# from langmem import create_memory_from_config # Removed langmem dependency
from dotenv import load_dotenv
import os

load_dotenv()

# Use a more capable model if possible for the main agent logic
llm = ChatGroq(model_name=os.getenv("GROQ_MODEL_NAME", "llama3-70b-8192"))
# Keep a smaller/faster model for potential utility tasks like extraction/summarization if needed
utility_llm = ChatGroq(model_name=os.getenv("GROQ_UTILITY_MODEL_NAME", "llama3-8b-8192")) 

# # Initialize LangMem Memory - Removed
# memory_instance = create_memory_from_config({
#     "memory_type": "basic",
#     "llm": memory_llm, # Was memory_llm
# }) 