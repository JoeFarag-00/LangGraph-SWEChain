from ..state import AgentState
# Import the utility LLM for extraction
from ..llm_config import utility_llm, llm
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
import json # Import json for parsing potentially structured output

# --- Special Node Definitions ---

def memory_extraction_node(state: AgentState):
    """Extracts user information from the latest message using an LLM call and updates state."""
    print("---\nEXTRACTING MEMORY")
    # Ensure messages exist and are not empty
    if not state.get('messages'):
        print("Memory Extractor: No messages in state to process.")
        return {}
    
    last_message = state['messages'][-1]
    # Only extract from HumanMessage for now
    if isinstance(last_message, HumanMessage):
        try:
            extraction_prompt = ChatPromptTemplate.from_messages([
                # Escape the literal curly braces for the example empty JSON object
                ("system", "You are an information extraction assistant. Extract key details about the user (name, preferences, location, explicit requests, etc.) from the following message. Output the extracted information as a JSON object. If no specific user details are mentioned, return an empty JSON object {{}}."),
                ("human", "{user_message}")
            ])
            
            # Use the utility_llm for this task
            extraction_chain = extraction_prompt | utility_llm 
            extracted_data_str = extraction_chain.invoke({"user_message": last_message.content}).content
            
            print(f"Raw Extracted Data String: {extracted_data_str}")
            
            # Attempt to parse the JSON output
            try:
                # Clean potential markdown code fences
                if extracted_data_str.startswith("```json"):
                     extracted_data_str = extracted_data_str[7:-3].strip()
                elif extracted_data_str.startswith("```"):
                     extracted_data_str = extracted_data_str[3:-3].strip()
                
                extracted_data = json.loads(extracted_data_str)
                if not isinstance(extracted_data, dict):
                     print("Extraction Error: Parsed data is not a dictionary.")
                     extracted_data = {} # Default to empty dict if not a dict
                     
            except json.JSONDecodeError as json_e:
                print(f"Extraction JSON Parsing Error: {json_e}")
                # Keep the raw string if parsing fails? Or return empty?
                extracted_data = {"raw_extraction": extracted_data_str} # Store raw if parsing fails
                
            print(f"Parsed Extracted Data: {extracted_data}")
            
            # Merge with existing user_info if necessary (simple overwrite here)
            current_info = state.get('user_info', {})
            # Only update if extracted_data is a dict
            if isinstance(extracted_data, dict):
                 current_info.update(extracted_data)
                 return {"user_info": current_info}
            else:
                 # If extraction didn't yield a dict, return no update to user_info
                 return {}
                 
        except Exception as e:
            print(f"Error during memory extraction LLM call: {e}")
            return {}
            
    return {} # Return empty dict if no extraction or not a human message

def summary_node(state: AgentState):
    """Checks message length and summarizes if it exceeds the threshold."""
    print("---\nCHECKING MESSAGE LENGTH FOR SUMMARY")
    # Ensure messages exist
    messages = state.get('messages', [])
    if not messages:
        print("Summary Node: No messages in state to summarize.")
        return {}
        
    message_count = len(messages)
    summary_threshold = 25 # Configurable threshold

    if message_count > summary_threshold:
        print(f"Message count ({message_count}) exceeds threshold ({summary_threshold}). Summarizing.")
        # Keep the first message if it's a System prompt
        first_message = messages[0] if messages and isinstance(messages[0], SystemMessage) else None
        
        summary_prompt_messages = [
            SystemMessage(content="Summarize the following conversation history concisely, capturing the key tasks, decisions, and outcomes."),
            # Maybe include user info? state.get('user_info', {})
            HumanMessage(content="\n".join([f"{type(m).__name__}: {m.content}" for m in messages]))
        ]
        
        # Using llm imported from llm_config
        summary_chain = ChatPromptTemplate.from_messages(summary_prompt_messages) | llm
        summary = summary_chain.invoke({})
        
        print(f"Summary: {summary.content}")
        
        # Construct new message list with potential system prompt and the summary
        new_messages = [summary] # Summary is an AIMessage
        if first_message:
            new_messages.insert(0, first_message)
            
        return {"messages": new_messages}
    else:
        print(f"Message count ({message_count}) is within threshold. No summary needed.")
        return {} # No change to state 