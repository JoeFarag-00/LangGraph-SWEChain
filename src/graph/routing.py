from langgraph.graph import END
from ..state import AgentState
from langchain_core.messages import AIMessage

# --- Routing Logic Definitions ---

def route_from_project_manager(state: AgentState):
    """Routes from ProjectManager to other agents or END."""
    # Ensure messages exist
    messages = state.get('messages', [])
    if not messages:
        print("Routing Error: No messages in state to route from Project Manager.")
        return END # Default to end if state is unexpected
        
    last_message = messages[-1]
    # Assuming the PM's response clearly indicates the next step
    # Simple routing based on keywords in the PM's message
    if isinstance(last_message, AIMessage): # PM is an AI agent
        content = last_message.content.lower()
        if "architect" in content:
            print("Routing: PM -> Architect")
            return "Architect"
        elif "developer" in content or "code" in content or "implement" in content:
            print("Routing: PM -> Developer")
            return "Developer"
        elif "tester" in content or "test" in content:
            print("Routing: PM -> Tester")
            return "Tester"
        elif "finish" in content or "complete" in content or "done" in content:
            print("Routing: PM -> END")
            return END
            
    # Default or fallback if no clear routing instruction or not an AIMessage
    print("Routing: PM -> END (Default/Fallback)")
    return END 

def route_to_summary_or_pm(state: AgentState):
    """Routes to Summary node if message count exceeds threshold, otherwise to ProjectManager."""
    messages = state.get('messages', [])
    message_count = len(messages)
    summary_threshold = 25 # Must match the threshold in summary_node
    
    if message_count > summary_threshold:
        print(f"Routing: -> Summary (Count: {message_count})")
        return "Summary"
    else:
        print(f"Routing: -> ProjectManager (Count: {message_count})")
        return "ProjectManager" 