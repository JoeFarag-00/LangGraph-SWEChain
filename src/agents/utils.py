import functools
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from ..state import AgentState # Relative import for AgentState

# Helper function to create a node
def create_agent_node(role_prompt: str, llm):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", role_prompt),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    chain = prompt | llm
    # Pass the chain explicitly to agent_node_func
    return functools.partial(agent_node_func, chain=chain) 

def agent_node_func(state: AgentState, chain):
    # Make sure state['messages'] exists and is not empty if needed by the LLM/chain
    # The state mechanism usually handles accumulation, but good to be aware
    if not state.get('messages'):
         # Handle case with no messages if necessary, maybe return default response or raise error
         # For now, let's assume messages will be populated by the time an agent node is called
         print("Warning: agent_node_func called with empty messages state.")
         # Depending on the chain, invoking with empty messages might be okay or might error
         # If it errors frequently, add more robust handling here.

    response = chain.invoke(state) 
    # Return the standard state update format
    return {"messages": [response]} 