import sys
import os

# Add src directory to the Python path
# This allows importing modules from src like src.graph.builder
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from langchain_core.messages import HumanMessage
from src.graph.builder import compiled_app # Import the compiled application

def run_interaction(user_input: str):
    """Runs a single interaction with the multi-agent system."""
    initial_state = {
        "messages": [HumanMessage(content=user_input)],
        "user_info": {} # Initialize user_info
    }

    print(f"\n--- Running Interaction ---")
    print(f"Input: {user_input}")
    print("-------------------------")
    
    # Stream the execution for visualization
    for event in compiled_app.stream(initial_state, {"recursion_limit": 50}): 
        for node_name, output in event.items():
            # Print output from each node
            print(f"--- Node: {node_name} ---")
            # Output is the state dict update, print messages if available
            if isinstance(output, dict) and 'messages' in output and output['messages']:
                print(f"Messages: {output['messages']}")
            elif isinstance(output, dict) and 'user_info' in output:
                 print(f"User Info Extracted: {output['user_info']}")
            else:
                # Print the raw output if it's not the expected message format
                print(f"Raw Output: {output}") 
            print("\n=====================\n")
    
    # Get the final state
    final_state = compiled_app.invoke(initial_state, {"recursion_limit": 50})
    print("--- Final State ---")
    # Ensure the final state message exists and is accessible
    if final_state and 'messages' in final_state and isinstance(final_state['messages'], list) and final_state['messages']:
         final_msg_content = final_state['messages'][-1].content
         print(f"Final Message: {final_msg_content}")
    else:
        print("Could not retrieve final message from state.")
    print("-------------------")
    return final_state

if __name__ == "__main__":
    # Example: Get input from command line argument or use default
    if len(sys.argv) > 1:
        request = " ".join(sys.argv[1:])
    else:
        request = "Please design and implement a simple Python function that adds two numbers. Then test it."
    
    run_interaction(request) 