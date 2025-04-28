from langsmith.schemas import Run, Example
from langsmith.evaluation import EvaluationResult
from langchain_core.messages import AIMessage
import re

# --- Custom Evaluator Definitions ---

def check_task_completion(run: Run, example: Example | None = None) -> EvaluationResult:
    """Checks if the final message indicates task completion."""
    # Check if outputs exist and have the expected structure
    if not run.outputs or 'messages' not in run.outputs or not isinstance(run.outputs['messages'], list) or not run.outputs['messages']:
        return EvaluationResult(key="task_completion", score=0, comment="Output format invalid or missing final message.")
    
    # Get the last message, ensuring it's an AIMessage
    final_message = run.outputs['messages'][-1]
    if isinstance(final_message, AIMessage):
        final_content = final_message.content.lower()
        # Simple keyword check for completion indicators
        completion_keywords = ["finish", "complete", "done", "tested successfully", "implemented", "task is complete"]
        if any(keyword in final_content for keyword in completion_keywords):
            return EvaluationResult(key="task_completion", score=1, comment="Final message indicates completion.")
        else:
            return EvaluationResult(key="task_completion", score=0, comment="Final message did not clearly indicate completion.")
            
    # If the last message isn't from the AI, task is likely not complete from agent's perspective
    return EvaluationResult(key="task_completion", score=0, comment="Final message not from AI agent.")

def check_code_generation(run: Run, example: Example | None = None) -> EvaluationResult:
    """Checks if the Developer node produced a code block if the request likely required it."""
    if not example or 'user_request' not in example.inputs:
        return EvaluationResult(key="code_generation_checked", score=None, comment="Missing example or user_request in example inputs.")
        
    # Basic check: Does the input request mention code/function/implement/write?
    input_request = example.inputs['user_request'].lower()
    requires_code_keywords = ["code", "function", "implement", "write", "create", "rest api", "flask", "python", "javascript"]
    requires_code = any(kw in input_request for kw in requires_code_keywords)
    
    # If code is not expected, skip this evaluation for this run
    if not requires_code:
        return EvaluationResult(key="code_generation_checked", score=None, comment="Code generation not expected for this input.")

    # Check run trace for Developer node output containing a markdown code block
    developer_produced_code = False
    # Need to recursively search child runs as the structure might be nested
    queue = list(run.child_runs) if run.child_runs else []
    visited_run_ids = set()

    while queue:
        child_run = queue.pop(0)
        if not child_run or child_run.id in visited_run_ids:
            continue
        visited_run_ids.add(child_run.id)

        # Check if this run is the Developer node
        # Comparing names, ensure node names in the graph builder match this string
        if child_run.name == "Developer":
            if child_run.outputs and isinstance(child_run.outputs, dict) and 'messages' in child_run.outputs:
                dev_messages = child_run.outputs['messages']
                # Check if messages list exists and is not empty
                if dev_messages and isinstance(dev_messages, list) and dev_messages:
                     # Check the first message (usually the AI response) in the list
                    if isinstance(dev_messages[0], AIMessage):
                        # Simple regex for markdown code block (```python ... ``` or ``` ... ```)
                        if re.search(r"```(python)?\n.*?\n```", dev_messages[0].content, re.DOTALL | re.IGNORECASE):
                            developer_produced_code = True
                            break # Found code, no need to search further

        # Add grandchildren to the queue if they exist
        if child_run.child_runs:
            queue.extend(child_run.child_runs)
                        
    score = 1 if developer_produced_code else 0
    comment = "Developer produced code as expected." if score == 1 else "Developer did not produce valid code block when expected."
    return EvaluationResult(key="code_generation_checked", score=score, comment=comment)

# Potential future evaluators:
# - Correct Routing Check (needs analysis of PM output and next actual node)
# - Code Correctness (requires execution or LLM-as-judge)
# - Test Effectiveness (requires analysis of Test node output vs. Dev code) 