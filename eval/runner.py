from langsmith import Client
from langsmith.evaluation import evaluate
from langsmith.schemas import Example, Dataset
from langchain_core.messages import HumanMessage
import os

# Import the compiled graph application
# Assuming the graph is built and compiled in src.graph.builder
from src.graph.builder import compiled_app
# Import evaluators
from .evaluators import check_task_completion, check_code_generation

# Import LLM config details for metadata
from src.llm_config import llm, memory_llm 

class EvaluationRunner:
    """Handles the setup and execution of LangSmith evaluations."""
    
    def __init__(self, dataset_name: str = "Software Dev Agent Evals"):
        self.client = Client()
        self.dataset_name = dataset_name
        self.dataset = self._ensure_dataset()

    def _ensure_dataset(self) -> Dataset:
        """Checks if dataset exists, creates it with default examples if not."""
        if self.client.has_dataset(dataset_name=self.dataset_name):
            print(f"Dataset '{self.dataset_name}' found.")
            # Optionally: Add more examples if needed or check existing ones
            return self.client.read_dataset(dataset_name=self.dataset_name)
        else:
            print(f"Dataset '{self.dataset_name}' not found. Creating...")
            dataset = self.client.create_dataset(
                dataset_name=self.dataset_name, 
                description="Evaluating the multi-agent software dev team."
            )
            print(f"Dataset created with ID: {dataset.id}")
            # Add default examples
            default_examples = [
                {"user_request": "Create a Python function `multiply(a, b)` that returns the product of two numbers."}, 
                {"user_request": "Design a simple REST API endpoint using Flask that takes a name and returns a greeting."}, 
                {"user_request": "Write a function to calculate the factorial of a number, including basic error handling for negative inputs."}, 
                {"user_request": "Refactor this code for clarity: def process(d): return d['value'] * 2"}, # Requires PM to ask for code maybe?
                {"user_request": "Test the `add(a,b)` function previously created."} # Requires context/memory
            ]
            self.client.create_examples(
                inputs=[ex for ex in default_examples],
                # No outputs needed if using custom evaluators on the run trace
                dataset_id=dataset.id,
            )
            print(f"Added {len(default_examples)} default examples to dataset '{self.dataset_name}'.")
            return dataset

    def _system_under_test(self, inputs: dict) -> dict:
        """Function wrapper for the LangGraph app to be evaluated."""
        if 'user_request' not in inputs:
             raise ValueError("Input dictionary must contain 'user_request' key.")
             
        initial_state = {
            "messages": [HumanMessage(content=inputs['user_request'])],
            "user_info": {} 
        }
        # Invoke the compiled LangGraph app
        # Set a recursion limit
        return compiled_app.invoke(initial_state, {"recursion_limit": 50})

    def run(self):
        """Executes the evaluation process."""
        print(f"\n--- Running Evaluations on Dataset: {self.dataset_name} ---")

        # Define evaluators to use
        evaluators = [
            check_task_completion,
            check_code_generation,
            # Add more evaluators from evaluators.py here
        ]

        # Metadata for the experiment
        experiment_metadata = {
            "agent_model": llm.model_name if llm else "Unknown",
            "memory_model": memory_llm.model_name if memory_llm else "Unknown",
            "graph_structure": "Mem -> PM -> [Arch|Dev|Test] -> Summary? -> PM -> END",
            # Add other relevant metadata like date, version, etc.
        }
        
        evaluation_results = evaluate(
            self._system_under_test, # The SUT function
            dataset_name=self.dataset_name, # The dataset to run evaluation on
            evaluators=evaluators,
            experiment_prefix="sw-dev-agent-eval", # Prefix for experiment name in LangSmith
            metadata=experiment_metadata,
            # Optional: Add concurrency level, progress bars etc.
            # max_concurrency=5, 
        )
        print("--- Evaluation Complete --- Results logged to LangSmith.")
        # The results object contains detailed scores, but they are also viewable in LangSmith UI
        # print(evaluation_results) 
        return evaluation_results 