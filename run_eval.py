import sys
import os

# Add base directory and src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from eval.runner import EvaluationRunner

if __name__ == "__main__":
    print("Initializing Evaluation Runner...")
    # Specify a dataset name if you want to use a different one
    # runner = EvaluationRunner(dataset_name="My Custom Eval Dataset") 
    runner = EvaluationRunner() 
    
    # Run the evaluations
    runner.run()
    
    print("Evaluation process initiated. Check LangSmith for results.") 