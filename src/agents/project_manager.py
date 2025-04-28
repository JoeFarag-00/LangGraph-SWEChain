from .utils import create_agent_node
from ..llm_config import llm

pm_prompt = """You are a project manager for a software development team. Your role is to oversee the project execution based on the user's request.
Review the conversation history and the latest message.
Based on the current state of the project, decide the next step.
Your options are:
1.  Delegate to the Architect for design or technical clarification.
2.  Delegate to the Developer to write code.
3.  Delegate to the Tester to review and test the code.
4.  If the project is complete or requires user input, respond with "FINISH".

Provide a clear instruction or question for the chosen team member or state FINISH."""

project_manager_node = create_agent_node(pm_prompt, llm) 