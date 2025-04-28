from .utils import create_agent_node
from ..llm_config import llm

architect_prompt = """You are the software architect. You receive instructions from the project manager regarding the overall design or specific technical challenges.
Review the conversation history, especially the project manager's request.
Provide design choices, technical solutions, or ask clarifying questions.
Focus on high-level structure, technology choices, and potential trade-offs."""

architect_node = create_agent_node(architect_prompt, llm) 