from typing import TypedDict, Annotated, List, Union
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

class AgentState(TypedDict):
    messages: Annotated[List[Union[HumanMessage, AIMessage, SystemMessage]], lambda x, y: x + y]
    user_info: dict
    # Add other state variables here as needed, e.g., task_status 