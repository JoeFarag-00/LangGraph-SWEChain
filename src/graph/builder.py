from langgraph.graph import StateGraph, END
from ..state import AgentState

# Import agent nodes
from ..agents.project_manager import project_manager_node
from ..agents.architect import architect_node
from ..agents.developer import developer_node
from ..agents.tester import tester_node

# Import special nodes
from .nodes import memory_extraction_node, summary_node

# Import routing functions
from .routing import route_from_project_manager, route_to_summary_or_pm

def build_graph():
    """Builds and compiles the LangGraph workflow."""
    workflow = StateGraph(AgentState)

    # Add nodes
    print("Building Graph: Adding nodes...")
    workflow.add_node("MemoryExtractor", memory_extraction_node)
    workflow.add_node("ProjectManager", project_manager_node)
    workflow.add_node("Architect", architect_node)
    workflow.add_node("Developer", developer_node)
    workflow.add_node("Tester", tester_node)
    workflow.add_node("Summary", summary_node)

    # Set entry point
    workflow.set_entry_point("MemoryExtractor")
    print("Building Graph: Entry point set to MemoryExtractor.")

    # Add edges
    print("Building Graph: Adding edges...")
    workflow.add_edge("MemoryExtractor", "ProjectManager")
    workflow.add_edge("Summary", "ProjectManager") # Summary node goes back to PM

    # Add conditional edge from Project Manager
    workflow.add_conditional_edges(
        "ProjectManager",
        route_from_project_manager,
        {
            "Architect": "Architect",
            "Developer": "Developer",
            "Tester": "Tester",
            END: END
        }
    )
    print("Building Graph: Added conditional edges from ProjectManager.")

    # Add conditional edges for Summary check
    # These route from the worker agents (Arch, Dev, Test) to either Summary or PM
    workflow.add_conditional_edges(
        "Architect",
        route_to_summary_or_pm,
        {"Summary": "Summary", "ProjectManager": "ProjectManager"}
    )
    workflow.add_conditional_edges(
        "Developer",
        route_to_summary_or_pm,
        {"Summary": "Summary", "ProjectManager": "ProjectManager"}
    )
    workflow.add_conditional_edges(
        "Tester",
        route_to_summary_or_pm,
        {"Summary": "Summary", "ProjectManager": "ProjectManager"}
    )
    print("Building Graph: Added conditional edges for summary check.")

    # Compile the graph
    print("Building Graph: Compiling...")
    app = workflow.compile()
    print("Building Graph: Compilation complete.")
    return app

# Compile the app when this module is imported
compiled_app = build_graph() 