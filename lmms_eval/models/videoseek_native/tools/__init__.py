from .answer import answer_tool, execute_answer
from .focus import execute_focus, focus_tool
from .overview import execute_overview, overview_tool
from .registry import ToolRegistry
from .skim import execute_skim, skim_tool


TOOLS = {
    "overview": overview_tool,
    "skim": skim_tool,
    "focus": focus_tool,
    "answer": answer_tool,
}


TOOL_FUNCTIONS = {
    "overview": execute_overview,
    "skim": execute_skim,
    "focus": execute_focus,
    "answer": execute_answer,
}


DEFAULT_TOOL_REGISTRY = ToolRegistry(tools=TOOLS, tool_functions=TOOL_FUNCTIONS)

__all__ = [
    "DEFAULT_TOOL_REGISTRY",
    "TOOL_FUNCTIONS",
    "TOOLS",
    "ToolRegistry",
    "answer_tool",
    "execute_answer",
    "execute_focus",
    "execute_overview",
    "execute_skim",
    "focus_tool",
    "overview_tool",
    "skim_tool",
]
