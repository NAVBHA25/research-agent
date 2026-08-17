"""
Tool definitions for the agent.

Each tool needs two things:
  1. A JSON schema, so Claude knows the tool exists and how to call it
  2. A Python function that actually performs the action

Swap `web_search` for a real API (Tavily, SerpAPI, Brave Search, Anthropic's
built-in web_search tool, etc.) once you move past the toy stage.
"""

import ast
import operator as op

# Tool schemas in Ollama's format (same shape as OpenAI function calling —
# a "type": "function" wrapper around name/description/parameters).
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a basic arithmetic expression, e.g. '12.5 * 3 + 1'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "A math expression using + - * / ( ) and numbers.",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web and return a short summary of results for a query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query."}
                },
                "required": ["query"],
            },
        },
    },
]

_OPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.USub: op.neg,
}


def _safe_eval(node):
    """Evaluate a parsed math expression without ever calling eval()."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.BinOp):
        return _OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp):
        return _OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("Unsupported expression")


def calculator(expression: str) -> str:
    try:
        tree = ast.parse(expression, mode="eval").body
        return str(_safe_eval(tree))
    except Exception as e:
        return f"Error evaluating expression: {e}"


def web_search(query: str) -> str:
    # PLACEHOLDER — plug in a real search API here. Example with Tavily:
    #   from tavily import TavilyClient
    #   client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    #   return json.dumps(client.search(query)["results"][:3])
    return (
        f"[stub] No live search connected. Replace tools.web_search() with a "
        f"real API to actually search for: '{query}'."
    )


_DISPATCH = {"calculator": calculator, "web_search": web_search}


def execute_tool(name: str, tool_input: dict) -> str:
    fn = _DISPATCH.get(name)
    if fn is None:
        return f"Unknown tool: {name}"
    return fn(**tool_input)
