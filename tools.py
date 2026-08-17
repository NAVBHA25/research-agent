"""
Tool definitions for the agent.

Each tool needs two things:
  1. A JSON schema, so the model knows the tool exists and how to call it
  2. A Python function that actually performs the action

web_search is wired up to Tavily (https://tavily.com), which has a free tier
of 1,000 search credits/month, no card required. Set TAVILY_API_KEY as an
environment variable (or put it in a .env file in the project root) before
running the agent.
"""

import ast
import operator as op
import os

from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()  # reads a .env file in the project root, if present

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


_tavily_api_key = os.environ.get("TAVILY_API_KEY")
_tavily_client = TavilyClient(api_key=_tavily_api_key) if _tavily_api_key else None


def web_search(query: str) -> str:
    """Real web search via Tavily's free tier (1,000 searches/month, no card)."""
    if _tavily_client is None:
        return (
            "Error: TAVILY_API_KEY is not set. Get a free key at "
            "https://tavily.com, then export it as an environment variable "
            "(export TAVILY_API_KEY=tvly-...) or add it to a .env file."
        )

    try:
        response = _tavily_client.search(query=query, max_results=3)
    except Exception as e:
        return f"Error calling Tavily: {e}"

    results = response.get("results", [])
    if not results:
        return f"No results found for '{query}'."

    # Keep it short and model-friendly: title + a trimmed snippet per result.
    lines = []
    for r in results:
        title = r.get("title", "Untitled")
        content = (r.get("content") or "")[:300]
        lines.append(f"- {title}: {content}")
    return "\n".join(lines)


_DISPATCH = {"calculator": calculator, "web_search": web_search}


def execute_tool(name: str, tool_input: dict) -> str:
    fn = _DISPATCH.get(name)
    if fn is None:
        return f"Unknown tool: {name}"
    return fn(**tool_input)
