"""
Minimal ReAct-style agentic loop using a LOCAL model via Ollama.

Runs entirely on your machine, completely free, no API key.
Install Ollama (https://ollama.com) and pull a tool-capable model first:
    ollama pull llama3.1

PROBLEM
    Answer multi-step questions that need BOTH looking things up (web search)
    and doing calculations (calculator). A single LLM call can't do this
    reliably on its own — it either guesses numbers or can't fetch facts.

APPROACH (ReAct: Reason -> Act -> Observe, repeat)
    1. Give the model a system prompt describing its goal and available tools.
    2. Send the user's question.
    3. If the model's reply contains tool_calls, run those tools locally,
       append the results as "tool" messages, and let the model continue.
    4. Repeat until the model replies with no tool_calls (final answer).
"""

import ollama
from tools import TOOL_SCHEMAS, execute_tool

MODEL = "llama3.1"  # swap for "qwen2.5:7b" if your machine is memory-limited

SYSTEM_PROMPT = """You are a careful research assistant agent.
You have access to two tools: `web_search` and `calculator`.
Break multi-step questions into smaller steps. Use `web_search` to find facts
you don't know, and `calculator` for any arithmetic. Never guess a number you
could look up or compute instead. When you have the final answer, state it
clearly and briefly explain how you got it."""


def run_agent(question: str, max_steps: int = 8, verbose: bool = True) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    for step in range(max_steps):
        response = ollama.chat(model=MODEL, messages=messages, tools=TOOL_SCHEMAS)
        msg = response["message"]
        messages.append(msg)

        tool_calls = msg.get("tool_calls")
        if not tool_calls:
            return msg.get("content", "")

        for call in tool_calls:
            name = call["function"]["name"]
            args = call["function"]["arguments"]
            if verbose:
                print(f"[step {step}] calling {name}({args})")
            result = execute_tool(name, args)
            messages.append({"role": "tool", "content": str(result)})

    return "Stopped: reached max_steps without a final answer."


if __name__ == "__main__":
    q = input("Ask the agent something: ")
    print("\n" + run_agent(q))
