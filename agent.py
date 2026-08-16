"""
Minimal ReAct-style agentic loop using the Anthropic API.

PROBLEM
    Answer multi-step questions that need BOTH looking things up (web search)
    and doing calculations (calculator). A single LLM call can't do this
    reliably on its own — it either guesses numbers or can't fetch facts.

APPROACH (ReAct: Reason -> Act -> Observe, repeat)
    1. Give Claude a system prompt describing its goal and available tools.
    2. Send the user's question.
    3. If Claude's reply contains a tool_use block, run that tool locally,
       send the result back as a tool_result, and let Claude continue.
    4. Repeat until Claude gives a final answer (stop_reason != "tool_use").
"""

import os
from anthropic import Anthropic
from tools import TOOL_SCHEMAS, execute_tool

client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

MODEL = "claude-sonnet-4-6"

SYSTEM_PROMPT = """You are a careful research assistant agent.
You have access to two tools: `web_search` and `calculator`.
Break multi-step questions into smaller steps. Use `web_search` to find facts
you don't know, and `calculator` for any arithmetic. Never guess a number you
could look up or compute instead. When you have the final answer, state it
clearly and briefly explain how you got it."""


def run_agent(question: str, max_steps: int = 8, verbose: bool = True) -> str:
    messages = [{"role": "user", "content": question}]

    for step in range(max_steps):
        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            tools=TOOL_SCHEMAS,
            messages=messages,
        )

        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            return "".join(
                block.text for block in response.content if block.type == "text"
            )

        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                if verbose:
                    print(f"[step {step}] calling {block.name}({block.input})")
                result = execute_tool(block.name, block.input)
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    }
                )

        messages.append({"role": "user", "content": tool_results})

    return "Stopped: reached max_steps without a final answer."


if __name__ == "__main__":
    q = input("Ask the agent something: ")
    print("\n" + run_agent(q))
