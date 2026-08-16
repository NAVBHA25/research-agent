# Research Agent — a minimal agentic AI example

A from-scratch example of the **ReAct** pattern (Reason → Act → Observe, loop)
using the Anthropic API. No frameworks (no LangChain/LangGraph) — just the
raw loop, so you can see exactly what "agentic" means under the hood.

## The problem

A single LLM call is unreliable at questions that need **both** external
facts and **precise math**, e.g.:

> "What's the population of France divided by Germany's, times 100?"

Ask a plain chatbot this and it may hallucinate the populations or make an
arithmetic slip. An agent instead: looks the numbers up with a tool, hands
the math to a calculator tool, and only then answers.

## The approach

```
 ┌────────────┐   tool_use   ┌──────────────┐
 │   Claude    │ ───────────▶│  Run locally  │
 │ (reasoning) │              │ (calculator,  │
 │             │◀─────────── │  web_search)  │
 └────────────┘  tool_result  └──────────────┘
        │
        ▼ (no more tools needed)
   Final answer
```

1. `agent.py` sends the question + tool schemas to Claude.
2. If Claude wants a tool, we execute it in Python (`tools.py`) and send
   the result back as a `tool_result` block.
3. Loop until Claude stops asking for tools and gives a final answer.
4. `max_steps` caps runaway loops — always set one.

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then add your ANTHROPIC_API_KEY
export $(cat .env | xargs)   # or use python-dotenv / direnv
python agent.py
```

## Files

| File | Purpose |
|---|---|
| `agent.py` | The ReAct loop itself |
| `tools.py` | Tool schemas + implementations (calculator, web_search stub) |
| `requirements.txt` | Dependencies |
| `.env.example` | Env var template |

## Extending this

- **Real search**: swap the `web_search` stub in `tools.py` for Tavily,
  SerpAPI, Brave Search, or Anthropic's built-in `web_search` tool type.
- **Memory**: persist `messages` to disk/DB between runs for multi-turn agents.
- **More tools**: add a schema + function pair — code execution, a database
  query, an API call, file read/write. The loop doesn't change.
- **Guardrails**: validate tool inputs, add a human-approval step before any
  tool with real-world side effects (sending email, making a purchase, etc.)
- **Multi-agent**: once one loop works, try a "planner" agent that delegates
  sub-tasks to specialized "worker" agents — same loop, nested.

## Why build it this way first

Frameworks like LangGraph or CrewAI are useful once you have >1 agent or
complex state machines, but they hide the loop that makes agents work. Build
this raw version first so debugging a framework later isn't debugging magic.
