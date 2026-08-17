# Research Agent — a minimal agentic AI example

A from-scratch example of the **ReAct** pattern (Reason → Act → Observe, loop)
using [Ollama](https://ollama.com) to run a model **locally, for free, with
no API key**. No frameworks (no LangChain/LangGraph) — just the raw loop, so
you can see exactly what "agentic" means under the hood.

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
# 1. Install Ollama: https://ollama.com/download (or `brew install ollama`)
# 2. Pull a tool-capable model
ollama pull llama3.1   # or qwen2.5:7b

# 3. Get a free Tavily API key (1,000 searches/month, no card): https://tavily.com
cp .env.example .env
# then paste your real key into .env, e.g. TAVILY_API_KEY=tvly-...

# 4. Install Python deps and run
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python agent.py
```

The LLM itself is free and local (Ollama, no API key, no billing). Only
`web_search` talks to an external service (Tavily's free tier) — if you skip
the `.env` step, `web_search` will return a clear error telling you the key
is missing, and `calculator` still works fine on its own.

## Files

| File | Purpose |
|---|---|
| `agent.py` | The ReAct loop itself |
| `tools.py` | Tool schemas + implementations (calculator, real Tavily web_search) |
| `requirements.txt` | Dependencies |
| `.env.example` | Template for your Tavily API key |

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
