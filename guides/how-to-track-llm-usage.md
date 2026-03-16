# How to Track LLM Token Usage in Deep Agents

## Architecture Overview

When a Deep Agents orchestrator runs with compiled LangGraph subagents, LLM calls happen at multiple layers:

```
┌─────────────────────────────────────────────────────────────────┐
│  with get_usage_metadata_callback() as cb:                      │
│                                                                  │
│  ┌─ Deep Agents Orchestrator ─────────────────────────────────┐ │
│  │                                                             │ │
│  │  Middleware Stack:                                          │ │
│  │    ├─ SummarizationMW ──── LLM call (summarize context)   │ │
│  │    ├─ Custom MW ────────── LLM call (if applicable)       │ │
│  │    └─ Main Agent Model ─── LLM call (reasoning loop)      │ │
│  │                                                             │ │
│  │  SubAgentMiddleware → task() tool:                         │ │
│  │    ├─ Compiled Subagent A (LangGraph workflow)             │ │
│  │    │    ├─ node_1 ──── LLM call                           │ │
│  │    │    └─ node_2 ──── LLM call                           │ │
│  │    ├─ Compiled Subagent B (LangGraph workflow)             │ │
│  │    │    ├─ node_1 ──── LLM call                           │ │
│  │    │    └─ node_2 ──── LLM call                           │ │
│  │    └─ Compiled Subagent C (LangGraph workflow)             │ │
│  │         ├─ node_1 ──── LLM call                           │ │
│  │         └─ node_2 ──── LLM call                           │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  cb.usage_metadata ← captures ALL of the above                  │
└──────────────────────────────────────────────────────────────────┘
```

**Every LLM call in the diagram above is captured by a single `get_usage_metadata_callback()` context manager.**

---

## Recommended Approach: `get_usage_metadata_callback()`

This is the simplest and most complete mechanism. One context manager wraps the entire agent invocation and captures token usage across every LLM call — including those inside compiled LangGraph subagents.

### Basic Usage

```python
from langchain_core.callbacks import get_usage_metadata_callback
from langchain_core.messages import HumanMessage

async def run_agent(user_query: str, thread_id: str):
    with get_usage_metadata_callback() as cb:
        result = await deep_agent.ainvoke(
            {"messages": [HumanMessage(content=user_query)]},
            config={"configurable": {"thread_id": thread_id}},
        )

    # Aggregated usage across orchestrator + all subagents + all middleware
    print(cb.usage_metadata)
    # {
    #   "gpt-4o-2024-08-06": {
    #     "input_tokens": 250000,
    #     "output_tokens": 45000,
    #     "total_tokens": 295000,
    #     "input_token_details": {"cache_read": 180000, "cache_creation": 20000},
    #     "output_token_details": {"reasoning": 5000},
    #   },
    #   "gpt-4o-mini-2024-07-18": {
    #     "input_tokens": 80000,
    #     "output_tokens": 12000,
    #     "total_tokens": 92000,
    #   },
    # }
```

### Full Example with `create_deep_agent` and Compiled Subagents

```python
from langchain_core.callbacks import get_usage_metadata_callback
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState
from langgraph.store.memory import InMemoryStore
from langgraph.checkpoint.memory import InMemorySaver

from deepagents import create_deep_agent
from deepagents.middleware.subagents import CompiledSubAgent


# --- Build compiled LangGraph subagent workflows ---

def build_research_subagent() -> CompiledSubAgent:
    """A custom LangGraph workflow passed as a compiled subagent."""
    llm = ChatOpenAI(model="gpt-4o-mini")

    def research_node(state: MessagesState):
        response = llm.invoke(state["messages"])
        return {"messages": [response]}

    graph = StateGraph(MessagesState)
    graph.add_node("research", research_node)
    graph.set_entry_point("research")
    graph.set_finish_point("research")

    return {
        "name": "researcher",
        "description": "Researches a topic and returns findings",
        "runnable": graph.compile(),
    }


def build_writer_subagent() -> CompiledSubAgent:
    llm = ChatOpenAI(model="gpt-4o")

    def write_node(state: MessagesState):
        response = llm.invoke(state["messages"])
        return {"messages": [response]}

    graph = StateGraph(MessagesState)
    graph.add_node("write", write_node)
    graph.set_entry_point("write")
    graph.set_finish_point("write")

    return {
        "name": "writer",
        "description": "Writes content based on research findings",
        "runnable": graph.compile(),
    }


# --- Create the orchestrator ---

store = InMemoryStore()
checkpointer = InMemorySaver()

agent = create_deep_agent(
    model="openai:gpt-4o",
    subagents=[build_research_subagent(), build_writer_subagent()],
    store=store,
    checkpointer=checkpointer,
)


# --- Run with token tracking ---

async def run_with_tracking(query: str, thread_id: str):
    with get_usage_metadata_callback() as cb:
        result = await agent.ainvoke(
            {"messages": [HumanMessage(content=query)]},
            config={"configurable": {"thread_id": thread_id}},
        )

    # cb.usage_metadata contains ALL token usage:
    #   - Orchestrator main model calls
    #   - SummarizationMiddleware calls (if triggered)
    #   - researcher subagent LLM calls
    #   - writer subagent LLM calls
    return result, cb.usage_metadata
```

### Persisting Usage to a Database

```python
async def run_and_persist(query: str, thread_id: str, db):
    with get_usage_metadata_callback() as cb:
        result = await agent.ainvoke(
            {"messages": [HumanMessage(content=query)]},
            config={"configurable": {"thread_id": thread_id}},
        )

    for model_name, usage in cb.usage_metadata.items():
        await db.execute(
            """
            INSERT INTO token_usage (thread_id, model_name, input_tokens, output_tokens, total_tokens)
            VALUES ($1, $2, $3, $4, $5)
            """,
            thread_id,
            model_name,
            usage["input_tokens"],
            usage["output_tokens"],
            usage["total_tokens"],
        )

    return result
```

---

## Why This Works for Compiled LangGraph Subagents

The usage callback propagates into compiled subagents through **two independent mechanisms**. Both are active simultaneously — belt and suspenders.

### Mechanism 1: ContextVar Inheritance (Implicit)

`get_usage_metadata_callback()` registers itself via `register_configure_hook(var, inheritable=True)` using a Python `ContextVar`. When any LangGraph compiled graph calls `Pregel.stream()`, it calls `get_callback_manager_for_config()`, which runs `_configure()`, which scans `_configure_hooks` and finds the ContextVar still set.

This works because:

- **Sync path**: The `task()` tool function calls `subagent.invoke()` on the same thread. ContextVars are inherited.
- **Async path**: The `atask()` tool function `await`s `subagent.ainvoke()` in the same async task. ContextVars are inherited.
- **Thread pool crossings**: LangGraph's `BackgroundExecutor` uses `copy_context()` before dispatching to thread pool workers (`langgraph/pregel/_executor.py:64`), so ContextVars survive even concurrent node execution.

### Mechanism 2: Explicit `callbacks` Propagation (Config-Based)

`_build_subagent_config()` in `subagents.py:430-473` copies `runtime.config["callbacks"]` into the subagent's config:

```python
if "callbacks" in runtime.config:
    subagent_config["callbacks"] = runtime.config["callbacks"]
```

The subagent is then invoked with this config at `subagents.py:491` (sync) / `subagents.py:512` (async):

```python
result = subagent.invoke(subagent_state, subagent_config)
```

This means callback handlers from the parent — including the usage tracker — are explicitly passed through the config, independent of ContextVar propagation.

### What Gets Captured

| LLM Call Point | Captured? | Via |
|---|---|---|
| Main orchestrator model | Yes | Both mechanisms |
| SummarizationMiddleware (orchestrator) | Yes | ContextVar |
| Custom middleware LLM calls | Yes | ContextVar |
| Compiled subagent — all nodes | Yes | Both mechanisms |
| SummarizationMiddleware (inside subagent, if present) | Yes | ContextVar |
| Standalone LLM calls outside the agent | Yes | If inside the `with` block |

---

## When It Does NOT Work

The ContextVar mechanism breaks if the subagent invocation crosses a context boundary:

| Scenario | Works? | Why |
|---|---|---|
| `subagent.invoke()` — same thread | Yes | Same ContextVar scope |
| `await subagent.ainvoke()` — same async task | Yes | Same ContextVar scope |
| `executor.submit(subagent.invoke, ...)` — thread pool | No | New thread, no `copy_context()` |
| HTTP/RPC to a separate service | No | Separate process |

The default Deep Agents `task()` / `atask()` invocation path does **not** use `executor.submit()` or any thread pool hop. Both sync and async paths invoke the subagent directly. So **for standard Deep Agents usage, tracking always works**.

---

## Per-Subagent Token Breakdown

`get_usage_metadata_callback()` gives one aggregated total. If you need per-subagent breakdowns, use one of these approaches:

### Option A: Nested Callback Per Subagent (Requires Framework Modification)

Wrap each subagent invocation in its own `get_usage_metadata_callback()` inside the `task()` closure. This requires modifying `subagents.py`.

### Option B: Model-Level Callbacks (No Framework Changes)

Attach a dedicated callback handler to each subagent's model object:

```python
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage
from langchain_core.messages.ai import add_usage
import threading


class SubagentTokenTracker(BaseCallbackHandler):
    def __init__(self, name: str):
        self.name = name
        self.usage = {}
        self._lock = threading.Lock()

    def on_llm_end(self, response, **kwargs):
        gen = response.generations[0][0]
        if hasattr(gen, "message") and isinstance(gen.message, AIMessage):
            usage = gen.message.usage_metadata
            model = gen.message.response_metadata.get("model_name", "unknown")
            if usage and model:
                with self._lock:
                    self.usage[model] = add_usage(self.usage.get(model), usage)


# Create trackers for each subagent
research_tracker = SubagentTokenTracker("researcher")
writer_tracker = SubagentTokenTracker("writer")

# Attach to model objects used inside each subagent's nodes
research_llm = ChatOpenAI(model="gpt-4o-mini", callbacks=[research_tracker])
writer_llm = ChatOpenAI(model="gpt-4o", callbacks=[writer_tracker])

# Use these models inside the subagent node functions
# After the run:
#   research_tracker.usage → {"gpt-4o-mini-...": {"input_tokens": ..., ...}}
#   writer_tracker.usage   → {"gpt-4o-...": {"input_tokens": ..., ...}}
```

### Option C: Hybrid — Total + Per-Subagent

Use `get_usage_metadata_callback()` for the total and model-level callbacks for the breakdown:

```python
async def run_with_detailed_tracking(query: str, thread_id: str):
    with get_usage_metadata_callback() as cb:
        result = await agent.ainvoke(
            {"messages": [HumanMessage(content=query)]},
            config={"configurable": {"thread_id": thread_id}},
        )

    total = cb.usage_metadata  # everything
    research = research_tracker.usage  # just the researcher subagent
    writer = writer_tracker.usage  # just the writer subagent
    orchestrator = subtract_usage(total, research, writer)  # the remainder

    return {
        "total": total,
        "orchestrator": orchestrator,
        "researcher": research,
        "writer": writer,
    }
```

---

## Alternative Mechanisms (Comparison)

| Mechanism | Orchestrator | SummarizationMW | Subagents | Complexity | Notes |
|---|---|---|---|---|---|
| **`get_usage_metadata_callback()`** | Yes | Yes | Yes | Low | Recommended. Single context manager. |
| **Model-level callbacks** | Yes | Yes* | Yes | Medium | Requires attaching tracker to every model object. |
| **`wrap_model_call` middleware** | Yes | No | No | Low | Only sees main agent model calls. |
| **`after_model` middleware** | Yes | No | No | Low | Same scope as `wrap_model_call`. |
| **`astream_events`** | Yes | Yes | Partial | High | Subagent events need `subgraphs=True`. |
| **Post-hoc `AIMessage.usage_metadata`** | Yes | No | No | Low | Only messages in final state. |

\* Only if the same model object is shared (true by default in `create_deep_agent` for SummarizationMiddleware).

---

## Caveats

### 1. `_configure_hooks` List Growth

Every call to `get_usage_metadata_callback()` appends a new entry to a module-level `_configure_hooks` list in `langchain_core.tracers.context`. This list is never cleaned up. In a long-running server process, this causes increasing iteration cost on every `_configure()` call (which runs on every LLM invocation).

**Mitigation**: This is a langchain-core issue, not a Deep Agents issue. For long-running servers, consider restarting workers periodically or using model-level callbacks (which don't have this problem).

### 2. Aggregated Output Only

`get_usage_metadata_callback()` returns usage aggregated by model name across the entire `with` block. There is no built-in way to attribute tokens to specific subagents or middleware calls. See the "Per-Subagent Token Breakdown" section above for workarounds.

### 3. Nested Context Managers

Do not nest `get_usage_metadata_callback()` calls. The implementation does not save/restore the ContextVar token properly — the outer handler is overwritten with `None` on inner exit rather than being restored:

```python
# DO NOT DO THIS:
with get_usage_metadata_callback() as outer:
    with get_usage_metadata_callback() as inner:
        result = await agent.ainvoke(...)
    # outer is now broken — its ContextVar was set to None
```

---

## Database Schema (Optional)

```sql
CREATE TABLE token_usage (
    id SERIAL PRIMARY KEY,
    thread_id TEXT,
    category TEXT NOT NULL,
    model_name TEXT NOT NULL,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    total_tokens INTEGER NOT NULL,
    input_cache_read INTEGER DEFAULT 0,
    input_cache_creation INTEGER DEFAULT 0,
    output_reasoning INTEGER DEFAULT 0,
    recorded_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_token_usage_category ON token_usage(category);
CREATE INDEX idx_token_usage_thread ON token_usage(thread_id);
```

---

## Source File Reference

| Concept | File | Key Lines |
|---|---|---|
| `get_usage_metadata_callback()` | `langchain_core/callbacks/usage.py` | 92-149 |
| `register_configure_hook()` | `langchain_core/tracers/context.py` | 171-202 |
| `_configure()` hook scan | `langchain_core/callbacks/manager.py` | 2461-2481 |
| `BackgroundExecutor` with `copy_context()` | `langgraph/pregel/_executor.py` | 48-75 |
| `Pregel.stream()` → callback setup | `langgraph/pregel/main.py` | 2523 |
| `_build_subagent_config()` | `deepagents/middleware/subagents.py` | 430-473 |
| Sync subagent invocation | `deepagents/middleware/subagents.py` | 491 |
| Async subagent invocation | `deepagents/middleware/subagents.py` | 512 |
| `callbacks` propagation | `deepagents/middleware/subagents.py` | 470-471 |
| `create_deep_agent()` store parameter | `deepagents/graph.py` | 62, 279 |
| CompiledSubAgent pass-through | `deepagents/graph.py` | 183-186 |
