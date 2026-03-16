# Comprehensive Token Usage Tracking Guide

## Your System Architecture — Every LLM Call Point

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        YOUR APPLICATION                                 │
│                                                                         │
│  ┌─────────────────┐   ┌──────────────────────────────────────────┐    │
│  │ INGESTION       │   │ DEEP AGENTS ORCHESTRATOR                 │    │
│  │ SERVICE         │   │                                          │    │
│  │                 │   │  ┌─ Middleware Stack ──────────────────┐  │    │
│  │ LLM Call #1:    │   │  │                                    │  │    │
│  │ Chunk           │   │  │ LLM Call #2: SummarizationMW       │  │    │
│  │ Enhancement     │   │  │   summarization.py:602 (sync)      │  │    │
│  │                 │   │  │   summarization.py:628 (async)     │  │    │
│  │ (langchain-     │   │  │                                    │  │    │
│  │  openai direct) │   │  │ LLM Call #3: AssetRoutingMW        │  │    │
│  │                 │   │  │   (your custom middleware)          │  │    │
│  └─────────────────┘   │  │                                    │  │    │
│                         │  │ LLM Call #4: Main Agent Model      │  │    │
│  ┌─────────────────┐   │  │   factory.py:1233 (sync)           │  │    │
│  │ ARBITRARY       │   │  │   factory.py:1281 (async)          │  │    │
│  │ TASKS           │   │  │                                    │  │    │
│  │                 │   │  │ Dynamic Prompts via:                │  │    │
│  │ LLM Call #7:    │   │  │   wrap_model_call (5 middlewares)   │  │    │
│  │ Any standalone  │   │  └────────────────────────────────────┘  │    │
│  │ LLM usage       │   │                                          │    │
│  │                 │   │  ┌─ SubAgentMiddleware (task tool) ───┐  │    │
│  └─────────────────┘   │  │                                    │  │    │
│                         │  │  subagents.py:442 (sync)           │  │    │
│                         │  │  subagents.py:460 (async)          │  │    │
│                         │  │                                    │  │    │
│                         │  │  ┌─ Subagent A (LangGraph WF) ──┐ │  │    │
│                         │  │  │ LLM Call #5a: Agent model     │ │  │    │
│                         │  │  │ LLM Call #5b: SummarizationMW │ │  │    │
│                         │  │  └───────────────────────────────┘ │  │    │
│                         │  │                                    │  │    │
│                         │  │  ┌─ Subagent B (LangGraph WF) ──┐ │  │    │
│                         │  │  │ LLM Call #6a: Agent model     │ │  │    │
│                         │  │  │ LLM Call #6b: SummarizationMW │ │  │    │
│                         │  │  └───────────────────────────────┘ │  │    │
│                         │  └────────────────────────────────────┘  │    │
│                         └──────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
```

### Complete LLM Call Inventory

| # | Call Point | Location | Who Calls It | Config/Callbacks Passed? |
|---|-----------|----------|-------------|-------------------------|
| 1 | Ingestion chunk enhancement | Your service code | You directly | You control this |
| 2 | SummarizationMiddleware | `summarization.py:602/628` | `before_model` hook | `config={"metadata": {"lc_source": "summarization"}}` — NO callbacks |
| 3 | AssetRoutingMiddleware | Your middleware | `after_model` hook | You control this |
| 4 | Main agent model | `factory.py:1233/1281` | `model_node` in agent loop | `model_.invoke(messages)` — NO explicit config |
| 5a | Subagent agent model | `factory.py:1233/1281` (subagent's copy) | Subagent's `model_node` | Same pattern, NO explicit config |
| 5b | Subagent SummarizationMW | `summarization.py:602/628` (subagent's copy) | Subagent's `before_model` | NO callbacks |
| 6a-b | Additional subagents | Same as 5a-b | Per subagent | Same pattern |
| 7 | Arbitrary standalone tasks | Your service code | You directly | You control this |

### The Critical Gap

**`subagents.py:441-442`** — `subagent.invoke(subagent_state)` passes NO `config` argument. `runtime.config` IS available and contains parent callbacks, but it's discarded. This means **any callback-based tracking on the parent agent does NOT automatically propagate to subagents.**

---

## All Available Token Tracking Mechanisms

### Mechanism 1: `get_usage_metadata_callback()` Context Manager

**Source:** `langchain_core/callbacks/usage.py` (added in langchain-core 0.3.49)

```python
from langchain_core.callbacks import get_usage_metadata_callback

with get_usage_metadata_callback() as cb:
    result = graph.invoke(input_state)
    print(cb.usage_metadata)
    # {"gpt-4o-mini-2024-07-18": {"input_tokens": 850, "output_tokens": 120, "total_tokens": 970, ...}}
```

**How it works:**
- Uses `register_configure_hook(var, inheritable=True)` — auto-injected into ALL `CallbackManager` instances created during the `with` block
- `on_llm_end` extracts `AIMessage.usage_metadata` and accumulates by `model_name`
- Thread-safe via `threading.Lock()`

**What it captures:**
- Main agent model calls (Call #4) — YES (within the graph's Pregel runtime, callbacks propagate via `manager.get_child()`)
- SummarizationMiddleware calls (Call #2) — DEPENDS: The summarization model is called with `self.model.invoke(prompt, config={"metadata": ...})`. Since `get_usage_metadata_callback` uses a ContextVar, it IS picked up by `_configure()` when the model creates its callback manager. **YES, captured** as long as the `with` block wraps the entire graph invocation.
- Subagent calls (Calls #5, #6) — **NO** by default. `subagent.invoke(subagent_state)` starts a fresh Pregel run. However, since `get_usage_metadata_callback` uses a ContextVar (not config-based propagation), **it DOES propagate if the subagent runs in the same Python thread/async context**. The ContextVar is inherited by the subagent's `_configure()` call.
- Ingestion / arbitrary tasks (Calls #1, #7) — YES, if wrapped in the same `with` block

**Verdict:** This is the BEST single mechanism. It crosses most boundaries automatically via ContextVar inheritance. The key question is whether subagent invocations happen in the same async context.

**Subagent ContextVar behavior:**
- **Sync `subagent.invoke()`:** Runs in the SAME thread → ContextVar IS inherited → **captured**
- **Async `await subagent.ainvoke()`:** Runs in the SAME async task → ContextVar IS inherited → **captured**
- If subagent is invoked in a separate thread (e.g., `executor.submit()`) → ContextVar is NOT inherited → **NOT captured**

---

### Mechanism 2: `AIMessage.usage_metadata` — Post-Hoc State Inspection

**Source:** `langchain_core/messages/ai.py:176`

Every `AIMessage` returned by `langchain-openai` has:
```python
class UsageMetadata(TypedDict):
    input_tokens: int
    output_tokens: int
    total_tokens: int
    input_token_details: NotRequired[InputTokenDetails]   # audio, cache_creation, cache_read
    output_token_details: NotRequired[OutputTokenDetails]  # audio, reasoning
```

**How to use it:**
```python
# After agent run, sum all AIMessage usage in the message history
from langchain_core.messages import AIMessage

final_state = graph.invoke(input_state)
total_input = 0
total_output = 0
for msg in final_state["messages"]:
    if isinstance(msg, AIMessage) and msg.usage_metadata:
        total_input += msg.usage_metadata["input_tokens"]
        total_output += msg.usage_metadata["output_tokens"]
```

**What it captures:**
- Main agent model calls — YES (AIMessages stored in `state["messages"]`)
- SummarizationMiddleware — **NO** (the response's usage_metadata is discarded; only `response.text.strip()` is used)
- Subagent model calls — **NO** (subagent messages are in subagent state, not parent state; only a summary `ToolMessage` returns to parent)
- Ingestion / arbitrary — N/A (no shared state)

**Verdict:** Only useful for the main agent's own model calls. Misses middleware LLM calls and subagent calls entirely.

---

### Mechanism 3: `wrap_model_call` Middleware Hook

**Source:** `langchain/libs/langchain_v1/langchain/agents/middleware/types.py`

```python
class TokenTrackingMiddleware(AgentMiddleware):
    def wrap_model_call(self, request: ModelRequest, handler):
        response = handler(request)  # calls the actual LLM
        ai_msg = response.result[0]  # AIMessage
        if ai_msg.usage_metadata:
            # Track: ai_msg.usage_metadata["input_tokens"], etc.
            # Can write to state:
            return ExtendedModelResponse(
                model_response=response,
                command=Command(update={"token_usage": ai_msg.usage_metadata})
            )
        return response
```

**What it captures:**
- Main agent model calls (Call #4) — YES (wraps every LLM call in the agent loop)
- SummarizationMiddleware — **NO** (`before_model` fires BEFORE `wrap_model_call`; and summarization calls its own internal model, not through the wrap chain)
- Subagent model calls — **NO** (subagent has its own middleware stack)

**Verdict:** Only captures the main agent's model calls. Clean, explicit, and can write to state. Does NOT see middleware-internal or subagent LLM calls.

---

### Mechanism 4: `after_model` Middleware Hook

**Source:** `langchain/libs/langchain_v1/langchain/agents/middleware/types.py`

```python
class TokenTrackingMiddleware(AgentMiddleware):
    def after_model(self, state, runtime):
        last_ai = next(
            (m for m in reversed(state["messages"]) if isinstance(m, AIMessage)), None
        )
        if last_ai and last_ai.usage_metadata:
            return {"accumulated_tokens": add_usage(
                state.get("accumulated_tokens"), last_ai.usage_metadata
            )}
        return None
```

**What it captures:** Same scope as `wrap_model_call` — only the main agent model calls. Cannot see SummarizationMiddleware or subagent LLM calls.

**Verdict:** Alternative to `wrap_model_call`. Fires as a separate graph node (proper state commit). Good for accumulating in state.

---

### Mechanism 5: Custom `BaseCallbackHandler` on the Model Object

**Source:** `langchain_core/callbacks/base.py`

```python
from langchain_core.callbacks import BaseCallbackHandler

class TokenTracker(BaseCallbackHandler):
    def __init__(self):
        self.total_usage = {}  # {model_name: UsageMetadata}

    def on_llm_end(self, response, **kwargs):
        gen = response.generations[0][0]
        if hasattr(gen, 'message') and gen.message.usage_metadata:
            model = gen.message.response_metadata.get("model_name", "unknown")
            self.total_usage[model] = add_usage(
                self.total_usage.get(model), gen.message.usage_metadata
            )

# Attach to model BEFORE passing to create_deep_agent:
tracker = TokenTracker()
model = ChatOpenAI(model="gpt-4o-mini", callbacks=[tracker])
# or: model = model.with_config({"callbacks": [tracker]})
```

**What it captures:**
- Main agent model calls — YES (callbacks are part of the model object)
- SummarizationMiddleware — **YES if same model object** (Deep Agents passes the same model instance to `SummarizationMiddleware` by default, graph.py:158-164)
- AssetRoutingMiddleware — YES if you construct it with the same model
- Subagent model calls — DEPENDS on whether subagents use the same model object

**Verdict:** Good for capturing everything that goes through a specific model object. If you share one model instance across orchestrator + summarization, you get both. Subagents need the same model instance (or a model with the same callback).

---

### Mechanism 6: `astream_events` — Event-Based Token Tracking

**Source:** `langchain_core/tracers/event_stream.py`

```python
async for event in graph.astream_events(input_state, version="v2"):
    if event["event"] == "on_chat_model_end":
        ai_msg = event["data"]["output"]
        if ai_msg.usage_metadata:
            model = event["name"]
            usage = ai_msg.usage_metadata
            parent_ids = event["parent_ids"]  # trace hierarchy
            # Track by category using parent_ids or tags
```

**What it captures:**
- Main agent model calls — YES
- SummarizationMiddleware — YES (fires `on_chat_model_end` for the summarization model call)
- Subagent calls — YES if `subgraphs=True` is set (for LangGraph subgraphs), but since subagents are invoked via `.invoke()` without config, events may not propagate

**Verdict:** Powerful for real-time streaming. Event hierarchy (`parent_ids`) enables categorization. But subagent boundary is still the weak point.

---

### Mechanism 7: `stream_mode="messages"` in LangGraph

**Source:** `langgraph/libs/langgraph/langgraph/pregel/_messages.py`

```python
async for chunk, metadata in graph.astream(input_state, stream_mode="messages"):
    if chunk.usage_metadata:
        # Final message from each LLM call has usage
        print(metadata)  # includes langgraph_node, langgraph_step, etc.
```

**What it captures:** Only the current graph's model node calls. Does not cross subagent boundaries.

---

## Recommended Strategy: Categorized Token Tracking

Given your categories (`ingestion`, `agents`, `arbitrary-tasks`), here is the recommended approach:

### Architecture

```
┌──────────────────────────────────────────────────────┐
│  TokenUsageService (singleton)                        │
│                                                       │
│  ┌─────────────────────────────────────────────────┐  │
│  │ category_usage: dict[str, dict[str, UsageMetadata]]│
│  │                                                   │  │
│  │ {                                                 │  │
│  │   "ingestion": {"gpt-4o-mini": {in: 50K, ...}},  │  │
│  │   "agents":    {"gpt-4o": {in: 200K, ...}},      │  │
│  │   "arbitrary": {"gpt-4o-mini": {in: 10K, ...}},  │  │
│  │ }                                                 │  │
│  └─────────────────────────────────────────────────┘  │
│                                                       │
│  write_to_db(thread_id, category, usage)             │
└──────────────────────────────────────────────────────┘
```

### Strategy A: `get_usage_metadata_callback()` Per Category (Simplest)

This is the **simplest and most complete** approach. It leverages ContextVar inheritance to capture ALL LLM calls within a Python execution context.

```python
from langchain_core.callbacks import get_usage_metadata_callback

# --- INGESTION ---
async def run_ingestion(chunks: list[str]):
    with get_usage_metadata_callback() as cb:
        for chunk in chunks:
            enhanced = await llm.ainvoke(f"Enhance: {chunk}")

        # cb.usage_metadata has ALL token usage for ingestion
        save_to_db(category="ingestion", usage=cb.usage_metadata)

# --- AGENTS (orchestrator + all subagents) ---
async def run_agent(user_query: str, thread_id: str):
    with get_usage_metadata_callback() as cb:
        result = await deep_agent.ainvoke(
            {"messages": [HumanMessage(content=user_query)]},
            config={"configurable": {"thread_id": thread_id}}
        )

        # cb.usage_metadata captures:
        #   - Main agent model calls (Call #4)
        #   - SummarizationMiddleware calls (Call #2) — via ContextVar
        #   - AssetRoutingMiddleware calls (Call #3) — via ContextVar
        #   - Subagent model calls (Calls #5, #6) — via ContextVar
        #     (ONLY if subagent.invoke() runs in same async context)
        save_to_db(category="agents", thread_id=thread_id, usage=cb.usage_metadata)

# --- ARBITRARY TASKS ---
async def run_arbitrary_task(task_description: str):
    with get_usage_metadata_callback() as cb:
        result = await llm.ainvoke(task_description)
        save_to_db(category="arbitrary-tasks", usage=cb.usage_metadata)
```

**Why this works for subagents:** `get_usage_metadata_callback()` uses `register_configure_hook(var, inheritable=True)`, which registers a ContextVar. When `subagent.invoke(subagent_state)` is called (even without passing `config`), the subagent's internal `Pregel.stream()` calls `get_callback_manager_for_config(config)` which calls `_configure()` which scans `_configure_hooks` and finds the ContextVar still set. The handler is auto-injected.

**This works because:**
1. Sync path: `subagent.invoke()` runs in the same thread → same ContextVar
2. Async path: `await subagent.ainvoke()` runs in the same async task → same ContextVar (Python asyncio ContextVars are inherited by child tasks)

**When this BREAKS:**
- If subagent invocation is wrapped in `loop.run_in_executor()` (runs in a thread pool) → ContextVar not inherited
- If subagent is invoked via HTTP/RPC to a separate service → completely separate process

### Strategy B: Model-Level Callbacks + Per-Category Tracker (Most Explicit)

For maximum control and guaranteed coverage regardless of ContextVar behavior:

```python
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage
from langchain_core.messages.ai import add_usage, UsageMetadata
import threading

class CategoryTokenTracker(BaseCallbackHandler):
    """Callback handler that tracks tokens for a specific category."""

    def __init__(self, category: str, db_writer):
        self.category = category
        self.db_writer = db_writer
        self.usage: dict[str, UsageMetadata] = {}
        self._lock = threading.Lock()

    def on_llm_end(self, response, **kwargs):
        gen = response.generations[0][0]
        if hasattr(gen, 'message') and isinstance(gen.message, AIMessage):
            usage = gen.message.usage_metadata
            model = gen.message.response_metadata.get("model_name", "unknown")
            if usage and model:
                with self._lock:
                    self.usage[model] = add_usage(self.usage.get(model), usage)

    def flush(self, thread_id: str = None):
        with self._lock:
            self.db_writer.write(
                category=self.category,
                thread_id=thread_id,
                usage=dict(self.usage)
            )
            self.usage.clear()

# --- Setup: Attach trackers to model objects ---

# For ingestion
ingestion_tracker = CategoryTokenTracker("ingestion", db_writer)
ingestion_llm = ChatOpenAI(model="gpt-4o-mini", callbacks=[ingestion_tracker])

# For agents — create model with tracker baked in
agent_tracker = CategoryTokenTracker("agents", db_writer)
agent_model = ChatOpenAI(model="gpt-4o", callbacks=[agent_tracker])

# Pass this model to create_deep_agent — it will be shared with SummarizationMiddleware
deep_agent = create_deep_agent(
    model=agent_model,          # main agent model — tracker fires on every call
    # SummarizationMiddleware gets the same model object by default (graph.py:158-164)
    middleware=[
        AssetRoutingMiddleware(
            model=ChatOpenAI(model="gpt-4o-mini", callbacks=[agent_tracker]),
            # ^^^ Same tracker on asset routing model
        ),
    ],
    subagents={
        "pdf-subagent": create_agent(
            model=ChatOpenAI(model="gpt-4o", callbacks=[agent_tracker]),
            # ^^^ Same tracker on subagent model
        ),
        "csv-subagent": create_agent(
            model=ChatOpenAI(model="gpt-4o", callbacks=[agent_tracker]),
        ),
        "postgres-subagent": create_agent(
            model=ChatOpenAI(model="gpt-4o", callbacks=[agent_tracker]),
        ),
    }
)

# For arbitrary tasks
arbitrary_tracker = CategoryTokenTracker("arbitrary-tasks", db_writer)
arbitrary_llm = ChatOpenAI(model="gpt-4o-mini", callbacks=[arbitrary_tracker])
```

**Pros:**
- Guaranteed coverage — every model object has its tracker baked in
- Works regardless of ContextVar propagation, threading, or config gaps
- No framework modifications needed
- Categorization is inherent (each model has its category tracker)

**Cons:**
- Must attach tracker to every model object explicitly
- If someone creates a model without the tracker, those calls are missed
- Slightly more boilerplate

### Strategy C: Hybrid — ContextVar + `wrap_model_call` State Accumulation

Combine `get_usage_metadata_callback()` for total tracking with `wrap_model_call` for per-step state accumulation:

```python
# Custom middleware for state-level tracking
class TokenAccumulatorMiddleware(AgentMiddleware):
    """Accumulates token usage in graph state per agent loop iteration."""

    def wrap_model_call(self, request, handler):
        response = handler(request)
        ai_msg = response.result[0]
        if ai_msg.usage_metadata:
            current = request.state.get("token_usage_total")
            new_total = add_usage(current, ai_msg.usage_metadata)
            return ExtendedModelResponse(
                model_response=response,
                command=Command(update={"token_usage_total": new_total})
            )
        return response

    async def awrap_model_call(self, request, handler):
        response = await handler(request)
        ai_msg = response.result[0]
        if ai_msg.usage_metadata:
            current = request.state.get("token_usage_total")
            new_total = add_usage(current, ai_msg.usage_metadata)
            return ExtendedModelResponse(
                model_response=response,
                command=Command(update={"token_usage_total": new_total})
            )
        return response

# Usage: wrap the entire run
async def run_agent(user_query, thread_id):
    with get_usage_metadata_callback() as cb:
        result = await deep_agent.ainvoke(...)

    # cb.usage_metadata — total across everything (orchestrator + subagents + middleware)
    # result["token_usage_total"] — only main agent model calls (from wrap_model_call)
    # Difference = middleware + subagent tokens
```

---

## Strategy Comparison Matrix

| Strategy | Ingestion | Main Agent Model | SummarizationMW | AssetRoutingMW | Subagents | Arbitrary Tasks | Framework Changes | Complexity |
|----------|:---------:|:----------------:|:----------------:|:--------------:|:---------:|:---------------:|:-----------------:|:----------:|
| **A: `get_usage_metadata_callback()`** | YES | YES | YES | YES | YES* | YES | None | Low |
| **B: Model-level callbacks** | YES | YES | YES** | YES | YES | YES | None | Medium |
| **C: Hybrid (A + wrap_model_call)** | YES | YES | YES | YES | YES* | YES | None | Medium |
| **D: `astream_events`** | N/A | YES | YES | YES | Partial | N/A | None | High |
| **E: Post-hoc state inspection** | N/A | YES | NO | NO | NO | N/A | None | Low |

\* Requires same-thread/async-context execution (true for default Deep Agents sync/async paths)
\** Requires same model object shared with SummarizationMiddleware (true by default in `create_deep_agent`)

---

## Recommendation

### For your use case, use **Strategy A** (`get_usage_metadata_callback()`) as the primary approach:

1. **It's the simplest** — one context manager wrapping each category's execution
2. **It crosses the subagent boundary** via ContextVar inheritance (confirmed: `_configure()` in `manager.py:2451-2471` scans `_configure_hooks` which picks up the ContextVar)
3. **It captures SummarizationMiddleware** — the ContextVar is present when `self.model.invoke()` creates its callback manager
4. **It's the officially recommended mechanism** (added in langchain-core 0.3.49)
5. **Thread-safe** with `threading.Lock()`
6. **Works with streaming** (fires on `on_llm_end` which always has final usage)

### Add **Strategy B** (model-level callbacks) as a safety net:

- Attach a shared `CategoryTokenTracker` to model objects used by subagents
- This guarantees coverage even if ContextVar propagation breaks (e.g., thread pool executor)

### For the DB schema:

```sql
CREATE TABLE token_usage (
    id SERIAL PRIMARY KEY,
    thread_id TEXT,
    category TEXT NOT NULL,          -- 'ingestion', 'agents', 'arbitrary-tasks'
    model_name TEXT NOT NULL,        -- 'gpt-4o-mini-2024-07-18', etc.
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

## Appendix: Fixing the Subagent Config Gap (Optional)

If you want callbacks to propagate through config (not just ContextVar), patch `subagents.py:441-442`:

```python
# BEFORE (current — config lost):
result = subagent.invoke(subagent_state)

# AFTER (config propagated):
result = subagent.invoke(subagent_state, runtime.config)
```

This makes `runtime.config` (which carries the parent's callback manager hierarchy) flow into the subagent's Pregel runtime. Combined with `get_usage_metadata_callback()`, this provides double-coverage.

**Note:** This requires modifying Deep Agents source code. Strategy A works without this change as long as ContextVar propagation holds.

---

## Source File Reference
(Use the framework reference skill)
| Concept | File | Lines |
|---------|------|-------|
| `UsageMetadata` TypedDict | `langchain_core/messages/ai.py` | 104-157 |
| `AIMessage.usage_metadata` | `langchain_core/messages/ai.py` | 176 |
| `add_usage()` | `langchain_core/messages/ai.py` | 721 |
| `UsageMetadataCallbackHandler` | `langchain_core/callbacks/usage.py` | 18-89 |
| `get_usage_metadata_callback()` | `langchain_core/callbacks/usage.py` | 92-149 |
| `register_configure_hook()` | `langchain_core/tracers/context.py` | 171-202 |
| `_configure()` hook scan | `langchain_core/callbacks/manager.py` | 2451-2471 |
| `BaseCallbackHandler.on_llm_end` | `langchain_core/callbacks/base.py` | 89 |
| `BaseChatModel.generate()` callback wiring | `langchain_core/language_models/chat_models.py` | 842-963 |
| `_create_usage_metadata()` (OpenAI) | `langchain_openai/chat_models/base.py` | 3623 |
| Streaming usage extraction | `langchain_openai/chat_models/base.py` | 1067-1127 |
| Main agent `model_.invoke()` | `langchain/agents/factory.py` | 1233 (sync), 1281 (async) |
| SummarizationMW LLM call | `langchain/agents/middleware/summarization.py` | 602 (sync), 628 (async) |
| `wrap_model_call` chain | `langchain/agents/middleware/types.py` | 580-798 |
| Subagent invocation (GAP) | `deepagents/middleware/subagents.py` | 441-442 (sync), 459-460 (async) |
| `runtime.config` available | `langgraph/prebuilt/tool_node.py` | 1591 |
| `add_messages` reducer (preserves usage) | `langgraph/graph/message.py` | 61-244 |
| Pregel callback propagation | `langgraph/pregel/_algo.py` | 681-690 |
| `StreamMessagesHandler` | `langgraph/pregel/_messages.py` | 42-250 |
