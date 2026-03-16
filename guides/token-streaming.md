# Custom Streaming from Subagent Nodes — All Possible Approaches

## The Problem

You have a Deep Agents orchestrator with multiple compiled LangGraph workflow subagents invoked via the `task` tool. Each subagent has nodes that produce custom status dictionaries (progress updates, intermediate results, etc.). You need these custom dicts to stream back to your frontend in real-time while the subagent is running, to keep the UI active. The system must also support a non-streaming (default) mode where `.invoke()` works as before.

## System Architecture

```
Frontend (WebSocket / SSE)
    │
    ▼
Your Service Layer
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  Deep Agents Orchestrator (CompiledStateGraph)          │
│  create_deep_agent() → graph.stream() or graph.invoke() │
│                                                         │
│  ┌─ Middleware Stack ─────────────────────────────────┐  │
│  │  wrap_model_call, before_model, after_model        │  │
│  │  runtime.stream_writer → available in all hooks    │  │
│  └────────────────────────────────────────────────────┘  │
│                                                         │
│  ┌─ ToolNode ──────────────────────────────────────────┐ │
│  │  task tool (SubAgentMiddleware)                     │ │
│  │  subagents.py:442 → subagent.invoke(state)  ← GAP  │ │
│  │  runtime.config ← has CONFIG_KEY_STREAM             │ │
│  │  runtime.stream_writer ← has live writer            │ │
│  │                                                     │ │
│  │  ┌─ Subagent A (Compiled LangGraph Workflow) ─────┐ │ │
│  │  │  Node 1: writer({"step": "parsing", ...})      │ │ │
│  │  │  Node 2: writer({"step": "processing", ...})   │ │ │
│  │  │  Node 3: writer({"step": "done", ...})         │ │ │
│  │  └────────────────────────────────────────────────┘ │ │
│  └─────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
    │
    ▼
Frontend receives: ("custom", {"step": "parsing", "subagent": "pdf-subagent"})
```

## The Core Blocker

**File:** `deepagents/libs/deepagents/deepagents/middleware/subagents.py`, lines 442 / 460

```python
# Current code — streaming breaks here:
result = subagent.invoke(subagent_state)          # sync, line 442
result = await subagent.ainvoke(subagent_state)   # async, line 460
```

**Why streaming breaks:** No `config` is passed. Without config:
- `CONFIG_KEY_STREAM` is absent → subagent's `stream_writer` becomes a **no-op**
- Callbacks are not explicitly propagated → `dispatch_custom_event` relies on ContextVar (works on Python >= 3.11 only)
- The subagent starts a completely fresh Pregel loop with no connection to the parent's stream

**What IS available inside the task tool:**
- `runtime.config` — the parent's `RunnableConfig`, which contains `CONFIG_KEY_STREAM` (when parent uses `subgraphs=True`) and inherited callbacks
- `runtime.stream_writer` — the parent's live `StreamWriter` callable (when parent uses `stream_mode="custom"`)

---

## Two Parallel Streaming Systems in LangGraph

Before discussing approaches, it's important to understand that LangGraph has **two completely independent streaming systems**:

| System | Mechanism | Activated by | What it carries |
|--------|-----------|-------------|-----------------|
| **LangGraph Native Stream** | `StreamProtocol` queue, `StreamWriter` callable | `graph.stream(stream_mode="custom")` | Raw `Any` data written via `StreamWriter` |
| **LangChain Callback Events** | `_AstreamEventsCallbackHandler` in callback chain | `graph.astream_events()` | Structured `StreamEvent` dicts (on_custom_event, on_chain_*, etc.) |

These are NOT the same thing. `StreamWriter` writes to the Pregel loop's output queue. `dispatch_custom_event` fires through the LangChain callback system. A consumer using `graph.stream(stream_mode="custom")` will NOT see `dispatch_custom_event` events, and vice versa.

---

## All Possible Approaches

### Approach 1: Pass `runtime.config` to `subagent.invoke()` + `StreamWriter` in Subagent Nodes

**The Mechanism:**

Change one line in `subagents.py` to pass the parent's config to the subagent. The subagent's Pregel loop detects `CONFIG_KEY_STREAM` in the config and inherits the parent's `stream_writer`. Subagent nodes use `StreamWriter` to emit custom dicts.

**What to Change:**

| File | Change |
|------|--------|
| `deepagents/middleware/subagents.py:442` | `subagent.invoke(subagent_state)` → `subagent.invoke(subagent_state, runtime.config)` |
| `deepagents/middleware/subagents.py:460` | `await subagent.ainvoke(subagent_state)` → `await subagent.ainvoke(subagent_state, runtime.config)` |
| Each subagent node function | Add `writer: StreamWriter` param OR call `get_stream_writer()` |
| Your service layer (caller) | Call `graph.stream(input, stream_mode=["custom", "updates"], subgraphs=True)` |

**Subagent Node Code:**
```python
from langgraph.types import StreamWriter

def my_subagent_node(state: MyState, *, writer: StreamWriter):
    writer({"step": "parsing", "progress": 0.3, "detail": "Reading PDF pages..."})
    # ... do work ...
    writer({"step": "parsing", "progress": 0.8, "detail": "Extracting tables..."})
    return {"result": "parsed data"}
```

**Caller Code (streaming mode):**
```python
async for ns, mode, chunk in graph.astream(
    input_state, config=config,
    stream_mode=["custom", "updates", "messages"],
    subgraphs=True,
):
    if mode == "custom":
        # chunk is whatever was passed to writer()
        await websocket.send_json(chunk)
    elif mode == "messages":
        # LLM token streaming
        ...
    elif mode == "updates":
        # State updates per node
        ...
```

**Caller Code (non-streaming mode — unchanged):**
```python
result = graph.invoke(input_state, config=config)
# StreamWriter calls are no-ops when not streaming — zero overhead
```

**How It Works Internally:**

1. Parent calls `graph.stream(..., stream_mode=["custom", ...], subgraphs=True)`
2. `main.py:2612`: sets `loop.config[CONF][CONFIG_KEY_STREAM] = loop.stream`
3. `main.py:2547`: creates `stream_writer` closure that puts `(namespace, "custom", data)` into stream queue
4. ToolNode runs the `task` tool → `runtime.config` has `CONFIG_KEY_STREAM`
5. Task tool calls `subagent.invoke(subagent_state, runtime.config)` ← **the fix**
6. Subagent's `main.py:2561`: detects `CONFIG_KEY_STREAM` in config → inherits parent's `stream_writer` via `Runtime.merge()`
7. Subagent node calls `writer({"step": "parsing"})` → data flows to parent's stream queue
8. Parent yields `(namespace_tuple, "custom", {"step": "parsing"})` to caller

**Why `invoke()` still works:** When the parent is called with `.invoke()` (non-streaming), no `stream_mode` is active, `CONFIG_KEY_STREAM` is not set, `stream_writer` is a no-op. Subagent nodes call `writer(...)` which silently drops the data. Zero behavior change.

**Reasoning:** This is the most natural LangGraph approach. `StreamWriter` is the framework's own mechanism for custom streaming from nodes. The only missing piece is the config propagation — a one-line fix.

---

### Approach 2: `dispatch_custom_event` + `astream_events`

**The Mechanism:**

Subagent nodes call `adispatch_custom_event(name, data)` (from `langchain-core`). The parent graph is consumed via `graph.astream_events()` which captures all events including custom ones from subagent nodes.

**What to Change:**

| File | Change |
|------|--------|
| `deepagents/middleware/subagents.py:442` | `subagent.invoke(subagent_state)` → `subagent.invoke(subagent_state, runtime.config)` |
| `deepagents/middleware/subagents.py:460` | Same for async |
| Each subagent node function | Call `adispatch_custom_event("progress", {"step": "parsing", ...})` |
| Your service layer (caller) | Use `graph.astream_events(input, version="v2")` instead of `graph.stream()` |

**Subagent Node Code:**
```python
from langchain_core.callbacks import adispatch_custom_event

async def my_subagent_node(state: MyState):
    await adispatch_custom_event(
        "subagent_progress",
        {"step": "parsing", "progress": 0.3, "subagent": "pdf-subagent"}
    )
    # ... do work ...
    await adispatch_custom_event(
        "subagent_progress",
        {"step": "done", "progress": 1.0, "subagent": "pdf-subagent"}
    )
    return {"result": "parsed data"}
```

**Caller Code (streaming mode):**
```python
async for event in graph.astream_events(input_state, config=config, version="v2"):
    if event["event"] == "on_custom_event" and event["name"] == "subagent_progress":
        await websocket.send_json(event["data"])
    elif event["event"] == "on_chat_model_stream":
        # LLM tokens
        await websocket.send_json({"type": "token", "content": event["data"]["chunk"].content})
```

**Non-streaming mode:** Use `graph.invoke()` as before. `adispatch_custom_event` calls are effectively no-ops (no event handler to receive them).

**Why config propagation is still needed:** `adispatch_custom_event` reads the callback manager from the config's context var. While Python >= 3.11 propagates context vars across `await` calls, explicitly passing `runtime.config` ensures:
- Callbacks are reliably propagated regardless of Python version
- The `_AstreamEventsCallbackHandler` is in the subagent's callback chain
- `parent_ids` in events correctly reflect the nesting hierarchy

**Without config propagation (Python >= 3.11 only):** `adispatch_custom_event` reads from `var_child_runnable_config` context var, which IS set by the parent's ToolNode execution. The subagent's `Pregel.invoke()` calls `ensure_config(None)` which reads this context var. So callbacks MAY propagate without the one-line fix — but only on Python >= 3.11, and the `parent_ids` hierarchy may be incorrect.

---

### Approach 3: Manual Stream Forwarding via `runtime.stream_writer` in a Modified Task Tool

**The Mechanism:**

Instead of relying on LangGraph's config propagation, modify the `task` tool to call `subagent.stream()` instead of `subagent.invoke()`, iterate over chunks, and forward custom events to the parent via `runtime.stream_writer`.

**What to Change:**

| File | Change |
|------|--------|
| `deepagents/middleware/subagents.py` | Rewrite `task()` / `atask()` to use `.stream()` and forward chunks |
| Each subagent node function | Add `writer: StreamWriter` param |
| Your service layer (caller) | Call `graph.stream(input, stream_mode=["custom", "updates"])` |

**Modified Task Tool (conceptual):**
```python
def task(subagent_type: str, description: str, *, runtime: ToolRuntime) -> Command:
    subagent, subagent_state = _validate_and_prepare_state(subagent_type, description, runtime)

    result = None
    for chunk in subagent.stream(
        subagent_state,
        stream_mode=["custom", "values"],
    ):
        mode, data = chunk
        if mode == "custom":
            # Forward custom events to parent stream with subagent context
            runtime.stream_writer({
                "source": subagent_type,
                **data,
            })
        elif mode == "values":
            result = data  # Last "values" chunk is the final state

    return _return_command_with_state_update(result, subagent_type, description)
```

**Async version:**
```python
async def atask(subagent_type: str, description: str, *, runtime: ToolRuntime) -> Command:
    subagent, subagent_state = _validate_and_prepare_state(subagent_type, description, runtime)

    result = None
    async for chunk in subagent.astream(
        subagent_state,
        stream_mode=["custom", "values"],
    ):
        mode, data = chunk
        if mode == "custom":
            runtime.stream_writer({
                "source": subagent_type,
                **data,
            })
        elif mode == "values":
            result = data

    return _return_command_with_state_update(result, subagent_type, description)
```

**Reasoning:** This approach gives you full control over what gets forwarded and how. You can enrich events with subagent metadata, filter events, transform data, etc. The subagent doesn't need config from the parent — it streams locally and the task tool explicitly forwards.

**Non-streaming mode compatibility:** When the parent uses `.invoke()`, `runtime.stream_writer` is a no-op. The forwarding calls silently drop data. The subagent's `.stream()` still works fine (it just iterates chunks locally). The final `result` is collected from the last `"values"` chunk. Functionally equivalent to `.invoke()`.

---

### Approach 4: Middleware `wrap_tool_call` Interception with Streaming Subagent

**The Mechanism:**

Create a custom middleware that intercepts the `task` tool call via `wrap_tool_call`, replaces the subagent invocation with a streaming one, and forwards events via `runtime.stream_writer`.

**What to Change:**

| File | Change |
|------|--------|
| Your middleware file | New `SubagentStreamingMiddleware` with `wrap_tool_call` |
| Each subagent node | Use `StreamWriter` or `dispatch_custom_event` |
| Your service layer | Add middleware to `create_deep_agent(middleware=[...])` |

**Middleware Code:**
```python
class SubagentStreamingMiddleware(AgentMiddleware):
    def wrap_tool_call(self, request, handler):
        if request.tool_call["name"] != "task":
            return handler(request)

        # Let the original handler run — but we could intercept before/after
        # For streaming, we'd need deeper integration
        response = handler(request)
        return response
```

**Limitation:** `wrap_tool_call` wraps the entire tool execution. You can't easily replace `.invoke()` with `.stream()` from inside `wrap_tool_call` without re-implementing the task tool logic. This approach is better for adding metadata to events or filtering, not for fundamentally changing the invocation pattern.

---

### Approach 5: Register Subagents as LangGraph Subgraph Nodes (Architectural Change)

**The Mechanism:**

Instead of invoking subagents via a `task` tool inside `ToolNode`, register each subagent as an actual **subgraph node** in the parent graph. LangGraph natively handles subgraph streaming — `subgraphs=True` propagates `CONFIG_KEY_STREAM` and all streaming just works.

**What to Change:**

| File | Change |
|------|--------|
| Your agent construction code | Register subagents as graph nodes instead of using SubAgentMiddleware |
| Routing logic | Replace `task` tool-based routing with conditional edges or `Command(goto=...)` |
| State schema | Subagent state schemas must align with parent graph state |

**Conceptual Code:**
```python
from langgraph.graph import StateGraph

builder = StateGraph(OrchestratorState)
builder.add_node("orchestrator", orchestrator_node)
builder.add_node("pdf_subagent", pdf_workflow.compile())    # subgraph node
builder.add_node("csv_subagent", csv_workflow.compile())    # subgraph node
builder.add_node("postgres_subagent", pg_workflow.compile())# subgraph node

# Conditional routing from orchestrator to subagents
builder.add_conditional_edges("orchestrator", route_to_subagent)
builder.add_edge("pdf_subagent", "orchestrator")
builder.add_edge("csv_subagent", "orchestrator")
builder.add_edge("postgres_subagent", "orchestrator")

graph = builder.compile()

# Streaming just works:
async for ns, mode, chunk in graph.astream(
    input, stream_mode=["custom", "updates"], subgraphs=True
):
    if mode == "custom":
        print(f"From {ns}: {chunk}")
```

**Why this works:** When a compiled graph is used as a **node** (not a tool), LangGraph's `get_subgraphs()` recognizes it. The `subgraphs=True` parameter injects `CONFIG_KEY_STREAM` into the subgraph's config. The `DuplexStream` fans out chunks to both the subgraph's local stream and the parent's stream. Streaming propagation is automatic — no manual forwarding needed.

**Reasoning:** This is the "LangGraph-native" way to do multi-agent with streaming. But it requires significant restructuring — you lose the `task` tool's dynamic routing and the SubAgentMiddleware's state preparation logic.

---

### Approach 6: Hybrid — `runtime.config` Propagation + `dispatch_custom_event` Fallback

**The Mechanism:**

Combine Approach 1 (config propagation) with Approach 2 (dispatch_custom_event) for maximum compatibility. Subagent nodes use both `StreamWriter` (for `stream_mode="custom"` consumers) and `dispatch_custom_event` (for `astream_events` consumers).

**Subagent Node Code:**
```python
from langgraph.types import StreamWriter
from langchain_core.callbacks import adispatch_custom_event

async def my_node(state: MyState, *, writer: StreamWriter):
    event_data = {"step": "parsing", "progress": 0.5}

    # StreamWriter — for graph.stream(stream_mode="custom") consumers
    writer(event_data)

    # dispatch_custom_event — for graph.astream_events() consumers
    await adispatch_custom_event("progress", event_data)

    return {"result": "done"}
```

**Why:** Your service layer might use `graph.stream()` in some contexts and `graph.astream_events()` in others (e.g., LangGraph Platform uses `astream_events`). This approach ensures custom events are visible in both systems.

---

## Comparison Table

| Approach | Source Changes | Subagent Node Changes | Streaming System | Non-Streaming Compatible | Subagent Namespace Visible | Complexity | Cross-Boundary Reliable |
|----------|:-------------:|:--------------------:|:----------------:|:------------------------:|:-------------------------:|:----------:|:----------------------:|
| **1. Config propagation + StreamWriter** | 2 lines in `subagents.py` | Add `writer: StreamWriter` param | LangGraph native (`stream_mode="custom"`) | Yes (writer is no-op) | Yes (with `subgraphs=True`) | **Low** | Yes |
| **2. Config propagation + dispatch_custom_event** | 2 lines in `subagents.py` | Add `adispatch_custom_event()` calls | LangChain callbacks (`astream_events`) | Yes (no handler = no-op) | Yes (via `parent_ids`) | **Low** | Yes (with config) |
| **3. Manual stream forwarding** | Rewrite `task()` (~30 lines) | Add `writer: StreamWriter` param | LangGraph native | Yes (writer is no-op) | Yes (manual enrichment) | **Medium** | Yes (explicit) |
| **4. wrap_tool_call middleware** | New middleware (~20 lines) | Depends on inner approach | Either | Yes | Partial | **Medium** | Partial |
| **5. Subgraph nodes** | Major restructuring | Add `writer: StreamWriter` param | LangGraph native (automatic) | Yes | Yes (automatic) | **High** | Yes (native) |
| **6. Hybrid (StreamWriter + dispatch)** | 2 lines in `subagents.py` | Both writer + dispatch | Both systems | Yes | Yes | **Low-Medium** | Yes |

## Detailed Pros and Cons

| Approach | Pros | Cons |
|----------|------|------|
| **1. Config + StreamWriter** | Minimal code change (2 lines). Framework-native mechanism. No-op when not streaming. Namespace-aware with `subgraphs=True`. Works sync and async. | Requires modifying Deep Agents source (`subagents.py`). Only works with `stream_mode="custom"` consumers. Not visible in `astream_events`. |
| **2. Config + dispatch_custom_event** | Minimal code change. Works with `astream_events` (standard LangGraph Platform API). Structured event schema with `name` + `data`. Filterable by name. | Requires modifying Deep Agents source. Requires Python >= 3.11 for reliable async without explicit config. Slightly more verbose in nodes. Not visible in `stream_mode="custom"`. |
| **3. Manual forwarding** | Full control over what gets forwarded. Can enrich/filter/transform events. Can add subagent metadata. No dependency on internal `CONFIG_KEY_STREAM`. | More code to maintain. Must handle `.stream()` iteration + result collection. Slightly more overhead (local stream + forward). More invasive change to `subagents.py`. |
| **4. wrap_tool_call middleware** | No changes to `subagents.py` (framework untouched). Pluggable via middleware stack. | Cannot easily change `.invoke()` to `.stream()` from within `wrap_tool_call`. Limited interception capability. Better for post-processing than streaming. |
| **5. Subgraph nodes** | Zero streaming plumbing needed — LangGraph handles everything. Native namespace hierarchy. Best for LangGraph Studio/Platform. | Major architectural restructuring. Loses `task` tool's dynamic routing. Must align state schemas. Loses SubAgentMiddleware features (state preparation, excluded keys). |
| **6. Hybrid** | Maximum compatibility — works with both `stream()` and `astream_events()` consumers. Future-proof. | Dual emission in every node (minor code duplication). Slightly more node-level boilerplate. |

---

## Recommendation

**Use Approach 1 (Config propagation + StreamWriter)** as the primary solution:

1. **Smallest change** — 2 lines in `subagents.py`
2. **Framework-native** — uses LangGraph's own `StreamWriter` mechanism
3. **Zero overhead in non-streaming mode** — `StreamWriter` is a no-op when `stream_mode="custom"` is not active
4. **Namespace-aware** — with `subgraphs=True`, each subagent's events are tagged with the subagent's namespace
5. **Already supported by CLI** — the Deep Agents CLI already uses `subgraphs=True`

**Add Approach 2 as a complement** if you also need `astream_events` support (e.g., for LangGraph Platform deployment).

### Implementation Steps

**Step 1: One-line fix in `subagents.py` (both sync and async):**
```python
# subagents.py:442
result = subagent.invoke(subagent_state, runtime.config)

# subagents.py:460
result = await subagent.ainvoke(subagent_state, runtime.config)
```

**Step 2: Add StreamWriter to your subagent nodes:**
```python
from langgraph.types import StreamWriter

def my_node(state: MyState, *, writer: StreamWriter):
    writer({"type": "progress", "step": "parsing", "pct": 30})
    # ... do work ...
    writer({"type": "progress", "step": "complete", "pct": 100})
    return {"parsed_data": result}
```

**Step 3: Call with streaming from your service layer:**
```python
# Streaming mode (for frontend):
async for ns, mode, chunk in graph.astream(
    input_state,
    config=config,
    stream_mode=["custom", "messages", "updates"],
    subgraphs=True,
):
    if mode == "custom":
        await send_to_frontend({"type": "subagent_event", "namespace": ns, **chunk})
    elif mode == "messages":
        msg_chunk, metadata = chunk
        if msg_chunk.content:
            await send_to_frontend({"type": "token", "content": msg_chunk.content})

# Non-streaming mode (unchanged):
result = graph.invoke(input_state, config=config)
```

---

## Source File Reference

| Concept | File | Lines |
|---------|------|-------|
| `StreamWriter` type | `langgraph/types.py` | 111 |
| `get_stream_writer()` | `langgraph/config.py` | 126 |
| `Runtime.stream_writer` | `langgraph/runtime.py` | dataclass field |
| `Runtime.merge()` (inherits parent writer) | `langgraph/runtime.py` | ~118 |
| `stream_mode` options | `langgraph/types.py` | 95-109 |
| Stream writer setup (3 cases) | `langgraph/pregel/main.py` | 2546-2580 |
| `CONFIG_KEY_STREAM` propagation (subgraphs) | `langgraph/pregel/main.py` | 2611-2613 |
| `DuplexStream` (parent+child fan-out) | `langgraph/pregel/_loop.py` | 131, 253 |
| `StreamProtocol` | `langgraph/pregel/protocol.py` | 148-164 |
| `_output()` tuple assembly | `langgraph/pregel/main.py` | 3252-3292 |
| `StreamMessagesHandler` (messages mode) | `langgraph/pregel/_messages.py` | 42-250 |
| Node param injection (writer) | `langgraph/_internal/_runnable.py` | 130-184 |
| `ToolRuntime.stream_writer` field | `langgraph/prebuilt/tool_node.py` | 1589-1594 |
| ToolRuntime construction (copies writer) | `langgraph/prebuilt/tool_node.py` | 806, 838 |
| **Subagent invocation (THE GAP)** | `deepagents/middleware/subagents.py` | 442, 460 |
| `runtime.config` available in task tool | `langgraph/prebuilt/tool_node.py` | 1591 |
| `_validate_and_prepare_state` | `deepagents/middleware/subagents.py` | 422-428 |
| `dispatch_custom_event` | `langchain_core/callbacks/manager.py` | ~2475 |
| `adispatch_custom_event` | `langchain_core/callbacks/manager.py` | ~2600 |
| `on_custom_event` handler | `langchain_core/callbacks/base.py` | ~447 |
| `CustomStreamEvent` schema | `langchain_core/runnables/schema.py` | ~176 |
| `_AstreamEventsCallbackHandler` custom events | `langchain_core/tracers/event_stream.py` | ~405 |
| CLI streaming usage | `deepagents_cli/textual_adapter.py` | 279 |
| CLI stream_mode | `["messages", "updates"]` with `subgraphs=True` | — |
| ACP streaming usage | `deepagents_acp/agent.py` | 456 |
| `create_deep_agent` return type | `deepagents/graph.py` | 68 |
| Middleware `runtime.stream_writer` docs | `langchain/agents/middleware/types.py` | 1325-1362 |
