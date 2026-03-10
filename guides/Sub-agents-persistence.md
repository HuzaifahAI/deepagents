# Sub-Agent Config Propagation & Persistence Analysis

## 1. Parent `runtime.config` — Full Contents During Streaming

When the parent Deep Agent streams with `stream_mode="custom"` and `subgraphs=True`, the `runtime.config` available inside the `task()` / `atask()` tool closures contains:

### Top-Level Keys

| Key                | Type              | Example Value                                                        | Description                                      |
|--------------------|-------------------|----------------------------------------------------------------------|--------------------------------------------------|
| `callbacks`        | `CallbackManager` | `<CallbackManager object>`                                           | Parent's LangChain callback chain for tracing     |
| `metadata`         | `dict`            | `{lc_agent_name: "supervisor", thread_id: "t1", langgraph_step: 2}` | Parent identity and execution metadata            |
| `tags`             | `list`            | `["my-tag"]`                                                         | User-supplied tags for observability              |
| `recursion_limit`  | `int`             | `25`                                                                 | Max recursion depth for nested graph calls        |

### `configurable` Dict Keys

| Key                      | Type                        | Description                                                                 |
|--------------------------|-----------------------------|-----------------------------------------------------------------------------|
| `__pregel_stream`        | `StreamProtocol`            | Parent's stream queue — the pipe through which subagent events flow         |
| `__pregel_runtime`       | `Runtime`                   | Holds `stream_writer` closure, `context`, `store`                           |
| `__pregel_checkpointer`  | `InMemorySaver`             | Parent's checkpointer instance                                              |
| `__pregel_call`          | `functools.partial`         | Internal Pregel call dispatch function                                      |
| `__pregel_read`          | `functools.partial`         | Internal Pregel state reader                                                |
| `__pregel_send`          | `builtin_function_or_method`| Internal Pregel send queue (deque.extend)                                   |
| `__pregel_scratchpad`    | `PregelScratchpad`          | Step counter, call counter, execution bookkeeping                           |
| `__pregel_task_id`       | `str`                       | Parent's current task ID (UUID)                                             |
| `thread_id`              | `str`                       | Checkpoint thread identifier (e.g. `"t1"`)                                  |
| `checkpoint_id`          | `NoneType`                  | Current checkpoint ID (None during execution)                               |
| `checkpoint_ns`          | `str`                       | Checkpoint namespace (e.g. `"tools:<task_id>"`)                             |
| `checkpoint_map`         | `dict`                      | Checkpoint hierarchy mapping                                                |

---

## 2. What Gets Extracted and Passed to Sub-Agents

The `_build_subagent_config()` helper in `subagents.py` extracts **only 2 keys** from the parent's `configurable` dict:

| Key                 | Type             | Passed? | Why                                                                              |
|---------------------|------------------|---------|----------------------------------------------------------------------------------|
| `__pregel_stream`   | `StreamProtocol` | **Yes** | Core streaming infrastructure — without this, `StreamWriter` is a no-op          |
| `__pregel_runtime`  | `Runtime`        | **Yes** | Subagent's Pregel loop merges this to inherit the live `stream_writer` closure    |

Everything else is **intentionally excluded**:

| Key                      | Passed? | Why excluded                                                              |
|--------------------------|---------|---------------------------------------------------------------------------|
| `callbacks`              | No      | Would override subagent's own `lc_agent_name` with parent's name         |
| `metadata`               | No      | Contains parent identity (`lc_agent_name: "supervisor"`)                  |
| `tags`                   | No      | Not needed for streaming; subagent can have its own                       |
| `recursion_limit`        | No      | Subagent uses its own default                                             |
| `__pregel_checkpointer`  | No      | Subagents don't have their own checkpointer (primary use case)            |
| `__pregel_call`          | No      | Parent's internal execution state                                         |
| `__pregel_read`          | No      | Parent's internal execution state                                         |
| `__pregel_send`          | No      | Parent's internal execution state                                         |
| `__pregel_scratchpad`    | No      | Parent's internal execution state                                         |
| `__pregel_task_id`       | No      | Parent's internal execution state                                         |
| `thread_id`              | No      | Checkpoint identity — not needed without checkpointer                     |
| `checkpoint_id`          | No      | Checkpoint identity                                                       |
| `checkpoint_ns`          | No      | Checkpoint namespace                                                      |
| `checkpoint_map`         | No      | Checkpoint hierarchy                                                      |

### Resulting config passed to subagent

**When parent is streaming** (`stream_mode="custom"`, `subgraphs=True`):
```python
{"configurable": {"__pregel_stream": <StreamProtocol>, "__pregel_runtime": <Runtime>}}
```

**When parent is not streaming** (`.invoke()`):
```python
{}  # empty dict — StreamWriter remains a no-op in the subagent
```

---

## 3. Checkpoint Compatibility Tests

### Test Setup

A subagent graph with a `StreamWriter` node:

```python
class SubState(TypedDict):
    messages: list[BaseMessage]

def streaming_node(state: SubState, writer: StreamWriter) -> SubState:
    writer({"event": "step-1"})
    return {"messages": [AIMessage(content="Result.")]}

g = StateGraph(SubState)
g.add_node("work", streaming_node)
g.add_edge(START, "work")
g.add_edge("work", END)
```

### Case 1: Subagent WITHOUT checkpointer

```python
subagent = g.compile()  # No checkpointer
```

| Aspect                        | Result                                                     |
|-------------------------------|------------------------------------------------------------|
| Streaming (`StreamWriter`)    | **Works** — custom events flow to parent stream            |
| State copy (parent→sub→parent)| **Works** — unchanged by config propagation                |
| Subagent identity             | **Preserved** — keeps its own `run_name`, `run_id`         |
| Errors                        | None                                                       |

**Output:**
```
Custom events received: [{'event': 'no-checkpoint-step-1'}]
Streaming works: True
```

### Case 2: Subagent WITH its own checkpointer

```python
subagent = g.compile(checkpointer=InMemorySaver())  # Has checkpointer
```

| Aspect          | Before streaming change (no config passed)                                               | After streaming change (streaming config passed)                                   |
|-----------------|------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------|
| Error           | `ValueError: Checkpointer requires one or more of the following 'configurable' keys: thread_id, checkpoint_ns, checkpoint_id` | `KeyError: 'thread_id'`                                                           |
| Streaming       | N/A (invocation fails)                                                                    | N/A (invocation fails)                                                             |
| Root cause      | No config passed at all → checkpointer can't find `thread_id`                            | Config has only streaming keys → checkpointer still can't find `thread_id`         |

---

## 4. Why Sub-Agents With Checkpointers Raise Errors

### Root Cause

When a LangGraph `CompiledStateGraph` is compiled with a checkpointer (e.g. `InMemorySaver`), its Pregel execution loop **requires** a `thread_id` in `config["configurable"]` to know where to save/load checkpoints. This is enforced in `langgraph/pregel/main.py` during the Pregel loop setup.

### The Error Chain

1. **Old behavior** (no config passed): `subagent.invoke(state)` → Pregel loop starts → checkpointer is present → looks for `thread_id` in `config["configurable"]` → `configurable` dict is empty → raises `ValueError: Checkpointer requires one or more of the following 'configurable' keys: thread_id, checkpoint_ns, checkpoint_id`

2. **New behavior** (streaming config passed): `subagent.invoke(state, {"configurable": {"__pregel_stream": ..., "__pregel_runtime": ...}})` → Pregel loop starts → checkpointer is present → looks for `thread_id` in `config["configurable"]` → `configurable` dict has streaming keys but no `thread_id` → raises `KeyError: 'thread_id'`

### Key Insight

**Sub-agents with their own checkpointers were never supported.** The error existed before the streaming change — only the error type changed (`ValueError` → `KeyError`). The streaming change does not introduce a new failure mode.

### What Would Be Needed to Support Checkpointer Sub-Agents

To support sub-agents with their own checkpointers, `_build_subagent_config()` would need to:

1. Generate a unique `thread_id` for each subagent invocation (to avoid checkpoint collisions with the parent)
2. Include it in the config: `{"configurable": {"thread_id": "<unique-id>", "__pregel_stream": ..., "__pregel_runtime": ...}}`

This is a separate feature and out of scope for the streaming propagation change.

---

## 5. State Copy Pipeline (Unaffected)

The state copy flow between parent and subagent is **completely independent** of the config change:

```
Parent state (runtime.state)
    │
    ▼ _validate_and_prepare_state() — line 426
    Filter out _EXCLUDED_STATE_KEYS (messages, todos, structured_response,
    skills_metadata, memory_contents), inject fresh HumanMessage
    │
    ▼ subagent.invoke(subagent_state, subagent_config) — line 476
    Config only affects Pregel execution loop (streaming, checkpointing).
    State channels and schema are untouched.
    │
    ▼ _return_command_with_state_update(result, tool_call_id) — line 412
    Filter result through _EXCLUDED_STATE_KEYS, extract last message as
    ToolMessage, return Command(update={...})
    │
    ▼ Parent state updated via Command
```

The streaming config keys (`__pregel_stream`, `__pregel_runtime`) are consumed by the Pregel execution loop and **never appear in the state dict**. The state copy pipeline operates on `runtime.state` and the subagent's returned state dict — both are orthogonal to the config.
