# LangGraph: All Ways to Stream + Get the Final Result

## The Problem

In your `test_streaming_subagents.py`, you're calling the graph **twice**:
1. First with `.stream(stream_mode="custom", subgraphs=True)` to get the custom StreamWriter events
2. Then with `.invoke()` to get the final message content

This is wasteful — it re-runs the LLM. Here are **all the possible ways** to stream AND capture the final result in a single invocation.

---

## Way 1: ✅ Multiple `stream_mode` (RECOMMENDED)

> [!TIP]
> This is the cleanest solution. You can pass **a list** of stream modes to get both custom events AND state updates in a single stream.

```python
final_state = None
custom_events = []

for chunk in agent.stream(
    {"messages": [HumanMessage(content=query)]},
    config={"configurable": {"thread_id": "test"}},
    stream_mode=["custom", "values"],  # ← list of modes!
    subgraphs=True,
):
    # With multiple modes + subgraphs, chunk is (namespace, data)
    # But with v2 format, chunk is a dict with "type" key
    # Without v2, the output format is: (stream_mode, data) tuples
    mode, data = chunk  # or namespace handling with subgraphs

    if mode == "custom" and isinstance(data, dict) and "node" in data:
        custom_events.append(data)
        print(format_event(data))
    elif mode == "values":
        final_state = data  # keep overwriting — last one is the final state

# After the loop, final_state contains the complete final state
final_answer = final_state["messages"][-1].content
```

### With v2 streaming format (newer LangGraph versions):
```python
for chunk in agent.stream(
    {"messages": [HumanMessage(content=query)]},
    config={"configurable": {"thread_id": "test"}},
    stream_mode=["custom", "values"],
    version="v2",
):
    if chunk["type"] == "custom":
        custom_events.append(chunk["data"])
    elif chunk["type"] == "values":
        final_state = chunk["data"]

final_answer = final_state["messages"][-1].content
```

**Pros:** Single invocation, clean separation of event types, officially supported
**Cons:** Slightly more data streamed (full state per step with `values`)

---

## Way 2: ✅ Multiple `stream_mode` with `updates` instead of `values`

Same as Way 1 but using `"updates"` instead of `"values"` — less data per chunk since it only sends deltas:

```python
all_updates = {}
custom_events = []

for chunk in agent.stream(
    {"messages": [HumanMessage(content=query)]},
    config={"configurable": {"thread_id": "test"}},
    stream_mode=["custom", "updates"],
    subgraphs=True,
):
    mode, data = chunk

    if mode == "custom" and isinstance(data, dict) and "node" in data:
        custom_events.append(data)
        print(format_event(data))
    elif mode == "updates":
        # data is {node_name: {state_key: value}}
        # The last update from the final node contains the result
        all_updates = data  # keep the last update

# You'd need to reconstruct or just use get_state (see Way 3)
```

**Pros:** Less overhead than `values` mode
**Cons:** You only get deltas, not the full state — reconstruction is needed

---

## Way 3: ✅ `get_state()` After Streaming (No Second Invoke)

Use `get_state()` instead of `invoke()` after streaming. This reads from the **checkpoint** — it does NOT re-run the graph:

```python
config = {"configurable": {"thread_id": "test"}}
custom_events = []

for namespace, data in agent.stream(
    {"messages": [HumanMessage(content=query)]},
    config=config,
    stream_mode="custom",
    subgraphs=True,
):
    if isinstance(data, dict) and "node" in data:
        custom_events.append(data)
        print(format_event(data))

# ✅ This reads from the checkpoint — does NOT re-execute the graph!
state_snapshot = agent.get_state(config)
final_answer = state_snapshot.values["messages"][-1].content
```

> [!IMPORTANT]
> `get_state()` requires a **checkpointer** to be configured (you already have `InMemorySaver`). This is the simplest fix to your current code — just replace `agent.invoke(...)` with `agent.get_state(config)`.

**Pros:** Minimal change to existing code, no re-execution, very clean
**Cons:** Requires checkpointer (you already have one)

---

## Way 4: ✅ Track the Last `values` Chunk in the Loop

Stream with `stream_mode="values"` only (no custom), and the **last chunk** emitted will be the complete final state:

```python
final_state = None

for chunk in agent.stream(
    {"messages": [HumanMessage(content=query)]},
    config={"configurable": {"thread_id": "test"}},
    stream_mode="values",
    subgraphs=True,
):
    final_state = chunk  # last one will be the final state

final_answer = final_state["messages"][-1].content
```

**Pros:** Very simple
**Cons:** You lose the `custom` StreamWriter events — not suitable if you need both custom events AND final state (unless combined with Way 1)

---

## Way 5: ✅ `stream_mode="checkpoints"`

The `checkpoints` stream mode emits checkpoint data (equivalent to `get_state()` output) after each step:

```python
final_checkpoint = None
custom_events = []

for chunk in agent.stream(
    {"messages": [HumanMessage(content=query)]},
    config={"configurable": {"thread_id": "test"}},
    stream_mode=["custom", "checkpoints"],
    subgraphs=True,
):
    mode, data = chunk

    if mode == "custom" and isinstance(data, dict) and "node" in data:
        custom_events.append(data)
        print(format_event(data))
    elif mode == "checkpoints":
        final_checkpoint = data  # last one = final checkpoint

# final_checkpoint contains the state snapshot
final_answer = final_checkpoint["values"]["messages"][-1].content
```

**Pros:** Very explicit, gives you checkpoint metadata (timestamps, etc.)
**Cons:** More data than `values` mode, includes checkpoint metadata you may not need

---

## Way 6: ✅ Async `astream()` with Multiple Modes

If you're in an async context:

```python
import asyncio

async def run_streaming():
    config = {"configurable": {"thread_id": "test"}}
    custom_events = []
    final_state = None

    async for chunk in agent.astream(
        {"messages": [HumanMessage(content=query)]},
        config=config,
        stream_mode=["custom", "values"],
        subgraphs=True,
    ):
        mode, data = chunk
        if mode == "custom" and isinstance(data, dict) and "node" in data:
            custom_events.append(data)
            print(format_event(data))
        elif mode == "values":
            final_state = data

    final_answer = final_state["messages"][-1].content
    return final_answer

result = asyncio.run(run_streaming())
```

**Pros:** Non-blocking, same multi-mode support
**Cons:** Requires async context

---

## Way 7: ✅ `astream_events()` (Most Granular)

The most granular API — gives you lifecycle events for every node, LLM call, and tool invocation:

```python
async def run_with_events():
    config = {"configurable": {"thread_id": "test"}}
    final_message = None

    async for event in agent.astream_events(
        {"messages": [HumanMessage(content=query)]},
        config=config,
        version="v2",
    ):
        kind = event["event"]

        # Custom events from StreamWriter
        if kind == "on_custom_event":
            print(format_event(event["data"]))

        # Capture final AI message from the last node
        if kind == "on_chat_model_end":
            final_message = event["data"]["output"]

    return final_message
```

**Pros:** Maximum granularity, token-by-token streaming possible
**Cons:** Complex event filtering, verbose, harder to work with

---

## Summary Comparison

| # | Approach | Re-runs Graph? | Gets Custom Events? | Gets Final State? | Complexity |
|---|----------|:--------------:|:-------------------:|:-----------------:|:----------:|
| **1** | `stream_mode=["custom", "values"]` | ❌ No | ✅ Yes | ✅ Yes | ⭐ Low |
| **2** | `stream_mode=["custom", "updates"]` | ❌ No | ✅ Yes | ⚠️ Deltas only | ⭐ Low |
| **3** | `stream` + `get_state()` after | ❌ No | ✅ Yes | ✅ Yes | ⭐ Low |
| **4** | `stream_mode="values"` only | ❌ No | ❌ No | ✅ Yes | ⭐ Low |
| **5** | `stream_mode=["custom", "checkpoints"]` | ❌ No | ✅ Yes | ✅ Yes | ⭐⭐ Med |
| **6** | `astream()` with multiple modes | ❌ No | ✅ Yes | ✅ Yes | ⭐⭐ Med |
| **7** | `astream_events(version="v2")` | ❌ No | ✅ Yes | ✅ Yes | ⭐⭐⭐ High |
| ❌ | **Your current approach** (`stream` + `invoke`) | **✅ Yes!** | ✅ Yes | ✅ Yes | ⭐ Low |

> [!WARNING]
> Your current approach calls `invoke()` after `stream()`, which **re-executes the entire graph** — meaning a second LLM call, second subagent invocations, etc. All approaches above avoid this.

---

## My Recommendations

### Quickest Fix → **Way 3** (`get_state()`)
Just replace your `invoke()` call with `get_state()`. One-line change:
```diff
-    result = agent.invoke(
-        {"messages": [HumanMessage(content="Summarize what you found")]},
-        config={"configurable": {"thread_id": "test_both_subagents"}},
-    )
-    final_answer = result["messages"][-1].content
+    state = agent.get_state({"configurable": {"thread_id": "test_both_subagents"}})
+    final_answer = state.values["messages"][-1].content
```

### Best Overall → **Way 1** (`stream_mode=["custom", "values"]`)
Everything in one loop, one invocation, full state available at the end.
