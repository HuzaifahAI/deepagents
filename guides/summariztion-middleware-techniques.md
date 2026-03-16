Final Synthesized Strategy: Displaying All Messages for a Thread

  New Key Findings from Research

  1. StoreBackend already exists in Deep Agents (deepagents/backends/store.py) — it wraps LangGraph's BaseStore. If you
  configure the agent with a PostgresStore, the StoreBackend automatically writes to Postgres. No custom backend needed.
  2. REMOVE_ALL_MESSAGES = "__remove_all__" — it's a hard replacement in add_messages. The reducer returns
  right[remove_all_idx + 1:], completely discarding left. No tombstoning, no soft-delete.
  3. PostgresSaver is append-only — every checkpoint gets a unique checkpoint_id. Old checkpoints are NEVER deleted. The
  checkpoint_blobs table retains old channel values even after messages are wiped from the current state.
  4. Backend offload is LOSSY — get_buffer_string() produces plain text. Message metadata, tool_calls, IDs are all lost. Not
  suitable for UI reconstruction.
  5. ShallowPostgresSaver (deprecated) and durability='exit' mode only keep 1 checkpoint — no history recovery possible with
  those.

  ---
  The Complete Picture

  ┌─────────────────────────── YOUR SETUP ───────────────────────────────┐
  │                                                                       │
  │  Deep Agents (create_deep_agent)                                      │
  │  ├── PostgresSaver (checkpointer)  → checkpoints table                │
  │  ├── PostgresStore (store)         → store table                      │
  │  └── SummarizationMiddleware                                          │
  │       ├── backend = StoreBackend(runtime)  → writes to store table    │
  │       └── before_model → RemoveMessage(REMOVE_ALL_MESSAGES)           │
  │                                                                       │
  │  POSTGRES DB                                                          │
  │  ┌─────────────────────────────────────────────────────────────────┐  │
  │  │ checkpoints table                                               │  │
  │  │   (thread_id, checkpoint_ns, checkpoint_id) → checkpoint JSONB  │  │
  │  │   ALL versions retained (append-only)                           │  │
  │  │                                                                 │  │
  │  │ checkpoint_blobs table                                          │  │
  │  │   (thread_id, checkpoint_ns, channel, version) → blob BYTEA    │  │
  │  │   Old message blobs retained until explicit prune               │  │
  │  │                                                                 │  │
  │  │ store table                                                     │  │
  │  │   (prefix, key) → value JSONB                                   │  │
  │  │   Flat key-value, no history per key (UPSERT)                   │  │
  │  └─────────────────────────────────────────────────────────────────┘  │
  └───────────────────────────────────────────────────────────────────────┘

  ---
  The 4 Strategies (Updated with Worker Findings)

  Strategy A: Checkpoint History Traversal (Zero Code Changes)

  PostgresSaver.list(config) → iterate ALL checkpoints (newest → oldest)

    Checkpoint 50 (step=49):  messages = [H1, A1, T1, ..., A50]  ← FULL (150 msgs)
    Checkpoint 51 (step=50):  messages = [Summary1, A41, ..., A50] ← post-summarization
    ...
    Checkpoint 80 (step=79):  messages = [Summary1, A41, ..., A80] ← grown again
    Checkpoint 81 (step=80):  messages = [Summary2, A71, ..., A80] ← post-summarization
    Checkpoint 95 (step=94):  messages = [Summary2, A71, ..., A95] ← current

  Algorithm:
  async def get_full_history(checkpointer, thread_id: str) -> list[AnyMessage]:
      config = {"configurable": {"thread_id": thread_id}}

      # Collect all checkpoints (newest first)
      all_ckpts = []
      async for ckpt_tuple in checkpointer.alist(config):
          all_ckpts.append(ckpt_tuple)

      # Reverse to oldest-first
      all_ckpts.reverse()

      # Walk forward, detect summarization boundaries
      all_messages_by_id: dict[str, AnyMessage] = {}
      message_order: list[str] = []

      for ckpt in all_ckpts:
          msgs = ckpt.checkpoint["channel_values"].get("messages", [])
          for msg in msgs:
              if (isinstance(msg, HumanMessage)
                  and msg.additional_kwargs.get("lc_source") == "summarization"):
                  continue  # skip summary messages
              if msg.id not in all_messages_by_id:
                  all_messages_by_id[msg.id] = msg
                  message_order.append(msg.id)

      return [all_messages_by_id[mid] for mid in message_order]

  ┌──────────────┬──────────────────────────────────────────────────────────────────────────────┐
  │    Aspect    │                                    Detail                                    │
  ├──────────────┼──────────────────────────────────────────────────────────────────────────────┤
  │ Data loss    │ None — full message objects with all metadata                                │
  ├──────────────┼──────────────────────────────────────────────────────────────────────────────┤
  │ Code changes │ None                                                                         │
  ├──────────────┼──────────────────────────────────────────────────────────────────────────────┤
  │ Performance  │ Terrible — must load/deserialize EVERY checkpoint. Could be 1000s per thread │
  ├──────────────┼──────────────────────────────────────────────────────────────────────────────┤
  │ Works with   │ PostgresSaver (default durability) ONLY                                      │
  ├──────────────┼──────────────────────────────────────────────────────────────────────────────┤
  │ Fails with   │ ShallowPostgresSaver, durability='exit'                                      │
  └──────────────┴──────────────────────────────────────────────────────────────────────────────┘

  ---
  Strategy B: PostgresStore Pre-Archive (Recommended)

  Since StoreBackend already exists and wraps BaseStore, and the summarization middleware already accepts a backend parameter,
   you can configure it to use StoreBackend. But the current offload is LOSSY (get_buffer_string).

  The fix: Subclass to archive structured message objects to the store directly via runtime.store.put():

  from datetime import UTC, datetime
  from langchain.agents.middleware.summarization import SummarizationMiddleware as BaseSummarizationMiddleware
  from langchain_core.messages import AnyMessage, HumanMessage, RemoveMessage
  from langgraph.graph.message import REMOVE_ALL_MESSAGES


  class ArchivingSummarizationMiddleware(SummarizationMiddleware):
      """SummarizationMiddleware that archives full message objects to BaseStore."""

      def _archive_to_store(self, runtime, messages: list[AnyMessage]) -> None:
          """Save evicted messages as structured dicts to runtime.store."""
          if runtime.store is None:
              return

          thread_id = self._get_thread_id()
          timestamp = datetime.now(UTC).isoformat()
          filtered = self._filter_summary_messages(messages)

          if not filtered:
              return

          # Each batch gets a unique key — store is UPSERT per (namespace, key)
          runtime.store.put(
              namespace=("message_archive", thread_id),
              key=f"batch_{timestamp}",
              value={
                  "messages": [msg.dict() for msg in filtered],  # LOSSLESS
                  "summarized_at": timestamp,
                  "message_count": len(filtered),
              },
          )

      def before_model(self, state, runtime):
          messages = state["messages"]
          self._ensure_message_ids(messages)

          truncated_messages, args_were_truncated = self._truncate_args(messages)
          total_tokens = self.token_counter(truncated_messages)
          should_summarize = self._should_summarize(truncated_messages, total_tokens)

          if args_were_truncated and not should_summarize:
              return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *truncated_messages]}

          if not should_summarize:
              return None

          cutoff_index = self._determine_cutoff_index(truncated_messages)
          if cutoff_index <= 0:
              if args_were_truncated:
                  return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *truncated_messages]}
              return None

          messages_to_summarize, preserved = self._partition_messages(truncated_messages, cutoff_index)

          # ★ Archive to store (LOSSLESS) ★
          self._archive_to_store(runtime, messages_to_summarize)

          # Original offload (LOSSY but human-readable — keep as backup)
          backend = self._get_backend(state, runtime)
          file_path = self._offload_to_backend(backend, messages_to_summarize)

          summary = self._create_summary(messages_to_summarize)
          new_messages = self._build_new_messages_with_path(summary, file_path)

          return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *new_messages, *preserved]}

  Retrieval endpoint:
  async def get_full_history(store, checkpointer, thread_id: str) -> list[dict]:
      """Reconstruct complete message history for a thread."""

      # 1. Get archived batches from PostgresStore
      batches = await store.asearch(
          ("message_archive", thread_id),
          limit=1000
      )
      batches.sort(key=lambda b: b.value["summarized_at"])

      # 2. Get current messages from latest checkpoint
      config = {"configurable": {"thread_id": thread_id}}
      latest = await checkpointer.aget_tuple(config)
      current_msgs = latest.checkpoint["channel_values"]["messages"]

      # 3. Merge: archived (chronological) + current (minus summaries)
      seen_ids: set[str] = set()
      full_history: list[dict] = []

      # Archived batches first (oldest → newest)
      for batch in batches:
          for msg_dict in batch.value["messages"]:
              msg_id = msg_dict.get("id") or msg_dict.get("kwargs", {}).get("id")
              if msg_id and msg_id not in seen_ids:
                  seen_ids.add(msg_id)
                  full_history.append(msg_dict)

      # Then current messages (skip summary HumanMessages)
      for msg in current_msgs:
          if (isinstance(msg, HumanMessage)
              and msg.additional_kwargs.get("lc_source") == "summarization"):
              continue
          if msg.id and msg.id not in seen_ids:
              seen_ids.add(msg.id)
              full_history.append(msg.dict())

      return full_history

  Data flow:

  WRITE (during summarization):
    ┌─────────────┐     ┌────────────────────────┐
    │ before_model │────▶│ runtime.store.put()     │──▶ store table
    │              │     │ ns=("message_archive",  │    (prefix, key) → JSONB
    │              │     │     thread_id)           │    msg.dict() = lossless
    │              │     │ key="batch_{timestamp}"  │
    │              │     └────────────────────────┘
    │              │
    │              │────▶ backend.write()  ──▶ /conversation_history/{tid}.md
    │              │     (existing lossy offload — human-readable backup)
    │              │
    │              │────▶ RemoveMessage(REMOVE_ALL_MESSAGES) + summary + preserved
    └─────────────┘      → committed to checkpoints table


  READ (user requests full history):
    ┌──────────────────┐
    │ GET /full-history │
    │                   │
    │  store.search(    │──▶ [batch_0, batch_1, ...] → archived msg dicts
    │   ("message_      │
    │    archive", tid)) │
    │                   │
    │  checkpointer.    │──▶ [summary, recent_msg1, ..., recent_msgN]
    │   get_tuple(cfg)  │
    │                   │
    │  MERGE + DEDUP    │──▶ [H1, A1, T1, ..., A95]  ← COMPLETE HISTORY
    │  by message ID    │
    └──────────────────┘

  ┌──────────────────┬───────────────────────────────────────────────────────────────┐
  │      Aspect      │                            Detail                             │
  ├──────────────────┼───────────────────────────────────────────────────────────────┤
  │ Data loss        │ None — msg.dict() preserves everything                        │
  ├──────────────────┼───────────────────────────────────────────────────────────────┤
  │ Code changes     │ One middleware subclass + one retrieval function              │
  ├──────────────────┼───────────────────────────────────────────────────────────────┤
  │ Performance      │ Fast — store.search() is a single indexed query               │
  ├──────────────────┼───────────────────────────────────────────────────────────────┤
  │ Storage overhead │ Moderate — messages stored in both store + checkpoint history │
  ├──────────────────┼───────────────────────────────────────────────────────────────┤
  │ Works with       │ Any checkpointer + PostgresStore                              │
  └──────────────────┴───────────────────────────────────────────────────────────────┘

  ---
  Strategy C: Current Backend Offload (Existing — Lossy)

  As-is today. get_buffer_string() → markdown text. Not suitable for UI reconstruction — no message IDs, no metadata, no
  tool_call structure. Only useful as a human-readable log for the agent itself.

  ---
  Strategy D: Direct SQL Query on checkpoint_blobs (Escape Hatch)

  Since PostgresSaver keeps all blob versions, you could query checkpoint_blobs directly:

  SELECT cb.blob, c.checkpoint_id, c.metadata
  FROM checkpoint_blobs cb
  JOIN checkpoints c ON c.thread_id = cb.thread_id
    AND c.checkpoint_ns = cb.checkpoint_ns
  WHERE cb.thread_id = 'my-thread'
    AND cb.channel = 'messages'
  ORDER BY c.checkpoint_id ASC;

  This bypasses the Python API but gives you every version of the messages channel ever saved. Requires deserializing the
  blobs (serde format).

  ---
  Final Recommendation

  ┌────────────────────────────────────────────────────────────────┐
  │  RECOMMENDED: Strategy B (PostgresStore Pre-Archive)           │
  │                                                                │
  │  WHY:                                                          │
  │  ✅ Lossless (msg.dict() preserves all fields)                 │
  │  ✅ Fast retrieval (store.search = single indexed query)       │
  │  ✅ Minimal code change (one subclass, ~30 lines)              │
  │  ✅ StoreBackend already exists in Deep Agents                 │
  │  ✅ Works with any BaseStore impl (Postgres, SQLite, memory)   │
  │  ✅ Cross-thread accessible (store is independent of ckpts)    │
  │  ✅ Keeps existing lossy offload as agent-readable backup      │
  │                                                                │
  │  FALLBACK: Strategy A (checkpoint traversal) requires          │
  │  zero code changes and works retroactively on existing data    │
  │                                                                │
  │  AVOID: Strategy C alone (lossy) or D (fragile SQL coupling)  │
  └────────────────────────────────────────────────────────────────┘
