"""CheckpointerService -- A hierarchical interface for LangGraph checkpoint data.

Provides read-only access to conversation threads stored in a LangGraph-compatible
checkpointer (e.g., MongoDBSaver) following a four-level hierarchy:

    Organization -> Project -> User -> Thread

Each thread_id follows the convention:
    {org_id}_{project_id}_{user_id}_{session_id}

Additionally provides full message history retrieval by reading lossless message
archives written by :class:`~deepagents.middleware.summarization.SummarizationMiddleware`
into LangGraph's BaseStore.

Usage:
    from deepagents.services.checkpointer_service import CheckpointerService
    from langgraph.checkpoint.mongodb import MongoDBSaver
    from pymongo import MongoClient

    client = MongoClient("mongodb://admin:password123@localhost:9099", directConnection=True)
    checkpointer = MongoDBSaver(client)

    service = CheckpointerService(
        checkpointer=checkpointer,
        org_id="alhai",
        project_id="construction_docs",
        user_id="john_doe",
    )

    # Get all threads for this user
    threads = service.get_user_threads()

    # Get messages from a specific thread
    messages = service.get_thread_messages(thread_id="alhai_construction_docs_john_doe_session001")

    # Get FULL history including archived (summarized-away) messages
    full_history = service.get_full_thread_messages(
        thread_id="alhai_construction_docs_john_doe_session001"
    )
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any, Dict, List, Literal, Optional, Sequence, Union

from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    messages_from_dict,
)
from langchain_core.messages.utils import count_tokens_approximately

from deepagents.middleware.summarization import build_archive_namespace

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
MessageType = Literal["human", "ai", "tool", "system", "all"]

ThreadSummary = Dict[str, Any]
"""A lightweight dict describing a conversation thread."""

TokenReport = Dict[str, int]
"""Token-count breakdown for a thread."""

_ARCHIVE_PAGE_SIZE = 100


# ---------------------------------------------------------------------------
# CheckpointerService
# ---------------------------------------------------------------------------
class CheckpointerService:
    """Read-only, hierarchical access to LangGraph checkpoint data.

    Parameters
    ----------
    checkpointer : object
        A LangGraph-compatible checkpointer instance (e.g., ``MongoDBSaver``).
        Must expose ``.list()`` and ``.get_tuple()`` methods.
    org_id : str, optional
        Organization identifier. Required for org/project/user scoped queries.
    project_id : str, optional
        Project identifier within the organization.
    user_id : str, optional
        User identifier within the project.
    session_id : str, optional
        Session identifier. When provided with org/project/user, used to
        build a default ``thread_id``.
    store : object, optional
        A LangGraph-compatible BaseStore instance. Required for
        ``get_full_thread_messages()`` which reads archived message batches.
    default_limit : int
        Maximum number of checkpoints to scan per query (default 500).
    """

    # -- Constructor -------------------------------------------------------

    def __init__(
        self,
        checkpointer: Any,
        org_id: Optional[str] = None,
        project_id: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        store: Any = None,
        default_limit: int = 500,
    ) -> None:
        self._checkpointer = checkpointer
        self.org_id = org_id
        self.project_id = project_id
        self.user_id = user_id
        self.session_id = session_id
        self._store = store
        self.default_limit = default_limit

    # -- Thread ID helpers -------------------------------------------------

    @staticmethod
    def build_thread_id(
        org_id: str,
        project_id: str,
        user_id: str,
        session_id: str,
    ) -> str:
        """Build a canonical thread ID from hierarchy components.

        Returns
        -------
        str
            ``"{org_id}_{project_id}_{user_id}_{session_id}"``
        """
        return f"{org_id}_{project_id}_{user_id}_{session_id}"

    @staticmethod
    def parse_thread_id(thread_id: str) -> Dict[str, str]:
        """Parse a canonical thread ID into its hierarchy components.

        Returns
        -------
        dict
            Keys: ``org_id``, ``project_id``, ``user_id``, ``session_id``.

        Raises
        ------
        ValueError
            If the thread_id does not contain exactly 4 underscore-separated parts.
        """
        parts = thread_id.split("_", 3)
        if len(parts) != 4:
            raise ValueError(
                f"Expected thread_id with 4 parts (org_project_user_session), "
                f"got {len(parts)} from '{thread_id}'"
            )
        return {
            "org_id": parts[0],
            "project_id": parts[1],
            "user_id": parts[2],
            "session_id": parts[3],
        }

    @property
    def default_thread_id(self) -> Optional[str]:
        """The thread ID built from constructor args, or ``None`` if incomplete."""
        if all([self.org_id, self.project_id, self.user_id, self.session_id]):
            return self.build_thread_id(
                self.org_id, self.project_id, self.user_id, self.session_id
            )
        return None

    # -- LangChain config builder ------------------------------------------

    def build_config(
        self,
        thread_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Build a LangGraph-compatible config dict.

        Parameters
        ----------
        thread_id : str, optional
            If not provided, falls back to ``self.default_thread_id``.

        Returns
        -------
        dict
            ``{"configurable": {"thread_id": ...}, "metadata": {...}}``
        """
        tid = thread_id or self.default_thread_id
        if tid is None:
            raise ValueError(
                "No thread_id provided and cannot build a default one "
                "(org_id, project_id, user_id, and session_id are all required)."
            )
        return {
            "configurable": {"thread_id": tid},
            "metadata": {
                "org_id": self.org_id or "",
                "project_id": self.project_id or "",
                "person_id": self.user_id or "",
            },
        }

    # ======================================================================
    # HIERARCHY QUERIES
    # ======================================================================

    def _list_threads(
        self,
        metadata_filter: Dict[str, str],
        limit: Optional[int] = None,
    ) -> List[ThreadSummary]:
        """Internal: scan checkpoints and deduplicate by thread_id."""
        seen: Dict[str, ThreadSummary] = {}
        for cp in self._checkpointer.list(
            config=None,
            filter=metadata_filter,
            limit=limit or self.default_limit,
        ):
            tid = cp.config["configurable"]["thread_id"]
            if tid not in seen:
                seen[tid] = {
                    "thread_id": tid,
                    "checkpoint_id": cp.config["configurable"].get("checkpoint_id"),
                    "metadata": cp.metadata,
                    "step": cp.metadata.get("step"),
                    "source": cp.metadata.get("source"),
                }
        return list(seen.values())

    # -- Organization level ------------------------------------------------

    def get_org_threads(
        self,
        org_id: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[ThreadSummary]:
        """Get all conversation threads across an entire organization."""
        oid = org_id or self.org_id
        if not oid:
            raise ValueError("org_id is required.")
        return self._list_threads({"org_id": oid}, limit)

    # -- Project level -----------------------------------------------------

    def get_project_threads(
        self,
        org_id: Optional[str] = None,
        project_id: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[ThreadSummary]:
        """Get all conversation threads within a specific project."""
        oid = org_id or self.org_id
        pid = project_id or self.project_id
        if not oid or not pid:
            raise ValueError("Both org_id and project_id are required.")
        return self._list_threads({"org_id": oid, "project_id": pid}, limit)

    # -- User level --------------------------------------------------------

    def get_user_threads(
        self,
        org_id: Optional[str] = None,
        project_id: Optional[str] = None,
        user_id: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[ThreadSummary]:
        """Get all conversation threads for a specific user."""
        oid = org_id or self.org_id
        pid = project_id or self.project_id
        uid = user_id or self.user_id
        if not all([oid, pid, uid]):
            raise ValueError("org_id, project_id, and user_id are all required.")
        return self._list_threads(
            {"org_id": oid, "project_id": pid, "person_id": uid}, limit
        )

    # ======================================================================
    # THREAD-LEVEL ACCESS
    # ======================================================================

    def get_checkpoint(
        self, thread_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Retrieve the latest checkpoint for a thread."""
        tid = thread_id or self.default_thread_id
        if not tid:
            raise ValueError("thread_id is required.")
        config = {"configurable": {"thread_id": tid}}
        tup = self._checkpointer.get_tuple(config)
        if tup is None:
            return None
        return tup.checkpoint

    def get_thread_messages(
        self, thread_id: Optional[str] = None
    ) -> List[BaseMessage]:
        """Get current messages in a thread (post-summarization state only)."""
        cp = self.get_checkpoint(thread_id)
        if cp is None:
            return []
        return cp.get("channel_values", {}).get("messages", [])

    async def aget_checkpoint(
        self, thread_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Retrieve the latest checkpoint for a thread (async)."""
        tid = thread_id or self.default_thread_id
        if not tid:
            raise ValueError("thread_id is required.")
        config = {"configurable": {"thread_id": tid}}
        tup = await self._checkpointer.aget_tuple(config)
        if tup is None:
            return None
        return tup.checkpoint

    async def aget_thread_messages(
        self, thread_id: Optional[str] = None
    ) -> List[BaseMessage]:
        """Get current messages in a thread (async)."""
        cp = await self.aget_checkpoint(thread_id)
        if cp is None:
            return []
        return cp.get("channel_values", {}).get("messages", [])

    # ======================================================================
    # FULL MESSAGE HISTORY (Store Archive Retrieval)
    # ======================================================================

    def get_full_thread_messages(
        self,
        thread_id: Optional[str] = None,
        include_current: bool = True,
    ) -> List[AnyMessage]:
        """Retrieve the complete message history including archived messages.

        Reads lossless message archives from the store (written by
        ``SummarizationMiddleware._archive_to_store``) and optionally appends
        the current thread messages from the latest checkpoint.

        The result is a chronologically ordered list of all messages that have
        ever been part of this thread, including those that were summarized away.

        Parameters
        ----------
        thread_id : str, optional
            Falls back to ``self.default_thread_id``.
        include_current : bool
            If True (default), appends current checkpoint messages after
            archived messages, skipping summary placeholder messages and
            deduplicating by message ID.

        Returns
        -------
        list[AnyMessage]
            Complete message history in chronological order.

        Raises
        ------
        ValueError
            If no store was provided at construction time.
        """
        if self._store is None:
            raise ValueError(
                "A store is required for get_full_thread_messages(). "
                "Pass store=<BaseStore> when constructing CheckpointerService."
            )

        tid = thread_id or self.default_thread_id
        if not tid:
            raise ValueError("thread_id is required.")

        # 1. Read archived batches from the store
        namespace = build_archive_namespace(tid)
        items = self._search_archive_paginated(namespace)

        # Sort by archived_at (ISO timestamps sort lexicographically for UTC)
        valid_items = [
            item for item in items
            if isinstance(item.value, dict)
            and "messages" in item.value
            and "archived_at" in item.value
        ]
        valid_items.sort(key=lambda item: item.value["archived_at"])

        # 2. Deserialize archived messages
        seen_ids: set[str] = set()
        all_messages: List[AnyMessage] = []

        for item in valid_items:
            try:
                batch = messages_from_dict(item.value["messages"])
                for msg in batch:
                    msg_id = getattr(msg, "id", None)
                    if msg_id and msg_id not in seen_ids:
                        seen_ids.add(msg_id)
                        all_messages.append(msg)
                    elif not msg_id:
                        all_messages.append(msg)
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "Failed to deserialize archive batch %s: %s: %s",
                    getattr(item, "key", "unknown"),
                    type(e).__name__,
                    e,
                )

        # 3. Optionally append current messages (skip summary placeholders, dedup)
        if include_current:
            current_msgs = self.get_thread_messages(tid)
            for msg in current_msgs:
                # Skip summary placeholder messages
                if (
                    isinstance(msg, HumanMessage)
                    and msg.additional_kwargs.get("lc_source") == "summarization"
                ):
                    continue
                msg_id = getattr(msg, "id", None)
                if msg_id and msg_id not in seen_ids:
                    seen_ids.add(msg_id)
                    all_messages.append(msg)
                elif not msg_id:
                    all_messages.append(msg)

        return all_messages

    async def aget_full_thread_messages(
        self,
        thread_id: Optional[str] = None,
        include_current: bool = True,
    ) -> List[AnyMessage]:
        """Async version of :meth:`get_full_thread_messages`."""
        if self._store is None:
            raise ValueError(
                "A store is required for aget_full_thread_messages(). "
                "Pass store=<BaseStore> when constructing CheckpointerService."
            )

        tid = thread_id or self.default_thread_id
        if not tid:
            raise ValueError("thread_id is required.")

        namespace = build_archive_namespace(tid)
        items = await self._asearch_archive_paginated(namespace)

        valid_items = [
            item for item in items
            if isinstance(item.value, dict)
            and "messages" in item.value
            and "archived_at" in item.value
        ]
        valid_items.sort(key=lambda item: item.value["archived_at"])

        seen_ids: set[str] = set()
        all_messages: List[AnyMessage] = []

        for item in valid_items:
            try:
                batch = messages_from_dict(item.value["messages"])
                for msg in batch:
                    msg_id = getattr(msg, "id", None)
                    if msg_id and msg_id not in seen_ids:
                        seen_ids.add(msg_id)
                        all_messages.append(msg)
                    elif not msg_id:
                        all_messages.append(msg)
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "Failed to deserialize archive batch %s: %s: %s",
                    getattr(item, "key", "unknown"),
                    type(e).__name__,
                    e,
                )

        if include_current:
            current_msgs = await self.aget_thread_messages(tid)
            for msg in current_msgs:
                if (
                    isinstance(msg, HumanMessage)
                    and msg.additional_kwargs.get("lc_source") == "summarization"
                ):
                    continue
                msg_id = getattr(msg, "id", None)
                if msg_id and msg_id not in seen_ids:
                    seen_ids.add(msg_id)
                    all_messages.append(msg)
                elif not msg_id:
                    all_messages.append(msg)

        return all_messages

    def _search_archive_paginated(self, namespace: tuple[str, ...]) -> list:
        """Paginated synchronous search for all archive items."""
        all_items: list = []
        offset = 0
        while True:
            page = self._store.search(
                namespace,
                limit=_ARCHIVE_PAGE_SIZE,
                offset=offset,
            )
            if not page:
                break
            all_items.extend(page)
            if len(page) < _ARCHIVE_PAGE_SIZE:
                break
            offset += _ARCHIVE_PAGE_SIZE
        return all_items

    async def _asearch_archive_paginated(self, namespace: tuple[str, ...]) -> list:
        """Paginated async search for all archive items."""
        all_items: list = []
        offset = 0
        while True:
            page = await self._store.asearch(
                namespace,
                limit=_ARCHIVE_PAGE_SIZE,
                offset=offset,
            )
            if not page:
                break
            all_items.extend(page)
            if len(page) < _ARCHIVE_PAGE_SIZE:
                break
            offset += _ARCHIVE_PAGE_SIZE
        return all_items

    # ======================================================================
    # MESSAGE FILTERING
    # ======================================================================

    @staticmethod
    def _msg_type_match(msg: BaseMessage, msg_type: MessageType) -> bool:
        """Check if a message matches the requested type."""
        if msg_type == "all":
            return True
        type_map = {
            "human": HumanMessage,
            "ai": AIMessage,
            "tool": ToolMessage,
            "system": SystemMessage,
        }
        return isinstance(msg, type_map.get(msg_type, BaseMessage))

    def filter_messages(
        self,
        thread_id: Optional[str] = None,
        msg_type: MessageType = "all",
        contains: Optional[str] = None,
        after: Optional[datetime] = None,
        before: Optional[datetime] = None,
        has_tool_calls: Optional[bool] = None,
    ) -> List[BaseMessage]:
        """Filter messages from a thread by type, content, or time range."""
        messages = self.get_thread_messages(thread_id)
        result: List[BaseMessage] = []

        for msg in messages:
            if not self._msg_type_match(msg, msg_type):
                continue
            if contains is not None:
                text = msg.content if isinstance(msg.content, str) else str(msg.content)
                if contains.lower() not in text.lower():
                    continue
            ts_raw = getattr(msg, "additional_kwargs", {}).get("timestamp")
            if ts_raw is not None:
                try:
                    ts = datetime.fromisoformat(str(ts_raw))
                    if after and ts <= after:
                        continue
                    if before and ts >= before:
                        continue
                except (ValueError, TypeError):
                    pass
            if has_tool_calls is not None:
                if isinstance(msg, AIMessage):
                    has_tc = bool(getattr(msg, "tool_calls", None))
                    if has_tool_calls != has_tc:
                        continue
                elif has_tool_calls:
                    continue
            result.append(msg)

        return result

    def get_tool_calls(
        self, thread_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Extract all tool calls from AI messages in a thread."""
        messages = self.get_thread_messages(thread_id)
        tool_calls: List[Dict[str, Any]] = []

        for idx, msg in enumerate(messages):
            if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
                for tc in msg.tool_calls:
                    tool_calls.append({
                        "tool_name": tc.get("name", ""),
                        "tool_args": tc.get("args", {}),
                        "tool_call_id": tc.get("id", ""),
                        "message_index": idx,
                    })
        return tool_calls

    # ======================================================================
    # TOKEN COUNTING
    # ======================================================================

    def count_tokens(
        self, thread_id: Optional[str] = None
    ) -> TokenReport:
        """Count tokens spent in a conversation thread."""
        messages = self.get_thread_messages(thread_id)

        total_input = 0
        total_output = 0
        total_tokens = 0
        ai_with_usage = 0

        for msg in messages:
            if isinstance(msg, AIMessage):
                usage = getattr(msg, "usage_metadata", None)
                if usage:
                    total_input += usage.get("input_tokens", 0)
                    total_output += usage.get("output_tokens", 0)
                    total_tokens += usage.get("total_tokens", 0)
                    ai_with_usage += 1

        approx = count_tokens_approximately(messages) if messages else 0

        return {
            "input_tokens_exact": total_input,
            "output_tokens_exact": total_output,
            "total_tokens_exact": total_tokens,
            "approx_prompt_tokens": approx,
            "message_count": len(messages),
            "ai_messages_with_usage": ai_with_usage,
        }

    # ======================================================================
    # EXPORT
    # ======================================================================

    @staticmethod
    def _message_to_dict(msg: BaseMessage, index: int) -> Dict[str, Any]:
        """Convert a single LangChain message to a JSON-serializable dict."""
        base: Dict[str, Any] = {
            "index": index,
            "type": msg.type,
            "content": msg.content,
            "id": getattr(msg, "id", None),
        }

        if isinstance(msg, AIMessage):
            base["tool_calls"] = getattr(msg, "tool_calls", []) or []
            usage = getattr(msg, "usage_metadata", None)
            base["usage_metadata"] = dict(usage) if usage else None

        if isinstance(msg, ToolMessage):
            base["tool_call_id"] = getattr(msg, "tool_call_id", None)
            base["name"] = getattr(msg, "name", None)

        extra = getattr(msg, "additional_kwargs", {})
        if extra:
            base["additional_kwargs"] = extra

        return base

    def export_thread(
        self,
        thread_id: Optional[str] = None,
        include_metadata: bool = True,
    ) -> Dict[str, Any]:
        """Export a full conversation thread as a JSON-serializable dict."""
        tid = thread_id or self.default_thread_id
        messages = self.get_thread_messages(tid)
        token_report = self.count_tokens(tid)

        export: Dict[str, Any] = {
            "thread_id": tid,
            "message_count": len(messages),
            "messages": [
                self._message_to_dict(msg, i) for i, msg in enumerate(messages)
            ],
            "token_report": token_report,
            "exported_at": datetime.now(UTC).isoformat(),
        }

        if include_metadata:
            cp = self.get_checkpoint(tid)
            if cp:
                export["metadata"] = {
                    k: v
                    for k, v in cp.items()
                    if k != "channel_values"
                }

        return export

    def export_thread_json(
        self,
        thread_id: Optional[str] = None,
        filepath: Optional[str] = None,
        indent: int = 2,
    ) -> str:
        """Export a thread to a JSON string, optionally writing to a file."""
        data = self.export_thread(thread_id)
        json_str = json.dumps(data, indent=indent, default=str)

        if filepath:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(json_str)

        return json_str

    # ======================================================================
    # SUMMARY / REPR
    # ======================================================================

    def thread_summary(
        self, thread_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get a concise summary of a conversation thread."""
        tid = thread_id or self.default_thread_id
        messages = self.get_thread_messages(tid)

        human_count = sum(1 for m in messages if isinstance(m, HumanMessage))
        ai_count = sum(1 for m in messages if isinstance(m, AIMessage))
        tool_count = sum(1 for m in messages if isinstance(m, ToolMessage))
        tc_count = sum(
            len(getattr(m, "tool_calls", []) or [])
            for m in messages
            if isinstance(m, AIMessage)
        )

        def preview(msg: BaseMessage, max_len: int = 100) -> str:
            text = msg.content if isinstance(msg.content, str) else str(msg.content)
            return text[:max_len] + ("..." if len(text) > max_len else "")

        return {
            "thread_id": tid,
            "message_count": len(messages),
            "human_messages": human_count,
            "ai_messages": ai_count,
            "tool_messages": tool_count,
            "tool_calls_count": tc_count,
            "token_report": self.count_tokens(tid),
            "first_message_preview": preview(messages[0]) if messages else None,
            "last_message_preview": preview(messages[-1]) if messages else None,
        }

    def __repr__(self) -> str:
        parts = [f"CheckpointerService(org={self.org_id!r}"]
        if self.project_id:
            parts.append(f"project={self.project_id!r}")
        if self.user_id:
            parts.append(f"user={self.user_id!r}")
        if self.session_id:
            parts.append(f"session={self.session_id!r}")
        return ", ".join(parts) + ")"


# ---------------------------------------------------------------------------
# Singleton helpers
# ---------------------------------------------------------------------------
_persistence_service: Optional[CheckpointerService] = None


def initialize_persistence_service(
    checkpointer: Any,
    org_id: Optional[str] = None,
    project_id: Optional[str] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    store: Any = None,
    default_limit: int = 500,
) -> CheckpointerService:
    """Initialize the CheckpointerService singleton with required dependencies.

    Must be called once before ``get_persistence_service()``.
    """
    global _persistence_service
    _persistence_service = CheckpointerService(
        checkpointer=checkpointer,
        org_id=org_id,
        project_id=project_id,
        user_id=user_id,
        session_id=session_id,
        store=store,
        default_limit=default_limit,
    )
    return _persistence_service


def get_persistence_service() -> CheckpointerService:
    """Get the initialized CheckpointerService singleton.

    Raises
    ------
    RuntimeError
        If ``initialize_persistence_service()`` has not been called yet.
    """
    if _persistence_service is None:
        raise RuntimeError(
            "CheckpointerService not initialized. "
            "Call initialize_persistence_service(checkpointer=...) first."
        )
    return _persistence_service
