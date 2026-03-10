"""Unit tests for CheckpointerService and message archive functionality."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, message_to_dict, messages_from_dict
from langgraph.store.memory import InMemoryStore

from deepagents.middleware.summarization import build_archive_namespace
from deepagents.services.checkpointer_service import CheckpointerService


# ---------------------------------------------------------------------------
# build_archive_namespace tests
# ---------------------------------------------------------------------------


class TestBuildArchiveNamespace:
    def test_four_part_thread_id(self) -> None:
        ns = build_archive_namespace("org1_proj1_user1_sess1")
        assert ns == ("message_archive", "org1", "proj1", "user1", "sess1")

    def test_single_part_fallback(self) -> None:
        ns = build_archive_namespace("simple-id")
        assert ns == ("message_archive", "simple-id")

    def test_two_part_fallback(self) -> None:
        ns = build_archive_namespace("ab")
        assert ns == ("message_archive", "ab")

    def test_three_part_fallback(self) -> None:
        ns = build_archive_namespace("a_b_c")
        assert ns == ("message_archive", "a_b_c")

    def test_session_with_underscores(self) -> None:
        """Session ID containing underscores is kept as one part (split maxsplit=3)."""
        ns = build_archive_namespace("org_proj_user_sess_with_underscores")
        assert ns == ("message_archive", "org", "proj", "user", "sess_with_underscores")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_store_with_archives(
    thread_id: str,
    batches: list[tuple[str, list[dict]]],
) -> InMemoryStore:
    """Create an InMemoryStore pre-loaded with archive batches.

    Args:
        thread_id: Thread identifier for namespace construction.
        batches: List of (archived_at_timestamp, messages_as_dicts) tuples.
    """
    store = InMemoryStore()
    namespace = build_archive_namespace(thread_id)
    for timestamp, msg_dicts in batches:
        store.put(
            namespace,
            f"archive_{timestamp}",
            {
                "messages": msg_dicts,
                "archived_at": timestamp,
                "thread_id": thread_id,
            },
        )
    return store


def _msg_to_dict(msg) -> dict:
    """Serialize a message using the same method as the middleware."""
    return message_to_dict(msg)


# ---------------------------------------------------------------------------
# CheckpointerService.get_full_thread_messages tests
# ---------------------------------------------------------------------------


class TestGetFullThreadMessages:
    def test_empty_store_returns_empty(self) -> None:
        store = InMemoryStore()
        checkpointer = MagicMock()
        checkpointer.get_tuple.return_value = None

        svc = CheckpointerService(
            checkpointer=checkpointer,
            store=store,
            org_id="org",
            project_id="proj",
            user_id="user",
            session_id="sess",
        )
        result = svc.get_full_thread_messages()
        assert result == []

    def test_single_batch_round_trip(self) -> None:
        """Messages survive archive -> store -> retrieval."""
        thread_id = "org_proj_user_sess"
        msgs = [
            HumanMessage(content="Hello", id="h1"),
            AIMessage(content="Hi there", id="a1"),
        ]
        msg_dicts = [_msg_to_dict(m) for m in msgs]
        store = _make_store_with_archives(thread_id, [("2026-01-01T00:00:00", msg_dicts)])

        checkpointer = MagicMock()
        checkpointer.get_tuple.return_value = None

        svc = CheckpointerService(
            checkpointer=checkpointer,
            store=store,
            org_id="org",
            project_id="proj",
            user_id="user",
            session_id="sess",
        )
        result = svc.get_full_thread_messages(include_current=False)
        assert len(result) == 2
        assert result[0].content == "Hello"
        assert result[1].content == "Hi there"

    def test_multiple_batches_ordering(self) -> None:
        """Batches are returned in chronological order regardless of insertion order."""
        thread_id = "org_proj_user_sess"
        batch1_msgs = [HumanMessage(content="First", id="h1")]
        batch2_msgs = [HumanMessage(content="Second", id="h2")]

        # Insert batch2 before batch1
        store = _make_store_with_archives(
            thread_id,
            [
                ("2026-01-02T00:00:00", [_msg_to_dict(m) for m in batch2_msgs]),
                ("2026-01-01T00:00:00", [_msg_to_dict(m) for m in batch1_msgs]),
            ],
        )

        checkpointer = MagicMock()
        checkpointer.get_tuple.return_value = None

        svc = CheckpointerService(
            checkpointer=checkpointer,
            store=store,
            org_id="org",
            project_id="proj",
            user_id="user",
            session_id="sess",
        )
        result = svc.get_full_thread_messages(include_current=False)
        assert len(result) == 2
        assert result[0].content == "First"
        assert result[1].content == "Second"

    def test_deduplicates_by_message_id(self) -> None:
        """Same message ID in archive and current state is not duplicated."""
        thread_id = "org_proj_user_sess"
        msg = HumanMessage(content="Hello", id="h1")
        store = _make_store_with_archives(
            thread_id,
            [("2026-01-01T00:00:00", [_msg_to_dict(msg)])],
        )

        # Current state also has the same message
        checkpoint = {"channel_values": {"messages": [msg]}}
        tup = MagicMock()
        tup.checkpoint = checkpoint
        checkpointer = MagicMock()
        checkpointer.get_tuple.return_value = tup

        svc = CheckpointerService(
            checkpointer=checkpointer,
            store=store,
            org_id="org",
            project_id="proj",
            user_id="user",
            session_id="sess",
        )
        result = svc.get_full_thread_messages(include_current=True)
        assert len(result) == 1
        assert result[0].content == "Hello"

    def test_skips_summary_placeholders_from_current(self) -> None:
        """Summary HumanMessages from current state are excluded."""
        thread_id = "org_proj_user_sess"
        store = InMemoryStore()

        summary_msg = HumanMessage(
            content="Summary of conversation",
            id="summary1",
            additional_kwargs={"lc_source": "summarization"},
        )
        real_msg = AIMessage(content="Real response", id="a1")
        checkpoint = {"channel_values": {"messages": [summary_msg, real_msg]}}
        tup = MagicMock()
        tup.checkpoint = checkpoint
        checkpointer = MagicMock()
        checkpointer.get_tuple.return_value = tup

        svc = CheckpointerService(
            checkpointer=checkpointer,
            store=store,
            org_id="org",
            project_id="proj",
            user_id="user",
            session_id="sess",
        )
        result = svc.get_full_thread_messages(include_current=True)
        assert len(result) == 1
        assert result[0].content == "Real response"

    def test_skips_corrupt_archive_items(self) -> None:
        """Corrupt items in the store are skipped with a warning."""
        thread_id = "org_proj_user_sess"
        store = InMemoryStore()
        namespace = build_archive_namespace(thread_id)

        # Valid batch
        valid_msg = HumanMessage(content="Valid", id="h1")
        store.put(
            namespace,
            "archive_2026-01-01T00:00:00",
            {
                "messages": [_msg_to_dict(valid_msg)],
                "archived_at": "2026-01-01T00:00:00",
                "thread_id": thread_id,
            },
        )
        # Corrupt batch (missing messages key)
        store.put(
            namespace,
            "archive_2026-01-02T00:00:00",
            {
                "archived_at": "2026-01-02T00:00:00",
                "thread_id": thread_id,
            },
        )

        checkpointer = MagicMock()
        checkpointer.get_tuple.return_value = None

        svc = CheckpointerService(
            checkpointer=checkpointer,
            store=store,
            org_id="org",
            project_id="proj",
            user_id="user",
            session_id="sess",
        )
        result = svc.get_full_thread_messages(include_current=False)
        # Only the valid batch's message is returned; corrupt one is filtered
        assert len(result) == 1
        assert result[0].content == "Valid"

    def test_raises_without_store(self) -> None:
        checkpointer = MagicMock()
        svc = CheckpointerService(
            checkpointer=checkpointer,
            store=None,
            org_id="org",
            project_id="proj",
            user_id="user",
            session_id="sess",
        )
        with pytest.raises(ValueError, match="store is required"):
            svc.get_full_thread_messages()

    def test_fallback_namespace_for_non_hierarchical_id(self) -> None:
        """Non-4-part thread IDs use fallback namespace and still work."""
        thread_id = "simple-hex-id"
        msg = HumanMessage(content="Hello", id="h1")
        store = _make_store_with_archives(
            thread_id,
            [("2026-01-01T00:00:00", [_msg_to_dict(msg)])],
        )

        checkpointer = MagicMock()
        checkpointer.get_tuple.return_value = None

        svc = CheckpointerService(
            checkpointer=checkpointer,
            store=store,
        )
        result = svc.get_full_thread_messages(thread_id=thread_id, include_current=False)
        assert len(result) == 1
        assert result[0].content == "Hello"


# ---------------------------------------------------------------------------
# SummarizationMiddleware._archive_to_store tests
# ---------------------------------------------------------------------------


class TestArchiveToStore:
    def _make_middleware(self):
        """Create a SummarizationMiddleware with a mock model."""
        from deepagents.middleware.summarization import SummarizationMiddleware

        model = MagicMock()
        model.profile = {"max_input_tokens": 200000}
        backend = MagicMock()
        return SummarizationMiddleware(
            model=model,
            backend=backend,
            trigger=("messages", 100),
            keep=("messages", 10),
        )

    @patch("deepagents.middleware.summarization.get_config")
    def test_silent_skip_when_no_store(self, mock_get_config) -> None:
        mock_get_config.return_value = {"configurable": {"thread_id": "org_proj_user_sess"}}
        mw = self._make_middleware()
        runtime = MagicMock()
        runtime.store = None

        msgs = [HumanMessage(content="Test", id="h1")]
        # Should not raise
        mw._archive_to_store(msgs, runtime)

    @patch("deepagents.middleware.summarization.get_config")
    def test_writes_to_store(self, mock_get_config) -> None:
        mock_get_config.return_value = {"configurable": {"thread_id": "org_proj_user_sess"}}
        mw = self._make_middleware()
        store = MagicMock()
        runtime = MagicMock()
        runtime.store = store

        msgs = [HumanMessage(content="Test", id="h1"), AIMessage(content="Reply", id="a1")]
        mw._archive_to_store(msgs, runtime)

        store.put.assert_called_once()
        call_args = store.put.call_args
        namespace = call_args[0][0]
        key = call_args[0][1]
        value = call_args[0][2]

        assert namespace == ("message_archive", "org", "proj", "user", "sess")
        assert key.startswith("archive_")
        assert "messages" in value
        assert "archived_at" in value
        assert "thread_id" in value
        assert len(value["messages"]) == 2

    @patch("deepagents.middleware.summarization.get_config")
    def test_filters_summary_messages(self, mock_get_config) -> None:
        mock_get_config.return_value = {"configurable": {"thread_id": "org_proj_user_sess"}}
        mw = self._make_middleware()
        store = MagicMock()
        runtime = MagicMock()
        runtime.store = store

        msgs = [
            HumanMessage(
                content="Previous summary",
                id="s1",
                additional_kwargs={"lc_source": "summarization"},
            ),
            HumanMessage(content="Real message", id="h1"),
        ]
        mw._archive_to_store(msgs, runtime)

        value = store.put.call_args[0][2]
        assert len(value["messages"]) == 1  # Summary message filtered out

    @patch("deepagents.middleware.summarization.get_config")
    def test_warns_on_store_failure(self, mock_get_config) -> None:
        mock_get_config.return_value = {"configurable": {"thread_id": "org_proj_user_sess"}}
        mw = self._make_middleware()
        store = MagicMock()
        store.put.side_effect = RuntimeError("Connection lost")
        runtime = MagicMock()
        runtime.store = store

        msgs = [HumanMessage(content="Test", id="h1")]
        # Should not raise
        mw._archive_to_store(msgs, runtime)

    @patch("deepagents.middleware.summarization.get_config")
    def test_skips_when_all_messages_are_summaries(self, mock_get_config) -> None:
        mock_get_config.return_value = {"configurable": {"thread_id": "org_proj_user_sess"}}
        mw = self._make_middleware()
        store = MagicMock()
        runtime = MagicMock()
        runtime.store = store

        msgs = [
            HumanMessage(
                content="Summary only",
                id="s1",
                additional_kwargs={"lc_source": "summarization"},
            ),
        ]
        mw._archive_to_store(msgs, runtime)
        store.put.assert_not_called()


# ---------------------------------------------------------------------------
# Round-trip serialization test
# ---------------------------------------------------------------------------


class TestRoundTripSerialization:
    def test_messages_survive_dict_roundtrip(self) -> None:
        """Various message types survive message_to_dict() -> messages_from_dict()."""
        original_msgs = [
            HumanMessage(content="Hello", id="h1"),
            AIMessage(
                content="I'll help",
                id="a1",
                tool_calls=[{"name": "read_file", "args": {"path": "/foo"}, "id": "tc1"}],
            ),
            ToolMessage(content="file contents", id="t1", tool_call_id="tc1"),
        ]

        serialized = [message_to_dict(msg) for msg in original_msgs]
        restored = messages_from_dict(serialized)

        assert len(restored) == 3
        assert restored[0].content == "Hello"
        assert isinstance(restored[0], HumanMessage)
        assert restored[1].content == "I'll help"
        assert isinstance(restored[1], AIMessage)
        assert len(restored[1].tool_calls) == 1
        assert restored[1].tool_calls[0]["name"] == "read_file"
        assert restored[2].content == "file contents"
        assert isinstance(restored[2], ToolMessage)
        assert restored[2].tool_call_id == "tc1"
