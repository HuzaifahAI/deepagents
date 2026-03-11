"""Test script: StreamWriter propagation from compiled LangGraph subagent workflows.

Creates a Deep Agent with two compiled LangGraph subagents (history_writer, bio_researcher),
each using StreamWriter to emit status dicts. Verifies that custom events flow to the
parent stream when using stream_mode="custom" with subgraphs=True.

Usage:
    # Main test — query that invokes both subagents, pretty streaming output
    conda run -n deep_agents_test python test_streaming_subagents.py

    # Raw debug mode — original single-subagent tests with raw event output
    conda run -n deep_agents_test python test_streaming_subagents.py --debug
"""

import os
import sys
import time
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import StreamWriter
from typing_extensions import TypedDict

from deepagents.graph import create_deep_agent
from deepagents.middleware.subagents import CompiledSubAgent
from langchain_core.callbacks import get_usage_metadata_callback
# ---------------------------------------------------------------------------
# LLM config
# ---------------------------------------------------------------------------
LLM_API_BASE_URL = os.environ.get("LLM_API_BASE_URL", "")
LLM_MODEL_NAME = os.environ.get("LLM_MODEL_NAME", "qwen3-235b")
LLM_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0.1"))
LLM_MAX_ITERATIONS = int(os.environ.get("LLM_MAX_ITERATIONS", "8"))

# Shared LLM instance — used by both subagent nodes and the parent agent
LLM = ChatOpenAI(
    base_url=LLM_API_BASE_URL,
    model=LLM_MODEL_NAME,
    temperature=LLM_TEMPERATURE,
    api_key="not-needed",
)

# ---------------------------------------------------------------------------
# Terminal formatting
# ---------------------------------------------------------------------------
BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[32m"
CYAN = "\033[36m"
YELLOW = "\033[33m"
MAGENTA = "\033[35m"
RESET = "\033[0m"
CHECK = f"{GREEN}\u2713{RESET}"
ARROW = f"{CYAN}\u25b6{RESET}"
DOT = f"{DIM}\u2502{RESET}"

# Map node names to friendly labels and colors
NODE_LABELS: dict[str, tuple[str, str]] = {
    "history_research": ("History Research", CYAN),
    "history_drafting": ("History Drafting", CYAN),
    "bio_claim_gathering": ("Bio: Claim Gathering", MAGENTA),
    "evaluation_of_claims": ("Bio: Claim Evaluation", MAGENTA),
    "bio_synthesis": ("Bio: Synthesis", MAGENTA),
}

STATUS_ICONS: dict[str, str] = {
    "started": f"{YELLOW}\u25cb{RESET}",
    "analyzing": f"{YELLOW}\u2026{RESET}",
    "processing": f"{YELLOW}\u2026{RESET}",
    "writing": f"{YELLOW}\u270e{RESET}",
    "verified": f"{GREEN}\u2713{RESET}",
    "completed": f"{GREEN}\u25cf{RESET}",
}


def format_event(event: dict[str, Any]) -> str:
    """Format a StreamWriter event dict into a readable terminal line."""
    node = event.get("node", "unknown")
    status = event.get("status", "")
    detail = event.get("detail", "")
    claim = event.get("claim", "")
    confidence = event.get("confidence")

    label, color = NODE_LABELS.get(node, (node, DIM))
    icon = STATUS_ICONS.get(status, " ")

    parts = [f"  {icon} {color}{BOLD}{label}{RESET}"]

    if claim and status == "verified" and confidence is not None:
        parts.append(f'{DIM}verified{RESET} "{claim}" {GREEN}({confidence:.0%}){RESET}')
    elif claim:
        parts.append(f'{DIM}{status}{RESET} "{claim}"')
    elif detail and status not in ("started", "completed"):
        parts.append(f"{DIM}{detail}{RESET}")
    elif status == "started":
        parts.append(f"{DIM}starting...{RESET}")
    elif status == "completed":
        parts.append(f"{GREEN}done{RESET}")

    return " \u2014 ".join(parts)


# ---------------------------------------------------------------------------
# Subagent state schema (messages-based, required by CompiledSubAgent)
# ---------------------------------------------------------------------------
class SubAgentState(TypedDict):
    messages: list[BaseMessage]


# ---------------------------------------------------------------------------
# Subagent 1: History Writer
# ---------------------------------------------------------------------------
def history_research_node(state: SubAgentState, writer: StreamWriter) -> SubAgentState:
    """Research historical context using the LLM and emit progress via StreamWriter."""
    writer({"node": "history_research", "status": "started", "detail": "Gathering historical sources"})

    topic = state["messages"][-1].content if state["messages"] else "unknown topic"

    writer({"node": "history_research", "status": "analyzing", "detail": f"Analyzing historical context for: {topic[:80]}"})

    response = LLM.invoke([
        HumanMessage(
            content=(
                "Research the following historical topic and list the key events, dates, "
                "and figures involved. Be concise and factual.\n\n"
                f"Topic: {topic}"
            )
        )
    ])

    writer({"node": "history_research", "status": "completed", "detail": "Historical research complete"})
    return {"messages": [response]}


def history_drafting_node(state: SubAgentState, writer: StreamWriter) -> SubAgentState:
    """Draft a historical narrative from research output using the LLM."""
    writer({"node": "history_drafting", "status": "started", "detail": "Drafting historical narrative"})

    research = state["messages"][-1].content

    writer({"node": "history_drafting", "status": "writing", "detail": "Composing narrative structure"})

    response = LLM.invoke([
        HumanMessage(
            content=(
                "Based on this research, write a well-structured historical narrative "
                "with proper chronology. Keep it to 2-3 paragraphs.\n\n"
                f"Research:\n{research}"
            )
        )
    ])

    writer({"node": "history_drafting", "status": "completed", "detail": "Draft complete"})
    return {"messages": [response]}


def build_history_writer() -> StateGraph:
    """Build the history writer subagent workflow."""
    graph = StateGraph(SubAgentState)
    graph.add_node("research", history_research_node)
    graph.add_node("draft", history_drafting_node)
    graph.add_edge(START, "research")
    graph.add_edge("research", "draft")
    graph.add_edge("draft", END)
    return graph


# ---------------------------------------------------------------------------
# Subagent 2: Bio Researcher
# ---------------------------------------------------------------------------
def bio_claim_gathering_node(state: SubAgentState, writer: StreamWriter) -> SubAgentState:
    """Gather biographical claims using the LLM and stream each discovered claim."""
    topic = state["messages"][-1].content if state["messages"] else "unknown person"

    writer({"node": "bio_claim_gathering", "status": "started", "detail": f"Collecting claims about: {topic[:80]}"})

    response = LLM.invoke([
        HumanMessage(
            content=(
                "List 5-7 key biographical claims about the following person. "
                "Format each as a numbered claim (e.g., '1. Born on ...').\n\n"
                f"Person: {topic}"
            )
        )
    ])

    # Stream progress for each claim found
    claims = [line.strip() for line in response.content.split("\n") if line.strip() and line.strip()[0].isdigit()]
    for claim in claims:
        writer({"node": "bio_claim_gathering", "status": "processing", "claim": claim[:60]})

    writer({"node": "bio_claim_gathering", "status": "completed", "detail": f"{len(claims)} claims collected"})
    return {"messages": [response]}


def bio_evaluation_node(state: SubAgentState, writer: StreamWriter) -> SubAgentState:
    """Evaluate biographical claims for accuracy using the LLM."""
    claims_text = state["messages"][-1].content

    writer({"node": "evaluation_of_claims", "status": "started", "detail": "Evaluating claim validity"})

    response = LLM.invoke([
        HumanMessage(
            content=(
                "Evaluate each of these biographical claims for factual accuracy. "
                "For each claim, rate confidence (0-100%) and note if correct/incorrect.\n\n"
                f"{claims_text}"
            )
        )
    ])

    writer({"node": "evaluation_of_claims", "status": "completed", "detail": "All claims evaluated"})
    return {"messages": [response]}


def bio_synthesis_node(state: SubAgentState, writer: StreamWriter) -> SubAgentState:
    """Synthesize verified claims into a cohesive biography using the LLM."""
    evaluated_claims = state["messages"][-1].content

    writer({"node": "bio_synthesis", "status": "started", "detail": "Synthesizing verified claims into biography"})

    response = LLM.invoke([
        HumanMessage(
            content=(
                "Based on these evaluated claims, write a concise, well-structured "
                "biography (2-3 paragraphs).\n\n"
                f"{evaluated_claims}"
            )
        )
    ])

    writer({"node": "bio_synthesis", "status": "completed", "detail": "Biography synthesized"})
    return {"messages": [response]}


def build_bio_researcher() -> StateGraph:
    """Build the bio researcher subagent workflow."""
    graph = StateGraph(SubAgentState)
    graph.add_node("gather_claims", bio_claim_gathering_node)
    graph.add_node("evaluate_claims", bio_evaluation_node)
    graph.add_node("synthesize", bio_synthesis_node)
    graph.add_edge(START, "gather_claims")
    graph.add_edge("gather_claims", "evaluate_claims")
    graph.add_edge("evaluate_claims", "synthesize")
    graph.add_edge("synthesize", END)
    return graph


# ---------------------------------------------------------------------------
# Shared agent builder
# ---------------------------------------------------------------------------
def build_agent() -> Any:
    """Build the deep agent with both compiled subagent workflows."""
    history_writer = build_history_writer().compile()
    bio_researcher = build_bio_researcher().compile()

    llm = LLM

    return create_deep_agent(
        model=llm,
        checkpointer=InMemorySaver(),
        subagents=[
            CompiledSubAgent(
                name="history_writer",
                description="A subagent that researches and writes historical narratives. Use this when the user asks about historical topics, events, or timelines.",
                runnable=history_writer,
            ),
            CompiledSubAgent(
                name="bio_researcher",
                description="A subagent that researches biographical information, gathers claims, verifies them, and synthesizes biographies. Use this when the user asks about a person's biography or life history.",
                runnable=bio_researcher,
            ),
        ],
    )


# ---------------------------------------------------------------------------
# Main test: both subagents, pretty streaming
# ---------------------------------------------------------------------------
def test_both_subagents_streaming() -> None:
    """Invoke a query that triggers both subagents with pretty streaming output."""
    agent = build_agent()

    query = (
        "I need two things: "
        "First, use the history_writer to write about the history of radioactivity discovery. "
        "Second, use the bio_researcher to research Marie Curie's biography."
    )

    print(f"\n{BOLD}Query:{RESET} {query}\n")
    print(f"{DIM}{'─' * 70}{RESET}")
    print(f"{BOLD}Streaming events:{RESET}\n")

    custom_events: list[dict[str, Any]] = []
    final_answer = ""
    current_subagent = ""
    with get_usage_metadata_callback() as cb:
        for namespace, data in agent.stream(
            {"messages": [HumanMessage(content=query)]},
            config={"configurable": {"thread_id": "test_both_subagents"}},
            stream_mode="custom",
            subgraphs=True,
        ):
            if not isinstance(data, dict) or "node" not in data:
                continue

            custom_events.append(data)
            node = data.get("node", "")

            # Print a separator when switching between subagent workflows
            label, color = NODE_LABELS.get(node, (node, DIM))
            if node in ("history_research", "bio_claim_gathering") and node != current_subagent:
                if current_subagent:
                    print()
                subagent_name = "History Writer" if "history" in node else "Bio Researcher"
                print(f"  {ARROW} {BOLD}{subagent_name}{RESET}")
                current_subagent = node

            line = format_event(data)
            print(line, flush=True)


        print(f"USAGE: {cb.usage_metadata}")
    # Collect final answer from state
    # result = agent.invoke(
    #     {"messages": [HumanMessage(content="Summarize what you found")]},
    #     config={"configurable": {"thread_id": "test_both_subagents"}},
    # )
    # final_answer = result["messages"][-1].content
    state = agent.get_state({"configurable": {"thread_id": "test_both_subagents"}})
    final_answer = state.values["messages"][-1].content



    print(f"\n{DIM}{'─' * 70}{RESET}")
    print(f"\n{BOLD}Streaming summary:{RESET}")
    print(f"  Events received: {len(custom_events)}")

    # Count per subagent
    history_events = [e for e in custom_events if e.get("node", "").startswith("history")]
    bio_events = [e for e in custom_events if not e.get("node", "").startswith("history")]
    print(f"  History Writer events: {history_events and len(history_events) or 0}")
    print(f"  Bio Researcher events: {bio_events and len(bio_events) or 0}")

    if history_events and bio_events:
        print(f"  {CHECK} Both subagents invoked successfully")
    elif history_events:
        print(f"  {YELLOW}Only History Writer was invoked{RESET}")
    elif bio_events:
        print(f"  {YELLOW}Only Bio Researcher was invoked{RESET}")
    else:
        print(f"  {YELLOW}No subagent events received{RESET}")

    print(f"\n{DIM}{'─' * 70}{RESET}")
    print(f"\n{BOLD}Final Answer:{RESET}\n")
    print(final_answer)
    print()


# ---------------------------------------------------------------------------
# Debug test: original raw-output tests (--debug flag)
# ---------------------------------------------------------------------------
def test_debug() -> None:
    """Original test with raw event dicts — for debugging."""
    agent = build_agent()

    print("=" * 70)
    print("DEBUG: StreamWriter propagation — raw event output")
    print("=" * 70)

    # ---- Test 1: Streaming mode with custom events ----
    print("\n--- Test 1: Streaming with stream_mode='custom', subgraphs=True ---\n")

    custom_events: list[dict[str, Any]] = []
    all_stream_items: list[Any] = []

    for namespace, data in agent.stream(
        {"messages": [HumanMessage(content="Write a biography of Marie Curie including her historical significance")]},
        config={"configurable": {"thread_id": "test_streaming_debug"}},
        stream_mode="custom",
        subgraphs=True,
    ):
        all_stream_items.append({"namespace": namespace, "data": data})
        if isinstance(data, dict) and "node" in data:
            custom_events.append(data)
            print(f"  [STREAM] namespace={namespace} | {data}")

    print(f"\n  Total stream items: {len(all_stream_items)}")
    print(f"  Custom events with 'node' key: {len(custom_events)}")

    if custom_events:
        print("\n  Custom events received:")
        for evt in custom_events:
            print(f"    - node={evt.get('node')}, status={evt.get('status')}, detail={evt.get('detail', evt.get('claim', ''))}")
        print("\n  RESULT: StreamWriter propagation WORKS")
    else:
        print("\n  WARNING: No custom events received from subagent nodes!")

    # ---- Test 2: Invoke mode (writer should be no-op) ----
    print("\n--- Test 2: Invoke mode (non-streaming, writer should be no-op) ---\n")

    result = agent.invoke(
        {"messages": [HumanMessage(content="Tell me about Albert Einstein's life")]},
        config={"configurable": {"thread_id": "test_invoke_debug"}},
    )

    tool_messages = [msg for msg in result["messages"] if msg.type == "tool"]
    final_msg = result["messages"][-1]

    print(f"  Total messages: {len(result['messages'])}")
    print(f"  Tool messages: {len(tool_messages)}")
    print(f"  Final message type: {final_msg.type}")
    print(f"  Final message (truncated): {final_msg.content[:200]}...")
    print("\n  RESULT: Invoke mode completed without errors")

    print("\n" + "=" * 70)
    print("ALL DEBUG TESTS COMPLETE")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    debug = "--debug" in sys.argv

    print(f"\n{BOLD}Deep Agents — StreamWriter Subagent Test{RESET}")
    print(f"{DIM}{'─' * 70}{RESET}")
    print(f"  Model:       {LLM_MODEL_NAME}")
    print(f"  Temperature: {LLM_TEMPERATURE}")
    print(f"  Max iter:    {LLM_MAX_ITERATIONS}")
    print(f"  Mode:        {'debug (raw events)' if debug else 'streaming (pretty)'}")
    print(f"{DIM}{'─' * 70}{RESET}")

    if debug:
        test_debug()
    else:
        test_both_subagents_streaming()
