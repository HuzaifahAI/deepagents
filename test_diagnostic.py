"""Diagnostic: compare state after stream vs invoke."""
import os
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import StreamWriter
from typing_extensions import TypedDict
from typing import Any
from langchain_core.messages import BaseMessage

from deepagents.graph import create_deep_agent
from deepagents.middleware.subagents import CompiledSubAgent

LLM = ChatOpenAI(
    base_url=os.environ.get("LLM_API_BASE_URL", ""),
    model=os.environ.get("LLM_MODEL_NAME", "qwen3-235b"),
    temperature=0.1,
    api_key="not-needed",
)

class SubAgentState(TypedDict):
    messages: list[BaseMessage]

def research_node(state: SubAgentState, writer: StreamWriter) -> SubAgentState:
    writer({"node": "research", "status": "started"})
    response = LLM.invoke([HumanMessage(content=f"Briefly list 3 key facts about: {state['messages'][-1].content}")])
    writer({"node": "research", "status": "completed"})
    return {"messages": [response]}

def build_simple_subagent():
    graph = StateGraph(SubAgentState)
    graph.add_node("research", research_node)
    graph.add_edge(START, "research")
    graph.add_edge("research", END)
    return graph.compile()

QUERY = "Use the researcher to find facts about Marie Curie."

def test_stream_then_state():
    print("=" * 70)
    print("TEST 1: stream_mode='custom' then get_state()")
    print("=" * 70)
    agent = create_deep_agent(
        model=LLM,
        checkpointer=InMemorySaver(),
        subagents=[CompiledSubAgent(
            name="researcher",
            description="Researches facts about a topic.",
            runnable=build_simple_subagent(),
        )],
    )

    custom_events = []
    for namespace, data in agent.stream(
        {"messages": [HumanMessage(content=QUERY)]},
        config={"configurable": {"thread_id": "stream_test"}},
        stream_mode="custom",
        subgraphs=True,
    ):
        if isinstance(data, dict) and "node" in data:
            custom_events.append(data)
            print(f"  [CUSTOM] {data}")

    state = agent.get_state({"configurable": {"thread_id": "stream_test"}})
    msgs = state.values.get("messages", [])
    print(f"\n  Total messages in state: {len(msgs)}")
    for i, m in enumerate(msgs):
        print(f"  [{i}] type={m.type}, len={len(m.content)}")
        if i == len(msgs) - 1:
            print(f"  FINAL MESSAGE:\n{m.content[:500]}")
    print(f"\n  Custom events: {len(custom_events)}")
    print(f"  State next: {state.next}")

def test_stream_values_mode():
    print("\n" + "=" * 70)
    print("TEST 2: stream_mode=['custom', 'values'] (both)")
    print("=" * 70)
    agent = create_deep_agent(
        model=LLM,
        checkpointer=InMemorySaver(),
        subagents=[CompiledSubAgent(
            name="researcher",
            description="Researches facts about a topic.",
            runnable=build_simple_subagent(),
        )],
    )

    custom_events = []
    final_values = None
    for item in agent.stream(
        {"messages": [HumanMessage(content=QUERY)]},
        config={"configurable": {"thread_id": "stream_values_test"}},
        stream_mode=["custom", "values"],
        subgraphs=True,
    ):
        namespace, (mode, data) = item[0], item[1] if len(item) > 2 else (item[0], item[1])
        # With multiple stream modes, format is (stream_mode, data)
        print(f"  [{type(item)}] {str(item)[:200]}")

    state = agent.get_state({"configurable": {"thread_id": "stream_values_test"}})
    msgs = state.values.get("messages", [])
    print(f"\n  Total messages in state: {len(msgs)}")
    if msgs:
        print(f"  FINAL MESSAGE:\n{msgs[-1].content[:500]}")

def test_invoke():
    print("\n" + "=" * 70)
    print("TEST 3: invoke() (non-streaming)")
    print("=" * 70)
    agent = create_deep_agent(
        model=LLM,
        checkpointer=InMemorySaver(),
        subagents=[CompiledSubAgent(
            name="researcher",
            description="Researches facts about a topic.",
            runnable=build_simple_subagent(),
        )],
    )

    result = agent.invoke(
        {"messages": [HumanMessage(content=QUERY)]},
        config={"configurable": {"thread_id": "invoke_test"}},
    )
    msgs = result["messages"]
    print(f"  Total messages: {len(msgs)}")
    for i, m in enumerate(msgs):
        print(f"  [{i}] type={m.type}, len={len(m.content)}")
    print(f"\n  FINAL MESSAGE:\n{msgs[-1].content[:500]}")

if __name__ == "__main__":
    print("Running diagnostic: stream vs invoke final message comparison\n")
    test_stream_then_state()
    test_invoke()
