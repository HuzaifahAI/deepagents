"""Compare full message state between stream and invoke modes."""
import os
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import StreamWriter
from typing_extensions import TypedDict

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
    response = LLM.invoke([HumanMessage(content=f"List 3 key facts about: {state['messages'][-1].content}")])
    writer({"node": "research", "status": "completed"})
    return {"messages": [response]}

def build_subagent():
    graph = StateGraph(SubAgentState)
    graph.add_node("research", research_node)
    graph.add_edge(START, "research")
    graph.add_edge("research", END)
    return graph.compile()

QUERY = (
    "I need two things: "
    "First, use the history_writer to find facts about World War 2. "
    "Second, use the bio_researcher to find facts about Albert Einstein."
)

def dump_messages(label, msgs):
    print(f"\n{'=' * 70}")
    print(f"{label}: {len(msgs)} messages")
    print('=' * 70)
    for i, m in enumerate(msgs):
        print(f"\n--- [{i}] type={m.type}, content_len={len(m.content)} ---")
        # For tool messages, show full content (this is what the LLM sees)
        if m.type == "tool":
            print(f"  tool_call_id: {getattr(m, 'tool_call_id', 'N/A')}")
            print(f"  FULL CONTENT:\n{m.content[:1000]}")
        elif m.type == "ai":
            tc = getattr(m, 'tool_calls', [])
            if tc:
                print(f"  tool_calls: {[t['name'] + '(' + str(t['args'])[:100] + ')' for t in tc]}")
            print(f"  CONTENT:\n{m.content[:1000]}")
        else:
            print(f"  CONTENT:\n{m.content[:500]}")

def test_stream():
    print("\n\nRUNNING: stream_mode='custom' + get_state()")
    agent = create_deep_agent(
        model=LLM,
        checkpointer=InMemorySaver(),
        subagents=[
            CompiledSubAgent(name="history_writer", description="Finds historical facts.", runnable=build_subagent()),
            CompiledSubAgent(name="bio_researcher", description="Researches biographical facts.", runnable=build_subagent()),
        ],
    )
    events = []
    for ns, data in agent.stream(
        {"messages": [HumanMessage(content=QUERY)]},
        config={"configurable": {"thread_id": "t1"}},
        stream_mode="custom",
        subgraphs=True,
    ):
        if isinstance(data, dict) and "node" in data:
            events.append(data)

    state = agent.get_state({"configurable": {"thread_id": "t1"}})
    dump_messages("STREAM MODE", state.values["messages"])
    print(f"\nCustom events: {len(events)}")
    return state.values["messages"]

def test_invoke():
    print("\n\nRUNNING: invoke()")
    agent = create_deep_agent(
        model=LLM,
        checkpointer=InMemorySaver(),
        subagents=[
            CompiledSubAgent(name="history_writer", description="Finds historical facts.", runnable=build_subagent()),
            CompiledSubAgent(name="bio_researcher", description="Researches biographical facts.", runnable=build_subagent()),
        ],
    )
    result = agent.invoke(
        {"messages": [HumanMessage(content=QUERY)]},
        config={"configurable": {"thread_id": "t2"}},
    )
    dump_messages("INVOKE MODE", result["messages"])
    return result["messages"]

if __name__ == "__main__":
    stream_msgs = test_stream()
    invoke_msgs = test_invoke()

    print("\n\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print(f"Stream: {len(stream_msgs)} msgs, final AI len={len(stream_msgs[-1].content)}")
    print(f"Invoke: {len(invoke_msgs)} msgs, final AI len={len(invoke_msgs[-1].content)}")

    # Compare tool message lengths
    stream_tools = [m for m in stream_msgs if m.type == "tool"]
    invoke_tools = [m for m in invoke_msgs if m.type == "tool"]
    print(f"\nStream tool msgs: {[len(m.content) for m in stream_tools]}")
    print(f"Invoke tool msgs: {[len(m.content) for m in invoke_tools]}")
