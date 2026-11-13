from typing import Annotated, TypedDict, Sequence
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv
load_dotenv()


# Define State with Reducer
class ReActState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# Custom Tools with Docstrings
@tool
def add(a: int, b: int) -> int:
    """Adds two integers together."""
    return a + b

@tool
def multiply(a: int, b: int) -> int:
    """Multiplies two integers together."""
    return a * b

tools = [add, multiply]
llm = ChatGroq(
    model="openai/gpt-oss-120b"
).bind_tools(tools)

def agent_node(state: ReActState) -> dict:
    sys_msg = SystemMessage(content="You are a helpful assistant equipped with math tools.")
    response = llm.invoke([sys_msg] + list(state["messages"]))
    return {"messages": [response]}

def should_continue(state: ReActState) -> str:
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END

builder = StateGraph(ReActState)
builder.add_node("agent", agent_node)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", should_continue, ["tools", END])
builder.add_edge("tools", "agent") # Feedback loop

app = builder.compile()