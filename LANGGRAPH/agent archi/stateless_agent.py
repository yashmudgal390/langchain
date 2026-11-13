from typing import TypedDict, List
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END

load_dotenv()

class BotState(TypedDict):
    messages: List[HumanMessage]

llm = ChatGroq(
    model="openai/gpt-oss-120b"
)

def process(state: BotState) -> dict:
    response = llm.invoke(state["messages"])
    print(f"AI: {response.content}")
    return state

builder = StateGraph(BotState)
builder.add_node("process", process)
builder.add_edge(START, "process")
builder.add_edge("process", END)

app = builder.compile()
# Stateless execution per invocation
app.invoke({"messages": [HumanMessage(content="Hello, who are you?")]})