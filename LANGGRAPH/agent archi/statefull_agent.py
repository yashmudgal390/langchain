from dotenv import load_dotenv
from langchain_groq import ChatGroq
from typing import TypedDict, List, Union
from langchain_core.messages import HumanMessage,AIMessage
from langgraph.graph import StateGraph, START, END

load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-120b"
)

class MemoryState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]

def chat_node(state:MemoryState)-> dict:
    response = llm.invoke(state['messages'])
    updated_messages = state['messages']+[AIMessage(content= response.content)]
    return {"messages": updated_messages}
builder = StateGraph(MemoryState)
builder.add_node("chat_node", chat_node)
builder.add_edge(START, "chat_node")
builder.add_edge("chat_node", END)

app = builder.compile()

history = [HumanMessage(content="Hello, my name is yash ")]
result = app.invoke({"messages": history})

history = result['messages']+ [HumanMessage(content="What is my name?")]
app.invoke({"messages": history})
print(result['messages'][-1].content)  # Output: Hello, my name is yash