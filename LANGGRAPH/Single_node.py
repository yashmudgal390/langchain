from typing import TypedDict
from langgraph.graph import START, END, StateGraph 
class agent(TypedDict):
    name: str
def greet(state : agent)->dict:
    """Modifies state by appending a greeting string."""
    return{"name":f"hello {state['name']},how is your day going?"}

#initialize the graph schema
builder = StateGraph(agent)
builder.add_node("greeter", greet)
builder.add_edge(START, "greeter")
builder.add_edge("greeter", END)

app = builder.compile()
result = app.invoke({"name":"John"})
print(result["name"])