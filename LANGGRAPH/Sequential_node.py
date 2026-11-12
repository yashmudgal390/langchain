#[START] ──> (First Node: Name) ──> (Second Node: Age) ──> [END]
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class SequentialState(TypedDict):
    name: str
    age: str
    final: str

def first_node(state: SequentialState) -> dict:
    return {"final": f"Hi {state['name']}! "}

def second_node(state: SequentialState) -> dict:
    # Concatenate onto existing key payload instead of overwriting
    updated_final = state["final"] + f"You are {state['age']} years old."
    return {"final": updated_final}

builder = StateGraph(SequentialState)
builder.add_node("first", first_node)
builder.add_node("second", second_node)

builder.add_edge(START, "first")
builder.add_edge("first", "second")
builder.add_edge("second", END)

app = builder.compile()
res = app.invoke({"name": "Charlie", "age": "20"})
print(res["final"]) # Output: Hi Charlie! You are 20 years old.