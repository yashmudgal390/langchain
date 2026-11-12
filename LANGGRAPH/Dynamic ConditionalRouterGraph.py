#Direct workflow execution dynamically to arithmetic nodes based on state inspection inside a router function
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class MathState(TypedDict):
    num1: int
    num2: int
    operation: str
    final_number: int

def adder(state: MathState) -> dict:
    return {"final_number": state["num1"] + state["num2"]}

def subtractor(state: MathState) -> dict:
    return {"final_number": state["num1"] - state["num2"]}

def router_condition(state: MathState) -> str:
    if state["operation"] == "+":
        return "addition"
    elif state["operation"] == "-":
        return "subtraction"
    return "subtraction"

builder = StateGraph(MathState)
builder.add_node("add_node", adder)
builder.add_node("sub_node", subtractor)
builder.add_node("router", lambda state: state) # Pass-through router node

builder.add_edge(START, "router")
builder.add_conditional_edges(
    "router",
    router_condition,
    {
        "addition": "add_node",
        "subtraction": "sub_node"
    }
)
builder.add_edge("add_node", END)
builder.add_edge("sub_node", END)

app = builder.compile()
res = app.invoke({"num1": 10, "num2": 5, "operation": "-"})
print(res["final_number"]) # Output: 5