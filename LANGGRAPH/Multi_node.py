from typing import TypedDict, List
from langgraph.graph import StateGraph, START, END

class MultiInputState(TypedDict):
    values: List[int]
    name: str
    result: str

def process_values(state: MultiInputState) -> dict:
    total_sum = sum(state["values"])
    formatted_msg = f"Hi there {state['name']}, your sum is equal to {total_sum}."
    return {"result": formatted_msg}

builder = StateGraph(MultiInputState)
builder.add_node("processor", process_values)
builder.add_edge(START, "processor")
builder.add_edge("processor", END)

app = builder.compile()
output = app.invoke({"values": [1, 2, 3, 4], "name": "Steve"})
print(output["result"]) # Output: Hi there Steve, your sum is equal to 10.