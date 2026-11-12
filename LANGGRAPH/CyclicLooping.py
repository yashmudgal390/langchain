#Construct stateful execution cycles that evaluate condition metrics repeatedly until a threshold triggers graph exit.
import random
from typing import TypedDict, List
from langgraph.graph import StateGraph, START, END

class LoopState(TypedDict):
    name: str
    numbers: List[int]
    counter: int

def greeting_node(state: LoopState) -> dict:
    return {"name": f"Hi {state['name']}", "counter": 0, "numbers": []}

def random_generator_node(state: LoopState) -> dict:
    rand_val = random.randint(0, 10)
    new_numbers = state["numbers"] + [rand_val]
    return {"numbers": new_numbers, "counter": state["counter"] + 1}

def loop_decision(state: LoopState) -> str:
    if state["counter"] < 5:
        return "loop"
    return "exit"

builder = StateGraph(LoopState)
builder.add_node("greet", greeting_node)
builder.add_node("gen_num", random_generator_node)

builder.add_edge(START, "greet")
builder.add_edge("greet", "gen_num")

builder.add_conditional_edges(
    "gen_num",
    loop_decision,
    {
        "loop": "gen_num",
        "exit": END
    }
)

app = builder.compile()
res = app.invoke({"name": "Yash", "numbers": [], "counter": -1})
print(res)
# Generates 5 random numbers through controlled state cycling