from typing import TypedDict 

class person(TypedDict):
    name : str
    age: int
new_person : person ={'name': 'Alice', 'age': 30}
print(new_person)   