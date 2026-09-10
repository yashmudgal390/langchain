from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()
model = ChatGroq(model="llama-3.3-70b-versatile",temperature=0.1,max_tokens=500)
result = model.invoke("what is the capital of India?")
print(result.content)