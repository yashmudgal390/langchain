from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()

model = ChatGroq(
    model="openai/gpt-oss-120b")

temp1 = PromptTemplate(
    template="write a detailed report on {topic}",
    input_variables=["topic"])
temp2 = PromptTemplate(
    template="write a 5 line summary on \n{text}",
    input_variables=["text"])

prompt = temp1.invoke({"topic":"black holes"})
result = model.invoke(prompt)
prompt2 = temp2.invoke({"text":result.content})
result2 = model.invoke(prompt2)
print(result2.content)