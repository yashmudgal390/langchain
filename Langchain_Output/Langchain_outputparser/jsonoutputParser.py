from langchain_groq import ChatGroq
from dotenv import load_dotenv, parser
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser


load_dotenv()
parser = JsonOutputParser()
model = ChatGroq(
    model="openai/gpt-oss-120b")
temp1 = PromptTemplate(
    template="give me name , age , city of random person \n {format_instructions}",
    input_variables=[],
    partial_variables={"format_instructions": parser.get_format_instructions()})

chain= temp1 | model | parser 
result  = chain.invoke({})
print(result)