from langchain_groq import ChatGroq
from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel,RunnableBranch,RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal
load_dotenv()

model = ChatGroq(
    model="openai/gpt-oss-120b"
)
parser = StrOutputParser()
class Feedback(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(description="The sentiment of the feedback, either 'positive' or 'negative'.}")
    
parser2 = PydanticOutputParser(pydantic_object=Feedback)
prompt1  = PromptTemplate(
    template = "clasify the following feedback into positive or negative \n {feedback} \n {format_instructions}",
    input_variables=["feedback"],
    partial_variables={"format_instructions": parser2.get_format_instructions()}
)
classify_chain = prompt1 | model | parser2

prompt2 = PromptTemplate(
    template  = "Generate a response based on the positive feedback: \n {feedback}",
    input_variables=["feedback"]
)
prompt3 = PromptTemplate(
    template  = "Generate a response based on the negative feedback: \n {feedback}",
    input_variables=["feedback"]
)

branch_chain = RunnableBranch(
    (lambda x: x.sentiment == "positive", prompt2 | model | parser),
    (lambda x: x.sentiment == "negative", prompt3 | model | parser),
    RunnableLambda(lambda x: "Invalid sentiment")
)

chain = classify_chain | branch_chain

print(chain.invoke({"feedback": "I love the new features in your product! It's so user-friendly and intuitive."}) )