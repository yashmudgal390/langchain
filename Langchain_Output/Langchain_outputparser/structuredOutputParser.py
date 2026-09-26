from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_classic.output_parsers import StructuredOutputParser, ResponseSchema

load_dotenv()

model = ChatGroq(
    model="openai/gpt-oss-120b"
)

schema = [
    ResponseSchema(
        name="fact1",
        description="fact 1 about the topic"
    ),
    ResponseSchema(
        name="fact2",
        description="fact 2 about the topic"
    ),
]

parser = StructuredOutputParser.from_response_schemas(schema)

temp = PromptTemplate(
    template="give 2 facts about the {topic}\n{format_instructions}",
    input_variables=["topic"],
    partial_variables={
        "format_instructions": parser.get_format_instructions()
    }
)

prompt = temp.invoke({
    "topic": "black holes"
})

result = model.invoke(prompt)

parsed = parser.parse(result.content)

print(parsed)