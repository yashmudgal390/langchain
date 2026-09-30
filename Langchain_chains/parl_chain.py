from langchain_groq import ChatGroq
from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

load_dotenv()

# Models
model1 = ChatGroq(
    model="openai/gpt-oss-120b"
)

model2 = ChatGroq(
    model="openai/gpt-oss-120b"
)

# Parser
parser = StrOutputParser()

# Prompt 1: Generate notes
prompt1 = PromptTemplate(
    template="Generate short and simple notes from the following text:\n{text}",
    input_variables=["text"]
)

# Prompt 2: Generate Q&A
prompt2 = PromptTemplate(
    template="Generate 5 question answers from the following text:\n{text}",
    input_variables=["text"]
)

# Prompt 3: Combine both
prompt3 = PromptTemplate(
    template="""Combine the following notes and question answers
into a single set of notes.

Notes:
{notes}

Question Answers:
{qna}
""",
    input_variables=["notes", "qna"]
)

# Run both chains in parallel
parallel_chain = RunnableParallel({
    "notes": prompt1 | model1 | parser,
    "qna": prompt2 | model2 | parser,
})

# Merge the results
merge_chain = prompt3 | model1 | parser

# Complete chain
chain = parallel_chain | merge_chain

# Input text
text = """A support vector machine is a supervised machine learning algorithm
often used for classification and regression problems in applications such as
signal processing, natural language processing (NLP), and speech and image
recognition.

The objective of the SVM algorithm is to find a hyperplane that, to the best
degree possible, separates data points of one class from those of another
class. This hyperplane can be a line for 2D space or a plane for an
n-dimensional space, where n is the number of features for each observation
in the data set.

There can be multiple hyperplanes that separate classes in the data. The
optimal hyperplane, derived by the SVM algorithm, is the one that maximizes
the margin between the two classes.

The margin is the maximal width of the slab parallel to the hyperplane that
has no interior data points. The data points that mark the boundary of this
parallel slab and are closest to the separating hyperplane are the support
vectors.

Support vectors refer to a subset of the training observations that identify
the location of the separating hyperplane.
"""

# Run the chain
result = chain.invoke({
    "text": text
})

print(result)