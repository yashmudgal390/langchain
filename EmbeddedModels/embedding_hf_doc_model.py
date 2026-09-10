from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
load_dotenv()

embedding_model = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")

document =[
    "The capital of France is Paris.",
    "The largest planet in our solar system is Jupiter.",
    "The tallest mountain in the world is Mount Everest."
]

vector = embedding_model.embed_documents(document)
print(str(vector))