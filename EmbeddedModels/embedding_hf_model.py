from langchain_huggingface import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
text = "The capital of France is Paris."
embedding = embedding_model.embed_query(text)
print(str(embedding))