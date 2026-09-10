from langchain_huggingface import  ChatHuggingFace , HuggingFacePipeline

llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs=dict(max_new_tokens=10, temperature=0.1)
)
model = ChatHuggingFace(llm=llm)
result = model.invoke("What is the capital of France?")
print(result.content)