import os
from typing import Annotated, TypedDict, Sequence
from dotenv import load_dotenv
from groq import Groq

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

# 0. Load GROQ_API_KEY from .env
load_dotenv()



# 1. Document Indexing Pipeline with Local Hugging Face Embeddings
loader = PyPDFLoader(r"d:\langchain\LANGGRAPH\agent archi\Stock_Market_Performance_2024.pdf")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# Runs locally on CPU/GPU (384-dimensional embeddings, no API key required)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
    collection_name="stock_market_hf",
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})


# 2. Retriever Tool Construction
@tool
def retriever_tool(query: str) -> str:
    """Searches the knowledge base for relevant context chunks to answer user questions."""
    retrieved_docs = retriever.invoke(query)
    if not retrieved_docs:
        return "No relevant context found."
    return "\n\n".join(
        [f"[Page {d.metadata.get('page', 'N/A')}]: {d.page_content}" for d in retrieved_docs]
    )


tools = [retriever_tool]

llm = ChatGroq(
    model=('openai/gpt-oss-120b'),
    temperature=0,
).bind_tools(tools)


# 3. Graph Construction
class RAGState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


def rag_agent(state: RAGState) -> dict:
    sys_msg = SystemMessage(
        content=(
            "You are an intelligent financial research assistant. "
            "Always use `retriever_tool` to look up facts before answering, and cite specific facts "
            "and page numbers from the retrieved documents. If the document does not contain the answer, "
            "state that clearly without guessing."
        )
    )
    res = llm.invoke([sys_msg] + list(state["messages"]))
    return {"messages": [res]}


def should_retrieve(state: RAGState) -> str:
    last_msg = state["messages"][-1]
    if getattr(last_msg, "tool_calls", None):
        return "tools"
    return END


builder = StateGraph(RAGState)
builder.add_node("agent", rag_agent)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", should_retrieve, ["tools", END])
builder.add_edge("tools", "agent")

rag_app = builder.compile()


# 4. Interactive CLI Runner
if __name__ == "__main__":
    print("=== Groq + HuggingFace RAG Agent Ready (type 'exit' to quit) ===")
    while True:
        user_query = input("\nAsk a question: ").strip()
        if not user_query:
            continue
        if user_query.lower() in {"exit", "quit"}:
            break

        result = rag_app.invoke({"messages": [HumanMessage(content=user_query)]})
        print(f"\nAssistant:\n{result['messages'][-1].content}")