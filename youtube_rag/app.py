import os
import re
from typing import Tuple, Optional

import streamlit as st
from dotenv import load_dotenv
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable
)

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# =====================================================
# 1. CONFIGURATION & STATE
# =====================================================
load_dotenv()

st.set_page_config(
    page_title="YouTube AI Assistant",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

def init_session_state():
    defaults = {
        "vector_store": None,
        "transcript": "",
        "video_id": None,
        "messages": [],
        "summary": None,
        "processed": False
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# =====================================================
# 2. CUSTOM CSS
# =====================================================
def inject_custom_css():
    st.markdown("""
    <style>
        .main-title {
            font-size: 2.5rem;
            font-weight: 800;
            background: -webkit-linear-gradient(45deg, #FF0000, #ff7b00);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }
        .subtitle {
            font-size: 1.1rem;
            color: #9ca3af;
            margin-bottom: 2rem;
        }
        div[data-testid="stMetric"] {
            background-color: #1a1c23;
            padding: 1rem;
            border: 1px solid #2e323a;
            border-radius: 0.75rem;
        }
        div[data-testid="stChatMessage"] {
            background-color: #1a1c23;
            border: 1px solid #2e323a;
            border-radius: 0.75rem;
            padding: 1rem;
            margin-bottom: 1rem;
        }
    </style>
    """, unsafe_allow_html=True)

# =====================================================
# 3. CORE LOGIC (INTEGRATED FROM YOUR IPYNB)
# =====================================================
@st.cache_resource
def load_models() -> Tuple[ChatGroq, HuggingFaceEmbeddings]:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error("GROQ_API_KEY not found. Please add it to your .env file.")
        st.stop()

    # Using the exact Groq model from your notebook
    llm = ChatGroq(
        model="openai/gpt-oss-120b", 
        temperature=0,
        api_key=api_key
    )
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return llm, embeddings

def extract_video_id(url: str) -> Optional[str]:
    patterns = [
        r"(?:v=|\/)([0-9A-Za-z_-]{11}).*",
        r"(?:youtu\.be\/)([0-9A-Za-z_-]{11})",
        r"(?:shorts\/)([0-9A-Za-z_-]{11})",
        r"(?:embed\/)([0-9A-Za-z_-]{11})"
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def process_video(video_id: str):
    # EXACT logic from your Jupyter Notebook
    transcript = YouTubeTranscriptApi().fetch(
        video_id,
        languages=["en"]
    )
    
    transcript_text = " ".join(
        snippet.text for snippet in transcript.snippets
    )

    if not transcript_text.strip():
        raise ValueError("The transcript is empty.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([transcript_text])

    _, embeddings = load_models()
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    return transcript_text, vector_store

def get_rag_chain():
    """Exact RAG pipeline from your Jupyter Notebook"""
    llm, _ = load_models()
    
    # 1. Set up retriever
    retriever = st.session_state.vector_store.as_retriever(
        search_type="similarity", 
        search_kwargs={"k": 4}
    )
    
    # 2. Exact PromptTemplate from your notebook
    prompt = PromptTemplate(
        template="""
          You are a helpful assistant.
          Answer ONLY from the provided transcript context.
          If the context is insufficient, just say you don't know.

          {context}
          Question: {question}
        """,
        input_variables=['context', 'question']
    )
    
    # 3. Format docs function
    def format_docs(retrieved_docs):
        return "\n\n".join(doc.page_content for doc in retrieved_docs)
    
    # 4. LCEL Parallel Chain setup
    parallel_chain = RunnableParallel({
        'context': retriever | RunnableLambda(format_docs),
        'question': RunnablePassthrough()
    })
    
    parser = StrOutputParser()
    
    # 5. Final Main Chain
    main_chain = parallel_chain | prompt | llm | parser
    return main_chain

# =====================================================
# 4. UI COMPONENTS
# =====================================================
def render_sidebar():
    with st.sidebar:
        st.title("🎬 YouTube AI")
        st.caption("Your personal video knowledge assistant")
        st.markdown("---")
        
        st.subheader("About")
        st.write("Transform YouTube videos into searchable knowledge.")
        
        st.markdown("---")
        if st.session_state.processed:
            st.success("✅ Knowledge base ready")
        else:
            st.info("ℹ️ No video loaded")
            
        st.markdown("---")
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
            
        if st.button("🔄 Reset Entire Session", use_container_width=True, type="primary"):
            for key in st.session_state.keys():
                del st.session_state[key]
            st.rerun()

def render_dashboard():
    st.markdown("### 📊 Video Dashboard")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Transcript Characters", f"{len(st.session_state.transcript):,}")
    with col2:
        st.metric("Video ID", st.session_state.video_id)
    with col3:
        st.metric("Status", "Ready ⚡")

    st.video(f"https://www.youtube.com/watch?v={st.session_state.video_id}")

    with st.expander("📄 View Full Transcript"):
        st.write(st.session_state.transcript)

def handle_chat():
    st.markdown("### 💬 Chat With Your Video")
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_question := st.chat_input("Ask anything about this video..."):
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        with st.chat_message("assistant"):
            try:
                with st.spinner("Searching transcript..."):
                    rag_chain = get_rag_chain()
                    response = rag_chain.invoke(user_question)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"Something went wrong: {str(e)}")

# =====================================================
# 5. MAIN APP EXECUTION
# =====================================================
def main():
    init_session_state()
    inject_custom_css()
    render_sidebar()

    st.markdown('<div class="main-title">YouTube AI Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Watch less. Understand more. Chat with any YouTube video.</div>', unsafe_allow_html=True)

    if not st.session_state.processed:
        with st.form("video_form"):
            video_url = st.text_input("YouTube Video URL", placeholder="https://www.youtube.com/watch?v=...")
            process_button = st.form_submit_button("🚀 Process Video", use_container_width=True)

        if process_button:
            video_id = extract_video_id(video_url.strip())
            if not video_id:
                st.error("Please enter a valid YouTube URL.")
            else:
                with st.status("Processing Video...", expanded=True) as status:
                    try:
                        st.write("📥 Fetching transcript...")
                        transcript_text, vector_store = process_video(video_id)
                        
                        st.write("🧠 Building knowledge base...")
                        st.session_state.update({
                            "transcript": transcript_text,
                            "vector_store": vector_store,
                            "video_id": video_id,
                            "messages": [],
                            "summary": None,
                            "processed": True
                        })
                        status.update(label="Processing Complete!", state="complete", expanded=False)
                        st.rerun()
                        
                    except TranscriptsDisabled:
                        status.update(label="Failed: Captions disabled", state="error")
                        st.error("Captions are disabled for this video.")
                    except NoTranscriptFound:
                        status.update(label="Failed: No English transcript", state="error")
                        st.error("No English transcript found.")
                    except Exception as e:
                        status.update(label="Processing failed", state="error")
                        st.error(f"Error: {str(e)}")

    if st.session_state.processed:
        render_dashboard()
        st.markdown("---")
        handle_chat()

if __name__ == "__main__":
    main()