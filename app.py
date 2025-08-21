import streamlit as st
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader
import tempfile
from dotenv import load_dotenv
import os

# Load API key
load_dotenv()
api = os.environ.get("gro_api_key")

# Initialize Groq LLM
llm = ChatGroq(model="llama-3.1-8b-instant", api_key=api)

# ------------------ UI CONFIG ------------------
st.set_page_config(page_title="ChatGroq Q&A", page_icon="🤖", layout="wide")

# Enhanced Custom CSS with modern styling
st.markdown("""
    <style>
    /* User Message with gradient background */
    .chat-message.user {
        background: linear-gradient(135deg, #1e3c72, #2a5298);
        color: white;
        padding: 12px;
        border-radius: 12px;
        margin: 8px 0;
        font-weight: bold;
        text-align: right;
        box-shadow: 0 2px 8px rgba(30, 60, 114, 0.3);
    }
    
    /* Bot Message with clean styling and accent border */
    .chat-message.bot {
        background: #f0f4f8;
        color: #333333;
        padding: 12px;
        border-radius: 12px;
        margin: 8px 0;
        border-left: 5px solid #00c853;
        text-align: left;
        box-shadow: 0 1px 4px rgba(0, 0, 0, 0.1);
    }
    
    /* Enhanced File Uploader Styling */
    .stFileUploader {
        background: #ffffff;
        border: 2px dashed #00bcd4;
        border-radius: 12px;
        padding: 12px;
        color: #333333;
        transition: border-color 0.3s ease;
    }
    
    .stFileUploader:hover {
        border-color: #00acc1;
        background: #f8fdff;
    }
    
    /* Input Styling */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        background-color: white;
        color: black;
        padding: 10px;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        transition: border-color 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: #2a5298;
        box-shadow: 0 0 0 2px rgba(42, 82, 152, 0.2);
    }
    
    /* Success Message Styling */
    .stSuccess {
        background: #e8f5e9 !important;
        color: #2e7d32 !important;
        padding: 10px;
        border-radius: 10px;
        border-left: 4px solid #4caf50;
    }
    
    /* Chat Input Styling */
    .stChatInput > div > div > input {
        background: white;
        border-radius: 20px;
        border: 1px solid #e0e0e0;
        padding: 12px 16px;
    }
    
    /* Main Title Styling */
    .main h1 {
        background: linear-gradient(135deg, #1e3c72, #2a5298);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: bold;
    }
    
    /* Footer with highlighted name */
    .footer {
        text-align: center;
        margin-top: 30px;
        font-size: 14px;
        color: grey;
        padding: 20px;
        border-top: 1px solid #e0e0e0;
    }
    
    .footer b {
        color: #ff5722;
        font-weight: 600;
    }
    
    /* Sidebar Styling */
    .css-1d391kg {
        background: #fafafa;
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #1e3c72, #2a5298);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 8px 16px;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #2a5298, #1e3c72);
        box-shadow: 0 4px 12px rgba(30, 60, 114, 0.3);
    }
    </style>
    """, unsafe_allow_html=True)

# ------------------ HEADER ------------------
st.title("🤖 ChatGroq — Multi-Document Q&A")
st.caption("Upload your documents (PDF, DOCX, TXT) and ask questions just like ChatGPT!")

# ------------------ FILE UPLOAD ------------------
uploaded_files = st.file_uploader("📂 Upload Documents", type=["pdf", "docx", "txt"], accept_multiple_files=True)

docs = []
if uploaded_files:
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name

        if uploaded_file.name.endswith(".pdf"):
            loader = PyPDFLoader(tmp_file_path)
            docs.extend(loader.load())
        elif uploaded_file.name.endswith(".docx"):
            loader = UnstructuredWordDocumentLoader(tmp_file_path)
            docs.extend(loader.load())
        elif uploaded_file.name.endswith(".txt"):
            loader = TextLoader(tmp_file_path)
            docs.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(docs)

    st.success(f"✅ Loaded and split {len(split_docs)} chunks from {len(uploaded_files)} file(s).")

    # ------------------ CHAT SECTION ------------------
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.chat_input("💬 Ask a question about your documents...")

    if user_input:
        combined_text = " ".join([doc.page_content for doc in split_docs])
        prompt = f"Answer the following question based on the documents:\n\n{user_input}\n\nDocuments:\n{combined_text[:4000]}"

        response = llm.invoke(prompt)
        bot_reply = response.content

        # Save chat history
        st.session_state.chat_history.append(("user", user_input))
        st.session_state.chat_history.append(("bot", bot_reply))

    # Display chat history
    for role, message in st.session_state.chat_history:
        if role == "user":
            st.markdown(f"<div class='chat-message user'>{message}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='chat-message bot'>{message}</div>", unsafe_allow_html=True)

# ------------------ FOOTER ------------------
st.markdown("<div class='footer'>🚀 Built by <b>Shashi Reddy</b> | Powered with Groq & LangChain</div>", unsafe_allow_html=True)