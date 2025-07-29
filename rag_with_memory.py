###################### load libraries ######################

import streamlit as st
import os
import nest_asyncio, asyncio
import pandas as pd
import json

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from openai import embeddings

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain.chains.combine_documents  import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_community.document_loaders import WebBaseLoader, PlaywrightURLLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import CSVLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import JSONLoader
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory

from vs_utils.vector_store_manage import VectorStoreManage
from vs_utils.memory_model import MemoryModel


###################### Set up environment ######################

load_dotenv()

nest_asyncio.apply()

## openai
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

## langsmith tracking
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")

## huggingface
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

# put into a sesion state to avoid reinitializing the vector store and the llm model
if "vector_store" not in st.session_state:
    ## vector store path
    VECTOR_STORE_PATH = "data/vector_store"

    ###################### AI Engine ######################
    ## LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    ## Embeddings
    #embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    hf_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


    ###################### Vector Store ######################
    st.session_state["vector_store"] = VectorStoreManage(vector_store_path=VECTOR_STORE_PATH, embedding_method=hf_embeddings, llm_model=llm)



###################### Set up Streamlit app ######################

st.title("RAG Q&A Chatbot with Memory")
st.write("This is a RAG Q&A Chatbot built with Streamlit and OpenAI.")

st.divider()

# Ensure memory JSON file exists and contains valid JSON
memory_path = "conversation_memory_data/memory.json"
os.makedirs(os.path.dirname(memory_path), exist_ok=True)
if not os.path.exists(memory_path) or os.path.getsize(memory_path) == 0:
    with open(memory_path, "w") as f:
        json.dump({}, f)

# Initialize MemoryModel in session state
if "memory_model" not in st.session_state:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    # Use the same vector store for both ingestion and retrieval to ensure matching embedding dimensions
    st.session_state["memory_model"] = MemoryModel(
        memory_json_file_path=memory_path,
        llm_model=llm,
        vector_store_manage=st.session_state["vector_store"],
    )
if "current_session_id" not in st.session_state:
    st.session_state["current_session_id"] = None
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# Session Management UI
st.sidebar.header("Session Management")
session_type = st.sidebar.radio("Session Type", ["New Session", "Existing Session"], key="session_type")
if session_type == "New Session":
    new_question = st.sidebar.text_input("Enter first question", key="new_session_input")
    if st.sidebar.button("Start Session", key="start_session"):
        st.session_state["memory_model"].new_session(new_question)
        st.session_state["current_session_id"] = new_question
        st.session_state["chat_history"] = []
elif session_type == "Existing Session":
    sessions = st.session_state["memory_model"].get_session_ids()
    if sessions:
        selected_session = st.sidebar.selectbox("Select session", sessions, key="existing_session_select")
        if st.sidebar.button("Load Session", key="load_session"):
            st.session_state["current_session_id"] = selected_session
            # Load structured history and flatten for UI display
            raw = st.session_state["memory_model"].get_chat_history(selected_session)
            flat = []
            for rec in raw:
                if isinstance(rec, dict):
                    flat.append(rec.get("input", ""))
                    flat.append(rec.get("answer", ""))
                elif isinstance(rec, str):
                    flat.append(rec)
            st.session_state["chat_history"] = flat
            # Sync model state (optional)
            st.session_state["memory_model"].session_id = selected_session
            st.session_state["memory_model"].chat_history = flat
    else:
        st.sidebar.write("No existing sessions found.")

# Display Chat History and Input
if st.session_state["current_session_id"]:
    st.write(f"### Session: {st.session_state['current_session_id']}")
    for idx, message in enumerate(st.session_state["chat_history"]):
        role = "You" if idx % 2 == 0 else "Bot"
        st.write(f"**{role}:** {message}")
    user_input = st.text_input("Your question:", key="chat_input")
    # Define callback to handle sending and ensure state updates before rerun
    def send_question():
        q = st.session_state.chat_input
        # Append user question
        st.session_state.chat_history.append(q)
        # Get and append response
        resp = st.session_state["memory_model"].get_the_response(q)
        st.session_state.chat_history.append(resp)
    # Use on_click callback so state changes trigger rerun and display updated chat
    st.button("Send", key="send_button", on_click=send_question)

else:
    st.write("Please start or load a session to chat.")

st.divider()

st.write("## Vector Store Management")
st.write("### Add a new document")

uploaded_file = st.file_uploader("Upload a file", type=["pdf", "txt", "csv", "json", "docx"], accept_multiple_files=True)

if uploaded_file:
    for file in uploaded_file:
        with st.spinner(f"Adding {file.name} to the vector store..."):
            st.session_state["vector_store"].add_new_document(file)
        st.success(f"{file.name} added to the vector store")

read_url = st.text_input("Enter a URL to read")
if read_url:
    with st.spinner(f"Adding {read_url} to the vector store..."):
        st.session_state["vector_store"].add_new_document_from_url(read_url)
    st.success(f"{read_url} added to the vector store")

st.write("### Remove a document")
document_name = st.selectbox("Select a document to remove", st.session_state["vector_store"].get_list_of_documents())
if st.button("Remove"):
    with st.spinner(f"Removing {document_name} from the vector store..."):
        st.session_state["vector_store"].remove_document(document_name)
    st.success(f"{document_name} removed from the vector store")

st.write("### Visualize the vector store")
st.dataframe(st.session_state["vector_store"].visualize_vector_store())
st.divider()