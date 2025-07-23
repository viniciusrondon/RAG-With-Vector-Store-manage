###################### load libraries ######################
from openai import embeddings
import streamlit as st
import os
import nest_asyncio, asyncio
import pandas as pd

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.chains.combine_documents  import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.document_loaders import WebBaseLoader, PlaywrightURLLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import CSVLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import JSONLoader
from vs_utils.vector_store_manage import VectorStoreManage





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



## Prompt Template

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """
            You are a helpful assistant. 
            Please answer the question to the user queries.
            You will be given a question and a context.
            You will need to answer the question based on the context.
            If you don't know the answer, please say "I don't know".
            If the question is not related to the context, please say "I don't know".
            If the question is not clear, please ask the user to clarify.
            If the question is not related to the context, please say "I don't know".
            Please answer in the same language as the question.
            Please provide the answer in a concise and to the point manner with the most accurate information based on the context and the question.
            <context> 
            {context}
            Question : {question}      
        """),
        ("user", "Question: {question}"),
    ]
)

parser = StrOutputParser()

def generate_response(question):
    document_chain = st.session_state["vector_store"].get_stuff_documents_chain(prompt = prompt) # create a chain of documents for passing to the llm model
    retriever = st.session_state["vector_store"].get_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    answer = retrieval_chain.invoke({"input": question, "question": question})
    return answer


###################### Set up Streamlit app ######################

st.title("RAG Q&A Chatbot")
st.write("This is a RAG Q&A Chatbot built with Streamlit and OpenAI.")

st.divider()

question = st.text_input("Enter a question")
if question:
    if st.session_state["vector_store"].vector_store is not None:
        with st.spinner(f"Answering {question}..."):
            answer = generate_response(question)
        st.write(answer)
    else:
        st.write("No documents in the vector store")


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

