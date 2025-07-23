###################### load libraries ######################
import streamlit as st
import os


from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough


###################### Set up environment ######################

load_dotenv()

## openai
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

## langsmith tracking
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")

## huggingface
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

###################### AI Engine ######################
## LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

## Prompt Template

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please answer the question to the user queries"),
        ("user", "Question: {question}"),
    ]
)

parser = StrOutputParser()

def generate_response(question):
    chain = prompt | llm | parser
    answer = chain.invoke({"question": question})
    return answer


###################### Set up Streamlit app ######################

st.title("Q&A Chatbot")
st.write("This is a Q&A Chatbot built with Streamlit and OpenAI.")

st.divider()

user_input = st.text_input("Enter your question:")

if user_input:
    st.write("Generating response...")
    response = generate_response(user_input)
    st.write(response)

st.divider()








