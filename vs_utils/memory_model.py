###################### load libraries ######################

import streamlit as st
import os
import nest_asyncio, asyncio
import pandas as pd
import json

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from openai import embeddings

from typing import Any

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


class MemoryModel:
    '''
    Memory Model

    This class is used to manage the memory of the chat history.

    This class will have the following methods:
    1. add_new_session: add a new session to the memory json file
    2. get_session_id: get the session id of the current session
    3. get_chat_history: get the chat history of the current session
    4. get_memory: get the memory of the current session
    5. get_memory_json: get the memory json of the current session
    6. get_memory_json_file: get the memory json file of the current session
    '''

    load_dotenv()

    ## openai
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

    ## langsmith tracking
    os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")

    ## huggingface
    os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

    ## memory json file
    DEFAULT_MEMORY_JSON_FILE_PATH = "conversation_memory_data/memory.json"



    def __init__(
        self,
        memory_json_file_path: str | None = None,
        memory_json_file: dict | None = None,
        llm_model: ChatOpenAI | Any | None = None,
        session_id: str | None = None,
        chat_history: list[str] | None = None,
        context_system_prompt: str | None = None,
        system_prompt: str | None = None,
        question: str | None = None,
        vector_store_manage: VectorStoreManage | None = None,
        ) -> None:
        """
            Initialize the MemoryModel class

            Parameters
            ----------
            memory_json_file: str | None, optional
                The path to the memory json file. If *None*, a default path is used.
            llm_model: ChatOpenAI | Any | None, optional
                The llm model to be used. If *None*, a default model is used.
            session_id: str | None, optional
                The session id of the current session. If *None*, a default session id is used.
            chat_history: list[str] | None, optional
                The chat history of the current session. If *None*, a default chat history is used.
            context_system_prompt: str | None, optional
        """
        self.memory_json_file_path = memory_json_file_path or self.DEFAULT_MEMORY_JSON_FILE_PATH
        self.llm_model = llm_model or ChatOpenAI(model="gpt-4o-mini", temperature=0)

        # Safely load or initialize memory JSON
        if os.path.exists(self.memory_json_file_path):
            try:
                with open(self.memory_json_file_path, "r") as f:
                    self.memory_json_file = json.load(f)
            except (json.JSONDecodeError, ValueError):
                # File empty or invalid JSON, reset to empty dict
                self.memory_json_file = {}
                self._save_memory_file()
        else:
            # Create a new memory JSON file
            self.memory_json_file = {}
            os.makedirs(os.path.dirname(self.memory_json_file_path), exist_ok=True)
            with open(self.memory_json_file_path, "w") as f:
                json.dump(self.memory_json_file, f, indent=4)
        # Initialize memory model state attributes
        self.session_id = None
        self.chat_history = []
        self.question = None

        self.context_system_prompt = context_system_prompt or """
        Given a chat history and the latest user message, you need to answer the question based on the context.
        You are a helpful sales person that can answer questions about the website and try to convert the user into a customer.
        Use the following pieces of retrieved context to answer the question. 
        About the questions, if you don't know the answer, just say that you don't know.
        Use three sentences maximum and keep the answer concise.
        Always start your response cordially" 
        Always answer in the same language as the question.
        """
        self.system_prompt = system_prompt or """
        You are a helpful sales person that can answer questions about the website and try to convert the user into a customer.
        Use the following pieces of retrieved context to answer the question. 
        About the questions, if you don't know the answer, just say that you don't know.
        Use three sentences maximum and keep the answer concise.
        Always start your response cordially" 
        Always answer in the same language as the question.
        """
        self.vector_store_manage = vector_store_manage or VectorStoreManage()

    def _save_memory_file(self) -> None:
        """Persist the memory JSON to disk."""
        os.makedirs(os.path.dirname(self.memory_json_file_path), exist_ok=True)
        with open(self.memory_json_file_path, "w") as f:
            json.dump(self.memory_json_file, f, indent=4)

    def get_chat_history(self, session_id: str) -> BaseChatMessageHistory:
        '''
        Get the chat history of the current session
        '''
        session = self.memory_json_file.get(session_id, {})
        return session.get("configurable", {}).get("chat_history", [])

    def new_session(self, question: str):
        '''
        Create a new session
        '''
        session_id = str(question)
        # Initialize session with empty chat history
        self.memory_json_file[session_id] = {
            "configurable": {
                "session_id": session_id,
                "chat_history": []
            }
        }
        self.session_id = session_id
        self.chat_history = []
        self.question = question
        self._save_memory_file()

    def get_session_ids(self) -> list[str]:
        """
        Return a list of all stored session IDs.
        """
        return list(self.memory_json_file.keys())

    def get_chat_history(self, session_id: str) -> list[str]:
        """
        Return the chat history for a given session ID.
        """
        session = self.memory_json_file.get(session_id, {})
        return session.get("configurable", {}).get("chat_history", [])

    def get_the_response(self, question: str):
        """Run RAG with history, record structured memory (input, context, answer), and return answer."""
        if self.session_id is None:
            raise ValueError("No active session. Call new_session() first.")
        # Retrieve existing structured records
        records = self.memory_json_file.setdefault(self.session_id, {}).setdefault("configurable", {}).setdefault("chat_history", [])
        # Flatten history for RAG chain, handling old plain strings and new dict records
        history_texts: list[str] = []
        for rec in records:
            if isinstance(rec, dict):
                # Structured record
                history_texts.append(rec.get("input", ""))
                history_texts.append(rec.get("answer", ""))
            elif isinstance(rec, str):
                # Legacy plain entry
                history_texts.append(rec)
            # ignore other types
        # Build prompts and chains
        context_prompt = ChatPromptTemplate.from_messages([
            ("system", self.context_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"{self.system_prompt}\nContext:\n{{context}}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])
        history_aware_chain = create_history_aware_retriever(
            llm=self.llm_model,
            retriever=self.vector_store_manage.get_retriever(),
            prompt=context_prompt
        )
        question_answer_chain = create_stuff_documents_chain(self.llm_model, prompt)
        rag_chain = create_retrieval_chain(history_aware_chain, question_answer_chain)
        # Execute RAG
        result = rag_chain.invoke({"input": question, "chat_history": history_texts})
        # Extract answer
        answer = result.get("answer", str(result)) if isinstance(result, dict) else str(result)
        # Structure context metadata
        structured_context = []
        for doc in result.get("context", []) if isinstance(result, dict) else []:
            structured_context.append({
                "id": getattr(doc, "id", None),
                "metadata": getattr(doc, "metadata", {}),
                "page_content": getattr(doc, "page_content", None)
            })
        # Append new record and persist
        record = {"input": question, "context": structured_context, "answer": answer}
        records.append(record)
        self._save_memory_file()
        return answer



    def add_new_session(self, session_id: str, chat_history: list[str]):
        """
        Add a new session with the provided chat history to memory and persist it.
        """
        self.memory_json_file[session_id] = {
            "configurable": {
                "session_id": session_id,
                "chat_history": chat_history
            }
        }
        self._save_memory_file()

    def get_memory(self, session_id: str) -> dict:
        """
        Return the full memory record for a given session ID.
        """
        return self.memory_json_file.get(session_id, {})

    def get_memory_json(self) -> dict:
        """
        Return the in-memory representation of the entire memory JSON.
        """
        return self.memory_json_file

    def get_memory_json_file(self) -> str:
        """
        Return the path to the memory JSON file.
        """
        return self.memory_json_file_path