###################### load libraries ######################

import os
import streamlit as st
import pandas as pd
from typing import Any
import PyPDF2

from dotenv import load_dotenv

# LangChain & ecosystem ----------------------------------------------------
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    RecursiveJsonSplitter,
)
from langchain_core.documents import Document
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import (
    PlaywrightURLLoader,
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    Docx2txtLoader,
    JSONLoader,
)
from langchain_core.documents import Document



class VectorStoreManage:
    '''
    Vector Store manage


    Create a class to manage the vector store.

    This class will have the following methods:
    1. visualize_vector_store: visualize the archieve that was vectorized by its index
    2. Adding a new document based on the type of document : PDF, TXT, CSV, JSON, DOCX, Webpage, etc.
    3. the class will receive te embedding method and the llm model to be used.
    4. the class will be able to be called as a retriever as a stuff documents chain.
    
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

    ## Embeddings

    #hf_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    # Default directory where a FAISS index will be stored if the caller does not provide one
    DEFAULT_VECTOR_STORE_DIR = "data/vector_store"

    def __init__(
        self,
        vector_store_path: str | None = None,
        embedding_method: OpenAIEmbeddings | Any | None = None,
        llm_model: ChatOpenAI | Any | None = None,
    ) -> None:
        """Create (or load) a FAISS vector store.

        Parameters
        ----------
        vector_store_path : str | None, optional
            Directory where the FAISS index lives. If *None*, a default directory
            under ``data/vector_store`` is used.
        embedding_method : Embeddings, optional
            Any langchain-compatible embeddings class. When *None* the OpenAI
            ``text-embedding-3-small`` model is used.
        llm_model : LLM, optional
            A langchain-compatible LLM to be used later for chains. Defaults to
            ``gpt-4o-mini`` with temperature **0**.
        """

        # Store path & models -------------------------------------------------
        self.vector_store_path: str = vector_store_path or self.DEFAULT_VECTOR_STORE_DIR
        self.embedding_method = embedding_method or OpenAIEmbeddings(
            model="text-embedding-3-small"
        )
        self.llm_model = llm_model or ChatOpenAI(model="gpt-4o-mini", temperature=0)

        # Try to load an existing index; otherwise defer creation until the
        # first document is added. -------------------------------------------------
        index_file = os.path.join(self.vector_store_path, "index.faiss")
        if os.path.exists(index_file):
            # Existing index found — load it
            self.vector_store = FAISS.load_local(
                self.vector_store_path,
                self.embedding_method,
                allow_dangerous_deserialization=True,
            )
        else:
            # No index yet — will be instantiated on first `add_*` call
            self.vector_store = None

        # Simple in-memory registry of documents we ingested (filenames / URLs)
        self.list_of_documents: list[str] = []

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------

    def _upsert_documents(self, docs: list[Document]) -> None:
        """Add *docs* to the index, creating the index if it does not exist."""

        if self.vector_store is None:
            # First time — create a new index.
            self.vector_store = FAISS.from_documents(docs, self.embedding_method)
        else:
            self.vector_store.add_documents(docs)

        # Persist to disk so that a future session can reload the same index.
        self.vector_store.save_local(self.vector_store_path)

    def add_new_document(self, uploaded_file):
        import tempfile
        """
        Add a new document to the vector store from an uploaded file-like object.
        The uploaded_file should be a file-like object (e.g., from Streamlit's st.file_uploader).
        """

        filename = uploaded_file.name.lower()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=400,
        )

        if filename.endswith(".pdf"):

            # Save uploaded file to a temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                temp_file.write(uploaded_file.read())
                temp_file_path = temp_file.name

            docs = []
            try:
                # Try to use PyPDFLoader (LangChain) for robust PDF parsing
                try:
                    docs = PyPDFLoader(temp_file_path).load()
                except Exception as e:
                    # If PyPDFLoader fails, fallback to PyPDF2
                    reader = PyPDF2.PdfReader(temp_file_path)
                    for i, page in enumerate(reader.pages):
                        text = page.extract_text() or ""
                        if text.strip():
                            docs.append(Document(page_content=text, metadata={"source": filename, "page": i+1}))
                # If docs is empty, raise an error
                if not docs:
                    raise ValueError("No text could be extracted from the PDF.")
            finally:
                os.remove(temp_file_path)

            # Defensive: filter out empty docs
            docs = [d for d in docs if d.page_content and d.page_content.strip()]

            if not docs:
                raise ValueError("No valid content found in the PDF document.")

            # Split into chunks
            chunks = text_splitter.split_documents(docs)
            # Store the source filename inside chunk metadata for later retrieval
            for c in chunks:
                c.metadata["source"] = filename
            self._upsert_documents(chunks)
            self.list_of_documents.append(filename)
            return

        elif filename.endswith(".txt"):
            text = uploaded_file.read().decode("utf-8")
            # Use TextLoader to create Document objects
            with tempfile.NamedTemporaryFile(delete=False, suffix='.txt', mode='w', encoding='utf-8') as temp_file:
                temp_file.write(text)
                temp_file_path = temp_file.name
            loader = TextLoader(temp_file_path)
            docs = loader.load()
            os.remove(temp_file_path)
            chunks = text_splitter.split_documents(docs)
            print(chunks[0].page_content)
            for c in chunks:
                c.metadata["source"] = filename
            self._upsert_documents(chunks)
            self.list_of_documents.append(filename)
            return 

        elif filename.endswith(".csv"):
            with tempfile.NamedTemporaryFile(delete=False, suffix='.csv', mode='wb') as temp_file:
                temp_file.write(uploaded_file.read())
                temp_file_path = temp_file.name
            loader = CSVLoader(temp_file_path)
            docs = loader.load()
            os.remove(temp_file_path)
            chunks = text_splitter.split_documents(docs)
            for c in chunks:
                c.metadata["source"] = filename
            self._upsert_documents(chunks)
            self.list_of_documents.append(filename)
            return

        elif filename.endswith(".json"):
            Json_text_splitter = RecursiveJsonSplitter(
                chunk_size=1000,
                chunk_overlap=400,
            )
            with tempfile.NamedTemporaryFile(delete=False, suffix='.json', mode='wb') as temp_file:
                temp_file.write(uploaded_file.read())
                temp_file_path = temp_file.name
            loader = JSONLoader(temp_file_path)
            docs = loader.load()
            os.remove(temp_file_path)
            chunks = Json_text_splitter.split_documents(docs)
            for c in chunks:
                c.metadata["source"] = filename
            self._upsert_documents(chunks)
            self.list_of_documents.append(filename)
            return 

        elif filename.endswith(".docx"):
            with tempfile.NamedTemporaryFile(delete=False, suffix='.docx', mode='wb') as temp_file:
                temp_file.write(uploaded_file.read())
                temp_file_path = temp_file.name
            loader = Docx2txtLoader(temp_file_path)
            docs = loader.load()
            os.remove(temp_file_path)
            chunks = text_splitter.split_documents(docs)
            for c in chunks:
                c.metadata["source"] = filename
            self._upsert_documents(chunks)
            self.list_of_documents.append(filename)
            return 
        else:
            raise ValueError(f"Unsupported file type: {filename}")

    def add_new_document_from_url(self, url):
        """
        Add a new document to the vector store from a URL.
        """
        # ── 1. one-time setup ────────────────────────────────────────────────
        import nest_asyncio, asyncio
        nest_asyncio.apply()                      # let asyncio.run() work in notebooks

        # ── 2. configure the loader (Sync API) ───────────────────────────────
        loader = PlaywrightURLLoader(
            urls=[url],
            remove_selectors=["script", "style", "noscript"],   # strip JS & CSS
            headless=True,                                      # default = True
        )

        # ── 3. run .load() in a worker thread ────────────────────────────────
        async def fetch_docs():
            # .load() is blocking; run it off the main event-loop
            return await asyncio.to_thread(loader.load)

        docs = asyncio.run(fetch_docs())
        # Divide the text into chunks
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=150)

        docs_split = text_splitter.split_documents(docs)
        for c in docs_split:
            c.metadata["source"] = url
        self._upsert_documents(docs_split)
        self.list_of_documents.append(url)
        return 
    
    def get_list_of_documents(self):
        """
        Return a list of unique document sources (filenames or URLs) currently in the vector store.
        """
        if self.vector_store is None:
            return []
        sources = set()
        for _, doc in self.vector_store.docstore._dict.items():
            source = doc.metadata.get("source")
            if source:
                sources.add(source)
        # Return a sorted list for consistency
        return sorted(sources)
    
    def remove_document(self, document_name):
        """Remove every vector whose metadata['source'] matches *document_name*."""

        if self.vector_store is None:
            raise ValueError("Vector store is not initialised.")

        # Collect IDs to delete by inspecting the underlying docstore
        ids_to_delete: list[str] = [
            doc_id
            for doc_id, doc in self.vector_store.docstore._dict.items()
            if doc.metadata.get("source") == document_name
        ]

        if not ids_to_delete:
            raise ValueError(f"No document with source '{document_name}' found in the store.")

        self.vector_store.delete(ids_to_delete)
        self.vector_store.save_local(self.vector_store_path)

        # Keep our in-memory registry in sync
        if document_name in self.list_of_documents:
            self.list_of_documents.remove(document_name)

        return 

    # retriever
    def get_retriever(self):
        if self.vector_store is None:
            raise ValueError("Vector store is empty – add documents first.")
        return self.vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    
    # stuff documents chain
    def get_stuff_documents_chain(self, prompt=None):
        """Return a simple StuffDocumentsChain using the configured LLM."""
        if prompt is None:
            return create_stuff_documents_chain(self.llm_model)
        return create_stuff_documents_chain(self.llm_model, prompt)

    # ------------------------------------------------------------------
    # Visualisation helpers
    # ------------------------------------------------------------------

    def visualize_vector_store(self):
        """Return a list (or DataFrame if *pandas* is installed) describing the docs.

        Each entry contains the internal document id, its source (file/url), and a
        short preview of the text. Useful for debugging or displaying inside a
        Streamlit app.
        """

        if self.vector_store is None:
            return []

        rows = []
        for doc_id, doc in self.vector_store.docstore._dict.items():
            row = {
                "id": doc_id,
                "source": doc.metadata.get("source", "unknown"),
                "preview": doc.page_content[:120] + ("…" if len(doc.page_content) > 120 else ""),
            }
            rows.append(row)

        try:
            import pandas as pd

            return pd.DataFrame(rows)
        except ImportError:
            return rows
    
def main():
    print("Hello World")

if __name__ == "__main__":
    main()