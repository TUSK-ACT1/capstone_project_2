"""
Module 3 - Multimodal RAG
--------------------------
Loads the small domain knowledge base (college, tourism, healthcare,
agriculture, library, museums, historical monuments) and builds a
retrieval pipeline using LangChain + a vector store (FAISS) with
sentence-transformer embeddings.

The retrieved context is then passed to the LLM (Mixtral / Llama via
an inference provider, or any Chat model you configure) to generate a
grounded answer.
"""

import os
import json
import glob
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

KB_DIR = os.path.join(os.path.dirname(__file__), "..", "knowledge_base")
INDEX_DIR = os.path.join(os.path.dirname(__file__), "..", "assets", "faiss_index")

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_knowledge_base(kb_dir: str = KB_DIR):
    """Reads every *.json file in the knowledge_base folder and returns
    a list of langchain Document objects, tagged with their domain."""
    documents = []
    for filepath in glob.glob(os.path.join(kb_dir, "*.json")):
        domain = os.path.splitext(os.path.basename(filepath))[0]
        with open(filepath, "r", encoding="utf-8") as f:
            entries = json.load(f)
        for entry in entries:
            text = f"{entry['title']}: {entry['content']}"
            documents.append(
                Document(
                    page_content=text,
                    metadata={"domain": domain, "id": entry["id"], "title": entry["title"]},
                )
            )
    return documents


def build_vector_store(documents=None, persist: bool = True):
    """Splits documents into chunks, embeds them, and builds/saves a FAISS index."""
    if documents is None:
        documents = load_knowledge_base()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vector_store = FAISS.from_documents(chunks, embeddings)

    if persist:
        os.makedirs(INDEX_DIR, exist_ok=True)
        vector_store.save_local(INDEX_DIR)

    return vector_store


def load_vector_store():
    """Loads a previously saved FAISS index, building it fresh if missing."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    if os.path.exists(INDEX_DIR):
        return FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
    return build_vector_store()


class KnowledgeRetriever:
    """Thin wrapper exposing a simple `.retrieve(query, k)` interface,
    so the rest of the app doesn't need to know about LangChain internals."""

    def __init__(self):
        self.vector_store = load_vector_store()

    def retrieve(self, query: str, k: int = 3):
        results = self.vector_store.similarity_search(query, k=k)
        return [
            {
                "domain": r.metadata.get("domain"),
                "title": r.metadata.get("title"),
                "content": r.page_content,
            }
            for r in results
        ]

    def retrieve_as_context(self, query: str, k: int = 3) -> str:
        hits = self.retrieve(query, k=k)
        context = "\n\n".join(
            f"[{h['domain']}] {h['content']}" for h in hits
        )
        return context


if __name__ == "__main__":
    # Quick manual test: builds the index and runs a sample query.
    retriever = KnowledgeRetriever()
    print(retriever.retrieve_as_context("What are the library timings?"))
