import os
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import faiss

class VectorStoreUtil:
    def __init__(self, index_path="data/faiss_index"):
        self.index_path = Path(index_path)
        self.index_path.mkdir(parents=True, exist_ok=True)

        # Embeddings
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        self.dim = len(self.embeddings.embed_query("Hello world"))

        # Check if FAISS index exists
        index_file = self.index_path / "index.faiss"
        if index_file.exists():
            try:
                self.vector_store = FAISS.load_local(
                    str(self.index_path),
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                print(f"✅ Loaded existing FAISS index from {index_file}")
            except Exception as e:
                print(f"⚠ Failed to load existing FAISS index: {e}")
                self._create_empty_index()
        else:
            self._create_empty_index()

    def _create_empty_index(self):
        index = faiss.IndexFlatL2(self.dim)
        self.vector_store = FAISS(
            embedding_function=self.embeddings,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )
        print(f"🆕 Created new empty FAISS index at {self.index_path}")

    def add_documents(self, texts, metadatas=None):
        """Add a list of strings as documents with optional metadata"""
        docs = []
        if metadatas is None:
            metadatas = [{} for _ in range(len(texts))]
        for text, meta in zip(texts, metadatas):
            docs.append(Document(page_content=text, metadata=meta))
        self.vector_store.add_documents(docs)

    def save(self):
        """Save FAISS index to disk"""
        self.vector_store.save_local(str(self.index_path))
        print(f"💾 FAISS index saved to {self.index_path}")

    def as_retriever(self, k=2):
        """Return a retriever object for querying"""
        return self.vector_store.as_retriever(search_type="mmr", search_kwargs={"k": k})
