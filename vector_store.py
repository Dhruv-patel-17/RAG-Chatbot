# vector_store.py

import os
import uuid
import chromadb


class VectorStore:
    """Manages document embeddings in a ChromaDB vector store"""

    def __init__(
        self,
        collection_name: str = "pdf_documents",
        persist_directory: str = "./data/vector_store",
    ):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self._initialize_store()

    def _initialize_store(self):
        try:
            os.makedirs(self.persist_directory, exist_ok=True)

            self.client = chromadb.PersistentClient(
                path=self.persist_directory
            )

            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "PDF document embeddings for RAG"},
            )

            print(f"Vector store ready: {self.collection_name}")
            print(f"Documents in collection: {self.collection.count()}")

        except Exception as e:
            raise RuntimeError(f"Error initializing vector store: {e}")

    def add_documents(self, documents, embeddings):
        """
        documents: list of LangChain Document objects
        embeddings: numpy array
        """

        if len(documents) != len(embeddings):
            raise ValueError("Documents and embeddings length mismatch")

        ids = []
        texts = []
        metadatas = []

        for doc, emb in zip(documents, embeddings):
            doc_id = str(uuid.uuid4())
            ids.append(doc_id)

            texts.append(doc.page_content)

            metadata = dict(doc.metadata or {})
            metadata["content_length"] = len(doc.page_content)
            metadatas.append(metadata)

        self.collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings.tolist(),
        )

        print(f"Added {len(documents)} documents to vector store")
