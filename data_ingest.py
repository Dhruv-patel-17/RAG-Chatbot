import os
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader,PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


from embedding_manager import EmbeddingManager
from vector_store import VectorStore


DATA_DIR = "./data"


# ---------------------------------------------------------
# 1. Load all PDFs and attach metadata
# ---------------------------------------------------------
def process_all_pdfs(base_directory: str):
    """
    Load all PDFs recursively and attach subject metadata
    (folder name = subject)
    """
    all_documents = []
    base_path = Path(base_directory)

    pdf_files = list(base_path.rglob("*.pdf"))
    print(f"Found {len(pdf_files)} PDF files to process")

    for pdf_file in pdf_files:
        try:
            print(f"Processing: {pdf_file.name}")

            loader = PyPDFLoader(str(pdf_file))
            documents = loader.load()

            subject = pdf_file.parent.name

            for doc in documents:
                doc.metadata["source_file"] = pdf_file.name
                doc.metadata["file_type"] = "pdf"
                doc.metadata["subject"] = subject

            all_documents.extend(documents)
            print(f"Loaded {len(documents)} pages")

        except Exception as e:
            print(f"Error processing {pdf_file.name}: {e}")

    print(f"Total documents loaded: {len(all_documents)}")
    return all_documents


# ---------------------------------------------------------
# 2. Split documents into chunks
# ---------------------------------------------------------
def split_documents(
    documents,
    chunk_size: int = 500,
    chunk_overlap: int = 100
):
    """
    Split documents into smaller chunks for RAG
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    split_docs = splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(split_docs)} chunks")

    return split_docs


# ---------------------------------------------------------
# 3. Ingest into Vector Store
# ---------------------------------------------------------
def ingest_data():
    print("Starting data ingestion pipeline...")

    # Load PDFs
    documents = process_all_pdfs(DATA_DIR)

    # Split into chunks
    chunks = split_documents(documents)

    # Generate embeddings
    embedding_manager = EmbeddingManager()
    texts = [doc.page_content for doc in chunks]
    embeddings = embedding_manager.generate_embeddings(texts)

    # Store in vector DB
    vector_store = VectorStore()
    vector_store.add_documents(chunks, embeddings)

    print("Data ingestion completed successfully ✅")


# ---------------------------------------------------------
# Run once
# ---------------------------------------------------------
if __name__ == "__main__":
    ingest_data()
