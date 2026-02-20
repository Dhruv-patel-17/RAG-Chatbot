import os
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import re

from embedding_manager import EmbeddingManager
from vector_store import VectorStore

DATA_DIR = "./data"


def extract_header(text: str):
    header = {}

    try:
        clean_text = text.replace("\n", " ")

        # University
        uni_match = re.search(r"(GUJARAT TECHNOLOGICAL UNIVERSITY)", clean_text, re.I)
        header["university"] = uni_match.group(1).strip() if uni_match else "Unknown University"

        # Exam Name
        exam_match = re.search(r"(BE\s*-\s*SEMESTER.*?EXAMINATION\s*-\s*\w+\s*\d{4})", clean_text, re.I)
        header["exam"] = exam_match.group(1).strip() if exam_match else "Unknown Exam"

        # Subject Code
        code_match = re.search(r"Subject\s*Code\s*:\s*(\d{5,10})", clean_text, re.I)
        header["subject_code"] = code_match.group(1) if code_match else "Unknown Code"

        # Subject Name
        name_match = re.search(r"Subject\s*Name\s*:\s*([A-Za-z0-9 ,\-()]+)", clean_text, re.I)
        header["subject_name"] = name_match.group(1).strip() if name_match else "Unknown Subject"

        # Date
        date_match = re.search(r"Date\s*:\s*([\d/\-]+)", clean_text)
        header["date"] = date_match.group(1) if date_match else "Unknown Date"

        # Time
        time_match = re.search(r"Time\s*:\s*([^T]+?)Total", clean_text, re.I)
        header["time"] = time_match.group(1).strip() if time_match else "Unknown Time"

        # Total Marks
        marks_match = re.search(r"Total\s*Marks\s*:\s*(\d+)", clean_text, re.I)
        header["marks"] = marks_match.group(1) if marks_match else "Unknown Marks"

    except Exception as e:
        print("Header extraction error:", e)

    return header

def extract_year(filename: str):
    year_match = re.search(r"(19|20)\d{2}", filename)
    return int(year_match.group()) if year_match else None


def extract_session(filename: str):
    name = filename.lower()
    if "winter" in name:
        return "Winter"
    if "summer" in name:
        return "Summer"
    return None


def process_all_pdfs(base_directory: str):
    all_documents = []
    base_path = Path(base_directory)

    pdf_files = list(base_path.rglob("*.pdf"))
    print(f"Found {len(pdf_files)} PDF files to process")

    for pdf_file in pdf_files:
        try:
            print(f"Processing: {pdf_file.name}")

            loader = PyPDFLoader(str(pdf_file))
            documents = loader.load()

            # remove empty pages
            documents = [doc for doc in documents if doc.page_content.strip()]
            if not documents:
                print(f"Skipping empty PDF: {pdf_file.name}")
                continue

            # Extract year/session/subject
            year = extract_year(pdf_file.name)
            session = extract_session(pdf_file.name)
            subject = pdf_file.parent.name

            if not year or not session:
                print(f" Invalid filename format: {pdf_file.name}")
                continue

            # Extract header from FIRST PAGE
            header = extract_header(documents[0].page_content)

            # Generate unique paper id
            paper_id = f"{subject}_{year}_{session}"

            # Attach metadata to ALL pages
            for doc in documents:

                doc.metadata.update({
                    "source_file": pdf_file.name,
                    "file_type": "pdf",
                    "subject": subject,
                    "year": year,
                    "session": session,
                    "paper_id": paper_id,

                    # header metadata
                    "university": header.get("university"),
                    "exam": header.get("exam"),
                    "subject_code": header.get("subject_code"),
                    "subject_name": header.get("subject_name"),
                    "date": header.get("date"),
                    "time": header.get("time"),
                    "marks": header.get("marks")
                })

                # REMOVE original Q numbering (we will reformat later)
                doc.page_content = re.sub(r"\bQ\.?\s*\d+\b", "", doc.page_content)

            all_documents.extend(documents)

            print(f"Loaded {len(documents)} pages → {subject} ({year} {session})")

        except Exception as e:
            print(f"Error processing {pdf_file.name}: {e}")

    print(f"\nTotal documents loaded: {len(all_documents)}")
    return all_documents


def split_documents(documents, chunk_size: int = 500, chunk_overlap: int = 100):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = splitter.split_documents(documents)

    # Remove empty chunks
    chunks = [c for c in chunks if c.page_content.strip()]

    print(f"Total chunks created: {len(chunks)}")
    return chunks

def ingest_data():
    print("Starting ingestion pipeline...")

    documents = process_all_pdfs(DATA_DIR)
    if not documents:
        print(" No documents found. Aborting.")
        return

    chunks = split_documents(documents)
    if not chunks:
        print(" No chunks created. Aborting.")
        return

    embedding_manager = EmbeddingManager()
    texts = [doc.page_content for doc in chunks]

    embeddings = embedding_manager.generate_embeddings(texts)
    if len(embeddings) == 0:
        print("Embedding generation failed.")
        return

    vector_store = VectorStore()
    vector_store.add_documents(chunks, embeddings)

    print("\n Data ingestion completed successfully")


if __name__ == "__main__":
    ingest_data()

