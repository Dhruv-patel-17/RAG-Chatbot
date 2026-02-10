from flask import Flask, render_template, request
from rag import rag_simple
from retriever import RAGRetriever  # your retriever object
from langchain_groq import ChatGroq
import os

app = Flask(__name__)

# Initialize LLM
llm = ChatGroq(
    groq_api_key="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",
    model_name="llama-3.1-8b-instant",
    temperature=0.3,
    max_tokens=1200
)

from vector_store import VectorStore
from embedding_manager import EmbeddingManager

embedding_manager = EmbeddingManager()
vector_store = VectorStore()

retriever = RAGRetriever(
    vector_store=vector_store,
    embedding_manager=embedding_manager
)

@app.route("/", methods=["GET", "POST"])
def index():
    response = ""
    if request.method == "POST":
        query = request.form.get("query")
        subject = request.form.get("subject")

        response = rag_simple(
            query=query,
            retriever=retriever,
            subject=subject,
            llm=llm
        )

    return render_template("index.html", response=response)


if __name__ == "__main__":
    app.run(debug=True)
