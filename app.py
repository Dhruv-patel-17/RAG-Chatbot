


from flask import Flask, render_template, request
from rag import rag_simple
from retriever import RAGRetriever
from langchain_groq import ChatGroq


from vector_store import VectorStore
from embedding_manager import EmbeddingManager


import os
# ------------------ LLM ------------------
from flask import send_file,session
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Preformatted
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
import io
from dotenv import load_dotenv
load_dotenv()
app = Flask(__name__)
app.secret_key = "b3619bf36809d94d647846c9a2bbbc0deab2931be00f164a5d5d7da131240c56"
# ------------------ PDF DOWNLOAD ROUTE ------------------
@app.route("/download_pdf")
def download_pdf():

    paper_text = session.get("generated_paper", "No paper generated yet.")

    buffer = io.BytesIO()

    doc = SimpleDocTemplate(buffer, pagesize=letter)

    styles = getSampleStyleSheet()
    elements = []

    for line in paper_text.split("\n"):
        elements.append(Preformatted(line, styles["Normal"]))
        elements.append(Spacer(1, 6))

    doc.build(elements)

    buffer.seek(0)

    return send_file(
        buffer,
        as_attachment=True,
        download_name="question_paper.pdf",
        mimetype="application/pdf"
    )




llm = ChatGroq(
    api_key = os.getenv("GROQ_API_KEY"),
    model_name="llama-3.1-8b-instant",
    temperature=0.3,
    max_tokens=1200
)



# ------------------ RAG SETUP ------------------
embedding_manager = EmbeddingManager()
vector_store = VectorStore()

retriever = RAGRetriever(
    vector_store=vector_store,
    embedding_manager=embedding_manager
)

# ------------------ ROUTES ------------------
@app.route("/", methods=["GET", "POST"])
def index():
    response = ""

    if request.method == "POST":
        subject = request.form.get("subject")
        query_type = request.form.get("query_type")
        year = request.form.get("year")
      
        topic = request.form.get("topic")
        custom_query = request.form.get("custom_query")

        # ---------- BUILD USER QUERY ----------
        user_query = ""

        if query_type == "probable":
            user_query = "Generate a probable full question paper"

        elif query_type == "topic":
            if not topic:
                response = "Please enter a topic."
                return render_template("index.html", response=response)
            user_query = f"Give me the questions related to {topic}"

      
            

        else:
            response = "Invalid query type selected."
            return render_template("index.html", response=response)

        # ---------- RAG CALL ----------
        response = rag_simple(
            query=user_query,
            retriever=retriever,
            subject=subject,
            llm=llm,
            top_k=20 
        )

        session["generated_paper"] = response

    return render_template("index.html", response=response)


# ------------------ MAIN ------------------
if __name__ == "__main__":
    app.run(debug=True)