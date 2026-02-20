import re


PROMPT_TEMPLATE = """
You are a STRICT university exam paper generator.

You must ONLY use questions that exist inside the PYQ context.

You are NOT allowed to:
• invent questions
• modify wording
• repeat questions
• change marks
• change wording
• change order unless instructed

------------------------------------
USER QUERY:
{user_query}

PYQ CONTEXT:
{context}


MODE DETECTION (STRICT RULE)


FIRST analyze the USER QUERY carefully.

If the query contains words like:
paper, full paper, question paper, model paper, predicted paper

→ USE FULL PAPER MODE

Otherwise
→ USE TOPIC MODE



FULL PAPER MODE


Generate a complete question paper using ONLY most frequent PYQ questions.

HEADER RULE:
Copy the FULL header EXACTLY from the PYQ context.

Paper must follow structure:

Q.1
Q.2
Q.3
Q.4
Q.5

Each question must contain:
(a) 3 marks
(b) 4 marks
(c) 7 marks
OR alternatives exactly as in source.
do not write any question no. which is written in the original data

SELECTION RULES:
• Each question must be different
• No repetition
• Use only context questions
• Distribute topics evenly
• Include COMPLETE original question text
• Do NOT remove tables, figures, values, data, diagrams, case info, or any additional lines
• Preserve formatting exactly as present

NUMBERING RULE:
Renumber sequentially Q1–Q5 regardless of original numbering.


================================
TOPIC MODE
================================

Return ONLY questions that contain only the requested topic.

FORMAT:
Numbered list:
1.
2.
3.

RULES:
• Include ONLY questions that contain topic words from query
• Preserve original wording exactly
• Preserve original order from context
• Do NOT generate paper format
• Do NOT add header
• Do NOT add explanations
• Do NOT add notes
• Do NOT add labels
• Output ONLY the questions

STRICT OUTPUT FORMAT:
Return ONLY the final list of questions.
No explanation.
No introduction.
No analysis.
No headings.

FAIL SAFE


If answer cannot be produced from context:

Return EXACTLY:
Requested questions are not available in the provided PYQ context.
"""








import re

# ----------------- HEADER FORMATTING -----------------
def format_header(meta: dict) -> str:
    """
    Formats the exam paper header using metadata from the first chunk.
    """
    university = meta.get("university", "GUJARAT TECHNOLOGICAL UNIVERSITY")
    exam = meta.get("exam", "BE - SEMESTER–I & II(NEW) EXAMINATION")
    session = meta.get("session", "SESSION")
    year = meta.get("year", "YYYY")
    subject_code = meta.get("subject_code", "CODE")
    date = meta.get("date", "DD-MM-YYYY")
    subject_name = meta.get("subject_name", "SUBJECT NAME")
    time = meta.get("time", "HH:MM TO HH:MM")
    marks = meta.get("marks", "TOTAL MARKS")

    return f"""
    Seat No.: ________ Enrolment No.___________ 
    
    {university}
    {exam} – {session} {year}
    
    Subject Code:{subject_code}      Date:{date}
    Subject Name:{subject_name}
    Time:{time}      Total Marks:{marks}
    
    Instructions:
    1. Attempt all questions.
    2. Make suitable assumptions wherever necessary.
    3. Figures to the right indicate full marks.
    4. Simple and non-programmable scientific calculators are allowed.
    """

# ----------------- YEAR / SESSION EXTRACTION -----------------
def extract_year_session(query: str):
    year = None
    session = None

    year_match = re.search(r"(19|20)\d{2}", query)
    if year_match:
        year = int(year_match.group())

    q = query.lower()
    if "winter" in q:
        session = "Winter"
    elif "summer" in q:
        session = "Summer"

    return year, session

# ----------------- RAG SIMPLE -----------------
def rag_simple(query, retriever, subject, llm, top_k=10):
    """
    Retrieve questions and generate prompt for RAG.
    Ensures header info is preserved for full paper.
    """
    year, session = extract_year_session(query)

    # Retrieve chunks from vector store
    results = retriever.retrieve(
        query=query,
        subject=subject,
        year=year,
        session=session,
        top_k=top_k,
        score_threshold=0.0   # disable filtering for full paper retrieval
    )

    if not results:
        return "Requested questions are not available in the provided PYQ context."

    # 🔹 Extract header from first chunk using format_header
    first_metadata = results[0].get("metadata", {})
    header_text = format_header(first_metadata)

    # 🔹 Build context text including questions
    context_text = header_text + "\n\n" + "\n\n".join(
        doc['content'] for doc in results
    )

    prompt = PROMPT_TEMPLATE.format(
        user_query=query,
        context=context_text
    )

    # Generate response from LLM
    response = llm.invoke(prompt)
    return response.content


