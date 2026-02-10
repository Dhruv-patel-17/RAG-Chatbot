PROMPT_TEMPLATE = """
You are an experienced university question-paper setter.

USER QUERY:
{user_query}

TASK:
Using ONLY the provided Previous Year Question (PYQ) context,
respond according to the USER QUERY.

RULES:
- ONLY generate QUESTIONS
- Do NOT explain concepts
- Do NOT add answers
- Do NOT summarize

MODE 1: FULL QUESTION PAPER
- If the user asks for a question paper or predicted paper,
  generate a COMPLETE, WELL-FORMATTED question paper.

EXAM FORMAT (ONLY FOR MODE 1):
--------------------------------
SECTION A - Very Short Answer (2 marks × 10)
SECTION B - Short Answer (5 marks × 6)
SECTION C - Long Answer (10 marks × 4)
--------------------------------

MODE 2: TOPIC-ONLY QUESTIONS
- If the user asks for a PARTICULAR TOPIC,
  return ONLY the questions related to that topic.
- Do NOT format as a paper.
- Return a numbered list only.

CONTEXT (PYQs):
{context}

NOW RESPOND.
"""


def rag_simple(query, retriever, subject, llm, top_k=5):
    results = retriever.retrieve(
        query,
        subject=subject,
        top_k=top_k
    )

    context = "\n\n".join(
        doc["content"] for doc in results
    ) if results else ""

    if not context.strip():
        return "No relevant questions found."

    prompt = PROMPT_TEMPLATE.format(
        user_query=query,
        context=context
    )

    response = llm.invoke(prompt)
    return response.content
