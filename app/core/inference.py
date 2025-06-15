from transformers import pipeline
from app.core.retriever import retrieve_relevant_chunks

# qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")


def answer_query(query: str) -> str:
    print(f"Received query: {query}")  # Debugging output
    context = retrieve_relevant_chunks(query)
    print(f"Retrieved context: {context[:500]}...")  # Debugging output
    if not context:
        return "Sorry, I couldn't find relevant information."

    answer = qa_pipeline({
        "question": query,
        "context": context
    })
    print(f"Generated answer: {answer}")  # Debugging output
    return answer.get("answer", "No answer generated.")