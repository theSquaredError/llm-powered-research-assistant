import os
import traceback
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from app.core.retriever import retrieve_relevant_chunks
import requests
import json


def generate_answer_stream(query: str, history: list = None):

    if history is None:
        history = []

    context = retrieve_relevant_chunks(query)
    print(f"[DEBUG] Retrieved context: {context[:200]}...\n")

    if not context:
        yield "Sorry, I couldn't find relevant information."
        return
    conversation = ""
    for q, a in history:
        conversation += f"User: {q}\nAssistant: {a}\n"
    conversation+=f"User: {query}\nAssistant:"
    system_prompt = """You are a research paper assistant. Answer questions ONLY based on the provided context from the research paper. 
    Do not make up facts, assumptions, or information not explicitly stated in the paper. If the answer is not in the provided context, clearly say "This information is not available in the provided paper." Cite relevant parts of the paper when possible."""
    prompt = (
        f"{system_prompt}\n\n"
        f"### Context:\n{context}\n\n"
        f"### Conversation:\n{conversation}\n\n"
        f"### Assistant Response:"
    )
    print(f"[DEBUG] Prompt for model:\n{prompt}\n")
    payload = {
                "model": "my-llama3-gguf",
                "prompt": prompt,
                "stream": True,
                "temperature":0.2,
                "num_ctx":4096,
                "num_predict":256
            }
    try:
        
        with requests.post(
            "http://localhost:11434/api/generate",
            json=payload,
            stream=True,          
        ) as r:
            r.raise_for_status()

            for line in r.iter_lines():
                if line:
                    data = json.loads(line.decode("utf-8"))
                    if data.get("done"):
                        break
                    token = data.get("response")
                    if token:
                        yield token

    except Exception as e:
        print("[ERROR] Exception during generation:")
        traceback.print_exc()
        yield f"Error: {type(e).__name__} - {str(e)}"



