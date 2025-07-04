import os
import traceback
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from app.core.retriever import retrieve_relevant_chunks
from together import Together

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# client = InferenceClient(
#     provider="hf-inference",
#     api_key=HF_TOKEN
# )
# client = InferenceClient(
#     provider="together",
#     api_key=HF_TOKEN
# )
# from together import Together
client = Together()
os.getenv("TOGETHER_API_KEY")
# response = client.chat.completions.create(
#     model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
#     messages=[],
#     stream=True
# )
# for token in response:
#     if hasattr(token, 'choices'):
#         print(token.choices[0].delta.content, end='', flush=True)

MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"

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
    conversation+=f"User: {query}\Assistant:"
    
    prompt = (
        f"### Context:\n{context}\n\n"
        f"### User Question:\n{query}\n\n"
        f"### Assistant Response:"
    )
    print(f"[DEBUG] Prompt for model:\n{prompt}\n")

    try:
        full_response = ""
        stream = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "user", "content": prompt}
            ],
            stream=True,
        )

        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta:
                token = chunk.choices[0].delta.content
                if token:
                    full_response += token
                    yield token
        print(f'[DEBUG] Assistant response: \n{full_response}')
    
    except Exception as e:
        print("[ERROR] Exception during generation:")
        traceback.print_exc()
        yield f"Error: {type(e).__name__} - {str(e)}"


# load_dotenv()
# hf_token = os.getenv("HUGGINGFACE_TOKEN")

# def generate_answer_stream(query: str):
#     context = retrieve_relevant_chunks(query)
#     if not context:
#         yield "Sorry, I couldn't find relevant information."
#         return

#     # qa_pipeline = pipeline(
#     #     "question-answering",
#     #     model="deepset/roberta-base-squad2",  # or any gated model
#     #     use_auth_token=hf_token
#     # )
#     qa_pipeline = pipeline(
#         "text2text-generation",
#         model="google/flan-t5-xl",
#         max_length=512,
#         temperature=0.7
#     )
#     answer = qa_pipeline({
#         "question": query,
#         "context": context
#     }).get("answer", "No answer generated.")

#     for token in answer.split():
#         yield token + " "
#         sleep(0.05)
