import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer

INDEX_PATH = 'data/faiss_index.index'
DOCSTORE_PATH = 'data/docstore.json'
EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'

class Retriever:
    def __init__(self, top_k=5):
        self.index = faiss.read_index(INDEX_PATH)
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        with open(DOCSTORE_PATH, 'r', encoding='utf-8') as f:
            self.docstore = json.load(f)
        self.top_k = top_k

    def retrieve(self, query):
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(query_embedding, self.top_k)
        results = []
        for idx in indices[0]:
            if idx < len(self.docstore):
                results.append(self.docstore[idx]["text"])
        return "\n".join(results)

retriever_instance = Retriever()

def retrieve_relevant_chunks(query):
    return retriever_instance.retrieve(query)
