import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
import os
# INDEX_PATH = 'data/faiss_index.index'
# DOCSTORE_PATH = 'data/docstore.json'
EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
QDRANT_HOST = os.getenv('QDRANT_HOST', 'localhost')
QDRANT_PORT = int(os.getenv('QDRANT_PORT', '6333'))
COLLECTION_NAME = "papers"

class Retriever:
    def __init__(self, top_k=5):
        self.client = QdrantClient(host=QDRANT_HOST, port = QDRANT_PORT)
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.top_k = top_k
        self.collection_name = COLLECTION_NAME

    def retrieve(self, query):
        """Retrieve relevant chunks from Qdrant"""

        query_vec = self.model.encode(query).tolist()

        try:
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vec,
                limit=self.top_k,
                with_payload=True
            )

            # extract contents from results
            chunks = []
            for result in results.points:
                if result.payload:
                    chunks.append(result.payload.get('content', ''))
            return "\n".join(chunks)
        
        except Exception as e:
            print(f"[ERROR] Qdrant retrieval failed: {e}")

retriever_instance = Retriever()

def retrieve_relevant_chunks(query):
    return retriever_instance.retrieve(query)
