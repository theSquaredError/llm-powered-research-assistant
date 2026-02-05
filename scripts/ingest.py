import os
import json
import faiss
import glob
import numpy as np
from sentence_transformers import SentenceTransformer
from app.core.utils import chunk_text, extract_text_from_pdf

EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
DATA_DIR = 'data/arxiv_papers'
INDEX_PATH = 'data/faiss_index.index'
DOCSTORE_PATH = 'data/docstore.json'
OUT_DIR = 'data/'

def load_documents(directory):
    documents = []
    for filepath in glob.glob(os.path.join(directory, '*.pdf')):
        text = extract_text_from_pdf(filepath)
        documents.append({"filename": os.path.basename(filepath), "content": text})
    return documents

def load_documents(directory):
    
    pass

def build_index():
    model = SentenceTransformer(EMBEDDING_MODEL)
    docs = load_documents(DATA_DIR)
    all_chunks = []
    metadata = []

    for doc in docs:
        chunks = chunk_text(doc['content'])
        all_chunks.extend(chunks)
        metadata.extend([{"text": chunk, "source": doc['filename']} for chunk in chunks])

    embeddings = model.encode(all_chunks, convert_to_numpy=True)
    dim = embeddings.shape[1]

    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    faiss.write_index(index, INDEX_PATH)
    with open(DOCSTORE_PATH, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

    print(f"Indexed {len(all_chunks)} chunks from {len(docs)} documents.")


if __name__ == '__main__':
    build_index()