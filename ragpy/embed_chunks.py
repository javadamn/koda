# rag/embed_chunks.py
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import numpy as np

def embed_chunks(chunks, model_name='all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    texts = [chunk["text"] for chunk in chunks]
    return model.encode(texts, convert_to_numpy=True)

def build_faiss_index(chunks, embeddings, index_path="outputs/faiss_index"):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    faiss.write_index(index, f"{index_path}.index")
    with open(f"{index_path}_chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

def load_faiss_index(index_path="outputs/faiss_index"):
    index = faiss.read_index(f"{index_path}.index")
    with open(f"{index_path}_chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    return index, chunks
