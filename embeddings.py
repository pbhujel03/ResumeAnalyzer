from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

#load the embeddings
model= SentenceTransformer("all-MiniLM-L6-v2")

def create_embeddings(chunks):
    """
    Convert list of text chunks into embeddings
    """
    return model.encode(chunks,convert_to_numpy=True)

def build_faiss_index(embeddings):
    """
    Create Faiss Index from embeddings
    """
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index
    
def search_faiss(index, query_embeddings, top_k = 5):
    """
    Search FAISS index from closest Vectors
    """    
    distances, indices = index.search(query_embeddings, top_k)
    return distances, indices






