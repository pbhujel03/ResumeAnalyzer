from sentence_transformers import SentenceTransformer
from textUtils import extract_text, chunk_text
import numpy as np

#load the embeddings
model= SentenceTransformer("all-MiniLM-L6-v2")

def create_embeddings(chunks):
    """
    Convert list of text chunks into embeddings
    """
    return model.encode(chunks,convert_to_numpy=True)

if __name__ == "__main__":
    resume_text = extract_text(resume_file)
    job_text = extract_text(job_file)

    resume_chunks = chunk_text(resume_text, max_length=20)
    job_chunks = chunk_text(job_text, max_length=20)
