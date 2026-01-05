import numpy as np

def cosine_similarity(vec1, vec2):
    """
    Compute Cosine Similarity between two vectors
    """
    dot_product = np.dot(vec1,vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)

    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0

    return dot_product / (norm_vec1*norm_vec2)
   
def compare_embeddings(resume_embeddings, job_embeddings):
    """
    Compare each resume chunk with each job chunk
    """
    scores = []

    for i,r_vec in enumerate(resume_embeddings):
        for j, j_vec in enumerate(job_embeddings):
            score = cosine_similarity(r_vec,j_vec)
            scores.append({
                "resume_chunk": i,
                "job_chunk": j,
                "score": score
            })

return scores
