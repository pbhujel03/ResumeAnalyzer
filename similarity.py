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

def sort_scores(scores):
    return sorted(scores, key= lambda x: x["score"],reverse=True)

#pick top matches
def get_top_matches(sorted_scores, top_k = 5):
    return sorted_scores[:top_k]

#compute overall match percentage
def compute_match_percentage(top_matches):
    if not top_matches:
        return 0
    
    avg_score = sum(item["score"] for item in top_matches)/len(top_matches)
    return round(avg_score *100, 2)

def explain_result(match_percentage):
    if match_percentage >= 80:
        return "Excellent Match! Your resume aligns very well with this job"
    elif match_percentage >= 60:
        return "Good Match. Adding more relevant projects can improve your chances"
    elif match_percentage >= 40:
        return "Average match. Consider improving skills mentioned in the job Description "
    else:
        return "Low match. Your resume needs significant improvements for the role"

def suggest_improvements(resume_text, job_text):
    resume_words = set(resume_text.lower().split())
    job_words = set(job_text.lower().split())

    missing = job_words - resume_words

    important = [w for w in missing if len(w)>4]
    return important[:5]

