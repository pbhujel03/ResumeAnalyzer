import streamlit as st
import faiss
import numpy as np
from textUtils import extract_text, chunk_text
from embeddings import create_embeddings, build_faiss_index, search_faiss 
import ollama

st.title("Resume Analyzer")

resume_file = st.file_uploader("Upload the Resume", type = ["pdf","docx"])
job_file = st.file_uploader("Upload job Description", type=["pdf","txt"])

if resume_file and job_file:

    resume_text = extract_text(resume_file)
    job_text = extract_text(job_file)

    resume_chunks = chunk_text(resume_text)
    job_chunks = chunk_text(job_text)

    resume_embeddings = create_embeddings(resume_chunks)
    job_embeddings = create_embeddings(job_chunks)

    job_index = build_faiss_index(job_embeddings)
    distances, indices = search_faiss(job_index, resume_embeddings, top_k = 1)

    #compute match percentage
    top_scores = [1- d/2 for d in distances.flatten()]
    match_percentage = round(np.mean(top_scores)* 100,2)

    #LLM explanation using Ollama
    prompt = f"""
    I have a resume and a job description.
    The resume match percentage is {match_percentage}%.
    Resume: {resume_text}
    job Description: {job_text}
    

    Explain in a clear way:
    1. why this resume matches the job.
    2. what improvements can be made to better match the job.
    """

    response = ollama.chat(
        model="llama3",
        messages=[{"role": "user", "content": prompt}]
    )
    llm_response = response["message"]["content"]


    #Display results
    st.subheader(f"Match Percentage: {match_percentage}%")
    # explanation = explain_result(match_percentage)
    # st.write(explanation)

    st.subheader("LLM explanation:")
    st.write(llm_response)
    






    