import streamlit as st

from textUtils import extract_text, chunk_text
from embeddings import create_embeddings
from similarity import (
    compare_embeddings,
    sort_scores,
    get_top_matches,
    compute_match_percentage,
    explain_result,
    suggest_improvements
)

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

    #compare similarity
    scores = compare_embeddings(resume_embeddings, job_embeddings)
    sorted_scores = sort_scores(scores)
    top_matches = get_top_matches(sorted_scores)

    #final score
    match_percentage = compute_match_percentage(top_matches)
    explanation = explain_result(match_percentage)

    #suggestions
    improvements = suggest_improvements(resume_text, job_text)

    #Display results
    st.subheader(f"Match Percentage: {match_percentage}%")
    st.write(explanation)

    if improvements:
        st.subheader("Suggestions to Improve:")
        for skill in improvements:
            st.write(f"- {skill}")






    