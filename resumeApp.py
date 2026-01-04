from textUtils import extract_text
import streamlit as st

st.title("Resume Analyzer")

resume_file = st.file_uploader("Upload the Resume", type = ["pdf","docx"])
job_file = st.file_uploader("Upload job Description", type=["pdf","txt"])






    