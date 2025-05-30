
import streamlit as st
import os
import re
import PyPDF2
import pdfplumber
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Function to extract text from PDF using pdfplumber
def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text

# Function to calculate similarity score
def calculate_score(resume_text, jd_text):
    documents = [resume_text, jd_text]
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
    score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return round(score * 100, 2)

# Load JD dataset
@st.cache_data
def load_dataset():
    return pd.read_csv('job-descriptions/training_data.csv')

# Streamlit App UI
st.set_page_config(page_title="Resume Scorer", layout="centered")
st.title("📄 Resume vs Job Description Scorer")

uploaded_resume = st.file_uploader("Upload your Resume (PDF)", type=["pdf"])
jd_option = st.radio("Choose how to input Job Description:", ["Upload from CSV", "Paste manually"])

if jd_option == "Upload from CSV":
    data = load_dataset()
    st.write("Sample Job Descriptions from Dataset:")
    st.dataframe(data.head())
    jd_index = st.number_input("Enter the index of the JD you want to use", min_value=0, max_value=len(data)-1, step=1)
    jd_text = data.iloc[jd_index]['Job Description']
else:
    jd_text = st.text_area("Paste the Job Description here", height=200)

if uploaded_resume is not None and jd_text:
    resume_text = extract_text_from_pdf(uploaded_resume)
    score = calculate_score(resume_text, jd_text)
    st.success(f"✅ Resume Match Score: {score}%")
