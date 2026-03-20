import re
import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="AI Resume Match Engine", page_icon="🧠", layout="wide")

st.title("🧠 AI Resume ↔ Job Match Engine")
st.write("Paste a resume and a job description. Get a match score, missing skills, and improvement tips.")

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

SKILLS = [
    "python", "machine learning", "deep learning", "nlp", "computer vision", "pandas",
    "numpy", "scikit-learn", "tensorflow", "pytorch", "streamlit", "fastapi",
    "sql", "mysql", "postgresql", "data analysis", "feature engineering",
    "model deployment", "docker", "git", "github", "aws", "azure", "gcp",
    "classification", "regression", "clustering", "time series", "api", "rest api"
]

def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()

def extract_skills(text: str):
    text = clean_text(text)
    found = []
    for skill in SKILLS:
        if re.search(rf"\b{re.escape(skill)}\b", text):
            found.append(skill)
    return sorted(set(found))

def embedding_score(resume_text: str, job_text: str) -> float:
    vectors = model.encode([resume_text, job_text], normalize_embeddings=True)
    score = cosine_similarity([vectors[0]], [vectors[1]])[0][0]
    return float(score)

col1, col2 = st.columns(2)

with col1:
    resume = st.text_area(
        "Resume text",
        height=320,
        placeholder="Paste the candidate resume here..."
    )

with col2:
    job_desc = st.text_area(
        "Job description",
        height=320,
        placeholder="Paste the job description here..."
    )

if st.button("Analyze match", use_container_width=True):
    if not resume.strip() or not job_desc.strip():
        st.error("Please paste both resume and job description.")
    else:
        resume_skills = extract_skills(resume)
        job_skills = extract_skills(job_desc)

        common = sorted(set(resume_skills) & set(job_skills))
        missing = sorted(set(job_skills) - set(resume_skills))
        score = embedding_score(resume, job_desc)

        final_score = round((score * 100), 2)

        st.subheader("Match Result")
        st.metric("Overall Match Score", f"{final_score}%")

        st.progress(min(max(score, 0), 1))

        c1, c2, c3 = st.columns(3)
        c1.metric("Resume skills found", len(resume_skills))
        c2.metric("Job skills found", len(job_skills))
        c3.metric("Common skills", len(common))

        st.markdown("### Matched skills")
        if common:
            st.success(", ".join(common))
        else:
            st.warning("No direct skill overlap detected.")

        st.markdown("### Missing skills to improve fit")
        if missing:
            st.info(", ".join(missing))
        else:
            st.success("No obvious missing skills detected.")

        st.markdown("### Improvement tips")
        tips = []
        if "docker" in missing:
            tips.append("Add Docker to show deployment readiness.")
        if "fastapi" in missing or "api" in missing:
            tips.append("Build one API-based ML project with FastAPI.")
        if "streamlit" in missing:
            tips.append("Deploy one project with Streamlit for portfolio visibility.")
        if "pytorch" in missing and "tensorflow" in missing:
            tips.append("Show one deep learning project in either PyTorch or TensorFlow.")
        if not tips:
            tips.append("Add measurable outcomes, datasets, and deployment links to your resume.")

        for tip in tips:
            st.write(f"- {tip}")

        st.markdown("### Resume-ready summary")
        st.write(
            f"This profile shows an approximate {final_score}% alignment with the target role. "
            f"It matches {len(common)} core skills and highlights {len(missing)} skill gaps."
        )