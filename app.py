from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import pickle
import pdfplumber
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

# ---------------- CORS FIX ----------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Local dev
        "https://job-matcher-frontend.onrender.com",  # Deployed frontend
        "https://job-matcher-backend-s5cf.onrender.com"  # Deployed backend
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- LOAD MODELS ----------------
with open("model/job_model.pkl", "rb") as f:
    config = pickle.load(f)

model = SentenceTransformer("model/my_model")
common_skills = config["common_skills"]

# ---------------- HELPERS ----------------
def extract_text(file):
    text = ""
    with pdfplumber.open(file.file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9 ]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

def extract_skills(text):
    return [s for s in common_skills if s in text]

def skill_match(a, b):
    if not b:
        return 0
    return len(set(a).intersection(set(b))) / len(b)

# ---------------- API ----------------
@app.post("/match")
async def match(
    cv: UploadFile = File(...),
    job_description: str = Form(...)
):

    cv_text = extract_text(cv)

    cv_clean = clean_text(cv_text)
    job_clean = clean_text(job_description)

    cv_emb = model.encode([cv_clean])
    job_emb = model.encode([job_clean])

    semantic_score = cosine_similarity(cv_emb, job_emb)[0][0]

    cv_skills = extract_skills(cv_clean)
    job_skills = extract_skills(job_clean)

    skill_score = skill_match(cv_skills, job_skills)

    final_score = 0.7 * semantic_score + 0.3 * skill_score

    return {
        "semantic_score": float(semantic_score),
        "skill_score": float(skill_score),
        "final_score": float(final_score),
        "cv_skills": cv_skills,
        "job_skills": job_skills
    }