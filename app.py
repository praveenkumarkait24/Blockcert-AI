from flask import Flask, render_template, request, redirect, url_for, session, send_file
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np
import os
import io # Required for handling binary file streams

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

from db import init_db, db
from models import Candidate, Job, Match

os.environ["TOKENIZERS_PARALLELISM"] = "false"

app = Flask(__name__)
app.secret_key = "super_secret_key_blockcert_ai" 
init_db(app)

with app.app_context():
    db.create_all()

model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L3-v2")

SKILL_VOCAB = [
    "python", "java", "c", "c++", "c#", "javascript", "typescript",
    "go", "rust", "kotlin", "swift", "php", "ruby", "html", "css", 
    "react", "angular", "vue", "nextjs", "nodejs", "express", "django", 
    "flask", "spring boot", "sql", "mysql", "postgresql", "mongodb", 
    "redis", "machine learning", "deep learning", "nlp", "data analysis",
    "pandas", "numpy", "pytorch", "tensorflow", "computer vision",
    "git", "github", "gitlab", "docker", "kubernetes", "aws", "azure", 
    "gcp", "ci/cd", "power bi", "tableau", "excel", "problem solving", 
    "communication", "leadership", "teamwork", "time management"
]

def embed(text: str) -> np.ndarray:
    if not text:
        return np.zeros((1, 384))
    return np.array(model.encode([text]))

def compute_match_score(resume_text: str, job_text: str) -> float:
    e1 = embed(resume_text)
    e2 = embed(job_text)
    sim = cosine_similarity(e1, e2)[0][0]
    score = float(max(0.0, min(1.0, (sim + 1) / 2))) * 100
    return round(score, 2)

def extract_skills(text: str):
    text_lower = text.lower()
    found = []
    for s in SKILL_VOCAB:
        if s in text_lower:
            found.append(s)
    return sorted(list(set(found)))

def compute_skill_gap(candidate_skills, job_skills):
    candidate_set = {s.lower() for s in candidate_skills}
    job_set = {s.lower() for s in job_skills}
    missing = list(job_set - candidate_set)
    return sorted(missing)

def extract_text_from_pdf(pdf_file):
    if not PdfReader:
        return ""
    try:
        reader = PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return ""

# --- NEW ROUTE TO VIEW PDF ---
@app.route('/resume/<int:candidate_id>')
def view_resume(candidate_id):
    candidate = Candidate.query.get_or_404(candidate_id)
    if not candidate.resume_data:
        return "No resume uploaded for this candidate", 404
    
    # Serve the binary data as a PDF file
    return send_file(
        io.BytesIO(candidate.resume_data),
        mimetype='application/pdf',
        as_attachment=False, # False opens in browser, True downloads
        download_name=f"{candidate.name}_resume.pdf"
    )

@app.route("/", methods=["GET", "POST"])
def index():
    error_message = None
    
    if request.method == "POST":
        candidate_name = request.form.get("candidate_name", "Anonymous")
        candidate_email = request.form.get("candidate_email", "")
        job_text = request.form.get("job_text", "")
        
        resume_text = ""
        resume_binary = None # Variable to hold the PDF binary data

        if "resume_pdf" in request.files:
            file = request.files["resume_pdf"]
            if file.filename != "":
                # 1. Read the binary data to save in DB
                resume_binary = file.read()
                # 2. Reset cursor to start so PDF reader can read it
                file.seek(0)
                # 3. Extract text
                resume_text = extract_text_from_pdf(file)
        
        if not resume_text:
             resume_text = request.form.get("resume_text", "")

        if not resume_text or not job_text:
            error_message = "Please provide both a resume (PDF) and a job description."
        else:
            cand_skills = extract_skills(resume_text)
            job_skills = extract_skills(job_text)
            match_score = compute_match_score(resume_text, job_text)
            missing_skills = compute_skill_gap(cand_skills, job_skills)

            candidate = Candidate.query.filter_by(email=candidate_email).first()
            if not candidate:
                candidate = Candidate(
                    name=candidate_name,
                    email=candidate_email,
                    resume_text=resume_text,
                    skills=",".join(cand_skills),
                    resume_data=resume_binary # Save PDF to DB
                )
                db.session.add(candidate)
            else:
                candidate.name = candidate_name
                candidate.resume_text = resume_text
                candidate.skills = ",".join(cand_skills)
                if resume_binary:
                    candidate.resume_data = resume_binary # Update PDF if new one uploaded
            
            job = Job.query.filter_by(title="Hackathon Job Role").first()
            if not job:
                job = Job(title="Hackathon Job Role", description=job_text, skills_required=",".join(job_skills))
                db.session.add(job)
            else:
                job.description = job_text
                job.skills_required = ",".join(job_skills)

            db.session.commit()

            match = Match.query.filter_by(candidate_id=candidate.id, job_id=job.id).first()
            if not match:
                match = Match(
                    candidate_id=candidate.id, 
                    job_id=job.id, 
                    match_score=match_score, 
                    missing_skills=",".join(missing_skills)
                )
                db.session.add(match)
            else:
                match.match_score = match_score
                match.missing_skills = ",".join(missing_skills)
            
            db.session.commit()

            session['analysis_result'] = {
                "match_score": match_score,
                "cand_skills": cand_skills,
                "job_skills": job_skills,
                "missing_skills": missing_skills
            }
            
            return redirect(url_for('index'))

    result = session.pop('analysis_result', None)
    matches = Match.query.order_by(Match.match_score.desc()).limit(10).all()

    return render_template("index.html", result=result, matches=matches, error_message=error_message)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)