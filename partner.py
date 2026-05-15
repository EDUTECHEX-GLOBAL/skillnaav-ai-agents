from dotenv import load_dotenv
load_dotenv()

import os
import io
import json
import fitz
import boto3
import asyncio
import tempfile
import docx2txt

from fastapi import FastAPI, Form, HTTPException, Query, status, Request, Path, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from datetime import datetime
from urllib.parse import urlparse
from pymongo import MongoClient, UpdateOne
from bson import ObjectId
from bson.errors import InvalidId
# sentence_transformers/PyTorch deferred — imported lazily via embedding_model proxy
import requests
import httpx
from fastapi import BackgroundTasks
from pydantic import BaseModel
from typing import List
from embedding_model import embedder, util  # lazy proxies; PyTorch loads on first request

class ShortlistRequest(BaseModel):
    internship_id: str
    job_description: str
    job_skills: List[str]
    resumes: List[str]



# === Utility ===
def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def convert_object_ids(obj):
    if isinstance(obj, list):
        return [convert_object_ids(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: (str(v) if isinstance(v, ObjectId) else convert_object_ids(v)) for k, v in obj.items()}
    else:
        return obj

def extract_school_admin_id(application):
    for key in ["schoolAdmin", "school_admin_id", "schoolAdminId"]:
        val = application.get(key)
        if isinstance(val, dict) and "$oid" in val:
            return val["$oid"]
        if val:
            return val
    return None

# === Setup ===
print(f"[{now()}] Partner module loaded (DB connects lazily on first request)")

# Lazy MongoDB initialisation — a missing/invalid MONGO_URI would crash the worker
# on import if we connected here.  We defer until first use instead.
_partner_db = None

def _get_partner_db():
    global _partner_db
    if _partner_db is None:
        uri = os.getenv("MONGO_URI", "")
        if not uri:
            raise RuntimeError("MONGO_URI environment variable is not set.")
        _partner_db = MongoClient(uri).get_default_database()
        _partner_db["shortlisted_candidates"].create_index(
            [("internship_id", 1), ("school_admin_id", 1)]
        )
        print(f"[{now()}] Connected to MongoDB (partner): {_partner_db.name}")
    return _partner_db

def _shortlist_collection():    return _get_partner_db()["shortlisted_candidates"]
def _applications_collection(): return _get_partner_db()["applications"]
def _candidate_pipeline():      return _get_partner_db()["candidatepipelines"]

# === Resume Utilities ===
def download_resume_from_s3(resume_url: str):
    print(f"[{now()}] Downloading resume from: {resume_url}")
    try:
        parsed = urlparse(resume_url)
        bucket = parsed.netloc.split('.')[0]
        key = parsed.path.lstrip('/')
        s3 = boto3.client(
            's3',
            aws_access_key_id=os.getenv("Resume_AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("Resume_AWS_SECRET_ACCESS_KEY"),
            region_name=os.getenv("Resume_AWS_REGION")
        )
        buf = io.BytesIO()
        s3.download_fileobj(bucket, key, buf)
        buf.seek(0)
        return buf
    except Exception as e:
        print(f"[{now()}] S3 Download Error: {e}")
        return None

def extract_text_from_pdf(pdf_file):
    try:
        pdf_file.seek(0)
        text = ""
        with fitz.open(stream=pdf_file.read(), filetype="pdf") as doc:
            for page in doc:
                text += page.get_text("text")
        return text
    except Exception as e:
        print(f"[{now()}] PDF Extract Error: {e}")
        return ""

def extract_text_from_docx(docx_file):
    try:
        docx_file.seek(0)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as temp_file:
            temp_file.write(docx_file.read())
            temp_path = temp_file.name
        text = docx2txt.process(temp_path)
        os.remove(temp_path)
        return text
    except Exception as e:
        print(f"[{now()}] DOCX Extract Error: {e}")
        return ""

# === Core Resume Processing ===
def compute_ats_similarity(text: str, job_embedding) -> float:
    """
    Smarter ATS scoring:
    1. Extract skill keywords found in the resume and encode them as a dense phrase.
    2. Split the resume into overlapping ~200-word chunks and encode each.
    3. Return the MAX cosine similarity across all chunks + keyword phrase.
    This avoids the 'full-doc dilution' problem where a long resume scores low
    against a short job description.
    """
    if not text or not text.strip():
        return 0.0

    # ── Skill keyword extraction ───────────────────────────────────────────────
    lower = text.lower()
    found_skills = [kw for kw in SKILL_KEYWORDS if re.search(rf"\b{re.escape(kw)}\b", lower)]
    skill_phrase = " ".join(found_skills) if found_skills else ""

    # ── Chunk the resume (200-word windows, 50-word stride) ───────────────────
    words = text.split()
    chunks = []
    window, stride = 200, 50
    for i in range(0, max(1, len(words) - window + 1), stride):
        chunk = " ".join(words[i: i + window])
        if chunk.strip():
            chunks.append(chunk)
    # Always add the first 300 words (header / summary area) as a separate chunk
    chunks.append(" ".join(words[:300]))
    if skill_phrase:
        chunks.append(skill_phrase)

    # Deduplicate
    chunks = list(dict.fromkeys(c for c in chunks if c.strip()))

    # ── Encode all chunks at once (batched → fast) ────────────────────────────
    chunk_embeddings = embedder.encode(chunks, convert_to_tensor=True, batch_size=32)
    sims = util.cos_sim(chunk_embeddings, job_embedding)   # shape: (N, 1)
    best = float(sims.max().item())

    return best


async def process_resume(resume_url, job_embedding):
    application = await asyncio.get_event_loop().run_in_executor(
        None, lambda: _applications_collection().find_one({"resumeUrl": resume_url})
    )

    if not application:
        print(f"[{now()}] No application found for resume: {resume_url}")
        return None

    name = application.get("userName") or application.get("name")
    email = application.get("userEmail") or application.get("email")
    applied_date = application.get("appliedDate") or application.get("applied_date") or application.get("appliedOn")
    student_id = application.get("studentId") or application.get("student_id") or application.get("studentID")
    school_admin_id = extract_school_admin_id(application)

    if not school_admin_id:
        print(f"[{now()}] ⚠️ Missing schoolAdmin in application: {resume_url}")

    file_stream = await asyncio.get_event_loop().run_in_executor(None, download_resume_from_s3, resume_url)
    if not file_stream:
        return None

    ext = os.path.splitext(urlparse(resume_url).path)[-1].lower()
    print(f"[{now()}] Extracting resume as {ext}")
    if ext == ".pdf":
        text = await asyncio.get_event_loop().run_in_executor(None, extract_text_from_pdf, file_stream)
    elif ext == ".docx":
        text = await asyncio.get_event_loop().run_in_executor(None, extract_text_from_docx, file_stream)
    else:
        print(f"[{now()}] Unsupported file type: {ext}")
        return None

    similarity = await asyncio.get_event_loop().run_in_executor(
        None, compute_ats_similarity, text, job_embedding
    )
    print(f"[{now()}] ATS score for {email}: {similarity:.4f} ({similarity*100:.1f}%)")

    return {
        "student_id": student_id,
        "name": name,
        "email": email,
        "appliedDate": applied_date,
        "resumeUrl": resume_url,
        "similarity_score": similarity,
        "text": text,
        "school_admin_id": school_admin_id
    }

def sync_shortlisted_to_pipeline(candidates, internship_obj_id):
    now = datetime.utcnow()
    ops = []

    for c in candidates:
        sid = c.get("student_id")
        if not sid or not ObjectId.is_valid(str(sid)):
            continue

        ops.append(
            UpdateOne(
                {
                    "internshipId": internship_obj_id,
                    "studentId": ObjectId(sid),
                },
                {
                    "$set": {
                        "stage": "L2",
                        "l2.enabled": True,
                        "l2.status": "not_sent",
                        "l2.updatedAt": now,
                    },
                    "$setOnInsert": {
                        "internshipId": internship_obj_id,
                        "studentId": ObjectId(sid),
                    },
                },
                upsert=True,
            )
        )

    if ops:
        _candidate_pipeline().bulk_write(ops)

# === FastAPI App Init ===
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://www.skillnaav.com", "https://skillnaav.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    print(f"\n[{now()}] 🔄 Incoming request: {request.method} {request.url}")
    response = await call_next(request)
    print(f"[{now()}] 🔚 Response status: {response.status_code}")
    return response


SERVER_BASE_URL = os.getenv("SERVER_BASE_URL", "http://localhost:5000")
FRONTEND_BASE_URL = os.getenv("FRONTEND_BASE_URL", "http://localhost:3000")

async def notify_rejection(app_doc):
    student_id_raw = app_doc.get("studentId")
    student_id = str(student_id_raw) if student_id_raw else None

    job_title = app_doc.get("jobTitle") or "the internship"
    student_email = app_doc.get("userEmail")

    if not student_id:
        print(f"[{now()}] ⚠ missing studentId in app_doc")
        return

    # Relative internal link
    relative_link = "/user-main-page?openTab=recommendations&from=auto_reject"

    notif_payload = {
        "studentId": student_id,
        "email": student_email,
        "title": "Application Rejected",
        "message": f"Your application for {job_title} was rejected. Please check recommendations.",
        "link": relative_link,
        "type": "recommendation",
        "skipEmail": True
    }

    notifications_url = f"{SERVER_BASE_URL}/api/notifications"

    async with httpx.AsyncClient(timeout=8) as client:
        try:
            resp = await client.post(notifications_url, json=notif_payload)
            if resp.status_code in (200, 201):
                print(f"[{now()}] ✅ Notification created for {student_id}")
            else:
                print(f"[{now()}] ❌ Error {resp.status_code}: {resp.text}")
                # Retry once
                resp2 = await client.post(notifications_url, json=notif_payload)
                print(f"Retry → {resp2.status_code}: {resp2.text}")

        except Exception as e:
            print(f"[{now()}] ❌ Failed to contact Node notifications API: {e}")


@app.post("/partner/shortlist")
async def shortlist_candidates(
    background_tasks: BackgroundTasks,
    request: Request,
):
    """Accepts multipart/form-data from Node AiServices.js"""
    form = await request.form()
    internship_id   = form.get("internship_id", "")
    job_description = form.get("job_description", "")
    raw_skills      = form.get("job_skills", "[]")
    resumes         = form.getlist("resumes")

    try:
        job_skills_list = json.loads(raw_skills) if isinstance(raw_skills, str) else raw_skills
    except (json.JSONDecodeError, TypeError):
        job_skills_list = []

    print(f"[{now()}] shortlist -> internship_id='{internship_id}' resumes={len(resumes)} skills={job_skills_list}")

    # ── ATS threshold (partner input, 0–100 → converted to 0–1) ──────────────
    try:
        ats_pct = float(form.get("ats_threshold", 30))
        ats_pct = max(0.0, min(100.0, ats_pct))
    except (ValueError, TypeError):
        ats_pct = 30.0
    ats_threshold = ats_pct / 100.0
    print(f"[{now()}] ATS threshold → {ats_pct}% ({ats_threshold:.2f})")

    # Validate internship_id
    try:
        internship_obj_id = ObjectId(internship_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid internship_id: {e}")

    if not resumes:
        raise HTTPException(status_code=400, detail="No resumes provided.")

    # Compose enriched job text — repeat skills 3× so they dominate the embedding
    skills_text = " ".join(job_skills_list)
    job_text = f"{job_description} {skills_text} {skills_text} {skills_text}".strip()
    job_embedding = embedder.encode(job_text, convert_to_tensor=True)

    # ── FIX 1: Skip resumes whose application is already Shortlisted ─────────
    # Prevents duplicate shortlist entries and re-processing on repeated clicks.
    already_shortlisted_urls = set(
        doc["resumeUrl"]
        for doc in _applications_collection().find(
            {
                "internshipId": internship_obj_id,
                "resumeUrl": {"$in": resumes},
                "status": "Shortlisted",
            },
            {"resumeUrl": 1},
        )
    )

    pending_resumes = [url for url in resumes if url not in already_shortlisted_urls]

    if already_shortlisted_urls:
        print(f"[{now()}] ⏭ Skipping {len(already_shortlisted_urls)} already-shortlisted resume(s).")

    if not pending_resumes:
        existing = list(_shortlist_collection().find({"internship_id": internship_obj_id}))
        print(f"[{now()}] ✅ All resumes already shortlisted — returning cached results.")
        return {"shortlisted_candidates": convert_object_ids(existing)}

    # ── Process only pending resumes ──────────────────────────────────────────
    tasks = [process_resume(url, job_embedding) for url in pending_resumes]
    results = await asyncio.gather(*tasks)

    # Filter by partner-defined ATS threshold
    candidates = [c for c in results if c and c["similarity_score"] >= ats_threshold]

    for cand in candidates:
        cand["internship_id"] = internship_obj_id
        cand["ats_score_pct"] = round(cand["similarity_score"] * 100, 1)
        if cand.get("school_admin_id") and ObjectId.is_valid(str(cand["school_admin_id"])):
            cand["school_admin_id"] = ObjectId(cand["school_admin_id"])

    candidates = sorted(candidates, key=lambda x: x["similarity_score"], reverse=True)

    # Shortlisted resume URLs from this run
    shortlisted_resume_urls = [c["resumeUrl"] for c in candidates]

    # All applications for this internship
    all_applications = list(_applications_collection().find({"internshipId": internship_obj_id}))
    all_resume_urls  = [app["resumeUrl"] for app in all_applications]

    # ── FIX 2: Rejected = everyone not shortlisted (now OR previously) ────────
    # Excludes already_shortlisted_urls so previous shortlists are never downgraded.
    rejected_resume_urls = list(
        set(all_resume_urls)
        - set(shortlisted_resume_urls)
        - already_shortlisted_urls
    )

    if candidates:
        _shortlist_collection().insert_many(candidates)

        _applications_collection().update_many(
            {"resumeUrl": {"$in": shortlisted_resume_urls}},
            {"$set": {"status": "Shortlisted"}}
        )

        sync_shortlisted_to_pipeline(candidates, internship_obj_id)

    # ── FIX 3: Always mark non-shortlisted as Rejected ────────────────────────
    # This runs even when NO candidate clears 0.3, so applications never stay
    # stuck in "Applied" forever.
    if rejected_resume_urls:
        _applications_collection().update_many(
            {
                "resumeUrl": {"$in": rejected_resume_urls},
                "status": {"$nin": ["Shortlisted"]},  # safety: never downgrade
            },
            {"$set": {"status": "Rejected"}}
        )
        print(f"[{now()}] 🚫 Marked {len(rejected_resume_urls)} application(s) as Rejected.")

        # Send rejection notifications
        for resume_url in rejected_resume_urls:
            app_doc = _applications_collection().find_one({"resumeUrl": resume_url})
            if app_doc:
                background_tasks.add_task(notify_rejection, app_doc)

    return {"shortlisted_candidates": convert_object_ids(candidates)}
def enrich_shortlist_docs(docs: list) -> list:
    """
    Ensure every shortlisted record has ats_score_pct.
    Backfills old records from similarity_score.
    Removes heavy resume text field from response.
    """
    enriched = []

    for doc in docs:
        if doc is None:
            continue

        # Backfill ATS %
        if doc.get("ats_score_pct") is None and doc.get("similarity_score") is not None:
            try:
                doc["ats_score_pct"] = round(float(doc["similarity_score"]) * 100, 1)
            except (TypeError, ValueError):
                doc["ats_score_pct"] = None

        # Remove large raw resume text
        doc.pop("text", None)

        enriched.append(doc)

    return enriched


@app.get("/partner/shortlisted/by-admin")
async def get_shortlisted_by_admin(
    internship_id: str = Query(...),
    school_admin_id: str = Query(...)
):
    print(f"\n[{now()}] === /partner/shortlisted/by-admin Called ===")
    print(
        f"[{now()}] Raw Query Params → "
        f"internship_id: '{internship_id}', "
        f"school_admin_id: '{school_admin_id}'"
    )

    if not internship_id or not ObjectId.is_valid(internship_id):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid internship_id: '{internship_id}'"
        )

    if not school_admin_id or not ObjectId.is_valid(school_admin_id):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid school_admin_id: '{school_admin_id}'"
        )

    query = {
        "internship_id": ObjectId(internship_id),
        "school_admin_id": ObjectId(school_admin_id)
    }

    try:
        docs = list(_shortlist_collection().find(query))

        # Fix old records + strip text
        docs = enrich_shortlist_docs(docs)

        return {
            "shortlisted_candidates": convert_object_ids(docs)
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Database error occurred: {str(e)}"
        )
    
@app.get("/partner/shortlisted/{internship_id}")
async def get_shortlisted_candidates(
    internship_id: str = Path(..., pattern="^[a-fA-F0-9]{24}$")
):
    try:
        internship_obj_id = ObjectId(internship_id)

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid internship_id: {str(e)}"
        )

    try:
        docs = list(
            _shortlist_collection().find(
                {"internship_id": internship_obj_id}
            )
        )

        # Fix old records + strip text
        docs = enrich_shortlist_docs(docs)

        return {
            "shortlisted_candidates": convert_object_ids(docs)
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
    
@app.get("/partner/fetch-applications/{job_id}")
async def fetch_applications(job_id: str):
    try:
        apps = list(_applications_collection().find({"job_id": job_id}, {"_id": 0}))
        return {"applications": convert_object_ids(apps)}
    except Exception as e:
        return {"error": str(e)}

from pydantic import BaseModel
import re

# ═══════════════════════════════════════════════════════════════════════════════
# RESUME INTELLIGENCE — Full structured extraction
# Replaces the old skills-only endpoint.
# Returns: name, email, phone, linkedin, portfolio, summary,
#          skills[], education[], experience[], projects[],
#          certifications[], languages[]
# ═══════════════════════════════════════════════════════════════════════════════

class ResumeRequest(BaseModel):
    resume_url: str


# ── Expanded skill dictionary ─────────────────────────────────────────────────
SKILL_KEYWORDS = [
    # Languages
    "python", "javascript", "typescript", "java", "c++", "c#", "go", "rust",
    "ruby", "php", "swift", "kotlin", "scala", "r", "matlab", "dart",
    # Frontend
    "react", "vue", "angular", "next.js", "svelte", "tailwind", "html", "css",
    "redux", "webpack", "vite",
    # Backend
    "node", "express", "fastapi", "django", "flask", "spring", "laravel",
    "graphql", "rest api", "microservices",
    # Data / AI / ML
    "machine learning", "deep learning", "ai", "tensorflow", "pytorch",
    "pandas", "numpy", "scikit-learn", "nlp", "computer vision", "data science",
    "keras", "opencv", "llm",
    # Cloud / DevOps
    "aws", "azure", "gcp", "docker", "kubernetes", "ci/cd", "terraform",
    "linux", "git", "jenkins", "github actions",
    # Databases
    "mongodb", "postgresql", "mysql", "redis", "elasticsearch", "sqlite",
    "firebase", "dynamodb", "sql",
    # Other
    "agile", "scrum", "figma", "unity", "android", "ios", "flutter",
]


# ── Section splitter ──────────────────────────────────────────────────────────
SECTION_PATTERNS = {
    "summary":        r"(summary|objective|profile|about me|professional summary|career objective)",
    "experience":     r"(experience|employment|work history|internship|positions?|work experience)",
    "education":      r"(education|academic|qualification|degree|university|college|school)",
    "projects":       r"(projects?|portfolio|personal projects?|academic projects?|key projects?)",
    "certifications": r"(certif|licens|award|achievement|accomplishment)",
    "skills":         r"(skills?|technical skills?|core competencies|technologies)",
    "contact":        r"(contact|email|phone|linkedin|portfolio|website|links?)",
}

def split_sections(text: str) -> dict:
    lines = text.split("\n")
    sections = {k: [] for k in SECTION_PATTERNS}
    sections["other"] = []
    current = "other"
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        matched = False
        for section, pattern in SECTION_PATTERNS.items():
            if re.search(pattern, stripped, re.IGNORECASE) and len(stripped) < 60:
                current = section
                matched = True
                break
        if not matched:
            sections[current].append(stripped)
    return {k: "\n".join(v) for k, v in sections.items()}


# ── Field extractors ──────────────────────────────────────────────────────────
def extract_skills_from_text(text: str) -> list:
    detected = []
    lower_text = text.lower()
    for skill in SKILL_KEYWORDS:
        if re.search(rf"\b{re.escape(skill)}\b", lower_text):
            detected.append(skill.title())
    return list(set(detected))

def extract_email(text: str) -> str:
    m = re.search(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", text)
    return m.group(0) if m else ""

def extract_phone(text: str) -> str:
    m = re.search(r"(\+?\d[\d\s\-().]{7,}\d)", text)
    return m.group(0).strip() if m else ""

def extract_linkedin(text: str) -> str:
    m = re.search(r"(https?://)?(www\.)?linkedin\.com/in/[\w\-]+/?", text, re.IGNORECASE)
    return m.group(0) if m else ""

def extract_portfolio(text: str) -> str:
    m = re.search(
        r"(https?://)?(www\.)?(github\.com/[\w\-]+|[\w\-]+\.(dev|io|me|netlify\.app|vercel\.app)[^\s,)\"']*)",
        text, re.IGNORECASE
    )
    return m.group(0) if m else ""

def extract_name(text: str) -> str:
    """First short line with 2–5 words and no special chars is usually the name."""
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped and 2 <= len(stripped.split()) <= 5 and len(stripped) < 50:
            if not re.search(r"[@/\d|•@#]", stripped):
                return stripped
    return ""

def parse_summary(section_text: str, full_text: str) -> str:
    if section_text.strip():
        return section_text.strip()[:600]
    paragraphs = [p.strip() for p in full_text.split("\n\n") if p.strip()]
    for para in paragraphs[1:4]:
        if len(para) > 80 and not re.search(r"[@\d]{2,}", para[:30]):
            return para[:600]
    return ""

def parse_education(section_text: str) -> list:
    results = []
    if not section_text.strip():
        return results
    chunks = re.split(
        r"\n{2,}|(?=\b(B\.?Tech|B\.?E|B\.?Sc|M\.?Tech|M\.?Sc|MBA|PhD|Bachelor|Master|High School|Diploma)\b)",
        section_text
    )
    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        entry = {}
        deg = re.search(
            r"(B\.?Tech|B\.?E|B\.?Sc|M\.?Tech|M\.?Sc|MBA|PhD|Bachelor[^\n,]*|Master[^\n,]*|High School[^\n,]*|Diploma[^\n,]*)",
            chunk, re.IGNORECASE
        )
        if deg:
            entry["degree"] = deg.group(0).strip()
        uni = re.search(r"(university|college|institute|school|academy)[^\n,]*", chunk, re.IGNORECASE)
        if uni:
            entry["university"] = uni.group(0).strip()
        years = re.findall(r"\b(20\d{2})\b", chunk)
        if len(years) >= 2:
            entry["startYear"] = years[0]
            entry["endYear"] = years[-1]
        elif len(years) == 1:
            entry["startYear"] = years[0]
        grade = re.search(r"(CGPA|GPA|Grade|Percentage)[:\s]+([0-9.]+\s*%?)", chunk, re.IGNORECASE)
        if grade:
            entry["grade"] = f"{grade.group(1)} {grade.group(2)}".strip()
        if entry.get("degree") or entry.get("university"):
            results.append(entry)
    return results[:3]

def parse_experience(section_text: str) -> list:
    results = []
    if not section_text.strip():
        return results
    chunks = re.split(r"\n{2,}", section_text)
    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk or len(chunk) < 20:
            continue
        lines = [l.strip() for l in chunk.split("\n") if l.strip()]
        entry = {}
        if lines:
            entry["title"] = lines[0]
        if len(lines) > 1:
            entry["company"] = lines[1]
        date_matches = re.findall(
            r"(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
            r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
            r"[,\s]+\d{4}",
            chunk, re.IGNORECASE
        )
        year_matches = re.findall(r"\b(20\d{2})\b", chunk)
        if date_matches:
            entry["startDate"] = date_matches[0]
            entry["endDate"] = date_matches[1] if len(date_matches) > 1 else ""
        elif year_matches:
            entry["startDate"] = year_matches[0]
            entry["endDate"] = year_matches[1] if len(year_matches) > 1 else ""
        entry["current"] = bool(re.search(r"\b(present|current|now)\b", chunk, re.IGNORECASE))
        entry["location"] = ""
        entry["description"] = " ".join(lines[2:])[:400] if len(lines) > 2 else ""
        if entry.get("title"):
            results.append(entry)
    return results[:5]

def parse_projects(section_text: str) -> list:
    results = []
    if not section_text.strip():
        return results
    chunks = re.split(r"\n{2,}", section_text)
    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk or len(chunk) < 15:
            continue
        lines = [l.strip() for l in chunk.split("\n") if l.strip()]
        entry = {}
        if lines:
            entry["name"] = lines[0]
        entry["description"] = " ".join(lines[1:])[:400]
        tech_match = re.search(
            r"(?:tech(?:nologies)?|stack|built with|tools|using)[:\s]+([\w,\s.+#]+)",
            chunk, re.IGNORECASE
        )
        if tech_match:
            entry["techStack"] = [t.strip() for t in re.split(r"[,|·•]", tech_match.group(1)) if t.strip()]
        else:
            entry["techStack"] = extract_skills_from_text(chunk)
        link = re.search(r"https?://[^\s\"'>]+", chunk)
        entry["link"] = link.group(0) if link else ""
        if entry.get("name"):
            results.append(entry)
    return results[:5]

def parse_certifications(section_text: str) -> list:
    results = []
    if not section_text.strip():
        return results
    for line in section_text.split("\n"):
        line = line.strip()
        if not line or len(line) < 5:
            continue
        entry = {"name": line, "issuer": "", "issueDate": "", "expiryDate": "", "credentialUrl": ""}
        issuer = re.search(
            r"(Google|AWS|Amazon|Microsoft|Coursera|Udemy|LinkedIn|Meta|IBM|Oracle|Cisco|HackerRank|MongoDB|Nptel|edX)",
            line, re.IGNORECASE
        )
        if issuer:
            entry["issuer"] = issuer.group(0)
        date = re.search(r"(20\d{2})", line)
        if date:
            entry["issueDate"] = date.group(0)
        results.append(entry)
    return results[:6]

def parse_languages(text: str) -> list:
    common = [
        "English", "Hindi", "Telugu", "Tamil", "Kannada", "Malayalam",
        "Spanish", "French", "German", "Chinese", "Japanese", "Arabic",
        "Portuguese", "Russian", "Korean", "Italian", "Dutch", "Bengali",
        "Marathi", "Urdu", "Gujarati", "Punjabi",
    ]
    results = []
    for lang in common:
        if re.search(rf"\b{lang}\b", text, re.IGNORECASE):
            prof_match = re.search(
                rf"{lang}[^.\n]*(native|fluent|advanced|intermediate|beginner)",
                text, re.IGNORECASE
            )
            proficiency = "Intermediate"
            if prof_match:
                p = prof_match.group(1).lower()
                if p in ("native", "fluent"):    proficiency = "Native"
                elif p == "advanced":            proficiency = "Advanced"
                elif p == "beginner":            proficiency = "Beginner"
            results.append({"language": lang, "proficiency": proficiency})
    return results[:5]


# ── Main endpoint ─────────────────────────────────────────────────────────────
@app.post("/extract-resume")
async def extract_resume(data: ResumeRequest):
    resume_url = data.resume_url
    print(f"[{now()}] 🧠 Extract-resume called for: {resume_url}")

    # 1️⃣ Download
    file_stream = await asyncio.get_event_loop().run_in_executor(
        None, download_resume_from_s3, resume_url
    )
    if not file_stream:
        raise HTTPException(status_code=400, detail="Failed to download resume")

    # 2️⃣ Extract raw text
    ext = os.path.splitext(urlparse(resume_url).path)[-1].lower()
    if ext == ".pdf":
        text = await asyncio.get_event_loop().run_in_executor(None, extract_text_from_pdf, file_stream)
    elif ext == ".docx":
        text = await asyncio.get_event_loop().run_in_executor(None, extract_text_from_docx, file_stream)
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")

    if not text.strip():
        raise HTTPException(status_code=422, detail="Could not extract text from resume")

    print(f"[{now()}] 📄 Extracted {len(text)} characters")

    # 3️⃣ Split into sections then extract every field
    sections   = split_sections(text)
    name           = extract_name(text)
    email          = extract_email(text)
    phone          = extract_phone(text)
    linkedin       = extract_linkedin(text)
    portfolio      = extract_portfolio(text)
    summary        = parse_summary(sections.get("summary", ""), text)
    skills         = extract_skills_from_text(text)
    education      = parse_education(sections.get("education", ""))
    experience     = parse_experience(sections.get("experience", ""))
    projects       = parse_projects(sections.get("projects", ""))
    certifications = parse_certifications(sections.get("certifications", ""))
    languages      = parse_languages(text)

    print(
        f"[{now()}] ✅ Done → skills:{len(skills)}, exp:{len(experience)}, "
        f"proj:{len(projects)}, edu:{len(education)}, certs:{len(certifications)}, "
        f"langs:{len(languages)}, linkedin:{'✓' if linkedin else '✗'}, "
        f"summary:{'✓' if summary else '✗'}"
    )

    return {
        "name":           name,
        "email":          email,
        "phone":          phone,
        "linkedin":       linkedin,
        "portfolio":      portfolio,
        "summary":        summary,
        "skills":         skills,
        "education":      education,
        "experience":     experience,
        "projects":       projects,
        "certifications": certifications,
        "languages":      languages,
    }

# ═══════════════════════════════════════════════════════════════════════════════
# CV GENERATOR — /cv/generate
# Accepts the merged StudentProfile shape from the Node API and returns a PDF.
# ═══════════════════════════════════════════════════════════════════════════════

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.pdfgen import canvas as rl_canvas
from reportlab.lib.units import mm

from pydantic import BaseModel
from typing import List, Optional, Any
from fastapi.responses import StreamingResponse

# ── Pydantic models matching the merged profile shape ─────────────────────────
class CVRequest(BaseModel):
    # Userwebapp fields (flat — from merged GET /api/student-profile/:userId)
    name:              Optional[str] = ""
    email:             Optional[str] = ""
    phone:             Optional[str] = ""
    linkedin:          Optional[str] = ""
    portfolio:         Optional[str] = ""
    universityName:    Optional[str] = ""
    fieldOfStudy:      Optional[str] = ""
    educationLevel:    Optional[str] = ""
    country:           Optional[str] = ""
    skills:            Optional[List[str]] = []

    # StudentProfile extension fields
    summary:           Optional[str] = ""
    experience:        Optional[List[Any]] = []
    projects:          Optional[List[Any]] = []
    certifications:    Optional[List[Any]] = []
    languages:         Optional[List[Any]] = []


# ── Layout constants ──────────────────────────────────────────────────────────
W, H      = A4                      # 595 x 842 pt
MARGIN_X  = 18 * mm
MARGIN_B  = 14 * mm
HEADER_H  = 46 * mm
ACCENT_H  = 5  * mm

# Brand colours
INDIGO    = colors.HexColor("#4F46E5")
INDIGO_LT = colors.HexColor("#818CF8")
SLATE_800 = colors.HexColor("#1E293B")
SLATE_600 = colors.HexColor("#475569")
SLATE_400 = colors.HexColor("#94A3B8")
SLATE_100 = colors.HexColor("#F1F5F9")
WHITE     = colors.white


def build_cv_pdf(p: CVRequest) -> bytes:
    buf = io.BytesIO()
    c   = rl_canvas.Canvas(buf, pagesize=A4)
    c.setTitle(f"Skillnaav CV — {p.name or 'Resume'}")

    # ── state ──────────────────────────────────────────────────────────────
    y = [H]   # mutable cursor (list so inner funcs can mutate)

    def new_page():
        c.showPage()
        y[0] = H - 10 * mm
        # light top rule on continuation pages
        c.setStrokeColor(INDIGO)
        c.setLineWidth(1)
        c.line(MARGIN_X, y[0], W - MARGIN_X, y[0])
        y[0] -= 6 * mm

    def check_page(need=14):
        if y[0] < MARGIN_B + need * mm:
            new_page()

    # ── HEADER BAND ────────────────────────────────────────────────────────
    # Indigo background
    c.setFillColor(INDIGO)
    c.rect(0, H - HEADER_H, W, HEADER_H, fill=1, stroke=0)
    # Bottom accent stripe
    c.setFillColor(INDIGO_LT)
    c.rect(0, H - HEADER_H, W, ACCENT_H, fill=1, stroke=0)

    # Watermark text
    c.saveState()
    c.setFillColor(WHITE)
    c.setFillAlpha(0.06)
    c.setFont("Helvetica-Bold", 72)
    c.drawCentredString(W / 2, H - HEADER_H + 6 * mm, "SKILLNAAV")
    c.restoreState()

    # Name
    name_txt = p.name or "Your Name"
    c.setFillColor(WHITE)
    c.setFont("Helvetica-Bold", 22)
    c.drawString(MARGIN_X, H - 18 * mm, name_txt)

    # Contact row
    contact_parts = [x for x in [p.email, p.phone, p.country] if x]
    contact_str = "  ·  ".join(contact_parts)
    c.setFont("Helvetica", 8)
    c.setFillColor(colors.HexColor("#E0E7FF"))
    c.drawString(MARGIN_X, H - 27 * mm, contact_str)

    # Links row
    links = [x for x in [p.linkedin, p.portfolio] if x]
    if links:
        c.setFont("Helvetica-Oblique", 7.5)
        c.setFillColor(colors.HexColor("#C7D2FE"))
        c.drawString(MARGIN_X, H - 34 * mm, "  |  ".join(links))

    # Skillnaav badge (top-right)
    c.setFont("Helvetica-Bold", 8)
    c.setFillColor(WHITE)
    c.drawRightString(W - MARGIN_X, H - 14 * mm, "skillnaav")
    c.setFont("Helvetica", 6.5)
    c.setFillColor(colors.HexColor("#C7D2FE"))
    c.drawRightString(W - MARGIN_X, H - 20 * mm, "skillnaav.com")

    y[0] = H - HEADER_H - 10 * mm

    # ── SECTION HELPER ────────────────────────────────────────────────────
    def section_title(title):
        check_page(20)
        y[0] -= 4 * mm
        c.setFont("Helvetica-Bold", 8.5)
        c.setFillColor(INDIGO)
        c.drawString(MARGIN_X, y[0], title.upper())
        # ruled line
        c.setStrokeColor(SLATE_100)
        c.setLineWidth(0.8)
        c.line(MARGIN_X + 3 * mm + c.stringWidth(title.upper(), "Helvetica-Bold", 8.5),
               y[0] + 1.5 * mm, W - MARGIN_X, y[0] + 1.5 * mm)
        y[0] -= 5 * mm

    def body_text(txt, indent=0, color=SLATE_600, size=8.5, max_width=None):
        if not txt:
            return
        check_page(6)
        mw = max_width or (W - 2 * MARGIN_X - indent)
        # Simple word-wrap
        words = str(txt).split()
        line_buf, lines = [], []
        c.setFont("Helvetica", size)
        for w_t in words:
            test = " ".join(line_buf + [w_t])
            if c.stringWidth(test, "Helvetica", size) > mw:
                lines.append(" ".join(line_buf))
                line_buf = [w_t]
            else:
                line_buf.append(w_t)
        if line_buf:
            lines.append(" ".join(line_buf))
        for ln in lines:
            check_page(5)
            c.setFillColor(color)
            c.drawString(MARGIN_X + indent, y[0], ln)
            y[0] -= 4.5 * mm

    def meta_text(txt, indent=0):
        body_text(txt, indent=indent, color=SLATE_400, size=7.5)

    def bold_text(txt, indent=0, size=9, color=SLATE_800):
        if not txt:
            return
        check_page(6)
        c.setFont("Helvetica-Bold", size)
        c.setFillColor(color)
        c.drawString(MARGIN_X + indent, y[0], str(txt))
        y[0] -= 4.5 * mm

    # ── SUMMARY ───────────────────────────────────────────────────────────
    if p.summary:
        section_title("Professional Summary")
        body_text(p.summary)

    # ── EDUCATION (from Userwebapp fields) ────────────────────────────────
    if p.universityName or p.educationLevel:
        section_title("Education")
        bold_text(p.universityName or "Institution")
        meta_text(
            "  ·  ".join(filter(None, [p.educationLevel, p.fieldOfStudy])),
            indent=0
        )

    # ── EXPERIENCE ────────────────────────────────────────────────────────
    if p.experience:
        section_title("Experience")
        for exp in p.experience:
            check_page(18)
            bold_text(exp.get("title", ""))
            co = exp.get("company", "")
            loc = exp.get("location", "")
            dates = "  ·  ".join(filter(None, [
                co, loc,
                f"{exp.get('startDate','')} – {'Present' if exp.get('current') else exp.get('endDate','')}"
            ]))
            meta_text(dates)
            body_text(exp.get("description", ""))
            y[0] -= 2 * mm

    # ── SKILLS (pill badges) ──────────────────────────────────────────────
    if p.skills:
        section_title("Skills")
        pill_x = MARGIN_X
        pill_y = y[0]
        pill_h = 5 * mm
        gap_x  = 2.5 * mm
        gap_y  = 2 * mm
        c.setFont("Helvetica", 7.5)
        for skill in p.skills:
            tw = c.stringWidth(skill, "Helvetica", 7.5)
            pw = tw + 5 * mm
            if pill_x + pw > W - MARGIN_X:
                pill_x  = MARGIN_X
                pill_y -= pill_h + gap_y
                check_page(8)
            # bg
            c.setFillColor(SLATE_100)
            c.roundRect(pill_x, pill_y - 1 * mm, pw, pill_h, 2 * mm, fill=1, stroke=0)
            # text
            c.setFillColor(INDIGO)
            c.drawString(pill_x + 2.5 * mm, pill_y + 0.6 * mm, skill)
            pill_x += pw + gap_x
        y[0] = pill_y - pill_h - 4 * mm

    # ── PROJECTS ─────────────────────────────────────────────────────────
    if p.projects:
        section_title("Projects")
        for proj in p.projects:
            check_page(18)
            name_str = proj.get("name", "")
            link = proj.get("link", "")
            display = f"{name_str}  —  {link}" if link else name_str
            bold_text(display)
            stack = proj.get("techStack", [])
            if stack:
                meta_text("Stack: " + ", ".join(stack[:8]))
            body_text(proj.get("description", ""))
            y[0] -= 2 * mm

    # ── CERTIFICATIONS ────────────────────────────────────────────────────
    if p.certifications:
        section_title("Certifications")
        for cert in p.certifications:
            check_page(10)
            bold_text(cert.get("name", ""))
            meta_text("  ·  ".join(filter(None, [cert.get("issuer",""), cert.get("issueDate","")])))

    # ── LANGUAGES ─────────────────────────────────────────────────────────
    if p.languages:
        section_title("Languages")
        lang_str = "   ·   ".join(
            f"{l.get('language','')} ({l.get('proficiency','')})" for l in p.languages
        )
        body_text(lang_str)

    # ── FOOTER ────────────────────────────────────────────────────────────
    for page_num in range(1, c.getPageNumber() + 1):
        c.setFont("Helvetica", 6.5)
        c.setFillColor(SLATE_400)
        footer = f"Generated by Skillnaav  ·  {p.name or ''}  ·  skillnaav.com"
        c.drawCentredString(W / 2, MARGIN_B - 5 * mm, footer)

    c.save()
    buf.seek(0)
    return buf.read()


@app.post("/cv/generate")
async def generate_cv(profile: CVRequest):
    """
    Accepts the merged profile (flat shape from GET /api/student-profile/:userId)
    and returns a Skillnaav-branded PDF.
    """
    print(f"[{now()}] 📄 CV generate for: {profile.name}")

    try:
        pdf_bytes = await asyncio.get_event_loop().run_in_executor(
            None, build_cv_pdf, profile
        )
    except Exception as e:
        print(f"[{now()}] ❌ CV generation error: {e}")
        raise HTTPException(status_code=500, detail=f"CV generation failed: {str(e)}")

    safe_name = (profile.name or "Resume").replace(" ", "_")
    filename  = f"Skillnaav_CV_{safe_name}.pdf"

    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )