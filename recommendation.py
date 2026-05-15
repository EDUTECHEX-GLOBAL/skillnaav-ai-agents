import os
import orjson # type: ignore
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import ORJSONResponse
from motor.motor_asyncio import AsyncIOMotorClient # type: ignore
from bson import ObjectId
from typing import List, Dict, Any, Union
from embedding_model import embedder, util   # lazy proxies — PyTorch loads on first use

# Load environment variables from .env filessss
load_dotenv()

app = FastAPI()

# MongoDB setup
MONGO_URI = os.getenv("MONGO_URI", "")
DB_NAME = os.getenv("MONGO_DB_NAME", "skillnaav")

if not MONGO_URI:
    import logging as _log
    _log.warning("MONGO_URI is not set — recommendation DB calls will fail at runtime.")

# Lazy DB initialisation so missing env vars don't crash the worker on boot
_mongo_client = None
_db = None

def _get_db():
    global _mongo_client, _db
    if _db is None:
        if not MONGO_URI:
            raise RuntimeError("MONGO_URI environment variable is not set.")
        _mongo_client = AsyncIOMotorClient(MONGO_URI)
        _db = _mongo_client[DB_NAME]
    return _db

# Collection accessors (evaluated lazily)
def _application_collection():  return _get_db().applications
def _user_collection():          return _get_db().userwebapps
def _internship_collection():    return _get_db().internshippostings
def _personality_collection():   return _get_db().personalityresponses


# Level ranking for career progression logic
LEVEL_RANK = {'basic': 1, 'intermediate': 2, 'advanced': 3}

# --- RIASEC-to-sector mapping ---
RIASEC_SECTOR_MAP = {
    "R": ["engineering", "mechanical", "electrical", "construction", "it support"],
    "I": ["research", "data", "analysis", "scientific", "technical", "programming"],
    "A": ["design", "creative", "art", "music", "fashion", "writer", "graphic"],
    "S": ["teaching", "counseling", "healthcare", "social work", "customer support"],
    "E": ["marketing", "sales", "entrepreneurship", "management", "leadership"],
    "C": ["accounting", "finance", "administration", "data entry", "project management"],
}

# --- Helper Functions ---

def convert_object_ids(data: Union[Dict, List]) -> Union[Dict, List]:
    """Recursively converts all ObjectId instances to strings."""
    if isinstance(data, dict):
        return {key: convert_object_ids(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_object_ids(item) for item in data]
    elif isinstance(data, ObjectId):
        return str(data)
    else:
        return data

def norm(text: str) -> str:
    """Normalizes text by converting to lowercase and stripping whitespace."""
    if not text:
        return ""
    return text.lower().strip()

def arr(v: Any) -> List[Any]:
    """Ensures the input is a list."""
    if v is None:
        return []
    if isinstance(v, list):
        return v
    return [v]

async def derive_signals(student: Dict[str, Any]) -> Dict[str, List[str]]:
    """Derives skills, roles, and locations from student profile."""
    skills = [norm(s) for s in arr(student.get('skills'))]
    roles = [norm(r) for r in arr(student.get('desiredRole', [])) + arr(student.get('interests', [])) if r]
    locations = [norm(l) for l in arr(student.get('preferredLocations', [])) + arr([student.get('city')]) if l]
    return {'skills': skills, 'roles': roles, 'locations': locations}

async def infer_from_recent_applications(student_id: ObjectId) -> Dict[str, Any]:
    """Infers skills, roles, and highest level from a student's recent applications."""
    last_apps_cursor = _application_collection().find({
        'studentId': student_id,
        'status': 'Completed'
    }).sort('appliedDate', -1).limit(10)
    last_apps = await last_apps_cursor.to_list(length=10)

    if not last_apps:
        return {'skills': [], 'roles': [], 'locations': [], 'highestLevel': 0}

    skills = set()
    titles = set()
    roles = set()
    locations = set()
    classifications = []

    for app in last_apps:
        internship = app.get('internshipId') or {}
        job_skills = internship.get('qualifications', [])
        skills.update([norm(s) for s in job_skills if s])
        job_title = norm(internship.get('jobTitle'))
        if job_title:
            titles.add(job_title)
        job_sector = norm(internship.get('sector'))
        if job_sector:
            roles.add(job_sector)
        job_location = norm(internship.get('location'))
        if job_location:
            locations.add(job_location)
        classification = norm(internship.get('classification'))
        if classification:
            classifications.append(classification)

    highest_level = max([LEVEL_RANK.get(c, 0) for c in classifications], default=0)

    return {
        'skills': list(skills)[:10],
        'roles': list(titles.union(roles))[:10],
        'locations': list(locations)[:5],
        'highestLevel': highest_level
    }

async def get_personality(student_id_obj: ObjectId) -> Dict[str, Any]:
    """Fetches RIASEC personality test results for a student."""
    personality = await _personality_collection().find_one({'userId': student_id_obj})
    if not personality:
        return {'hollandCode': '', 'dominantTraits': []}
    holland_code = personality.get('hollandCode', '') or ''
    return {
        'hollandCode': holland_code,
        'dominantTraits': list(holland_code) if holland_code else []
    }

def score_job(job: Dict[str, Any], signals: Dict[str, Any], student: Dict[str, Any], dominant_traits: List[str]) -> float:
    """Scores a single job based on various matching criteria and RIASEC traits."""
    score = 0.0
    job_skills = [norm(s) for s in job.get('qualifications', [])]
    job_title = norm(job.get('jobTitle', ''))
    job_desc = norm(job.get('jobDescription', ''))
    job_cat = norm(job.get('sector', ''))
    job_loc = norm(job.get('location', ''))
    work_mode = norm(job.get('internshipMode', ''))
    job_level = LEVEL_RANK.get(norm(job.get('classification', '')), 0)

    skill_hits = len([s for s in signals['skills'] if s in job_skills])
    score += skill_hits * 3

    role_hit = any(r for r in signals['roles'] if r in job_title or r in job_cat or r in job_desc)
    if role_hit:
        score += 5

    fields = [norm(student.get('fieldOfStudy', '')), norm(student.get('desiredField', ''))]
    for f in fields:
        if not f:
            continue
        if f in job_title or f in job_desc or f in job_cat:
            score += 5
        else:
            sim_scores = [
                util.cos_sim(embedder.encode(f), embedder.encode(text)).item()
                for text in [job_title, job_desc, job_cat]
            ]
            if max(sim_scores) > 0.6:
                score += 4

    loc_hit = 'online' in work_mode or 'remote' in work_mode or any(l for l in signals['locations'] if l in job_loc)
    if loc_hit:
        score += 3

    student_level = signals.get('highestLevel', 1) or 1
    if job_level == student_level:
        score += 3
    elif job_level == student_level + 1:
        score += 10
    elif job_level > student_level + 1:
        score -= 3
    else:
        score -= 1

    # RIASEC personality boost
    for trait in dominant_traits:
        if job_cat in RIASEC_SECTOR_MAP.get(trait, []):
            score += 8  # substantial boost for a direct personality/sector match

    return score

async def embed_text(text: str):
    """Encodes text into a vector embedding."""
    return embedder.encode(text, convert_to_tensor=True)

async def score_job_with_embedding(job: Dict[str, Any], signals: Dict[str, Any], student_embedding, student: Dict[str, Any], dominant_traits: List[str]) -> float:
    """Combines a base score, embedding similarity, and RIASEC trait boost."""
    base_score = score_job(job, signals, student, dominant_traits)
    job_text = ' '.join(filter(None, [job.get('jobTitle', ''), job.get('jobDescription', '')] + job.get('qualifications', [])))
    job_embedding = await embed_text(job_text)
    sim_score = util.cos_sim(student_embedding, job_embedding).item()
    return base_score + sim_score * 10

# --- API Endpoint ---

@app.get('/recommendations/{student_id}', response_class=ORJSONResponse)
async def get_personalized_recommendations(student_id: str, limit: int = 6) -> Dict[str, List[Dict[str, Any]]]:
    if not ObjectId.is_valid(student_id):
        raise HTTPException(status_code=400, detail="Invalid student ID")

    print("Generating fresh recommendations (no cache)")

    student_id_obj = ObjectId(student_id)
    student = await _user_collection().find_one({'_id': student_id_obj})
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")

    signals = await derive_signals(student)
    inferred = await infer_from_recent_applications(student_id_obj)

    highest_level = inferred.get('highestLevel', 1) or 1

    signals = {
        'skills': list(set(signals['skills']).union(inferred['skills'])),
        'roles': list(set(signals['roles']).union(inferred['roles'])),
        'locations': list(set(signals['locations']).union(inferred['locations'])),
        'highestLevel': highest_level
    }

    personality = await get_personality(student_id_obj)
    dominant_traits = personality.get('dominantTraits', [])

    applied_ids = await _application_collection().distinct('internshipId', {'studentId': student_id_obj})

    matched_sectors = set()
    for trait in dominant_traits:
        matched_sectors.update(RIASEC_SECTOR_MAP.get(trait, []))

    query = {
        'applicationOpen': True,
        '_id': {'$nin': applied_ids}
    }
    if matched_sectors:
        query['sector'] = {'$in': list(matched_sectors)}

    candidates_cursor = _internship_collection().find(query).limit(100)
    candidates = await candidates_cursor.to_list(length=100)

    if not candidates:
        query = {
            'applicationOpen': True,
            '_id': {'$nin': applied_ids}
        }
        candidates_cursor = _internship_collection().find(query).limit(100)
        candidates = await candidates_cursor.to_list(length=100)

    student_text = ' '.join(signals['skills'] + signals['roles'] + signals['locations'])
    student_embedding = await embed_text(student_text)

    scored_jobs = []
    for job in candidates:
        score = await score_job_with_embedding(job, signals, student_embedding, student, dominant_traits)
        scored_jobs.append({'job': job, 'score': score})
    scored_jobs.sort(key=lambda x: x['score'], reverse=True)

    final_list = [item['job'] for item in scored_jobs[:limit]]

    if not final_list or (scored_jobs and scored_jobs[0]['score'] <= 0):
        fallback_cursor = _internship_collection().find({
            'applicationOpen': True,
            '_id': {'$nin': applied_ids},
            'classification': {'$in': ['basic', 'intermediate']}
        }).sort('createdAt', -1).limit(limit)
        fallback_docs = await fallback_cursor.to_list(length=limit)
        final_list = fallback_docs

    final_list = convert_object_ids(final_list)

    return {'recommendations': final_list}