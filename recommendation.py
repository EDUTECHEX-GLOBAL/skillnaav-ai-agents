import os
import json
import asyncio
import orjson  # type: ignore
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import ORJSONResponse
from motor.motor_asyncio import AsyncIOMotorClient  # type: ignore
from bson import ObjectId
from typing import List, Dict, Any, Union, Optional
from embedding_model import embedder, util  # lazy proxies — PyTorch loads on first use
import anthropic

# Load environment variables
load_dotenv()

app = FastAPI()

# ── In-flight request deduplication ──────────────────────────────────────────
# FIX: When two requests for the same student arrive before either completes
# (e.g. React StrictMode double-invoke, browser retry), both would miss the
# cache, run the full pipeline in parallel, and race on the cache write.
# This dict maps student_id → asyncio.Event so the second caller simply waits
# for the first to finish, then reads the freshly-written cache.
_inflight: Dict[str, asyncio.Event] = {}

# ── MongoDB setup ─────────────────────────────────────────────────────────────
MONGO_URI = os.getenv("MONGO_URI", "")
DB_NAME = os.getenv("MONGO_DB_NAME", "skillnaav")

# How long cached recommendations stay valid before recomputing
CACHE_TTL_HOURS = int(os.getenv("REC_CACHE_TTL_HOURS", "6"))

# ── Anthropic client (used for Claude re-ranking) ─────────────────────────────
# Reads ANTHROPIC_API_KEY from environment automatically.
# If the key is missing, Claude re-ranking is silently skipped and the system
# falls back to pure embedding scores — nothing breaks.
_anthropic_client: Optional[anthropic.AsyncAnthropic] = None

def _get_anthropic_client() -> Optional[anthropic.AsyncAnthropic]:
    global _anthropic_client
    if _anthropic_client is None:
        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        if not api_key:
            return None
        _anthropic_client = anthropic.AsyncAnthropic(api_key=api_key)
    return _anthropic_client

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
def _application_collection():    return _get_db().applications
def _user_collection():            return _get_db().userwebapps
def _internship_collection():      return _get_db().internshippostings
def _personality_collection():     return _get_db().personalityresponses
def _cache_collection():           return _get_db().recommendation_cache  # NEW


# ── Level ranking for career progression logic ────────────────────────────────
LEVEL_RANK = {'basic': 1, 'intermediate': 2, 'advanced': 3}

# ── RIASEC-to-sector mapping ──────────────────────────────────────────────────
RIASEC_SECTOR_MAP = {
    "R": ["engineering", "mechanical", "electrical", "construction", "it support"],
    "I": ["research", "data", "analysis", "scientific", "technical", "programming"],
    "A": ["design", "creative", "art", "music", "fashion", "writer", "graphic"],
    "S": ["teaching", "counseling", "healthcare", "social work", "customer support"],
    "E": ["marketing", "sales", "entrepreneurship", "management", "leadership"],
    "C": ["accounting", "finance", "administration", "data entry", "project management"],
}


# ── Startup: warm up the embedding model so the first real request is fast ────
@app.on_event("startup")
async def warmup_embedding_model():
    """
    FIX #2 (Performance) — Model warm-up
    PyTorch / sentence-transformers can take 15-30 s to load on first use.
    Running a dummy encode during startup means every subsequent request
    hits an already-loaded model and skips that cold-start penalty entirely.
    """
    loop = asyncio.get_event_loop()
    print("[startup] Warming up embedding model — this may take ~20 s on first boot...")
    await loop.run_in_executor(None, lambda: embedder.encode("warmup ping"))
    print("[startup] Embedding model is ready. Accepting requests.")


# ── Helper functions ──────────────────────────────────────────────────────────

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


def _precompute_field_embeddings(student: Dict[str, Any]) -> Dict[str, Any]:
    """
    FIX #2 — Pre-compute field-of-study embeddings ONCE per request, not
    once per job. Previously score_job called embedder.encode() twice per
    field per job inside the scoring loop, producing hundreds of serial
    encode calls (visible as 300+ 'Batches: 100%' log lines).

    Now we encode all unique field strings here, once, and pass the results
    as a lookup table into score_job so it can call util.cos_sim() directly
    without any further encode calls.
    """
    fields = list({
        norm(student.get('fieldOfStudy', '')),
        norm(student.get('desiredField', '')),
    } - {''})

    if not fields:
        return {}

    embeddings = embedder.encode(fields, batch_size=64)
    return {field: embeddings[i] for i, field in enumerate(fields)}


def score_job(
    job: Dict[str, Any],
    signals: Dict[str, Any],
    student: Dict[str, Any],
    dominant_traits: List[str],
    field_embeddings: Dict[str, Any],      # pre-computed — no encode calls inside
    job_embedding = None,                  # pre-computed job embedding tensor
) -> float:
    """
    Scores a single internship posting against the student profile.

    All embeddings are pre-computed before this function is called, so
    there are zero embedder.encode() calls inside — pure arithmetic only.
    """
    score = 0.0
    job_skills = [norm(s) for s in job.get('qualifications', [])]
    job_title  = norm(job.get('jobTitle', ''))
    job_desc   = norm(job.get('jobDescription', ''))
    job_cat    = norm(job.get('sector', ''))
    job_loc    = norm(job.get('location', ''))
    work_mode  = norm(job.get('internshipMode', ''))
    job_level  = LEVEL_RANK.get(norm(job.get('classification', '')), 0)

    # Skill overlap
    skill_hits = sum(1 for s in signals['skills'] if s in job_skills)
    score += skill_hits * 3

    # Role / sector match
    if any(r for r in signals['roles'] if r in job_title or r in job_cat or r in job_desc):
        score += 5

    # Field of study — uses pre-computed embeddings, no encode() calls here
    job_fields_text = [job_title, job_desc, job_cat]
    for field, field_emb in field_embeddings.items():
        if field in job_title or field in job_desc or field in job_cat:
            score += 5
        elif job_embedding is not None:
            # Compare student field embedding against pre-computed job embedding
            sim = util.cos_sim(field_emb, job_embedding).item()
            if sim > 0.6:
                score += 4

    # Location / remote
    if 'online' in work_mode or 'remote' in work_mode or any(l for l in signals['locations'] if l in job_loc):
        score += 3

    # Career level progression
    student_level = signals.get('highestLevel', 1) or 1
    if job_level == student_level:
        score += 3
    elif job_level == student_level + 1:
        score += 10       # stretch goal — strongest signal
    elif job_level > student_level + 1:
        score -= 3
    else:
        score -= 1

    # RIASEC personality boost
    for trait in dominant_traits:
        if job_cat in RIASEC_SECTOR_MAP.get(trait, []):
            score += 8

    return score


def batch_score_jobs(
    candidates: List[Dict[str, Any]],
    signals: Dict[str, Any],
    student_embedding,
    student: Dict[str, Any],
    dominant_traits: List[str],
) -> List[Dict[str, Any]]:
    """
    FIX #2 — Score ALL candidate internships in a single batched embedding call.

    Before this fix the pipeline was:
      for each job:
          embedder.encode(job_text)          # 1 call per job  → 100+ serial calls
          embedder.encode(fieldOfStudy)      # 2 calls per field per job → 200+ more
      Total: 300+ individual encode calls, one 'Batches: 100%' line each.

    After:
      1. Pre-compute field embeddings once (already done by _precompute_field_embeddings)
      2. Build all 100 job texts → ONE batched encode call
      3. Cosine similarity for all jobs at once via util.cos_sim (matrix op)
      4. score_job() uses the pre-computed tensors — no encode() calls inside
    Result: 300+ serial calls → 2 batched calls total (student + all jobs).
    """
    # Pre-compute field embeddings once for this request
    field_embeddings = _precompute_field_embeddings(student)

    # Build all job texts at once
    job_texts = []
    for job in candidates:
        text = ' '.join(filter(
            None,
            [job.get('jobTitle', ''), job.get('jobDescription', '')] +
            job.get('qualifications', [])
        ))
        job_texts.append(text or 'internship')

    # ONE batched encode for all 100 jobs
    job_embeddings = embedder.encode(job_texts, batch_size=64)

    # Cosine similarity: student vs all jobs at once → shape (N,)
    sim_scores = util.cos_sim(student_embedding, job_embeddings)[0]

    scored = []
    for i, job in enumerate(candidates):
        base = score_job(job, signals, student, dominant_traits,
                         field_embeddings=field_embeddings,
                         job_embedding=job_embeddings[i])
        total = base + float(sim_scores[i].item()) * 10
        scored.append({'job': job, 'score': total})

    return scored


# ── Claude re-ranking ─────────────────────────────────────────────────────────

async def claude_rerank(
    top_candidates: List[Dict[str, Any]],
    student: Dict[str, Any],
    signals: Dict[str, Any],
    dominant_traits: List[str],
    limit: int,
) -> List[Dict[str, Any]]:
    """
    Uses Claude to re-rank the embedding-scored shortlist and attach a human-readable
    match_reason to each job.

    Strategy
    --------
    1. Your embedding + RIASEC scoring already filters 100 → top ~20 good candidates.
    2. Claude receives only those ~20, reads the student profile, and returns:
       - A re-ranked order (best fit first)
       - A 1-2 sentence match_reason per job explaining WHY it fits
    3. If Claude fails for any reason (API error, timeout, bad JSON), the function
       returns the original embedding-ranked list unchanged — zero impact on users.
    4. Results are cached like everything else, so Claude is called at most once
       per CACHE_TTL_HOURS per student. Cost is negligible.

    Token budget
    ------------
    Each job is summarised to ~80 tokens before sending. With 20 jobs that's
    ~1600 tokens in + ~600 tokens out = ~2200 tokens per call ≈ $0.001 per student
    per 6 hours at Sonnet pricing.
    """
    client = _get_anthropic_client()
    if not client:
        print("[claude] ANTHROPIC_API_KEY not set — skipping re-rank, using embedding order")
        return top_candidates[:limit]

    # ── Build a compact student profile summary for the prompt ───────────────
    student_summary = {
        "name":           student.get("name", "Student"),
        "field_of_study": student.get("fieldOfStudy", ""),
        "desired_field":  student.get("desiredField", ""),
        "skills":         signals.get("skills", [])[:15],
        "desired_roles":  signals.get("roles", [])[:8],
        "locations":      signals.get("locations", [])[:5],
        "education_level": student.get("educationLevel", ""),
        "riasec_traits":  dominant_traits,
        "career_level":   signals.get("highestLevel", 1),
    }

    # ── Build compact job summaries — only what Claude needs ─────────────────
    # We intentionally strip heavy fields (full description) to stay token-lean.
    job_summaries = []
    for i, job in enumerate(top_candidates):
        job_summaries.append({
            "index":          i,                                          # used to map back
            "id":             str(job.get("_id", "")),
            "title":          job.get("jobTitle", ""),
            "company":        job.get("companyName", ""),
            "sector":         job.get("sector", ""),
            "location":       job.get("location", ""),
            "mode":           job.get("internshipMode", ""),
            "level":          job.get("classification", ""),
            "skills_needed":  job.get("qualifications", [])[:8],
            "description":    (job.get("jobDescription", "") or "")[:300],  # first 300 chars only
        })

    prompt = f"""You are a career advisor helping match students to internships.

STUDENT PROFILE:
{json.dumps(student_summary, indent=2)}

INTERNSHIP CANDIDATES (already pre-filtered by an embedding model — all are broadly relevant):
{json.dumps(job_summaries, indent=2)}

YOUR TASK:
1. Re-rank these internships from best to worst fit for THIS specific student.
2. For each internship write a match_reason (1 short sentence, max 15 words).
3. Only return the top {limit} internships.

RULES:
- Prioritise skill overlap, career level fit, field alignment, and RIASEC personality traits.
- If the student has no skills/roles yet, prioritise entry-level and broad roles.
- Return ONLY valid JSON. No explanation, no markdown, no code fences, no trailing commas.

RESPONSE FORMAT (strict JSON array, exactly {limit} items):
[
  {{
    "index": <original index from the candidates list>,
    "match_reason": "<1 sentence, max 15 words>"
  }},
  ...
]"""

    try:
        print(f"[claude] Re-ranking {len(top_candidates)} candidates → top {limit} ...")
        response = await asyncio.wait_for(
            client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=1200,
                messages=[{"role": "user", "content": prompt}],
            ),
            timeout=25.0,
        )

        raw_text = response.content[0].text.strip()

        # Strip markdown fences if Claude adds them despite instructions
        if raw_text.startswith("```"):
            raw_text = raw_text.split("```")[1]
            if raw_text.startswith("json"):
                raw_text = raw_text[4:]
        raw_text = raw_text.strip()

        ranked = json.loads(raw_text)

        # ── Map Claude's ranking back to the original job dicts ───────────────
        result = []
        seen_indices = set()
        for item in ranked:
            idx = item.get("index")
            if idx is None or idx in seen_indices or idx >= len(top_candidates):
                continue
            seen_indices.add(idx)
            job = dict(top_candidates[idx])                  # copy so original is untouched
            job["match_reason"] = item.get("match_reason", "")
            result.append(job)

        # Safety: if Claude returned fewer than limit, pad with remaining embedding-ranked jobs
        if len(result) < limit:
            for i, job in enumerate(top_candidates):
                if i not in seen_indices and len(result) < limit:
                    padded = dict(job)
                    padded["match_reason"] = ""
                    result.append(padded)

        print(f"[claude] Re-ranking complete. Returning {len(result)} jobs.")
        return result

    except json.JSONDecodeError as e:
        print(f"[claude] JSON parse error — falling back to embedding order. Error: {e}")
        return top_candidates[:limit]
    except anthropic.APIError as e:
        print(f"[claude] API error — falling back to embedding order. Error: {e}")
        return top_candidates[:limit]
    except Exception as e:
        print(f"[claude] Unexpected error — falling back to embedding order. Error: {e}")
        return top_candidates[:limit]


# ── API Endpoints ─────────────────────────────────────────────────────────────

@app.get('/recommendations/{student_id}', response_class=ORJSONResponse)
async def get_personalized_recommendations(
    student_id: str,
    limit: int = 6
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Returns personalised internship recommendations for a student.

    FIX #1  — Only approved internships are returned.
              All three queries (primary, sector-fallback, score-fallback) now
              include adminApproved: True so unapproved postings are never
              surfaced, regardless of whether applicationOpen is True.

    FIX #2  — Results are cached in MongoDB for CACHE_TTL_HOURS (default 6 h).
              Repeated visits within that window are served in <200 ms instead
              of triggering a full 20-40 s recompute.  The cache is busted
              automatically via DELETE /recommendations/{student_id}/cache
              (called by the Node service whenever the student submits a new
              application or updates their profile).
    """
    if not ObjectId.is_valid(student_id):
        raise HTTPException(status_code=400, detail="Invalid student ID")

    student_id_obj = ObjectId(student_id)

    # ── Cache check ───────────────────────────────────────────────────────────
    async def _read_cache():
        try:
            cached = await _cache_collection().find_one({'studentId': student_id_obj})
            if cached:
                cached_at: datetime = cached.get('cachedAt')
                if cached_at:
                    if cached_at.tzinfo is None:
                        cached_at = cached_at.replace(tzinfo=timezone.utc)
                    age = datetime.now(timezone.utc) - cached_at
                    if age < timedelta(hours=CACHE_TTL_HOURS):
                        return cached['recommendations'][:limit]
        except Exception as cache_err:
            print(f"[cache] Read error (non-fatal): {cache_err}")
        return None

    cached_result = await _read_cache()
    if cached_result is not None:
        print(f"[cache HIT]  student={student_id}")
        return {'recommendations': cached_result}

    # ── FIX: in-flight deduplication ─────────────────────────────────────────
    # If another coroutine is already computing for this student, wait for it
    # then return from cache — avoids double pipeline run + double Claude call.
    if student_id in _inflight:
        print(f"[inflight] Waiting for in-progress request for student={student_id}...")
        await _inflight[student_id].wait()
        cached_result = await _read_cache()
        if cached_result is not None:
            return {'recommendations': cached_result}
        # If cache still empty after wait, fall through and compute anyway
    
    event = asyncio.Event()
    _inflight[student_id] = event

    print(f"[cache MISS] Generating fresh recommendations for student={student_id}")

    # ── Load student profile ──────────────────────────────────────────────────
    student = await _user_collection().find_one({'_id': student_id_obj})
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")

    # ── Build signals from profile + completed applications ───────────────────
    signals = await derive_signals(student)
    inferred = await infer_from_recent_applications(student_id_obj)

    highest_level = inferred.get('highestLevel', 1) or 1

    signals = {
        'skills':       list(set(signals['skills']).union(inferred['skills'])),
        'roles':        list(set(signals['roles']).union(inferred['roles'])),
        'locations':    list(set(signals['locations']).union(inferred['locations'])),
        'highestLevel': highest_level
    }

    personality = await get_personality(student_id_obj)
    dominant_traits = personality.get('dominantTraits', [])

    applied_ids = await _application_collection().distinct('internshipId', {'studentId': student_id_obj})

    # ── Map RIASEC traits → sectors ───────────────────────────────────────────
    matched_sectors = set()
    for trait in dominant_traits:
        matched_sectors.update(RIASEC_SECTOR_MAP.get(trait, []))

    # ── PRIMARY QUERY — adminApproved + open + sector filter ────────────────
    # adminApproved: True  → only postings explicitly approved by an admin
    # adminReviewed: True  → posting has been reviewed (not just sitting pending)
    # applicationOpen: True → still accepting applicants
    # deleted: False        → not soft-deleted
    query: Dict[str, Any] = {
        'adminApproved':   True,
        'adminReviewed':   True,
        'deleted':         False,
        'applicationOpen': True,
        '_id':             {'$nin': applied_ids}
    }
    if matched_sectors:
        query['sector'] = {'$in': list(matched_sectors)}

    candidates_cursor = _internship_collection().find(query).limit(100)
    candidates = await candidates_cursor.to_list(length=100)
    print(f"[rec] Primary query returned {len(candidates)} candidates (sectors={list(matched_sectors) or 'none'})")

    # ── SECTOR FALLBACK — adminApproved + open, no sector restriction ─────────
    # Reached when the RIASEC-filtered query returns nothing.
    if not candidates:
        print(f"[rec] No sector matches — falling back to full approved open pool")
        query = {
            'adminApproved':   True,
            'adminReviewed':   True,
            'deleted':         False,
            'applicationOpen': True,
            '_id':             {'$nin': applied_ids}
        }
        candidates_cursor = _internship_collection().find(query).limit(100)
        candidates = await candidates_cursor.to_list(length=100)
        print(f"[rec] Sector fallback returned {len(candidates)} candidates")

    # ── Score every candidate — batched, no per-job encode calls ────────────
    student_text = ' '.join(signals['skills'] + signals['roles'] + signals['locations'])
    if not student_text.strip():
        student_text = "internship opportunity"

    # Encode the student profile (one call)
    loop = asyncio.get_event_loop()
    student_embedding = await loop.run_in_executor(
        None, lambda: embedder.encode(student_text)
    )

    # Batch-encode all job texts + score in one executor call (no event loop blocking)
    scored_jobs = await loop.run_in_executor(
        None, batch_score_jobs, candidates, signals, student_embedding, student, dominant_traits
    )
    scored_jobs.sort(key=lambda x: x['score'], reverse=True)

    if scored_jobs:
        print(f"[rec] Score range: {scored_jobs[-1]['score']:.2f} – {scored_jobs[0]['score']:.2f}")

    # ── Take top 20 from embedding scorer → feed to Claude re-ranker ──────────
    # We pass 20 (not just `limit`) so Claude has enough candidates to make
    # intelligent ranking decisions before trimming to the requested limit.
    PRE_RANK_POOL = 12  # reduced from 20 — fewer candidates = shorter Claude response, less chance of truncation
    embedding_top = [item['job'] for item in scored_jobs[:PRE_RANK_POOL]]

    # ── Claude re-ranks the pool and attaches match_reason per job ────────────
    # Falls back silently to embedding order if Claude is unavailable or errors.
    if embedding_top:
        final_list = await claude_rerank(
            top_candidates=embedding_top,
            student=student,
            signals=signals,
            dominant_traits=dominant_traits,
            limit=limit,
        )
    else:
        final_list = []

    # ── RECENCY FALLBACK — only when the candidate pool was truly empty ───────
    # This means adminApproved+applicationOpen returned zero documents.
    # We drop the classification restriction so ANY approved open job is shown.
    if not final_list:
        print(f"[rec] Candidate pool was empty — using recency fallback (no classification filter)")
        fallback_cursor = _internship_collection().find({
            'adminApproved':   True,
            'adminReviewed':   True,
            'deleted':         False,
            'applicationOpen': True,
            '_id':             {'$nin': applied_ids}
        }).sort('createdAt', -1).limit(limit)
        fallback_docs = await fallback_cursor.to_list(length=limit)
        final_list = fallback_docs
        print(f"[rec] Recency fallback returned {len(final_list)} docs")

    # Convert ObjectIds to strings for JSON serialisation
    final_list = convert_object_ids(final_list)

    # ── Persist to cache — only when we have actual results ──────────────────
    if final_list:
        try:
            await _cache_collection().update_one(
                {'studentId': student_id_obj},
                {'$set': {
                    'studentId':       student_id_obj,
                    'recommendations': final_list,
                    'cachedAt':        datetime.now(timezone.utc)
                }},
                upsert=True
            )
            print(f"[cache] Saved {len(final_list)} recommendations for student={student_id}")
        except Exception as cache_err:
            print(f"[cache] Write error (non-fatal): {cache_err}")
    else:
        print(f"[cache] Skipping cache — empty result set (will retry on next request)")

    # ── FIX: release in-flight lock so any waiting requests can read from cache
    if student_id in _inflight:
        _inflight.pop(student_id).set()   # unblock all waiters

    return {'recommendations': final_list}


@app.delete('/recommendations/{student_id}/cache', response_class=ORJSONResponse)
async def bust_recommendation_cache(student_id: str):
    """
    FIX #2 — Cache invalidation endpoint.

    Call this from Node.js (AiServices.js) whenever an event makes the cached
    recommendations stale:
      • Student submits a new application
      • Student updates their profile / skills / desired roles
      • An internship posting is approved or closed by admin

    The next GET /recommendations/{student_id} will then recompute fresh results.
    """
    if not ObjectId.is_valid(student_id):
        raise HTTPException(status_code=400, detail="Invalid student ID")

    result = await _cache_collection().delete_one({'studentId': ObjectId(student_id)})
    deleted = result.deleted_count > 0
    print(f"[cache] Busted cache for student={student_id}  (found={deleted})")
    return {'status': 'cache cleared', 'was_cached': deleted}