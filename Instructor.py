from __future__ import annotations

# FastAPI service to run on :8003
from fastapi import FastAPI, Body, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import os
import sys
import json
from datetime import datetime, time
from typing import Any, Dict, List, Optional, Tuple, DefaultDict
from collections import defaultdict
from fastapi.responses import RedirectResponse, Response

# 👇 Load env before using os.getenv
from dotenv import load_dotenv
load_dotenv()

try:
    from pymongo import MongoClient
    from bson import ObjectId
except Exception as e:  # pragma: no cover
    print(json.dumps({"ok": False, "error": f"Missing dependency: {e}"}))
    sys.exit(1)

# -----------------------------
# Environment & configuration
# -----------------------------
# Use server host only in URI; pick DB by DB_NAME
MONGO_URI = os.getenv("MONGO_URI", "mongodb://127.0.0.1:27017")
DB_NAME   = os.getenv("DB_NAME", "skillnaav-land")  # <-- your DB in Compass

INSTRUCTORS_COLL = os.getenv("INSTRUCTORS_COLL", "")
SCHEDULES_COLL   = os.getenv("SCHEDULES_COLL", "")
OFFERS_COLL      = os.getenv("OFFERS_COLL", "")
USERS_COLL       = os.getenv("USERS_COLL", "")

# Fallback guesses (auto-resolved against db.list_collection_names())
GUESSES = {
    "instructors": [
        "instructors", "instructures", "instructor", "Instructure",
        # 👇 common for mongoose.model('InstructureManagement', ...) -> 'instructuremanagements'
        "instructuremanagements", "InstructureManagement",
        "instructorManagements", "InstructorManagement"
    ],
    "schedules": [
        "internshipschedules", "internshipSchedules",
        "internship_schedule", "InternshipSchedule"
    ],
    "offers": ["offerletters", "offerLetters", "offers", "OfferLetters"],
    "users":  ["users", "students", "user", "student", "Users"],
}

# -----------------------------
# Small helpers
# -----------------------------
def _resolve_collection(db, explicit: str, guesses: List[str]) -> str:
    cols = set(db.list_collection_names())
    if explicit and explicit in cols:
        return explicit
    for g in guesses:
        if g in cols:
            return g
    return guesses[0]

def _is_blank(x: Any) -> bool:
    return x is None or (isinstance(x, str) and not x.strip())

def _display_name(instr: Dict[str, Any]) -> str:
    for k in ("instructorName", "name", "fullName", "displayName"):
        v = instr.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    # 👇 add this line for your schema
    if "firstName" not in instr and "lastName" not in instr:
        if "Specializations" in instr or "Skills" in instr:
            return instr.get("email") or str(instr.get("_id"))
    fn = instr.get("firstName") or instr.get("firstname") or ""
    ln = instr.get("lastName")  or instr.get("lastname")  or ""
    nm = f"{fn} {ln}".strip()
    return nm or str(instr.get("_id"))

def _parse_hhmm(s: Optional[str]) -> Optional[time]:
    if not s or not isinstance(s, str): return None
    s = s.strip()
    for f in ("%H:%M", "%I:%M %p"):
        try:
            return datetime.strptime(s, f).time()
        except Exception:
            pass
    return None

def _extract_availability_window(instr: Dict[str, Any]) -> Tuple[Optional[time], Optional[time]]:
    """
    Optional availability window on the instructor. If absent, we treat as fully available.
    Supported keys:
      - availabilityStart / availabilityEnd
      - availableStart / availableEnd
      - availableFrom / availableTo
      - startTime / endTime
      - workHours: { start|from, end|to }
    """
    start = (
        instr.get("availabilityStart")
        or instr.get("availableStart")
        or instr.get("availableFrom")
        or instr.get("startTime")
        or instr.get("workStart")
    )
    end = (
        instr.get("availabilityEnd")
        or instr.get("availableEnd")
        or instr.get("availableTo")
        or instr.get("endTime")
        or instr.get("workEnd")
    )
    if not start or not end:
        avail = instr.get("availability") or instr.get("workHours") or {}
        if isinstance(avail, dict):
            start = start or avail.get("start") or avail.get("from")
            end   = end   or avail.get("end")   or avail.get("to")
    return _parse_hhmm(start), _parse_hhmm(end)

def _parse_session_time(v: Any) -> Optional[time]:
    if not v: return None
    if isinstance(v, time): return v
    if isinstance(v, (int, float)):
        s = f"{int(v):04d}"
        hh, mm = int(s[:-2]), int(s[-2:])
        return time(hh, mm)
    if isinstance(v, str):
        v = v.strip()
        for f in ("%H:%M", "%I:%M %p"):
            try:
                return datetime.strptime(v, f).time()
            except Exception:
                pass
    return None

def _session_times(sess: Dict[str, Any]) -> Tuple[Optional[time], Optional[time]]:
    s = sess.get("startTime") or sess.get("start") or sess.get("from")
    e = sess.get("endTime")   or sess.get("end")   or sess.get("to")
    return _parse_session_time(s), _parse_session_time(e)

def _within(a_start: Optional[time], a_end: Optional[time], s: Optional[time], e: Optional[time]) -> bool:
    # If either side lacks times, don't block assignment.
    if s is None or e is None: return True
    if a_start is None or a_end is None: return True
    si, ei = s.hour*60 + s.minute, e.hour*60 + e.minute
    ai, bi = a_start.hour*60 + a_start.minute, a_end.hour*60 + a_end.minute
    return ai <= si and ei <= bi

# -----------------------------
# Core preparation & picking
# -----------------------------
def _load_instructors(coll, partner_id: Optional[str] = None) -> List[Dict[str, Any]]:
    # If instructors are tagged with partnerId, filter; otherwise fall back to all
    query: Dict[str, Any] = {}
    if partner_id:
        ors = []
        try:
            ors.append({"partnerId": ObjectId(partner_id)})
        except Exception:
            pass
        ors.append({"partnerId": partner_id})
        query = {"$or": ors}

    docs = list(coll.find(query))
    if not docs and partner_id:
        # Partner-specific list empty? fall back to all instructors.
        docs = list(coll.find({}))

    prepared = []
    for ins in docs:
        a_start, a_end = _extract_availability_window(ins)
        prepared.append({
            "_id": ins["_id"],
            "name": _display_name(ins),  # builds "ASRITH CHOWDARY" from firstName+lastName if 'name' missing
            "a_start": a_start,
            "a_end":   a_end,
            "raw": ins
        })
    return prepared

def _initial_load_for_internship(instructors: List[Dict[str, Any]], timetable: List[Dict[str, Any]]) -> DefaultDict[str, int]:
    load = defaultdict(int)
    for sess in timetable or []:
        nm = sess.get("instructor") or sess.get("instructorName")
        if isinstance(nm, str) and nm.strip():
            load[nm.strip()] += 1
    for ins in instructors:
        load[ins["name"]] += 0
    return load

def _pick_instructor(instructors: List[Dict[str, Any]],
                     load_map: DefaultDict[str, int],
                     s_time: Optional[time],
                     e_time: Optional[time]) -> Optional[Dict[str, Any]]:
    candidates = [c for c in instructors if _within(c["a_start"], c["a_end"], s_time, e_time)]
    if not candidates:
        candidates = instructors[:]  # fallback if no availability data matches
    candidates.sort(key=lambda c: (load_map[c["name"]], c["name"]))
    return candidates[0] if candidates else None

# -----------------------------
# Assignment engine
# -----------------------------
def assign_instructors(partner_id: Optional[str] = None) -> Dict[str, Any]:
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]

    instructors_coll = db[_resolve_collection(db, INSTRUCTORS_COLL, GUESSES["instructors"])]
    schedules_coll   = db[_resolve_collection(db, SCHEDULES_COLL,   GUESSES["schedules"])]

    instructors = _load_instructors(instructors_coll, partner_id)

    # 👉 Fetch ALL schedules with timetables (ignore partnerId, restrictions, etc.)
    schedules = list(schedules_coll.find({"timetable": {"$exists": True, "$ne": []}}))

    total_sessions_assigned = 0
    per_schedule_reports: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []

    for sched in schedules:
        timetable: List[Dict[str, Any]] = sched.get("timetable") or []
        if not timetable:
            continue

        internship_id = sched.get("internshipId")
        load_map = _initial_load_for_internship(instructors, timetable)

        sessions_updated = 0
        changed = False

        for sess in timetable:
            # 👉 Always assign (overwrite if already exists)
            # inside the for sess in timetable loop (no “already has instructor” checks)
            s_time, e_time = _session_times(sess)
            chosen = _pick_instructor(instructors, load_map, s_time, e_time)
            if not chosen:
                skipped.append({
                    "internshipId": str(internship_id),
                    "scheduleId": str(sched.get("_id")),
                    "date": sess.get("date"),
                    "reason": "No available instructors"
                })
                continue

            raw = chosen.get("raw", {}) if isinstance(chosen.get("raw"), dict) else {}
            first = (raw.get("firstName") or "").strip()
            last  = (raw.get("lastName")  or "").strip()
            full_name = f"{first} {last}".strip() or _display_name(raw)

            # ✅ set both fields – some UI uses instructorName
            sess["instructor"] = full_name
            sess["instructorName"] = full_name
            sess["instructorInfo"] = {
                "firstName": first,
                "lastName":  last,
                "instructorId": str(raw.get("_id") or "")
            }
            try:
                sess["instructorId"] = raw["_id"]
            except Exception:
                pass

            load_map[chosen["name"]] += 1
            sessions_updated += 1
            total_sessions_assigned += 1
            changed = True

        if changed:
            res = schedules_coll.update_one(
                {"_id": sched["_id"]},
                {"$set": {"timetable": timetable, "updatedAt": datetime.utcnow()}}
            )
            print(f"[assign] schedule={sched['_id']} matched={res.matched_count} modified={res.modified_count}")
            per_schedule_reports.append({
                "internshipId": str(internship_id),
                "scheduleId": str(sched["_id"]),
                "sessionsUpdated": sessions_updated,
                "fieldsSet": ["instructor", "instructorName"]
            })

    return {
    "ok": True,
    "instructors": len(instructors),
    "schedules_scanned": len(schedules),
    "assignments_made": total_sessions_assigned,
    "assignments": per_schedule_reports,
    "skipped": skipped,
    "debug": {
        "db": DB_NAME,
        "instructors_coll": instructors_coll.name,
        "schedules_coll": schedules_coll.name,
        "instructors_found": len(instructors),
        "schedules_found": len(schedules),
    },
}

# -----------------------------
# FastAPI service
# -----------------------------
class AssignPayload(BaseModel):
    partnerId: Optional[str] = None

instructor_app = FastAPI(title="Instructor Assignment API", version="2.0.0")
instructor_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@instructor_app.get("/health")
def health():
    return {"ok": True, "service": "instructor-assignment", "port": 8000}

@instructor_app.post("/assign-instructors")
def assign_instructors_http(payload: AssignPayload = Body(default=None),
                            partnerId: Optional[str] = Query(default=None)):
    pid = partnerId or (payload.partnerId if payload else None)
    return assign_instructors(partner_id=pid)

@instructor_app.get("/assign-instructors")
def assign_instructors_http_get(partnerId: Optional[str] = Query(default=None)):
    return assign_instructors(partner_id=partnerId)

@instructor_app.get("/")
def root():
    # Send browser to the interactive docs instead of 404
    return RedirectResponse(url="/docs")

@instructor_app.get("/favicon.ico")
def favicon():
    # Empty 200 response so the browser doesn't log a 404 for the tab icon
    return Response(content=b"", media_type="image/x-icon",
                    headers={"Cache-Control": "public, max-age=86400"})

if __name__ == "__main__":
    # Local debug (unified): uvicorn app:app --reload --port 8000
    print(json.dumps(assign_instructors(partner_id=None), default=str))