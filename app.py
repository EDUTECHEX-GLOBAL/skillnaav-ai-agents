"""
SkillNaav Unified API  –  v3.1
================================
Combines all 4 agents into ONE FastAPI app on ONE port (default 8000).

Strategy
--------
Instead of mount() (which creates isolated ASGI sub-apps and breaks
middleware/route resolution), we copy every route from each module's
`app` object directly onto a single root FastAPI instance.
This keeps ALL existing URL paths exactly as they were – zero frontend changes.
for changes for commit

Existing URL map (unchanged)
-----------------------------
  POST /analyze-skills/                  <- main.py        (was :8000)
  GET  /health                           <- main.py
  GET  /recommendations/{student_id}     <- recommendation.py (was :8002)
  POST /partner/shortlist                <- partner.py     (was :8001)
  GET  /partner/shortlisted/{id}         <- partner.py
  GET  /partner/shortlisted/by-admin     <- partner.py
  GET  /partner/fetch-applications/{id}  <- partner.py
  POST /extract-resume                   <- partner.py
  POST /cv/generate                      <- partner.py
  POST /assign-instructors               <- Instructor.py  (was :8003)
  GET  /assign-instructors               <- Instructor.py

Run with:
  uvicorn app:app --reload --port 8000
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.routing import APIRoute

# Import every module so their top-level code runs (DB connections, model
# loading, etc.) and their `app` objects are populated with routes.
import main           as _main        # Bedrock / skill-gap analysis   (was :8000)
import partner        as _partner     # Shortlisting, CV, resume parse  (was :8001)
import recommendation as _recommend   # Personalised recommendations    (was :8002)
import Instructor     as _instructor  # Instructor assignment engine     (was :8003)

# ── Root application ──────────────────────────────────────────────────────────
app = FastAPI(
    title="SkillNaav Unified API",
    description=(
        "All four AI agents on a single port — no URL changes needed.\n\n"
        "| Routes | Agent |\n"
        "|--------|-------|\n"
        "| `/analyze-skills/`, `/health` | Resume skill analysis (Bedrock) |\n"
        "| `/partner/*`, `/extract-resume`, `/cv/generate` | Partner / CV |\n"
        "| `/recommendations/{id}` | Personalised recommendations |\n"
        "| `/assign-instructors` | Instructor assignment |\n"
    ),
    version="3.1.0",
)

# ── CORS – union of every allowed origin across all four original services ────
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://www.skillnaav.com",
        "https://skillnaav.com",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Helper: absorb all routes from a sub-app into root app ───────────────────
def _absorb(source_app, label: str):
    """
    Walk source_app.routes and register each APIRoute on the root app.
    Skips built-in FastAPI meta-routes (/openapi.json, /docs, /redoc).
    """
    skip = {"/openapi.json", "/docs", "/docs/oauth2-redirect", "/redoc", "/favicon.ico"}
    for route in source_app.routes:
        if not isinstance(route, APIRoute):
            continue
        if route.path in skip:
            continue
        # Prefix the name so collisions are visible, not silent
        route.name = f"{label}__{route.name}"
        app.routes.append(route)
        print(f"  [unified] + {sorted(route.methods)} {route.path}  ({label})")

print("\n[unified] Registering routes ...")
_absorb(_main.app,        "skills")
_absorb(_partner.app,     "partner")
_absorb(_recommend.app,   "recommend")
_absorb(_instructor.instructor_app,  "instructor")
print("[unified] All routes registered.\n")

# ── Root convenience route ────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")

# ── Entry-point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)