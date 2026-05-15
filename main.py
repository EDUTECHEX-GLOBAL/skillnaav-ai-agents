from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
import os
import fitz  # PyMuPDF for PDFs
import docx  # python-docx for DOCX files
import logging
import spacy
import boto3 # type: ignore
from dotenv import load_dotenv
import time
import json
import re
import traceback  # For error logging
from datetime import datetime  # For timestamp utility

# Load environment variables
load_dotenv()
aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
aws_region = os.getenv("AWS_REGION")
if not all([aws_access_key_id, aws_secret_access_key, aws_region]):
    raise ValueError("Missing AWS credentials or region. Check your .env file!")

# Initialize Bedrock client
bedrock_client = boto3.client(
    service_name="bedrock-runtime",
    region_name=aws_region,
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key
)

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://www.skillnaav.com", "https://skillnaav.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load NLP model
nlp = spacy.load("en_core_web_sm")

# Utility function for current timestamp
def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Extract text from PDF
def extract_text_from_pdf(pdf_file):
    pdf_file.file.seek(0)  # Reset file pointer before reading
    doc = fitz.open(stream=pdf_file.file.read(), filetype="pdf")
    return "\n".join(page.get_text() for page in doc)

# Extract text from DOCX
def extract_text_from_docx(docx_file):
    doc = docx.Document(docx_file.file)
    return "\n".join([para.text for para in doc.paragraphs])

# Expanded predefined technical skills to improve extraction
TECH_SKILLS = {
    # Programming Languages
    "python", "java", "javascript", "c++", "c#", "go", "rust", "swift", "kotlin", "scala",
    "ruby", "php", "typescript", "r", "matlab", "perl", "shell", "bash",
    
    # Web Technologies
    "html", "css", "react", "angular", "vue", "svelte", "node.js", "express.js", 
    "django", "flask", "spring", "laravel", "rails", "asp.net", "jquery",
    "bootstrap", "tailwind", "sass", "less", "webpack", "vite",
    
    # Databases
    "sql", "mysql", "postgresql", "mongodb", "redis", "elasticsearch", "cassandra",
    "oracle", "sqlite", "dynamodb", "neo4j", "influxdb",
    
    # Cloud & DevOps
    "aws", "azure", "gcp", "docker", "kubernetes", "terraform", "ansible",
    "jenkins", "gitlab", "github", "ci/cd", "nginx", "apache", "microservices",
    
    # Data Science & AI
    "machine learning", "deep learning", "tensorflow", "pytorch", "keras", "pandas", 
    "numpy", "scikit-learn", "matplotlib", "seaborn", "jupyter", "anaconda",
    "spark", "hadoop", "kafka", "airflow", "mlflow",
    
    # Quantum Computing
    "quantum computing", "quantum algorithms", "qubits", "circuit simulation",
    "quantum gates", "quantum entanglement", "quantum superposition", "qiskit",
    "cirq", "quantum annealing", "quantum cryptography",
    
    # Emerging Technologies
    "blockchain", "ethereum", "solidity", "web3", "nft", "defi", "smart contracts",
    "iot", "edge computing", "5g", "ar", "vr", "metaverse",
    
    # Tools & Methodologies
    "git", "agile", "scrum", "kanban", "jira", "confluence", "slack", "teams",
    "figma", "sketch", "photoshop", "illustrator", "unity", "unreal engine",
    
    # APIs & Protocols
    "rest api", "graphql", "grpc", "websocket", "oauth", "jwt", "soap", "xml", "json",
    
    # Testing & Quality
    "unit testing", "integration testing", "selenium", "cypress", "jest", "mocha",
    "pytest", "junit", "tdd", "bdd", "code review",
    
    # Variations for common skills
    "expressjs", "express.js", "react.js", "reactjs", "nodejs", "node.js",
    "vue.js", "vuejs", "angular.js", "angularjs"
}

# Enhanced normalize skill names for better comparison
def normalize_skill_name(skill):
    """Normalize skill names for accurate comparison"""
    if not skill:
        return ""
    
    # Convert to lowercase and strip whitespace
    skill = skill.lower().strip()
    
    # Remove special characters but keep dots, plus signs, and numbers
    skill = re.sub(r'[^\w\s.#+]', '', skill)
    
    # Replace multiple spaces with single space
    skill = re.sub(r'\s+', ' ', skill)
    
    # Remove spaces around dots
    skill = re.sub(r'\s*\.\s*', '.', skill)
    
    # Handle common variations and synonyms
    skill_mappings = {
        "nodejs": "node.js",
        "node js": "node.js",
        "expressjs": "express.js",
        "express js": "express.js",
        "reactjs": "react.js",
        "react js": "react.js",
        "vuejs": "vue.js",
        "vue js": "vue.js",
        "angularjs": "angular.js",
        "angular js": "angular.js",
        "c sharp": "c#",
        "c plus plus": "c++",
        "cpp": "c++",
        "javascript": "javascript",
        "js": "javascript",
        "typescript": "typescript",
        "ts": "typescript",
        "artificial intelligence": "machine learning",
        "ai": "machine learning",
        "ml": "machine learning",
        "deep learning": "deep learning",
        "dl": "deep learning",
        "quantum computing": "quantum computing",
        "quantum algorithms": "quantum algorithms",
        "circuit simulation": "circuit simulation",
        "rest": "rest api",
        "restful": "rest api",
        "restful api": "rest api",
        "continuous integration": "ci/cd",
        "continuous deployment": "ci/cd",
        "amazon web services": "aws",
        "microsoft azure": "azure",
        "google cloud": "gcp",
        "google cloud platform": "gcp"
    }
    
    return skill_mappings.get(skill, skill)

# Invoke Amazon Bedrock model using provided payload and model parameters
def invoke_bedrock(prompt_text):
    try:
        body = {
            "prompt": prompt_text,
            "max_gen_len": 2048,
            "temperature": 0.3,  # Lower temperature for more consistent outputs
            "top_p": 0.9
        }
        response = bedrock_client.invoke_model(
            modelId="meta.llama3-8b-instruct-v1:0",
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json"
        )
        response_body = json.loads(response['body'].read())
        return response_body.get("generation", "")
    except Exception as e:
        logger.error(f"Bedrock Error: {str(e)}")
        logger.error(f"Full Traceback: {traceback.format_exc()}")
        return ""

# Enhanced skill extraction from resume text
def extract_skills_from_resume(text):
    """Extract skills from resume text using both regex and AI"""
    found_skills = set()
    
    # First, extract skills using predefined list
    for skill in TECH_SKILLS:
        # Use word boundaries for better matching
        pattern = rf"\b{re.escape(skill)}\b"
        if re.search(pattern, text, re.IGNORECASE):
            found_skills.add(normalize_skill_name(skill))
    
    # Use Bedrock to extract additional skills with improved prompt
    try:
        prompt = f"""
Extract technical skills from the following resume text. Return only the skills as a comma-separated list.
Focus on:
- Programming languages (Python, Java, JavaScript, etc.)
- Frameworks and libraries (React, Django, TensorFlow, etc.)
- Databases (MySQL, MongoDB, PostgreSQL, etc.)
- Cloud platforms (AWS, Azure, GCP, etc.)
- Tools and technologies (Docker, Kubernetes, Git, etc.)
- Methodologies (Agile, Scrum, DevOps, etc.)
- Emerging technologies (Quantum Computing, Blockchain, etc.)

Resume Text:
{text}

Return format: skill1, skill2, skill3
"""
        response_text = invoke_bedrock(prompt)
        if response_text:
            # Parse the response more carefully
            skills_line = response_text.strip().split('\n')[0]  # Take first line
            extracted_skills = [skill.strip() for skill in skills_line.split(',')]
            
            for skill in extracted_skills:
                if skill and len(skill) > 1:  # Avoid single characters
                    normalized_skill = normalize_skill_name(skill)
                    if normalized_skill:
                        found_skills.add(normalized_skill)
        
        logger.info("Additional skills extracted using Bedrock.")
    except Exception as e:
        logger.error(f"Bedrock Error (Extract Skills): {str(e)}")
        logger.error(f"Full Traceback: {traceback.format_exc()}")
    
    # Convert to list and remove empty strings
    final_skills = [skill for skill in found_skills if skill]
    logger.info(f"Found Skills: {final_skills}")
    return final_skills

# Identify skill gaps with enhanced normalization
def identify_skill_gaps(user_skills, job_skills):
    """Identify missing skills with improved normalization"""
    user_skills_normalized = set()
    job_skills_normalized = set()
    
    # Normalize user skills
    for skill in user_skills:
        normalized = normalize_skill_name(skill)
        if normalized:
            user_skills_normalized.add(normalized)
    
    # Normalize job skills
    for skill in job_skills:
        normalized = normalize_skill_name(skill)
        if normalized:
            job_skills_normalized.add(normalized)
    
    logger.info(f"User Skills (Normalized): {user_skills_normalized}")
    logger.info(f"Job Skills (Normalized): {job_skills_normalized}")
    
    # Find gaps
    skill_gaps = list(job_skills_normalized - user_skills_normalized)
    logger.info(f"Skill Gaps Identified: {skill_gaps}")
    
    return skill_gaps

# Enhanced readiness score calculation
def calculate_readiness_score(user_skills, job_skills):
    """Calculate readiness score with improved matching"""
    if not job_skills:
        return 100  # If no job skills required, user is 100% ready
    
    user_skills_normalized = set()
    job_skills_normalized = set()
    
    # Normalize user skills
    for skill in user_skills:
        normalized = normalize_skill_name(skill)
        if normalized:
            user_skills_normalized.add(normalized)
    
    # Normalize job skills
    for skill in job_skills:
        normalized = normalize_skill_name(skill)
        if normalized:
            job_skills_normalized.add(normalized)
    
    # Calculate intersection
    matching_skills = user_skills_normalized & job_skills_normalized
    
    # Calculate score
    if job_skills_normalized:
        match_score = (len(matching_skills) / len(job_skills_normalized)) * 100
    else:
        match_score = 0
    
    logger.info(f"Matching Skills: {matching_skills}")
    logger.info(f"Match Score: {match_score}")
    
    return round(match_score, 2)

# Enhanced course recommendations with structured output
def generate_course_recommendations(skill_gaps):
    """Generate structured course recommendations with robust JSON parsing"""
    if not skill_gaps:
        return {"message": "No skill gaps detected. You're ready for this role!"}
    
    prompt = f"""
Generate exactly 3 online course recommendations for these skills: {', '.join(skill_gaps)}

Return ONLY a valid JSON array. Do not include any other text, explanations, or markdown formatting.

[
  {{
    "platform": "Coursera",
    "title": "Course Title Here",
    "description": "Brief description here.",
    "duration": "4 weeks"
  }}
]
"""
    
    try:
        time.sleep(1)
        response_text = invoke_bedrock(prompt)
        logger.info(f"Raw Bedrock Response (Courses): {response_text}")
        
        # Step 1: Clean the response by removing markdown blocks
        cleaned_text = response_text.strip()
        
        # Remove markdown code blocks using safe string splitting
        backtick_marker = '`' * 3  # Creates ```
        if backtick_marker in cleaned_text:
            parts = cleaned_text.split(backtick_marker)
            # Take content between first and second occurrence of ```
            if len(parts) >= 3:
                cleaned_text = parts[1]
                # Remove any language identifier (like 'json')
                if cleaned_text.startswith('json'):
                    cleaned_text = cleaned_text[4:].strip()
            else:
                cleaned_text = parts[0] if parts else cleaned_text
        
        # Step 2: Extract JSON array more robustly
        start_bracket = cleaned_text.find('[')
        end_bracket = cleaned_text.rfind(']')
        
        if start_bracket != -1 and end_bracket != -1 and end_bracket > start_bracket:
            json_text = cleaned_text[start_bracket:end_bracket + 1]
            logger.info(f"Extracted JSON: {json_text}")
            
            try:
                courses = json.loads(json_text)
                
                # Step 3: Validate and filter courses
                valid_courses = []
                valid_platforms = {"coursera", "udemy", "edx", "pluralsight", "linkedin learning"}
                
                for course in courses:
                    if (isinstance(course, dict) and 
                        all(key in course for key in ["platform", "title", "description", "duration"]) and
                        course["platform"].lower() in valid_platforms):
                        valid_courses.append(course)
                
                if valid_courses:
                    logger.info(f"Successfully parsed {len(valid_courses)} valid courses")
                    return {"courses": valid_courses}
                    
            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing failed: {e}")
                logger.error(f"Attempted to parse: {json_text[:200]}...")
        
        # Step 4: Fallback to structured text parsing
        logger.warning("JSON parsing failed, using fallback")
        fallback_courses = [
            {
                "platform": "Coursera",
                "title": f"Advanced {skill_gaps[0].title()} Course",
                "description": f"Comprehensive course covering {skill_gaps[0]} fundamentals and advanced concepts.",
                "duration": "4-6 weeks"
            },
            {
                "platform": "Udemy",
                "title": f"Complete {skill_gaps[0] if skill_gaps else 'Technical Skills'} Bootcamp",
                "description": f"Hands-on practical course for mastering {skill_gaps[0] if skill_gaps else 'required skills'}.",
                "duration": "10-15 hours"
            },
            {
                "platform": "edX",
                "title": f"Professional {skill_gaps[0].title() if skill_gaps else 'Development'} Certificate",
                "description": f"Industry-recognized certification program for {skill_gaps[0] if skill_gaps else 'professional skills'}.",
                "duration": "8-12 weeks"
            }
        ]
        return {"courses": fallback_courses}
        
    except Exception as e:
        logger.error(f"Course recommendation error: {e}")
        logger.error(f"Full Traceback: {traceback.format_exc()}")
        return {
            "error": "Unable to generate recommendations",
            "fallback": [
                "Visit Coursera.org for comprehensive courses",
                "Check Udemy.com for practical tutorials", 
                "Browse edX.org for university-level content"
            ]
        }


# Enhanced quiz generation with better validation
def generate_quizzes(skill_gaps):
    if not skill_gaps:
        return []

    selected_skills = skill_gaps[:3]

    prompt = f"""
Generate exactly 3 multiple-choice quiz questions for these skills: {', '.join(selected_skills)}.

Format like this:
Question: ...
Options:
A. ...
B. ...
C. ...
D. ...
Answer: A
Skill: ...

Do not use code blocks or JSON. Return plain text only.
"""

    try:
        time.sleep(1)
        response_text = invoke_bedrock(prompt)
        logger.info(f"Raw Bedrock Response (Plain Quiz): {response_text}")

        # Split questions
        pattern = r"Question:\s*(.*?)\nOptions:\s*(A\..*?)\nAnswer:\s*([A-D])\nSkill:\s*(.*?)\n?(?=Question:|$)"
        matches = re.findall(pattern, response_text, re.DOTALL)

        quizzes = []
        for question, options_block, answer, skill in matches:
            options = [opt.strip() for opt in options_block.strip().split('\n')]
            quizzes.append({
                "question": question.strip(),
                "options": options,
                "answer": answer.strip(),
                "skill": skill.strip()
            })

        if quizzes:
            logger.info(f"Parsed {len(quizzes)} quizzes from plain text.")
            return quizzes
        else:
            return {"error": "No quizzes found in plain text response", "raw": response_text}

    except Exception as e:
        logger.error(f"Quiz parsing failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {"error": "Error generating quizzes", "raw": ""}

# Enhanced main API endpoint
@app.post("/analyze-skills/")
async def analyze_skills(
    file: UploadFile = File(...),
    job_description: str = Form(...),
    required_skills: str = Form(...)
):
    """Analyze skills from resume and compare with job requirements"""
    try:
        # Validate file type
        if not file.filename.endswith((".pdf", ".docx")):
            raise HTTPException(
                status_code=400, 
                detail="Invalid file type. Only PDF and DOCX files are supported."
            )
        
        # Extract text from file
        if file.filename.endswith(".pdf"):
            resume_text = extract_text_from_pdf(file)
        else:
            resume_text = extract_text_from_docx(file)
        
        # Validate extracted text
        if not resume_text or len(resume_text.strip()) < 50:
            raise HTTPException(
                status_code=400, 
                detail="Unable to extract sufficient text from resume. Please check the file."
            )
        
        logger.info(f"Extracted Resume Text Length: {len(resume_text)}")
        
        # Extract skills from resume
        user_skills = extract_skills_from_resume(resume_text)
        logger.info(f"Extracted User Skills: {user_skills}")
        
        if not user_skills:
            raise HTTPException(
                status_code=400, 
                detail="No technical skills found in the resume. Please ensure your resume contains relevant technical skills."
            )
        
        # Parse job skills
        job_skills = [skill.strip() for skill in required_skills.split(",") if skill.strip()]
        
        if not job_skills:
            raise HTTPException(
                status_code=400, 
                detail="No job skills provided. Please specify required skills."
            )
        
        # Calculate metrics
        skill_gaps = identify_skill_gaps(user_skills, job_skills)
        readiness_score = calculate_readiness_score(user_skills, job_skills)
        
        # Generate recommendations and quizzes
        recommendations = generate_course_recommendations(skill_gaps)
        quizzes = generate_quizzes(skill_gaps)
        
        # Prepare response
        response = {
            "readiness_score": readiness_score,
            "user_skills": user_skills,
            "job_skills": job_skills,
            "skill_gaps": skill_gaps,
            "recommendations": recommendations,
            "quizzes": quizzes,
            "analysis_timestamp": now()
        }
        
        logger.info(f"Analysis completed. Readiness Score: {readiness_score}%")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected Error: {str(e)}")
        logger.error(f"Full Traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500, 
            detail=f"An unexpected error occurred during analysis: {str(e)}"
        )

@app.get("/")
def read_root():
    """Health check endpoint"""
    return {
        "message": "SkillNaav API is running!",
        "timestamp": now(),
        "version": "2.0.0"
    }

@app.get("/health")
def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "timestamp": now(),
        "services": {
            "bedrock": "connected",
            "spacy": "loaded"
        }
    }
