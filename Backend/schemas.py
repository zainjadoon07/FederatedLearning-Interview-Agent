from pydantic import BaseModel, EmailStr
from typing import List

# -------------------------
# Company Authentication Schemas
# -------------------------
class CompanyCreate(BaseModel):
    name: str
    email: EmailStr
    password: str

class CompanyLogin(BaseModel):
    email: EmailStr
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class CompanyUpdate(BaseModel):
    name: str | None = None
    new_password: str | None = None

class CompanyDelete(BaseModel):
    # Enforces password matching before deletion
    password: str

class CompanySettings(BaseModel):
    passing_threshold: int = 70               # Default: must score 70% to pass
    ai_strictness: str = "strict"             # Options: lenient, strict, hyper-strict
    theme_color: str = "#2563EB"              # For frontend branding (blue default)
    company_site_url: str | None = None

# -------------------------
# Interview Template Schemas
# -------------------------
class InterviewTemplateCreate(BaseModel):
    role: str
    skills_required: List[str]
    difficulty: str = "medium"  # easy, medium, hard
    total_questions: int = 5

class InterviewTemplateResponse(BaseModel):
    interview_id: str
    company_id: str
    role: str
    skills_required: List[str]
    difficulty: str
    total_questions: int
    shareable_link: str
    created_at: str

# -------------------------
# Interview Session Schemas
# -------------------------
class SessionStart(BaseModel):
    interview_id: str
    candidate_name: str
    candidate_email: str

class CheatingFlags(BaseModel):
    tab_switches: int = 0
    copy_paste_attempts: int = 0
    time_taken_seconds: int = 0

class BatchAnswersSubmit(BaseModel):
    session_id: str
    answers: list[str]
    cheating_flags: CheatingFlags | None = None

class InterviewSessionStartResponse(BaseModel):
    session_id: str
    all_questions: list[str]
