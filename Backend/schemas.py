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
