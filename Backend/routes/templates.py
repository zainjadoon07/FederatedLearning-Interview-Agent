from fastapi import APIRouter, HTTPException, status, Depends
from schemas import InterviewTemplateCreate, InterviewTemplateResponse
from database import db
from utils.security import get_current_company
from datetime import datetime, timezone
import uuid
import os

router = APIRouter()

@router.post("/create", response_model=InterviewTemplateResponse)
async def create_interview_template(
    payload: InterviewTemplateCreate,
    current_company: dict = Depends(get_current_company)
):
    """
    Creates a new interview configuration template tied to the logged-in company.
    Returns the uniquely generated shareable link for the candidate.
    """
    interview_id = str(uuid.uuid4())
    
    # The base URL where your Next.js application will live
    frontend_base_url = os.getenv("FRONTEND_URL")
    if not frontend_base_url:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="FRONTEND_URL not configured in environment variables."
        )
    shareable_link = f"{frontend_base_url}/interview/{interview_id}"
    
    timestamp = datetime.now(timezone.utc).isoformat()
    
    template_doc = {
        "interview_id": interview_id,
        "company_id": current_company["company_id"],
        "role": payload.role,
        "skills_required": payload.skills_required,
        "difficulty": payload.difficulty,
        "total_questions": payload.total_questions,
        "shareable_link": shareable_link,
        "created_at": timestamp
    }
    
    # Store strictly inside the MongoDB 'templates' collection
    await db.db.templates.insert_one(template_doc)
    
    return template_doc


@router.get("/list", response_model=list[InterviewTemplateResponse])
async def list_company_templates(current_company: dict = Depends(get_current_company)):
    """
    Returns all interview templates created by the logged-in company.
    Enforces strict tenant isolation by filtering via the JWT's company_id.
    """
    # MongoDB query isolated strictly to the company requesting it
    cursor = db.db.templates.find({"company_id": current_company["company_id"]})
    templates = await cursor.to_list(length=100)
    
    # Clean out the internal MongoDB '_id' ObjectId before returning JSON
    for t in templates:
        t.pop("_id", None)
        
    return templates
