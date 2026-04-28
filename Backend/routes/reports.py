from fastapi import APIRouter, HTTPException, Depends
from database import db
from utils.security import get_current_company

router = APIRouter()

@router.get("/candidates")
async def get_all_company_sessions(current_company: dict = Depends(get_current_company)):
    """
    Returns a high-level summary of every candidate that has ever taken
    an interview belonging to the logged-in company.
    """
    company_id = current_company["company_id"]
    
    # Securely retrieve only sessions tied to this company's templates
    cursor = db.db.sessions.find(
        {"company_id": company_id},
        {"_id": 0, "session_id": 1, "interview_id": 1, "candidate_name": 1, "candidate_email": 1, "status": 1, "final_grade_percentage": 1, "created_at": 1} # Exclude the massive Q&A arrays for speed
    ).sort("created_at", -1) # Newest first
    
    sessions = await cursor.to_list(length=500)
    return sessions


@router.get("/candidate/{session_id}")
async def get_detailed_candidate_report(session_id: str, current_company: dict = Depends(get_current_company)):
    """
    Returns the complete, detailed breakdown of a specific candidate's interview,
    including every question, their answer, and the DistilBERT vs Gemini scores.
    """
    company_id = current_company["company_id"]
    
    session = await db.db.sessions.find_one(
        {"session_id": session_id, "company_id": company_id}, 
        {"_id": 0}
    )
    
    if not session:
        raise HTTPException(status_code=404, detail="Candidate session not found or unauthorized.")
        
    return session
