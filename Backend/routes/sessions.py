from fastapi import APIRouter, HTTPException, status, BackgroundTasks
from schemas import SessionStart, BatchAnswersSubmit, InterviewSessionStartResponse
from database import db
from services.langgraph_agent import generate_batch_questions, evaluate_batch_interview
import uuid
from datetime import datetime, timezone

router = APIRouter()

@router.post("/start", response_model=InterviewSessionStartResponse)
async def start_interview(payload: SessionStart):
    """
    Candidate opens the link and submits their Form (Email & Name).
    It generates all questions upfront and returns the ENTIRE array to the frontend.
    """
    template = await db.db.templates.find_one({"interview_id": payload.interview_id})
    if not template:
        raise HTTPException(status_code=404, detail="Interview template not found")
        
    session_id = str(uuid.uuid4())
    
    # 1. Generate ALL questions securely at once
    try:
        questions_array = generate_batch_questions(
            role=template["role"],
            skills=template["skills_required"],
            difficulty=template["difficulty"],
            total=template["total_questions"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI Question Generation Failed: {str(e)}")
        
    # 2. Save session strictly once at initialization
    session_doc = {
        "session_id": session_id,
        "interview_id": payload.interview_id,
        "candidate_name": payload.candidate_name,
        "candidate_email": payload.candidate_email,
        "status": "in_progress",
        "questions": questions_array,   # The pre-baked list
        "answers": [],                  # Empty array; the frontend will hold answers locally
        "created_at": datetime.now(timezone.utc).isoformat()
    }
    await db.db.sessions.insert_one(session_doc)
    
    # 3. Hand everything over to React!
    return {
        "session_id": session_id,
        "all_questions": questions_array
    }


@router.post("/submit_all")
async def submit_all_answers(payload: BatchAnswersSubmit, background_tasks: BackgroundTasks):
    """
    Called EXACTLY ONCE at the very end of the interview.
    The React frontend passes the array of all the applicant's answers.
    We save them to DB and trigger the Background Evaluation.
    """
    session = await db.db.sessions.find_one({"session_id": payload.session_id})
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
        
    if session["status"] in ["analyzing", "completed"]:
        raise HTTPException(status_code=400, detail="Interview is already finished!")
        
    if len(payload.answers) != len(session["questions"]):
        raise HTTPException(
            status_code=400, 
            detail=f"Expected {len(session['questions'])} answers, but received {len(payload.answers)}."
        )
        
    # 1. Update the DB exactly once with all the answers
    await db.db.sessions.update_one(
        {"session_id": payload.session_id},
        {"$set": {
            "answers": payload.answers,
            "status": "analyzing"
        }}
    )
    
    # 2. 💥 Fire the heavy AI evaluation silently into a background runner thread!
    background_tasks.add_task(evaluate_batch_interview, payload.session_id, session["questions"], payload.answers)
    
    return {"message": "All answers submitted successfully! Interview evaluation implies is running in the background."}
