from fastapi import APIRouter, HTTPException, status, BackgroundTasks, File, UploadFile, Form
from schemas import BatchAnswersSubmit, InterviewSessionStartResponse
from database import db
from services.langgraph_agent import generate_batch_questions, evaluate_batch_interview
import uuid
import PyPDF2
from io import BytesIO
from datetime import datetime, timezone

router = APIRouter()

@router.post("/start", response_model=InterviewSessionStartResponse)
async def start_interview(
    interview_id: str = Form(...),
    candidate_name: str = Form(...),
    candidate_email: str = Form(...),
    resume: UploadFile = File(None)   # Optional Resume upload!
):
    """
    Candidate opens the link and submits their Form. If a PDF resume is attached,
    we extract the text and dynamically inject it into the Gemini prompt!
    """
    template = await db.db.templates.find_one({"interview_id": interview_id})
    if not template:
        raise HTTPException(status_code=404, detail="Interview template not found")
        
    existing_session = await db.db.sessions.find_one({
        "interview_id": interview_id,
        "candidate_email": candidate_email
    })
    if existing_session:
        raise HTTPException(status_code=400, detail="A session with this email already exists for this interview.")
        
    session_id = str(uuid.uuid4())
    
    # 1. OPTIONAL: Parse the PDF Resume if they uploaded one
    resume_text = None
    if resume and resume.filename.endswith(".pdf"):
        try:
            pdf_reader = PyPDF2.PdfReader(BytesIO(await resume.read()))
            resume_text = ""
            for page in pdf_reader.pages:
                resume_text += page.extract_text()
        except Exception as e:
            print(f"Failed to parse resume: {e}")
            # If it fails, we gracefully continue without it
            resume_text = None
    
    # 2. Generate questions securely (Now dynamically contextualized if resume_text exists!)
    try:
        questions_array = generate_batch_questions(
            role=template["role"],
            skills=template["skills_required"],
            difficulty=template["difficulty"],
            total=template["total_questions"],
            resume_text=resume_text
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI Question Generation Failed: {str(e)}")
        
    # 3. Save session locally
    session_doc = {
        "session_id": session_id,
        "interview_id": interview_id,
        "candidate_name": candidate_name,
        "candidate_email": candidate_email,
        "company_id": template["company_id"],
        "status": "in_progress",
        "questions": questions_array, 
        "answers": [],                  
        "created_at": datetime.now(timezone.utc).isoformat()
    }
    await db.db.sessions.insert_one(session_doc)
    
    return {
        "session_id": session_id,
        "all_questions": questions_array
    }


@router.post("/submit_all")
async def submit_all_answers(payload: BatchAnswersSubmit, background_tasks: BackgroundTasks):
    """
    Final Submission. Checks strict lengths and triggers background evaluator.
    Absorbs cheating metrics (tab_switches, time_taken) collected by the frontend!
    """
    session = await db.db.sessions.find_one({"session_id": payload.session_id})
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
        
    if session["status"] in ["analyzing", "completed"]:
        raise HTTPException(status_code=400, detail="Interview is already finished!")
        
    if len(payload.answers) != len(session["questions"]):
        raise HTTPException(status_code=400, detail=f"Expected {len(session['questions'])} answers.")
        
    # Inject the cheating analytics metrics collected from the React JS Window directly into their permanent record!
    cheating_data = payload.cheating_flags.dict() if payload.cheating_flags else None

    # Update DB exactly once with answers AND metrics
    await db.db.sessions.update_one(
        {"session_id": payload.session_id},
        {"$set": {
            "answers": payload.answers,
            "cheating_flags": cheating_data,
            "status": "analyzing"
        }}
    )
    
    # Fire evaluator
    background_tasks.add_task(evaluate_batch_interview, payload.session_id, session["questions"], payload.answers)
    
    return {"message": "All answers submitted successfully! Evaluation is running in the background."}
