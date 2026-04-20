import os
from typing import List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel
from services.ai_service import ai_evaluator
from database import db

# Initialize Gemini
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.7)

# Pydantic schema to force Gemini to output a perfect JSON array
class QuestionList(BaseModel):
    questions: List[str]


def generate_batch_questions(role: str, skills: List[str], difficulty: str, total: int) -> List[str]:
    """
    Called exactly once at the start of the interview. 
    Generates all questions upfront so the user experiences zero latency during the interview.
    """
    print(f"Generating {total} {difficulty} questions for {role} upfront...")
    
    prompt = f"""
    You are an expert technical interviewer hiring a {role}.
    The required skills are: {', '.join(skills)}.
    The core difficulty is '{difficulty.upper()}'.
    
    Generate EXACTLY {total} unique, atomic interview questions.
    Ensure they are direct and can be answered in a standard paragraph. Do not ask multi-part essays.You can give exlanation or examples for the questions though, for example if its a coding question , then you have to give user inputs and examples as well!
    """
    
    # Enforces perfect JSON array extraction from the LLM response
    structured_llm = llm.with_structured_output(QuestionList)
    
    response = structured_llm.invoke([
        SystemMessage(content="You are a strict technical interviewer. You output only structured JSON arrays of questions."),
        HumanMessage(content=prompt)
    ])
    
    return response.questions


async def evaluate_batch_interview(session_id: str, questions: List[str], answers: List[str]):
    """
    Runs completely silently in the background AFTER the final question is answered.
    Iterates over all Q&A pairs, runs DistilBERT, runs Gemini, and computes the final aggregate grade.
    """
    print(f" Starting background batch evaluation for session {session_id}...")
    
    history = []
    total_score = 0
    
    for i in range(len(questions)):
        q = questions[i]
        a = answers[i] if i < len(answers) else ""
        
        # 1. DistilBERT Baseline
        model_score = ai_evaluator.evaluate(q, a)
        
        # 2. Gemini Verification
        eval_prompt = f"""
        Question Asked: "{q}"
        Candidate Answer: "{a}"
        
        Evaluate the candidate's answer and score it on this scale, Be sure to be unbiased and look for absolute clarity to humans , text length matters as well . be judge as if you are a human trying to understand the answer which you don't know , the best practice would be to look for clarity and correctness , don't rely too heavily on the language errors or grammatical errors , just pure core answer of the question :
        0 = Poor / "I don't know" / Completely irrelevant / mostly wrong
        1 = Average / Partially correct
        2 = Excellent / Highly accurate and well reasoned
        
        Respond WITH ONLY A SINGLE DIGIT (0, 1, or 2). Absolutely no other text.
        """
        
        try:
            response = llm.invoke([SystemMessage(content="You output only 0, 1, or 2."), HumanMessage(content=eval_prompt)])
            llm_score = int(response.content.strip())
            final_score = llm_score  # We trust LLM overrides for accuracy
        except Exception as e:
            print(f" Gemini Eval failed for Q{i+1}: {e}")
            llm_score = None
            final_score = model_score
            
        history.append({
            "question": q,
            "answer": a,
            "final_score": final_score,
            "model_score": model_score,
            "gemini_score": llm_score
        })
        total_score += final_score

    # 3. Calculate Final Grade Metric
    max_possible = len(questions) * 2
    percentage = (total_score / max_possible) * 100 if max_possible > 0 else 0
    
    # 4. Save Final Annotated Results to MongoDB Data Warehouse
    await db.db.sessions.update_one(
        {"session_id": session_id},
        {"$set": {
            "status": "completed",
            "evaluation_history": history,
            "final_grade_percentage": round(percentage, 2)
        }}
    )
    print(f"Batch evaluation FINISHED for {session_id}. Final Grade: {round(percentage, 2)}%")
