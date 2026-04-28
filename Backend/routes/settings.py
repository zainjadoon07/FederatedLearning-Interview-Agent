from fastapi import APIRouter, HTTPException, Depends
from schemas import CompanySettings
from database import db
from utils.security import get_current_company

router = APIRouter()

@router.get("/get", response_model=CompanySettings)
async def get_settings(current_company: dict = Depends(get_current_company)):
    """
    Retrieves the company's preferences (e.g. AI strictness, branding colors).
    """
    company_id = current_company["company_id"]
    
    settings = await db.db.company_settings.find_one({"company_id": company_id}, {"_id": 0})
    if not settings:
        # Return universal defaults seamlessly if the company hasn't modified anything yet
        return CompanySettings()
        
    return settings


@router.put("/update", response_model=dict)
async def update_settings(payload: CompanySettings, current_company: dict = Depends(get_current_company)):
    """
    Allows the company to permanently persist their HR workflows and strictness preferences.
    """
    company_id = current_company["company_id"]
    
    # Convert Pydantic model to raw dict for MongoDB
    update_data = payload.dict()
    update_data["company_id"] = company_id
    
    # 'upsert=True' is a smart MongoDB command that inserts a new document if one 
    # doesn't exist, or safely overrides the existing document if it does!
    await db.db.company_settings.update_one(
        {"company_id": company_id},
        {"$set": update_data},
        upsert=True
    )
    
    return {"message": "Company preference settings securely updated!"}
