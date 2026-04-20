from fastapi import APIRouter, HTTPException, status, Depends
from schemas import CompanyCreate, CompanyLogin, Token, CompanyUpdate, CompanyDelete
from database import db
from utils.security import get_password_hash, verify_password, create_access_token, get_current_company
import uuid

router = APIRouter()

@router.post("/register", response_model=Token)
async def register_company(company: CompanyCreate):
    """
    Register a new company to use the SaaS platform.
    Hashes the password and generates a unique company_id.
    """
    try:
        # 1. Check if the company email is already taken
        existing_company = await db.db.companies.find_one({"email": company.email})
        if existing_company:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="A company with this email is already registered."
            )
        
        # 2. Create the company document
        company_id = str(uuid.uuid4())
        hashed_password = get_password_hash(company.password)
        
        company_doc = {
            "company_id": company_id,
            "name": company.name,
            "email": company.email,
            "hashed_password": hashed_password
        }
        
        # 3. Save to MongoDB
        await db.db.companies.insert_one(company_doc)
        
        # 4. Generate Session Token (JWT)
        access_token = create_access_token(data={"sub": str(company_id)})
        
        return {"access_token": access_token, "token_type": "bearer"}
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        raise HTTPException(status_code=500, detail=error_trace)

@router.post("/login", response_model=Token)
async def login_company(credentials: CompanyLogin):
    """
    Authenticate a company.
    Returns a JWT token to be used on all subsequent requests.
    """
    # 1. Look up company by email
    company_doc = await db.db.companies.find_one({"email": credentials.email})
    
    if not company_doc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password.",
        )
        
    # 2. Verify hashed password
    if not verify_password(credentials.password, company_doc["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password.",
        )
        
    # 3. Generate Session Token (JWT)
    access_token = create_access_token(data={"sub": str(company_doc["company_id"])})
    
    return {"access_token": access_token, "token_type": "bearer"}


# ---------------------------------------------------------
# PROTECTED ROUTES (Require valid JWT Token)
# ---------------------------------------------------------

@router.put("/profile", status_code=status.HTTP_200_OK)
async def update_company(
    payload: CompanyUpdate, 
    current_company: dict = Depends(get_current_company)
):
    """
    Update company details (name, password).
    Requires the client to send the Bearer Token in the headers.
    """
    update_data = {}
    
    if payload.name:
        update_data["name"] = payload.name
        
    if payload.new_password:
        update_data["hashed_password"] = get_password_hash(payload.new_password)
        
    if not update_data:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No fields provided to update.")
        
    # Apply changes specifically to the scoped company_id extracted securely from the JWT
    result = await db.db.companies.update_one(
        {"company_id": current_company["company_id"]},
        {"$set": update_data}
    )
    
    if result.modified_count == 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No changes made to profile.")
        
    return {"message": "Company profile successfully updated."}


@router.delete("/delete", status_code=status.HTTP_200_OK)
async def delete_company(
    payload: CompanyDelete, 
    current_company: dict = Depends(get_current_company)
):
    """
    Permanently delete the company account.
    Requires both the Bearer Token AND the explicit typed-in password for maximum safety.
    """
    # 1. Fetch the exact company document using the secure JWT context
    company_doc = await db.db.companies.find_one({"company_id": current_company["company_id"]})
    
    if not company_doc:
         raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Company not found")
         
    # 2. Verify their password input immediately before deletion
    if not verify_password(payload.password, company_doc["hashed_password"]):
         raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect password. Deletion aborted.")
    
    # 3. Permanently remove the document from MongoDB
    await db.db.companies.delete_one({"company_id": current_company["company_id"]})
    
    # Normally, the frontend will see this 200 OK and wipe the user's tokens!
    return {"message": "Your company account has been permanently deleted. You will now be logged out."}
