from pydantic import BaseModel, EmailStr

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
