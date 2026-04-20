import bcrypt
import jwt
import os
from datetime import datetime, timedelta, timezone
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

# JWT Configuration
# In production, ALWAYS load SECRET_KEY from .env
SECRET_KEY = os.getenv("SECRET_KEY", "your-super-secret-development-key")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 1440  # 24 Hours

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Compare plain password against the hashed database value using raw bcrypt."""
    return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))

def get_password_hash(password: str) -> str:
    """Create a bcrypt hash for a new password using raw bcrypt."""
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed.decode('utf-8')

def create_access_token(data: dict) -> str:
    """Generate a JWT encoding the user claims (like company_id)."""
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# JWT Verification Dependency
security = HTTPBearer()

def get_current_company(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    """
    Validates the Bearer token in the request headers and extracts the company_id.
    Attach this to any route that requires a logged-in user!
    """
    token = credentials.credentials
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        company_id = payload.get("sub")
        if company_id is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid auth token")
        return {"company_id": company_id}
        
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token has expired. Please log in again.")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid auth token format.")
