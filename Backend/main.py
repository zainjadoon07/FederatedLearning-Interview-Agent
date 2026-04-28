from fastapi import FastAPI
from contextlib import asynccontextmanager
from database import connect_to_mongo, close_mongo_connection, db
from routes import auth, templates, sessions, reports, settings
from services.ai_service import ai_evaluator


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup ---
    await connect_to_mongo()
    ai_evaluator.load_model()
    yield
    # --- Shutdown ---
    await close_mongo_connection()


app = FastAPI(
    title="Federated AI Interview Platform API",
    description="Multi-tenant backend for LLM-driven interviews",
    lifespan=lifespan
)

# --- Register Routers ---
app.include_router(auth.router, prefix="/api/auth", tags=["Authentication"])
app.include_router(settings.router, prefix="/api/settings", tags=["Company Preferences"])
app.include_router(templates.router, prefix="/api/templates", tags=["Interview Templates"])
app.include_router(sessions.router, prefix="/api/sessions", tags=["Candidate Interview Sessions"])
app.include_router(reports.router, prefix="/api/reports", tags=["Recruiter Analytics & Reports"])

@app.get("/")
async def health_check():
    """Simple endpoint to verify the API is running and connected to DB."""
    return {
        "status": "online",
        "database_connected": db.client is not None,
        "message": "Welcome to the Federated AI Interview Platform API!"
    }
