from fastapi import FastAPI
from contextlib import asynccontextmanager
from database import connect_to_mongo, close_mongo_connection, db

# Lifespan context manager ensures early database connection on startup
# and clean disconnection on shutdown.
@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup ---
    await connect_to_mongo()
    yield
    # --- Shutdown ---
    await close_mongo_connection()

app = FastAPI(
    title="Federated AI Interview Platform API",
    description="Multi-tenant backend for LLM-driven interviews",
    lifespan=lifespan
)

@app.get("/")
async def health_check():
    """Simple endpoint to verify the API is running and connected to DB."""
    return {
        "status": "online",
        "database_connected": db.client is not None,
        "message": "Welcome to the Federated AI Interview Platform API!"
    }
