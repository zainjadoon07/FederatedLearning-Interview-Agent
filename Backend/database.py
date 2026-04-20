import os
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME", "FedInterviewAI")

if not MONGO_URI:
    raise ValueError("MONGO_URI not found in .env file!")

class Database:
    client: AsyncIOMotorClient = None
    db = None

db = Database()

async def connect_to_mongo():
    """Establish connection to MongoDB asynchronously"""
    try:
        print(" Attempting to connect to MongoDB...")
        db.client = AsyncIOMotorClient(MONGO_URI)
        db.db = db.client[DB_NAME]
        
        # Ping the database to verify the connection actually works
        await db.client.admin.command('ping')
        print(f" Successfully connected to MongoDB! (Database: {DB_NAME})")
    except Exception as e:
        print(f" Failed to connect to MongoDB: {e}")
        raise e

async def close_mongo_connection():
    """Close the MongoDB connection pool"""
    if db.client:
        db.client.close()
        print(" MongoDB connection successfully closed.")
