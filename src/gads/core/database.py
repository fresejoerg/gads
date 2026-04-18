import os
from sqlmodel import SQLModel, create_engine, Session
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL not set in environment")

# Create the engine
# Note: we don't need connect_args={"check_same_thread": False} for Postgres
engine = create_engine(DATABASE_URL, echo=False)

def init_db():
    """Create database tables."""
    SQLModel.metadata.create_all(engine)

def get_session():
    """Get a new database session."""
    return Session(engine)
