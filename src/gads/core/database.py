import os
from sqlmodel import SQLModel, create_engine, Session
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

DATABASE_URL = os.getenv("GADS_DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("GADS_DATABASE_URL not set in environment")

# Create the engine
engine = create_engine(DATABASE_URL, echo=False)

def init_db():
    """Create database tables."""
    # IMPORTANT: Import models here so they are registered with SQLModel.metadata
    from gads.core.models import Project, Task, Artifact, OutboxEvent
    SQLModel.metadata.create_all(engine)

def get_session():
    """Get a new database session."""
    return Session(engine)
