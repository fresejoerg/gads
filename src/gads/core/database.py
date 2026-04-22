import os
import time
from sqlmodel import SQLModel, create_engine, Session
from sqlalchemy.exc import OperationalError
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

DATABASE_URL = os.getenv("GADS_DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("GADS_DATABASE_URL not set in environment")

# Create the engine
engine = create_engine(DATABASE_URL, echo=False)

def init_db():
    """Create database tables with retry logic."""
    # IMPORTANT: Import models here so they are registered with SQLModel.metadata
    from gads.core.models import Project, Task, Artifact, OutboxEvent
    
    max_retries = 5
    for attempt in range(max_retries):
        try:
            print(f"  [Database] Initializing tables (Attempt {attempt+1}/{max_retries})...")
            SQLModel.metadata.create_all(engine)
            print("  [Database] Tables initialized successfully.")
            return
        except OperationalError as e:
            if attempt == max_retries - 1:
                print(f"  [Database] Failed to connect after {max_retries} attempts.")
                raise e
            print(f"  [Database] Database not ready, retrying in 2s... ({e})")
            time.sleep(2)

def get_session():
    """Get a new database session."""
    return Session(engine)
