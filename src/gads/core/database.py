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
    from gads.core.models import Project, Task, Artifact, OutboxEvent, Instruction, ExecutionLog, User, ProviderKey
    
    max_retries = 5
    for attempt in range(max_retries):
        try:
            print(f"  [Database] Initializing tables (Attempt {attempt+1}/{max_retries})...")
            SQLModel.metadata.create_all(engine)
            migrate_tenancy()
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


def migrate_tenancy():
    """One-off, idempotent schema step for HOSTED-1 (there is no Alembic).

    create_all makes the new `gads_user` table but never adds a column to an existing table,
    so `project.owner_id` is added here, then every ownerless project is given to the admin.
    Runs on every startup; on an up-to-date DB it changes nothing.
    """
    from sqlalchemy import text
    from sqlmodel import select
    from gads.core import config
    from gads.core.models import User

    with Session(engine) as session:
        admin_email = config.admin_email()
        admin = session.exec(select(User).where(User.email == admin_email)).first()
        if admin is None:
            admin = User(email=admin_email, role="admin")
            session.add(admin)
            session.commit()
            session.refresh(admin)
            print(f"  [Database] Created admin user ({admin_email}).")
        elif admin.role != "admin":
            admin.role = "admin"
            session.add(admin)
            session.commit()
        session.execute(text(
            "ALTER TABLE project ADD COLUMN IF NOT EXISTS owner_id UUID REFERENCES gads_user(id)"))
        session.execute(text("CREATE INDEX IF NOT EXISTS ix_project_owner_id ON project (owner_id)"))
        n = session.execute(text("UPDATE project SET owner_id = :aid WHERE owner_id IS NULL")
                         .bindparams(aid=admin.id)).rowcount
        session.commit()
        if n:
            print(f"  [Database] Backfilled owner_id for {n} project(s) to the admin.")
