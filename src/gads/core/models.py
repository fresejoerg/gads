from typing import Optional, List, Dict, Any
from datetime import datetime
import uuid
from sqlmodel import SQLModel, Field, JSON, Column, Relationship
from sqlalchemy import UniqueConstraint

class User(SQLModel, table=True):
    """A login (approach_docs/032 §4). Table `gads_user`: `user` is reserved in Postgres."""
    __tablename__ = "gads_user"
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    email: str = Field(index=True, unique=True)
    role: str = "tester"  # admin | tester
    created_at: datetime = Field(default_factory=datetime.now)
    # Run-shaping overrides over the global /config layer (registry.RUN_SETTING_KEYS).
    settings_json: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(JSON))


class ProviderKey(SQLModel, table=True):
    """A user's own LLM provider key (BYOK, HOSTED-2a), Fernet-encrypted with GADS_SECRET_KEY.

    The raw key is never returned by the API; only `last4` is shown. One row per (user, provider).
    """
    __tablename__ = "gads_provider_key"
    __table_args__ = (UniqueConstraint("user_id", "provider", name="uq_provider_key_user_provider"),)
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    user_id: uuid.UUID = Field(foreign_key="gads_user.id", index=True)
    provider: str  # anthropic | gemini | openai
    ciphertext: str
    last4: str
    created_at: datetime = Field(default_factory=datetime.now)


class Project(SQLModel, table=True):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    name: str = Field(index=True)
    # Owner (HOSTED-1). Nullable: rows from before tenancy are backfilled to the admin in
    # init_db, and a NULL owner means admin-only.
    owner_id: Optional[uuid.UUID] = Field(default=None, foreign_key="gads_user.id", index=True)
    objective: str
    narrative: Optional[str] = None
    takeaways: Optional[List[str]] = Field(default=None, sa_column=Column(JSON))
    last_state_json: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    instructions: List["Instruction"] = Relationship(back_populates="project")
    tasks: List["Task"] = Relationship(back_populates="project")
    artifacts: List["Artifact"] = Relationship(back_populates="project")

class Instruction(SQLModel, table=True):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    project_id: uuid.UUID = Field(foreign_key="project.id")
    content: str
    created_at: datetime = Field(default_factory=datetime.now)
    
    project: Project = Relationship(back_populates="instructions")
    tasks: List["Task"] = Relationship(back_populates="instruction")

class Task(SQLModel, table=True):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    project_id: uuid.UUID = Field(foreign_key="project.id")
    instruction_id: Optional[uuid.UUID] = Field(default=None, foreign_key="instruction.id")
    description: str
    assigned_to: str
    status: str = "pending"  # pending, running, completed, failed, bypassed
    escalation_count: int = 0
    estimated_runtime_s: Optional[float] = None
    postcondition_json: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(JSON))
    attached_skills: Optional[List[str]] = Field(default=None, sa_column=Column(JSON))
    heartbeat: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.now)
    error: Optional[str] = None
    result_json: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(JSON))
    
    project: Project = Relationship(back_populates="tasks")
    instruction: Optional[Instruction] = Relationship(back_populates="tasks")

class ExecutionLog(SQLModel, table=True):
    """Historical data for the Runtime Oracle's learned estimator."""
    id: Optional[int] = Field(default=None, primary_key=True)
    estimator_class: str = Field(index=True)
    n_rows: int
    m_cols: int
    params_json: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(JSON))
    hardware_id: str
    actual_runtime_s: float
    created_at: datetime = Field(default_factory=datetime.now)

class Artifact(SQLModel, table=True):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    project_id: uuid.UUID = Field(foreign_key="project.id")
    type: str  # plot, dataframe, entities, code
    description: str
    content_json: Dict[str, Any] = Field(sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=datetime.now)
    agent_id: str
    
    project: Project = Relationship(back_populates="artifacts")

class OutboxEvent(SQLModel, table=True):
    id: int = Field(default=None, primary_key=True)
    sequence: int = Field(unique=True, index=True)
    type: str  # TASK_STARTED, TASK_COMPLETED, ARTIFACT_CREATED, etc.
    payload_json: Dict[str, Any] = Field(sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=datetime.now)
    processed: bool = False
