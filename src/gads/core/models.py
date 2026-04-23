from typing import Optional, List, Dict, Any
from datetime import datetime
import uuid
from sqlmodel import SQLModel, Field, JSON, Column, Relationship

class Project(SQLModel, table=True):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    name: str = Field(index=True)
    objective: str
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    tasks: List["Task"] = Relationship(back_populates="project")
    artifacts: List["Artifact"] = Relationship(back_populates="project")

class Task(SQLModel, table=True):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    project_id: uuid.UUID = Field(foreign_key="project.id")
    description: str
    assigned_to: str
    status: str = "pending"  # pending, running, completed, failed
    escalation_count: int = 0
    postcondition_json: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(JSON))
    heartbeat: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.now)
    error: Optional[str] = None
    result_json: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(JSON))
    
    project: Project = Relationship(back_populates="tasks")

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
