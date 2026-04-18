from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime

class Artifact(BaseModel):
    """A unit of work produced by an agent."""
    id: str
    type: str  # e.g., "plot", "dataframe_summary", "entities", "code"
    description: str
    content: Any
    created_at: datetime = Field(default_factory=datetime.now)
    agent_id: str

class Task(BaseModel):
    """A discrete step in the project plan."""
    id: str
    description: str
    assigned_to: str  # Agent model name or alias
    status: str = "pending"  # "pending", "in_progress", "completed", "failed"
    result_artifact_ids: List[str] = []
    error: Optional[str] = None

class Blackboard(BaseModel):
    """The shared state for the multi-agent system."""
    project_name: str
    objective: str
    plan: List[Task] = []
    artifacts: Dict[str, Artifact] = {}
    metadata: Dict[str, Any] = {}

    def add_artifact(self, artifact: Artifact):
        self.artifacts[artifact.id] = artifact

    def get_artifact(self, artifact_id: str) -> Optional[Artifact]:
        return self.artifacts.get(artifact_id)
