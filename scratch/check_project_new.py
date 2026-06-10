import uuid
import sys
from sqlmodel import Session, select
from gads.core.database import engine
from gads.core.models import Project, Task

def main():
    project_id = uuid.UUID("2b4a2683-34c6-44fa-bd9e-a5ac7d377314")
    with Session(engine) as session:
        project = session.get(Project, project_id)
        if not project:
            print("Project not found")
            return
            
        print(f"Project Name: {project.name}")
        print(f"Narrative: {project.narrative[:200] if project.narrative else 'None'}")
        
        tasks = session.exec(select(Task).where(Task.project_id == project_id).order_by(Task.created_at)).all()
        print(f"\nTasks status summary:")
        completed = [t for t in tasks if t.status == "completed"]
        failed = [t for t in tasks if t.status == "failed"]
        running = [t for t in tasks if t.status == "running"]
        pending = [t for t in tasks if t.status == "pending"]
        bypassed = [t for t in tasks if t.status == "bypassed"]
        print(f"  Pending: {len(pending)}")
        print(f"  Running: {len(running)}")
        print(f"  Completed: {len(completed)}")
        print(f"  Failed: {len(failed)}")
        print(f"  Bypassed: {len(bypassed)}")
        print(f"  Total: {len(tasks)}")
        
        if running:
            print("\nCurrently Running Tasks:")
            for t in running:
                print(f"  - {t.id}: {t.description[:100]} (assigned: {t.assigned_to})")
                
        if failed:
            print("\nFailed Tasks:")
            for t in failed:
                print(f"  - {t.id}: {t.description[:100]} | Error: {t.error}")

if __name__ == "__main__":
    main()
