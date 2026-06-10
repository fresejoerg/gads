import uuid
from sqlmodel import Session, select
from gads.core.database import engine
from gads.core.models import Project, Task

def main():
    project_id = uuid.UUID("79c9c753-010b-4720-9a62-b2867a2f3057")
    with Session(engine) as session:
        project = session.get(Project, project_id)
        if not project:
            print("Project not found")
            return
            
        print(f"Project: {project.name}")
        
        tasks = session.exec(
            select(Task).where(Task.project_id == project_id).order_by(Task.created_at)
        ).all()
        
        print(f"Total tasks: {len(tasks)}")
        for i, t in enumerate(tasks):
            print(f"{i+1:02d}. ID: {t.id} | Status: {t.status} | Assigned: {t.assigned_to}")
            print(f"    Desc: {t.description[:100]}")
            if t.error:
                print(f"    Error: {t.error}")
            if t.result_json and "recipe_id" in t.result_json:
                print(f"    Matched Recipe: {t.result_json['recipe_id']}")

if __name__ == "__main__":
    main()
