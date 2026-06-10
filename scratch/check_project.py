import uuid
from sqlmodel import Session, select
from gads.core.database import engine
from gads.core.models import Project, Task

def main():
    project_id = uuid.UUID("79c9c753-010b-4720-9a62-b2867a2f3057")
    with Session(engine) as session:
        project = session.get(Project, project_id)
        if project:
            print(f"Project ID: {project.id}")
            print(f"Name: {project.name}")
            print(f"Narrative summary: {project.narrative[:300] if project.narrative else 'None'}")
            print(f"Takeaways: {project.takeaways}")
            
            # Check for failed tasks
            tasks = session.exec(select(Task).where(Task.project_id == project_id)).all()
            completed_tasks = [t for t in tasks if t.status == "completed"]
            failed_tasks = [t for t in tasks if t.status == "failed"]
            bypassed_tasks = [t for t in tasks if t.status == "bypassed"]
            print(f"\nTasks status:")
            print(f"  Completed: {len(completed_tasks)}")
            print(f"  Failed: {len(failed_tasks)}")
            print(f"  Bypassed: {len(bypassed_tasks)}")
            print(f"  Total: {len(tasks)}")
            
            # Print failed tasks
            if failed_tasks:
                print("\nFailed Tasks:")
                for t in failed_tasks[:5]:
                    print(f"  - ID: {t.id}")
                    print(f"    Desc: {t.description[:120]}...")
                    print(f"    Error: {t.error}")
            
            # Print completed tasks with code
            print("\nCompleted Tasks with Code:")
            for t in completed_tasks:
                if t.result_json and "code" in t.result_json:
                    print(f"  - ID: {t.id}")
                    print(f"    Desc: {t.description[:100]}...")
                    print(f"    Code length: {len(t.result_json['code'])}")
        else:
            print("Project not found")

if __name__ == "__main__":
    main()
