import uuid
from sqlmodel import Session, select
from gads.core.database import engine
from gads.core.models import Project, Task

def main():
    with Session(engine) as session:
        projects = session.exec(select(Project)).all()
        print(f"Total projects: {len(projects)}")
        for p in projects:
            print(f"\nProject ID: {p.id}")
            print(f"Name: {p.name}")
            print(f"Objective: {p.objective[:200]}...")
            
            tasks = session.exec(select(Task).where(Task.project_id == p.id)).all()
            print(f"Tasks: {len(tasks)}")
            for t in tasks:
                print(f"  - Task ID: {t.id}")
                print(f"    Status: {t.status}")
                print(f"    Description: {t.description[:150]}...")
                if t.result_json:
                    print(f"    Has code: {'code' in t.result_json}")

if __name__ == "__main__":
    main()
