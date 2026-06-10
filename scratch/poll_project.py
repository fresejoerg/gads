import uuid
import time
from sqlmodel import Session, select
from gads.core.database import engine
from gads.core.models import Project, Task

def main():
    project_id = uuid.UUID("2b4a2683-34c6-44fa-bd9e-a5ac7d377314")
    
    last_status = {}
    start_time = time.time()
    
    while time.time() - start_time < 90:
        with Session(engine) as session:
            project = session.get(Project, project_id)
            if not project:
                print("Project not found")
                return
                
            tasks = session.exec(select(Task).where(Task.project_id == project_id).order_by(Task.created_at)).all()
            
            changed = False
            for t in tasks:
                t_id = str(t.id)
                if last_status.get(t_id) != t.status:
                    print(f"Task: {t.description[:60]}... | Status change: {last_status.get(t_id)} -> {t.status} | Assigned: {t.assigned_to}")
                    last_status[t_id] = t.status
                    changed = True
                    
            if changed:
                completed = len([t for t in tasks if t.status == 'completed'])
                failed = len([t for t in tasks if t.status == 'failed'])
                running = len([t for t in tasks if t.status == 'running'])
                pending = len([t for t in tasks if t.status == 'pending'])
                print(f"Summary: Pending={pending}, Running={running}, Completed={completed}, Failed={failed}")
                
            # If all tasks are completed or failed, we can stop polling (unless planner is still generating more)
            # Actually, let's just sleep a bit and continue
            time.sleep(2)

if __name__ == "__main__":
    main()
