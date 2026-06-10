import uuid
import time
from sqlmodel import Session, select
from gads.core.database import engine
from gads.core.models import Project, Task

def main():
    project_id = uuid.UUID("ccb1b9ca-939e-435d-8983-905453c5aa13")
    
    last_status = {}
    start_time = time.time()
    
    # Run for up to 5 minutes to follow execution closely
    while time.time() - start_time < 300:
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
                    print(f"Task: {t.description[:65]}... | Status change: {last_status.get(t_id)} -> {t.status} | Assigned: {t.assigned_to}", flush=True)
                    if t.error:
                        print(f"    Error: {t.error}", flush=True)
                    last_status[t_id] = t.status
                    changed = True
                    
            if changed:
                completed = len([t for t in tasks if t.status == 'completed'])
                failed = len([t for t in tasks if t.status == 'failed'])
                running = len([t for t in tasks if t.status == 'running'])
                pending = len([t for t in tasks if t.status == 'pending'])
                print(f"Summary: Pending={pending}, Running={running}, Completed={completed}, Failed={failed}", flush=True)
                
            # Stop if the final report/dashboard tasks are complete or if it has failed tasks
            if len(tasks) > 5 and all(t.status in ('completed', 'failed', 'bypassed') for t in tasks):
                # Check if final task is completed
                if tasks[-1].status == 'completed':
                    print("All tasks completed! Exiting poll.", flush=True)
                    break
                
            time.sleep(2)

if __name__ == "__main__":
    main()
