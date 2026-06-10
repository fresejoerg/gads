import uuid
from sqlmodel import Session
from gads.core.database import engine
from gads.core.models import Task

def main():
    task_id = uuid.UUID("e98cad5b-eada-4ff7-b6be-ca20bfe9cc60")
    with Session(engine) as session:
        task = session.get(Task, task_id)
        if task:
            print(f"Task ID: {task.id}")
            print(f"Status: {task.status}")
            print(f"Result JSON: {task.result_json}")
        else:
            print("Task not found")

if __name__ == "__main__":
    main()
